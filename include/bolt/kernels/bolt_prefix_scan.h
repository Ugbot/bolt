// bolt/kernels/bolt_prefix_scan.h — blocked parallel INCLUSIVE prefix sum
// (scan) for int64 / float64, the classic three-phase decomposition:
//
//   Phase 1 (parallel): each fixed-size block runs a serial left-to-right
//                       inclusive scan of its own rows into `out`, and
//                       records its block total.
//   Phase 2 (serial):   exclusive scan over the block totals — each
//                       block's additive offset (the sum of every block
//                       before it).
//   Phase 3 (parallel): each block (except block 0) adds its offset to
//                       its slice of `out`.
//
// The intended consumer is the window-family cumulative frame
// (`SUM(x) OVER (ORDER BY t ROWS UNBOUNDED PRECEDING)`, plan E2 phase 2):
// a single-partition cumulative window is a pure prefix scan, previously
// pinned serial in chukonu's window_agg_op.
//
// DETERMINISM: the block width is the FIXED constant k_scan_block_rows —
// never derived from the worker count — so the result is identical for
// any pool size (including a pool of one). For int64 the result is exact
// and equal to the serial fold. For float64 the blocked association
// `offsets[b] + local[i]` performs ONE extra rounding step per row vs
// the strict serial left fold — a deterministic, well-defined summation
// order (the same class of reassociation every blocked/SIMD summation
// performs), but NOT bit-identical to the serial fold for general
// floats. Callers that require bit-equality with a serial running fold
// (or whose doubles are integer-valued below 2^53, e.g. widened Int64
// columns, for which the blocked result IS exact and bit-identical)
// must make that call — see chukonu window_agg_op.cpp's gating.
//
// Tiger Style: noexcept, bounded loops, ≥2 asserts per function, no heap
// on the hot path (block totals come from the caller's arena), serial
// fallback on any resource miss — never a silent wrong answer.

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_scheduler.h"

namespace bolt {
namespace kernels {
namespace prefix_scan {

// Fixed block width. 65,536 rows (512 KiB of float64) balances per-task
// scheduling overhead against work-stealing granularity; being a fixed
// constant (not n/workers) is what makes the float64 result independent
// of the pool size.
inline constexpr std::int64_t k_scan_block_rows = 1 << 16;

// Hard cap on block count (Tiger Style upper bound): 2^20 blocks =
// 2^36 rows. submit_range's uint32 row index caps engagement well
// before this; the constant guards the arithmetic.
inline constexpr std::int64_t k_scan_max_blocks = 1 << 20;

// Serial inclusive scan — the strict left fold. Reference semantics for
// int64; for float64 this is the "serial running fold" association.
// NOTE: `in`/`out` deliberately NOT restrict-qualified — in-place
// (in == out) is a legal call shape for every function in this header.
template <typename T>
inline void inclusive_scan_serial(const T*     in,
                                  std::int64_t n,
                                  T*           out) noexcept {
    assert((in != nullptr && out != nullptr) || n == 0);
    assert(n >= 0);
    T acc = T{0};
    for (std::int64_t i = 0; i < n; ++i) {
        acc += in[i];
        out[i] = acc;
    }
}

template <typename T>
struct ScanBlockCtx {
    const T* in;        // may equal `out` (in-place)
    T*       out;
    T*       totals;    // one per block
};

// Phase 1: local inclusive scan of one block; record the block total.
// Blocks are disjoint slices of `out` — no cross-task aliasing.
template <typename T>
inline void scan_block_local(void* user_data, std::uint32_t start,
                             std::uint32_t end,
                             std::uint32_t /*thread_id*/) noexcept {
    assert(user_data != nullptr);
    assert(end > start);
    auto* ctx = static_cast<ScanBlockCtx<T>*>(user_data);
    const std::uint32_t b = start / static_cast<std::uint32_t>(k_scan_block_rows);
    T acc = T{0};
    for (std::uint32_t i = start; i < end; ++i) {
        acc += ctx->in[i];
        ctx->out[i] = acc;
    }
    ctx->totals[b] = acc;
}

template <typename T>
struct ScanAddCtx {
    T*       out;
    const T* offsets;   // exclusive scan of block totals (arena-owned,
                        // never aliases `out`)
};

// Phase 3: add the block's offset to its slice. Block 0's offset is
// T{0}; the branchless add is cheaper than skipping it.
template <typename T>
inline void scan_block_add(void* user_data, std::uint32_t start,
                           std::uint32_t end,
                           std::uint32_t /*thread_id*/) noexcept {
    assert(user_data != nullptr);
    assert(end > start);
    auto* ctx = static_cast<ScanAddCtx<T>*>(user_data);
    const std::uint32_t b = start / static_cast<std::uint32_t>(k_scan_block_rows);
    const T off = ctx->offsets[b];
    for (std::uint32_t i = start; i < end; ++i) {
        ctx->out[i] += off;
    }
}

// Blocked parallel inclusive scan. Returns true when the parallel path
// ran; false means the caller's inputs were served by the SERIAL fold
// (no scheduler / single-threaded pool / n too small / n past uint32 /
// arena scratch miss) — the output is correct either way; the return
// value only reports which association produced it (relevant for
// float64 bit-comparisons, see the header comment).
//
// `in` and `out` may not alias unless equal (in-place scan is legal:
// each phase reads a slot only before or at the write to it).
template <typename T>
inline bool inclusive_scan(Scheduler*   sched,
                           const T*     in,
                           std::int64_t n,
                           T*           out,
                           Arena*       arena) noexcept {
    static_assert(sizeof(T) == 8, "int64 / float64 lanes only");
    assert((in != nullptr && out != nullptr) || n == 0);
    assert(n >= 0);
    if (sched == nullptr || sched->thread_count() <= 1 ||
        n <= k_scan_block_rows ||
        n > static_cast<std::int64_t>(0xFFFFFFFFu)) {
        inclusive_scan_serial(in, n, out);
        return false;
    }
    const std::int64_t num_blocks =
        (n + k_scan_block_rows - 1) / k_scan_block_rows;
    assert(num_blocks >= 2);
    if (num_blocks > k_scan_max_blocks || arena == nullptr) {
        inclusive_scan_serial(in, n, out);
        return false;
    }
    T* totals  = arena->allocate_array<T>(static_cast<std::size_t>(num_blocks));
    T* offsets = arena->allocate_array<T>(static_cast<std::size_t>(num_blocks));
    if (totals == nullptr || offsets == nullptr) {
        inclusive_scan_serial(in, n, out);
        return false;
    }

    // Phase 1: per-block local scans (grain == block width, so morsel
    // boundaries ARE block boundaries and start / block_rows is exact).
    ScanBlockCtx<T> c1{ in, out, totals };
    sched->submit_range(&scan_block_local<T>, &c1,
                        static_cast<std::uint32_t>(n),
                        static_cast<std::uint32_t>(k_scan_block_rows));
    sched->wait_all();

    // Phase 2: serial exclusive scan of block totals -> per-block offset.
    T acc = T{0};
    for (std::int64_t b = 0; b < num_blocks; ++b) {
        offsets[b] = acc;
        acc += totals[b];
    }

    // Phase 3: parallel offset add (block 0 adds T{0}).
    ScanAddCtx<T> c3{ out, offsets };
    sched->submit_range(&scan_block_add<T>, &c3,
                        static_cast<std::uint32_t>(n),
                        static_cast<std::uint32_t>(k_scan_block_rows));
    sched->wait_all();
    return true;
}

}  // namespace prefix_scan
}  // namespace kernels
}  // namespace bolt
