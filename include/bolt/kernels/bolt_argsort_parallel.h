// bolt_argsort_parallel.h — Parallel multi-block indirect (argsort) sort.
//
// A PARALLEL companion to merge_sort_indirect_u32 (bolt_argsort.h). It keeps
// the single-threaded kernel untouched and adds a multi-block variant that:
//   1. splits [0,n) into K disjoint blocks (K = bolt hardware concurrency via
//      the caller's Scheduler, capped at kParallelSortMaxBlocks),
//   2. sorts every block in parallel on the Scheduler's worker pool, each task
//      using the EXISTING stable single-threaded merge_sort_indirect_u32 over
//      its slice (disjoint perm/scratch ranges — no cross-task sharing), then
//   3. bottom-up merges the K sorted runs, ping-ponging perm<->scratch, reusing
//      the existing stable merge_sort_run_u32 primitive at block granularity.
//
// The result is byte-identical to the single-threaded kernel: same STABLE
// order (ties keep original index), same permutation. Determinism is preserved
// because the block split is a fixed function of n and K, and every merge uses
// the same "<= takes left" stable rule.
//
// This header is SEPARATE from bolt_argsort.h so the pure kernel header stays
// free of <thread>/<atomic>/scheduler dependencies — callers that only need
// the single-threaded path never pull in the scheduler.
//
// Scratch / no-heap
// -----------------
//   The driver NEVER calls malloc/new. The merge path needs a uint32 scratch
//   buffer of exactly n entries; the caller owns it. The Arena* overload
//   allocates that scratch from a caller-supplied bolt::Arena (still zero
//   malloc on the hot path — arena bump allocation), so callers that already
//   hold an Arena don't have to manage the buffer themselves.
//
// Tiger Style:
//   * noexcept; no exceptions, no RTTI, no allocation on the hot path.
//   * BOLT_RESTRICT on pointer parameters; branchless inner merge (inherited
//     from merge_sort_run_u32).
//   * Bounded K (kParallelSortMaxBlocks) and bounded merge-tree depth
//     (<= ceil(log2 K)). ≥2 assertions per function (preconditions +
//     a fully-sorted postcondition; permutation invariant asserted in tests).
//   * Functions ≤70 lines.
//
// Caveat: must be called from a thread that is NOT one of `sched`'s workers
// (it blocks on wait_all()). Call it from the submitting/control thread.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/kernels/bolt_argsort.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace kernels {

// One block per worker, capped. Bounded (Tiger Style): keeps the scratch
// partitioning and the bottom-up merge-tree depth bounded regardless of the
// host's logical-CPU count.
inline constexpr int64_t kParallelSortMaxBlocks = 64;

// Below this row count the per-task + merge overhead loses to the single
// threaded merge sort, so small inputs route straight through it.
inline constexpr int64_t kParallelSortMinRows = 1 << 15;  // 32768

// Debug-only postcondition helper: true iff `perm` is non-decreasing under
// `less` (no element sorts strictly before its predecessor). Read-only — safe
// inside assert(). Bounded single pass.
template <typename Compare>
BOLT_FORCE_INLINE bool parallel_argsort_sorted_u32(
        const uint32_t* BOLT_RESTRICT perm, int64_t n, Compare less) noexcept {
    assert(perm != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 1; i < n; ++i) {
        if (less(perm[i], perm[i - 1])) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Phase 1 — per-block sort task. submit_range tiles [0,n) into blocks of
// `block_size`; each invocation owns exactly one block [start,end). It writes
// identity into its perm slice then stable-sorts it with the single-threaded
// kernel over disjoint perm/scratch ranges (no sharing across tasks).
// ---------------------------------------------------------------------------
template <typename Compare>
struct ParallelArgsortCtx {
    uint32_t* BOLT_RESTRICT perm;
    uint32_t* BOLT_RESTRICT scratch;
    Compare less;
};

template <typename Compare>
inline void parallel_argsort_block_task(void* user_data, uint32_t start,
                                        uint32_t end,
                                        uint32_t /*thread_id*/) noexcept {
    assert(user_data != nullptr);
    assert(end >= start);
    auto* c = static_cast<ParallelArgsortCtx<Compare>*>(user_data);
    for (uint32_t i = start; i < end; ++i) c->perm[i] = i;
    const int64_t len = static_cast<int64_t>(end) - static_cast<int64_t>(start);
    merge_sort_indirect_u32(c->perm + start, c->scratch + start, len, c->less);
}

// ---------------------------------------------------------------------------
// Phase 2 — one merge task per adjacent run-pair in a bottom-up pass. Pairs
// tile every block, so [0,n) is fully written into `dst` each pass (a pair
// with no right run copies its left run through). Reuses merge_sort_run_u32.
// ---------------------------------------------------------------------------
template <typename Compare>
struct ParallelMergeCtx {
    const uint32_t* BOLT_RESTRICT src;
    uint32_t* BOLT_RESTRICT dst;
    Compare less;
    int64_t half_span;  // rows per run  = width_blocks * block_size
    int64_t pair_span;  // rows per pair = 2 * half_span
    int64_t n;
};

template <typename Compare>
inline void parallel_argsort_merge_task(void* user_data, uint32_t start,
                                        uint32_t end,
                                        uint32_t /*thread_id*/) noexcept {
    assert(user_data != nullptr);
    assert(end >= start);
    auto* c = static_cast<ParallelMergeCtx<Compare>*>(user_data);
    for (uint32_t p = start; p < end; ++p) {
        const int64_t base = static_cast<int64_t>(p) * c->pair_span;
        const int64_t lo  = base < c->n ? base : c->n;
        const int64_t m0  = base + c->half_span;
        const int64_t mid = m0 < c->n ? m0 : c->n;
        const int64_t h0  = base + c->pair_span;
        const int64_t hi  = h0 < c->n ? h0 : c->n;
        merge_sort_run_u32(c->src, c->dst, lo, mid, hi, c->less);
    }
}

// ===========================================================================
// Public API — explicit-comparator PARALLEL permutation merge sort (uint32)
// ===========================================================================

// PARALLEL stable argsort by `less` (true iff row a sorts strictly before b).
// `perm` is filled with the sorted permutation of [0,n); the caller need NOT
// pre-initialise it to identity (the block tasks do). `scratch` is a caller
// owned uint32 buffer of n entries. `sched` provides the worker pool. Falls
// back to the single-threaded kernel for small or degenerate inputs. STABLE,
// deterministic, byte-identical to merge_sort_indirect_u32. No allocation.
template <typename Compare>
inline void parallel_merge_sort_indirect_u32(
        uint32_t* BOLT_RESTRICT perm, uint32_t* BOLT_RESTRICT scratch,
        int64_t n, Compare less, Scheduler* sched) noexcept {
    assert(perm != nullptr || n == 0);
    assert(n >= 0 && n <= static_cast<int64_t>(UINT32_MAX));

    int64_t k = (sched != nullptr) ? static_cast<int64_t>(sched->thread_count()) : 1;
    if (k > kParallelSortMaxBlocks) k = kParallelSortMaxBlocks;

    // Small / single-worker: route straight through the single-threaded kernel.
    if (n < kParallelSortMinRows || k <= 1) {
        for (int64_t i = 0; i < n; ++i) perm[i] = static_cast<uint32_t>(i);
        merge_sort_indirect_u32(perm, scratch, n, less);
        assert(n <= 1 || parallel_argsort_sorted_u32(perm, n, less));
        return;
    }
    assert(scratch != nullptr);

    const int64_t block_size = (n + k - 1) / k;
    const int64_t num_blocks = (n + block_size - 1) / block_size;

    // Phase 1: sort each block in parallel (identity init happens per slice).
    ParallelArgsortCtx<Compare> sctx{perm, scratch, less};
    sched->submit_range(&parallel_argsort_block_task<Compare>, &sctx,
                        static_cast<uint32_t>(n),
                        static_cast<uint32_t>(block_size));
    sched->wait_all();

    // Phase 2: bottom-up merge of the sorted runs, ping-ponging perm<->scratch.
    // Bounded passes: width doubles each pass, <= ceil(log2 num_blocks).
    const uint32_t* src = perm;
    uint32_t* dst = scratch;
    for (int64_t width = 1; width < num_blocks; width <<= 1) {
        const int64_t half_span = width * block_size;
        const int64_t pair_span = half_span << 1;
        const int64_t num_pairs = (num_blocks + (width << 1) - 1) / (width << 1);
        ParallelMergeCtx<Compare> mctx{src, dst, less, half_span, pair_span, n};
        sched->submit_range(&parallel_argsort_merge_task<Compare>, &mctx,
                            static_cast<uint32_t>(num_pairs), 1u);
        sched->wait_all();
        const uint32_t* tmp = src; src = dst; dst = const_cast<uint32_t*>(tmp);
    }
    if (src != perm) {  // result landed in scratch — copy back into perm.
        for (int64_t i = 0; i < n; ++i) perm[i] = src[i];
    }
    assert(n <= 1 || parallel_argsort_sorted_u32(perm, n, less));
}

// Arena overload: allocates the n-entry uint32 scratch from a caller-supplied
// bolt::Arena (zero malloc — arena bump alloc). Convenience for callers that
// already hold an Arena and don't want to manage the scratch buffer.
template <typename Compare>
inline bool parallel_merge_sort_indirect_u32_arena(
        uint32_t* BOLT_RESTRICT perm, int64_t n, Compare less,
        Scheduler* sched, Arena* arena) noexcept {
    assert(perm != nullptr || n == 0);
    assert(arena != nullptr);
    uint32_t* scratch = nullptr;
    // Real scratch is needed unless the driver takes the insertion-only fast
    // path inside merge_sort_indirect_u32 (n <= kArgsortInsertionCap).
    if (n > kArgsortInsertionCap) {
        scratch = arena->allocate_array<uint32_t>(static_cast<size_t>(n));
        if (scratch == nullptr) return false;  // OOM — runtime condition.
    }
    parallel_merge_sort_indirect_u32(perm, scratch, n, less, sched);
    return true;
}

}  // namespace kernels
}  // namespace bolt
