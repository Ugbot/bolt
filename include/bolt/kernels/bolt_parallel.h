// bolt/kernels/bolt_parallel.h — Morsel-parallel wrappers over serial kernels.
//
// Design notes (Tiger Style, Wave F+):
//   - Every function is noexcept; no exceptions, no RTTI, no smart pointers.
//   - All scratch memory comes from the caller-supplied Arena (or the
//     scheduler's per-worker arenas for the groupby variant). No heap on the
//     hot path.
//   - We cannot rely on scheduler worker-id routing: bolt's submit_range pushes
//     morsels onto a SPMC ring and every morsel's `thread_id` argument is 0.
//     Therefore the partial-buffer design here is **per-morsel**, indexed by
//     the morsel's `start / grain` offset, not per-worker. Each morsel owns a
//     disjoint slice of scratch and never contends with its peers — no
//     atomics in the inner loop.
//   - Serial fallback when sched == nullptr or thread_count() <= 1.
//
// Public surface:
//   bolt::kernels::parallel_filter_gt<T>  — returns total row count.
//   bolt::kernels::parallel_sum<T>        — widened scalar.
//   bolt::parallel_groupby_sum_int64      — BoltColumn in/out, mirrors serial.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_config.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/bolt_types.h"
#include "bolt/join/bolt_groupby.h"
#include "bolt/join/bolt_swiss.h"
#include "bolt/kernels/bolt_numeric.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace kernels {

// ---------------------------------------------------------------------------
// Scratch pad for vectorized tail writes. Matches the +L slack the SIMD
// filter kernels use internally so per-morsel slices never alias.
// ---------------------------------------------------------------------------
inline constexpr int64_t kSimdTail = 32;

// ---------------------------------------------------------------------------
// parallel_filter_gt
// ---------------------------------------------------------------------------
// Per-morsel design:
//   1. Allocate one flat `partial` int32 buffer of size n + kSimdTail * M
//      (M = number of morsels) from the caller arena. Each morsel owns the
//      slice `[start, start + grain + kSimdTail)` — the kSimdTail pad isolates
//      morsels from each other so the branchless speculative write is safe.
//   2. Allocate one `int64_t counts[M]` array.
//   3. Each morsel runs `filter_gt<T>` on its slice then re-biases the emitted
//      indices by `+ start` so indices are absolute.
//   4. Wait. Serial compaction copies indices in ascending morsel order ->
//      output ordering is deterministic and ascending.

template <typename T>
struct ParallelFilterGtCtx {
    const T*      data;
    T             scalar;
    uint32_t      grain;
    int32_t*      partial;    // size = num_morsels * (grain + kSimdTail)
    int64_t*      counts;     // size = num_morsels
};

template <typename T>
inline void parallel_filter_gt_morsel(void* user_data, uint32_t start,
                                      uint32_t end,
                                      uint32_t /*thread_id*/) noexcept {
    assert(user_data != nullptr);
    assert(end >= start);
    auto* ctx = static_cast<ParallelFilterGtCtx<T>*>(user_data);
    const uint32_t morsel_idx = start / ctx->grain;
    const int64_t  slice_cap  = static_cast<int64_t>(ctx->grain) + kSimdTail;
    int32_t* slot = ctx->partial + morsel_idx * slice_cap;

    const int64_t slice_len = static_cast<int64_t>(end - start);
    int64_t c = filter_gt<T>(ctx->data + start, slice_len, ctx->scalar, slot);

    // Re-bias indices from slice-relative to absolute.
    const int32_t bias = static_cast<int32_t>(start);
    for (int64_t i = 0; i < c; ++i) slot[i] += bias;

    ctx->counts[morsel_idx] = c;
}

template <typename T>
inline int64_t parallel_filter_gt(Scheduler*           sched,
                                  const T* BOLT_RESTRICT data,
                                  int64_t              n,
                                  T                    scalar,
                                  int32_t* BOLT_RESTRICT out,
                                  Arena*               arena) noexcept {
    assert(data != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    assert(n >= 0);

    if (sched == nullptr || sched->thread_count() <= 1 || n == 0 ||
        n < config::kParallelMinRowsFilter) {
        return filter_gt<T>(data, n, scalar, out);
    }
    assert(arena != nullptr);

    const uint32_t grain = bolt_grain_rows(sched->grain_bytes(), sizeof(T));
    const uint64_t num_morsels64 =
        (static_cast<uint64_t>(n) + grain - 1) / grain;
    if (num_morsels64 > 0xFFFFFFFFu) return -1;  // Tiger Style upper bound
    const uint32_t num_morsels = static_cast<uint32_t>(num_morsels64);

    const int64_t slice_cap = static_cast<int64_t>(grain) + kSimdTail;
    const size_t  pbytes    =
        static_cast<size_t>(num_morsels) * static_cast<size_t>(slice_cap)
      * sizeof(int32_t);
    int32_t* partial = static_cast<int32_t*>(arena->allocate(pbytes, 64));
    int64_t* counts  = arena->allocate_array<int64_t>(num_morsels);
    if (!partial || !counts) return -1;
    memset(counts, 0, num_morsels * sizeof(int64_t));

    ParallelFilterGtCtx<T> ctx{ data, scalar, grain, partial, counts };
    sched->submit_range(&parallel_filter_gt_morsel<T>, &ctx,
                        static_cast<uint32_t>(n), grain);
    sched->wait_all();

    // Serial compaction — ascending morsel order preserves ascending indices.
    int64_t total = 0;
    for (uint32_t m = 0; m < num_morsels; ++m) {
        const int32_t* src = partial + static_cast<int64_t>(m) * slice_cap;
        const int64_t c = counts[m];
        assert(c >= 0);
        if (c > 0) memcpy(out + total, src, static_cast<size_t>(c) * sizeof(int32_t));
        total += c;
    }
    return total;
}

// ---------------------------------------------------------------------------
// parallel_sum
// ---------------------------------------------------------------------------
// Per-morsel accumulator, summed serially after wait. No atomics in the inner
// loop. The caller does not supply an arena: partials live in the scheduler's
// worker-0 arena (startup-sized, reused every call). This matches the
// "scheduler owns its scratch" convention.

template <typename T>
struct ParallelSumCtx {
    const T*      data;
    uint32_t      grain;
    accum_t<T>*   partials;   // size = num_morsels
};

template <typename T>
inline void parallel_sum_morsel(void* user_data, uint32_t start, uint32_t end,
                                uint32_t /*thread_id*/) noexcept {
    assert(user_data != nullptr);
    assert(end >= start);
    auto* ctx = static_cast<ParallelSumCtx<T>*>(user_data);
    const uint32_t morsel_idx = start / ctx->grain;
    const int64_t  len        = static_cast<int64_t>(end - start);
    ctx->partials[morsel_idx] = sum<T>(ctx->data + start, len);
}

template <typename T>
inline accum_t<T> parallel_sum(Scheduler*           sched,
                               const T* BOLT_RESTRICT data,
                               int64_t              n) noexcept {
    assert(data != nullptr || n == 0);
    assert(n >= 0);

    if (sched == nullptr || sched->thread_count() <= 1 || n == 0 ||
        n < config::kParallelMinRowsSum) {
        return sum<T>(data, n);
    }

    const uint32_t grain = bolt_grain_rows(sched->grain_bytes(), sizeof(T));
    const uint64_t num_morsels64 =
        (static_cast<uint64_t>(n) + grain - 1) / grain;
    assert(num_morsels64 <= 0xFFFFFFFFu);
    const uint32_t num_morsels = static_cast<uint32_t>(num_morsels64);

    Arena* scratch = sched->worker_arena(0);
    assert(scratch != nullptr);
    accum_t<T>* partials = scratch->allocate_array<accum_t<T>>(num_morsels);
    if (!partials) return sum<T>(data, n);  // Fall back to serial on OOM.
    memset(partials, 0, num_morsels * sizeof(accum_t<T>));

    ParallelSumCtx<T> ctx{ data, grain, partials };
    sched->submit_range(&parallel_sum_morsel<T>, &ctx,
                        static_cast<uint32_t>(n), grain);
    sched->wait_all();

    accum_t<T> total = accum_t<T>{0};
    for (uint32_t m = 0; m < num_morsels; ++m) total += partials[m];
    return total;
}

}  // namespace kernels

// ---------------------------------------------------------------------------
// Override the prefetch_ahead value applied to per-morsel partials in the
// next call to parallel_groupby_agg_int64. Pass 0 to disable. Not thread-
// safe; intended for benchmark drivers that want to sweep this knob.
// In production callers should set GroupByTable::prefetch_ahead directly
// when they own the table.
// ---------------------------------------------------------------------------
inline uint16_t& parallel_groupby_prefetch_ahead_override() noexcept {
    static uint16_t value = 0;
    return value;
}

// ---------------------------------------------------------------------------
// Toggle the two-pass hoisted-probe ingest path in
// parallel_groupby_agg_int64. Default off (single-pass `ingest_unchecked`
// — best for low-cardinality workloads where the hit/miss branch is
// already well-predicted). Set to non-zero for high-cardinality cases
// where Pass A's branchless probe + Pass B's branchless accumulate-via-
// dummy-slot wins. See docs/research/branchless-hashing.md.
// ---------------------------------------------------------------------------
inline uint8_t& parallel_groupby_two_pass_override() noexcept {
    static uint8_t value = 0;
    return value;
}

// ---------------------------------------------------------------------------
// parallel_groupby_sum_int64 — DuckDB-style thread-local-aggregates + merge.
// ---------------------------------------------------------------------------
// We build one GroupByTable per morsel (not per worker; see design note above)
// and then a single-threaded merge pass walks all partials into a global
// table. For 10k rows with grain ≈ 32k this is one morsel = serial; for
// larger inputs the phase-1 cost scales with cores while phase-2 merge is
// linear in total-distinct-groups-across-partials.

struct ParallelGroupByCtx {
    const uint64_t*       keys;
    const int64_t*        values;
    uint32_t              grain;
    GroupByTable*         partials;       // size = num_morsels
    int32_t*              existing_buf;   // [num_morsels * (grain+slack)] or null
    uint32_t*             miss_idx_buf;   // [num_morsels * (grain+slack)] or null
    uint8_t               two_pass;       // 1 = use ingest_two_pass
    std::atomic<uint32_t>* error_flag;
};

inline void parallel_groupby_morsel(void* user_data, uint32_t start,
                                    uint32_t end,
                                    uint32_t /*thread_id*/) noexcept {
    assert(user_data != nullptr);
    assert(end >= start);
    auto* ctx = static_cast<ParallelGroupByCtx*>(user_data);
    const uint32_t m = start / ctx->grain;
    GroupByTable& tbl = ctx->partials[m];

    // Two-pass hoisted-probe variant (opt-in via ctx->two_pass).
    // Uses pre-allocated per-morsel scratch buffers from ctx.
    if (ctx->two_pass != 0 && ctx->existing_buf != nullptr) {
        const int64_t slice_n = static_cast<int64_t>(end - start);
        int32_t*  ex = ctx->existing_buf + static_cast<size_t>(m) * ctx->grain;
        uint32_t* mq = ctx->miss_idx_buf + static_cast<size_t>(m) * ctx->grain;
        tbl.ingest_two_pass(ctx->keys + start,
                            ctx->values + start,
                            slice_n, ex, mq);
        return;
    }

    // Default single-pass loop-peeled — main body has no per-row
    // prefetch bound check and uses ingest_unchecked (caller pre-sized
    // hint=grain so capacity can never overflow inside one morsel).
    const uint32_t pf = tbl.prefetch_ahead;
    if (pf > 0 && end > start + pf) {
        const uint32_t main_end = end - pf;
        for (uint32_t i = start; i < main_end; ++i) {
            tbl.ht.prefetch(ctx->keys[i + pf]);
            tbl.ingest_unchecked(ctx->keys[i], ctx->values[i]);
        }
        for (uint32_t i = main_end; i < end; ++i) {
            tbl.ingest_unchecked(ctx->keys[i], ctx->values[i]);
        }
    } else {
        for (uint32_t i = start; i < end; ++i) {
            tbl.ingest_unchecked(ctx->keys[i], ctx->values[i]);
        }
    }
    (void)ctx;
}

// ---------------------------------------------------------------------------
// Radix-partitioned parallel merge (DuckDB pattern).
// ---------------------------------------------------------------------------
// Phase 2 in three sub-phases:
//   2a. Scatter — walk each per-morsel partial; route each (key, sum, count)
//       triple into one of P shard buffers via a high-bit slice of swiss_mix.
//       Shards are disjoint in key-space so Phase 2b is embarrassingly parallel.
//   2b. Partition merge — for each partition p in [0, P), one worker builds a
//       final SwissTable and upserts all triples routed to shard p.
//   2c. Emit — serial walk over P partitions appending (key, sum) rows.
//
// Memory: an auxiliary counting pass sizes each shard exactly (no slack), so
// the scatter buffer totals `total_partial_groups * sizeof(MergeTriple)` bytes
// across all shards — no multiplicative blowup over the old serial merge, which
// also allocated a global table of this order.
// ---------------------------------------------------------------------------

inline constexpr uint32_t kGroupbyMergeRadixMax = 64;  // hard upper bound on P
static_assert((kGroupbyMergeRadixMax & (kGroupbyMergeRadixMax - 1)) == 0,
              "kGroupbyMergeRadixMax must be a power of two");

// 64 bytes (cache-line aligned). Tried 40-byte naturally-aligned variant
// during Wave N; measured slower. Likely the radix-merge consumer prefers
// each triple in its own cache line for prefetcher behaviour.
struct alignas(64) MergeTriple {
    uint64_t key;
    int64_t  sum;
    int64_t  count;
    int64_t  min;
    int64_t  max;
    uint64_t pad[3];
};
static_assert(sizeof(MergeTriple) == 64, "MergeTriple must fit one cache line");

// Choose radix partition count: next-pow-2 >= threads, clamped to [1, P_MAX].
BOLT_FORCE_INLINE uint32_t groupby_merge_choose_p(uint32_t threads) noexcept {
    assert(threads > 0);
    uint32_t p = 1;
    while (p < threads && p < kGroupbyMergeRadixMax) p <<= 1;
    assert(p >= 1);
    assert(p <= kGroupbyMergeRadixMax);
    assert((p & (p - 1)) == 0);
    return p;
}

// Route a key to one of P shards using the high bits of swiss_mix.
// High bits to keep routing decorrelated from SwissTable's low-bit index mask.
BOLT_FORCE_INLINE uint32_t groupby_merge_shard(uint64_t key, uint32_t p_mask) noexcept {
    assert((p_mask & (p_mask + 1)) == 0);  // p_mask == P - 1, power-of-two mask
    const uint64_t h = swiss_mix(key);
    return static_cast<uint32_t>(h >> 32) & p_mask;
}

// Scatter-phase context. One shard_bufs[s] buffer per shard, each pre-sized
// exactly via shard_counts[s] from the counting pre-pass. shard_heads[s] is an
// atomic claim index used by workers to carve out contiguous write slots.
struct ParallelGroupByScatterCtx {
    const GroupByTable*    partials;
    uint32_t               num_morsels;
    uint32_t               p_mask;        // P - 1
    MergeTriple**          shard_bufs;    // size = P
    std::atomic<uint32_t>* shard_heads;   // size = P; claim cursors
};

// Scatter one partial (one "morsel" in [start, end) with end == start + 1).
inline void parallel_groupby_scatter_morsel(void* user_data, uint32_t start,
                                            uint32_t end,
                                            uint32_t /*thread_id*/) noexcept {
    assert(user_data != nullptr);
    assert(end >= start);
    auto* ctx = static_cast<ParallelGroupByScatterCtx*>(user_data);
    for (uint32_t m = start; m < end; ++m) {
        assert(m < ctx->num_morsels);
        const GroupByTable& p = ctx->partials[m];
        for (uint32_t g = 0; g < p.num_groups; ++g) {
            const uint64_t k = p.group_keys[g];
            const uint32_t s = groupby_merge_shard(k, ctx->p_mask);
            const uint32_t slot = ctx->shard_heads[s].fetch_add(
                1u, std::memory_order_relaxed);
            MergeTriple& t = ctx->shard_bufs[s][slot];
            t.key   = k;
            t.sum   = p.payload[g].sum;
            t.count = p.payload[g].count;
            t.min   = p.payload[g].min;
            t.max   = p.payload[g].max;
        }
    }
}

// Partition-merge context. One final GroupByTable per shard.
struct ParallelGroupByMergeCtx {
    const MergeTriple* const* shard_bufs;      // size = P
    const uint32_t*           shard_sizes;     // size = P
    GroupByTable*             shard_finals;    // size = P
    std::atomic<uint32_t>*    error_flag;
};

// Merge one shard's triples into its final table. Range = one shard per task.
inline void parallel_groupby_merge_shard(void* user_data, uint32_t start,
                                         uint32_t end,
                                         uint32_t /*thread_id*/) noexcept {
    assert(user_data != nullptr);
    assert(end >= start);
    auto* ctx = static_cast<ParallelGroupByMergeCtx*>(user_data);
    for (uint32_t s = start; s < end; ++s) {
        GroupByTable& dst = ctx->shard_finals[s];
        const MergeTriple* src = ctx->shard_bufs[s];
        const uint32_t n = ctx->shard_sizes[s];
        for (uint32_t i = 0; i < n; ++i) {
            const uint64_t k = src[i].key;
            const int32_t existing = dst.ht.find(k);
            if (existing >= 0) {
                GroupAggregate& g = dst.payload[existing];
                g.sum   += src[i].sum;
                g.count += src[i].count;
                if (src[i].min < g.min) g.min = src[i].min;
                if (src[i].max > g.max) g.max = src[i].max;
                continue;
            }
            if (dst.num_groups >= dst.capacity) {
                ctx->error_flag->store(1, std::memory_order_relaxed);
                return;
            }
            const uint32_t gid = dst.num_groups++;
            if (!dst.ht.insert(k, gid)) {
                --dst.num_groups;
                ctx->error_flag->store(1, std::memory_order_relaxed);
                return;
            }
            GroupAggregate& g = dst.payload[gid];
            g.sum   = src[i].sum;
            g.count = src[i].count;
            g.min   = src[i].min;
            g.max   = src[i].max;
            dst.group_keys[gid] = k;
        }
    }
}

// Radix-partitioned parallel merge: Phase 2a scatter + 2b partition-merge.
// Returns false on OOM or capacity overflow. On success, shard_finals[0..P)
// hold disjoint key→(sum,count) aggregates and `total_groups_out` is the sum.
inline bool parallel_groupby_radix_merge(Scheduler* sched,
                                         const GroupByTable* partials,
                                         uint32_t num_morsels,
                                         uint32_t P,
                                         Arena* arena,
                                         GroupByTable* shard_finals,
                                         uint32_t* total_groups_out) noexcept {
    assert(sched != nullptr);
    assert(partials != nullptr);
    assert(shard_finals != nullptr);
    assert(total_groups_out != nullptr);
    assert(P >= 1 && P <= kGroupbyMergeRadixMax);
    assert((P & (P - 1)) == 0);

    const uint32_t p_mask = P - 1;

    // ---- Counting pre-pass: exact shard sizes (no slack) -----------------
    uint32_t* shard_sizes = arena->allocate_array<uint32_t>(P);
    if (!shard_sizes) return false;
    memset(shard_sizes, 0, P * sizeof(uint32_t));
    uint64_t total_triples = 0;
    for (uint32_t m = 0; m < num_morsels; ++m) {
        const GroupByTable& p = partials[m];
        for (uint32_t g = 0; g < p.num_groups; ++g) {
            const uint32_t s = groupby_merge_shard(p.group_keys[g], p_mask);
            ++shard_sizes[s];
        }
        total_triples += p.num_groups;
    }

    // ---- Allocate per-shard buffers + atomic head cursors ----------------
    MergeTriple** shard_bufs = arena->allocate_array<MergeTriple*>(P);
    std::atomic<uint32_t>* shard_heads =
        arena->allocate_array<std::atomic<uint32_t>>(P);
    if (!shard_bufs || !shard_heads) return false;
    for (uint32_t s = 0; s < P; ++s) {
        if (shard_sizes[s] == 0) {
            shard_bufs[s] = nullptr;
        } else {
            shard_bufs[s] = arena->allocate_array<MergeTriple>(shard_sizes[s]);
            if (!shard_bufs[s]) return false;
        }
        shard_heads[s].store(0, std::memory_order_relaxed);
    }

    // ---- Phase 2a: parallel scatter --------------------------------------
    ParallelGroupByScatterCtx sctx{
        partials, num_morsels, p_mask, shard_bufs, shard_heads
    };
    sched->submit_range(&parallel_groupby_scatter_morsel, &sctx,
                        num_morsels, /*grain=*/1);
    sched->wait_all();

    // Sanity: every shard filled to its counted size.
    for (uint32_t s = 0; s < P; ++s) {
        assert(shard_heads[s].load(std::memory_order_relaxed) == shard_sizes[s]);
    }

    // ---- Build per-shard final tables ------------------------------------
    // Capacity hint = shard_sizes[s]; worst case every triple is a new key.
    for (uint32_t s = 0; s < P; ++s) {
        uint64_t hint = shard_sizes[s];
        if (hint == 0) hint = 1;
        if (!GroupByTable::create(&shard_finals[s], hint, arena)) return false;
    }

    // ---- Phase 2b: parallel partition merge ------------------------------
    std::atomic<uint32_t> merr{0};
    ParallelGroupByMergeCtx mctx{
        shard_bufs, shard_sizes, shard_finals, &merr
    };
    sched->submit_range(&parallel_groupby_merge_shard, &mctx,
                        P, /*grain=*/1);
    sched->wait_all();
    if (merr.load(std::memory_order_relaxed) != 0) return false;

    uint32_t total = 0;
    for (uint32_t s = 0; s < P; ++s) total += shard_finals[s].num_groups;
    *total_groups_out = total;
    (void)total_triples;
    return true;
}

inline bool parallel_groupby_agg_int64(Scheduler*         sched,
                                       const BoltColumn&  keys,
                                       const BoltColumn&  values,
                                       Arena*             arena,
                                       BoltColumn*        out_keys,
                                       BoltColumn*        out_sums,
                                       BoltColumn*        out_counts,
                                       BoltColumn*        out_mins,
                                       BoltColumn*        out_maxes) noexcept {
    assert(arena    != nullptr);
    assert(out_keys != nullptr);
    assert(out_sums != nullptr);

    // Serial fallback — also covers Constant-column fast path transparently.
    if (sched == nullptr || sched->thread_count() <= 1 ||
        keys.format == ColumnFormat::Constant ||
        keys.length < config::kParallelMinRowsGroupby) {
        return groupby_agg_int64(keys, values, arena, out_keys, out_sums,
                                 out_counts, out_mins, out_maxes);
    }
    if (keys.length != values.length) return false;
    if (values.type != BoltType::Int64) return false;
    if (keys.format != ColumnFormat::Flat && keys.format != ColumnFormat::View)
        return false;
    if (keys.type != BoltType::Int64 && keys.type != BoltType::UInt64)
        return false;
    if (values.format != ColumnFormat::Flat && values.format != ColumnFormat::View)
        return false;

    const int64_t n = keys.length;
    if (n == 0) {
        return groupby_agg_int64(keys, values, arena, out_keys, out_sums,
                                 out_counts, out_mins, out_maxes);
    }

    const uint32_t grain = bolt_grain_rows(sched->grain_bytes(), sizeof(int64_t));
    const uint64_t num_morsels64 =
        (static_cast<uint64_t>(n) + grain - 1) / grain;
    if (num_morsels64 == 0 || num_morsels64 > 0xFFFFu) {
        // Single-morsel or pathological fan-out: delegate to serial.
        return groupby_agg_int64(keys, values, arena, out_keys, out_sums,
                                 out_counts, out_mins, out_maxes);
    }
    const uint32_t num_morsels = static_cast<uint32_t>(num_morsels64);

    GroupByTable* partials = arena->allocate_array<GroupByTable>(num_morsels);
    if (!partials) return false;

    // Size hint per morsel — worst case every row is a distinct key in a
    // single morsel. Bound by grain to avoid over-allocation.
    const uint16_t pf_override = parallel_groupby_prefetch_ahead_override();
    for (uint32_t m = 0; m < num_morsels; ++m) {
        uint64_t hint = grain;
        if (!GroupByTable::create(&partials[m], hint, arena)) return false;
        if (pf_override != 0) partials[m].prefetch_ahead = pf_override;
    }

    // Optional two-pass scratch buffers — only allocated when the
    // override is set. Keeps memory footprint identical to the
    // single-pass default when the experiment isn't requested.
    const uint8_t two_pass = parallel_groupby_two_pass_override();
    int32_t*  existing_buf = nullptr;
    uint32_t* miss_idx_buf = nullptr;
    if (two_pass != 0) {
        const size_t scratch_n =
            static_cast<size_t>(num_morsels) * static_cast<size_t>(grain);
        existing_buf = arena->allocate_array<int32_t>(scratch_n);
        miss_idx_buf = arena->allocate_array<uint32_t>(scratch_n);
        if (!existing_buf || !miss_idx_buf) return false;
    }

    std::atomic<uint32_t> err{0};
    ParallelGroupByCtx ctx{
        static_cast<const uint64_t*>(keys.data),
        static_cast<const int64_t*>(values.data),
        grain,
        partials,
        existing_buf,
        miss_idx_buf,
        two_pass,
        &err,
    };
    sched->submit_range(&parallel_groupby_morsel, &ctx,
                        static_cast<uint32_t>(n), grain);
    sched->wait_all();
    if (err.load(std::memory_order_relaxed) != 0) return false;

    // ---- Cardinality-adaptive Phase 2 ---------------------------------
    // For low-cardinality inputs (1BRC: 413 stations × W workers ≈ 3300
    // total triples) the radix scatter + atomic-CAS shard-claim costs
    // more than a single-thread walk over the partials. Sum the partials'
    // num_groups; below the threshold, do a serial merge into one global
    // table. Above, fall through to the radix path.
    uint64_t total_partial_groups = 0;
    for (uint32_t m = 0; m < num_morsels; ++m) {
        total_partial_groups += partials[m].num_groups;
    }
    if (total_partial_groups <= config::kGroupbySerialMergeThreshold) {
        GroupByTable global;
        // Worst case: all partial groups are distinct → cap by sum.
        if (!GroupByTable::create(&global, total_partial_groups + 1, arena))
            return false;
        for (uint32_t m = 0; m < num_morsels; ++m) {
            const GroupByTable& p = partials[m];
            for (uint32_t g = 0; g < p.num_groups; ++g) {
                uint64_t k = p.group_keys[g];
                int32_t  ex = global.ht.find(k);
                if (ex >= 0) {
                    GroupAggregate& dst = global.payload[ex];
                    const GroupAggregate& src = p.payload[g];
                    dst.sum   += src.sum;
                    dst.count += src.count;
                    if (src.min < dst.min) dst.min = src.min;
                    if (src.max > dst.max) dst.max = src.max;
                } else {
                    if (global.num_groups >= global.capacity) return false;
                    uint32_t gid = global.num_groups++;
                    if (!global.ht.insert(k, gid)) {
                        --global.num_groups;
                        return false;
                    }
                    global.payload[gid]    = p.payload[g];
                    global.group_keys[gid] = k;
                }
            }
        }
        const int64_t out_n = static_cast<int64_t>(global.num_groups);
        if (!gb_alloc_flat_i64(out_keys, out_n, arena)) return false;
        if (!gb_alloc_flat_i64(out_sums, out_n, arena)) return false;
        if (out_counts && !gb_alloc_flat_i64(out_counts, out_n, arena)) return false;
        if (out_mins   && !gb_alloc_flat_i64(out_mins,   out_n, arena)) return false;
        if (out_maxes  && !gb_alloc_flat_i64(out_maxes,  out_n, arena)) return false;
        int64_t* ok = static_cast<int64_t*>(out_keys->data);
        int64_t* os = static_cast<int64_t*>(out_sums->data);
        int64_t* oc = out_counts ? static_cast<int64_t*>(out_counts->data) : nullptr;
        int64_t* on = out_mins   ? static_cast<int64_t*>(out_mins->data)   : nullptr;
        int64_t* ox = out_maxes  ? static_cast<int64_t*>(out_maxes->data)  : nullptr;
        for (int64_t i = 0; i < out_n; ++i) {
            ok[i] = static_cast<int64_t>(global.group_keys[i]);
            const GroupAggregate& g = global.payload[i];
            os[i] = g.sum;
            if (oc) oc[i] = g.count;
            if (on) on[i] = g.min;
            if (ox) ox[i] = g.max;
        }
        return true;
    }

    // ---- Radix-partitioned parallel merge (Phase 2a/2b) -----------------
    // Choose P = next-pow-2 >= thread_count, capped at kGroupbyMergeRadixMax.
    // With at least one partition per worker, Phase 2b is embarrassingly
    // parallel: each worker owns one shard and sees no cross-thread writes.
    const uint32_t P = groupby_merge_choose_p(sched->thread_count());
    assert(P >= 1 && P <= kGroupbyMergeRadixMax);

    GroupByTable* shard_finals = arena->allocate_array<GroupByTable>(P);
    if (!shard_finals) return false;
    uint32_t total_groups = 0;
    if (!parallel_groupby_radix_merge(sched, partials, num_morsels, P, arena,
                                      shard_finals, &total_groups)) {
        return false;
    }

    // ---- Phase 3: emit output columns (serial walk across shards) -------
    // NOTE: output row order now reflects shard order, then within-shard
    // insertion order. This differs from the previous single-global-table
    // walk, but existing tests compare by {key -> sum} map (see
    // test_bolt_parallel.cpp::to_map), so this change is transparent.
    const int64_t out_n = static_cast<int64_t>(total_groups);
    if (!gb_alloc_flat_i64(out_keys, out_n, arena)) return false;
    if (!gb_alloc_flat_i64(out_sums, out_n, arena)) return false;
    if (out_counts && !gb_alloc_flat_i64(out_counts, out_n, arena)) return false;
    if (out_mins   && !gb_alloc_flat_i64(out_mins,   out_n, arena)) return false;
    if (out_maxes  && !gb_alloc_flat_i64(out_maxes,  out_n, arena)) return false;
    int64_t* ok = static_cast<int64_t*>(out_keys->data);
    int64_t* os = static_cast<int64_t*>(out_sums->data);
    int64_t* oc = out_counts ? static_cast<int64_t*>(out_counts->data) : nullptr;
    int64_t* on = out_mins   ? static_cast<int64_t*>(out_mins->data)   : nullptr;
    int64_t* ox = out_maxes  ? static_cast<int64_t*>(out_maxes->data)  : nullptr;
    int64_t w = 0;
    for (uint32_t s = 0; s < P; ++s) {
        const GroupByTable& f = shard_finals[s];
        for (uint32_t i = 0; i < f.num_groups; ++i) {
            assert(w < out_n);
            ok[w] = static_cast<int64_t>(f.group_keys[i]);
            const GroupAggregate& g = f.payload[i];
            os[w] = g.sum;
            if (oc) oc[w] = g.count;
            if (on) on[w] = g.min;
            if (ox) ox[w] = g.max;
            ++w;
        }
    }
    assert(w == out_n);
    return true;
}

// ---------------------------------------------------------------------------
// v0.1 sum-only entry point — thin shim over parallel_groupby_agg_int64.
// ---------------------------------------------------------------------------
inline bool parallel_groupby_sum_int64(Scheduler*         sched,
                                       const BoltColumn&  keys,
                                       const BoltColumn&  values,
                                       Arena*             arena,
                                       BoltColumn*        out_keys,
                                       BoltColumn*        out_sums) noexcept {
    return parallel_groupby_agg_int64(sched, keys, values, arena,
                                      out_keys, out_sums,
                                      nullptr, nullptr, nullptr);
}

}  // namespace bolt
