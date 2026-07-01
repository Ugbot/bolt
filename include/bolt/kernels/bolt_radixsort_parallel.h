// bolt_radixsort_parallel.h — PARALLEL multi-key LSD radix argsort.
//
// A parallel companion to the multi-key radix built from bolt_radixsort.h's
// bias/refine primitives. Serial indirect radix is single-core and random-
// scatter-bound, so it LOSES to parallel_merge_sort_indirect_u32 at scale
// (measured ~0.5-2 s vs ~0.16 s at 1.2M rows). This kernel wins by fanning the
// per-bucket work across the caller's Scheduler:
//
//   1. MSD-partition [0,n) into 256 buckets by the TOP byte (byte 7) of the
//      primary key keys[0] — one stable counting-sort pass. Buckets are
//      contiguous and globally ordered by that byte.
//   2. GATHER the partitioned rows' keys into contiguous per-key arrays so that,
//      within a bucket's slice, the keys are laid out 0..blen and a bucket sort
//      uses a LOCAL [0,blen) permutation. (This is REQUIRED: radix_refine_perm_u64
//      histograms its key array positionally, which only matches the scattered
//      rows when the permutation covers exactly the same index range — true for a
//      local perm over contiguous gathered keys, false for a global-row subset.)
//   3. Sort every bucket IN PARALLEL: each owns a DISJOINT slice; sort a local
//      perm by the full multi-key (radix_refine, least-significant key first —
//      the top byte of keys[0] is constant in a bucket, so that pass self-skips),
//      then map the local perm back to global rows.
//
// Result is byte-identical to the stable comparison merge: the MSD scatter and
// every per-bucket refine are stable, so the total order is lexicographic by
// keys[0]..keys[nk-1] with ties keeping original row order.
//
// keys[k] are PRE-BIASED uint64 arrays (unsigned-ascending == the desired order
// for that key incl. asc/desc — see radix_fill_biased_{i64,f64}), indexed by
// original row id, never reordered. `arena` supplies the gather/local-perm
// scratch (bump allocation — zero malloc on the hot path). `scratch` is a
// caller-owned uint32[n] buffer.
//
// Tiger Style: noexcept, no RTTI/exceptions/hot-path malloc; BOLT_RESTRICT;
// bounded (256 buckets, <=kParallelRadixMaxKeys keys); >=2 assertions/fn.
// Caveat: call from a thread that is NOT one of sched's workers (blocks on
// wait_all()).

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/kernels/bolt_radixsort.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace kernels {

inline constexpr int     kParallelRadixMaxKeys = 8;
inline constexpr int64_t kParallelRadixMinRows = 1 << 15;   // 32768

// Serial multi-key refine of the FULL perm (a permutation of [0,n), so the
// positional histogram matches the scattered rows). Used by the fallback and as
// the per-bucket primitive once keys are gathered contiguously.
inline void radix_refine_full(uint32_t* BOLT_RESTRICT perm,
                              uint32_t* BOLT_RESTRICT scratch,
                              const uint64_t* const* keys, int nk, int64_t n) noexcept {
    assert(nk >= 1 && (perm != nullptr || n == 0));
    for (int64_t i = 0; i < n; ++i) perm[i] = static_cast<uint32_t>(i);
    if (n < 2) return;
    int32_t* p  = reinterpret_cast<int32_t*>(perm);
    int32_t* ps = reinterpret_cast<int32_t*>(scratch);
    for (int k = nk - 1; k >= 0; --k) radix_refine_perm_u64(p, keys[k], ps, n);
}

// Per-bucket task context. Each bucket owns the slice [bucket_start[b],
// bucket_start[b+1]) of every array below — no task touches another's memory.
struct ParallelRadixBucketCtx {
    uint32_t*      out_perm;      // global perm, written per bucket
    const uint32_t* part_rows;    // partitioned global row ids (scratch copy)
    const uint64_t* pk;           // gathered keys, nk stacked planes of n
    uint32_t*      lperm;         // local perm scratch (per-bucket slice)
    uint32_t*      lscratch;      // local ping-pong scratch (per-bucket slice)
    const int64_t* bucket_start;  // [257]; bucket_start[256] == n
    int64_t        n;             // plane stride for pk
    int            nk;
};

inline void parallel_radix_bucket_task(void* user_data, uint32_t start,
                                       uint32_t end, uint32_t /*thread_id*/) noexcept {
    assert(user_data != nullptr && end >= start);
    auto* c = static_cast<ParallelRadixBucketCtx*>(user_data);
    for (uint32_t b = start; b < end; ++b) {
        const int64_t bs   = c->bucket_start[b];
        const int64_t blen = c->bucket_start[b + 1] - bs;
        if (blen <= 0) continue;
        int32_t* lp  = reinterpret_cast<int32_t*>(c->lperm) + bs;
        int32_t* lps = reinterpret_cast<int32_t*>(c->lscratch) + bs;
        for (int64_t j = 0; j < blen; ++j) lp[j] = static_cast<int32_t>(j);  // local identity
        // Sort the local perm by the full multi-key. pk plane k, bucket slice
        // [bs,bs+blen) is CONTIGUOUS, so the positional histogram is exact.
        for (int k = c->nk - 1; k >= 0; --k) {
            radix_refine_perm_u64(lp, c->pk + static_cast<int64_t>(k) * c->n + bs, lps, blen);
        }
        // Map local order back to global rows: out[bs+j] = part_rows[bs + lp[j]].
        for (int64_t j = 0; j < blen; ++j)
            c->out_perm[bs + j] = c->part_rows[bs + static_cast<int64_t>(lp[j])];
    }
}

// PARALLEL stable multi-key radix argsort. Fills `perm` with the sorted
// permutation of [0,n). `scratch` = caller-owned uint32[n]. `keys[0..nk)` are
// pre-biased uint64[n] arrays (keys[0] primary). `sched` = pool; `arena` = scratch
// for the gather + local perms. Byte-identical to the stable merge.
inline bool parallel_radix_argsort_multikey_u64(
        uint32_t* BOLT_RESTRICT perm, uint32_t* BOLT_RESTRICT scratch,
        const uint64_t* const* keys, int nk, int64_t n,
        Scheduler* sched, Arena* arena) noexcept {
    assert((perm != nullptr && keys != nullptr) || n == 0);
    assert(nk >= 1 && nk <= kParallelRadixMaxKeys);
    assert(n >= 0 && n <= static_cast<int64_t>(UINT32_MAX));

    int64_t k = (sched != nullptr) ? static_cast<int64_t>(sched->thread_count()) : 1;
    if (n < kParallelRadixMinRows || k <= 1 || sched == nullptr || arena == nullptr) {
        radix_refine_full(perm, scratch, keys, nk, n);
        return true;
    }
    assert(scratch != nullptr);

    // MSD partition by keys[0] top byte (shift=56): bucket sizes → exclusive
    // prefix sum bucket_start[257]; a mutable copy `offsets` drives the scatter.
    int64_t counts[kRadixBuckets] = {0};
    radix_histogram_u64(keys[0], n, 56, counts);
    int64_t bucket_start[kRadixBuckets + 1];
    int64_t offsets[kRadixBuckets];
    int64_t sum = 0, max_count = 0;
    for (int b = 0; b < kRadixBuckets; ++b) {
        bucket_start[b] = sum; offsets[b] = sum; sum += counts[b];
        if (counts[b] > max_count) max_count = counts[b];
    }
    bucket_start[kRadixBuckets] = sum;
    assert(sum == n);

    // MSD parallelism is bounded by n / max_bucket: a primary key with low top-
    // byte entropy (narrow float range — e.g. mid-prices) collapses to one giant
    // bucket that sorts SERIALLY, losing to the parallel merge. Decline (return
    // false) so the caller keeps the merge; only proceed when the split gives at
    // least ~8-way parallelism. Cheap gate — before any scatter/gather work.
    if (max_count > n / 8) return false;

    // Stable scatter of the identity perm into `scratch` by the top byte →
    // `scratch` now holds the partitioned global row ids.
    for (int64_t i = 0; i < n; ++i) perm[i] = static_cast<uint32_t>(i);
    radix_scatter_indirect(reinterpret_cast<int32_t*>(perm), keys[0],
                           reinterpret_cast<int32_t*>(scratch), n, 56, offsets);

    // Gather the partitioned rows' keys contiguously: pk[k*n + i] = keys[k][row].
    uint64_t* pk = arena->allocate_array<uint64_t>(static_cast<size_t>(nk) * static_cast<size_t>(n));
    uint32_t* lperm    = arena->allocate_array<uint32_t>(static_cast<size_t>(n));
    uint32_t* lscratch = arena->allocate_array<uint32_t>(static_cast<size_t>(n));
    if (pk == nullptr || lperm == nullptr || lscratch == nullptr) {
        radix_refine_full(perm, scratch, keys, nk, n);   // OOM → correct serial path
        return true;
    }
    for (int kk = 0; kk < nk; ++kk) {
        const uint64_t* src = keys[kk];
        uint64_t* dst = pk + static_cast<int64_t>(kk) * n;
        for (int64_t i = 0; i < n; ++i) dst[i] = src[scratch[i]];
    }

    // Sort every bucket in parallel; `scratch` is the read-only partitioned rows,
    // `perm` is written. Disjoint slices, keys shared read-only.
    ParallelRadixBucketCtx ctx{perm, scratch, pk, lperm, lscratch, bucket_start, n, nk};
    sched->submit_range(&parallel_radix_bucket_task, &ctx,
                        static_cast<uint32_t>(kRadixBuckets), 1u);
    sched->wait_all();
    return true;
}

}  // namespace kernels
}  // namespace bolt
