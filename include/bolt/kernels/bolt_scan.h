// bolt_scan.h — Scan / scatter array primitives (the substrate under CSR
// build + any "histogram → exclusive-scan → scatter" group-by layout).
//
// Three flat-array kernels, all int64_t, all caller-owns-all-buffers:
//   - exclusive_scan_i64  : exclusive prefix sum (out has n+1 slots)
//   - histogram_i64       : bucket counts (counts[keys[i]]++)
//   - scatter_i64         : out[dst_idx[i]] = src[i]   (inverse of gather)
//
// These are the array primitives chukonu's hand-rolled CSR build loop
// (degree-count -> exclusive-scan -> scatter, in
// src/graph/csr_adjacency.cpp) used to spell out inline. Lifting them here
// gives one perf surface to optimise + audit, and the CSR graph kernels in
// bolt_csr.h compose them.
//
// Shape rules (Tiger Style — matches bolt_binsearch.h exactly):
//   - every function >=2 asserts, <=70 lines
//   - all noexcept, no exceptions, no RTTI, no heap allocation
//   - POD + free functions; pointer + length, no std::span / std::vector
//   - BOLT_RESTRICT on every data pointer so the compiler knows the
//     in / out ranges do not alias (the scatter loop in particular wants
//     this — src, dst_idx and out are three disjoint caller buffers)
//   - explicit int sizes throughout (int64_t payloads, int32_t scatter
//     indices to match Bolt's selection-vector / gather index convention)
//
// ns/row floors (scalar, /O2, L1/L2-resident, client Intel):
//   - exclusive_scan_i64 : ~0.5 ns/row — one add + one store per element,
//     a strict serial dependency chain on `acc` (not vectorisable; the
//     running sum is inherently sequential). Bound by store throughput.
//   - histogram_i64      : ~0.7 ns/row on a small dense bucket set
//     (L1-resident counts), degrading toward ~3-4 ns/row once the bucket
//     array spills L2 (random-access increment). Same shape as the CSR
//     degree-count pass.
//   - scatter_i64        : ~1-2 ns/row — one indexed load of dst_idx + one
//     indexed store into out; the store is a random-access write so the
//     floor tracks the store-buffer / cache-line-fill rate, not the count.
//     (Mirror of gather_branchless's indexed-load floor, on the store side.)

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace kernels {

// ============================================================================
// exclusive_scan_i64
// ============================================================================
//
// Exclusive prefix sum. `out` has n+1 slots: out[0] = 0,
// out[i] = sum(in[0..i)), out[n] = grand total. The running sum is a
// strict serial dependency (acc threads through every iteration), so this
// is intentionally a plain scalar loop — there is no branch to predicate
// and the dependency chain defeats auto-vectorisation. Caller owns both
// buffers; `in` and `out` must not alias (BOLT_RESTRICT).
//
// Matches csr_adjacency.cpp Pass 2 byte-for-byte: offsets[k] = acc;
// acc += degree[k]; with offsets[n] = total at the end.
BOLT_FORCE_INLINE void exclusive_scan_i64(
        const int64_t* BOLT_RESTRICT in, int64_t n,
        int64_t* BOLT_RESTRICT out) noexcept {
    assert(n >= 0);
    assert(n == 0 || (in != nullptr && out != nullptr));
    assert(n != 0 || out != nullptr);  // out[0] is always written

    int64_t acc = 0;
    out[0] = 0;
    for (int64_t i = 0; i < n; ++i) {
        acc += in[i];
        assert(acc >= out[i] && "exclusive_scan_i64: counts must be non-negative");
        out[i + 1] = acc;
    }
    assert(out[0] == 0);
    assert(n == 0 || out[n] == acc);
}

// ============================================================================
// histogram_i64
// ============================================================================
//
// Bucket histogram: counts[keys[i]] += 1 for every key in [0, n_buckets).
// The caller zeroes and owns `counts` (n_buckets slots). Every key MUST be
// a dense bucket id in [0, n_buckets) — asserted on the hot path because an
// out-of-range key is a programmer error (the CSR build relies on dense
// node ids), not a data condition. Matches csr_adjacency.cpp Pass 1
// (cursor[k] += 1) exactly: a plain indexed increment, no branch.
BOLT_FORCE_INLINE void histogram_i64(
        const int64_t* BOLT_RESTRICT keys, int64_t n, int64_t n_buckets,
        int64_t* BOLT_RESTRICT counts) noexcept {
    assert(n >= 0);
    assert(n_buckets >= 0);
    assert(n == 0 || (keys != nullptr && counts != nullptr));

    for (int64_t i = 0; i < n; ++i) {
        const int64_t k = keys[i];
        assert(k >= 0 && k < n_buckets && "histogram_i64: key out of bucket range");
        counts[k] += 1;
    }
}

// ============================================================================
// scatter_i64
// ============================================================================
//
// Scatter: out[dst_idx[i]] = src[i] for i in [0, n). The exact inverse of
// gather_branchless (which does out[i] = data[indices[i]]). Caller owns
// `out`, sized to (max dst_idx) + 1. Indices are int32_t to match Bolt's
// selection-vector / gather index convention. No branch: every iteration
// is one indexed load + one indexed store. `src`, `dst_idx` and `out` are
// three disjoint caller buffers (BOLT_RESTRICT).
BOLT_FORCE_INLINE void scatter_i64(
        const int64_t* BOLT_RESTRICT src, const int32_t* BOLT_RESTRICT dst_idx,
        int64_t n, int64_t* BOLT_RESTRICT out) noexcept {
    assert(n >= 0);
    assert(n == 0 || (src != nullptr && dst_idx != nullptr && out != nullptr));

    for (int64_t i = 0; i < n; ++i) {
        const int32_t d = dst_idx[i];
        assert(d >= 0 && "scatter_i64: destination index must be non-negative");
        out[d] = src[i];
    }
}

}  // namespace kernels
}  // namespace bolt
