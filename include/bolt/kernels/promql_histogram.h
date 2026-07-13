// bolt/kernels/promql_histogram.h — Prometheus classic histogram_quantile.
//
//   promql_histogram_quantile(phi, buckets) — the phi-quantile of a classic
//   `le`-bucketed histogram, by linear interpolation within the bucket that
//   contains the phi*total rank.
//
// Input shape (DIFFERENT from the rate/deriv range kernels — this is a
// cross-bucket op at ONE timestamp, not a time window):
//   le_upper[]   — the `le` upper bounds of each bucket (the +Inf bucket's
//                  bound is +infinity). MUST NOT be pre-sorted by the caller;
//                  this kernel sorts a private working copy.
//   cum_count[]  — the CUMULATIVE observation count at each `le` (Prometheus
//                  histogram buckets are cumulative).
//   n_buckets    — number of (le, count) pairs (>= 0).
//   phi          — the quantile in [0,1].
//   scratch      — caller storage of >= n_buckets HistBucket (working copy;
//                  the kernel never allocates).
//   scratch_cap  — capacity of `scratch`.
//
// Semantics replicate Prometheus exactly (promql/quantile.go ::
// bucketQuantile + coalesceBuckets + ensureMonotonicAndIgnoreSmallDeltas):
//   phi is NaN                    -> NaN
//   phi < 0                       -> -Inf
//   phi > 1                       -> +Inf
//   highest bucket is not +Inf    -> NaN
//   fewer than 2 buckets          -> NaN
//   zero total observations       -> NaN
//   otherwise: sort by le, coalesce equal-le buckets (sum counts), clamp to
//   monotonic non-decreasing counts (ignoring sub-tolerance float noise),
//   locate the rank bucket, and linearly interpolate within it.
//
// Oracle: chukonu/tests/promql/testdata/upstream/histograms.test (READ-ONLY).
// Worked cases pinned in tests/test_promql_kernels.cpp.
//
// Tiger Style: noexcept, >= 2 asserts/fn, fixed-bound loops, no allocation
// (caller-provided scratch), explicit integer sizes, functions <= 70 lines.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>
#include <cmath>
#include <limits>

namespace bolt {
namespace promql {

// One classic-histogram bucket: an `le` upper bound and its cumulative count.
struct HistBucket {
    double upper;
    double count;
};

// Default tolerance for treating a tiny count decrease as float noise rather
// than a real non-monotonicity (Prometheus smallDeltaTolerance = 1e-12).
inline constexpr double kPromqlSmallDeltaTolerance = 1e-12;

namespace detail {

// Insertion sort scratch[0..n) ascending by `upper`. n is small (bucket
// counts are tiny), so O(n^2) is optimal and branch-predictable here.
inline void hist_sort_by_upper(HistBucket* BOLT_RESTRICT b, int64_t n) noexcept {
    assert(b != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 1; i < n; ++i) {          // bounded by n
        const HistBucket key = b[i];
        int64_t j = i - 1;
        while (j >= 0 && b[j].upper > key.upper) {  // bounded by i
            b[j + 1] = b[j];
            --j;
        }
        b[j + 1] = key;
    }
}

// Merge adjacent equal-`upper` buckets by summing counts. Returns the new
// (compacted) length. Requires b sorted ascending by upper.
inline int64_t hist_coalesce(HistBucket* BOLT_RESTRICT b, int64_t n) noexcept {
    assert(b != nullptr || n == 0);
    assert(n >= 0);
    if (n == 0) return 0;
    int64_t w = 0;
    for (int64_t i = 1; i < n; ++i) {          // bounded by n
        if (b[i].upper == b[w].upper) {
            b[w].count += b[i].count;
        } else {
            ++w;
            b[w] = b[i];
        }
    }
    const int64_t len = w + 1;
    assert(len >= 1 && len <= n);
    return len;
}

// Force counts monotonic non-decreasing, ignoring sub-tolerance decreases.
inline void hist_ensure_monotonic(HistBucket* BOLT_RESTRICT b, int64_t n,
                                  double tolerance) noexcept {
    assert(b != nullptr || n == 0);
    assert(tolerance >= 0.0);
    if (n == 0) return;
    double prev = b[0].count;
    for (int64_t i = 1; i < n; ++i) {          // bounded by n
        double curr = b[i].count;
        if (curr != prev && std::fabs(curr - prev) < tolerance) {
            b[i].count = prev;                 // numerically insignificant
            curr = prev;
        }
        if (curr < prev) {
            b[i].count = prev;                 // force monotonic
            curr = prev;
        }
        prev = curr;
    }
}

}  // namespace detail

// histogram_quantile(phi, buckets). See header comment for the full contract.
inline double promql_histogram_quantile(double phi,
                                        const double* BOLT_RESTRICT le_upper,
                                        const double* BOLT_RESTRICT cum_count,
                                        int64_t n_buckets,
                                        HistBucket* BOLT_RESTRICT scratch,
                                        int64_t scratch_cap,
                                        double tolerance =
                                            kPromqlSmallDeltaTolerance) noexcept {
    assert(le_upper != nullptr || n_buckets == 0);
    assert(cum_count != nullptr || n_buckets == 0);
    assert(scratch != nullptr || n_buckets == 0);
    assert(scratch_cap >= n_buckets);
    assert(n_buckets >= 0);
    (void)scratch_cap;  // consumed only by the assert above (NDEBUG-stripped)

    if (std::isnan(phi)) return std::numeric_limits<double>::quiet_NaN();
    if (phi < 0.0) return -std::numeric_limits<double>::infinity();
    if (phi > 1.0) return std::numeric_limits<double>::infinity();

    for (int64_t i = 0; i < n_buckets; ++i) {  // bounded by n_buckets
        scratch[i].upper = le_upper[i];
        scratch[i].count = cum_count[i];
    }
    detail::hist_sort_by_upper(scratch, n_buckets);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (n_buckets == 0) return nan;
    // The highest bucket must be the +Inf catch-all, else the data is invalid.
    if (!(std::isinf(scratch[n_buckets - 1].upper) &&
          scratch[n_buckets - 1].upper > 0.0)) {
        return nan;
    }

    const int64_t m = detail::hist_coalesce(scratch, n_buckets);
    detail::hist_ensure_monotonic(scratch, m, tolerance);
    if (m < 2) return nan;

    const double observations = scratch[m - 1].count;
    if (observations == 0.0) return nan;
    double rank = phi * observations;

    // First bucket (excluding +Inf) whose cumulative count >= rank.
    int64_t b = 0;
    while (b < m - 1 && scratch[b].count < rank) {  // bounded by m
        ++b;
    }
    assert(b >= 0 && b <= m - 1);

    if (b == m - 1) return scratch[m - 2].upper;
    if (b == 0 && scratch[0].upper <= 0.0) return scratch[0].upper;

    double bucket_start = 0.0;
    const double bucket_end = scratch[b].upper;
    double count = scratch[b].count;
    if (b > 0) {
        bucket_start = scratch[b - 1].upper;
        count -= scratch[b - 1].count;
        rank -= scratch[b - 1].count;
    }
    assert(count != 0.0);
    return bucket_start + (bucket_end - bucket_start) * (rank / count);
}

}  // namespace promql
}  // namespace bolt
