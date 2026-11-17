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
//   monotonic non-decreasing counts (a count that differs from the running
//   one only by float noise takes the running one, whichever way it moved:
//   upstream's util/almost.Equal, relative to the two counts), locate the
//   rank bucket, and linearly interpolate within it.
//
// fused_muladd (PromqlNumerics::fused_muladd; default off): an arm64 Go build
// contracts two multiply-adds of bucketQuantile into one fma each, the rank
// inside the bucket (phi * total - previous count) and the interpolation
// (start + width * fraction). On, the kernel does the same and answers that
// build bit for bit (checked against Prometheus 3.5.0 darwin/arm64 over 400
// random histograms x 13 quantiles: 5200 answers equal fused, 802 of them
// differ unfused). Off is an amd64 build's arithmetic.
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

// Contraction is chosen by PromqlNumerics::fused_muladd, never by the compiler:
// an implicit fma would make answers depend on the CPU (MSVC never contracts
// under /fp:precise).
#if defined(__clang__)
#pragma float_control(push)
#pragma clang fp contract(off)
#endif

namespace bolt {
namespace promql {

// One classic-histogram bucket: an `le` upper bound and its cumulative count.
struct HistBucket {
    double upper;
    double count;
};

// Default tolerance for treating a tiny count difference as float noise rather
// than a real step (Prometheus smallDeltaTolerance = 1e-12). It is RELATIVE:
// the difference is held against the sum of the two counts' magnitudes.
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

// Prometheus util/almost.Equal(a, b, epsilon): are two counts apart only by
// float noise. Two NaNs are equal. Beside a zero (or when the magnitudes sum
// to a subnormal) the difference is held against epsilon times the smallest
// normal double; otherwise against the sum of the magnitudes.
inline bool hist_almost_equal(double a, double b, double epsilon) noexcept {
    assert(epsilon >= 0.0);
    assert(epsilon < 1.0);
    constexpr double k_min_normal = 2.2250738585072014e-308;
    if (std::isnan(a) && std::isnan(b)) return true;
    if (a == b) return true;
    const double abs_sum = std::fabs(a) + std::fabs(b);
    const double diff = std::fabs(a - b);
    if (a == 0.0 || b == 0.0 || abs_sum < k_min_normal) {
        return diff < epsilon * k_min_normal;
    }
    const double max = std::numeric_limits<double>::max();
    return diff / (abs_sum < max ? abs_sum : max) < epsilon;
}

// Force counts monotonic non-decreasing (upstream
// ensureMonotonicAndIgnoreSmallDeltas): a count within `tolerance` of the
// running one (almost.Equal, either direction) or below it takes the running
// one; only a real increase moves the running count.
inline void hist_ensure_monotonic(HistBucket* BOLT_RESTRICT b, int64_t n,
                                  double tolerance) noexcept {
    assert(b != nullptr || n == 0);
    assert(tolerance >= 0.0);
    if (n == 0) return;
    double prev = b[0].count;
    for (int64_t i = 1; i < n; ++i) {          // bounded by n
        const double curr = b[i].count;
        if (curr == prev) continue;
        if (hist_almost_equal(prev, curr, tolerance) || curr < prev) {
            b[i].count = prev;                 // noise, or a dip: not a step
            continue;
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
                                            kPromqlSmallDeltaTolerance,
                                        bool fused_muladd = false) noexcept {
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
    const double rank = phi * observations;

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
    double in_bucket = rank;                   // the rank inside bucket b
    if (b > 0) {
        const double below = scratch[b - 1].count;
        bucket_start = scratch[b - 1].upper;
        count -= below;
        in_bucket = fused_muladd ? std::fma(phi, observations, -below)
                                 : rank - below;
    }
    assert(b < m - 1);                         // 0 / 0 is NaN here, as upstream
    const double width = bucket_end - bucket_start;
    const double fraction = in_bucket / count;
    if (fused_muladd) return std::fma(width, fraction, bucket_start);
    const double step = width * fraction;
    return bucket_start + step;
}

}  // namespace promql
}  // namespace bolt

#if defined(__clang__)
#pragma float_control(pop)
#endif
