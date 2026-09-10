// bolt/kernels/promql_native_histogram.h — Prometheus NATIVE histograms.
//
// A native histogram is a SINGLE sample whose value is a sparse bucket
// distribution, not a float: schema (resolution), a zero bucket, and
// positive/negative bucket runs. Prometheus 2.40+ carries them alongside the
// classic `_bucket`/`_count`/`_sum` triple that promql_histogram.h serves.
// Schema -53 is the "custom buckets" (NHCB) variant, whose bounds come from an
// explicit `custom_values` list rather than from a power-of-two ladder — that
// is what Prometheus synthesises from a classic histogram, so an NHCB answer
// must agree with the classic answer bucket-for-bucket.
//
// REPRESENTATION (deliberately DENSE, not spans). Prometheus stores buckets as
// (span, delta) runs because a scrape carries thousands of them over the wire.
// Here the whole histogram is one arena-resident POD with a dense count array
// per side plus the index of its first bucket, because every consumer in this
// tree materialises all buckets anyway (quantile, fraction and variance each
// walk the complete ascending sequence) and a dense array makes that walk a
// straight scan with no span bookkeeping. The cost is bounded by
// k_nh_max_side; a histogram wider than that is REFUSED at parse, never
// truncated — a truncated histogram answers confidently and wrongly.
//
// BUCKET INDEXING, pinned twice over:
//   idx 0 is always the bucket whose UPPER bound is 1, idx 1 the next to the
//   right, idx -1 the one to the left. That is `getBoundExponential(idx,
//   schema)` in prometheus/model/histogram/generic.go, and it is also stated
//   in prose by the corpus itself (native_histograms.test:111-113). Both
//   agree, and `histogram_quantile(0.5, {{schema:0 sum:5 count:4
//   buckets:[1 2 1]}})` = sqrt(2) only under this reading (the off-by-one
//   reading yields 2.828).
//
// SEMANTICS are a faithful port of, not an interpretation of:
//   promql/quantile.go        HistogramQuantile, HistogramFraction
//   promql/functions.go       histogramVariance (histogram_stddev/_stdvar)
//   model/histogram/generic.go getBound, getBoundExponential, FractionBelow
//   model/histogram/float_histogram.go allFloatBucketIterator.Next
// The quantile's forward/REVERSE iterator split (q < 0.5 vs q >= 0.5) is part
// of the contract, not an optimisation: it changes which floating-point
// subtraction forms the in-bucket rank and therefore the last digits of the
// answer the corpus pins.
//
// Oracle: chukonu/tests/promql/testdata/upstream/native_histograms.test and
// histograms.test (READ-ONLY, Prometheus's own). Worked cases pinned in
// bolt/tests/test_promql_native_histogram.cpp.
//
// Tiger Style: noexcept, >= 2 asserts/fn, fixed-bound loops, no allocation
// (caller-provided scratch), explicit integer sizes, functions <= 70 lines.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace bolt {
namespace promql {

// Dense bucket counts held per side. 128 is 2.3x the widest run in the whole
// Prometheus corpus (55 positive / 47 negative), so it is a real ceiling with
// headroom rather than a number chosen to fit the tests.
inline constexpr int32_t k_nh_max_side   = 128;
inline constexpr int32_t k_nh_max_custom = 64;
// Negatives + the zero bucket + positives.
inline constexpr int32_t k_nh_max_all_buckets = 2 * k_nh_max_side + 1;

// Prometheus's sentinel schema for "custom buckets" (NHCB).
inline constexpr int32_t k_nh_schema_custom = -53;
// Exponential schemas Prometheus admits.
inline constexpr int32_t k_nh_schema_min = -4;
inline constexpr int32_t k_nh_schema_max = 8;

// Counter-reset hint carried with the sample. Only `Gauge` changes arithmetic
// (a gauge histogram is never reset-corrected); the rest are informational.
enum class NhResetHint : uint8_t {
    Unknown      = 0,
    CounterReset = 1,
    NotReset     = 2,
    Gauge        = 3,
};

// One native-histogram sample. POD, arena-resident, no ownership.
struct NativeHistogram {
    int32_t schema;      // -4..8 exponential, or k_nh_schema_custom
    int32_t pos_offset;  // bucket index of pos[0]
    int32_t n_pos;
    int32_t neg_offset;  // bucket index of neg[0]
    int32_t n_neg;
    int32_t n_custom;    // custom_values length (schema == custom only)
    uint8_t reset_hint;  // NhResetHint
    uint8_t _pad[7];
    double  zero_threshold;
    double  zero_count;
    double  count;
    double  sum;
    double  pos[k_nh_max_side];
    double  neg[k_nh_max_side];
    double  custom[k_nh_max_custom];
};

// One materialised bucket: the half-open interval (lower, upper] and its count.
struct NhBucket {
    double lower;
    double upper;
    double count;
};

inline void nh_init(NativeHistogram* h) noexcept {
    assert(h != nullptr);
    h->schema = 0;
    h->pos_offset = 0; h->n_pos = 0;
    h->neg_offset = 0; h->n_neg = 0;
    h->n_custom = 0;
    h->reset_hint = static_cast<uint8_t>(NhResetHint::Unknown);
    for (int32_t i = 0; i < 7; ++i) h->_pad[i] = 0;
    h->zero_threshold = 0.0;
    h->zero_count = 0.0;
    h->count = 0.0;
    h->sum = 0.0;
    assert(h->n_pos == 0 && h->n_neg == 0);
}

inline bool nh_uses_custom(const NativeHistogram* h) noexcept {
    assert(h != nullptr);
    assert(h->n_custom >= 0);
    return h->schema == k_nh_schema_custom;
}

namespace detail {

// prometheus/model/histogram/generic.go :: getBoundExponential. Reproduced
// bit-for-bit apart from the precomputed `exponentialBounds` table, which is
// 0.5 * 2^(i / 2^schema) and is computed here instead (exp2 of an exact
// dyadic rational; the two agree to within 1 ULP, far inside the corpus's
// comparison tolerance).
inline double nh_bound_exponential(int32_t idx, int32_t schema) noexcept {
    assert(schema >= k_nh_schema_min && schema <= k_nh_schema_max);
    assert(idx > -(1 << 22) && idx < (1 << 22));  // sane index range
    if (schema < 0) {
        const int exponent = static_cast<int>(idx) << (-schema);
        if (exponent == 1024) return DBL_MAX;
        return std::ldexp(1.0, exponent);
    }
    const int32_t mask     = (1 << schema) - 1;
    const int32_t frac_idx = idx & mask;
    const double  frac =
        0.5 * std::exp2(static_cast<double>(frac_idx) /
                        static_cast<double>(int32_t{1} << schema));
    const int exponent = (static_cast<int>(idx) >> schema) + 1;
    if (frac == 0.5 && exponent == 1025) return DBL_MAX;
    return std::ldexp(frac, exponent);
}

// prometheus/model/histogram/generic.go :: getBound. `idx` outside
// [-1, n_custom] is a caller bug for the custom-bucket case (Go panics);
// here it asserts and clamps to an infinity so a release build cannot read
// out of bounds.
inline double nh_bound(int32_t idx, int32_t schema, const double* cv,
                       int32_t n_cv) noexcept {
    assert(schema == k_nh_schema_custom || cv == nullptr || n_cv >= 0);
    assert(n_cv >= 0 && n_cv <= k_nh_max_custom);
    if (schema == k_nh_schema_custom) {
        const double inf = std::numeric_limits<double>::infinity();
        if (idx >= n_cv) return inf;
        if (idx <= -1)   return -inf;
        assert(cv != nullptr);
        return cv[idx];
    }
    return nh_bound_exponential(idx, schema);
}

// Kahan-Babuska-Neumaier increment, matching prometheus/util/kahansum.
inline void nh_kahan_inc(double inc, double* sum, double* c) noexcept {
    assert(sum != nullptr && c != nullptr);
    const double t = *sum + inc;
    if (std::fabs(*sum) >= std::fabs(inc)) *c += (*sum - t) + inc;
    else                                   *c += (inc - t) + *sum;
    *sum = t;
}

}  // namespace detail

// Materialise every bucket in ASCENDING VALUE order — negatives (most negative
// first), then the zero bucket if its count is non-zero, then positives — into
// `out`. Mirrors allFloatBucketIterator.Next, including its clipping of a
// bucket bound that falls strictly inside the zero threshold.
//
// Returns the number written, or -1 if `cap` is too small (never a partial
// answer: a dropped bucket silently changes every quantile).
inline int32_t nh_all_buckets(const NativeHistogram* h, NhBucket* BOLT_RESTRICT out,
                              int32_t cap) noexcept {
    assert(h != nullptr);
    assert(out != nullptr || cap == 0);
    const bool   custom = nh_uses_custom(h);
    const double* cv    = custom ? h->custom : nullptr;
    const int32_t n_cv  = custom ? h->n_custom : 0;
    const double  zt    = h->zero_threshold;
    int32_t n = 0;

    // Negative side, descending index == ascending value.
    for (int32_t j = h->n_neg - 1; j >= 0; --j) {   // bounded by n_neg
        if (n >= cap) return -1;
        const int32_t idx = h->neg_offset + j;
        NhBucket b;
        b.lower = -detail::nh_bound(idx, h->schema, cv, n_cv);
        b.upper = -detail::nh_bound(idx - 1, h->schema, cv, n_cv);
        b.count = h->neg[j];
        if (b.upper < 0.0 && b.upper > -zt) b.upper = -zt;
        else if (b.lower > 0.0 && b.lower < zt) b.lower = zt;
        out[n++] = b;
    }
    if (h->zero_count > 0.0) {
        if (n >= cap) return -1;
        NhBucket b;
        b.lower = -zt; b.upper = zt; b.count = h->zero_count;
        out[n++] = b;
    }
    for (int32_t j = 0; j < h->n_pos; ++j) {        // bounded by n_pos
        if (n >= cap) return -1;
        const int32_t idx = h->pos_offset + j;
        NhBucket b;
        b.lower = detail::nh_bound(idx - 1, h->schema, cv, n_cv);
        b.upper = detail::nh_bound(idx, h->schema, cv, n_cv);
        b.count = h->pos[j];
        if (b.lower > 0.0 && b.lower < zt) b.lower = zt;
        else if (b.upper < 0.0 && b.upper > -zt) b.upper = -zt;
        out[n++] = b;
    }
    assert(n <= cap);
    return n;
}

namespace detail {

// The zero-bucket bound fixup both quantile and fraction apply before
// interpolating: a bucket straddling zero is closed at zero on the side where
// the histogram has no buckets at all.
inline void nh_close_zero_bucket(const NativeHistogram* h, NhBucket* b) noexcept {
    assert(h != nullptr && b != nullptr);
    if (h->n_neg == 0 && h->n_pos > 0)      b->lower = 0.0;
    else if (h->n_pos == 0 && h->n_neg > 0) b->upper = 0.0;
}

// prometheus/model/histogram/generic.go :: Bucket.FractionBelow.
inline double nh_fraction_below(const NhBucket* b, double v, bool linear) noexcept {
    assert(b != nullptr);
    assert(!std::isnan(v));
    if (linear) return (v - b->lower) / (b->upper - b->lower);
    const double log_lower = std::log2(std::fabs(b->lower));
    const double log_upper = std::log2(std::fabs(b->upper));
    const double log_v     = std::log2(std::fabs(v));
    if (v > 0.0) return (log_v - log_lower) / (log_upper - log_lower);
    return 1.0 - ((log_v - log_upper) / (log_lower - log_upper));
}

}  // namespace detail

// histogram_quantile(q, h) over a NATIVE histogram.
// `scratch` must hold >= nh_all_buckets()'s output; on overflow returns NaN
// rather than a quantile computed from a truncated bucket set.
inline double nh_quantile(const NativeHistogram* h, double q,
                          NhBucket* BOLT_RESTRICT scratch,
                          int32_t scratch_cap) noexcept {
    assert(h != nullptr);
    assert(scratch != nullptr || scratch_cap == 0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (q < 0.0) return -std::numeric_limits<double>::infinity();
    if (q > 1.0) return std::numeric_limits<double>::infinity();
    if (h->count == 0.0 || std::isnan(q)) return nan;

    const int32_t nb = nh_all_buckets(h, scratch, scratch_cap);
    if (nb < 0) return nan;

    const bool custom  = nh_uses_custom(h);
    const bool nan_sum = std::isnan(h->sum);
    const bool forward = nan_sum || q < 0.5;
    double rank  = forward ? q * h->count : (1.0 - q) * h->count;
    double count = 0.0;
    NhBucket bucket{0.0, 0.0, 0.0};
    for (int32_t k = 0; k < nb; ++k) {              // bounded by nb
        bucket = forward ? scratch[k] : scratch[nb - 1 - k];
        if (bucket.count == 0.0) continue;
        count += bucket.count;
        if (count >= rank) break;
    }

    if (!custom && bucket.lower < 0.0 && bucket.upper > 0.0) {
        detail::nh_close_zero_bucket(h, &bucket);
    } else if (custom) {
        const double inf = std::numeric_limits<double>::infinity();
        if (bucket.lower == -inf) {
            if (bucket.upper <= 0.0) return bucket.upper;
            bucket.lower = 0.0;
        } else if (bucket.upper == inf) {
            return bucket.lower;
        }
    }
    if (count > h->count) count = h->count;
    if (count < rank) return nan_sum ? nan : bucket.upper;

    rank = forward ? rank - (count - bucket.count) : count - rank;
    const double fraction = rank / bucket.count;

    if (custom || (bucket.lower <= 0.0 && bucket.upper >= 0.0))
        return bucket.lower + (bucket.upper - bucket.lower) * fraction;

    const double log_lower = std::log2(std::fabs(bucket.lower));
    const double log_upper = std::log2(std::fabs(bucket.upper));
    if (bucket.lower > 0.0)
        return std::exp2(log_lower + (log_upper - log_lower) * fraction);
    return -std::exp2(log_upper + (log_lower - log_upper) * (1.0 - fraction));
}

// histogram_fraction(lower, upper, h) — the estimated share of observations in
// (lower, upper]. Faithful port of promql/quantile.go :: HistogramFraction.
inline double nh_fraction(const NativeHistogram* h, double lower, double upper,
                          NhBucket* BOLT_RESTRICT scratch,
                          int32_t scratch_cap) noexcept {
    assert(h != nullptr);
    assert(scratch != nullptr || scratch_cap == 0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (h->count == 0.0 || std::isnan(lower) || std::isnan(upper)) return nan;
    if (lower >= upper) return 0.0;

    const int32_t nb = nh_all_buckets(h, scratch, scratch_cap);
    if (nb < 0) return nan;
    const bool custom = nh_uses_custom(h);

    double rank = 0.0, lower_rank = 0.0, upper_rank = 0.0, count = 0.0;
    bool lower_set = false, upper_set = false;
    int32_t k = 0;
    for (; k < nb; ++k) {                            // bounded by nb
        NhBucket b = scratch[k];
        count += b.count;
        bool zero_bucket = false;
        if (b.lower <= 0.0 && b.upper >= 0.0) {
            zero_bucket = true;
            detail::nh_close_zero_bucket(h, &b);
        }
        if (!lower_set && b.lower >= lower) { lower_rank = rank; lower_set = true; }
        if (!upper_set && b.lower >= upper) { upper_rank = rank; upper_set = true; }
        if (lower_set && upper_set) break;
        const bool linear = custom || zero_bucket;
        const double inf = std::numeric_limits<double>::infinity();
        if (!lower_set && b.lower < lower && b.upper > lower) {
            lower_rank = (linear && b.lower == -inf)
                ? b.count
                : rank + b.count * detail::nh_fraction_below(&b, lower, linear);
            lower_set = true;
        }
        if (!upper_set && b.lower < upper && b.upper > upper) {
            upper_rank = (linear && b.lower == -inf)
                ? b.count
                : rank + b.count * detail::nh_fraction_below(&b, upper, linear);
            upper_set = true;
        }
        if (lower_set && upper_set) break;
        rank += b.count;
    }
    if (std::isnan(h->sum)) {
        for (int32_t j = k + 1; j < nb; ++j) count += scratch[j].count;
    } else {
        count = h->count;
    }
    if (!lower_set || lower_rank > count) lower_rank = count;
    if (!upper_set || upper_rank > count) upper_rank = count;
    return (upper_rank - lower_rank) / h->count;
}

// histogram_stdvar(h) — variance of the observations estimated from bucket
// representatives (geometric mean for exponential buckets, arithmetic mean for
// custom buckets, zero inside the zero bucket). promql/functions.go ::
// histogramVariance.
inline double nh_stdvar(const NativeHistogram* h,
                        NhBucket* BOLT_RESTRICT scratch,
                        int32_t scratch_cap) noexcept {
    assert(h != nullptr);
    assert(scratch != nullptr || scratch_cap == 0);
    const int32_t nb = nh_all_buckets(h, scratch, scratch_cap);
    if (nb < 0) return std::numeric_limits<double>::quiet_NaN();
    const bool   custom = nh_uses_custom(h);
    const double mean   = h->sum / h->count;
    double variance = 0.0, c_variance = 0.0;
    for (int32_t k = 0; k < nb; ++k) {               // bounded by nb
        const NhBucket& b = scratch[k];
        if (b.count == 0.0) continue;
        double val;
        if (custom)                                  val = (b.upper + b.lower) / 2.0;
        else if (b.lower <= 0.0 && b.upper >= 0.0)   val = 0.0;
        else {
            val = std::sqrt(b.upper * b.lower);
            if (b.upper < 0.0) val = -val;
        }
        const double delta = val - mean;
        detail::nh_kahan_inc(b.count * delta * delta, &variance, &c_variance);
    }
    // A non-finite running sum makes the Kahan compensation NaN (Inf - Inf),
    // which would turn a legitimately INFINITE variance into NaN. Prometheus's
    // kahansum has the same guard; histogram_stdvar over a sum:Inf histogram
    // is +Inf, not NaN (native_histograms.test's stddev_stdvar_7).
    if (std::isfinite(variance)) variance += c_variance;
    variance /= h->count;
    assert(!(variance < 0.0));  // a sum of squares is never negative
    return variance;
}

inline double nh_stddev(const NativeHistogram* h,
                        NhBucket* BOLT_RESTRICT scratch,
                        int32_t scratch_cap) noexcept {
    assert(h != nullptr);
    assert(scratch != nullptr || scratch_cap == 0);
    return std::sqrt(nh_stdvar(h, scratch, scratch_cap));
}

namespace detail {

// Widen one side's dense run to cover [lo, hi) in place. Returns false if the
// span exceeds k_nh_max_side — REFUSED, never truncated.
inline bool nh_widen_side(double* BOLT_RESTRICT buf, int32_t* offset, int32_t* n,
                          int32_t lo, int32_t hi) noexcept {
    assert(buf != nullptr && offset != nullptr && n != nullptr);
    assert(hi >= lo);
    const int32_t width = hi - lo;
    if (width > k_nh_max_side) return false;
    if (width == *n && lo == *offset) return true;
    double tmp[k_nh_max_side];
    for (int32_t i = 0; i < width; ++i) tmp[i] = 0.0;   // bounded by width
    for (int32_t i = 0; i < *n; ++i) {                  // bounded by n
        const int32_t at = (*offset + i) - lo;
        assert(at >= 0 && at < width);
        tmp[at] = buf[i];
    }
    for (int32_t i = 0; i < width; ++i) buf[i] = tmp[i];
    *offset = lo;
    *n = width;
    return true;
}

}  // namespace detail

// h *= factor. prometheus/model/histogram/float_histogram.go :: Mul — every
// count (zero bucket, positives, negatives) and the sum scale by the same
// factor; the bucket LADDER (schema, offsets, custom values) is untouched
// because scaling does not move a boundary. A factor of 0 keeps the buckets
// present with a count of 0, which is what upstream's Mul does and what
// `histogram_mul_div*0` in native_histograms.test pins.
inline void nh_mul(NativeHistogram* h, double factor) noexcept {
    assert(h != nullptr);
    assert(h->n_pos >= 0 && h->n_neg >= 0);
    h->zero_count *= factor;
    h->count *= factor;
    h->sum *= factor;
    for (int32_t i = 0; i < h->n_pos; ++i) h->pos[i] *= factor;   // bounded
    for (int32_t i = 0; i < h->n_neg; ++i) h->neg[i] *= factor;   // bounded
    assert(h->n_pos <= k_nh_max_side && h->n_neg <= k_nh_max_side);
}

// h /= scalar. prometheus/model/histogram/float_histogram.go :: Div — NOT
// `nh_mul(h, 1/scalar)`: upstream special-cases a divisor of zero by REMOVING
// every bucket (a bucket count of ±Inf is meaningless) while still dividing
// the zero bucket, count and sum, so the result carries only ±Inf/NaN
// aggregates. `histogram_mul_div/0` and `histogram_mul_div*0/0` pin exactly
// that shape, buckets absent.
inline void nh_div(NativeHistogram* h, double scalar) noexcept {
    assert(h != nullptr);
    assert(h->n_pos >= 0 && h->n_neg >= 0);
    h->zero_count /= scalar;
    h->count /= scalar;
    h->sum /= scalar;
    if (scalar == 0.0) { h->n_pos = 0; h->n_neg = 0; h->pos_offset = 0; h->neg_offset = 0; return; }
    for (int32_t i = 0; i < h->n_pos; ++i) h->pos[i] /= scalar;   // bounded
    for (int32_t i = 0; i < h->n_neg; ++i) h->neg[i] /= scalar;   // bounded
    assert(h->n_pos <= k_nh_max_side && h->n_neg <= k_nh_max_side);
}

namespace detail {

// prometheus/model/histogram/generic.go :: targetIdx — the bucket index in the
// coarser `target` schema that the `origin`-schema bucket `idx` falls into.
// The `-1`/`+1` is not cosmetic: bucket idx 0 has UPPER bound 1, so the runs
// group around the boundary at 1, not around index 0. Verified against
// native_histograms.test's own answer key: reducing schema 1 `[0 2 1]` to
// schema 0 gives `{0:0, 1:3}`, which is exactly what
// `histogram_sub_2{idx="0"} - ignoring(idx) histogram_sub_2{idx="1"}` requires.
inline int32_t nh_target_idx(int32_t idx, int32_t shift) noexcept {
    assert(shift >= 0 && shift < 31);
    return ((idx - 1) >> shift) + 1;   // C++20: >> on signed is arithmetic
}

// Collapse one dense side onto the coarser ladder `shift` levels down.
// Returns false — leaving the side untouched — when the reduced run would
// exceed k_nh_max_side; never a partial reduction.
inline bool nh_reduce_side(double* BOLT_RESTRICT buf, int32_t* offset,
                           int32_t* n, int32_t shift) noexcept {
    assert(buf != nullptr && offset != nullptr && n != nullptr);
    if (*n == 0) return true;
    const int32_t t_lo = nh_target_idx(*offset, shift);
    const int32_t t_hi = nh_target_idx(*offset + *n - 1, shift);
    assert(t_hi >= t_lo);
    const int32_t width = t_hi - t_lo + 1;
    if (width > k_nh_max_side) return false;
    double tmp[k_nh_max_side];
    for (int32_t i = 0; i < width; ++i) tmp[i] = 0.0;   // bounded by width
    for (int32_t i = 0; i < *n; ++i) {                  // bounded by n
        const int32_t at = nh_target_idx(*offset + i, shift) - t_lo;
        assert(at >= 0 && at < width);
        tmp[at] += buf[i];
    }
    for (int32_t i = 0; i < width; ++i) buf[i] = tmp[i];
    *offset = t_lo;
    *n = width;
    return true;
}

// prometheus/model/histogram/float_histogram.go :: intersectCustomBucketBounds.
// Both ladders ascend, so the intersection is one merge walk. Upstream compares
// with `==` here (its CustomBucketBoundsMatch uses the bit pattern; this does
// not) and returns nil — length 0 — when either side is empty.
inline int32_t nh_intersect_custom(const double* BOLT_RESTRICT a, int32_t na,
                                   const double* BOLT_RESTRICT b, int32_t nb,
                                   double* BOLT_RESTRICT out) noexcept {
    assert((a != nullptr || na == 0) && (b != nullptr || nb == 0));
    assert(na >= 0 && na <= k_nh_max_custom && nb >= 0 && nb <= k_nh_max_custom);
    assert(out != nullptr);
    if (na == 0 || nb == 0) return 0;
    int32_t i = 0, j = 0, n = 0;
    while (i < na && j < nb) {                       // bounded by na + nb
        if      (a[i] == b[j]) { out[n++] = a[i]; ++i; ++j; }
        else if (a[i] <  b[j]) { ++i; }
        else                   { ++j; }
    }
    assert(n <= na && n <= nb);
    return n;
}

// Fold one NHCB's dense positive buckets onto the INTERSECTED ladder,
// accumulating into `target[0 .. n_int]` with sign `sgn`. `target` has n_int+1
// slots: one per intersected bound plus the trailing +Inf bucket.
//
// Mirrors the `mapBuckets` closure inside upstream's
// addCustomBucketsWithMismatches: a source bucket whose upper bound is
// `bounds[idx]` lands in the FIRST intersected bucket whose bound is >= that,
// and in the +Inf bucket when there is none. `k` is monotone because both
// ladders ascend and `inter` is a subsequence of `bounds`.
//
// Upstream folds with Kahan. Its compensation term is DISCARDED by both Add and
// Sub (each takes `_` for it) and Kahan's running total is `sum + inc` — the
// same value a plain `+=` writes — so this is bit-identical to upstream for
// those two callers. It is NOT identical to KahanAdd, which keeps the term;
// that path is not what this kernel implements (see nh_combine).
inline void nh_map_custom_onto(const double* BOLT_RESTRICT src, int32_t off,
                               int32_t n, const double* BOLT_RESTRICT bounds,
                               int32_t n_bounds,
                               const double* BOLT_RESTRICT inter, int32_t n_int,
                               double sgn, double* BOLT_RESTRICT target) noexcept {
    assert((src != nullptr && bounds != nullptr) || n == 0);
    assert(n >= 0 && n <= k_nh_max_side);
    assert(target != nullptr && n_int >= 0 && n_int <= k_nh_max_custom);
    int32_t k = 0;
    for (int32_t j = 0; j < n; ++j) {                // bounded by n
        const int32_t idx = off + j;
        assert(idx >= 0);
        int32_t t = n_int;                           // the +Inf bucket
        if (idx >= 0 && idx < n_bounds) {
            const double b = bounds[idx];
            while (k < n_int) {                      // bounded by n_int
                if (inter[k] >= b) { t = k; break; }
                ++k;
            }
        }
        target[t] += sgn * src[j];
    }
}

}  // namespace detail

// Reduce `h` to the coarser exponential schema `target`, in place.
// prometheus/model/histogram/generic.go :: reduceResolution. Refuses (false,
// h untouched) for a custom-bucket histogram — an NHCB ladder has no coarser
// form — and for a target finer than the current schema.
inline bool nh_reduce_resolution(NativeHistogram* h, int32_t target) noexcept {
    assert(h != nullptr);
    assert(h->n_pos >= 0 && h->n_neg >= 0);
    if (nh_uses_custom(h)) return false;
    if (target < k_nh_schema_min || target > k_nh_schema_max) return false;
    if (target > h->schema) return false;
    if (target == h->schema) return true;
    const int32_t shift = h->schema - target;
    NativeHistogram out = *h;
    if (!detail::nh_reduce_side(out.pos, &out.pos_offset, &out.n_pos, shift)) return false;
    if (!detail::nh_reduce_side(out.neg, &out.neg_offset, &out.n_neg, shift)) return false;
    out.schema = target;
    *h = out;
    assert(h->schema == target);
    return true;
}

// dst = dst (+/-) src for two NHCBs whose custom ladders DIFFER.
//
// Upstream does NOT refuse this pair. Since v3.6 (Add/Sub in
// float_histogram.go, via addCustomBucketsWithMismatches) it RECONCILES the two
// ladders onto the INTERSECTION of their bound sets and folds every bucket of
// each operand into the intersected bucket that contains it. Refusing was this
// kernel's behaviour through W25 and it is what made 22 corpus cells RED.
//
// Note what the intersection is NOT: it is not a union, and it is not the finer
// ladder. Dropping a bound merges the two buckets it separated, so mass is only
// ever coarsened — never split, never interpolated. That is why the answer is
// exact rather than estimated, and why an empty intersection is legal: the
// result is then a single (-Inf, +Inf] bucket, which is exactly what upstream's
// `nil` CustomValues plus one bucket means.
inline bool nh_combine_custom_mismatch(NativeHistogram* BOLT_RESTRICT dst,
                                       const NativeHistogram* BOLT_RESTRICT src,
                                       bool subtract) noexcept {
    assert(dst != nullptr && src != nullptr);
    assert(nh_uses_custom(dst) && nh_uses_custom(src));
    double inter[k_nh_max_custom];
    const int32_t n_int = detail::nh_intersect_custom(
        dst->custom, dst->n_custom, src->custom, src->n_custom, inter);
    double target[k_nh_max_custom + 1];
    for (int32_t i = 0; i <= n_int; ++i) target[i] = 0.0;  // bounded by n_int
    const double sgn = subtract ? -1.0 : 1.0;
    detail::nh_map_custom_onto(dst->pos, dst->pos_offset, dst->n_pos,
                               dst->custom, dst->n_custom, inter, n_int, 1.0, target);
    detail::nh_map_custom_onto(src->pos, src->pos_offset, src->n_pos,
                               src->custom, src->n_custom, inter, n_int, sgn, target);
    // Upstream drops every zero-population bucket when it rebuilds the spans. A
    // dense run cannot express an interior gap, so the equivalent here is to
    // trim the zero prefix and suffix; a surviving interior zero reads as a
    // zero-count bucket, which every consumer in this tree already treats as an
    // absent one (nh_quantile skips it, nh_all_buckets emits it with count 0).
    int32_t lo = 0, hi = n_int + 1;
    while (lo < hi && target[lo] == 0.0) ++lo;              // bounded by n_int+1
    while (hi > lo && target[hi - 1] == 0.0) --hi;          // bounded by n_int+1
    if (hi - lo > k_nh_max_side) return false;
    for (int32_t i = lo; i < hi; ++i) dst->pos[i - lo] = target[i];
    dst->pos_offset = lo;
    dst->n_pos = hi - lo;
    for (int32_t i = 0; i < n_int; ++i) dst->custom[i] = inter[i];
    dst->n_custom = n_int;
    dst->count += sgn * src->count;
    dst->sum   += sgn * src->sum;
    assert(dst->n_pos >= 0 && dst->n_pos <= k_nh_max_side);
    assert(dst->n_custom >= 0 && dst->n_custom <= k_nh_max_custom);
    return true;
}

// dst = dst + src (subtract == false) or dst - src (subtract == true).
// prometheus/model/histogram/float_histogram.go :: Add / Sub — both operands
// are first brought onto the COARSER of the two schemas, then combined bucket
// for bucket. Two NHCBs with different custom bounds are reconciled onto the
// intersection of those bounds (nh_combine_custom_mismatch). Returns false,
// leaving `dst` untouched, for any pair this cannot represent exactly:
//   * one custom-bucket and one exponential — the ladders are not comparable,
//     and upstream errors here too (checkSchemaAndBounds);
//   * different zero thresholds — upstream widens the zero bucket by absorbing
//     adjacent buckets until the thresholds meet, which MOVES counts between
//     buckets; refusing is honest, guessing is a confident wrong distribution;
//   * a union or reduction wider than k_nh_max_side.
inline bool nh_combine(NativeHistogram* BOLT_RESTRICT dst,
                       const NativeHistogram* BOLT_RESTRICT src,
                       bool subtract) noexcept {
    assert(dst != nullptr && src != nullptr);
    assert(dst != src);
    const bool cd = nh_uses_custom(dst), cs = nh_uses_custom(src);
    if (cd != cs) return false;
    if (cd) {
        bool same = dst->n_custom == src->n_custom;
        for (int32_t i = 0; same && i < dst->n_custom; ++i)  // bounded by n_custom
            if (dst->custom[i] != src->custom[i]) same = false;
        if (!same) return nh_combine_custom_mismatch(dst, src, subtract);
    }
    if (dst->zero_threshold != src->zero_threshold) return false;
    NativeHistogram a = *dst;
    NativeHistogram b = *src;
    if (!cd && a.schema != b.schema) {
        if (a.schema > b.schema) { if (!nh_reduce_resolution(&a, b.schema)) return false; }
        else                     { if (!nh_reduce_resolution(&b, a.schema)) return false; }
    }
    assert(a.schema == b.schema);
    const double sgn = subtract ? -1.0 : 1.0;
    if (b.n_pos > 0) {
        const int32_t lo = (a.n_pos == 0) ? b.pos_offset
            : (a.pos_offset < b.pos_offset ? a.pos_offset : b.pos_offset);
        const int32_t a_hi = a.pos_offset + a.n_pos, b_hi = b.pos_offset + b.n_pos;
        const int32_t hi = (a.n_pos == 0) ? b_hi : (a_hi > b_hi ? a_hi : b_hi);
        if (!detail::nh_widen_side(a.pos, &a.pos_offset, &a.n_pos, lo, hi)) return false;
        for (int32_t i = 0; i < b.n_pos; ++i)         // bounded by b.n_pos
            a.pos[(b.pos_offset + i) - a.pos_offset] += sgn * b.pos[i];
    }
    if (b.n_neg > 0) {
        const int32_t lo = (a.n_neg == 0) ? b.neg_offset
            : (a.neg_offset < b.neg_offset ? a.neg_offset : b.neg_offset);
        const int32_t a_hi = a.neg_offset + a.n_neg, b_hi = b.neg_offset + b.n_neg;
        const int32_t hi = (a.n_neg == 0) ? b_hi : (a_hi > b_hi ? a_hi : b_hi);
        if (!detail::nh_widen_side(a.neg, &a.neg_offset, &a.n_neg, lo, hi)) return false;
        for (int32_t i = 0; i < b.n_neg; ++i)         // bounded by b.n_neg
            a.neg[(b.neg_offset + i) - a.neg_offset] += sgn * b.neg[i];
    }
    a.count += sgn * b.count;
    a.sum += sgn * b.sum;
    a.zero_count += sgn * b.zero_count;
    *dst = a;
    assert(dst->n_pos <= k_nh_max_side && dst->n_neg <= k_nh_max_side);
    return true;
}

// ---- counter-reset detection ------------------------------------------
//
// prometheus/model/histogram/float_histogram.go :: DetectReset + detectReset,
// as of v3.12.0 — the version whose `native_histograms.test` this tree vendors
// (that file differs from ours by ONE comment typo; v3.7.3's is 842 lines
// shorter, so the docker oracle is materially behind the answer key).
//
// A reset is NOT "the count went down". Upstream's rule is a conjunction over
// FOUR independent things — the explicit CounterResetHint on the sample, the
// observation count, the zero bucket under a possibly-different threshold, and
// EVERY bucket compared on a COMMON schema — and getting any of them wrong is a
// silent wrong rate on every histogram latency panel, never an error. That is
// why this returns a THREE-valued answer: a shape whose reset cannot be decided
// exactly (upstream reconciles mismatched NHCB bounds; this kernel does not)
// answers `Undecidable`, and the caller must refuse the query.
enum class NhReset : uint8_t { No = 0, Yes = 1, Undecidable = 2 };

namespace detail {

// One side of a histogram as prometheus's `floatBucketIterator(positive,
// absoluteStartValue = zt, targetSchema)` yields it: merged onto `target` and
// with the LEADING buckets whose upper bound is <= `zt` dropped. Dense, so an
// index outside [offset, offset+n) is an absent bucket, exactly as a gap in
// upstream's spans is.
struct NhSideView {
    double  buf[k_nh_max_side];
    int32_t offset;
    int32_t n;
};

// Build the view. Returns false when the merge would exceed k_nh_max_side —
// never a partial view, because a truncated side answers confidently and
// wrongly (a dropped bucket reads as a missing bucket, i.e. a fabricated reset).
inline bool nh_side_view(const double* BOLT_RESTRICT src, int32_t off, int32_t n,
                         int32_t schema, int32_t target, double zt,
                         NhSideView* BOLT_RESTRICT out) noexcept {
    assert(src != nullptr && out != nullptr);
    assert(n >= 0 && n <= k_nh_max_side);
    out->offset = off;
    out->n = n;
    for (int32_t i = 0; i < n; ++i) out->buf[i] = src[i];   // bounded by n
    if (schema != target) {
        if (target > schema) return false;
        if (!nh_reduce_side(out->buf, &out->offset, &out->n, schema - target))
            return false;
    }
    // `boundReachedStartValue` starts true when zt == 0, so a zero threshold
    // skips nothing. Bounds rise with the index, so the skipped set is a
    // prefix; custom-bucket ladders are never skipped (upstream ignores
    // absoluteStartValue for them).
    if (zt != 0.0 && target != k_nh_schema_custom) {
        int32_t k = 0;
        while (k < out->n &&
               nh_bound_exponential(out->offset + k, target) <= zt) ++k;
        if (k > 0) {
            for (int32_t i = k; i < out->n; ++i) out->buf[i - k] = out->buf[i];
            out->offset += k;
            out->n -= k;
        }
    }
    assert(out->n >= 0 && out->n <= k_nh_max_side);
    return true;
}

// prometheus's `detectReset(currIt, prevIt)` over two dense views. Walking
// prev's indices and treating an index outside curr's run as ABSENT reproduces
// upstream's three branches exactly — including the one that differs from a
// plain `curr < prev`: a bucket prev HAS and curr LACKS is a reset whenever its
// count is non-zero, which is not the same test when that count is negative.
inline bool nh_side_reset(const NhSideView* curr, const NhSideView* prev) noexcept {
    assert(curr != nullptr && prev != nullptr);
    assert(curr->n >= 0 && curr->n <= k_nh_max_side);
    assert(prev->n >= 0 && prev->n <= k_nh_max_side);
    for (int32_t i = 0; i < prev->n; ++i) {          // bounded by prev->n
        const int32_t idx = prev->offset + i;
        const double  p   = prev->buf[i];
        if (idx >= curr->offset && idx < curr->offset + curr->n) {
            if (curr->buf[idx - curr->offset] < p) return true;
        } else if (p != 0.0) {
            return true;
        }
    }
    return false;
}

// prometheus's `detectResetWithMismatchedCustomBounds` over two NHCBs.
//
// Upstream walks both ladders with two bucket iterators, and at every bound the
// two ladders SHARE it rolls up the mass each has accumulated since the last
// shared bound, then compares those two roll-ups. The set of shared bounds is
// by definition nh_intersect_custom's output, and "the mass since the previous
// shared bound" is by definition the intersected bucket — so mapping both onto
// the intersected ladder and comparing element-wise is the same partition and
// the same comparison, with the same trailing +Inf slot that upstream reaches
// when both bound indices run off their ends.
//
// It must NOT fall through to nh_side_reset: that compares by ABSOLUTE bucket
// index, and two different ladders give the same index different bounds.
inline bool nh_reset_mismatched_custom(const NativeHistogram* curr,
                                       const NativeHistogram* prev) noexcept {
    assert(curr != nullptr && prev != nullptr);
    assert(curr->schema == k_nh_schema_custom && prev->schema == k_nh_schema_custom);
    double inter[k_nh_max_custom];
    const int32_t n_int = nh_intersect_custom(curr->custom, curr->n_custom,
                                              prev->custom, prev->n_custom, inter);
    double c[k_nh_max_custom + 1], p[k_nh_max_custom + 1];
    for (int32_t i = 0; i <= n_int; ++i) { c[i] = 0.0; p[i] = 0.0; }
    nh_map_custom_onto(curr->pos, curr->pos_offset, curr->n_pos,
                       curr->custom, curr->n_custom, inter, n_int, 1.0, c);
    nh_map_custom_onto(prev->pos, prev->pos_offset, prev->n_pos,
                       prev->custom, prev->n_custom, inter, n_int, 1.0, p);
    for (int32_t i = 0; i <= n_int; ++i)             // bounded by n_int + 1
        if (c[i] < p[i]) return true;
    return false;
}

}  // namespace detail

// prometheus's `zeroCountForLargerThreshold`: what `h`'s zero count would be if
// its zero threshold were the larger `*threshold`. If that threshold lands
// INSIDE a populated bucket it is raised to that bucket's outer bound and
// written back — the caller compares the returned threshold against the one it
// asked for, and a change means "reset" (upstream's own reading).
// Returns false when the fixpoint does not settle within the bucket bound.
inline bool nh_zero_count_for_larger_threshold(const NativeHistogram* h,
                                               double* BOLT_RESTRICT threshold,
                                               double* BOLT_RESTRICT out) noexcept {
    assert(h != nullptr && threshold != nullptr && out != nullptr);
    assert(!(*threshold < h->zero_threshold));
    if (*threshold == h->zero_threshold) { *out = h->zero_count; return true; }
    if (nh_uses_custom(h)) return false;     // no exponential ladder to walk
    double lt = *threshold;
    // The negative side can raise `lt` once per bucket, and each raise redoes
    // the walk; k_nh_max_side + 1 passes is therefore a real bound.
    for (int32_t pass = 0; pass <= k_nh_max_side; ++pass) {
        double zc = h->zero_count;
        bool restart = false;
        for (int32_t j = 0; j < h->n_pos; ++j) {          // bounded by n_pos
            const int32_t idx = h->pos_offset + j;
            if (detail::nh_bound_exponential(idx - 1, h->schema) >= lt) break;
            zc += h->pos[j];
            const double upper = detail::nh_bound_exponential(idx, h->schema);
            if (upper > lt) { if (h->pos[j] != 0.0) lt = upper; break; }
        }
        for (int32_t j = 0; j < h->n_neg; ++j) {          // bounded by n_neg
            const int32_t idx = h->neg_offset + j;
            if (-detail::nh_bound_exponential(idx - 1, h->schema) <= -lt) break;
            zc += h->neg[j];
            const double lower = -detail::nh_bound_exponential(idx, h->schema);
            if (lower < -lt) {
                if (h->neg[j] != 0.0) { lt = -lower; restart = true; }
                break;
            }
        }
        if (!restart) { *threshold = lt; *out = zc; return true; }
    }
    return false;
}

// True when two NHCB ladders are the same. Upstream compares the bound lists
// outright; a differing ladder is not a differently-spelled same histogram.
inline bool nh_custom_bounds_match(const NativeHistogram* a,
                                   const NativeHistogram* b) noexcept {
    assert(a != nullptr && b != nullptr);
    assert(a->n_custom >= 0 && a->n_custom <= k_nh_max_custom);
    assert(b->n_custom >= 0 && b->n_custom <= k_nh_max_custom);
    if (a->n_custom != b->n_custom) return false;
    for (int32_t i = 0; i < a->n_custom; ++i)      // bounded by n_custom
        if (a->custom[i] != b->custom[i]) return false;
    return true;
}

// Does `h` come after a counter reset relative to `prev`?
inline NhReset nh_detect_reset(const NativeHistogram* h,
                               const NativeHistogram* prev) noexcept {
    assert(h != nullptr && prev != nullptr);
    assert(h->n_pos <= k_nh_max_side && prev->n_pos <= k_nh_max_side);
    if (h->reset_hint == static_cast<uint8_t>(NhResetHint::CounterReset))
        return NhReset::Yes;
    if (h->reset_hint == static_cast<uint8_t>(NhResetHint::NotReset))
        return NhReset::No;
    // Unknown and Gauge both fall through: PromQL still lets a counter
    // function run over a gauge histogram, and warns rather than refusing.
    if (h->count < prev->count) return NhReset::Yes;
    if (nh_uses_custom(h)) {
        if (!nh_uses_custom(prev)) return NhReset::Yes;
        // Mismatched ladders are reconciled bucket by bucket and the answer
        // RETURNS here — upstream's DetectReset returns from this branch too,
        // and the checks below it (schema, zero threshold, per-index bucket
        // walk) all assume a shared ladder.
        if (!nh_custom_bounds_match(h, prev))
            return detail::nh_reset_mismatched_custom(h, prev) ? NhReset::Yes
                                                               : NhReset::No;
    }
    if (h->schema > prev->schema) return NhReset::Yes;
    if (h->zero_threshold < prev->zero_threshold) return NhReset::Yes;
    double thr = h->zero_threshold, pzc = 0.0;
    if (!nh_zero_count_for_larger_threshold(prev, &thr, &pzc))
        return NhReset::Undecidable;
    if (thr != h->zero_threshold) return NhReset::Yes;   // inside a live bucket
    if (h->zero_count < pzc) return NhReset::Yes;
    detail::NhSideView cv{}, pv{};
    for (int side = 0; side < 2; ++side) {                // positives, negatives
        const bool pos = (side == 0);
        const double* cs = pos ? h->pos : h->neg;
        const double* ps = pos ? prev->pos : prev->neg;
        const int32_t co = pos ? h->pos_offset : h->neg_offset;
        const int32_t po = pos ? prev->pos_offset : prev->neg_offset;
        const int32_t cn = pos ? h->n_pos : h->n_neg;
        const int32_t pn = pos ? prev->n_pos : prev->n_neg;
        if (!detail::nh_side_view(cs, co, cn, h->schema, h->schema,
                                  h->zero_threshold, &cv) ||
            !detail::nh_side_view(ps, po, pn, prev->schema, h->schema,
                                  h->zero_threshold, &pv))
            return NhReset::Undecidable;
        if (detail::nh_side_reset(&cv, &pv)) return NhReset::Yes;
    }
    return NhReset::No;
}

// `dst = src` brought onto the coarser exponential schema `target`
// (prometheus's CopyToSchema). Refuses rather than truncating.
inline bool nh_copy_to_schema(NativeHistogram* BOLT_RESTRICT dst,
                              const NativeHistogram* BOLT_RESTRICT src,
                              int32_t target) noexcept {
    assert(dst != nullptr && src != nullptr);
    assert(dst != src);
    *dst = *src;
    if (target == src->schema) return true;
    return nh_reduce_resolution(dst, target);
}

// prometheus's `FloatHistogram.Equals` — DATA equality, not mathematical
// equality: count/sum/zero_count are compared by BIT PATTERN (so NaN == NaN and
// +0 != -0), and the bucket LAYOUT must match, not merely the distribution.
// This tree's dense (offset, n) run is the direct analogue of upstream's single
// span, which is what `changes()` compares: its inputs are always samples as
// they were written, never a histogram this engine computed.
inline bool nh_equals(const NativeHistogram* a, const NativeHistogram* b) noexcept {
    assert(a != nullptr && b != nullptr);
    assert(a->n_pos <= k_nh_max_side && a->n_neg <= k_nh_max_side);
    assert(b->n_pos <= k_nh_max_side && b->n_neg <= k_nh_max_side);
    auto bits = [](double v) noexcept {
        uint64_t u = 0; std::memcpy(&u, &v, sizeof(u)); return u;
    };
    if (a->schema != b->schema) return false;
    if (bits(a->count) != bits(b->count)) return false;
    if (bits(a->sum) != bits(b->sum)) return false;
    if (nh_uses_custom(a) && !nh_custom_bounds_match(a, b)) return false;
    if (a->zero_threshold != b->zero_threshold) return false;
    if (bits(a->zero_count) != bits(b->zero_count)) return false;
    if (a->n_pos != b->n_pos || a->n_neg != b->n_neg) return false;
    if (a->n_pos != 0 && a->pos_offset != b->pos_offset) return false;
    if (a->n_neg != 0 && a->neg_offset != b->neg_offset) return false;
    for (int32_t i = 0; i < a->n_pos; ++i)              // bounded by n_pos
        if (bits(a->pos[i]) != bits(b->pos[i])) return false;
    for (int32_t i = 0; i < a->n_neg; ++i)              // bounded by n_neg
        if (bits(a->neg[i]) != bits(b->neg[i])) return false;
    return true;
}

// histogram_avg(h). NaN for an empty histogram (0/0), matching upstream.
inline double nh_avg(const NativeHistogram* h) noexcept {
    assert(h != nullptr);
    assert(!(h->count < 0.0));
    return h->sum / h->count;
}

// ---- trim operators (`h </ x`, `h >/ x`) -------------------------------
//
// PromQL's native-histogram bucket-slice operators. `h </ x` keeps the part
// of the distribution at or below `x`; `h >/ x` keeps the part above it. The
// result is a HISTOGRAM, not a float, and satisfies the identity
// native_histograms.test states for itself:
//
//     histogram_count(h </ x) == histogram_fraction(-Inf, x, h) * histogram_count(h)
//     histogram_count(h >/ x) == histogram_fraction(x, +Inf, h) * histogram_count(h)
//
// `count` is the retained mass. `sum` cannot be sliced — the original sum is
// a scalar over observations we no longer have — so it is RE-ESTIMATED from
// the retained buckets using the same representative Prometheus uses for
// histogram_stdvar: the geometric mean of an exponential bucket's bounds, the
// arithmetic midpoint of a custom bucket's, and the midpoint of the retained
// slice when a bucket is cut. That re-estimate differs from the recorded sum
// even for a full histogram, which is why a trim that removes NOTHING returns
// the input untouched rather than a recomputed near-miss.
namespace detail {

// The representative value of the interval (lo, hi]: arithmetic midpoint for
// a linear (custom-bucket / zero-bucket) interval, geometric mean for an
// exponential one. Mirrors promql/functions.go :: histogramVariance's choice.
inline double nh_trim_rep(double lo, double hi, bool linear) noexcept {
    assert(!(hi < lo));
    if (linear) return (lo + hi) / 2.0;
    double v = std::sqrt(lo * hi);
    if (hi < 0.0) v = -v;
    return v;
}

// One bucket's fate under a trim. `lo`/`hi` are its bounds (an infinity is a
// real possibility for the first and last CUSTOM buckets), `n` its count.
// Writes the retained count and, when that is non-zero, its representative.
//
// An UNBOUNDED bucket cannot be split — there is no interpolation across an
// infinite span — so upstream counts the whole of it on the unbounded side
// (nh_fraction has the same rule, `linear && b.lower == -inf -> b.count`).
// Kept whole, its retained slice is (-inf, min(hi,cut)] or (max(lo,cut), +inf],
// whose only finite end is the representative.
inline void nh_trim_bucket(double lo, double hi, double n, double cut,
                           bool keep_above, bool linear,
                           double* BOLT_RESTRICT kept,
                           double* BOLT_RESTRICT rep) noexcept {
    assert(kept != nullptr && rep != nullptr);
    assert(!std::isnan(cut));
    const double inf = std::numeric_limits<double>::infinity();
    *kept = 0.0; *rep = 0.0;
    if (keep_above) {
        if (lo == -inf) return;                          // all of it is below
        if (hi == inf) { *kept = n; *rep = (cut > lo) ? cut : lo; return; }
        if (cut <= lo) { *kept = n; *rep = nh_trim_rep(lo, hi, linear); return; }
        if (cut >= hi) return;
        NhBucket b; b.lower = lo; b.upper = hi; b.count = n;
        *kept = n * (1.0 - nh_fraction_below(&b, cut, linear));
        *rep  = nh_trim_rep(cut, hi, linear);
        return;
    }
    if (hi == inf) return;                               // all of it is above
    if (lo == -inf) { *kept = n; *rep = (cut < hi) ? cut : hi; return; }
    if (cut >= hi) { *kept = n; *rep = nh_trim_rep(lo, hi, linear); return; }
    if (cut <= lo) return;
    NhBucket b; b.lower = lo; b.upper = hi; b.count = n;
    *kept = n * nh_fraction_below(&b, cut, linear);
    *rep  = nh_trim_rep(lo, cut, linear);
}

// The zero bucket's interval. It is uniform (linear) over [-zt, zt], closed at
// zero on whichever side the histogram carries no buckets at all — the same
// bias rule nh_close_zero_bucket applies to quantile and fraction.
inline void nh_zero_span(const NativeHistogram* h, double* lo, double* hi) noexcept {
    assert(h != nullptr && lo != nullptr && hi != nullptr);
    *lo = -h->zero_threshold; *hi = h->zero_threshold;
    if (h->n_neg == 0 && h->n_pos > 0)      *lo = 0.0;
    else if (h->n_pos == 0 && h->n_neg > 0) *hi = 0.0;
}

}  // namespace detail

// `h </ cutoff` (keep_above == false) or `h >/ cutoff` (keep_above == true),
// applied in place. Returns false — leaving `h` untouched — for a NaN cutoff,
// which upstream has no case for and which no ordering can answer.
inline bool nh_trim(NativeHistogram* h, double cutoff, bool keep_above) noexcept {
    assert(h != nullptr);
    assert(h->n_pos <= k_nh_max_side && h->n_neg <= k_nh_max_side);
    if (std::isnan(cutoff)) return false;
    const double inf = std::numeric_limits<double>::infinity();
    // Nothing is on the far side of an infinity in the keeping direction.
    if (keep_above ? (cutoff == -inf) : (cutoff == inf)) return true;
    const bool custom = nh_uses_custom(h);
    const double* cv  = custom ? h->custom : nullptr;
    const int32_t ncv = custom ? h->n_custom : 0;
    const double  zt  = h->zero_threshold;

    NativeHistogram out = *h;
    double total = 0.0, sum = 0.0, c_sum = 0.0;
    bool removed = false;
    // Everything is trimmed away by an infinity in the discarding direction.
    const bool drop_all = keep_above ? (cutoff == inf) : (cutoff == -inf);

    for (int32_t j = 0; j < h->n_pos; ++j) {             // bounded by n_pos
        const int32_t idx = h->pos_offset + j;
        double lo = detail::nh_bound(idx - 1, h->schema, cv, ncv);
        double hi = detail::nh_bound(idx, h->schema, cv, ncv);
        if (lo > 0.0 && lo < zt) lo = zt;
        else if (hi < 0.0 && hi > -zt) hi = -zt;
        // A custom bucket unbounded BELOW but reaching above zero is closed at
        // zero, exactly as nh_close_zero_bucket closes a zero-straddling
        // bucket: the histogram records no negative observations, so its first
        // bucket really starts at 0. That is what makes `cbh </ 15` estimate
        // its lowest bucket at 2.5 (the midpoint of [0,5]) rather than at its
        // upper bound, and `cbh >/ 0` a no-op rather than a drop.
        //
        // The closure only asserts where the mass sits AT OR ABOVE zero, so it
        // cannot answer `>/ c` for a NEGATIVE c: that asks how much lies above
        // a point the closure says nothing about, inside a span nothing can
        // split. Upstream falls back to the unbounded rule there and the
        // bucket contributes nothing — `cbh_two_buckets_split_at_positive
        // >/ -10.0` keeps 100 of 101, dropping the whole [0,5] bucket, while
        // the same histogram `>/ 0.0` keeps all 101. Both are in
        // native_histograms.test; the pair is what fixes this asymmetry.
        if (lo == -inf && hi < inf && hi > 0.0 && (!keep_above || cutoff >= 0.0))
            lo = 0.0;
        double kept = 0.0, rep = 0.0;
        if (!drop_all)
            detail::nh_trim_bucket(lo, hi, h->pos[j], cutoff, keep_above,
                                   custom, &kept, &rep);
        if (kept != h->pos[j]) removed = true;
        out.pos[j] = kept;
        total += kept;
        detail::nh_kahan_inc(kept * rep, &sum, &c_sum);
    }
    for (int32_t j = 0; j < h->n_neg; ++j) {             // bounded by n_neg
        const int32_t idx = h->neg_offset + j;
        double lo = -detail::nh_bound(idx, h->schema, cv, ncv);
        double hi = -detail::nh_bound(idx - 1, h->schema, cv, ncv);
        if (hi < 0.0 && hi > -zt) hi = -zt;
        else if (lo > 0.0 && lo < zt) lo = zt;
        double kept = 0.0, rep = 0.0;
        if (!drop_all)
            detail::nh_trim_bucket(lo, hi, h->neg[j], cutoff, keep_above,
                                   custom, &kept, &rep);
        if (kept != h->neg[j]) removed = true;
        out.neg[j] = kept;
        total += kept;
        detail::nh_kahan_inc(kept * rep, &sum, &c_sum);
    }
    if (h->zero_count != 0.0) {
        double lo = 0.0, hi = 0.0;
        detail::nh_zero_span(h, &lo, &hi);
        double kept = 0.0, rep = 0.0;
        if (!drop_all)
            detail::nh_trim_bucket(lo, hi, h->zero_count, cutoff, keep_above,
                                   /*linear=*/true, &kept, &rep);
        if (kept != h->zero_count) removed = true;
        out.zero_count = kept;
        total += kept;
        detail::nh_kahan_inc(kept * rep, &sum, &c_sum);
    }
    // A trim that removed nothing is the identity, not a re-estimate: the
    // recorded sum is exact and the bucket-representative sum is not.
    if (!removed) return true;
    if (std::isfinite(sum)) sum += c_sum;
    out.count = total;
    out.sum   = sum;
    *h = out;
    assert(h->n_pos <= k_nh_max_side && h->n_neg <= k_nh_max_side);
    assert(!(h->count < 0.0));
    return true;
}

}  // namespace promql
}  // namespace bolt
