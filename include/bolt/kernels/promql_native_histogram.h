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

// dst += src, for two histograms of the SAME schema (and, for custom buckets,
// the same custom_values) and the same zero threshold. Returns false — leaving
// `dst` untouched — for any other pair: mixing schemas needs a resolution
// reduction that changes bucket boundaries, and silently adding across
// incompatible boundaries would produce a confident wrong distribution.
inline bool nh_add(NativeHistogram* BOLT_RESTRICT dst,
                   const NativeHistogram* BOLT_RESTRICT src) noexcept {
    assert(dst != nullptr && src != nullptr);
    assert(dst != src);
    if (dst->schema != src->schema) return false;
    if (dst->zero_threshold != src->zero_threshold) return false;
    if (nh_uses_custom(dst)) {
        if (dst->n_custom != src->n_custom) return false;
        for (int32_t i = 0; i < dst->n_custom; ++i)   // bounded by n_custom
            if (dst->custom[i] != src->custom[i]) return false;
    }
    NativeHistogram out = *dst;
    if (src->n_pos > 0) {
        const int32_t lo = (out.n_pos == 0) ? src->pos_offset
            : (out.pos_offset < src->pos_offset ? out.pos_offset : src->pos_offset);
        const int32_t a_hi = out.pos_offset + out.n_pos;
        const int32_t b_hi = src->pos_offset + src->n_pos;
        const int32_t hi = (out.n_pos == 0) ? b_hi : (a_hi > b_hi ? a_hi : b_hi);
        if (!detail::nh_widen_side(out.pos, &out.pos_offset, &out.n_pos, lo, hi))
            return false;
        for (int32_t i = 0; i < src->n_pos; ++i)      // bounded by src n_pos
            out.pos[(src->pos_offset + i) - out.pos_offset] += src->pos[i];
    }
    if (src->n_neg > 0) {
        const int32_t lo = (out.n_neg == 0) ? src->neg_offset
            : (out.neg_offset < src->neg_offset ? out.neg_offset : src->neg_offset);
        const int32_t a_hi = out.neg_offset + out.n_neg;
        const int32_t b_hi = src->neg_offset + src->n_neg;
        const int32_t hi = (out.n_neg == 0) ? b_hi : (a_hi > b_hi ? a_hi : b_hi);
        if (!detail::nh_widen_side(out.neg, &out.neg_offset, &out.n_neg, lo, hi))
            return false;
        for (int32_t i = 0; i < src->n_neg; ++i)      // bounded by src n_neg
            out.neg[(src->neg_offset + i) - out.neg_offset] += src->neg[i];
    }
    out.count += src->count;
    out.sum += src->sum;
    out.zero_count += src->zero_count;
    *dst = out;
    assert(dst->n_pos <= k_nh_max_side && dst->n_neg <= k_nh_max_side);
    return true;
}

// histogram_avg(h). NaN for an empty histogram (0/0), matching upstream.
inline double nh_avg(const NativeHistogram* h) noexcept {
    assert(h != nullptr);
    assert(!(h->count < 0.0));
    return h->sum / h->count;
}

}  // namespace promql
}  // namespace bolt
