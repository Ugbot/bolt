// bolt/kernels/bolt_stat_agg.h -- Spark-exact statistical aggregate kernels.
//
//   * 4th-central-moment cell: skewness / kurtosis (Spark CentralMomentAgg:
//     incremental update and Pebay's pairwise merge, so partition-parallel
//     partials combine exactly).
//   * co-moment finalizers: covar_samp and regr_avgx/avgy/sxx/sxy/syy/slope/
//     intercept/r2 read the Welford co-moment cell the Corr/Covar aggregates
//     already fold (agg_f64::comoment_update): no second pass, no new state.
//   * percentile: Spark's exact percentile interpolation over a sorted slice.
//   * Greenwald-Khanna QuantileSummaries (Spark's percentile_approx), the same
//     insert / head-buffer / compress / query rules as Spark's
//     QuantileSummaries, over caller-supplied scratch.
//
// Tiger Style: noexcept, no heap (the caller supplies every buffer), bounded
// loops, >= 2 asserts per non-trivial function, no exceptions / RTTI.
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"
#include "bolt/kernels/group_stats.h"

namespace bolt {
namespace kernels {
namespace stat_agg {

using group_stats::CorrMoments;

// ---------------------------------------------------------------------------
// 4th central moment. The CorrMoments cell is reused field for field:
//   n = count, sx = mean, sxx = M2, sxy = M3, syy = M4 (sy unused).
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE void cm4_update(CorrMoments* BOLT_RESTRICT m, double x) noexcept {
    assert(m != nullptr);
    assert(m->n >= 0.0);
    const double n1 = m->n + 1.0;
    const double delta = x - m->sx;
    const double delta_n = delta / n1;
    const double delta2 = delta * delta;
    const double delta_n2 = delta_n * delta_n;
    const double m2 = m->sxx + delta * (delta - delta_n);
    const double m3 = m->sxy - 3.0 * delta_n * m2 + delta * (delta2 - delta_n2);
    const double m4 = m->syy - 4.0 * delta_n * m3 - 6.0 * delta_n2 * m2 +
                      delta * (delta * delta2 - delta_n * delta_n2);
    m->n = n1;
    m->sx += delta_n;
    m->sxx = m2;
    m->sxy = m3;
    m->syy = m4;
}

// Merge a partial over a disjoint row set into `d` (Pebay 2008).
inline void cm4_combine(CorrMoments* BOLT_RESTRICT d,
                        const CorrMoments* BOLT_RESTRICT s) noexcept {
    assert(d != nullptr && s != nullptr);
    assert(d->n >= 0.0 && s->n >= 0.0);
    if (s->n == 0.0) return;
    if (d->n == 0.0) { *d = *s; return; }
    const double n1 = d->n, n2 = s->n, n = n1 + n2;
    const double delta = s->sx - d->sx;
    const double dn = delta / n;
    const double dn2 = dn * dn;
    const double m2 = d->sxx + s->sxx + delta * dn * n1 * n2;
    const double m3 = d->sxy + s->sxy + dn2 * delta * n1 * n2 * (n1 - n2) +
                      3.0 * dn * (n1 * s->sxx - n2 * d->sxx);
    const double m4 = d->syy + s->syy +
                      dn2 * dn * delta * n1 * n2 * (n1 * n1 - n1 * n2 + n2 * n2) +
                      6.0 * dn2 * (n1 * n1 * s->sxx + n2 * n2 * d->sxx) +
                      4.0 * dn * (n1 * s->sxy - n2 * d->sxy);
    d->sx += dn * n2;
    d->sxx = m2;
    d->sxy = m3;
    d->syy = m4;
    d->n = n;
}

// skewness = sqrt(n) * M3 / sqrt(M2^3); NULL (false) when n < 1 or M2 == 0
// (Spark 4: spark.sql.legacy.statisticalAggregate=false). NaN propagates.
inline bool skewness_finalize(const CorrMoments* BOLT_RESTRICT m,
                              double* BOLT_RESTRICT out) noexcept {
    assert(m != nullptr && out != nullptr);
    if (m->n < 1.0 || m->sxx == 0.0) return false;
    *out = std::sqrt(m->n) * m->sxy / std::sqrt(m->sxx * m->sxx * m->sxx);
    return true;
}

// kurtosis (excess) = n * M4 / M2^2 - 3; NULL as for skewness.
inline bool kurtosis_finalize(const CorrMoments* BOLT_RESTRICT m,
                              double* BOLT_RESTRICT out) noexcept {
    assert(m != nullptr && out != nullptr);
    if (m->n < 1.0 || m->sxx == 0.0) return false;
    *out = m->n * m->syy / (m->sxx * m->sxx) - 3.0;
    return true;
}

// ---------------------------------------------------------------------------
// Co-moment finalizers over the Welford cell (comoment_update(x, y)):
//   n, sx = mean_x, sy = mean_y, sxy = C_xy, sxx = M2_x, syy = M2_y.
// Spark regr_f(y, x): the cell's x is Spark's second argument.
// ---------------------------------------------------------------------------
enum class RegrFn : std::uint8_t {
    CovarSamp = 0, AvgX, AvgY, Sxx, Syy, Sxy, Slope, Intercept, R2, kCount,
};

// false = SQL NULL (no pair, or the quantity is undefined).
inline bool regr_finalize(RegrFn f, const CorrMoments* BOLT_RESTRICT m,
                          double* BOLT_RESTRICT out) noexcept {
    assert(m != nullptr && out != nullptr);
    assert(static_cast<std::uint8_t>(f) < static_cast<std::uint8_t>(RegrFn::kCount));
    if (m->n < 1.0) return false;
    switch (f) {
        case RegrFn::CovarSamp:
            if (m->n < 2.0) return false;
            *out = m->sxy / (m->n - 1.0);
            return true;
        case RegrFn::AvgX: *out = m->sx; return true;
        case RegrFn::AvgY: *out = m->sy; return true;
        case RegrFn::Sxx:  *out = m->sxx; return true;
        case RegrFn::Syy:  *out = m->syy; return true;
        case RegrFn::Sxy:  *out = m->sxy; return true;
        case RegrFn::Slope:
            if (m->sxx == 0.0) return false;
            *out = m->sxy / m->sxx;
            return true;
        case RegrFn::Intercept:
            if (m->sxx == 0.0) return false;
            *out = m->sy - (m->sxy / m->sxx) * m->sx;
            return true;
        case RegrFn::R2:
            // Spark: NULL when the first argument (y) has no variance, 1.0
            // when only x has none, else the squared Pearson correlation.
            if (m->syy == 0.0) return false;
            if (m->sxx == 0.0) { *out = 1.0; return true; }
            *out = m->sxy * m->sxy / (m->sxx * m->syy);
            return true;
        case RegrFn::kCount: break;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Exact percentile as Spark's `percentile`: position = (n-1) * p over the
// ascending values; between two ranks, (hi - pos) * v[lo] + (pos - lo) * v[hi].
// Sort scratch as group_stats::quantile_f64 (NaN sorts last). n <= 0: 0.0.
// ---------------------------------------------------------------------------
inline double percentile_spark_f64(const double* BOLT_RESTRICT vals, std::int64_t n, double p,
                                   std::int64_t* BOLT_RESTRICT key_scratch,
                                   std::int32_t* BOLT_RESTRICT perm,
                                   std::int32_t* BOLT_RESTRICT sort_scratch) noexcept {
    assert(vals != nullptr || n == 0);
    assert(n <= 0x7FFFFFFF);
    if (n <= 0) return 0.0;
    if (n == 1) return vals[0];
    assert(key_scratch != nullptr && perm != nullptr && sort_scratch != nullptr);
    for (std::int64_t i = 0; i < n; ++i) key_scratch[i] = group_stats::f64_sortable_key(vals[i]);
    sort_indirect_i64_scratch(perm, key_scratch, sort_scratch, n, /*ascending=*/true);
    const double position = static_cast<double>(n - 1) * p;
    const std::int64_t lo = static_cast<std::int64_t>(std::floor(position));
    const std::int64_t hi = static_cast<std::int64_t>(std::ceil(position));
    const double vlo = vals[perm[lo]];
    if (hi == lo) return vlo;
    const double vhi = vals[perm[hi]];
    return (static_cast<double>(hi) - position) * vlo + (position - static_cast<double>(lo)) * vhi;
}

// ---------------------------------------------------------------------------
// Greenwald-Khanna quantile summary (Spark QuantileSummaries). One value
// stream in arrival order -> a compressed summary -> quantile queries.
// ---------------------------------------------------------------------------
struct GkStat {
    double       v;
    std::int64_t g;
    std::int64_t delta;
};

struct GkWork {
    GkStat*        a;           // two sample arrays, each `cap` entries
    GkStat*        b;
    std::uint32_t  cap;         // >= threshold + head_size
    std::uint32_t  head_size;   // Spark defaultHeadSize (50000)
    std::uint32_t  threshold;   // Spark defaultCompressThreshold (10000)
    double*        head;        // head_size values
    std::int64_t*  key;         // head_size
    std::int32_t*  perm;        // head_size
    std::int32_t*  tmp;         // head_size
};

constexpr std::uint32_t k_gk_head_size = 50000;
constexpr std::uint32_t k_gk_compress_threshold = 10000;

struct GkState {
    GkStat*        cur;
    std::uint32_t  len;
    std::int64_t   count;
    double         rel_err;
};

// Sort the head buffer and insert it into the samples (insertHeadBuffer).
inline void gk_flush(GkWork* w, GkState* s, std::uint32_t h) noexcept {
    assert(w != nullptr && s != nullptr);
    assert(h <= w->head_size);
    if (h == 0) return;
    for (std::uint32_t i = 0; i < h; ++i) w->key[i] = group_stats::f64_sortable_key(w->head[i]);
    if (h > 1) sort_indirect_i64_scratch(w->perm, w->key, w->tmp, h, /*ascending=*/true);
    else w->perm[0] = 0;
    GkStat* out = (s->cur == w->a) ? w->b : w->a;
    std::uint32_t o = 0, si = 0;
    std::int64_t cc = s->count;
    for (std::uint32_t oi = 0; oi < h; ++oi) {
        const double cs = w->head[w->perm[oi]];
        while (si < s->len && s->cur[si].v <= cs) out[o++] = s->cur[si++];
        ++cc;
        const bool edge = (o == 0) || (si == s->len && oi == h - 1);
        const std::int64_t delta =
            edge ? 0 : static_cast<std::int64_t>(std::floor(2.0 * s->rel_err * static_cast<double>(cc)));
        out[o++] = GkStat{cs, 1, delta};
        assert(o <= w->cap);
    }
    while (si < s->len) out[o++] = s->cur[si++];
    assert(o <= w->cap);
    s->cur = out;
    s->len = o;
    s->count = cc;
}

// compressImmut: merge neighbours while g + g' + delta' < mergeThreshold.
inline void gk_compress(GkWork* w, GkState* s) noexcept {
    assert(w != nullptr && s != nullptr);
    if (s->len == 0) return;
    const double merge_threshold = 2.0 * s->rel_err * static_cast<double>(s->count);
    GkStat* out = (s->cur == w->a) ? w->b : w->a;
    const GkStat* in = s->cur;
    std::uint32_t o = 0;            // filled in reverse, then reversed in place
    GkStat head = in[s->len - 1];
    for (std::int64_t i = static_cast<std::int64_t>(s->len) - 2; i >= 1; --i) {
        const GkStat s1 = in[i];
        if (static_cast<double>(s1.g + head.g + head.delta) < merge_threshold) {
            head.g += s1.g;
        } else {
            out[o++] = head;
            head = s1;
        }
    }
    out[o++] = head;
    if (in[0].v <= head.v && s->len > 1) out[o++] = in[0];
    for (std::uint32_t i = 0, j = o - 1; i < j; ++i, --j) { const GkStat t = out[i]; out[i] = out[j]; out[j] = t; }
    s->cur = out;
    s->len = o;
}

// Feed `n` values (arrival order). Returns false when `w` is too small.
inline bool gk_build(const double* BOLT_RESTRICT vals, std::int64_t n, double rel_err,
                     GkWork* w, GkState* s) noexcept {
    assert(w != nullptr && s != nullptr);
    assert(vals != nullptr || n == 0);
    if (w->cap < w->threshold + w->head_size) return false;
    s->cur = w->a; s->len = 0; s->count = 0; s->rel_err = rel_err;
    std::uint32_t h = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        w->head[h++] = vals[i];
        if (h >= w->head_size) {
            gk_flush(w, s, h);
            h = 0;
            if (s->len >= w->threshold) gk_compress(w, s);
        }
    }
    gk_flush(w, s, h);
    gk_compress(w, s);
    return true;
}

// Spark QuantileSummaries.query over a built summary; len must be > 0.
inline double gk_query(const GkState* BOLT_RESTRICT s, double q) noexcept {
    assert(s != nullptr && s->len > 0);
    assert(q >= 0.0 && q <= 1.0);
    if (q <= s->rel_err) return s->cur[0].v;
    if (q >= 1.0 - s->rel_err) return s->cur[s->len - 1].v;
    const std::int64_t rank = static_cast<std::int64_t>(std::ceil(q * static_cast<double>(s->count)));
    const double target = s->rel_err * static_cast<double>(s->count);
    std::int64_t min_rank = 0;
    for (std::uint32_t i = 0; i + 1 < s->len; ++i) {
        min_rank += s->cur[i].g;
        const std::int64_t max_rank = min_rank + s->cur[i].delta;
        if (static_cast<double>(max_rank) - target <= static_cast<double>(rank) &&
            static_cast<double>(rank) <= static_cast<double>(min_rank) + target)
            return s->cur[i].v;
    }
    return s->cur[s->len - 1].v;
}

}  // namespace stat_agg
}  // namespace kernels
}  // namespace bolt
