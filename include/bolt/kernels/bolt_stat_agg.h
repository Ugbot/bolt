// bolt/kernels/bolt_stat_agg.h -- Spark-exact statistical aggregate kernels.
//
//   * 4th-central-moment cell: skewness / kurtosis (Spark CentralMomentAgg:
//     incremental update and Pebay's pairwise merge, so partition-parallel
//     partials combine exactly).
//   * co-moment finalizers: covar_samp and regr_avgx/avgy/sxx/sxy/syy/slope/
//     intercept/r2 read the Welford co-moment cell the Corr/Covar aggregates
//     already fold (agg_f64::comoment_update): no second pass, no new state.
//   * percentile: Spark's exact percentile interpolation over a sorted slice.
//   * Greenwald-Khanna QuantileSummaries (Spark's percentile_approx): see
//     bolt_gk_spark.h (included below).
//
// Tiger Style: noexcept, no heap (the caller supplies every buffer), bounded
// loops, >= 2 asserts per non-trivial function, no exceptions / RTTI.
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

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
    if (m->n == 0.0) {
        // The first value: Spark's update from the empty cell has delta == delta_n == x, so M2 = x * (x - x) is
        // exactly 0 for a finite x and NaN for NaN / +-Inf; M3 and M4 are 0 unless x^2 / x^3 overflow (inf - inf).
        // Written out so no FMA contraction turns the exact zeros into rounding residue.
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const bool fin = std::isfinite(x);
        m->n = 1.0; m->sx = x;
        m->sxx = fin ? 0.0 : nan;
        m->sxy = (fin && std::isfinite(x * x)) ? 0.0 : nan;
        m->syy = (fin && std::isfinite(x * x) && std::isfinite(x * x * x)) ? 0.0 : nan;
        return;
    }
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

}  // namespace stat_agg
}  // namespace kernels
}  // namespace bolt

#include "bolt/kernels/bolt_gk_spark.h"
