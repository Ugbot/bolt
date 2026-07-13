// bolt/kernels/group_stats.h — holistic + two-input streaming statistics
// kernels for SQL group-by aggregates the engine was missing:
//   * exact MEDIAN / QUANTILE over a per-group value slice (sort-based),
//   * Pearson CORRELATION + COVARIANCE over two parallel value slices
//     (single-pass moments).
//
// These back chukonu's PhysicalHashAggregate holistic + 2-arg aggregate
// path. The operator collects each group's values into an arena-backed
// growable buffer (median) or folds a running moment struct (corr/covar);
// this header owns the math (Decision 19 — all compute lives in Bolt).
//
// Shape rules (Tiger Style): noexcept, >=2 asserts, bounded loops, no heap
// (caller supplies scratch), no exceptions/RTTI, explicit int sizes,
// BOLT_RESTRICT on data pointers.
//
// MEDIAN/QUANTILE are EXACT (full sort). An approximate t-digest path for
// very large groups is a tracked follow-up; the result of this kernel is
// the exact type-7 (linear-interpolation) quantile.

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_argsort.h"  // sort_indirect_i64_scratch

namespace bolt {
namespace kernels {
namespace group_stats {

// IEEE-754 double -> monotonic-ordered int64 key. Ascending integer order of
// the transformed keys equals ascending order of the doubles (NaN excluded):
// flip the sign bit for non-negatives, flip all bits for negatives. Lets us
// reuse the tested integer argsort instead of a separate float sort.
BOLT_FORCE_INLINE int64_t f64_sortable_key(double v) noexcept {
    std::uint64_t u = 0;
    static_assert(sizeof(u) == sizeof(v), "ieee754 double is 8 bytes");
    std::memcpy(&u, &v, sizeof(u));
    // mask = all-ones for negatives (top bit set), else just the sign bit.
    const std::uint64_t mask =
        (static_cast<std::uint64_t>(-static_cast<std::int64_t>(u >> 63))) |
        0x8000000000000000ULL;
    return static_cast<std::int64_t>(u ^ mask);
}

// Exact quantile (type-7 / linear interpolation — the NumPy/pandas/R-7
// default) over `vals[0..n)`. `q` is clamped to [0,1]. Sort-based: reuses
// bolt::kernels::sort_indirect_i64_scratch over the sortable-key transform.
// Caller supplies scratch (no heap allocation here):
//   key_scratch  : int64[>=n]  sortable-key transform of vals
//   perm         : int32[>=n]  output permutation (ascending)
//   sort_scratch : int32[>=n]  mergesort sibling buffer
// Returns 0.0 for n<=0; vals[0] for n==1.
inline double quantile_f64(const double* BOLT_RESTRICT vals, std::int64_t n,
                           double q,
                           std::int64_t* BOLT_RESTRICT key_scratch,
                           std::int32_t* BOLT_RESTRICT perm,
                           std::int32_t* BOLT_RESTRICT sort_scratch) noexcept {
    assert(vals != nullptr || n == 0);
    assert(n <= 0x7FFFFFFF && "quantile group too large for int32 argsort perm");
    if (n <= 0) return 0.0;
    if (q < 0.0) q = 0.0;
    if (q > 1.0) q = 1.0;
    if (n == 1) return vals[0];
    assert(key_scratch != nullptr && perm != nullptr && sort_scratch != nullptr);
    for (std::int64_t i = 0; i < n; ++i) key_scratch[i] = f64_sortable_key(vals[i]);
    sort_indirect_i64_scratch(perm, key_scratch, sort_scratch, n, /*ascending=*/true);
    const double h  = q * static_cast<double>(n - 1);
    const double hf = std::floor(h);
    std::int64_t lo = static_cast<std::int64_t>(hf);
    if (lo < 0)     lo = 0;
    if (lo > n - 1) lo = n - 1;
    const std::int64_t hi = (lo < n - 1) ? (lo + 1) : lo;
    const double frac = h - hf;
    const double vlo = vals[perm[lo]];
    const double vhi = vals[perm[hi]];
    return vlo + (vhi - vlo) * frac;
}

// MEDIAN == quantile 0.5.
inline double median_f64(const double* BOLT_RESTRICT vals, std::int64_t n,
                         std::int64_t* BOLT_RESTRICT key_scratch,
                         std::int32_t* BOLT_RESTRICT perm,
                         std::int32_t* BOLT_RESTRICT sort_scratch) noexcept {
    assert(vals != nullptr || n == 0);
    assert(n >= 0);
    return quantile_f64(vals, n, 0.5, key_scratch, perm, sort_scratch);
}

// ---------------------------------------------------------------------------
// Two-input streaming moments for Pearson correlation + covariance.
// Single pass over (x, y) pairs: n, Sx, Sy, Sxy, Sxx, Syy.
// ---------------------------------------------------------------------------
struct CorrMoments {
    double n;     // count of (x,y) pairs
    double sx;    // sum x
    double sy;    // sum y
    double sxy;   // sum x*y
    double sxx;   // sum x*x
    double syy;   // sum y*y
};

BOLT_FORCE_INLINE void corr_moments_init(CorrMoments* BOLT_RESTRICT m) noexcept {
    assert(m != nullptr);
    m->n = 0.0; m->sx = 0.0; m->sy = 0.0;
    m->sxy = 0.0; m->sxx = 0.0; m->syy = 0.0;
}

BOLT_FORCE_INLINE void corr_moments_update(CorrMoments* BOLT_RESTRICT m,
                                           double x, double y) noexcept {
    assert(m != nullptr);
    assert(m->n >= 0.0);
    m->n  += 1.0;
    m->sx += x;
    m->sy += y;
    m->sxy += x * y;
    m->sxx += x * x;
    m->syy += y * y;
}

// COVAR_POP = (Sxy - Sx*Sy/n) / n. Returns 0.0 for n<1.
inline double covar_pop_finalize(const CorrMoments* BOLT_RESTRICT m) noexcept {
    assert(m != nullptr);
    assert(m->n >= 0.0);
    if (m->n < 1.0) return 0.0;
    return (m->sxy - m->sx * m->sy / m->n) / m->n;
}

// COVAR_SAMP = (Sxy - Sx*Sy/n) / (n-1). Returns 0.0 for n<2.
inline double covar_samp_finalize(const CorrMoments* BOLT_RESTRICT m) noexcept {
    assert(m != nullptr);
    if (m->n < 2.0) return 0.0;
    return (m->sxy - m->sx * m->sy / m->n) / (m->n - 1.0);
}

// Single-input POPULATION variance/stddev via Welford's online algorithm.
// Reuses a CorrMoments' (n, sx, sxx) fields as Welford's (n, running mean,
// running M2) so VarPop/StddevPop need NO separate per-group state from
// Corr/Covar — a fresh corr_moments_init zeroes them (n=0, mean=0, M2=0). This
// is IDENTICAL to Prometheus's stdvar/stddev recurrence and, crucially,
// numerically stable: a zero-variance group (all inputs equal) yields M2
// EXACTLY 0. The raw-moment form (Sxx - Sx^2/n) does NOT — it leaves a ~1e-13
// residue that sqrt amplifies to ~1e-6, so it cannot be used for stddev. A NaN
// sample propagates (mean/M2 become NaN -> variance NaN).
BOLT_FORCE_INLINE void var_moments_update(CorrMoments* BOLT_RESTRICT m,
                                          double x) noexcept {
    assert(m != nullptr);
    assert(m->n >= 0.0);
    m->n += 1.0;
    const double delta = x - m->sx;      // sx := running mean
    m->sx += delta / m->n;
    m->sxx += delta * (x - m->sx);       // sxx := running M2 (>= 0)
}

// VAR_POP = M2 / n. Returns 0.0 for n<1 (moot: a live group always has n>=1;
// a single value yields exactly 0). A single value's M2 is exactly 0.
inline double var_pop_welford_finalize(const CorrMoments* BOLT_RESTRICT m) noexcept {
    assert(m != nullptr);
    assert(m->n >= 0.0);
    if (m->n < 1.0) return 0.0;
    return m->sxx / m->n;
}

// STDDEV_POP = sqrt(VAR_POP). M2 >= 0 by construction (Welford), so sqrt is
// always defined; a zero-variance group yields exactly 0.
inline double stddev_pop_welford_finalize(const CorrMoments* BOLT_RESTRICT m) noexcept {
    assert(m != nullptr);
    return std::sqrt(var_pop_welford_finalize(m));
}

// Pearson r = (n*Sxy - Sx*Sy) / sqrt((n*Sxx - Sx^2)(n*Syy - Sy^2)).
// Returns 0.0 for n<2 or a degenerate (zero-variance) input.
inline double corr_finalize(const CorrMoments* BOLT_RESTRICT m) noexcept {
    assert(m != nullptr);
    assert(m->n >= 0.0);
    if (m->n < 2.0) return 0.0;
    const double cov = m->n * m->sxy - m->sx * m->sy;
    const double vx  = m->n * m->sxx - m->sx * m->sx;
    const double vy  = m->n * m->syy - m->sy * m->sy;
    const double denom = std::sqrt(vx * vy);
    return (denom > 0.0) ? (cov / denom) : 0.0;
}

}  // namespace group_stats
}  // namespace kernels
}  // namespace bolt
