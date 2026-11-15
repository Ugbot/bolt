// bolt/kernels/promql_over_time.h — PromQL *_over_time math kernels.
//
// Pure, allocation-free. The caller windows samples to Prometheus's
// half-open (mint, maxt] and passes the values (quantile/mad sort in place).
// Semantics match Prometheus promql/quantile.go and functions.go
// (stddevOverTime / stdvarOverTime / funcMadOverTime).
//
// Tiger Style: noexcept, >=2 asserts/fn, no allocation, no std:: containers.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/promql_rate.h"

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

// Insertion-sort `vals[0..n)` ascending. NaN sorts first, as Prometheus's
// vectorByValueHeap.Less does.
inline void promql_sort_values(double* BOLT_RESTRICT vals, int64_t n) noexcept {
    assert(vals != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t a = 1; a < n; ++a) {
        const double vk = vals[a];
        int64_t b = a - 1;
        while (b >= 0 && (std::isnan(vals[b]) ? false
                          : (std::isnan(vk) || vals[b] > vk))) {
            vals[b + 1] = vals[b];
            --b;
        }
        vals[b + 1] = vk;
    }
}

// Which Prometheus the kernels answer like where upstream semantics moved,
// and whether multiply-adds are fused. Default: latest release, unfused, so
// every CPU answers the same bits.
//   compat V3_5   quantile always interpolates (a lone -Inf gives NaN) and
//                 mad_over_time ignores NaN; latest has #19899 (v3.16, exact
//                 rank answers the sample) and #19040 (v3.14, mad NaN -> NaN).
//                 #20180 (equal neighbours skip interpolation) is unreleased.
//   fused_muladd  Go's arm64 contraction of a*b+c into one fma; only for
//                 bit parity with an arm64 Prometheus build.
enum class PromqlCompat : uint8_t { Latest = 0, V3_5 = 1 };

struct PromqlNumerics {
    PromqlCompat compat = PromqlCompat::Latest;
    bool fused_muladd = false;
};

// a * b + c, fused only in fused mode; the separate statements keep the
// compiler from contracting it.
inline double promql_mul_add(double a, double b, double c,
                             const PromqlNumerics& num) noexcept {
    if (num.fused_muladd) return std::fma(a, b, c);
    const double p = a * b;
    return p + c;
}

// Prometheus quantile(q, values): rank = q*(n-1), linear interpolate.
// q < 0 => -Inf; q > 1 => +Inf; n==0 or NaN q => NaN. Mutates vals.
inline double promql_quantile(double* BOLT_RESTRICT vals, int64_t n, double q,
                              const PromqlNumerics& num = {}) noexcept {
    assert(vals != nullptr || n == 0);
    assert(n >= 0);
    if (n <= 0) return promql_no_sample();
    // A NaN quantile is an invalid ARGUMENT: Prometheus annotates and still
    // emits a sample whose value is NaN, so this is a value, not an absence.
    if (std::isnan(q)) return promql_nan();
    if (q < 0.0) return -std::numeric_limits<double>::infinity();
    if (q > 1.0) return  std::numeric_limits<double>::infinity();
    promql_sort_values(vals, n);
    const double rank = q * static_cast<double>(n - 1);
    const double lo_d = std::floor(rank);
    const int64_t lo = static_cast<int64_t>(lo_d);
    int64_t hi = lo + 1;
    if (hi >= n) hi = n - 1;
    const double weight = rank - lo_d;
    if (weight == 0.0 && num.compat != PromqlCompat::V3_5) return vals[lo];
    const double low_part = vals[lo] * (1.0 - weight);
    return promql_mul_add(vals[hi], weight, low_part, num);
}

// Prometheus kahanSumInc: Neumaier-compensated sum + inc.
inline void promql_kahan_inc(double inc, double* sum, double* c) noexcept {
    assert(sum != nullptr && c != nullptr);
    const double t = *sum + inc;
    if (std::isinf(t)) {
        *c = 0.0;
    } else if (std::fabs(*sum) >= std::fabs(inc)) {
        *c += (*sum - t) + inc;
    } else {
        *c += (inc - t) + *sum;
    }
    *sum = t;
}

// promql_kahan_inc(a * b, ...). Fused mode feeds the product to the sum and
// the compensation as fmas, as Go compiles it on arm64.
inline void promql_kahan_inc_mul(double a, double b, double* sum, double* c,
                                 const PromqlNumerics& num) noexcept {
    assert(sum != nullptr && c != nullptr);
    const double inc = a * b;
    if (!num.fused_muladd) {
        promql_kahan_inc(inc, sum, c);
        return;
    }
    const double t = std::fma(a, b, *sum);
    if (std::isinf(t)) {
        *c = 0.0;
    } else if (std::fabs(*sum) >= std::fabs(inc)) {
        *c += std::fma(a, b, *sum - t);
    } else {
        *c += std::fma(a, b, -t) + *sum;
    }
    *sum = t;
}

// Population variance exactly as Prometheus varianceOverTime: Welford with
// Kahan-compensated mean and M2. NaN poisons.
inline double promql_stdvar(const double* BOLT_RESTRICT vals, int64_t n) noexcept {
    assert(vals != nullptr || n == 0);
    assert(n >= 0);
    if (n <= 0) return promql_no_sample();
    double mean = 0.0, c_mean = 0.0, aux = 0.0, c_aux = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double v = vals[i];
        const double count = static_cast<double>(i + 1);
        const double delta = v - (mean + c_mean);
        promql_kahan_inc(delta / count, &mean, &c_mean);
        promql_kahan_inc(delta * (v - (mean + c_mean)), &aux, &c_aux);
    }
    return (aux + c_aux) / static_cast<double>(n);
}

inline double promql_stddev(const double* BOLT_RESTRICT vals, int64_t n) noexcept {
    const double v = promql_stdvar(vals, n);
    return std::isnan(v) ? v : std::sqrt(v);
}

// Median of |x - median(x)|. Latest: any NaN => NaN (#19040); V3_5: NaN
// sorts first and takes part in both medians. Mutates vals.
inline double promql_mad(double* BOLT_RESTRICT vals, int64_t n,
                         const PromqlNumerics& num = {}) noexcept {
    assert(vals != nullptr || n == 0);
    assert(n >= 0);
    if (n <= 0) return promql_no_sample();
    if (num.compat != PromqlCompat::V3_5) {
        for (int64_t i = 0; i < n; ++i)
            if (std::isnan(vals[i])) return promql_nan();  // a NaN VALUE, emitted
    }
    const double med = promql_quantile(vals, n, 0.5, num);
    for (int64_t i = 0; i < n; ++i)
        vals[i] = std::fabs(vals[i] - med);
    return promql_quantile(vals, n, 0.5, num);
}

// ---------------------------------------------------------------------------
// Simple *_over_time reductions over one frame's float samples (ascending ts).
// These are the Prometheus-exact folds the engine's range functions share:
// Neumaier-compensated sum, the overflow-safe incremental mean, NaN-ignoring
// min / max, and the sample timestamp of the extreme (latest wins a tie).
// ---------------------------------------------------------------------------

// One Neumaier step: sum += v with the lost low bits kept in *c.
inline void promql_neumaier_inc(double* sum, double* c, double v) noexcept {
    assert(sum != nullptr && c != nullptr);
    const double t = *sum + v;
    if (std::fabs(*sum) >= std::fabs(v)) *c += (*sum - t) + v;
    else                                 *c += (v - t) + *sum;
    *sum = t;
}

// An overflowed sum stays +-Inf; adding the compensation would make it NaN.
inline double promql_neumaier_result(double sum, double c) noexcept {
    return std::isinf(sum) ? sum : sum + c;
}

// sum_over_time. n <= 0 is no sample.
inline double promql_sum_over_time(const double* BOLT_RESTRICT vals, int64_t n) noexcept {
    assert(vals != nullptr || n == 0);
    assert(n >= 0);
    if (n <= 0) return promql_no_sample();
    double sum = 0.0, c = 0.0;
    for (int64_t i = 0; i < n; ++i) promql_neumaier_inc(&sum, &c, vals[i]);
    return promql_neumaier_result(sum, c);
}

// Prometheus's incremental mean after `count` samples (this one included),
// Neumaier-compensated; once the mean is +-Inf a finite or same-sign-Inf
// sample keeps it. *c is the running compensation: mean = returned + *c.
inline double promql_mean_inc(double mean, double* c, int64_t count, double v) noexcept {
    assert(c != nullptr && count > 0);
    if (std::isinf(mean)) {
        if (std::isinf(v) && std::signbit(v) == std::signbit(mean)) return mean;
        if (!std::isinf(v) && !std::isnan(v)) return mean;
    }
    const double fc = static_cast<double>(count);
    const double q = static_cast<double>(count - 1) / fc;
    const double base = q * mean, inc = v / fc;
    double bc = q * (*c);
    const double t = base + inc;
    if (!std::isfinite(t)) { *c = 0.0; return t; }
    bc += (std::fabs(base) >= std::fabs(inc)) ? (base - t) + inc : (inc - t) + base;
    *c = bc;
    return t;
}

// avg_over_time: the compensated sum / n while the sum is finite, else the
// incremental mean (an overflowing sum such as 4 x 9.99e307 still averages).
inline double promql_avg_over_time(const double* BOLT_RESTRICT vals, int64_t n) noexcept {
    assert(vals != nullptr || n == 0);
    assert(n >= 0);
    if (n <= 0) return promql_no_sample();
    double sum = 0.0, sc = 0.0, mean = 0.0, mc = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        promql_neumaier_inc(&sum, &sc, vals[i]);
        mean = promql_mean_inc(mean, &mc, i + 1, vals[i]);
    }
    const double s = promql_neumaier_result(sum, sc);
    const double m = mean + mc;
    if (std::isnan(s) || std::isnan(m)) return promql_nan();
    return std::isfinite(s) ? s / static_cast<double>(n) : m;
}

// min_over_time / max_over_time: NaN never wins; an all-NaN frame is NaN.
// The first of equal values keeps its sign of zero.
inline double promql_min_over_time(const double* BOLT_RESTRICT vals, int64_t n) noexcept {
    assert(vals != nullptr || n == 0);
    assert(n >= 0);
    if (n <= 0) return promql_no_sample();
    bool any = false;
    double m = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        if (std::isnan(vals[i])) continue;
        if (!any || vals[i] < m) m = vals[i];
        any = true;
    }
    return any ? m : promql_nan();
}

inline double promql_max_over_time(const double* BOLT_RESTRICT vals, int64_t n) noexcept {
    assert(vals != nullptr || n == 0);
    assert(n >= 0);
    if (n <= 0) return promql_no_sample();
    bool any = false;
    double m = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        if (std::isnan(vals[i])) continue;
        if (!any || vals[i] > m) m = vals[i];
        any = true;
    }
    return any ? m : promql_nan();
}

// ts_of_min / ts_of_max _over_time: the timestamp, in seconds, of the extreme
// sample. The LATEST sample wins a tie; NaN never wins and an all-NaN frame
// answers the last sample's timestamp.
inline double promql_ts_of_extreme_over_time(const int64_t* BOLT_RESTRICT ts,
                                             const double* BOLT_RESTRICT vals, int64_t n,
                                             bool want_min, double ts_per_second) noexcept {
    assert((ts != nullptr && vals != nullptr) || n == 0);
    assert(ts_per_second > 0.0);
    if (n <= 0) return promql_no_sample();
    bool seen = false;
    double best = 0.0;
    int64_t best_ts = ts[n - 1];
    for (int64_t i = 0; i < n; ++i) {
        if (std::isnan(vals[i])) continue;
        if (!seen || (want_min ? (vals[i] <= best) : (vals[i] >= best))) {
            best = vals[i];
            best_ts = ts[i];
            seen = true;
        }
    }
    return static_cast<double>(best_ts) / ts_per_second;
}

// ts_of_last / ts_of_first _over_time.
inline double promql_ts_of_edge_over_time(const int64_t* BOLT_RESTRICT ts, int64_t n,
                                          bool last, double ts_per_second) noexcept {
    assert(ts != nullptr || n == 0);
    assert(ts_per_second > 0.0);
    if (n <= 0) return promql_no_sample();
    return static_cast<double>(last ? ts[n - 1] : ts[0]) / ts_per_second;
}

}  // namespace promql
}  // namespace bolt

#if defined(__clang__)
#pragma float_control(pop)
#endif
