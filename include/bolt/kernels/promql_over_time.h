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

// Prometheus quantile(q, values): rank = q*(n-1), linear interpolate.
// q < 0 => -Inf; q > 1 => +Inf; n==0 or NaN q => NaN. Mutates vals.
inline double promql_quantile(double* BOLT_RESTRICT vals, int64_t n,
                              double q) noexcept {
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
    return vals[lo] * (1.0 - weight) + vals[hi] * weight;
}

// Go fuses a*b+c into one multiply-add on arm64 and not on x86-64
// (GOAMD64=v1), so Prometheus's last bits depend on its build target. Match
// the Prometheus built for this target; -DBOLT_PROMQL_FUSED_MULADD=0/1 overrides.
#ifndef BOLT_PROMQL_FUSED_MULADD
#if defined(__aarch64__) || defined(_M_ARM64)
#define BOLT_PROMQL_FUSED_MULADD 1
#else
#define BOLT_PROMQL_FUSED_MULADD 0
#endif
#endif
inline constexpr bool kPromqlFusedMulAdd = BOLT_PROMQL_FUSED_MULADD != 0;

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

// promql_kahan_inc(a * b, ...) as Go compiles it: on arm64 the product feeds
// the sum and the compensation as fused multiply-adds.
inline void promql_kahan_inc_mul(double a, double b, double* sum, double* c) noexcept {
    assert(sum != nullptr && c != nullptr);
    if constexpr (!kPromqlFusedMulAdd) {
        promql_kahan_inc(a * b, sum, c);
        return;
    }
    const double inc = a * b;
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

// a * b + c, fused where Go fuses it.
inline double promql_mul_add(double a, double b, double c) noexcept {
    if constexpr (kPromqlFusedMulAdd) return std::fma(a, b, c);
    const double p = a * b;
    return p + c;
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

// Median of |x - median(x)|, NaN ordered first as Prometheus does. Mutates vals.
inline double promql_mad(double* BOLT_RESTRICT vals, int64_t n) noexcept {
    assert(vals != nullptr || n == 0);
    assert(n >= 0);
    if (n <= 0) return promql_no_sample();
    const double med = promql_quantile(vals, n, 0.5);
    for (int64_t i = 0; i < n; ++i)
        vals[i] = std::fabs(vals[i] - med);
    return promql_quantile(vals, n, 0.5);
}

}  // namespace promql
}  // namespace bolt
