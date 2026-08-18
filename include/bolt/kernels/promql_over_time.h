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

// Insertion-sort `vals[0..n)` ascending. NaN sorts to the end (Prometheus).
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
    if (n <= 0 || std::isnan(q)) return promql_nan();
    if (q < 0.0) return -std::numeric_limits<double>::infinity();
    if (q > 1.0) return  std::numeric_limits<double>::infinity();
    promql_sort_values(vals, n);
    if (n == 1) return vals[0];
    const double rank = q * static_cast<double>(n - 1);
    const double lo_d = std::floor(rank);
    const int64_t lo = static_cast<int64_t>(lo_d);
    int64_t hi = lo + 1;
    if (hi >= n) hi = n - 1;
    const double weight = rank - lo_d;
    return vals[lo] * (1.0 - weight) + vals[hi] * weight;
}

// Population variance (Welford / Prometheus stdvarOverTime). NaN poisons.
inline double promql_stdvar(const double* BOLT_RESTRICT vals, int64_t n) noexcept {
    assert(vals != nullptr || n == 0);
    assert(n >= 0);
    if (n <= 0) return promql_nan();
    double mean = 0.0, m2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double v = vals[i];
        const double count = static_cast<double>(i + 1);
        const double delta = v - mean;
        mean += delta / count;
        m2 += delta * (v - mean);
    }
    return m2 / static_cast<double>(n);
}

inline double promql_stddev(const double* BOLT_RESTRICT vals, int64_t n) noexcept {
    const double v = promql_stdvar(vals, n);
    return std::isnan(v) ? v : std::sqrt(v);
}

// Median of |x - median(x)|. Any NaN => NaN. Mutates vals.
inline double promql_mad(double* BOLT_RESTRICT vals, int64_t n) noexcept {
    assert(vals != nullptr || n == 0);
    assert(n >= 0);
    if (n <= 0) return promql_nan();
    for (int64_t i = 0; i < n; ++i)
        if (std::isnan(vals[i])) return promql_nan();
    const double med = promql_quantile(vals, n, 0.5);
    for (int64_t i = 0; i < n; ++i)
        vals[i] = std::fabs(vals[i] - med);
    return promql_quantile(vals, n, 0.5);
}

}  // namespace promql
}  // namespace bolt
