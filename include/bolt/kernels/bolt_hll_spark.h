// bolt_hll_spark.h -- Spark 4.0.1 HyperLogLog++ (approx_count_distinct), bit for bit.
//
// The sketch is bolt_hll.h's runtime-precision register array (one byte per
// register, same index / rank / max-merge); Spark packs ten 6-bit registers
// per long, which holds the same values. This header adds what Spark's
// HyperLogLogPlusPlusHelper adds on top:
//   * p = ceil(2 * log2(1.106 / rsd)), p >= 4 (hll_spark_precision);
//   * the hashed value: XxHash64Function seed 42 of the value, typed as Spark
//     types it (the caller hashes; bolt_hash_spark.h xx_hash_*);
//   * the estimate: Z^-1 summed register by register in index order, the
//     linear-counting thresholds and the k=6 nearest-neighbour bias
//     correction over Spark's tables for p < 19, Java Math.round.
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "bolt/kernels/bolt_hll.h"
#include "bolt/kernels/bolt_hll_spark_tables.h"

namespace bolt::kernels::hll {

inline constexpr uint64_t k_hll_spark_seed = 42u;
inline constexpr uint32_t k_hll_spark_knn = 6;
inline constexpr uint32_t k_hll_spark_bias_max_p = 18;

// Spark's p for a relative standard deviation; 0 when rsd yields p < 4 (Spark
// refuses) or the value is not finite / positive. May exceed k_hll_rt_max_p:
// the caller budgets it.
inline uint32_t hll_spark_precision(double rsd) noexcept {
    if (!(rsd > 0.0) || !std::isfinite(rsd)) return 0;
    const double p = std::ceil(2.0 * std::log(1.106 / rsd) / std::log(2.0));
    assert(!std::isnan(p));
    if (!(p >= static_cast<double>(k_hll_rt_min_p))) return 0;
    return p > 64.0 ? 64u : static_cast<uint32_t>(p);
}

// java.util.Arrays.binarySearch(double[], from, to, key), including its
// doubleToLongBits tie-break for -0.0 / NaN.
inline int32_t hll_java_binary_search(const double* a, uint32_t n, double key) noexcept {
    assert(a != nullptr && n > 0);
    int64_t lo = 0, hi = static_cast<int64_t>(n) - 1;
    while (lo <= hi) {
        const int64_t mid = static_cast<int64_t>(static_cast<uint64_t>(lo + hi) >> 1);
        const double v = a[mid];
        if (v < key) { lo = mid + 1; continue; }
        if (v > key) { hi = mid - 1; continue; }
        int64_t vb = 0, kb = 0;
        const double vv = std::isnan(v) ? std::numeric_limits<double>::quiet_NaN() : v;
        const double kk = std::isnan(key) ? std::numeric_limits<double>::quiet_NaN() : key;
        std::memcpy(&vb, &vv, 8);
        std::memcpy(&kb, &kk, 8);
        if (vb == kb) return static_cast<int32_t>(mid);
        if (vb < kb) lo = mid + 1; else hi = mid - 1;
    }
    return static_cast<int32_t>(-(lo + 1));
}

// HyperLogLogPlusPlusHelper.estimateBias.
inline double hll_spark_bias(uint32_t p, double e) noexcept {
    assert(p >= k_hll_rt_min_p && p <= k_hll_spark_bias_max_p);
    const double* est = spark_tables::k_raw[p - 4u];
    const double* bias = spark_tables::k_bias[p - 4u];
    const uint32_t n = spark_tables::k_len[p - 4u];
    assert(n >= k_hll_spark_knn);
    const int32_t ix = hll_java_binary_search(est, n, e);
    const int64_t nearest = ix < 0 ? -(static_cast<int64_t>(ix) + 1) : ix;
    const auto dist = [&](int64_t i) { const double d = e - est[i]; return d * d; };
    int64_t low = nearest - static_cast<int64_t>(k_hll_spark_knn) + 1;
    if (low < 0) low = 0;
    int64_t high = low + k_hll_spark_knn;
    if (high > n) high = n;
    while (high < n && dist(high) < dist(low)) { ++low; ++high; }   // bounded by n
    double sum = 0.0;
    for (int64_t i = low; i < high; ++i) sum += bias[i];
    return sum / static_cast<double>(high - low);
}

// Java Math.round(double) (floor(x + 1/2) computed exactly) for a finite,
// non-negative estimate.
inline int64_t hll_java_round(double x) noexcept {
    assert(!std::isnan(x) && x >= 0.0);
    const double f = std::floor(x);
    if (f >= 9.223372036854775807e18) return INT64_MAX;
    const int64_t r = static_cast<int64_t>(f);
    return (x - f) >= 0.5 ? r + 1 : r;
}

// HyperLogLogPlusPlusHelper.query over m = 2^p registers.
inline int64_t hll_spark_estimate(const uint8_t* regs, uint32_t p) noexcept {
    assert(regs != nullptr);
    assert(p >= k_hll_rt_min_p && p <= k_hll_rt_max_p);
    const uint32_t m = 1u << p;
    const double* inv = hll_inv_pow2_table();
    double z_inv = 0.0, v = 0.0;
    for (uint32_t i = 0; i < m; ++i) {   // Spark's order: one accumulator, index order
        const uint8_t r = regs[i] > 64u ? 64u : regs[i];
        z_inv += inv[r];
        if (r == 0) v += 1.0;
    }
    const double md = static_cast<double>(m);
    const double alpha = p == 4 ? 0.673 : p == 5 ? 0.697 : p == 6 ? 0.709 : 0.7213 / (1.0 + 1.079 / md);
    const double e = alpha * md * md / z_inv;
    const bool hllpp = p <= k_hll_spark_bias_max_p;
    const double corrected = (hllpp && e < 5.0 * md) ? e - hll_spark_bias(p, e) : e;
    double est = corrected;
    if (v > 0.0) {
        const double h = md * std::log(md / v);
        if ((hllpp && h <= spark_tables::k_thresholds[p - 4u]) || e <= 2.5 * md) est = h;
    }
    return hll_java_round(est);
}

}  // namespace bolt::kernels::hll
