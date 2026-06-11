// bolt/kernels/bolt_decimal64.h — Decimal64 (int64-mantissa exact decimal)
// lane kernels. W-DEC: a Decimal64 column is a plain int64 mantissa array
// plus the column's decimal_scale, so almost all of its math IS the int64
// SIMD kernel set (bolt_numeric.h add/sub/mul/broadcast/sum, branchless
// filters). This header holds only the conversions that have no int64
// equivalent.
//
// Tiger Style: noexcept, no allocation, BOLT_RESTRICT, >=2 asserts,
// <=70-line functions, branch-free inner loops.

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_decimal.h"

namespace bolt {
namespace kernels {
namespace dec64 {

// Widen int64 mantissas to Decimal128 (sign-extend the high limb). Used
// (a) at expression leaves of trees that did NOT prove the i64-lane
// certificate and (b) when feeding d128-only consumers. Branch-free;
// auto-vectorizes (two stores per row). Floor target: <= 0.6 ns/row.
inline void widen_to_d128(const int64_t* BOLT_RESTRICT a, int64_t n,
                          decimal::Decimal128* BOLT_RESTRICT out) noexcept {
    assert(a != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t v = a[i];
        out[i].lo = v;
        out[i].hi = v >> 63;   // arithmetic shift: 0 or -1
    }
}

// Exact scale-UP by 10^pow (pow in 0..18): out[i] = a[i] * 10^pow. The
// CALLER proved no overflow (plan-time interval certificate) — this
// kernel asserts the contract in debug and never checks in release.
// Floor target: <= 0.5 ns/row (single multiply by a hoisted constant).
inline constexpr int64_t k_dec64_pow10[19] = {
    1LL, 10LL, 100LL, 1000LL, 10000LL, 100000LL, 1000000LL, 10000000LL,
    100000000LL, 1000000000LL, 10000000000LL, 100000000000LL,
    1000000000000LL, 10000000000000LL, 100000000000000LL,
    1000000000000000LL, 10000000000000000LL, 100000000000000000LL,
    1000000000000000000LL,
};

inline void rescale_up(const int64_t* BOLT_RESTRICT a, int64_t n,
                       int pow, int64_t* BOLT_RESTRICT out) noexcept {
    assert(a != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(pow >= 0 && pow <= 18);
    const int64_t m = k_dec64_pow10[pow];
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] * m;
#if !defined(NDEBUG)
        // Overflow contract: the plan-time proof guarantees the product
        // fits; division re-check catches a violated certificate.
        assert(a[i] == 0 || out[i] / m == a[i]);
#endif
    }
}

// Scalar widen for per-row paths (Value construction at expression
// leaves). Kept here so the layout knowledge lives in ONE place.
BOLT_FORCE_INLINE decimal::Decimal128 widen_one(int64_t v) noexcept {
    decimal::Decimal128 r;
    r.lo = v;
    r.hi = v >> 63;
    return r;
}

// Narrow Decimal128 values to int64 mantissas. The CALLER proved every
// value fits (plan-time interval certificate) — debug-asserts the
// contract, never checks in release. Inverse of widen_to_d128.
// Floor target: <= 0.6 ns/row (strided load + store).
inline void narrow_from_d128(const decimal::Decimal128* BOLT_RESTRICT a,
                             int64_t n,
                             int64_t* BOLT_RESTRICT out) noexcept {
    assert(a != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    for (int64_t i = 0; i < n; ++i) {
        assert(((a[i].hi == 0 && a[i].lo >= 0) ||
                (a[i].hi == -1 && a[i].lo < 0)) &&
               "narrow_from_d128: caller's int64 proof violated");
        out[i] = a[i].lo;
    }
}

}  // namespace dec64
}  // namespace kernels
}  // namespace bolt
