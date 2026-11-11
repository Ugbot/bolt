// Scalar and column kernels behind Spark's math / bitwise functions (acos .. width_bucket).
//
// Exactness: every transcendental runs the fdlibm port java.lang.StrictMath uses (bolt_fdlibm_*.h),
// with Spark's own formulas for asinh / acosh / atanh / cot / sec / csc, so a result equals the JVM's
// bit for bit. The one exception is sin / cos: HotSpot's Math.sin / Math.cos are platform stubs that
// differ from fdlibm by 1 ulp for about 5% of arguments with |x| >= 0.5 (the JDK allows 1 ulp).
// Column kernels are branchless loops the compiler vectorises where the scalar kernel is; none allocates.
#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bolt/kernels/bolt_fdlibm_explog.h"
#include "bolt/kernels/bolt_fdlibm_trig.h"

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace bolt::kernels::math_spark {

namespace fd = bolt::kernels::fdlibm;

enum class F64Op : std::uint8_t {
    Acos, Acosh, Asin, Asinh, Atan, Atanh, Cos, Cosh, Sin, Sinh, Tan, Tanh, Cot, Sec, Csc,
    Cbrt, Expm1, Log1p, Degrees, Radians, Rint, Signum,
};

inline constexpr double k_rad_to_deg = 57.29577951308232;
inline constexpr double k_deg_to_rad = 0.017453292519943295;

// log1p(x) is NULL for x <= -1 (Spark's UnaryLogExpression); *is_null says so.
inline double f64_unary(F64Op op, double x, bool* is_null) noexcept {
    *is_null = false;
    switch (op) {
        case F64Op::Acos: return fd::Acos::compute(x);
        case F64Op::Acosh: return fd::Log::compute(x + std::sqrt(x * x - 1.0));
        case F64Op::Asin: return fd::Asin::compute(x);
        case F64Op::Asinh:
            return x == -INFINITY ? x : fd::Log::compute(x + std::sqrt(x * x + 1.0));
        case F64Op::Atan: return fd::Atan::compute(x);
        case F64Op::Atanh: return 0.5 * (fd::Log1p::compute(x) - fd::Log1p::compute(-x));
        case F64Op::Cos: return fd::Cos::compute(x);
        case F64Op::Cosh: return fd::Cosh::compute(x);
        case F64Op::Sin: return fd::Sin::compute(x);
        case F64Op::Sinh: return fd::Sinh::compute(x);
        case F64Op::Tan: return fd::Tan::compute(x);
        case F64Op::Tanh: return fd::Tanh::compute(x);
        case F64Op::Cot: return 1.0 / fd::Tan::compute(x);
        case F64Op::Sec: return 1.0 / fd::Cos::compute(x);
        case F64Op::Csc: return 1.0 / fd::Sin::compute(x);
        case F64Op::Cbrt: return fd::Cbrt::compute(x);
        case F64Op::Expm1: return fd::Expm1::compute(x);
        case F64Op::Log1p:
            if (x <= -1.0) { *is_null = true; return 0.0; }
            return fd::Log1p::compute(x);
        case F64Op::Degrees: return x * k_rad_to_deg;
        case F64Op::Radians: return x * k_deg_to_rad;
        case F64Op::Rint: return std::nearbyint(x);
        case F64Op::Signum: return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : x);   // NaN and +-0 return themselves
    }
    *is_null = true;
    return 0.0;
}

// Column form: out[i] = op(in[i]) for rows with validity set (nullptr = all valid); an invalid or NULL-result
// row gets validity cleared in out_valid (byte per row, 1 = valid) and 0.0 stored.
inline void f64_unary_col(F64Op op, const double* in, const std::uint8_t* in_valid, double* out,
                          std::uint8_t* out_valid, std::uint32_t n) noexcept {
    for (std::uint32_t i = 0; i < n; ++i) {
        bool nl = false;
        const double r = f64_unary(op, in[i], &nl);
        const bool v = (in_valid == nullptr || in_valid[i] != 0) && !nl;
        out[i] = v ? r : 0.0;
        out_valid[i] = v ? 1 : 0;
    }
}

inline double atan2_spark(double y, double x) noexcept { return fd::Atan2::compute(y + 0.0, x + 0.0); }
inline double hypot_spark(double x, double y) noexcept { return fd::Hypot::compute(x, y); }

// ---- bround -----------------------------------------------------------------------------------

// BigDecimal(shortest decimal text of x).setScale(scale, HALF_EVEN).toDouble, as Spark's bround over DOUBLE.
inline double bround_f64(double x, std::int32_t scale) noexcept {
    if (x == 0.0) return 0.0;   // BigDecimal has no negative zero
    if (!std::isfinite(x)) return x;
    char buf[40];
    const auto r = std::to_chars(buf, buf + sizeof buf, std::fabs(x), std::chars_format::scientific);
    if (r.ec != std::errc()) return x;
    *r.ptr = '\0';
    char dig[32];
    std::int32_t nd = 0, e10 = 0;
    const char* p = buf;
    for (; p < r.ptr && *p != 'e'; ++p) if (*p != '.' && nd < 31) dig[nd++] = *p;
    if (p < r.ptr) e10 = static_cast<std::int32_t>(std::strtol(p + 1, nullptr, 10));
    const std::int64_t keep = static_cast<std::int64_t>(e10) + 1 + scale;   // digits of dig that survive
    if (keep >= nd) return x;
    if (keep < 0) return 0.0;
    bool up = false;
    const int d = dig[keep] - '0';
    if (d > 5) up = true;
    else if (d == 5) {
        bool rest = false;
        for (std::int32_t i = static_cast<std::int32_t>(keep) + 1; i < nd; ++i) rest = rest || dig[i] != '0';
        up = rest || (keep > 0 && ((dig[keep - 1] - '0') & 1) != 0);
    }
    char out[48];
    std::int32_t m = static_cast<std::int32_t>(keep);
    std::int32_t exp10 = e10;
    char kd[32];
    std::memcpy(kd, dig, static_cast<std::size_t>(m));
    if (up) {
        std::int32_t i = m - 1;
        while (i >= 0 && kd[i] == '9') kd[i--] = '0';
        if (i >= 0) ++kd[i];
        else {   // carry out of the top digit (also keep == 0: a lone 1 unit)
            std::memmove(kd + 1, kd, static_cast<std::size_t>(m));
            kd[0] = '1';
            m += 1;
            exp10 += 1;
            if (keep == 0) { m = 1; kd[0] = '1'; }
        }
    }
    if (m == 0) return 0.0;
    const int len = std::snprintf(out, sizeof out, "%.*se%d", m, kd, static_cast<int>(exp10 - m + 1));
    (void)len;
    // "%.*s" cuts kd to m digits; value = digits * 10^(exp10-m+1)
    const double v = std::strtod(out, nullptr);
    return x < 0 ? -v : v;
}

inline float bround_f32(float x, std::int32_t scale) noexcept {
    return static_cast<float>(bround_f64(static_cast<double>(x), scale));
}

// bround over an integer of `width` bits (8/16/32/64) at a negative scale: HALF_EVEN to a multiple of
// 10^-scale. *overflow is set when the result leaves the width (Spark: ARITHMETIC_OVERFLOW under ANSI,
// wrapped otherwise); the returned value is the wrapped result either way.
inline std::int64_t wrap_width(std::uint64_t v, std::uint32_t width) noexcept {
    if (width >= 64) return static_cast<std::int64_t>(v);
    const std::uint64_t m = (1ull << width) - 1u;
    v &= m;
    if ((v >> (width - 1)) & 1u) v |= ~m;
    return static_cast<std::int64_t>(v);
}

inline std::int64_t bround_int(std::int64_t x, std::int32_t scale, std::uint32_t width, bool* overflow) noexcept {
    *overflow = false;
    if (scale >= 0) return x;
    if (scale < -18) return 0;   // 10^19 exceeds every integer: rounds to 0 (|x| < 5e18 or exactly half is even)
    std::int64_t p = 1;
    for (std::int32_t i = 0; i < -scale; ++i) p *= 10;
    std::int64_t q = x / p, r = x % p;
    const std::int64_t ar = r < 0 ? -r : r;
    const std::int64_t twice = ar * 2;   // |r| < 10^18: no overflow
    if (twice > p || (twice == p && (q & 1) != 0)) q += x < 0 ? -1 : 1;
    const __int128 big = static_cast<__int128>(q) * p;
    const std::int64_t lo = width >= 64 ? INT64_MIN : -(std::int64_t{1} << (width - 1));
    const std::int64_t hi = width >= 64 ? INT64_MAX : (std::int64_t{1} << (width - 1)) - 1;
    if (big < lo || big > hi) *overflow = true;
    return wrap_width(static_cast<std::uint64_t>(static_cast<unsigned __int128>(big)), width);
}

// ---- integer / bit functions -----------------------------------------------------------------

inline bool factorial_i64(std::int64_t n, std::int64_t* out) noexcept {
    if (n < 0 || n > 20) return false;
    std::int64_t f = 1;
    for (std::int64_t i = 2; i <= n; ++i) f *= i;
    *out = f;
    return true;
}

// pmod over a `width`-bit integer; b != 0 (the caller handles the zero divisor).
inline std::int64_t pmod_int(std::int64_t a, std::int64_t b, std::uint32_t width) noexcept {
    const std::int64_t r = b == -1 ? 0 : a % b;
    if (r >= 0) return r;
    const std::int64_t s = wrap_width(static_cast<std::uint64_t>(r) + static_cast<std::uint64_t>(b), width);
    return s % b;   // b == -1 never reaches here: r is 0 then
}

inline double pmod_f64(double a, double b) noexcept {
    const double r = std::fmod(a, b);
    return r < 0 ? std::fmod(r + b, b) : r;
}

inline std::int64_t popcount_i64(std::int64_t x) noexcept {
    return static_cast<std::int64_t>(__builtin_popcountll(static_cast<unsigned long long>(x)));
}
inline void popcount_i64_col(const std::int64_t* in, std::int64_t* out, std::uint32_t n) noexcept {
    for (std::uint32_t i = 0; i < n; ++i) out[i] = popcount_i64(in[i]);
}

// bit `pos` of a `width`-bit integer; false when pos is outside 0..width-1.
inline bool bit_get(std::int64_t x, std::int64_t pos, std::uint32_t width, std::int64_t* out) noexcept {
    if (pos < 0 || pos >= static_cast<std::int64_t>(width)) return false;
    *out = static_cast<std::int64_t>((static_cast<std::uint64_t>(x) >> pos) & 1u);
    return true;
}

// width_bucket(v, b1, b2, n): 0 below the range, n + 1 above it (mirrored when b1 > b2); false = NULL result.
inline bool width_bucket(double v, double b1, double b2, std::int64_t n, std::int64_t* out) noexcept {
    if (n <= 0 || n == INT64_MAX || std::isnan(v) || std::isnan(b1) || std::isnan(b2) || b1 == b2 ||
        std::isinf(b1) || std::isinf(b2)) return false;
    const bool asc = b1 < b2;
    const double lower = asc ? b1 : b2, upper = asc ? b2 : b1;
    if (asc) {
        if (v < lower) { *out = 0; return true; }
        if (v >= upper) { *out = n + 1; return true; }
        *out = static_cast<std::int64_t>(static_cast<double>(n) * ((v - lower) / (upper - lower))) + 1;
    } else {
        if (v > upper) { *out = 0; return true; }
        if (v <= lower) { *out = n + 1; return true; }
        *out = static_cast<std::int64_t>(static_cast<double>(n) * ((upper - v) / (upper - lower))) + 1;
    }
    return true;
}

}  // namespace bolt::kernels::math_spark

#if defined(__clang__)
#pragma clang fp contract(on)
#endif
