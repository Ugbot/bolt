// bolt/kernels/bolt_float_chars.h — locale-independent double <-> text with
// std::to_chars / std::from_chars (plain / general format) semantics.
//
// Apple gates the floating-point <charconv> overloads behind macOS 13.3
// (to_chars) and 26.0 (from_chars), so code that must build at an older
// deployment target (the wheels) cannot call them. These kernels are the
// owned replacement:
//   * f64_to_chars: SHORTEST round-trip digits (the closest such, ties to
//     even) via exact big-integer free-format generation (Steele-White /
//     Burger-Dybvig), laid out exactly as std::to_chars(first, last, v).
//   * f64_from_chars (bolt_float_parse.h): the std::from_chars grammar,
//     correctly rounded.
//
// Tiger Style: noexcept, no allocation, bounded loops, >=2 asserts.

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "bolt/kernels/bolt_bignum_small.h"

namespace bolt {
namespace kernels {
namespace float_chars {

namespace detail {

namespace bn = ::bolt::kernels::bignum;

inline uint32_t bitlen64(uint64_t v) noexcept {
    uint32_t n = 0;
    while (v != 0) { v >>= 1; ++n; }
    return n;
}

struct Scaled {
    bn::Big r, s, mp, mm;   // value, scale, upper and lower half-gaps
    int32_t k;
    bool incl;              // boundaries round-trip (even mantissa)
};

// Burger-Dybvig setup: v = r/s, gaps mp/s and mm/s, scaled so that
// (r + mp)/s < 1 (<= when inclusive) with the smallest such 10^k.
inline void scale(double v, Scaled* st) noexcept {
    assert(v > 0.0 && std::isfinite(v));
    assert(st != nullptr);
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t be = static_cast<uint32_t>((bits >> 52) & 0x7ffu);
    const uint64_t frac = bits & ((uint64_t{1} << 52) - 1u);
    const uint64_t f = be == 0 ? frac : (frac | (uint64_t{1} << 52));
    const int32_t e = be == 0 ? -1074 : static_cast<int32_t>(be) - 1075;
    const bool unequal = frac == 0 && be > 1;
    st->incl = (f & 1u) == 0;
    bn::set_u64(&st->r, f);
    bn::set_u64(&st->s, 1);
    bn::set_u64(&st->mp, 1);
    bn::set_u64(&st->mm, 1);
    if (e >= 0) {
        bn::shl(&st->r, static_cast<uint32_t>(e) + (unequal ? 2u : 1u));
        bn::set_u64(&st->s, unequal ? 4u : 2u);
        bn::shl(&st->mp, static_cast<uint32_t>(e) + (unequal ? 1u : 0u));
        bn::shl(&st->mm, static_cast<uint32_t>(e));
    } else {
        bn::shl(&st->r, unequal ? 2u : 1u);
        bn::shl(&st->s, static_cast<uint32_t>(-e) + (unequal ? 2u : 1u));
        if (unequal) bn::set_u64(&st->mp, 2);
    }
    const int32_t lg2 = e + static_cast<int32_t>(bitlen64(f)) - 1;
    int32_t k = static_cast<int32_t>(
        std::ceil(lg2 * 0.30102999566398114 - 1e-10));
    if (k >= 0) {
        bn::mul_pow10(&st->s, static_cast<uint32_t>(k));
    } else {
        bn::mul_pow10(&st->r, static_cast<uint32_t>(-k));
        bn::mul_pow10(&st->mp, static_cast<uint32_t>(-k));
        bn::mul_pow10(&st->mm, static_cast<uint32_t>(-k));
    }
    for (int32_t fix = 0; fix < 4; ++fix) {
        const int c = bn::compare_sum(&st->r, &st->mp, &st->s);
        if (!(st->incl ? c >= 0 : c > 0)) break;
        bn::mul_small(&st->s, 10);
        ++k;
    }
    st->k = k;
    assert(!st->r.overflow && !st->s.overflow);
}

// Last digit: d or d+1 (closest; exact tie to even), carry propagated.
inline uint32_t finish(Scaled* st, char* digits, uint32_t n, int d,
                       bool tc1, bool tc2) noexcept {
    assert(st != nullptr && digits != nullptr);
    assert(n < 20 && d >= 0 && d <= 9);
    int up = tc2 ? 1 : 0;
    if (tc1 && tc2) {
        bn::Big r2;
        bn::copy(&r2, &st->r);
        bn::shl(&r2, 1);
        const int c = bn::compare(&r2, &st->s);
        up = c > 0 ? 1 : (c < 0 ? 0 : (d & 1));
    }
    digits[n] = static_cast<char>('0' + d + up);
    for (uint32_t j = n + 1; j > 0 && digits[j - 1] > '9'; --j) {
        digits[j - 1] = '0';
        if (j == 1) { digits[0] = '1'; ++st->k; break; }
        ++digits[j - 2];
    }
    uint32_t i = n + 1;
    while (i > 1 && digits[i - 1] == '0') --i;
    assert(i >= 1 && i <= 17);
    return i;
}

#if defined(__SIZEOF_INT128__)
// Same algorithm in 128-bit words for normal v with -120 <= e <= 37. Every
// quantity stays below 10 * s; s is ~40 * mantissa (< 2^60) when k >= 0,
// 2^(2-e) <= 2^122 when k < 0, and 2 * 10^k <= 2^96 when e >= 0.
inline bool shortest_digits_u128(double v, char* digits, uint32_t* nd,
                                 int32_t* k_out) noexcept {
    assert(v > 0.0 && std::isfinite(v));
    assert(digits != nullptr && nd != nullptr && k_out != nullptr);
    using u128 = unsigned __int128;
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t be = static_cast<uint32_t>((bits >> 52) & 0x7ffu);
    const uint64_t frac = bits & ((uint64_t{1} << 52) - 1u);
    const int32_t e = static_cast<int32_t>(be) - 1075;
    if (be == 0 || e < -120 || e > 37) return false;
    const uint64_t f = frac | (uint64_t{1} << 52);
    const bool unequal = frac == 0 && be > 1;
    const bool incl = (f & 1u) == 0;
    const uint32_t u = unequal ? 1u : 0u;
    u128 r, s, mp, mm;
    if (e >= 0) {
        r = u128{f} << (static_cast<uint32_t>(e) + 1u + u);
        s = u128{2} << u;
        mp = u128{1} << (static_cast<uint32_t>(e) + u);
        mm = u128{1} << static_cast<uint32_t>(e);
    } else {
        r = u128{f} << (1u + u);
        s = u128{1} << (static_cast<uint32_t>(-e) + 1u + u);
        mp = u128{1} << u;
        mm = 1;
    }
    int32_t k = static_cast<int32_t>(
        std::ceil((e + 52) * 0.30102999566398114 - 1e-10));
    for (int32_t i = 0; i < k; ++i) s *= 10u;
    for (int32_t i = k; i < 0; ++i) { r *= 10u; mp *= 10u; mm *= 10u; }
    for (int32_t fix = 0; fix < 4 && (incl ? r + mp >= s : r + mp > s); ++fix) {
        s *= 10u;
        ++k;
    }
    for (uint32_t n = 0; n < 20; ++n) {
        r *= 10u; mp *= 10u; mm *= 10u;
        int d = 0;
        while (d < 10 && r >= s) { r -= s; ++d; }
        const bool tc1 = incl ? r <= mm : r < mm;
        const bool tc2 = incl ? r + mp >= s : r + mp > s;
        if (!tc1 && !tc2) { digits[n] = static_cast<char>('0' + d); continue; }
        int up = tc2 ? 1 : 0;
        if (tc1 && tc2) up = 2 * r > s ? 1 : (2 * r < s ? 0 : (d & 1));
        digits[n] = static_cast<char>('0' + d + up);
        for (uint32_t j = n + 1; j > 0 && digits[j - 1] > '9'; --j) {
            digits[j - 1] = '0';
            if (j == 1) { digits[0] = '1'; ++k; break; }
            ++digits[j - 2];
        }
        uint32_t i = n + 1;
        while (i > 1 && digits[i - 1] == '0') --i;
        *nd = i;
        *k_out = k;
        return true;
    }
    return false;
}
#endif

// v > 0, finite. Writes the shortest digits and k such that
// v ~= 0.d1d2...dn * 10^k. Returns n (1..17).
inline uint32_t shortest_digits(double v, char* digits, int32_t* k_out) noexcept {
    assert(v > 0.0 && std::isfinite(v));
    assert(digits != nullptr && k_out != nullptr);
#if defined(__SIZEOF_INT128__)
    uint32_t fast_nd = 0;
    if (shortest_digits_u128(v, digits, &fast_nd, k_out)) return fast_nd;
#endif
    Scaled st;
    scale(v, &st);
    for (uint32_t n = 0; n < 20; ++n) {
        bn::mul_small(&st.r, 10);
        bn::mul_small(&st.mp, 10);
        bn::mul_small(&st.mm, 10);
        int d = 0;
        while (d < 10 && bn::compare(&st.r, &st.s) >= 0) {
            bn::sub(&st.r, &st.s);
            ++d;
        }
        const int cl = bn::compare(&st.r, &st.mm);
        const int ch = bn::compare_sum(&st.r, &st.mp, &st.s);
        const bool tc1 = st.incl ? cl <= 0 : cl < 0;
        const bool tc2 = st.incl ? ch >= 0 : ch > 0;
        if (!tc1 && !tc2) { digits[n] = static_cast<char>('0' + d); continue; }
        const uint32_t nd = finish(&st, digits, n, d, tc1, tc2);
        *k_out = st.k;
        return nd;
    }
    assert(false && "shortest_digits: no termination");
    return 0;
}

// std::to_chars prints a fixed-notation integer D * 10^re (re > 0) as the
// double's EXACT integer value unless D * 10^re is itself exactly
// representable: (D without trailing binary zeros) * 5^re < 2^53.
inline bool fixed_needs_exact(const char* digits, uint32_t nd, int32_t re) noexcept {
    assert(digits != nullptr && nd >= 1 && nd <= 17);
    assert(re > 0);
    static constexpr uint64_t max_shifted[23] = {
        9007199254740991u, 1801439850948198u, 360287970189639u,
        72057594037927u, 14411518807585u, 2882303761517u, 576460752303u,
        115292150460u, 23058430092u, 4611686018u, 922337203u, 184467440u,
        36893488u, 7378697u, 1475739u, 295147u, 59029u, 11805u, 2361u, 472u,
        94u, 18u, 3u};
    if (re > 22) return true;
    uint64_t m = 0;
    for (uint32_t i = 0; i < nd; ++i) m = m * 10u + static_cast<uint64_t>(digits[i] - '0');
    while ((m & 1u) == 0) m >>= 1;
    return m > max_shifted[re];
}

// Exact decimal digits of an integral v (< 2^80); returns the count.
inline uint32_t exact_integer(double v, char* out, uint32_t cap) noexcept {
    assert(v >= 1.0 && v == std::floor(v));
    assert(out != nullptr && cap >= 25);
    int exp2 = 0;
    const double fr = std::frexp(v, &exp2);
    const uint64_t f = static_cast<uint64_t>(std::ldexp(fr, 53));
    bn::Big b;
    bn::set_u64(&b, f);
    if (exp2 > 53) bn::shl(&b, static_cast<uint32_t>(exp2 - 53));
    else { for (int i = exp2; i < 53; ++i) bn::shr1(&b); }
    char rev[32];
    uint32_t n = 0;
    while (!bn::is_zero(&b) && n < sizeof(rev)) {
        rev[n++] = static_cast<char>('0' + bn::div_small(&b, 10));
    }
    assert(n <= cap);
    for (uint32_t i = 0; i < n; ++i) out[i] = rev[n - 1 - i];
    return n;
}

}  // namespace detail

// Formats v like std::to_chars(first, last, v) (no format argument). Returns
// the byte count, or 0 when cap is too small (std::errc::value_too_large).
inline uint32_t f64_to_chars(char* out, uint32_t cap, double v) noexcept {
    assert(out != nullptr);
    assert(cap <= (1u << 20));
    char t[40];
    uint32_t n = 0;
    if (std::signbit(v)) t[n++] = '-';
    if (std::isnan(v) || std::isinf(v)) {
        std::memcpy(t + n, std::isnan(v) ? "nan" : "inf", 3);
        n += 3;
    } else if (v == 0.0) {
        t[n++] = '0';
    } else {
        char d[20];
        int32_t k = 0;
        const uint32_t nd = detail::shortest_digits(std::fabs(v), d, &k);
        const int32_t re = k - static_cast<int32_t>(nd);   // v = D * 10^re
        const int32_t lo = nd == 1 ? -3 : -static_cast<int32_t>(nd + 3);
        const int32_t hi = nd == 1 ? 4 : 5;
        if (re >= lo && re <= hi) {
            const uint32_t uk = static_cast<uint32_t>(k);
            if (re > 0 && detail::fixed_needs_exact(d, nd, re)) {
                n += detail::exact_integer(std::fabs(v), t + n, sizeof(t) - n);
            } else if (re >= 0) {
                std::memcpy(t + n, d, nd); n += nd;
                for (int32_t i = 0; i < re; ++i) t[n++] = '0';
            } else if (k > 0) {
                std::memcpy(t + n, d, uk); n += uk;
                t[n++] = '.';
                std::memcpy(t + n, d + uk, nd - uk); n += nd - uk;
            } else {
                t[n++] = '0'; t[n++] = '.';
                for (int32_t i = 0; i < -k; ++i) t[n++] = '0';
                std::memcpy(t + n, d, nd); n += nd;
            }
        } else {
            t[n++] = d[0];
            if (nd > 1) { t[n++] = '.'; std::memcpy(t + n, d + 1, nd - 1); n += nd - 1; }
            int32_t x = k - 1;
            t[n++] = 'e';
            t[n++] = x < 0 ? '-' : '+';
            if (x < 0) x = -x;
            if (x >= 100) t[n++] = static_cast<char>('0' + x / 100);
            t[n++] = static_cast<char>('0' + (x / 10) % 10);
            t[n++] = static_cast<char>('0' + x % 10);
        }
    }
    assert(n <= sizeof(t));
    if (n > cap) return 0;
    std::memcpy(out, t, n);
    return n;
}

}  // namespace float_chars
}  // namespace kernels
}  // namespace bolt
