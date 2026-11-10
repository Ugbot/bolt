// bolt/kernels/bolt_ryu.h — shortest round-trip decimal of a double (Ryu,
// Ulf Adams, PLDI 2018).
//
// d2d(v) is the shortest decimal m * 10^e that reads back as v; among the
// shortest, the closest to v, ties to even. That is the digit string CPython
// repr(float) (dtoa mode 0) and std::to_chars print. Two 64x128-bit products
// per value against generated tables (bolt_ryu_tables.h), no bignum, no
// allocation; the Steele-White generator in bolt_float_chars.h stays as the
// exact reference the tests sweep it against.
//
// Tiger Style: noexcept, bounded loops (<= 17 digit removals), >=2 asserts.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_ryu_tables.h"

namespace bolt {
namespace kernels {
namespace ryu {

struct Decimal64 {
    uint64_t mantissa;   // no trailing zeros required; 1..17 digits
    int32_t  exponent;   // value = mantissa * 10^exponent
};

namespace detail {

BOLT_FORCE_INLINE uint32_t pow5bits(int32_t e) noexcept {
    assert(e >= 0 && e <= 3528);
    return static_cast<uint32_t>(((static_cast<uint32_t>(e) * 1217359u) >> 19) + 1u);
}
BOLT_FORCE_INLINE uint32_t log10_pow2(int32_t e) noexcept {
    assert(e >= 0 && e <= 1650);
    return (static_cast<uint32_t>(e) * 78913u) >> 18;
}
BOLT_FORCE_INLINE uint32_t log10_pow5(int32_t e) noexcept {
    assert(e >= 0 && e <= 2620);
    return (static_cast<uint32_t>(e) * 732923u) >> 20;
}

BOLT_FORCE_INLINE uint32_t pow5_factor(uint64_t v) noexcept {
    assert(v != 0);
    uint32_t n = 0;
    while (v % 5u == 0) {   // <= 27 for a 64-bit value
        v /= 5u;
        ++n;
    }
    return n;
}
BOLT_FORCE_INLINE bool multiple_of_pow5(uint64_t v, uint32_t p) noexcept { return pow5_factor(v) >= p; }
BOLT_FORCE_INLINE bool multiple_of_pow2(uint64_t v, uint32_t p) noexcept {
    assert(p < 64);
    return (v & ((uint64_t{1} << p) - 1u)) == 0;
}

// (m * mul) >> j for a 64-bit m, a 128-bit mul and j >= 64.
BOLT_FORCE_INLINE uint64_t mul_shift64(uint64_t m, const uint64_t* mul, int32_t j) noexcept {
    assert(j >= 64 && j < 192);
#if defined(__SIZEOF_INT128__)
    using u128 = unsigned __int128;
    const u128 b0 = static_cast<u128>(m) * mul[0];
    const u128 b2 = static_cast<u128>(m) * mul[1];
    return static_cast<uint64_t>(((b0 >> 64) + b2) >> (j - 64));
#else
    const auto mul64 = [](uint64_t a, uint64_t b, uint64_t* hi) noexcept {
        const uint64_t al = a & 0xFFFFFFFFu, ah = a >> 32, bl = b & 0xFFFFFFFFu, bh = b >> 32;
        const uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
        const uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFu) + (hl & 0xFFFFFFFFu);
        *hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
        return (mid << 32) | (ll & 0xFFFFFFFFu);
    };
    uint64_t h0 = 0, h2 = 0;
    (void)mul64(m, mul[0], &h0);
    const uint64_t l2 = mul64(m, mul[1], &h2);
    const uint64_t lo = l2 + h0;
    const uint64_t hi = h2 + (lo < l2 ? 1u : 0u);   // 128-bit (b0 >> 64) + b2
    const int32_t s = j - 64;
    return s == 0 ? lo : s < 64 ? (hi << (64 - s)) | (lo >> s) : hi >> (s - 64);
#endif
}

BOLT_FORCE_INLINE uint32_t decimal_length17(uint64_t v) noexcept {
    assert(v < 100000000000000000ull);
    if (v >= 10000000000000000ull) return 17;
    if (v >= 1000000000000000ull) return 16;
    if (v >= 100000000000000ull) return 15;
    if (v >= 10000000000000ull) return 14;
    if (v >= 1000000000000ull) return 13;
    if (v >= 100000000000ull) return 12;
    if (v >= 10000000000ull) return 11;
    if (v >= 1000000000ull) return 10;
    if (v >= 100000000ull) return 9;
    if (v >= 10000000ull) return 8;
    if (v >= 1000000ull) return 7;
    if (v >= 100000ull) return 6;
    if (v >= 10000ull) return 5;
    if (v >= 1000ull) return 4;
    if (v >= 100ull) return 3;
    if (v >= 10ull) return 2;
    return 1;
}

}  // namespace detail

// v > 0, finite (subnormals included).
inline Decimal64 d2d(double v) noexcept {
    assert(v > 0.0);
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    const uint64_t ieee_m = bits & ((uint64_t{1} << 52) - 1u);
    const uint32_t ieee_e = static_cast<uint32_t>((bits >> 52) & 0x7FFu);
    assert(ieee_e != 0x7FFu);
    int32_t e2;
    uint64_t m2;
    if (ieee_e == 0) {
        e2 = 1 - 1023 - 52 - 2;
        m2 = ieee_m;
    } else {
        e2 = static_cast<int32_t>(ieee_e) - 1023 - 52 - 2;
        m2 = (uint64_t{1} << 52) | ieee_m;
    }
    const bool accept_bounds = (m2 & 1u) == 0;   // round-half-even reading back
    const uint64_t mv = 4u * m2;
    const uint32_t mm_shift = (ieee_m != 0 || ieee_e <= 1) ? 1u : 0u;

    uint64_t vr, vp, vm;
    int32_t e10;
    bool vm_tz = false, vr_tz = false;
    if (e2 >= 0) {
        const uint32_t q = detail::log10_pow2(e2) - (e2 > 3 ? 1u : 0u);
        e10 = static_cast<int32_t>(q);
        const int32_t k = kPow5InvBitcount + static_cast<int32_t>(detail::pow5bits(static_cast<int32_t>(q))) - 1;
        const int32_t i = -e2 + static_cast<int32_t>(q) + k;
        assert(q < kPow5InvCount);
        vr = detail::mul_shift64(4u * m2, kPow5InvSplit[q], i);
        vp = detail::mul_shift64(4u * m2 + 2u, kPow5InvSplit[q], i);
        vm = detail::mul_shift64(4u * m2 - 1u - mm_shift, kPow5InvSplit[q], i);
        if (q <= 21) {
            if (mv % 5u == 0) vr_tz = detail::multiple_of_pow5(mv, q);
            else if (accept_bounds) vm_tz = detail::multiple_of_pow5(mv - 1u - mm_shift, q);
            else vp -= detail::multiple_of_pow5(mv + 2u, q) ? 1u : 0u;
        }
    } else {
        const uint32_t q = detail::log10_pow5(-e2) - (-e2 > 1 ? 1u : 0u);
        e10 = static_cast<int32_t>(q) + e2;
        const int32_t i = -e2 - static_cast<int32_t>(q);
        const int32_t k = static_cast<int32_t>(detail::pow5bits(i)) - kPow5Bitcount;
        const int32_t j = static_cast<int32_t>(q) - k;
        assert(i >= 0 && static_cast<uint32_t>(i) < kPow5Count);
        vr = detail::mul_shift64(4u * m2, kPow5Split[i], j);
        vp = detail::mul_shift64(4u * m2 + 2u, kPow5Split[i], j);
        vm = detail::mul_shift64(4u * m2 - 1u - mm_shift, kPow5Split[i], j);
        if (q <= 1) {
            vr_tz = true;
            if (accept_bounds) vm_tz = mm_shift == 1u;
            else --vp;
        } else if (q < 63) {
            vr_tz = detail::multiple_of_pow2(mv, q);
        }
    }

    int32_t removed = 0;
    uint64_t out;
    if (vm_tz || vr_tz) {   // rare: exact ties decide
        uint8_t last = 0;
        while (vp / 10u > vm / 10u) {
            vm_tz &= vm % 10u == 0;
            vr_tz &= last == 0;
            last = static_cast<uint8_t>(vr % 10u);
            vr /= 10u; vp /= 10u; vm /= 10u;
            ++removed;
        }
        if (vm_tz) {
            while (vm % 10u == 0) {
                vr_tz &= last == 0;
                last = static_cast<uint8_t>(vr % 10u);
                vr /= 10u; vp /= 10u; vm /= 10u;
                ++removed;
            }
        }
        if (vr_tz && last == 5 && vr % 2u == 0) last = 4;   // exactly ...50..0: to even
        out = vr + (((vr == vm && (!accept_bounds || !vm_tz)) || last >= 5) ? 1u : 0u);
    } else {
        bool round_up = false;
        if (vp / 100u > vm / 100u) {
            round_up = vr % 100u >= 50u;
            vr /= 100u; vp /= 100u; vm /= 100u;
            removed += 2;
        }
        while (vp / 10u > vm / 10u) {
            round_up = vr % 10u >= 5u;
            vr /= 10u; vp /= 10u; vm /= 10u;
            ++removed;
        }
        out = vr + ((vr == vm || round_up) ? 1u : 0u);
    }
    assert(out != 0 && out < 100000000000000000ull);
    return Decimal64{out, e10 + removed};
}

// Digits of d2d(v) (no trailing zeros) into digits[0..n) and k with
// v ~= 0.d1..dn * 10^k, as float_chars::detail::shortest_digits reports them.
inline uint32_t shortest_digits(double v, char* digits, int32_t* k_out) noexcept {
    assert(digits != nullptr && k_out != nullptr);
    static constexpr char kPairs[201] =
        "00010203040506070809101112131415161718192021222324252627282930313233343536373839"
        "40414243444546474849505152535455565758596061626364656667686970717273747576777879"
        "8081828384858687888990919293949596979899";
    Decimal64 d = d2d(v);
    if (d.mantissa % 10u == 0) {   // rare (integers, short decimals): <= 16 steps
        do {
            d.mantissa /= 10u;
            ++d.exponent;
        } while (d.mantissa % 10u == 0);
    }
    uint64_t m = d.mantissa;
    const uint32_t n = detail::decimal_length17(m);
    uint32_t i = n;
    while (m >= 100u) {
        const uint32_t r = static_cast<uint32_t>(m % 100u);
        m /= 100u;
        i -= 2;
        std::memcpy(digits + i, kPairs + 2u * r, 2);
    }
    if (m >= 10u) {
        i -= 2;
        std::memcpy(digits + i, kPairs + 2u * m, 2);
    } else {
        digits[--i] = static_cast<char>('0' + m);
    }
    assert(i == 0);
    *k_out = d.exponent + static_cast<int32_t>(n);
    assert(n >= 1 && n <= 17);
    return n;
}

}  // namespace ryu
}  // namespace kernels
}  // namespace bolt
