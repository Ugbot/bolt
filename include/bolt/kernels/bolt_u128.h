// bolt/kernels/bolt_u128.h — portable unsigned 128-bit helpers (MSVC has no __int128): a value as (hi, lo) and the
// few operations decimal text and variant decimals need. Division by a small divisor is decimal::div_u128_u64.
#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/kernels/bolt_decimal.h"

namespace bolt {
namespace kernels {
namespace u128 {

struct U128 {
    uint64_t hi = 0;
    uint64_t lo = 0;
};

inline bool is_zero(U128 x) noexcept { return x.hi == 0 && x.lo == 0; }

// Two's complement negation (wraps at 128 bits).
inline U128 negate(U128 x) noexcept {
    U128 r{~x.hi, ~x.lo};
    if (++r.lo == 0) ++r.hi;
    return r;
}

// A two's-complement little-endian integer of `n` bytes (1..16): its magnitude, and whether it is negative.
inline U128 from_le_signed(const uint8_t* p, uint32_t n, bool* neg) noexcept {
    assert(p != nullptr && neg != nullptr && n >= 1u && n <= 16u);
    U128 x{};
    for (uint32_t i = 0; i < n; ++i) {
        if (i < 8u) x.lo |= uint64_t{p[i]} << (8u * i);
        else x.hi |= uint64_t{p[i]} << (8u * (i - 8u));
    }
    *neg = (p[n - 1u] & 0x80u) != 0;
    if (!*neg) return x;
    if (n < 8u) { x.lo |= ~uint64_t{0} << (8u * n); x.hi = ~uint64_t{0}; }
    else if (n == 8u) x.hi = ~uint64_t{0};
    else if (n < 16u) x.hi |= ~uint64_t{0} << (8u * (n - 8u));
    return negate(x);
}

// x * 10 + d; false when the result passes 128 bits.
inline bool mul10_add(U128* x, uint32_t d) noexcept {
    assert(x != nullptr && d < 10u);
    const uint64_t a = (x->lo & 0xFFFFFFFFull) * 10u, b = (x->lo >> 32) * 10u;
    const uint64_t lo = a + (b << 32);
    const uint64_t carry = (lo < a ? 1u : 0u) + (b >> 32);
    const uint64_t h0 = x->hi;
    if (h0 > (~uint64_t{0} - carry) / 10u) return false;
    U128 r{h0 * 10u + carry, lo};
    r.lo += d;
    if (r.lo < d) { if (r.hi == ~uint64_t{0}) return false; ++r.hi; }
    *x = r;
    return true;
}

// x + d; false when the result passes 128 bits.
inline bool add_small(U128* x, uint64_t d) noexcept {
    assert(x != nullptr);
    x->lo += d;
    if (x->lo >= d) return true;
    if (x->hi == ~uint64_t{0}) return false;
    ++x->hi;
    return true;
}

inline bool less(U128 a, U128 b) noexcept { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }

// 10^n for n <= 38.
inline U128 pow10(uint32_t n) noexcept {
    assert(n <= 38u);
    U128 r{0, 1};
    for (uint32_t k = 0; k < n; ++k) (void)mul10_add(&r, 0);   // 10^38 < 2^127
    return r;
}

// The 16-byte little-endian two's complement of (neg ? -mag : mag).
inline void to_le_signed(U128 mag, bool neg, uint8_t* out16) noexcept {
    assert(out16 != nullptr);
    const U128 v = neg ? negate(mag) : mag;
    for (uint32_t i = 0; i < 8u; ++i) {
        out16[i] = static_cast<uint8_t>(v.lo >> (8u * i));
        out16[8u + i] = static_cast<uint8_t>(v.hi >> (8u * i));
    }
}

// a * b, exact.
inline U128 mul_u64(uint64_t a, uint64_t b) noexcept {
    const uint64_t al = a & 0xFFFFFFFFu, ah = a >> 32, bl = b & 0xFFFFFFFFu, bh = b >> 32;
    const uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    const uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFu) + (hl & 0xFFFFFFFFu);
    return U128{hh + (lh >> 32) + (hl >> 32) + (mid >> 32), (mid << 32) | (ll & 0xFFFFFFFFu)};
}

// x * m; false when the result passes 128 bits.
inline bool mul_small(U128 x, uint64_t m, U128* out) noexcept {
    assert(out != nullptr);
    const U128 lo = mul_u64(x.lo, m);
    const U128 hi = mul_u64(x.hi, m);
    if (hi.hi != 0) return false;
    const uint64_t h = hi.lo + lo.hi;
    if (h < lo.hi) return false;
    *out = U128{h, lo.lo};
    return true;
}

// x >> k, x << k and 2^k for k < 128.
inline U128 shr(U128 x, uint32_t k) noexcept {
    assert(k < 128u);
    if (k == 0) return x;
    if (k >= 64u) return U128{0, x.hi >> (k - 64u)};
    return U128{x.hi >> k, (x.lo >> k) | (x.hi << (64u - k))};
}
inline U128 shl(U128 x, uint32_t k) noexcept {
    assert(k < 128u);
    if (k == 0) return x;
    if (k >= 64u) return U128{x.lo << (k - 64u), 0};
    return U128{(x.hi << k) | (x.lo >> (64u - k)), x.lo << k};
}
inline U128 bit(uint32_t k) noexcept { return shl(U128{0, 1}, k); }

// a - b (wraps at 128 bits).
inline U128 sub(U128 a, U128 b) noexcept {
    U128 r{a.hi - b.hi, a.lo - b.lo};
    if (a.lo < b.lo) --r.hi;
    return r;
}

inline bool equal(U128 a, U128 b) noexcept { return a.hi == b.hi && a.lo == b.lo; }

// a + b (wraps at 128 bits).
inline U128 add(U128 a, U128 b) noexcept {
    U128 r{a.hi + b.hi, a.lo + b.lo};
    if (r.lo < a.lo) ++r.hi;
    return r;
}

// a * b; false when the result passes 128 bits.
inline bool mul(U128 a, U128 b, U128* out) noexcept {
    assert(out != nullptr);
    if (a.hi != 0 && b.hi != 0) return false;
    if (a.hi != 0) return mul_small(a, b.lo, out);
    return mul_small(b, a.lo, out);
}

// x / d, the remainder in *rem.
inline U128 div_small(U128 x, uint64_t d, uint64_t* rem) noexcept {
    assert(d != 0 && rem != nullptr);
    U128 q{};
    decimal::div_u128_u64(x.hi, x.lo, d, &q.hi, &q.lo, rem);
    return q;
}

}  // namespace u128
}  // namespace kernels
}  // namespace bolt
