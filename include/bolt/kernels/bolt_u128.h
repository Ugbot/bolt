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
