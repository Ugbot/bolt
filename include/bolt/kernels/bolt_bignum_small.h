// bolt/kernels/bolt_bignum_small.h — fixed-capacity unsigned big integer for
// the exact float <-> decimal slow paths (bolt_float_chars.h). Little-endian
// 32-bit limbs, no allocation; every operation that would exceed the
// capacity sets `overflow` instead of growing.
//
// Tiger Style: noexcept, no allocation, bounded loops, >=2 asserts.

#pragma once

#include <cassert>
#include <cstdint>

namespace bolt {
namespace kernels {
namespace bignum {

// 5120 bits: 10^1130 (the widest scale the parser admits) times a 2^66
// quotient window times an 800-digit mantissa bound stays inside.
inline constexpr uint32_t k_limbs = 160;

struct Big {
    uint32_t limb[k_limbs];
    uint32_t n;          // used limbs; limb[n-1] != 0 unless n == 0
    bool overflow;
};

inline void set_u64(Big* b, uint64_t v) noexcept {
    assert(b != nullptr);
    b->overflow = false;
    b->limb[0] = static_cast<uint32_t>(v);
    b->limb[1] = static_cast<uint32_t>(v >> 32);
    b->n = (v >> 32) != 0 ? 2u : (v != 0 ? 1u : 0u);
    assert(b->n <= 2);
}

inline void copy(Big* dst, const Big* src) noexcept {
    assert(dst != nullptr && src != nullptr);
    assert(src->n <= k_limbs);
    for (uint32_t i = 0; i < src->n; ++i) dst->limb[i] = src->limb[i];
    dst->n = src->n;
    dst->overflow = src->overflow;
}

inline bool is_zero(const Big* b) noexcept {
    assert(b != nullptr);
    assert(b->n <= k_limbs);
    return b->n == 0;
}

inline uint32_t bit_length(const Big* b) noexcept {
    assert(b != nullptr);
    assert(b->n <= k_limbs);
    if (b->n == 0) return 0;
    uint32_t top = b->limb[b->n - 1];
    uint32_t bits = 0;
    while (top != 0) { top >>= 1; ++bits; }
    return (b->n - 1) * 32u + bits;
}

inline void mul_small(Big* b, uint32_t m) noexcept {
    assert(b != nullptr);
    assert(b->n <= k_limbs);
    uint64_t carry = 0;
    for (uint32_t i = 0; i < b->n; ++i) {
        const uint64_t p = static_cast<uint64_t>(b->limb[i]) * m + carry;
        b->limb[i] = static_cast<uint32_t>(p);
        carry = p >> 32;
    }
    if (carry != 0) {
        if (b->n == k_limbs) { b->overflow = true; return; }
        b->limb[b->n++] = static_cast<uint32_t>(carry);
    }
    if (m == 0) b->n = 0;
}

inline void add_small(Big* b, uint32_t a) noexcept {
    assert(b != nullptr);
    assert(b->n <= k_limbs);
    uint64_t carry = a;
    for (uint32_t i = 0; i < b->n && carry != 0; ++i) {
        const uint64_t s = static_cast<uint64_t>(b->limb[i]) + carry;
        b->limb[i] = static_cast<uint32_t>(s);
        carry = s >> 32;
    }
    if (carry != 0) {
        if (b->n == k_limbs) { b->overflow = true; return; }
        b->limb[b->n++] = static_cast<uint32_t>(carry);
    }
}

// b *= 10^e, nine decimal digits per limb multiply.
inline void mul_pow10(Big* b, uint32_t e) noexcept {
    assert(b != nullptr);
    assert(e <= 4000);
    while (e >= 9) { mul_small(b, 1000000000u); e -= 9; }
    static constexpr uint32_t p10[9] = {1u, 10u, 100u, 1000u, 10000u,
                                         100000u, 1000000u, 10000000u,
                                         100000000u};
    if (e > 0) mul_small(b, p10[e]);
}

inline void shl(Big* b, uint32_t s) noexcept {
    assert(b != nullptr);
    assert(b->n <= k_limbs);
    if (b->n == 0 || s == 0) return;
    const uint32_t words = s / 32u;
    const uint32_t bits = s % 32u;
    const uint32_t top_extra =
        (bits != 0 && (b->limb[b->n - 1] >> (32u - bits)) != 0) ? 1u : 0u;
    const uint32_t nn = b->n + words + top_extra;
    if (nn > k_limbs) { b->overflow = true; return; }
    if (top_extra) b->limb[nn - 1] = b->limb[b->n - 1] >> (32u - bits);
    for (uint32_t i = b->n; i > 0; --i) {
        const uint32_t src = i - 1;
        uint32_t v = b->limb[src] << bits;
        if (bits != 0 && src > 0) v |= b->limb[src - 1] >> (32u - bits);
        b->limb[src + words] = v;
    }
    for (uint32_t i = 0; i < words; ++i) b->limb[i] = 0;
    b->n = nn;
}

inline void shr1(Big* b) noexcept {
    assert(b != nullptr);
    assert(b->n <= k_limbs);
    for (uint32_t i = 0; i < b->n; ++i) {
        uint32_t v = b->limb[i] >> 1;
        if (i + 1 < b->n) v |= b->limb[i + 1] << 31;
        b->limb[i] = v;
    }
    while (b->n > 0 && b->limb[b->n - 1] == 0) --b->n;
}

inline int compare(const Big* a, const Big* b) noexcept {
    assert(a != nullptr && b != nullptr);
    assert(a->n <= k_limbs && b->n <= k_limbs);
    if (a->n != b->n) return a->n < b->n ? -1 : 1;
    for (uint32_t i = a->n; i > 0; --i) {
        if (a->limb[i - 1] != b->limb[i - 1])
            return a->limb[i - 1] < b->limb[i - 1] ? -1 : 1;
    }
    return 0;
}

// a + c compared with b, without materialising the sum.
inline int compare_sum(const Big* a, const Big* c, const Big* b) noexcept {
    assert(a != nullptr && b != nullptr && c != nullptr);
    assert(a->n <= k_limbs && c->n <= k_limbs);
    Big s;
    copy(&s, a);
    uint64_t carry = 0;
    const uint32_t n = s.n > c->n ? s.n : c->n;
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t x = (i < s.n ? s.limb[i] : 0u);
        const uint64_t y = (i < c->n ? c->limb[i] : 0u);
        const uint64_t t = x + y + carry;
        s.limb[i] = static_cast<uint32_t>(t);
        carry = t >> 32;
    }
    s.n = n;
    if (carry != 0) {
        if (n == k_limbs) return 1;
        s.limb[s.n++] = static_cast<uint32_t>(carry);
    }
    return compare(&s, b);
}

// b /= d; returns the remainder.
inline uint32_t div_small(Big* b, uint32_t d) noexcept {
    assert(b != nullptr && d != 0);
    assert(b->n <= k_limbs);
    uint64_t rem = 0;
    for (uint32_t i = b->n; i > 0; --i) {
        const uint64_t cur = (rem << 32) | b->limb[i - 1];
        b->limb[i - 1] = static_cast<uint32_t>(cur / d);
        rem = cur % d;
    }
    while (b->n > 0 && b->limb[b->n - 1] == 0) --b->n;
    return static_cast<uint32_t>(rem);
}

// a -= b; requires a >= b.
inline void sub(Big* a, const Big* b) noexcept {
    assert(a != nullptr && b != nullptr);
    assert(compare(a, b) >= 0);
    int64_t borrow = 0;
    for (uint32_t i = 0; i < a->n; ++i) {
        int64_t d = static_cast<int64_t>(a->limb[i]) - borrow -
                    (i < b->n ? static_cast<int64_t>(b->limb[i]) : 0);
        borrow = d < 0 ? 1 : 0;
        if (d < 0) d += (int64_t{1} << 32);
        a->limb[i] = static_cast<uint32_t>(d);
    }
    while (a->n > 0 && a->limb[a->n - 1] == 0) --a->n;
}

}  // namespace bignum
}  // namespace kernels
}  // namespace bolt
