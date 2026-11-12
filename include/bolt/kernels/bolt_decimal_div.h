// bolt/kernels/bolt_decimal_div.h — exact Decimal128 division and remainder at
// a chosen result scale.
//
// d128_div_round(a, sa, b, sb, s) is round_half_up(a / b) at scale `s`, exact:
// the operands are aligned in 256 bits, so no intermediate truncates. HALF_UP
// rounds away from zero on a discarded half (SQL / Spark / Java BigDecimal
// ROUND_HALF_UP). d128_rem_exact is the remainder with the dividend's sign at
// scale max(sa, sb). Both report a zero divisor and a result past 127 bits;
// the caller owns the precision check and what each failure means.
#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/kernels/bolt_decimal.h"

namespace bolt::kernels::decimal {

enum class DivStatus : uint8_t { Ok, DivZero, Overflow };

// How a quotient drops the digits past its scale.
enum class DecRound : uint8_t { HalfUp, TowardZero };

// The rounding of a decimal AVG at its result scale, for every finalizer
// (bolt's grouped ones, chukonu's global one). HalfUp is Spark's and
// Postgres'; TowardZero was the G2CHK-736 contract.
inline constexpr DecRound k_decimal_avg_round = DecRound::HalfUp;

namespace div_detail {

struct U256 { uint64_t w[4]; };   // little-endian limbs

inline bool u256_is_zero(const U256& x) noexcept { return (x.w[0] | x.w[1] | x.w[2] | x.w[3]) == 0; }

inline U256 u256_from(Decimal128 v) noexcept {
    assert(v.hi >= 0);
    return U256{{static_cast<uint64_t>(v.lo), static_cast<uint64_t>(v.hi), 0, 0}};
}

// x *= m; false when the product does not fit 256 bits.
inline bool u256_mul_small(U256* x, uint64_t m) noexcept {
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t lo = 0, hi = 0;
        mul_u64_u64(x->w[i], m, &lo, &hi);
        const uint64_t s = lo + carry;
        hi += s < lo ? 1u : 0u;
        x->w[i] = s;
        carry = hi;
    }
    return carry == 0;
}

// x *= 10^n; false on overflow.
inline bool u256_mul_pow10(U256* x, int n) noexcept {
    assert(n >= 0);
    static constexpr uint64_t kPow10[19] = {
        1ull, 10ull, 100ull, 1000ull, 10000ull, 100000ull, 1000000ull, 10000000ull,
        100000000ull, 1000000000ull, 10000000000ull, 100000000000ull, 1000000000000ull,
        10000000000000ull, 100000000000000ull, 1000000000000000ull, 10000000000000000ull,
        100000000000000000ull, 1000000000000000000ull,
    };
    for (int rem = n; rem > 0;) {
        const int step = rem <= 18 ? rem : 18;
        if (!u256_mul_small(x, kPow10[step])) return false;
        rem -= step;
    }
    return true;
}

inline int u256_cmp(const U256& a, const U256& b) noexcept {
    for (int i = 3; i >= 0; --i) {
        if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1;
    }
    return 0;
}

inline void u256_sub(U256* a, const U256& b) noexcept {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        const uint64_t d = a->w[i] - b.w[i];
        const uint64_t nb = (a->w[i] < b.w[i]) || (d < borrow) ? 1u : 0u;
        a->w[i] = d - borrow;
        borrow = nb;
    }
}

inline void u256_shl1(U256* a, uint64_t in) noexcept {
    for (int i = 3; i > 0; --i) a->w[i] = (a->w[i] << 1) | (a->w[i - 1] >> 63);
    a->w[0] = (a->w[0] << 1) | in;
}

inline int u256_bits(const U256& a) noexcept {
    for (int i = 3; i >= 0; --i) {
        if (a.w[i] != 0) {
            int b = 64;
            uint64_t v = a.w[i];
            while ((v >> 63) == 0) { v <<= 1; --b; }
            return i * 64 + b;
        }
    }
    return 0;
}

// q = n / d, r = n % d (d != 0). Shift-subtract over n's significant bits.
inline void u256_divmod(const U256& n, const U256& d, U256* q, U256* r) noexcept {
    assert(!u256_is_zero(d));
    *q = U256{{0, 0, 0, 0}};
    *r = U256{{0, 0, 0, 0}};
    if (n.w[2] == 0 && n.w[3] == 0 && d.w[1] == 0 && d.w[2] == 0 && d.w[3] == 0) {
        uint64_t qh = 0, ql = 0, rem = 0;
        div_u128_u64(n.w[1], n.w[0], d.w[0], &qh, &ql, &rem);
        q->w[0] = ql; q->w[1] = qh; r->w[0] = rem;
        return;
    }
    const int bits = u256_bits(n);
    for (int i = bits - 1; i >= 0; --i) {
        u256_shl1(r, (n.w[i / 64] >> (i % 64)) & 1u);
        u256_shl1(q, 0);
        if (u256_cmp(*r, d) >= 0) { u256_sub(r, d); q->w[0] |= 1u; }
    }
}

inline Decimal128 abs128(Decimal128 v) noexcept { return v.hi < 0 ? d128_neg(v) : v; }

// A non-negative 256-bit value as a signed Decimal128; false past 2^127 - 1.
inline bool to_d128(const U256& x, bool neg, Decimal128* out) noexcept {
    if (x.w[2] != 0 || x.w[3] != 0 || (x.w[1] >> 63) != 0) return false;
    Decimal128 v;
    v.lo = static_cast<int64_t>(x.w[0]);
    v.hi = static_cast<int64_t>(x.w[1]);
    *out = neg ? d128_neg(v) : v;
    return true;
}

}  // namespace div_detail

// (a / 10^sa) / (b / 10^sb) as a mantissa at scale `s`, rounded by `mode`.
inline DivStatus d128_div_scaled(Decimal128 a, int sa, Decimal128 b, int sb, int s,
                                 DecRound mode, Decimal128* out) noexcept {
    using namespace div_detail;
    assert(out != nullptr);
    assert(sa >= 0 && sb >= 0 && s >= 0 && sa <= 76 && sb <= 76 && s <= 76);
    if (b.lo == 0 && b.hi == 0) return DivStatus::DivZero;
    const bool neg = (a.hi < 0) != (b.hi < 0);
    // a / b at scale s: (|a| * 10^(s - sa + sb)) / |b|, the power on whichever
    // side keeps it non-negative.
    U256 n = u256_from(abs128(a));
    U256 d = u256_from(abs128(b));
    const int k = s - sa + sb;
    if (k >= 0) {
        if (!u256_mul_pow10(&n, k)) return DivStatus::Overflow;
    } else if (!u256_mul_pow10(&d, -k)) {
        *out = Decimal128{0, 0};   // |a| < 2^127 over a divisor past 2^256: rounds to zero
        return DivStatus::Ok;
    }
    U256 q{}, r{};
    u256_divmod(n, d, &q, &r);
    // HALF_UP: up when the remainder is at least half the divisor (2r >= d).
    U256 r2 = r;
    const bool r2_fits = u256_mul_small(&r2, 2);
    if (mode == DecRound::HalfUp && (!r2_fits || u256_cmp(r2, d) >= 0)) {
        for (int i = 0; i < 4 && ++q.w[i] == 0; ++i) {}
    }
    return to_d128(q, neg, out) ? DivStatus::Ok : DivStatus::Overflow;
}

// round_half_up((a / 10^sa) / (b / 10^sb)) as a mantissa at scale `s`.
inline DivStatus d128_div_round(Decimal128 a, int sa, Decimal128 b, int sb, int s,
                                Decimal128* out) noexcept {
    return d128_div_scaled(a, sa, b, sb, s, DecRound::HalfUp, out);
}

// |v| < 10^38: v is a DECIMAL(38) mantissa.
inline bool d128_within_38_digits(Decimal128 v) noexcept {
    const Decimal128 m = div_detail::abs128(v);
    return m.hi >= 0 && d128_cmp(m, d128_pow10(38)) < 0;
}

// a + b in two's complement; true when the signed 128-bit word wrapped.
BOLT_FORCE_INLINE bool d128_add_wrapped(Decimal128 a, Decimal128 b, Decimal128* out) noexcept {
    const uint64_t lo = static_cast<uint64_t>(a.lo) + static_cast<uint64_t>(b.lo);
    const uint64_t hi = static_cast<uint64_t>(a.hi) + static_cast<uint64_t>(b.hi) +
                        (lo < static_cast<uint64_t>(a.lo) ? 1u : 0u);
    out->lo = static_cast<int64_t>(lo);
    out->hi = static_cast<int64_t>(hi);
    return ((static_cast<uint64_t>(a.hi) ^ hi) & (static_cast<uint64_t>(b.hi) ^ hi)) >> 63 != 0;
}

// The AVG of `count` >= 1 values summing to `sum` (scale in_s), at scale
// out_s, rounded by k_decimal_avg_round. Overflow when the sum or the result
// needs more than 38 digits.
inline DivStatus d128_avg(Decimal128 sum, int in_s, int64_t count, int out_s,
                          Decimal128* out) noexcept {
    assert(count > 0 && out != nullptr);
    assert(in_s >= 0 && out_s >= in_s && out_s <= 38);
    *out = Decimal128{0, 0};
    if (!d128_within_38_digits(sum)) return DivStatus::Overflow;
    Decimal128 q{};
    const DivStatus ds = d128_div_scaled(sum, in_s, d128_from_i64(count), 0, out_s,
                                         k_decimal_avg_round, &q);
    if (ds != DivStatus::Ok) return ds;
    if (!d128_within_38_digits(q)) return DivStatus::Overflow;
    *out = q;
    return DivStatus::Ok;
}

// a % b with the dividend's sign, at scale max(sa, sb).
inline DivStatus d128_rem_exact(Decimal128 a, int sa, Decimal128 b, int sb,
                                Decimal128* out) noexcept {
    using namespace div_detail;
    assert(out != nullptr);
    assert(sa >= 0 && sb >= 0 && sa <= 76 && sb <= 76);
    if (b.lo == 0 && b.hi == 0) return DivStatus::DivZero;
    const bool neg = a.hi < 0;
    U256 n = u256_from(abs128(a));
    U256 d = u256_from(abs128(b));
    const int t = sa > sb ? sa : sb;
    if (!u256_mul_pow10(&n, t - sa) || !u256_mul_pow10(&d, t - sb)) return DivStatus::Overflow;
    U256 q{}, r{};
    u256_divmod(n, d, &q, &r);
    return to_d128(r, neg, out) ? DivStatus::Ok : DivStatus::Overflow;
}

}  // namespace bolt::kernels::decimal
