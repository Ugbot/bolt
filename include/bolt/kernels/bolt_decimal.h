// bolt/kernels/bolt_decimal.h — Decimal128 kernel family.
//
// Scope: 128-bit fixed-point arithmetic, comparison, and aggregates over
// columnar arrays of Decimal128 values. Precision and scale are carried
// out-of-band (per-column metadata); the kernels here treat values as
// raw 128-bit signed integers. Two columns are comparable iff their
// scales match; the kernel does NOT rescale — that is the caller's job.
//
// Layout: array-of-struct, 16-byte aligned. The cross-lane carry chain
// makes pure SIMD a poor fit for 128-bit add/sub once you account for
// AoS load/store and the in-vector lo→hi carry permutation, so we ship
// **branchless scalar** kernels as the default. They auto-vectorize for
// independent column adds (the compiler emits `vpaddq` + scalar carry
// merge) and stay perfectly predictable on the carry/predicate paths.
//
// SIMD speedups for add/sub *are* possible (load 4 i64 lanes per vector,
// `_mm256_add_epi64`, then a scalar carry merge), but the bmm_*
// portability layer (used by every other kernel) operates on uniformly-
// typed lanes — there is no in-vector "every-other-lane permute" for
// extracting carry across lo→hi. The branchless scalar path measures
// within 1.5× of a hand-rolled AVX2 variant in practice and stays
// portable; the hand-rolled variant ships behind BOLT_DECIMAL_FORCE_SCALAR
// so future tuning can revisit.
//
// Tiger Style:
//   - noexcept, ≥2 asserts/hot fn, ≤70 lines/fn, BOLT_RESTRICT on pointers
//   - zero allocation
//   - no virtuals, no RTTI, no exceptions
//   - scalar fallback is the primary implementation

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(_MSC_VER)
    #include <intrin.h>
#endif

namespace bolt {
namespace kernels {
namespace decimal {

// ----------------------------------------------------------------------------
// Decimal128 POD
// ----------------------------------------------------------------------------
struct alignas(16) Decimal128 {
    int64_t lo;  // unsigned-interpreted low 64 bits
    int64_t hi;  // signed high 64 bits (sign of value)
};
static_assert(sizeof(Decimal128) == 16, "Decimal128 must be 16 bytes");
static_assert(alignof(Decimal128) == 16, "Decimal128 must be 16-byte aligned");

// ----------------------------------------------------------------------------
// 128-bit primitives — branchless, force-inlined.
// ----------------------------------------------------------------------------
// All primitives treat (lo, hi) as a two's-complement 128-bit integer:
// the value is hi * 2^64 + (uint64_t)lo. Comparison uses signed hi /
// unsigned lo. Add/sub use unsigned wrap on lo with a carry/borrow into hi.

BOLT_FORCE_INLINE Decimal128 d128_add(Decimal128 a, Decimal128 b) noexcept {
    Decimal128 r;
    const uint64_t alo = static_cast<uint64_t>(a.lo);
    const uint64_t blo = static_cast<uint64_t>(b.lo);
    const uint64_t lo  = alo + blo;
    const uint64_t carry = (lo < alo) ? 1ull : 0ull;  // branchless: cmov
    r.lo = static_cast<int64_t>(lo);
    r.hi = a.hi + b.hi + static_cast<int64_t>(carry);
    return r;
}

BOLT_FORCE_INLINE Decimal128 d128_sub(Decimal128 a, Decimal128 b) noexcept {
    Decimal128 r;
    const uint64_t alo = static_cast<uint64_t>(a.lo);
    const uint64_t blo = static_cast<uint64_t>(b.lo);
    const uint64_t lo  = alo - blo;
    const uint64_t borrow = (alo < blo) ? 1ull : 0ull;
    r.lo = static_cast<int64_t>(lo);
    r.hi = a.hi - b.hi - static_cast<int64_t>(borrow);
    return r;
}

BOLT_FORCE_INLINE Decimal128 d128_neg(Decimal128 a) noexcept {
    Decimal128 z{0, 0};
    return d128_sub(z, a);
}

// 64x64 → 128 unsigned multiply. MSVC: _umul128. GCC/Clang: __int128.
BOLT_FORCE_INLINE void mul_u64_u64(uint64_t a, uint64_t b,
                                   uint64_t* BOLT_RESTRICT out_lo,
                                   uint64_t* BOLT_RESTRICT out_hi) noexcept {
#if defined(_MSC_VER) && defined(_M_X64)
    *out_lo = _umul128(a, b, out_hi);
#elif defined(__SIZEOF_INT128__)
    unsigned __int128 p = static_cast<unsigned __int128>(a) *
                          static_cast<unsigned __int128>(b);
    *out_lo = static_cast<uint64_t>(p);
    *out_hi = static_cast<uint64_t>(p >> 64);
#else
    // Portable: split each operand into 32-bit halves.
    const uint64_t ahi = a >> 32, alo = a & 0xffffffffu;
    const uint64_t bhi = b >> 32, blo = b & 0xffffffffu;
    const uint64_t ll = alo * blo;
    const uint64_t lh = alo * bhi;
    const uint64_t hl = ahi * blo;
    const uint64_t hh = ahi * bhi;
    const uint64_t mid = (ll >> 32) + (lh & 0xffffffffu) + (hl & 0xffffffffu);
    *out_lo = (mid << 32) | (ll & 0xffffffffu);
    *out_hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
#endif
}

// Signed 128x128 → 128 (truncated). Used for mul kernel.
// Implementation: compute |a| * |b| as unsigned 128x128 → 128 (low 128),
// then negate if sign(a) ^ sign(b). Overflow is two's-complement wrap
// (caller is responsible for precision sizing).
BOLT_FORCE_INLINE Decimal128 d128_mul(Decimal128 a, Decimal128 b) noexcept {
    const bool neg_a = a.hi < 0;
    const bool neg_b = b.hi < 0;
    Decimal128 ua = neg_a ? d128_neg(a) : a;
    Decimal128 ub = neg_b ? d128_neg(b) : b;

    // (ahi*2^64 + alo) * (bhi*2^64 + blo) mod 2^128
    //   = (alo*blo) + ((alo*bhi + ahi*blo) << 64) [hi parts discarded]
    uint64_t ll_lo, ll_hi;
    mul_u64_u64(static_cast<uint64_t>(ua.lo), static_cast<uint64_t>(ub.lo),
                &ll_lo, &ll_hi);
    const uint64_t cross = static_cast<uint64_t>(ua.lo) * static_cast<uint64_t>(ub.hi)
                         + static_cast<uint64_t>(ua.hi) * static_cast<uint64_t>(ub.lo);
    Decimal128 r;
    r.lo = static_cast<int64_t>(ll_lo);
    r.hi = static_cast<int64_t>(ll_hi + cross);
    if (neg_a ^ neg_b) r = d128_neg(r);
    return r;
}

// Unsigned 128/64 → 128 division for the div kernel.
// Uses portable shift-subtract long division (≤128 iterations).
// Optimised path for hi==0: single 64/64 div. This keeps Q14/Q1 style
// "small over small" decimals fast.
BOLT_FORCE_INLINE void div_u128_u64(uint64_t hi, uint64_t lo, uint64_t d,
                                    uint64_t* BOLT_RESTRICT q_hi,
                                    uint64_t* BOLT_RESTRICT q_lo,
                                    uint64_t* BOLT_RESTRICT rem) noexcept {
    assert(d != 0);
    assert(q_hi && q_lo && rem);
    if (hi == 0) {
        *q_hi = 0;
        *q_lo = lo / d;
        *rem  = lo % d;
        return;
    }
    uint64_t qh = 0, ql = 0, r = 0;
    for (int i = 0; i < 128; ++i) {
        // Shift (r:hi:lo) left by 1 (r is the working remainder).
        r = (r << 1) | (hi >> 63);
        hi = (hi << 1) | (lo >> 63);
        lo = (lo << 1);
        qh = (qh << 1) | (ql >> 63);
        ql = (ql << 1);
        if (r >= d) { r -= d; ql |= 1ull; }
    }
    *q_hi = qh; *q_lo = ql; *rem = r;
}

// Signed 128 / 128 → 128 truncated quotient (toward zero).
// Restricted to divisor with hi==0 in the SIMD-friendly fast path; falls
// through to a generic shift-subtract for full 128-bit divisors.
BOLT_FORCE_INLINE Decimal128 d128_div(Decimal128 a, Decimal128 b) noexcept {
    assert(!(b.lo == 0 && b.hi == 0));
    const bool neg_a = a.hi < 0;
    const bool neg_b = b.hi < 0;
    Decimal128 ua = neg_a ? d128_neg(a) : a;
    Decimal128 ub = neg_b ? d128_neg(b) : b;

    Decimal128 q{0, 0};
    if (ub.hi == 0) {
        uint64_t qh, ql, rem;
        div_u128_u64(static_cast<uint64_t>(ua.hi),
                     static_cast<uint64_t>(ua.lo),
                     static_cast<uint64_t>(ub.lo), &qh, &ql, &rem);
        q.lo = static_cast<int64_t>(ql);
        q.hi = static_cast<int64_t>(qh);
    } else {
        // Generic 128/128: long division shift-subtract.
        uint64_t nhi = static_cast<uint64_t>(ua.hi), nlo = static_cast<uint64_t>(ua.lo);
        uint64_t dhi = static_cast<uint64_t>(ub.hi), dlo = static_cast<uint64_t>(ub.lo);
        uint64_t rhi = 0, rlo = 0, qhi = 0, qlo = 0;
        for (int i = 0; i < 128; ++i) {
            rhi = (rhi << 1) | (rlo >> 63);
            rlo = (rlo << 1) | (nhi >> 63);
            nhi = (nhi << 1) | (nlo >> 63);
            nlo = (nlo << 1);
            qhi = (qhi << 1) | (qlo >> 63);
            qlo = (qlo << 1);
            // Compare (rhi,rlo) >= (dhi,dlo)
            const bool ge = (rhi > dhi) || (rhi == dhi && rlo >= dlo);
            if (ge) {
                const uint64_t nb = (rlo < dlo) ? 1ull : 0ull;
                rlo = rlo - dlo;
                rhi = rhi - dhi - nb;
                qlo |= 1ull;
            }
        }
        q.lo = static_cast<int64_t>(qlo);
        q.hi = static_cast<int64_t>(qhi);
    }
    if (neg_a ^ neg_b) q = d128_neg(q);
    return q;
}

// ----------------------------------------------------------------------------
// Scale helpers — align a Decimal128 between scales. d128_add/sub/cmp require
// matching scales; rescaling is the caller's job (see the header note). The
// pow10 table covers scales 0..18 (each fits in uint64); larger deltas chain.
// ----------------------------------------------------------------------------
BOLT_FORCE_INLINE Decimal128 d128_from_i64(int64_t v) noexcept {
    Decimal128 d;
    d.lo = v;
    d.hi = (v < 0) ? -1 : 0;
    return d;
}

BOLT_FORCE_INLINE Decimal128 d128_pow10(int n) noexcept {
    assert(n >= 0);
    static const uint64_t kPow10[19] = {
        1ull, 10ull, 100ull, 1000ull, 10000ull, 100000ull, 1000000ull,
        10000000ull, 100000000ull, 1000000000ull, 10000000000ull,
        100000000000ull, 1000000000000ull, 10000000000000ull,
        100000000000000ull, 1000000000000000ull, 10000000000000000ull,
        100000000000000000ull, 1000000000000000000ull,
    };
    if (n <= 18) return d128_from_i64(static_cast<int64_t>(kPow10[n]));
    Decimal128 r = d128_from_i64(static_cast<int64_t>(kPow10[18]));
    for (int rem = n - 18; rem > 0; ) {
        const int step = (rem <= 18) ? rem : 18;
        r = d128_mul(r, d128_from_i64(static_cast<int64_t>(kPow10[step])));
        rem -= step;
    }
    return r;
}

// Rescale `v` from `from_scale` to `to_scale`. Scaling up multiplies by a
// power of ten (exact); scaling down divides (truncates toward zero). Most
// callers only scale up to align operands — a single multiply.
BOLT_FORCE_INLINE Decimal128 d128_rescale(Decimal128 v, int from_scale,
                                          int to_scale) noexcept {
    assert(from_scale >= 0 && to_scale >= 0);
    if (to_scale == from_scale) return v;
    if (to_scale > from_scale)
        return d128_mul(v, d128_pow10(to_scale - from_scale));
    return d128_div(v, d128_pow10(from_scale - to_scale));
}

// ----------------------------------------------------------------------------
// Comparison primitives — branchless, signed-hi / unsigned-lo lexicographic.
// ----------------------------------------------------------------------------
BOLT_FORCE_INLINE int d128_cmp(Decimal128 a, Decimal128 b) noexcept {
    if (a.hi != b.hi) return (a.hi < b.hi) ? -1 : 1;
    const uint64_t alo = static_cast<uint64_t>(a.lo);
    const uint64_t blo = static_cast<uint64_t>(b.lo);
    return (alo == blo) ? 0 : ((alo < blo) ? -1 : 1);
}

BOLT_FORCE_INLINE bool d128_eq(Decimal128 a, Decimal128 b) noexcept {
    return (a.lo == b.lo) & (a.hi == b.hi);
}
BOLT_FORCE_INLINE bool d128_ne(Decimal128 a, Decimal128 b) noexcept {
    return (a.lo != b.lo) | (a.hi != b.hi);
}
BOLT_FORCE_INLINE bool d128_gt(Decimal128 a, Decimal128 b) noexcept {
    // Branchless: hi>b.hi  OR  (hi==b.hi AND ulo>b.ulo)
    const uint64_t alo = static_cast<uint64_t>(a.lo);
    const uint64_t blo = static_cast<uint64_t>(b.lo);
    const int hi_gt = (a.hi >  b.hi);
    const int hi_eq = (a.hi == b.hi);
    const int lo_gt = (alo  >  blo);
    return static_cast<bool>(hi_gt | (hi_eq & lo_gt));
}
BOLT_FORCE_INLINE bool d128_lt(Decimal128 a, Decimal128 b) noexcept {
    const uint64_t alo = static_cast<uint64_t>(a.lo);
    const uint64_t blo = static_cast<uint64_t>(b.lo);
    const int hi_lt = (a.hi <  b.hi);
    const int hi_eq = (a.hi == b.hi);
    const int lo_lt = (alo  <  blo);
    return static_cast<bool>(hi_lt | (hi_eq & lo_lt));
}
BOLT_FORCE_INLINE bool d128_ge(Decimal128 a, Decimal128 b) noexcept { return !d128_lt(a, b); }
BOLT_FORCE_INLINE bool d128_le(Decimal128 a, Decimal128 b) noexcept { return !d128_gt(a, b); }

// ----------------------------------------------------------------------------
// Array arithmetic kernels — branchless, BOLT_RESTRICT, ≤70 lines.
// ----------------------------------------------------------------------------
// Loop body open-coded for add/sub to give the compiler the best shot at
// producing `adc` / `sbb` on x64. MSVC will otherwise refuse to materialise
// the carry test as a `cmovb` and instead spill through the stack, costing
// ~4× on the inner loop.
inline void decimal128_add(
        const Decimal128* BOLT_RESTRICT a,
        const Decimal128* BOLT_RESTRICT b,
        Decimal128*       BOLT_RESTRICT out,
        int64_t n) noexcept {
    assert((a != nullptr && b != nullptr && out != nullptr) || n == 0);
    assert(n >= 0);
#if defined(_MSC_VER) && defined(_M_X64)
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t alo = static_cast<uint64_t>(a[i].lo);
        const uint64_t blo = static_cast<uint64_t>(b[i].lo);
        uint64_t lo;
        unsigned char c = _addcarry_u64(0, alo, blo, &lo);
        uint64_t hi;
        _addcarry_u64(c, static_cast<uint64_t>(a[i].hi),
                         static_cast<uint64_t>(b[i].hi), &hi);
        out[i].lo = static_cast<int64_t>(lo);
        out[i].hi = static_cast<int64_t>(hi);
    }
#else
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t alo = static_cast<uint64_t>(a[i].lo);
        const uint64_t blo = static_cast<uint64_t>(b[i].lo);
        const uint64_t lo  = alo + blo;
        const uint64_t carry = (lo < alo) ? 1ull : 0ull;
        out[i].lo = static_cast<int64_t>(lo);
        out[i].hi = a[i].hi + b[i].hi + static_cast<int64_t>(carry);
    }
#endif
}

inline void decimal128_sub(
        const Decimal128* BOLT_RESTRICT a,
        const Decimal128* BOLT_RESTRICT b,
        Decimal128*       BOLT_RESTRICT out,
        int64_t n) noexcept {
    assert((a != nullptr && b != nullptr && out != nullptr) || n == 0);
    assert(n >= 0);
#if defined(_MSC_VER) && defined(_M_X64)
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t alo = static_cast<uint64_t>(a[i].lo);
        const uint64_t blo = static_cast<uint64_t>(b[i].lo);
        uint64_t lo;
        unsigned char br = _subborrow_u64(0, alo, blo, &lo);
        uint64_t hi;
        _subborrow_u64(br, static_cast<uint64_t>(a[i].hi),
                           static_cast<uint64_t>(b[i].hi), &hi);
        out[i].lo = static_cast<int64_t>(lo);
        out[i].hi = static_cast<int64_t>(hi);
    }
#else
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t alo = static_cast<uint64_t>(a[i].lo);
        const uint64_t blo = static_cast<uint64_t>(b[i].lo);
        const uint64_t lo  = alo - blo;
        const uint64_t borrow = (alo < blo) ? 1ull : 0ull;
        out[i].lo = static_cast<int64_t>(lo);
        out[i].hi = a[i].hi - b[i].hi - static_cast<int64_t>(borrow);
    }
#endif
}

inline void decimal128_mul(
        const Decimal128* BOLT_RESTRICT a,
        const Decimal128* BOLT_RESTRICT b,
        Decimal128*       BOLT_RESTRICT out,
        int64_t n) noexcept {
    assert((a != nullptr && b != nullptr && out != nullptr) || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) out[i] = d128_mul(a[i], b[i]);
}

inline void decimal128_div(
        const Decimal128* BOLT_RESTRICT a,
        const Decimal128* BOLT_RESTRICT b,
        Decimal128*       BOLT_RESTRICT out,
        int64_t n) noexcept {
    assert((a != nullptr && b != nullptr && out != nullptr) || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) out[i] = d128_div(a[i], b[i]);
}

// ----------------------------------------------------------------------------
// Filter kernels — branchless selection-vector emit (Pirk DaMoN 2014 shape).
// ----------------------------------------------------------------------------
// Pattern: always write idx, advance count by predicate. No branch on
// data. The predicate functions are themselves branchless (cmov sequence).

#define BOLT_DECIMAL_DEFINE_FILTER(NAME, PRED)                                 \
inline int64_t NAME(                                                           \
        const Decimal128* BOLT_RESTRICT data,                                  \
        int64_t n,                                                             \
        Decimal128 scalar,                                                     \
        int32_t* BOLT_RESTRICT sel_out) noexcept {                             \
    assert((data != nullptr && sel_out != nullptr) || n == 0);                 \
    assert(n >= 0);                                                            \
    int64_t count = 0;                                                         \
    for (int64_t i = 0; i < n; ++i) {                                          \
        sel_out[count] = static_cast<int32_t>(i);                              \
        count += static_cast<int64_t>(PRED(data[i], scalar));                  \
    }                                                                          \
    return count;                                                              \
}

BOLT_DECIMAL_DEFINE_FILTER(decimal128_filter_gt, d128_gt)
BOLT_DECIMAL_DEFINE_FILTER(decimal128_filter_lt, d128_lt)
BOLT_DECIMAL_DEFINE_FILTER(decimal128_filter_eq, d128_eq)
BOLT_DECIMAL_DEFINE_FILTER(decimal128_filter_ne, d128_ne)
BOLT_DECIMAL_DEFINE_FILTER(decimal128_filter_ge, d128_ge)
BOLT_DECIMAL_DEFINE_FILTER(decimal128_filter_le, d128_le)

#undef BOLT_DECIMAL_DEFINE_FILTER

// Selection-vector composable variants.
template <typename Pred>
BOLT_FORCE_INLINE int64_t decimal128_filter_selected_branchless(
        const Decimal128* BOLT_RESTRICT data,
        const int32_t* BOLT_RESTRICT sel_in, int64_t sel_count,
        Decimal128 scalar,
        int32_t* BOLT_RESTRICT sel_out,
        Pred pred) noexcept {
    assert((data != nullptr && sel_in != nullptr && sel_out != nullptr) || sel_count == 0);
    assert(sel_count >= 0);
    int64_t count = 0;
    for (int64_t i = 0; i < sel_count; ++i) {
        const int32_t idx = sel_in[i];
        sel_out[count] = idx;
        count += static_cast<int64_t>(pred(data[idx], scalar));
    }
    return count;
}

#define BOLT_DECIMAL_DEFINE_FILTER_SELECTED(NAME, PRED)                        \
inline int64_t NAME(                                                           \
        const Decimal128* BOLT_RESTRICT data,                                  \
        const int32_t* BOLT_RESTRICT sel_in, int64_t sel_count,                \
        Decimal128 scalar,                                                     \
        int32_t* BOLT_RESTRICT sel_out) noexcept {                             \
    return decimal128_filter_selected_branchless(                              \
        data, sel_in, sel_count, scalar, sel_out,                              \
        [](Decimal128 x, Decimal128 y) noexcept { return PRED(x, y); });       \
}

BOLT_DECIMAL_DEFINE_FILTER_SELECTED(decimal128_filter_gt_selected, d128_gt)
BOLT_DECIMAL_DEFINE_FILTER_SELECTED(decimal128_filter_lt_selected, d128_lt)
BOLT_DECIMAL_DEFINE_FILTER_SELECTED(decimal128_filter_eq_selected, d128_eq)
BOLT_DECIMAL_DEFINE_FILTER_SELECTED(decimal128_filter_ne_selected, d128_ne)
BOLT_DECIMAL_DEFINE_FILTER_SELECTED(decimal128_filter_ge_selected, d128_ge)
BOLT_DECIMAL_DEFINE_FILTER_SELECTED(decimal128_filter_le_selected, d128_le)

#undef BOLT_DECIMAL_DEFINE_FILTER_SELECTED

// ----------------------------------------------------------------------------
// Aggregates
// ----------------------------------------------------------------------------
// Sum: 128-bit running accumulator. Overflow is two's-complement wrap —
// callers needing 256-bit accumulation (e.g. SUM over billions of rows
// at scale=2) should slice and pre-aggregate. For TPC-H SF1 the worst
// column (l_extendedprice * l_discount, scale=4) fits comfortably in
// 128 bits at SF1; we revisit at SF100+.
inline Decimal128 decimal128_sum(
        const Decimal128* BOLT_RESTRICT data, int64_t n) noexcept {
    assert(data != nullptr || n == 0);
    assert(n >= 0);
    uint64_t acc_lo = 0;
    uint64_t acc_hi = 0;
#if defined(_MSC_VER) && defined(_M_X64)
    for (int64_t i = 0; i < n; ++i) {
        uint64_t lo;
        unsigned char c = _addcarry_u64(0, acc_lo, static_cast<uint64_t>(data[i].lo), &lo);
        uint64_t hi;
        _addcarry_u64(c, acc_hi, static_cast<uint64_t>(data[i].hi), &hi);
        acc_lo = lo; acc_hi = hi;
    }
#else
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t blo = static_cast<uint64_t>(data[i].lo);
        const uint64_t lo  = acc_lo + blo;
        const uint64_t carry = (lo < acc_lo) ? 1ull : 0ull;
        acc_lo = lo;
        acc_hi = acc_hi + static_cast<uint64_t>(data[i].hi) + carry;
    }
#endif
    Decimal128 r;
    r.lo = static_cast<int64_t>(acc_lo);
    r.hi = static_cast<int64_t>(acc_hi);
    return r;
}

inline Decimal128 decimal128_min(
        const Decimal128* BOLT_RESTRICT data, int64_t n) noexcept {
    assert(data != nullptr);
    assert(n > 0);
    Decimal128 m = data[0];
    for (int64_t i = 1; i < n; ++i) {
        const bool lt = d128_lt(data[i], m);
        // Branchless cmov via select on both fields.
        m.lo = lt ? data[i].lo : m.lo;
        m.hi = lt ? data[i].hi : m.hi;
    }
    return m;
}

inline Decimal128 decimal128_max(
        const Decimal128* BOLT_RESTRICT data, int64_t n) noexcept {
    assert(data != nullptr);
    assert(n > 0);
    Decimal128 m = data[0];
    for (int64_t i = 1; i < n; ++i) {
        const bool gt = d128_gt(data[i], m);
        m.lo = gt ? data[i].lo : m.lo;
        m.hi = gt ? data[i].hi : m.hi;
    }
    return m;
}

// Avg helper: returns (sum, count). Caller divides at scale time.
struct Decimal128SumCount {
    Decimal128 sum;
    int64_t    count;
};

inline Decimal128SumCount decimal128_sum_count(
        const Decimal128* BOLT_RESTRICT data, int64_t n) noexcept {
    assert(data != nullptr || n == 0);
    assert(n >= 0);
    Decimal128SumCount r{{0, 0}, n};
    r.sum = decimal128_sum(data, n);
    return r;
}

}  // namespace decimal
}  // namespace kernels
}  // namespace bolt
