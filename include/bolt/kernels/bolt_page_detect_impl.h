// bolt_page_detect_impl.h — internals of bolt_page_detect.h (MSEG K4).
// Include bolt_page_detect.h, not this file.
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt::page::detail {


BOLT_FORCE_INLINE uint64_t bit_mask(const uint8_t* v, uint64_t bit) noexcept {
    return 0ull - static_cast<uint64_t>((v[bit >> 3] >> (bit & 7)) & 1u);
}

BOLT_FORCE_INLINE uint64_t valid_mask(const uint8_t* v, uint64_t bit) noexcept {
    return v == nullptr ? ~0ull : bit_mask(v, bit);
}

BOLT_FORCE_INLINE uint64_t load_lo(const uint8_t* p, uint32_t w) noexcept {
    uint64_t x = 0;
    std::memcpy(&x, p, w < 8 ? w : 8);
    return x;
}

BOLT_FORCE_INLINE uint64_t load_hi(const uint8_t* p, uint32_t w) noexcept {
    uint64_t x = 0;
    if (w > 8) std::memcpy(&x, p + 8, w - 8);
    return x;
}


BOLT_FORCE_INLINE uint32_t nulls_in(const uint8_t* v, uint64_t voff, uint32_t n) noexcept {
    if (v == nullptr) return 0;
    return n - static_cast<uint32_t>(bitmap_popcount(v, voff, voff + n));
}

// Validity bits [bit, bit + 8) (all inside the bitmap).
BOLT_FORCE_INLINE uint32_t valid_byte(const uint8_t* v, uint64_t bit) noexcept {
    const uint32_t sh = static_cast<uint32_t>(bit & 7);
    uint32_t b = static_cast<uint32_t>(v[bit >> 3]) >> sh;
    if (sh != 0) b |= static_cast<uint32_t>(v[(bit >> 3) + 1]) << (8 - sh);
    return b & 0xFFu;
}

template <uint32_t W> struct UintOf { using T = uint64_t; };
template <> struct UintOf<1> { using T = uint8_t; };
template <> struct UintOf<2> { using T = uint16_t; };
template <> struct UintOf<4> { using T = uint32_t; };

// A row's difference from the first value (f0, f1) over the bytes that carry
// the value (hm, tm: all for fixed widths; length + covered bytes for an inline
// StringView); zero iff equal. Narrow widths stay in their own width so the
// loops vectorise.
template <uint32_t W>
BOLT_FORCE_INLINE typename UintOf<W>::T row_diff(const uint8_t* BOLT_RESTRICT s, uint64_t f0,
                                                 uint64_t f1, uint64_t hm, uint64_t tm) noexcept {
    using T = typename UintOf<W>::T;
    if constexpr (W <= 8) {
        T x;
        std::memcpy(&x, s, W);
        return static_cast<T>((x ^ static_cast<T>(f0)) & static_cast<T>(hm));
    } else {
        static_assert(W == 16, "fixed widths are 1, 2, 4, 8 or 16 bytes");
        uint64_t lo, hi;
        std::memcpy(&lo, s, 8);
        std::memcpy(&hi, s + 8, 8);
        (void)f1;
        return ((lo ^ f0) & hm) | ((hi ^ f1) & tm);
    }
}

// Dense rows (no bitmap): copy (when d) and OR the differences; vectorises.
template <uint32_t W, bool kCopy>
inline uint64_t dense_rows(const uint8_t* BOLT_RESTRICT s, uint32_t n, uint8_t* BOLT_RESTRICT d,
                           uint64_t f0, uint64_t f1, uint64_t hm, uint64_t tm) noexcept {
    using T = typename UintOf<W>::T;
    assert(kCopy == (d != nullptr));
    assert(n > 0 && s != nullptr);
    T diff = 0;
    for (uint32_t i = 0; i < n; ++i) {  // bounded: n
        const uint8_t* p = s + static_cast<size_t>(i) * W;
        if (kCopy) std::memcpy(d + static_cast<size_t>(i) * W, p, W);
        diff = static_cast<T>(diff | row_diff<W>(p, f0, f1, hm, tm));
    }
    return diff;
}

template <uint32_t W>
BOLT_FORCE_INLINE void masked_row(const uint8_t* BOLT_RESTRICT s, uint8_t* BOLT_RESTRICT d,
                                  uint64_t m) noexcept {
    uint64_t lo = 0, hi = 0;
    std::memcpy(&lo, s, W < 8 ? W : 8);
    if (W > 8) std::memcpy(&hi, s + 8, W - 8);
    lo &= m; hi &= m;
    std::memcpy(d, &lo, W < 8 ? W : 8);
    if (W > 8) std::memcpy(d + 8, &hi, W - 8);
}

// The copy alone, null slots zeroed (8 rows per validity byte).
template <uint32_t W>
inline void masked_copy_rows(const uint8_t* s, const uint8_t* v, uint64_t voff, uint32_t n,
                             uint8_t* d) noexcept {
    assert(s != nullptr && d != nullptr);
    assert(n > 0);
    if (v == nullptr) {
        std::memcpy(d, s, static_cast<size_t>(n) * W);
        return;
    }
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {  // bounded: n / 8
        const uint32_t bits = valid_byte(v, voff + i);
        for (uint32_t j = 0; j < 8; ++j)  // bounded: 8
            masked_row<W>(s + static_cast<size_t>(i + j) * W, d + static_cast<size_t>(i + j) * W,
                          0ull - ((bits >> j) & 1u));
    }
    for (; i < n; ++i)  // bounded: n
        masked_row<W>(s + static_cast<size_t>(i) * W, d + static_cast<size_t>(i) * W,
                      bit_mask(v, voff + i));
}

// Rows with a bitmap: copy, flag each row that differs (ignoring validity;
// vectorises), then AND the flags with the validity 8 rows at a time.
template <uint32_t W>
inline uint64_t masked_rows(const uint8_t* s, const uint8_t* v, uint64_t voff, uint32_t n,
                            uint8_t* d, uint64_t f0, uint64_t f1, uint64_t hm,
                            uint64_t tm) noexcept {
    assert(v != nullptr && s != nullptr);
    assert(n > 0 && n <= stats::kPageDetectBlockRows);
    if (d != nullptr) masked_copy_rows<W>(s, v, voff, n, d);
    alignas(8) uint8_t ne[stats::kPageDetectBlockRows];
    for (uint32_t i = 0; i < n; ++i)  // bounded: n
        ne[i] = row_diff<W>(s + static_cast<size_t>(i) * W, f0, f1, hm, tm) != 0;
    uint64_t diff = 0;
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {  // bounded: n / 8
        uint64_t x;
        std::memcpy(&x, ne + i, 8);
        diff |= ((x * 0x0102040810204080ull) >> 56) & valid_byte(v, voff + i);
    }
    for (; i < n; ++i) diff |= ne[i] & (bit_mask(v, voff + i) & 1u);  // bounded: n
    return diff;
}

// One block of at most kPageDetectBlockRows rows.
template <uint32_t W>
BOLT_FORCE_INLINE uint64_t block_rows(const uint8_t* s, const uint8_t* v, uint64_t voff,
                                      uint32_t n, uint8_t* d, uint64_t f0, uint64_t f1,
                                      uint64_t hm, uint64_t tm) noexcept {
    if (v != nullptr) return masked_rows<W>(s, v, voff, n, d, f0, f1, hm, tm);
    return d != nullptr ? dense_rows<W, true>(s, n, d, f0, f1, hm, tm)
                        : dense_rows<W, false>(s, n, nullptr, f0, f1, hm, tm);
}

// Rows [0, n) once the first value is known: compare block by block while
// the page is a candidate, then only copy.
template <uint32_t W>
inline void fixed_detect_rows(FixedDetectState* st, const uint8_t* s, const uint8_t* v,
                              uint64_t voff, uint32_t n, uint8_t* d) noexcept {
    assert(st->have_first && st->width == W);
    assert(n > 0);
    uint32_t i = 0;
    while (i < n && st->diff == 0) {  // bounded: n / kPageDetectBlockRows
        const uint32_t k = n - i < stats::kPageDetectBlockRows ? n - i : stats::kPageDetectBlockRows;
        st->diff |= block_rows<W>(s + static_cast<size_t>(i) * W, v, voff + i, k,
                                  d != nullptr ? d + static_cast<size_t>(i) * W : nullptr,
                                  st->first_lo, st->first_hi, ~0ull, ~0ull);
        i += k;
    }
    if (i < n && d != nullptr)
        masked_copy_rows<W>(s + static_cast<size_t>(i) * W, v, voff + i, n - i,
                            d + static_cast<size_t>(i) * W);
    st->nulls += nulls_in(v, voff, n);
}

template <uint32_t W>
inline void fixed_detect_dispatch(FixedDetectState* st, const uint8_t* s, const uint8_t* v,
                                  uint64_t voff, uint32_t n, uint8_t* d) noexcept {
    uint32_t i = 0;
    while (!st->have_first && i < n) {  // bounded: n
        const uint8_t* p = s + static_cast<size_t>(i) * W;
        if (valid_mask(v, voff + i) != 0) {
            st->first_lo = load_lo(p, W);
            st->first_hi = load_hi(p, W);
            st->have_first = true;
            break;
        }
        ++st->nulls;
        if (d != nullptr) std::memset(d + static_cast<size_t>(i) * W, 0, W);
        ++i;
    }
    if (i < n)
        fixed_detect_rows<W>(st, s + static_cast<size_t>(i) * W, v, voff + i, n - i,
                             d != nullptr ? d + static_cast<size_t>(i) * W : nullptr);
}


BOLT_FORCE_INLINE uint64_t byte_mask(uint32_t nbytes) noexcept {
    return nbytes >= 8 ? ~0ull : ((1ull << (nbytes * 8)) - 1);
}

inline void sv_set_first(SvDetectState* st, const StringView& v, const char* overflow) noexcept {
    assert(!st->have_first);
    st->first = v;
    st->have_first = true;
    const uint32_t len = v.length;
    st->hdr_mask = byte_mask(4 + (len < 4 ? len : 4));
    st->tail_mask = len <= 12 ? byte_mask(len > 4 ? len - 4 : 0) : 0;
    st->first_bytes = len <= 12 ? st->first.prefix : overflow + v.ref.offset;
    assert(len <= 12 || overflow != nullptr);
}

// A long view equal to first in length + prefix but with another reference.
inline bool sv_long_differs(const SvDetectState* st, const StringView& v,
                            const char* overflow) noexcept {
    assert(v.length == st->first.length && v.length > 12);
    assert(overflow != nullptr);
    if (v.ref.offset == st->first.ref.offset && overflow + v.ref.offset == st->first_bytes)
        return false;
    return std::memcmp(overflow + v.ref.offset, st->first_bytes, v.length) != 0;
}

// First value inline (<= 12 B): length + the bytes it covers decide.
inline uint64_t sv_rows_inline(const SvDetectState* st, const StringView* s, const uint8_t* v,
                               uint64_t voff, uint32_t n, StringView* d) noexcept {
    assert(st->have_first && st->first.length <= 12);
    assert(n > 0 && n <= stats::kPageDetectBlockRows);
    uint64_t f[2];
    std::memcpy(f, &st->first, sizeof(f));
    return block_rows<sizeof(StringView)>(reinterpret_cast<const uint8_t*>(s), v, voff, n,
                                          reinterpret_cast<uint8_t*>(d), f[0], f[1],
                                          st->hdr_mask, st->tail_mask);
}

// First value long: equal length + prefix, then the bytes while a candidate.
inline void sv_rows_long(SvDetectState* st, const StringView* s, const uint8_t* validity,
                         uint64_t voff, uint32_t n, StringView* d, const char* overflow) noexcept {
    assert(st->have_first && st->first.length > 12);
    if (st->differs) {
        if (d != nullptr)
            masked_copy_rows<sizeof(StringView)>(reinterpret_cast<const uint8_t*>(s), validity, voff,
                                                 n, reinterpret_cast<uint8_t*>(d));
        st->nulls += nulls_in(validity, voff, n);
        return;
    }
    uint64_t f[2];
    std::memcpy(f, &st->first, sizeof(f));
    bool differs = false;
    for (uint32_t i = 0; i < n; ++i) {  // bounded: n
        const uint64_t m = valid_mask(validity, voff + i);
        uint64_t x[2];
        std::memcpy(x, s + i, sizeof(x));
        x[0] &= m; x[1] &= m;
        if (d != nullptr) std::memcpy(d + i, x, sizeof(x));
        if (m != 0) differs = x[0] != f[0] || (x[1] != f[1] && sv_long_differs(st, s[i], overflow));
        if (differs) {
            if (i + 1 < n && d != nullptr)
                masked_copy_rows<sizeof(StringView)>(reinterpret_cast<const uint8_t*>(s + i + 1),
                                                     validity, voff + i + 1, n - i - 1,
                                                     reinterpret_cast<uint8_t*>(d + i + 1));
            break;
        }
    }
    st->differs = differs;
    st->nulls += nulls_in(validity, voff, n);
}

inline void sv_detect_rows(SvDetectState* st, const StringView* s, const uint8_t* validity,
                           uint64_t voff, uint32_t n, StringView* d,
                           const char* overflow) noexcept {
    assert(st->have_first);
    assert(n > 0 && s != nullptr);
    if (st->first.length > 12) {
        sv_rows_long(st, s, validity, voff, n, d, overflow);
        return;
    }
    uint32_t i = 0;
    while (i < n && !st->differs) {  // bounded: n / kPageDetectBlockRows
        const uint32_t k = n - i < stats::kPageDetectBlockRows ? n - i : stats::kPageDetectBlockRows;
        StringView* dk = d != nullptr ? d + i : nullptr;
        st->differs = sv_rows_inline(st, s + i, validity, voff + i, k, dk) != 0;
        i += k;
    }
    if (i < n && d != nullptr)
        masked_copy_rows<sizeof(StringView)>(reinterpret_cast<const uint8_t*>(s + i), validity,
                                             voff + i, n - i, reinterpret_cast<uint8_t*>(d + i));
    st->nulls += nulls_in(validity, voff, n);
}

}  // namespace bolt::page::detail
