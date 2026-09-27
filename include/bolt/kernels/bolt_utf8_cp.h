// bolt_utf8_cp.h — UTF-8 code-point primitives shared by every string kernel
// that counts, slices, matches or case-maps characters (substring, LIKE '_',
// GLOB '?', regex '.', upper/lower).
//
// A "unit" is one byte plus every continuation byte (10xxxxxx) that follows
// it. On valid UTF-8 a unit is exactly one code point; on invalid input the
// definition still partitions the bytes (no read past `len`, no split inside
// a well-formed sequence), and it is the same rule utf8_count_codepoints
// uses, so LENGTH and SUBSTRING agree on every input.
//
// Case mapping is Unicode simple (1:1 code point) mapping, table-driven from
// DuckDB's upper()/lower() (bolt_utf8_case_tables.h), covering U+0000..U+058F,
// U+1E00..U+1FFF and the Kelvin/Ohm/Angstrom signs. Code points outside those
// blocks map to themselves. Mapping can change a unit's byte length (ß ->
// ẞ grows 2 -> 3, İ -> i shrinks 2 -> 1); the output never exceeds
// utf8_case_map_bound(len). A unit that is not well-formed UTF-8 is copied
// through unchanged.
//
// RULES: No exceptions, no heap, all fns noexcept, bounded loops.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_utf8_case_tables.h"

namespace bolt {
namespace kernels {
namespace utf8 {

constexpr uint32_t kUtf8Invalid = 0xFFFFFFFFu;

BOLT_FORCE_INLINE bool utf8_is_cont(uint8_t b) noexcept {
    return (b & 0xC0u) == 0x80u;
}

// True iff p[0..len) is 7-bit ASCII: one OR-reduction over 16-byte blocks
// (vectorised by the compiler) and a single high-bit test.
BOLT_FORCE_INLINE bool utf8_is_ascii(const char* BOLT_RESTRICT p,
                                     uint32_t len) noexcept {
    assert(p != nullptr || len == 0);
    uint64_t acc = 0;
    uint32_t i = 0;
    for (; i + 16u <= len; i += 16u) {
        uint64_t a, b;
        memcpy(&a, p + i, 8);
        memcpy(&b, p + i + 8, 8);
        acc |= a | b;
    }
    for (; i < len; ++i) acc |= static_cast<uint8_t>(p[i]);
    return (acc & 0x8080808080808080ull) == 0;
}

// Byte length of the unit starting at p[pos]; pos < len.
BOLT_FORCE_INLINE uint32_t utf8_unit_len(const char* BOLT_RESTRICT p,
                                         uint32_t len, uint32_t pos) noexcept {
    assert(p != nullptr);
    assert(pos < len);
    if (static_cast<uint8_t>(p[pos]) < 0x80u) return 1u;
    uint32_t k = pos + 1u;
    while (k < len && utf8_is_cont(static_cast<uint8_t>(p[k]))) ++k;
    return k - pos;
}

// Byte offset of the unit start before `pos` (pos > floor). Mirrors
// utf8_unit_len so forward and backward stepping visit the same boundaries.
BOLT_FORCE_INLINE uint32_t utf8_unit_back(const char* BOLT_RESTRICT p,
                                          uint32_t floor, uint32_t pos) noexcept {
    assert(p != nullptr);
    assert(pos > floor);
    uint32_t k = pos - 1u;
    while (k > floor && utf8_is_cont(static_cast<uint8_t>(p[k]))) --k;
    return k;
}

// Byte offset reached by advancing `n` units from byte offset `pos`,
// clamped to `len`. ASCII runs advance 8 bytes per step.
BOLT_FORCE_INLINE uint32_t utf8_cp_advance(const char* BOLT_RESTRICT p,
                                           uint32_t len, uint32_t pos,
                                           uint32_t n) noexcept {
    assert(p != nullptr || len == 0);
    assert(pos <= len);
    while (n > 0 && pos < len) {
        if (n >= 8u && pos + 8u <= len) {
            uint64_t w;
            memcpy(&w, p + pos, 8);
            if ((w & 0x8080808080808080ull) == 0) { pos += 8u; n -= 8u; continue; }
        }
        pos += utf8_unit_len(p, len, pos);
        --n;
    }
    assert(pos <= len);
    return pos;
}

// Strict decode of the unit at p[pos]. Returns the unit's byte length and
// sets *cp to the code point, or to kUtf8Invalid when the unit is not one
// well-formed scalar value (bad lead, wrong length, overlong, surrogate,
// > U+10FFFF).
BOLT_FORCE_INLINE uint32_t utf8_cp_decode(const char* BOLT_RESTRICT p,
                                          uint32_t len, uint32_t pos,
                                          uint32_t* cp) noexcept {
    assert(cp != nullptr);
    const uint32_t u = utf8_unit_len(p, len, pos);
    const uint8_t b0 = static_cast<uint8_t>(p[pos]);
    *cp = kUtf8Invalid;
    if (u == 1u) { if (b0 < 0x80u) *cp = b0; return 1u; }
    uint32_t need, v, min;
    if ((b0 & 0xE0u) == 0xC0u)      { need = 2; v = b0 & 0x1Fu; min = 0x80u; }
    else if ((b0 & 0xF0u) == 0xE0u) { need = 3; v = b0 & 0x0Fu; min = 0x800u; }
    else if ((b0 & 0xF8u) == 0xF0u) { need = 4; v = b0 & 0x07u; min = 0x10000u; }
    else return u;
    if (u != need) return u;
    for (uint32_t k = 1; k < need; ++k) {
        v = (v << 6) | (static_cast<uint8_t>(p[pos + k]) & 0x3Fu);
    }
    if (v < min || v > 0x10FFFFu || (v >= 0xD800u && v <= 0xDFFFu)) return u;
    *cp = v;
    return u;
}

// Encode a scalar value; returns bytes written (1..4) to out[0..4).
BOLT_FORCE_INLINE uint32_t utf8_cp_encode(uint32_t cp, char* out) noexcept {
    assert(out != nullptr);
    assert(cp <= 0x10FFFFu && !(cp >= 0xD800u && cp <= 0xDFFFu));
    if (cp < 0x80u) { out[0] = static_cast<char>(cp); return 1; }
    if (cp < 0x800u) {
        out[0] = static_cast<char>(0xC0u | (cp >> 6));
        out[1] = static_cast<char>(0x80u | (cp & 0x3Fu));
        return 2;
    }
    if (cp < 0x10000u) {
        out[0] = static_cast<char>(0xE0u | (cp >> 12));
        out[1] = static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu));
        out[2] = static_cast<char>(0x80u | (cp & 0x3Fu));
        return 3;
    }
    out[0] = static_cast<char>(0xF0u | (cp >> 18));
    out[1] = static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu));
    out[2] = static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu));
    out[3] = static_cast<char>(0x80u | (cp & 0x3Fu));
    return 4;
}

BOLT_FORCE_INLINE uint32_t utf8_cp_upper(uint32_t cp) noexcept {
    namespace t = case_tables;
    if (cp < t::kAHi) return t::kUpperA[cp];
    if (cp >= t::kBLo && cp < t::kBHi) return t::kUpperB[cp - t::kBLo];
    return cp;
}

BOLT_FORCE_INLINE uint32_t utf8_cp_lower(uint32_t cp) noexcept {
    namespace t = case_tables;
    if (cp < t::kAHi) return t::kLowerA[cp];
    if (cp >= t::kBLo && cp < t::kBHi) return t::kLowerB[cp - t::kBLo];
    for (uint32_t i = 0; i < sizeof(t::kLowerSparse) / sizeof(t::kLowerSparse[0]); ++i) {
        if (t::kLowerSparse[i][0] == cp) return t::kLowerSparse[i][1];
    }
    return cp;
}

// True when the code point's full Unicode case mapping (SpecialCasing: ß ->
// SS, İ -> i̇, final sigma, Greek iota subscripts, ...) differs from the
// simple mapping utf8_case_map applies. Callers whose reference semantics
// are the full mapping (XPath fn:lower-case) refuse such input.
BOLT_FORCE_INLINE bool utf8_cp_has_special_casing(uint32_t cp) noexcept {
    namespace t = case_tables;
    constexpr uint32_t n = sizeof(t::kSpecialCasing) / sizeof(t::kSpecialCasing[0]);
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (t::kSpecialCasing[mid] < cp) lo = mid + 1; else hi = mid;
    }
    assert(lo <= n);
    return lo < n && t::kSpecialCasing[lo] == cp;
}

BOLT_FORCE_INLINE bool utf8_has_special_casing(const char* BOLT_RESTRICT p,
                                               uint32_t len) noexcept {
    assert(p != nullptr || len == 0);
    for (uint32_t i = 0; i < len;) {
        if (static_cast<uint8_t>(p[i]) < 0x80u) { ++i; continue; }
        uint32_t cp;
        i += utf8_cp_decode(p, len, i, &cp);
        if (cp != kUtf8Invalid && utf8_cp_has_special_casing(cp)) return true;
    }
    return false;
}

// Upper bound on case-mapped output bytes: only 2-byte units grow (to 3).
BOLT_FORCE_INLINE uint32_t utf8_case_map_bound(uint32_t len) noexcept {
    return len + (len >> 1);
}

// Case-map src[0..len) into dst (capacity >= utf8_case_map_bound(len), or
// >= len when src is ASCII). Returns bytes written. dst == nullptr only
// measures. src and dst must not overlap.
BOLT_FORCE_INLINE uint32_t utf8_case_map(const char* BOLT_RESTRICT src,
                                         uint32_t len, bool upper,
                                         char* BOLT_RESTRICT dst) noexcept {
    assert(src != nullptr || len == 0);
    uint32_t w = 0;
    uint32_t i = 0;
    while (i < len) {
        const uint8_t c = static_cast<uint8_t>(src[i]);
        if (c < 0x80u) {
            const bool hit = upper ? (c >= 'a' && c <= 'z') : (c >= 'A' && c <= 'Z');
            if (dst != nullptr) dst[w] = static_cast<char>(hit ? (c ^ 0x20u) : c);
            ++w; ++i;
            continue;
        }
        uint32_t cp;
        const uint32_t u = utf8_cp_decode(src, len, i, &cp);
        if (cp == kUtf8Invalid) {
            if (dst != nullptr) memcpy(dst + w, src + i, u);
            w += u;
        } else {
            char enc[4];
            const uint32_t m = utf8_cp_encode(upper ? utf8_cp_upper(cp)
                                                    : utf8_cp_lower(cp), enc);
            if (dst != nullptr) memcpy(dst + w, enc, m);
            w += m;
        }
        i += u;
    }
    assert(w <= utf8_case_map_bound(len));
    return w;
}

// GLOB match: `*` any run, `?` exactly one unit, `escape` (a byte, or -1)
// makes the next pattern unit literal. Iterative with single-star
// backtracking (O(slen * plen), no recursion). A trailing lone escape
// matches nothing.
BOLT_FORCE_INLINE bool utf8_glob_match(const char* BOLT_RESTRICT s, uint32_t slen,
                                       const char* BOLT_RESTRICT p, uint32_t plen,
                                       int32_t escape = -1) noexcept {
    assert(s != nullptr || slen == 0);
    assert(p != nullptr || plen == 0);
    assert(escape >= -1 && escape <= 127);
    uint32_t si = 0, pi = 0;
    uint32_t star_p = 0xFFFFFFFFu, star_s = 0;
    while (si < slen) {
        if (pi < plen && p[pi] == '*') {
            star_p = pi++;
            star_s = si;
            continue;
        }
        if (pi < plen && p[pi] == '?') {
            si += utf8_unit_len(s, slen, si);
            ++pi;
            continue;
        }
        if (pi < plen) {
            const bool esc = static_cast<int32_t>(static_cast<uint8_t>(p[pi])) == escape;
            const uint32_t lit = esc ? pi + 1u : pi;
            if (lit < plen) {
                const uint32_t n = utf8_unit_len(p, plen, lit);
                if (si + n <= slen && memcmp(s + si, p + lit, n) == 0) {
                    si += n;
                    pi = lit + n;
                    continue;
                }
            }
        }
        if (star_p == 0xFFFFFFFFu) return false;
        pi = star_p + 1u;
        star_s += utf8_unit_len(s, slen, star_s);
        si = star_s;
    }
    while (pi < plen && p[pi] == '*') ++pi;
    assert(pi <= plen);
    return pi == plen;
}

}  // namespace utf8
}  // namespace kernels
}  // namespace bolt
