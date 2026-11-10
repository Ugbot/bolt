// bolt_text_spark.h — scalar text kernels for Spark's string function family
// (ascii/chr, trim with a character set, overlay, mask, translate, repeat,
// substring_index, split_part, find_in_set, locate/instr, levenshtein, soundex,
// UTF-8 validation and repair).
//
// Semantics follow Spark 4 (UTF8String / the Catalyst string expressions),
// checked against a running Spark 4.0.1: lengths and positions count code
// points, a "unit" is one lead byte plus its continuation bytes (the rule
// bolt_utf8_cp.h defines), byte-level functions (substring_index, split_part,
// find_in_set) search UTF-8 bytes. Null handling, ANSI errors and result types
// are the caller's; every kernel here is a pure function of its bytes.
//
// RULES: No exceptions, no heap, all fns noexcept, bounded loops. Outputs go
// to a caller buffer sized by the matching *_bound function; a result that
// does not fit returns kTextOverflow. Kernels that produce a substring of
// their input return the byte range instead of copying.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/kernels/bolt_text_fn.h"
#include "bolt/kernels/bolt_unicode_class.h"
#include "bolt/kernels/bolt_utf8_cp.h"

namespace bolt {
namespace kernels {
namespace text {

constexpr int64_t kTextInvalid = -4;       // input the function rejects (ANSI error / NULL)
constexpr int64_t kTextTooLong = -6;       // an operand past a kernel's fixed working size
constexpr uint32_t kEditMaxChars = 8192;   // levenshtein: code points of the shorter string
constexpr uint32_t kReplacementCp = 0xFFFDu;

// ---- code point helpers ---------------------------------------------------

// The code point of the unit at s[pos]; U+FFFD when the unit is not well formed.
inline uint32_t unit_cp(const char* s, uint32_t len, uint32_t pos, uint32_t* ulen) noexcept {
    assert(s != nullptr && pos < len && ulen != nullptr);
    uint32_t cp = 0;
    *ulen = utf8::utf8_cp_decode(s, len, pos, &cp);
    assert(*ulen >= 1u && pos + *ulen <= len);
    return cp == utf8::kUtf8Invalid ? kReplacementCp : cp;
}

// Byte offset of the start of the unit that ends at `end` (end > 0).
inline uint32_t unit_before(const char* s, uint32_t end) noexcept {
    assert(s != nullptr && end > 0);
    uint32_t p = end - 1u;
    for (uint32_t k = 0; k < 3u && p > 0 && utf8::utf8_is_cont(static_cast<uint8_t>(s[p])); ++k) --p;
    return p;
}

// ascii(s): the first code point; 0 for "". A malformed first unit is U+FFFD (Spark
// decodes the first character through String).
inline int32_t ascii_first(const char* s, uint32_t len) noexcept {
    if (len == 0) return 0;
    uint32_t u = 0;
    return static_cast<int32_t>(unit_cp(s, len, 0, &u));
}

// chr(n) / char(n): n < 0 is "", otherwise the code point (n mod 256). Writes 1-2 bytes.
inline uint32_t chr_encode(int64_t n, char* out) noexcept {
    assert(out != nullptr);
    if (n < 0) return 0;
    return utf8::utf8_cp_encode(static_cast<uint32_t>(n & 0xFF), out);
}

// ---- position / substring -------------------------------------------------

// Byte range [*b, *e) of Spark's substringSQL(pos, length): 1-based, 0 acts as 1, a
// negative pos counts from the end, int32 arithmetic as in the JVM.
inline void substr_sql(const char* s, uint32_t slen, int32_t pos, int32_t length,
                       uint32_t* b, uint32_t* e) noexcept {
    assert(s != nullptr || slen == 0);
    assert(b != nullptr && e != nullptr);
    const int64_t n = utf8_chars(s, slen);
    int64_t start = pos > 0 ? static_cast<int64_t>(pos) - 1 : (pos < 0 ? n + pos : 0);
    int64_t end = start + length;
    if (end > INT32_MAX) end = INT32_MAX;
    if (end < INT32_MIN) end = INT32_MIN;
    if (start < 0) start = 0;
    if (end <= start || start >= n) { *b = *e = 0; return; }
    if (end > n) end = n;
    *b = utf8_prefix_bytes(s, slen, start);
    *e = *b + utf8_prefix_bytes(s + *b, slen - *b, end - start);
    assert(*b <= *e && *e <= slen);
}

// left(s, n): the first n code points; n <= 0 is "".
inline uint32_t left_bytes(const char* s, uint32_t len, int64_t n) noexcept {
    assert(s != nullptr || len == 0);
    return n <= 0 ? 0u : utf8_prefix_bytes(s, len, n);
}

// right(s, n): byte offset where the last n code points start.
inline uint32_t right_start(const char* s, uint32_t len, int64_t n) noexcept {
    assert(s != nullptr || len == 0);
    if (n <= 0) return len;
    const int64_t have = utf8_chars(s, len);
    if (n >= have) return 0;
    return utf8_prefix_bytes(s, len, have - n);
}

// locate / instr / position: 1-based code point index of the first match at or after
// code point `pos` (1-based), 0 when absent. pos < 1 is 0; an empty needle is 1.
inline int32_t locate(const char* sub, uint32_t sublen, const char* s, uint32_t slen,
                      int64_t pos) noexcept {
    assert(sub != nullptr || sublen == 0);
    assert(s != nullptr || slen == 0);
    if (pos < 1) return 0;
    if (sublen == 0) return 1;
    uint32_t i = 0;
    int64_t c = 0;
    while (i < slen && c < pos - 1) { i += utf8::utf8_unit_len(s, slen, i); ++c; }
    for (uint32_t guard = 0; guard <= slen; ++guard) {          // bounded: slen units
        if (i + sublen > slen) return 0;
        if (std::memcmp(s + i, sub, sublen) == 0) return static_cast<int32_t>(c + 1);
        if (i >= slen) return 0;
        i += utf8::utf8_unit_len(s, slen, i);
        ++c;
    }
    return 0;
}

// ---- trim with a character set -------------------------------------------

namespace detail {
struct CharSet {
    uint64_t ascii[2];
    const char* p;
    uint32_t len;
};
inline CharSet charset_make(const char* set, uint32_t len) noexcept {
    CharSet c{{0, 0}, set, len};
    for (uint32_t i = 0; i < len; ++i) {
        const uint8_t b = static_cast<uint8_t>(set[i]);
        if (b < 0x80u) c.ascii[b >> 6] |= 1ull << (b & 63u);
    }
    return c;
}
inline bool charset_has(const CharSet& c, const char* s, uint32_t ulen) noexcept {
    const uint8_t b = static_cast<uint8_t>(s[0]);
    if (ulen == 1u && b < 0x80u) return ((c.ascii[b >> 6] >> (b & 63u)) & 1ull) != 0;
    for (uint32_t i = 0; i < c.len;) {                           // bounded: set bytes
        const uint32_t u = utf8::utf8_unit_len(c.p, c.len, i);
        if (u == ulen && std::memcmp(c.p + i, s, ulen) == 0) return true;
        i += u;
    }
    return false;
}
}  // namespace detail

// btrim / ltrim / rtrim with a character set: `mode` 0 both ends, 1 leading, 2 trailing.
inline void trim_set(const char* s, uint32_t slen, const char* set, uint32_t setlen, int mode,
                     uint32_t* b, uint32_t* e) noexcept {
    assert(s != nullptr || slen == 0);
    assert(set != nullptr || setlen == 0);
    assert(b != nullptr && e != nullptr && mode >= 0 && mode <= 2);
    const detail::CharSet cs = detail::charset_make(set, setlen);
    uint32_t lo = 0, hi = slen;
    if (mode != 2) {
        while (lo < hi) {
            const uint32_t u = utf8::utf8_unit_len(s, slen, lo);
            if (!detail::charset_has(cs, s + lo, u)) break;
            lo += u;
        }
    }
    if (mode != 1) {
        while (hi > lo) {
            const uint32_t p = unit_before(s, hi);
            if (p < lo || !detail::charset_has(cs, s + p, hi - p)) break;
            hi = p;
        }
    }
    *b = lo;
    *e = hi;
}

// ---- overlay --------------------------------------------------------------

// Both pieces of s can be the whole of s (int32 wrap-around of pos + len, as on the JVM).
constexpr uint64_t overlay_bound(uint32_t slen, uint32_t rlen) noexcept {
    return 2ull * slen + rlen;
}

// overlay(s PLACING r FROM pos [FOR len]); len < 0 means chars(r) (Spark's default).
inline int64_t overlay(const char* s, uint32_t slen, const char* r, uint32_t rlen, int32_t pos,
                       int32_t len, char* out, uint32_t cap) noexcept {
    assert(s != nullptr || slen == 0);
    assert(r != nullptr || rlen == 0);
    assert(out != nullptr);
    const int32_t length = len >= 0 ? len : static_cast<int32_t>(utf8_chars(r, rlen));
    const int32_t from = static_cast<int32_t>(static_cast<uint32_t>(pos) + static_cast<uint32_t>(length));
    uint32_t b1 = 0, e1 = 0, b2 = 0, e2 = 0;
    substr_sql(s, slen, 1, static_cast<int32_t>(static_cast<uint32_t>(pos) - 1u), &b1, &e1);
    substr_sql(s, slen, from, INT32_MAX, &b2, &e2);
    const uint64_t total = static_cast<uint64_t>(e1 - b1) + rlen + (e2 - b2);
    if (total > cap) return kTextOverflow;
    uint32_t w = 0;
    std::memcpy(out + w, s + b1, e1 - b1); w += e1 - b1;
    std::memcpy(out + w, r, rlen); w += rlen;
    std::memcpy(out + w, s + b2, e2 - b2); w += e2 - b2;
    return w;
}

// ---- mask -----------------------------------------------------------------

// A 1-byte character can become 4 bytes; a 4-byte one becomes two replacements of 4.
constexpr uint64_t mask_bound(uint32_t slen) noexcept { return static_cast<uint64_t>(slen) * 8u; }

// mask(s, upper, lower, digit, other): each replacement is a code point, or kMaskKeep to
// leave that class unchanged. Classes are Java's Character.isUpperCase / isLowerCase / isDigit,
// applied per UTF-16 unit as Spark does: a supplementary character is two surrogate units,
// neither of which is a letter or a digit, so it is "other" twice over; a supplementary
// replacement character is its high surrogate alone, which encodes as '?'.
constexpr uint32_t kMaskKeep = 0xFFFFFFFFu;

inline int64_t mask(const char* s, uint32_t slen, uint32_t up, uint32_t lo, uint32_t dg,
                    uint32_t ot, char* out, uint32_t cap) noexcept {
    assert(s != nullptr || slen == 0);
    assert(out != nullptr);
    auto put = [&](uint32_t cp, uint32_t* w) noexcept -> bool {
        if (cp > 0xFFFFu) cp = '?';
        if (*w + 4u > cap) return false;
        *w += utf8::utf8_cp_encode(cp, out + *w);
        return true;
    };
    uint32_t w = 0;
    for (uint32_t i = 0; i < slen;) {                            // bounded: slen
        uint32_t cp = 0;
        const uint32_t u = utf8::utf8_cp_decode(s, slen, i, &cp);
        uint32_t to = ot;
        if (cp == utf8::kUtf8Invalid) to = ot;
        else if (cp > 0xFFFFu) to = ot;
        else if (unicode::is_upper(cp)) to = up;
        else if (unicode::is_lower(cp)) to = lo;
        else if (unicode::is_digit(cp)) to = dg;
        if (to == kMaskKeep) {
            if (w + u > cap) return kTextOverflow;
            std::memcpy(out + w, s + i, u);
            w += u;
        } else {
            if (!put(to, &w)) return kTextOverflow;
            if (cp != utf8::kUtf8Invalid && cp > 0xFFFFu && !put(to, &w)) return kTextOverflow;
        }
        i += u;
    }
    return w;
}

// ---- translate ------------------------------------------------------------

// translate(s, from, to): a code point found in `from` at index i becomes the i-th code
// point of `to`, or is dropped when `to` is shorter; the first occurrence in `from` wins.
// Output never exceeds slen * 4.
inline int64_t translate(const char* s, uint32_t slen, const char* from, uint32_t flen,
                         const char* to, uint32_t tlen, char* out, uint32_t cap) noexcept {
    assert(s != nullptr || slen == 0);
    assert(out != nullptr);
    uint32_t w = 0;
    for (uint32_t i = 0; i < slen;) {                            // bounded: slen
        const uint32_t u = utf8::utf8_unit_len(s, slen, i);
        uint32_t fi = 0, idx = 0;
        bool found = false;
        for (uint32_t f = 0; f < flen;) {                        // bounded: flen
            const uint32_t fu = utf8::utf8_unit_len(from, flen, f);
            if (fu == u && std::memcmp(from + f, s + i, u) == 0) { found = true; fi = idx; break; }
            f += fu;
            ++idx;
        }
        if (!found) {
            if (w + u > cap) return kTextOverflow;
            std::memcpy(out + w, s + i, u);
            w += u;
        } else {
            uint32_t t = 0;
            for (uint32_t k = 0; k < fi && t < tlen; ++k) t += utf8::utf8_unit_len(to, tlen, t);
            if (t < tlen) {
                const uint32_t tu = utf8::utf8_unit_len(to, tlen, t);
                if (w + tu > cap) return kTextOverflow;
                std::memcpy(out + w, to + t, tu);
                w += tu;
            }
        }
        i += u;
    }
    return w;
}

// ---- repeat ---------------------------------------------------------------

// repeat(s, n): n <= 0 is "". Returns bytes, or kTextOverflow when the result passes `cap`.
inline int64_t repeat(const char* s, uint32_t slen, int64_t n, char* out, uint64_t cap) noexcept {
    assert(s != nullptr || slen == 0);
    assert(out != nullptr || n <= 0 || slen == 0);
    if (n <= 0 || slen == 0) return 0;
    const uint64_t total = static_cast<uint64_t>(slen) * static_cast<uint64_t>(n);
    if (static_cast<uint64_t>(n) > cap || total > cap) return kTextOverflow;
    for (int64_t k = 0; k < n; ++k) std::memcpy(out + static_cast<uint64_t>(k) * slen, s, slen);
    return static_cast<int64_t>(total);
}

// ---- substring_index / split_part / find_in_set --------------------------

namespace detail {
// First occurrence of d in s at or after `from`, or -1.
inline int64_t find_from(const char* s, uint32_t slen, const char* d, uint32_t dlen,
                         uint32_t from) noexcept {
    if (dlen == 0 || slen < dlen) return -1;
    const uint32_t last = slen - dlen;
    for (uint32_t i = from; i <= last; ++i) {                    // bounded: slen
        if (s[i] == d[0] && std::memcmp(s + i, d, dlen) == 0) return i;
    }
    return -1;
}
// Last occurrence of d in s starting at or before `from` (may be negative), or -1.
inline int64_t rfind_from(const char* s, uint32_t slen, const char* d, uint32_t dlen,
                          int64_t from) noexcept {
    if (dlen == 0 || slen < dlen) return -1;
    int64_t i = from;
    if (i > static_cast<int64_t>(slen - dlen)) i = static_cast<int64_t>(slen - dlen);
    for (; i >= 0; --i) {                                        // bounded: slen
        if (s[i] == d[0] && std::memcmp(s + i, d, dlen) == 0) return i;
    }
    return -1;
}
}  // namespace detail

// substring_index(s, delim, count): everything before the count-th delimiter from the
// left (count > 0) or after the |count|-th from the right (count < 0); the whole string
// when there are fewer; "" for count 0 or an empty delimiter. Byte search, Spark's order.
inline void substring_index(const char* s, uint32_t slen, const char* d, uint32_t dlen,
                            int32_t count, uint32_t* b, uint32_t* e) noexcept {
    assert(b != nullptr && e != nullptr);
    *b = 0;
    *e = slen;
    if (dlen == 0 || count == 0) { *e = 0; return; }
    if (count > 0) {
        int64_t idx = -1;
        for (int64_t c = count; c > 0; --c) {                    // bounded: occurrences <= slen
            idx = detail::find_from(s, slen, d, dlen, static_cast<uint32_t>(idx + 1));
            if (idx < 0) return;
        }
        *e = static_cast<uint32_t>(idx);
        return;
    }
    int64_t idx = static_cast<int64_t>(slen) - dlen + 1;
    for (int64_t c = -static_cast<int64_t>(count); c > 0; --c) {
        idx = detail::rfind_from(s, slen, d, dlen, idx - 1);
        if (idx < 0) return;
    }
    *b = static_cast<uint32_t>(idx) + dlen;
    *e = slen;
}

// split_part(s, delim, n): the n-th piece of a literal split (n counts from the end when
// negative), "" when out of range. n == 0 is the caller's error. An empty delimiter does
// not split. Returns false and *b = *e = 0 for "" .
inline void split_part(const char* s, uint32_t slen, const char* d, uint32_t dlen, int64_t n,
                       uint32_t* b, uint32_t* e) noexcept {
    assert(b != nullptr && e != nullptr && n != 0);
    *b = *e = 0;
    uint64_t parts = 1;
    if (dlen != 0) {
        for (uint32_t i = 0; i < slen;) {                        // bounded: slen
            const int64_t at = detail::find_from(s, slen, d, dlen, i);
            if (at < 0) break;
            ++parts;
            i = static_cast<uint32_t>(at) + dlen;
        }
    }
    int64_t want = n > 0 ? n - 1 : static_cast<int64_t>(parts) + n;
    if (want < 0 || want >= static_cast<int64_t>(parts)) return;
    uint32_t start = 0;
    for (int64_t k = 0; k < want; ++k) {                         // bounded: parts
        start = static_cast<uint32_t>(detail::find_from(s, slen, d, dlen, start)) + dlen;
    }
    const int64_t at = dlen == 0 ? -1 : detail::find_from(s, slen, d, dlen, start);
    *b = start;
    *e = at < 0 ? slen : static_cast<uint32_t>(at);
}

// find_in_set(str, list): 1-based index of an exact element of the comma list, 0 when
// absent or when `str` itself contains a comma.
inline int32_t find_in_set(const char* m, uint32_t mlen, const char* list, uint32_t llen) noexcept {
    assert(m != nullptr || mlen == 0);
    assert(list != nullptr || llen == 0);
    for (uint32_t i = 0; i < mlen; ++i) if (m[i] == ',') return 0;
    int32_t n = 1;
    int64_t last = -1;
    for (uint32_t i = 0; i < llen; ++i) {                        // bounded: llen
        if (list[i] != ',') continue;
        if (i - static_cast<uint32_t>(last + 1) == mlen &&
            std::memcmp(list + last + 1, m, mlen) == 0) return n;
        ++n;
        last = i;
    }
    if (llen - static_cast<uint32_t>(last + 1) == mlen &&
        std::memcmp(list + last + 1, m, mlen) == 0) return n;
    return 0;
}

// ---- replace ----------------------------------------------------------------

// replace(s, search, repl): every non-overlapping occurrence, left to right; an empty
// search leaves s unchanged. replace_size is the exact output length.
inline uint64_t replace_size(const char* s, uint32_t slen, const char* f, uint32_t flen,
                             uint32_t rlen) noexcept {
    assert(s != nullptr || slen == 0);
    assert(f != nullptr || flen == 0);
    if (flen == 0) return slen;
    uint64_t total = slen;
    for (uint32_t i = 0; i < slen;) {                            // bounded: slen
        const int64_t at = detail::find_from(s, slen, f, flen, i);
        if (at < 0) break;
        total = total - flen + rlen;
        i = static_cast<uint32_t>(at) + flen;
    }
    return total;
}

inline int64_t replace_all(const char* s, uint32_t slen, const char* f, uint32_t flen, const char* r,
                           uint32_t rlen, char* out, uint32_t cap) noexcept {
    assert(s != nullptr || slen == 0);
    assert(out != nullptr);
    uint64_t w = 0;
    uint32_t i = 0;
    while (flen != 0 && i < slen) {                              // bounded: slen
        const int64_t at = detail::find_from(s, slen, f, flen, i);
        if (at < 0) break;
        const uint32_t a = static_cast<uint32_t>(at);
        if (w + (a - i) + rlen > cap) return kTextOverflow;
        std::memcpy(out + w, s + i, a - i); w += a - i;
        std::memcpy(out + w, r, rlen); w += rlen;
        i = a + flen;
    }
    if (w + (slen - i) > cap) return kTextOverflow;
    std::memcpy(out + w, s + i, slen - i);
    return static_cast<int64_t>(w + (slen - i));
}

// ---- levenshtein ----------------------------------------------------------

// Scratch words levenshtein needs for a shorter string of `chars` code points.
constexpr uint32_t levenshtein_scratch(uint32_t chars) noexcept { return 3u * (chars + 1u); }

namespace detail {
inline uint32_t decode_all(const char* s, uint32_t len, uint32_t* cps) noexcept {
    uint32_t n = 0;
    for (uint32_t i = 0; i < len; ++n) {
        uint32_t u = 0;
        cps[n] = unit_cp(s, len, i, &u);
        i += u;
    }
    return n;
}
}  // namespace detail

// Edit distance over code points. `work` holds levenshtein_scratch(min chars) words.
// Returns the distance, or kTextTooLong when the shorter string passes kEditMaxChars.
// threshold >= 0 turns a distance past it into -1 (Spark's levenshtein(a, b, threshold)).
inline int64_t levenshtein(const char* a, uint32_t alen, const char* b, uint32_t blen,
                           int64_t threshold, bool has_threshold, uint32_t* work) noexcept {
    assert(a != nullptr || alen == 0);
    assert(b != nullptr || blen == 0);
    assert(work != nullptr);
    uint32_t na = utf8_chars(a, alen), nb = utf8_chars(b, blen);
    if (na < nb) { const char* t = a; a = b; b = t; const uint32_t tl = alen; alen = blen; blen = tl; na ^= nb; nb ^= na; na ^= nb; }
    // b is now the shorter string (nb code points); a the longer (na).
    if (nb > kEditMaxChars) return kTextTooLong;
    int64_t result = 0;
    if (nb == 0) {
        result = na;
    } else {
        uint32_t* cps = work;
        uint32_t* prev = work + (nb + 1u);
        uint32_t* cur = prev + (nb + 1u);
        detail::decode_all(b, blen, cps);
        for (uint32_t j = 0; j <= nb; ++j) prev[j] = j;
        uint32_t i = 0;
        for (uint32_t pos = 0; pos < alen; ++i) {                // bounded: na
            uint32_t u = 0;
            const uint32_t ca = unit_cp(a, alen, pos, &u);
            pos += u;
            cur[0] = i + 1u;
            for (uint32_t j = 1; j <= nb; ++j) {
                const uint32_t sub = prev[j - 1] + (cps[j - 1] == ca ? 0u : 1u);
                const uint32_t del = prev[j] + 1u, ins = cur[j - 1] + 1u;
                uint32_t m = sub < del ? sub : del;
                cur[j] = m < ins ? m : ins;
            }
            uint32_t* t = prev; prev = cur; cur = t;
        }
        result = prev[nb];
    }
    if (has_threshold && result > threshold) return -1;
    return result;
}

// ---- soundex --------------------------------------------------------------

// Spark's soundex: the first byte must be an ASCII letter, else the input is returned
// unchanged (returns -1, the caller passes the input through). Otherwise out[0..4).
inline int32_t soundex(const char* s, uint32_t len, char* out) noexcept {
    assert(s != nullptr || len == 0);
    assert(out != nullptr);
    static const char kMap[26] = {'0','1','2','3','0','1','2','7','0','2','2','4','5',
                                  '5','0','1','2','6','2','3','0','1','7','2','0','2'};
    if (len == 0) return -1;
    uint8_t b = static_cast<uint8_t>(s[0]);
    if (b >= 'a' && b <= 'z') b = static_cast<uint8_t>(b - 32);
    else if (b < 'A' || b > 'Z') return -1;
    out[0] = static_cast<char>(b);
    out[1] = out[2] = out[3] = '0';
    uint32_t sxi = 1;
    char last = kMap[b - 'A'];
    for (uint32_t i = 1; i < len; ++i) {                         // bounded: len
        b = static_cast<uint8_t>(s[i]);
        if (b >= 'a' && b <= 'z') b = static_cast<uint8_t>(b - 32);
        else if (b < 'A' || b > 'Z') { last = '0'; continue; }
        const char code = kMap[b - 'A'];
        if (code == '7') continue;
        if (code != '0' && code != last) {
            out[sxi++] = code;
            if (sxi > 3) break;
        }
        last = code;
    }
    return 4;
}

// ---- UTF-8 validation -----------------------------------------------------

namespace detail {
// Length of the maximal well-formed prefix of the sequence at s[i]: >0 = a valid scalar
// of that many bytes; otherwise the negative length of the maximal invalid subpart
// (Unicode's "maximal subpart" practice, what Java's decoder replaces with one U+FFFD).
inline int32_t utf8_seq(const char* s, uint32_t len, uint32_t i) noexcept {
    const uint8_t b = static_cast<uint8_t>(s[i]);
    if (b < 0x80u) return 1;
    uint32_t need;
    uint8_t lo = 0x80, hi = 0xBF;
    if (b >= 0xC2u && b <= 0xDFu) need = 1;
    else if (b >= 0xE0u && b <= 0xEFu) { need = 2; if (b == 0xE0u) lo = 0xA0; if (b == 0xEDu) hi = 0x9F; }
    else if (b >= 0xF0u && b <= 0xF4u) { need = 3; if (b == 0xF0u) lo = 0x90; if (b == 0xF4u) hi = 0x8F; }
    else return -1;
    uint32_t k = 1;
    for (; k <= need; ++k) {
        if (i + k >= len) return -static_cast<int32_t>(k);
        const uint8_t c = static_cast<uint8_t>(s[i + k]);
        if (c < lo || c > hi) return -static_cast<int32_t>(k);
        lo = 0x80; hi = 0xBF;
    }
    return static_cast<int32_t>(need + 1u);
}
}  // namespace detail

inline bool utf8_is_valid(const char* s, uint32_t len) noexcept {
    assert(s != nullptr || len == 0);
    for (uint32_t i = 0; i < len;) {                             // bounded: len
        const int32_t r = detail::utf8_seq(s, len, i);
        if (r < 0) return false;
        i += static_cast<uint32_t>(r);
    }
    return true;
}

constexpr uint64_t make_valid_bound(uint32_t len) noexcept { return static_cast<uint64_t>(len) * 3u; }

// make_valid_utf8: every maximal invalid subpart becomes one U+FFFD.
inline int64_t utf8_make_valid(const char* s, uint32_t len, char* out, uint32_t cap) noexcept {
    assert(s != nullptr || len == 0);
    assert(out != nullptr);
    uint32_t w = 0;
    for (uint32_t i = 0; i < len;) {                             // bounded: len
        const int32_t r = detail::utf8_seq(s, len, i);
        if (r > 0) {
            if (w + static_cast<uint32_t>(r) > cap) return kTextOverflow;
            std::memcpy(out + w, s + i, static_cast<uint32_t>(r));
            w += static_cast<uint32_t>(r);
            i += static_cast<uint32_t>(r);
        } else {
            if (w + 3u > cap) return kTextOverflow;
            out[w++] = static_cast<char>(0xEF); out[w++] = static_cast<char>(0xBF); out[w++] = static_cast<char>(0xBD);
            i += static_cast<uint32_t>(-r);
        }
    }
    return w;
}

}  // namespace text
}  // namespace kernels
}  // namespace bolt
