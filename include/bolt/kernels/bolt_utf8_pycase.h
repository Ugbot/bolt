// Python-exact case mapping (str.lower / str.upper) over UTF-8 (G2LAUNCH-152).
//
// Full Unicode case mapping as CPython applies it: SpecialCasing expansions
// ("ß".upper() == "SS", "İ".lower() == "i̇", "ﬁ".upper() == "FI") and the
// Final_Sigma context for U+03A3 in lower(). Tables are generated from the
// CPython Unicode database (tools/gen_pycase_tables.py); kUnicodeVersion
// names it, and a caller compiling Python code must check that the
// interpreter whose semantics it reproduces has the same version. Invalid
// UTF-8 units are copied unchanged, as utf8_case_map does.
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_utf8_cp.h"
#include "bolt/kernels/bolt_utf8_pycase_tables.h"

namespace bolt::kernels::utf8 {

// A mapped code point is at most 3 code points of at most 4 bytes; any
// non-ASCII input unit is at least 2 bytes, so 6x bounds the output.
BOLT_FORCE_INLINE uint32_t utf8_pycase_bound(uint32_t len) noexcept { return len * 6u; }

namespace pycase {

template <class T, uint32_t N>
BOLT_FORCE_INLINE const T* find_map(const T (&tab)[N], uint32_t cp) noexcept {
    uint32_t lo = 0, hi = N;
    while (lo < hi) {   // bounded: log2(N) steps
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (tab[mid].cp < cp) lo = mid + 1u;
        else hi = mid;
    }
    return (lo < N && tab[lo].cp == cp) ? &tab[lo] : nullptr;
}

template <uint32_t N>
BOLT_FORCE_INLINE bool in_ranges(const PyCaseRange (&tab)[N], uint32_t cp) noexcept {
    uint32_t lo = 0, hi = N;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (tab[mid].hi < cp) lo = mid + 1u;
        else hi = mid;
    }
    return lo < N && tab[lo].lo <= cp && cp <= tab[lo].hi;
}

// Start of the unit before `pos` (pos > 0).
BOLT_FORCE_INLINE uint32_t prev_unit(const char* p, uint32_t pos) noexcept {
    assert(pos > 0);
    uint32_t q = pos - 1u;
    uint32_t steps = 0;
    while (q > 0 && steps < 3u && (static_cast<uint8_t>(p[q]) & 0xC0u) == 0x80u) { --q; ++steps; }
    return q;
}

// CPython's handle_capital_sigma: final when a cased letter precedes it and
// none follows, case-ignorable code points skipped on both sides.
BOLT_FORCE_INLINE bool final_sigma(const char* p, uint32_t len, uint32_t at, uint32_t after) noexcept {
    uint32_t cp = kUtf8Invalid;
    uint32_t pos = at;
    bool found = false;
    while (pos > 0) {   // bounded by `at` units
        pos = prev_unit(p, pos);
        (void)utf8_cp_decode(p, len, pos, &cp);
        if (cp == kUtf8Invalid || !in_ranges(kSigmaIgnorable, cp)) { found = true; break; }
    }
    if (!found || cp == kUtf8Invalid || !in_ranges(kSigmaCased, cp)) return false;
    pos = after;
    while (pos < len) {   // bounded by len - after units
        const uint32_t u = utf8_cp_decode(p, len, pos, &cp);
        if (cp == kUtf8Invalid || !in_ranges(kSigmaIgnorable, cp)) {
            return cp == kUtf8Invalid || !in_ranges(kSigmaCased, cp);
        }
        pos += u;
    }
    return true;
}

}  // namespace pycase

// Map src[0..len) into dst (capacity >= utf8_pycase_bound(len)); returns the
// bytes written. dst == nullptr only measures.
BOLT_FORCE_INLINE uint32_t utf8_case_map_python(const char* BOLT_RESTRICT src, uint32_t len,
                                                bool upper, char* BOLT_RESTRICT dst) noexcept {
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
            if (dst != nullptr) std::memcpy(dst + w, src + i, u);
            w += u;
            i += u;
            continue;
        }
        uint32_t out[3] = {cp, 0, 0};
        uint32_t n = 1;
        if (!upper && cp == 0x03A3u) {
            out[0] = pycase::final_sigma(src, len, i, i + u) ? 0x03C2u : 0x03C3u;
        } else {
            const pycase::PyCaseMap* m = upper ? pycase::find_map(pycase::kUpper, cp)
                                               : pycase::find_map(pycase::kLower, cp);
            if (m != nullptr) {
                n = m->n;
                out[0] = m->m[0]; out[1] = m->m[1]; out[2] = m->m[2];
            }
        }
        for (uint32_t k = 0; k < n; ++k) {
            char enc[4];
            const uint32_t e = utf8_cp_encode(out[k], enc);
            if (dst != nullptr) std::memcpy(dst + w, enc, e);
            w += e;
        }
        i += u;
    }
    assert(w <= utf8_pycase_bound(len) || len == 0);
    return w;
}

}  // namespace bolt::kernels::utf8
