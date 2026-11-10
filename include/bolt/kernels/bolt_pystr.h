// Python str / float text semantics over UTF-8 and doubles (G2LAUNCH-152):
// str.strip / lstrip / rstrip (whitespace = str.isspace(), or an explicit
// character set) and repr(float) (shortest round-trip digits, Python's
// fixed / exponent switch). Tables come from the CPython Unicode database
// (bolt_utf8_pycase_tables.h).
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_float_chars.h"
#include "bolt/kernels/bolt_utf8_cp.h"
#include "bolt/kernels/bolt_utf8_pycase_tables.h"

namespace bolt::kernels::utf8 {

BOLT_FORCE_INLINE bool py_isspace(uint32_t cp) noexcept {
    for (uint32_t s : pycase::kSpace) {   // 29 entries
        if (s == cp) return true;
        if (s > cp) return false;
    }
    return false;
}

// Is `cp` one of the code points of `chars` (UTF-8, `clen` bytes)?
BOLT_FORCE_INLINE bool cp_in(uint32_t cp, const char* chars, uint32_t clen) noexcept {
    uint32_t i = 0;
    while (i < clen) {   // bounded by clen
        uint32_t c;
        const uint32_t u = utf8_cp_decode(chars, clen, i, &c);
        if (c == cp) return true;
        i += u;
    }
    return false;
}

enum class PyStrip : uint8_t { Both = 0, Left = 1, Right = 2 };

// The [*lo, *hi) byte window str.strip leaves. chars == nullptr: whitespace.
BOLT_FORCE_INLINE void py_strip(const char* p, uint32_t len, PyStrip mode, const char* chars,
                                uint32_t clen, uint32_t* lo, uint32_t* hi) noexcept {
    assert(lo != nullptr && hi != nullptr);
    uint32_t a = 0, b = len;
    const auto strip = [&](uint32_t cp) noexcept {
        return chars == nullptr ? py_isspace(cp) : cp_in(cp, chars, clen);
    };
    if (mode != PyStrip::Right) {
        while (a < b) {   // bounded by len
            uint32_t cp;
            const uint32_t u = utf8_cp_decode(p, len, a, &cp);
            if (cp == kUtf8Invalid || !strip(cp)) break;
            a += u;
        }
    }
    if (mode != PyStrip::Left) {
        while (b > a) {   // bounded by len
            uint32_t q = b - 1;
            uint32_t steps = 0;
            while (q > a && steps < 3 && (static_cast<uint8_t>(p[q]) & 0xC0u) == 0x80u) { --q; ++steps; }
            uint32_t cp;
            (void)utf8_cp_decode(p, len, q, &cp);
            if (cp == kUtf8Invalid || !strip(cp)) break;
            b = q;
        }
    }
    *lo = a;
    *hi = b;
}

// repr(float) into out (capacity >= 32); returns the length.
BOLT_FORCE_INLINE uint32_t f64_python_repr(double v, char* out) noexcept {
    assert(out != nullptr);
    if (std::isnan(v)) { std::memcpy(out, "nan", 3); return 3; }
    uint32_t n = 0;
    if (std::signbit(v)) out[n++] = '-';
    if (std::isinf(v)) { std::memcpy(out + n, "inf", 3); return n + 3; }
    if (v == 0.0) { std::memcpy(out + n, "0.0", 3); return n + 3; }
    char d[20];
    int32_t k = 0;
    const uint32_t nd = float_chars::detail::shortest_digits(std::fabs(v), d, &k);
    assert(nd >= 1 && nd <= 17);
    if (k <= -4 || k > 16) {   // Python's switch to exponent notation
        out[n++] = d[0];
        if (nd > 1) { out[n++] = '.'; std::memcpy(out + n, d + 1, nd - 1); n += nd - 1; }
        int32_t e = k - 1;
        out[n++] = 'e';
        out[n++] = e < 0 ? '-' : '+';
        if (e < 0) e = -e;
        char ed[4];
        uint32_t m = 0;
        do { ed[m++] = static_cast<char>('0' + e % 10); e /= 10; } while (e != 0);   // <= 3 digits
        if (m < 2) ed[m++] = '0';
        while (m > 0) out[n++] = ed[--m];
        return n;
    }
    if (k <= 0) {
        out[n++] = '0';
        out[n++] = '.';
        for (int32_t i = 0; i < -k; ++i) out[n++] = '0';
        std::memcpy(out + n, d, nd);
        return n + nd;
    }
    const uint32_t ku = static_cast<uint32_t>(k);
    if (ku >= nd) {
        std::memcpy(out + n, d, nd);
        n += nd;
        for (uint32_t i = nd; i < ku; ++i) out[n++] = '0';
        out[n++] = '.';
        out[n++] = '0';
        return n;
    }
    std::memcpy(out + n, d, ku);
    n += ku;
    out[n++] = '.';
    std::memcpy(out + n, d + ku, nd - ku);
    return n + nd - ku;
}

}  // namespace bolt::kernels::utf8
