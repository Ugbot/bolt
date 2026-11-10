// bolt_encode_spark.h -- binary/text encodings for the Spark functions hex / unhex / bin / base64 /
// unbase64 / conv / encode / decode / to_binary, header-only, caller-owned output, no heap.
// Bit-exact with Spark 4 (and the JDK codecs it calls):
//   * base64 : java.util.Base64 basic and MIME encoders (MIME = 76-char lines joined by CRLF); the
//              decoders replay Base64.Decoder.decode0 (lenient MIME skips illegal characters, the
//              basic decoder rejects them), reporting the JDK's failure instead of guessing.
//   * conv   : Spark's NumberConverter (unsigned 64-bit accumulate, signed output for a negative
//              to-base, overflow saturates to 2^64-1 or reports it, scanning stops at the first
//              digit that is not valid in the base).
//   * charset: US-ASCII, ISO-8859-1, UTF-8, UTF-16 (BOM FEFF on encode, BOM detected on decode),
//              UTF-16BE/LE, UTF-32 (BE; BOM detected on decode), all strict (REPORT): an
//              unmappable or malformed unit is a failure, never a replacement character.
// RULES: No exceptions. No RTTI. No heap. All fns noexcept. Bounded loops.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"

namespace bolt {
namespace kernels {
namespace encx {

// ---- hex ------------------------------------------------------------------------------------
// Uppercase hex of p[0, n) into dst (2n bytes).
inline void hex_upper(const std::uint8_t* p, std::uint64_t n, char* dst) noexcept {
    assert(dst != nullptr && (p != nullptr || n == 0));
    static constexpr char d[] = "0123456789ABCDEF";
    for (std::uint64_t i = 0; i < n; ++i) {
        dst[2 * i] = d[p[i] >> 4];
        dst[2 * i + 1] = d[p[i] & 15u];
    }
}

// Spark hex(long): uppercase digits of the two's-complement value, no leading zeros ("0" for 0).
// dst holds 16 bytes; returns the digit count.
inline std::uint32_t hex_long(std::int64_t v, char* dst) noexcept {
    assert(dst != nullptr);
    static constexpr char d[] = "0123456789ABCDEF";
    std::uint64_t u = static_cast<std::uint64_t>(v);
    char tmp[16];
    std::uint32_t n = 0;
    do { tmp[n++] = d[u & 15u]; u >>= 4; } while (u != 0);
    for (std::uint32_t i = 0; i < n; ++i) dst[i] = tmp[n - 1 - i];
    return n;
}

// Spark bin(long): binary digits of the two's-complement value, no leading zeros. dst holds 64.
inline std::uint32_t bin_long(std::int64_t v, char* dst) noexcept {
    assert(dst != nullptr);
    std::uint64_t u = static_cast<std::uint64_t>(v);
    char tmp[64];
    std::uint32_t n = 0;
    do { tmp[n++] = static_cast<char>('0' + (u & 1u)); u >>= 1; } while (u != 0);
    for (std::uint32_t i = 0; i < n; ++i) dst[i] = tmp[n - 1 - i];
    return n;
}

inline int hex_digit(std::uint8_t c) noexcept {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Spark unhex(s): (n+1)/2 bytes into dst; an odd count pads a leading '0'. False on any non-hex byte.
inline bool unhex(const std::uint8_t* s, std::uint64_t n, std::uint8_t* dst, std::uint64_t* out_len) noexcept {
    assert(dst != nullptr && out_len != nullptr && (s != nullptr || n == 0));
    std::uint64_t i = 0, o = 0;
    if ((n & 1u) != 0) {
        const int v = hex_digit(s[0]);
        if (v < 0) return false;
        dst[o++] = static_cast<std::uint8_t>(v);
        i = 1;
    }
    for (; i < n; i += 2) {
        const int a = hex_digit(s[i]);
        const int b = hex_digit(s[i + 1]);
        if (a < 0 || b < 0) return false;
        dst[o++] = static_cast<std::uint8_t>((a << 4) | b);
    }
    *out_len = o;
    return true;
}

// ---- base64 ---------------------------------------------------------------------------------
inline std::uint64_t base64_encoded_cap(std::uint64_t n) noexcept { return ((n + 2u) / 3u) * 4u + (((n + 2u) / 3u) * 4u) / 76u * 2u + 4u; }

// Base64 of p[0, n) with '=' padding; `mime` breaks lines at 76 characters with CRLF (no trailing
// separator), as java.util.Base64.getMimeEncoder. Returns the length written.
inline std::uint64_t base64_encode(const std::uint8_t* p, std::uint64_t n, bool mime, char* dst) noexcept {
    assert(dst != nullptr && (p != nullptr || n == 0));
    static constexpr char t[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::uint64_t o = 0, line = 0;
    for (std::uint64_t i = 0; i < n; i += 3) {
        if (mime && line == 76u) { dst[o++] = '\r'; dst[o++] = '\n'; line = 0; }
        const std::uint32_t b0 = p[i], b1 = i + 1 < n ? p[i + 1] : 0u, b2 = i + 2 < n ? p[i + 2] : 0u;
        dst[o++] = t[b0 >> 2];
        dst[o++] = t[((b0 & 3u) << 4) | (b1 >> 4)];
        dst[o++] = i + 1 < n ? t[((b1 & 15u) << 2) | (b2 >> 6)] : '=';
        dst[o++] = i + 2 < n ? t[b2 & 63u] : '=';
        line += 4;
    }
    return o;
}

enum class B64Status : std::uint8_t {
    Ok = 0,
    WrongEndingUnit,   // "Input byte array has wrong 4-byte ending unit"
    LastUnitShort,     // "Last unit does not have enough valid bits"
    IllegalChar,       // "Illegal base64 character <hex>" (basic decoder)
    TooShort,          // "Input byte[] should at least have 2 bytes for base64 bytes" (basic decoder)
    IncorrectEnding,   // "Input byte array has incorrect ending byte at <n>"
};

// The JDK's message for a failed decode (`at` is the byte value or position it names).
inline const char* b64_message(B64Status s) noexcept {
    switch (s) {
        case B64Status::WrongEndingUnit: return "Input byte array has wrong 4-byte ending unit";
        case B64Status::LastUnitShort: return "Last unit does not have enough valid bits";
        case B64Status::IllegalChar: return "Illegal base64 character";
        case B64Status::TooShort: return "Input byte[] should at least have 2 bytes for base64 bytes";
        case B64Status::IncorrectEnding: return "Input byte array has incorrect ending byte";
        default: return "";
    }
}

inline int b64_value(std::uint8_t c) noexcept {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    if (c == '=') return -2;
    return -1;
}

// java.util.Base64 decoding of s[0, n) into dst (>= 3n/4 + 3 bytes). `mime` is the lenient decoder
// (Base64.getMimeDecoder), otherwise the strict basic one.
inline B64Status base64_decode(const std::uint8_t* s, std::uint64_t n, bool mime, std::uint8_t* dst,
                               std::uint64_t* out_len) noexcept {
    assert(dst != nullptr && out_len != nullptr && (s != nullptr || n == 0));
    *out_len = 0;
    if (!mime) {
        if (n == 0) return B64Status::Ok;
        if (n < 2) return B64Status::TooShort;
    }
    std::uint64_t sp = 0, dp = 0;
    std::uint32_t bits = 0;
    int shiftto = 18;
    while (sp < n) {
        int b = b64_value(s[sp++]);
        if (b < 0) {
            if (b == -2) {
                if ((shiftto == 6 && (sp == n || s[sp++] != '=')) || shiftto == 18) return B64Status::WrongEndingUnit;
                break;
            }
            if (mime) continue;
            return B64Status::IllegalChar;
        }
        bits |= static_cast<std::uint32_t>(b) << shiftto;
        shiftto -= 6;
        if (shiftto < 0) {
            dst[dp++] = static_cast<std::uint8_t>(bits >> 16);
            dst[dp++] = static_cast<std::uint8_t>(bits >> 8);
            dst[dp++] = static_cast<std::uint8_t>(bits);
            shiftto = 18;
            bits = 0;
        }
    }
    if (shiftto == 6) {
        dst[dp++] = static_cast<std::uint8_t>(bits >> 16);
    } else if (shiftto == 0) {
        dst[dp++] = static_cast<std::uint8_t>(bits >> 16);
        dst[dp++] = static_cast<std::uint8_t>(bits >> 8);
    } else if (shiftto == 12) {
        return B64Status::LastUnitShort;
    }
    while (sp < n) {
        if (mime && b64_value(s[sp++]) < 0) continue;
        return B64Status::IncorrectEnding;
    }
    *out_len = dp;
    return B64Status::Ok;
}

// ---- conv -----------------------------------------------------------------------------------
enum class ConvStatus : std::uint8_t { Ok = 0, Null, Overflow };

inline int radix_digit(std::uint8_t c, int radix) noexcept {
    int d;
    if (c >= '0' && c <= '9') d = c - '0';
    else if (c >= 'a' && c <= 'z') d = c - 'a' + 10;
    else if (c >= 'A' && c <= 'Z') d = c - 'A' + 10;
    else return -1;
    return d < radix ? d : -1;
}

// Spark conv(num, from_base, to_base) over the (already trimmed) digits s[0, n). dst holds 65 bytes.
// Null: empty input or a base out of 2..36 (to_base by magnitude). Overflow: the digits exceed
// 2^64-1 (the caller reports it under ANSI, else `saturate` has produced 2^64-1 and Ok is returned).
inline ConvStatus conv(const std::uint8_t* s, std::uint64_t n, std::int64_t from_base, std::int64_t to_base,
                       bool ansi, char* dst, std::uint32_t* out_len) noexcept {
    assert(dst != nullptr && out_len != nullptr && (s != nullptr || n == 0));
    const std::int64_t abs_to = to_base < 0 ? -to_base : to_base;
    if (from_base < 2 || from_base > 36 || abs_to < 2 || abs_to > 36 || n == 0) return ConvStatus::Null;
    const bool negative = s[0] == '-';
    std::uint64_t v = 0;
    const std::uint64_t radix = static_cast<std::uint64_t>(from_base);
    for (std::uint64_t i = negative ? 1u : 0u; i < n; ++i) {
        const int d = radix_digit(s[i], static_cast<int>(from_base));
        if (d < 0) break;
        const std::uint64_t ud = static_cast<std::uint64_t>(d);
        if (v > (UINT64_MAX - ud) / radix) {
            if (ansi) return ConvStatus::Overflow;
            v = UINT64_MAX;
            break;
        }
        v = v * radix + ud;
    }
    bool neg_out = false;
    if (negative && to_base > 0) {
        v = (v >> 63) != 0 ? UINT64_MAX : (~v + 1u);
    }
    if (to_base < 0 && (v >> 63) != 0) {
        v = ~v + 1u;
        neg_out = true;
    }
    if (negative && to_base < 0) neg_out = true;
    static constexpr char dig[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    char tmp[64];
    std::uint32_t k = 0;
    const std::uint64_t base = static_cast<std::uint64_t>(abs_to);
    do { tmp[k++] = dig[v % base]; v /= base; } while (v != 0);
    std::uint32_t o = 0;
    if (neg_out) dst[o++] = '-';
    for (std::uint32_t i = 0; i < k; ++i) dst[o++] = tmp[k - 1 - i];
    *out_len = o;
    return ConvStatus::Ok;
}

// ---- charsets -------------------------------------------------------------------------------
enum class Charset : std::uint8_t { None = 0, UsAscii, Latin1, Utf8, Utf16, Utf16Be, Utf16Le, Utf32 };

// Spark's accepted names, compared in lower case (so "UTF-8" and "utf-8" are fine, "Utf8" is not).
inline Charset charset_of(const char* s, std::uint32_t n) noexcept {
    char b[16];
    if (n >= sizeof b) return Charset::None;
    for (std::uint32_t i = 0; i < n; ++i) b[i] = (s[i] >= 'A' && s[i] <= 'Z') ? static_cast<char>(s[i] + 32) : s[i];
    b[n] = '\0';
    if (!std::strcmp(b, "us-ascii")) return Charset::UsAscii;
    if (!std::strcmp(b, "iso-8859-1")) return Charset::Latin1;
    if (!std::strcmp(b, "utf-8")) return Charset::Utf8;
    if (!std::strcmp(b, "utf-16")) return Charset::Utf16;
    if (!std::strcmp(b, "utf-16be")) return Charset::Utf16Be;
    if (!std::strcmp(b, "utf-16le")) return Charset::Utf16Le;
    if (!std::strcmp(b, "utf-32")) return Charset::Utf32;
    return Charset::None;
}

// Strict UTF-8 well-formedness (Unicode Table 3-7: no overlongs, no surrogates, <= U+10FFFF).
// On success *cp holds the code point and *len its byte length.
inline bool utf8_next(const std::uint8_t* p, std::uint64_t n, std::uint32_t* cp, std::uint32_t* len) noexcept {
    assert(n > 0);
    const std::uint32_t b0 = p[0];
    if (b0 < 0x80u) { *cp = b0; *len = 1; return true; }
    std::uint32_t need, lo = 0x80u, hi = 0xBFu, c;
    if (b0 >= 0xC2u && b0 <= 0xDFu) { need = 1; c = b0 & 0x1Fu; }
    else if (b0 >= 0xE0u && b0 <= 0xEFu) { need = 2; c = b0 & 0x0Fu; if (b0 == 0xE0u) lo = 0xA0u; if (b0 == 0xEDu) hi = 0x9Fu; }
    else if (b0 >= 0xF0u && b0 <= 0xF4u) { need = 3; c = b0 & 0x07u; if (b0 == 0xF0u) lo = 0x90u; if (b0 == 0xF4u) hi = 0x8Fu; }
    else return false;
    if (n < need + 1u) return false;
    for (std::uint32_t i = 1; i <= need; ++i) {
        const std::uint32_t b = p[i];
        if (b < (i == 1 ? lo : 0x80u) || b > (i == 1 ? hi : 0xBFu)) return false;
        c = (c << 6) | (b & 0x3Fu);
    }
    *cp = c;
    *len = need + 1u;
    return true;
}

inline bool utf8_valid(const std::uint8_t* p, std::uint64_t n) noexcept {
    std::uint64_t i = 0;
    while (i < n) {
        std::uint32_t cp, l;
        if (!utf8_next(p + i, n - i, &cp, &l)) return false;
        i += l;
    }
    return true;
}

inline std::uint32_t utf8_put(std::uint32_t cp, std::uint8_t* d) noexcept {
    if (cp < 0x80u) { d[0] = static_cast<std::uint8_t>(cp); return 1; }
    if (cp < 0x800u) { d[0] = static_cast<std::uint8_t>(0xC0u | (cp >> 6)); d[1] = static_cast<std::uint8_t>(0x80u | (cp & 0x3Fu)); return 2; }
    if (cp < 0x10000u) {
        d[0] = static_cast<std::uint8_t>(0xE0u | (cp >> 12)); d[1] = static_cast<std::uint8_t>(0x80u | ((cp >> 6) & 0x3Fu));
        d[2] = static_cast<std::uint8_t>(0x80u | (cp & 0x3Fu)); return 3;
    }
    d[0] = static_cast<std::uint8_t>(0xF0u | (cp >> 18)); d[1] = static_cast<std::uint8_t>(0x80u | ((cp >> 12) & 0x3Fu));
    d[2] = static_cast<std::uint8_t>(0x80u | ((cp >> 6) & 0x3Fu)); d[3] = static_cast<std::uint8_t>(0x80u | (cp & 0x3Fu));
    return 4;
}

inline std::uint64_t charset_encode_cap(std::uint64_t n) noexcept { return 4u * n + 4u; }
inline std::uint64_t charset_decode_cap(std::uint64_t n) noexcept { return 3u * n + 4u; }

enum class CodingStatus : std::uint8_t { Ok = 0, Malformed, InvalidSource };

// encode(string, charset): the UTF-8 bytes s[0, n) re-encoded. A code point the charset cannot
// map is Malformed (Spark reports it); a source that is not well-formed UTF-8 is InvalidSource
// (Spark first repairs it with U+FFFD, which this kernel does not reproduce).
inline CodingStatus charset_encode(const std::uint8_t* s, std::uint64_t n, Charset cs, std::uint8_t* dst,
                                   std::uint64_t* out_len) noexcept {
    assert(dst != nullptr && out_len != nullptr && (s != nullptr || n == 0) && cs != Charset::None);
    if (!utf8_valid(s, n)) return CodingStatus::InvalidSource;
    std::uint64_t i = 0, o = 0;
    if (cs == Charset::Utf8) { if (n != 0) std::memcpy(dst, s, n); *out_len = n; return CodingStatus::Ok; }
    if (cs == Charset::Utf16) { dst[o++] = 0xFEu; dst[o++] = 0xFFu; }
    while (i < n) {
        std::uint32_t cp, l;
        (void)utf8_next(s + i, n - i, &cp, &l);
        i += l;
        switch (cs) {
            case Charset::UsAscii:
                if (cp > 0x7Fu) return CodingStatus::Malformed;
                dst[o++] = static_cast<std::uint8_t>(cp);
                break;
            case Charset::Latin1:
                if (cp > 0xFFu) return CodingStatus::Malformed;
                dst[o++] = static_cast<std::uint8_t>(cp);
                break;
            case Charset::Utf32:
                dst[o++] = static_cast<std::uint8_t>(cp >> 24); dst[o++] = static_cast<std::uint8_t>(cp >> 16);
                dst[o++] = static_cast<std::uint8_t>(cp >> 8); dst[o++] = static_cast<std::uint8_t>(cp);
                break;
            default: {   // UTF-16 family
                std::uint32_t u[2];
                std::uint32_t k = 1;
                if (cp >= 0x10000u) { const std::uint32_t v = cp - 0x10000u; u[0] = 0xD800u | (v >> 10); u[1] = 0xDC00u | (v & 0x3FFu); k = 2; }
                else u[0] = cp;
                for (std::uint32_t j = 0; j < k; ++j) {
                    if (cs == Charset::Utf16Le) { dst[o++] = static_cast<std::uint8_t>(u[j]); dst[o++] = static_cast<std::uint8_t>(u[j] >> 8); }
                    else { dst[o++] = static_cast<std::uint8_t>(u[j] >> 8); dst[o++] = static_cast<std::uint8_t>(u[j]); }
                }
            }
        }
    }
    *out_len = o;
    return CodingStatus::Ok;
}

// decode(binary, charset) into UTF-8. Malformed input (invalid sequence, unpaired surrogate,
// truncated unit, byte >= 0x80 for US-ASCII) is Malformed.
inline CodingStatus charset_decode(const std::uint8_t* s, std::uint64_t n, Charset cs, std::uint8_t* dst,
                                   std::uint64_t* out_len) noexcept {
    assert(dst != nullptr && out_len != nullptr && (s != nullptr || n == 0) && cs != Charset::None);
    std::uint64_t o = 0, i = 0;
    switch (cs) {
        case Charset::Utf8:
            if (!utf8_valid(s, n)) return CodingStatus::Malformed;
            if (n != 0) std::memcpy(dst, s, n);
            *out_len = n;
            return CodingStatus::Ok;
        case Charset::UsAscii:
            for (; i < n; ++i) { if (s[i] > 0x7Fu) return CodingStatus::Malformed; dst[o++] = s[i]; }
            break;
        case Charset::Latin1:
            for (; i < n; ++i) o += utf8_put(s[i], dst + o);
            break;
        case Charset::Utf32: {
            bool le = false;
            if (n >= 4) {
                const std::uint32_t w = (static_cast<std::uint32_t>(s[0]) << 24) | (static_cast<std::uint32_t>(s[1]) << 16) |
                                        (static_cast<std::uint32_t>(s[2]) << 8) | s[3];
                if (w == 0x0000FEFFu) i = 4;
                else if (w == 0xFFFE0000u) { i = 4; le = true; }
            }
            if ((n - i) % 4u != 0) return CodingStatus::Malformed;
            for (; i < n; i += 4) {
                std::uint32_t cp = le ? (static_cast<std::uint32_t>(s[i + 3]) << 24) | (static_cast<std::uint32_t>(s[i + 2]) << 16) |
                                        (static_cast<std::uint32_t>(s[i + 1]) << 8) | s[i]
                                      : (static_cast<std::uint32_t>(s[i]) << 24) | (static_cast<std::uint32_t>(s[i + 1]) << 16) |
                                        (static_cast<std::uint32_t>(s[i + 2]) << 8) | s[i + 3];
                if (cp > 0x10FFFFu || (cp >= 0xD800u && cp <= 0xDFFFu)) return CodingStatus::Malformed;
                o += utf8_put(cp, dst + o);
            }
            break;
        }
        default: {   // UTF-16 family
            bool le = cs == Charset::Utf16Le;
            if (cs == Charset::Utf16 && n >= 2) {
                if (s[0] == 0xFEu && s[1] == 0xFFu) i = 2;
                else if (s[0] == 0xFFu && s[1] == 0xFEu) { i = 2; le = true; }
            }
            if ((n - i) % 2u != 0) return CodingStatus::Malformed;
            while (i < n) {
                const std::uint32_t u = le ? (static_cast<std::uint32_t>(s[i + 1]) << 8) | s[i]
                                           : (static_cast<std::uint32_t>(s[i]) << 8) | s[i + 1];
                i += 2;
                if (u >= 0xDC00u && u <= 0xDFFFu) return CodingStatus::Malformed;
                if (u >= 0xD800u && u <= 0xDBFFu) {
                    if (i + 2 > n) return CodingStatus::Malformed;
                    const std::uint32_t w = le ? (static_cast<std::uint32_t>(s[i + 1]) << 8) | s[i]
                                               : (static_cast<std::uint32_t>(s[i]) << 8) | s[i + 1];
                    if (w < 0xDC00u || w > 0xDFFFu) return CodingStatus::Malformed;
                    i += 2;
                    o += utf8_put(0x10000u + ((u - 0xD800u) << 10) + (w - 0xDC00u), dst + o);
                } else {
                    o += utf8_put(u, dst + o);
                }
            }
        }
    }
    *out_len = o;
    return CodingStatus::Ok;
}

}  // namespace encx
}  // namespace kernels
}  // namespace bolt
