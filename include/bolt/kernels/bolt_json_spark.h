// bolt/kernels/bolt_json_spark.h — Spark 4 JSON text semantics: a Jackson-compatible
// pull parser and the get_json_object / json_array_length / json_object_keys kernels.
//
// What Spark does, exactly, is what Jackson (2.18) does under Spark's factory options:
// single-quoted strings and raw control characters inside strings are accepted; comments,
// unquoted names, NaN/Infinity, leading zeros, trailing commas and `\x` escapes other than
// the JSON ones are errors. An error anywhere on the path the evaluator walks makes the
// answer NULL (never an exception). Nothing after the root value is read.
//
// Everything is over caller buffers: no allocation, no exceptions, nesting bounded by
// kMaxNesting (Jackson's StreamReadConstraints default), outputs bounded by the caller's
// buffer (overflow is reported, never silently truncated).
//
// Tiger Style: noexcept, >=2 asserts per non-trivial function, loops bounded by input size.
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bolt/kernels/bolt_float_chars.h"
#include "bolt/kernels/bolt_float_parse.h"

namespace bolt {
namespace kernels {
namespace jsonspark {

inline constexpr uint32_t kMaxNesting = 1000;       // Jackson StreamReadConstraints.maxNestingDepth
inline constexpr uint32_t kMaxNumberChars = 1000;   // StreamReadConstraints.maxNumberLength
inline constexpr uint32_t kMaxPathInstr = 256;      // compiled path instructions (budget json_path_instructions)
inline constexpr uint32_t kMaxPathBytes = 4096;     // bytes of names in one compiled path
inline constexpr uint32_t kMaxEvalDepth = 256;      // path-evaluator recursion (flattened array nesting); past it: budget refusal

enum class Tok : uint8_t {
    None = 0, StartObject, EndObject, StartArray, EndArray, FieldName,
    String, Int, Float, True, False, Null, Error,
};

// ---- output buffer ------------------------------------------------------------------------------------------------------

struct Out {
    char*    p;
    uint32_t cap;
    uint32_t n;
    bool     over;                       // a write past cap (the answer is refused by the caller)
    void put(char c) noexcept { if (n < cap) p[n] = c; else over = true; ++n; }
    void put(const char* s, uint32_t len) noexcept {
        if (len == 0) return;
        if (len <= cap && n <= cap - len) std::memcpy(p + n, s, len); else over = true;
        n += len;
    }
};

// ---- pull parser --------------------------------------------------------------------------------------------------------

struct Parser {
    const uint8_t* s;
    uint32_t       n;
    uint32_t       i;
    Tok            cur;
    uint32_t       depth;                // open containers
    bool           has_elem;             // the current container already holds an element
    bool           in_obj;               // the current container is an object
    uint64_t       obj_bits[(kMaxNesting + 63) / 64];   // per open container below the top: object?
    uint64_t       elem_bits[(kMaxNesting + 63) / 64];  // per open container below the top: had an element?
    uint32_t       tb, te;               // string content / number text, byte range in s
    uint8_t        quote;                // the quote char of the current string / name
    bool           esc;                  // the token contains a backslash
    bool           ctrl_ok;              // raw control characters inside strings (get_json_object: yes; from_json: no)
    bool           nan_ok;               // NaN / Infinity / +-INF number tokens (from_json, schema_of_json: yes)
};

inline void parser_init(Parser* p, const uint8_t* s, uint32_t n, bool ctrl_ok = true, bool nan_ok = false) noexcept {
    assert(p != nullptr && (s != nullptr || n == 0));
    p->s = s; p->n = n; p->i = 0; p->cur = Tok::None; p->depth = 0;
    p->has_elem = false; p->in_obj = false; p->tb = p->te = 0; p->quote = 0; p->esc = false;
    p->ctrl_ok = ctrl_ok; p->nan_ok = nan_ok;
    std::memset(p->obj_bits, 0, sizeof p->obj_bits);
    std::memset(p->elem_bits, 0, sizeof p->elem_bits);
}

namespace detail {

inline bool is_ws(uint8_t c) noexcept { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
inline void skip_ws(Parser* p) noexcept { while (p->i < p->n && is_ws(p->s[p->i])) ++p->i; }
inline Tok fail(Parser* p) noexcept { p->cur = Tok::Error; return Tok::Error; }
inline bool is_digit(uint8_t c) noexcept { return c >= '0' && c <= '9'; }

inline int hexv(uint8_t c) noexcept {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

inline void push(Parser* p, bool obj) noexcept {
    const uint32_t d = p->depth;
    const uint64_t bit = uint64_t{1} << (d & 63u);
    if (p->in_obj) p->obj_bits[d >> 6] |= bit; else p->obj_bits[d >> 6] &= ~bit;
    if (p->has_elem) p->elem_bits[d >> 6] |= bit; else p->elem_bits[d >> 6] &= ~bit;
    ++p->depth;
    p->in_obj = obj;
    p->has_elem = false;
}
inline void pop(Parser* p) noexcept {
    assert(p->depth > 0);
    --p->depth;
    const uint32_t d = p->depth;
    const uint64_t bit = uint64_t{1} << (d & 63u);
    p->in_obj = (p->obj_bits[d >> 6] & bit) != 0;
    p->has_elem = (p->elem_bits[d >> 6] & bit) != 0;
}

// Scan a quoted string from s[i] (the opening quote); validates escapes and UTF-8. Sets tb/te.
inline bool scan_string(Parser* p) noexcept {
    assert(p->i < p->n);
    const uint8_t q = p->s[p->i];
    assert(q == '"' || q == '\'');
    p->quote = q;
    p->esc = false;
    uint32_t i = p->i + 1;
    p->tb = i;
    while (i < p->n) {
        const uint8_t c = p->s[i];
        if (c == q) { p->te = i; p->i = i + 1; return true; }
        if (c == '\\') {
            p->esc = true;
            if (i + 1 >= p->n) return false;
            const uint8_t e = p->s[i + 1];
            if (e == 'u') {
                if (i + 5 >= p->n) return false;
                for (uint32_t k = 2; k < 6; ++k) if (hexv(p->s[i + k]) < 0) return false;
                i += 6;
            } else if (e == '"' || e == '\'' || e == '\\' || e == '/' || e == 'b' || e == 'f' || e == 'n' || e == 'r' || e == 't') {
                i += 2;
            } else {
                return false;
            }
        } else if (c < 0x80) {
            if (c < 0x20 && !p->ctrl_ok) return false;
            ++i;
        } else {                                                    // multi-byte UTF-8: continuation bytes must follow
            const uint32_t len = c >= 0xF0 ? 4u : c >= 0xE0 ? 3u : c >= 0xC2 ? 2u : 0u;
            if (len == 0 || c > 0xF4 || i + len > p->n) return false;
            for (uint32_t k = 1; k < len; ++k) if ((p->s[i + k] & 0xC0) != 0x80) return false;
            i += len;
        }
    }
    return false;                                                   // unterminated
}

inline bool ident_part(uint8_t c) noexcept {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == '$' ||
           c >= 0x7F;
}

inline bool scan_literal(Parser* p, const char* w, uint32_t len) noexcept {
    if (p->n - p->i < len || std::memcmp(p->s + p->i, w, len) != 0) return false;
    p->i += len;
    if (p->i < p->n) {
        const uint8_t c = p->s[p->i];
        if (c >= '0' && c != ']' && c != '}' && ident_part(c)) return false;
    }
    return true;
}

// A JSON number (Jackson's grammar: no leading zeros, digits around '.', digits after an exponent).
inline Tok scan_number(Parser* p) noexcept {
    const uint32_t b = p->i;
    uint32_t i = b;
    if (p->s[i] == '-') ++i;
    if (i >= p->n || !is_digit(p->s[i])) return fail(p);
    if (p->s[i] == '0') {
        ++i;
        if (i < p->n && is_digit(p->s[i])) return fail(p);
    } else {
        while (i < p->n && is_digit(p->s[i])) ++i;
    }
    bool flt = false;
    if (i < p->n && p->s[i] == '.') {
        ++i;
        if (i >= p->n || !is_digit(p->s[i])) return fail(p);
        while (i < p->n && is_digit(p->s[i])) ++i;
        flt = true;
    }
    if (i < p->n && (p->s[i] == 'e' || p->s[i] == 'E')) {
        ++i;
        if (i < p->n && (p->s[i] == '+' || p->s[i] == '-')) ++i;
        if (i >= p->n || !is_digit(p->s[i])) return fail(p);
        while (i < p->n && is_digit(p->s[i])) ++i;
        flt = true;
    }
    if (i - b > kMaxNumberChars) return fail(p);
    p->tb = b; p->te = i; p->i = i;
    p->cur = flt ? Tok::Float : Tok::Int;
    return p->cur;
}

// The value starting at s[i] (whitespace consumed).
inline Tok scan_value(Parser* p, bool root) noexcept {
    assert(p != nullptr && p->i < p->n);
    const uint8_t c = p->s[p->i];
    switch (c) {
        case '"': case '\'':
            if (!scan_string(p)) return fail(p);
            p->cur = Tok::String;
            return p->cur;
        case '{': case '[':
            if (p->depth >= kMaxNesting) return fail(p);
            ++p->i;
            push(p, c == '{');
            p->cur = c == '{' ? Tok::StartObject : Tok::StartArray;
            return p->cur;
        case 't': if (!scan_literal(p, "true", 4)) return fail(p); p->cur = Tok::True; return p->cur;
        case 'f': if (!scan_literal(p, "false", 5)) return fail(p); p->cur = Tok::False; return p->cur;
        case 'n': if (!scan_literal(p, "null", 4)) return fail(p); p->cur = Tok::Null; return p->cur;
        default:
            if (p->nan_ok && (c == 'N' || c == 'I' || ((c == '-' || c == '+') && p->i + 1 < p->n && p->s[p->i + 1] == 'I') ||
                              (c == 'N'))) {
                static const char* const kw[] = {"NaN", "Infinity", "+Infinity", "-Infinity", "+INF", "-INF"};
                for (const char* w : kw) {
                    if (scan_literal(p, w, static_cast<uint32_t>(std::strlen(w)))) {
                        p->tb = p->i - static_cast<uint32_t>(std::strlen(w)); p->te = p->i;
                        p->cur = Tok::Float;
                        return p->cur;
                    }
                }
                return fail(p);
            }
            if (c == '-' || is_digit(c)) {
                const Tok t = scan_number(p);
                if (t == Tok::Error) return t;
                if (root && p->i < p->n && !is_ws(p->s[p->i])) return fail(p);   // Jackson: root-level values need a separator
                return t;
            }
            return fail(p);
    }
}

}  // namespace detail

// Advance to the next token (Jackson's nextToken). Tok::None at the end of a root-level input.
inline Tok next(Parser* p) noexcept {
    assert(p != nullptr && p->s != nullptr);
    assert(p->depth <= kMaxNesting);
    using namespace detail;
    if (p->cur == Tok::Error) return Tok::Error;
    if (p->cur == Tok::FieldName) {                                  // ':' value
        skip_ws(p);
        if (p->i >= p->n || p->s[p->i] != ':') return fail(p);
        ++p->i;
        skip_ws(p);
        if (p->i >= p->n) return fail(p);
        return scan_value(p, false);
    }
    skip_ws(p);
    if (p->i >= p->n) {
        if (p->depth != 0) return fail(p);
        p->cur = Tok::None;
        return Tok::None;
    }
    const uint8_t c = p->s[p->i];
    if (c == ']' || c == '}') {
        if (p->depth == 0 || p->in_obj != (c == '}')) return fail(p);
        ++p->i;
        pop(p);
        p->has_elem = true;
        p->cur = c == '}' ? Tok::EndObject : Tok::EndArray;
        return p->cur;
    }
    if (p->depth > 0 && p->has_elem) {
        if (c != ',') return fail(p);
        ++p->i;
        skip_ws(p);
        if (p->i >= p->n) return fail(p);
    }
    if (p->depth > 0 && p->in_obj) {                                 // a field name
        const uint8_t q = p->s[p->i];
        if (q != '"' && q != '\'') return fail(p);
        if (!scan_string(p)) return fail(p);
        p->has_elem = true;
        p->cur = Tok::FieldName;
        return p->cur;
    }
    if (p->depth > 0) p->has_elem = true;
    return scan_value(p, p->depth == 0);
}

// Jackson's skipChildren: past the matching end of the current container (a no-op elsewhere).
inline bool skip_children(Parser* p) noexcept {
    assert(p != nullptr);
    if (p->cur != Tok::StartObject && p->cur != Tok::StartArray) return p->cur != Tok::Error;
    const uint32_t target = p->depth - 1u;
    for (uint64_t guard = 0; guard <= uint64_t{p->n} + 1u; ++guard) {   // each token consumes >= 1 byte
        const Tok t = next(p);
        if (t == Tok::Error || t == Tok::None) return false;
        if ((t == Tok::EndObject || t == Tok::EndArray) && p->depth == target) return true;
    }
    return false;
}

// ---- string tokens ------------------------------------------------------------------------------------------------------

// Walk the decoded UTF-16 code units of the current String / FieldName token: `unit(u)` per code unit.
template <class F>
inline void for_each_unit(const Parser& p, F unit) noexcept {
    uint32_t i = p.tb;
    while (i < p.te) {
        const uint8_t c = p.s[i];
        if (c == '\\') {
            const uint8_t e = p.s[i + 1];
            if (e == 'u') {
                const uint32_t u = (static_cast<uint32_t>(detail::hexv(p.s[i + 2])) << 12) |
                                   (static_cast<uint32_t>(detail::hexv(p.s[i + 3])) << 8) |
                                   (static_cast<uint32_t>(detail::hexv(p.s[i + 4])) << 4) |
                                   static_cast<uint32_t>(detail::hexv(p.s[i + 5]));
                unit(u);
                i += 6;
                continue;
            }
            const uint32_t m = e == 'b' ? 8u : e == 'f' ? 12u : e == 'n' ? 10u : e == 'r' ? 13u : e == 't' ? 9u : e;
            unit(m);
            i += 2;
        } else if (c < 0x80) {
            unit(c);
            ++i;
        } else if (c < 0xE0) {
            unit(((c & 0x1Fu) << 6) | (p.s[i + 1] & 0x3Fu));
            i += 2;
        } else if (c < 0xF0) {
            unit(((c & 0x0Fu) << 12) | ((p.s[i + 1] & 0x3Fu) << 6) | (p.s[i + 2] & 0x3Fu));
            i += 3;
        } else {
            const uint32_t cp = ((c & 0x07u) << 18) | ((p.s[i + 1] & 0x3Fu) << 12) | ((p.s[i + 2] & 0x3Fu) << 6) |
                                (p.s[i + 3] & 0x3Fu);
            const uint32_t v = cp - 0x10000u;
            unit(0xD800u + (v >> 10));
            unit(0xDC00u + (v & 0x3FFu));
            i += 4;
        }
    }
}

inline void put_utf8(Out* o, uint32_t cp) noexcept {
    if (cp < 0x80) { o->put(static_cast<char>(cp)); return; }
    if (cp < 0x800) { o->put(static_cast<char>(0xC0 | (cp >> 6))); o->put(static_cast<char>(0x80 | (cp & 0x3F))); return; }
    if (cp < 0x10000) {
        o->put(static_cast<char>(0xE0 | (cp >> 12)));
        o->put(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        o->put(static_cast<char>(0x80 | (cp & 0x3F)));
        return;
    }
    o->put(static_cast<char>(0xF0 | (cp >> 18)));
    o->put(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    o->put(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    o->put(static_cast<char>(0x80 | (cp & 0x3F)));
}

// The current string / name as UTF-8 text (a lone surrogate is `lone`: '?' as Java's String -> UTF-8 does, or an
// error for the raw writer). Returns false when a lone surrogate met strict mode.
inline bool token_utf8(const Parser& p, Out* o, bool strict) noexcept {
    uint32_t hi = 0;
    bool ok = true;
    for_each_unit(p, [&](uint32_t u) noexcept {
        if (hi != 0) {
            if (u >= 0xDC00 && u <= 0xDFFF) { put_utf8(o, 0x10000u + ((hi - 0xD800u) << 10) + (u - 0xDC00u)); hi = 0; return; }
            if (strict) ok = false; else o->put('?');
            hi = 0;
        }
        if (u >= 0xD800 && u <= 0xDBFF) { hi = u; return; }
        if (u >= 0xDC00 && u <= 0xDFFF) { if (strict) ok = false; else o->put('?'); return; }
        put_utf8(o, u);
    });
    if (hi != 0) { if (strict) ok = false; else o->put('?'); }
    return ok;
}

// Jackson's JSON string escaping of the current token (quotes added by the caller).
inline void put_unit_escaped(Out* o, uint32_t u) noexcept {
    static const char hexd[] = "0123456789ABCDEF";
    if (u == '"') { o->put("\\\"", 2); return; }
    if (u == '\\') { o->put("\\\\", 2); return; }
    if (u < 0x20) {
        switch (u) {
            case 8: o->put("\\b", 2); return;
            case 9: o->put("\\t", 2); return;
            case 10: o->put("\\n", 2); return;
            case 12: o->put("\\f", 2); return;
            case 13: o->put("\\r", 2); return;
            default: break;
        }
        o->put("\\u00", 4); o->put(hexd[u >> 4]); o->put(hexd[u & 15]);
        return;
    }
    if (u >= 0xD800 && u <= 0xDFFF) {
        o->put("\\u", 2);
        o->put(hexd[(u >> 12) & 15]); o->put(hexd[(u >> 8) & 15]); o->put(hexd[(u >> 4) & 15]); o->put(hexd[u & 15]);
        return;
    }
    put_utf8(o, u);
}

// `chars`: the char-based generator Spark buffers wildcard matches in (supplementary characters stay whole, a lone
// surrogate cannot be re-encoded: *bad). Otherwise the byte generator (a surrogate becomes a \uXXXX escape).
inline void token_json_string(const Parser& p, Out* o, bool chars = false, bool* bad = nullptr) noexcept {
    o->put('"');
    if (!chars) {
        for_each_unit(p, [&](uint32_t u) noexcept { put_unit_escaped(o, u); });
    } else {
        uint32_t hi = 0;
        for_each_unit(p, [&](uint32_t u) noexcept {
            if (hi != 0) {
                if (u >= 0xDC00 && u <= 0xDFFF) { put_utf8(o, 0x10000u + ((hi - 0xD800u) << 10) + (u - 0xDC00u)); hi = 0; return; }
                if (bad != nullptr) *bad = true;
                hi = 0;
            }
            if (u >= 0xD800 && u <= 0xDBFF) { hi = u; return; }
            if (u >= 0xDC00 && u <= 0xDFFF) { if (bad != nullptr) *bad = true; return; }
            put_unit_escaped(o, u);
        });
        if (hi != 0 && bad != nullptr) *bad = true;
    }
    o->put('"');
}

// Does the current token's decoded text equal name[0..len) (UTF-8)? A lone surrogate never equals valid UTF-8.
inline bool token_equals(const Parser& p, const char* name, uint32_t len) noexcept {
    if (!p.esc) {                                                    // raw bytes are the text
        return (p.te - p.tb) == len && std::memcmp(p.s + p.tb, name, len) == 0;
    }
    char buf[kMaxPathBytes + 8];
    Out o{buf, sizeof buf, 0, false};
    if (!token_utf8(p, &o, true) || o.over) return false;
    return o.n == len && std::memcmp(buf, name, len) == 0;
}

// ---- numbers ------------------------------------------------------------------------------------------------------------

// Java's Double.toString (JDK 19+: the shortest decimal that rounds to the double; at least one fraction digit).
inline uint32_t java_double_chars(char* out, uint32_t cap, double v) noexcept {
    assert(out != nullptr && cap >= 32);
    if (v != v) { std::memcpy(out, "NaN", 3); return 3; }
    uint32_t w = 0;
    if (std::signbit(v)) { out[w++] = '-'; v = -v; }
    if (v == 0.0) { std::memcpy(out + w, "0.0", 3); return w + 3; }
    if (v > 1.7976931348623157e308) { std::memcpy(out + w, "Infinity", 8); return w + 8; }
    char dig[24];
    int32_t k = 0;
    uint32_t nd = float_chars::detail::shortest_digits(v, dig, &k);          // v ~= 0.d1d2..dn * 10^k
    if (nd == 1 && v < 2.2250738585072014e-308) {                    // JDK renders >= 2 digits: the closest 2-digit one
        char t[40];
        std::snprintf(t, sizeof t, "%.1e", v);
        if (std::strtod(t, nullptr) == v) { dig[0] = t[0]; dig[1] = t[2]; nd = 2; k = std::atoi(t + 4) + 1; }
    }
    const int32_t sci = k - 1;                                       // d.ddd * 10^sci
    if (v >= 1e-3 && v < 1e7) {
        if (k <= 0) {                                                // 0.00ddd
            out[w++] = '0'; out[w++] = '.';
            for (int32_t z = 0; z < -k; ++z) out[w++] = '0';
            for (uint32_t d = 0; d < nd; ++d) out[w++] = dig[d];
        } else if (static_cast<uint32_t>(k) >= nd) {                 // ddd000.0
            for (uint32_t d = 0; d < nd; ++d) out[w++] = dig[d];
            for (uint32_t z = nd; z < static_cast<uint32_t>(k); ++z) out[w++] = '0';
            out[w++] = '.'; out[w++] = '0';
        } else {                                                     // dd.ddd
            for (int32_t d = 0; d < k; ++d) out[w++] = dig[d];
            out[w++] = '.';
            for (uint32_t d = static_cast<uint32_t>(k); d < nd; ++d) out[w++] = dig[d];
        }
        return w;
    }
    out[w++] = dig[0]; out[w++] = '.';
    if (nd == 1) out[w++] = '0';
    for (uint32_t d = 1; d < nd; ++d) out[w++] = dig[d];
    out[w++] = 'E';
    w += static_cast<uint32_t>(std::snprintf(out + w, cap - w, "%d", sci));
    return w;
}

// The value of an Int / Float token as a double (correctly rounded; NaN / Infinity / +-INF tokens included).
inline double token_double(const Parser& p) noexcept {
    assert(p.cur == Tok::Int || p.cur == Tok::Float);
    const char* t = reinterpret_cast<const char*>(p.s + p.tb);
    const uint32_t len = p.te - p.tb;
    if (len > 0 && (t[0] == 'N')) return std::nan("");
    if (len > 1 && (t[0] == 'I' || t[1] == 'I')) return t[0] == '-' ? -HUGE_VAL : HUGE_VAL;
    if (p.cur == Tok::Int && len == 2 && t[0] == '-' && t[1] == '0') return 0.0;        // an integer token has no negative zero
    double d = 0;
    (void)float_chars::f64_from_json_chars(t, t + len, &d);
    return d;
}

// The number token as Jackson's copyCurrentStructure writes it: int / long / big integer text, or a double.
inline void put_number(const Parser& p, Out* o) noexcept {
    assert(p.cur == Tok::Int || p.cur == Tok::Float);
    const char* t = reinterpret_cast<const char*>(p.s + p.tb);
    const uint32_t len = p.te - p.tb;
    if (p.cur == Tok::Int) {
        if (len == 2 && t[0] == '-' && t[1] == '0') { o->put('0'); return; }
        o->put(t, len);
        return;
    }
    const double d = token_double(p);
    if (d != d) { o->put("\"NaN\"", 5); return; }
    if (d > 1.7976931348623157e308) { o->put("\"Infinity\"", 10); return; }
    if (d < -1.7976931348623157e308) { o->put("\"-Infinity\"", 11); return; }
    char b[48];
    o->put(b, java_double_chars(b, sizeof b, d));
}

// ---- JSON writer (Jackson's generator: commas and escaping) --------------------------------------------------------------

struct Gen {
    Out*     o;
    bool     chars;     // the char-based generator (see token_json_string)
    bool     bad;       // a string it could not encode
    uint32_t depth;
    bool     first[kMaxNesting + 8];     // the container at [depth] has no element yet
    bool     after_name;
};

inline void gen_init(Gen* g, Out* o) noexcept {
    g->o = o; g->chars = false; g->bad = false; g->depth = 0; g->first[0] = true; g->after_name = false;
}
inline bool gen_value_prefix(Gen* g) noexcept {
    if (g->after_name) { g->after_name = false; return true; }
    if (g->depth > 0) {
        if (!g->first[g->depth]) g->o->put(',');
        g->first[g->depth] = false;
    } else {
        if (!g->first[0]) return false;                              // one root value
        g->first[0] = false;
    }
    return true;
}
inline bool gen_start(Gen* g, char c) noexcept {
    if (g->depth + 1 >= kMaxNesting + 8 || !gen_value_prefix(g)) return false;
    g->o->put(c);
    g->first[++g->depth] = true;
    return true;
}
inline void gen_end(Gen* g, char c) noexcept { assert(g->depth > 0); --g->depth; g->o->put(c); }
inline bool gen_name(Gen* g, const Parser& p) noexcept {
    if (g->depth > 0) {
        if (!g->first[g->depth]) g->o->put(',');
        g->first[g->depth] = false;
    }
    token_json_string(p, g->o, g->chars, &g->bad);
    g->o->put(':');
    g->after_name = true;
    return true;
}
inline bool gen_raw_value(Gen* g, const char* s, uint32_t len) noexcept {
    if (!gen_value_prefix(g)) return false;
    g->o->put(s, len);
    return true;
}

// copyCurrentStructure: the current token's whole value (scalar, or container to its matching end).
inline bool copy_structure(Parser* p, Gen* g) noexcept {
    assert(p != nullptr && g != nullptr);
    const bool cont = p->cur == Tok::StartObject || p->cur == Tok::StartArray;
    const uint32_t target = cont ? p->depth - 1u : p->depth;
    Tok t = p->cur;
    for (uint64_t guard = 0; guard <= uint64_t{p->n} * 2u + 4u; ++guard) {
        switch (t) {
            case Tok::StartObject: if (!gen_start(g, '{')) return false; break;
            case Tok::StartArray: if (!gen_start(g, '[')) return false; break;
            case Tok::EndObject: gen_end(g, '}'); break;
            case Tok::EndArray: gen_end(g, ']'); break;
            case Tok::FieldName: gen_name(g, *p); break;
            case Tok::String:
                if (!gen_value_prefix(g)) return false;
                token_json_string(*p, g->o, g->chars, &g->bad);
                break;
            case Tok::Int: case Tok::Float:
                if (!gen_value_prefix(g)) return false;
                put_number(*p, g->o);
                break;
            case Tok::True: if (!gen_raw_value(g, "true", 4)) return false; break;
            case Tok::False: if (!gen_raw_value(g, "false", 5)) return false; break;
            case Tok::Null: if (!gen_raw_value(g, "null", 4)) return false; break;
            default: return false;
        }
        if (g->bad) return false;
        if (!cont) return true;
        if ((t == Tok::EndObject || t == Tok::EndArray) && p->depth == target) return true;
        t = next(p);
        if (t == Tok::Error || t == Tok::None) return false;
    }
    return false;
}

// ---- Spark's JSON path (JsonPathParser) -----------------------------------------------------------------------------------

enum class Ins : uint8_t { Key = 0, Named, Wildcard, Subscript, Index };

struct PathIns {
    Ins      kind;
    uint16_t name_off, name_len;    // Named: into SparkPath::names
    int64_t  index;                 // Index
};
struct SparkPath {
    uint32_t n;                     // instructions (0 = the root `$`)
    uint32_t name_bytes;
    uint32_t wildcards;
    PathIns  ins[kMaxPathInstr];
    char     names[kMaxPathBytes];
};

// $ then rep( .name | ['name'] | .* | ['*'] | [*] | [n] ), whole input, no whitespace. False: the path is invalid
// (every row is NULL). `over` set when a budget (instructions / name bytes) was exceeded instead.
inline bool path_compile(const char* s, uint32_t len, SparkPath* out, bool* over) noexcept {
    assert(out != nullptr && over != nullptr);
    out->n = 0; out->name_bytes = 0; out->wildcards = 0; *over = false;
    if (len == 0 || s[0] != '$') return false;
    uint32_t i = 1;
    for (uint32_t guard = 0; i < len && guard <= len; ++guard) {
        if (out->n + 2 > kMaxPathInstr) { *over = true; return false; }
        PathIns* a = &out->ins[out->n];
        if (len - i >= 2 && s[i] == '.' && s[i + 1] == '*') {        // `.*`
            a[0] = PathIns{Ins::Key, 0, 0, 0}; a[1] = PathIns{Ins::Wildcard, 0, 0, 0};
            out->n += 2; i += 2; continue;
        }
        if (len - i >= 5 && std::memcmp(s + i, "['*']", 5) == 0) {
            a[0] = PathIns{Ins::Key, 0, 0, 0}; a[1] = PathIns{Ins::Wildcard, 0, 0, 0};
            out->n += 2; i += 5; continue;
        }
        if (s[i] == '.') {                                           // .name : [^.\[]+
            uint32_t j = i + 1;
            while (j < len && s[j] != '.' && s[j] != '[') ++j;
            if (j == i + 1) return false;
            if (out->name_bytes + (j - i - 1) > kMaxPathBytes) { *over = true; return false; }
            std::memcpy(out->names + out->name_bytes, s + i + 1, j - i - 1);
            a[0] = PathIns{Ins::Key, 0, 0, 0};
            a[1] = PathIns{Ins::Named, static_cast<uint16_t>(out->name_bytes), static_cast<uint16_t>(j - i - 1), 0};
            out->name_bytes += j - i - 1; out->n += 2; i = j; continue;
        }
        if (s[i] != '[') return false;
        if (len - i >= 2 && s[i + 1] == '\'') {                      // ['name'] : [^'?]+
            uint32_t j = i + 2;
            while (j < len && s[j] != '\'' && s[j] != '?') ++j;
            if (j == i + 2 || j + 1 >= len || s[j] != '\'' || s[j + 1] != ']') return false;
            if (out->name_bytes + (j - i - 2) > kMaxPathBytes) { *over = true; return false; }
            std::memcpy(out->names + out->name_bytes, s + i + 2, j - i - 2);
            a[0] = PathIns{Ins::Key, 0, 0, 0};
            a[1] = PathIns{Ins::Named, static_cast<uint16_t>(out->name_bytes), static_cast<uint16_t>(j - i - 2), 0};
            out->name_bytes += j - i - 2; out->n += 2; i = j + 2; continue;
        }
        if (len - i >= 3 && s[i + 1] == '*' && s[i + 2] == ']') {    // [*]
            a[0] = PathIns{Ins::Subscript, 0, 0, 0}; a[1] = PathIns{Ins::Wildcard, 0, 0, 0};
            ++out->wildcards; out->n += 2; i += 3; continue;
        }
        uint32_t j = i + 1;
        int64_t v = 0;
        while (j < len && detail::is_digit(static_cast<uint8_t>(s[j]))) {
            if (v > (INT64_MAX - 9) / 10) return false;              // Scala's toLong fails: the path never matches
            v = v * 10 + (s[j] - '0'); ++j;
        }
        if (j == i + 1 || j >= len || s[j] != ']') return false;
        a[0] = PathIns{Ins::Subscript, 0, 0, 0}; a[1] = PathIns{Ins::Index, 0, 0, v};
        out->n += 2; i = j + 1;
    }
    return true;
}

// ---- get_json_object ----------------------------------------------------------------------------------------------------

enum class Style : uint8_t { Raw, Quoted, Flatten };

struct Eval {
    Parser*          p;
    const SparkPath* path;
    uint32_t         rec;       // recursion depth (bounded by nesting + path length)
};

namespace detail {

inline bool eval_path(Eval* e, Gen* g, Style st, uint32_t k, bool* dirty) noexcept;

// One path step over the current token; false = a parse error (the answer is NULL).
inline bool eval_path(Eval* e, Gen* g, Style st, uint32_t k, bool* dirty) noexcept {
    assert(e != nullptr && g != nullptr && dirty != nullptr);
    Parser* p = e->p;
    *dirty = false;
    if (++e->rec > kMaxEvalDepth) { g->o->over = true; return false; }
    const SparkPath& pa = *e->path;
    const uint32_t rest = pa.n - k;                                   // instructions left
    const PathIns* ins = pa.ins + k;
    const Tok t = p->cur;
    bool ok = true;
    if (rest == 0) {
        if (t == Tok::String && st == Style::Raw) {
            Out* o = g->o;
            if (!gen_value_prefix(g)) { ok = false; }
            else if (!token_utf8(*p, o, true)) ok = false;
            else *dirty = true;
        } else if (t == Tok::StartArray && st == Style::Flatten) {
            for (uint64_t guard = 0; ok; ++guard) {
                const Tok nt = next(p);
                if (nt == Tok::Error || nt == Tok::None) { ok = false; break; }
                if (nt == Tok::EndArray) break;
                bool d = false;
                ok = eval_path(e, g, st, k, &d);
                *dirty |= d;
                if (guard > p->n) { ok = false; }
            }
        } else {
            ok = copy_structure(p, g);
            *dirty = ok;
        }
    } else if (t == Tok::StartObject && ins[0].kind == Ins::Key) {
        for (uint64_t guard = 0;; ++guard) {
            const Tok nt = next(p);
            if (nt == Tok::Error || nt == Tok::None) { ok = false; break; }
            if (nt == Tok::EndObject) break;
            if (guard > p->n) { ok = false; break; }
            if (*dirty) { if (!skip_children(p)) { ok = false; break; } continue; }
            bool d = false;
            ok = eval_path(e, g, st, k + 1, &d);
            if (!ok) break;
            *dirty = d;
        }
    } else if (t == Tok::StartArray && rest >= 4 && ins[0].kind == Ins::Subscript && ins[1].kind == Ins::Wildcard &&
               ins[2].kind == Ins::Subscript && ins[3].kind == Ins::Wildcard) {
        gen_start(g, '[');
        for (uint64_t guard = 0; ok; ++guard) {
            const Tok nt = next(p);
            if (nt == Tok::Error || nt == Tok::None) { ok = false; break; }
            if (nt == Tok::EndArray) break;
            bool d = false;
            ok = eval_path(e, g, Style::Flatten, k + 4, &d);
            *dirty |= d;
            if (guard > p->n) ok = false;
        }
        gen_end(g, ']');
    } else if (t == Tok::StartArray && rest >= 2 && ins[0].kind == Ins::Subscript && ins[1].kind == Ins::Wildcard &&
               st != Style::Quoted) {
        const Style ns = st == Style::Raw ? Style::Quoted : Style::Flatten;
        Out* o = g->o;
        const bool comma = !g->after_name && g->depth > 0 && !g->first[g->depth];
        const uint32_t base = o->n;
        const uint32_t sb = base + (comma ? 1u : 0u);
        Out sub{o->p, o->cap, sb, o->over};
        Gen sg;
        gen_init(&sg, &sub);
        sg.chars = true;
        gen_start(&sg, '[');
        uint32_t matches = 0;
        for (uint64_t guard = 0; ok; ++guard) {
            const Tok nt = next(p);
            if (nt == Tok::Error || nt == Tok::None) { ok = false; break; }
            if (nt == Tok::EndArray) break;
            bool d = false;
            ok = eval_path(e, &sg, ns, k + 2, &d);
            matches += d ? 1u : 0u;
            if (guard > p->n) ok = false;
        }
        gen_end(&sg, ']');
        o->over = sub.over;
        if (ok && matches > 0 && !sub.over) {
            // [e1,e2,...]: all of it when > 1 match, the single element's text when exactly one.
            uint32_t end = sub.n;
            if (matches == 1) { std::memmove(o->p + sb, o->p + sb + 1, sub.n - sb - 2u); end = sub.n - 2u; }
            if (comma) o->p[base] = ',';
            o->n = end;
            if (g->after_name) g->after_name = false;
            else g->first[g->depth] = false;
            *dirty = true;
        } else if (ok && matches > 0) {
            o->n = sub.n;
            *dirty = true;
        } else {
            o->n = base;
        }
    } else if (t == Tok::StartArray && rest >= 2 && ins[0].kind == Ins::Subscript && ins[1].kind == Ins::Wildcard) {
        // Quoted style inside a wildcard: every element, as an array.
        if (!gen_start(g, '[')) return false;
        for (uint64_t guard = 0; ok; ++guard) {
            const Tok nt = next(p);
            if (nt == Tok::Error || nt == Tok::None) { ok = false; break; }
            if (nt == Tok::EndArray) break;
            bool d = false;
            ok = eval_path(e, g, st, k + 2, &d);
            *dirty |= d;
            if (guard > p->n) ok = false;
        }
        gen_end(g, ']');
    } else if (t == Tok::StartArray && rest >= 2 && ins[0].kind == Ins::Subscript && ins[1].kind == Ins::Index) {
        int64_t idx = ins[1].index;
        Tok nt = next(p);
        for (; ok && idx > 0 && nt != Tok::EndArray; --idx) {
            if (nt == Tok::Error || nt == Tok::None || !skip_children(p)) { ok = false; break; }
            nt = next(p);
        }
        if (ok && (nt == Tok::Error || nt == Tok::None)) ok = false;
        if (ok && nt != Tok::EndArray) {
            bool d = false;
            ok = eval_path(e, g, st, k + 2, &d);
            *dirty = d;
            for (uint64_t guard = 0; ok; ++guard) {                  // the rest of the array is read to its end
                const Tok rt = next(p);
                if (rt == Tok::Error || rt == Tok::None) { ok = false; break; }
                if (rt == Tok::EndArray) break;
                if (!skip_children(p)) { ok = false; break; }
                if (guard > p->n) ok = false;
            }
        }
    } else if (t == Tok::FieldName && ins[0].kind == Ins::Named && token_equals(*p, pa.names + ins[0].name_off, ins[0].name_len)) {
        const Tok nt = next(p);
        if (nt == Tok::Error || nt == Tok::None) ok = false;
        else if (nt != Tok::Null) ok = eval_path(e, g, st, k + 1, dirty);
    } else {
        ok = skip_children(p);
    }
    --e->rec;
    return ok;
}

}  // namespace detail

// get_json_object(json, path): 1 = matched (text in o), 0 = NULL, -1 = o overflowed (a budget).
inline int get_json_object(const uint8_t* json, uint32_t len, const SparkPath& path, Out* o) noexcept {
    assert(o != nullptr && o->n == 0);
    assert(path.n <= kMaxPathInstr);
    Parser ps;
    Parser* p = &ps;
    parser_init(p, json, len);
    Gen gs;
    Gen* g = &gs;
    gen_init(g, o);
    Eval e{p, &path, 0};
    bool dirty = false;
    bool ok = false;
    const Tok t = next(p);
    if (t != Tok::Error && t != Tok::None) ok = detail::eval_path(&e, g, Style::Raw, 0, &dirty);
    if (o->over) return -1;
    return ok && dirty ? 1 : 0;
}

// json_array_length: >= 0 the count, -1 NULL.
inline int64_t json_array_length(const uint8_t* json, uint32_t len) noexcept {
    assert(json != nullptr || len == 0);
    Parser ps;
    Parser* p = &ps;
    parser_init(p, json, len);
    int64_t cnt = -1;
    if (next(p) == Tok::StartArray) {
        cnt = 0;
        for (uint64_t guard = 0; guard <= uint64_t{len} + 1u; ++guard) {
            const Tok t = next(p);
            if (t == Tok::Error || t == Tok::None) { cnt = -1; break; }
            if (t == Tok::EndArray) break;
            if (!skip_children(p)) { cnt = -1; break; }
            ++cnt;
        }
    }
    return cnt;
}

// json_object_keys: the top-level names written as [u32 length][utf8]...; count in *keys. 1 ok, 0 NULL. A write past
// the buffer sets o->over (a counting Out, cap 0, measures the bytes: o->n).
inline int json_object_keys(const uint8_t* json, uint32_t len, Out* o, uint32_t* keys) noexcept {
    assert(o != nullptr && keys != nullptr);
    Parser ps;
    Parser* p = &ps;
    parser_init(p, json, len);
    *keys = 0;
    int rc = 0;
    if (next(p) == Tok::StartObject) {
        rc = 1;
        for (uint64_t guard = 0; guard <= uint64_t{len} + 1u; ++guard) {
            const Tok t = next(p);
            if (t == Tok::Error || t == Tok::None) { rc = 0; break; }
            if (t == Tok::EndObject) break;
            const uint32_t at = o->n;
            o->put("\0\0\0\0", 4);
            token_utf8(*p, o, false);
            const uint32_t klen = o->n - at - 4u;
            if (!o->over) std::memcpy(o->p + at, &klen, 4);
            ++*keys;
            const Tok v = next(p);
            if (v == Tok::Error || v == Tok::None || !skip_children(p)) { rc = 0; break; }
        }
    }
    return rc;
}

}  // namespace jsonspark
}  // namespace kernels
}  // namespace bolt
