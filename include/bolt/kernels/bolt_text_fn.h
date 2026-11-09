// bolt_text_fn.h — scalar text kernels for SQL / DataFrame string functions:
// pad, initcap, and a bounded java.util.Formatter subset (format_string).
//
// Semantics follow Spark (which follows Java): lengths count code points, a
// pad truncates a longer input to its first `target` code points, initcap
// lower-cases the string then upper-cases the first unit after each space.
//
// RULES: No exceptions, no heap, all fns noexcept, bounded loops. Outputs go
// to a caller buffer; a result that does not fit returns kTextOverflow.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "bolt/kernels/bolt_float_chars.h"
#include "bolt/kernels/bolt_utf8_cp.h"

namespace bolt {
namespace kernels {
namespace text {

constexpr int64_t kTextOverflow    = -1;   // output buffer too small
constexpr int64_t kTextBadFormat   = -2;   // malformed / unsupported spec
constexpr int64_t kTextBadArgument = -3;   // conversion does not fit the arg
constexpr uint32_t kFormatMaxWidth = 4096; // width / precision ceiling

// Byte offset just past the first `chars` code points of s[0..len).
inline uint32_t utf8_prefix_bytes(const char* s, uint32_t len, int64_t chars) noexcept {
    assert(s != nullptr || len == 0);
    uint32_t pos = 0;
    for (int64_t c = 0; c < chars && pos < len; ++c) pos += utf8::utf8_unit_len(s, len, pos);
    assert(pos <= len);
    return pos;
}

inline uint32_t utf8_chars(const char* s, uint32_t len) noexcept {
    uint32_t n = 0;
    for (uint32_t pos = 0; pos < len; ++n) pos += utf8::utf8_unit_len(s, len, pos);
    return n;
}

// lpad / rpad. target <= 0 gives "". An empty pad only truncates.
inline int64_t utf8_pad(const char* s, uint32_t slen, int64_t target, const char* pad,
                        uint32_t plen, bool left, char* out, uint32_t cap) noexcept {
    assert(s != nullptr || slen == 0);
    assert(pad != nullptr || plen == 0);
    if (target <= 0) return 0;
    const uint32_t have = utf8_chars(s, slen);
    if (static_cast<int64_t>(have) >= target || plen == 0) {
        const uint32_t keep = utf8_prefix_bytes(s, slen, target);
        if (keep > cap) return kTextOverflow;
        std::memcpy(out, s, keep);
        return keep;
    }
    const int64_t fill = target - have;
    uint32_t w = 0;
    if (!left) {
        if (slen > cap) return kTextOverflow;
        std::memcpy(out, s, slen);
        w = slen;
    }
    uint32_t pp = 0;
    for (int64_t c = 0; c < fill; ++c) {                     // bounded: target
        const uint32_t u = utf8::utf8_unit_len(pad, plen, pp);
        if (w + u > cap) return kTextOverflow;
        std::memcpy(out + w, pad + pp, u);
        w += u;
        pp += u;
        if (pp >= plen) pp = 0;
    }
    if (left) {
        if (w + slen > cap) return kTextOverflow;
        std::memcpy(out + w, s, slen);
        w += slen;
    }
    return w;
}

// initcap: dst capacity >= utf8_case_map_bound(len). Returns bytes written.
inline uint32_t utf8_initcap(const char* s, uint32_t len, char* dst) noexcept {
    assert(s != nullptr || len == 0);
    assert(dst != nullptr);
    uint32_t w = 0;
    bool start = true;
    for (uint32_t i = 0; i < len;) {                         // bounded: len
        uint32_t cp = 0;
        const uint32_t u = utf8::utf8_cp_decode(s, len, i, &cp);
        if (cp == utf8::kUtf8Invalid) {
            std::memcpy(dst + w, s + i, u);
            w += u;
        } else {
            w += utf8::utf8_cp_encode(start ? utf8::utf8_cp_upper(cp) : utf8::utf8_cp_lower(cp), dst + w);
        }
        start = (cp == ' ');
        i += u;
    }
    assert(w <= utf8::utf8_case_map_bound(len));
    return w;
}

// ---- format_string ---------------------------------------------------------

enum class FormatKind : uint8_t { Null = 0, Int = 1, Float = 2, Str = 3, Bool = 4 };

struct FormatArg {
    FormatKind  kind;
    int64_t     i;
    double      d;
    const char* s;
    uint32_t    len;
};

struct FormatSpec {
    int32_t arg_index;     // 1-based explicit index, 0 = next
    bool    left;          // '-'
    bool    zero;          // '0'
    bool    plus;          // '+'
    bool    space;         // ' '
    bool    group;         // ','
    int32_t width;         // -1 = none
    int32_t precision;     // -1 = none
    char    conv;
};

struct FormatOut {
    char*    p;
    uint32_t cap;
    uint32_t n;
    bool put(const char* src, uint32_t len) noexcept {
        if (len > cap - n) return false;
        std::memcpy(p + n, src, len);
        n += len;
        return true;
    }
    bool fill(char c, int64_t count) noexcept {
        for (int64_t k = 0; k < count; ++k) {
            if (n >= cap) return false;
            p[n++] = c;
        }
        return true;
    }
};

namespace detail {
inline int32_t read_int(const char* f, uint32_t len, uint32_t* i) noexcept {
    int64_t v = 0;
    uint32_t start = *i;
    while (*i < len && f[*i] >= '0' && f[*i] <= '9' && *i - start < 6) v = v * 10 + (f[(*i)++] - '0');
    return *i == start ? -1 : static_cast<int32_t>(v);
}

inline bool parse_spec(const char* f, uint32_t len, uint32_t* i, FormatSpec* s) noexcept {
    assert(f != nullptr && i != nullptr && s != nullptr);
    *s = FormatSpec{0, false, false, false, false, false, -1, -1, 0};
    const uint32_t save = *i;
    const int32_t n = read_int(f, len, i);
    if (n > 0 && *i < len && f[*i] == '$') { s->arg_index = n; ++*i; }
    else *i = save;
    for (uint32_t guard = 0; *i < len && guard < 8; ++guard) {
        const char c = f[*i];
        if (c == '-') s->left = true;
        else if (c == '0') s->zero = true;
        else if (c == '+') s->plus = true;
        else if (c == ' ') s->space = true;
        else if (c == ',') s->group = true;
        else break;
        ++*i;
    }
    s->width = read_int(f, len, i);
    if (*i < len && f[*i] == '.') {
        ++*i;
        s->precision = read_int(f, len, i);
        if (s->precision < 0) return false;
    }
    if (*i >= len) return false;
    s->conv = f[(*i)++];
    return s->width <= static_cast<int32_t>(kFormatMaxWidth) &&
           s->precision <= static_cast<int32_t>(kFormatMaxWidth);
}

inline bool emit_padded(FormatOut* o, const FormatSpec& s, const char* body, uint32_t blen,
                        uint32_t chars, uint32_t sign_len) noexcept {
    const int64_t pad = s.width > static_cast<int32_t>(chars) ? s.width - chars : 0;
    if (s.left) return o->put(body, blen) && o->fill(' ', pad);
    if (s.zero) return o->put(body, sign_len) && o->fill('0', pad) && o->put(body + sign_len, blen - sign_len);
    return o->fill(' ', pad) && o->put(body, blen);
}

inline uint32_t group_digits(const char* in, uint32_t len, char* out) noexcept {
    uint32_t w = 0, lead = (len > 0 && in[0] == '-') ? 1u : 0u;
    for (uint32_t k = 0; k < lead; ++k) out[w++] = in[k];
    const uint32_t digits = len - lead;
    for (uint32_t k = 0; k < digits; ++k) {
        if (k != 0 && (digits - k) % 3 == 0) out[w++] = ',';
        out[w++] = in[lead + k];
    }
    return w;
}

inline int64_t emit_int(FormatOut* o, const FormatSpec& s, const FormatArg& a) noexcept {
    if (a.kind != FormatKind::Int) return kTextBadArgument;
    char raw[32], buf[48];
    int n = 0;
    if (s.conv == 'x' || s.conv == 'X') {
        n = std::snprintf(raw, sizeof(raw), s.conv == 'x' ? "%llx" : "%llX",
                          static_cast<unsigned long long>(a.i));
    } else if (s.conv == 'o') {
        n = std::snprintf(raw, sizeof(raw), "%llo", static_cast<unsigned long long>(a.i));
    } else {
        n = std::snprintf(raw, sizeof(raw), "%lld", static_cast<long long>(a.i));
    }
    if (n <= 0) return kTextBadArgument;
    uint32_t len = static_cast<uint32_t>(n);
    if (s.group && s.conv == 'd') len = group_digits(raw, len, buf);
    else std::memcpy(buf, raw, len);
    char body[64];
    uint32_t bl = 0;
    if (s.conv == 'd' && a.i >= 0 && (s.plus || s.space)) body[bl++] = s.plus ? '+' : ' ';
    std::memcpy(body + bl, buf, len);
    bl += len;
    const uint32_t sign = (body[0] == '-' || body[0] == '+' || body[0] == ' ') ? 1u : 0u;
    return emit_padded(o, s, body, bl, bl, sign) ? 0 : kTextOverflow;
}

// Java's %.Nf: the double's shortest round-trip digits, rounded HALF_UP to
// `prec` fraction digits (java.util.Formatter), not the binary value's
// nearest-even rounding printf applies. Returns bytes, or 0 when out is short.
inline uint32_t fixed_half_up(double v, int32_t prec, char* out, uint32_t cap) noexcept {
    assert(out != nullptr && prec >= 0);
    char d[40] = {};
    int32_t k = 0;
    const bool neg = v < 0.0;
    const double a = neg ? -v : v;
    uint32_t nd = 0;
    if (a == 0.0) { d[0] = '0'; nd = 1; k = 1; }
    else nd = float_chars::detail::shortest_digits(a, d, &k);   // a = 0.d1..dn * 10^k
    // digit at position p (p = 0 is the first integer digit of 10^(k-1)).
    const int32_t int_digits = k > 0 ? k : 0;
    const int32_t total = int_digits + prec;              // digits kept
    char kept[kFormatMaxWidth + 400];
    if (static_cast<uint32_t>(total) + 2u > sizeof(kept)) return 0;
    auto digit_at = [&](int32_t i) -> int {               // i-th kept digit, i in [0, total]
        const int32_t src = i - (int_digits - k);         // index into d[]
        return (src >= 0 && src < static_cast<int32_t>(nd)) ? d[src] - '0' : 0;
    };
    for (int32_t i = 0; i < total; ++i) kept[i] = static_cast<char>('0' + digit_at(i));
    bool carry = digit_at(total) >= 5;
    for (int32_t i = total - 1; carry && i >= 0; --i) {
        if (kept[i] == '9') { kept[i] = '0'; } else { ++kept[i]; carry = false; }
    }
    uint32_t w = 0;
    auto put = [&](char c) { if (w < cap) out[w] = c; ++w; };
    if (neg) put('-');
    if (carry) put('1');
    if (int_digits == 0 && !carry) put('0');
    for (int32_t i = 0; i < int_digits; ++i) put(kept[i]);
    if (prec > 0) put('.');
    for (int32_t i = int_digits; i < total; ++i) put(kept[i]);
    return w <= cap ? w : 0;
}

inline int64_t emit_float(FormatOut* o, const FormatSpec& s, const FormatArg& a) noexcept {
    double v = 0;
    if (a.kind == FormatKind::Float) v = a.d;
    else return kTextBadArgument;
    const int prec = s.precision < 0 ? 6 : s.precision;
    char spec[8];
    uint32_t k = 0;
    spec[k++] = '%';
    if (s.plus) spec[k++] = '+';
    spec[k++] = '.';
    spec[k++] = '*';
    spec[k++] = (s.conv == 'e' || s.conv == 'E') ? s.conv : 'f';
    spec[k] = 0;
    char raw[kFormatMaxWidth + 400];
    int n = 0;
    if (v != v) { n = 3; std::memcpy(raw, "NaN", 3); }
    else if (v == 1.0 / 0.0 || v == -1.0 / 0.0) {
        n = v > 0 ? 8 : 9;
        std::memcpy(raw, v > 0 ? "Infinity" : "-Infinity", static_cast<size_t>(n));
    } else if (s.conv == 'f') {
        uint32_t off = 0;
        if (s.plus && v >= 0.0) raw[off++] = '+';
        const uint32_t w = fixed_half_up(v, prec, raw + off, sizeof(raw) - off);
        if (w == 0) return kTextOverflow;
        n = static_cast<int>(w + off);
    } else {
        n = std::snprintf(raw, sizeof(raw), spec, prec, v);
    }
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(raw)) return kTextOverflow;
    const uint32_t sign = (raw[0] == '-' || raw[0] == '+') ? 1u : 0u;
    return emit_padded(o, s, raw, static_cast<uint32_t>(n), static_cast<uint32_t>(n), sign) ? 0 : kTextOverflow;
}

inline int64_t emit_text(FormatOut* o, const FormatSpec& s, const FormatArg& a) noexcept {
    char tmp[64];
    const char* body = tmp;
    uint32_t bl = 0;
    switch (a.kind) {
        case FormatKind::Null:  body = "null"; bl = 4; break;
        case FormatKind::Str:   body = a.s; bl = a.len; break;
        case FormatKind::Bool:  body = a.i ? "true" : "false"; bl = a.i ? 4u : 5u; break;
        case FormatKind::Int:   bl = static_cast<uint32_t>(std::snprintf(tmp, sizeof(tmp), "%lld",
                                                                         static_cast<long long>(a.i))); break;
        case FormatKind::Float: bl = static_cast<uint32_t>(std::snprintf(tmp, sizeof(tmp), "%.17g", a.d)); break;
    }
    if (s.conv == 'b' || s.conv == 'B') {
        const bool t = a.kind == FormatKind::Null ? false : a.kind == FormatKind::Bool ? a.i != 0 : true;
        body = t ? "true" : "false"; bl = t ? 4u : 5u;
    }
    if (s.precision >= 0) bl = utf8_prefix_bytes(body, bl, s.precision);
    char up[kFormatMaxWidth];
    if ((s.conv == 'S' || s.conv == 'B') && bl <= kFormatMaxWidth / 2) {
        bl = utf8::utf8_case_map(body, bl, true, up);
        body = up;
    }
    return emit_padded(o, s, body, bl, utf8_chars(body, bl), 0) ? 0 : kTextOverflow;
}
}  // namespace detail

// java.util.Formatter subset: %[n$][-0+ ,][width][.prec](s S b B d x X o f e E %% n).
// A null argument prints "null" for every conversion, as Java does. Returns
// bytes written, kTextOverflow, kTextBadFormat or kTextBadArgument.
inline int64_t text_format(const char* fmt, uint32_t flen, const FormatArg* args, uint32_t nargs,
                           char* out, uint32_t cap) noexcept {
    assert(fmt != nullptr || flen == 0);
    assert(args != nullptr || nargs == 0);
    FormatOut o{out, cap, 0};
    uint32_t next = 0;
    for (uint32_t i = 0; i < flen;) {                        // bounded: flen
        if (fmt[i] != '%') {
            uint32_t j = i;
            while (j < flen && fmt[j] != '%') ++j;
            if (!o.put(fmt + i, j - i)) return kTextOverflow;
            i = j;
            continue;
        }
        ++i;
        FormatSpec s;
        if (!detail::parse_spec(fmt, flen, &i, &s)) return kTextBadFormat;
        if (s.conv == '%') { if (!o.put("%", 1)) return kTextOverflow; continue; }
        if (s.conv == 'n') { if (!o.put("\n", 1)) return kTextOverflow; continue; }
        const uint32_t idx = s.arg_index > 0 ? static_cast<uint32_t>(s.arg_index - 1) : next++;
        if (idx >= nargs) return kTextBadFormat;
        const FormatArg& a = args[idx];
        int64_t r = 0;
        switch (s.conv) {
            case 's': case 'S': case 'b': case 'B': r = detail::emit_text(&o, s, a); break;
            case 'd': case 'x': case 'X': case 'o':
                r = a.kind == FormatKind::Null ? detail::emit_text(&o, s, a) : detail::emit_int(&o, s, a); break;
            case 'f': case 'e': case 'E':
                r = a.kind == FormatKind::Null ? detail::emit_text(&o, s, a) : detail::emit_float(&o, s, a); break;
            default: return kTextBadFormat;
        }
        if (r < 0) return r;
    }
    assert(o.n <= cap);
    return o.n;
}

}  // namespace text
}  // namespace kernels
}  // namespace bolt
