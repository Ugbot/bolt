// bolt_url.h — URL kernels for Spark's url_encode / url_decode / parse_url.
//
//  * url_encode / url_decode follow java.net.URLEncoder / URLDecoder
//    (application/x-www-form-urlencoded, UTF-8): space <-> '+', the
//    unreserved set is A-Z a-z 0-9 . - * _, everything else is %XX per byte,
//    uppercase hex. A decoded run of %XX bytes is read as UTF-8 with each
//    malformed unit replaced by one U+FFFD exactly as the JDK's String(bytes, UTF_8) does.
//    A '%' not followed by two valid characters is an error (kTextInvalid).
//
//  * java_uri_parse is a port of java.net.URI's parser (RFC 2396 with the JDK's
//    deviations: IPv6 literals, empty authority, non-ASCII "other" characters):
//    which strings are URIs, and what scheme / authority / userinfo / host /
//    path / query / fragment they split into. A server-based authority that
//    does not parse as host[:port] falls back to a registry authority (host
//    and userinfo absent), as the JDK does.
//
// RULES: No exceptions, no heap, all fns noexcept, bounded loops, ranges not copies.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/kernels/bolt_text_fn.h"
#include "bolt/kernels/bolt_utf8_cp.h"

namespace bolt {
namespace kernels {
namespace url {

using text::kTextInvalid;
using text::kTextOverflow;

constexpr int64_t kUrlUnsupported = -5;     // a form this port does not implement

// ---- url_encode / url_decode ----------------------------------------------

constexpr uint64_t url_encode_bound(uint32_t len) noexcept { return static_cast<uint64_t>(len) * 3u; }

inline bool encode_keeps(uint8_t b) noexcept {
    return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9') ||
           b == '.' || b == '-' || b == '*' || b == '_';
}

inline int64_t url_encode(const char* s, uint32_t len, char* out, uint32_t cap) noexcept {
    assert(s != nullptr || len == 0);
    assert(out != nullptr || len == 0);
    static const char kHex[] = "0123456789ABCDEF";
    uint64_t w = 0;
    for (uint32_t i = 0; i < len; ++i) {                         // bounded: len
        const uint8_t b = static_cast<uint8_t>(s[i]);
        if (encode_keeps(b)) { if (w + 1 > cap) return kTextOverflow; out[w++] = static_cast<char>(b); }
        else if (b == ' ') { if (w + 1 > cap) return kTextOverflow; out[w++] = '+'; }
        else {
            if (w + 3 > cap) return kTextOverflow;
            out[w++] = '%'; out[w++] = kHex[b >> 4]; out[w++] = kHex[b & 15u];
        }
    }
    return static_cast<int64_t>(w);
}

namespace detail {
inline int hexval(uint8_t c) noexcept {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
// Integer.parseInt(two chars, 16) as URLDecoder uses it: [sign] hex digits; -1 = rejected.
inline int pct_byte(uint8_t c1, uint8_t c2) noexcept {
    if (c1 == '+' || c1 == '-') {
        const int h = hexval(c2);
        return h < 0 ? -1 : h;                                   // "-N" parses negative, "-0" is 0
    }
    const int a = hexval(c1), b = hexval(c2);
    return (a < 0 || b < 0) ? -1 : a * 16 + b;
}
inline bool pct_negative(uint8_t c1, uint8_t c2) noexcept { return c1 == '-' && hexval(c2) > 0; }
}  // namespace detail

namespace detail {
// java.lang.String(bytes, UTF_8) as the JDK's String.decodeUTF8 runs it with replacement: how
// many bytes the unit at p takes, and whether they are a valid character (copied) or one U+FFFD.
// A port of its branches, not the Unicode "maximal subpart" rule: an encoded surrogate is ONE
// replacement for its three bytes, and a truncated sequence at the end of the input can swallow
// all that is left.
inline bool j_cont(uint8_t b) noexcept { return (b & 0xC0u) == 0x80u; }
inline bool j_bad3_2(uint8_t b1, uint8_t b2) noexcept { return (b1 == 0xE0u && (b2 & 0xE0u) == 0x80u) || !j_cont(b2); }
inline bool j_bad4_2(uint8_t b1, uint8_t b2) noexcept {
    return (b1 == 0xF0u && (b2 < 0x90u || b2 > 0xBFu)) || (b1 == 0xF4u && (b2 & 0xF0u) != 0x80u) || !j_cont(b2);
}
inline uint32_t java_utf8_step(const uint8_t* s, uint32_t len, uint32_t p, bool* valid) noexcept {
    assert(p < len);
    const uint8_t b1 = s[p];
    const uint32_t rem = len - p;
    *valid = false;
    if (b1 < 0x80u) { *valid = true; return 1; }
    if (b1 >= 0xC2u && b1 <= 0xDFu) {
        if (rem < 2) return rem;                                 // truncated at the end: one U+FFFD
        if (!j_cont(s[p + 1])) return 1;
        *valid = true;
        return 2;
    }
    if (b1 >= 0xE0u && b1 <= 0xEFu) {
        if (rem >= 3) {
            const uint8_t b2 = s[p + 1], b3 = s[p + 2];
            if (j_bad3_2(b1, b2) || !j_cont(b3)) return j_bad3_2(b1, b2) ? 1u : 2u;
            if (b1 == 0xEDu && b2 >= 0xA0u) return 3;            // an encoded surrogate: one U+FFFD
            *valid = true;
            return 3;
        }
        if (rem == 2 && j_bad3_2(b1, s[p + 1])) return 1;
        return rem;
    }
    if (b1 >= 0xF0u && b1 <= 0xF7u) {
        if (rem >= 4) {
            const uint8_t b2 = s[p + 1], b3 = s[p + 2], b4 = s[p + 3];
            const uint32_t uc = (static_cast<uint32_t>(b1 & 7u) << 18) | (static_cast<uint32_t>(b2 & 0x3Fu) << 12) |
                                (static_cast<uint32_t>(b3 & 0x3Fu) << 6) | (b4 & 0x3Fu);
            if (!j_cont(b2) || !j_cont(b3) || !j_cont(b4) || uc < 0x10000u || uc > 0x10FFFFu) {
                if (b1 > 0xF4u || j_bad4_2(b1, b2)) return 1;
                return j_cont(b3) ? 3u : 2u;
            }
            *valid = true;
            return 4;
        }
        if (b1 > 0xF4u || (rem > 1 && j_bad4_2(b1, s[p + 1]))) return 1;
        if (rem > 2 && !j_cont(s[p + 2])) return 2;              // the lead and its second byte, one U+FFFD
        return rem;
    }
    return 1;                                                    // 80..C1, F8..FF
}
}  // namespace detail

// url_decode into out (cap >= len). Returns bytes, kTextInvalid for a malformed escape.
inline int64_t url_decode(const char* s, uint32_t len, char* out, uint32_t cap) noexcept {
    assert(s != nullptr || len == 0);
    assert(out != nullptr || len == 0);
    if (cap < len) return kTextOverflow;
    uint32_t w = 0;
    for (uint32_t i = 0; i < len;) {                             // bounded: len
        const uint8_t c = static_cast<uint8_t>(s[i]);
        if (c == '+') { out[w++] = ' '; ++i; continue; }
        if (c != '%') { out[w++] = static_cast<char>(c); ++i; continue; }
        // A run of %XX: stage the raw bytes at the tail of the buffer, then repair into place.
        uint32_t rb = 0, j = i;
        while (j < len && s[j] == '%') {                         // bounded: len
            if (j + 2 >= len) return kTextInvalid;                // incomplete trailing escape
            const uint8_t c1 = static_cast<uint8_t>(s[j + 1]), c2 = static_cast<uint8_t>(s[j + 2]);
            const int v = detail::pct_byte(c1, c2);
            if (v < 0 || detail::pct_negative(c1, c2)) return kTextInvalid;
            ++rb;
            j += 3;
        }
        char* raw = out + (cap - rb);
        uint32_t k = 0;
        for (uint32_t p = i; p < j; p += 3) {
            raw[k++] = static_cast<char>(detail::pct_byte(static_cast<uint8_t>(s[p + 1]),
                                                           static_cast<uint8_t>(s[p + 2])));
        }
        for (uint32_t p = 0; p < rb;) {                          // bounded: rb
            bool valid = false;
            const uint32_t take = detail::java_utf8_step(reinterpret_cast<const uint8_t*>(raw), rb, p, &valid);
            if (valid) { for (uint32_t q = 0; q < take; ++q) out[w + q] = raw[p + q]; w += take; }
            else { out[w++] = static_cast<char>(0xEF); out[w++] = static_cast<char>(0xBF); out[w++] = static_cast<char>(0xBD); }
            p += take;
            assert(w <= cap - rb + p);       // output stays behind the unread staged bytes (w + 3*rb <= len)
        }
        i = j;
    }
    return w;
}

// ---- java.net.URI ----------------------------------------------------------

struct Span { uint32_t off = 0; uint32_t len = 0; bool has = false; };

struct UriParts {
    Span scheme, authority, userinfo, host, path, query, fragment;
};

namespace detail {

// Class membership of one ASCII character, by the JDK's masks.
enum Cls : uint8_t { kAlpha, kDigit, kAlnum, kHex, kScheme, kUric, kPath, kUserinfo, kRegName,
                     kServer, kServerPercent, kAlnumDash, kDigitDot };

inline bool in_set(const char* set, char c) noexcept {
    for (; *set != '\0'; ++set) if (*set == c) return true;
    return false;
}
inline bool is_alnum(char c) noexcept {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}
inline bool is_unreserved(char c) noexcept { return is_alnum(c) || in_set("-_.!~*'()", c); }

inline bool ascii_in(Cls cls, char c) noexcept {
    switch (cls) {
        case kAlpha: return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        case kDigit: return c >= '0' && c <= '9';
        case kAlnum: return is_alnum(c);
        case kHex: return hexval(static_cast<uint8_t>(c)) >= 0;
        case kScheme: return is_alnum(c) || in_set("+-.", c);
        case kUric: return is_unreserved(c) || in_set(";/?:@&=+$,[]", c);
        case kPath: return is_unreserved(c) || in_set(":@&=+$,;/", c);
        case kUserinfo: return is_unreserved(c) || in_set(";:&=+$,", c);
        case kRegName: return is_unreserved(c) || in_set("$,;:@&=+", c);
        case kServer: return is_unreserved(c) || in_set(";:&=+$,.:@[]", c);
        case kServerPercent: return is_unreserved(c) || in_set(";:&=+$,.:@[]%", c);
        case kAlnumDash: return is_alnum(c) || c == '-';
        case kDigitDot: return (c >= '0' && c <= '9') || c == '.';
    }
    return false;
}
// Classes that admit %XX escapes and unescaped non-ASCII "other" characters.
inline bool admits_escaped(Cls cls) noexcept {
    return cls == kUric || cls == kPath || cls == kUserinfo || cls == kRegName || cls == kServer;
}
inline bool is_space_or_control(uint32_t cp) noexcept {
    if (cp <= 0x1Fu || (cp >= 0x7Fu && cp <= 0x9Fu)) return true;                     // ISO control
    return cp == 0x20u || cp == 0xA0u || cp == 0x1680u || (cp >= 0x2000u && cp <= 0x200Au) ||
           cp == 0x2028u || cp == 0x2029u || cp == 0x202Fu || cp == 0x205Fu || cp == 0x3000u;
}

struct Parser {
    const char* s;
    uint32_t n;
    UriParts out;
    int ipv6_bytes = 0;

    bool at(uint32_t p, char c) const noexcept { return p < n && s[p] == c; }

    // JDK scan(start, n, mask): index of the first character not in the class. A malformed
    // %XX escape in an escape-admitting class is a parse failure (*bad = true).
    uint32_t scan(uint32_t start, uint32_t end, Cls cls, bool* bad) const noexcept {
        uint32_t p = start;
        while (p < end) {                                        // bounded: end
            const uint8_t c = static_cast<uint8_t>(s[p]);
            if (c < 0x80u && ascii_in(cls, static_cast<char>(c))) { ++p; continue; }
            if (admits_escaped(cls)) {
                if (c == '%') {
                    if (p + 3 <= end && hexval(static_cast<uint8_t>(s[p + 1])) >= 0 &&
                        hexval(static_cast<uint8_t>(s[p + 2])) >= 0) { p += 3; continue; }
                    *bad = true;
                    return p;
                }
                if (c >= 0x80u) {
                    uint32_t cp = 0;
                    const uint32_t u = utf8::utf8_cp_decode(s, end, p, &cp);
                    if (cp == utf8::kUtf8Invalid) cp = 0xFFFDu;
                    if (cp > 0x80u && !is_space_or_control(cp)) { p += u; continue; }
                }
            }
            break;
        }
        return p;
    }
    // scan(start, n, stopset): index of the first stop character, or n.
    uint32_t scan_to(uint32_t start, uint32_t end, const char* stop) const noexcept {
        uint32_t p = start;
        while (p < end && !in_set(stop, s[p])) ++p;
        return p;
    }
    // scan(start, n, errset, stopset): -1 on an err character.
    int64_t scan_err(uint32_t start, uint32_t end, const char* err, const char* stop) const noexcept {
        uint32_t p = start;
        while (p < end) {
            if (in_set(err, s[p])) return -1;
            if (in_set(stop, s[p])) break;
            ++p;
        }
        return p;
    }
    // All of [a, b) in `cls`?
    bool all_in(uint32_t a, uint32_t b, Cls cls, bool* bad) const noexcept {
        return scan(a, b, cls, bad) == b;
    }
};

inline void set_span(Span* sp, uint32_t a, uint32_t b) noexcept { sp->off = a; sp->len = b - a; sp->has = true; }

// ---- IPv4 / IPv6 / hostname ----

inline uint32_t scan_byte(const Parser& ps, uint32_t p, uint32_t m) noexcept {
    uint32_t q = p;
    while (q < m && ps.s[q] >= '0' && ps.s[q] <= '9') ++q;
    if (q <= p) return q;
    uint64_t v = 0;
    for (uint32_t k = p; k < q && v < 100000u; ++k) v = v * 10u + static_cast<uint32_t>(ps.s[k] - '0');
    return v > 255u ? p : q;
}

// Parses d.d.d.d at [start, n); returns the end, or -1 when malformed (strict: must end at n).
inline int64_t scan_ipv4(const Parser& ps, uint32_t start, uint32_t n, bool strict) noexcept {
    bool bad = false;
    const uint32_t m = ps.scan(start, n, kDigitDot, &bad);
    if (m <= start || (strict && m != n)) return -1;
    uint32_t p = start;
    for (int part = 0; part < 4; ++part) {
        const uint32_t q = scan_byte(ps, p, m);
        if (q <= p) return -1;
        p = q;
        if (part == 3) break;
        if (p >= m || ps.s[p] != '.') return -1;
        ++p;
    }
    return p < m ? -1 : static_cast<int64_t>(p);
}

inline uint32_t hex_run(const Parser& ps, uint32_t p, uint32_t n) noexcept {
    uint32_t q = p;
    while (q < n && hexval(static_cast<uint8_t>(ps.s[q])) >= 0) ++q;
    return q;
}

// scanHexSeq: returns the end (>= start) or -1 when no group starts here / an IPv4 begins.
inline int64_t scan_hex_seq(Parser* ps, uint32_t start, uint32_t n, bool* fail) noexcept {
    uint32_t p = start;
    uint32_t q = hex_run(*ps, p, n);
    if (q <= p) return -1;
    if (ps->at(q, '.')) return -1;
    if (q > p + 4) { *fail = true; return -1; }
    ps->ipv6_bytes += 2;
    p = q;
    while (p < n) {                                              // bounded: n
        if (!ps->at(p, ':')) break;
        if (ps->at(p + 1, ':')) break;
        ++p;
        q = hex_run(*ps, p, n);
        if (q <= p) { *fail = true; return -1; }
        if (ps->at(q, '.')) { --p; break; }
        if (q > p + 4) { *fail = true; return -1; }
        ps->ipv6_bytes += 2;
        p = q;
    }
    return p;
}

inline bool take_ipv4(Parser* ps, uint32_t p, uint32_t n, uint32_t* end) noexcept {
    const int64_t q = scan_ipv4(*ps, p, n, true);
    if (q <= static_cast<int64_t>(p)) return false;
    *end = static_cast<uint32_t>(q);
    ps->ipv6_bytes += 4;
    return true;
}

inline bool scan_hex_post(Parser* ps, uint32_t start, uint32_t n, uint32_t* end) noexcept {
    uint32_t p = start;
    if (p == n) { *end = p; return true; }
    bool fail = false;
    const int64_t q = scan_hex_seq(ps, p, n, &fail);
    if (fail) return false;
    if (q > static_cast<int64_t>(p)) {
        p = static_cast<uint32_t>(q);
        if (ps->at(p, ':')) { if (!take_ipv4(ps, p + 1, n, &p)) return false; }
    } else if (!take_ipv4(ps, p, n, &p)) {
        return false;
    }
    *end = p;
    return true;
}

// parseIPv6Reference over [start, n): true when the whole range is a valid IPv6 address.
inline bool parse_ipv6(Parser* ps, uint32_t start, uint32_t n) noexcept {
    ps->ipv6_bytes = 0;
    uint32_t p = start;
    bool compressed = false, fail = false;
    const int64_t q = scan_hex_seq(ps, p, n, &fail);
    if (fail) return false;
    if (q > static_cast<int64_t>(p)) {
        p = static_cast<uint32_t>(q);
        if (ps->at(p, ':') && ps->at(p + 1, ':')) { compressed = true; if (!scan_hex_post(ps, p + 2, n, &p)) return false; }
        else if (ps->at(p, ':')) { if (!take_ipv4(ps, p + 1, n, &p)) return false; }
    } else if (ps->at(p, ':') && ps->at(p + 1, ':')) {
        compressed = true;
        if (!scan_hex_post(ps, p + 2, n, &p)) return false;
    }
    if (p < n) return false;
    if (ps->ipv6_bytes > 16) return false;
    if (!compressed && ps->ipv6_bytes < 16) return false;
    if (compressed && ps->ipv6_bytes == 16) return false;
    return true;
}

// parseHostname: returns the end of the host name, or -1.
inline int64_t parse_hostname(const Parser& ps, uint32_t start, uint32_t n) noexcept {
    bool bad = false;
    uint32_t p = start;
    int64_t last = -1;
    for (uint32_t guard = 0; guard <= n; ++guard) {              // bounded: labels
        uint32_t q = ps.scan(p, n, kAlnum, &bad);
        if (q <= p) break;
        last = p;
        p = q;
        q = ps.scan(p, n, kAlnumDash, &bad);
        if (q > p) {
            if (ps.s[q - 1] == '-') return -1;
            p = q;
        }
        if (!ps.at(p, '.')) break;
        ++p;
        if (p >= n) break;
    }
    if (p < n && !ps.at(p, ':')) return -1;
    if (last < 0) return -1;
    if (last > static_cast<int64_t>(start) && !ascii_in(kAlpha, ps.s[last])) return -1;
    return p;
}

// parseServer: [userinfo@]host[:port] over [start, n). On success fills userinfo / host and
// returns true; the whole range must be consumed.
inline bool parse_server(Parser* ps, uint32_t start, uint32_t n) noexcept {
    bool bad = false;
    uint32_t p = start;
    int64_t q = ps->scan_err(p, n, "/?#", "@");
    if (q >= static_cast<int64_t>(p) && ps->at(static_cast<uint32_t>(q), '@')) {
        if (!ps->all_in(p, static_cast<uint32_t>(q), kUserinfo, &bad) || bad) return false;
        set_span(&ps->out.userinfo, p, static_cast<uint32_t>(q));
        p = static_cast<uint32_t>(q) + 1u;
    }
    if (ps->at(p, '[')) {
        ++p;
        q = ps->scan_err(p, n, "/?#", "]");
        if (q > static_cast<int64_t>(p) && ps->at(static_cast<uint32_t>(q), ']')) {
            const uint32_t qq = static_cast<uint32_t>(q);
            const uint32_t r = ps->scan_to(p, qq, "%");
            if (r < qq && r > p) {                               // a %scope id
                if (!parse_ipv6(ps, p, r)) return false;
                if (r + 1 == qq) return false;
                if (!ps->all_in(r + 1, qq, kAlnum, &bad)) return false;
            } else if (!parse_ipv6(ps, p, qq)) {
                return false;
            }
            set_span(&ps->out.host, p - 1u, qq + 1u);
            p = qq + 1u;
        } else {
            return false;
        }
    } else {
        int64_t e = scan_ipv4(*ps, p, n, false);
        if (e > static_cast<int64_t>(p) && static_cast<uint32_t>(e) < n && ps->s[e] != ':') e = -1;
        if (e > static_cast<int64_t>(p)) {
            set_span(&ps->out.host, p, static_cast<uint32_t>(e));
            p = static_cast<uint32_t>(e);
        } else {
            e = parse_hostname(*ps, p, n);
            if (e < 0) return false;
            set_span(&ps->out.host, p, static_cast<uint32_t>(e));
            p = static_cast<uint32_t>(e);
        }
    }
    if (ps->at(p, ':')) {
        ++p;
        if (p < n) {
            const uint32_t qq = ps->scan_to(p, n, "/");
            if (qq > p) {
                uint64_t port = 0;
                for (uint32_t k = p; k < qq; ++k) {
                    if (ps->s[k] < '0' || ps->s[k] > '9') return false;
                    port = port * 10u + static_cast<uint32_t>(ps->s[k] - '0');
                    if (port > 2147483647u) return false;
                }
                p = qq;
            }
        }
    }
    return p >= n;
}

// parseAuthority over [start, n): false when the URI is invalid.
inline bool parse_authority(Parser* ps, uint32_t start, uint32_t n) noexcept {
    bool bad = false;
    const uint32_t p = start;
    const bool has_bracket = ps->scan_to(p, n, "]") < n;
    const bool server_chars = ps->scan(p, n, has_bracket ? kServerPercent : kServer, &bad) == n && !bad;
    bad = false;
    const bool reg_chars = ps->scan(p, n, kRegName, &bad) == n;
    if (bad) return false;                                       // malformed escape pair
    if (reg_chars && !server_chars) { set_span(&ps->out.authority, p, n); return true; }
    bool server_ok = false;
    if (server_chars) {
        server_ok = parse_server(ps, p, n);
        if (server_ok) set_span(&ps->out.authority, p, n);
        else { ps->out.userinfo = Span{}; ps->out.host = Span{}; }
    }
    if (server_ok) return true;
    if (reg_chars) { set_span(&ps->out.authority, p, n); return true; }
    return false;
}

inline bool parse_hierarchical(Parser* ps, uint32_t start, uint32_t* end) noexcept {
    bool bad = false;
    uint32_t p = start;
    const uint32_t n = ps->n;
    if (ps->at(p, '/') && ps->at(p + 1, '/')) {
        p += 2;
        const uint32_t q = ps->scan_to(p, n, "/?#");
        if (q > p) { if (!parse_authority(ps, p, q)) return false; p = q; }
        else if (q >= n) return false;                           // "scheme://" with nothing after
    }
    uint32_t q = ps->scan_to(p, n, "?#");
    if (!ps->all_in(p, q, kPath, &bad) || bad) return false;
    set_span(&ps->out.path, p, q);
    p = q;
    if (ps->at(p, '?')) {
        ++p;
        q = ps->scan_to(p, n, "#");
        if (!ps->all_in(p, q, kUric, &bad) || bad) return false;
        set_span(&ps->out.query, p, q);
        p = q;
    }
    *end = p;
    return true;
}

}  // namespace detail

// Parses `s` as java.net.URI(String) does. Returns false when the JDK throws
// URISyntaxException (an invalid URL). Spans index into s.
inline bool java_uri_parse(const char* s, uint32_t n, UriParts* out) noexcept {
    assert(out != nullptr && (s != nullptr || n == 0));
    detail::Parser ps{s, n, UriParts{}, 0};
    bool bad = false;
    uint32_t p = 0;
    const int64_t c = ps.scan_err(0, n, "/?#", ":");
    if (c >= 0 && ps.at(static_cast<uint32_t>(c), ':')) {
        const uint32_t colon = static_cast<uint32_t>(c);
        if (colon == 0) return false;
        if (!detail::ascii_in(detail::kAlpha, s[0])) return false;
        if (!ps.all_in(1, colon, detail::kScheme, &bad)) return false;
        detail::set_span(&ps.out.scheme, 0, colon);
        p = colon + 1u;
        if (ps.at(p, '/')) {
            if (!detail::parse_hierarchical(&ps, p, &p)) return false;
        } else {
            const uint32_t q = ps.scan_to(p, n, "#");
            if (q <= p) return false;
            if (!ps.all_in(p, q, detail::kUric, &bad) || bad) return false;
            p = q;                                               // opaque: no path / query
        }
    } else if (!detail::parse_hierarchical(&ps, 0, &p)) {
        return false;
    }
    if (ps.at(p, '#')) {
        if (!ps.all_in(p + 1, n, detail::kUric, &bad) || bad) return false;
        detail::set_span(&ps.out.fragment, p + 1u, n);
        p = n;
    }
    if (p < n) return false;
    *out = ps.out;
    return true;
}

// ---- parse_url ------------------------------------------------------------

enum class UrlPart : uint8_t { Host, Path, Query, Ref, Protocol, File, Authority, UserInfo, Other };

inline UrlPart url_part_of(const char* p, uint32_t len) noexcept {
    assert(p != nullptr || len == 0);
    struct Row { const char* name; UrlPart part; };
    static const Row kRows[] = {{"HOST", UrlPart::Host}, {"PATH", UrlPart::Path}, {"QUERY", UrlPart::Query},
                                {"REF", UrlPart::Ref}, {"PROTOCOL", UrlPart::Protocol}, {"FILE", UrlPart::File},
                                {"AUTHORITY", UrlPart::Authority}, {"USERINFO", UrlPart::UserInfo}};
    for (const Row& r : kRows) {
        if (std::strlen(r.name) == len && std::memcmp(r.name, p, len) == 0) return r.part;
    }
    return UrlPart::Other;
}

// A regular-expression metacharacter: Spark compiles the QUERY key into "(&|^)key=([^&]*)".
inline bool key_is_plain(const char* k, uint32_t len) noexcept {
    for (uint32_t i = 0; i < len; ++i) if (detail::in_set("\\^$.|?*+()[]{}", k[i])) return false;
    return true;
}

// parse_url(url, part[, key]). Returns 1 with the byte range in [*b, *e) of `url` (the
// result is always a substring of it, except FILE which spans path "?" query and is also
// a contiguous range), 0 for SQL NULL, kTextInvalid for an invalid URL, 2 when `key` holds
// regular-expression metacharacters (the range is then the QUERY text; the caller matches
// "(&|^)key=([^&]*)" over it, as Spark compiles the key).
inline int parse_url(const char* url, uint32_t ulen, UrlPart part, const char* key,
                     uint32_t klen, bool has_key, uint32_t* b, uint32_t* e) noexcept {
    assert(url != nullptr || ulen == 0);
    assert(b != nullptr && e != nullptr);
    UriParts u;
    if (!java_uri_parse(url, ulen, &u)) return static_cast<int>(kTextInvalid);
    Span sp{};
    switch (part) {
        case UrlPart::Host: sp = u.host; break;
        case UrlPart::Path: sp = u.path; break;
        case UrlPart::Query: sp = u.query; break;
        case UrlPart::Ref: sp = u.fragment; break;
        case UrlPart::Protocol: sp = u.scheme; break;
        case UrlPart::Authority: sp = u.authority; break;
        case UrlPart::UserInfo: sp = u.userinfo; break;
        case UrlPart::File:
            sp = u.path;
            if (u.path.has && u.query.has) sp.len = u.query.off + u.query.len - u.path.off;   // path ?query
            break;
        case UrlPart::Other: return 0;
    }
    if (has_key && part != UrlPart::Query) return 0;            // a key only applies to QUERY
    if (!sp.has) return 0;
    if (!(part == UrlPart::Query && has_key)) { *b = sp.off; *e = sp.off + sp.len; return 1; }
    if (!key_is_plain(key, klen)) { *b = sp.off; *e = sp.off + sp.len; return 2; }   // regex key: caller searches the query
    const char* q = url + sp.off;
    const uint32_t ql = sp.len;
    for (uint32_t i = 0; i <= ql; ++i) {                         // bounded: query bytes
        const bool start = (i == 0) || q[i - 1] == '&';
        if (!start) continue;
        if (i + klen + 1 > ql) continue;
        if (std::memcmp(q + i, key, klen) != 0 || q[i + klen] != '=') continue;
        uint32_t v = i + klen + 1, w = v;
        while (w < ql && q[w] != '&') ++w;
        *b = sp.off + v;
        *e = sp.off + w;
        return 1;
    }
    return 0;
}

}  // namespace url
}  // namespace kernels
}  // namespace bolt
