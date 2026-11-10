// bolt::parse::yaml — see bolt/parse/bolt_yaml.h (G2GRAPH-453).
//
// A recursive-descent reader over the raw bytes: block structure is decided
// by the column of each content line, flow structure by indicators. Every
// value is written straight to the JSON output buffer as it is recognised,
// so there is no intermediate tree. Recursion depth is the collection depth,
// bounded by YamlLimits::depth (<= 256).

#include "bolt/parse/bolt_yaml.h"

#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bolt/bolt_arena.h"

namespace bolt::parse::yaml {

namespace {

constexpr uint32_t kMaxDepth = 256u;

struct P {
    const char*  s;
    uint32_t     n;
    uint32_t     i;          // cursor
    Arena*       arena;
    YamlLimits   lim;
    YamlError*   err;
    char*        out;
    uint32_t     out_len;
    uint32_t     out_cap;
    uint32_t     depth;
    uint32_t     nodes;
};

bool fail(P* p, YamlCode code, const char* what) noexcept {
    assert(p != nullptr && what != nullptr);
    if (p->err->code != YamlCode::Ok) return false;
    uint32_t line = 1, col = 1;
    for (uint32_t k = 0; k < p->i && k < p->n; ++k) {           // bounded by n
        if (p->s[k] == '\n') { ++line; col = 1; } else { ++col; }
    }
    p->err->code = code;
    p->err->line = line;
    p->err->col  = col;
    p->err->what = what;
    return false;
}

// ---- output ------------------------------------------------------------------

bool reserve(P* p, uint32_t need) noexcept {
    assert(p != nullptr);
    if (p->out_len + need + 1u <= p->out_cap) return true;
    if (static_cast<uint64_t>(p->out_len) + need > p->lim.bytes) {
        return fail(p, YamlCode::Limit, "output bytes");
    }
    uint64_t cap = p->out_cap == 0 ? 256u : static_cast<uint64_t>(p->out_cap) * 2u;
    while (cap < static_cast<uint64_t>(p->out_len) + need + 1u) cap *= 2u;   // bounded: doubles
    if (cap > static_cast<uint64_t>(p->lim.bytes) + 1u) cap = static_cast<uint64_t>(p->lim.bytes) + 1u;
    char* d = static_cast<char*>(p->arena->allocate(static_cast<size_t>(cap), 1));
    if (d == nullptr) return fail(p, YamlCode::NoMemory, "arena exhausted");
    if (p->out_len != 0) std::memcpy(d, p->out, p->out_len);
    p->out = d;
    p->out_cap = static_cast<uint32_t>(cap);
    return true;
}

bool emit(P* p, const char* b, uint32_t n) noexcept {
    if (!reserve(p, n)) return false;
    if (n != 0) std::memcpy(p->out + p->out_len, b, n);
    p->out_len += n;
    return true;
}

bool emit_c(P* p, char c) noexcept { return emit(p, &c, 1); }

bool count_node(P* p) noexcept {
    if (++p->nodes > p->lim.nodes) return fail(p, YamlCode::Limit, "nodes");
    return true;
}

bool emit_json_string(P* p, const char* b, uint32_t n) noexcept {
    assert(p != nullptr && (b != nullptr || n == 0));
    if (!reserve(p, n * 6u + 2u)) return false;
    char* w = p->out + p->out_len;
    *w++ = '"';
    for (uint32_t k = 0; k < n; ++k) {                          // bounded by n
        const auto c = static_cast<unsigned char>(b[k]);
        if (c == '"' || c == '\\') { *w++ = '\\'; *w++ = static_cast<char>(c); continue; }
        if (c >= 0x20) { *w++ = static_cast<char>(c); continue; }
        static constexpr char kHex[] = "0123456789abcdef";
        if (c == '\n') { *w++ = '\\'; *w++ = 'n'; continue; }
        if (c == '\t') { *w++ = '\\'; *w++ = 't'; continue; }
        if (c == '\r') { *w++ = '\\'; *w++ = 'r'; continue; }
        *w++ = '\\'; *w++ = 'u'; *w++ = '0'; *w++ = '0';
        *w++ = kHex[c >> 4]; *w++ = kHex[c & 15];
    }
    *w++ = '"';
    p->out_len = static_cast<uint32_t>(w - p->out);
    return true;
}

// Growable scratch for a scalar under construction (quoted / folded text).
struct Buf {
    char*    b;
    uint32_t n;
    uint32_t cap;
};

bool buf_put(P* p, Buf* x, const char* b, uint32_t n) noexcept {
    assert(p != nullptr && x != nullptr);
    if (x->n + n > x->cap) {
        uint64_t cap = x->cap == 0 ? 64u : static_cast<uint64_t>(x->cap) * 2u;
        while (cap < static_cast<uint64_t>(x->n) + n) cap *= 2u;          // bounded: doubles
        if (cap > p->lim.bytes) cap = p->lim.bytes;
        if (cap < static_cast<uint64_t>(x->n) + n) return fail(p, YamlCode::Limit, "input bytes");
        char* d = static_cast<char*>(p->arena->allocate(static_cast<size_t>(cap), 1));
        if (d == nullptr) return fail(p, YamlCode::NoMemory, "arena exhausted");
        if (x->n != 0) std::memcpy(d, x->b, x->n);
        x->b = d;
        x->cap = static_cast<uint32_t>(cap);
    }
    if (n != 0) std::memcpy(x->b + x->n, b, n);
    x->n += n;
    return true;
}

bool buf_c(P* p, Buf* x, char c) noexcept { return buf_put(p, x, &c, 1); }

// ---- lines ------------------------------------------------------------------

bool at_end(const P* p) noexcept { return p->i >= p->n; }
char cur(const P* p) noexcept { return p->i < p->n ? p->s[p->i] : '\0'; }
char at(const P* p, uint32_t k) noexcept { return k < p->n ? p->s[k] : '\0'; }
bool is_ws(char c) noexcept { return c == ' ' || c == '\t'; }
bool is_brk(char c) noexcept { return c == '\n' || c == '\r' || c == '\0'; }
bool is_blank_or_brk(char c) noexcept { return is_ws(c) || is_brk(c); }

uint32_t line_start(const P* p, uint32_t k) noexcept {
    while (k > 0 && p->s[k - 1] != '\n') --k;                   // bounded by k
    return k;
}

uint32_t column(const P* p, uint32_t k) noexcept { return k - line_start(p, k); }

void skip_ws(P* p) noexcept {
    while (!at_end(p) && is_ws(cur(p))) ++p->i;                  // bounded by n
}

void skip_to_eol(P* p) noexcept {
    while (!at_end(p) && cur(p) != '\n') ++p->i;                 // bounded by n
}

// Skip spaces and a comment to the end of the line. False if anything else
// is left on the line.
bool rest_of_line_empty(P* p) noexcept {
    skip_ws(p);
    if (cur(p) == '#') skip_to_eol(p);
    if (cur(p) == '\r') ++p->i;
    return is_brk(cur(p));
}

// The next content line from `k` (a line start or a position whose rest is
// empty): returns the position of its first non-space byte, *indent its
// column; n when none. Blank and comment-only lines are skipped. A tab in
// indentation is refused.
bool next_content(P* p, uint32_t k, uint32_t* pos, uint32_t* indent) noexcept {
    while (k < p->n && p->s[k] != '\n') ++k;                     // to end of this line
    while (k < p->n) {                                           // bounded by n
        ++k;                                                     // past '\n'
        uint32_t j = k;
        while (j < p->n && p->s[j] == ' ') ++j;
        if (j < p->n && p->s[j] == '\t') {
            uint32_t t = j;
            while (t < p->n && is_ws(p->s[t])) ++t;
            if (t < p->n && !is_brk(p->s[t]) && p->s[t] != '#') {
                p->i = j;
                return fail(p, YamlCode::Unsupported, "tab indentation");
            }
        }
        while (j < p->n && is_ws(p->s[j])) ++j;
        if (j >= p->n) break;
        if (is_brk(p->s[j]) || p->s[j] == '#') { k = j; while (k < p->n && p->s[k] != '\n') ++k; continue; }
        *pos = j;
        *indent = column(p, j);
        return true;
    }
    *pos = p->n;
    *indent = 0;
    return true;
}

bool doc_marker(const P* p, uint32_t k, const char* m) noexcept {
    return column(p, k) == 0 && k + 3 <= p->n && std::memcmp(p->s + k, m, 3) == 0 &&
           is_blank_or_brk(at(p, k + 3));
}

// ---- scalars ----------------------------------------------------------------

bool is_dec(const char* b, uint32_t n) noexcept {
    uint32_t k = (n > 0 && (b[0] == '-' || b[0] == '+')) ? 1u : 0u;
    if (k == n) return false;
    for (; k < n; ++k) if (b[k] < '0' || b[k] > '9') return false;
    return true;
}

bool is_float_text(const char* b, uint32_t n) noexcept {   // [-+]?(\.[0-9]+|[0-9]+(\.[0-9]*)?)([eE][-+]?[0-9]+)?
    uint32_t k = (n > 0 && (b[0] == '-' || b[0] == '+')) ? 1u : 0u;
    uint32_t digits = 0;
    while (k < n && b[k] >= '0' && b[k] <= '9') { ++k; ++digits; }
    if (k < n && b[k] == '.') {
        ++k;
        while (k < n && b[k] >= '0' && b[k] <= '9') { ++k; ++digits; }
    }
    if (digits == 0) return false;
    if (k < n && (b[k] == 'e' || b[k] == 'E')) {
        ++k;
        if (k < n && (b[k] == '-' || b[k] == '+')) ++k;
        uint32_t ed = 0;
        while (k < n && b[k] >= '0' && b[k] <= '9') { ++k; ++ed; }
        if (ed == 0) return false;
    }
    return k == n;
}

bool word_in(const char* b, uint32_t n, const char* const* words, uint32_t count) noexcept {
    for (uint32_t w = 0; w < count; ++w) {                        // bounded by count
        const uint32_t l = static_cast<uint32_t>(std::strlen(words[w]));
        if (l == n && std::memcmp(b, words[w], n) == 0) return true;
    }
    return false;
}

bool emit_int(P* p, const char* b, uint32_t n, int base) noexcept {
    char tmp[80];
    if (n >= sizeof(tmp)) return fail(p, YamlCode::Unsupported, "integer beyond int64");
    std::memcpy(tmp, b, n);
    tmp[n] = '\0';
    errno = 0;
    char* end = nullptr;
    const char* digits = base == 10 ? tmp : tmp + 2;
    const long long v = std::strtoll(digits, &end, base);
    if (errno == ERANGE || end != tmp + n) return fail(p, YamlCode::Unsupported, "integer beyond int64");
    const int w = std::snprintf(tmp, sizeof(tmp), "%lld", v);
    return emit(p, tmp, static_cast<uint32_t>(w));
}

bool emit_float(P* p, const char* b, uint32_t n) noexcept {
    char tmp[400];
    if (n >= sizeof(tmp)) return fail(p, YamlCode::Unsupported, "float literal too long");
    std::memcpy(tmp, b, n);
    tmp[n] = '\0';
    const double v = std::strtod(tmp, nullptr);
    if (!std::isfinite(v)) return fail(p, YamlCode::Unsupported, "non-finite float");
    int w = std::snprintf(tmp, sizeof(tmp), "%.17g", v);
    bool has_dot = false;
    for (int k = 0; k < w; ++k) has_dot = has_dot || tmp[k] == '.' || tmp[k] == 'e' || tmp[k] == 'n';
    if (!has_dot) { tmp[w++] = '.'; tmp[w++] = '0'; }
    return emit(p, tmp, static_cast<uint32_t>(w));
}

// A plain scalar under the YAML 1.2 core schema.
bool emit_plain(P* p, const char* b, uint32_t n) noexcept {
    static const char* const kNull[]  = {"~", "null", "Null", "NULL"};
    static const char* const kTrue[]  = {"true", "True", "TRUE"};
    static const char* const kFalse[] = {"false", "False", "FALSE"};
    static const char* const kInf[]   = {".inf", ".Inf", ".INF", "+.inf", "+.Inf", "+.INF",
                                         "-.inf", "-.Inf", "-.INF", ".nan", ".NaN", ".NAN"};
    if (!count_node(p)) return false;
    if (n == 0 || word_in(b, n, kNull, 4)) return emit(p, "null", 4);
    if (word_in(b, n, kTrue, 3)) return emit(p, "true", 4);
    if (word_in(b, n, kFalse, 3)) return emit(p, "false", 5);
    if (is_dec(b, n)) return emit_int(p, b, n, 10);
    if (n > 2 && b[0] == '0' && b[1] == 'o') {
        bool ok = true;
        for (uint32_t k = 2; k < n; ++k) ok = ok && b[k] >= '0' && b[k] <= '7';
        if (ok) return emit_int(p, b, n, 8);
    }
    if (n > 2 && b[0] == '0' && b[1] == 'x') {
        bool ok = true;
        for (uint32_t k = 2; k < n; ++k) {
            const char c = b[k];
            ok = ok && ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'));
        }
        if (ok) return emit_int(p, b, n, 16);
    }
    if (word_in(b, n, kInf, 12)) return fail(p, YamlCode::Unsupported, "non-finite float");
    if (is_float_text(b, n)) return emit_float(p, b, n);
    return emit_json_string(p, b, n);
}

bool flow_indicator(char c) noexcept { return c == ',' || c == '[' || c == ']' || c == '{' || c == '}'; }

// Where a plain scalar ends on this line: before " #", ": " (or ':' at end),
// and in flow context before an indicator.
uint32_t plain_end(const P* p, uint32_t k, bool flow) noexcept {
    uint32_t e = k;
    while (e < p->n && !is_brk(p->s[e])) {                        // bounded by n
        const char c = p->s[e];
        if (c == '#' && e > k && is_ws(p->s[e - 1])) break;
        if (c == ':' && (is_blank_or_brk(at(p, e + 1)) || (flow && flow_indicator(at(p, e + 1))))) break;
        if (flow && flow_indicator(c)) break;
        ++e;
    }
    while (e > k && is_ws(p->s[e - 1])) --e;
    return e;
}

bool refused_start(P* p, char c) noexcept {
    if (c == '&' || c == '*') return fail(p, YamlCode::Unsupported, "anchors and aliases");
    if (c == '!') return fail(p, YamlCode::Unsupported, "tags");
    if (c == '%') return fail(p, YamlCode::Unsupported, "directives");
    if (c == '?' && is_blank_or_brk(at(p, p->i + 1))) return fail(p, YamlCode::Unsupported, "complex keys");
    if (c == '@' || c == '`') return fail(p, YamlCode::Syntax, "reserved indicator");
    return true;
}

// Multi-line plain scalar in block context; continuation lines must be
// indented more than `parent`.
bool block_plain(P* p, int parent, Buf* out) noexcept {
    uint32_t e = plain_end(p, p->i, false);
    if (!buf_put(p, out, p->s + p->i, e - p->i)) return false;
    p->i = e;
    for (;;) {                                                    // bounded: advances lines
        uint32_t save = p->i;
        if (!rest_of_line_empty(p)) { p->i = save; return true; }
        if (cur(p) == '\0' && at_end(p)) return true;
        // Count blank lines and find the next content.
        uint32_t k = p->i, blanks = 0;
        bool comment = false;
        uint32_t pos = p->n, ind = 0;
        while (k < p->n) {
            ++k;
            uint32_t j = k;
            while (j < p->n && is_ws(p->s[j])) ++j;
            if (j >= p->n) { k = p->n; break; }
            if (is_brk(p->s[j])) { ++blanks; k = j; while (k < p->n && p->s[k] != '\n') ++k; continue; }
            if (p->s[j] == '#') comment = true;
            pos = j; ind = column(p, j);
            break;
        }
        if (pos >= p->n || comment || static_cast<int>(ind) <= parent ||
            doc_marker(p, line_start(p, pos), "---") || doc_marker(p, line_start(p, pos), "...")) {
            p->i = save;
            return true;
        }
        uint32_t pe = plain_end(p, pos, false);
        if (pe < p->n && p->s[pe] == ':') { p->i = pos; return fail(p, YamlCode::Syntax, "mapping value not allowed here"); }
        if (blanks == 0) { if (!buf_c(p, out, ' ')) return false; }
        for (uint32_t b = 0; b < blanks; ++b) if (!buf_c(p, out, '\n')) return false;
        if (!buf_put(p, out, p->s + pos, pe - pos)) return false;
        p->i = pe;
    }
}

// Fold a line break inside a quoted scalar at p->i ('\n'): trailing spaces of
// the line are dropped, blank lines become '\n', a single break a space.
bool quoted_break(P* p, Buf* out) noexcept {
    while (out->n > 0 && is_ws(out->b[out->n - 1])) --out->n;
    uint32_t blanks = 0;
    ++p->i;
    for (;;) {                                                    // bounded by n
        skip_ws(p);
        if (cur(p) == '\r') ++p->i;
        if (cur(p) != '\n') break;
        ++blanks;
        ++p->i;
    }
    if (blanks == 0) return buf_c(p, out, ' ');
    for (uint32_t b = 0; b < blanks; ++b) if (!buf_c(p, out, '\n')) return false;
    return true;
}

bool utf8_put(P* p, Buf* out, uint32_t cp) noexcept {
    char u[4];
    uint32_t n = 0;
    if (cp < 0x80) { u[n++] = static_cast<char>(cp); }
    else if (cp < 0x800) { u[n++] = static_cast<char>(0xC0 | (cp >> 6)); u[n++] = static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        u[n++] = static_cast<char>(0xE0 | (cp >> 12)); u[n++] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        u[n++] = static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x110000) {
        u[n++] = static_cast<char>(0xF0 | (cp >> 18)); u[n++] = static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        u[n++] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); u[n++] = static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        return fail(p, YamlCode::Syntax, "invalid escape");
    }
    return buf_put(p, out, u, n);
}

bool hex_escape(P* p, Buf* out, uint32_t digits) noexcept {
    uint32_t cp = 0;
    for (uint32_t k = 0; k < digits; ++k) {                       // bounded: <= 8
        const char c = at(p, p->i + 1 + k);
        const int v = (c >= '0' && c <= '9') ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10
                    : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : -1;
        if (v < 0) return fail(p, YamlCode::Syntax, "invalid escape");
        cp = cp * 16u + static_cast<uint32_t>(v);
    }
    p->i += digits;
    return utf8_put(p, out, cp);
}

bool dq_escape(P* p, Buf* out) noexcept {
    ++p->i;                                                       // past '\\'
    const char c = cur(p);
    static constexpr char kFrom[] = {'0', 'a', 'b', 't', 'n', 'v', 'f', 'r', 'e', ' ', '"', '/', '\\'};
    static constexpr char kTo[]   = {'\0', '\a', '\b', '\t', '\n', '\v', '\f', '\r', '\x1b', ' ', '"', '/', '\\'};
    if (c == '\n') { ++p->i; skip_ws(p); return true; }           // escaped break joins
    if (c == 'x') return hex_escape(p, out, 2) && (++p->i, true);
    if (c == 'u') return hex_escape(p, out, 4) && (++p->i, true);
    if (c == 'U') return hex_escape(p, out, 8) && (++p->i, true);
    if (c == 'N') { ++p->i; return utf8_put(p, out, 0x85); }
    if (c == '_') { ++p->i; return utf8_put(p, out, 0xA0); }
    if (c == 'L') { ++p->i; return utf8_put(p, out, 0x2028); }
    if (c == 'P') { ++p->i; return utf8_put(p, out, 0x2029); }
    if (c == '\t') { ++p->i; return buf_c(p, out, '\t'); }
    for (uint32_t k = 0; k < sizeof(kFrom); ++k) {                // bounded: table
        if (kFrom[k] == c) { ++p->i; return buf_c(p, out, kTo[k]); }
    }
    return fail(p, YamlCode::Syntax, "invalid escape");
}

bool quoted(P* p, Buf* out) noexcept {
    const char q = cur(p);
    ++p->i;
    for (;;) {                                                    // bounded by n
        if (at_end(p)) return fail(p, YamlCode::Syntax, "unterminated quoted scalar");
        const char c = cur(p);
        if (c == q) {
            if (q == '\'' && at(p, p->i + 1) == '\'') { if (!buf_c(p, out, '\'')) return false; p->i += 2; continue; }
            ++p->i;
            return true;
        }
        if (c == '\\' && q == '"') { if (!dq_escape(p, out)) return false; continue; }
        if (c == '\r' && at(p, p->i + 1) == '\n') { ++p->i; continue; }
        if (c == '\n') { if (!quoted_break(p, out)) return false; continue; }
        if (!buf_c(p, out, c)) return false;
        ++p->i;
    }
}

// ---- block scalars ------------------------------------------------------------

struct BlockHeader { char style; char chomp; uint32_t explicit_indent; };

bool block_header(P* p, BlockHeader* h) noexcept {
    h->style = cur(p);
    h->chomp = 'c';
    h->explicit_indent = 0;
    ++p->i;
    for (int k = 0; k < 2; ++k) {                                 // bounded: two indicators
        const char c = cur(p);
        if ((c == '+' || c == '-') && h->chomp == 'c') { h->chomp = c; ++p->i; continue; }
        if (c >= '1' && c <= '9' && h->explicit_indent == 0) { h->explicit_indent = static_cast<uint32_t>(c - '0'); ++p->i; continue; }
    }
    if (!rest_of_line_empty(p)) return fail(p, YamlCode::Syntax, "block scalar header");
    return true;
}

// Content lines of a block scalar: those with indent >= `ind` or blank.
bool block_scalar(P* p, int parent, Buf* out) noexcept {
    BlockHeader h{};
    if (!block_header(p, &h)) return false;
    uint32_t k = p->i < p->n ? p->i + 1 : p->n;                  // first content line start
    uint32_t ind = h.explicit_indent != 0 ? static_cast<uint32_t>(parent < 0 ? 0 : parent) + h.explicit_indent : 0;
    if (ind == 0) {                                               // detect from first non-empty line
        for (uint32_t t = k; t < p->n;) {
            uint32_t j = t;
            while (j < p->n && p->s[j] == ' ') ++j;
            if (j < p->n && !is_brk(p->s[j])) { ind = j - t; break; }
            while (j < p->n && p->s[j] != '\n') ++j;
            t = j + 1;
        }
        if (static_cast<int>(ind) <= parent) ind = static_cast<uint32_t>(parent + 1);
    }
    uint32_t trailing = 0;      // pending '\n's (blank lines) not yet written
    bool any = false, prev_more = false;
    while (k < p->n) {                                            // bounded by n
        uint32_t j = k, sp = 0;
        while (j < p->n && p->s[j] == ' ') { ++j; ++sp; }
        const bool blank = j >= p->n || is_brk(p->s[j]);
        if (!blank && sp < ind) break;
        uint32_t e = blank ? j : j;
        while (e < p->n && p->s[e] != '\n') ++e;
        uint32_t le = e;
        if (le > k && p->s[le - 1] == '\r') --le;
        if (blank) { ++trailing; k = e < p->n ? e + 1 : e; continue; }
        const bool more = sp > ind || p->s[k + ind] == '\t';
        if (any) {
            if (h.style == '|' || more || prev_more) {
                if (!buf_c(p, out, '\n')) return false;
                for (uint32_t b = 0; b < trailing; ++b) if (!buf_c(p, out, '\n')) return false;
            } else if (trailing == 0) {
                if (!buf_c(p, out, ' ')) return false;
            } else {
                for (uint32_t b = 0; b < trailing; ++b) if (!buf_c(p, out, '\n')) return false;
            }
        } else {
            for (uint32_t b = 0; b < trailing; ++b) if (!buf_c(p, out, '\n')) return false;
        }
        trailing = 0;
        if (!buf_put(p, out, p->s + k + ind, le - (k + ind))) return false;
        any = true;
        prev_more = more;
        k = e < p->n ? e + 1 : e;
    }
    if (h.chomp != '-' && any) { if (!buf_c(p, out, '\n')) return false; }
    if (h.chomp == '+') for (uint32_t b = 0; b < trailing; ++b) if (!buf_c(p, out, '\n')) return false;
    p->i = k > 0 ? k - 1 : 0;                                     // end of the last consumed line
    if (p->i < p->n && p->s[p->i] != '\n') p->i = k;
    return true;
}

// ---- collections ----------------------------------------------------------------

bool enter(P* p) noexcept {
    if (++p->depth > p->lim.depth) return fail(p, YamlCode::Limit, "depth");
    return count_node(p);
}

bool flow_node(P* p) noexcept;

// Whitespace, line breaks and comments inside a flow collection.
void flow_ws(P* p) noexcept {
    for (;;) {                                                    // bounded by n
        while (!at_end(p) && (is_ws(cur(p)) || cur(p) == '\n' || cur(p) == '\r')) ++p->i;
        if (cur(p) == '#' && (p->i == 0 || is_blank_or_brk(at(p, p->i - 1)))) { skip_to_eol(p); continue; }
        return;
    }
}

bool flow_scalar(P* p, bool as_key) noexcept {
    const char c = cur(p);
    if (!refused_start(p, c)) return false;
    if (c == '"' || c == '\'') {
        Buf b{};
        if (!quoted(p, &b)) return false;
        if (!as_key && !count_node(p)) return false;
        return emit_json_string(p, b.b, b.n);
    }
    const uint32_t e = plain_end(p, p->i, true);
    if (e == p->i && !as_key) return fail(p, YamlCode::Syntax, "empty flow entry");
    const uint32_t b = p->i;
    p->i = e;
    uint32_t t = p->i;
    while (t < p->n && is_ws(p->s[t])) ++t;
    if (t < p->n && is_brk(p->s[t])) {
        p->i = t;
        flow_ws(p);
        if (!at_end(p) && !flow_indicator(cur(p)) && cur(p) != ':') {
            return fail(p, YamlCode::Unsupported, "multi-line plain scalar in a flow collection");
        }
    }
    return as_key ? emit_json_string(p, p->s + b, e - b) : emit_plain(p, p->s + b, e - b);
}

bool flow_seq(P* p) noexcept {
    if (!enter(p) || !emit_c(p, '[')) return false;
    ++p->i;
    for (bool first = true;; first = false) {                     // bounded by n
        flow_ws(p);
        if (at_end(p)) return fail(p, YamlCode::Syntax, "unterminated flow sequence");
        if (cur(p) == ']') { ++p->i; break; }
        if (!first) {
            if (cur(p) != ',') return fail(p, YamlCode::Syntax, "expected ',' or ']'");
            ++p->i;
            flow_ws(p);
            if (cur(p) == ']') { ++p->i; break; }
            if (!emit_c(p, ',')) return false;
        }
        if (at_end(p)) return fail(p, YamlCode::Syntax, "unterminated flow sequence");
        if (!flow_node(p)) return false;
        flow_ws(p);
        if (cur(p) == ':') return fail(p, YamlCode::Unsupported, "single-pair mapping in a flow sequence");
    }
    --p->depth;
    return emit_c(p, ']');
}

bool flow_map(P* p) noexcept {
    if (!enter(p) || !emit_c(p, '{')) return false;
    ++p->i;
    for (bool first = true;; first = false) {                     // bounded by n
        flow_ws(p);
        if (at_end(p)) return fail(p, YamlCode::Syntax, "unterminated flow mapping");
        if (cur(p) == '}') { ++p->i; break; }
        if (!first) {
            if (cur(p) != ',') return fail(p, YamlCode::Syntax, "expected ',' or '}'");
            ++p->i;
            flow_ws(p);
            if (cur(p) == '}') { ++p->i; break; }
            if (!emit_c(p, ',')) return false;
        }
        if (at_end(p)) return fail(p, YamlCode::Syntax, "unterminated flow mapping");
        if (cur(p) == '[' || cur(p) == '{') return fail(p, YamlCode::Unsupported, "complex keys");
        if (!flow_scalar(p, true) || !emit_c(p, ':')) return false;
        flow_ws(p);
        if (cur(p) == ':') {
            ++p->i;
            flow_ws(p);
            if (cur(p) == ',' || cur(p) == '}') { if (!count_node(p) || !emit(p, "null", 4)) return false; continue; }
            if (!flow_node(p)) return false;
        } else {
            if (!count_node(p) || !emit(p, "null", 4)) return false;
        }
    }
    --p->depth;
    return emit_c(p, '}');
}

bool flow_node(P* p) noexcept {
    if (cur(p) == '[') return flow_seq(p);
    if (cur(p) == '{') return flow_map(p);
    return flow_scalar(p, false);
}

bool block_node(P* p, int parent) noexcept;
bool block_map(P* p, uint32_t col) noexcept;
bool block_seq(P* p, uint32_t col) noexcept;

// Is the content at p->i a mapping key on this line (`key:` + blank/EOL)?
bool looks_like_key(const P* p) noexcept {
    uint32_t k = p->i;
    const char c = at(p, k);
    if (c == '"' || c == '\'') {
        ++k;
        while (k < p->n && !is_brk(p->s[k])) {                    // bounded by n
            if (p->s[k] == '\\' && c == '"') { k += 2; continue; }
            if (p->s[k] == c) {
                if (c == '\'' && at(p, k + 1) == '\'') { k += 2; continue; }
                break;
            }
            ++k;
        }
        if (k >= p->n || p->s[k] != c) return false;
        ++k;
        while (k < p->n && is_ws(p->s[k])) ++k;
        return at(p, k) == ':' && is_blank_or_brk(at(p, k + 1));
    }
    if (c == '[' || c == '{' || c == '-' || c == '|' || c == '>' || c == '#') {
        if (!(c == '-' && !is_blank_or_brk(at(p, k + 1)))) return false;
    }
    const uint32_t e = plain_end(p, k, false);
    return e < p->n && p->s[e] == ':' && e > k;
}

// A value written after `key:` / `- ` on the same line.
bool inline_value(P* p, int parent) noexcept {
    const char c = cur(p);
    if (!refused_start(p, c)) return false;
    if (c == '[' || c == '{') {
        if (!flow_node(p)) return false;
        if (!rest_of_line_empty(p)) return fail(p, YamlCode::Syntax, "content after a flow collection");
        return true;
    }
    if (c == '|' || c == '>') {
        Buf b{};
        if (!block_scalar(p, parent, &b) || !count_node(p)) return false;
        return emit_json_string(p, b.b, b.n);
    }
    if (c == '"' || c == '\'') {
        Buf b{};
        if (!quoted(p, &b)) return false;
        const uint32_t save = p->i;
        if (!rest_of_line_empty(p)) { p->i = save; return fail(p, YamlCode::Syntax, "content after a quoted scalar"); }
        if (!count_node(p)) return false;
        return emit_json_string(p, b.b, b.n);
    }
    Buf b{};
    if (!block_plain(p, parent, &b)) return false;
    const uint32_t save = p->i;
    if (!rest_of_line_empty(p)) {
        p->i = save;
        return fail(p, YamlCode::Syntax, cur(p) == ':' ? "mapping value not allowed here" : "unexpected content");
    }
    return emit_plain(p, b.b, b.n);
}

// The value of `key:` / `- ` whose rest of line is empty: a block node on
// the following lines (or an indentless sequence at `col` for a mapping).
bool value_below(P* p, uint32_t col, bool indentless_ok) noexcept {
    uint32_t pos = 0, ind = 0;
    if (!next_content(p, p->i, &pos, &ind)) return false;
    const bool seq_here = indentless_ok && pos < p->n && ind == col && p->s[pos] == '-' &&
                          is_blank_or_brk(at(p, pos + 1));
    if (pos >= p->n || (ind <= col && !seq_here) || doc_marker(p, line_start(p, pos), "...") ||
        doc_marker(p, line_start(p, pos), "---")) {
        return count_node(p) && emit(p, "null", 4);
    }
    p->i = pos;
    if (seq_here) return block_seq(p, col);
    return block_node(p, static_cast<int>(col));
}

bool map_key(P* p) noexcept {
    const char c = cur(p);
    if (!refused_start(p, c)) return false;
    if (c == '"' || c == '\'') {
        Buf b{};
        if (!quoted(p, &b) || !emit_json_string(p, b.b, b.n)) return false;
        skip_ws(p);
    } else {
        const uint32_t e = plain_end(p, p->i, false);
        if (!emit_json_string(p, p->s + p->i, e - p->i)) return false;
        p->i = e;
    }
    if (cur(p) != ':') return fail(p, YamlCode::Syntax, "expected ':' after a mapping key");
    ++p->i;
    return emit_c(p, ':');
}

// Advance to the next sibling at `col`: true and p->i there, or false with
// *done set when the collection ends (or an error was recorded).
bool next_sibling(P* p, uint32_t col, bool* done) noexcept {
    uint32_t pos = 0, ind = 0;
    *done = true;
    if (!rest_of_line_empty(p)) return fail(p, YamlCode::Syntax, "unexpected content");
    if (!next_content(p, p->i, &pos, &ind)) return false;
    if (pos >= p->n || ind < col || doc_marker(p, line_start(p, pos), "...") ||
        doc_marker(p, line_start(p, pos), "---")) {
        p->i = pos >= p->n ? p->n : line_start(p, pos) - (line_start(p, pos) > 0 ? 1u : 0u);
        return true;
    }
    if (ind > col) { p->i = pos; return fail(p, YamlCode::Syntax, "bad indentation"); }
    p->i = pos;
    *done = false;
    return true;
}

bool block_map(P* p, uint32_t col) noexcept {
    if (!enter(p) || !emit_c(p, '{')) return false;
    for (bool first = true;; first = false) {                     // bounded: advances lines
        if (!first && !emit_c(p, ',')) return false;
        if (!looks_like_key(p)) return fail(p, YamlCode::Syntax, "expected a mapping key");
        if (!map_key(p)) return false;
        skip_ws(p);
        const uint32_t save = p->i;
        if (rest_of_line_empty(p)) {
            if (!value_below(p, col, true)) return false;
        } else {
            p->i = save;
            if (!inline_value(p, static_cast<int>(col))) return false;
        }
        bool done = false;
        if (!next_sibling(p, col, &done)) return false;
        if (done) break;
        if (cur(p) == '-' && is_blank_or_brk(at(p, p->i + 1))) {
            return fail(p, YamlCode::Syntax, "sequence entry where a mapping key was expected");
        }
    }
    --p->depth;
    return emit_c(p, '}');
}

bool block_seq(P* p, uint32_t col) noexcept {
    if (!enter(p) || !emit_c(p, '[')) return false;
    for (bool first = true;; first = false) {                     // bounded: advances lines
        if (!first && !emit_c(p, ',')) return false;
        assert(cur(p) == '-');
        ++p->i;
        skip_ws(p);
        const uint32_t save = p->i;
        if (rest_of_line_empty(p)) {
            if (!value_below(p, col, false)) return false;
        } else {
            p->i = save;
            if (!block_node(p, static_cast<int>(col))) return false;
        }
        bool done = false;
        if (!next_sibling(p, col, &done)) return false;
        if (done) break;
        if (!(cur(p) == '-' && is_blank_or_brk(at(p, p->i + 1)))) {
            p->i = line_start(p, p->i) > 0 ? line_start(p, p->i) - 1 : 0;   // a mapping key: the parent's
            break;
        }
    }
    --p->depth;
    return emit_c(p, ']');
}

// A node starting at p->i (its first byte) inside a block collection whose
// indent is `parent` (-1 at the top).
bool block_node(P* p, int parent) noexcept {
    const char c = cur(p);
    const uint32_t col = column(p, p->i);
    if (c == '-' && is_blank_or_brk(at(p, p->i + 1))) return block_seq(p, col);
    if (!refused_start(p, c)) return false;
    if (looks_like_key(p)) return block_map(p, col);
    return inline_value(p, parent);
}

bool document(P* p) noexcept {
    uint32_t pos = 0, ind = 0;
    p->i = 0;
    // Treat the input as if preceded by a line break so the first line is examined.
    uint32_t k = 0;
    while (k < p->n && (is_ws(p->s[k]))) ++k;
    if (k < p->n && (is_brk(p->s[k]) || p->s[k] == '#')) {
        if (!next_content(p, k, &pos, &ind)) return false;
    } else {
        pos = k; ind = column(p, k);
    }
    if (pos < p->n && at(p, pos) == '%') { p->i = pos; return fail(p, YamlCode::Unsupported, "directives"); }
    if (pos < p->n && doc_marker(p, pos, "---")) {
        p->i = pos + 3;
        if (!rest_of_line_empty(p)) return fail(p, YamlCode::Unsupported, "content on the --- line");
        if (!next_content(p, p->i, &pos, &ind)) return false;
    }
    if (pos >= p->n || doc_marker(p, pos, "...")) return count_node(p) && emit(p, "null", 4);
    p->i = pos;
    if (!block_node(p, -1)) return false;
    if (!rest_of_line_empty(p)) return fail(p, YamlCode::Syntax, "unexpected content");
    if (!next_content(p, p->i, &pos, &ind)) return false;
    if (pos < p->n && doc_marker(p, pos, "...")) {
        p->i = pos + 3;
        if (!next_content(p, p->i, &pos, &ind)) return false;
    }
    if (pos < p->n) {
        p->i = pos;
        return fail(p, YamlCode::Unsupported, doc_marker(p, pos, "---") ? "more than one document" : "content after the document");
    }
    return true;
}

}  // namespace

YamlCode to_json(const char* src, uint32_t len, Arena* arena, YamlLimits limits,
                 const char** out, uint32_t* out_len, YamlError* err) noexcept {
    assert(arena != nullptr && out != nullptr && out_len != nullptr && err != nullptr);
    *err = YamlError{YamlCode::Ok, 0, 0, ""};
    *out = nullptr;
    *out_len = 0;
    if (src == nullptr && len != 0) { err->code = YamlCode::Syntax; err->what = "null input"; return err->code; }
    if (limits.bytes == 0 || limits.bytes > 0x7FFFFFFFu || limits.depth == 0 || limits.depth > kMaxDepth ||
        limits.nodes == 0) {
        err->code = YamlCode::Limit; err->what = "invalid limits"; return err->code;
    }
    if (len > limits.bytes) { err->code = YamlCode::Limit; err->what = "input bytes"; return err->code; }
    P p{};
    p.s = src == nullptr ? "" : src; p.n = len; p.arena = arena; p.lim = limits; p.err = err;
    if (!document(&p)) {
        if (err->code == YamlCode::Ok) { err->code = YamlCode::Syntax; err->what = "invalid YAML"; }
        return err->code;
    }
    if (!reserve(&p, 0)) return err->code;
    p.out[p.out_len] = '\0';
    *out = p.out;
    *out_len = p.out_len;
    return YamlCode::Ok;
}

}  // namespace bolt::parse::yaml
