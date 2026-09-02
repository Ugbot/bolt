// bolt_json_path.h — bounded JSON path compile + extract kernel
// (Gestalt2 json-functions lane: json_extract / -> / ->> SQL surface).
//
// This is NOT a general JSONPath engine. It supports exactly the subset the
// SQL scalar functions need, compiled ONCE (at SQL lowering / plan-build
// time, never per row) into a fixed-size JsonPath, then evaluated per row
// with a hand-rolled bounded scanner over the raw JSON bytes:
//
//   $                       root (identity — the whole document)
//   $.key                   object member access (chained: $.a.b.c)
//   $[N]  /  $.a[N]         array index access (N >= 0, decimal)
//   bare key ("a", "a.b")   a SINGLE object key, VERBATIM — this is the
//                           `col -> 'a'` operator form; note a bare key
//                           containing '.' is one literal key (PostgreSQL
//                           `->` semantics), NOT a nested path.
//   [N] (bare)              a single array index — the `col -> 2` form.
//
// NOT supported (documented gap, refused at compile, never mis-evaluated):
// wildcards `$.*` / `$[*]`, recursive descent `..`, quoted bracket keys
// `$['a']`, negative indices, filters. Path depth is capped at
// kMaxPathSegs (8); keys at kMaxKeyLen (48 bytes) — over-cap fails compile.
//
// Evaluation semantics: a missing key / index out of range / type mismatch
// (indexing an object with [N], keying an array) / MALFORMED JSON all
// return "not found" — the SQL layer surfaces SQL NULL, never a crash and
// never a fabricated value. Nesting while skipping values is bounded by
// kMaxSkipDepth; deeper documents fail the lookup cleanly.
//
// Tiger Style: noexcept, no exceptions, no heap allocation (caller-buffer
// unescape only), fixed caps, >=2 asserts/fn, functions <=70 lines.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace bolt {
namespace kernels {
namespace json {

constexpr std::uint8_t kMaxPathSegs  = 8;    // bounded path depth
constexpr std::uint8_t kMaxKeyLen    = 48;   // bytes per object key
constexpr int          kMaxSkipDepth = 32;   // {}/[] nesting bound in skips

struct JsonPathSeg {
    std::uint8_t is_index;          // 0 = object key, 1 = array index
    std::uint8_t key_len;           // bytes in key[] (is_index == 0)
    char         key[kMaxKeyLen];   // NOT NUL-terminated; key_len governs
    std::int64_t index;             // array index (is_index == 1)
};

struct JsonPath {
    std::uint8_t seg_count;         // 0 = identity (whole document)
    std::uint8_t _pad[7];
    JsonPathSeg  segs[kMaxPathSegs];
};

// A borrowed byte range inside the caller's JSON text. Never owns memory.
struct JsonSpan {
    const char*   p;
    std::uint32_t len;
};

// ---------------------------------------------------------------------------
// Path compile (once, at plan build — never per row).
// ---------------------------------------------------------------------------

// Parse one `.key` or `[N]` step starting at s[*i]. Returns false on any
// syntax error / cap breach.
inline bool jp_compile_step(const char* s, std::uint32_t n, std::uint32_t* i,
                            JsonPath* out) noexcept {
    assert(s != nullptr && i != nullptr && out != nullptr);
    assert(*i < n);
    if (out->seg_count >= kMaxPathSegs) return false;   // depth cap
    JsonPathSeg* seg = &out->segs[out->seg_count];
    std::memset(seg, 0, sizeof(*seg));
    if (s[*i] == '.') {
        ++*i;
        std::uint32_t k = 0;
        while (*i < n && s[*i] != '.' && s[*i] != '[') {
            if (k >= kMaxKeyLen) return false;          // key length cap
            seg->key[k++] = s[(*i)++];
        }
        if (k == 0) return false;                       // `$.` / `a..b`
        // `$.*` is JSONPath's wildcard — REFUSE it rather than silently
        // matching a member literally named "*" (a genuine "*" key is
        // still reachable via the bare-key `-> '*'` operator form).
        if (k == 1 && seg->key[0] == '*') return false;
        seg->key_len = static_cast<std::uint8_t>(k);
        seg->is_index = 0;
    } else if (s[*i] == '[') {
        ++*i;
        if (*i >= n || s[*i] < '0' || s[*i] > '9') return false;
        std::int64_t v = 0;
        std::uint32_t digits = 0;
        while (*i < n && s[*i] >= '0' && s[*i] <= '9') {
            if (digits++ >= 18) return false;           // bounded index
            v = v * 10 + (s[(*i)++] - '0');
        }
        if (*i >= n || s[*i] != ']') return false;
        ++*i;
        seg->is_index = 1;
        seg->index    = v;
    } else {
        return false;
    }
    ++out->seg_count;
    return true;
}

// Compile a path string. Accepts `$`, `$.a.b[0]`, `[N]...`, or a BARE key
// taken verbatim as one object-key segment (the `->` operator form).
inline bool json_path_compile(const char* path, std::uint32_t len,
                              JsonPath* out) noexcept {
    assert(path != nullptr);
    assert(out != nullptr);
    std::memset(out, 0, sizeof(*out));
    if (len == 0) return false;
    if (path[0] == '$') {
        std::uint32_t i = 1;
        while (i < len) {
            if (!jp_compile_step(path, len, &i, out)) return false;
        }
        return true;                                    // `$` alone: identity
    }
    if (path[0] == '[') {
        std::uint32_t i = 0;
        while (i < len) {
            if (!jp_compile_step(path, len, &i, out)) return false;
        }
        return out->seg_count > 0;
    }
    // Bare key: the WHOLE string, verbatim (may contain '.') — PG `->`.
    if (len > kMaxKeyLen) return false;
    out->segs[0].is_index = 0;
    out->segs[0].key_len  = static_cast<std::uint8_t>(len);
    std::memcpy(out->segs[0].key, path, len);
    out->seg_count = 1;
    return true;
}

// ---------------------------------------------------------------------------
// Bounded per-row scanner (no allocation, no recursion).
// ---------------------------------------------------------------------------

inline void jp_skip_ws(const char* s, std::uint32_t n,
                       std::uint32_t* i) noexcept {
    assert(s != nullptr && i != nullptr);
    while (*i < n && (s[*i] == ' ' || s[*i] == '\t' || s[*i] == '\n' ||
                      s[*i] == '\r')) {
        ++*i;
    }
}

// s[*i] must be the opening quote. Advances past the closing quote.
inline bool jp_skip_string(const char* s, std::uint32_t n,
                           std::uint32_t* i) noexcept {
    assert(s != nullptr && i != nullptr);
    assert(*i < n && s[*i] == '"');
    ++*i;
    while (*i < n) {
        const char c = s[(*i)++];
        if (c == '\\') {
            if (*i >= n) return false;                  // dangling escape
            ++*i;                                       // skip escaped char
        } else if (c == '"') {
            return true;
        }
    }
    return false;                                       // unterminated
}

// Skip one JSON value starting at s[*i] (ws already consumed). Iterative,
// depth-bounded; loop is bounded by n (every iteration advances *i).
inline bool jp_skip_value(const char* s, std::uint32_t n,
                          std::uint32_t* i) noexcept {
    assert(s != nullptr && i != nullptr);
    if (*i >= n) return false;
    int depth = 0;
    do {
        jp_skip_ws(s, n, i);
        if (*i >= n) return false;
        const char c = s[*i];
        if (c == '{' || c == '[') {
            if (++depth > kMaxSkipDepth) return false;  // nesting bound
            ++*i;
        } else if (c == '}' || c == ']') {
            if (--depth < 0) return false;
            ++*i;
        } else if (c == '"') {
            if (!jp_skip_string(s, n, i)) return false;
        } else if (c == ',' || c == ':') {
            if (depth == 0) return false;               // bare separator
            ++*i;
        } else {
            // scalar: number / true / false / null — scan to a delimiter
            const std::uint32_t start = *i;
            while (*i < n && s[*i] != ',' && s[*i] != '}' && s[*i] != ']' &&
                   s[*i] != ' ' && s[*i] != '\t' && s[*i] != '\n' &&
                   s[*i] != '\r') {
                ++*i;
            }
            if (*i == start) return false;              // no progress
        }
    } while (depth > 0 && *i <= n);
    return depth == 0;
}

// s[*i] at '{'. Find member `key` and set *out to its raw value span.
inline bool jp_object_find(const char* s, std::uint32_t n, std::uint32_t i,
                           const char* key, std::uint8_t klen,
                           JsonSpan* out) noexcept {
    assert(s != nullptr && key != nullptr && out != nullptr);
    assert(i < n && s[i] == '{');
    ++i;
    for (std::uint32_t iter = 0; iter <= n; ++iter) {   // bounded walk
        jp_skip_ws(s, n, &i);
        if (i >= n) return false;
        if (s[i] == '}') return false;                  // end: not found
        if (s[i] != '"') return false;                  // malformed key
        const std::uint32_t kstart = i + 1;
        if (!jp_skip_string(s, n, &i)) return false;
        const std::uint32_t kend = i - 1;               // excl. close quote
        jp_skip_ws(s, n, &i);
        if (i >= n || s[i] != ':') return false;
        ++i;
        jp_skip_ws(s, n, &i);
        const std::uint32_t vstart = i;
        if (!jp_skip_value(s, n, &i)) return false;
        // Match on RAW key bytes (escaped keys never match — documented).
        if (kend - kstart == klen &&
            std::memcmp(s + kstart, key, klen) == 0) {
            out->p   = s + vstart;
            out->len = i - vstart;
            return true;
        }
        jp_skip_ws(s, n, &i);
        if (i < n && s[i] == ',') { ++i; continue; }
        if (i < n && s[i] == '}') return false;         // end: not found
        return false;                                   // malformed
    }
    return false;
}

// s[i] at '['. Find element `idx` and set *out to its raw value span.
inline bool jp_array_at(const char* s, std::uint32_t n, std::uint32_t i,
                        std::int64_t idx, JsonSpan* out) noexcept {
    assert(s != nullptr && out != nullptr);
    assert(i < n && s[i] == '[');
    if (idx < 0) return false;
    ++i;
    std::int64_t at = 0;
    for (std::uint32_t iter = 0; iter <= n; ++iter) {   // bounded walk
        jp_skip_ws(s, n, &i);
        if (i >= n || s[i] == ']') return false;        // out of range / end
        const std::uint32_t vstart = i;
        if (!jp_skip_value(s, n, &i)) return false;
        if (at == idx) {
            out->p   = s + vstart;
            out->len = i - vstart;
            return true;
        }
        ++at;
        jp_skip_ws(s, n, &i);
        if (i < n && s[i] == ',') { ++i; continue; }
        return false;                                   // end or malformed
    }
    return false;
}

// Resolve `path` against the JSON text [s, s+n). On success *out is the raw
// (ws-trimmed) value span. Missing / mismatched / malformed => false.
inline bool json_path_lookup(const JsonPath* path, const char* s,
                             std::uint32_t n, JsonSpan* out) noexcept {
    assert(path != nullptr && out != nullptr);
    assert(path->seg_count <= kMaxPathSegs);
    if (s == nullptr || n == 0) return false;
    std::uint32_t i = 0;
    jp_skip_ws(s, n, &i);
    const std::uint32_t vstart = i;
    if (!jp_skip_value(s, n, &i)) return false;         // malformed root
    JsonSpan cur{s + vstart, i - vstart};
    // Trailing garbage after the root value = malformed document.
    jp_skip_ws(s, n, &i);
    if (i != n) return false;
    for (std::uint8_t d = 0; d < path->seg_count; ++d) {
        const JsonPathSeg& seg = path->segs[d];
        if (cur.len == 0) return false;
        JsonSpan next{};
        if (seg.is_index == 0) {
            if (cur.p[0] != '{') return false;          // keying a non-object
            if (!jp_object_find(cur.p, cur.len, 0, seg.key, seg.key_len,
                                &next)) {
                return false;
            }
        } else {
            if (cur.p[0] != '[') return false;          // indexing non-array
            if (!jp_array_at(cur.p, cur.len, 0, seg.index, &next)) {
                return false;
            }
        }
        cur = next;
    }
    *out = cur;
    return true;
}

// ---------------------------------------------------------------------------
// Typed accessors over a resolved span.
// ---------------------------------------------------------------------------

inline bool json_span_is_null(const JsonSpan& sp) noexcept {
    assert(sp.len == 0 || sp.p != nullptr);
    return sp.len == 4 && std::memcmp(sp.p, "null", 4) == 0;
}

// Strict JSON integer (optional '-', digits, no fraction/exponent).
inline bool json_span_to_i64(const JsonSpan& sp, std::int64_t* out) noexcept {
    assert(out != nullptr);
    if (sp.len == 0 || sp.p == nullptr) return false;
    std::uint32_t i = 0;
    bool neg = false;
    if (sp.p[0] == '-') { neg = true; i = 1; }
    if (i >= sp.len || sp.len - i > 19) return false;   // bounded digits
    std::uint64_t v = 0;                                // unsigned: no UB
    for (; i < sp.len; ++i) {
        const char c = sp.p[i];
        if (c < '0' || c > '9') return false;           // float/exp/garbage
        const std::uint64_t d = static_cast<std::uint64_t>(c - '0');
        if (v > (UINT64_C(0xFFFFFFFFFFFFFFFF) - d) / 10) return false;
        v = v * 10 + d;
    }
    const std::uint64_t lim = neg ? (UINT64_C(1) << 63)
                                  : ((UINT64_C(1) << 63) - 1);
    if (v > lim) return false;                          // int64 overflow
    *out = neg ? -static_cast<std::int64_t>(v - 1) - 1
               : static_cast<std::int64_t>(v);
    return true;
}

// JSON number (int or float) to double via strtod on a bounded local copy.
inline bool json_span_to_f64(const JsonSpan& sp, double* out) noexcept {
    assert(out != nullptr);
    if (sp.len == 0 || sp.p == nullptr || sp.len > 63) return false;
    const char c0 = sp.p[0];
    if (c0 != '-' && (c0 < '0' || c0 > '9')) return false;   // not a number
    char buf[64];
    std::memcpy(buf, sp.p, sp.len);
    buf[sp.len] = '\0';
    char* end = nullptr;
    const double v = std::strtod(buf, &end);
    if (end != buf + sp.len) return false;              // trailing garbage
    *out = v;
    return true;
}

// Decode one \uXXXX (surrogate-pair aware) at s[*i] (past the 'u').
// Emits UTF-8 into dst; returns bytes written or -1.
inline std::int32_t jp_unesc_u(const char* s, std::uint32_t n,
                               std::uint32_t* i, char* dst,
                               std::uint32_t cap) noexcept {
    assert(s != nullptr && i != nullptr && dst != nullptr);
    auto hex4 = [&](std::uint32_t at, std::uint32_t* v) noexcept -> bool {
        if (at + 4 > n) return false;
        std::uint32_t r = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s[at + k];
            std::uint32_t d;
            if (c >= '0' && c <= '9') d = static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') d = static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') d = static_cast<std::uint32_t>(c - 'A' + 10);
            else return false;
            r = (r << 4) | d;
        }
        *v = r;
        return true;
    };
    std::uint32_t cp = 0;
    if (!hex4(*i, &cp)) return -1;
    *i += 4;
    if (cp >= 0xD800 && cp <= 0xDBFF) {                 // high surrogate
        if (*i + 6 > n || s[*i] != '\\' || s[*i + 1] != 'u') return -1;
        std::uint32_t lo = 0;
        if (!hex4(*i + 2, &lo)) return -1;
        if (lo < 0xDC00 || lo > 0xDFFF) return -1;
        *i += 6;
        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
        return -1;                                      // lone low surrogate
    }
    if (cp < 0x80) {
        if (cap < 1) return -1;
        dst[0] = static_cast<char>(cp);
        return 1;
    }
    if (cp < 0x800) {
        if (cap < 2) return -1;
        dst[0] = static_cast<char>(0xC0 | (cp >> 6));
        dst[1] = static_cast<char>(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        if (cap < 3) return -1;
        dst[0] = static_cast<char>(0xE0 | (cp >> 12));
        dst[1] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        dst[2] = static_cast<char>(0x80 | (cp & 0x3F));
        return 3;
    }
    if (cap < 4) return -1;
    dst[0] = static_cast<char>(0xF0 | (cp >> 18));
    dst[1] = static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    dst[2] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    dst[3] = static_cast<char>(0x80 | (cp & 0x3F));
    return 4;
}

// ->> text form: a string span is UNQUOTED (escapes decoded); any other
// value's raw JSON text is copied verbatim. Returns bytes written or -1
// on overflow / bad escape. Callers must special-case JSON null FIRST
// (json_span_is_null) — Postgres ->> returns SQL NULL for it.
inline std::int32_t json_span_unquote(const JsonSpan& sp, char* dst,
                                      std::uint32_t cap) noexcept {
    assert(dst != nullptr);
    assert(sp.len == 0 || sp.p != nullptr);
    if (sp.len == 0) return -1;
    if (sp.p[0] != '"') {                               // non-string: verbatim
        if (sp.len > cap) return -1;
        std::memcpy(dst, sp.p, sp.len);
        return static_cast<std::int32_t>(sp.len);
    }
    if (sp.len < 2 || sp.p[sp.len - 1] != '"') return -1;
    const char* s = sp.p;
    const std::uint32_t n = sp.len - 1;                 // excl. close quote
    std::uint32_t i = 1;                                // past open quote
    std::uint32_t o = 0;
    while (i < n) {
        const char c = s[i++];
        if (c != '\\') {
            if (o >= cap) return -1;
            dst[o++] = c;
            continue;
        }
        if (i >= n) return -1;
        const char e = s[i++];
        char one = 0;
        switch (e) {
            case '"':  one = '"';  break;
            case '\\': one = '\\'; break;
            case '/':  one = '/';  break;
            case 'b':  one = '\b'; break;
            case 'f':  one = '\f'; break;
            case 'n':  one = '\n'; break;
            case 'r':  one = '\r'; break;
            case 't':  one = '\t'; break;
            case 'u': {
                const std::int32_t w =
                    jp_unesc_u(s, n, &i, dst + o, cap - o);
                if (w < 0) return -1;
                o += static_cast<std::uint32_t>(w);
                continue;
            }
            default: return -1;                         // invalid escape
        }
        if (o >= cap) return -1;
        dst[o++] = one;
    }
    return static_cast<std::int32_t>(o);
}

}  // namespace json
}  // namespace kernels
}  // namespace bolt
