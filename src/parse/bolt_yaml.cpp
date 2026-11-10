// bolt::parse::yaml — see bolt/parse/bolt_yaml.h (G2GRAPH-453).
//
// Stage 1 (prescan) builds a line table word-at-a-time: line starts from an
// exact SWAR newline mask, indentation from an exact SWAR leading-space count,
// and each line's class. Stage 2 is recursive descent whose block decisions
// read that table (next content line, its indent, its class) instead of
// rescanning bytes; recursion depth is the collection depth, bounded by
// YamlLimits::depth (<= 256). Tokens go straight onto a json::Token tape.
// Scalars: bolt_yaml_scalars.inc; collections and the document:
// bolt_yaml_blocks.inc.

#include "bolt/parse/bolt_yaml.h"

#include <bit>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/parse/bolt_json_writer.h"

namespace bolt::parse::yaml {

namespace {

namespace bj = bolt::parse::json;

constexpr uint32_t kMaxDepth = 256u;

// ---- stage 1: word-at-a-time line table ---------------------------------------

constexpr uint64_t kOnes = 0x0101010101010101ull;
constexpr uint64_t kLow7 = 0x7F7F7F7F7F7F7F7Full;
constexpr uint64_t kHigh = 0x8080808080808080ull;

// 0x80 in each byte of `w` equal to `c`, 0 elsewhere (exact: no borrow leak).
inline uint64_t eq_mask(uint64_t w, uint8_t c) noexcept {
    const uint64_t x = w ^ (kOnes * c);
    return ~(((x & kLow7) + kLow7) | x | kLow7);
}

// Little-endian load of up to 8 bytes, padded with `pad`.
inline uint64_t load8(const char* p, uint32_t avail, char pad) noexcept {
    char b[8];
    if (avail >= 8) { std::memcpy(b, p, 8); }
    else { std::memset(b, pad, 8); std::memcpy(b, p, avail); }
    uint64_t w = 0;
    std::memcpy(&w, b, 8);
    if constexpr (std::endian::native == std::endian::big) w = __builtin_bswap64(w);
    return w;
}

enum : uint8_t { kLineContent = 0, kLineBlank = 1, kLineComment = 2, kLineTab = 3 };

struct Line {
    uint32_t start;     // first byte
    uint32_t end;       // the '\n' (or '\r' before it, or n)
    uint32_t content;   // first non-space byte (== start + indent)
    uint8_t  kind;
    uint8_t  _pad[3];
};

// Newline positions in bulk. TODO(SIMD): NEON/AVX2 16/32-byte compare +
// movemask, as bolt_json's deferred SIMD stage; this SWAR path is the reference.
uint32_t count_newlines(const char* s, uint32_t n) noexcept {
    uint32_t c = 0;
    for (uint32_t i = 0; i < n; i += 8) {                         // bounded by n
        const uint64_t m = eq_mask(load8(s + i, n - i, 0), '\n');
        c += static_cast<uint32_t>(std::popcount(m));
    }
    return c;
}

void fill_starts(const char* s, uint32_t n, Line* lines) noexcept {
    uint32_t k = 1;
    lines[0].start = 0;
    for (uint32_t i = 0; i < n; i += 8) {                         // bounded by n
        uint64_t m = eq_mask(load8(s + i, n - i, 0), '\n');
        while (m != 0) {                                          // bounded: <= 8 bits
            const uint32_t b = static_cast<uint32_t>(std::countr_zero(m)) >> 3;
            lines[k++].start = i + b + 1u;
            m &= m - 1u;
        }
    }
}

// Leading spaces of [p, p + avail), word at a time.
uint32_t leading_spaces(const char* p, uint32_t avail) noexcept {
    uint32_t k = 0;
    while (k < avail) {                                           // bounded by avail
        const uint64_t not_space = ~eq_mask(load8(p + k, avail - k, 'x'), ' ') & kHigh;
        if (not_space != 0) return k + (static_cast<uint32_t>(std::countr_zero(not_space)) >> 3);
        k += 8;
    }
    return avail;
}

void classify(const char* s, Line* L) noexcept {
    const uint32_t c = L->start + leading_spaces(s + L->start, L->end - L->start);
    uint32_t j = c;
    while (j < L->end && (s[j] == ' ' || s[j] == '\t')) ++j;      // bounded by end
    L->content = c;
    L->kind = j == L->end ? kLineBlank : s[j] == '#' ? kLineComment : s[c] == '\t' ? kLineTab : kLineContent;
}

// ---- parser state -------------------------------------------------------------------

struct P {
    const char*             s;
    uint32_t                n;
    uint32_t                i;
    Line*                   lines;
    uint32_t                nlines;
    Arena*                  arena;
    YamlLimits              lim;
    YamlError*              err;
    bj::Token*              tape;
    uint32_t                ntok, tcap;
    char*                   aux;
    uint32_t                naux, acap;
    uint32_t                depth, nodes;
    uint32_t                ln_hint;   // last line ln_of answered (cursor locality)
    uint32_t                key_pos, key_end;   // looks_like_key's scan, reused by map_key
    bool                    key_json;
    bool                    emit;
    const bj::PathFilter*   filter;
    int32_t                 path_len, nseg;
    int32_t                 seg_start[bj::kJsonMaxDepth];
    char                    path[bj::kJsonMaxDepth * (bj::kJsonMaxPathBytes + 1)];
};

// The line holding `pos`. The cursor moves forward through lines, so the
// previous answer or its successor almost always hits; else binary search.
uint32_t ln_of(const P* p, uint32_t pos) noexcept {
    uint32_t& hint = const_cast<P*>(p)->ln_hint;
    for (uint32_t h = hint; h < p->nlines && h <= hint + 1u; ++h) {   // bounded: 2 probes
        if (p->lines[h].start <= pos && (h + 1u == p->nlines || p->lines[h + 1u].start > pos)) return h;
    }
    uint32_t lo = 0, hi = p->nlines;                              // largest start <= pos
    while (hi - lo > 1) {                                         // bounded: log2(lines)
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (p->lines[mid].start <= pos) lo = mid; else hi = mid;
    }
    hint = lo;
    return lo;
}

bool fail(P* p, YamlCode code, const char* what) noexcept {
    assert(p != nullptr && what != nullptr);
    if (p->err->code != YamlCode::Ok) return false;
    const uint32_t pos = p->i < p->n ? p->i : p->n;
    const uint32_t l = p->lines == nullptr ? 0 : ln_of(p, pos);
    p->err->code = code;
    p->err->line = l + 1u;
    p->err->col = p->lines == nullptr ? 1u : pos - p->lines[l].start + 1u;
    p->err->what = what;
    return false;
}

bool prescan(P* p) noexcept {
    p->nlines = count_newlines(p->s, p->n) + 1u;
    p->lines = p->arena->allocate_array<Line>(p->nlines);
    if (p->lines == nullptr) return fail(p, YamlCode::NoMemory, "arena exhausted");
    fill_starts(p->s, p->n, p->lines);
    for (uint32_t l = 0; l < p->nlines; ++l) {                    // bounded by lines
        Line& L = p->lines[l];
        L.end = l + 1 < p->nlines ? p->lines[l + 1].start - 1u : p->n;
        if (L.end > L.start && p->s[L.end - 1] == '\r') --L.end;
        classify(p->s, &L);
    }
    return true;
}

// ---- tape and aux ---------------------------------------------------------------

bool tok(P* p, bj::TokenType t, uint32_t start, uint32_t len, uint8_t flags) noexcept {
    if (!p->emit) return true;
    if (p->ntok == p->tcap) {
        // First size: ~1 token per 8 source bytes (frontmatter averages ~1 per 12).
        const uint64_t cap = p->tcap == 0 ? 64u + p->n / 8u : static_cast<uint64_t>(p->tcap) * 2u;
        if (cap > 0x7FFFFFFFu) return fail(p, YamlCode::Limit, "nodes");
        auto* nt = p->arena->allocate_array<bj::Token>(static_cast<size_t>(cap));
        if (nt == nullptr) return fail(p, YamlCode::NoMemory, "arena exhausted");
        if (p->ntok != 0) std::memcpy(nt, p->tape, sizeof(bj::Token) * p->ntok);
        p->tape = nt;
        p->tcap = static_cast<uint32_t>(cap);
    }
    bj::Token& k = p->tape[p->ntok++];
    k.type = t; k.flags = flags; k._pad[0] = 0; k._pad[1] = 0;
    k.start = static_cast<int32_t>(start);
    k.length = static_cast<int32_t>(len);
    return true;
}

bool aux_reserve(P* p, uint32_t need) noexcept {
    if (p->naux + static_cast<uint64_t>(need) <= p->acap) return true;
    uint64_t cap = p->acap == 0 ? 256u : static_cast<uint64_t>(p->acap) * 2u;
    while (cap < p->naux + static_cast<uint64_t>(need)) cap *= 2u;      // bounded: doubles
    if (cap > 0x7FFFFFFFu) return fail(p, YamlCode::Limit, "input bytes");
    char* d = static_cast<char*>(p->arena->allocate(static_cast<size_t>(cap), 1));
    if (d == nullptr) return fail(p, YamlCode::NoMemory, "arena exhausted");
    if (p->naux != 0) std::memcpy(d, p->aux, p->naux);
    p->aux = d;
    p->acap = static_cast<uint32_t>(cap);
    return true;
}

// Append `b` JSON-escaped to aux; *off its offset, *len its escaped length.
bool aux_escaped(P* p, const char* b, uint32_t n, uint32_t* off, uint32_t* len) noexcept {
    if (!aux_reserve(p, n * 6u)) return false;
    static constexpr char kHex[] = "0123456789abcdef";
    *off = p->naux;
    char* w = p->aux + p->naux;
    for (uint32_t k = 0; k < n; ++k) {                            // bounded by n
        const auto c = static_cast<unsigned char>(b[k]);
        if (c == '"' || c == '\\') { *w++ = '\\'; *w++ = static_cast<char>(c); continue; }
        if (c >= 0x20) { *w++ = static_cast<char>(c); continue; }
        *w++ = '\\';
        if (c == '\n') { *w++ = 'n'; continue; }
        if (c == '\t') { *w++ = 't'; continue; }
        if (c == '\r') { *w++ = 'r'; continue; }
        *w++ = 'u'; *w++ = '0'; *w++ = '0'; *w++ = kHex[c >> 4]; *w++ = kHex[c & 15];
    }
    *len = static_cast<uint32_t>(w - (p->aux + p->naux));
    p->naux += *len;
    return true;
}

bool aux_raw(P* p, const char* b, uint32_t n, uint32_t* off) noexcept {
    if (!aux_reserve(p, n)) return false;
    *off = p->naux;
    std::memcpy(p->aux + p->naux, b, n);
    p->naux += n;
    return true;
}

bool count_node(P* p) noexcept {
    if (++p->nodes > p->lim.nodes) return fail(p, YamlCode::Limit, "nodes");
    return true;
}

// ---- scalar text ------------------------------------------------------------------

// Growable scratch for text that is not a source slice.
struct Buf {
    char*    b;
    uint32_t n, cap;
};

bool buf_put(P* p, Buf* x, const char* b, uint32_t n) noexcept {
    if (x->n + static_cast<uint64_t>(n) > x->cap) {
        uint64_t cap = x->cap == 0 ? 64u : static_cast<uint64_t>(x->cap) * 2u;
        while (cap < static_cast<uint64_t>(x->n) + n) cap *= 2u;          // bounded: doubles
        if (cap > static_cast<uint64_t>(p->lim.bytes) * 2u + 64u) return fail(p, YamlCode::Limit, "input bytes");
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

// A scalar's text: a source slice [b, e) (JSON-inner already when `json`), or
// materialised bytes in `buf` (raw, to be JSON-escaped).
struct Text {
    uint32_t b, e;
    bool     slice;
    bool     json;     // the slice is already JSON string content
    Buf      buf;
};

const char* text_ptr(const P* p, const Text& t) noexcept { return t.slice ? p->s + t.b : t.buf.b; }
uint32_t text_len(const Text& t) noexcept { return t.slice ? t.e - t.b : t.buf.n; }

bool json_safe(const char* b, uint32_t n) noexcept {
    for (uint32_t k = 0; k < n; ++k) {                            // bounded by n
        const auto c = static_cast<unsigned char>(b[k]);
        if (c < 0x20 || c == '"' || c == '\\') return false;
    }
    return true;
}

// A Key / String token for `t`: zero-copy when the slice is JSON content.
bool emit_text(P* p, bj::TokenType type, const Text& t) noexcept {
    if (!p->emit) return true;
    if (t.slice && (t.json || json_safe(p->s + t.b, t.e - t.b))) return tok(p, type, t.b, t.e - t.b, 0);
    uint32_t off = 0, len = 0;
    return aux_escaped(p, text_ptr(p, t), text_len(t), &off, &len) && tok(p, type, off, len, bj::kTokenAux);
}

// ---- line navigation ---------------------------------------------------------------

bool at_end(const P* p) noexcept { return p->i >= p->n; }
char cur(const P* p) noexcept { return p->i < p->n ? p->s[p->i] : '\0'; }
char at(const P* p, uint32_t k) noexcept { return k < p->n ? p->s[k] : '\0'; }
bool is_ws(char c) noexcept { return c == ' ' || c == '\t'; }
bool is_brk(char c) noexcept { return c == '\n' || c == '\r' || c == '\0'; }
bool blank_or_brk(char c) noexcept { return is_ws(c) || is_brk(c); }
uint32_t line_end(const P* p, uint32_t pos) noexcept { return p->lines[ln_of(p, pos)].end; }
uint32_t column(const P* p, uint32_t pos) noexcept { return pos - p->lines[ln_of(p, pos)].start; }

void skip_ws(P* p) noexcept { while (!at_end(p) && is_ws(cur(p))) ++p->i; }

// Skip spaces and a comment to the end of the line; false if anything else is left.
bool rest_empty(P* p) noexcept {
    skip_ws(p);
    const uint32_t e = line_end(p, p->i);
    if (p->i < e && cur(p) == '#') p->i = e;
    return p->i >= e;
}

// The next content line after the line holding `k`: *pos its first byte,
// *indent its column; *pos == n when none. A tab-indented line is refused.
bool next_content(P* p, uint32_t k, uint32_t* pos, uint32_t* indent) noexcept {
    for (uint32_t l = ln_of(p, k) + 1u; l < p->nlines; ++l) {      // bounded by lines
        const Line& L = p->lines[l];
        if (L.kind == kLineBlank || L.kind == kLineComment) continue;
        if (L.kind == kLineTab) { p->i = L.content; return fail(p, YamlCode::Unsupported, "tab indentation"); }
        *pos = L.content;
        *indent = L.content - L.start;
        return true;
    }
    *pos = p->n;
    *indent = 0;
    return true;
}

bool doc_marker(const P* p, uint32_t k, const char* m) noexcept {
    return k < p->n && column(p, k) == 0 && k + 3 <= p->n && std::memcmp(p->s + k, m, 3) == 0 &&
           blank_or_brk(at(p, k + 3));
}

bool doc_boundary(const P* p, uint32_t k) noexcept { return doc_marker(p, k, "---") || doc_marker(p, k, "..."); }

#include "bolt_yaml_scalars.inc"
#include "bolt_yaml_blocks.inc"

}  // namespace

#include "bolt_yaml_api.inc"

}  // namespace bolt::parse::yaml
