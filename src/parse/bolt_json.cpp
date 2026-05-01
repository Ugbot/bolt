// ---------------------------------------------------------------------------
// bolt::parse::json — skip-aware JSON parser implementation.
//
// Layer 1.3. Four stages composed in this file:
//   1. structural-index scan — single linear pass over `src`, scalar
//      fallback. UTF-8 validated DFA-style.
//   2. tape — Token entries written into the arena.
//   3. iterator — cursor over tokens with O(tokens-skipped) skip_to_close.
//   4. path filter — perfect-hash table over slash-prefixed interest paths;
//      build_index_filtered traverses without emitting tokens for subtrees
//      that aren't a prefix of any interest path.
//
// SIMD is deferred. See docs/research/json-skip-architecture.md.
// ---------------------------------------------------------------------------
#include "bolt/parse/bolt_json.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace bolt {
namespace parse {
namespace json {

// ===========================================================================
// PathFilter — closed-addressing FNV-1a hash table.
// ===========================================================================

namespace {

struct PathEntry {
    const char* bytes;  // arena-owned; not null-terminated
    int32_t     length;
    int32_t     occupied;  // 0 = empty slot, 1 = occupied
};

struct PathFilterImpl {
    PathEntry* table;     // arena-allocated; size == capacity
    int32_t    capacity;  // power-of-two
    int32_t    count;
};

BOLT_FORCE_INLINE uint64_t fnv1a(const char* BOLT_RESTRICT b, int32_t n) noexcept {
    assert(n >= 0);
    assert(n == 0 || b != nullptr);
    uint64_t h = 0xCBF29CE484222325ULL;
    for (int32_t i = 0; i < n; ++i) {
        h ^= static_cast<uint8_t>(b[i]);
        h *= 0x100000001B3ULL;
    }
    return h;
}

BOLT_FORCE_INLINE int32_t next_pow2(int32_t v) noexcept {
    assert(v > 0);
    int32_t r = 1;
    while (r < v) r <<= 1;
    return r;
}

// Insert (b,n) into the table. Returns false if the table is full (which
// shouldn't happen if capacity was sized to count*4).
bool path_table_insert(PathFilterImpl* impl, const char* b, int32_t n) noexcept {
    assert(impl != nullptr);
    assert(impl->capacity > 0);
    const uint64_t h = fnv1a(b, n);
    const uint32_t mask = static_cast<uint32_t>(impl->capacity - 1);
    uint32_t slot = static_cast<uint32_t>(h) & mask;
    for (int32_t probes = 0; probes < impl->capacity; ++probes) {
        PathEntry& e = impl->table[slot];
        if (!e.occupied) {
            e.bytes = b;
            e.length = n;
            e.occupied = 1;
            ++impl->count;
            return true;
        }
        if (e.length == n && memcmp(e.bytes, b, static_cast<size_t>(n)) == 0) {
            return true;  // duplicate — silently merge
        }
        slot = (slot + 1u) & mask;
    }
    return false;
}

bool path_table_contains(const PathFilterImpl* impl, const char* b, int32_t n) noexcept {
    assert(impl != nullptr);
    if (impl->count == 0) return false;
    const uint64_t h = fnv1a(b, n);
    const uint32_t mask = static_cast<uint32_t>(impl->capacity - 1);
    uint32_t slot = static_cast<uint32_t>(h) & mask;
    for (int32_t probes = 0; probes < impl->capacity; ++probes) {
        const PathEntry& e = impl->table[slot];
        if (!e.occupied) return false;
        if (e.length == n && memcmp(e.bytes, b, static_cast<size_t>(n)) == 0) {
            return true;
        }
        slot = (slot + 1u) & mask;
    }
    return false;
}

// True iff `cur` (length cn) is a prefix of any path in `impl`. We test by
// walking each occupied entry — the table is small (≤4096 slots) and the
// path-stack walks the prefix once per descent, so this is fine for v1.
// A precomputed prefix-set is the obvious upgrade.
bool path_is_prefix(const PathFilterImpl* impl, const char* cur, int32_t cn) noexcept {
    assert(impl != nullptr);
    if (impl->count == 0) return false;
    for (int32_t i = 0; i < impl->capacity; ++i) {
        const PathEntry& e = impl->table[i];
        if (!e.occupied) continue;
        if (e.length < cn) continue;
        if (cn == 0) return true;
        if (memcmp(e.bytes, cur, static_cast<size_t>(cn)) != 0) continue;
        // Boundary: the next char in the entry must be '/' or end-of-path.
        if (e.length == cn) return true;
        if (e.bytes[cn] == '/') return true;
    }
    return false;
}

}  // namespace

bool compile_paths(const char* const* paths, const int32_t* path_lens,
                   int32_t count, Arena* arena, PathFilter* out) noexcept {
    assert(arena != nullptr);
    assert(out != nullptr);
    if (count < 0 || count > kJsonMaxPaths) return false;
    if (count > 0 && (paths == nullptr || path_lens == nullptr)) return false;

    PathFilterImpl* impl = arena->allocate_array<PathFilterImpl>(1);
    if (impl == nullptr) return false;
    impl->capacity = (count == 0) ? 1 : next_pow2(count * 4);
    impl->count = 0;
    impl->table = arena->allocate_array<PathEntry>(static_cast<size_t>(impl->capacity));
    if (impl->table == nullptr) return false;
    memset(impl->table, 0, sizeof(PathEntry) * static_cast<size_t>(impl->capacity));

    for (int32_t i = 0; i < count; ++i) {
        const char* p = paths[i];
        const int32_t n = path_lens[i];
        if (n <= 0 || n > kJsonMaxPathBytes) return false;
        if (p == nullptr) return false;
        if (p[0] != '/') return false;
        char* copy = static_cast<char*>(arena->copy_into(p, static_cast<size_t>(n), 1));
        if (copy == nullptr) return false;
        if (!path_table_insert(impl, copy, n)) return false;
    }

    out->impl = impl;
    out->path_count = count;
    out->_pad = 0;
    return true;
}

// ===========================================================================
// UTF-8 validation DFA (Bjoern Hoehrmann, public-domain).
// We use the single-table form to reject overlong sequences and surrogates.
// ===========================================================================

namespace {

constexpr uint8_t kUtf8Accept = 0;
constexpr uint8_t kUtf8Reject = 12;

const uint8_t kUtf8d[] = {
    // The first part maps bytes to character classes.
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7, 7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    8,8,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    10,3,3,3,3,3,3,3,3,3,3,3,3,4,3,3, 11,6,6,6,5,8,8,8,8,8,8,8,8,8,8,8,
    // The second part is a transition table.
    0,12,24,36,60,96,84,12,12,12,48,72, 12,12,12,12,12,12,12,12,12,12,12,12,
    12, 0,12,12,12,12,12, 0,12, 0,12,12, 12,24,12,12,12,12,12,24,12,24,12,12,
    12,12,12,12,12,12,12,24,12,12,12,12, 12,24,12,12,12,12,12,12,12,24,12,12,
    12,12,12,12,12,12,12,36,12,36,12,12, 12,36,12,12,12,12,12,36,12,36,12,12,
    12,36,12,12,12,12,12,12,12,12,12,12,
};

BOLT_FORCE_INLINE uint8_t utf8_decode_step(uint8_t* state, uint8_t byte) noexcept {
    assert(state != nullptr);
    const uint8_t type = kUtf8d[byte];
    *state = kUtf8d[256 + (*state) + type];
    return *state;
}

}  // namespace

// ===========================================================================
// Stage 1+2 — single-pass parser that emits Tokens to the tape.
// ===========================================================================

namespace {

struct Parser {
    const uint8_t* src;
    int32_t        src_len;
    int32_t        pos;
    Token*         tokens;
    int32_t        token_cap;
    int32_t        token_count;

    // Path-filter state. nullptr → unfiltered.
    const PathFilterImpl* filter;
    char                  path_stack[kJsonMaxDepth * (kJsonMaxPathBytes + 1)];
    int32_t               seg_start[kJsonMaxDepth];   // start offset within path_stack
    int32_t               seg_count;
    int32_t               path_len;                   // current bytes used in path_stack
    int32_t               array_index[kJsonMaxDepth]; // -1 == not in array
    int32_t               depth;
};

BOLT_FORCE_INLINE bool emit(Parser* p, TokenType t, int32_t s, int32_t l) noexcept {
    assert(p != nullptr);
    if (p->token_count >= p->token_cap) return false;
    Token& tok = p->tokens[p->token_count++];
    tok.type = t;
    tok._pad[0] = 0; tok._pad[1] = 0; tok._pad[2] = 0;
    tok.start = s;
    tok.length = l;
    return true;
}

BOLT_FORCE_INLINE void skip_ws(Parser* p) noexcept {
    assert(p != nullptr);
    while (p->pos < p->src_len) {
        const uint8_t c = p->src[p->pos];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') return;
        ++p->pos;
    }
}

// Consume a JSON string starting at `*p->pos == '"'`. Sets out_start /
// out_length to the bytes inside the quotes (exclusive of the quotes).
// Validates UTF-8 and rejects bare control bytes < 0x20.
bool scan_string(Parser* p, int32_t* out_start, int32_t* out_length) noexcept {
    assert(p != nullptr);
    assert(p->pos < p->src_len && p->src[p->pos] == '"');
    ++p->pos;  // opening quote
    const int32_t start = p->pos;
    uint8_t state = kUtf8Accept;
    while (p->pos < p->src_len) {
        const uint8_t c = p->src[p->pos];
        if (c == '"' && state == kUtf8Accept) {
            *out_start = start;
            *out_length = p->pos - start;
            ++p->pos;  // closing quote
            return true;
        }
        if (c == '\\') {
            // Escape consumes one extra byte. We don't validate inner forms
            // beyond requiring at least one follower (\uXXXX is parsed
            // lazily by the consumer).
            if (p->pos + 1 >= p->src_len) return false;
            p->pos += 2;
            state = kUtf8Accept;
            continue;
        }
        if (c < 0x20 && state == kUtf8Accept) return false;  // bare control
        utf8_decode_step(&state, c);
        if (state == kUtf8Reject) return false;
        ++p->pos;
    }
    return false;  // unterminated
}

bool scan_number(Parser* p, int32_t* out_start, int32_t* out_length, bool* is_float) noexcept {
    assert(p != nullptr);
    const int32_t start = p->pos;
    *is_float = false;
    if (p->src[p->pos] == '-') ++p->pos;
    if (p->pos >= p->src_len) return false;
    if (p->src[p->pos] < '0' || p->src[p->pos] > '9') return false;
    while (p->pos < p->src_len && p->src[p->pos] >= '0' && p->src[p->pos] <= '9') ++p->pos;
    if (p->pos < p->src_len && p->src[p->pos] == '.') {
        *is_float = true;
        ++p->pos;
        if (p->pos >= p->src_len || p->src[p->pos] < '0' || p->src[p->pos] > '9') return false;
        while (p->pos < p->src_len && p->src[p->pos] >= '0' && p->src[p->pos] <= '9') ++p->pos;
    }
    if (p->pos < p->src_len && (p->src[p->pos] == 'e' || p->src[p->pos] == 'E')) {
        *is_float = true;
        ++p->pos;
        if (p->pos < p->src_len && (p->src[p->pos] == '+' || p->src[p->pos] == '-')) ++p->pos;
        if (p->pos >= p->src_len || p->src[p->pos] < '0' || p->src[p->pos] > '9') return false;
        while (p->pos < p->src_len && p->src[p->pos] >= '0' && p->src[p->pos] <= '9') ++p->pos;
    }
    *out_start = start;
    *out_length = p->pos - start;
    return true;
}

bool match_keyword(Parser* p, const char* kw, int32_t kw_len) noexcept {
    assert(p != nullptr);
    if (p->src_len - p->pos < kw_len) return false;
    if (memcmp(p->src + p->pos, kw, static_cast<size_t>(kw_len)) != 0) return false;
    p->pos += kw_len;
    return true;
}

// Path-stack helpers — used only by build_index_filtered; cheap enough to
// always update in the unfiltered path too, but we early-out via filter==null.

void path_push_key(Parser* p, const char* key, int32_t key_len) noexcept {
    assert(p != nullptr);
    if (p->seg_count >= kJsonMaxDepth) return;
    if (key_len > kJsonMaxPathBytes) key_len = kJsonMaxPathBytes;
    if (p->path_len + 1 + key_len > static_cast<int32_t>(sizeof(p->path_stack))) return;
    p->seg_start[p->seg_count++] = p->path_len;
    p->path_stack[p->path_len++] = '/';
    memcpy(p->path_stack + p->path_len, key, static_cast<size_t>(key_len));
    p->path_len += key_len;
}

void path_push_index(Parser* p, int32_t idx) noexcept {
    assert(p != nullptr);
    if (p->seg_count >= kJsonMaxDepth) return;
    if (p->path_len + 12 > static_cast<int32_t>(sizeof(p->path_stack))) return;
    p->seg_start[p->seg_count++] = p->path_len;
    p->path_stack[p->path_len++] = '/';
    char buf[12];
    int32_t n = 0;
    if (idx == 0) { buf[n++] = '0'; }
    else {
        char tmp[12]; int32_t m = 0;
        int32_t v = idx;
        while (v > 0) { tmp[m++] = static_cast<char>('0' + (v % 10)); v /= 10; }
        for (int32_t i = m - 1; i >= 0; --i) buf[n++] = tmp[i];
    }
    memcpy(p->path_stack + p->path_len, buf, static_cast<size_t>(n));
    p->path_len += n;
}

void path_pop(Parser* p) noexcept {
    assert(p != nullptr);
    if (p->seg_count <= 0) return;
    --p->seg_count;
    p->path_len = p->seg_start[p->seg_count];
}

// Forward decl.
bool parse_value(Parser* p, bool emit_tokens) noexcept;

bool parse_object(Parser* p, bool emit_tokens) noexcept {
    assert(p != nullptr);
    assert(p->src[p->pos] == '{');
    if (p->depth >= kJsonMaxDepth) return false;
    ++p->depth;
    if (emit_tokens) {
        if (!emit(p, TokenType::BeginObject, p->pos, 0)) { --p->depth; return false; }
    }
    ++p->pos;
    p->array_index[p->depth - 1] = -1;
    skip_ws(p);
    if (p->pos < p->src_len && p->src[p->pos] == '}') {
        if (emit_tokens) {
            if (!emit(p, TokenType::EndObject, p->pos, 0)) { --p->depth; return false; }
        }
        ++p->pos;
        --p->depth;
        return true;
    }
    while (p->pos < p->src_len) {
        skip_ws(p);
        if (p->pos >= p->src_len || p->src[p->pos] != '"') return false;
        int32_t ks, kl;
        if (!scan_string(p, &ks, &kl)) return false;
        // Decide whether this key's subtree is interesting.
        bool child_emit = emit_tokens;
        if (p->filter != nullptr) {
            path_push_key(p, reinterpret_cast<const char*>(p->src + ks), kl);
            child_emit = emit_tokens && path_is_prefix(p->filter,
                                                       p->path_stack, p->path_len);
        }
        if (child_emit) {
            if (!emit(p, TokenType::Key, ks, kl)) { if (p->filter) path_pop(p); --p->depth; return false; }
        }
        skip_ws(p);
        if (p->pos >= p->src_len || p->src[p->pos] != ':') {
            if (p->filter) path_pop(p); --p->depth; return false;
        }
        ++p->pos;
        skip_ws(p);
        if (!parse_value(p, child_emit)) {
            if (p->filter) path_pop(p); --p->depth; return false;
        }
        if (p->filter) path_pop(p);
        skip_ws(p);
        if (p->pos >= p->src_len) { --p->depth; return false; }
        if (p->src[p->pos] == ',') { ++p->pos; continue; }
        if (p->src[p->pos] == '}') {
            if (emit_tokens) {
                if (!emit(p, TokenType::EndObject, p->pos, 0)) { --p->depth; return false; }
            }
            ++p->pos;
            --p->depth;
            return true;
        }
        --p->depth;
        return false;
    }
    --p->depth;
    return false;
}

bool parse_array(Parser* p, bool emit_tokens) noexcept {
    assert(p != nullptr);
    assert(p->src[p->pos] == '[');
    if (p->depth >= kJsonMaxDepth) return false;
    ++p->depth;
    if (emit_tokens) {
        if (!emit(p, TokenType::BeginArray, p->pos, 0)) { --p->depth; return false; }
    }
    ++p->pos;
    int32_t idx = 0;
    skip_ws(p);
    if (p->pos < p->src_len && p->src[p->pos] == ']') {
        if (emit_tokens) {
            if (!emit(p, TokenType::EndArray, p->pos, 0)) { --p->depth; return false; }
        }
        ++p->pos;
        --p->depth;
        return true;
    }
    while (p->pos < p->src_len) {
        skip_ws(p);
        bool child_emit = emit_tokens;
        if (p->filter != nullptr) {
            path_push_index(p, idx);
            child_emit = emit_tokens && path_is_prefix(p->filter,
                                                       p->path_stack, p->path_len);
        }
        if (!parse_value(p, child_emit)) {
            if (p->filter) path_pop(p); --p->depth; return false;
        }
        if (p->filter) path_pop(p);
        ++idx;
        skip_ws(p);
        if (p->pos >= p->src_len) { --p->depth; return false; }
        if (p->src[p->pos] == ',') { ++p->pos; continue; }
        if (p->src[p->pos] == ']') {
            if (emit_tokens) {
                if (!emit(p, TokenType::EndArray, p->pos, 0)) { --p->depth; return false; }
            }
            ++p->pos;
            --p->depth;
            return true;
        }
        --p->depth;
        return false;
    }
    --p->depth;
    return false;
}

bool parse_value(Parser* p, bool emit_tokens) noexcept {
    assert(p != nullptr);
    skip_ws(p);
    if (p->pos >= p->src_len) return false;
    const uint8_t c = p->src[p->pos];
    if (c == '{') return parse_object(p, emit_tokens);
    if (c == '[') return parse_array(p, emit_tokens);
    if (c == '"') {
        int32_t s, l;
        if (!scan_string(p, &s, &l)) return false;
        if (emit_tokens) return emit(p, TokenType::String, s, l);
        return true;
    }
    if (c == 't') {
        if (!match_keyword(p, "true", 4)) return false;
        if (emit_tokens) return emit(p, TokenType::BoolTrue, p->pos - 4, 4);
        return true;
    }
    if (c == 'f') {
        if (!match_keyword(p, "false", 5)) return false;
        if (emit_tokens) return emit(p, TokenType::BoolFalse, p->pos - 5, 5);
        return true;
    }
    if (c == 'n') {
        if (!match_keyword(p, "null", 4)) return false;
        if (emit_tokens) return emit(p, TokenType::Null, p->pos - 4, 4);
        return true;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        int32_t s, l; bool is_float;
        if (!scan_number(p, &s, &l, &is_float)) return false;
        if (emit_tokens) {
            return emit(p, is_float ? TokenType::Float64 : TokenType::Int64, s, l);
        }
        return true;
    }
    return false;
}

}  // namespace

// ===========================================================================
// Public surface — build_index / build_index_filtered.
// ===========================================================================

static bool build_index_impl(const uint8_t* src, int32_t src_len,
                             const PathFilter* filter,
                             Arena* arena, StructuralIndex* out) noexcept {
    assert(arena != nullptr);
    assert(out != nullptr);
    if (src_len < 0) return false;
    if (src_len > 0 && src == nullptr) return false;

    // Worst-case tape: every non-whitespace byte is a structural single. We
    // size to src_len + 2 (for End plus a small slack). This bound holds for
    // legal JSON (each token covers ≥1 source byte).
    const int32_t cap = (src_len < 4 ? 8 : src_len + 2);
    Token* tokens = arena->allocate_array<Token>(static_cast<size_t>(cap));
    if (tokens == nullptr) return false;

    Parser p;
    memset(&p, 0, sizeof(p));
    p.src = src;
    p.src_len = src_len;
    p.tokens = tokens;
    p.token_cap = cap;
    p.filter = (filter != nullptr) ? static_cast<const PathFilterImpl*>(filter->impl)
                                   : nullptr;

    skip_ws(&p);
    if (p.pos >= p.src_len) {
        // Empty doc — emit only End.
        if (!emit(&p, TokenType::End, p.pos, 0)) return false;
        out->src = src;
        out->src_len = src_len;
        out->tokens = tokens;
        out->token_count = p.token_count;
        return true;
    }
    if (!parse_value(&p, /*emit_tokens=*/true)) return false;
    skip_ws(&p);
    if (p.pos != p.src_len) return false;  // trailing junk
    if (!emit(&p, TokenType::End, p.pos, 0)) return false;

    out->src = src;
    out->src_len = src_len;
    out->tokens = tokens;
    out->token_count = p.token_count;
    return true;
}

bool build_index(const uint8_t* src, int32_t src_len,
                 Arena* arena, StructuralIndex* out) noexcept {
    return build_index_impl(src, src_len, nullptr, arena, out);
}

bool build_index_filtered(const uint8_t* src, int32_t src_len,
                          const PathFilter* filter,
                          Arena* arena, StructuralIndex* out) noexcept {
    return build_index_impl(src, src_len, filter, arena, out);
}

// ===========================================================================
// Iterator surface.
// ===========================================================================

bool iter_init(const StructuralIndex* idx, Iterator* it) noexcept {
    assert(idx != nullptr);
    assert(it != nullptr);
    if (idx->tokens == nullptr || idx->token_count <= 0) return false;
    it->idx = idx;
    it->cursor = 0;
    it->_pad = 0;
    return true;
}

TokenType iter_peek(const Iterator* it) noexcept {
    assert(it != nullptr);
    assert(it->idx != nullptr);
    if (it->cursor < 0 || it->cursor >= it->idx->token_count) return TokenType::End;
    return it->idx->tokens[it->cursor].type;
}

bool iter_advance(Iterator* it) noexcept {
    assert(it != nullptr);
    assert(it->idx != nullptr);
    if (it->cursor >= it->idx->token_count) return false;
    ++it->cursor;
    return true;
}

bool iter_skip_to_close(Iterator* it) noexcept {
    assert(it != nullptr);
    assert(it->idx != nullptr);
    const int32_t n = it->idx->token_count;
    if (it->cursor >= n) return false;
    const Token* toks = it->idx->tokens;
    const TokenType t0 = toks[it->cursor].type;
    if (t0 != TokenType::BeginObject && t0 != TokenType::BeginArray) {
        // For scalars, advancing once is the "skip".
        ++it->cursor;
        return true;
    }
    int32_t depth = 0;
    int32_t i = it->cursor;
    while (i < n) {
        const TokenType t = toks[i].type;
        if (t == TokenType::BeginObject || t == TokenType::BeginArray) ++depth;
        else if (t == TokenType::EndObject || t == TokenType::EndArray) {
            --depth;
            if (depth == 0) { it->cursor = i + 1; return true; }
        }
        ++i;
    }
    return false;
}

bool iter_next_key(Iterator* it, StringView* key) noexcept {
    assert(it != nullptr);
    assert(it->idx != nullptr);
    assert(key != nullptr);
    if (it->cursor >= it->idx->token_count) return false;
    const Token& t = it->idx->tokens[it->cursor];
    if (t.type == TokenType::EndObject) return false;
    if (t.type != TokenType::Key) return false;
    const char* base = reinterpret_cast<const char*>(it->idx->src);
    const uint32_t len = static_cast<uint32_t>(t.length);
    memset(key, 0, sizeof(*key));
    key->length = len;
    if (len > 0) {
        const char* src = base + t.start;
        memcpy(key->prefix, src, (len < 4) ? len : 4);
        if (len <= 12) {
            if (len > 4) memcpy(key->inline_data, src + 4, len - 4);
        } else {
            // Spilled — caller resolves via (buf_idx, offset). For our v1
            // surface we encode offset only; buf_idx==0 means "the source".
            key->ref.buf_idx = 0;
            key->ref.offset = static_cast<uint32_t>(t.start);
        }
    }
    ++it->cursor;
    return true;
}

bool iter_string(const Iterator* it, StringView* out) noexcept {
    assert(it != nullptr);
    assert(it->idx != nullptr);
    assert(out != nullptr);
    if (it->cursor >= it->idx->token_count) return false;
    const Token& t = it->idx->tokens[it->cursor];
    if (t.type != TokenType::String && t.type != TokenType::Key) return false;
    const char* base = reinterpret_cast<const char*>(it->idx->src);
    const uint32_t len = static_cast<uint32_t>(t.length);
    memset(out, 0, sizeof(*out));
    out->length = len;
    if (len > 0) {
        const char* s = base + t.start;
        memcpy(out->prefix, s, (len < 4) ? len : 4);
        if (len <= 12) {
            if (len > 4) memcpy(out->inline_data, s + 4, len - 4);
        } else {
            out->ref.buf_idx = 0;
            out->ref.offset = static_cast<uint32_t>(t.start);
        }
    }
    return true;
}

bool iter_int64(const Iterator* it, int64_t* out) noexcept {
    assert(it != nullptr);
    assert(out != nullptr);
    if (it->cursor >= it->idx->token_count) return false;
    const Token& t = it->idx->tokens[it->cursor];
    if (t.type != TokenType::Int64 && t.type != TokenType::Float64) return false;
    if (t.length <= 0 || t.length > 64) return false;
    char buf[65];
    memcpy(buf, it->idx->src + t.start, static_cast<size_t>(t.length));
    buf[t.length] = '\0';
    char* end = nullptr;
    long long v = std::strtoll(buf, &end, 10);
    if (end != buf + t.length) return false;
    *out = static_cast<int64_t>(v);
    return true;
}

bool iter_float64(const Iterator* it, double* out) noexcept {
    assert(it != nullptr);
    assert(out != nullptr);
    if (it->cursor >= it->idx->token_count) return false;
    const Token& t = it->idx->tokens[it->cursor];
    if (t.type != TokenType::Int64 && t.type != TokenType::Float64) return false;
    if (t.length <= 0 || t.length > 64) return false;
    char buf[65];
    memcpy(buf, it->idx->src + t.start, static_cast<size_t>(t.length));
    buf[t.length] = '\0';
    char* end = nullptr;
    double v = std::strtod(buf, &end);
    if (end != buf + t.length) return false;
    *out = v;
    return true;
}

}  // namespace json
}  // namespace parse
}  // namespace bolt
