// ---------------------------------------------------------------------------
// bolt::parse::json — skip-aware JSON parser implementation.
//
// Layer 1.3 (+ fionn-parity wave). Stages composed in this file:
//   1. structural-index scan — single linear pass over `src`. String
//      bodies ride a SIMD/SWAR plain-run accelerator (AVX2 when
//      compiled in, SWAR everywhere); UTF-8 validated DFA-style on the
//      slow path only.
//   2. sinks — the SAME scanner feeds two sinks via compile-time
//      dispatch: TapeSink (Token entries into the arena → tape) and
//      SaxSink (event callbacks, no tape, O(depth) memory — fionn's
//      native streaming mode).
//   3. iterator — cursor over tokens with O(tokens-skipped) skip_to_close.
//   4. path filter — hash table over slash-prefixed interest paths;
//      build_index_filtered traverses without emitting tokens for
//      subtrees that aren't a prefix of any interest path.
//   5. NDJSON — SWAR newline framing + per-record tape (scratch arena
//      reset per record) or per-record SAX (no arena).
// ---------------------------------------------------------------------------
#include "bolt/parse/bolt_json.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if BOLT_SIMD_AVX2
#include <immintrin.h>
#endif

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

// Initialise only the fields the parse reads before writing. Zeroing
// the whole Parser (memset) writes the ~17 KB path_stack on every call —
// a 100x write amplification for small NDJSON records that dominated
// the first bench run. path_stack / seg_start / array_index are always
// written before they are read (path_len / seg_count gate them).
BOLT_FORCE_INLINE void parser_init(Parser* p, const uint8_t* src,
                                   int32_t src_len, Token* tokens,
                                   int32_t token_cap,
                                   const PathFilterImpl* filter) noexcept {
    assert(p != nullptr);
    p->src         = src;
    p->src_len     = src_len;
    p->pos         = 0;
    p->tokens      = tokens;
    p->token_cap   = token_cap;
    p->token_count = 0;
    p->filter      = filter;
    p->seg_count   = 0;
    p->path_len    = 0;
    p->depth       = 0;
}

BOLT_FORCE_INLINE bool tape_emit(Parser* p, TokenType t, int32_t s, int32_t l) noexcept {
    assert(p != nullptr);
    if (p->token_count >= p->token_cap) return false;
    Token& tok = p->tokens[p->token_count++];
    tok.type = t;
    tok._pad[0] = 0; tok._pad[1] = 0; tok._pad[2] = 0;
    tok.start = s;
    tok.length = l;
    return true;
}

// Defined with the scanner helpers below; the SAX sink materialises
// numbers through the same slice parsers the tape iterators use.
bool parse_int64_slice(const uint8_t* src, int32_t start, int32_t length,
                       int64_t* out) noexcept;
bool parse_float64_slice(const uint8_t* src, int32_t start, int32_t length,
                         double* out) noexcept;

// ---------------------------------------------------------------------------
// Sinks — compile-time dispatch so ONE scanner serves both output modes
// (Bolt's X-macro/template rule: no runtime function-pointer dispatch on
// the hot path).
//
// TapeSink  → Token entries on the arena tape (build_index*).
// SaxSink   → event callbacks, no tape, O(depth) memory (sax_parse).
//             `aborted` distinguishes a consumer abort (callback said
//             stop) from malformed input — NDJSON ignore_errors must
//             not swallow consumer aborts.
// ---------------------------------------------------------------------------

struct TapeSink {
    BOLT_FORCE_INLINE bool emit(Parser* p, TokenType t, int32_t s,
                                int32_t l) noexcept {
        return tape_emit(p, t, s, l);
    }
};

struct SaxSink {
    const SaxHandler* h;
    const uint8_t*    src;
    bool              aborted;

    bool emit(Parser* p, TokenType t, int32_t s, int32_t l) noexcept {
        assert(h != nullptr);
        (void)p;
        const char* bytes = reinterpret_cast<const char*>(src) + s;
        bool ok = true;
        switch (t) {
            case TokenType::BeginObject:
                if (h->on_begin_object) ok = h->on_begin_object(h->ctx);
                break;
            case TokenType::EndObject:
                if (h->on_end_object) ok = h->on_end_object(h->ctx);
                break;
            case TokenType::BeginArray:
                if (h->on_begin_array) ok = h->on_begin_array(h->ctx);
                break;
            case TokenType::EndArray:
                if (h->on_end_array) ok = h->on_end_array(h->ctx);
                break;
            case TokenType::Key:
                if (h->on_key) ok = h->on_key(h->ctx, bytes, l);
                break;
            case TokenType::String:
                if (h->on_string) ok = h->on_string(h->ctx, bytes, l);
                break;
            case TokenType::Int64: {
                if (h->on_int64) {
                    int64_t v = 0;
                    if (!parse_int64_slice(src, s, l, &v)) return false;
                    ok = h->on_int64(h->ctx, v, bytes, l);
                }
                break;
            }
            case TokenType::Float64: {
                if (h->on_float64) {
                    double v = 0;
                    if (!parse_float64_slice(src, s, l, &v)) return false;
                    ok = h->on_float64(h->ctx, v, bytes, l);
                }
                break;
            }
            case TokenType::BoolTrue:
                if (h->on_bool) ok = h->on_bool(h->ctx, true);
                break;
            case TokenType::BoolFalse:
                if (h->on_bool) ok = h->on_bool(h->ctx, false);
                break;
            case TokenType::Null:
                if (h->on_null) ok = h->on_null(h->ctx);
                break;
            case TokenType::End:
                break;
        }
        if (!ok) aborted = true;
        return ok;
    }
};

BOLT_FORCE_INLINE void skip_ws(Parser* p) noexcept {
    assert(p != nullptr);
    while (p->pos < p->src_len) {
        const uint8_t c = p->src[p->pos];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') return;
        ++p->pos;
    }
}

// ---------------------------------------------------------------------------
// SIMD/SWAR plain-run accelerator (fionn-parity wave).
//
// Counts leading bytes that are "plain" string content: not '"', not
// '\\', not a control byte (< 0x20), not >= 0x80 (UTF-8 lead/cont).
// Plain bytes need no DFA step and no per-byte branching, and they are
// the overwhelming majority of real-world JSON string content — this
// is where fionn's skip-scanners earn their GiB/s.
//
// AVX2 path (when compiled with /arch:AVX2 or -mavx2) processes 32
// bytes per step with exact byte compares. The SWAR path processes 8
// bytes per step with Mycroft has-byte masks; Mycroft false positives
// can only appear in lanes AFTER a real hit (the borrow chain starts
// at a real-hit lane), so (a) hits == 0 proves the whole word plain and
// (b) ctz lands on a real hit. Both facts keep the fast path exact.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE int32_t string_plain_run(const uint8_t* BOLT_RESTRICT s,
                                           int32_t n) noexcept {
    assert(s != nullptr || n == 0);
    assert(n >= 0);
    int32_t i = 0;
#if BOLT_SIMD_AVX2
    const __m256i vq  = _mm256_set1_epi8('"');
    const __m256i vbs = _mm256_set1_epi8('\\');
    const __m256i vsp = _mm256_set1_epi8(0x20);
    while (i + 32 <= n) {
        const __m256i c = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(s + i));
        // Signed compare: bytes >= 0x80 are negative, so (0x20 > c) also
        // flags them — harmless, they're flagged via movemask(c) anyway.
        const __m256i hit = _mm256_or_si256(
            _mm256_or_si256(_mm256_cmpeq_epi8(c, vq),
                            _mm256_cmpeq_epi8(c, vbs)),
            _mm256_cmpgt_epi8(vsp, c));
        const uint32_t m =
            static_cast<uint32_t>(_mm256_movemask_epi8(hit)) |
            static_cast<uint32_t>(_mm256_movemask_epi8(c));
        if (m != 0) return i + bolt_ctz32(m);
        i += 32;
    }
#endif
    while (i + 8 <= n) {
        uint64_t w;
        memcpy(&w, s + i, 8);
        const uint64_t high  = w & 0x8080808080808080ull;
        const uint64_t quote = bolt_swar_has_byte_u64(w, '"');
        const uint64_t bsl   = bolt_swar_has_byte_u64(w, '\\');
        const uint64_t ctrl  =
            (w - 0x2020202020202020ull) & ~w & 0x8080808080808080ull;
        const uint64_t hits = high | quote | bsl | ctrl;
        if (hits != 0) return i + (bolt_ctz64(hits) >> 3);
        i += 8;
    }
    while (i < n) {
        const uint8_t c = s[i];
        if (c == '"' || c == '\\' || c < 0x20 || c >= 0x80) return i;
        ++i;
    }
    return i;
}

// Materialise a number slice. Shared by the tape iterators and the SAX
// sink — one parser, two consumers. Slices are bounded at 64 bytes
// (no legal JSON number the scanner emits is longer in practice; the
// bound keeps the stack copy fixed).
bool parse_int64_slice(const uint8_t* src, int32_t start, int32_t length,
                       int64_t* out) noexcept {
    assert(src != nullptr);
    assert(out != nullptr);
    if (length <= 0 || length > 64) return false;
    char buf[65];
    memcpy(buf, src + start, static_cast<size_t>(length));
    buf[length] = '\0';
    char* end = nullptr;
    const long long v = std::strtoll(buf, &end, 10);
    if (end != buf + length) return false;
    *out = static_cast<int64_t>(v);
    return true;
}

bool parse_float64_slice(const uint8_t* src, int32_t start, int32_t length,
                         double* out) noexcept {
    assert(src != nullptr);
    assert(out != nullptr);
    if (length <= 0 || length > 64) return false;
    char buf[65];
    memcpy(buf, src + start, static_cast<size_t>(length));
    buf[length] = '\0';
    char* end = nullptr;
    const double v = std::strtod(buf, &end);
    if (end != buf + length) return false;
    *out = v;
    return true;
}

// Consume a JSON string starting at `*p->pos == '"'`. Sets out_start /
// out_length to the bytes inside the quotes (exclusive of the quotes).
// Validates UTF-8 and rejects bare control bytes < 0x20. Plain ASCII
// runs are skipped via string_plain_run (SIMD/SWAR); the byte-at-a-time
// DFA path only runs for quotes, escapes, controls, and UTF-8 bytes.
bool scan_string(Parser* p, int32_t* out_start, int32_t* out_length) noexcept {
    assert(p != nullptr);
    assert(p->pos < p->src_len && p->src[p->pos] == '"');
    ++p->pos;  // opening quote
    const int32_t start = p->pos;
    uint8_t state = kUtf8Accept;
    while (p->pos < p->src_len) {
        if (state == kUtf8Accept) {
            p->pos += string_plain_run(p->src + p->pos, p->src_len - p->pos);
            if (p->pos >= p->src_len) break;   // unterminated
        }
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

// SWAR digit-run: count leading ASCII digits in s[0..n). A byte b is a
// digit iff (b ^ 0x30) <= 9; flag = ((x + 0x06) | x) & 0xF0 per lane.
// The x-term is carry-free and exact; the (x + 0x06) carry can only
// contaminate lanes ABOVE a lane whose own x-term already flags, so
// flags == 0 proves all-digits and ctz lands on a real non-digit.
BOLT_FORCE_INLINE int32_t digit_run(const uint8_t* BOLT_RESTRICT s,
                                    int32_t n) noexcept {
    assert(s != nullptr || n == 0);
    int32_t i = 0;
    while (i + 8 <= n) {
        uint64_t w;
        memcpy(&w, s + i, 8);
        const uint64_t x = w ^ 0x3030303030303030ull;
        const uint64_t flags =
            ((x + 0x0606060606060606ull) | x) & 0xF0F0F0F0F0F0F0F0ull;
        if (flags != 0) return i + (bolt_ctz64(flags) >> 3);
        i += 8;
    }
    while (i < n && s[i] >= '0' && s[i] <= '9') ++i;
    return i;
}

bool scan_number(Parser* p, int32_t* out_start, int32_t* out_length, bool* is_float) noexcept {
    assert(p != nullptr);
    const int32_t start = p->pos;
    *is_float = false;
    if (p->src[p->pos] == '-') ++p->pos;
    if (p->pos >= p->src_len) return false;
    {
        const int32_t run = digit_run(p->src + p->pos, p->src_len - p->pos);
        if (run == 0) return false;
        p->pos += run;
    }
    if (p->pos < p->src_len && p->src[p->pos] == '.') {
        *is_float = true;
        ++p->pos;
        const int32_t run = digit_run(p->src + p->pos, p->src_len - p->pos);
        if (run == 0) return false;
        p->pos += run;
    }
    if (p->pos < p->src_len && (p->src[p->pos] == 'e' || p->src[p->pos] == 'E')) {
        *is_float = true;
        ++p->pos;
        if (p->pos < p->src_len && (p->src[p->pos] == '+' || p->src[p->pos] == '-')) ++p->pos;
        const int32_t run = (p->pos < p->src_len)
            ? digit_run(p->src + p->pos, p->src_len - p->pos) : 0;
        if (run == 0) return false;
        p->pos += run;
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

// Forward decl. Templated on the sink so the tape and SAX modes share
// one scanner with zero runtime dispatch.
template <typename Sink>
bool parse_value(Parser* p, Sink* sink, bool emit_tokens) noexcept;

template <typename Sink>
bool parse_object(Parser* p, Sink* sink, bool emit_tokens) noexcept {
    assert(p != nullptr);
    assert(p->src[p->pos] == '{');
    if (p->depth >= kJsonMaxDepth) return false;
    ++p->depth;
    if (emit_tokens) {
        if (!sink->emit(p, TokenType::BeginObject, p->pos, 0)) { --p->depth; return false; }
    }
    ++p->pos;
    p->array_index[p->depth - 1] = -1;
    skip_ws(p);
    if (p->pos < p->src_len && p->src[p->pos] == '}') {
        if (emit_tokens) {
            if (!sink->emit(p, TokenType::EndObject, p->pos, 0)) { --p->depth; return false; }
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
            if (!sink->emit(p, TokenType::Key, ks, kl)) { if (p->filter) path_pop(p); --p->depth; return false; }
        }
        skip_ws(p);
        if (p->pos >= p->src_len || p->src[p->pos] != ':') {
            if (p->filter) path_pop(p); --p->depth; return false;
        }
        ++p->pos;
        skip_ws(p);
        if (!parse_value(p, sink, child_emit)) {
            if (p->filter) path_pop(p); --p->depth; return false;
        }
        if (p->filter) path_pop(p);
        skip_ws(p);
        if (p->pos >= p->src_len) { --p->depth; return false; }
        if (p->src[p->pos] == ',') { ++p->pos; continue; }
        if (p->src[p->pos] == '}') {
            if (emit_tokens) {
                if (!sink->emit(p, TokenType::EndObject, p->pos, 0)) { --p->depth; return false; }
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

template <typename Sink>
bool parse_array(Parser* p, Sink* sink, bool emit_tokens) noexcept {
    assert(p != nullptr);
    assert(p->src[p->pos] == '[');
    if (p->depth >= kJsonMaxDepth) return false;
    ++p->depth;
    if (emit_tokens) {
        if (!sink->emit(p, TokenType::BeginArray, p->pos, 0)) { --p->depth; return false; }
    }
    ++p->pos;
    int32_t idx = 0;
    skip_ws(p);
    if (p->pos < p->src_len && p->src[p->pos] == ']') {
        if (emit_tokens) {
            if (!sink->emit(p, TokenType::EndArray, p->pos, 0)) { --p->depth; return false; }
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
        if (!parse_value(p, sink, child_emit)) {
            if (p->filter) path_pop(p); --p->depth; return false;
        }
        if (p->filter) path_pop(p);
        ++idx;
        skip_ws(p);
        if (p->pos >= p->src_len) { --p->depth; return false; }
        if (p->src[p->pos] == ',') { ++p->pos; continue; }
        if (p->src[p->pos] == ']') {
            if (emit_tokens) {
                if (!sink->emit(p, TokenType::EndArray, p->pos, 0)) { --p->depth; return false; }
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

template <typename Sink>
bool parse_value(Parser* p, Sink* sink, bool emit_tokens) noexcept {
    assert(p != nullptr);
    skip_ws(p);
    if (p->pos >= p->src_len) return false;
    const uint8_t c = p->src[p->pos];
    if (c == '{') return parse_object(p, sink, emit_tokens);
    if (c == '[') return parse_array(p, sink, emit_tokens);
    if (c == '"') {
        int32_t s, l;
        if (!scan_string(p, &s, &l)) return false;
        if (emit_tokens) return sink->emit(p, TokenType::String, s, l);
        return true;
    }
    if (c == 't') {
        if (!match_keyword(p, "true", 4)) return false;
        if (emit_tokens) return sink->emit(p, TokenType::BoolTrue, p->pos - 4, 4);
        return true;
    }
    if (c == 'f') {
        if (!match_keyword(p, "false", 5)) return false;
        if (emit_tokens) return sink->emit(p, TokenType::BoolFalse, p->pos - 5, 5);
        return true;
    }
    if (c == 'n') {
        if (!match_keyword(p, "null", 4)) return false;
        if (emit_tokens) return sink->emit(p, TokenType::Null, p->pos - 4, 4);
        return true;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        int32_t s, l; bool is_float;
        if (!scan_number(p, &s, &l, &is_float)) return false;
        if (emit_tokens) {
            return sink->emit(p, is_float ? TokenType::Float64 : TokenType::Int64, s, l);
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
    parser_init(&p, src, src_len, tokens, cap,
                (filter != nullptr)
                    ? static_cast<const PathFilterImpl*>(filter->impl)
                    : nullptr);

    TapeSink sink{};
    skip_ws(&p);
    if (p.pos >= p.src_len) {
        // Empty doc — emit only End.
        if (!tape_emit(&p, TokenType::End, p.pos, 0)) return false;
        out->src = src;
        out->src_len = src_len;
        out->tokens = tokens;
        out->token_count = p.token_count;
        return true;
    }
    if (!parse_value(&p, &sink, /*emit_tokens=*/true)) return false;
    skip_ws(&p);
    if (p.pos != p.src_len) return false;  // trailing junk
    if (!tape_emit(&p, TokenType::End, p.pos, 0)) return false;

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
// SAX surface — event-driven streaming parse, no tape, no arena.
// ===========================================================================

namespace {

// Internal tri-state so NDJSON can tell "malformed record" (skippable
// under ignore_errors) apart from "consumer aborted" (never skippable).
enum class SaxResult : uint8_t { Ok = 0, Malformed = 1, Aborted = 2 };

SaxResult sax_parse_impl(const uint8_t* src, int32_t src_len,
                         const SaxHandler* handler) noexcept {
    assert(handler != nullptr);
    if (src_len < 0) return SaxResult::Malformed;
    if (src_len > 0 && src == nullptr) return SaxResult::Malformed;

    Parser p;
    // No tape: tokens/token_cap null/0 — TapeSink is never invoked.
    parser_init(&p, src, src_len, nullptr, 0, nullptr);

    SaxSink sink{handler, src, false};
    skip_ws(&p);
    if (p.pos >= p.src_len) return SaxResult::Ok;   // empty doc, no events
    if (!parse_value(&p, &sink, /*emit_tokens=*/true)) {
        return sink.aborted ? SaxResult::Aborted : SaxResult::Malformed;
    }
    skip_ws(&p);
    if (p.pos != p.src_len) return SaxResult::Malformed;  // trailing junk
    return SaxResult::Ok;
}

// SWAR newline scan — the SimdLineSeparator role. Returns the index of
// the next '\n' at/after `pos`, or `len` when none remains. Mycroft's
// first hit is exact, so the returned index is always a real newline.
int64_t find_newline(const uint8_t* s, int64_t len, int64_t pos) noexcept {
    assert(s != nullptr || len == 0);
    assert(pos >= 0);
    while (pos + 8 <= len) {
        uint64_t w;
        memcpy(&w, s + pos, 8);
        const int idx = bolt_swar_find_byte_u64(w, '\n');
        if (idx >= 0) return pos + idx;
        pos += 8;
    }
    while (pos < len && s[pos] != '\n') ++pos;
    return pos;
}

// Trim one NDJSON line to its record payload: drop leading/trailing
// spaces, tabs, and a trailing '\r' (CRLF input). Returns false when
// the line is blank.
bool trim_line(const uint8_t* s, int64_t* b, int64_t* t) noexcept {
    assert(s != nullptr);
    assert(b != nullptr && t != nullptr);
    while (*b < *t && (s[*b] == ' ' || s[*b] == '\t' || s[*b] == '\r')) ++*b;
    while (*t > *b && (s[*t - 1] == ' ' || s[*t - 1] == '\t' ||
                       s[*t - 1] == '\r')) --*t;
    return *b < *t;
}

}  // namespace

bool sax_parse(const uint8_t* src, int32_t src_len,
               const SaxHandler* handler) noexcept {
    if (handler == nullptr) return false;
    return sax_parse_impl(src, src_len, handler) == SaxResult::Ok;
}

// ===========================================================================
// NDJSON surface.
// ===========================================================================

bool ndjson_for_each(const uint8_t* src, int64_t src_len,
                     bool ignore_errors, Arena* scratch,
                     void* ctx, NdjsonRecordFn on_record,
                     NdjsonStats* out_stats) noexcept {
    assert(scratch != nullptr);
    assert(on_record != nullptr);
    if (src_len < 0) return false;
    if (src_len > 0 && src == nullptr) return false;
    NdjsonStats local{0, 0};
    NdjsonStats* st = (out_stats != nullptr) ? out_stats : &local;
    st->records = 0;
    st->skipped = 0;

    int64_t pos = 0;
    while (pos < src_len) {
        const int64_t eol = find_newline(src, src_len, pos);
        int64_t b = pos;
        int64_t t = eol;
        pos = eol + 1;
        if (!trim_line(src, &b, &t)) continue;
        if (t - b > INT32_MAX) return false;
        scratch->reset();   // record tape valid only inside the callback
        StructuralIndex rec;
        if (!build_index(src + b, static_cast<int32_t>(t - b),
                         scratch, &rec)) {
            if (!ignore_errors) return false;
            st->skipped += 1;
            continue;
        }
        if (!on_record(ctx, &rec, st->records)) return false;
        st->records += 1;
    }
    return true;
}

bool ndjson_for_each_sax(const uint8_t* src, int64_t src_len,
                         bool ignore_errors, const SaxHandler* handler,
                         NdjsonStats* out_stats) noexcept {
    assert(handler != nullptr);
    if (src_len < 0) return false;
    if (src_len > 0 && src == nullptr) return false;
    NdjsonStats local{0, 0};
    NdjsonStats* st = (out_stats != nullptr) ? out_stats : &local;
    st->records = 0;
    st->skipped = 0;

    int64_t pos = 0;
    while (pos < src_len) {
        const int64_t eol = find_newline(src, src_len, pos);
        int64_t b = pos;
        int64_t t = eol;
        pos = eol + 1;
        if (!trim_line(src, &b, &t)) continue;
        if (t - b > INT32_MAX) return false;
        const SaxResult r = sax_parse_impl(
            src + b, static_cast<int32_t>(t - b), handler);
        if (r == SaxResult::Aborted) return false;
        if (r == SaxResult::Malformed) {
            if (!ignore_errors) return false;
            st->skipped += 1;
            continue;
        }
        st->records += 1;
    }
    return true;
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
    return parse_int64_slice(it->idx->src, t.start, t.length, out);
}

bool iter_float64(const Iterator* it, double* out) noexcept {
    assert(it != nullptr);
    assert(out != nullptr);
    if (it->cursor >= it->idx->token_count) return false;
    const Token& t = it->idx->tokens[it->cursor];
    if (t.type != TokenType::Int64 && t.type != TokenType::Float64) return false;
    return parse_float64_slice(it->idx->src, t.start, t.length, out);
}

}  // namespace json
}  // namespace parse
}  // namespace bolt
