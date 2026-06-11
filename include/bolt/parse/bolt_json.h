#pragma once
// ---------------------------------------------------------------------------
// bolt::parse::json — skip-aware JSON parser.
//
// Layer 1.3 of plan `this-was-a-freach-hashed-crab.md`. The architecture is
// lifted (not the code) from Fionn: a structural-index tape generated in a
// single pass over the source, an iterator with a `skip_to_close` primitive
// that advances over the tape (not the byte stream) at O(tokens-skipped),
// and a precompiled path filter so consumers only see subtrees they care
// about.
//
// Conventions:
//   - All allocation through bolt::Arena. No std::*.
//   - noexcept everywhere. Failure → false / nullptr / TokenType::End.
//   - Strings are zero-copy slices into the caller's source buffer; the
//     source must outlive the returned StructuralIndex.
//   - Numbers materialise lazily via iter_int64 / iter_float64.
//
// Caps:
//   - Nesting depth ≤ 64.
//   - PathFilter ≤ 1024 paths, each ≤ 256 bytes.
//
// SIMD: scalar fallback only in this wave. AVX2 / SSE4.2 structural-index
// scan is a TODO — see docs/research/json-skip-architecture.md.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"

namespace bolt {
namespace parse {
namespace json {

enum class TokenType : uint8_t {
    BeginObject = 0,
    EndObject   = 1,
    BeginArray  = 2,
    EndArray    = 3,
    Key         = 4,
    String      = 5,
    Int64       = 6,
    Float64     = 7,
    BoolTrue    = 8,
    BoolFalse   = 9,
    Null        = 10,
    End         = 11,
};

// Compact tape entry. Points back into the caller's source buffer.
struct Token {
    TokenType type;
    uint8_t   _pad[3];
    int32_t   start;   // byte offset into source
    int32_t   length;  // byte span (string / number); 0 for structural
};
static_assert(sizeof(Token) == 12, "Token is 12 bytes");
static_assert(alignof(Token) == 4, "Token aligns on 4 bytes");

struct StructuralIndex {
    const uint8_t* src;
    int32_t        src_len;
    Token*         tokens;
    int32_t        token_count;
};

// Caps for the PathFilter perfect-hash backing store.
static constexpr int32_t kJsonMaxPaths       = 1024;
static constexpr int32_t kJsonMaxPathBytes   = 256;
static constexpr int32_t kJsonMaxDepth       = 64;

// Opaque body lives in the arena. The struct itself is 16 bytes so callers
// can hold one by value without indirection in their hot paths.
struct PathFilter {
    const void* impl;
    int32_t     path_count;
    int32_t     _pad;
};
static_assert(sizeof(PathFilter) == 16, "PathFilter is 16 bytes");

// Iterator over a built index. Cheap to copy.
struct Iterator {
    const StructuralIndex* idx;
    int32_t                cursor;  // index into tokens[]
    int32_t                _pad;
};
static_assert(sizeof(Iterator) == 16, "Iterator is 16 bytes");

// Build the index over the entire source. Returns false on malformed JSON,
// invalid UTF-8, depth overflow, or arena exhaustion.
bool build_index(const uint8_t* src, int32_t src_len,
                 Arena* arena, StructuralIndex* out) noexcept;

// Build the index, but skip subtrees whose path is not a prefix of any path
// in `filter`. The skip happens at the structural level — the parser still
// reads bytes to find the matching close, but emits NO tokens for the skipped
// subtree. Filter == nullptr behaves like build_index.
bool build_index_filtered(const uint8_t* src, int32_t src_len,
                          const PathFilter* filter,
                          Arena* arena, StructuralIndex* out) noexcept;

// Compile a path-set. Each path must start with '/' and segments are split
// on '/'. `paths[i]` is `path_lens[i]` bytes long. Paths copied into the
// arena. Returns false on cap overflow, malformed input, or arena failure.
bool compile_paths(const char* const* paths, const int32_t* path_lens,
                   int32_t count, Arena* arena, PathFilter* out) noexcept;

// Iterator-style navigation.
bool      iter_init(const StructuralIndex* idx, Iterator* it) noexcept;
TokenType iter_peek(const Iterator* it) noexcept;
bool      iter_advance(Iterator* it) noexcept;
bool      iter_skip_to_close(Iterator* it) noexcept;
bool      iter_next_key(Iterator* it, StringView* key) noexcept;

// ---------------------------------------------------------------------------
// SAX — event-driven streaming parse (fionn's native mode). NO tape, NO
// arena: O(depth) stack memory, one pass over the bytes, events fired as
// they are scanned. Same validation as build_index (UTF-8 DFA, depth cap,
// trailing-junk rejection); the SIMD plain-run accelerator applies here
// too since both modes share one scanner.
//
// Any callback may be nullptr — that event is simply not delivered.
// A callback returning false aborts the parse (sax_parse returns false).
// String/key slices point into the caller's source buffer (zero-copy);
// escape sequences are NOT decoded. Numbers arrive materialised AND as
// raw slices, so consumers can keep fionn's number-as-byte-slice model.
// ---------------------------------------------------------------------------
struct SaxHandler {
    void* ctx;
    bool (*on_begin_object)(void* ctx) noexcept;
    bool (*on_end_object)(void* ctx) noexcept;
    bool (*on_begin_array)(void* ctx) noexcept;
    bool (*on_end_array)(void* ctx) noexcept;
    bool (*on_key)(void* ctx, const char* bytes, int32_t len) noexcept;
    bool (*on_string)(void* ctx, const char* bytes, int32_t len) noexcept;
    bool (*on_int64)(void* ctx, int64_t v,
                     const char* bytes, int32_t len) noexcept;
    bool (*on_float64)(void* ctx, double v,
                       const char* bytes, int32_t len) noexcept;
    bool (*on_bool)(void* ctx, bool v) noexcept;
    bool (*on_null)(void* ctx) noexcept;
};

bool sax_parse(const uint8_t* src, int32_t src_len,
               const SaxHandler* handler) noexcept;

// ---------------------------------------------------------------------------
// NDJSON — newline-delimited records (fionn's SimdLineSeparator role).
// Lines are split with the SWAR newline scan; blank lines are skipped;
// a trailing '\r' (CRLF input) is trimmed per record.
//
// ndjson_for_each: per record, a tape is built into `scratch` and handed
// to `on_record`; the scratch arena is RESET between records, so memory
// stays O(largest record), and the record tape is only valid inside the
// callback. With `ignore_errors`, records that fail to tokenize are
// counted into stats->skipped instead of failing the run. A callback
// returning false aborts (returns false).
//
// ndjson_for_each_sax: same record framing, but each record is delivered
// as SAX events — no arena at all. A handler-callback returning false
// aborts the whole run; tokenize failures honour `ignore_errors`.
// ---------------------------------------------------------------------------
struct NdjsonStats {
    int64_t records;   // records delivered
    int64_t skipped;   // records dropped by ignore_errors
};

using NdjsonRecordFn = bool (*)(void* ctx, const StructuralIndex* record,
                                int64_t record_index) noexcept;

bool ndjson_for_each(const uint8_t* src, int64_t src_len,
                     bool ignore_errors, Arena* scratch,
                     void* ctx, NdjsonRecordFn on_record,
                     NdjsonStats* out_stats) noexcept;

bool ndjson_for_each_sax(const uint8_t* src, int64_t src_len,
                         bool ignore_errors, const SaxHandler* handler,
                         NdjsonStats* out_stats) noexcept;

// Materialise the scalar at the cursor. The cursor is not advanced — caller
// uses iter_advance() afterward. Numbers parsed lazily here.
bool iter_string  (const Iterator* it, StringView* out) noexcept;
bool iter_int64   (const Iterator* it, int64_t*    out) noexcept;
bool iter_float64 (const Iterator* it, double*     out) noexcept;

}  // namespace json
}  // namespace parse
}  // namespace bolt
