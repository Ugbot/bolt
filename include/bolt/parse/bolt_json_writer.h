#pragma once
// ---------------------------------------------------------------------------
// bolt::parse::json — byte-faithful JSON writer / value builder.
//
// The WRITE side of bolt::parse::json. bolt_json.h is a read-only structural
// -index tape (build_index / iter_*); it has no way to emit JSON text. This
// header adds:
//
//   1. A small arena-backed value tree (JsonValue) — object / array / string
//      / number / bool / null — whose object members keep INSERTION ORDER
//      (there is NO key sorting anywhere in this file, by design).
//   2. `serialize()` — walk a value tree and emit compact JSON text into a
//      single arena buffer (two-pass: exact-length count, then one write).
//   3. `value_from_index()` — lift a parsed StructuralIndex into a value tree
//      whose scalars are VERBATIM byte slices into the original source.
//
// BYTE-FAITHFUL round-trip
// ------------------------
// serialize(value_from_index(build_index(src))) reproduces `src` byte-for-
// byte WHEN `src` is in this writer's canonical compact form: no insignificant
// whitespace, object members in file order. The guarantee rests on two facts:
//   * scalars (numbers, strings, true/false/null) are re-emitted from their
//     ORIGINAL source bytes — never reparsed and reformatted, so "1.50",
//     "1e10", "-0", and escape sequences inside strings survive unchanged;
//   * object member order is the tape order, never sorted.
// This is exactly what a prompt-cache-identity-preserving edit needs: parse,
// mutate one text node, re-serialize, and every untouched byte is identical,
// so a provider's cached prefix is not silently invalidated by key reordering
// or numeric reformatting.
//
// The one thing this mode does NOT preserve is insignificant whitespace that
// was present in the ORIGINAL between tokens (the tape discards it). Provider
// payloads are compact, so this is a non-issue in practice; a caller needing
// bit-exact reproduction of a pretty-printed document with arbitrary spacing
// must use a byte-offset splice against the original buffer instead.
//
// Conventions (match bolt_json.cpp):
//   - All allocation through bolt::Arena. No std::*. noexcept everywhere.
//   - Failure -> false / nullptr. No exceptions, no throw.
//   - Scalar/string/key bytes are zero-copy slices; for a tree produced by
//     value_from_index the SOURCE buffer must outlive the tree AND any
//     serialize() call that reads it.
//   - Caps: object/array nesting <= kJsonMaxDepth (shared with the reader).
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/parse/bolt_json.h"

namespace bolt {
namespace parse {
namespace json {

// How a JsonValue serialises. The distinction between the two string kinds is
// what makes byte-faithfulness possible: a string lifted from the tape carries
// its escape sequences already-encoded, so it must be emitted VERBATIM, while
// a string built from raw user bytes must be ESCAPED on the way out.
enum class ValueKind : uint8_t {
    Literal        = 0,  // raw bytes emitted as-is, NO quotes: numbers, true/false/null
    StringVerbatim = 1,  // inner bytes emitted as-is between quotes (escapes preserved)
    StringEscaped  = 2,  // inner bytes JSON-escaped on emit, between quotes
    Object         = 3,  // ordered member list
    Array          = 4,  // ordered element list
};

struct JsonValue;

// A member of an object (key != nullptr) or an element of an array
// (key == nullptr). Members form a singly-linked, tail-appended list so
// iteration order == insertion order.
struct JsonNode {
    JsonNode*   next;
    const char* key;      // object member key inner bytes; nullptr for array elements
    int32_t     key_len;
    int32_t     _pad;
    JsonValue*  value;
};

// A JSON value node. Cheap to allocate from the arena. Scalars use
// (bytes,len); containers use the (head,tail,count) member list.
struct JsonValue {
    ValueKind   kind;
    uint8_t     _pad[7];
    const char* bytes;    // Literal / String*: slice bytes (may be nullptr iff len==0)
    int32_t     len;      // Literal / String*: slice length
    int32_t     count;    // Object / Array: child count
    JsonNode*   head;     // Object / Array: first child (insertion order)
    JsonNode*   tail;     // Object / Array: last child (append cursor)
};

// ---------------------------------------------------------------------------
// Constructors. Each returns nullptr on arena exhaustion.
// ---------------------------------------------------------------------------
JsonValue* make_null  (Arena* arena) noexcept;
JsonValue* make_bool  (Arena* arena, bool v) noexcept;
JsonValue* make_int   (Arena* arena, int64_t v) noexcept;
// Non-finite doubles (NaN / +-Inf) have no JSON representation and are encoded
// as `null` (matching nlohmann's default), since emitting "nan"/"inf" would be
// invalid JSON. Finite doubles round-trip through "%.17g".
JsonValue* make_double(Arena* arena, double v) noexcept;

// make_string  — `bytes` is raw text; it is JSON-ESCAPED on serialize.
// make_string_raw — `bytes` is already valid JSON string inner content
//                   (escapes pre-encoded); emitted VERBATIM. This is the
//                   byte-faithful variant used to preserve a parsed string.
// make_literal — `bytes` is emitted verbatim with NO surrounding quotes;
//                caller guarantees it is a valid JSON number or keyword.
// None of these copy `bytes`; the buffer must outlive the value.
JsonValue* make_string    (Arena* arena, const char* bytes, int32_t len) noexcept;
JsonValue* make_string_raw(Arena* arena, const char* bytes, int32_t len) noexcept;
JsonValue* make_literal   (Arena* arena, const char* bytes, int32_t len) noexcept;

JsonValue* make_object(Arena* arena) noexcept;
JsonValue* make_array (Arena* arena) noexcept;

// ---------------------------------------------------------------------------
// Container mutation. Append preserves insertion order (no sorting, no dedup).
// object_add COPIES the key bytes into the arena (safe for transient keys);
// `key` must be valid JSON string inner content. Returns false on bad args
// (wrong kind / nullptr) or arena exhaustion.
// ---------------------------------------------------------------------------
bool object_add(Arena* arena, JsonValue* obj,
                const char* key, int32_t key_len, JsonValue* val) noexcept;
bool array_add (Arena* arena, JsonValue* arr, JsonValue* val) noexcept;

// First member whose key equals (key,key_len), or nullptr. O(members).
JsonValue* object_get(const JsonValue* obj,
                      const char* key, int32_t key_len) noexcept;

// ---------------------------------------------------------------------------
// In-place scalar mutation — retarget an existing node to a new string value.
// This is the ProxyTransform edit primitive: navigate to a text node, swap its
// bytes, re-serialize; every other node re-emits from its untouched slice.
// `bytes` is NOT copied.
// ---------------------------------------------------------------------------
void set_string    (JsonValue* v, const char* bytes, int32_t len) noexcept;  // escaped on emit
void set_string_raw(JsonValue* v, const char* bytes, int32_t len) noexcept;  // verbatim on emit

// ---------------------------------------------------------------------------
// Lift a parsed structural index into a value tree. Scalars (numbers, strings,
// booleans, null) become VERBATIM slices into idx->src, so serialize()
// reproduces them byte-for-byte. Requires a COMPLETE (unfiltered) index — a
// path-filtered index omits skipped subtrees and is rejected as malformed.
// Returns false on a null/empty index, a truncated tape, or arena exhaustion.
// ---------------------------------------------------------------------------
bool value_from_index(const StructuralIndex* idx, Arena* arena,
                      JsonValue** out_root) noexcept;

// ---------------------------------------------------------------------------
// Serialise a value tree to compact JSON text in one freshly-allocated arena
// buffer. Two passes over the tree: pass 1 computes the exact byte length,
// pass 2 fills a single allocation (no reallocation, no incidental
// whitespace). On success *out_bytes/*out_len describe the arena-owned buffer
// (NOT null-terminated). Returns false on null args, depth overflow
// (> kJsonMaxDepth), a size exceeding INT32_MAX, or arena exhaustion.
// ---------------------------------------------------------------------------
bool serialize(const JsonValue* root, Arena* arena,
               const uint8_t** out_bytes, int32_t* out_len) noexcept;

}  // namespace json
}  // namespace parse
}  // namespace bolt
