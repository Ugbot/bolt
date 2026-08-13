// bolt/ingest/bolt_avro.h — Avro 1.11 Object Container File (OCF) reader+writer.
//
// Backs Iceberg manifests + manifest lists (Avro) and Schema Registry payloads.
// Scope of W1: the container format + the primitive/record/array/map/union
// decoders + the four standard OCF codecs (null / deflate / snappy / zstd).
//
// OCF layout (spec: avro.apache.org/docs/1.11.1/specification/#object-container-files):
//   magic  : "Obj" 0x01
//   header : an Avro-encoded map<string,bytes> metadata block carrying
//            "avro.schema" (JSON) + "avro.codec", then a 16-byte sync marker.
//   blocks : repeated { long object_count, long block_size, <serialized
//            objects, codec-compressed>, 16-byte sync }.
//
// Primitive encodings:
//   null   : 0 bytes
//   boolean: 1 byte (0/1)
//   int/long: zig-zag varint (long up to 10 bytes)
//   float  : 4 bytes little-endian IEEE-754
//   double : 8 bytes little-endian IEEE-754
//   bytes/string: long length prefix then raw bytes
//   record : fields concatenated in declared order
//   array/map: blocks of { long count (neg = count + size prefix), items }, 0 ends
//   union  : long branch index then the branch value
//   enum   : int symbol index
//   fixed  : N raw bytes
//
// Reader API: parse the header (schema JSON + codec), then iterate rows. Each
// row is delivered to a callback as a flat array of `AvroValue` (one per
// top-level record field) — a simpler shape than BoltBatch for W1; the
// lakehouse manifest reader maps these into columns at the call site.
//
// Tiger Style: PODs, Arena allocs, ≥2 asserts/fn, bounded loops, no exceptions.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_types.h"

namespace bolt {
namespace ingest {

static constexpr uint32_t kAvroMagicLen   = 4u;       // "Obj" 0x01
static constexpr uint32_t kAvroSyncLen     = 16u;
static constexpr uint32_t kAvroMaxFields   = 256u;    // top-level record fields
static constexpr uint32_t kAvroMaxName     = 128u;
// Extra OCF metadata entries a writer may add beside avro.schema/avro.codec.
// Iceberg uses five (schema, partition-spec, partition-spec-id,
// format-version, content); the cap leaves room without being unbounded.
static constexpr uint32_t kAvroMaxMetaEntries = 16u;

enum class AvroType : uint8_t {
    kNull = 0, kBoolean = 1, kInt = 2, kLong = 3, kFloat = 4, kDouble = 5,
    kBytes = 6, kString = 7,
    // Container types. These never produce a VALUE -- they decode as null --
    // but they are SKIPPED on the wire so the fields after them stay aligned.
    // Without them a schema containing an array was unreadable entirely.
    kArray = 8, kMap = 9,
};

// AvroField::item_type when the element of an array/map is not a primitive
// and not a modelled record (a union, or another container). Such a block can
// still be skipped when the writer emits the byte-size form; otherwise the
// decode FAILS rather than guessing an element width.
static constexpr uint8_t kAvroItemOpaque = 0xFFu;

// AvroField::item_type when the element is a RECORD whose fields are modelled.
// The element's flattened field descriptors occupy the `elem_fields` entries
// IMMEDIATELY FOLLOWING this one in the same field array, so an element's width
// is discovered by walking them — the only way to advance past a positive-count
// block, which is the form Avro's Java writer actually emits.
//
// Iceberg needs exactly this: its INT-keyed maps (column_sizes, value_counts,
// null_value_counts, lower_bounds, upper_bounds) are array<record<key,value>>,
// because an Avro map may only key on string.
static constexpr uint8_t kAvroItemRecord = 0xFEu;

// Element-record fields modelled per container. Bounded; a wider element fails
// the parse rather than truncating (a truncated element mis-computes the width
// and would mis-align every following field).
static constexpr uint32_t kAvroMaxElemFields = 32u;

enum class AvroCodec : uint8_t {
    kNull = 0, kDeflate = 1, kSnappy = 2, kZstd = 3, kUnknown = 4,
};

// One decoded scalar. Bytes/string point into the arena (decoded block).
struct AvroValue {
    AvroType type;
    bool     is_null;          // true ⇒ value absent (null union branch)
    uint8_t  _pad[2];
    union {
        int64_t  i64;          // int / long
        double   f64;          // float (widened) / double
        uint32_t b_dummy;
    } num;
    const uint8_t* bytes;      // bytes / string payload
    uint32_t       bytes_len;
    uint32_t       _pad2;
};

// One field of the flattened top-level record schema.
struct AvroField {
    char     name[kAvroMaxName];
    AvroType type;            // resolved primitive (union → underlying type)
    // Union with "null". This applies to CONTAINERS too: `union{null, array}`
    // is how Iceberg declares every optional repeated field (lower_bounds,
    // split_offsets, partitions, ...), so a nullable kArray/kMap is normal —
    // the branch index is read first and, when it selects null, the container
    // is absent from the wire entirely.
    bool     nullable;
    // Which union branch index encodes null. Avro writes the branch index
    // before the value, and BOTH orderings occur in the wild: ["null",T]
    // (the convention when a field has a null default) gives 0, ["T","null"]
    // gives 1. Only meaningful when `nullable`; a zeroed field therefore
    // means "null-first", matching the pre-existing behaviour exactly.
    uint8_t  null_branch;
    // Element type for kArray / kMap: a primitive AvroType, kAvroItemRecord,
    // or kAvroItemOpaque. Meaningless for other types.
    uint8_t  item_type;
    // ONLY when item_type == kAvroItemRecord: how many entries immediately
    // AFTER this one in the same field array describe the element record's
    // (flattened) fields. Those entries are NOT part of the enclosing record's
    // positional wire sequence — a decoder must step over them — and they
    // always decode to null, because an array element repeats per row and the
    // flat one-value-per-field model cannot carry N values. They exist so the
    // element's WIDTH is computable. Zero for every other field.
    uint16_t elem_fields;
};

// Parsed OCF header: schema fields + codec + sync marker.
struct AvroHeader {
    AvroField field[kAvroMaxFields];
    uint32_t  n_fields;
    AvroCodec codec;
    uint8_t   _pad[3];
    uint8_t   sync[kAvroSyncLen];
};

// Per-row callback. `vals` has `n` entries (== header n_fields). Return false
// to abort the scan. `ctx` is opaque.
using AvroRowFn = bool (*)(void* ctx, const AvroValue* vals, uint32_t n,
                           int64_t row_index) noexcept;

// Per-ELEMENT callback for a repeated field (array / map).
//
// The flat one-value-per-field row model cannot carry a repeated field's N
// values, so a container publishes as null in the row callback (and always
// has). This callback is where those values are delivered: it fires once per
// element, DURING the walk of the enclosing row, before that row's `AvroRowFn`
// runs.
//
//   field_index : index of the CONTAINER field in the header field array, so a
//                 caller can tell `lower_bounds` from `upper_bounds`.
//   elem_index  : 0-based position within the container, across all blocks.
//   vals/n_vals : for a RECORD element (item_type == kAvroItemRecord), one
//                 value per element field, in declared order — i.e. exactly
//                 the `elem_fields` descriptors that follow the container.
//                 For a PRIMITIVE element, exactly one value.
//                 For a MAP, the string key is prepended as vals[0], so a map
//                 record element yields 1 + elem_fields values.
//
// Iceberg needs precisely this: column_sizes / value_counts /
// null_value_counts / lower_bounds / upper_bounds are array<record<key,value>>
// (an Avro map may only key on string), and `split_offsets` / `equality_ids`
// are array<long>.
//
// ZERO-COPY LIFETIME: identical to AvroRowFn — bytes/string point into the
// decoded block buffer and are only valid for the duration of the call.
// Return false to abort the scan.
using AvroElemFn = bool (*)(void* ctx, uint32_t field_index, int64_t row_index,
                            int64_t elem_index, const AvroValue* vals,
                            uint32_t n_vals) noexcept;

// Parse the OCF header (magic + metadata + sync). Returns false on malformed
// input, an unsupported codec, or a schema we cannot flatten to primitives.
bool avro_read_header(const uint8_t* src, uint64_t src_len,
                      Arena* arena, AvroHeader* out,
                      uint64_t* body_offset) noexcept;

// Read every object block, decoding rows and invoking `on_row`. Decompresses
// blocks per the header codec (snappy/deflate/zstd require the matching codec
// to be compiled in — see bolt_{zstd,gzip}.h). Returns false on corruption or
// when a callback aborts; *out_rows gets the count delivered.
bool avro_read(const uint8_t* src, uint64_t src_len, Arena* arena,
               void* ctx, AvroRowFn on_row, int64_t* out_rows) noexcept;

// As `avro_read`, plus per-element delivery for repeated fields. `on_elem` may
// be null, in which case this is exactly `avro_read` (which is implemented as
// this call with a null `on_elem` — there is one walker, not two).
bool avro_read_ex(const uint8_t* src, uint64_t src_len, Arena* arena,
                  void* ctx, AvroRowFn on_row, AvroElemFn on_elem,
                  int64_t* out_rows) noexcept;

// ---------------------------------------------------------------------------
// Bare-datum surface — schema-driven, NO code generation.
//
// The OCF functions above assume a container file (magic + metadata + sync +
// framed blocks). Streaming carriers do not use that: a Kafka/Confluent value
// is `[magic 0x00][4-byte big-endian schema id][BARE datum]` — a single record
// body with no header and no block framing, whose schema is looked up out of
// band (registry, or supplied by the caller). The two entry points below are
// that path: parse a schema *as JSON text*, then decode one datum against it.
// Everything is resolved at runtime from the parsed field table, so a new
// schema needs no rebuild.
//
// Both share the OCF implementation internals — there is exactly one varint /
// union / primitive decoder in this translation unit, so the streaming and
// container paths cannot drift apart.
// ---------------------------------------------------------------------------

// Parse Avro schema JSON (exactly the text a Schema Registry returns for a
// subject version) into the flattened top-level field table. Writes at most
// `max_fields` entries and sets *out_n. Returns false when the JSON is
// malformed or resolves to zero fields — never a partial success.
//
// Union handling: a two-branch union with "null" marks the field `nullable`
// and records which branch index is the null one, so BOTH ["null",T] and
// [T,"null"] decode correctly.
bool avro_parse_schema(const uint8_t* json, uint32_t json_len,
                       AvroField* out_fields, uint32_t max_fields,
                       uint32_t* out_n) noexcept;

// Decode ONE bare datum: `n_fields` values written to out_vals[0..n_fields).
// *out_consumed (optional) receives the bytes read, so a caller can walk a
// buffer holding several datums back-to-back.
//
// ZERO-COPY LIFETIME: bytes/string values point INTO `src`, they are not
// copied — identical to the OCF row callback's contract. The caller must
// consume (or copy out) before `src` is reused. This is why no Arena is
// taken: the decoder allocates nothing.
//
// Fails closed on truncation or a malformed varint and never reads past
// src[src_len); a partially-filled out_vals must not be used when it
// returns false.
bool avro_decode_datum(const uint8_t* src, uint64_t src_len,
                       const AvroField* fields, uint32_t n_fields,
                       AvroValue* out_vals, uint64_t* out_consumed) noexcept;

// ---------------------------------------------------------------------------
// Writer — encode rows into a single-block OCF (codec = null) into a caller
// buffer. The schema is the flattened field list; each row is a contiguous
// AvroValue array of length n_fields. Sufficient for manifest emission in W1;
// codec-compressed writes follow once the read path proves out.
// ---------------------------------------------------------------------------

// Upper bound on the encoded size for `n_rows` rows of `n_fields` (worst case
// 10 bytes/long + value bytes). Conservative; caller sizes its buffer from it.
uint64_t avro_write_max_len(const AvroField* fields, uint32_t n_fields,
                            uint64_t total_value_bytes, int64_t n_rows) noexcept;

// Write an OCF with the given schema + rows into dst[0..*dst_len). On entry
// *dst_len is the capacity; on success it is the bytes written. `rows` is a
// row-major AvroValue matrix [n_rows][n_fields]. Returns false on overflow.
bool avro_write(const AvroField* fields, uint32_t n_fields,
                const AvroValue* rows, int64_t n_rows,
                const uint8_t sync[kAvroSyncLen],
                uint8_t* dst, uint64_t* dst_len) noexcept;

// One extra OCF metadata entry. `key` is NUL-terminated; the value is opaque
// bytes (Avro metadata values are `bytes`, not `string`).
struct AvroMetaKV {
    const char*    key;
    const uint8_t* val;
    uint32_t       val_len;
    uint32_t       _pad;
};

// As `avro_write`, but the caller supplies the schema JSON and any extra
// metadata entries. Both exist for Iceberg manifests, which the generated
// schema cannot express:
//
//   * The generated schema is a FLAT record named "r" carrying primitives and
//     no `field-id` annotations. Iceberg requires `manifest_entry` /
//     `manifest_file` with nested `data_file` + `partition` records and a
//     field-id on every field. The WIRE BYTES are unaffected by that nesting —
//     an Avro record's fields are inline and positional, so a flattened field
//     list encodes identically — but the reader must be told the real shape.
//   * A real `manifest_entry` schema measures ~3.8 KB for a minimal two-column
//     unpartitioned table, against `build_schema_json`'s 4096-byte buffer. The
//     caller's string is referenced, never copied into that buffer.
//   * Iceberg readers expect `schema`, `partition-spec`, `partition-spec-id`,
//     `format-version` and `content` alongside `avro.schema` / `avro.codec`.
//
// `schema_json` == nullptr falls back to the generated schema. Container-typed
// fields (kArray / kMap) are writable ONLY as a null union branch: the encoder
// emits the branch index and skips the value, which is exactly how Iceberg's
// optional repeated fields (`lower_bounds`, `split_offsets`, ...) are omitted.
// A non-null container value is rejected rather than mis-encoded.
bool avro_write_ex(const AvroField* fields, uint32_t n_fields,
                   const AvroValue* rows, int64_t n_rows,
                   const char* schema_json, uint32_t schema_len,
                   const AvroMetaKV* extra_meta, uint32_t n_extra_meta,
                   const uint8_t sync[kAvroSyncLen],
                   uint8_t* dst, uint64_t* dst_len) noexcept;

// Upper bound for `avro_write_ex`. `meta_bytes` is the total of the caller's
// schema JSON plus every extra metadata key and value; the fixed allowance in
// `avro_write_max_len` covers only the generated schema.
uint64_t avro_write_ex_max_len(const AvroField* fields, uint32_t n_fields,
                               uint64_t total_value_bytes, int64_t n_rows,
                               uint64_t meta_bytes) noexcept;

}  // namespace ingest
}  // namespace bolt
