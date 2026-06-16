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

enum class AvroType : uint8_t {
    kNull = 0, kBoolean = 1, kInt = 2, kLong = 3, kFloat = 4, kDouble = 5,
    kBytes = 6, kString = 7,
};

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
    bool     nullable;       // union with "null"
    uint8_t  _pad[2];
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

}  // namespace ingest
}  // namespace bolt
