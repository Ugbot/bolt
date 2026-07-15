// bolt/ingest/bolt_parquet_meta.h — W-PQ: zero-dependency Parquet footer
// metadata reader (hand-rolled Thrift Compact Protocol subset).
//
// Scope v1 (read path, FLAT schemas only):
//   - footer locate/verify: [..data..][metadata][u32 len LE]["PAR1"]
//   - FileMetaData -> schema (root + leaf columns), num_rows, row groups
//   - per column-chunk: codec, encodings seen, num_values, page offsets,
//     compressed/uncompressed sizes, min/max statistics bytes
// Anything outside the subset (nested schemas, unknown field types) is
// SKIPPED structurally (thrift lets us skip unknown fields safely) or
// rejected with `false` — never undefined behaviour on corrupt input:
// every read is bounds-checked against the footer slice.
//
// Tiger Style: POD outputs with fixed caps, noexcept, no allocation (the
// caller's structs are the storage), >=2 asserts per function, <=70-line
// functions, bounded loops everywhere (corrupt varints/lists cannot spin).

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"

namespace bolt {
namespace ingest {
namespace parquet {

// ---- thrift compact protocol primitives -----------------------------------

// Bounded cursor over the footer slice. All reads check remaining bytes.
struct TcCursor {
    const uint8_t* p;
    const uint8_t* end;
};

inline bool tc_u8(TcCursor* c, uint8_t* out) noexcept {
    assert(c != nullptr && out != nullptr);
    if (c->p >= c->end) return false;
    *out = *c->p++;
    return true;
}

// ULEB128 varint, bounded to 10 bytes (u64).
inline bool tc_varint(TcCursor* c, uint64_t* out) noexcept {
    assert(c != nullptr && out != nullptr);
    uint64_t v = 0;
    for (uint32_t shift = 0; shift < 70; shift += 7) {
        uint8_t b;
        if (!tc_u8(c, &b)) return false;
        v |= static_cast<uint64_t>(b & 0x7Fu) << shift;
        if ((b & 0x80u) == 0) { *out = v; return true; }
    }
    return false;   // >10 bytes: corrupt
}

inline bool tc_zigzag(TcCursor* c, int64_t* out) noexcept {
    assert(c != nullptr && out != nullptr);
    uint64_t u = 0;
    if (!tc_varint(c, &u)) return false;
    *out = static_cast<int64_t>((u >> 1) ^ (~(u & 1) + 1));
    return true;
}

// Compact-protocol element types.
enum : uint8_t {
    kTcStop = 0, kTcTrue = 1, kTcFalse = 2, kTcByte = 3, kTcI16 = 4,
    kTcI32 = 5, kTcI64 = 6, kTcDouble = 7, kTcBinary = 8, kTcList = 9,
    kTcSet = 10, kTcMap = 11, kTcStruct = 12,
};

// Field header: returns false at STOP or on corrupt input.
// *field_id accumulates (short-form deltas add; long form sets).
inline bool tc_field(TcCursor* c, int16_t* field_id, uint8_t* type) noexcept {
    assert(c != nullptr && field_id != nullptr && type != nullptr);
    uint8_t b;
    if (!tc_u8(c, &b)) return false;
    if (b == kTcStop) return false;
    const uint8_t delta = b >> 4;
    *type = b & 0x0Fu;
    if (delta != 0) {
        *field_id = static_cast<int16_t>(*field_id + delta);
        return true;
    }
    int64_t fid = 0;
    if (!tc_zigzag(c, &fid)) return false;
    *field_id = static_cast<int16_t>(fid);
    return true;
}

// List header: element type + size (bounded by caller-supplied cap).
inline bool tc_list(TcCursor* c, uint8_t* elem_type, uint32_t* size) noexcept {
    assert(c != nullptr && elem_type != nullptr && size != nullptr);
    uint8_t b;
    if (!tc_u8(c, &b)) return false;
    *elem_type = b & 0x0Fu;
    uint32_t n = b >> 4;
    if (n == 15u) {
        uint64_t v = 0;
        if (!tc_varint(c, &v) || v > 0x7FFFFFFFull) return false;
        n = static_cast<uint32_t>(v);
    }
    *size = n;
    return true;
}

// Binary/string: yields a view into the footer slice (no copy).
inline bool tc_binary(TcCursor* c, const uint8_t** out, uint32_t* len) noexcept {
    assert(c != nullptr && out != nullptr && len != nullptr);
    uint64_t n = 0;
    if (!tc_varint(c, &n)) return false;
    if (n > static_cast<uint64_t>(c->end - c->p)) return false;
    *out = c->p;
    *len = static_cast<uint32_t>(n);
    c->p += n;
    return true;
}

// Structurally skip one value of `type` (bounded recursion via depth cap —
// corrupt nesting cannot blow the stack).
bool tc_skip(TcCursor* c, uint8_t type, uint32_t depth) noexcept;

// ---- parquet metadata subset ----------------------------------------------

inline constexpr uint32_t kPqMaxColumns   = 128;   // flat leaf columns
                                                    // (bumped 64->128 for
                                                    // ClickBench's 105-col
                                                    // `hits` table; PqMeta
                                                    // stays arena-only, see
                                                    // bolt_parquet_read.cpp)
inline constexpr uint32_t kPqMaxRowGroups = 4096;
inline constexpr uint32_t kPqMaxNameBytes = 64;
inline constexpr uint32_t kPqMaxStatBytes = 64;    // min/max value bytes kept

// parquet::Type (physical).
enum class PqType : int32_t {
    Boolean = 0, Int32 = 1, Int64 = 2, Int96 = 3, Float = 4, Double = 5,
    ByteArray = 6, FixedLenByteArray = 7,
};

// parquet::CompressionCodec.
enum class PqCodec : int32_t {
    Uncompressed = 0, Snappy = 1, Gzip = 2, Lzo = 3, Brotli = 4, Lz4 = 5,
    Zstd = 6, Lz4Raw = 7,
};

// One leaf column of a FLAT schema.
struct PqColumn {
    char     name[kPqMaxNameBytes];   // NUL-terminated, truncated if longer
    PqType   physical;
    int32_t  type_length;             // FIXED_LEN_BYTE_ARRAY width
    int32_t  converted;               // ConvertedType, -1 = none
    int32_t  scale;                   // DECIMAL scale (converted/logical)
    int32_t  precision;               // DECIMAL precision
    uint8_t  optional;                // repetition: 1 = OPTIONAL (def levels)
    uint8_t  _pad[7];
};

// PqChunk::stats_flags bits (G2FEAT-21): record WHICH thrift Statistics
// field each kept stat came from. The modern `min_value`/`max_value`
// (fields 6/5) are defined against the column's logical sort order and a
// spec-compliant writer keeps them valid bounds even when truncated; the
// legacy `min`/`max` (fields 2/1) were written by old writers with
// inconsistent orderings for BYTE_ARRAY (signed-byte vs unsigned-lex), so
// string pruning must never trust them. Integer stats are safe from either
// source (signed order is the one defined order for INT32/INT64).
inline constexpr uint32_t kPqStatMinIsValueField = 1u << 0;  // from min_value
inline constexpr uint32_t kPqStatMaxIsValueField = 1u << 1;  // from max_value

// One column chunk inside a row group.
struct PqChunk {
    int64_t  data_page_offset;
    int64_t  dictionary_page_offset;  // 0 = none
    int64_t  num_values;
    int64_t  total_compressed_size;
    int64_t  total_uncompressed_size;
    PqCodec  codec;
    uint32_t stats_flags;             // kPqStat* bits (was padding)
    // Statistics (min_value/max_value preferred; legacy min/max fallback).
    // A stat longer than kPqMaxStatBytes is recorded as ABSENT (len 0) —
    // bolt never keeps a self-truncated stat, so a present stat is the
    // writer's full bytes (see copy_stat).
    uint8_t  min_bytes[kPqMaxStatBytes];
    uint8_t  max_bytes[kPqMaxStatBytes];
    uint32_t min_len;                 // 0 = absent
    uint32_t max_len;                 // 0 = absent
    int64_t  null_count;              // -1 = absent
};

struct PqRowGroup {
    int64_t  num_rows;
    int64_t  total_byte_size;
    uint32_t chunk_off;               // index into PqMeta::chunks
    uint32_t chunk_count;             // == column count for flat schemas
};

struct PqMeta {
    int64_t     num_rows;
    uint32_t    n_columns;
    uint32_t    n_row_groups;
    uint32_t    n_chunks;             // n_row_groups * n_columns
    int32_t     version;
    PqColumn    columns[kPqMaxColumns];
    PqRowGroup  row_groups[kPqMaxRowGroups];
    // chunks live OUT-OF-LINE (kPqMaxRowGroups * kPqMaxColumns PqChunk
    // would be ~100 MB inline): the caller supplies the array.
    PqChunk*    chunks;
    uint32_t    chunks_cap;
};

// Locate + verify the footer in a whole-file buffer. On success *meta_off /
// *meta_len bound the thrift FileMetaData slice. Rejects truncated files,
// bad magic, and lengths that escape the buffer.
bool pq_locate_footer(const uint8_t* buf, uint64_t len,
                      uint64_t* meta_off, uint32_t* meta_len) noexcept;

// Parse FileMetaData (the located slice) into *out. `out->chunks` /
// `chunks_cap` must be set by the caller beforehand. Returns false on any
// corrupt/unsupported shape (nested schema, too many columns/row groups,
// chunk overflow) — never UB on hostile input.
bool pq_parse_file_meta(const uint8_t* meta, uint32_t meta_len,
                        PqMeta* out) noexcept;

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
