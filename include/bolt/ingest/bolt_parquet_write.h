// bolt/ingest/bolt_parquet_write.h — W-PQ-W: streaming Parquet writer.
//
// Mirror of bolt_parquet_read.h: produces files the existing reader
// (parquet_read_meta + parquet_read_row_group + parquet_read_file) can
// round-trip without round-trip-only divergence.
//
// Scope v1 (FLAT schemas only):
//   - file layout: "PAR1" + row_groups + Thrift FileMetaData + u32 LE len + "PAR1"
//   - encodings:   PLAIN only (no dictionary, no RLE / DELTA on values)
//   - codecs:      UNCOMPRESSED, SNAPPY (others -> false on open)
//   - types:       Int32, Int64, Float32, Float64, Utf8, Binary,
//                  Decimal128 (FLBA(16) BE two's-complement),
//                  Date32 (INT32), Timestamp[us] (INT64 isAdjustedToUTC=true),
//                  Bool (PLAIN bit-packed LE).
//   - nulls:       max_def_level == 1 column gets a hybrid-encoded def
//                  level stream prefixed to the page payload. Non-nullable
//                  columns omit def levels altogether.
//   - statistics:  per column chunk min/max/null_count when
//                  ParquetWriteOpts::emit_statistics is set; skipped
//                  otherwise. Encoded as Statistics fields 3,5,6 — the
//                  shape the reader's parse_statistics consumes first.
//
// Out of scope (open returns false, or the option is silently a no-op):
//   - Dictionary encoding, RLE / DELTA, nested types, LIST / MAP / STRUCT
//   - Page index, bloom filters, encryption
//   - GZIP / ZSTD / LZ4 codecs
//
// Tiger Style: noexcept everywhere, no exceptions / RTTI / smart pointers,
// PODs at the API edge, asserts >= 2 per non-trivial function, functions
// <= 70 lines, bounded loops. One new/delete pair lives at open/close for
// the opaque ParquetWriter; everything else is fixed-cap or std::vector
// confined to the at-close footer build (writer's outer edge, not hot).

#pragma once

#include <cstdint>

#include "bolt/bolt_types.h"      // BoltType, kMaxBatchColumns
#include "bolt/bolt_column.h"     // BoltBatch / BoltColumn

namespace bolt {
namespace ingest {
namespace parquet {

// One column's worth of writer-side schema. Fixed-size POD.
struct ParquetWriteColumn {
    char         name[64];      // NUL-terminated; truncated past 63 chars.
    BoltType     type;          // Source column type (see scope above).
    std::uint8_t precision;     // Decimal128 only.
    std::uint8_t scale;         // Decimal128 only.
    bool         nullable;      // OPTIONAL when true; REQUIRED otherwise.
    std::uint8_t _pad[5];       // explicit alignment / future use
};

// Writer options. POD; copied into the writer at open.
struct ParquetWriteOpts {
    ParquetWriteColumn columns[kMaxBatchColumns];
    std::uint32_t      n_columns;
    std::uint32_t      row_group_target_bytes;   // capped at 64 MiB internally
    std::uint8_t       compression;              // 0=none, 1=SNAPPY
                                                 // (2=GZIP, 3=ZSTD, 4=LZ4_RAW
                                                 //  rejected at open: false)
    bool               emit_statistics;
    bool               emit_bloom_filter;        // accepted, ignored (stub)
    bool               emit_page_index;          // accepted, ignored (stub)
    std::uint8_t       _pad[4];
};

// Opaque writer state. One new at open, one delete at close.
struct ParquetWriter;

// Open `path` for binary writing, validate opts, write the "PAR1" magic.
// Returns nullptr on bad path, unsupported codec / type, or failed open.
ParquetWriter* parquet_write_open(const char* path,
                                  const ParquetWriteOpts* opts) noexcept;

// Append one row group built from `batch`. `batch->num_cols` must match
// the writer's column count and each column's type / nullability must
// match the schema declared at open. Returns false on any mismatch or
// IO failure; the writer is left in an unspecified state — call close
// to release resources.
bool parquet_write_row_group(ParquetWriter* w,
                             const BoltBatch* batch) noexcept;

// Flush the Thrift FileMetaData footer + 4-byte length + "PAR1", close
// the file, and delete the writer. Returns false on IO failure.
// Always safe to call exactly once per successful open.
bool parquet_write_close(ParquetWriter* w) noexcept;

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
