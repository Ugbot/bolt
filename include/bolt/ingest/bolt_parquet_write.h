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
//                  shape the reader's parse_statistics consumes first
//                  (and the modern min_value/max_value fields pyarrow /
//                  DuckDB trust for pruning). Coverage (G2FEAT-24):
//                    * null_count is ALWAYS emitted — 0 for REQUIRED
//                      columns (nulls structurally impossible), the
//                      counted value for OPTIONAL ones.
//                    * min/max for Int32/Date32, Int64/Timestamp,
//                      Float32, Float64, Bool (single byte 0/1), and
//                      Utf8/Binary (full bytes or omitted — never a
//                      truncated bound; see compute_stats_utf8).
//                    * Float32/Float64: NaN values are skipped (spec:
//                      NaN must not appear in bounds); an all-NaN chunk
//                      omits min/max. A zero min is written as -0.0 and
//                      a zero max as +0.0 so both signed zeros are
//                      inside the bounds (spec recommendation).
//                    * Decimal128: min/max still omitted (v1 gap).
//   - row groups:  one call to parquet_write_row_group appends one row
//                  group by default; setting
//                  ParquetWriteOpts::row_group_max_rows > 0 splits each
//                  call's batch into consecutive row groups of at most
//                  that many rows (scan-skip granularity control).
//
// Out of scope (open returns false, or the option is silently a no-op):
//   - Dictionary encoding, RLE / DELTA, nested types, LIST / MAP / STRUCT
//   - Page index (ColumnIndex/OffsetIndex — tracked as a follow-up; the
//     writer emits exactly one data page per column chunk, so chunk
//     Statistics already carry page-granularity bounds today),
//     bloom filters, encryption
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
    // G2FEAT-47: kMaxFixedColumns (256), decoupled from the raised in-memory
    // kMaxBatchColumns; kPwMaxColumns (impl) matches. A batch wider than this
    // cannot be Parquet-row-group-written — the exact prior 256-col behaviour.
    ParquetWriteColumn columns[kMaxFixedColumns];
    std::uint32_t      n_columns;
    std::uint32_t      row_group_target_bytes;   // capped at 64 MiB internally.
                                                 // NOTE: advisory only today —
                                                 // no byte-based splitting is
                                                 // performed; use
                                                 // row_group_max_rows for real
                                                 // rowgroup size control.
    std::uint8_t       compression;              // 0=none, 1=SNAPPY
                                                 // (2=GZIP, 3=ZSTD, 4=LZ4_RAW
                                                 //  rejected at open: false)
    bool               emit_statistics;
    bool               emit_bloom_filter;        // accepted, ignored (stub)
    bool               emit_page_index;          // accepted, ignored (stub)
    // G2FEAT-24: rowgroup size control. 0 (the zero-init default) keeps the
    // legacy contract — one parquet_write_row_group call == one row group.
    // >0: each call's batch is split into consecutive row groups of at most
    // this many rows (the last one carries the remainder). Occupies the old
    // _pad[4] bytes, so struct size / layout are unchanged.
    std::uint32_t      row_group_max_rows;
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
