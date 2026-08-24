// bolt/ingest/bolt_parquet_write.h — W-PQ-W: streaming Parquet writer.
//
// Mirror of bolt_parquet_read.h: produces files the existing reader
// (parquet_read_meta + parquet_read_row_group + parquet_read_file) can
// round-trip without round-trip-only divergence.
//
// Scope v1 (FLAT schemas only):
//   - file layout: "PAR1" + row_groups + [page index] + Thrift FileMetaData
//                  + u32 LE len + "PAR1"
//   - encodings:   PLAIN, RLE_DICTIONARY (the parquet-mr / Arrow default,
//                  selected by ParquetWriteOpts::use_dictionary),
//                  DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY,
//                  DELTA_BYTE_ARRAY and BYTE_STREAM_SPLIT -- the full set
//                  bolt's reader decodes, so anything bolt can read it can
//                  now also write. Chosen per column via
//                  ParquetWriteColumn::encoding (see PqWriteEncoding).
//                  A column whose dictionary would exceed
//                  dictionary_max_bytes falls back to PLAIN for the whole
//                  chunk -- never mid-chunk, so a chunk's data pages all
//                  carry one encoding.
//   - pages:       a column chunk is split into data pages of at most
//                  ParquetWriteOpts::data_page_target_bytes (default 1 MiB,
//                  matching parquet-mr / Arrow). Previously one page carried
//                  a whole chunk, which silently truncated the int32 page
//                  size fields past 2 GiB.
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
//   - DATA_PAGE_V2 (v1 data pages only)
//   - nested types, LIST / MAP / STRUCT
//   - bloom filters, encryption
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

// Per-column value encoding. `Auto` is the only value that consults
// ParquetWriteOpts::use_dictionary; naming an encoding explicitly overrides
// it for that column, which is how Arrow's column_encoding option behaves.
//
// An encoding that does not apply to the column's type is rejected at
// parquet_write_open (which returns nullptr) rather than silently ignored --
// a caller who asked for DELTA_BYTE_ARRAY on an integer column has a bug,
// and quietly writing PLAIN would hide it.
enum class PqWriteEncoding : std::uint8_t {
    Auto                 = 0,
    Plain                = 1,
    // Dictionary: DICTIONARY_PAGE + RLE_DICTIONARY. Falls back to PLAIN for
    // the whole chunk if the dictionary exceeds its ceiling -- so asking for
    // it is a preference, not a guarantee, exactly as in parquet-mr.
    Dictionary           = 2,
    // DELTA_BINARY_PACKED. INT32 / INT64 (and Date32 / Timestamp, which are
    // those physically). The integer encoding of the parquet V2 writer.
    DeltaBinaryPacked    = 3,
    // DELTA_LENGTH_BYTE_ARRAY: delta-packed lengths, then the bytes.
    DeltaLengthByteArray = 4,
    // DELTA_BYTE_ARRAY: incremental (prefix, suffix) encoding. The usual
    // choice for sorted or common-prefix strings.
    DeltaByteArray       = 5,
    // BYTE_STREAM_SPLIT: a pure transpose. FLOAT / DOUBLE, and since parquet
    // 2.9 the fixed-width integer and FLBA types (so Decimal128 too).
    // Increasingly the recommended encoding for floating point.
    ByteStreamSplit      = 6,
};

// One column's worth of writer-side schema. Fixed-size POD.
struct ParquetWriteColumn {
    char         name[64];      // NUL-terminated; truncated past 63 chars.
    BoltType     type;          // Source column type (see scope above).
    std::uint8_t precision;     // Decimal128 only.
    std::uint8_t scale;         // Decimal128 only.
    bool         nullable;      // OPTIONAL when true; REQUIRED otherwise.
    std::uint8_t encoding;      // PqWriteEncoding; 0 (Auto) is the default.
    std::uint8_t _pad[4];       // explicit alignment / future use
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
    // Emit a ColumnIndex + OffsetIndex per column chunk, in the index region
    // between the last row group and the footer, located by ColumnChunk
    // fields 4-7. Per-page min/max/null_count let a reader skip whole pages
    // on a predicate without decoding them. Costs a per-page statistics pass
    // at write time, so it is opt-in.
    bool               emit_page_index;
    // G2FEAT-24: rowgroup size control. 0 (the zero-init default) keeps the
    // legacy contract — one parquet_write_row_group call == one row group.
    // >0: each call's batch is split into consecutive row groups of at most
    // this many rows (the last one carries the remainder). Occupies the old
    // _pad[4] bytes, so struct size / layout are unchanged.
    std::uint32_t      row_group_max_rows;

    // ---- page splitting -------------------------------------------------
    // Maximum UNCOMPRESSED value bytes per data page. 0 selects the default
    // (kPwDefaultPageBytes, 1 MiB -- parquet-mr's and Arrow's default). There
    // is deliberately no "unlimited" setting: the PageHeader's size fields are
    // int32, so a single page carrying a multi-GiB chunk truncates them and
    // emits a corrupt file. Clamped internally to [4 KiB, 512 MiB].
    // A page always holds at least one row, so a single row wider than the
    // budget still writes (as its own page) rather than failing.
    std::uint32_t      data_page_target_bytes;

    // ---- dictionary encoding --------------------------------------------
    // When set, each column chunk is dictionary-encoded: a DICTIONARY_PAGE of
    // PLAIN distinct values followed by RLE_DICTIONARY data pages of indices.
    // This is what parquet-mr and Arrow emit by default and is typically a
    // large size win on low-cardinality columns.
    //
    // Fallback is per CHUNK, never mid-chunk: if the distinct values exceed
    // dictionary_max_bytes (or kPwMaxDictEntries) the chunk is rewritten
    // PLAIN, so every data page in a chunk shares one encoding. Deciding
    // mid-chunk is legal parquet but leaves a chunk whose pages disagree,
    // which is a well-known source of reader bugs; we decline to produce it.
    bool               use_dictionary;
    std::uint8_t       _pad2[3];
    // Dictionary size ceiling in bytes of PLAIN-encoded distinct values.
    // 0 selects the default (kPwDefaultDictBytes, 1 MiB -- Arrow's default).
    std::uint32_t      dictionary_max_bytes;
};

// Defaults referenced by the option comments above.
inline constexpr std::uint32_t kPwDefaultPageBytes = 1u << 20;   // 1 MiB
inline constexpr std::uint32_t kPwMinPageBytes     = 4u << 10;   // 4 KiB
inline constexpr std::uint32_t kPwMaxPageBytes     = 512u << 20; // 512 MiB
inline constexpr std::uint32_t kPwDefaultDictBytes = 1u << 20;   // 1 MiB
inline constexpr std::uint32_t kPwMaxDictEntries   = 1u << 20;

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

// ---------------------------------------------------------------------------
// Memory sink — the same writer with no filesystem involved.
//
// For a producer that SYNTHESIZES parquet per request (serving a file that was
// never stored), a temp file is the whole cost: two syscall-heavy copies and a
// path to clean up, for bytes that are about to be written to a socket.
//
// The bytes are identical to the file path's. Every offset in the footer comes
// from `file_pos`, which counts bytes rather than asking the sink where it is,
// so the sink choice cannot change the output. `test_bolt_parquet_write_mem`
// pins that as an equality, not an assumption.
// ---------------------------------------------------------------------------

// As `parquet_write_open`, accumulating into memory. Same validation, same
// "PAR1" prologue. Finish with `parquet_write_close_mem`, NOT
// `parquet_write_close` (which would drop the bytes on the floor).
// `reserve_bytes` pre-sizes the accumulator ONCE. Without it the buffer grows
// by repeated reallocation as row groups are appended -- for a synthesis path
// that produces a file per request, that is realloc churn and a memcpy of
// everything written so far on each growth. A caller that can estimate the
// output (rows x row width is enough; the encoding is PLAIN and uncompressed)
// should pass it. 0 means "grow as needed" and is always correct, just slower.
ParquetWriter* parquet_write_open_mem(const ParquetWriteOpts* opts,
                                      std::uint64_t reserve_bytes = 0) noexcept;

// Finish the file and hand back its bytes, copied into `arena` so they outlive
// the writer. Deletes the writer in every case, success or not — like
// `parquet_write_close`, exactly one call per successful open.
//
// Returns false (and writes nothing) if `w` came from `parquet_write_open`:
// those bytes already went to disk, and an empty buffer would be
// indistinguishable from a valid zero-row file.
bool parquet_write_close_mem(ParquetWriter* w, Arena* arena,
                             const std::uint8_t** out,
                             std::uint64_t* out_len) noexcept;

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
