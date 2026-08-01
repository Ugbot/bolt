// bolt/ingest/bolt_parquet_read.h — W-PQ increment 2: Parquet page reader.
//
// Decodes the FLAT-schema subset located by bolt_parquet_meta.h into
// BoltBatch columns (caller Arena owns all storage).
//
// Scope v1:
//   - pages:     DATA_PAGE (v1) + DICTIONARY_PAGE. DATA_PAGE_V2 rejected.
//   - codecs:    UNCOMPRESSED, SNAPPY (bolt_snappy.h).
//   - encodings: PLAIN; RLE/bit-packed hybrid for definition levels
//                (max_def_level <= 1), for RLE_DICTIONARY /
//                PLAIN_DICTIONARY indices (bit-width byte prefix), and
//                for RLE-encoded BOOLEAN data pages (u32 len prefix).
//   - types:     INT64 -> Int64 (DECIMAL converted -> Decimal64 mantissa),
//                INT32 -> Int32 (DATE converted -> Date32),
//                DOUBLE -> Float64,
//                FLOAT -> Float64 (f32 widened at decode; dictionary
//                  entries converted once at dict-page decode — G2FEAT-346),
//                BOOLEAN -> Int64 0/1 (PLAIN bit-packed LSB-first, or an
//                  RLE data page),
//                BYTE_ARRAY -> Utf8 StringViews (inline <= 12 bytes; longer
//                  values spill into ONE per-column overflow buffer,
//                  exposed via BoltColumn::str_overflow_base),
//                FIXED_LEN_BYTE_ARRAY DECIMAL(p<=18) -> Decimal64 (W-DEC:
//                  big-endian two's-complement mantissa -> int64,
//                  decimal_scale stamped on the column); p>18 -> Decimal128.
//   - OPTIONAL columns: validity bitmap built from definition levels.
//     All-valid shortcut: when every chunk of a column reports
//     statistics null_count == 0 the bitmap is not allocated at all.
//
// Safety contract: every read is bounds-checked against the file slice;
// corrupt/truncated/hostile input returns false, never UB (fuzzed under
// ASAN in tests/test_bolt_parquet_read.cpp).
//
// Tiger Style: noexcept, no allocation outside the caller Arena, fixed
// caps, bounded loops, >=2 asserts and <=70 lines per function.

#pragma once

#include <cstdint>

#include "bolt/bolt_types.h"
#include "bolt/ingest/bolt_parquet_meta.h"

namespace bolt {

class Arena;
struct BoltBatch;
struct BoltColumn;

namespace ingest {
namespace parquet {

// Map one parquet leaf column to its Bolt materialization (FLOAT widens
// to Float64, BOOLEAN lands as Int64 0/1). Returns false when the column
// shape is outside the supported subset (INT96, non-DECIMAL FLBA, INT32
// DECIMAL, FLBA wider than 16 bytes).
bool parquet_map_type(const PqColumn* col, BoltType* out_type,
                      uint8_t* out_scale) noexcept;

// Locate the footer and parse FileMetaData. The chunk table behind
// out->chunks is allocated from `arena` (sized 4096 first, retried once
// at the kPqMaxRowGroups * kPqMaxColumns hard cap for chunk-heavy files).
bool parquet_read_meta(const uint8_t* buf, uint64_t len, Arena* arena,
                       PqMeta* out) noexcept;

// Decode ONE row group into caller-preallocated column descriptors
// (out_cols[meta->n_columns]; the structs are descriptors only — all
// data buffers come from `arena`). *out_rows receives the group's row
// count. Designed so a scheduler can submit_range over row groups:
// every call touches disjoint output storage.
bool parquet_read_row_group(const uint8_t* buf, uint64_t len,
                            const PqMeta* meta, uint32_t row_group,
                            Arena* arena, BoltColumn* out_cols,
                            int64_t* out_rows) noexcept;

// Projection-pushdown variant (G2FEAT-8): decode ONLY the columns named by
// `col_idx[0..n_idx)` (file-schema indices, caller-deduplicated) from ONE
// row group. out_cols is indexed by PROJECTION position — out_cols[j] holds
// the column for col_idx[j] — so a scan operator decodes exactly the
// referenced columns into a dense output. Same disjoint-output contract as
// parquet_read_row_group; that function is now the n_idx == n_columns
// identity case. Rejects out-of-range indices and n_idx == 0.
bool parquet_read_row_group_cols(const uint8_t* buf, uint64_t len,
                                 const PqMeta* meta, uint32_t row_group,
                                 const uint16_t* col_idx, uint32_t n_idx,
                                 Arena* arena, BoltColumn* out_cols,
                                 int64_t* out_rows) noexcept;

// G2FEAT-49: resumable single-column page decode. Decodes WHOLE data pages of
// `col` in `row_group` from byte offset `start_off` (0 = first data page) until
// >= `max_rows` rows (whole pages, may overshoot), into a FRESH Flat `out_col`
// sized to exactly the decoded rows; returns the row count in *out_rows and the
// next undecoded page offset in *next_off (0 = chunk exhausted). Walk `next_off`
// to stream a chunk in bounded sub-chunks. v1: PLAIN chunks only (returns false
// on a dictionary page). Byte-exact (value-level) vs a whole-chunk decode of the
// same pages. Enables bounded per-worker footprint for wide single-column scans.
bool parquet_read_col_chunk_pages(const uint8_t* buf, uint64_t len,
                                  const PqMeta* meta, uint32_t row_group,
                                  uint16_t col, uint64_t start_off,
                                  int64_t max_rows, Arena* arena,
                                  BoltColumn* out_col, int64_t* out_rows,
                                  uint64_t* next_off) noexcept;

// Whole file: locate + parse meta, allocate full-length columns, then
// decode every row group into its row offset (serial v1; the row-group
// function above is the future parallel unit).
bool parquet_read_file(const uint8_t* buf, uint64_t len, Arena* arena,
                       BoltBatch* out_batch) noexcept;

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
