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
//   - types:     INT64 -> Int64 (DECIMAL -> Decimal64 mantissa;
//                  TIMESTAMP {millis,micros,nanos} -> Timestamp[us],
//                  normalized to microseconds at decode; TIME -> Duration[us];
//                  UINT_64 -> UInt64 — G2FEAT-46),
//                INT32 -> Int32 (DATE -> Date32; DECIMAL -> Decimal64;
//                  UINT_8/16/32 -> UInt8/16/32, INT_8/16 -> Int8/16;
//                  TIME_MILLIS -> Int32 raw — G2FEAT-46),
//                INT96 -> Timestamp[us] (legacy Impala/Spark day+nanos,
//                  converted at decode — G2FEAT-46),
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
//     LogicalType (SchemaElement field 10) is honored over ConvertedType,
//     so nanosecond timestamps/times and un/signed IntType round-trip even
//     when the writer omitted the legacy ConvertedType.
//     Still unsupported (parquet_map_type returns false): non-DECIMAL FLBA,
//     FLBA wider than 16 bytes, and nested LIST/MAP/STRUCT.
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

// Map one parquet leaf column to its Bolt materialization (FLOAT widens to
// Float64, BOOLEAN lands as Int64 0/1, TIMESTAMP/INT96 -> Timestamp[us],
// TIME -> Duration[us], INT32-DECIMAL -> Decimal64, un/signed IntType ->
// the matching Int*/UInt* width — G2FEAT-46). Returns false only for the
// still-unsupported shapes: non-DECIMAL FLBA, FLBA wider than 16 bytes, and
// nested types. `*out_scale` carries the DECIMAL scale for Decimal columns.
bool parquet_map_type(const PqColumn* col, BoltType* out_type,
                      uint8_t* out_scale) noexcept;

// Derive a COMPLETE BoltSchema straight from a parsed footer. The parquet
// metadata already states everything a consumer needs — name, logical type,
// DECIMAL precision/scale, integer width/signedness, nullability — so no
// caller should be re-deriving it, and none should have to squeeze it through
// a lossy hand-maintained enum (a 4-value {int64,utf8,float32,float64} hint
// cannot express DATE or DECIMAL at all, which is why TPC-H could not be
// declared this way).
//
// This routes through `parquet_map_type` — the SAME mapping the reader
// decodes with — so a consumer's declared schema cannot drift from what the
// scan actually produces. That drift is a real, previously-shipped bug class
// (G2FEAT-112: catalog said Int64 while the buffer was 2-byte), and it had
// three separate hand-rolled copies of this loop to go wrong in.
//
// `out` must already own storage for >= meta->n_columns fields (see
// BoltSchema::set_storage). `lowercase_names` folds A-Z for consumers whose
// catalog is always-lowercased. No allocation. Returns false if the schema is
// empty, `out` is under-sized, or any column has no supported mapping (the
// caller learns WHICH via a subsequent parquet_map_type probe).
bool parquet_schema_from_meta(const PqMeta* meta, BoltSchema* out,
                              bool lowercase_names) noexcept;

// Fold one column's per-chunk min/max statistics into a single file-level
// range, decoded into the units a consumer's catalog wants:
//   - integer-like  -> the raw integer value
//   - DECIMAL       -> the int64 MANTISSA (scale is on the column; the caller
//                      already knows it from parquet_schema_from_meta)
//   - DATE          -> days since epoch
// The footer carries these EXACTLY and for free — no page decode, no scan —
// yet a consumer that re-derives them has to read every row.
//
// FAILS CLOSED. Returns false (and leaves *out_min/*out_max untouched) unless
// EVERY chunk of the column carries both stats and every one decodes. A single
// chunk missing statistics means the file-level range is unproven, and a
// half-proven range is worse than none: callers gate value-range lowering on
// this, so a wrong bound silently produces wrong answers where an absent bound
// merely forgoes an optimization.
//
// Types with no meaningful int64 range (Utf8/Binary/FLBA-non-decimal, floats)
// return false — a consumer wanting string ranges needs a different accessor.
bool parquet_column_int_range(const PqMeta* meta, uint32_t col_idx,
                              int64_t* out_min, int64_t* out_max) noexcept;

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
// to stream a chunk in bounded sub-chunks. Byte-exact (value-level) vs a
// whole-chunk decode of the same pages. Enables bounded per-worker footprint
// for wide single-column scans.
//
// Dictionary chunks are supported: the DICTIONARY_PAGE is decoded once per
// call, before the requested data pages, and the chunk's byte region is taken
// to start there (total_compressed_size spans from the dictionary page, not
// from data_page_offset). This is what makes the function usable as the jump
// target for PAGE-level skipping driven by bolt_parquet_pageindex.h -- which
// matters because dictionary encoding is the writer's recommended default and
// what parquet-mr and Arrow emit by default too, so a PLAIN-only resumable
// decoder could not skip pages in most real files.
//
// The cost of resuming mid-chunk on a dictionary column is re-decoding the
// dictionary page on every call. That is the correct trade for a skipping
// reader (it decodes ONE dictionary instead of every data page it skipped),
// and callers streaming a whole chunk in order should keep using
// parquet_read_row_group_cols, which decodes the dictionary once.
bool parquet_read_col_chunk_pages(const uint8_t* buf, uint64_t len,
                                  const PqMeta* meta, uint32_t row_group,
                                  uint16_t col, uint64_t start_off,
                                  int64_t max_rows, Arena* arena,
                                  BoltColumn* out_col, int64_t* out_rows,
                                  uint64_t* next_off) noexcept;

// Decode ONE repeated (LIST / MAP-leaf) column of a row group into a
// ColumnFormat::Nested BoltColumn of BoltType::List. This is the path a leaf
// under a REPEATED group takes; every other entry point here refuses such a
// column, because it produces a VARIABLE number of values per row and cannot
// be materialised one-value-per-row.
//
// The result carries `out_rows` rows: `col.list_offsets()` gives rows+1
// element offsets and `col.list_element()` the typed element column. An EMPTY
// list is a valid row whose offsets are equal; a NULL list is marked in
// `col.validity`. Those are different values and both are represented.
//
// Scope: max_rep == 1 -- one level of repetition, which is `list<T>` and each
// leaf of a `map<K,V>`. A list OF lists (max_rep >= 2) returns false rather
// than being guessed at. Every value encoding and both page formats the flat
// path supports work here, because the two share one encoding dispatch.
bool parquet_read_list_column(const uint8_t* buf, uint64_t len,
                              const PqMeta* meta, uint32_t row_group,
                              uint16_t col, Arena* arena,
                              BoltColumn* out_col, int64_t* out_rows) noexcept;

// Whole file: locate + parse meta, allocate full-length columns, then
// decode every row group into its row offset (serial v1; the row-group
// function above is the future parallel unit).
bool parquet_read_file(const uint8_t* buf, uint64_t len, Arena* arena,
                       BoltBatch* out_batch) noexcept;

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
