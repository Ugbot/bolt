// src/lakehouse/iceberg_dml_read.h — internal: the read half of the Iceberg
// writer's row-level operations (G2ICE-280).
//
// table_delete / table_overwrite / table_compact need the table's LIVE rows
// (every delete already applied) together with each row's file position. They
// read them through the ordinary Iceberg scan (iceberg_scan.cpp), so a row the
// table returns and a row these operations see are the same row. This TU is
// separate from iceberg_writer.cpp because the scan and the writer each define
// their own opaque `TableHandle` (see new_table_handle in both); it only ever
// holds the scan's.

#pragma once

#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/lakehouse/format.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {
namespace dml {

enum class ReadStatus : uint8_t { kOk = 0, kFailed = 1, kUnsupported = 2 };

// One batch of live rows of one data file (one row group, deletes applied).
struct RowBatch {
    const char*     file_path;   // as the data file's manifest entry records it
    BoltBatch*      batch;       // the table's current-schema columns; `_pos` already stripped
    const int64_t*  pos;         // file position of each row of `batch`
    const uint32_t* match;       // rows of `batch` the predicate accepts (nullptr without a predicate)
    uint32_t        n_match;
    uint32_t        _pad;
    const BoltColumn* lineage;   // with `lineage`: each row's _row_id and _last_updated_sequence_number (two Int64 columns)
};

using RowVisitFn = bool (*)(void* ctx, const RowBatch* b) noexcept;
using FileKeepFn = bool (*)(void* ctx, const DataFileRef* e) noexcept;

// Read snapshot `snapshot_id` of the table whose metadata document is
// `meta_json`, through `store` (rooted at the table location). Every batch and
// buffer lives in `arena` and stays valid until the caller resets it. `pred`
// (nullable) prunes files by statistics and selects RowBatch::match; `keep`
// (nullable) restricts which data files are read. A false `visit` stops the
// read with kFailed.
ReadStatus read_live_rows(Arena* arena, const ObjectStore* store,
                          const uint8_t* meta_json, uint32_t meta_len,
                          int64_t snapshot_id, const Predicate* pred,
                          FileKeepFn keep, void* keep_ctx,
                          RowVisitFn visit, void* ctx, bool lineage = false) noexcept;

// Rows of `b` (first `n_cols` columns are searched for the predicate column by
// name) where `pred` is TRUE under SQL three-valued logic: a NULL value never
// matches a comparison. Comparison ops: integer/date/timestamp/bool columns
// against an integer literal, float columns against an integer or float
// literal (IEEE: NaN matches only kNe), string/binary columns against a byte
// string (unsigned byte order). kUnsupported for any other pairing; kFailed
// when the column does not exist.
ReadStatus eval_predicate(const BoltBatch* b, uint32_t n_cols,
                          const Predicate* pred, Arena* arena,
                          uint32_t** out, uint32_t* n_out) noexcept;

// The distinct data-file paths a position-delete parquet file names
// (`file_path`, field 2147483546). Paths are arena-owned, NUL-terminated.
bool position_delete_targets(Arena* arena, const uint8_t* body, uint64_t len,
                             const char*** out, uint32_t* n_out) noexcept;

}  // namespace dml
}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
