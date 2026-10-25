// bolt/lakehouse/iceberg/statistics.h — per-data-file stats for prune.
//
// Iceberg manifest carries lower/upper/null_value_counts keyed by field-id.
// W4 stores them as ASCII strings keyed by field-id; comparison interprets
// against the predicate's BoltType, matching the delta path. Tiger Style.

#pragma once

#include <cstdint>

#include "bolt/lakehouse/format.h"

namespace bolt { namespace ingest { namespace parquet { struct PqMeta; } } }

namespace bolt {
namespace lakehouse {
namespace iceberg {

static constexpr uint32_t kIcebergMaxStatCols = 32u;

struct ColumnStatEntry {
    int32_t  field_id;
    int64_t  null_count;
    bool     has_lower;
    bool     has_upper;
    // Byte lengths of `lower` / `upper`. A REAL Iceberg manifest stores bounds
    // in Iceberg's binary single-value encoding — a double bound is 8 raw
    // bytes, and `250.5` is 00 00 00 00 00 50 6F 40, which STARTS with NUL.
    // Treating those as C strings (what the JSON path can get away with, since
    // JSON bounds are text) would truncate almost every numeric bound to
    // empty. Carving the lengths out of the existing padding keeps the struct
    // the same size and leaves the JSON path's behaviour unchanged.
    uint8_t  lower_len;
    uint8_t  upper_len;
    // G2ICE-296 (spec Appendix F): which of the other per-column metrics are present. Carved from the padding, so a
    // zeroed entry (every older producer) still means "null_count only", as before.
    bool     has_value_count;
    bool     has_nan_count;
    bool     has_column_size;
    bool     no_null_count;        // true: null_value_counts carries no entry for this field (metrics mode none, unknown)
    char     lower[kLakeMaxValBytes];
    char     upper[kLakeMaxValBytes];
    int64_t  value_count;          // value_counts (nulls included); valid with has_value_count
    int64_t  nan_count;            // nan_value_counts (float / double only); valid with has_nan_count
    int64_t  column_size;          // column_sizes: the column's bytes on disk; valid with has_column_size
};
// 4 field_id + 4 alignment + 8 null_count + 8 flags/lengths + bounds + 3 x 8 counts.
static_assert(sizeof(ColumnStatEntry) == 48u + 2u * kLakeMaxValBytes,
              "ColumnStatEntry layout pinned: flags came from padding");

// G2ICE-296: split_offsets kept per data file (the row groups' start offsets). A file with more row groups writes none:
// the field is optional and a reader only loses split planning.
static constexpr uint32_t kIcebergMaxSplitOffsets = 64u;

struct FileStats {
    int64_t          record_count;
    int64_t          file_size_in_bytes;
    uint32_t         n_cols;
    uint32_t         n_split_offsets;   // G2ICE-296: 0 = split_offsets written null
    ColumnStatEntry  cols[kIcebergMaxStatCols];
    int64_t          split_offsets[kIcebergMaxSplitOffsets];
    // sort_order_id (field 140): the table sort order the file's rows are in; 0 is the unsorted order (what Iceberg-Java
    // writes for a file whose rows follow no table order). has_sort_order_id false = null.
    bool             has_sort_order_id;
    uint8_t          _pad2[3];
    int32_t          sort_order_id;
};

struct DataFileRef;
struct Schema;

// G2ICE-43 — fold a parquet footer's per-row-group chunk statistics into the
// Iceberg per-column stats of that ONE file: null_value_counts from the summed
// null counts, lower/upper bounds as Iceberg's binary single-value encoding.
// `field_ids[c]` names parquet column c's Iceberg field id (<= 0 skips it).
// Bounds are emitted only where the parquet stat bytes ARE the Iceberg
// encoding (plain int32/int64/float/double/boolean, UTF-8 byte arrays via
// min_value/max_value) and every non-all-null row group carries them; any
// doubt drops the bound, never guesses one. A column whose null count is
// unknown in any row group is left out entirely. Writes only `n_cols` and
// `cols` — record_count / file_size_in_bytes stay the caller's.
bool file_stats_from_parquet_meta(const ::bolt::ingest::parquet::PqMeta* meta,
                                  const int32_t* field_ids,
                                  uint32_t n_field_ids,
                                  FileStats* out) noexcept;

// Per-file predicate prune via lower/upper/null_value_counts. Conservative:
// returns true on anything it can't evaluate. Numeric comparisons only at W4.
bool stats_pass(const DataFileRef* f, const Schema* sch,
                const ::bolt::lakehouse::Predicate* preds,
                uint32_t n_preds) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
