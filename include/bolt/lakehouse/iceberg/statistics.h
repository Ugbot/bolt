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
    uint8_t  _pad[4];
    char     lower[kLakeMaxValBytes];
    char     upper[kLakeMaxValBytes];
};
// 4 field_id + 4 alignment + 8 null_count + 4 flags/lengths + 4 pad + bounds.
static_assert(sizeof(ColumnStatEntry) == 24u + 2u * kLakeMaxValBytes,
              "ColumnStatEntry layout pinned: lengths came from padding");

struct FileStats {
    int64_t          record_count;
    int64_t          file_size_in_bytes;
    uint32_t         n_cols;
    uint32_t         _pad;
    ColumnStatEntry  cols[kIcebergMaxStatCols];
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
