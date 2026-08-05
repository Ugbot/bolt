// bolt/lakehouse/iceberg/statistics.h — per-data-file stats for prune.
//
// Iceberg manifest carries lower/upper/null_value_counts keyed by field-id.
// W4 stores them as ASCII strings keyed by field-id; comparison interprets
// against the predicate's BoltType, matching the delta path. Tiger Style.

#pragma once

#include <cstdint>

#include "bolt/lakehouse/format.h"

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

// Per-file predicate prune via lower/upper/null_value_counts. Conservative:
// returns true on anything it can't evaluate. Numeric comparisons only at W4.
bool stats_pass(const DataFileRef* f, const Schema* sch,
                const ::bolt::lakehouse::Predicate* preds,
                uint32_t n_preds) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
