// bolt/lakehouse/iceberg/statistics.h — per-data-file stats for prune.
//
// Iceberg manifest carries lower/upper/null_value_counts keyed by field-id.
// W4 stores them as ASCII strings keyed by field-id; comparison interprets
// against the predicate's BoltType, matching the delta path. Tiger Style.

#pragma once

#include <cstdint>

#include "bolt/lakehouse/format.h"

namespace bolt { class Arena; }
namespace bolt { namespace ingest { namespace parquet { struct PqMeta; } } }

namespace bolt {
namespace lakehouse {
namespace iceberg {

// G2ICE-336: a data file's per-column metrics and split offsets live in an arena, sized to the file. How many columns
// one file may carry and how long one bound may be are budgets in the bolt Limits table (bolt_budget.h:
// iceberg_stat_columns, iceberg_bound_bytes); past either the producer refuses (ResourceExhausted naming the knob),
// never drops or cuts a metric. A FileStats is a shallow value: copies share the arena storage, which must outlive
// them (stats_clone / stats_encode for a copy that outlives its arena).

struct ColumnStatEntry {
    int32_t  field_id;
    uint32_t lower_len;            // bytes of `lower` / `upper`: Iceberg's binary single-value encoding (a double is
    uint32_t upper_len;            // 8 raw bytes and may start with NUL), the JSON path's text
    uint32_t _pad;
    int64_t  null_count;
    bool     has_lower;
    bool     has_upper;
    bool     has_value_count;      // G2ICE-296 (spec Appendix F): which of the other metrics are present
    bool     has_nan_count;
    bool     has_column_size;
    bool     no_null_count;        // true: null_value_counts carries no entry for this field (metrics mode none, unknown)
    uint8_t  _pad2[2];
    const char* lower;             // arena-owned; null when absent
    const char* upper;
    int64_t  value_count;          // value_counts (nulls included); valid with has_value_count
    int64_t  nan_count;            // nan_value_counts (float / double only); valid with has_nan_count
    int64_t  column_size;          // column_sizes: the column's bytes on disk; valid with has_column_size
};

struct FileStats {
    int64_t          record_count;
    int64_t          file_size_in_bytes;
    uint32_t         n_cols;
    uint32_t         cap_cols;
    ColumnStatEntry* cols;              // cap_cols entries, arena-owned
    uint32_t         n_split_offsets;   // 0 = split_offsets written null
    uint32_t         _pad;
    int64_t*         split_offsets;     // arena-owned
    // sort_order_id (field 140): the table sort order the file's rows are in; 0 is the unsorted order (what Iceberg-Java
    // writes for a file whose rows follow no table order). has_sort_order_id false = null.
    bool             has_sort_order_id;
    uint8_t          _pad2[3];
    int32_t          sort_order_id;
};

// The entry for `field_id`, appended (zeroed, null_count -1 = unknown) when absent. Grows `cols` in `a` by doubling;
// nullptr with ResourceExhausted("iceberg_stat_columns") past the budget, or on arena exhaustion.
ColumnStatEntry* stats_col(FileStats* st, Arena* a, int32_t field_id) noexcept;
// Append a new entry for `field_id` (no lookup: the caller knows it is new). Same refusals.
ColumnStatEntry* stats_add_col(FileStats* st, Arena* a, int32_t field_id) noexcept;
// The entry for `field_id`, or nullptr.
const ColumnStatEntry* stats_find(const FileStats* st, int32_t field_id) noexcept;
// Copy `n` bytes into `a` as e's lower (upper) bound. False with ResourceExhausted("iceberg_bound_bytes") when n is
// over the budget, or on arena exhaustion; the entry then has no such bound and the caller must refuse.
bool stats_set_bound(ColumnStatEntry* e, Arena* a, bool lower, const void* b, uint32_t n) noexcept;
// Copy the split offsets into `a`.
bool stats_set_splits(FileStats* st, Arena* a, const int64_t* offs, uint32_t n) noexcept;
// A deep copy of `src` whose storage is in `a`.
bool stats_clone(const FileStats& src, Arena* a, FileStats* out) noexcept;
// Serialized form (for caches that outlive the arena): stats_encoded_size bytes; stats_decode allocates in `a`.
uint64_t stats_encoded_size(const FileStats& st) noexcept;
bool stats_encode(const FileStats& st, uint8_t* out, uint64_t cap, uint64_t* len) noexcept;
bool stats_decode(const uint8_t* b, uint64_t len, Arena* a, FileStats* out) noexcept;

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
// unknown in any row group is left out entirely. Appends to `cols` in `a`;
// record_count / file_size_in_bytes stay the caller's. False on a malformed
// footer or a refused budget (ResourceExhausted names it).
bool file_stats_from_parquet_meta(const ::bolt::ingest::parquet::PqMeta* meta,
                                  const int32_t* field_ids,
                                  uint32_t n_field_ids,
                                  Arena* a, FileStats* out) noexcept;

// Per-file predicate prune via lower/upper/null_value_counts. Conservative:
// returns true on anything it can't evaluate. Numeric comparisons only at W4.
bool stats_pass(const DataFileRef* f, const Schema* sch,
                const ::bolt::lakehouse::Predicate* preds,
                uint32_t n_preds) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
