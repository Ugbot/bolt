// bolt/lakehouse/scan_optimizer.h — W8 vectorized-scan primitives.
//
// Three small PODs + free functions. The Morsel is the vector-of-rows the
// optimised scan path emits; downstream operators consume Morsels until
// the existing `scan_next_batch` shim materialises a contiguous BoltBatch
// at the egress boundary.
//
// Tiger Style: PODs, ≥2 asserts/fn, ≤70 lines/fn, no exceptions, no smart
// pointers, fixed caps.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "bolt/bolt_column.h"
#include "bolt/lakehouse/format.h"

namespace bolt {
namespace lakehouse {

// Hard caps (W8 contract).
static constexpr uint32_t kScanMorselMaxRows     = 1u << 20;    // 1M rows
static constexpr uint32_t kScanMaxParallelism    = 16u;
static constexpr uint32_t kScanDefaultParallel   = 4u;
static constexpr uint32_t kScanMaxLookahead      = 32u;
static constexpr uint32_t kScanDefaultLookahead  = 8u;
static constexpr uint32_t kScanMaxInflight       = 64u;

// A Morsel = a (sub)set of rows in a BoltBatch, addressed by selection
// vector. `sel == nullptr` AND `sel_len == batch->rows` is the
// "everything matches, no filter" fast path — caller must handle both.
struct Morsel {
    BoltBatch* batch;     // borrowed; owned by scan arena
    uint32_t*  sel;       // ascending row indices, or nullptr for all-rows
    uint32_t   sel_len;   // length of sel (or all-rows count)
    uint32_t   _pad;
};

// Initialise selection vector to the identity sel[i] = i. Used at the
// start of a morsel before predicates/DV filter it down.
void sel_identity(uint32_t* sel, uint32_t n) noexcept;

// Filter selection vector by a runtime predicate over `sel_in[]`. Returns
// the post-filter length written into `sel_out[]`. `sel_out` may alias
// `sel_in` (in-place). The pred call is intentionally a free function
// pointer so the optimiser can inline the simple cases.
uint32_t sel_filter_passthrough(
    const uint32_t* sel_in, uint32_t sel_len,
    uint32_t* sel_out) noexcept;

// Drop rows whose index appears in `mark[]` (bool-per-row tombstone mask).
// `mark[i] != 0` removes row `sel_in[i]`. Returns new length. In-place ok.
uint32_t sel_filter_mark(
    const uint32_t* sel_in, uint32_t sel_len,
    const uint8_t* mark, uint32_t* sel_out) noexcept;

// Result of a column-pruning pass against a `ReadOptions::projection[]`.
// Materialised columns are listed by their *source* index into the
// table schema. Empty projection (n_projection == 0) means "all".
struct ColumnPruneResult {
    uint32_t keep[kLakeMaxProjection];   // source col indices, ascending
    uint32_t n_keep;
    uint32_t n_source;                   // total columns in source schema
    bool     all_columns;                // true ⇔ projection == "all"
    uint8_t  _pad[3];
};

// Build the column-prune plan from a ReadOptions + source column names.
// `src_names[i]` is the name of source col `i`. Names not present in the
// projection are dropped; names listed but missing from the schema are
// silently ignored (caller responsibility to validate at table-open).
bool column_prune_plan(
    const ReadOptions* opts,
    const char* const* src_names, uint32_t n_src,
    ColumnPruneResult* out) noexcept;

// Predicate pushdown verdict at the row-group level. Three levels are
// supported in principle (stats / page-index / bloom); only `stats` is
// wired through today — page-index + bloom are upstream-gated and
// return `kPushdownIndeterminate` for now.
enum class PushdownVerdict : uint8_t {
    kSkip            = 0,   // proven no rows match → skip whole row group
    kAllPass         = 1,   // proven all rows match → no per-row filter needed
    kIndeterminate   = 2,   // need to scan + filter at row level
};

struct RowGroupStats {
    // Per-column min/max as i64 (the dominant case in W2/W4 tests). Strings
    // / floats fall back to indeterminate.
    int64_t  min_i64[kLakeMaxProjection];
    int64_t  max_i64[kLakeMaxProjection];
    int64_t  null_count[kLakeMaxProjection];
    uint64_t num_rows;
    uint32_t n_cols;
    bool     has_minmax;
    uint8_t  _pad[3];
};

// Apply predicate-pushdown stats analysis. `col_names[i]` lines up with the
// stats columns. Returns the strongest verdict we can prove. Upstream
// backlog: page-index + bloom-filter readers.
PushdownVerdict predicate_pushdown_stats(
    const RowGroupStats* stats,
    const char* const* col_names,
    const Predicate* preds, uint32_t n_preds) noexcept;

// Centralised partition-prune helper. Same as
// `delta::delta_file_passes` / `iceberg::partition_passes` were doing
// per-format — now callable from both. Returns `out_keep_idx[]` (up to
// `*out_n` on entry, set to actual count on exit) listing the indices
// of `files[]` that survive partition pruning. Predicates are matched
// by column name against `parts[]` key/value pairs.
struct PartitionEntry {
    const char* key;
    const char* value;
    bool        is_null;
    uint8_t     _pad[7];
};

struct PartitionedFile {
    const PartitionEntry* parts;   // n_parts entries
    uint32_t              n_parts;
    uint32_t              _pad;
};

bool partition_prune(
    const PartitionedFile* files, uint32_t n_files,
    const Predicate* preds, uint32_t n_preds,
    uint32_t* out_keep_idx, uint32_t* out_n) noexcept;

}  // namespace lakehouse
}  // namespace bolt
