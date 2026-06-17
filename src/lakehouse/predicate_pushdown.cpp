// bolt/lakehouse/predicate_pushdown.cpp — three-level predicate pushdown.
//
// Level 1: row-group stats (min/max/null_count). Implemented inline in
//          `scan_optimizer.cpp::predicate_pushdown_stats`. This TU exposes
//          the multi-level driver.
// Level 2: Parquet page index (PageHeader / ColumnIndex / OffsetIndex).
// Level 3: Parquet bloom filter.
//
// Levels 2 + 3 are upstream-gated — bolt::ingest does not (yet) expose
// readers for ColumnIndex or BloomFilter. The driver falls back through
// the levels and returns kIndeterminate when nothing harder than stats
// can be proven.

#include "bolt/lakehouse/scan_optimizer.h"

#include <cassert>

namespace bolt {
namespace lakehouse {

// Forward — implemented in scan_optimizer.cpp.
PushdownVerdict predicate_pushdown_stats(
    const RowGroupStats* stats,
    const char* const* col_names,
    const Predicate* preds, uint32_t n_preds) noexcept;

// Multi-level driver.
//
// Today: tries stats; returns whatever stats decided. When the page-index
// reader and bloom-filter reader land upstream, plug them in below in the
// `kIndeterminate` arm.
//
// Upstream backlog:
//   - `bolt::ingest::parquet_read_column_index(meta, rg, col, out)` →
//     per-page min/max + null counts. Lets us prove `kSkip` at a finer
//     granularity than the row group.
//   - `bolt::ingest::parquet_read_bloom_filter(meta, rg, col, out)` →
//     SplitBlock bloom filter. Lets us prove `kSkip` for Eq predicates
//     on high-cardinality keys where min/max is too wide.
PushdownVerdict predicate_pushdown_multi(
    const RowGroupStats* stats,
    const char* const* col_names,
    const Predicate* preds, uint32_t n_preds) noexcept {
    assert(stats != nullptr);
    assert(preds != nullptr || n_preds == 0);
    const PushdownVerdict v = predicate_pushdown_stats(
        stats, col_names, preds, n_preds);
    if (v != PushdownVerdict::kIndeterminate) return v;
    // Page-index + bloom layers go here; upstream-gated. The verdict
    // stays kIndeterminate until those readers ship in bolt::ingest.
    return PushdownVerdict::kIndeterminate;
}

}  // namespace lakehouse
}  // namespace bolt
