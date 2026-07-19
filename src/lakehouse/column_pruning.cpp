// bolt/lakehouse/column_pruning.cpp — projection-pushdown helpers.
//
// Today the planning logic lives in `scan_optimizer.cpp::column_prune_plan`.
// This TU hosts the *materialisation* helper: gather the kept columns out
// of a source BoltBatch into a projected output BoltBatch.
//
// The ideal path — projected Parquet decode — is upstream-gated: bolt's
// ingest layer does not yet expose `parquet_read_row_group_projected`.
// Until that lands, callers materialise the full row group and we drop
// the non-projected columns post-decode. Throughput cost is bounded
// because pruning happens before any downstream operator sees the batch.

#include "bolt/lakehouse/scan_optimizer.h"

#include <cassert>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"

namespace bolt {
namespace lakehouse {

// Gather kept columns from `src` into `dst`. `dst` must be `init_empty`-ed
// by the caller. Column payload pointers are aliased (zero-copy); the
// helper writes into `dst->mut_col(i)`.
bool column_pruning_gather(
    const BoltBatch* src, const ColumnPruneResult* plan,
    BoltBatch* dst) noexcept {
    assert(src != nullptr);
    assert(plan != nullptr);
    assert(dst != nullptr);
    if (src == nullptr || plan == nullptr || dst == nullptr) return false;
    if (plan->n_keep > kMaxBatchColumns) return false;
    dst->num_rows = src->num_rows;
    // G2FEAT-47: this helper writes via dst->mut_col(i), so dst needs the COW
    // dirty mask — construct mutable. Sizes columns[2] to the kept width.
    const uint32_t n_out = plan->all_columns ? src->num_cols : plan->n_keep;
    if (!BoltBatch::alloc_columns_mutable(dst, dst->arena, n_out)) return false;
    if (plan->all_columns) {
        for (uint32_t i = 0; i < src->num_cols; ++i) {
            dst->mut_col(i) = src->col(i);
        }
        return true;
    }
    for (uint32_t i = 0; i < plan->n_keep; ++i) {
        const uint32_t s = plan->keep[i];
        if (s >= src->num_cols) return false;
        dst->mut_col(i) = src->col(s);
    }
    // Upstream backlog: `bolt::ingest::parquet_read_row_group_projected`
    // — would skip decoding non-projected columns entirely. Wired here
    // once that lands; falls back to gather (above) today.
    return true;
}

}  // namespace lakehouse
}  // namespace bolt
