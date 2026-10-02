// bolt_stats_limits.h — capacities of the fused page/stripe stats kernel
// (MSEG K7, kernels/bolt_stats_fused.h).
#pragma once

#include <cstdint>

namespace bolt::stats {

// Kind A invariant: the float sum is defined as kStatsSumLanes running sums,
// lane = global row index mod kStatsSumLanes, folded pairwise. Changing it
// changes stored f32/f64 sums.
inline constexpr uint32_t kStatsSumLanes = 8;

// Rows per dense sub-block of one update: a block whose rows are all valid
// (and, for floats, NaN-free) takes the vectorised path. Multiple of
// kStatsSumLanes. Tuning only; results do not depend on it.
inline constexpr uint32_t kStatsBlockRows = 1024;

// Rows one FusedStats can describe (row_count / null_count are uint32).
inline constexpr uint64_t kStatsMaxRows = 0xFFFFFFFFull;

static_assert(kStatsBlockRows % kStatsSumLanes == 0, "block must hold whole lane groups");
static_assert((kStatsSumLanes & (kStatsSumLanes - 1)) == 0, "lane count is a power of two");

}  // namespace bolt::stats
