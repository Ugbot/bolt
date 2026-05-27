// bolt_groupby.h (kernels tier) — re-export of the typed multi-key GROUP BY
// kernel that physically lives under bolt/join/bolt_groupby.h.
//
// K-AGG-A scope:
//   Keys     : Int64 / Int32 / Decimal128 / Date32 (composite up to 4).
//   Aggs     : SUM / COUNT / COUNT(*) / MIN / MAX / AVG
//              over Int64 / Decimal128 / Float64.
//   Single-threaded, branch-free apply (CMOV-style bmin/bmax for integer
//   MIN/MAX), Tiger Style — no hot-path allocs, ≤70 lines/fn, ≥2 asserts.
//
// K-AGG-B (deferred): Utf8 spilled keys, COUNT(DISTINCT), partitioned-parallel.

#pragma once

#include "bolt/join/bolt_groupby.h"

namespace bolt::kernels::groupby {

using ::bolt::AggKind;
using ::bolt::AggSpec;
using ::bolt::GbCell16;
using ::bolt::groupby_agg_multi_key_typed;
using ::bolt::kGbEntryCap;
using ::bolt::kGbMaxAggs;
using ::bolt::kGbMaxKeys;

}  // namespace bolt::kernels::groupby
