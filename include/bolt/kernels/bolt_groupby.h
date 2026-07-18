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
// K-AGG-B: Utf8 spilled (>12-byte) GROUP BY keys are handled — the stateful
//   _begin/_ingest/_finalize + merge path deep-copies them into its own arena
//   (Card S), and the one-shot groupby_agg_multi_key_typed re-anchors + owns
//   them at emit. Still deferred: COUNT(DISTINCT), partitioned-parallel.

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
