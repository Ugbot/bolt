// bolt/kernels/promql.h — aggregator for the PromQL range/histogram kernels.
//
// Prometheus counter/gauge/histogram MATH as pure bolt kernels. See the
// individual headers for per-function contracts and the Prometheus source
// each replicates. The chukonu window operator wires these in as a later,
// separate step; bolt owns only the math + its tests.
//
//   promql_rate.h       rate / increase / delta / irate / idelta /
//                       resets / changes  (range-vector functions)
//   promql_deriv.h      deriv / predict_linear + linear_regression
//   promql_histogram.h  histogram_quantile (classic le-bucket interpolation)
//   promql_native_histogram.h
//                       NATIVE histograms (Prometheus 2.40+): the sparse
//                       schema/zero-bucket/positive/negative sample type and
//                       histogram_quantile / _fraction / _stdvar / _stddev /
//                       _avg over it, including the custom-buckets (NHCB)
//                       schema -53 that a classic histogram converts into,
//                       plus counter-reset detection (nh_detect_reset) and the
//                       schema-aligning copy every cross-sample comparison
//                       needs.
//   promql_nh_rate.h    rate / increase / delta / irate / idelta / resets /
//                       changes over NATIVE histograms — the histogram half of
//                       promql_rate.h, which is different arithmetic and not
//                       the same arithmetic on a different type.

#pragma once

#include "bolt/kernels/promql_rate.h"
#include "bolt/kernels/promql_deriv.h"
#include "bolt/kernels/promql_histogram.h"
#include "bolt/kernels/promql_native_histogram.h"
#include "bolt/kernels/promql_nh_rate.h"
#include "bolt/kernels/promql_over_time.h"
