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

#pragma once

#include "bolt/kernels/promql_rate.h"
#include "bolt/kernels/promql_deriv.h"
#include "bolt/kernels/promql_histogram.h"
#include "bolt/kernels/promql_over_time.h"
