// bolt/kernels/promql_time_fn.h: every PromQL range function behind one frame
// signature, the rows of the TimeWindow registry. A frame is the contiguous
// (ts, v) slice of the samples inside one evaluation window, ts ascending.
//
//   rate / increase / delta extrapolate to the frame bounds [lo, hi];
//   predict_linear measures from eval_ts; irate / idelta, resets, changes and
//   the over_time folds read the slice only.
//
// ts_per_second turns the ts column's unit into seconds (PromQL tables: 1000).
// The result is a value plus presence: a structural absence (too few samples,
// zero interval, bad smoothing factor) is present = false, a NaN is a value.
// Kernels that sort mutate v (quantile, mad); `promql_tf_mutates` says which.
//
// Tiger Style: noexcept, no allocation, no std:: containers.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/promql_deriv.h"
#include "bolt/kernels/promql_over_time.h"
#include "bolt/kernels/promql_rate.h"
#include "bolt/kernels/promql_smoothing.h"

#include <cassert>
#include <cmath>
#include <cstdint>

#if defined(__clang__)
#pragma float_control(push)
#pragma clang fp contract(off)
#endif

namespace bolt {
namespace promql {

struct TimeFnFrame {
    const int64_t* ts;     // ascending, n samples
    double*        v;      // n values (mutated by quantile / mad)
    int64_t        n;
    int64_t        lo, hi; // frame bounds in ts units, hi >= lo
    int64_t        eval_ts;
    double         p0, p1;
    double         ts_per_second;
    PromqlNumerics num;
};

struct TimeFnOut {
    double value;
    bool   present;
};

using TimeFnKernel = TimeFnOut (*)(const TimeFnFrame&) noexcept;

BOLT_FORCE_INLINE TimeFnOut promql_tf_wrap(double v) noexcept {
    return TimeFnOut{v, !promql_is_no_sample(v)};
}

BOLT_FORCE_INLINE TimeFnOut promql_tf_check(const TimeFnFrame& f) noexcept {
    assert(f.n >= 0 && f.hi >= f.lo && f.ts_per_second > 0.0);
    assert(f.ts != nullptr || f.n == 0);
    return TimeFnOut{0.0, f.n > 0};
}

#define BOLT_PROMQL_TF(NAME, EXPR)                                            \
    inline TimeFnOut NAME(const TimeFnFrame& f) noexcept {                    \
        const TimeFnOut chk = promql_tf_check(f);                             \
        if (!chk.present) return chk;                                         \
        return promql_tf_wrap(EXPR);                                          \
    }

BOLT_PROMQL_TF(promql_tf_rate,
               promql_rate(f.ts, f.v, f.n, f.lo, f.hi, f.ts_per_second))
BOLT_PROMQL_TF(promql_tf_increase,
               promql_increase(f.ts, f.v, f.n, f.lo, f.hi, f.ts_per_second))
BOLT_PROMQL_TF(promql_tf_delta,
               promql_delta(f.ts, f.v, f.n, f.lo, f.hi, f.ts_per_second))
BOLT_PROMQL_TF(promql_tf_irate, promql_irate(f.ts, f.v, f.n, f.ts_per_second))
BOLT_PROMQL_TF(promql_tf_idelta, promql_idelta(f.ts, f.v, f.n, f.ts_per_second))
BOLT_PROMQL_TF(promql_tf_deriv, promql_deriv(f.ts, f.v, f.n, f.num, f.ts_per_second))
BOLT_PROMQL_TF(promql_tf_predict_linear,
               promql_predict_linear(f.ts, f.v, f.n, f.eval_ts, f.p0, f.num,
                                     f.ts_per_second))
BOLT_PROMQL_TF(promql_tf_resets, static_cast<double>(promql_resets(f.v, f.n)))
BOLT_PROMQL_TF(promql_tf_changes, static_cast<double>(promql_changes(f.v, f.n)))
BOLT_PROMQL_TF(promql_tf_stddev, promql_stddev(f.v, f.n))
BOLT_PROMQL_TF(promql_tf_stdvar, promql_stdvar(f.v, f.n))
BOLT_PROMQL_TF(promql_tf_quantile, promql_quantile(f.v, f.n, f.p0, f.num))
BOLT_PROMQL_TF(promql_tf_mad, promql_mad(f.v, f.n, f.num))
BOLT_PROMQL_TF(promql_tf_sum, promql_sum_over_time(f.v, f.n))
BOLT_PROMQL_TF(promql_tf_avg, promql_avg_over_time(f.v, f.n))
BOLT_PROMQL_TF(promql_tf_min, promql_min_over_time(f.v, f.n))
BOLT_PROMQL_TF(promql_tf_max, promql_max_over_time(f.v, f.n))
BOLT_PROMQL_TF(promql_tf_present, 1.0)
BOLT_PROMQL_TF(promql_tf_ts_of_min,
               promql_ts_of_extreme_over_time(f.ts, f.v, f.n, true, f.ts_per_second))
BOLT_PROMQL_TF(promql_tf_ts_of_max,
               promql_ts_of_extreme_over_time(f.ts, f.v, f.n, false, f.ts_per_second))
BOLT_PROMQL_TF(promql_tf_ts_of_last,
               promql_ts_of_edge_over_time(f.ts, f.n, true, f.ts_per_second))
BOLT_PROMQL_TF(promql_tf_ts_of_first,
               promql_ts_of_edge_over_time(f.ts, f.n, false, f.ts_per_second))

// double_exponential_smoothing: p0 = smoothing factor, p1 = trend factor. A
// factor outside (0, 1) is a query error the caller rejects; here it is absent.
inline TimeFnOut promql_tf_double_exp(const TimeFnFrame& f) noexcept {
    const TimeFnOut chk = promql_tf_check(f);
    if (!chk.present) return chk;
    if (!promql_des_factor_valid(f.p0) || !promql_des_factor_valid(f.p1)) {
        return TimeFnOut{0.0, false};
    }
    return promql_tf_wrap(promql_double_exp_smoothing(f.v, f.n, f.p0, f.p1));
}

#undef BOLT_PROMQL_TF

// True when the kernel sorts or rewrites v: the caller passes a scratch copy.
inline bool promql_tf_mutates(TimeFnKernel k) noexcept {
    return k == &promql_tf_quantile || k == &promql_tf_mad;
}

}  // namespace promql
}  // namespace bolt

#if defined(__clang__)
#pragma float_control(pop)
#endif
