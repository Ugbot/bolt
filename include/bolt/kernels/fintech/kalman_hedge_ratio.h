// bolt/kernels/fintech/kalman_hedge_ratio.h — Phase 5.S.7 Tier 2 kernel.
//
// KalmanHedgeRatio: 1-D Kalman filter over the instantaneous hedge ratio
// beta = price_a / price_b, paired with the resulting Kalman spread
// `price_a - beta * price_b`.
//
// Per-row recurrence (identity state transition on beta):
//   predict:  beta_pred = beta;  P_pred = P + q
//   observe:  z = (|pb| > 1e-15) ? pa/pb : beta_pred
//             (near-zero pb → observation == prediction, filter coasts)
//   update:   K    = P_pred / (P_pred + r)
//             beta = beta_pred + K * (z - beta_pred)
//             P    = (1 - K) * P_pred
//   emit:     hedge_ratio[i] = beta
//             spread[i]      = pa[i] - beta * pb[i]
//
// Source: chukonu/fintech/kernels.h :: CreateKalmanHedgeRatioProcessor
// (line 2009). Chukonu seeds beta = 1.0, P = 1.0, and emits starting at
// i=0 (no seed-from-observation like Kalman1D — the first row already
// runs a full predict/update). We match that.
//
// Branchless note: the `|pb| > eps` guard is a cmov ternary, not a
// control-flow branch. The inner loop otherwise is straight-line arithmetic
// and vectorises cleanly subject to the beta/P loop-carry.
//
// Two outputs per call: hedge ratio (beta) and Kalman spread.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct KalmanHedgeRatioOpDesc {
    int32_t pa_col;
    int32_t pb_col;
    int32_t out_ratio_col;
    int32_t out_spread_col;
    double  q;              // process-noise variance
    double  r;              // measurement-noise variance
    double  beta_init;      // chukonu uses 1.0
    double  p_init;         // chukonu uses 1.0
};

struct KalmanHedgeRatioState {
    KalmanHedgeRatioOpDesc desc;
    double                 beta;
    double                 p;
};

inline void execute_kalman_hedge_ratio(KalmanHedgeRatioState* state,
                                       const double* BOLT_RESTRICT pa,
                                       const double* BOLT_RESTRICT pb,
                                       int64_t n,
                                       double* BOLT_RESTRICT out_ratio,
                                       double* BOLT_RESTRICT out_spread) noexcept {
    assert(state != nullptr);
    assert(pa         != nullptr || n == 0);
    assert(pb         != nullptr || n == 0);
    assert(out_ratio  != nullptr || n == 0);
    assert(out_spread != nullptr || n == 0);
    assert(n >= 0);
    assert(state->desc.q >= 0.0);
    assert(state->desc.r >= 0.0);

    const double q = state->desc.q;
    const double r = state->desc.r;
    double       beta = state->beta;
    double       p    = state->p;
    for (int64_t i = 0; i < n; ++i) {
        const double pa_i = pa[i];
        const double pb_i = pb[i];
        // Predict (identity state transition).
        const double beta_pred = beta;
        const double p_pred    = p + q;
        // Observation: pa/pb, or fall back to beta_pred near pb==0.
        // Ternary compiles to cmov — no mispredictable branch.
        const double obs = (std::fabs(pb_i) > 1e-15) ? (pa_i / pb_i) : beta_pred;
        // Update.
        const double k = p_pred / (p_pred + r);
        beta = beta_pred + k * (obs - beta_pred);
        p    = (1.0 - k) * p_pred;
        out_ratio[i]  = beta;
        out_spread[i] = pa_i - beta * pb_i;
    }
    state->beta = beta;
    state->p    = p;
}

inline KalmanHedgeRatioState* make_kalman_hedge_ratio_state(
    ::bolt::Arena* arena,
    int32_t pa_col,
    int32_t pb_col,
    int32_t out_ratio_col,
    int32_t out_spread_col,
    double q,
    double r,
    double beta_init = 1.0,
    double p_init    = 1.0) noexcept {
    assert(arena != nullptr);
    assert(q >= 0.0);
    assert(r >= 0.0);
    assert(p_init >= 0.0);
    KalmanHedgeRatioState* s = arena->allocate_array<KalmanHedgeRatioState>(1);
    if (s == nullptr) return nullptr;
    s->desc.pa_col         = pa_col;
    s->desc.pb_col         = pb_col;
    s->desc.out_ratio_col  = out_ratio_col;
    s->desc.out_spread_col = out_spread_col;
    s->desc.q              = q;
    s->desc.r              = r;
    s->desc.beta_init      = beta_init;
    s->desc.p_init         = p_init;
    s->beta                = beta_init;
    s->p                   = p_init;
    return s;
}

inline void kalman_hedge_ratio_reset(KalmanHedgeRatioState* state) noexcept {
    assert(state != nullptr);
    assert(state->desc.q >= 0.0);
    state->beta = state->desc.beta_init;
    state->p    = state->desc.p_init;
}

}  // namespace fintech
}  // namespace bolt
