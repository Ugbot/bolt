// bolt/kernels/fintech/kalman1d.h — Phase 5.S.6 Tier 2 kernel.
//
// Kalman1D: scalar single-state Kalman filter over an observation series.
// State = (x_hat, P, Q, R). Per-row recurrence (identity state-transition):
//   predict:
//       x_pred = x_hat
//       P_pred = P + Q
//   update:
//       K       = P_pred / (P_pred + R)
//       x_hat   = x_pred + K * (z - x_pred)
//       P       = (1 - K) * P_pred
//   emit x_hat.
//
// Source: chukonu/fintech/kernels.h :: CreateKalman1DProcessor (line 3659).
// chukonu's seed is x_hat = obs[0], P = 1.0 — the first row emits obs[0]
// verbatim and the filter steps from i=1. We match that exactly. P_0 is
// hard-coded to 1.0 in chukonu; exposing it as configurable costs nothing
// and lets callers drop the filter into a warm stream.
//
// Branchless note: the recurrence is straight-line arithmetic — no
// predicates, no warmup after the seed. The only branch is the one-time
// "initialized" check, which is hoisted out of the steady-state loop via
// the same i=0 seed pattern used by execute_ema.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct Kalman1DOpDesc {
    int32_t obs_col;
    int32_t out_col;
    double  q;          // process-noise variance
    double  r;          // measurement-noise variance
    double  p_init;     // initial P (chukonu uses 1.0)
};

struct Kalman1DState {
    Kalman1DOpDesc desc;
    double         x_hat;
    double         p;
    bool           initialized;
};

inline void execute_kalman1d(Kalman1DState* state,
                             const double* BOLT_RESTRICT z,
                             int64_t n,
                             double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(z   != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    assert(state->desc.q >= 0.0);
    assert(state->desc.r >= 0.0);

    int64_t i = 0;
    // Seed from the first observation (chukonu convention). Hoisted out
    // of the steady-state loop so the per-row body is branchless.
    if (!state->initialized && i < n) {
        state->x_hat       = z[i];
        state->p           = state->desc.p_init;
        state->initialized = true;
        out[i] = state->x_hat;
        ++i;
    }

    const double q = state->desc.q;
    const double r = state->desc.r;
    double       x = state->x_hat;
    double       p = state->p;
    for (; i < n; ++i) {
        const double p_pred = p + q;
        const double k      = p_pred / (p_pred + r);
        x     = x + k * (z[i] - x);
        p     = (1.0 - k) * p_pred;
        out[i] = x;
    }
    state->x_hat = x;
    state->p     = p;
}

inline Kalman1DState* make_kalman1d_state(::bolt::Arena* arena,
                                          int32_t obs_col,
                                          int32_t out_col,
                                          double q,
                                          double r,
                                          double p_init = 1.0) noexcept {
    assert(arena != nullptr);
    assert(q >= 0.0);
    assert(r >= 0.0);
    assert(p_init >= 0.0);
    Kalman1DState* s = arena->allocate_array<Kalman1DState>(1);
    if (s == nullptr) return nullptr;
    s->desc.obs_col = obs_col;
    s->desc.out_col = out_col;
    s->desc.q       = q;
    s->desc.r       = r;
    s->desc.p_init  = p_init;
    s->x_hat        = 0.0;
    s->p            = p_init;
    s->initialized  = false;
    return s;
}

inline void kalman1d_reset(Kalman1DState* state) noexcept {
    assert(state != nullptr);
    assert(state->desc.q >= 0.0);
    state->x_hat       = 0.0;
    state->p           = state->desc.p_init;
    state->initialized = false;
}

}  // namespace fintech
}  // namespace bolt
