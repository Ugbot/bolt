// bolt/kernels/fintech/ewcov.h — Phase 5.E.3.
//
// Exponentially-weighted covariance of two series x, y (RiskMetrics form).
// Tracks running means mu_x, mu_y and running cov by:
//   dx = x - mu_x
//   dy = y - mu_y
//   mu_x += (1 - lambda) * dx
//   mu_y += (1 - lambda) * dy
//   cov   = lambda * cov + (1 - lambda) * dx * dy
// First sample seeds mu_x = x[0], mu_y = y[0], cov = 0.
//
// Source: chukonu/fintech/kernels.h :: CreateEWCOVProcessor (line 3600).
//
// Convention note: chukonu parametrises by `lambda` (the "decay" weight
// on the running state), NOT the `alpha` used by EMA/EWMA. Under the EMA
// convention alpha = 1 - lambda. We match chukonu — the caller supplies
// `lambda` so the math is identical to RiskMetrics docs. A value of
// lambda = 0.94 is the RiskMetrics default.
//
// State is NOT an EmaState — the two-variable update has three running
// quantities. We declare a POD directly.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct EWCOVOpDesc {
    int32_t x_col;
    int32_t y_col;
    int32_t out_col;
    double  lambda;  // 0 <= lambda <= 1; decay on running state.
};

struct EWCOVState {
    EWCOVOpDesc desc;
    double mu_x;
    double mu_y;
    double cov;
    bool   initialized;
};

inline void execute_ewcov(EWCOVState* state,
                          const double* BOLT_RESTRICT* in,
                          int64_t n,
                          double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(in != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    assert(state->desc.lambda >= 0.0);
    assert(state->desc.lambda <= 1.0);

    const double* BOLT_RESTRICT x = in[state->desc.x_col];
    const double* BOLT_RESTRICT y = in[state->desc.y_col];
    assert(x != nullptr || n == 0);
    assert(y != nullptr || n == 0);

    int64_t i = 0;
    if (!state->initialized && i < n) {
        state->mu_x        = x[i];
        state->mu_y        = y[i];
        state->cov         = 0.0;
        state->initialized = true;
        out[i] = 0.0;
        ++i;
    }
    const double lambda      = state->desc.lambda;
    const double one_minus_l = 1.0 - lambda;
    double mu_x = state->mu_x;
    double mu_y = state->mu_y;
    double cov  = state->cov;
    for (; i < n; ++i) {
        const double dx = x[i] - mu_x;
        const double dy = y[i] - mu_y;
        mu_x += one_minus_l * dx;
        mu_y += one_minus_l * dy;
        cov   = lambda * cov + one_minus_l * dx * dy;
        out[i] = cov;
    }
    state->mu_x = mu_x;
    state->mu_y = mu_y;
    state->cov  = cov;
}

inline EWCOVState* make_ewcov_state(::bolt::Arena* arena,
                                    int32_t x_col,
                                    int32_t y_col,
                                    int32_t out_col,
                                    double lambda) noexcept {
    assert(arena != nullptr);
    assert(lambda >= 0.0);
    assert(lambda <= 1.0);
    EWCOVState* s = arena->allocate_array<EWCOVState>(1);
    if (s == nullptr) return nullptr;
    s->desc.x_col   = x_col;
    s->desc.y_col   = y_col;
    s->desc.out_col = out_col;
    s->desc.lambda  = lambda;
    s->mu_x         = 0.0;
    s->mu_y         = 0.0;
    s->cov          = 0.0;
    s->initialized  = false;
    return s;
}

// Clear running state without changing column bindings or lambda. Next
// execute_ewcov() call will re-seed from its first sample.
inline void ewcov_reset(EWCOVState* state) noexcept {
    assert(state != nullptr);
    assert(state->desc.lambda >= 0.0);
    state->mu_x        = 0.0;
    state->mu_y        = 0.0;
    state->cov         = 0.0;
    state->initialized = false;
}

}  // namespace fintech
}  // namespace bolt
