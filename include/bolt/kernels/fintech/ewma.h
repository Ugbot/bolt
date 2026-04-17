// bolt/kernels/fintech/ewma.h — Phase 5.E.2.
//
// EWMA (exponentially-weighted moving average) with a caller-supplied alpha.
// y[i] = alpha * x[i] + (1 - alpha) * y[i-1]; first sample seeds.
//
// The functional shape is identical to EMA — only the alpha-provision
// differs. EMA derives alpha from a period (2/(p+1)); EWMA takes alpha
// directly. Typical RiskMetrics convention: lambda = 0.94, alpha = 1 -
// lambda = 0.06; we follow chukonu's CreateEWMAProcessor which takes
// alpha directly (default 0.94 in chukonu — the "fast-decay" flavour).
//
// Source: chukonu/fintech/kernels.h :: CreateEWMAProcessor (line 809).
// Reuses Phase 2's EmaState substrate; we just bypass EmaState::init's
// 2/(p+1) formula and set alpha manually.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct EWMAOpDesc {
    int32_t in_col;
    int32_t out_col;
    double  alpha;  // 0.0 <= alpha <= 1.0; caller supplied.
};

struct EWMAState {
    EWMAOpDesc desc;
    ::bolt::fintech::EmaState ema;
};

inline void execute_ewma(EWMAState* state,
                         const double* BOLT_RESTRICT* in,
                         int64_t n,
                         double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(in != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    assert(state->ema.alpha >= 0.0);
    assert(state->ema.alpha <= 1.0);

    const double* BOLT_RESTRICT v = in[state->desc.in_col];
    assert(v != nullptr || n == 0);

    int64_t i = 0;
    if (!state->ema.initialized && i < n) {
        state->ema.value       = v[i];
        state->ema.initialized = true;
        out[i] = v[i];
        ++i;
    }
    const double alpha       = state->ema.alpha;
    const double one_minus_a = 1.0 - alpha;
    double       ema_value   = state->ema.value;
    for (; i < n; ++i) {
        ema_value = alpha * v[i] + one_minus_a * ema_value;
        out[i]    = ema_value;
    }
    state->ema.value = ema_value;
}

// Arena-pinned constructor. alpha must lie in [0, 1]. Returns nullptr on
// OOM.
inline EWMAState* make_ewma_state(::bolt::Arena* arena,
                                  int32_t in_col,
                                  int32_t out_col,
                                  double alpha) noexcept {
    assert(arena != nullptr);
    assert(alpha >= 0.0);
    assert(alpha <= 1.0);
    EWMAState* s = arena->allocate_array<EWMAState>(1);
    if (s == nullptr) return nullptr;
    s->desc.in_col  = in_col;
    s->desc.out_col = out_col;
    s->desc.alpha   = alpha;
    // Bypass EmaState::init's 2/(p+1) formula — set alpha directly.
    s->ema.alpha       = alpha;
    s->ema.value       = 0.0;
    s->ema.initialized = false;
    return s;
}

}  // namespace fintech
}  // namespace bolt
