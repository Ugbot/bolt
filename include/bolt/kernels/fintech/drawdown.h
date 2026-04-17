// bolt/kernels/fintech/drawdown.h — Phase 5.S.2.
//
// Drawdown (running, fractional) — tracks the maximum value seen and emits
// the fractional drawdown at every row:
//     peak[i] = max(peak[i-1], x[i])
//     out[i]  = (peak[i] - x[i]) / peak[i]      if peak[i] > 1e-15
//             = 0.0                              otherwise
//
// Guard: when peak <= 1e-15 (unlikely for real equity curves, but possible
// during warmup or with zero/negative series), we emit 0.0 instead of
// producing NaN or infinity. Matches chukonu exactly.
//
// Difference vs MaxDrawdown (P5.S.1):
//   MaxDrawdown emits an ABSOLUTE drop `peak - x`.
//   Drawdown   emits a FRACTIONAL drop `(peak - x) / peak`.
//
// Branchless audit: peak update is `(x > peak) ? x : peak` — cmov. The
// div-by-zero guard is `(peak > eps) ? num/denom : 0.0` — cmov. Zero
// data-dependent branches in the per-row steady-state loop. The warmup
// branch seeds peak from the first sample and is hoisted outside the
// row loop (warmup-split pattern).
//
// Source: chukonu/fintech/kernels.h :: CreateDrawdownProcessor (line 2901).

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

// Peak-guard epsilon — matches chukonu's literal 1e-15.
inline constexpr double kDrawdownPeakEps = 1e-15;

struct DrawdownOpDesc {
    int32_t in_col;
    int32_t out_col;
};

struct DrawdownState {
    DrawdownOpDesc desc;
    double peak;
    bool   initialized;
};

inline void execute_drawdown(DrawdownState* state,
                             const double* BOLT_RESTRICT* in,
                             int64_t n,
                             double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(in    != nullptr);
    assert(out   != nullptr || n == 0);
    assert(n >= 0);

    const double* BOLT_RESTRICT v = in[state->desc.in_col];
    assert(v != nullptr || n == 0);

    int64_t i = 0;
    // Warmup-split: seed peak on the first sample of the first call.
    if (!state->initialized && i < n) {
        state->peak        = v[i];
        state->initialized = true;
        // Fractional drawdown at seed is 0 by definition (peak == x).
        out[i]             = 0.0;
        ++i;
    }
    double peak = state->peak;
    for (; i < n; ++i) {
        const double x    = v[i];
        peak              = (x > peak) ? x : peak;
        // Branchless guard: compute the candidate, select 0.0 when the
        // denominator is too small. Both branches are cheap scalar ops so
        // we evaluate both and let cmov pick.
        const double num  = peak - x;
        const double safe = (peak > kDrawdownPeakEps) ? peak : 1.0;
        const double dd   = num / safe;
        out[i]            = (peak > kDrawdownPeakEps) ? dd : 0.0;
    }
    state->peak = peak;
}

inline DrawdownState* make_drawdown_state(::bolt::Arena* arena,
                                          int32_t in_col,
                                          int32_t out_col) noexcept {
    assert(arena != nullptr);
    assert(in_col >= 0);
    DrawdownState* s = arena->allocate_array<DrawdownState>(1);
    if (s == nullptr) return nullptr;
    s->desc.in_col     = in_col;
    s->desc.out_col    = out_col;
    s->peak            = 0.0;
    s->initialized     = false;
    return s;
}

inline void drawdown_reset(DrawdownState* state) noexcept {
    assert(state != nullptr);
    assert(state->desc.in_col >= 0);
    state->peak        = 0.0;
    state->initialized = false;
}

}  // namespace fintech
}  // namespace bolt
