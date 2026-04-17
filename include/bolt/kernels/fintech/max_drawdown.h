// bolt/kernels/fintech/max_drawdown.h — Phase 5.S.1.
//
// MaxDrawdown (running) — tracks the maximum value seen across the entire
// stream and emits the absolute drawdown at every row:
//     peak[i] = max(peak[i-1], x[i])
//     out[i]  = peak[i] - x[i]
//
// This is the streaming form of column_primitives::running_max_drawdown<T>
// with cross-batch state persistence: the `peak` field in MaxDrawdownState
// is carried across execute_* calls so that batch N+1 sees the cumulative
// peak from all prior batches, not just a per-batch reset.
//
// Shape note: chukonu's CreateMaxDrawdownProcessor emits a single scalar
// fraction (max of (peak - val) / peak) broadcast to every row. Bolt's
// equivalent is a per-row absolute drawdown — the streaming shape Phase 1
// settled on (see P1.5). The fractional per-row form lives in drawdown.h
// (P5.S.2). The "broadcast scalar" form from chukonu is not useful under
// streaming morsel-driven execution, where the true max may still be in
// the future; we drop it deliberately.
//
// Branchless audit: peak update is `(x > peak) ? x : peak` — cmov/csel.
// Output is `peak - x` with no conditional. Zero data-dependent branches
// in the per-row loop.
//
// Source reference: chukonu/fintech/kernels.h :: CreateMaxDrawdownProcessor
// (line 1141). Primitive reference: column_primitives::running_max_drawdown.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

// POD descriptor: column bindings. Pinned at graph compile.
struct MaxDrawdownOpDesc {
    int32_t in_col;
    int32_t out_col;
};

// Cross-batch state: running peak + a "has_seen_anything" flag so the very
// first sample seeds the peak with its own value (not 0, which would make
// the first drawdown incorrectly positive for any x[0] < 0).
struct MaxDrawdownState {
    MaxDrawdownOpDesc desc;
    double peak;
    bool   initialized;
};

// Execute MaxDrawdown over `n` rows. State persists across calls.
// Contract:
//   - state != nullptr, in != nullptr, out != nullptr || n == 0, n >= 0.
//   - in[desc.in_col] holds n doubles (prices / equity path).
//
// Branchless inner loop: peak update uses a ternary (compiles to cmov);
// emit is a subtraction. If the state is not yet initialised, we pay one
// branch to seed on the very first sample of the very first call. That
// branch is hoisted outside the steady-state loop (warmup-split pattern),
// so the steady-state path is 100% branchless.
inline void execute_max_drawdown(MaxDrawdownState* state,
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
    // Warmup branch: seed the peak from the first sample ever seen. This
    // is hoisted out of the steady-state loop — zero per-row branching.
    if (!state->initialized && i < n) {
        state->peak        = v[i];
        state->initialized = true;
        out[i]             = 0.0;
        ++i;
    }
    double peak = state->peak;
    for (; i < n; ++i) {
        const double x = v[i];
        // Branchless running max: cmov-style ternary.
        peak    = (x > peak) ? x : peak;
        out[i]  = peak - x;
    }
    state->peak = peak;
}

// Arena-pinned constructor. Returns nullptr on OOM.
inline MaxDrawdownState* make_max_drawdown_state(::bolt::Arena* arena,
                                                 int32_t in_col,
                                                 int32_t out_col) noexcept {
    assert(arena != nullptr);
    assert(in_col  >= 0);
    MaxDrawdownState* s = arena->allocate_array<MaxDrawdownState>(1);
    if (s == nullptr) return nullptr;
    s->desc.in_col     = in_col;
    s->desc.out_col    = out_col;
    s->peak            = 0.0;
    s->initialized     = false;
    return s;
}

inline void max_drawdown_reset(MaxDrawdownState* state) noexcept {
    assert(state != nullptr);
    assert(state->desc.in_col >= 0);
    state->peak        = 0.0;
    state->initialized = false;
}

}  // namespace fintech
}  // namespace bolt
