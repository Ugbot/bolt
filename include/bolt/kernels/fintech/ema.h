// bolt/kernels/fintech/ema.h — Phase 3 / P3.1 worked example.
//
// EMA: y[i] = alpha * x[i] + (1 - alpha) * y[i-1], alpha = 2 / (period + 1).
// First sample seeds the state (no warmup ramp).
//
// Source: chukonu/fintech/kernels.h :: CreateEMAProcessor (line 482).
// Template: docs/research/gestalt-kernel-adapter.md — "Operator definition
// (Tier 2 — stateful)" section.
//
// State persists across execute_ema calls for the graph's lifetime. The
// EMAState struct is POD, pinned in a bolt::Arena at graph-compile time.
// execute_ema does NOT re-init; callers who want a fresh series call
// state->ema.reset() first.
//
// Tiger Style:
//   - POD state, noexcept, BOLT_RESTRICT on hot pointers, >=2 asserts,
//     functions <= 70 lines. Zero allocation on the hot path; the only
//     allocation is a single arena bump at graph compile via
//     make_ema_state().
//
// Performance target: 5-20x vs the Arrow/chukonu version (per the
// gestalt-kernel-adapter.md benchmark claim). In-tree benchmark against
// Arrow is a followup — we don't link libarrow.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

// POD descriptor: column bindings + period. Pinned at graph compile.
struct EMAOpDesc {
    int32_t  in_col;
    int32_t  out_col;
    uint32_t period;
};

// Per-graph stateful kernel state. EmaState (Phase 2 substrate) holds the
// running value; EMAOpDesc holds the column binding. POD layout — zero-
// initialisable, trivially copyable.
struct EMAState {
    EMAOpDesc desc;
    ::bolt::fintech::EmaState ema;
};

// Execute the EMA kernel over `n` samples pulled from in[desc.in_col].
// `out` receives n doubles. State persists across calls.
//
// Contract:
//   - state != nullptr, state->ema.alpha already configured by make_ema_state.
//   - in != nullptr and in[desc.in_col] != nullptr when n > 0.
//   - out != nullptr when n > 0.
//   - n >= 0.
//
// This is the Tier 2 body from gestalt-kernel-adapter.md lines 118-143,
// retargeted at raw typed pointers (Phase 4a signature convention).
inline void execute_ema(EMAState* state,
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
    // Seed the state from the first sample if not yet initialised.
    if (!state->ema.initialized && i < n) {
        state->ema.value       = v[i];
        state->ema.initialized = true;
        out[i] = v[i];
        ++i;
    }
    // Steady-state recurrence; branchless in the inner loop, vectorises
    // as a scalar reduction (auto-vec restricted by the loop-carried
    // dependency on state->ema.value, but each iteration is ~3 FLOPs
    // and memory-bound anyway).
    const double alpha        = state->ema.alpha;
    const double one_minus_a  = 1.0 - alpha;
    double       ema_value    = state->ema.value;
    for (; i < n; ++i) {
        ema_value = alpha * v[i] + one_minus_a * ema_value;
        out[i]    = ema_value;
    }
    state->ema.value = ema_value;
}

// Arena-pinned constructor. Call once at graph compile time. Returns
// nullptr on OOM — caller must check before using the state. The arena
// must outlive every execute_ema() call that references this state.
inline EMAState* make_ema_state(::bolt::Arena* arena,
                                int32_t in_col,
                                int32_t out_col,
                                uint32_t period) noexcept {
    assert(arena != nullptr);
    assert(period >= 1u);
    EMAState* s = arena->allocate_array<EMAState>(1);
    if (s == nullptr) return nullptr;
    s->desc.in_col  = in_col;
    s->desc.out_col = out_col;
    s->desc.period  = period;
    s->ema.init(static_cast<int32_t>(period));
    assert(s->ema.alpha > 0.0);
    assert(s->ema.alpha <= 1.0);
    return s;
}

}  // namespace fintech
}  // namespace bolt
