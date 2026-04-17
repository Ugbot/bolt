// bolt/kernels/fintech/macd.h — Phase 5.E.4.
//
// MACD — Moving Average Convergence Divergence. Three outputs:
//   macd_line[i]      = EMA(x, fast)[i] - EMA(x, slow)[i]
//   macd_signal[i]    = EMA(macd_line, signal)[i]
//   macd_histogram[i] = macd_line[i] - macd_signal[i]
//
// Conventional periods: fast=12, slow=26, signal=9. Alphas derived via
// alpha = 2 / (period + 1) — classical EMA convention (not Wilder). Each
// of the three sub-EMAs seeds from the first sample (chukonu semantics:
// at i=0 ema_fast=ema_slow=x[0] so macd_line[0]=0, macd_signal[0]=0).
//
// Source: chukonu/fintech/kernels.h :: CreateMACDProcessor (line 904).
//
// Uses three EmaState substrates (Phase 2). The signal EMA is fed with
// the MACD-line value, not the raw input — so its update() happens
// inside the MACD loop body.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct MACDOpDesc {
    int32_t  in_col;
    int32_t  line_col;
    int32_t  signal_col;
    int32_t  hist_col;
    uint32_t fast_period;
    uint32_t slow_period;
    uint32_t signal_period;
};

struct MACDState {
    MACDOpDesc desc;
    ::bolt::fintech::EmaState ema_fast;
    ::bolt::fintech::EmaState ema_slow;
    ::bolt::fintech::EmaState ema_signal;
};

// Execute MACD. Emits three output streams. out_line/out_signal/out_hist
// all must have capacity >= n.
inline void execute_macd(MACDState* state,
                         const double* BOLT_RESTRICT* in,
                         int64_t n,
                         double* BOLT_RESTRICT out_line,
                         double* BOLT_RESTRICT out_signal,
                         double* BOLT_RESTRICT out_hist) noexcept {
    assert(state != nullptr);
    assert(in != nullptr);
    assert(n >= 0);
    assert(out_line   != nullptr || n == 0);
    assert(out_signal != nullptr || n == 0);
    assert(out_hist   != nullptr || n == 0);

    const double* BOLT_RESTRICT v = in[state->desc.in_col];
    assert(v != nullptr || n == 0);

    for (int64_t i = 0; i < n; ++i) {
        const double x         = v[i];
        const double ema_f     = state->ema_fast.update(x);
        const double ema_s     = state->ema_slow.update(x);
        const double macd_line = ema_f - ema_s;
        // Signal EMA feeds on the MACD line itself. Using EmaState::update
        // automatically seeds the signal at macd_line on the first call
        // (matches chukonu's "if i==0: macd_signal = macd_line" branch).
        const double signal    = state->ema_signal.update(macd_line);
        out_line[i]   = macd_line;
        out_signal[i] = signal;
        out_hist[i]   = macd_line - signal;
    }
}

inline MACDState* make_macd_state(::bolt::Arena* arena,
                                  int32_t in_col,
                                  int32_t line_col,
                                  int32_t signal_col,
                                  int32_t hist_col,
                                  uint32_t fast_period,
                                  uint32_t slow_period,
                                  uint32_t signal_period) noexcept {
    assert(arena != nullptr);
    assert(fast_period   >= 1u);
    assert(slow_period   >= 1u);
    assert(signal_period >= 1u);
    MACDState* s = arena->allocate_array<MACDState>(1);
    if (s == nullptr) return nullptr;
    s->desc.in_col        = in_col;
    s->desc.line_col      = line_col;
    s->desc.signal_col    = signal_col;
    s->desc.hist_col      = hist_col;
    s->desc.fast_period   = fast_period;
    s->desc.slow_period   = slow_period;
    s->desc.signal_period = signal_period;
    s->ema_fast.init(static_cast<int32_t>(fast_period));
    s->ema_slow.init(static_cast<int32_t>(slow_period));
    s->ema_signal.init(static_cast<int32_t>(signal_period));
    return s;
}

inline void macd_reset(MACDState* state) noexcept {
    assert(state != nullptr);
    state->ema_fast.reset();
    state->ema_slow.reset();
    state->ema_signal.reset();
}

}  // namespace fintech
}  // namespace bolt
