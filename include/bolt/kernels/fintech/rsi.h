// bolt/kernels/fintech/rsi.h — Phase 5.E.5.
//
// RSI (Relative Strength Index, Wilder 1978):
//   change[i]     = x[i] - x[i-1]
//   gain[i]       = max(change, 0)
//   loss[i]       = max(-change, 0)
//   avg_gain[i]   = EMA_wilder(gain, period)
//   avg_loss[i]   = EMA_wilder(loss, period)
//   rs[i]         = avg_gain / avg_loss  (or 100 when avg_loss == 0)
//   rsi[i]        = 100 - 100 / (1 + rs)
//
// Critical convention: Wilder's smoothing uses alpha = 1/period, NOT the
// classical EMA alpha = 2/(period+1). See the design-log entry for this
// port for why we branched from EmaState::init (which uses 2/(p+1)) and
// set alpha directly.
//
// Output contract (matches chukonu):
//   - The first sample in the *first ever* call emits 50.0 (neutral),
//     because there is no prior value to diff against.
//   - Subsequent samples emit the recurrence above. State carries the
//     previous value so the first sample of batch N+1 diffs correctly
//     against the last sample of batch N.
//
// Source: chukonu/fintech/kernels.h :: CreateRSIProcessor (line 519).

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct RSIOpDesc {
    int32_t  in_col;
    int32_t  out_col;
    uint32_t period;
};

// State: two Wilder-smoothed EMAs (of gains and losses) plus the
// previous sample value (so we can diff across batch boundaries).
struct RSIState {
    RSIOpDesc desc;
    ::bolt::fintech::EmaState avg_gain;  // alpha = 1 / period  (Wilder)
    ::bolt::fintech::EmaState avg_loss;  // alpha = 1 / period  (Wilder)
    double    prev_value;                // last x seen; valid iff has_prev
    bool      has_prev;
};

inline void execute_rsi(RSIState* state,
                        const double* BOLT_RESTRICT* in,
                        int64_t n,
                        double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(in != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    assert(state->avg_gain.alpha > 0.0);
    assert(state->avg_gain.alpha <= 1.0);

    const double* BOLT_RESTRICT v = in[state->desc.in_col];
    assert(v != nullptr || n == 0);

    int64_t i = 0;
    // First sample ever: no prior → emit neutral 50.0 and remember value.
    if (!state->has_prev && i < n) {
        out[i] = 50.0;
        state->prev_value = v[i];
        state->has_prev   = true;
        ++i;
    }
    double prev = state->prev_value;
    for (; i < n; ++i) {
        const double x      = v[i];
        const double change = x - prev;
        const double gain   = (change > 0.0) ?  change : 0.0;
        const double loss   = (change < 0.0) ? -change : 0.0;
        const double ag     = state->avg_gain.update(gain);
        const double al     = state->avg_loss.update(loss);
        // Matches chukonu: if avg_loss == 0, treat RS as 100 (very
        // strong up-move), which pushes RSI to 100 - 100/101 ≈ 99.0099.
        // This is the chukonu convention; canonical Wilder says "RSI = 100"
        // outright when avg_loss == 0. We follow the source we port.
        const double rs     = (al > 0.0) ? (ag / al) : 100.0;
        out[i]              = 100.0 - (100.0 / (1.0 + rs));
        prev                = x;
    }
    state->prev_value = prev;
}

inline RSIState* make_rsi_state(::bolt::Arena* arena,
                                int32_t in_col,
                                int32_t out_col,
                                uint32_t period) noexcept {
    assert(arena != nullptr);
    assert(period >= 1u);
    RSIState* s = arena->allocate_array<RSIState>(1);
    if (s == nullptr) return nullptr;
    s->desc.in_col  = in_col;
    s->desc.out_col = out_col;
    s->desc.period  = period;
    // Wilder smoothing: alpha = 1/period. Bypass EmaState::init's
    // 2/(p+1) formula. See design-log "RSI Wilder vs classical EMA".
    const double a = 1.0 / static_cast<double>(period);
    s->avg_gain.alpha       = a;
    s->avg_gain.value       = 0.0;
    s->avg_gain.initialized = true;   // seeded at 0, not from first sample
    s->avg_loss.alpha       = a;
    s->avg_loss.value       = 0.0;
    s->avg_loss.initialized = true;
    s->prev_value           = 0.0;
    s->has_prev             = false;
    return s;
}

inline void rsi_reset(RSIState* state) noexcept {
    assert(state != nullptr);
    // Keep alpha, re-arm as initialised-at-zero (matches make_rsi_state).
    state->avg_gain.value       = 0.0;
    state->avg_gain.initialized = true;
    state->avg_loss.value       = 0.0;
    state->avg_loss.initialized = true;
    state->prev_value           = 0.0;
    state->has_prev             = false;
}

}  // namespace fintech
}  // namespace bolt
