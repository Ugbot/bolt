// bolt/kernels/fintech/kyle_lambda.h — Phase 5.S.11 Tier 2 kernel.
//
// KyleLambda: rolling OLS slope of price_change on signed_volume over a
// window of size `w`. The slope is the market-impact coefficient from
// Kyle's (1985) model — how much price moves per unit of signed order flow.
//
//   mean_dp = mean(dp[i-w+1 .. i])
//   mean_sv = mean(sv[i-w+1 .. i])
//   cov     = Σ (dp_j - mean_dp) (sv_j - mean_sv)
//   var_sv  = Σ (sv_j - mean_sv)²
//   lambda  = cov / var_sv              (OLS slope)
//
// Source: chukonu/fintech/kernels.h :: CreateKyleLambdaProcessor (line 2508).
// Contract copied verbatim:
//   * w < 2 → output 0.0 (handled by warmup-split, not a per-row branch).
//   * var_sv <= 1e-15 → output 0.0 (cmov guard).
//
// State layout: two RollingRing<double, kCap> — one per series — and a
// rescan per row, matching rolling_correlation.h. Design-log "Rolling-
// window Welford — rescan vs decrement" applies identically.
//
// Branchless note: warmup rows are computed up-front and emitted in a
// tiny prefix loop (warmup-split pattern); the steady-state loop has no
// data-dependent control-flow branches, only cmov ternaries.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/state.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

template <uint32_t kCap>
struct KyleLambdaState {
    int32_t                      dp_col;
    int32_t                      sv_col;
    int32_t                      out_col;
    uint32_t                     window;
    RollingRing<double, kCap>    ring_dp;
    RollingRing<double, kCap>    ring_sv;

    inline void init(int32_t dp, int32_t sv, int32_t out, uint32_t w) noexcept {
        assert(w > 0u);
        assert(w <= kCap);
        dp_col  = dp;
        sv_col  = sv;
        out_col = out;
        window  = w;
        ring_dp.init(w);
        ring_sv.init(w);
    }
};

template <uint32_t kCap>
inline void execute_kyle_lambda(KyleLambdaState<kCap>* state,
                                const double* BOLT_RESTRICT dp,
                                const double* BOLT_RESTRICT sv,
                                int64_t n,
                                double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(dp  != nullptr || n == 0);
    assert(sv  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    // Warmup split — see rolling_correlation.h for the derivation.
    // chukonu emits 0.0 while ring fill < 2. Post-push size < need
    // → warmup row count = max(0, need - have - 1).
    const uint32_t have   = state->ring_dp.size();
    const uint32_t need   = 2u;
    const uint32_t warmup = (have + 1u >= need) ? 0u : (need - have - 1u);
    const int64_t warm_n = (n < static_cast<int64_t>(warmup))
                         ? n : static_cast<int64_t>(warmup);

    for (int64_t i = 0; i < warm_n; ++i) {
        state->ring_dp.push(dp[i]);
        state->ring_sv.push(sv[i]);
        out[i] = 0.0;
    }

    // Steady-state: ring size >= 2 guaranteed, no per-row warmup branch.
    for (int64_t i = warm_n; i < n; ++i) {
        state->ring_dp.push(dp[i]);
        state->ring_sv.push(sv[i]);
        const uint32_t w = state->ring_dp.size();
        assert(w >= 2u);
        assert(w == state->ring_sv.size());

        double sum_dp = 0.0, sum_sv = 0.0;
        for (uint32_t j = 0; j < w; ++j) {
            sum_dp += state->ring_dp.back(j);
            sum_sv += state->ring_sv.back(j);
        }
        const double mean_dp = sum_dp / static_cast<double>(w);
        const double mean_sv = sum_sv / static_cast<double>(w);

        double cov = 0.0, var_sv = 0.0;
        for (uint32_t j = 0; j < w; ++j) {
            const double d_dp = state->ring_dp.back(j) - mean_dp;
            const double d_sv = state->ring_sv.back(j) - mean_sv;
            cov    += d_dp * d_sv;
            var_sv += d_sv * d_sv;
        }
        // cmov guard: near-zero variance → 0.0. No control-flow branch.
        out[i] = (var_sv > 1e-15) ? (cov / var_sv) : 0.0;
    }
}

template <uint32_t kCap>
inline KyleLambdaState<kCap>* make_kyle_lambda_state(::bolt::Arena* arena,
                                                     int32_t dp_col,
                                                     int32_t sv_col,
                                                     int32_t out_col,
                                                     uint32_t window) noexcept {
    assert(arena != nullptr);
    assert(window > 0u);
    assert(window <= kCap);
    KyleLambdaState<kCap>* s = arena->allocate_array<KyleLambdaState<kCap>>(1);
    if (s == nullptr) return nullptr;
    s->init(dp_col, sv_col, out_col, window);
    return s;
}

}  // namespace fintech
}  // namespace bolt
