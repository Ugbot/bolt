// bolt/kernels/fintech/historical_cvar.h — Phase 5.S.5 Tier 2 kernel.
//
// HistoricalCVaR: Conditional Value-at-Risk (Expected Shortfall) — mean
// of the returns strictly worse than the VaR threshold, over a rolling
// window of size `w`.
//
//   sorted = sort(returns[i-w+1 .. i])
//   var_index = floor((1 - confidence) * n_window)
//   cvar = (var_index > 0) ? -mean(sorted[0 .. var_index)) : 0.0
//
// Source: chukonu/fintech/kernels.h :: CreateHistoricalCVaRProcessor
// (line 721). Contract copied verbatim:
//   * NO warmup NaN — partial-window CVaR emitted from row 0.
//   * cvar is positive (loss convention); the `-mean` flips sign.
//   * When var_index == 0 (nw too small for the tail), emit 0.0.
//
// Complexity: the tail sum is O(var_index) = O(alpha * w) per row.
// Combined with SortedRing's O(w) push, steady-state cost is O(w) per
// row — bounded and deterministic for the typical Tier-2 windows.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/sorted_ring.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

template <uint32_t kCap>
struct HistoricalCVaRState {
    int32_t                        in_col;
    int32_t                        out_col;
    uint32_t                       window;
    double                         confidence;
    SortedRing<double, kCap>       ring;

    inline void init(int32_t in, int32_t out,
                     uint32_t w, double conf) noexcept {
        assert(w > 0u);
        assert(w <= kCap);
        assert(conf > 0.0);
        assert(conf < 1.0);
        in_col     = in;
        out_col    = out;
        window     = w;
        confidence = conf;
        ring.init(w);
    }
};

// Execute HistoricalCVaR across `n` rows. The tail-sum loop runs a
// compile-time-branchless inner loop over [0, var_index); the outer
// row body selects between `-sum/idx` and `0.0` via cmov. No data-
// dependent control-flow branches inside the hot loop body.
template <uint32_t kCap>
inline void execute_historical_cvar(HistoricalCVaRState<kCap>* state,
                                    const double* BOLT_RESTRICT x,
                                    int64_t n,
                                    double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(x     != nullptr || n == 0);
    assert(out   != nullptr || n == 0);
    assert(n >= 0);
    const double alpha = 1.0 - state->confidence;
    assert(alpha > 0.0);
    assert(alpha < 1.0);
    for (int64_t i = 0; i < n; ++i) {
        state->ring.push(x[i]);
        const uint32_t nw = state->ring.size();
        assert(nw > 0u);
        const uint32_t var_idx = static_cast<uint32_t>(alpha * static_cast<double>(nw));

        // Tail sum over sorted[0..var_idx). If var_idx == 0 the loop is
        // a no-op; sum remains 0.0 and the output cmov selects 0.0.
        double sum = 0.0;
        for (uint32_t j = 0; j < var_idx; ++j) {
            sum += state->ring.order(j);
        }
        const double denom   = (var_idx > 0u) ? static_cast<double>(var_idx) : 1.0;
        const double mean_t  = sum / denom;
        out[i] = (var_idx > 0u) ? (-mean_t) : 0.0;
    }
}

}  // namespace fintech
}  // namespace bolt
