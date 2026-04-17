// bolt/kernels/fintech/historical_var.h — Phase 5.S.4 Tier 2 kernel.
//
// HistoricalVaR: historical Value-at-Risk — rolling `(1-confidence)`
// quantile of the returns distribution over a window of size `w`.
//
//   sorted = sort(returns[i-w+1 .. i])
//   var_index = floor((1 - confidence) * n_window)
//   var = (var_index < n_window) ? -sorted[var_index] : 0.0
//
// Source: chukonu/fintech/kernels.h :: CreateHistoricalVaRProcessor
// (line 676). Contract copied verbatim:
//   * NO warmup NaN — partial-window VaR emitted from row 0 (see note).
//   * var is a positive number (loss convention); emitted as `-sorted[idx]`.
//   * confidence in (0, 1); typical 0.95 or 0.99.
//
// Deviation from the plan one-liner: the plan said "emit NaN for warmup
// rows to match chukonu". Reading the chukonu source shows chukonu does
// NOT emit NaN — it emits the partial-window quantile from row 0, and
// only falls back to 0.0 when var_index >= sorted.size() (which happens
// on the very first row when n_window=1 and 1-confidence <= 0 rounds to
// index 0, actually n_window=1 → var_index=0 < 1, so sorted[0] is taken
// — chukonu never hits the zero branch under typical confidence). We
// match chukonu exactly.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/sorted_ring.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

// Pinned per-graph POD state. kCap is the compile-time max window size.
template <uint32_t kCap>
struct HistoricalVaRState {
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

// Execute HistoricalVaR across `n` rows. Per-row body is branchless: the
// `var_index < n_window` guard compiles to a cmov on x86-64 (double =
// cond ? a : b). No data-dependent control flow inside the row loop.
template <uint32_t kCap>
inline void execute_historical_var(HistoricalVaRState<kCap>* state,
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
        // var_index = floor(alpha * nw), clamped to [0, nw-1] via cmov.
        // floor of a positive number is trunc in C++ integer cast, and
        // the SortedRing guarantees sorted[nw-1] exists so the (idx < nw)
        // select is a cmov, not a branch on memory.
        const uint32_t raw_idx = static_cast<uint32_t>(alpha * static_cast<double>(nw));
        const bool     in_range = (raw_idx < nw);
        const uint32_t safe_idx = in_range ? raw_idx : 0u;
        const double   q        = state->ring.order(safe_idx);
        out[i] = in_range ? (-q) : 0.0;
    }
}

}  // namespace fintech
}  // namespace bolt
