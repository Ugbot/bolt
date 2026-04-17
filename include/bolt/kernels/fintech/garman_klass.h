// bolt/kernels/fintech/garman_klass.h — Phase 4b Tier 1 kernel.
//
// Garman-Klass (1980) OHLC volatility estimator, per-bar:
//   out[i] = 0.5 · ln(h/l)^2 - (2·ln2 - 1) · ln(c/o)^2
//
// Source: chukonu/fintech/kernels.h :: CreateGarmanKlassProcessor (3099).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct GarmanKlassOpDesc {
    int32_t high_col;
    int32_t low_col;
    int32_t open_col;
    int32_t close_col;
    int32_t out_col;
};

// Inputs: in[0]=high, in[1]=low, in[2]=open, in[3]=close.
inline void garman_klass_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                                double* BOLT_RESTRICT out,
                                const GarmanKlassOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT high  = in[0];
    const double* BOLT_RESTRICT low   = in[1];
    const double* BOLT_RESTRICT open  = in[2];
    const double* BOLT_RESTRICT close = in[3];
    assert(high  != nullptr || n == 0);
    assert(low   != nullptr || n == 0);
    assert(open  != nullptr || n == 0);
    assert(close != nullptr || n == 0);
    const double coeff = 2.0 * std::log(2.0) - 1.0;
    for (int64_t i = 0; i < n; ++i) {
        const double ln_hl = std::log(high[i]  / low[i]);
        const double ln_co = std::log(close[i] / open[i]);
        out[i] = 0.5 * ln_hl * ln_hl - coeff * ln_co * ln_co;
    }
}

}  // namespace fintech
}  // namespace bolt
