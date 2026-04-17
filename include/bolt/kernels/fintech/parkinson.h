// bolt/kernels/fintech/parkinson.h — Phase 4b Tier 1 kernel.
//
// Parkinson (1980) high-low volatility estimator, per-bar:
//   out[i] = (1 / (4 · ln2)) · ln(h/l)^2
//
// Source: chukonu/fintech/kernels.h :: CreateParkinsonProcessor (3146).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct ParkinsonOpDesc {
    int32_t high_col;
    int32_t low_col;
    int32_t out_col;
};

// Inputs: in[0] = high, in[1] = low.
inline void parkinson_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                             double* BOLT_RESTRICT out,
                             const ParkinsonOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT high = in[0];
    const double* BOLT_RESTRICT low  = in[1];
    assert(high != nullptr || n == 0);
    assert(low  != nullptr || n == 0);
    const double coeff = 1.0 / (4.0 * std::log(2.0));
    for (int64_t i = 0; i < n; ++i) {
        const double ln_hl = std::log(high[i] / low[i]);
        out[i] = coeff * ln_hl * ln_hl;
    }
}

}  // namespace fintech
}  // namespace bolt
