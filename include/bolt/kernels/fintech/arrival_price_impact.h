// bolt/kernels/fintech/arrival_price_impact.h — Phase 4 Tier 1 kernel.
//
// Arrival-price impact in basis points:
//   out[i] = 10_000 * (exec_price[i] - arrival_mid[i]) / arrival_mid[i]
//
// If |arrival_mid[i]| < 1e-15, out[i] = 0 (chukonu sentinel). Matches
// the buy-side TCA convention: positive bps = paid more than arrival mid.
//
// Source: chukonu/fintech/kernels.h :: CreateArrivalPriceImpactProcessor (2464).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct ArrivalPriceImpactOpDesc {
    int32_t exec_price_col;
    int32_t arrival_mid_col;
    int32_t out_col;
};

// Inputs: in[0] = exec_price, in[1] = arrival_mid.
inline void arrival_price_impact_kernel(const double* BOLT_RESTRICT* in,
                                        int64_t n,
                                        double* BOLT_RESTRICT out,
                                        const ArrivalPriceImpactOpDesc& desc)
    noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT exec = in[0];
    const double* BOLT_RESTRICT mid  = in[1];
    assert(exec != nullptr || n == 0);
    assert(mid  != nullptr || n == 0);
    for (int64_t i = 0; i < n; ++i) {
        const double m = mid[i];
        out[i] = (std::fabs(m) > 1e-15) ? 10000.0 * (exec[i] - m) / m : 0.0;
    }
}

}  // namespace fintech
}  // namespace bolt
