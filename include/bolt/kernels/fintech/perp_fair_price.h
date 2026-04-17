// bolt/kernels/fintech/perp_fair_price.h — Phase 4 Tier 1 kernel.
//
// Perpetual-swap fair price (crypto convention, 8-hour funding cadence):
//   out[i] = spot[i] * (1 + funding_rate[i] * hours[i] / 8)
//
// Source: chukonu/fintech/kernels.h :: CreatePerpFairPriceProcessor (3866).
// `hours` is hours-to-next-funding; `8` is the standard exchange cadence.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct PerpFairPriceOpDesc {
    int32_t spot_col;
    int32_t funding_rate_col;
    int32_t hours_col;
    int32_t out_col;
};

// Inputs: in[0] = spot, in[1] = funding_rate, in[2] = hours_to_funding.
inline void perp_fair_price_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                                   double* BOLT_RESTRICT out,
                                   const PerpFairPriceOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT spot  = in[0];
    const double* BOLT_RESTRICT fr    = in[1];
    const double* BOLT_RESTRICT hours = in[2];
    assert(spot  != nullptr || n == 0);
    assert(fr    != nullptr || n == 0);
    assert(hours != nullptr || n == 0);
    constexpr double kInv8 = 1.0 / 8.0;
    for (int64_t i = 0; i < n; ++i) {
        out[i] = spot[i] * (1.0 + fr[i] * hours[i] * kInv8);
    }
}

}  // namespace fintech
}  // namespace bolt
