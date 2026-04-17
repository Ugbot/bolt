// bolt/kernels/fintech/funding_cost.h — Phase 4 Tier 1 kernel.
//
// Funding cost (per-period):
//   out[i] = position[i] * price[i] * funding_rate[i]
//
// Source: chukonu/fintech/kernels.h :: CreateFundingCostProcessor (1618).
// The plan-doc one-liner `notional · rate · dt_fraction` is the chukonu
// shape with `notional = position * price` fused into two multiplications
// and `dt_fraction` already baked into the per-period `funding_rate`.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct FundingCostOpDesc {
    int32_t position_col;
    int32_t price_col;
    int32_t funding_rate_col;
    int32_t out_col;
};

// Inputs: in[0] = position, in[1] = price, in[2] = funding_rate.
inline void funding_cost_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                                double* BOLT_RESTRICT out,
                                const FundingCostOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT position = in[0];
    const double* BOLT_RESTRICT price    = in[1];
    const double* BOLT_RESTRICT rate     = in[2];
    assert(position != nullptr || n == 0);
    assert(price    != nullptr || n == 0);
    assert(rate     != nullptr || n == 0);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = position[i] * price[i] * rate[i];
    }
}

}  // namespace fintech
}  // namespace bolt
