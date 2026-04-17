// bolt/kernels/fintech/effective_spread.h — Phase 4 Tier 1 kernel.
//
// Effective spread: out[i] = 2 * |exec_price[i] - midprice[i]|
//
// Source: chukonu/fintech/kernels.h :: CreateEffectiveSpreadProcessor (218).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct EffectiveSpreadOpDesc {
    int32_t exec_price_col;
    int32_t midprice_col;
    int32_t out_col;
};

// Inputs: in[0] = exec_price, in[1] = midprice.
inline void effective_spread_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                                    double* BOLT_RESTRICT out,
                                    const EffectiveSpreadOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT exec = in[0];
    const double* BOLT_RESTRICT mid  = in[1];
    assert(exec != nullptr || n == 0);
    assert(mid  != nullptr || n == 0);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = 2.0 * std::fabs(exec[i] - mid[i]);
    }
}

}  // namespace fintech
}  // namespace bolt
