// bolt/kernels/fintech/funding_apr.h — Phase 4 Tier 1 kernel.
//
// Funding APR conversion (per-period funding rate -> annual % rate):
//   out[i] = rate[i] * periods_per_year * 100
//
// Source: chukonu/fintech/kernels.h :: CreateFundingAPRProcessor (3835).
// `periods_per_year` for common perps: 8h funding -> 365 * 3 = 1095,
// 1h funding -> 8760. Caller supplies via the desc.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct FundingAPROpDesc {
    int32_t rate_col;
    int32_t out_col;
    double  periods_per_year;
};

// Inputs: in[0] = rate. Single-input kernel; array-of-pointers form kept
// for uniformity with the rest of Phase 4.
inline void funding_apr_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                               double* BOLT_RESTRICT out,
                               const FundingAPROpDesc& desc) noexcept {
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT rate = in[0];
    assert(rate != nullptr || n == 0);
    const double scale = desc.periods_per_year * 100.0;
    for (int64_t i = 0; i < n; ++i) {
        out[i] = rate[i] * scale;
    }
}

}  // namespace fintech
}  // namespace bolt
