// bolt/kernels/fintech/quoted_spread.h — Phase 4 Tier 1 kernel.
//
// Quoted spread: out[i] = ask[i] - bid[i]
//
// Source: chukonu/fintech/kernels.h :: CreateQuotedSpreadProcessor (line 188).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct QuotedSpreadOpDesc {
    int32_t bid_col;
    int32_t ask_col;
    int32_t out_col;
};

// Inputs: in[0] = bid, in[1] = ask.
inline void quoted_spread_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                                 double* BOLT_RESTRICT out,
                                 const QuotedSpreadOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT bid = in[0];
    const double* BOLT_RESTRICT ask = in[1];
    assert(bid != nullptr || n == 0);
    assert(ask != nullptr || n == 0);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = ask[i] - bid[i];
    }
}

}  // namespace fintech
}  // namespace bolt
