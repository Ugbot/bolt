// bolt/kernels/fintech/l1_imbalance.h — Phase 4 Tier 1 kernel.
//
// L1 Order Book Imbalance:
//   out[i] = (bid_sz[i] - ask_sz[i]) / (bid_sz[i] + ask_sz[i])
//
// Range is [-1, +1]. When both sides are zero the result is NaN (0/0) per
// IEEE — callers who need a sentinel must filter.
//
// Source: chukonu/fintech/kernels.h :: CreateL1ImbalanceProcessor (line 154).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct L1ImbalanceOpDesc {
    int32_t bid_size_col;
    int32_t ask_size_col;
    int32_t out_col;
};

// Inputs: in[0] = bid_size, in[1] = ask_size.
inline void l1_imbalance_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                                double* BOLT_RESTRICT out,
                                const L1ImbalanceOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT bid_sz = in[0];
    const double* BOLT_RESTRICT ask_sz = in[1];
    assert(bid_sz != nullptr || n == 0);
    assert(ask_sz != nullptr || n == 0);
    for (int64_t i = 0; i < n; ++i) {
        const double bs = bid_sz[i];
        const double as = ask_sz[i];
        out[i] = (bs - as) / (bs + as);
    }
}

}  // namespace fintech
}  // namespace bolt
