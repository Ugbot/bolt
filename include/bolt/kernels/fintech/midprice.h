// bolt/kernels/fintech/midprice.h — Phase 4 Tier 1 kernel.
//
// Midprice: out[i] = (bid[i] + ask[i]) / 2
//
// Source: chukonu/fintech/kernels.h :: CreateMidpriceProcessor (line 83).
// Port shape: docs/research/gestalt-kernel-adapter.md (Tier 1 template).
//
// Phase 4 uses the simplified free-function signature (raw typed pointers
// + row count). BoltBatch/BoltColumn plumbing lands in Phase 4b.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

// POD descriptor: bound column indices only. Pinned at graph compile.
struct MidpriceOpDesc {
    int32_t bid_col;
    int32_t ask_col;
    int32_t out_col;
};

// Inputs: in[0] = bid (double), in[1] = ask (double).
// Output: out[i] = (bid[i] + ask[i]) / 2.
inline void midprice_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                            double* BOLT_RESTRICT out,
                            const MidpriceOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT bid = in[0];
    const double* BOLT_RESTRICT ask = in[1];
    assert(bid != nullptr || n == 0);
    assert(ask != nullptr || n == 0);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = (bid[i] + ask[i]) * 0.5;
    }
}

}  // namespace fintech
}  // namespace bolt
