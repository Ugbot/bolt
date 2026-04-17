// bolt/kernels/fintech/microprice.h — Phase 4 Tier 1 kernel.
//
// Microprice (size-weighted midprice):
//   out[i] = (bid[i] * ask_sz[i] + ask[i] * bid_sz[i]) / (bid_sz[i] + ask_sz[i])
//
// Source: chukonu/fintech/kernels.h :: CreateMicropriceProcessor (line 114).
// Port shape: docs/research/gestalt-kernel-adapter.md (Tier 1 template).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct MicropriceOpDesc {
    int32_t bid_col;
    int32_t ask_col;
    int32_t bid_size_col;
    int32_t ask_size_col;
    int32_t out_col;
};

// Inputs: in[0] = bid, in[1] = ask, in[2] = bid_size, in[3] = ask_size.
// Division by zero (bid_sz + ask_sz == 0) yields NaN/inf per IEEE —
// filtering degenerate quotes is the caller's job.
inline void microprice_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                              double* BOLT_RESTRICT out,
                              const MicropriceOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT bid = in[0];
    const double* BOLT_RESTRICT ask = in[1];
    const double* BOLT_RESTRICT bid_sz = in[2];
    const double* BOLT_RESTRICT ask_sz = in[3];
    assert(bid    != nullptr || n == 0);
    assert(ask    != nullptr || n == 0);
    assert(bid_sz != nullptr || n == 0);
    assert(ask_sz != nullptr || n == 0);
    for (int64_t i = 0; i < n; ++i) {
        const double bs = bid_sz[i];
        const double as = ask_sz[i];
        out[i] = (bid[i] * as + ask[i] * bs) / (bs + as);
    }
}

}  // namespace fintech
}  // namespace bolt
