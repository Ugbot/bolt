// bolt/kernels/fintech/log_returns.h — Phase 4b Tier 1 kernel.
//
// Log returns: out[0] = 0.0, out[i] = log(p[i] / p[i-1]).
//
// We evaluate the ratio directly (single div + single log) rather than
// materialising `log_column(prices)` then differencing — saves one pass
// over the input and keeps it branchless.
//
// Non-positive inputs yield NaN/-inf per IEEE; filtering is the caller's
// job (mirrors log_column's contract).
//
// Source: chukonu/fintech/kernels.h :: CreateLogReturnsProcessor (line 334).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct LogReturnsOpDesc {
    int32_t price_col;
    int32_t out_col;
};

// Inputs: in[0] = price (double).
// Output: out[0] = 0.0, out[i>=1] = log(p[i] / p[i-1]).
inline void log_returns_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                               double* BOLT_RESTRICT out,
                               const LogReturnsOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    if (n == 0) return;
    const double* BOLT_RESTRICT px = in[0];
    assert(px != nullptr);
    out[0] = 0.0;
    for (int64_t i = 1; i < n; ++i) {
        out[i] = std::log(px[i] / px[i - 1]);
    }
}

}  // namespace fintech
}  // namespace bolt
