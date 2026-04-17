// bolt/kernels/fintech/rogers_satchell.h — Phase 4b Tier 1 kernel.
//
// Rogers-Satchell (1991) drift-independent OHLC volatility, per-bar:
//   out[i] = ln(h/c)·ln(h/o) + ln(l/c)·ln(l/o)
//
// Source: chukonu/fintech/kernels.h :: CreateRogersSatchellProcessor (3186).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct RogersSatchellOpDesc {
    int32_t high_col;
    int32_t low_col;
    int32_t open_col;
    int32_t close_col;
    int32_t out_col;
};

// Inputs: in[0]=high, in[1]=low, in[2]=open, in[3]=close.
inline void rogers_satchell_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                                   double* BOLT_RESTRICT out,
                                   const RogersSatchellOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT high  = in[0];
    const double* BOLT_RESTRICT low   = in[1];
    const double* BOLT_RESTRICT open  = in[2];
    const double* BOLT_RESTRICT close = in[3];
    assert(high  != nullptr || n == 0);
    assert(low   != nullptr || n == 0);
    assert(open  != nullptr || n == 0);
    assert(close != nullptr || n == 0);
    for (int64_t i = 0; i < n; ++i) {
        const double h = high[i], l = low[i], o = open[i], c = close[i];
        out[i] = std::log(h / c) * std::log(h / o)
               + std::log(l / c) * std::log(l / o);
    }
}

}  // namespace fintech
}  // namespace bolt
