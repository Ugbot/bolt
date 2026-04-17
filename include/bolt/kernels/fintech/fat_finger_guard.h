// bolt/kernels/fintech/fat_finger_guard.h — Phase 4b Tier 1 kernel.
//
// Fat-finger guard: emit 1 when the order trips either the price band
// OR the notional ceiling, 0 when both are within limits. Matches the
// task-spec polarity (1 = trip) — inverse of chukonu's ok-flag.
//
// Trip condition (any of):
//   - |order_price / ref_price - 1| >= band_pct
//   - |ref_price| < 1e-15  (cannot validate price, treat as trip)
//   - order_price * order_size >= notional_cap
//
// Binary output: int32_t (1 = trip, 0 = ok).
//
// Source: chukonu/fintech/kernels.h :: CreateFatFingerGuardProcessor (2162).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct FatFingerGuardOpDesc {
    int32_t order_price_col;
    int32_t order_size_col;
    int32_t ref_price_col;
    int32_t out_col;
    double  band_pct;       // e.g. 0.05
    double  notional_cap;   // e.g. 10_000_000
};

// Inputs: in[0] = order_price, in[1] = order_size, in[2] = ref_price.
inline void fat_finger_guard_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                                    int32_t* BOLT_RESTRICT out,
                                    const FatFingerGuardOpDesc& desc) noexcept {
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT op = in[0];
    const double* BOLT_RESTRICT os = in[1];
    const double* BOLT_RESTRICT rp = in[2];
    assert(op != nullptr || n == 0);
    assert(os != nullptr || n == 0);
    assert(rp != nullptr || n == 0);
    const double band = desc.band_pct;
    const double cap  = desc.notional_cap;
    for (int64_t i = 0; i < n; ++i) {
        const double r = rp[i];
        const bool ref_ok = (std::fabs(r) > 1e-15);
        const double dev = ref_ok ? std::fabs(op[i] / r - 1.0) : 0.0;
        const bool price_trip = (!ref_ok) || (dev >= band);
        const double notional = op[i] * os[i];
        const bool notional_trip = (notional >= cap);
        out[i] = (price_trip || notional_trip) ? 1 : 0;
    }
}

}  // namespace fintech
}  // namespace bolt
