// bolt/kernels/fintech/price_band_guard.h — Phase 4b Tier 1 kernel.
//
// Price-band guard: emit 1 when the order price is OUTSIDE the accepted
// band around the reference, 0 when inside. The task-spec sense is
// "1 = trip" (risk flag), which is the inverse of chukonu's "ok flag";
// we match the task spec so the four Phase 4b guard kernels all agree
// on polarity.
//
// Trip condition: |order_px / ref_px - 1| >= band_pct
//                 (== chukonu's NOT-ok branch, plus the |ref|<1e-15 guard
//                 treated as "trip" since we cannot validate a zero ref)
//
// Binary output: int32_t (1 = trip, 0 = within-band).
//
// Source: chukonu/fintech/kernels.h :: CreatePriceBandGuardProcessor (2118).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct PriceBandGuardOpDesc {
    int32_t order_price_col;
    int32_t ref_price_col;
    int32_t out_col;
    double  band_pct;   // e.g. 0.05 == 5% band
};

// Inputs: in[0] = order_price, in[1] = ref_price.
inline void price_band_guard_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                                    int32_t* BOLT_RESTRICT out,
                                    const PriceBandGuardOpDesc& desc) noexcept {
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT op = in[0];
    const double* BOLT_RESTRICT rp = in[1];
    assert(op != nullptr || n == 0);
    assert(rp != nullptr || n == 0);
    const double band = desc.band_pct;
    for (int64_t i = 0; i < n; ++i) {
        const double r = rp[i];
        const bool ref_ok = (std::fabs(r) > 1e-15);
        const double dev = ref_ok ? std::fabs(op[i] / r - 1.0) : 0.0;
        const bool trip = (!ref_ok) || (dev >= band);
        out[i] = trip ? 1 : 0;
    }
}

}  // namespace fintech
}  // namespace bolt
