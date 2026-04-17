// bolt/kernels/fintech/forward_points.h — Phase 4 Tier 1 kernel.
//
// FX Forward Points (interest-rate-parity implied):
//   forward[i] = spot[i] * exp((r_dom[i] - r_for[i]) * T_years)
//   out[i]     = forward[i] - spot[i]
//
// Source: chukonu/fintech/kernels.h :: CreateForwardPointsProcessor (1577).
// The row-wise plan-doc formula `forward - spot` is the *output* once
// `forward` is synthesised from the rate differential — matching chukonu.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct ForwardPointsOpDesc {
    int32_t spot_col;
    int32_t r_dom_col;
    int32_t r_for_col;
    int32_t out_col;
    double  t_years;
};

// Inputs: in[0] = spot, in[1] = r_dom, in[2] = r_for.
inline void forward_points_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                                  double* BOLT_RESTRICT out,
                                  const ForwardPointsOpDesc& desc) noexcept {
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT spot  = in[0];
    const double* BOLT_RESTRICT r_dom = in[1];
    const double* BOLT_RESTRICT r_for = in[2];
    assert(spot  != nullptr || n == 0);
    assert(r_dom != nullptr || n == 0);
    assert(r_for != nullptr || n == 0);
    const double T = desc.t_years;
    for (int64_t i = 0; i < n; ++i) {
        const double s = spot[i];
        const double fwd = s * std::exp((r_dom[i] - r_for[i]) * T);
        out[i] = fwd - s;
    }
}

}  // namespace fintech
}  // namespace bolt
