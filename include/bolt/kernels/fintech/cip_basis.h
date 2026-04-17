// bolt/kernels/fintech/cip_basis.h — Phase 4 Tier 1 kernel.
//
// Covered interest-parity basis (log-form, matching chukonu):
//   implied_rate_diff = ln(forward / spot) / T    (0 if s or T tiny)
//   actual_rate_diff  = r_dom - r_for
//   out[i] = implied_rate_diff - actual_rate_diff
//
// Source: chukonu/fintech/kernels.h :: CreateCIPBasisProcessor (1656).
// This is the log-rate-differential variant (continuous compounding).
// The plan-doc one-liner `(f - s·(1 + r_d·τ)/(1 + r_f·τ)) / s` is the
// simple-compounding equivalent; chukonu ships the log form, so we match.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct CIPBasisOpDesc {
    int32_t spot_col;
    int32_t forward_col;
    int32_t r_dom_col;
    int32_t r_for_col;
    int32_t out_col;
    double  t_years;
};

// Inputs: in[0] = spot, in[1] = forward, in[2] = r_dom, in[3] = r_for.
inline void cip_basis_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                             double* BOLT_RESTRICT out,
                             const CIPBasisOpDesc& desc) noexcept {
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT spot    = in[0];
    const double* BOLT_RESTRICT forward = in[1];
    const double* BOLT_RESTRICT r_dom   = in[2];
    const double* BOLT_RESTRICT r_for   = in[3];
    assert(spot    != nullptr || n == 0);
    assert(forward != nullptr || n == 0);
    assert(r_dom   != nullptr || n == 0);
    assert(r_for   != nullptr || n == 0);
    const double T = desc.t_years;
    const bool t_ok = (T > 1e-12);
    for (int64_t i = 0; i < n; ++i) {
        const double s = spot[i];
        const double f = forward[i];
        const double implied = (s > 1e-12 && t_ok) ? std::log(f / s) / T : 0.0;
        const double actual  = r_dom[i] - r_for[i];
        out[i] = implied - actual;
    }
}

}  // namespace fintech
}  // namespace bolt
