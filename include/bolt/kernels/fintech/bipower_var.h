// bolt/kernels/fintech/bipower_var.h — Phase 4b Tier 1 kernel.
//
// Bipower variation (Barndorff-Nielsen & Shephard), jump-robust:
//   bpv = (π/2) · Σ_{i=1..n-1} |r[i]| · |r[i-1]|  /  (n - 1)
//   out[i] = bpv  (scalar broadcast over the batch)
//
// Degenerate case n < 2: output is all zero (no pair to accumulate).
//
// Source: chukonu/fintech/kernels.h :: CreateBipowerVarProcessor (401).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace bolt {
namespace fintech {

struct BipowerVarOpDesc {
    int32_t return_col;
    int32_t out_col;
};

// Inputs: in[0] = returns (double).
inline void bipower_var_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                               double* BOLT_RESTRICT out,
                               const BipowerVarOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    if (n == 0) return;
    const double* BOLT_RESTRICT r = in[0];
    assert(r != nullptr);
    double bpv = 0.0;
    if (n >= 2) {
        double sum = 0.0;
        for (int64_t i = 1; i < n; ++i) {
            sum += std::fabs(r[i]) * std::fabs(r[i - 1]);
        }
        bpv = (M_PI * 0.5) * sum / static_cast<double>(n - 1);
    }
    for (int64_t i = 0; i < n; ++i) {
        out[i] = bpv;
    }
}

}  // namespace fintech
}  // namespace bolt
