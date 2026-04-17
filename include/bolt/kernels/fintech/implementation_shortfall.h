// bolt/kernels/fintech/implementation_shortfall.h — Phase 4b Tier 1 kernel.
//
// Implementation shortfall in basis points:
//   out[i] = 10000 * (exec[i] - benchmark[i]) / benchmark[i]
//
// Note on the plan one-liner: the plan wrote `Σ(exec-arrival)·qty`
// (total-dollar cost form). Chukonu ships the per-row bps form — which
// is what execution desks actually consume — so we match chukonu. The
// dollar form is a trivial multiply-then-sum that composes on this
// kernel's output if ever needed.
//
// Sentinel: |benchmark| < 1e-15 → out = 0 (matches chukonu).
//
// Source: chukonu/fintech/kernels.h :: CreateImplementationShortfallProcessor
// (line 3939).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct ImplementationShortfallOpDesc {
    int32_t exec_col;
    int32_t benchmark_col;
    int32_t out_col;
};

// Inputs: in[0] = exec_price, in[1] = benchmark_price.
inline void implementation_shortfall_kernel(
    const double* BOLT_RESTRICT* in, int64_t n,
    double* BOLT_RESTRICT out,
    const ImplementationShortfallOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT exec  = in[0];
    const double* BOLT_RESTRICT bench = in[1];
    assert(exec  != nullptr || n == 0);
    assert(bench != nullptr || n == 0);
    for (int64_t i = 0; i < n; ++i) {
        const double b = bench[i];
        const double safe = (std::fabs(b) > 1e-15);
        out[i] = safe ? 10000.0 * (exec[i] - b) / b : 0.0;
    }
}

}  // namespace fintech
}  // namespace bolt
