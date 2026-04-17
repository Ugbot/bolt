// bolt/kernels/fintech/realized_var.h — Phase 4b Tier 1 kernel.
//
// Realized variance over a batch: scalar sum-of-squares of the return
// column, broadcast to every output row (matches chukonu's shape so
// downstream joins/aggregates see a uniform column).
//
//   rv = Σ r[i]^2  (over the whole batch)
//   out[i] = rv for all i
//
// Source: chukonu/fintech/kernels.h :: CreateRealizedVarProcessor (367).
// Composes on fintech::sum_of_squares from column_primitives.h.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/column_primitives.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct RealizedVarOpDesc {
    int32_t return_col;
    int32_t out_col;
};

// Inputs: in[0] = returns (double).
inline void realized_var_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                                double* BOLT_RESTRICT out,
                                const RealizedVarOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    if (n == 0) return;
    const double* BOLT_RESTRICT r = in[0];
    assert(r != nullptr);
    const double rv = sum_of_squares<double>(r, n);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = rv;
    }
}

}  // namespace fintech
}  // namespace bolt
