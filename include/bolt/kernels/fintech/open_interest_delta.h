// bolt/kernels/fintech/open_interest_delta.h — Phase 4b Tier 1 kernel.
//
// Open interest delta: out[0] = 0, out[i] = oi[i] - oi[i-1].
// Thin wrapper over `diff_column<double>`; carries a POD desc so the
// operator surface stays uniform with every other Phase 4 kernel.
//
// Source: chukonu/fintech/kernels.h :: CreateOpenInterestDeltaProcessor
// (line 3901). Matches verbatim.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/column_primitives.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct OpenInterestDeltaOpDesc {
    int32_t oi_col;
    int32_t out_col;
};

// Inputs: in[0] = open_interest (double).
// Output: out[0] = 0.0, out[i>=1] = oi[i] - oi[i-1].
inline void open_interest_delta_kernel(const double* BOLT_RESTRICT* in,
                                       int64_t n,
                                       double* BOLT_RESTRICT out,
                                       const OpenInterestDeltaOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT oi = in[0];
    assert(oi != nullptr || n == 0);
    diff_column<double>(oi, n, out);
}

}  // namespace fintech
}  // namespace bolt
