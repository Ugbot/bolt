// bolt/kernels/fintech/signed_volume.h — Phase 4 Tier 1 kernel.
//
// Signed volume: out[i] = sign(side[i]) * qty[i]
//
// Task-spec shape: `side` is int64 in {-1, 0, +1}; `qty` is double.
// (The chukonu variant multiplies two doubles — CreateSignedVolumeProcessor
// at line 3799 — but Bolt's tick feeds carry the side as an integer flag,
// so we take an int64 side column and the multiplication becomes a cast.)
//
// Source: chukonu/fintech/kernels.h :: CreateSignedVolumeProcessor (3799).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct SignedVolumeOpDesc {
    int32_t side_col;  // int64, {-1, 0, +1}
    int32_t qty_col;   // double
    int32_t out_col;
};

// Two-input signature with heterogeneous types — caller passes side as
// `const int64_t*` and qty as `const double*` through separate params to
// avoid union gymnastics in the `in[]` array-of-pointers form.
inline void signed_volume_kernel(const int64_t* BOLT_RESTRICT side,
                                 const double*  BOLT_RESTRICT qty,
                                 int64_t n,
                                 double* BOLT_RESTRICT out,
                                 const SignedVolumeOpDesc& desc) noexcept {
    (void)desc;
    assert(side != nullptr || n == 0);
    assert(qty  != nullptr || n == 0);
    assert(out  != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = static_cast<double>(side[i]) * qty[i];
    }
}

}  // namespace fintech
}  // namespace bolt
