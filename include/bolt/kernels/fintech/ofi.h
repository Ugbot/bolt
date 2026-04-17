// bolt/kernels/fintech/ofi.h — Phase 4b Tier 1 kernel.
//
// Order-flow imbalance (chukonu variant): out[i] = buy_vol[i] - sell_vol[i].
//
// Note on the plan one-liner: the plan-doc called this "OFI from L1 book
// deltas with signed aggregation". Chukonu's shipped kernel is simpler —
// just per-row signed-volume imbalance of already-classified buy/sell
// aggregates. We match chukonu; the book-delta variant would be a
// separate kernel when that data is available at the ingest layer.
//
// Source: chukonu/fintech/kernels.h :: CreateOFIProcessor (line 3710).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct OFIOpDesc {
    int32_t buy_vol_col;
    int32_t sell_vol_col;
    int32_t out_col;
};

// Inputs: in[0] = buy_vol, in[1] = sell_vol (both double).
// Output: out[i] = buy_vol[i] - sell_vol[i].
inline void ofi_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                       double* BOLT_RESTRICT out,
                       const OFIOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT buy  = in[0];
    const double* BOLT_RESTRICT sell = in[1];
    assert(buy  != nullptr || n == 0);
    assert(sell != nullptr || n == 0);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = buy[i] - sell[i];
    }
}

}  // namespace fintech
}  // namespace bolt
