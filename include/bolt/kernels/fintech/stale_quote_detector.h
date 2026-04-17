// bolt/kernels/fintech/stale_quote_detector.h — Phase 4b Tier 1 kernel.
//
// Stale-quote flag: 1 when (timestamp[i] - last_update[i]) > max_age, else 0.
// Two int64 timestamp columns (ns or ms — units are the caller's decision;
// max_age is in the same units).
//
// Binary output: int32_t (1 = stale, 0 = fresh). We use int32 rather than
// bool so the column stays SIMD-alignable for downstream compaction.
//
// Source: chukonu/fintech/kernels.h :: CreateStaleQuoteDetectorProcessor (2079).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct StaleQuoteDetectorOpDesc {
    int32_t ts_col;
    int32_t last_update_col;
    int32_t out_col;
    int64_t max_age;   // same unit as the two timestamp columns
};

// Inputs: in[0] = ts, in[1] = last_update (both int64).
// Output: out[i] = (ts[i] - last_update[i] > max_age) ? 1 : 0.
inline void stale_quote_detector_kernel(
    const int64_t* BOLT_RESTRICT* in, int64_t n,
    int32_t* BOLT_RESTRICT out,
    const StaleQuoteDetectorOpDesc& desc) noexcept {
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const int64_t* BOLT_RESTRICT ts = in[0];
    const int64_t* BOLT_RESTRICT lu = in[1];
    assert(ts != nullptr || n == 0);
    assert(lu != nullptr || n == 0);
    const int64_t max_age = desc.max_age;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t age = ts[i] - lu[i];
        out[i] = (age > max_age) ? 1 : 0;
    }
}

}  // namespace fintech
}  // namespace bolt
