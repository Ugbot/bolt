// bolt/kernels/fintech/circuit_breaker.h — Phase 4b Tier 1 kernel.
//
// Circuit breaker: emit 1 when |price[i] / ref[i] - 1| > threshold, else 0.
// Per-row, batch-local — no running state across batches (chukonu's
// CreateCircuitBreakerProcessor at 2219 is pure row-local arithmetic, so
// this kernel stays in Phase 4b rather than being pushed to Phase 5).
//
// Trip condition: |price / ref - 1| > threshold_pct
//                 (|ref| < 1e-15 → treated as "no trip" per chukonu;
//                 sentinels at the validator layer should catch zero refs)
//
// Binary output: int32_t (1 = tripped, 0 = within-band).
//
// Source: chukonu/fintech/kernels.h :: CreateCircuitBreakerProcessor (2219).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct CircuitBreakerOpDesc {
    int32_t price_col;
    int32_t ref_price_col;
    int32_t out_col;
    double  threshold_pct;   // e.g. 0.10 == 10% trip
};

// Inputs: in[0] = price, in[1] = ref_price.
inline void circuit_breaker_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                                   int32_t* BOLT_RESTRICT out,
                                   const CircuitBreakerOpDesc& desc) noexcept {
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    const double* BOLT_RESTRICT p  = in[0];
    const double* BOLT_RESTRICT rp = in[1];
    assert(p  != nullptr || n == 0);
    assert(rp != nullptr || n == 0);
    const double thr = desc.threshold_pct;
    for (int64_t i = 0; i < n; ++i) {
        const double r = rp[i];
        int32_t flag = 0;
        if (std::fabs(r) > 1e-15) {
            const double dev = std::fabs(p[i] / r - 1.0);
            flag = (dev > thr) ? 1 : 0;
        }
        out[i] = flag;
    }
}

}  // namespace fintech
}  // namespace bolt
