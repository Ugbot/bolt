// bolt/kernels/fintech/corwin_schultz.h — Phase 4b Tier 1 kernel.
//
// Corwin-Schultz (2012) high-low spread estimator over consecutive bars:
//   beta  = (ln(h0/l0)^2 + ln(h1/l1)^2) / 2
//   gamma = ln(max(h0,h1) / min(l0,l1))^2
//   k     = 3 - 2*sqrt(2)
//   alpha = (sqrt(2*beta) - sqrt(beta)) / k - sqrt(gamma / k)
//   S     = 2 * (exp(alpha) - 1) / (1 + exp(alpha))         (clipped to >=0)
//
// Row 0 has no predecessor, output is 0.
//
// Source: chukonu/fintech/kernels.h :: CreateCorwinSchultzProcessor (2653).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct CorwinSchultzOpDesc {
    int32_t high_col;
    int32_t low_col;
    int32_t out_col;
};

// Inputs: in[0] = high, in[1] = low.
inline void corwin_schultz_kernel(const double* BOLT_RESTRICT* in, int64_t n,
                                  double* BOLT_RESTRICT out,
                                  const CorwinSchultzOpDesc& desc) noexcept {
    (void)desc;
    assert(in  != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    if (n == 0) return;
    const double* BOLT_RESTRICT high = in[0];
    const double* BOLT_RESTRICT low  = in[1];
    assert(high != nullptr);
    assert(low  != nullptr);
    const double k_const = 3.0 - 2.0 * std::sqrt(2.0);
    out[0] = 0.0;
    for (int64_t i = 1; i < n; ++i) {
        const double h0 = high[i - 1], l0 = low[i - 1];
        const double h1 = high[i],     l1 = low[i];
        const double ln_hl0 = std::log(h0 / l0);
        const double ln_hl1 = std::log(h1 / l1);
        const double beta   = (ln_hl0 * ln_hl0 + ln_hl1 * ln_hl1) * 0.5;
        const double h_max  = (h0 > h1) ? h0 : h1;
        const double l_min  = (l0 < l1) ? l0 : l1;
        const double ln_rng = std::log(h_max / l_min);
        const double gamma  = ln_rng * ln_rng;
        const double sqrt_b  = std::sqrt((beta > 0.0) ? beta : 0.0);
        const double sqrt_2b = std::sqrt((2.0 * beta > 0.0) ? 2.0 * beta : 0.0);
        const double g_over_k = gamma / k_const;
        const double alpha = (sqrt_2b - sqrt_b) / k_const
                             - std::sqrt((g_over_k > 0.0) ? g_over_k : 0.0);
        const double e = std::exp(alpha);
        double s = 2.0 * (e - 1.0) / (1.0 + e);
        s = (s < 0.0) ? 0.0 : s;
        out[i] = s;
    }
}

}  // namespace fintech
}  // namespace bolt
