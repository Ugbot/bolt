// bolt/kernels/fintech/cornish_fisher_var.h — Phase 5.S.9 Tier 2 kernel.
//
// CornishFisherVaR: parametric Value-at-Risk with skew + excess-kurtosis
// adjustment. Computes (mean, pop_var, pop_skew, excess_kurt) over the
// batch, then broadcasts a single scalar VaR to every output row.
//
//   Moments over the whole batch (POPULATION, /n):
//       mean = (1/n) Σ r_i
//       m2   = (1/n) Σ (r_i - mean)²
//       m3   = (1/n) Σ (r_i - mean)³
//       m4   = (1/n) Σ (r_i - mean)⁴
//       stddev = sqrt(m2)
//       skew   = m3 / (m2 · stddev)       if m2 > 1e-15 else 0
//       kurt   = (m4 / m2²) - 3.0         if m2 > 1e-15 else 0    (excess)
//
//   Normal quantile at tail probability p = 1 - confidence, via the
//   Abramowitz-Stegun 26.2.3 rational approximation (the same one
//   chukonu uses; typo-corrected in chukonu as "26.2.23"):
//       q = min(p, 1 - p)
//       t = sqrt(-2 ln q)
//       z = t - (c0 + c1 t + c2 t²) / (1 + d1 t + d2 t² + d3 t³)
//       if p < 0.5: z = -z
//
//   Cornish-Fisher expansion (chukonu line 3007-3012):
//       cf_z = z
//            + (z² - 1) · skew / 6
//            + (z³ - 3z) · kurt / 24
//            - (2z³ - 5z) · skew² / 36
//
//   VaR = -(mean + cf_z · stddev)   (positive loss, Gaussian-tail sign
//                                    convention. Lower-tail quantile
//                                    gets negated so VaR > 0 on losses.)
//
// Source: chukonu/fintech/kernels.h :: CreateCornishFisherVaRProcessor
// (line 2945). chukonu broadcasts the scalar across the batch — this is
// a batch-global statistic, not a per-row rolling one. We match that.
//
// Branchless note: the inner moment-accumulation loop is straight-line
// arithmetic (compiler auto-vectorises). The final broadcast loop is a
// single-value fill, also branchless. The only data-dependent code is
// (a) guarding m2 > eps when forming skew/kurt (cmov ternary) and (b)
// the one-shot `p < 0.5` / `p > 0 && p < 1` normal-quantile branches,
// which are hoisted OUT of the per-row loop and run once per execute.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct CornishFisherVaROpDesc {
    int32_t return_col;
    int32_t out_col;
    double  confidence;   // e.g. 0.95 for 95% VaR
};

// Stateless — every call recomputes from scratch (batch-aggregate shape,
// just like chukonu). The struct exists to unify the make_<name>_state
// interface across the phase.
struct CornishFisherVaRState {
    CornishFisherVaROpDesc desc;
};

// Normal quantile via Abramowitz-Stegun 26.2.3 rational approximation.
// Same coefficients chukonu uses. Not per-row — called once per batch.
inline double cornish_fisher_normal_quantile(double p) noexcept {
    assert(p >= 0.0);
    assert(p <= 1.0);
    if (!(p > 0.0 && p < 1.0)) return 0.0;
    const double q = (p < 0.5) ? p : (1.0 - p);
    const double t = std::sqrt(-2.0 * std::log(q));
    const double c0 = 2.515517;
    const double c1 = 0.802853;
    const double c2 = 0.010328;
    const double d1 = 1.432788;
    const double d2 = 0.189269;
    const double d3 = 0.001308;
    const double num = c0 + c1 * t + c2 * t * t;
    const double den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    double z = t - num / den;
    if (p < 0.5) z = -z;
    return z;
}

inline void execute_cornish_fisher_var(CornishFisherVaRState* state,
                                       const double* BOLT_RESTRICT r,
                                       int64_t n,
                                       double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(r   != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    assert(state->desc.confidence >= 0.0);
    assert(state->desc.confidence <= 1.0);
    if (n == 0) return;

    // Pass 1: mean. Pure reduction, auto-vectorises.
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) sum += r[i];
    const double mean = sum / static_cast<double>(n);

    // Pass 2: m2, m3, m4 about the mean. Pure reduction.
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d  = r[i] - mean;
        const double d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= static_cast<double>(n);
    m3 /= static_cast<double>(n);
    m4 /= static_cast<double>(n);

    const double stddev = std::sqrt(m2);
    // cmov guards on degenerate-variance input (chukonu lines 2981-2982).
    const double skew = (m2 > 1e-15) ? (m3 / (m2 * stddev)) : 0.0;
    const double kurt = (m2 > 1e-15) ? ((m4 / (m2 * m2)) - 3.0) : 0.0;

    const double p    = 1.0 - state->desc.confidence;
    const double z    = cornish_fisher_normal_quantile(p);
    const double z2   = z * z;
    const double z3   = z2 * z;
    const double cf_z = z
                      + (z2 - 1.0) * skew / 6.0
                      + (z3 - 3.0 * z) * kurt / 24.0
                      - (2.0 * z3 - 5.0 * z) * skew * skew / 36.0;
    const double var_val = -(mean + cf_z * stddev);

    // Broadcast scalar across the batch. Branchless fill.
    for (int64_t i = 0; i < n; ++i) out[i] = var_val;
}

inline CornishFisherVaRState* make_cornish_fisher_var_state(
    ::bolt::Arena* arena,
    int32_t return_col,
    int32_t out_col,
    double confidence) noexcept {
    assert(arena != nullptr);
    assert(confidence >= 0.0);
    assert(confidence <= 1.0);
    CornishFisherVaRState* s =
        arena->allocate_array<CornishFisherVaRState>(1);
    if (s == nullptr) return nullptr;
    s->desc.return_col = return_col;
    s->desc.out_col    = out_col;
    s->desc.confidence = confidence;
    return s;
}

}  // namespace fintech
}  // namespace bolt
