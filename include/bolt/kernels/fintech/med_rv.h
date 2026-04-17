// bolt/kernels/fintech/med_rv.h — Phase 5.S.8 Tier 2 kernel.
//
// MedRV: median realised volatility estimator (Andersen, Dobrev, Schaumburg
// 2012). Jump-robust alternative to RealizedVar: uses the median of three
// adjacent squared returns to reject outliers.
//
//   scale           = π / (6 - 4·sqrt(3) + π)     (≈ 1.0416)
//   bias_correction = n / (n - 2)
//   sum_med_sq      = Σ_{i=1..n-2} median(|r_{i-1}|, |r_i|, |r_{i+1}|)²
//   medrv           = scale * bias_correction * sum_med_sq
//
// Source: chukonu/fintech/kernels.h :: CreateMedRVProcessor (line 2773).
// Contract copied verbatim:
//   * Batch-scalar broadcast — one medrv value computed over the whole
//     batch and written to every output row (not rolling).
//   * n < 3 → medrv = 0.0, broadcast to every row.
//   * Median of three via a 3-compare sorting network (branchless-ready).
//   * `scale` uses the chukonu form π/(6 - 4√3 + π), which is the
//     algebraic inverse of 6 - 4√3/π and produces the published
//     MedRV constant to full double precision.
//
// Deviation from the plan one-liner: the plan described a "rolling median
// of squared returns scaled by the MedRV constant". Chukonu does NOT do
// that — it does a batch-wide scalar reduction. We match chukonu so the
// numerics line up with the gestalt/chukonu reference output.
//
// No state carried across batches (chukonu doesn't either). The state
// struct exists only to hold column-id metadata; the execute_ function
// needs no ring / no accumulator.

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

struct MedRVState {
    int32_t in_col;
    int32_t out_col;

    inline void init(int32_t in, int32_t out) noexcept {
        in_col  = in;
        out_col = out;
    }
};

// Median of three via sorting network (branchless when the target has cmov).
// Each comparison uses the (a>b ? b : a), (a>b ? a : b) min/max idiom, which
// MSVC lowers to vminsd/vmaxsd (no branch). The bubble pattern leaves the
// median in `b`.
static inline double medrv_median3(double a, double b, double c) noexcept {
    // Sort (a,b) ascending.
    const double a0 = (a > b) ? b : a;
    const double b0 = (a > b) ? a : b;
    // Sort (b0,c) ascending.
    const double b1 = (b0 > c) ? c : b0;
    const double c1 = (b0 > c) ? b0 : c;
    (void)c1;  // not needed after final swap
    // Sort (a0,b1) ascending — b1 is now the median of the original triple.
    const double b2 = (a0 > b1) ? a0 : b1;
    return b2;
}

// Execute MedRV over the entire `n`-row batch and broadcast the scalar
// medrv to every output slot.
//
// Hot loops:
//   - aggregation loop (n-2 iterations): fully branchless — each step is
//     three fabs + a 3-element sorting network + multiply-accumulate.
//   - broadcast loop (n iterations): pure store, no branch.
inline void execute_med_rv(MedRVState* state,
                           const double* BOLT_RESTRICT r,
                           int64_t n,
                           double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    (void)state;  // metadata holder only; no state carried across batches
    assert(r     != nullptr || n == 0);
    assert(out   != nullptr || n == 0);
    assert(n >= 0);

    double medrv = 0.0;
    if (n >= 3) {
        const double sqrt3 = std::sqrt(3.0);
        const double scale = M_PI / (6.0 - 4.0 * sqrt3 + M_PI);
        const double bias  = static_cast<double>(n) / static_cast<double>(n - 2);

        double sum_med_sq = 0.0;
        // i walks [1 .. n-2]; branchless body.
        for (int64_t i = 1; i < n - 1; ++i) {
            const double a = std::fabs(r[i - 1]);
            const double b = std::fabs(r[i]);
            const double c = std::fabs(r[i + 1]);
            const double m = medrv_median3(a, b, c);
            sum_med_sq += m * m;
        }
        medrv = scale * bias * sum_med_sq;
    }

    // Scalar broadcast — match chukonu: every row gets the same value.
    for (int64_t i = 0; i < n; ++i) out[i] = medrv;
}

}  // namespace fintech
}  // namespace bolt
