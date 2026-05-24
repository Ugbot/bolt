// test_bolt_vector_bond.cpp — PDX-BOND lower-bound kernels.
//
// Wave 9.4 ζ.5b: probabilistic Chebyshev lower bound on remaining-dim
// L2² contribution. Tightens the box-LB block-skip in the canonical
// PDX-64 walker (recall-preserving when k_σ chosen so that the
// Chebyshev bound is ≤ the exact box bound).
//
// Covers:
//   1. ChebyshevLBNonNegative          — LB ≥ 0 always.
//   2. ChebyshevLBBoundedByMinMax      — chebyshev_lb ≤ box_lb when k_σ
//                                         is set high relative to box width.
//   3. ChebyshevLBAtKZeroIsCenter      — at k_σ = 0, kernel = Σ(q−μ)².
//   4. ChebyshevLBVisitedMaskRespected — masked dims contribute 0.

#include <gtest/gtest.h>

#include "bolt/kernels/bolt_vector_bond.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>

namespace {

// Bounded-loop helper to build a visited_mask bitmap (LSB-first per byte).
inline void set_visited_bit(uint8_t* mask, uint32_t d) noexcept {
    mask[d >> 3] = static_cast<uint8_t>(mask[d >> 3] | (1u << (d & 7u)));
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Non-negative for all valid (random) inputs.
// ---------------------------------------------------------------------------

TEST(BoltVectorBondChebyshevLB, ChebyshevLBNonNegative) {
    constexpr uint32_t kDim = 128u;
    constexpr uint32_t kTrials = 64u;  // bounded loop

    std::mt19937 rng(0xB0B0u);
    std::uniform_real_distribution<float> q_dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> mu_dist(-5.0f, 5.0f);
    std::uniform_real_distribution<float> var_dist(0.0f, 4.0f);
    std::uniform_real_distribution<float> k_dist(0.0f, 4.0f);

    float q[kDim], mu[kDim], var[kDim];

    for (uint32_t t = 0; t < kTrials; ++t) {
        for (uint32_t d = 0; d < kDim; ++d) {
            q  [d] = q_dist (rng);
            mu [d] = mu_dist(rng);
            var[d] = var_dist(rng);
        }
        const float k = k_dist(rng);
        const float lb = bolt::vec::sigma_bound_remaining_chebyshev_lb_f32(
            q, mu, var, /*visited=*/nullptr, kDim, k);
        EXPECT_GE(lb, 0.0f)
            << "trial " << t << " k=" << k << " lb=" << lb;
        EXPECT_TRUE(std::isfinite(lb)) << "trial " << t;
    }
}

// ---------------------------------------------------------------------------
// 2. chebyshev_lb ≤ box_lb when k_σ is high enough that the σ-tail
//    covers the box width on every dim.
//
//    We construct min/max around μ such that (max − μ) = (μ − min) = w.
//    With var = σ², setting k_σ ≥ w / σ guarantees that the Chebyshev
//    "best diff" |q − μ| − k_σ σ ≤ |q − μ| − w ≤ box "best diff" for
//    any q outside [min, max], and 0 ≤ 0 inside. So chebyshev_lb is
//    componentwise ≤ box_lb, and therefore the sum is ≤.
// ---------------------------------------------------------------------------

TEST(BoltVectorBondChebyshevLB, ChebyshevLBBoundedByMinMax) {
    constexpr uint32_t kDim = 64u;
    constexpr uint32_t kTrials = 32u;

    std::mt19937 rng(0xCAFEu);
    std::uniform_real_distribution<float> q_dist (-5.0f, 5.0f);
    std::uniform_real_distribution<float> mu_dist(-2.0f, 2.0f);
    // σ ∈ [0.25, 1.5]; box half-width w = σ × ρ with ρ ∈ [0.25, 1.0],
    // so k_σ = 1.5 × max(ρ) = 1.5 covers every dim with margin.
    std::uniform_real_distribution<float> sigma_dist(0.25f, 1.5f);
    std::uniform_real_distribution<float> rho_dist (0.25f, 1.0f);

    float q[kDim], mu[kDim], var[kDim], mn[kDim], mx[kDim];

    for (uint32_t t = 0; t < kTrials; ++t) {
        float max_ratio = 0.0f;
        for (uint32_t d = 0; d < kDim; ++d) {
            q   [d] = q_dist (rng);
            mu  [d] = mu_dist(rng);
            const float sigma = sigma_dist(rng);
            const float rho   = rho_dist  (rng);
            const float w     = sigma * rho;  // box half-width
            var [d] = sigma * sigma;
            mn  [d] = mu[d] - w;
            mx  [d] = mu[d] + w;
            if (rho > max_ratio) max_ratio = rho;
        }
        // k_σ ≥ max ρ guarantees k_σ × σ ≥ w on every dim.
        const float k = max_ratio + 0.5f;
        const float lb_box = bolt::vec::box_lower_bound_l2_f32(
            q, mn, mx, /*visited=*/nullptr, kDim);
        const float lb_che = bolt::vec::sigma_bound_remaining_chebyshev_lb_f32(
            q, mu, var, /*visited=*/nullptr, kDim, k);
        EXPECT_LE(lb_che, lb_box + 1e-4f)
            << "trial " << t << " k=" << k
            << " lb_che=" << lb_che << " lb_box=" << lb_box;
    }
}

// ---------------------------------------------------------------------------
// 3. At k_σ = 0, the kernel collapses to Σ (q − μ)² — the deterministic
//    centroid-distance lower bound.
// ---------------------------------------------------------------------------

TEST(BoltVectorBondChebyshevLB, ChebyshevLBAtKZeroIsCenter) {
    constexpr uint32_t kDim = 96u;
    constexpr uint32_t kTrials = 16u;

    std::mt19937 rng(0xDEADu);
    std::uniform_real_distribution<float> q_dist (-3.0f, 3.0f);
    std::uniform_real_distribution<float> mu_dist(-3.0f, 3.0f);
    std::uniform_real_distribution<float> var_dist(0.0f, 9.0f);

    float q[kDim], mu[kDim], var[kDim];

    for (uint32_t t = 0; t < kTrials; ++t) {
        float expected = 0.0f;
        for (uint32_t d = 0; d < kDim; ++d) {
            q  [d] = q_dist (rng);
            mu [d] = mu_dist(rng);
            var[d] = var_dist(rng);  // ignored at k=0
            const float df = q[d] - mu[d];
            expected += df * df;
        }
        const float lb = bolt::vec::sigma_bound_remaining_chebyshev_lb_f32(
            q, mu, var, /*visited=*/nullptr, kDim, /*k_sigma=*/0.0f);
        EXPECT_NEAR(lb, expected, 1e-3f * expected + 1e-4f)
            << "trial " << t;
    }
}

// ---------------------------------------------------------------------------
// 4. Visited-mask: dims flagged in the bitmap must contribute 0.
//    Compare full-pass kernel against a pass with half the dims masked,
//    and verify the masked total equals the kernel run over the unmasked
//    subset only.
// ---------------------------------------------------------------------------

TEST(BoltVectorBondChebyshevLB, ChebyshevLBVisitedMaskRespected) {
    constexpr uint32_t kDim = 64u;
    constexpr uint32_t kBytes = (kDim + 7u) / 8u;

    std::mt19937 rng(0xFEEDu);
    std::uniform_real_distribution<float> q_dist (-4.0f, 4.0f);
    std::uniform_real_distribution<float> mu_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> var_dist(0.0f, 1.0f);

    float q[kDim], mu[kDim], var[kDim];
    uint8_t mask[kBytes] = {0};

    for (uint32_t d = 0; d < kDim; ++d) {
        q  [d] = q_dist (rng);
        mu [d] = mu_dist(rng);
        var[d] = var_dist(rng);
        // Mark every odd dim as "visited" (contributes 0).
        if ((d & 1u) == 1u) set_visited_bit(mask, d);
    }
    const float k_sigma = 1.5f;

    const float lb_masked = bolt::vec::sigma_bound_remaining_chebyshev_lb_f32(
        q, mu, var, mask, kDim, k_sigma);

    // Reference: compute scalar LB over even dims only.
    float ref = 0.0f;
    for (uint32_t d = 0; d < kDim; d += 2u) {
        const float df  = q[d] - mu[d];
        const float ad  = (df >= 0.0f) ? df : -df;
        const float sig = std::sqrt(var[d]);
        const float gap = ad - k_sigma * sig;
        if (gap > 0.0f) ref += gap * gap;
    }
    EXPECT_NEAR(lb_masked, ref, 1e-3f * ref + 1e-4f)
        << "masked-vs-reference mismatch";

    // Fully-masked input must produce exactly 0.
    uint8_t full_mask[kBytes];
    std::memset(full_mask, 0xFF, sizeof(full_mask));
    const float lb_full = bolt::vec::sigma_bound_remaining_chebyshev_lb_f32(
        q, mu, var, full_mask, kDim, k_sigma);
    EXPECT_EQ(lb_full, 0.0f);
}
