// test_bolt_vector_distance.cpp — Unit tests for the lifted vector-distance
// kernels in `bolt/kernels/bolt_vector_distance.h` (L2 / IP / cosine, both
// pair and column-major vertical accumulator forms).

#include "bolt/kernels/bolt_vector_distance.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

// Scalar reference (fresh — does not call back into the kernel under test).
float ref_l2_pair(const float* a, const float* b, size_t D) {
    float acc = 0.0f;
    for (size_t i = 0; i < D; ++i) {
        const float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}
float ref_ip_pair(const float* a, const float* b, size_t D) {
    float acc = 0.0f;
    for (size_t i = 0; i < D; ++i) acc += a[i] * b[i];
    return acc;
}

}  // namespace

TEST(BoltVectorDistance, L2PairBasic) {
    const float a[3] = {1.0f, 2.0f, 3.0f};
    const float b[3] = {4.0f, 5.0f, 6.0f};
    EXPECT_FLOAT_EQ(bolt::l2_pair_f32(a, b, 3), 27.0f);
}

TEST(BoltVectorDistance, IPPairBasic) {
    const float a[3] = {1.0f, 2.0f, 3.0f};
    const float b[3] = {4.0f, 5.0f, 6.0f};
    EXPECT_FLOAT_EQ(bolt::ip_pair_f32(a, b, 3), 32.0f);
}

TEST(BoltVectorDistance, CosinePairMatchesIPOverNorms) {
    const float a[3] = {1.0f, 2.0f, 3.0f};
    const float b[3] = {4.0f, 5.0f, 6.0f};
    const float dot = ref_ip_pair(a, b, 3);
    const float na  = std::sqrt(ref_ip_pair(a, a, 3));
    const float nb  = std::sqrt(ref_ip_pair(b, b, 3));
    const float expected = dot / (na * nb);
    EXPECT_NEAR(bolt::cosine_pair_f32(a, b, 3), expected, 1e-6f);
}

TEST(BoltVectorDistance, CosinePairZeroNorm) {
    const float a[3] = {0.0f, 0.0f, 0.0f};
    const float b[3] = {1.0f, 2.0f, 3.0f};
    EXPECT_FLOAT_EQ(bolt::cosine_pair_f32(a, b, 3), 0.0f);
}

TEST(BoltVectorDistance, L2PairWideMatchesScalar) {
    // Length 19 — exercises the AVX2 8-lane body + scalar tail (3).
    constexpr size_t D = 19;
    float a[D], b[D];
    for (size_t i = 0; i < D; ++i) {
        a[i] = static_cast<float>(i) * 0.5f - 1.25f;
        b[i] = static_cast<float>(i) * -0.7f + 0.3f;
    }
    EXPECT_NEAR(bolt::l2_pair_f32(a, b, D), ref_l2_pair(a, b, D), 1e-3f);
}

// Vertical L2 accumulator across 3 dims. dim_col is column-major: each call
// passes one row of the column-major slab. dists must be pre-zeroed.
float ref_cosine_pair(const float* a, const float* b, size_t D) {
    const float dot = ref_ip_pair(a, b, D);
    const float na  = std::sqrt(ref_ip_pair(a, a, D));
    const float nb  = std::sqrt(ref_ip_pair(b, b, D));
    const float denom = na * nb;
    return (denom > 0.0f) ? (dot / denom) : 0.0f;
}

void fill_random_f32(std::vector<float>& v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& x : v) x = dist(rng);
}

// Parity gate for the 4-way multi-accumulator AVX2 pair kernels. Length
// 1021 is prime → it exercises the 32-wide unroll, the 8-wide remainder
// body, AND the scalar tail simultaneously. FP reassociation is allowed
// here (Bolt-local), so the gate is close-match (relative tol) against a
// left-to-right scalar recompute, not bit-identical.
TEST(BoltVectorDistance, PairMultiAccumMatchesScalarPrimeLen) {
    constexpr size_t D = 1021;
    std::vector<float> a(D), b(D);
    fill_random_f32(a, 0x1111);
    fill_random_f32(b, 0x2222);

    const float l2_ref = ref_l2_pair(a.data(), b.data(), D);
    const float ip_ref = ref_ip_pair(a.data(), b.data(), D);
    const float cs_ref = ref_cosine_pair(a.data(), b.data(), D);

    EXPECT_NEAR(bolt::l2_pair_f32(a.data(), b.data(), D), l2_ref,
                std::abs(l2_ref) * 1e-4f + 1e-3f);
    EXPECT_NEAR(bolt::ip_pair_f32(a.data(), b.data(), D), ip_ref,
                std::abs(ip_ref) * 1e-4f + 1e-3f);
    EXPECT_NEAR(bolt::cosine_pair_f32(a.data(), b.data(), D), cs_ref, 1e-4f);
}

TEST(BoltVectorDistance, VerticalL2AccumulateAcross3Dims) {
    constexpr size_t N = 4;
    constexpr size_t D = 3;
    // Vector i, dim d -> v[i][d] = i + 0.5 * d.
    // Query q[d] = d + 1.
    // Expected dists[i] = sum_d (q[d] - v[i][d])^2.
    float col0[N] = { 0.0f, 1.0f, 2.0f, 3.0f };  // d=0: v[i][0] = i
    float col1[N] = { 0.5f, 1.5f, 2.5f, 3.5f };  // d=1
    float col2[N] = { 1.0f, 2.0f, 3.0f, 4.0f };  // d=2
    const float q[D] = { 1.0f, 2.0f, 3.0f };

    float dists[N] = {0.0f, 0.0f, 0.0f, 0.0f};
    bolt::l2_vertical_f32_accumulate(col0, q[0], dists, N);
    bolt::l2_vertical_f32_accumulate(col1, q[1], dists, N);
    bolt::l2_vertical_f32_accumulate(col2, q[2], dists, N);

    for (size_t i = 0; i < N; ++i) {
        float expected = 0.0f;
        const float vi[D] = { col0[i], col1[i], col2[i] };
        for (size_t d = 0; d < D; ++d) {
            const float diff = q[d] - vi[d];
            expected += diff * diff;
        }
        EXPECT_NEAR(dists[i], expected, 1e-5f) << "i=" << i;
    }
}

TEST(BoltVectorDistance, VerticalL2WideMatchesScalar) {
    // 20 vectors — exercises AVX2 body (16) + scalar tail (4).
    constexpr size_t N = 20;
    float col[N];
    for (size_t i = 0; i < N; ++i) col[i] = static_cast<float>(i) - 7.0f;
    const float q = 3.5f;

    float dists[N] = {};
    bolt::l2_vertical_f32_accumulate(col, q, dists, N);
    for (size_t i = 0; i < N; ++i) {
        const float d = q - col[i];
        EXPECT_NEAR(dists[i], d * d, 1e-5f);
    }
}

TEST(BoltVectorDistance, VerticalIPAccumulateAcross3Dims) {
    constexpr size_t N = 4;
    constexpr size_t D = 3;
    float col0[N] = { 0.0f, 1.0f, 2.0f, 3.0f };
    float col1[N] = { 0.5f, 1.5f, 2.5f, 3.5f };
    float col2[N] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float q[D] = { 1.0f, 2.0f, 3.0f };

    float dists[N] = {0.0f, 0.0f, 0.0f, 0.0f};
    bolt::ip_vertical_f32_accumulate(col0, q[0], dists, N);
    bolt::ip_vertical_f32_accumulate(col1, q[1], dists, N);
    bolt::ip_vertical_f32_accumulate(col2, q[2], dists, N);

    for (size_t i = 0; i < N; ++i) {
        const float vi[D] = { col0[i], col1[i], col2[i] };
        float expected = 0.0f;
        for (size_t d = 0; d < D; ++d) expected += q[d] * vi[d];
        EXPECT_NEAR(dists[i], expected, 1e-5f) << "i=" << i;
    }
}

TEST(BoltVectorDistance, VerticalCosineAccumulateAcross3Dims) {
    constexpr size_t N = 4;
    constexpr size_t D = 3;
    float col0[N] = { 0.0f, 1.0f, 2.0f, 3.0f };
    float col1[N] = { 0.5f, 1.5f, 2.5f, 3.5f };
    float col2[N] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float q[D] = { 1.0f, 2.0f, 3.0f };

    float sxqx[N] = {};
    float sxx[N]  = {};
    bolt::cosine_vertical_f32_accumulate_sxqx(col0, q[0], sxqx, sxx, N);
    bolt::cosine_vertical_f32_accumulate_sxqx(col1, q[1], sxqx, sxx, N);
    bolt::cosine_vertical_f32_accumulate_sxqx(col2, q[2], sxqx, sxx, N);

    for (size_t i = 0; i < N; ++i) {
        const float vi[D] = { col0[i], col1[i], col2[i] };
        float expected_sxqx = 0.0f, expected_sxx = 0.0f;
        for (size_t d = 0; d < D; ++d) {
            expected_sxqx += q[d] * vi[d];
            expected_sxx  += vi[d] * vi[d];
        }
        EXPECT_NEAR(sxqx[i], expected_sxqx, 1e-5f) << "i=" << i;
        EXPECT_NEAR(sxx[i],  expected_sxx,  1e-5f) << "i=" << i;
    }
}
