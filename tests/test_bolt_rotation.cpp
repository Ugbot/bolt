// test_bolt_rotation.cpp - Tests for the rotation kernels
// (bolt/kernels/bolt_rotation.h): Householder QR apply + DCT-II with
// random sign flips.

#include "bolt/kernels/bolt_rotation.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>

namespace {

float vec_norm(const float* v, size_t D) {
    float s = 0.0f;
    for (size_t i = 0; i < D; ++i) s += v[i] * v[i];
    return std::sqrt(s);
}

}  // namespace

// Householder reflectors are involutions: applying the same reflector
// twice recovers the input vector (within fp tolerance).
TEST(BoltRotation, QRRoundtripSingleReflector) {
    constexpr size_t D = 16;
    float v[D];
    // Construct an arbitrary unit-norm reflector.
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < D; ++i) v[i] = dist(rng);
    const float vn = vec_norm(v, D);
    ASSERT_GT(vn, 0.0f);
    for (size_t i = 0; i < D; ++i) v[i] /= vn;

    float x_in[D];
    for (size_t i = 0; i < D; ++i) x_in[i] = dist(rng);

    float x_mid[D];
    float x_back[D];
    bolt::apply_qr_rotation_f32(v, D, 1, x_in, x_mid);
    bolt::apply_qr_rotation_f32(v, D, 1, x_mid, x_back);

    for (size_t i = 0; i < D; ++i) {
        EXPECT_NEAR(x_back[i], x_in[i], 1e-5f) << "i=" << i;
    }
}

// Householder reflectors preserve L2 norm.
TEST(BoltRotation, QRNormPreservation) {
    constexpr size_t D = 32;
    constexpr size_t R = 4;
    std::mt19937 rng(0xDECAFu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float reflectors[R * D];
    for (size_t r = 0; r < R; ++r) {
        float* vr = reflectors + r * D;
        for (size_t i = 0; i < D; ++i) vr[i] = dist(rng);
        const float n = vec_norm(vr, D);
        ASSERT_GT(n, 0.0f);
        for (size_t i = 0; i < D; ++i) vr[i] /= n;
    }

    float x_in[D];
    for (size_t i = 0; i < D; ++i) x_in[i] = dist(rng);
    float x_out[D];
    bolt::apply_qr_rotation_f32(reflectors, D, R, x_in, x_out);

    const float in_n = vec_norm(x_in, D);
    const float out_n = vec_norm(x_out, D);
    EXPECT_NEAR(in_n, out_n, 1e-4f);
}

// DCT-II of e_0 = {1,0,...,0} with all-positive signs has a known
// closed-form: X[k] = scale * c_k * cos(pi * k / (2D)) where c_0 =
// 1/sqrt(2) and c_k = 1 for k > 0.
TEST(BoltRotation, DCTHandComputedReferenceD8) {
    constexpr size_t D = 8;
    int8_t signs[D];
    for (size_t i = 0; i < D; ++i) signs[i] = 1;
    float x_in[D] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float x_out[D];
    bolt::apply_dct_rotation_f32(signs, D, x_in, x_out);

    constexpr float kPi = 3.14159265358979323846f;
    const float scale = std::sqrt(2.0f / static_cast<float>(D));
    for (size_t k = 0; k < D; ++k) {
        const float kk = (k == 0) ? (1.0f / std::sqrt(2.0f)) : 1.0f;
        const float angle = kPi * static_cast<float>(k) / (2.0f * static_cast<float>(D));
        const float expected = scale * kk * std::cos(angle);
        EXPECT_NEAR(x_out[k], expected, 1e-5f) << "k=" << k;
    }
}

// Orthonormal DCT-II preserves L2 norm. Random +/-1 signs are also a
// norm-preserving (orthogonal) pre-multiplication, so the composition
// preserves norm.
TEST(BoltRotation, DCTNormPreservation) {
    constexpr size_t D = 64;
    std::mt19937 rng(0xBADF00Du);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> sign_dist(0, 1);

    int8_t signs[D];
    for (size_t i = 0; i < D; ++i) signs[i] = sign_dist(rng) ? 1 : -1;

    float x_in[D];
    for (size_t i = 0; i < D; ++i) x_in[i] = dist(rng);
    float x_out[D];
    bolt::apply_dct_rotation_f32(signs, D, x_in, x_out);

    const float in_n = vec_norm(x_in, D);
    const float out_n = vec_norm(x_out, D);
    ASSERT_GT(in_n, 0.0f);
    const float rel = std::fabs(in_n - out_n) / in_n;
    EXPECT_LT(rel, 1e-3f) << "in_n=" << in_n << " out_n=" << out_n;
}
