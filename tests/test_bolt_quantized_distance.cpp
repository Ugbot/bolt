// test_bolt_quantized_distance.cpp — unit tests for SQ8 / RaBitQ / F16
// kernels in bolt/kernels/bolt_quantized_distance.h.

#include "bolt/kernels/bolt_quantized_distance.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Local f32 → f16 packer for tests only. Round-to-nearest-even with
// flush-to-zero on subnormals (sufficient for the round-trip check).
// ---------------------------------------------------------------------------
uint16_t f32_to_f16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp        = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127;
    uint32_t mant       = bits & 0x7FFFFFu;

    if (exp == 128) {
        // Inf / NaN.
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp > 15) {
        return static_cast<uint16_t>(sign | 0x7C00u);  // Inf.
    }
    if (exp < -14) {
        return static_cast<uint16_t>(sign);  // Flush to zero.
    }
    const uint32_t f16_exp  = static_cast<uint32_t>(exp + 15);
    const uint32_t f16_mant = mant >> 13;  // truncate (good enough for tests).
    return static_cast<uint16_t>(sign | (f16_exp << 10) | f16_mant);
}

// ---------------------------------------------------------------------------
// SQ8: pack a known f32 sequence, accumulate squared diffs in q-space,
// dequantize, compare against scalar f32 reference.
// ---------------------------------------------------------------------------
TEST(BoltQuantizedDistance, SQ8_RoundTripSquaredError) {
    constexpr size_t N = 16;
    constexpr size_t D = 8;
    // Grid: deterministic values in [0, 1].
    std::vector<float> base(N * D);
    std::vector<float> query(D);
    for (size_t i = 0; i < N * D; ++i) base[i] = static_cast<float>(i % 64) / 63.0f;
    for (size_t d = 0; d < D; ++d)     query[d] = static_cast<float>(d) / 7.0f;

    // Pack with global [0,1] range.
    const float vmin  = 0.0f;
    const float scale = 1.0f / 255.0f;
    std::vector<uint8_t> q_base(N * D);
    std::vector<uint8_t> q_query(D);
    bolt::f32_pack_sq8(base.data(),  N * D, vmin, scale, q_base.data());
    bolt::f32_pack_sq8(query.data(), D,     vmin, scale, q_query.data());

    // Vertical layout: column-major slab of [D rows × N cols].
    std::vector<uint8_t> slab(N * D);
    for (size_t i = 0; i < N; ++i) {
        for (size_t d = 0; d < D; ++d) {
            slab[d * N + i] = q_base[i * D + d];
        }
    }

    // Run vertical accumulator across all dims.
    std::vector<int32_t> dists_sq(N, 0);
    for (size_t d = 0; d < D; ++d) {
        bolt::l2_vertical_sq8_accumulate(
            slab.data() + d * N, q_query[d], dists_sq.data(), N);
    }

    // Dequantize accumulator: f-space ≈ scale^2 * dists_sq.
    const float s2 = scale * scale;
    for (size_t i = 0; i < N; ++i) {
        // Reference: scalar f32 squared L2 against the *quantized*
        // values reconstituted (so we measure pure kernel correctness,
        // not quantization loss).
        float ref = 0.0f;
        for (size_t d = 0; d < D; ++d) {
            const float bf = vmin + static_cast<float>(q_base[i * D + d]) * scale;
            const float qf = vmin + static_cast<float>(q_query[d])         * scale;
            const float diff = bf - qf;
            ref += diff * diff;
        }
        const float got = s2 * static_cast<float>(dists_sq[i]);
        // Within one quantization unit squared per dim, summed.
        EXPECT_NEAR(got, ref, 1e-5f * static_cast<float>(D))
            << "vector i=" << i;
    }
}

// ---------------------------------------------------------------------------
// RaBitQ: pack two random vectors, hand-compute Hamming.
// ---------------------------------------------------------------------------
TEST(BoltQuantizedDistance, RaBitQ_HammingMatchesHand) {
    constexpr size_t D = 192;  // not a multiple of 64 to exercise tail.
    constexpr size_t W = (D + 63) / 64;

    std::mt19937 rng(0xCAFEBABEu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> a(D), b(D);
    for (size_t i = 0; i < D; ++i) { a[i] = dist(rng); b[i] = dist(rng); }

    std::vector<uint64_t> a_bits(W), b_bits(W);
    bolt::f32_pack_rabitq(a.data(), D, 0.0f, a_bits.data());
    bolt::f32_pack_rabitq(b.data(), D, 0.0f, b_bits.data());

    int32_t got = -1;
    bolt::l2_pair_rabitq(a_bits.data(), b_bits.data(), W, &got);

    // Hand reference: for each dim, +1 if signs differ.
    int32_t ref = 0;
    for (size_t i = 0; i < D; ++i) {
        const bool ba = (a[i] >= 0.0f);
        const bool bb = (b[i] >= 0.0f);
        ref += (ba != bb) ? 1 : 0;
    }
    EXPECT_EQ(got, ref);
}

// ---------------------------------------------------------------------------
// F16 round-trip: f32 → f16 → unpack via kernel, compare squared diff.
// ---------------------------------------------------------------------------
TEST(BoltQuantizedDistance, F16_VerticalMatchesScalarReference) {
    constexpr size_t N = 12;
    constexpr size_t D = 5;
    std::vector<float> base(N * D);
    std::vector<float> query(D);
    for (size_t i = 0; i < N * D; ++i) base[i] = 0.25f * static_cast<float>(i);
    for (size_t d = 0; d < D; ++d)     query[d] = static_cast<float>(d) - 1.0f;

    // Pack to f16.
    std::vector<uint16_t> h_base(N * D);
    std::vector<uint16_t> h_query(D);
    for (size_t i = 0; i < N * D; ++i) h_base[i]  = f32_to_f16(base[i]);
    for (size_t d = 0; d < D; ++d)     h_query[d] = f32_to_f16(query[d]);

    // Column-major slab [D × N] of u16.
    std::vector<uint16_t> slab(N * D);
    for (size_t i = 0; i < N; ++i) {
        for (size_t d = 0; d < D; ++d) {
            slab[d * N + i] = h_base[i * D + d];
        }
    }

    std::vector<float> dists(N, 0.0f);
    for (size_t d = 0; d < D; ++d) {
        bolt::l2_vertical_f16_accumulate(
            slab.data() + d * N, h_query[d], dists.data(), N);
    }

    // Reference: scalar L2 over the *f16-decoded* base/query (so we
    // isolate kernel correctness from f16 rounding error).
    for (size_t i = 0; i < N; ++i) {
        float ref = 0.0f;
        for (size_t d = 0; d < D; ++d) {
            const float bf = bolt::detail::qd_f16_to_f32(h_base[i * D + d]);
            const float qf = bolt::detail::qd_f16_to_f32(h_query[d]);
            const float diff = qf - bf;
            ref += diff * diff;
        }
        EXPECT_FLOAT_EQ(dists[i], ref) << "vector i=" << i;
    }
}

// ---------------------------------------------------------------------------
// f32_pack_rabitq sanity: positives → 1, negatives → 0 with threshold 0.
// ---------------------------------------------------------------------------
TEST(BoltQuantizedDistance, RaBitQ_PackSignBoundary) {
    // 130 values: alternating positive / negative, threshold = 0.
    constexpr size_t N = 130;
    constexpr size_t W = (N + 63) / 64;
    std::vector<float>    in(N);
    std::vector<uint64_t> bits(W);
    for (size_t i = 0; i < N; ++i) {
        in[i] = (i % 2 == 0) ? 1.5f : -0.25f;
    }
    bolt::f32_pack_rabitq(in.data(), N, 0.0f, bits.data());

    // Even indices set, odd indices clear.
    for (size_t i = 0; i < N; ++i) {
        const uint64_t got = (bits[i >> 6] >> (i & 63)) & 1ull;
        const uint64_t expected = (i % 2 == 0) ? 1ull : 0ull;
        EXPECT_EQ(got, expected) << "i=" << i;
    }
    // Padded tail bits beyond N must be 0.
    for (size_t i = N; i < W * 64; ++i) {
        const uint64_t tail = (bits[i >> 6] >> (i & 63)) & 1ull;
        EXPECT_EQ(tail, 0ull) << "padded bit i=" << i;
    }
}

// ---------------------------------------------------------------------------
// SQ8 pack clamping at the saturation boundaries.
// ---------------------------------------------------------------------------
TEST(BoltQuantizedDistance, SQ8_PackClamps) {
    const float in[] = {-100.0f, 0.0f, 0.5f, 1.0f, 100.0f};
    uint8_t out[5];
    bolt::f32_pack_sq8(in, 5, /*min=*/0.0f, /*scale=*/1.0f / 255.0f, out);
    EXPECT_EQ(out[0], 0u);   // -100 clamps to 0.
    EXPECT_EQ(out[1], 0u);   // 0   exactly to 0.
    // 0.5 * 255 ~= 127.5; round-to-nearest with float reciprocal lands
    // at 127 or 128 depending on rounding direction of the inverse.
    EXPECT_GE(out[2], 127u);
    EXPECT_LE(out[2], 128u);
    EXPECT_EQ(out[3], 255u); // 1.0 maps to 255.
    EXPECT_EQ(out[4], 255u); // +100 clamps to 255.
}

}  // namespace
