// test_bolt_vector_distance_f16.cpp — Wave 9.4 ζ.3.
//
// Cross-check `bolt::vec::l2_pair_f16` and `bolt::vec::l2_vertical_f16_accumulate`
// against an f32 reference computed via the well-tested `bolt::l2_pair_f32`
// kernel after quantising the f32 input to f16 and back. f16 carries
// ~11 bits of mantissa precision; max relative error against the f32
// reference is bounded by the quantisation step, not by the kernel.

#include "bolt/kernels/bolt_vector_distance.h"
#include "bolt/kernels/bolt_vector_distance_f16.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

// Quantise → de-quantise round-trip; returns the f32 vector that the
// f16 kernel would actually see (i.e., includes the f16 rounding
// error). The f16 kernel result should match the f32 result computed
// on this dequantised input modulo accumulator-order rounding.
std::vector<float> f32_via_f16(const std::vector<float>& f32_in) {
    std::vector<uint16_t> halves(f32_in.size());
    bolt::vec::quantise_f32_to_f16(f32_in.data(), halves.data(),
                                    f32_in.size());
    std::vector<float> f32_out(f32_in.size());
    bolt::vec::dequantise_f16_to_f32(halves.data(), f32_out.data(),
                                      halves.size());
    return f32_out;
}

float ref_l2(const std::vector<float>& a, const std::vector<float>& b) {
    float acc = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

void fill_random(std::vector<float>& v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& x : v) x = dist(rng);
}

}  // namespace

TEST(BoltVectorDistanceF16, L2PairTinyBasic) {
    // Three exactly-representable f16 values; result should be exact.
    const float a_f32[3] = {1.0f, 2.0f, 3.0f};
    const float b_f32[3] = {4.0f, 5.0f, 6.0f};
    std::vector<uint16_t> a_f16(3), b_f16(3);
    bolt::vec::quantise_f32_to_f16(a_f32, a_f16.data(), 3);
    bolt::vec::quantise_f32_to_f16(b_f32, b_f16.data(), 3);
    EXPECT_FLOAT_EQ(bolt::vec::l2_pair_f16(a_f16.data(), b_f16.data(), 3),
                    27.0f);
}

TEST(BoltVectorDistanceF16, L2PairMatchesF32RefAt384) {
    // GMM-ish random; D=384 (Cohere embed-english-v3.0 / OpenAI ada-002).
    constexpr size_t D = 384;
    std::vector<float> a(D), b(D);
    fill_random(a, 0xCAFE);
    fill_random(b, 0xBEEF);

    const auto a_q = f32_via_f16(a);
    const auto b_q = f32_via_f16(b);
    const float ref = ref_l2(a_q, b_q);

    std::vector<uint16_t> a_f16(D), b_f16(D);
    bolt::vec::quantise_f32_to_f16(a.data(), a_f16.data(), D);
    bolt::vec::quantise_f32_to_f16(b.data(), b_f16.data(), D);

    const float got = bolt::vec::l2_pair_f16(a_f16.data(), b_f16.data(), D);
    // Tolerance: at D=384 with values in [-2, 2], absolute L2 is in
    // the hundreds. Allow 0.5% relative for SIMD-reduction-order
    // rounding.
    EXPECT_NEAR(got, ref, ref * 5e-3f);
}

TEST(BoltVectorDistanceF16, L2PairMatchesF32RefAt1536) {
    constexpr size_t D = 1536;  // OpenAI text-embedding-3 dim.
    std::vector<float> a(D), b(D);
    fill_random(a, 0xD00D);
    fill_random(b, 0xF00D);

    const auto a_q = f32_via_f16(a);
    const auto b_q = f32_via_f16(b);
    const float ref = ref_l2(a_q, b_q);

    std::vector<uint16_t> a_f16(D), b_f16(D);
    bolt::vec::quantise_f32_to_f16(a.data(), a_f16.data(), D);
    bolt::vec::quantise_f32_to_f16(b.data(), b_f16.data(), D);

    const float got = bolt::vec::l2_pair_f16(a_f16.data(), b_f16.data(), D);
    EXPECT_NEAR(got, ref, ref * 5e-3f);
}

TEST(BoltVectorDistanceF16, VerticalAccumulateMatchesPair) {
    // The vertical accumulator should produce the same per-row result
    // as N independent pair calls when the corpus is laid out
    // column-major (one row's full dim contribution comes from N
    // single-dim accumulator updates).
    constexpr size_t D = 256;
    constexpr size_t N = 64;
    std::vector<float> q(D);
    std::vector<float> corpus_rowmajor(N * D);
    fill_random(q, 0x1A2B);
    fill_random(corpus_rowmajor, 0x3C4D);

    std::vector<uint16_t> q_f16(D);
    std::vector<uint16_t> corpus_f16(N * D);
    bolt::vec::quantise_f32_to_f16(q.data(), q_f16.data(), D);
    bolt::vec::quantise_f32_to_f16(corpus_rowmajor.data(),
                                    corpus_f16.data(), N * D);

    // Pair: per-row L2 against q.
    std::vector<float> pair_dists(N);
    for (size_t i = 0; i < N; ++i) {
        pair_dists[i] = bolt::vec::l2_pair_f16(
            q_f16.data(),
            corpus_f16.data() + i * D, D);
    }

    // Vertical: rebuild corpus column-major (N rows × D dims →
    // D columns × N rows). For each dim, accumulate into dists[N].
    std::vector<uint16_t> corpus_colmajor(N * D);
    for (size_t i = 0; i < N; ++i) {
        for (size_t d = 0; d < D; ++d) {
            corpus_colmajor[d * N + i] = corpus_f16[i * D + d];
        }
    }
    std::vector<float> vertical_dists(N, 0.0f);
    for (size_t d = 0; d < D; ++d) {
        bolt::vec::l2_vertical_f16_accumulate(
            corpus_colmajor.data() + d * N,
            q_f16[d],
            vertical_dists.data(), N);
    }

    for (size_t i = 0; i < N; ++i) {
        // Reduction order differs (per-dim accumulation vs
        // SIMD-hsum of full pair); 0.5% relative tolerance.
        const float ref = pair_dists[i];
        EXPECT_NEAR(vertical_dists[i], ref, ref * 5e-3f + 1e-3f)
            << "row " << i;
    }
}
