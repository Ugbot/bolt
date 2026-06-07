// test_bolt_vector_distance_u8.cpp — Wave 9.4 ζ.4.
//
// Cross-check `bolt::vec::l2_pair_u8` and
// `bolt::vec::l2_vertical_u8_accumulate` against a scalar integer
// reference. u8 quantisation is lossy (8-bit per element); we compare
// against an integer reference computed on the SAME quantised input
// (no f32 round-trip — that would conflate quantisation error with
// kernel error).

#include "bolt/kernels/bolt_vector_distance_u8.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

namespace {

// Scalar integer reference. Same arithmetic the kernel does, hand-
// rolled here so a kernel bug can't paper over itself.
uint32_t ref_l2_u8(const std::vector<uint8_t>& a,
                    const std::vector<uint8_t>& b) {
    uint32_t acc = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const int32_t d = static_cast<int32_t>(a[i]) -
                          static_cast<int32_t>(b[i]);
        acc += static_cast<uint32_t>(d * d);
    }
    return acc;
}

void fill_random_u8(std::vector<uint8_t>& v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& x : v) x = static_cast<uint8_t>(dist(rng));
}

}  // namespace

TEST(BoltVectorDistanceU8, L2PairTinyBasic) {
    const uint8_t a[3] = {10u, 20u, 30u};
    const uint8_t b[3] = {15u, 25u, 35u};
    // (5)² + (5)² + (5)² = 75
    EXPECT_EQ(bolt::vec::l2_pair_u8(a, b, 3), 75u);
}

TEST(BoltVectorDistanceU8, L2PairMaxWidthSafe) {
    // Saturating u8 values: ensure no overflow when |a - b| = 255.
    std::vector<uint8_t> a(256, 0u);
    std::vector<uint8_t> b(256, 255u);
    // 255² * 256 = 65025 * 256 = 16646400 (fits in u32)
    EXPECT_EQ(bolt::vec::l2_pair_u8(a.data(), b.data(), 256),
              static_cast<uint32_t>(255u * 255u) * 256u);
}

TEST(BoltVectorDistanceU8, L2PairMatchesRefAt384) {
    constexpr size_t D = 384;
    std::vector<uint8_t> a(D), b(D);
    fill_random_u8(a, 0xDEAD);
    fill_random_u8(b, 0xBEEF);
    const uint32_t ref = ref_l2_u8(a, b);
    const uint32_t got = bolt::vec::l2_pair_u8(a.data(), b.data(), D);
    EXPECT_EQ(got, ref);    // exact integer arithmetic
}

TEST(BoltVectorDistanceU8, L2PairMatchesRefAt1536) {
    constexpr size_t D = 1536;
    std::vector<uint8_t> a(D), b(D);
    fill_random_u8(a, 0xC0DE);
    fill_random_u8(b, 0xFACE);
    const uint32_t ref = ref_l2_u8(a, b);
    const uint32_t got = bolt::vec::l2_pair_u8(a.data(), b.data(), D);
    EXPECT_EQ(got, ref);
}

TEST(BoltVectorDistanceU8, VerticalAccumulateMatchesPair) {
    constexpr size_t D = 256;
    constexpr size_t N = 64;
    std::vector<uint8_t> q(D);
    std::vector<uint8_t> corpus(N * D);
    fill_random_u8(q,      0xA1B2);
    fill_random_u8(corpus, 0xC3D4);

    // Pair: per-row L2.
    std::vector<uint32_t> pair_dists(N);
    for (size_t i = 0; i < N; ++i) {
        const std::vector<uint8_t> row(
            corpus.begin() + static_cast<long>(i * D),
            corpus.begin() + static_cast<long>((i + 1) * D));
        pair_dists[i] = ref_l2_u8(q, row);
    }

    // Vertical: column-major corpus + vertical accumulate per dim.
    std::vector<uint8_t> corpus_colmajor(N * D);
    for (size_t i = 0; i < N; ++i) {
        for (size_t d = 0; d < D; ++d) {
            corpus_colmajor[d * N + i] = corpus[i * D + d];
        }
    }
    std::vector<uint32_t> vertical_dists(N, 0u);
    for (size_t d = 0; d < D; ++d) {
        bolt::vec::l2_vertical_u8_accumulate(
            corpus_colmajor.data() + d * N,
            q[d],
            vertical_dists.data(), N);
    }
    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(vertical_dists[i], pair_dists[i]) << "row " << i;
    }
}

// Scalar reference for the vertical accumulate: per-element square of
// (q - col[i]) added into the running u32 accumulator. Independent of
// the kernel — a kernel bug cannot paper over itself.
uint32_t ref_vertical_one(uint32_t prev, uint8_t q, uint8_t x) {
    const int32_t d = static_cast<int32_t>(q) - static_cast<int32_t>(x);
    return prev + static_cast<uint32_t>(d * d);
}

// Bit-parity gate for the SIMD paths. Length 1021 is prime (NOT a
// multiple of any lane count: 16/32/64) so both the vector body and the
// scalar tail run. q and a column entry are forced to 0 and 255 so the
// diff is 255 (>181) — the exact case the old `mullo_epi16` square would
// have mis-handled had it not been backed by the d²<2^16 invariant. The
// rest is seeded-random. SIMD output must equal an all-scalar recompute.
TEST(BoltVectorDistanceU8, VerticalSimdBitParityWithLargeDiff) {
    constexpr size_t N = 1021;            // prime length → body + tail
    std::vector<uint8_t> col(N);
    fill_random_u8(col, 0x5151);
    col[0]     = 255u;                    // force a 255-diff vs q=0
    col[N - 1] = 255u;                    // force a 255-diff in the tail
    col[7]     = 200u;                    // diff 200 (>181) mid-body

    const uint8_t q = 0u;
    std::vector<uint32_t> got(N, 0u);
    bolt::vec::l2_vertical_u8_accumulate(col.data(), q, got.data(), N);

    for (size_t i = 0; i < N; ++i) {
        const uint32_t want = ref_vertical_one(0u, q, col[i]);
        ASSERT_EQ(got[i], want) << "i=" << i << " col=" << int(col[i]);
    }
}

// Pair bit-parity gate over the multi-accumulator AVX2 path. Length 1021
// exercises the 64-wide unroll, the 16-wide remainder, and the scalar
// tail in one shot. Integer arithmetic → exact equality.
TEST(BoltVectorDistanceU8, PairSimdBitParityMultiAccum) {
    constexpr size_t D = 1021;            // prime → all three code paths
    std::vector<uint8_t> a(D), b(D);
    fill_random_u8(a, 0x9001);
    fill_random_u8(b, 0x9002);
    a[3] = 255u; b[3] = 0u;               // 255-diff inside the unroll
    a[D - 1] = 0u; b[D - 1] = 255u;       // 255-diff in the scalar tail
    const uint32_t got = bolt::vec::l2_pair_u8(a.data(), b.data(), D);
    const uint32_t ref = ref_l2_u8(a, b);
    EXPECT_EQ(got, ref);
}

TEST(BoltVectorDistanceU8, QuantizationRoundTrip) {
    // f32 → u8 → f32: round-trip error bounded by half-step of the
    // u8 quantisation grid.
    constexpr size_t N = 100;
    std::vector<float> src(N);
    std::mt19937 rng(0x1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : src) x = dist(rng);

    const bolt::vec::QuantParams params =
        bolt::vec::compute_quant_params_global_f32(src.data(), src.size());
    EXPECT_GT(params.scale, 0.0f);

    std::vector<uint8_t> u8s(N);
    std::vector<float>   round_trip(N);
    bolt::vec::quantise_f32_to_u8(src.data(), u8s.data(), N, params);
    bolt::vec::dequantise_u8_to_f32(u8s.data(), round_trip.data(), N, params);

    const float step = 1.0f / params.scale;          // u8 quantum
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(round_trip[i], src[i], step)
            << "element " << i;
    }
}
