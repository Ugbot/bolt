// test_bolt_vector_distance_f16.cpp — Wave 9.4 ζ.3.
//
// Cross-check `bolt::vec::l2_pair_f16` and `bolt::vec::l2_vertical_f16_accumulate`
// against an f32 reference computed via the well-tested `bolt::l2_pair_f32`
// kernel after quantising the f32 input to f16 and back. f16 carries
// ~11 bits of mantissa precision; max relative error against the f32
// reference is bounded by the quantisation step, not by the kernel.

#include "bolt/kernels/bolt_vector_distance.h"
#include "bolt/kernels/bolt_vector_distance_f16.h"
#include "bolt/kernels/bolt_vector_distance_u8.h"

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

TEST(BoltVectorDistanceF16, ZoneWalkL2F16PruneAt384) {
    // f16 zone-walk: mirrors MarbleDB's f32 zone_walk_l2_with_prune
    // contract. Two probes — prune fires (returns -1) vs. prune does
    // not fire (returns ~ref).
    constexpr size_t   D         = 384;
    constexpr uint32_t zone_size = 64;
    constexpr uint32_t n_zones   = D / zone_size;   // 6
    static_assert(D % zone_size == 0,
                  "test assumes zones tile D exactly");

    std::vector<float> q_f32(D), row_f32(D);
    fill_random(q_f32,   0xF16AAAA);
    fill_random(row_f32, 0xF16BBBB);

    std::vector<uint16_t> q_f16(D), row_f16(D);
    bolt::vec::quantise_f32_to_f16(q_f32.data(),   q_f16.data(),   D);
    bolt::vec::quantise_f32_to_f16(row_f32.data(), row_f16.data(), D);

    // Reference: full L2² computed as the sum of per-zone l2_pair_f16
    // contributions. Matches what zone_walk would return at prune=+inf
    // exactly (same accumulation order, same kernel, no pruning).
    float ref = 0.0f;
    for (uint32_t zi = 0; zi < n_zones; ++zi) {
        const uint32_t d_lo = zi * zone_size;
        ref += bolt::vec::l2_pair_f16(q_f16.data() + d_lo,
                                       row_f16.data() + d_lo,
                                       zone_size);
    }
    ASSERT_GT(ref, 1.0f) << "test corpus must have non-trivial L2²";

    // Probe 1: prune fires (threshold strictly below the true value).
    const float pruned = bolt::vec::zone_walk_l2_f16_with_prune(
        q_f16.data(), row_f16.data(),
        static_cast<uint32_t>(D), zone_size,
        /*zone_order=*/nullptr, n_zones,
        /*prune_threshold=*/ref - 1.0f);
    EXPECT_FLOAT_EQ(pruned, -1.0f);

    // Probe 2: no prune (threshold strictly above the true value).
    const float full = bolt::vec::zone_walk_l2_f16_with_prune(
        q_f16.data(), row_f16.data(),
        static_cast<uint32_t>(D), zone_size,
        /*zone_order=*/nullptr, n_zones,
        /*prune_threshold=*/ref + 1.0f);
    EXPECT_FLOAT_EQ(full, ref);

    // Probe 3: BOND-style permutation (reverse) should produce the same
    // unpruned distance because the accumulation is commutative in f32
    // (modulo rounding, well within FLOAT_EQ for D=384).
    uint8_t perm[n_zones];
    for (uint32_t i = 0; i < n_zones; ++i) {
        perm[i] = static_cast<uint8_t>(n_zones - 1u - i);
    }
    const float full_perm = bolt::vec::zone_walk_l2_f16_with_prune(
        q_f16.data(), row_f16.data(),
        static_cast<uint32_t>(D), zone_size,
        perm, n_zones,
        /*prune_threshold=*/ref + 1.0f);
    EXPECT_NEAR(full_perm, ref, ref * 5e-6f + 1e-3f);
}

TEST(BoltVectorDistanceF16, ZoneWalkL2F16ShortTailZone) {
    // Last zone is short (zone_size doesn't divide D). Exercises the
    // d_hi clamp branch — the most common bug shape for this helper.
    constexpr uint32_t D         = 100;
    constexpr uint32_t zone_size = 32;
    constexpr uint32_t n_zones   = 4;   // 32 + 32 + 32 + 4
    static_assert(n_zones * zone_size > D, "tail zone must be short");

    std::vector<float> q_f32(D), row_f32(D);
    fill_random(q_f32,   0x1111);
    fill_random(row_f32, 0x2222);

    std::vector<uint16_t> q_f16(D), row_f16(D);
    bolt::vec::quantise_f32_to_f16(q_f32.data(),   q_f16.data(),   D);
    bolt::vec::quantise_f32_to_f16(row_f32.data(), row_f16.data(), D);

    // Reference: matches the kernel's clamping logic exactly.
    float ref = 0.0f;
    for (uint32_t zi = 0; zi < n_zones; ++zi) {
        const uint32_t d_lo = zi * zone_size;
        if (d_lo >= D) continue;
        const uint32_t d_hi = (d_lo + zone_size <= D) ? d_lo + zone_size : D;
        ref += bolt::vec::l2_pair_f16(q_f16.data() + d_lo,
                                       row_f16.data() + d_lo,
                                       d_hi - d_lo);
    }
    const float got = bolt::vec::zone_walk_l2_f16_with_prune(
        q_f16.data(), row_f16.data(),
        D, zone_size, /*zone_order=*/nullptr, n_zones,
        /*prune_threshold=*/ref + 100.0f);
    EXPECT_FLOAT_EQ(got, ref);
}

TEST(BoltVectorDistanceF16, L2PairI8MatchesL2PairU8ReinterpretCast) {
    // Locks in the i8-via-u8 reinterpret asymmetry: the naive
    // reinterpret_cast is wrong on sign-straddling inputs; the
    // XOR-128 bias trick is the correct path.
    constexpr size_t D = 256;
    std::vector<int8_t> a_i8(D), b_i8(D);
    std::mt19937 rng(0x18B1A5);
    std::uniform_int_distribution<int> dist(-128, 127);
    for (auto& x : a_i8) x = static_cast<int8_t>(dist(rng));
    for (auto& x : b_i8) x = static_cast<int8_t>(dist(rng));

    // True signed reference, computed in i32.
    uint64_t ref = 0u;
    for (size_t i = 0; i < D; ++i) {
        const int32_t d =
            static_cast<int32_t>(a_i8[i]) - static_cast<int32_t>(b_i8[i]);
        ref += static_cast<uint64_t>(d * d);
    }

    // (a) Naive reinterpret_cast through l2_pair_u8 — expected to be
    // wrong on a random sign-straddling corpus. Verify it diverges
    // from the signed reference (otherwise this test is silently
    // passing for the wrong reason).
    const uint32_t naive = bolt::vec::l2_pair_u8(
        reinterpret_cast<const uint8_t*>(a_i8.data()),
        reinterpret_cast<const uint8_t*>(b_i8.data()), D);
    EXPECT_NE(static_cast<uint64_t>(naive), ref)
        << "with random sign-straddling i8 inputs, naive reinterpret "
           "must NOT match the signed truth (else the asymmetry "
           "documentation is stale)";

    // (b) Bias trick (XOR 0x80 maps [-128,+127] → [0,255] monotone).
    // Pairwise diffs are preserved exactly, so l2_pair_u8 over biased
    // inputs equals the signed L2² of the original i8 inputs.
    std::vector<uint8_t> a_u8(D), b_u8(D);
    for (size_t i = 0; i < D; ++i) {
        a_u8[i] = static_cast<uint8_t>(a_i8[i]) ^ static_cast<uint8_t>(0x80);
        b_u8[i] = static_cast<uint8_t>(b_i8[i]) ^ static_cast<uint8_t>(0x80);
    }
    const uint32_t biased = bolt::vec::l2_pair_u8(
        a_u8.data(), b_u8.data(), D);
    EXPECT_EQ(static_cast<uint64_t>(biased), ref)
        << "XOR-128 bias trick must recover the signed L2² exactly";
}

TEST(BoltVectorDistanceF16, L2PairI8NaiveMatchesU8WhenSameSign) {
    // Companion edge-case: when all i8 values are non-negative, the
    // bit pattern equals the unsigned interpretation, so the naive
    // reinterpret IS correct. Documents the "harmless when no sign
    // straddle" boundary.
    constexpr size_t D = 64;
    std::vector<int8_t> a_i8(D), b_i8(D);
    std::mt19937 rng(0xCAFEF00D);
    std::uniform_int_distribution<int> dist(0, 127);   // i8 ≥ 0
    for (auto& x : a_i8) x = static_cast<int8_t>(dist(rng));
    for (auto& x : b_i8) x = static_cast<int8_t>(dist(rng));

    uint64_t ref = 0u;
    for (size_t i = 0; i < D; ++i) {
        const int32_t d =
            static_cast<int32_t>(a_i8[i]) - static_cast<int32_t>(b_i8[i]);
        ref += static_cast<uint64_t>(d * d);
    }
    const uint32_t naive = bolt::vec::l2_pair_u8(
        reinterpret_cast<const uint8_t*>(a_i8.data()),
        reinterpret_cast<const uint8_t*>(b_i8.data()), D);
    EXPECT_EQ(static_cast<uint64_t>(naive), ref)
        << "non-negative i8 inputs: bit pattern matches unsigned, naive "
           "reinterpret IS correct (no bias needed)";
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
