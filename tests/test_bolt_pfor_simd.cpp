// test_bolt_pfor_simd.cpp — bit-exact equivalence between the SIMD overlay
// and the scalar baseline for `bolt::kernels::pfor` (Layer 2.1b).
//
// The load-bearing correctness gate: the SIMD pack path MUST produce the
// same byte stream as the scalar pack path, and the SIMD unpack path MUST
// produce the same uint32 sequence as the scalar unpack path. Any drift is
// a wire-format divergence — every downstream consumer (BM25 codec, future
// column compression) reads bytes the writer emits, regardless of which
// path emitted them.
//
// Reference: docs/research/pfor-simd-overlay.md.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_pfor.h"
#include "bolt/kernels/bolt_pfor_simd.h"

namespace pfor = bolt::kernels::pfor;

namespace {

// Build a 128-element uint32 array whose values fit in `bpv` bits.
std::vector<uint32_t> make_uniform_block(uint8_t bpv, uint64_t seed) {
    std::vector<uint32_t> v(pfor::kBlockSize);
    std::mt19937_64 rng(seed);
    const uint64_t mask = (bpv == 0u) ? 0ull
                          : (bpv >= 32u ? ~0u
                                        : ((1ull << bpv) - 1ull));
    for (auto& x : v) {
        x = static_cast<uint32_t>(rng() & mask);
    }
    return v;
}

// True if either AVX2 or SSE4.2 SIMD path is active in this build.
constexpr bool kSimdActive =
#if BOLT_SIMD_AVX2 || BOLT_SIMD_SSE42
    true;
#else
    false;
#endif

}  // namespace

// --- Round-trip via the public API (SIMD selected at compile time) --------

TEST(BoltPforSimd, PackUnpackRoundTrip_AllSupportedBpv) {
    const uint8_t bpvs[] = {1, 2, 4, 8, 16, 32};
    for (uint8_t bpv : bpvs) {
        const auto in = make_uniform_block(bpv, 0xBEEF0000ull + bpv);
        std::vector<uint8_t> packed(
            pfor::pfor_payload_bytes(pfor::kBlockSize, bpv) +
            pfor::kBitpackTailSlack, 0);
        pfor::pack_bits(in.data(), pfor::kBlockSize, bpv, packed.data());

        std::vector<uint32_t> out(pfor::kBlockSize, 0xDEADBEEFu);
        pfor::unpack_bits(packed.data(), pfor::kBlockSize, bpv, out.data());
        EXPECT_EQ(out, in) << "bpv=" << static_cast<int>(bpv);
    }
}

// --- Bit-exact equivalence: scalar bytes == public (SIMD-first) bytes ----

TEST(BoltPforSimd, CrossPathEquivalence_PackBytesIdentical) {
    const uint8_t bpvs[] = {1, 2, 4, 8, 16, 32};
    for (uint8_t bpv : bpvs) {
        const auto in = make_uniform_block(bpv, 0xC0FFEE00ull + bpv);

        const int32_t bytes =
            pfor::pfor_payload_bytes(pfor::kBlockSize, bpv);
        std::vector<uint8_t> via_scalar(bytes + pfor::kBitpackTailSlack, 0);
        std::vector<uint8_t> via_public(bytes + pfor::kBitpackTailSlack, 0);

        pfor::internal::pack_bits_scalar(
            in.data(), pfor::kBlockSize, bpv, via_scalar.data());
        pfor::pack_bits(
            in.data(), pfor::kBlockSize, bpv, via_public.data());

        // Compare only the meaningful payload bytes — slack is uninitialised
        // for the SIMD path that does not extend writes past payload_bytes.
        for (int32_t i = 0; i < bytes; ++i) {
            ASSERT_EQ(via_scalar[i], via_public[i])
                << "bpv=" << static_cast<int>(bpv)
                << " byte=" << i
                << " scalar=" << static_cast<int>(via_scalar[i])
                << " public=" << static_cast<int>(via_public[i]);
        }
    }
}

TEST(BoltPforSimd, CrossPathEquivalence_UnpackValuesIdentical) {
    const uint8_t bpvs[] = {1, 2, 4, 8, 16, 32};
    for (uint8_t bpv : bpvs) {
        const auto in = make_uniform_block(bpv, 0xFEEDFACEull + bpv);

        std::vector<uint8_t> packed(
            pfor::pfor_payload_bytes(pfor::kBlockSize, bpv) +
            pfor::kBitpackTailSlack, 0);
        pfor::internal::pack_bits_scalar(
            in.data(), pfor::kBlockSize, bpv, packed.data());

        std::vector<uint32_t> via_scalar(pfor::kBlockSize, 0u);
        std::vector<uint32_t> via_public(pfor::kBlockSize, 0u);
        pfor::internal::unpack_bits_scalar(
            packed.data(), pfor::kBlockSize, bpv, via_scalar.data());
        pfor::unpack_bits(
            packed.data(), pfor::kBlockSize, bpv, via_public.data());

        EXPECT_EQ(via_scalar, via_public) << "bpv=" << static_cast<int>(bpv);
        EXPECT_EQ(via_scalar, in)         << "bpv=" << static_cast<int>(bpv);
    }
}

// --- Direct SIMD entry points (only when SIMD is active) ------------------

#if BOLT_SIMD_AVX2
TEST(BoltPforSimd, Avx2Direct_BitExactWithScalar) {
    const uint8_t bpvs[] = {1, 2, 4, 8, 16};
    for (uint8_t bpv : bpvs) {
        const auto in = make_uniform_block(bpv, 0xA5A5A5A5ull + bpv);

        const int32_t bytes =
            pfor::pfor_payload_bytes(pfor::kBlockSize, bpv);

        std::vector<uint8_t> via_scalar(bytes + pfor::kBitpackTailSlack, 0);
        std::vector<uint8_t> via_simd  (bytes + pfor::kBitpackTailSlack, 0);
        pfor::internal::pack_bits_scalar(
            in.data(), pfor::kBlockSize, bpv, via_scalar.data());
        const bool ok_pack = pfor::simd::pack_avx2(
            in.data(), pfor::kBlockSize, bpv, via_simd.data());
        ASSERT_TRUE(ok_pack) << "bpv=" << static_cast<int>(bpv);

        for (int32_t i = 0; i < bytes; ++i) {
            ASSERT_EQ(via_scalar[i], via_simd[i])
                << "bpv=" << static_cast<int>(bpv) << " byte=" << i;
        }

        std::vector<uint32_t> out_scalar(pfor::kBlockSize, 0u);
        std::vector<uint32_t> out_simd  (pfor::kBlockSize, 0u);
        pfor::internal::unpack_bits_scalar(
            via_scalar.data(), pfor::kBlockSize, bpv, out_scalar.data());
        const bool ok_unpack = pfor::simd::unpack_avx2(
            via_scalar.data(), pfor::kBlockSize, bpv, out_simd.data());
        ASSERT_TRUE(ok_unpack);
        EXPECT_EQ(out_scalar, out_simd) << "bpv=" << static_cast<int>(bpv);
        EXPECT_EQ(out_scalar, in)       << "bpv=" << static_cast<int>(bpv);
    }
}

TEST(BoltPforSimd, Avx2Direct_RejectsUnsupportedShape) {
    std::vector<uint32_t> in(pfor::kBlockSize, 0u);
    std::vector<uint8_t>  out(pfor::kBlockSize * 4, 0u);
    // Wrong n_values.
    EXPECT_FALSE(pfor::simd::pack_avx2(in.data(), 64, 4, out.data()));
    EXPECT_FALSE(pfor::simd::unpack_avx2(out.data(), 64, 4, in.data()));
    // Unsupported bpv.
    EXPECT_FALSE(pfor::simd::pack_avx2(in.data(), pfor::kBlockSize, 7, out.data()));
    EXPECT_FALSE(pfor::simd::unpack_avx2(out.data(), pfor::kBlockSize, 7, in.data()));
}
#endif  // BOLT_SIMD_AVX2

#if BOLT_SIMD_SSE42 && !BOLT_SIMD_AVX2
TEST(BoltPforSimd, Sse42Direct_BitExactWithScalar) {
    const uint8_t bpvs[] = {1, 2, 4, 8, 16};
    for (uint8_t bpv : bpvs) {
        const auto in = make_uniform_block(bpv, 0x1234ABCDull + bpv);
        const int32_t bytes =
            pfor::pfor_payload_bytes(pfor::kBlockSize, bpv);

        std::vector<uint8_t> via_scalar(bytes + pfor::kBitpackTailSlack, 0);
        std::vector<uint8_t> via_simd  (bytes + pfor::kBitpackTailSlack, 0);
        pfor::internal::pack_bits_scalar(
            in.data(), pfor::kBlockSize, bpv, via_scalar.data());
        ASSERT_TRUE(pfor::simd::pack_sse42(
            in.data(), pfor::kBlockSize, bpv, via_simd.data()));
        for (int32_t i = 0; i < bytes; ++i) {
            ASSERT_EQ(via_scalar[i], via_simd[i])
                << "bpv=" << static_cast<int>(bpv) << " byte=" << i;
        }

        std::vector<uint32_t> out_scalar(pfor::kBlockSize, 0u);
        std::vector<uint32_t> out_simd  (pfor::kBlockSize, 0u);
        pfor::internal::unpack_bits_scalar(
            via_scalar.data(), pfor::kBlockSize, bpv, out_scalar.data());
        ASSERT_TRUE(pfor::simd::unpack_sse42(
            via_scalar.data(), pfor::kBlockSize, bpv, out_simd.data()));
        EXPECT_EQ(out_scalar, out_simd);
        EXPECT_EQ(out_scalar, in);
    }
}
#endif  // BOLT_SIMD_SSE42 && !BOLT_SIMD_AVX2

// --- Block-level tests via pfor_pack_block / pfor_unpack_block ------------

TEST(BoltPforSimd, PforBlockRoundTripWithSimd_CommonBpv) {
    const uint8_t bpvs[] = {1, 2, 4, 8, 16};
    for (uint8_t bpv : bpvs) {
        const auto in = make_uniform_block(bpv, 0xBADC0DEull + bpv);

        std::vector<uint8_t> packed(pfor::kPforMaxPackedBytes, 0);
        const int32_t bytes = pfor::pfor_pack_block(
            in.data(), packed.data(), static_cast<int32_t>(packed.size()));
        ASSERT_GT(bytes, 0) << "bpv=" << static_cast<int>(bpv);

        const uint8_t token       = packed[0];
        const uint8_t token_bpv   = token & 0x1Fu;
        const uint8_t token_excs  = (token >> 5) & 0x07u;
        EXPECT_EQ(token_bpv, bpv) << "expected exact bpv (uniform input)";
        EXPECT_EQ(token_excs, 0u);

        std::vector<uint32_t> out(pfor::kBlockSize, 0xDEADBEEF);
        const int32_t consumed = pfor::pfor_unpack_block(
            packed.data(), bytes, out.data());
        ASSERT_EQ(consumed, bytes);
        EXPECT_EQ(out, in) << "bpv=" << static_cast<int>(bpv);
    }
}

TEST(BoltPforSimd, DeltaToDocIdsViaSimd) {
    // Delta sequence whose max bit-width forces bpv=8 (a SIMD-accelerated
    // path). Validate prefix-sum doc-ids match.
    std::mt19937 rng(0xD0CD0Cu);
    std::vector<uint32_t> deltas(pfor::kBlockSize);
    for (auto& d : deltas) { d = 1u + (rng() % 200u); }  // ≤ 200 fits in 8 bpv.

    std::vector<uint8_t> packed(pfor::kPforMaxPackedBytes, 0);
    const int32_t bytes = pfor::pfor_pack_block(
        deltas.data(), packed.data(), static_cast<int32_t>(packed.size()));
    ASSERT_GT(bytes, 0);
    EXPECT_EQ(packed[0] & 0x1Fu, 8u) << "expected bpv=8 path";

    const uint64_t prev = 100000ull;
    std::vector<uint64_t> docs(pfor::kBlockSize);
    const int32_t consumed = pfor::pfor_unpack_to_doc_ids(
        packed.data(), bytes, prev, docs.data());
    ASSERT_EQ(consumed, bytes);

    uint64_t expect = prev;
    for (int32_t i = 0; i < pfor::kBlockSize; ++i) {
        expect += deltas[i];
        ASSERT_EQ(docs[i], expect) << "i=" << i;
    }
}

TEST(BoltPforSimd, ScalarFallback_BpvNotInSimdSet) {
    // bpv=7 is intentionally NOT in the SIMD coverage table; the public
    // entry must fall through to scalar and still round-trip.
    std::vector<uint32_t> in(pfor::kBlockSize);
    std::mt19937 rng(7u);
    for (auto& x : in) { x = rng() & 0x7Fu; }  // 7 bits.

    std::vector<uint8_t> packed(
        pfor::pfor_payload_bytes(pfor::kBlockSize, 7) +
        pfor::kBitpackTailSlack, 0);
    pfor::pack_bits(in.data(), pfor::kBlockSize, 7u, packed.data());

    std::vector<uint32_t> out(pfor::kBlockSize, 0u);
    pfor::unpack_bits(packed.data(), pfor::kBlockSize, 7u, out.data());
    EXPECT_EQ(out, in);
}

TEST(BoltPforSimd, ReportsActiveSimdTier) {
    // Diagnostic — the test is informational. Logs which SIMD tier is
    // compiled in so a CI failure surfaces the build configuration.
#if BOLT_SIMD_AVX2
    std::cout << "[ INFO     ] BOLT_SIMD_AVX2 active" << std::endl;
#elif BOLT_SIMD_SSE42
    std::cout << "[ INFO     ] BOLT_SIMD_SSE42 active" << std::endl;
#else
    std::cout << "[ INFO     ] SIMD overlay inactive — scalar path only"
              << std::endl;
#endif
    EXPECT_TRUE(true);
    EXPECT_EQ(kSimdActive, kSimdActive);
}
