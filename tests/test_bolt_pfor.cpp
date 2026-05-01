// test_bolt_pfor.cpp - coverage for `bolt::kernels::pfor` block kernels.
//
// Layer 2 substrate of plan `this-was-a-freach-hashed-crab.md`. Tests use STL
// freely (test side); the kernel under test does not.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "bolt/kernels/bolt_pfor.h"

namespace pfor = bolt::kernels::pfor;

namespace {

// Generate kBlockSize values whose top `bpv` bits are random.
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

}  // namespace

TEST(BoltPfor, PackUnpackRoundTripUniformBitWidths) {
    const uint8_t bpvs[] = {0, 1, 2, 4, 8, 16, 24, 31};
    for (uint8_t bpv : bpvs) {
        const auto in = make_uniform_block(bpv, 0xC0FFEEull + bpv);
        std::vector<uint8_t> packed(pfor::kPforMaxPackedBytes, 0);
        const int32_t bytes = pfor::pfor_pack_block(
            in.data(), packed.data(), static_cast<int32_t>(packed.size()));
        ASSERT_GT(bytes, 0) << "bpv=" << static_cast<int>(bpv);

        std::vector<uint32_t> out(pfor::kBlockSize, 0xDEADBEEF);
        const int32_t consumed = pfor::pfor_unpack_block(
            packed.data(), bytes, out.data());
        ASSERT_EQ(consumed, bytes);
        EXPECT_EQ(out, in) << "bpv=" << static_cast<int>(bpv);
    }
}

TEST(BoltPfor, PackUnpackWithExceptions) {
    // 123 values fit in 4 bpv (<=15); 5 values need 16 bpv.
    std::vector<uint32_t> in(pfor::kBlockSize, 7);  // bit_width=3, fits in bpv=4.
    const int32_t exc_idx[] = {3, 17, 42, 88, 119};
    for (int32_t i : exc_idx) { in[i] = 0xABCDu; }  // 16-bit value.

    std::vector<uint8_t> packed(pfor::kPforMaxPackedBytes, 0);
    const int32_t bytes = pfor::pfor_pack_block(
        in.data(), packed.data(), static_cast<int32_t>(packed.size()));
    ASSERT_GT(bytes, 0);

    const uint8_t token = packed[0];
    const uint8_t bpv = token & 0x1F;
    const uint8_t num_exc = token >> 5;
    EXPECT_LE(bpv, 8u) << "bpv must drop with patch channel; got "
                       << static_cast<int>(bpv);
    EXPECT_GE(bpv, 4u);  // at least 4 to hold the 123 small values.
    EXPECT_EQ(num_exc, 5u);

    std::vector<uint32_t> out(pfor::kBlockSize);
    ASSERT_EQ(pfor::pfor_unpack_block(packed.data(), bytes, out.data()), bytes);
    EXPECT_EQ(out, in);
}

TEST(BoltPfor, AllEqualFastPath) {
    // 128 zeros -> bpv=0, num_exc=0, packed length = 1 byte (just token).
    std::vector<uint32_t> zeros(pfor::kBlockSize, 0);
    std::vector<uint8_t> packed(pfor::kPforMaxPackedBytes, 0xFF);
    const int32_t bytes = pfor::pfor_pack_block(
        zeros.data(), packed.data(), static_cast<int32_t>(packed.size()));
    EXPECT_EQ(bytes, 1);
    EXPECT_EQ(packed[0], 0u);

    std::vector<uint32_t> out(pfor::kBlockSize, 0xDEADBEEF);
    EXPECT_EQ(pfor::pfor_unpack_block(packed.data(), bytes, out.data()), 1);
    for (auto x : out) { EXPECT_EQ(x, 0u); }

    // 128 identical non-zero values: bpv = bit_width(value), no exceptions.
    std::vector<uint32_t> sevens(pfor::kBlockSize, 7);
    std::fill(packed.begin(), packed.end(), uint8_t{0});
    const int32_t bytes2 = pfor::pfor_pack_block(
        sevens.data(), packed.data(), static_cast<int32_t>(packed.size()));
    ASSERT_GT(bytes2, 0);
    const uint8_t token = packed[0];
    EXPECT_EQ(token >> 5, 0u);   // no exceptions.
    EXPECT_EQ(token & 0x1F, 3u); // bit_width(7) = 3.
    std::vector<uint32_t> out2(pfor::kBlockSize, 0);
    ASSERT_EQ(pfor::pfor_unpack_block(packed.data(), bytes2, out2.data()), bytes2);
    EXPECT_EQ(out2, sevens);
}

TEST(BoltPfor, MaxBitWidthFallback) {
    // 127 values with bit-width 1, one with bit-width 32. The 32-bit value
    // becomes an exception; chosen bpv must be in [24, 24] so the high 8 bits
    // fit the patch byte.
    std::vector<uint32_t> in(pfor::kBlockSize, 1);
    in[42] = 0xFFFFFFFFu;
    std::vector<uint8_t> packed(pfor::kPforMaxPackedBytes, 0);
    const int32_t bytes = pfor::pfor_pack_block(
        in.data(), packed.data(), static_cast<int32_t>(packed.size()));
    ASSERT_GT(bytes, 0);

    const uint8_t token = packed[0];
    const uint8_t bpv = token & 0x1F;
    const uint8_t num_exc = token >> 5;
    EXPECT_EQ(bpv, 24u);  // smallest b such that 32 <= b + 8.
    EXPECT_EQ(num_exc, 1u);

    std::vector<uint32_t> out(pfor::kBlockSize);
    ASSERT_EQ(pfor::pfor_unpack_block(packed.data(), bytes, out.data()), bytes);
    EXPECT_EQ(out, in);
}

TEST(BoltPfor, ExceptionThresholdRespected) {
    // 8 outliers at 16-bit; 120 at 4-bit. Must pick bpv such that <=7 exceptions.
    std::vector<uint32_t> in(pfor::kBlockSize, 7);  // bit_width 3.
    for (int i = 0; i < 8; ++i) { in[i * 16] = 0xABCDu; }
    std::vector<uint8_t> packed(pfor::kPforMaxPackedBytes, 0);
    const int32_t bytes = pfor::pfor_pack_block(
        in.data(), packed.data(), static_cast<int32_t>(packed.size()));
    ASSERT_GT(bytes, 0);

    const uint8_t token = packed[0];
    const uint8_t num_exc = token >> 5;
    EXPECT_LE(num_exc, 7u);

    std::vector<uint32_t> out(pfor::kBlockSize);
    ASSERT_EQ(pfor::pfor_unpack_block(packed.data(), bytes, out.data()), bytes);
    EXPECT_EQ(out, in);
}

TEST(BoltPfor, DeltaToDocIdsPrefixSum) {
    // Build a delta sequence; pack; prefix-sum-decode into doc-ids.
    std::mt19937 rng(42);
    std::vector<uint32_t> deltas(pfor::kBlockSize);
    for (auto& d : deltas) { d = 1u + (rng() % 50u); }

    std::vector<uint8_t> packed(pfor::kPforMaxPackedBytes, 0);
    const int32_t bytes = pfor::pfor_pack_block(
        deltas.data(), packed.data(), static_cast<int32_t>(packed.size()));
    ASSERT_GT(bytes, 0);

    const uint64_t prev = 1000;
    std::vector<uint64_t> docs(pfor::kBlockSize);
    const int32_t consumed = pfor::pfor_unpack_to_doc_ids(
        packed.data(), bytes, prev, docs.data());
    ASSERT_EQ(consumed, bytes);

    uint64_t expect = prev;
    for (int32_t i = 0; i < pfor::kBlockSize; ++i) {
        expect += deltas[i];
        EXPECT_EQ(docs[i], expect) << "i=" << i;
    }
}

TEST(BoltPfor, MalformedInputRejected) {
    // Encode then truncate; unpack must return 0 cleanly.
    std::vector<uint32_t> in(pfor::kBlockSize, 0xCAFE);
    std::vector<uint8_t> packed(pfor::kPforMaxPackedBytes, 0);
    const int32_t bytes = pfor::pfor_pack_block(
        in.data(), packed.data(), static_cast<int32_t>(packed.size()));
    ASSERT_GT(bytes, 1);
    std::vector<uint32_t> out(pfor::kBlockSize, 0xDEADBEEF);
    EXPECT_EQ(pfor::pfor_unpack_block(packed.data(), bytes - 1, out.data()), 0);
    // Empty buffer.
    EXPECT_EQ(pfor::pfor_unpack_block(packed.data(), 0, out.data()), 0);
}

TEST(BoltPfor, CompressionRatio) {
    // Zipf-ish deltas: median ~10, max ~1000. Should pack into <= 12 bits/delta.
    std::mt19937 rng(7);
    std::vector<uint32_t> deltas(pfor::kBlockSize);
    for (auto& d : deltas) {
        // log-uniform between 1 and 1000.
        const double u = (static_cast<double>(rng()) /
                          static_cast<double>(rng.max()));
        const double x = std::exp(u * std::log(1000.0));
        d = static_cast<uint32_t>(std::max(1.0, x));
    }
    std::vector<uint8_t> packed(pfor::kPforMaxPackedBytes, 0);
    const int32_t bytes = pfor::pfor_pack_block(
        deltas.data(), packed.data(), static_cast<int32_t>(packed.size()));
    ASSERT_GT(bytes, 0);
    // 12 bits/delta * 128 deltas / 8 = 192 bytes (generous).
    EXPECT_LE(bytes, 192) << "packed=" << bytes << " bytes for 128 deltas";
    std::cout << "[ INFO     ] CompressionRatio: " << bytes
              << " bytes / 128 deltas = "
              << (bytes * 8.0 / 128.0) << " bits/delta" << std::endl;

    std::vector<uint32_t> out(pfor::kBlockSize);
    ASSERT_EQ(pfor::pfor_unpack_block(packed.data(), bytes, out.data()), bytes);
    EXPECT_EQ(out, deltas);
}
