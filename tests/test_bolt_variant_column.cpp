// test_bolt_variant_column.cpp — coverage for `bolt::VariantColumn`.
//
// Layer 1.2b of plan `this-was-a-freach-hashed-crab.md` — sum-type
// column. UInt8 discriminator selects one of N typed variants; 254 =
// shared overflow, 255 = null. Compaction collapses uniform
// discriminators to `ColumnFormat::Constant`.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_variant_column.h"

using bolt::BoltColumn;
using bolt::BoltType;
using bolt::ColumnFormat;
using bolt::VariantColumn;

namespace {

bolt::Arena make_arena() {
    bolt::ArenaConfig cfg{};
    cfg.initial_block_size = 64u * 1024u;
    cfg.max_block_size     = 1u * 1024u * 1024u;
    cfg.alignment          = 64u;
    return bolt::Arena(cfg);
}

// Allocate a Flat UInt8 discriminator column from an arena and copy
// the supplied bytes into it.
BoltColumn make_disc_flat(const std::vector<uint8_t>& bytes,
                          bolt::Arena* arena) {
    BoltColumn c = BoltColumn::make_flat_alloc(
        static_cast<int64_t>(bytes.size()), BoltType::UInt8, arena);
    std::memcpy(c.data, bytes.data(), bytes.size());
    return c;
}

}  // namespace

TEST(BoltVariantColumn, SingleVariantUniform) {
    bolt::Arena arena = make_arena();
    constexpr int64_t kN = 1000;

    std::vector<uint8_t> disc(kN, 0u);
    BoltColumn discriminator = make_disc_flat(disc, &arena);

    BoltColumn variants[1];
    variants[0] = BoltColumn::make_flat_alloc(kN, BoltType::Int32, &arena);
    auto* v0 = static_cast<int32_t*>(variants[0].data);
    for (int64_t i = 0; i < kN; ++i) v0[i] = static_cast<int32_t>(i * 7);

    BoltColumn shared = BoltColumn::make_empty();
    VariantColumn vc = bolt::variant_column_make(
        discriminator, variants, 1, shared, &arena);
    ASSERT_EQ(vc.length, kN);
    ASSERT_EQ(vc.variant_count, 1);

    EXPECT_TRUE(bolt::variant_column_compact(&vc, &arena));
    EXPECT_EQ(vc.discriminator.format, ColumnFormat::Constant);

    for (int64_t row : {int64_t{0}, int64_t{1}, int64_t{500}, kN - 1}) {
        const BoltColumn* picked = nullptr;
        const uint8_t d = bolt::variant_column_at(&vc, row, &picked);
        EXPECT_EQ(d, 0u);
        ASSERT_NE(picked, nullptr);
        EXPECT_EQ(picked, &vc.variants[0]);
        EXPECT_EQ(picked->typed_data<int32_t>()[row],
                  static_cast<int32_t>(row * 7));
    }
}

TEST(BoltVariantColumn, TwoVariantBalanced) {
    bolt::Arena arena = make_arena();
    constexpr int64_t kN = 1000;

    std::vector<uint8_t> disc(kN);
    for (int64_t i = 0; i < kN; ++i) disc[i] = static_cast<uint8_t>(i & 1);
    BoltColumn discriminator = make_disc_flat(disc, &arena);

    BoltColumn variants[2];
    variants[0] = BoltColumn::make_flat_alloc(kN, BoltType::Int32,   &arena);
    variants[1] = BoltColumn::make_flat_alloc(kN, BoltType::Float32, &arena);
    auto* v0 = static_cast<int32_t*>(variants[0].data);
    auto* v1 = static_cast<float*>  (variants[1].data);
    for (int64_t i = 0; i < kN; ++i) {
        v0[i] = static_cast<int32_t>(i + 100);
        v1[i] = static_cast<float>(i) * 0.5f;
    }

    BoltColumn shared = BoltColumn::make_empty();
    VariantColumn vc = bolt::variant_column_make(
        discriminator, variants, 2, shared, &arena);
    ASSERT_EQ(vc.length, kN);
    ASSERT_EQ(vc.variant_count, 2);

    for (int64_t i = 0; i < kN; ++i) {
        const BoltColumn* picked = nullptr;
        const uint8_t d = bolt::variant_column_at(&vc, i, &picked);
        const uint8_t expected = static_cast<uint8_t>(i & 1);
        ASSERT_EQ(d, expected);
        ASSERT_NE(picked, nullptr);
        EXPECT_EQ(picked, &vc.variants[expected]);
        if (expected == 0u) {
            EXPECT_EQ(picked->typed_data<int32_t>()[i],
                      static_cast<int32_t>(i + 100));
        } else {
            EXPECT_FLOAT_EQ(picked->typed_data<float>()[i],
                            static_cast<float>(i) * 0.5f);
        }
    }
}

TEST(BoltVariantColumn, NullDiscriminator) {
    bolt::Arena arena = make_arena();
    std::vector<uint8_t> disc = {0u, bolt::kVariantDiscNull, 0u, bolt::kVariantDiscNull};
    BoltColumn discriminator = make_disc_flat(disc, &arena);

    BoltColumn variants[1];
    variants[0] = BoltColumn::make_flat_alloc(4, BoltType::Int32, &arena);

    VariantColumn vc = bolt::variant_column_make(
        discriminator, variants, 1, BoltColumn::make_empty(), &arena);
    ASSERT_EQ(vc.length, 4);

    const BoltColumn* picked = nullptr;
    EXPECT_EQ(bolt::variant_column_at(&vc, 0, &picked), 0u);
    EXPECT_NE(picked, nullptr);
    picked = nullptr;
    EXPECT_EQ(bolt::variant_column_at(&vc, 1, &picked),
              bolt::kVariantDiscNull);
    EXPECT_EQ(picked, nullptr);
    picked = nullptr;
    EXPECT_EQ(bolt::variant_column_at(&vc, 3, &picked),
              bolt::kVariantDiscNull);
    EXPECT_EQ(picked, nullptr);
}

TEST(BoltVariantColumn, OverflowDiscriminator) {
    bolt::Arena arena = make_arena();
    std::vector<uint8_t> disc = {bolt::kVariantDiscOverflow, 0u,
                                 bolt::kVariantDiscOverflow};
    BoltColumn discriminator = make_disc_flat(disc, &arena);

    BoltColumn variants[1];
    variants[0] = BoltColumn::make_flat_alloc(3, BoltType::Int32, &arena);

    VariantColumn vc = bolt::variant_column_make(
        discriminator, variants, 1, BoltColumn::make_empty(), &arena);
    ASSERT_EQ(vc.length, 3);

    const BoltColumn* picked = nullptr;
    EXPECT_EQ(bolt::variant_column_at(&vc, 0, &picked),
              bolt::kVariantDiscOverflow);
    EXPECT_EQ(picked, nullptr);
    picked = nullptr;
    EXPECT_EQ(bolt::variant_column_at(&vc, 2, &picked),
              bolt::kVariantDiscOverflow);
    EXPECT_EQ(picked, nullptr);
}

TEST(BoltVariantColumn, CompactDetectionUniform) {
    bolt::Arena arena = make_arena();
    constexpr int64_t kN = 257;
    std::vector<uint8_t> disc(kN, 3u);
    BoltColumn discriminator = make_disc_flat(disc, &arena);

    BoltColumn variants[4];
    for (int i = 0; i < 4; ++i) {
        variants[i] = BoltColumn::make_flat_alloc(kN, BoltType::Int32, &arena);
    }
    VariantColumn vc = bolt::variant_column_make(
        discriminator, variants, 4, BoltColumn::make_empty(), &arena);
    EXPECT_EQ(vc.discriminator.format, ColumnFormat::Flat);

    EXPECT_TRUE(bolt::variant_column_compact(&vc, &arena));
    EXPECT_EQ(vc.discriminator.format, ColumnFormat::Constant);
    EXPECT_EQ(vc.discriminator.get_constant<uint8_t>(), 3u);

    // Reads still work after compaction — every row routes to variants[3].
    for (int64_t row : {int64_t{0}, int64_t{128}, kN - 1}) {
        const BoltColumn* picked = nullptr;
        EXPECT_EQ(bolt::variant_column_at(&vc, row, &picked), 3u);
        ASSERT_NE(picked, nullptr);
        EXPECT_EQ(picked, &vc.variants[3]);
    }
}

TEST(BoltVariantColumn, CompactDetectionNonUniform) {
    bolt::Arena arena = make_arena();
    std::vector<uint8_t> disc = {1u, 1u, 1u, 0u, 1u};
    BoltColumn discriminator = make_disc_flat(disc, &arena);

    BoltColumn variants[2];
    variants[0] = BoltColumn::make_flat_alloc(5, BoltType::Int32, &arena);
    variants[1] = BoltColumn::make_flat_alloc(5, BoltType::Int32, &arena);
    VariantColumn vc = bolt::variant_column_make(
        discriminator, variants, 2, BoltColumn::make_empty(), &arena);
    ASSERT_EQ(vc.discriminator.format, ColumnFormat::Flat);

    EXPECT_FALSE(bolt::variant_column_compact(&vc, &arena));
    EXPECT_EQ(vc.discriminator.format, ColumnFormat::Flat);
}

TEST(BoltVariantColumn, MakeRejectsBadInput) {
    bolt::Arena arena = make_arena();

    // variant_count > kVariantMaxVariants
    {
        std::vector<uint8_t> disc(4, 0u);
        BoltColumn discriminator = make_disc_flat(disc, &arena);
        BoltColumn dummy = BoltColumn::make_flat_alloc(4, BoltType::Int32, &arena);
        VariantColumn vc = bolt::variant_column_make(
            discriminator, &dummy,
            static_cast<int32_t>(bolt::kVariantMaxVariants) + 1,
            BoltColumn::make_empty(), &arena);
        EXPECT_EQ(vc.length, 0);
        EXPECT_EQ(vc.variants, nullptr);
    }

    // variants==nullptr with variant_count > 0
    {
        std::vector<uint8_t> disc(4, 0u);
        BoltColumn discriminator = make_disc_flat(disc, &arena);
        VariantColumn vc = bolt::variant_column_make(
            discriminator, nullptr, 2,
            BoltColumn::make_empty(), &arena);
        EXPECT_EQ(vc.length, 0);
        EXPECT_EQ(vc.variants, nullptr);
    }

    // discriminator wrong type
    {
        BoltColumn bad = BoltColumn::make_flat_alloc(4, BoltType::Int32, &arena);
        BoltColumn dummy = BoltColumn::make_flat_alloc(4, BoltType::Int32, &arena);
        VariantColumn vc = bolt::variant_column_make(
            bad, &dummy, 1, BoltColumn::make_empty(), &arena);
        EXPECT_EQ(vc.length, 0);
        EXPECT_EQ(vc.variants, nullptr);
    }
}
