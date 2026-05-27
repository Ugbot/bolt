// Tests for bolt::DistinctCell / DistinctCellArray.
//
// Scope: per-cell dedup correctness, saturation behaviour, array indexing.
// Backs chukonu's K-AGG-B (COUNT(DISTINCT) / SUM(DISTINCT) honour the
// AggSpec::distinct bit at the kernel level rather than ignoring it).

#include "bolt/join/bolt_groupby_distinct.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

TEST(BoltGroupbyDistinct, EmptyCellHasZeroCount) {
    bolt::DistinctCell c{};
    c.init();
    EXPECT_EQ(c.n, 0u);
    EXPECT_EQ(c.overflow_seen, 0u);
    EXPECT_FALSE(c.saturated());
}

TEST(BoltGroupbyDistinct, InsertNewValuesAreCounted) {
    bolt::DistinctCell c{};
    c.init();
    EXPECT_TRUE(c.insert(1));
    EXPECT_TRUE(c.insert(2));
    EXPECT_TRUE(c.insert(3));
    EXPECT_EQ(c.n, 3u);
}

TEST(BoltGroupbyDistinct, InsertDuplicatesReturnFalse) {
    bolt::DistinctCell c{};
    c.init();
    EXPECT_TRUE(c.insert(7));
    EXPECT_FALSE(c.insert(7));
    EXPECT_FALSE(c.insert(7));
    EXPECT_EQ(c.n, 1u);
}

TEST(BoltGroupbyDistinct, CountDistinctOverDuplicateStream) {
    // Simulates: COUNT(DISTINCT col) over a stream {5, 1, 5, 2, 1, 5, 3, 2}.
    // Expected unique = 4 (values 1, 2, 3, 5).
    bolt::DistinctCell c{};
    c.init();
    const std::int64_t stream[] = {5, 1, 5, 2, 1, 5, 3, 2};
    std::int64_t count_distinct = 0;
    for (std::int64_t v : stream) {
        if (c.insert(v)) count_distinct += 1;
    }
    EXPECT_EQ(count_distinct, 4);
    EXPECT_EQ(c.n, 4u);
}

TEST(BoltGroupbyDistinct, SumDistinctOverDuplicateStream) {
    // Simulates: SUM(DISTINCT col) over {10, 10, 20, 10, 30, 20}.
    // Expected sum of uniques = 10 + 20 + 30 = 60.
    bolt::DistinctCell c{};
    c.init();
    const std::int64_t stream[] = {10, 10, 20, 10, 30, 20};
    std::int64_t sum_distinct = 0;
    for (std::int64_t v : stream) {
        if (c.insert(v)) sum_distinct += v;
    }
    EXPECT_EQ(sum_distinct, 60);
    EXPECT_EQ(c.n, 3u);
}

TEST(BoltGroupbyDistinct, SaturationMarksAndCountsOverflow) {
    bolt::DistinctCell c{};
    c.init();
    // Fill exactly to capacity.
    for (std::uint32_t i = 0; i < bolt::k_distinct_cell_cap; ++i) {
        EXPECT_TRUE(c.insert(static_cast<std::int64_t>(i)));
    }
    EXPECT_TRUE(c.saturated());
    EXPECT_EQ(c.n, bolt::k_distinct_cell_cap);
    // One more — saturated, must report false but track overflow.
    EXPECT_FALSE(c.insert(999));
    EXPECT_EQ(c.overflow_seen, 1u);
    EXPECT_EQ(c.n, bolt::k_distinct_cell_cap);
}

TEST(BoltGroupbyDistinct, ArrayIndexingPerCellIndependent) {
    constexpr std::uint32_t entry_cap = 4;
    constexpr std::uint16_t n_aggs    = 2;
    bolt::DistinctCell cells[n_aggs * entry_cap]{};
    bolt::DistinctCellArray arr{ cells, entry_cap, n_aggs };
    for (std::uint16_t j = 0; j < n_aggs; ++j)
        for (std::uint32_t g = 0; g < entry_cap; ++g)
            arr.cell_at(j, g)->init();

    // Distinct streams per (agg, group). Verify the cells don't leak across.
    EXPECT_TRUE(arr.cell_at(0, 0)->insert(42));
    EXPECT_TRUE(arr.cell_at(0, 1)->insert(42));   // same value, different group
    EXPECT_FALSE(arr.cell_at(0, 0)->insert(42));  // dup in cell (0, 0)
    EXPECT_TRUE(arr.cell_at(1, 0)->insert(42));   // same value, different agg
    EXPECT_EQ(arr.cell_at(0, 0)->n, 1u);
    EXPECT_EQ(arr.cell_at(0, 1)->n, 1u);
    EXPECT_EQ(arr.cell_at(1, 0)->n, 1u);
    EXPECT_EQ(arr.cell_at(1, 1)->n, 0u);
}

// ---- DistinctCell16 (Phase-B: inline-Utf8 keys) ----------------------------

namespace {
void fill16(std::uint8_t out[16], std::uint8_t a, std::uint8_t b) noexcept {
    std::memset(out, 0, 16);
    out[0] = a; out[1] = b;
}
}  // namespace

TEST(BoltGroupbyDistinct16, InsertNewAndDedupes) {
    bolt::DistinctCell16 c{};
    c.init();
    std::uint8_t v[16];
    fill16(v, 'A', 0); EXPECT_TRUE(bolt::distinct_cell16_insert(&c, v));
    fill16(v, 'B', 0); EXPECT_TRUE(bolt::distinct_cell16_insert(&c, v));
    fill16(v, 'C', 0); EXPECT_TRUE(bolt::distinct_cell16_insert(&c, v));
    fill16(v, 'D', 0); EXPECT_TRUE(bolt::distinct_cell16_insert(&c, v));
    fill16(v, 'E', 0); EXPECT_TRUE(bolt::distinct_cell16_insert(&c, v));
    // 3 duplicates — must all return false, n stays at 5.
    fill16(v, 'A', 0); EXPECT_FALSE(bolt::distinct_cell16_insert(&c, v));
    fill16(v, 'C', 0); EXPECT_FALSE(bolt::distinct_cell16_insert(&c, v));
    fill16(v, 'E', 0); EXPECT_FALSE(bolt::distinct_cell16_insert(&c, v));
    EXPECT_EQ(c.n, 5u);
}

// Two 16-byte payloads that share the first 8 bytes but differ in the tail
// — exactly the StringView shape (length+prefix shared, inline_data tail
// differs) that the chukonu Phase-A 8-byte dedup misclassifies as equal.
// DistinctCell16 must compare all 16 bytes and keep them distinct.
TEST(BoltGroupbyDistinct16, SharedFirst8BytesAreDistinct) {
    bolt::DistinctCell16 c{};
    c.init();
    std::uint8_t a[16];
    std::uint8_t b[16];
    std::memset(a, 0, 16);
    std::memset(b, 0, 16);
    // Shared first 8 bytes: pretend it's StringView{length=9,prefix="alph"}.
    const std::uint8_t shared[8] = {9, 0, 0, 0, 'a', 'l', 'p', 'h'};
    std::memcpy(a,     shared, 8);
    std::memcpy(b,     shared, 8);
    // Diverging tails — "abet1" vs "abet2".
    std::memcpy(a + 8, "abet1", 5);
    std::memcpy(b + 8, "abet2", 5);
    EXPECT_EQ(0, std::memcmp(a, b, 8));        // sanity: prefix collides
    EXPECT_NE(0, std::memcmp(a, b, 16));       // sanity: full payload differs
    EXPECT_TRUE(bolt::distinct_cell16_insert(&c, a));   // new
    EXPECT_TRUE(bolt::distinct_cell16_insert(&c, b));   // new — not a dup
    EXPECT_FALSE(bolt::distinct_cell16_insert(&c, a));  // dup of first
    EXPECT_FALSE(bolt::distinct_cell16_insert(&c, b));  // dup of second
    EXPECT_EQ(c.n, 2u);
    EXPECT_EQ(c.overflow_seen, 0u);
}

TEST(BoltGroupbyDistinct16, SaturationMarksOverflow) {
    bolt::DistinctCell16 c{};
    c.init();
    std::uint8_t v[16];
    for (std::uint32_t i = 0; i < bolt::k_distinct_cell_cap; ++i) {
        fill16(v, static_cast<std::uint8_t>(i >> 8),
                  static_cast<std::uint8_t>(i & 0xFF));
        EXPECT_TRUE(bolt::distinct_cell16_insert(&c, v));
    }
    EXPECT_TRUE(c.saturated());
    EXPECT_EQ(c.n, bolt::k_distinct_cell_cap);
    fill16(v, 0xFF, 0xFE);
    EXPECT_FALSE(bolt::distinct_cell16_insert(&c, v));
    EXPECT_EQ(c.overflow_seen, 1u);
    EXPECT_EQ(c.n, bolt::k_distinct_cell_cap);
}

}  // namespace
