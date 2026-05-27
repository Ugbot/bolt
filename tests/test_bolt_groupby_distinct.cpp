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

}  // namespace
