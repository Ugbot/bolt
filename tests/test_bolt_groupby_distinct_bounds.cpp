// Bounds contract for DistinctCellArray::cell_at (G2CHK-92).
//
// WHY THIS FILE IS COMPILED WITH -DNDEBUG
// ----------------------------------------
// `cell_at` computes `&cells[j * entry_cap + g]` and hands the pointer back
// for the caller to WRITE THROUGH; the only guard used to be
//     assert(j < n_aggs);
//     assert(g < entry_cap);
// absent from any -DNDEBUG build (a stock Release) — an out-of-range (j, g)
// would return a pointer past `cells[]`, an out-of-bounds write waiting to
// happen. This file forces NDEBUG so it exercises exactly the configuration
// where the old asserts vanished — mirroring test_bolt_join_bounds.cpp
// (G2GRAPH-27). The asserts THEMSELVES stay in bolt_groupby_distinct.h
// (Tiger Style: they are still the right tool for a programmer-error
// precondition in an assert-live build) — this test lives in its own
// NDEBUG-only translation unit rather than the normal suite precisely
// because exercising the out-of-range case would abort wherever the
// asserts are live.

#ifndef NDEBUG
#error "test_bolt_groupby_distinct_bounds.cpp must be compiled with -DNDEBUG (see header comment)"
#endif

#include "bolt/join/bolt_groupby_distinct.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

TEST(BoltGroupbyDistinctBounds, CellAtOutOfBoundsReturnsNullNotOobPointer) {
    constexpr std::uint32_t entry_cap = 4;
    constexpr std::uint16_t n_aggs    = 2;
    bolt::DistinctCell cells[n_aggs * entry_cap]{};
    bolt::DistinctCellArray arr{ cells, entry_cap, n_aggs };

    // Would-have-overflowed cases: group index >= entry_cap, agg index
    // >= n_aggs, and both at once.
    EXPECT_EQ(arr.cell_at(0, entry_cap), nullptr);
    EXPECT_EQ(arr.cell_at(0, entry_cap + 1000), nullptr);
    EXPECT_EQ(arr.cell_at(n_aggs, 0), nullptr);
    EXPECT_EQ(arr.cell_at(n_aggs, entry_cap), nullptr);

    // In-bounds neighbours of the boundary still resolve normally — the
    // fix must not have tightened the valid range.
    EXPECT_NE(arr.cell_at(0, entry_cap - 1), nullptr);
    EXPECT_NE(arr.cell_at(n_aggs - 1, entry_cap - 1), nullptr);
}

}  // namespace
