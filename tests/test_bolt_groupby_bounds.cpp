// Bounds contract for GroupByTable::ingest_two_pass (G2CHK-92).
//
// WHY THIS FILE IS COMPILED WITH -DNDEBUG
// ----------------------------------------
// `ingest_two_pass` used to guard its only capacity check with
//     assert(num_groups + n <= capacity);   // worst-case all distinct
// which is absent from any -DNDEBUG build (a stock Release). Unlike the
// per-row `ingest_unchecked` (kept assert-only — see the comment above it in
// bolt_groupby.h, verified safe by construction for every real caller), this
// check is PER-CALL, so promoting it to a real runtime check costs one
// comparison per call, not per row. This file forces NDEBUG so it exercises
// exactly the configuration where the old assert vanished — mirroring
// test_bolt_join_bounds.cpp (G2GRAPH-27).
//
// THE FIX
// -------
// ingest_two_pass now returns bool: true on success, false (and writes
// NOTHING — the check runs before Pass A even starts) when `n` would push
// num_groups past capacity.

#ifndef NDEBUG
#error "test_bolt_groupby_bounds.cpp must be compiled with -DNDEBUG (see header comment)"
#endif

#include "bolt/join/bolt_groupby.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

using namespace bolt;

// A call whose n would push num_groups past capacity must fail closed
// (return false) and touch NOTHING — no partial group insert, no dummy-slot
// write left dangling.
TEST(GroupByBounds, TwoPassOverflowFailsClosed) {
    Arena arena;
    GroupByTable tbl;
    // tight_sizing rounds up to kSwissMinCapacity (16) at minimum.
    ASSERT_TRUE(GroupByTable::create(&tbl, 16, &arena));
    ASSERT_EQ(tbl.capacity, 16u);

    // Fill 10 distinct groups first (well within capacity).
    std::vector<uint64_t> ks(10);
    std::vector<int64_t>  vs(10);
    for (int i = 0; i < 10; ++i) { ks[static_cast<size_t>(i)] = static_cast<uint64_t>(i); vs[static_cast<size_t>(i)] = i * 7; }
    std::vector<int32_t>  existing(10);
    std::vector<uint32_t> miss(10);
    ASSERT_TRUE(tbl.ingest_two_pass(ks.data(), vs.data(), 10,
                                    existing.data(), miss.data()));
    ASSERT_EQ(tbl.num_groups, 10u);

    // Now try to ingest 10 MORE distinct keys: 10 + 10 = 20 > capacity (16).
    // Pre-fix this was an assert only -- absent here -- and the loop below
    // would have written group records past `payload`/`group_keys`.
    std::vector<uint64_t> ks2(10);
    std::vector<int64_t>  vs2(10);
    for (int i = 0; i < 10; ++i) { ks2[static_cast<size_t>(i)] = static_cast<uint64_t>(1000 + i); vs2[static_cast<size_t>(i)] = i; }
    std::vector<int32_t>  existing2(10);
    std::vector<uint32_t> miss2(10);
    const bool ok = tbl.ingest_two_pass(ks2.data(), vs2.data(), 10,
                                        existing2.data(), miss2.data());
    EXPECT_FALSE(ok);
    // Nothing was written: num_groups is exactly what it was before the
    // failed call, and the 10 original groups are still intact.
    EXPECT_EQ(tbl.num_groups, 10u);
    for (int i = 0; i < 10; ++i) {
        const int32_t slot = tbl.ht.find(static_cast<uint64_t>(i));
        ASSERT_GE(slot, 0) << "key " << i;
        EXPECT_EQ(tbl.payload[slot].sum, i * 7);
    }
    // None of the new (overflowing) keys were inserted.
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(tbl.ht.find(static_cast<uint64_t>(1000 + i)), -1);
    }

    // The table is still usable after the failed call: a correctly-sized
    // follow-up ingest (6 more keys -> num_groups=16=capacity) succeeds.
    std::vector<uint64_t> ks3(6);
    std::vector<int64_t>  vs3(6);
    for (int i = 0; i < 6; ++i) { ks3[static_cast<size_t>(i)] = static_cast<uint64_t>(2000 + i); vs3[static_cast<size_t>(i)] = i; }
    std::vector<int32_t>  existing3(6);
    std::vector<uint32_t> miss3(6);
    EXPECT_TRUE(tbl.ingest_two_pass(ks3.data(), vs3.data(), 6,
                                    existing3.data(), miss3.data()));
    EXPECT_EQ(tbl.num_groups, 16u);
}

// A call that ingests EXACTLY up to capacity must still succeed (the fix
// must not have tightened the boundary by one).
TEST(GroupByBounds, TwoPassExactCapacityFits) {
    Arena arena;
    GroupByTable tbl;
    ASSERT_TRUE(GroupByTable::create(&tbl, 16, &arena));
    ASSERT_EQ(tbl.capacity, 16u);

    std::vector<uint64_t> ks(16);
    std::vector<int64_t>  vs(16);
    for (int i = 0; i < 16; ++i) { ks[static_cast<size_t>(i)] = static_cast<uint64_t>(i); vs[static_cast<size_t>(i)] = i; }
    std::vector<int32_t>  existing(16);
    std::vector<uint32_t> miss(16);
    EXPECT_TRUE(tbl.ingest_two_pass(ks.data(), vs.data(), 16,
                                    existing.data(), miss.data()));
    EXPECT_EQ(tbl.num_groups, 16u);
}

}  // namespace
