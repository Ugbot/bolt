// test_bolt_swiss_growable.cpp — SwissTableGrowable (growable/erasable
// SwissTable sibling, bolt_swiss_growable.h).
//
// Covers: insert/find/update, growth past the initial capacity, the hard
// ceiling failing cleanly (never silently), erase (both the never-full
// Empty-revert path and the tombstone path), tombstone reuse + purge under
// steady-state churn (the anti-tombstone-accumulation property), iteration,
// and a randomized stress cross-checked against a std::unordered_map oracle.
// (The oracle is test scaffolding — CLAUDE.md's container ban is on hot/engine
// code, not tests.)

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <unordered_map>

#include "bolt/bolt_arena.h"
#include "bolt/join/bolt_swiss_growable.h"

using bolt::Arena;
using bolt::ArenaConfig;
using bolt::SwissTableGrowable;

namespace {

ArenaConfig small_arena_cfg() {
    ArenaConfig cfg;
    cfg.initial_block_size = 1u << 16;  // registries are small; don't grab 4MB
    return cfg;
}

}  // namespace

TEST(SwissGrowable, InsertFindUpdate) {
    Arena arena(small_arena_cfg());
    SwissTableGrowable t;
    ASSERT_TRUE(SwissTableGrowable::create(&t, 8, &arena));

    EXPECT_EQ(t.find(42), -1);
    EXPECT_TRUE(t.insert(42, 7));
    EXPECT_TRUE(t.insert(43, 8));
    EXPECT_EQ(t.size, 2u);
    EXPECT_EQ(t.find(42), 7);
    EXPECT_EQ(t.find(43), 8);
    EXPECT_EQ(t.find(44), -1);

    // Update in place: size unchanged.
    EXPECT_TRUE(t.insert(42, 99));
    EXPECT_EQ(t.size, 2u);
    EXPECT_EQ(t.find(42), 99);
}

TEST(SwissGrowable, GrowsPastInitialCapacity) {
    Arena arena(small_arena_cfg());
    SwissTableGrowable t;
    ASSERT_TRUE(SwissTableGrowable::create(&t, 4, &arena));
    const uint32_t initial_cap = t.capacity;

    constexpr uint32_t kN = 10000;
    for (uint32_t i = 0; i < kN; ++i) {
        ASSERT_TRUE(t.insert(0x9E3779B97F4A7C15ull * (i + 1), i)) << i;
    }
    EXPECT_EQ(t.size, kN);
    EXPECT_GT(t.capacity, initial_cap);
    for (uint32_t i = 0; i < kN; ++i) {
        ASSERT_EQ(t.find(0x9E3779B97F4A7C15ull * (i + 1)),
                  static_cast<int32_t>(i)) << i;
    }
}

TEST(SwissGrowable, CeilingFailsCleanlyNeverSilently) {
    Arena arena(small_arena_cfg());
    SwissTableGrowable t;
    ASSERT_TRUE(SwissTableGrowable::create(&t, 4, &arena, /*max_cap=*/64));
    EXPECT_EQ(t.max_capacity, 64u);

    // Fill to the 7/8 load limit at the ceiling; the next NEW key must fail.
    uint32_t inserted = 0;
    for (uint32_t i = 0; i < 1000; ++i) {
        if (!t.insert(1000 + i, i)) break;
        ++inserted;
    }
    EXPECT_EQ(inserted, 64u - 64u / 8u);  // 56 = load limit at max capacity
    EXPECT_EQ(t.size, inserted);
    EXPECT_EQ(t.capacity, 64u);
    EXPECT_FALSE(t.insert(999999, 1));  // clean fail, no growth, no corrupt

    // Everything already in stays findable and updatable after the fail.
    for (uint32_t i = 0; i < inserted; ++i) {
        ASSERT_EQ(t.find(1000 + i), static_cast<int32_t>(i));
    }
    EXPECT_TRUE(t.insert(1000, 12345));  // update of existing key still works
    EXPECT_EQ(t.find(1000), 12345);
    EXPECT_EQ(t.size, inserted);

    // Erase one → exactly one new key fits again (tombstone or empty reuse).
    EXPECT_TRUE(t.erase(1001));
    EXPECT_TRUE(t.insert(999999, 1));
    EXPECT_EQ(t.find(999999), 1);
}

TEST(SwissGrowable, EraseBasicAndReinsert) {
    Arena arena(small_arena_cfg());
    SwissTableGrowable t;
    ASSERT_TRUE(SwissTableGrowable::create(&t, 32, &arena));

    for (uint64_t k = 1; k <= 20; ++k) ASSERT_TRUE(t.insert(k, uint32_t(k)));
    EXPECT_EQ(t.size, 20u);

    EXPECT_FALSE(t.erase(777));  // absent key
    EXPECT_TRUE(t.erase(10));
    EXPECT_EQ(t.size, 19u);
    EXPECT_EQ(t.find(10), -1);
    EXPECT_TRUE(t.erase(11));
    EXPECT_FALSE(t.erase(11));   // double-erase reports absent

    // Survivors intact.
    for (uint64_t k = 1; k <= 20; ++k) {
        if (k == 10 || k == 11) continue;
        ASSERT_EQ(t.find(k), static_cast<int32_t>(k)) << k;
    }

    // Reinsert an erased key.
    EXPECT_TRUE(t.insert(10, 1010));
    EXPECT_EQ(t.find(10), 1010);
    EXPECT_EQ(t.size, 19u);
}

// The anti-tombstone-accumulation property: steady-state churn (open/close
// sockets at a stable connection count — the fd-registry workload) must not
// degrade the table without bound. Occupancy (size + tombstones) is capped at
// 7/8 capacity by construction, and capacity must stabilize, not ratchet.
TEST(SwissGrowable, ChurnDoesNotAccumulateTombstonesUnbounded) {
    Arena arena(small_arena_cfg());
    SwissTableGrowable t;
    ASSERT_TRUE(SwissTableGrowable::create(&t, 64, &arena));

    // Reach steady state: 64 live keys.
    for (uint64_t k = 0; k < 64; ++k) ASSERT_TRUE(t.insert(k * 2654435761u + 1, uint32_t(k)));
    const uint32_t cap_after_fill = t.capacity;

    // 200k erase+insert pairs at constant size.
    std::mt19937_64 rng(1234);
    uint64_t next_key = 1000000;
    uint64_t live[64];
    for (uint64_t k = 0; k < 64; ++k) live[k] = k * 2654435761u + 1;
    for (uint32_t round = 0; round < 200000; ++round) {
        const uint32_t victim = static_cast<uint32_t>(rng() % 64);
        ASSERT_TRUE(t.erase(live[victim]));
        live[victim] = next_key++;
        ASSERT_TRUE(t.insert(live[victim], victim));
        ASSERT_LE(t.size + t.tombstones, t.load_limit());
    }
    EXPECT_EQ(t.size, 64u);
    // Capacity may have grown once past the fill point but must be bounded —
    // steady-state churn is not growth pressure.
    EXPECT_LE(t.capacity, cap_after_fill * 2);
    for (uint32_t k = 0; k < 64; ++k) {
        ASSERT_EQ(t.find(live[k]), static_cast<int32_t>(k));
    }
}

// Low-load erase should take the was-never-full path (revert to Empty) and
// leave zero tombstones — the common fd-registry case.
TEST(SwissGrowable, LowLoadEraseLeavesNoTombstones) {
    Arena arena(small_arena_cfg());
    SwissTableGrowable t;
    ASSERT_TRUE(SwissTableGrowable::create(&t, 256, &arena));
    ASSERT_GE(t.capacity, 512u);  // 2x-sized: plenty of empties everywhere

    for (uint64_t k = 1; k <= 32; ++k) ASSERT_TRUE(t.insert(k, uint32_t(k)));
    for (uint64_t k = 1; k <= 32; ++k) ASSERT_TRUE(t.erase(k));
    EXPECT_EQ(t.size, 0u);
    EXPECT_EQ(t.tombstones, 0u);
}

TEST(SwissGrowable, IterationEnumeratesExactlyLiveSet) {
    Arena arena(small_arena_cfg());
    SwissTableGrowable t;
    ASSERT_TRUE(SwissTableGrowable::create(&t, 64, &arena));

    std::unordered_map<uint64_t, uint32_t> expect;
    for (uint64_t k = 100; k < 180; ++k) {
        ASSERT_TRUE(t.insert(k, uint32_t(k * 3)));
        expect[k] = uint32_t(k * 3);
    }
    for (uint64_t k = 100; k < 180; k += 3) {
        ASSERT_TRUE(t.erase(k));
        expect.erase(k);
    }

    std::unordered_map<uint64_t, uint32_t> seen;
    for (uint32_t i = t.next_live(0); i < t.capacity; i = t.next_live(i + 1)) {
        ASSERT_TRUE(seen.emplace(t.slots[i].key, t.slots[i].value).second)
            << "duplicate key in iteration: " << t.slots[i].key;
    }
    EXPECT_EQ(seen, expect);
}

// Randomized stress vs a std::unordered_map oracle: mixed insert / update /
// erase / find, with growth from a tiny table, plus periodic full-iteration
// comparison. Any divergence (wrong value, ghost key, lost key, bad size)
// fails immediately.
TEST(SwissGrowable, OracleStress) {
    Arena arena(small_arena_cfg());
    SwissTableGrowable t;
    ASSERT_TRUE(SwissTableGrowable::create(&t, 4, &arena));

    std::unordered_map<uint64_t, uint32_t> oracle;
    std::mt19937_64 rng(20260916);

    constexpr uint32_t kOps = 300000;
    for (uint32_t op = 0; op < kOps; ++op) {
        const uint64_t key = (rng() % 4096) + 1;  // heavy key reuse → real churn
        const uint32_t action = static_cast<uint32_t>(rng() % 10);
        if (action < 5) {  // insert-or-update
            const uint32_t val = static_cast<uint32_t>(rng());
            ASSERT_TRUE(t.insert(key, val));
            oracle[key] = val;
        } else if (action < 8) {  // erase
            const bool expected = oracle.erase(key) > 0;
            ASSERT_EQ(t.erase(key), expected) << "op " << op << " key " << key;
        } else {  // find
            auto it = oracle.find(key);
            const int32_t expected =
                (it == oracle.end()) ? -1 : static_cast<int32_t>(it->second);
            ASSERT_EQ(t.find(key), expected) << "op " << op << " key " << key;
        }
        ASSERT_EQ(t.size, oracle.size());

        if ((op & 0xFFFFu) == 0xFFFFu) {  // periodic full sweep
            std::unordered_map<uint64_t, uint32_t> seen;
            for (uint32_t i = t.next_live(0); i < t.capacity;
                 i = t.next_live(i + 1)) {
                seen.emplace(t.slots[i].key, t.slots[i].value);
            }
            ASSERT_EQ(seen, oracle) << "iteration diverged at op " << op;
        }
    }

    // Final sweep.
    std::unordered_map<uint64_t, uint32_t> seen;
    for (uint32_t i = t.next_live(0); i < t.capacity; i = t.next_live(i + 1)) {
        seen.emplace(t.slots[i].key, t.slots[i].value);
    }
    EXPECT_EQ(seen, oracle);
}
