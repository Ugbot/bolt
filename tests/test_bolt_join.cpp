// test_bolt_join.cpp — GTest suite for bolt::join (swiss, hashjoin, groupby).

#include <gtest/gtest.h>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/join/bolt_sbbf.h"
#include "bolt/join/bolt_swiss.h"
#include "bolt/join/bolt_hashjoin.h"
#include "bolt/join/bolt_groupby.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace bolt;

// ---------------------------------------------------------------------------
// SwissTable — insert / find round-trip
// ---------------------------------------------------------------------------
TEST(BoltSwiss, InsertFindRoundTrip100) {
    Arena arena;
    SwissTable ht;
    ASSERT_TRUE(SwissTable::create(&ht, 256, &arena));

    for (uint32_t i = 0; i < 100; ++i) {
        uint64_t key = 0x1000ULL + i * 7u;
        ASSERT_TRUE(ht.insert(key, i));
    }
    EXPECT_EQ(ht.size, 100u);

    for (uint32_t i = 0; i < 100; ++i) {
        uint64_t key = 0x1000ULL + i * 7u;
        int32_t v = ht.find(key);
        EXPECT_EQ(v, static_cast<int32_t>(i)) << "key=" << key;
    }

    // Negative lookups.
    EXPECT_EQ(ht.find(0xDEADBEEFCAFEBABEULL), -1);
    EXPECT_EQ(ht.find(0), -1);
}

// ---------------------------------------------------------------------------
// SwissTable — vectorized probe (500 hits, 500 misses)
// ---------------------------------------------------------------------------
TEST(BoltSwiss, FindSimd1000KeysHalfHit) {
    Arena arena;
    SwissTable ht;
    ASSERT_TRUE(SwissTable::create(&ht, 2048, &arena));

    // Build: 500 even keys in [0, 1000).
    for (uint32_t i = 0; i < 500; ++i) {
        uint64_t key = static_cast<uint64_t>(i) * 2ULL + 1000ULL;
        ASSERT_TRUE(ht.insert(key, i));
    }

    // Probe: 1000 keys — even offsets hit, odd offsets miss.
    std::vector<uint64_t> probes(1000);
    for (int i = 0; i < 1000; ++i) {
        probes[i] = 1000ULL + static_cast<uint64_t>(i);
    }
    std::vector<int32_t> out(1000, -7);
    ht.find_simd(probes.data(), 1000, out.data());

    int hits = 0, misses = 0;
    for (int i = 0; i < 1000; ++i) {
        bool should_hit = (i % 2 == 0);
        if (should_hit) {
            EXPECT_EQ(out[i], i / 2) << "i=" << i;
            ++hits;
        } else {
            EXPECT_EQ(out[i], -1) << "i=" << i;
            ++misses;
        }
    }
    EXPECT_EQ(hits, 500);
    EXPECT_EQ(misses, 500);
}

// ---------------------------------------------------------------------------
// HashJoinBuild + Probe — small int64 equi-join
// ---------------------------------------------------------------------------
TEST(BoltHashJoin, SmallInt64Equijoin) {
    Arena arena;

    // Build side: 5 unique int64 keys.
    int64_t build_keys_data[5] = {10, 20, 30, 40, 50};
    BoltColumn build_col = BoltColumn::make_flat(
        build_keys_data, nullptr, 5, BoltType::Int64);

    HashJoinBuild b{};
    ASSERT_TRUE(b.build(build_col, &arena));

    // Probe side: 8 keys — some hit, some miss.
    int64_t probe_keys_data[8] = {20, 15, 30, 50, 99, 10, 40, 60};
    BoltColumn probe_col = BoltColumn::make_flat(
        probe_keys_data, nullptr, 8, BoltType::Int64);

    int32_t out_b[8] = {0};
    int32_t out_p[8] = {0};
    int64_t n = HashJoinProbe::probe(b, probe_col, out_b, out_p);
    ASSERT_EQ(n, 5);

    // Verify every emitted pair: build_keys[out_b[i]] == probe_keys[out_p[i]].
    for (int64_t i = 0; i < n; ++i) {
        ASSERT_GE(out_b[i], 0);
        ASSERT_LT(out_b[i], 5);
        ASSERT_GE(out_p[i], 0);
        ASSERT_LT(out_p[i], 8);
        EXPECT_EQ(build_keys_data[out_b[i]], probe_keys_data[out_p[i]]);
    }
}

// ---------------------------------------------------------------------------
// groupby_sum_int64 — 3 distinct keys
// ---------------------------------------------------------------------------
TEST(BoltGroupBy, SumThreeDistinctKeys) {
    Arena arena;

    // keys:   [1, 2, 1, 3, 2, 1]  -> groups {1:3rows, 2:2rows, 3:1row}
    // values: [10, 20, 30, 40, 50, 60]
    // sums:   {1: 10+30+60=100, 2: 20+50=70, 3: 40}
    int64_t keys_data[6]   = {1, 2, 1, 3, 2, 1};
    int64_t values_data[6] = {10, 20, 30, 40, 50, 60};

    BoltColumn keys_col = BoltColumn::make_flat(
        keys_data, nullptr, 6, BoltType::Int64);
    BoltColumn values_col = BoltColumn::make_flat(
        values_data, nullptr, 6, BoltType::Int64);

    BoltColumn out_keys{}, out_sums{};
    ASSERT_TRUE(groupby_sum_int64(keys_col, values_col, &arena,
                                   &out_keys, &out_sums));
    ASSERT_EQ(out_keys.length, 3);
    ASSERT_EQ(out_sums.length, 3);

    // Unordered check: build a map.
    int64_t* ok = static_cast<int64_t*>(out_keys.data);
    int64_t* os = static_cast<int64_t*>(out_sums.data);
    int64_t got[4] = {0, 0, 0, 0};  // index by key 1..3
    bool seen[4] = {false, false, false, false};
    for (int64_t i = 0; i < 3; ++i) {
        ASSERT_GE(ok[i], 1);
        ASSERT_LE(ok[i], 3);
        got[ok[i]] = os[i];
        seen[ok[i]] = true;
    }
    EXPECT_TRUE(seen[1]); EXPECT_EQ(got[1], 100);
    EXPECT_TRUE(seen[2]); EXPECT_EQ(got[2], 70);
    EXPECT_TRUE(seen[3]); EXPECT_EQ(got[3], 40);
}

// ---------------------------------------------------------------------------
// groupby_sum_int64 — Constant-column fast path
// ---------------------------------------------------------------------------
TEST(BoltGroupBy, ConstantKeyFastPath) {
    Arena arena;

    // All rows belong to group 42. Values are flat.
    int64_t values_data[4] = {1, 2, 3, 4};
    BoltColumn keys_col   = BoltColumn::make_constant<int64_t>(
        42, 4, BoltType::Int64);
    BoltColumn values_col = BoltColumn::make_flat(
        values_data, nullptr, 4, BoltType::Int64);

    BoltColumn out_keys{}, out_sums{};
    ASSERT_TRUE(groupby_sum_int64(keys_col, values_col, &arena,
                                   &out_keys, &out_sums));
    ASSERT_EQ(out_keys.length, 1);
    ASSERT_EQ(out_sums.length, 1);
    EXPECT_EQ(static_cast<int64_t*>(out_keys.data)[0], 42);
    EXPECT_EQ(static_cast<int64_t*>(out_sums.data)[0], 10);
}

// ---------------------------------------------------------------------------
// groupby_sum_int32 — int32 keys, int64 values (mirrors int64 test)
// ---------------------------------------------------------------------------
TEST(BoltGroupBy, SumThreeDistinctKeysInt32) {
    Arena arena;

    int32_t keys_data[6]   = {1, 2, 1, 3, 2, 1};
    int64_t values_data[6] = {10, 20, 30, 40, 50, 60};

    BoltColumn keys_col = BoltColumn::make_flat(
        keys_data, nullptr, 6, BoltType::Int32);
    BoltColumn values_col = BoltColumn::make_flat(
        values_data, nullptr, 6, BoltType::Int64);

    BoltColumn out_keys{}, out_sums{};
    ASSERT_TRUE(groupby_sum_int32(keys_col, values_col, &arena,
                                   &out_keys, &out_sums));
    ASSERT_EQ(out_keys.length, 3);
    ASSERT_EQ(out_sums.length, 3);
    ASSERT_EQ(out_keys.type, BoltType::Int32);
    ASSERT_EQ(out_sums.type, BoltType::Int64);

    const int32_t* ok = static_cast<const int32_t*>(out_keys.data);
    const int64_t* os = static_cast<const int64_t*>(out_sums.data);
    int64_t got[4] = {0, 0, 0, 0};
    bool    seen[4] = {false, false, false, false};
    for (int64_t i = 0; i < 3; ++i) {
        ASSERT_GE(ok[i], 1);
        ASSERT_LE(ok[i], 3);
        got[ok[i]] = os[i];
        seen[ok[i]] = true;
    }
    EXPECT_TRUE(seen[1]); EXPECT_EQ(got[1], 100);
    EXPECT_TRUE(seen[2]); EXPECT_EQ(got[2], 70);
    EXPECT_TRUE(seen[3]); EXPECT_EQ(got[3], 40);
}

// ---------------------------------------------------------------------------
// Branchless probe — edge cases (mirror scalar semantics exactly).
// ---------------------------------------------------------------------------
#include <algorithm>

namespace {
int64_t probe_scalar_ref(const HashJoinBuild& b,
                         const int64_t* keys, int64_t n,
                         int32_t* out_b, int32_t* out_p) {
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        uint64_t k = static_cast<uint64_t>(keys[i]);
        uint64_t h = swiss_mix(k);
        uint32_t p = hj_partition_of(h);
        int32_t  v = b.partitions[p].find(k);
        if (v >= 0) {
            out_b[count] = v;
            out_p[count] = static_cast<int32_t>(i);
            ++count;
        }
    }
    return count;
}
struct Pair { int32_t b; int32_t p; };
inline bool pair_lt(const Pair& a, const Pair& b) { return a.p < b.p; }
}  // namespace

TEST(BoltHashJoin, BranchlessProbeAllHit) {
    Arena arena;
    constexpr int64_t N = 257;  // non-multiple of 8 to exercise tail.
    std::vector<int64_t> bk(N), pk(N);
    for (int64_t i = 0; i < N; ++i) { bk[i] = 1000 + i * 3; pk[i] = bk[i]; }

    BoltColumn bc = BoltColumn::make_flat(bk.data(), nullptr, N, BoltType::Int64);
    BoltColumn pc = BoltColumn::make_flat(pk.data(), nullptr, N, BoltType::Int64);

    HashJoinBuild b{};
    ASSERT_TRUE(b.build(bc, &arena));

    std::vector<int32_t> out_b(N), out_p(N);
    int64_t got = HashJoinProbe::probe(b, pc, out_b.data(), out_p.data());
    ASSERT_EQ(got, N);
    for (int64_t i = 0; i < got; ++i) {
        ASSERT_GE(out_b[i], 0);
        ASSERT_LT(out_b[i], N);
        EXPECT_EQ(bk[out_b[i]], pk[out_p[i]]);
    }
}

TEST(BoltHashJoin, BranchlessProbeAllMiss) {
    Arena arena;
    constexpr int64_t N = 129;
    std::vector<int64_t> bk(N), pk(N);
    for (int64_t i = 0; i < N; ++i) { bk[i] = i * 2; pk[i] = i * 2 + 1; }

    BoltColumn bc = BoltColumn::make_flat(bk.data(), nullptr, N, BoltType::Int64);
    BoltColumn pc = BoltColumn::make_flat(pk.data(), nullptr, N, BoltType::Int64);

    HashJoinBuild b{};
    ASSERT_TRUE(b.build(bc, &arena));

    std::vector<int32_t> out_b(N, 7), out_p(N, 7);
    int64_t got = HashJoinProbe::probe(b, pc, out_b.data(), out_p.data());
    EXPECT_EQ(got, 0);
}

TEST(BoltHashJoin, BranchlessProbeAlternating) {
    Arena arena;
    constexpr int64_t BN = 200;
    constexpr int64_t PN = 400;
    std::vector<int64_t> bk(BN), pk(PN);
    for (int64_t i = 0; i < BN; ++i) bk[i] = 1000 + i;
    for (int64_t i = 0; i < PN; ++i) {
        pk[i] = (i % 2 == 0) ? (1000 + (i / 2)) : (-static_cast<int64_t>(i) - 1);
    }

    BoltColumn bc = BoltColumn::make_flat(bk.data(), nullptr, BN, BoltType::Int64);
    BoltColumn pc = BoltColumn::make_flat(pk.data(), nullptr, PN, BoltType::Int64);

    HashJoinBuild b{};
    ASSERT_TRUE(b.build(bc, &arena));

    std::vector<int32_t> out_b(PN), out_p(PN);
    int64_t got = HashJoinProbe::probe(b, pc, out_b.data(), out_p.data());
    ASSERT_EQ(got, PN / 2);
    for (int64_t i = 0; i < got; ++i) {
        EXPECT_EQ(out_p[i] % 2, 0);
        EXPECT_EQ(bk[out_b[i]], pk[out_p[i]]);
    }
}

// ---------------------------------------------------------------------------
// SwissTable — tight sizing (cardinality-aware factory)
// ---------------------------------------------------------------------------
TEST(BoltSwiss, SwissTableTightSizingExactCapacity) {
    Arena arena;
    SwissTable t;
    ASSERT_TRUE(SwissTable::create_with(&t, 413, &arena, /*tight_sizing=*/true));
    EXPECT_EQ(t.capacity, 512u);  // next pow2 >= 413
    EXPECT_EQ(t.size, 0u);
    EXPECT_EQ(t.mask, 511u);

    // All ctrl bytes (including mirrored tail group) are Empty.
    for (uint32_t i = 0; i < t.capacity + kSwissGroupWidth; ++i) {
        ASSERT_EQ(t.ctrl[i], kSwissCtrlEmpty) << "i=" << i;
    }

    for (uint32_t i = 0; i < 100; ++i) {
        uint64_t key = 0xC000ULL + i * 11u;
        ASSERT_TRUE(t.insert(key, i));
    }
    EXPECT_EQ(t.size, 100u);
    for (uint32_t i = 0; i < 100; ++i) {
        uint64_t key = 0xC000ULL + i * 11u;
        EXPECT_EQ(t.find(key), static_cast<int32_t>(i));
    }
}

TEST(BoltSwiss, SwissTableLooseSizingDefaults) {
    Arena arena;
    SwissTable t;
    ASSERT_TRUE(SwissTable::create(&t, 413, &arena));
    EXPECT_EQ(t.capacity, 1024u);  // next pow2 >= 826
    EXPECT_EQ(t.size, 0u);
    EXPECT_EQ(t.mask, 1023u);

    for (uint32_t i = 0; i < 100; ++i) {
        uint64_t key = 0xD000ULL + i * 13u;
        ASSERT_TRUE(t.insert(key, i));
    }
    EXPECT_EQ(t.size, 100u);
    for (uint32_t i = 0; i < 100; ++i) {
        uint64_t key = 0xD000ULL + i * 13u;
        EXPECT_EQ(t.find(key), static_cast<int32_t>(i));
    }
}

TEST(BoltSwiss, SwissTableTightSizingMatchesLooseFunctionally) {
    Arena arena;
    SwissTable tight, loose;
    ASSERT_TRUE(SwissTable::create_with(&tight, 512, &arena, /*tight_sizing=*/true));
    ASSERT_TRUE(SwissTable::create_with(&loose, 512, &arena, /*tight_sizing=*/false));

    // Generate 200 pseudo-random keys (LCG, deterministic).
    uint64_t s = 0xA3C59AC3B80E7CC1ULL;
    std::vector<uint64_t> keys(200);
    for (int i = 0; i < 200; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        keys[i] = s;
    }

    for (uint32_t i = 0; i < 200; ++i) {
        ASSERT_TRUE(tight.insert(keys[i], i));
        ASSERT_TRUE(loose.insert(keys[i], i));
    }
    EXPECT_EQ(tight.size, 200u);
    EXPECT_EQ(loose.size, 200u);

    for (uint32_t i = 0; i < 200; ++i) {
        int32_t vt = tight.find(keys[i]);
        int32_t vl = loose.find(keys[i]);
        EXPECT_EQ(vt, vl) << "i=" << i;
        EXPECT_EQ(vt, static_cast<int32_t>(i)) << "i=" << i;
    }

    // Negative lookups agree on misses.
    for (uint64_t probe : {0ULL, 1ULL, 2ULL, 0xDEADBEEFULL}) {
        EXPECT_EQ(tight.find(probe), loose.find(probe));
    }
}

TEST(BoltHashJoin, BranchlessProbeLargeInput) {
    Arena arena;
    constexpr int64_t BN = 1024;
    constexpr int64_t PN = 100000;
    std::vector<int64_t> bk(BN), pk(PN);
    for (int64_t i = 0; i < BN; ++i) bk[i] = 10000 + i;

    uint64_t s = 0x9E3779B97F4A7C15ULL;
    int64_t expected_hits = 0;
    for (int64_t i = 0; i < PN; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        bool hit = (s & 1ULL) == 0;
        if (hit) {
            pk[i] = 10000 + static_cast<int64_t>((s >> 1) % BN);
            ++expected_hits;
        } else {
            pk[i] = -1 - static_cast<int64_t>(i);
        }
    }

    BoltColumn bc = BoltColumn::make_flat(bk.data(), nullptr, BN, BoltType::Int64);
    BoltColumn pc = BoltColumn::make_flat(pk.data(), nullptr, PN, BoltType::Int64);

    HashJoinBuild b{};
    ASSERT_TRUE(b.build(bc, &arena));

    std::vector<int32_t> out_b(PN), out_p(PN);
    int64_t got = HashJoinProbe::probe(b, pc, out_b.data(), out_p.data());
    ASSERT_EQ(got, expected_hits);

    std::vector<int32_t> ref_b(PN), ref_p(PN);
    int64_t ref_got = probe_scalar_ref(b, pk.data(), PN, ref_b.data(), ref_p.data());
    ASSERT_EQ(got, ref_got);

    std::vector<Pair> got_pairs(got), ref_pairs(ref_got);
    for (int64_t i = 0; i < got; ++i) got_pairs[i] = {out_b[i], out_p[i]};
    for (int64_t i = 0; i < ref_got; ++i) ref_pairs[i] = {ref_b[i], ref_p[i]};
    std::sort(got_pairs.begin(), got_pairs.end(), pair_lt);
    std::sort(ref_pairs.begin(), ref_pairs.end(), pair_lt);
    for (int64_t i = 0; i < got; ++i) {
        EXPECT_EQ(got_pairs[i].b, ref_pairs[i].b) << "i=" << i;
        EXPECT_EQ(got_pairs[i].p, ref_pairs[i].p) << "i=" << i;
    }
}

// ---------- Bloom-gated probe (C2) ---------------------------------------
// `probe_with_bloom` must return the same pairs as the default `probe`,
// build-side constructed with `build_bloom = true`. Exercises both the
// SIMD block and the scalar tail path via a large mixed-match input.
TEST(BoltHashJoin, BloomGatedProbeMatchesDefault) {
    Arena arena;
    constexpr int64_t BN = 1024;
    constexpr int64_t PN = 100000;
    std::vector<int64_t> bk(BN), pk(PN);
    for (int64_t i = 0; i < BN; ++i) bk[i] = 20000 + i;

    // ~10% match rate — the regime where Bloom should be net-positive.
    uint64_t s = 0x9E3779B97F4A7C15ULL;
    int64_t expected_hits = 0;
    for (int64_t i = 0; i < PN; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        bool hit = ((s % 10u) == 0);
        if (hit) {
            pk[i] = 20000 + static_cast<int64_t>((s >> 3) % BN);
            ++expected_hits;
        } else {
            pk[i] = -1000000 - static_cast<int64_t>(i);
        }
    }

    BoltColumn bc = BoltColumn::make_flat(bk.data(), nullptr, BN, BoltType::Int64);
    BoltColumn pc = BoltColumn::make_flat(pk.data(), nullptr, PN, BoltType::Int64);

    HashJoinBuild b{};
    ASSERT_TRUE(b.build(bc, &arena, /*build_bloom=*/true));
    ASSERT_TRUE(b.has_bloom);

    std::vector<int32_t> out_b(PN), out_p(PN);
    int64_t got = HashJoinProbe::probe_with_bloom(
        b, pc, out_b.data(), out_p.data());
    ASSERT_EQ(got, expected_hits);

    // Cross-check against the non-Bloom path on the same build side.
    std::vector<int32_t> ref_b(PN), ref_p(PN);
    int64_t ref_got = HashJoinProbe::probe(b, pc, ref_b.data(), ref_p.data());
    ASSERT_EQ(got, ref_got);

    std::vector<Pair> got_pairs(got), ref_pairs(ref_got);
    for (int64_t i = 0; i < got; ++i) got_pairs[i] = {out_b[i], out_p[i]};
    for (int64_t i = 0; i < ref_got; ++i) ref_pairs[i] = {ref_b[i], ref_p[i]};
    std::sort(got_pairs.begin(), got_pairs.end(), pair_lt);
    std::sort(ref_pairs.begin(), ref_pairs.end(), pair_lt);
    for (int64_t i = 0; i < got; ++i) {
        EXPECT_EQ(got_pairs[i].b, ref_pairs[i].b) << "i=" << i;
        EXPECT_EQ(got_pairs[i].p, ref_pairs[i].p) << "i=" << i;
    }
}

// Bloom FPR sanity: build with 1000 keys, probe 10 000 never-inserted keys.
// Block-Bloom at 16 bits/key should give FPR well under 5%.
TEST(BoltHashJoin, BloomFprUnderFivePercent) {
    Arena arena;
    constexpr int64_t BN = 1000;
    std::vector<int64_t> bk(BN);
    std::mt19937_64 rng(0xB10F);
    for (int64_t i = 0; i < BN; ++i) bk[i] = static_cast<int64_t>(rng() | 1ULL);

    BoltColumn bc = BoltColumn::make_flat(bk.data(), nullptr, BN, BoltType::Int64);
    HashJoinBuild b{};
    ASSERT_TRUE(b.build(bc, &arena, /*build_bloom=*/true));
    ASSERT_TRUE(b.has_bloom);

    // Probe 10 000 keys that are definitely not in the build side
    // (top bit is 0 here, build had | 1 forcing low bit set only; so use
    // keys with the high bit set AND disjoint from the build distribution).
    constexpr int64_t PN = 10000;
    int64_t false_positives = 0;
    for (int64_t i = 0; i < PN; ++i) {
        uint64_t probe = (static_cast<uint64_t>(rng()) | (1ULL << 63));
        // Must not actually be in the build side; if it is, skip.
        bool in_build = false;
        for (int64_t j = 0; j < BN; ++j) {
            if (static_cast<uint64_t>(bk[j]) == probe) { in_build = true; break; }
        }
        if (in_build) { --i; continue; }
        const uint64_t h = swiss_mix(probe);
        if (bloom_test(b.bloom, h)) ++false_positives;
    }
    // 16 bits/key + 4 hashes per key → expected FPR ≈ 0.4%.
    // Be generous on the assert: under 5% is the contract.
    EXPECT_LT(false_positives * 100 / PN, 5)
        << "FPR too high: " << false_positives << "/" << PN;
}

TEST(BoltHashJoin, BloomNotBuiltByDefault) {
    Arena arena;
    int64_t bk[4] = {1, 2, 3, 4};
    BoltColumn bc = BoltColumn::make_flat(bk, nullptr, 4, BoltType::Int64);

    HashJoinBuild b{};
    ASSERT_TRUE(b.build(bc, &arena));
    EXPECT_FALSE(b.has_bloom);  // default is bloom-off
}

// ---------- SwissTableInterleaved (C3) -----------------------------------
// Build-from-flat + find must return the same value as the source
// SwissTable for every key that was inserted (and -1 for every miss).
TEST(BoltSwiss, InterleavedMatchesFlatFind) {
    Arena arena;
    SwissTable flat;
    ASSERT_TRUE(SwissTable::create(&flat, /*capacity_hint=*/256, &arena));

    // Insert 200 unique keys mapped to sequential values.
    std::mt19937_64 rng(0xC0FFEE);
    std::vector<uint64_t> keys;
    keys.reserve(200);
    for (int i = 0; i < 200; ++i) {
        uint64_t k = rng();
        // Avoid accidental collisions in the test fixture.
        while (flat.find(k) >= 0) k = rng();
        ASSERT_TRUE(flat.insert(k, static_cast<uint32_t>(i)));
        keys.push_back(k);
    }

    SwissTableInterleaved iv{};
    ASSERT_TRUE(SwissTableInterleaved::build_from(&iv, flat, &arena));
    EXPECT_EQ(iv.capacity, flat.capacity);
    EXPECT_EQ(iv.size, flat.size);

    // Hits: every inserted key must find at the same value in both.
    for (int i = 0; i < 200; ++i) {
        int32_t f = flat.find(keys[i]);
        int32_t g = iv.find(keys[i]);
        ASSERT_EQ(f, static_cast<int32_t>(i)) << "flat miss at i=" << i;
        EXPECT_EQ(g, f) << "interleaved mismatch at i=" << i;
    }
    // Misses: random never-inserted keys.
    for (int i = 0; i < 200; ++i) {
        uint64_t miss = rng() | 1ULL;  // perturb; extremely unlikely to collide
        int32_t f = flat.find(miss);
        int32_t g = iv.find(miss);
        EXPECT_EQ(g, f) << "miss disagreement at i=" << i;
    }
}

TEST(BoltSwiss, InterleavedEmptyTable) {
    Arena arena;
    SwissTable flat;
    ASSERT_TRUE(SwissTable::create(&flat, /*capacity_hint=*/16, &arena));
    SwissTableInterleaved iv{};
    ASSERT_TRUE(SwissTableInterleaved::build_from(&iv, flat, &arena));
    EXPECT_EQ(iv.find(42), -1);
    EXPECT_EQ(iv.find(0), -1);
    EXPECT_EQ(iv.size, 0u);
}

// High-cardinality stress — 10 000 unique keys exercise multi-group
// linear-probe chains on both FLAT and INTERLEAVED.  Also ensures the
// rebuild path copes when the interleaved table has capacity equal
// to the flat source (no growth; just reinsert).
TEST(BoltSwiss, InterleavedHighCardinalityMatchesFlat) {
    Arena arena;
    SwissTable flat;
    ASSERT_TRUE(SwissTable::create(&flat, /*capacity_hint=*/10000, &arena));

    std::mt19937_64 rng(0xCAFEB0BA);
    std::vector<uint64_t> keys;
    keys.reserve(10000);
    for (int i = 0; i < 10000; ++i) {
        uint64_t k = rng();
        while (flat.find(k) >= 0) k = rng();
        ASSERT_TRUE(flat.insert(k, static_cast<uint32_t>(i)));
        keys.push_back(k);
    }

    SwissTableInterleaved iv{};
    ASSERT_TRUE(SwissTableInterleaved::build_from(&iv, flat, &arena));
    EXPECT_EQ(iv.size, flat.size);

    // Shuffle lookup order so prefetch + locality aren't accidentally aligned.
    std::shuffle(keys.begin(), keys.end(), rng);
    for (auto k : keys) {
        int32_t g = iv.find(k);
        ASSERT_GE(g, 0) << "missing key " << k;
        EXPECT_EQ(flat.find(k), g);
    }
}

// ---------- Split Block Bloom (C2b / Parquet-spec) -----------------------
// Zero false-negatives: every key ever added must test true.
TEST(BoltSbbf, NoFalseNegatives) {
    Arena arena;
    SplitBlockBloom sbf{};
    ASSERT_TRUE(sbbf_create(&sbf, /*expected_n=*/1000, &arena));

    std::mt19937_64 rng(0xB10C);
    std::vector<uint64_t> keys;
    keys.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        uint64_t k = rng();
        sbbf_add(sbf, swiss_mix(k));
        keys.push_back(k);
    }
    for (uint64_t k : keys) {
        EXPECT_TRUE(sbbf_test(sbf, swiss_mix(k))) << "false negative for " << k;
    }
}

// FPR sanity: Parquet SBBF targets ~1% at 8 bits/key. Budget 2% to stay
// well inside the tolerance band on 10 000 never-inserted probes.
TEST(BoltSbbf, FprUnderTwoPercent) {
    Arena arena;
    SplitBlockBloom sbf{};
    ASSERT_TRUE(sbbf_create(&sbf, /*expected_n=*/1000, &arena));

    std::mt19937_64 rng(0x5BBF);
    std::vector<uint64_t> keys;
    keys.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        uint64_t k = rng() | 1ULL;  // low bit set — distinguishes from probes below
        sbbf_add(sbf, swiss_mix(k));
        keys.push_back(k);
    }

    constexpr int PN = 10000;
    int fps = 0;
    for (int i = 0; i < PN; ++i) {
        uint64_t probe = (rng() & ~1ULL);    // low bit clear — never in build
        if (sbbf_test(sbf, swiss_mix(probe))) ++fps;
    }
    EXPECT_LT(fps * 100 / PN, 2) << "SBBF FPR too high: " << fps << "/" << PN;
}

TEST(BoltSbbf, EmptyFilterAlwaysAbsent) {
    Arena arena;
    SplitBlockBloom sbf{};
    ASSERT_TRUE(sbbf_create(&sbf, /*expected_n=*/64, &arena));
    for (uint64_t k : {1ULL, 42ULL, 0xDEADBEEFULL, ~uint64_t{0}}) {
        EXPECT_FALSE(sbbf_test(sbf, swiss_mix(k)));
    }
}

TEST(BoltSbbf, SizingAtTenBitsPerKey) {
    Arena arena;
    SplitBlockBloom sbf{};
    ASSERT_TRUE(sbbf_create(&sbf, /*expected_n=*/1024, &arena));
    // 1024 keys × 10 bits/key = 10240 bits = 40 blocks → round up to 64
    // blocks (next pow2) × 32 bytes = 2048 bytes.
    EXPECT_GE(sbbf_size_bytes(sbf), 2048u);
    EXPECT_LE(sbbf_size_bytes(sbf), 4096u);
}

// Collision-heavy stress — hash_mix is deterministic, but we can force
// a crowded group by hand-picking keys whose `swiss_mix` falls in the
// same 16-wide group. Drives the linear-probe chain past the initial
// group into neighbours. Verifies INTERLEAVED's group-aligned probe
// handles overflow into subsequent groups.
TEST(BoltSwiss, InterleavedLinearProbeSpillover) {
    Arena arena;
    SwissTableInterleaved iv{};
    // Small table so collisions are dense. 16 groups × 16 lanes = 256 slots.
    ASSERT_TRUE(SwissTableInterleaved::create(&iv, /*hint=*/120, &arena));

    // Insert 100 distinct keys; at 50% load factor collisions are frequent.
    std::mt19937_64 rng(0x1234);
    std::vector<uint64_t> keys;
    keys.reserve(100);
    for (int i = 0; i < 100; ++i) {
        uint64_t k = rng();
        while (iv.find(k) >= 0) k = rng();
        ASSERT_TRUE(iv.insert(k, static_cast<uint32_t>(i)));
        keys.push_back(k);
    }
    EXPECT_EQ(iv.size, 100u);

    // All inserted keys must still be findable.
    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_EQ(iv.find(keys[i]), static_cast<int32_t>(i))
            << "lost key at i=" << i;
    }

    // Update-on-existing-key semantics: reinserting flips value without
    // growing size.
    const uint32_t old_size = iv.size;
    ASSERT_TRUE(iv.insert(keys[42], /*new value=*/9999));
    EXPECT_EQ(iv.size, old_size);
    EXPECT_EQ(iv.find(keys[42]), 9999);
}

// ---------------------------------------------------------------------------
// SwissTableSalted (G2FEAT-89/362) — salt-in-entry probe with inline payload.
// ---------------------------------------------------------------------------

TEST(BoltSwissSalted, FindOrInsertRoundTrip) {
    Arena arena;
    SwissTableSalted t{};
    ASSERT_TRUE(SwissTableSalted::create(&t, 256, &arena));

    for (uint32_t i = 0; i < 100; ++i) {
        uint64_t key = 0x1000ULL + i * 7u;
        int64_t* slot = t.find_or_insert(key, static_cast<int64_t>(i));
        ASSERT_NE(slot, nullptr);
        EXPECT_EQ(*slot, static_cast<int64_t>(i));
    }
    EXPECT_EQ(t.size, 100u);

    for (uint32_t i = 0; i < 100; ++i) {
        uint64_t key = 0x1000ULL + i * 7u;
        int64_t* slot = t.find(key);
        ASSERT_NE(slot, nullptr) << "key=" << key;
        EXPECT_EQ(*slot, static_cast<int64_t>(i));
    }

    // Negative lookups.
    EXPECT_EQ(t.find(0xDEADBEEFCAFEBABEULL), nullptr);
    EXPECT_EQ(t.find(0), nullptr);
}

// Repeated find_or_insert on the same key must return the SAME cell (not a
// fresh one) so `*slot += x` accumulates correctly — the whole point of the
// co-located layout.
TEST(BoltSwissSalted, AccumulateInPlaceMatchesGroundTruth) {
    Arena arena;
    SwissTableSalted t{};
    ASSERT_TRUE(SwissTableSalted::create(&t, 64, &arena));

    constexpr int kKeys = 20;
    constexpr int kRows = 5000;
    std::vector<int64_t> expected(kKeys, 0);

    std::mt19937_64 rng(0x5EED);
    for (int i = 0; i < kRows; ++i) {
        const uint64_t k = static_cast<uint64_t>(rng() % kKeys);
        const int64_t  v = static_cast<int64_t>((rng() % 97) + 1);
        int64_t* slot = t.find_or_insert(k, 0);
        ASSERT_NE(slot, nullptr);
        *slot += v;
        expected[static_cast<size_t>(k)] += v;
    }

    ASSERT_EQ(t.size, static_cast<uint64_t>(kKeys));
    for (int k = 0; k < kKeys; ++k) {
        int64_t* slot = t.find(static_cast<uint64_t>(k));
        ASSERT_NE(slot, nullptr) << "key=" << k;
        EXPECT_EQ(*slot, expected[static_cast<size_t>(k)]) << "key=" << k;
    }
}

// A table sized far too small for the workload must fail closed (bounded
// probe returns nullptr) rather than loop forever or silently corrupt state
// — Tiger Style: no growth on the hot path.
TEST(BoltSwissSalted, FullTableReturnsNullNeverLoopsForever) {
    Arena arena;
    SwissTableSalted t{};
    // create() rounds up to a power of two with ~66% target load; force a
    // tiny table so it's easy to genuinely fill.
    ASSERT_TRUE(SwissTableSalted::create(&t, 1, &arena));
    const uint64_t cap = t.capacity;

    uint32_t inserted = 0;
    for (uint64_t k = 0; k < cap * 4; ++k) {
        int64_t* slot = t.find_or_insert(k, static_cast<int64_t>(k));
        if (slot == nullptr) break;
        ++inserted;
    }
    // Must have stopped at or before full capacity — never silently grown
    // past it, never crashed, never spun.
    EXPECT_LE(inserted, cap);
    EXPECT_GT(inserted, 0u);
    EXPECT_LE(t.size, t.capacity);
}

// find_or_insert_batch (windowed-prefetch variant) must return byte-identical
// slot pointers / final values to the plain scalar find_or_insert loop over
// the SAME key stream — the prefetch window is purely a latency-hiding
// reordering trick and must never change which entry a key resolves to.
TEST(BoltSwissSalted, BatchWindowedPrefetchMatchesScalar) {
    constexpr int64_t kRows = 200'000;
    constexpr uint64_t kDistinct = 4000;

    std::vector<uint64_t> keys(static_cast<size_t>(kRows));
    std::mt19937_64 rng(0xC0FFEE);
    for (int64_t i = 0; i < kRows; ++i) {
        keys[static_cast<size_t>(i)] = rng() % kDistinct;
    }

    // Scalar reference.
    Arena arena_scalar;
    SwissTableSalted t_scalar{};
    ASSERT_TRUE(SwissTableSalted::create(&t_scalar, kDistinct, &arena_scalar));
    for (int64_t i = 0; i < kRows; ++i) {
        int64_t* slot = t_scalar.find_or_insert(keys[static_cast<size_t>(i)], 0);
        ASSERT_NE(slot, nullptr);
        *slot += 1;
    }

    // Batched, windowed-prefetch path over the identical key stream.
    Arena arena_batch;
    SwissTableSalted t_batch{};
    ASSERT_TRUE(SwissTableSalted::create(&t_batch, kDistinct, &arena_batch));
    std::vector<int64_t*> out_slots(static_cast<size_t>(kRows));
    t_batch.find_or_insert_batch(keys.data(), kRows, 0, out_slots.data());
    for (int64_t i = 0; i < kRows; ++i) {
        ASSERT_NE(out_slots[static_cast<size_t>(i)], nullptr);
        *out_slots[static_cast<size_t>(i)] += 1;
    }

    ASSERT_EQ(t_scalar.size, t_batch.size);
    ASSERT_EQ(t_scalar.size, kDistinct);
    for (uint64_t k = 0; k < kDistinct; ++k) {
        int64_t* a = t_scalar.find(k);
        int64_t* b = t_batch.find(k);
        ASSERT_NE(a, nullptr) << "key=" << k;
        ASSERT_NE(b, nullptr) << "key=" << k;
        EXPECT_EQ(*a, *b) << "key=" << k;
    }
}

// Cross-check against the REAL, unmodified bolt::SwissTable + a separate
// accumulator array (the layout SwissTableSalted exists to beat) — both
// must compute the identical (distinct groups, total) over the same input.
// This is the correctness half of the go/no-go: a faster wrong answer is
// not a win.
TEST(BoltSwissSalted, MatchesFlatSwissTablePlusSeparateArray) {
    constexpr int64_t kRows = 100'000;
    constexpr uint64_t kDistinct = 3000;

    std::vector<uint64_t> keys(static_cast<size_t>(kRows));
    std::mt19937_64 rng(0xA11CE);
    for (int64_t i = 0; i < kRows; ++i) {
        keys[static_cast<size_t>(i)] = rng() % kDistinct;
    }

    // Reference: FLAT SwissTable + separate accum[] array.
    Arena arena_flat;
    SwissTable flat{};
    ASSERT_TRUE(SwissTable::create(&flat, kDistinct, &arena_flat));
    std::vector<int64_t> accum(flat.capacity, 0);
    int32_t next_slot = 0;
    for (int64_t i = 0; i < kRows; ++i) {
        const uint64_t k = keys[static_cast<size_t>(i)];
        int32_t slot = flat.find(k);
        if (slot < 0) {
            slot = next_slot++;
            ASSERT_TRUE(flat.insert(k, static_cast<uint32_t>(slot)));
        }
        accum[static_cast<size_t>(slot)]++;
    }
    int64_t flat_total = 0;
    for (int32_t s = 0; s < next_slot; ++s) flat_total += accum[static_cast<size_t>(s)];

    // Salted table over the identical stream.
    Arena arena_salt;
    SwissTableSalted salted{};
    ASSERT_TRUE(SwissTableSalted::create(&salted, kDistinct, &arena_salt));
    for (int64_t i = 0; i < kRows; ++i) {
        int64_t* slot = salted.find_or_insert(keys[static_cast<size_t>(i)], 0);
        ASSERT_NE(slot, nullptr);
        (*slot)++;
    }
    int64_t salted_total = 0;
    for (uint64_t s = 0; s < salted.capacity; ++s) {
        salted_total += salted.entries[s].value;
    }

    EXPECT_EQ(static_cast<uint32_t>(next_slot), static_cast<uint32_t>(salted.size));
    EXPECT_EQ(flat_total, kRows);
    EXPECT_EQ(salted_total, kRows);
    EXPECT_EQ(flat_total, salted_total);
}
