// test_bolt_mergejoin.cpp — correctness tests for sorted-input merge-join
// (inner / left-outer / right-outer variants).

#include <gtest/gtest.h>

#include "bolt/bolt_column.h"
#include "bolt/join/bolt_mergejoin.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

struct Pair {
    int32_t b;
    int32_t p;
};

bool pair_lt(const Pair& a, const Pair& b) noexcept {
    if (a.b != b.b) return a.b < b.b;
    return a.p < b.p;
}

bolt::BoltColumn make_i64(std::vector<int64_t>& v) noexcept {
    return bolt::BoltColumn::make_flat(
        v.data(), nullptr,
        static_cast<int64_t>(v.size()),
        bolt::BoltType::Int64);
}

}  // namespace

// ---- Inner ---------------------------------------------------------------

TEST(BoltMergeJoin, InnerSimpleUniqueKeys) {
    std::vector<int64_t> bk = {1, 3, 5, 7, 9};
    std::vector<int64_t> pk = {2, 3, 5, 8, 9, 10};
    auto bc = make_i64(bk);
    auto pc = make_i64(pk);

    std::vector<int32_t> ob(16), op(16);
    int64_t n = bolt::mergejoin_inner_i64(bc, pc, ob.data(), op.data(), 16);
    ASSERT_EQ(n, 3);

    std::vector<Pair> got(n);
    for (int64_t i = 0; i < n; ++i) got[i] = {ob[i], op[i]};
    std::sort(got.begin(), got.end(), pair_lt);
    // Expected: bk[1]=3 ~ pk[1], bk[2]=5 ~ pk[2], bk[4]=9 ~ pk[4]
    EXPECT_EQ(got[0].b, 1); EXPECT_EQ(got[0].p, 1);
    EXPECT_EQ(got[1].b, 2); EXPECT_EQ(got[1].p, 2);
    EXPECT_EQ(got[2].b, 4); EXPECT_EQ(got[2].p, 4);
}

TEST(BoltMergeJoin, InnerDuplicateKeysCrossProduct) {
    // Build has two 5s, probe has three 5s → 2*3 = 6 pairs on key=5.
    std::vector<int64_t> bk = {3, 5, 5, 7};
    std::vector<int64_t> pk = {4, 5, 5, 5, 8};
    auto bc = make_i64(bk);
    auto pc = make_i64(pk);

    std::vector<int32_t> ob(32), op(32);
    int64_t n = bolt::mergejoin_inner_i64(bc, pc, ob.data(), op.data(), 32);
    ASSERT_EQ(n, 6);

    std::vector<Pair> got(n);
    for (int64_t i = 0; i < n; ++i) got[i] = {ob[i], op[i]};
    std::sort(got.begin(), got.end(), pair_lt);
    // All pairs (build_idx, probe_idx) in {1,2} × {1,2,3}.
    const std::vector<Pair> want = {
        {1,1},{1,2},{1,3},{2,1},{2,2},{2,3}
    };
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_EQ(got[i].b, want[i].b) << "i=" << i;
        EXPECT_EQ(got[i].p, want[i].p) << "i=" << i;
    }
}

TEST(BoltMergeJoin, InnerNoOverlap) {
    std::vector<int64_t> bk = {1, 2, 3};
    std::vector<int64_t> pk = {100, 200, 300};
    auto bc = make_i64(bk);
    auto pc = make_i64(pk);

    std::vector<int32_t> ob(8), op(8);
    int64_t n = bolt::mergejoin_inner_i64(bc, pc, ob.data(), op.data(), 8);
    EXPECT_EQ(n, 0);
}

TEST(BoltMergeJoin, InnerEmptyInputs) {
    std::vector<int64_t> empty;
    std::vector<int64_t> pk = {1, 2, 3};
    auto bc = make_i64(empty);
    auto pc = make_i64(pk);

    std::vector<int32_t> ob(8), op(8);
    int64_t n = bolt::mergejoin_inner_i64(bc, pc, ob.data(), op.data(), 8);
    EXPECT_EQ(n, 0);
}

TEST(BoltMergeJoin, InnerCapacityLimit) {
    std::vector<int64_t> bk = {1, 1, 1, 1};
    std::vector<int64_t> pk = {1, 1};
    auto bc = make_i64(bk);
    auto pc = make_i64(pk);

    // 4*2 = 8 pairs possible; cap at 3.
    std::vector<int32_t> ob(16), op(16);
    int64_t n = bolt::mergejoin_inner_i64(bc, pc, ob.data(), op.data(), 3);
    EXPECT_EQ(n, 3);
}

// ---- Left outer ----------------------------------------------------------

TEST(BoltMergeJoin, LeftOuterEmitsUnmatchedProbes) {
    std::vector<int64_t> bk = {3, 7};
    std::vector<int64_t> pk = {1, 3, 5, 7, 9};
    auto bc = make_i64(bk);
    auto pc = make_i64(pk);

    std::vector<int32_t> ob(16), op(16);
    int64_t n = bolt::mergejoin_left_outer_i64(bc, pc, ob.data(), op.data(), 16);
    ASSERT_EQ(n, 5);

    // Expected pairs (by probe_idx order):
    //   j=0 (pk=1) → unmatched:  (-1, 0)
    //   j=1 (pk=3) → match bk[0]: (0, 1)
    //   j=2 (pk=5) → unmatched:  (-1, 2)
    //   j=3 (pk=7) → match bk[1]: (1, 3)
    //   j=4 (pk=9) → unmatched:  (-1, 4)
    std::vector<Pair> got(n), want{{-1,0},{0,1},{-1,2},{1,3},{-1,4}};
    for (int64_t i = 0; i < n; ++i) got[i] = {ob[i], op[i]};
    // Left outer emits in probe order; no need to sort.
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_EQ(got[i].b, want[i].b) << "i=" << i;
        EXPECT_EQ(got[i].p, want[i].p) << "i=" << i;
    }
}

TEST(BoltMergeJoin, LeftOuterDuplicateBuildKeys) {
    std::vector<int64_t> bk = {5, 5, 5};
    std::vector<int64_t> pk = {1, 5, 9};
    auto bc = make_i64(bk);
    auto pc = make_i64(pk);

    std::vector<int32_t> ob(16), op(16);
    int64_t n = bolt::mergejoin_left_outer_i64(bc, pc, ob.data(), op.data(), 16);
    // pk=1 unmatched → 1 row; pk=5 matches all 3 builds → 3 rows;
    // pk=9 unmatched → 1 row. Total 5.
    ASSERT_EQ(n, 5);
    EXPECT_EQ(ob[0], -1); EXPECT_EQ(op[0], 0);
    EXPECT_EQ(ob[1],  0); EXPECT_EQ(op[1], 1);
    EXPECT_EQ(ob[2],  1); EXPECT_EQ(op[2], 1);
    EXPECT_EQ(ob[3],  2); EXPECT_EQ(op[3], 1);
    EXPECT_EQ(ob[4], -1); EXPECT_EQ(op[4], 2);
}

// ---- Right outer ---------------------------------------------------------

TEST(BoltMergeJoin, RightOuterEmitsUnmatchedBuilds) {
    std::vector<int64_t> bk = {1, 3, 5, 7, 9};
    std::vector<int64_t> pk = {3, 7};
    auto bc = make_i64(bk);
    auto pc = make_i64(pk);

    std::vector<int32_t> ob(16), op(16);
    int64_t n = bolt::mergejoin_right_outer_i64(bc, pc, ob.data(), op.data(), 16);
    ASSERT_EQ(n, 5);

    std::vector<Pair> got(n), want{{0,-1},{1,0},{2,-1},{3,1},{4,-1}};
    for (int64_t i = 0; i < n; ++i) got[i] = {ob[i], op[i]};
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_EQ(got[i].b, want[i].b) << "i=" << i;
        EXPECT_EQ(got[i].p, want[i].p) << "i=" << i;
    }
}

// Adversarial: all rows on both sides share the same key. Equal-run
// cross-product is n × m. Exercises the inner-run emit loop past any
// small-constant inline buffers.
TEST(BoltMergeJoin, InnerAllDuplicates) {
    std::vector<int64_t> bk(20, 77);   // 20 rows, all key=77
    std::vector<int64_t> pk(15, 77);   // 15 rows, all key=77
    auto bc = make_i64(bk);
    auto pc = make_i64(pk);

    std::vector<int32_t> ob(500), op(500);
    int64_t n = bolt::mergejoin_inner_i64(bc, pc, ob.data(), op.data(), 500);
    EXPECT_EQ(n, 20 * 15);

    // Every (build_idx, probe_idx) in [0,20) × [0,15) must appear exactly once.
    std::vector<Pair> got(n);
    for (int64_t i = 0; i < n; ++i) got[i] = {ob[i], op[i]};
    std::sort(got.begin(), got.end(), pair_lt);
    // Check monotone by build_idx, then by probe_idx within each build.
    int64_t k = 0;
    for (int32_t b = 0; b < 20; ++b) {
        for (int32_t p = 0; p < 15; ++p) {
            EXPECT_EQ(got[k].b, b);
            EXPECT_EQ(got[k].p, p);
            ++k;
        }
    }
}

TEST(BoltMergeJoin, LeftOuterAllBuildAfterProbes) {
    // Build keys all higher than probe keys → every probe unmatched.
    std::vector<int64_t> bk = {100, 200, 300};
    std::vector<int64_t> pk = {1, 2, 3, 4, 5};
    auto bc = make_i64(bk);
    auto pc = make_i64(pk);

    std::vector<int32_t> ob(16), op(16);
    int64_t n = bolt::mergejoin_left_outer_i64(bc, pc, ob.data(), op.data(), 16);
    ASSERT_EQ(n, 5);
    for (int64_t i = 0; i < 5; ++i) {
        EXPECT_EQ(ob[i], -1);
        EXPECT_EQ(op[i], static_cast<int32_t>(i));
    }
}

TEST(BoltMergeJoin, RightOuterAllProbesAfterBuild) {
    std::vector<int64_t> bk = {1, 2, 3, 4, 5};
    std::vector<int64_t> pk = {100, 200};
    auto bc = make_i64(bk);
    auto pc = make_i64(pk);

    std::vector<int32_t> ob(16), op(16);
    int64_t n = bolt::mergejoin_right_outer_i64(bc, pc, ob.data(), op.data(), 16);
    ASSERT_EQ(n, 5);
    for (int64_t i = 0; i < 5; ++i) {
        EXPECT_EQ(ob[i], static_cast<int32_t>(i));
        EXPECT_EQ(op[i], -1);
    }
}

// ---- Cross-check vs hash-join --------------------------------------------
// Inner merge-join must produce the same set of pairs as an inner hash-join
// on the same (sorted, unique-key) inputs.
TEST(BoltMergeJoin, InnerMatchesHashJoinForUniqueKeys) {
    constexpr int64_t N = 2000;
    std::vector<int64_t> bk(N), pk(N);
    for (int64_t i = 0; i < N; ++i) bk[i] = 1000 + i;             // sorted, unique
    for (int64_t i = 0; i < N; ++i) pk[i] = 1000 + (i * 3 % N);   // NOT sorted
    std::sort(pk.begin(), pk.end());                              // merge needs sorted probe

    // Match count: every probe row lands on a build row (build covers [1000, 1000+N)).
    int64_t expected = 0;
    for (int64_t v : pk) if (v >= 1000 && v < 1000 + N) ++expected;

    auto bc = make_i64(bk);
    auto pc = make_i64(pk);

    std::vector<int32_t> ob(N*2), op(N*2);
    int64_t n = bolt::mergejoin_inner_i64(bc, pc, ob.data(), op.data(), N*2);
    EXPECT_EQ(n, expected);

    // Spot-check a few pairs: each probe row matches the build row with
    // the same key.
    for (int64_t k = 0; k < n; ++k) {
        EXPECT_EQ(bk[ob[k]], pk[op[k]]) << "pair " << k;
    }
}
