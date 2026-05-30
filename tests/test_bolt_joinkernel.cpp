// Tests for bolt::join_build_typed / join_probe_typed / join_drain_build
// (bolt/join/bolt_joinkernel.h) — typed, all-shapes SwissTable join.
//
// Scope: kernel-output correctness on small typed inputs — inner single &
// multi-match, composite keys, Int32 keys, SEMI/ANTI/LEFT-OUTER drain via the
// matched bitmap, NULL-key non-matching, empty sides, and a brute-force
// O(n*m) inner cross-check.

#include "bolt/join/bolt_joinkernel.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

namespace {

using namespace bolt;

// Simulate the chukonu wrapper's no-residual matched-bit pass: every emitted
// pair counts as a match.
void mark_all(JoinBuildTyped* b, const int32_t* out_build, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) b->matched[out_build[i]] = 1;
}

TEST(BoltJoinKernel, InnerSingleInt64) {
    Arena a;
    int64_t bk[] = {1, 2, 3};
    int64_t pk[] = {2, 3, 4, 2};       // 2 matches twice, 3 once, 4 none
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, 3, BoltType::Int64);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, 4, BoltType::Int64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, 3, &a, &jb));
    int32_t ob[16], op[16];
    const size_t n = join_probe_typed(&jb, &p, 4, ob, op, 16);
    EXPECT_EQ(n, 3u);                  // (b=1,p=0),(b=2,p=1),(b=1,p=3)
    // Verify each pair's keys are equal.
    for (size_t i = 0; i < n; ++i) EXPECT_EQ(bk[ob[i]], pk[op[i]]);
}

TEST(BoltJoinKernel, InnerMultiMatch) {
    Arena a;
    int64_t bk[] = {7, 7, 7, 9};       // key 7 thrice
    int64_t pk[] = {7};
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, 4, BoltType::Int64);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, 1, BoltType::Int64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, 4, &a, &jb));
    int32_t ob[16], op[16];
    const size_t n = join_probe_typed(&jb, &p, 1, ob, op, 16);
    EXPECT_EQ(n, 3u);                  // one probe row -> 3 build matches
    for (size_t i = 0; i < n; ++i) { EXPECT_EQ(bk[ob[i]], 7); EXPECT_EQ(op[i], 0); }
}

TEST(BoltJoinKernel, CompositeTwoKey) {
    Arena a;
    int32_t b0[] = {1, 1, 2};
    int64_t b1[] = {10, 20, 10};
    int32_t p0[] = {1, 2};
    int64_t p1[] = {20, 10};
    BoltColumn bcols[2] = {BoltColumn::make_flat(b0, nullptr, 3, BoltType::Int32),
                           BoltColumn::make_flat(b1, nullptr, 3, BoltType::Int64)};
    BoltColumn pcols[2] = {BoltColumn::make_flat(p0, nullptr, 2, BoltType::Int32),
                           BoltColumn::make_flat(p1, nullptr, 2, BoltType::Int64)};
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(bcols, 2, 3, &a, &jb));
    int32_t ob[16], op[16];
    const size_t n = join_probe_typed(&jb, pcols, 2, ob, op, 16);
    EXPECT_EQ(n, 2u);                  // (1,20)->b1 ; (2,10)->b2
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(b0[ob[i]], p0[op[i]]);
        EXPECT_EQ(b1[ob[i]], p1[op[i]]);
    }
}

TEST(BoltJoinKernel, SemiAndAntiDrain) {
    Arena a;
    int64_t bk[] = {1, 2, 3};
    int64_t pk[] = {2};
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, 3, BoltType::Int64);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, 1, BoltType::Int64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, 3, &a, &jb));
    int32_t ob[16], op[16];
    const size_t n = join_probe_typed(&jb, &p, 1, ob, op, 16);
    mark_all(&jb, ob, n);              // wrapper marks matched build rows
    // SEMI: emit matched build rows once.
    uint32_t cur = 0; int32_t sb[16], sp[16];
    const size_t ns = join_drain_build(&jb, /*want_matched=*/1, &cur, sb, sp, 16);
    EXPECT_EQ(ns, 1u); EXPECT_EQ(bk[sb[0]], 2); EXPECT_EQ(sp[0], kJoinNullIndex);
    // ANTI: emit unmatched build rows once.
    cur = 0; int32_t ab[16], ap[16];
    const size_t na = join_drain_build(&jb, /*want_matched=*/0, &cur, ab, ap, 16);
    EXPECT_EQ(na, 2u);                 // keys 1 and 3
    EXPECT_EQ(ap[0], kJoinNullIndex);
}

TEST(BoltJoinKernel, LeftOuterUnmatchedDrain) {
    Arena a;
    int64_t bk[] = {1, 2, 3};
    int64_t pk[] = {2, 2};             // only key 2 matches (twice)
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, 3, BoltType::Int64);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, 2, BoltType::Int64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, 3, &a, &jb));
    int32_t ob[16], op[16];
    const size_t n = join_probe_typed(&jb, &p, 2, ob, op, 16);
    EXPECT_EQ(n, 2u);                  // (b=2,p=0),(b=2,p=1)
    mark_all(&jb, ob, n);
    uint32_t cur = 0; int32_t ub[16], up[16];
    const size_t nu = join_drain_build(&jb, /*want_matched=*/0, &cur, ub, up, 16);
    EXPECT_EQ(nu, 2u);                 // unmatched build rows 1,3 with NULL probe
    for (size_t i = 0; i < nu; ++i) EXPECT_EQ(up[i], kJoinNullIndex);
}

TEST(BoltJoinKernel, NullKeysDoNotMatch) {
    Arena a;
    int64_t bk[] = {1, 2, 3};
    uint8_t bvalid = 0b101;            // build row 1 (key 2) is NULL
    int64_t pk[] = {1, 2, 3};
    uint8_t pvalid = 0b110;            // probe row 0 (key 1) is NULL
    BoltColumn b = BoltColumn::make_flat(bk, &bvalid, 3, BoltType::Int64);
    BoltColumn p = BoltColumn::make_flat(pk, &pvalid, 3, BoltType::Int64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, 3, &a, &jb));
    int32_t ob[16], op[16];
    const size_t n = join_probe_typed(&jb, &p, 3, ob, op, 16);
    // probe key1 NULL (skip), key2 matches a NULL build row (no match), key3 ok.
    EXPECT_EQ(n, 1u);
    EXPECT_EQ(bk[ob[0]], 3); EXPECT_EQ(pk[op[0]], 3);
}

TEST(BoltJoinKernel, EmptySides) {
    Arena a;
    int64_t bk[] = {1};
    int64_t pk[] = {1};
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, 0, BoltType::Int64);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, 0, BoltType::Int64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, 0, &a, &jb));
    int32_t ob[4], op[4];
    EXPECT_EQ(join_probe_typed(&jb, &p, 0, ob, op, 4), 0u);
    // empty build, non-empty probe -> no matches.
    BoltColumn p2 = BoltColumn::make_flat(pk, nullptr, 1, BoltType::Int64);
    EXPECT_EQ(join_probe_typed(&jb, &p2, 1, ob, op, 4), 0u);
}

TEST(BoltJoinKernel, BruteForceInnerCrossCheck) {
    Arena a;
    constexpr int NB = 200, NP = 300;
    int64_t bk[NB], pk[NP];
    for (int i = 0; i < NB; ++i) bk[i] = (i * 7) % 23;   // dup keys
    for (int i = 0; i < NP; ++i) pk[i] = (i * 3) % 29;
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, NB, BoltType::Int64);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, NP, BoltType::Int64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, NB, &a, &jb));
    int32_t ob[NB * NP], op[NB * NP];
    const size_t n = join_probe_typed(&jb, &p, NP, ob, op, NB * NP);
    size_t expect = 0;
    for (int r = 0; r < NP; ++r)
        for (int s = 0; s < NB; ++s)
            if (pk[r] == bk[s]) ++expect;
    EXPECT_EQ(n, expect);
    for (size_t i = 0; i < n; ++i) EXPECT_EQ(bk[ob[i]], pk[op[i]]);
}

}  // namespace
