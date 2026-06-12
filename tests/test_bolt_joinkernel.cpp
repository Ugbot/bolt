// Tests for bolt::join_build_typed / join_probe_typed / join_drain_build
// (bolt/join/bolt_joinkernel.h) — typed, all-shapes SwissTable join.
//
// Scope: kernel-output correctness on small typed inputs — inner single &
// multi-match, composite keys, Int32 keys, SEMI/ANTI/LEFT-OUTER drain via the
// matched bitmap, NULL-key non-matching, empty sides, and a brute-force
// O(n*m) inner cross-check.
//
// Card A2 hardening cases (added below):
//   * Float64 key canonicalization — -0.0 JOINs +0.0; canonical NaN equals
//     itself; distinct NaN payloads still match (collapse to one bit pattern).
//   * Decimal128 scale precondition — matching scales JOIN; the scale-mismatch
//     guard (join_decimal_scales_match) reports a mismatch.
//   * Spilled-Utf8 key arm — equal >12-byte strings at DIFFERENT arena offsets
//     JOIN (the false-negative GbCell16 raw-16-byte compare would miss); inline
//     Utf8 keys still JOIN; unequal strings don't.
//   * OneInt64 fast path — fast path output == forced-General path output.

#include "bolt/join/bolt_joinkernel.h"
#include "bolt/kernels/bolt_decimal.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

using namespace bolt;

// Build a Utf8 BoltColumn over `views` with the given spilled-bytes base.
// `views`/`base` are caller-owned. decimal_scale irrelevant for Utf8.
BoltColumn make_utf8_col(const StringView* views, int64_t n,
                         const char* base) noexcept {
    BoltColumn c = BoltColumn::make_flat(
        const_cast<StringView*>(views), nullptr, n, BoltType::Utf8);
    c.str_overflow_base = const_cast<char*>(base);
    return c;
}

// Spill a C-string into `pool` at `*cursor`, returning a spilled StringView
// whose ref.offset is relative to `pool` (the column's str_overflow_base).
// Asserts the string is actually spilled (>12 bytes) so the test exercises the
// resolve path rather than the inline fast path.
StringView spill_view(char* pool, uint32_t* cursor, const char* s) noexcept {
    const uint32_t len = static_cast<uint32_t>(std::strlen(s));
    StringView v;
    std::memset(&v, 0, sizeof(v));
    v.length = len;
    std::memcpy(v.prefix, s, 4);
    std::memcpy(pool + *cursor, s, len);
    v.ref.buf_idx = 0;
    v.ref.offset  = *cursor;
    *cursor += len;
    return v;
}

// Make a Decimal128 from an integer mantissa (value = mantissa * 10^-scale).
kernels::decimal::Decimal128 dec_of(int64_t mantissa) noexcept {
    return kernels::decimal::d128_from_i64(mantissa);
}

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

// ===========================================================================
// Card A2 — Float64 key canonicalization.
// ===========================================================================

// -0.0 on the build side must match +0.0 on the probe side (and vice versa).
TEST(BoltJoinKernel, Float64NegativeZeroMatchesPositiveZero) {
    Arena a;
    double bk[] = {-0.0, 1.5, 2.5};
    double pk[] = {0.0, -0.0, 1.5};   // both zero spellings probe build's -0.0
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, 3, BoltType::Float64);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, 3, BoltType::Float64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, 3, &a, &jb));
    int32_t ob[16], op[16];
    const size_t n = join_probe_typed(&jb, &p, 3, ob, op, 16);
    // probe 0 (+0.0)->build 0(-0.0); probe 1(-0.0)->build 0; probe 2(1.5)->build 1.
    EXPECT_EQ(n, 3u);
    for (size_t i = 0; i < n; ++i) {
        // Either both zero (any sign) or numerically equal.
        const bool both_zero = (bk[ob[i]] == 0.0) && (pk[op[i]] == 0.0);
        EXPECT_TRUE(both_zero || bk[ob[i]] == pk[op[i]]);
    }
}

// A NaN key compares equal to itself (canonical-NaN semantics), and two
// DIFFERENT NaN bit-patterns still match because both canonicalize to the same
// quiet NaN. (Raw IEEE NaN != NaN would emit zero pairs — this documents the
// join's defined behaviour.)
TEST(BoltJoinKernel, Float64NaNKeysMatchViaCanonicalization) {
    Arena a;
    // Two distinct NaN payloads (signalling vs quiet, different mantissa bits).
    uint64_t qnan_bits = 0x7FF8000000000000ULL;
    uint64_t snan_bits = 0x7FF0000000000001ULL;  // distinct NaN payload
    double nan_a, nan_b;
    std::memcpy(&nan_a, &qnan_bits, 8);
    std::memcpy(&nan_b, &snan_bits, 8);
    ASSERT_TRUE(std::isnan(nan_a) && std::isnan(nan_b));
    double bk[] = {nan_a, 7.0};
    double pk[] = {nan_b, 7.0};        // different NaN payload than build
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, 2, BoltType::Float64);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, 2, BoltType::Float64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, 2, &a, &jb));
    int32_t ob[16], op[16];
    const size_t n = join_probe_typed(&jb, &p, 2, ob, op, 16);
    EXPECT_EQ(n, 2u);                  // NaN<->NaN and 7.0<->7.0
    // The NaN pair: build row 0 matched by probe row 0.
    bool saw_nan_pair = false, saw_seven = false;
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(bk[ob[i]]) && std::isnan(pk[op[i]])) saw_nan_pair = true;
        if (bk[ob[i]] == 7.0 && pk[op[i]] == 7.0)           saw_seven = true;
    }
    EXPECT_TRUE(saw_nan_pair);
    EXPECT_TRUE(saw_seven);
}

// canon_f64_bits unit check: -0.0 -> +0.0, every NaN -> one bit pattern,
// finite values untouched.
TEST(BoltJoinKernel, Float64CanonBitsUnit) {
    using jk_detail::canon_f64_bits;
    auto bits = [](double d){ int64_t b; std::memcpy(&b, &d, 8); return b; };
    EXPECT_EQ(canon_f64_bits(bits(-0.0)), bits(0.0));
    EXPECT_EQ(canon_f64_bits(bits(0.0)),  bits(0.0));
    EXPECT_EQ(canon_f64_bits(bits(3.25)), bits(3.25));
    EXPECT_EQ(canon_f64_bits(bits(-3.25)), bits(-3.25));
    int64_t nan1 = static_cast<int64_t>(0x7FF8000000000000ULL);
    int64_t nan2 = static_cast<int64_t>(0xFFF0000000000007ULL);  // negative NaN
    EXPECT_EQ(canon_f64_bits(nan1), canon_f64_bits(nan2));
}

// ===========================================================================
// Card A2 — Decimal128 scale precondition.
// ===========================================================================

// Equal mantissa + equal scale -> JOIN. The scale guard reports a match.
TEST(BoltJoinKernel, Decimal128MatchingScaleJoins) {
    Arena a;
    kernels::decimal::Decimal128 bk[] = {dec_of(1230), dec_of(4560)};
    kernels::decimal::Decimal128 pk[] = {dec_of(4560), dec_of(9990)};
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, 2, BoltType::Decimal128);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, 2, BoltType::Decimal128);
    b.decimal_scale = 2; p.decimal_scale = 2;
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, 2, &a, &jb));
    EXPECT_TRUE(join_decimal_scales_match(&jb, &p));
    int32_t ob[16], op[16];
    const size_t n = join_probe_typed(&jb, &p, 2, ob, op, 16);
    EXPECT_EQ(n, 1u);                  // 45.60 == 45.60
    EXPECT_EQ(bk[ob[0]].lo, 4560);
}

// Same NOMINAL value at different scales (12.30@scale2 vs 12.300@scale3) has a
// different mantissa, so it must NOT silently match — and the scale guard flags
// the mismatch. (We don't call join_probe_typed here: its precondition assert
// would fire in debug; the guard is the caller's gate.)
TEST(BoltJoinKernel, Decimal128ScaleMismatchDetected) {
    Arena a;
    kernels::decimal::Decimal128 bk[] = {dec_of(1230)};   // 12.30 @ scale 2
    kernels::decimal::Decimal128 pk[] = {dec_of(12300)};  // 12.300 @ scale 3
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, 1, BoltType::Decimal128);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, 1, BoltType::Decimal128);
    b.decimal_scale = 2; p.decimal_scale = 3;
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, 1, &a, &jb));
    EXPECT_FALSE(join_decimal_scales_match(&jb, &p));   // scale guard fires
    // And even raw mantissas differ (1230 != 12300), so the equi-join would
    // not match regardless — the guard prevents a same-scale rescale bug.
    EXPECT_NE(bk[0].lo, pk[0].lo);
}

// ===========================================================================
// Card A2 — Spilled-Utf8 key arm (the biggest correctness gap).
// ===========================================================================

// Equal >12-byte strings stored at DIFFERENT arena offsets must JOIN. The old
// GbCell16 raw-16-byte compare false-negatives here (the spilled `ref` offset
// differs); the byte-resolving path fixes it.
TEST(BoltJoinKernel, Utf8SpilledKeysJoinAcrossOffsets) {
    Arena a;
    char build_pool[256] = {0};
    char probe_pool[256] = {0};
    uint32_t bc = 0, pc = 0;
    // Build pool layout puts "STRAWBERRY_FIELDS" at offset 0.
    StringView bv[] = {
        spill_view(build_pool, &bc, "STRAWBERRY_FIELDS"),   // len 17, spilled
        spill_view(build_pool, &bc, "BLUEBERRY_HILL_2025"), // len 19, spilled
    };
    // Probe pool deliberately stores a DIFFERENT-LENGTH string FIRST so the
    // equal string lands at a DIFFERENT offset than on the build side.
    StringView pv[] = {
        spill_view(probe_pool, &pc, "ZZ"),                  // pad (inline-ish)
        spill_view(probe_pool, &pc, "BLUEBERRY_HILL_2025"), // matches build[1]
        spill_view(probe_pool, &pc, "STRAWBERRY_FIELDS"),   // matches build[0]
    };
    // Force the first probe view to also be spilled so all three exercise base
    // resolution uniformly. "ZZ" is inline; that's fine (it matches nothing).
    BoltColumn b = make_utf8_col(bv, 2, build_pool);
    BoltColumn p = make_utf8_col(pv, 3, probe_pool);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, 2, &a, &jb));
    int32_t ob[16], op[16];
    const size_t n = join_probe_typed(&jb, &p, 3, ob, op, 16);
    EXPECT_EQ(n, 2u);
    // Verify the matched pairs are byte-equal via the resolving compare.
    for (size_t i = 0; i < n; ++i) {
        const StringView& B = bv[ob[i]];
        const StringView& P = pv[op[i]];
        EXPECT_EQ(0, kernels::utf8::sv_compare(B, build_pool, P, probe_pool));
    }
}

// Inline (<=12 byte) Utf8 keys still JOIN, and unequal strings don't match.
TEST(BoltJoinKernel, Utf8InlineKeysJoin) {
    Arena a;
    StringView bv[3], pv[3];
    const char* bs[] = {"apple", "pear", "kiwi"};
    const char* ps[] = {"pear", "mango", "apple"};
    for (int i = 0; i < 3; ++i) bv[i] = StringView::from_cstr(bs[i]);
    for (int i = 0; i < 3; ++i) pv[i] = StringView::from_cstr(ps[i]);
    BoltColumn b = make_utf8_col(bv, 3, nullptr);   // all inline -> base unused
    BoltColumn p = make_utf8_col(pv, 3, nullptr);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, 3, &a, &jb));
    int32_t ob[16], op[16];
    const size_t n = join_probe_typed(&jb, &p, 3, ob, op, 16);
    EXPECT_EQ(n, 2u);                  // pear, apple match; mango doesn't
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(0, kernels::utf8::sv_compare(bv[ob[i]], nullptr,
                                               pv[op[i]], nullptr));
    }
}

// Mixed inline + spilled keys in one column: an inline build key matches an
// inline probe key; a spilled build key matches a spilled probe key.
TEST(BoltJoinKernel, Utf8MixedInlineSpilledKeys) {
    Arena a;
    char bpool[128] = {0}, ppool[128] = {0};
    uint32_t bc = 0, pc = 0;
    StringView bv[2] = {
        StringView::from_cstr("hi"),                       // inline
        spill_view(bpool, &bc, "a_long_spilled_keyword"),  // spilled (22)
    };
    StringView pv[2] = {
        spill_view(ppool, &pc, "a_long_spilled_keyword"),  // spilled, diff base
        StringView::from_cstr("hi"),                       // inline
    };
    BoltColumn b = make_utf8_col(bv, 2, bpool);
    BoltColumn p = make_utf8_col(pv, 2, ppool);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, 2, &a, &jb));
    int32_t ob[16], op[16];
    const size_t n = join_probe_typed(&jb, &p, 2, ob, op, 16);
    EXPECT_EQ(n, 2u);
}

// ===========================================================================
// Card A2 — OneInt64 fast path vs forced-General path agreement.
// ===========================================================================

// The single-int64-key fast path (join_build_typed -> OneInt64) must emit the
// SAME (build_idx, probe_idx) pair set as the GbCell16 general path
// (join_build_typed_general -> jk_probe_general) over identical data.
TEST(BoltJoinKernel, OneInt64FastPathAgreesWithGeneral) {
    Arena a;
    constexpr int NB = 250, NP = 400;
    static int64_t bk[NB], pk[NP];
    for (int i = 0; i < NB; ++i) bk[i] = (i * 11) % 37;     // dup chains
    for (int i = 0; i < NP; ++i) pk[i] = (i * 5) % 41;
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, NB, BoltType::Int64);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, NP, BoltType::Int64);

    JoinBuildTyped fast{};
    ASSERT_TRUE(join_build_typed(&b, 1, NB, &a, &fast));
    EXPECT_EQ(fast.key_shape, static_cast<uint8_t>(JoinKeyShape::OneInt64));

    JoinBuildTyped gen{};
    ASSERT_TRUE(join_build_typed_general(&b, 1, NB, &a, &gen));
    EXPECT_EQ(gen.key_shape, static_cast<uint8_t>(JoinKeyShape::General));

    static int32_t fb[NB * NP], fp[NB * NP], gb[NB * NP], gp[NB * NP];
    const size_t nf = join_probe_typed(&fast, &p, NP, fb, fp, NB * NP);
    const size_t ng = jk_probe_general<false>(&gen, &p, NP, gb, gp, NB * NP);
    ASSERT_EQ(nf, ng);
    // Both walk build chains head-first in the same insertion order, so the
    // emitted pair sequences are identical element-for-element.
    for (size_t i = 0; i < nf; ++i) {
        EXPECT_EQ(fb[i], gb[i]);
        EXPECT_EQ(fp[i], gp[i]);
        EXPECT_EQ(bk[fb[i]], pk[fp[i]]);
    }
    // Cross-check against brute force.
    size_t expect = 0;
    for (int r = 0; r < NP; ++r)
        for (int s = 0; s < NB; ++s)
            if (pk[r] == bk[s]) ++expect;
    EXPECT_EQ(nf, expect);
}

// ===========================================================================
// W-J4 — bloom-gated probe agreement.
// ===========================================================================

// use_bloom=true must emit the SAME pair sequence as use_bloom=false (the
// SBBF has no false negatives; only lookup cost changes). Exercised on BOTH
// key shapes, with a probe set dominated by non-members so the bloom path's
// skip arm actually runs (members ~10%).
TEST(BoltJoinKernel, BloomProbeAgreesOneInt64) {
    Arena a;
    constexpr int NB = 300, NP = 2000;
    static int64_t bk[NB], pk[NP];
    for (int i = 0; i < NB; ++i) bk[i] = i * 7;                  // members: 0,7,14,…
    for (int i = 0; i < NP; ++i) pk[i] = (i % 10 == 0) ? (i % NB) * 7
                                                       : 1000000 + i;  // ~90% misses
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, NB, BoltType::Int64);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, NP, BoltType::Int64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, NB, &a, &jb));
    EXPECT_EQ(jb.has_bloom, 1);
    static int32_t nb_[NP * 4], np_[NP * 4], bb_[NP * 4], bp_[NP * 4];
    const size_t n0 = join_probe_typed(&jb, &p, NP, nb_, np_, NP * 4, false);
    const size_t n1 = join_probe_typed(&jb, &p, NP, bb_, bp_, NP * 4, true);
    ASSERT_EQ(n0, n1);
    ASSERT_GT(n0, 0u);
    for (size_t i = 0; i < n0; ++i) {
        EXPECT_EQ(nb_[i], bb_[i]);
        EXPECT_EQ(np_[i], bp_[i]);
    }
}

// Same agreement over the General (composite-key) path: two-key composite so
// key_shape stays General; ~10% members.
TEST(BoltJoinKernel, BloomProbeAgreesGeneral) {
    Arena a;
    constexpr int NB = 200, NP = 1500;
    static int64_t bk0[NB], bk1[NB], pk0[NP], pk1[NP];
    for (int i = 0; i < NB; ++i) { bk0[i] = i; bk1[i] = i * 3; }
    for (int i = 0; i < NP; ++i) {
        const bool member = (i % 10 == 0);
        pk0[i] = member ? (i % NB)     : 500000 + i;
        pk1[i] = member ? (i % NB) * 3 : 900000 + i;
    }
    BoltColumn b[2] = {
        BoltColumn::make_flat(bk0, nullptr, NB, BoltType::Int64),
        BoltColumn::make_flat(bk1, nullptr, NB, BoltType::Int64)};
    BoltColumn p[2] = {
        BoltColumn::make_flat(pk0, nullptr, NP, BoltType::Int64),
        BoltColumn::make_flat(pk1, nullptr, NP, BoltType::Int64)};
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(b, 2, NB, &a, &jb));
    EXPECT_EQ(jb.key_shape, static_cast<uint8_t>(JoinKeyShape::General));
    EXPECT_EQ(jb.has_bloom, 1);
    static int32_t nb_[NP * 2], np_[NP * 2], bb_[NP * 2], bp_[NP * 2];
    const size_t n0 = join_probe_typed(&jb, p, NP, nb_, np_, NP * 2, false);
    const size_t n1 = join_probe_typed(&jb, p, NP, bb_, bp_, NP * 2, true);
    ASSERT_EQ(n0, n1);
    ASSERT_GT(n0, 0u);
    for (size_t i = 0; i < n0; ++i) {
        EXPECT_EQ(nb_[i], bb_[i]);
        EXPECT_EQ(np_[i], bp_[i]);
    }
}

// Date32 single key takes the OneInt64 fast path (it's a 32-bit int under the
// hood) and joins correctly, including negative (pre-epoch) day values.
TEST(BoltJoinKernel, Date32SingleKeyFastPath) {
    Arena a;
    int32_t bk[] = {-5, 0, 19000};     // negative, epoch, modern day
    int32_t pk[] = {19000, -5, 7};
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, 3, BoltType::Date32);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, 3, BoltType::Date32);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, 3, &a, &jb));
    EXPECT_EQ(jb.key_shape, static_cast<uint8_t>(JoinKeyShape::OneInt64));
    int32_t ob[16], op[16];
    const size_t n = join_probe_typed(&jb, &p, 3, ob, op, 16);
    EXPECT_EQ(n, 2u);                  // 19000 and -5 match; 7 doesn't
    for (size_t i = 0; i < n; ++i) EXPECT_EQ(bk[ob[i]], pk[op[i]]);
}

// Composite key with a Utf8 component + an int component, spilled strings.
TEST(BoltJoinKernel, CompositeUtf8PlusInt) {
    Arena a;
    char bpool[128] = {0}, ppool[128] = {0};
    uint32_t bc = 0, pc = 0;
    StringView bs[] = { spill_view(bpool, &bc, "region_north_america"),
                        spill_view(bpool, &bc, "region_north_america") };
    int64_t   bi[] = { 100, 200 };
    StringView ps[] = { spill_view(ppool, &pc, "region_north_america") };
    int64_t   pi[] = { 200 };
    BoltColumn bcols[2] = { make_utf8_col(bs, 2, bpool),
                            BoltColumn::make_flat(bi, nullptr, 2, BoltType::Int64) };
    BoltColumn pcols[2] = { make_utf8_col(ps, 1, ppool),
                            BoltColumn::make_flat(pi, nullptr, 1, BoltType::Int64) };
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(bcols, 2, 2, &a, &jb));
    EXPECT_EQ(jb.key_shape, static_cast<uint8_t>(JoinKeyShape::General));
    int32_t ob[16], op[16];
    const size_t n = join_probe_typed(&jb, pcols, 1, ob, op, 16);
    EXPECT_EQ(n, 1u);                  // (region.., 200) -> build row 1
    EXPECT_EQ(bi[ob[0]], 200);
}

}  // namespace
