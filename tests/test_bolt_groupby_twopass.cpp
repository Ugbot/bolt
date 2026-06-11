// Tests for K-AGG-C — the two-pass typed GROUP BY ingest
// (bolt_groupby_twopass.inc) against the per-row fallback contract.
//
// The hard correctness requirement is CROSS-PATH CONSISTENCY: a window may
// run the specialised two-pass path while the next window (same state, same
// SwissTable) runs the fallback, so hashes and stored canonical cells must
// be byte-exact between paths or the same logical key silently becomes two
// groups. Scope:
//   - hash equivalence fuzz: packed-register hash vs gb_detail::hash_keys
//     (inline Utf8 with garbage padding bytes, and int keys)
//   - garbage padding in StringView unused bytes must not split groups
//   - cross-path mixing: inline-only morsels (two-pass) interleaved with
//     spilled-key morsels (fallback) over the same logical keys
//   - TPC-H Q1 shape vs a scalar oracle (multi-window, 2 Utf8 keys,
//     d128 SUM / COUNT(*) / AVG(d128))
//   - NULL masking, sparse sel, empty sel through the two-pass path
//   - MRU pass-1 eviction churn; ring pass-1 with many groups
//   - banked pass-2 (low cardinality) incl. Avg divisor correctness
//   - DISTINCT falls back; spilled Card-S keys group by content
//   - >65536-row single ingest (window split); cap-overflow oom

#include "bolt/join/bolt_groupby.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

using namespace bolt;
namespace dec = bolt::kernels::decimal;

inline AggSpec make_spec(AggKind k, uint8_t in_col,
                         uint8_t distinct = 0) noexcept {
    AggSpec s{};
    s.kind = k; s.in_col = in_col; s.distinct = distinct;
    return s;
}

// Inline StringView with DETERMINISTIC GARBAGE in the unused padding bytes.
// The two-pass pack masks padding and the fallback canonicalises at insert;
// either way garbage must never influence grouping or hashing.
StringView sv_garbage(const char* s, uint32_t len, uint8_t garbage) noexcept {
    StringView v;
    std::memset(&v, garbage, sizeof(v));
    v.length = len;
    assert(len <= 12u);
    std::memcpy(v.prefix, s, len < 4u ? len : 4u);
    if (len > 4u) std::memcpy(v.inline_data, s + 4, len - 4u);
    // bytes past the content keep `garbage`
    return v;
}

StringView sv_clean(const char* s, uint32_t len) noexcept {
    StringView v{};
    v.length = len;
    assert(len <= 12u);
    std::memcpy(v.prefix, s, len < 4u ? len : 4u);
    if (len > 4u) std::memcpy(v.inline_data, s + 4, len - 4u);
    return v;
}

uint64_t splitmix64(uint64_t* s) noexcept {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// ---------------------------------------------------------------------------
// Hash equivalence fuzz: the packed register hash must be byte-exact vs
// gb_detail::hash_keys for every inline content, INDEPENDENT of padding.
// ---------------------------------------------------------------------------

TEST(BoltGroupbyTwopass, HashEquivalenceFuzzInlineUtf8) {
    GroupbyTypedState st{};
    st.n_keys = 1;
    st.key_types[0] = BoltType::Utf8;
    uint64_t rng = 0xC0FFEE0001ULL;
    for (int iter = 0; iter < 20000; ++iter) {
        char buf[12];
        const uint32_t len = static_cast<uint32_t>(splitmix64(&rng) % 13u);
        for (uint32_t i = 0; i < len; ++i) {
            buf[i] = static_cast<char>(splitmix64(&rng) & 0xFF);
        }
        const uint8_t garbage = static_cast<uint8_t>(splitmix64(&rng) | 1u);
        StringView sv = sv_garbage(buf, len, garbage);
        BoltColumn col = BoltColumn::make_flat(&sv, nullptr, 1, BoltType::Utf8);
        const uint64_t ref = gb_detail::hash_keys(&col, 1, 0);
        uint64_t a[1], b[1];
        gb_twopass::gb_pack_row<1, true>(&st, &col, 0, a, b);
        const uint64_t got = gb_twopass::gb_hash_packed<1, true>(a, b);
        ASSERT_EQ(ref, got) << "len=" << len << " iter=" << iter;
    }
}

TEST(BoltGroupbyTwopass, HashEquivalenceFuzzIntKeys) {
    GroupbyTypedState st{};
    st.n_keys = 2;
    st.key_types[0] = BoltType::Int32;
    st.key_types[1] = BoltType::Int64;
    uint64_t rng = 0xBADC0DE99ULL;
    for (int iter = 0; iter < 20000; ++iter) {
        int32_t k0 = static_cast<int32_t>(splitmix64(&rng));
        int64_t k1 = static_cast<int64_t>(splitmix64(&rng));
        BoltColumn cols[2];
        cols[0] = BoltColumn::make_flat(&k0, nullptr, 1, BoltType::Int32);
        cols[1] = BoltColumn::make_flat(&k1, nullptr, 1, BoltType::Int64);
        const uint64_t ref = gb_detail::hash_keys(cols, 2, 0);
        uint64_t a[2], b[2];
        gb_twopass::gb_pack_row<2, false>(&st, cols, 0, a, b);
        const uint64_t got = gb_twopass::gb_hash_packed<2, false>(a, b);
        ASSERT_EQ(ref, got) << "iter=" << iter;
    }
}

// ---------------------------------------------------------------------------
// End-to-end harness: begin/ingest.../finalize, returning results keyed for
// oracle comparison.
// ---------------------------------------------------------------------------

struct RunOut {
    uint32_t ng = 0;
    BoltColumn ok[2];
    BoltColumn oa[8];
};

// ---------------------------------------------------------------------------

TEST(BoltGroupbyTwopass, GarbagePaddingGroupsTogether) {
    Arena a;
    // Same logical keys, three different padding patterns, two morsels.
    StringView k1[3] = {sv_garbage("AB", 2, 0x11), sv_garbage("CD", 2, 0x22),
                        sv_garbage("AB", 2, 0x33)};
    StringView k2[3] = {sv_garbage("CD", 2, 0x44), sv_garbage("AB", 2, 0x55),
                        sv_clean("CD", 2)};
    int64_t v1[3] = {1, 2, 4};
    int64_t v2[3] = {8, 16, 32};
    BoltColumn kd = BoltColumn::make_flat(k1, nullptr, 0, BoltType::Utf8);
    BoltColumn vd = BoltColumn::make_flat(v1, nullptr, 0, BoltType::Int64);
    AggSpec spec = make_spec(AggKind::Sum, 0);
    GroupbyTypedState st{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&st, &a, &kd, 1, &vd, 1,
                                                  &spec, 1, 8));
    BoltColumn kc = BoltColumn::make_flat(k1, nullptr, 3, BoltType::Utf8);
    BoltColumn vc = BoltColumn::make_flat(v1, nullptr, 3, BoltType::Int64);
    groupby_agg_multi_key_typed_ingest(&st, &kc, &vc, nullptr, 0, 3);
    kc = BoltColumn::make_flat(k2, nullptr, 3, BoltType::Utf8);
    vc = BoltColumn::make_flat(v2, nullptr, 3, BoltType::Int64);
    groupby_agg_multi_key_typed_ingest(&st, &kc, &vc, nullptr, 0, 3);
    ASSERT_FALSE(st.oom);
    BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
    EXPECT_EQ(ng, 2u);
    const StringView* outk = static_cast<const StringView*>(ok[0].data);
    const int64_t* outv = static_cast<const int64_t*>(oa[0].data);
    for (uint32_t i = 0; i < ng; ++i) {
        if (std::memcmp(outk[i].prefix, "AB", 2) == 0) {
            EXPECT_EQ(outv[i], 1 + 4 + 16);
        } else {
            EXPECT_EQ(outv[i], 2 + 8 + 32);
        }
    }
}

// Cross-path mixing: an all-inline morsel runs two-pass; a morsel holding
// one spilled (>12) key runs the fallback for its whole window. Keys seen
// by both paths must land in the SAME groups, in both orders.
static void run_mixing(bool spilled_first) {
    Arena a;
    char longbuf[32];
    std::memset(longbuf, 'z', sizeof(longbuf));
    StringView lk{};
    lk.length = 20;
    std::memcpy(lk.prefix, longbuf, 4);
    lk.ref.buf_idx = 0;
    lk.ref.offset = 0;

    StringView inl[4] = {sv_garbage("AA", 2, 0xA5), sv_garbage("BB", 2, 0x5A),
                         sv_garbage("AA", 2, 0x77), sv_garbage("BB", 2, 0x00)};
    int64_t inl_v[4] = {1, 2, 4, 8};
    StringView mix[3] = {sv_clean("AA", 2), lk, sv_clean("BB", 2)};
    int64_t mix_v[3] = {16, 32, 64};

    BoltColumn kd = BoltColumn::make_flat(inl, nullptr, 0, BoltType::Utf8);
    BoltColumn vd = BoltColumn::make_flat(inl_v, nullptr, 0, BoltType::Int64);
    AggSpec spec = make_spec(AggKind::Sum, 0);
    GroupbyTypedState st{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&st, &a, &kd, 1, &vd, 1,
                                                  &spec, 1, 8));
    BoltColumn ki = BoltColumn::make_flat(inl, nullptr, 4, BoltType::Utf8);
    BoltColumn vi = BoltColumn::make_flat(inl_v, nullptr, 4, BoltType::Int64);
    BoltColumn km = BoltColumn::make_flat(mix, nullptr, 3, BoltType::Utf8);
    km.str_overflow_base = longbuf;
    BoltColumn vm = BoltColumn::make_flat(mix_v, nullptr, 3, BoltType::Int64);
    if (spilled_first) {
        groupby_agg_multi_key_typed_ingest(&st, &km, &vm, nullptr, 0, 3);
        groupby_agg_multi_key_typed_ingest(&st, &ki, &vi, nullptr, 0, 4);
    } else {
        groupby_agg_multi_key_typed_ingest(&st, &ki, &vi, nullptr, 0, 4);
        groupby_agg_multi_key_typed_ingest(&st, &km, &vm, nullptr, 0, 3);
    }
    ASSERT_FALSE(st.oom);
    BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
    ASSERT_EQ(ng, 3u);   // AA, BB, and the spilled key — never 4+
    const StringView* outk = static_cast<const StringView*>(ok[0].data);
    const int64_t* outv = static_cast<const int64_t*>(oa[0].data);
    int64_t sum_aa = -1, sum_bb = -1, sum_long = -1;
    for (uint32_t i = 0; i < ng; ++i) {
        if (outk[i].length == 20u) sum_long = outv[i];
        else if (std::memcmp(outk[i].prefix, "AA", 2) == 0) sum_aa = outv[i];
        else sum_bb = outv[i];
    }
    EXPECT_EQ(sum_aa, 1 + 4 + 16);
    EXPECT_EQ(sum_bb, 2 + 8 + 64);
    EXPECT_EQ(sum_long, 32);
}

TEST(BoltGroupbyTwopass, CrossPathMixingInlineThenSpilled) {
    run_mixing(/*spilled_first=*/false);
}

TEST(BoltGroupbyTwopass, CrossPathMixingSpilledThenInline) {
    run_mixing(/*spilled_first=*/true);
}

// ---------------------------------------------------------------------------
// Q1 shape vs scalar oracle: 2 one-char Utf8 keys (4 combos), 3 d128 SUMs,
// COUNT(*), AVG(d128); 50k rows ingested in uneven morsels so window
// boundaries and the xmorsel path are exercised.
// ---------------------------------------------------------------------------

TEST(BoltGroupbyTwopass, Q1ShapeMatchesScalarOracle) {
    constexpr int64_t N = 50000;
    static StringView rf[N], ls[N];
    static dec::Decimal128 p0[N], p1[N], p2[N];
    const char* rfv[3] = {"A", "N", "R"};
    const char* lsv[2] = {"F", "O"};
    uint64_t rng = 0x51AB5ULL;
    struct Acc { int64_t s0 = 0, s1 = 0, s2 = 0, n = 0; };
    std::map<std::string, Acc> oracle;
    for (int64_t i = 0; i < N; ++i) {
        const uint32_t r3 = static_cast<uint32_t>(splitmix64(&rng) % 3u);
        const uint32_t r2 = static_cast<uint32_t>(splitmix64(&rng) % 2u);
        rf[i] = sv_garbage(rfv[r3], 1, static_cast<uint8_t>(splitmix64(&rng)));
        ls[i] = sv_garbage(lsv[r2], 1, static_cast<uint8_t>(splitmix64(&rng)));
        const int64_t a0 = static_cast<int64_t>(splitmix64(&rng) % 100000u);
        const int64_t a1 = static_cast<int64_t>(splitmix64(&rng) % 100000u);
        const int64_t a2 = static_cast<int64_t>(splitmix64(&rng) % 100000u);
        p0[i] = dec::d128_from_i64(a0);
        p1[i] = dec::d128_from_i64(a1);
        p2[i] = dec::d128_from_i64(a2);
        Acc& acc = oracle[std::string(rfv[r3]) + lsv[r2]];
        acc.s0 += a0; acc.s1 += a1; acc.s2 += a2; acc.n += 1;
    }
    Arena a;
    BoltColumn kd[2], pd[3];
    kd[0] = BoltColumn::make_flat(rf, nullptr, 0, BoltType::Utf8);
    kd[1] = BoltColumn::make_flat(ls, nullptr, 0, BoltType::Utf8);
    pd[0] = BoltColumn::make_flat(p0, nullptr, 0, BoltType::Decimal128);
    pd[1] = BoltColumn::make_flat(p1, nullptr, 0, BoltType::Decimal128);
    pd[2] = BoltColumn::make_flat(p2, nullptr, 0, BoltType::Decimal128);
    pd[0].decimal_scale = 2; pd[1].decimal_scale = 2; pd[2].decimal_scale = 2;
    AggSpec specs[5];
    specs[0] = make_spec(AggKind::Sum, 0);
    specs[1] = make_spec(AggKind::Sum, 1);
    specs[2] = make_spec(AggKind::Sum, 2);
    specs[3] = make_spec(AggKind::CountStar, 0);
    specs[4] = make_spec(AggKind::Avg, 2);
    GroupbyTypedState st{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&st, &a, kd, 2, pd, 3,
                                                  specs, 5, 8));
    // Uneven morsels: 7, then 13000, then the rest.
    const int64_t cuts[3] = {7, 13000, N};
    int64_t off = 0;
    for (int c = 0; c < 3; ++c) {
        const int64_t len = cuts[c] - off;
        BoltColumn k[2], p[3];
        k[0] = BoltColumn::make_flat(rf + off, nullptr, len, BoltType::Utf8);
        k[1] = BoltColumn::make_flat(ls + off, nullptr, len, BoltType::Utf8);
        p[0] = BoltColumn::make_flat(p0 + off, nullptr, len, BoltType::Decimal128);
        p[1] = BoltColumn::make_flat(p1 + off, nullptr, len, BoltType::Decimal128);
        p[2] = BoltColumn::make_flat(p2 + off, nullptr, len, BoltType::Decimal128);
        p[0].decimal_scale = 2; p[1].decimal_scale = 2; p[2].decimal_scale = 2;
        groupby_agg_multi_key_typed_ingest(&st, k, p, nullptr, 0, len);
        ASSERT_FALSE(st.oom);
        off = cuts[c];
    }
    BoltColumn ok[2], oa[5];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
    ASSERT_EQ(ng, oracle.size());
    const StringView* okr = static_cast<const StringView*>(ok[0].data);
    const StringView* okl = static_cast<const StringView*>(ok[1].data);
    for (uint32_t i = 0; i < ng; ++i) {
        std::string key;
        key.push_back(okr[i].prefix[0]);
        key.push_back(okl[i].prefix[0]);
        ASSERT_TRUE(oracle.count(key)) << key;
        const Acc& acc = oracle[key];
        const auto* s0 = static_cast<const dec::Decimal128*>(oa[0].data);
        const auto* s1 = static_cast<const dec::Decimal128*>(oa[1].data);
        const auto* s2 = static_cast<const dec::Decimal128*>(oa[2].data);
        const auto* cs = static_cast<const int64_t*>(oa[3].data);
        const auto* av = static_cast<const dec::Decimal128*>(oa[4].data);
        EXPECT_EQ(dec::d128_cmp(s0[i], dec::d128_from_i64(acc.s0)), 0) << key;
        EXPECT_EQ(dec::d128_cmp(s1[i], dec::d128_from_i64(acc.s1)), 0) << key;
        EXPECT_EQ(dec::d128_cmp(s2[i], dec::d128_from_i64(acc.s2)), 0) << key;
        EXPECT_EQ(cs[i], acc.n) << key;
        // AVG replicates finalize's formula: rescale(+4) then divide.
        const dec::Decimal128 num =
            dec::d128_rescale(dec::d128_from_i64(acc.s2), 2, 6);
        const dec::Decimal128 q = dec::d128_div(num, dec::d128_from_i64(acc.n));
        EXPECT_EQ(dec::d128_cmp(av[i], q), 0) << key;
    }
}

// ---------------------------------------------------------------------------
// NULL masking through the two-pass loops (HasValid=true specialisations).
// ---------------------------------------------------------------------------

TEST(BoltGroupbyTwopass, NullMaskingThroughTwopass) {
    Arena a;
    int64_t ks[6] = {1, 1, 1, 2, 2, 2};
    dec::Decimal128 vs[6];
    for (int i = 0; i < 6; ++i) vs[i] = dec::d128_from_i64((i + 1) * 10);
    uint8_t validity[1] = {0b00101101};   // rows 1 and 4 NULL
    BoltColumn kd = BoltColumn::make_flat(ks, nullptr, 0, BoltType::Int64);
    BoltColumn vd = BoltColumn::make_flat(vs, nullptr, 0, BoltType::Decimal128);
    AggSpec specs[4];
    specs[0] = make_spec(AggKind::Sum, 0);
    specs[1] = make_spec(AggKind::Count, 0);
    specs[2] = make_spec(AggKind::CountStar, 0);
    specs[3] = make_spec(AggKind::Avg, 0);
    GroupbyTypedState st{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&st, &a, &kd, 1, &vd, 1,
                                                  specs, 4, 4));
    BoltColumn k = BoltColumn::make_flat(ks, nullptr, 6, BoltType::Int64);
    BoltColumn v = BoltColumn::make_flat(vs, validity, 6, BoltType::Decimal128);
    groupby_agg_multi_key_typed_ingest(&st, &k, &v, nullptr, 0, 6);
    ASSERT_FALSE(st.oom);
    BoltColumn ok[1], oa[4];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
    ASSERT_EQ(ng, 2u);
    const int64_t* outk = static_cast<const int64_t*>(ok[0].data);
    const auto* sums = static_cast<const dec::Decimal128*>(oa[0].data);
    const auto* cnts = static_cast<const int64_t*>(oa[1].data);
    const auto* css  = static_cast<const int64_t*>(oa[2].data);
    for (uint32_t i = 0; i < ng; ++i) {
        if (outk[i] == 1) {
            EXPECT_EQ(dec::d128_cmp(sums[i], dec::d128_from_i64(10 + 30)), 0);
            EXPECT_EQ(cnts[i], 2);
            EXPECT_EQ(css[i], 3);
        } else {
            EXPECT_EQ(dec::d128_cmp(sums[i], dec::d128_from_i64(40 + 60)), 0);
            EXPECT_EQ(cnts[i], 2);
            EXPECT_EQ(css[i], 3);
        }
    }
}

TEST(BoltGroupbyTwopass, SparseSelAndEmptySel) {
    Arena a;
    int64_t ks[8] = {1, 2, 1, 2, 1, 2, 1, 2};
    int64_t vs[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    uint32_t sel[4] = {1, 3, 5, 7};
    BoltColumn kd = BoltColumn::make_flat(ks, nullptr, 0, BoltType::Int64);
    BoltColumn vd = BoltColumn::make_flat(vs, nullptr, 0, BoltType::Int64);
    AggSpec spec = make_spec(AggKind::Sum, 0);
    GroupbyTypedState st{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&st, &a, &kd, 1, &vd, 1,
                                                  &spec, 1, 4));
    BoltColumn k = BoltColumn::make_flat(ks, nullptr, 8, BoltType::Int64);
    BoltColumn v = BoltColumn::make_flat(vs, nullptr, 8, BoltType::Int64);
    // Empty sel: sel != nullptr, sel_len == 0 must ingest NOTHING.
    groupby_agg_multi_key_typed_ingest(&st, &k, &v, sel, 0, 8);
    ASSERT_FALSE(st.oom);
    EXPECT_EQ(st.entry_count, 0u);
    groupby_agg_multi_key_typed_ingest(&st, &k, &v, sel, 4, 8);
    ASSERT_FALSE(st.oom);
    BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
    ASSERT_EQ(ng, 1u);
    EXPECT_EQ(static_cast<const int64_t*>(ok[0].data)[0], 2);
    EXPECT_EQ(static_cast<const int64_t*>(oa[0].data)[0], 200);
}

// ---------------------------------------------------------------------------
// Pass-1 variants: MRU eviction churn (8 cycling keys overflow the 4-entry
// MRU every row) and the ring path (window starts past kGbMruWindowMax).
// ---------------------------------------------------------------------------

TEST(BoltGroupbyTwopass, MruEvictionChurnIsExact) {
    constexpr int64_t N = 5000;
    static int64_t ks[N];
    static int64_t vs[N];
    int64_t expect[8] = {0};
    for (int64_t i = 0; i < N; ++i) {
        ks[i] = i % 8;          // cycle 8 keys through a 4-entry MRU
        vs[i] = i;
        expect[i % 8] += i;
    }
    Arena a;
    BoltColumn kd = BoltColumn::make_flat(ks, nullptr, 0, BoltType::Int64);
    BoltColumn vd = BoltColumn::make_flat(vs, nullptr, 0, BoltType::Int64);
    AggSpec spec = make_spec(AggKind::Sum, 0);
    GroupbyTypedState st{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&st, &a, &kd, 1, &vd, 1,
                                                  &spec, 1, 16));
    BoltColumn k = BoltColumn::make_flat(ks, nullptr, N, BoltType::Int64);
    BoltColumn v = BoltColumn::make_flat(vs, nullptr, N, BoltType::Int64);
    groupby_agg_multi_key_typed_ingest(&st, &k, &v, nullptr, 0, N);
    ASSERT_FALSE(st.oom);
    BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
    ASSERT_EQ(ng, 8u);
    const int64_t* outk = static_cast<const int64_t*>(ok[0].data);
    const int64_t* outv = static_cast<const int64_t*>(oa[0].data);
    for (uint32_t i = 0; i < ng; ++i) EXPECT_EQ(outv[i], expect[outk[i]]);
}

TEST(BoltGroupbyTwopass, RingPathManyGroupsIsExact) {
    constexpr int64_t N = 5000;   // > kGbMruWindowMax groups
    static int64_t ks[N];
    static int64_t vs[N];
    for (int64_t i = 0; i < N; ++i) { ks[i] = i; vs[i] = i * 3; }
    Arena a;
    BoltColumn kd = BoltColumn::make_flat(ks, nullptr, 0, BoltType::Int64);
    BoltColumn vd = BoltColumn::make_flat(vs, nullptr, 0, BoltType::Int64);
    AggSpec spec = make_spec(AggKind::Sum, 0);
    GroupbyTypedState st{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&st, &a, &kd, 1, &vd, 1,
                                                  &spec, 1, N * 2));
    BoltColumn k = BoltColumn::make_flat(ks, nullptr, N, BoltType::Int64);
    BoltColumn v = BoltColumn::make_flat(vs, nullptr, N, BoltType::Int64);
    // First ingest fills 5000 groups (MRU window at ec=0); the SECOND
    // ingest starts at ec=5000 -> ring/prefetch pass-1, every key a re-find.
    groupby_agg_multi_key_typed_ingest(&st, &k, &v, nullptr, 0, N);
    groupby_agg_multi_key_typed_ingest(&st, &k, &v, nullptr, 0, N);
    ASSERT_FALSE(st.oom);
    BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
    ASSERT_EQ(ng, static_cast<uint32_t>(N));
    const int64_t* outk = static_cast<const int64_t*>(ok[0].data);
    const int64_t* outv = static_cast<const int64_t*>(oa[0].data);
    for (uint32_t i = 0; i < ng; ++i) EXPECT_EQ(outv[i], outk[i] * 6);
}

// ---------------------------------------------------------------------------
// Banked pass-2: low cardinality + window large enough to amortise zeroing.
// Avg must divide by the exact per-group valid count via the banked counts.
// ---------------------------------------------------------------------------

TEST(BoltGroupbyTwopass, BankedAvgAndSumsAreExact) {
    constexpr int64_t N = 4096;   // 8 groups -> banked (N >= 8 * 64)
    static int64_t ks[N];
    static dec::Decimal128 dv[N];
    static int64_t iv[N];
    int64_t dsum[8] = {0}, isum[8] = {0}, n_g[8] = {0};
    for (int64_t i = 0; i < N; ++i) {
        ks[i] = (i * 2654435761u) % 8;
        const int64_t x = i % 1000;
        dv[i] = dec::d128_from_i64(x);
        iv[i] = x * 2;
        dsum[ks[i]] += x; isum[ks[i]] += x * 2; n_g[ks[i]] += 1;
    }
    Arena a;
    BoltColumn kd = BoltColumn::make_flat(ks, nullptr, 0, BoltType::Int64);
    BoltColumn pd[2];
    pd[0] = BoltColumn::make_flat(dv, nullptr, 0, BoltType::Decimal128);
    pd[0].decimal_scale = 2;
    pd[1] = BoltColumn::make_flat(iv, nullptr, 0, BoltType::Int64);
    AggSpec specs[5];
    specs[0] = make_spec(AggKind::Sum, 0);
    specs[1] = make_spec(AggKind::Avg, 0);
    specs[2] = make_spec(AggKind::Sum, 1);
    specs[3] = make_spec(AggKind::Avg, 1);
    specs[4] = make_spec(AggKind::Count, 1);
    GroupbyTypedState st{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&st, &a, &kd, 1, pd, 2,
                                                  specs, 5, 16));
    BoltColumn k = BoltColumn::make_flat(ks, nullptr, N, BoltType::Int64);
    BoltColumn p[2];
    p[0] = BoltColumn::make_flat(dv, nullptr, N, BoltType::Decimal128);
    p[0].decimal_scale = 2;
    p[1] = BoltColumn::make_flat(iv, nullptr, N, BoltType::Int64);
    groupby_agg_multi_key_typed_ingest(&st, &k, p, nullptr, 0, N);
    ASSERT_FALSE(st.oom);
    BoltColumn ok[1], oa[5];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
    ASSERT_EQ(ng, 8u);
    const int64_t* outk = static_cast<const int64_t*>(ok[0].data);
    for (uint32_t i = 0; i < ng; ++i) {
        const int64_t g = outk[i];
        const auto* ds = static_cast<const dec::Decimal128*>(oa[0].data);
        const auto* da = static_cast<const dec::Decimal128*>(oa[1].data);
        const auto* is = static_cast<const int64_t*>(oa[2].data);
        const auto* ia = static_cast<const double*>(oa[3].data);
        const auto* ic = static_cast<const int64_t*>(oa[4].data);
        EXPECT_EQ(dec::d128_cmp(ds[i], dec::d128_from_i64(dsum[g])), 0);
        const dec::Decimal128 num =
            dec::d128_rescale(dec::d128_from_i64(dsum[g]), 2, 6);
        const dec::Decimal128 q =
            dec::d128_div(num, dec::d128_from_i64(n_g[g]));
        EXPECT_EQ(dec::d128_cmp(da[i], q), 0);
        EXPECT_EQ(is[i], isum[g]);
        EXPECT_DOUBLE_EQ(ia[i], static_cast<double>(isum[g]) /
                                static_cast<double>(n_g[g]));
        EXPECT_EQ(ic[i], n_g[g]);
    }
}

// ---------------------------------------------------------------------------
// Specialised Min/Max + Date32 keys; Float64 keeps fallback-identical
// semantics through gb_p2_generic.
// ---------------------------------------------------------------------------

TEST(BoltGroupbyTwopass, MinMaxAndFloat64ThroughTwopass) {
    constexpr int64_t N = 1000;
    static int32_t ks[N];
    static int64_t v64[N];
    static int32_t v32[N];
    static double  vf[N];
    int64_t mn[4], mx[4];
    // KNOWN PRE-EXISTING BUG (preserved, not fixed silently): apply(Sum,
    // Float64) integer-adds the double's BIT PATTERN (slot->a += v.a). The
    // two-pass generic loop must be bit-identical to the fallback, so the
    // oracle below REPLAYS gb_detail::apply rather than summing doubles.
    GbCell16 frep[4] = {};
    for (int i = 0; i < 4; ++i) {
        mn[i] = INT64_MAX; mx[i] = INT64_MIN;
        frep[i] = gb_detail::agg_identity(AggKind::Sum, BoltType::Float64);
    }
    for (int64_t i = 0; i < N; ++i) {
        ks[i] = static_cast<int32_t>(18000 + (i % 4));   // Date32-style keys
        v64[i] = (i * 37) % 5000 - 2500;
        v32[i] = static_cast<int32_t>((i * 13) % 997);
        vf[i] = static_cast<double>(i % 100);
        const int g = static_cast<int>(i % 4);
        if (v64[i] < mn[g]) mn[g] = v64[i];
        if (v64[i] > mx[g]) mx[g] = v64[i];
        GbCell16 v{};
        std::memcpy(&v.a, &vf[i], 8);
        gb_detail::apply(AggKind::Sum, BoltType::Float64, &frep[g], v, true);
    }
    Arena a;
    BoltColumn kd = BoltColumn::make_flat(ks, nullptr, 0, BoltType::Date32);
    BoltColumn pd[3];
    pd[0] = BoltColumn::make_flat(v64, nullptr, 0, BoltType::Int64);
    pd[1] = BoltColumn::make_flat(v32, nullptr, 0, BoltType::Int32);
    pd[2] = BoltColumn::make_flat(vf, nullptr, 0, BoltType::Float64);
    AggSpec specs[4];
    specs[0] = make_spec(AggKind::Min, 0);
    specs[1] = make_spec(AggKind::Max, 0);
    specs[2] = make_spec(AggKind::Min, 1);
    specs[3] = make_spec(AggKind::Sum, 2);
    GroupbyTypedState st{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&st, &a, &kd, 1, pd, 3,
                                                  specs, 4, 8));
    BoltColumn k = BoltColumn::make_flat(ks, nullptr, N, BoltType::Date32);
    BoltColumn p[3];
    p[0] = BoltColumn::make_flat(v64, nullptr, N, BoltType::Int64);
    p[1] = BoltColumn::make_flat(v32, nullptr, N, BoltType::Int32);
    p[2] = BoltColumn::make_flat(vf, nullptr, N, BoltType::Float64);
    groupby_agg_multi_key_typed_ingest(&st, &k, p, nullptr, 0, N);
    ASSERT_FALSE(st.oom);
    BoltColumn ok[1], oa[4];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
    ASSERT_EQ(ng, 4u);
    const int32_t* outk = static_cast<const int32_t*>(ok[0].data);
    for (uint32_t i = 0; i < ng; ++i) {
        const int g = outk[i] - 18000;
        EXPECT_EQ(static_cast<const int64_t*>(oa[0].data)[i], mn[g]);
        EXPECT_EQ(static_cast<const int64_t*>(oa[1].data)[i], mx[g]);
        // Bit-exact parity with the fallback's apply() replay (see above).
        int64_t got_bits = 0;
        std::memcpy(&got_bits, &static_cast<const int64_t*>(oa[3].data)[i], 8);
        EXPECT_EQ(got_bits, frep[g].a);
    }
}

// ---------------------------------------------------------------------------
// DISTINCT falls back per-window and stays correct across windows.
// ---------------------------------------------------------------------------

TEST(BoltGroupbyTwopass, DistinctFallsBackAndStaysCorrect) {
    constexpr int64_t N = 3000;
    static int64_t ks[N];
    static int64_t vs[N];
    for (int64_t i = 0; i < N; ++i) { ks[i] = i % 2; vs[i] = i % 10; }
    Arena a;
    BoltColumn kd = BoltColumn::make_flat(ks, nullptr, 0, BoltType::Int64);
    BoltColumn vd = BoltColumn::make_flat(vs, nullptr, 0, BoltType::Int64);
    AggSpec spec = make_spec(AggKind::Sum, 0, /*distinct=*/1);
    GroupbyTypedState st{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&st, &a, &kd, 1, &vd, 1,
                                                  &spec, 1, 4));
    BoltColumn k = BoltColumn::make_flat(ks, nullptr, N, BoltType::Int64);
    BoltColumn v = BoltColumn::make_flat(vs, nullptr, N, BoltType::Int64);
    groupby_agg_multi_key_typed_ingest(&st, &k, &v, nullptr, 0, N);
    ASSERT_FALSE(st.oom);
    BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
    ASSERT_EQ(ng, 2u);
    // Group 0 sees even values 0,2,4,6,8 (sum 20); group 1 odd (sum 25).
    const int64_t* outk = static_cast<const int64_t*>(ok[0].data);
    const int64_t* outv = static_cast<const int64_t*>(oa[0].data);
    for (uint32_t i = 0; i < ng; ++i) {
        EXPECT_EQ(outv[i], outk[i] == 0 ? 20 : 25);
    }
}

// ---------------------------------------------------------------------------
// Window split: one ingest of > kGbIngestWindow rows.
// ---------------------------------------------------------------------------

TEST(BoltGroupbyTwopass, SingleIngestPastWindowBoundary) {
    constexpr int64_t N = 70000;   // > 65536
    static int64_t ks[N];
    static int64_t vs[N];
    int64_t expect[3] = {0, 0, 0};
    for (int64_t i = 0; i < N; ++i) {
        ks[i] = i % 3; vs[i] = i;
        expect[i % 3] += i;
    }
    Arena a;
    BoltColumn kd = BoltColumn::make_flat(ks, nullptr, 0, BoltType::Int64);
    BoltColumn vd = BoltColumn::make_flat(vs, nullptr, 0, BoltType::Int64);
    AggSpec specs[2];
    specs[0] = make_spec(AggKind::Sum, 0);
    specs[1] = make_spec(AggKind::CountStar, 0);
    GroupbyTypedState st{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&st, &a, &kd, 1, &vd, 1,
                                                  specs, 2, 8));
    BoltColumn k = BoltColumn::make_flat(ks, nullptr, N, BoltType::Int64);
    BoltColumn v = BoltColumn::make_flat(vs, nullptr, N, BoltType::Int64);
    groupby_agg_multi_key_typed_ingest(&st, &k, &v, nullptr, 0, N);
    ASSERT_FALSE(st.oom);
    BoltColumn ok[1], oa[2];
    uint32_t ng = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
    ASSERT_EQ(ng, 3u);
    const int64_t* outk = static_cast<const int64_t*>(ok[0].data);
    const int64_t* outv = static_cast<const int64_t*>(oa[0].data);
    const int64_t* outc = static_cast<const int64_t*>(oa[1].data);
    for (uint32_t i = 0; i < ng; ++i) {
        EXPECT_EQ(outv[i], expect[outk[i]]);
        EXPECT_EQ(outc[i], (N / 3) + ((outk[i] < (N % 3)) ? 1 : 0));
    }
}

// ---------------------------------------------------------------------------
// Cap overflow: more groups than the (tight) SwissTable capacity must set
// state->oom and finalize must fail — same contract as the fallback.
// ---------------------------------------------------------------------------

TEST(BoltGroupbyTwopass, CapOverflowSetsOom) {
    constexpr int64_t N = 100000;
    static int64_t ks[N];
    static int64_t vs[N];
    for (int64_t i = 0; i < N; ++i) { ks[i] = i; vs[i] = 1; }
    Arena a;
    BoltColumn kd = BoltColumn::make_flat(ks, nullptr, 0, BoltType::Int64);
    BoltColumn vd = BoltColumn::make_flat(vs, nullptr, 0, BoltType::Int64);
    AggSpec spec = make_spec(AggKind::Sum, 0);
    GroupbyTypedState st{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&st, &a, &kd, 1, &vd, 1,
                                                  &spec, 1, /*hint=*/64));
    BoltColumn k = BoltColumn::make_flat(ks, nullptr, N, BoltType::Int64);
    BoltColumn v = BoltColumn::make_flat(vs, nullptr, N, BoltType::Int64);
    groupby_agg_multi_key_typed_ingest(&st, &k, &v, nullptr, 0, N);
    EXPECT_TRUE(st.oom);
    BoltColumn ok[1], oa[1];
    uint32_t ng = 0;
    EXPECT_FALSE(groupby_agg_multi_key_typed_finalize(&st, ok, oa, &ng));
}

}  // namespace
