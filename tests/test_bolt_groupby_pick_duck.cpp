// MaxBy / MinBy with kGbPickDuckBy (DuckDB arg_max / arg_min): NULL values and NULL orderings are skipped, the first of
// tied orderings wins. Every value x ordering type pair, one state fed in chunks and four partials merged in input
// order, against a row-at-a-time reference.
#include "bolt/join/bolt_groupby.h"
#include <gtest/gtest.h>
#include <cstring>
#include <string>
#include <vector>

namespace {
using namespace bolt;
using Z = std::size_t;
constexpr Z kRows = 6000;
constexpr Z kGroups = 7;

struct Col {
    BoltType t;
    std::vector<std::string> s;   // Utf8 values
    std::vector<double> d;        // Float64
    std::vector<std::int64_t> i;  // Int64
    std::vector<bool> ok;
};

std::uint64_t rng(std::uint64_t* st) { *st = *st * 6364136223846793005ull + 1442695040888963407ull; return *st >> 33; }

Col make_col(BoltType t, std::uint64_t seed, int dom) {
    Col c; c.t = t;
    std::uint64_t st = seed;
    for (Z r = 0; r < kRows; ++r) {
        const std::int64_t v = static_cast<std::int64_t>(rng(&st) % static_cast<std::uint64_t>(dom));
        c.ok.push_back(rng(&st) % 100 >= 15);
        c.i.push_back(v);
        c.d.push_back(static_cast<double>(v) * 0.5);
        c.s.push_back("padding-past-twelve-" + std::string(1, static_cast<char>('a' + static_cast<int>(v % 26))) + std::to_string(v));
    }
    return c;
}

BoltColumn build(Arena* a, const Col& c, Z from, Z n) {
    BoltColumn col = BoltColumn::make_flat_alloc(static_cast<std::int64_t>(n), c.t, a);
    std::size_t total = 0;
    if (c.t == BoltType::Utf8) for (Z r = 0; r < n; ++r) total += c.s[from + r].size();
    char* pool = static_cast<char*>(a->allocate(total + 16, 16));
    std::size_t at = 0;
    for (Z r = 0; r < n; ++r) {
        const Z g = from + r;
        if (c.t == BoltType::Int64) static_cast<std::int64_t*>(col.data)[r] = c.i[g];
        else if (c.t == BoltType::Float64) static_cast<double*>(col.data)[r] = c.d[g];
        else {
            const std::string& s = c.s[g];
            std::memcpy(pool + at, s.data(), s.size());
            StringView sv; std::memset(&sv, 0, sizeof sv);
            sv.length = static_cast<std::uint32_t>(s.size());
            std::memcpy(sv.prefix, pool + at, 4);
            sv.ref.buf_idx = 0; sv.ref.offset = static_cast<std::uint32_t>(at);
            static_cast<StringView*>(col.data)[r] = sv;
            at += s.size();
        }
    }
    if (c.t == BoltType::Utf8) { col.str_overflow_base = pool; }
    auto* vb = static_cast<std::uint8_t*>(a->allocate(n / 8 + 2, 8));
    std::memset(vb, 0, n / 8 + 2);
    for (Z r = 0; r < n; ++r) if (c.ok[from + r]) vb[r >> 3] = static_cast<std::uint8_t>(vb[r >> 3] | (1u << (r & 7)));
    col.validity = vb; col.validity_offset = 0; col.stats.all_valid = false;
    return col;
}

std::string text(const Col& c, Z r) {
    return c.t == BoltType::Int64 ? std::to_string(c.i[r]) : c.t == BoltType::Float64 ? std::to_string(c.d[r]) : c.s[r];
}

bool begin(GroupbyTypedState* st, Arena* a, BoltType vt, BoltType ot, AggKind k, std::uint8_t flags) {
    std::memset(st, 0, sizeof(*st));
    BoltColumn kd{}, pay[2]{};
    kd.type = BoltType::Int64; pay[0].type = vt; pay[1].type = ot;
    AggSpec sp[2]{};
    sp[0].kind = k; sp[0].in_col = 0; sp[0].aux = 1; sp[0].flags = flags;
    sp[1].kind = AggKind::ByOrd; sp[1].in_col = 1;
    return groupby_agg_multi_key_typed_begin(st, a, &kd, 1, pay, 2, sp, 2, 16, 0);
}

bool ingest(GroupbyTypedState* st, Arena* a, const Col& v, const Col& o, Z from, Z n) {
    BoltColumn key = BoltColumn::make_flat_alloc(static_cast<std::int64_t>(n), BoltType::Int64, a);
    for (Z r = 0; r < n; ++r) static_cast<std::int64_t*>(key.data)[r] = static_cast<std::int64_t>((from + r) % kGroups);
    const BoltColumn pay[2] = {build(a, v, from, n), build(a, o, from, n)};
    groupby_agg_multi_key_typed_ingest(st, &key, pay, nullptr, 0, static_cast<std::int64_t>(n));
    return !st->oom && !st->null_key;
}

std::string cell(const GroupbyTypedState* st, std::int64_t g, const Col& v) {
    for (std::uint32_t gid = 0; gid < st->entry_count; ++gid) {
        if (st->keys_flat[gid].a != g) continue;
        if (st->counts[gid] != kGbPickValue) return "<null>";
        const GbCell16 c = st->accums[gid];
        if (v.t == BoltType::Int64) return std::to_string(static_cast<std::int64_t>(c.a));
        if (v.t == BoltType::Float64) { double d; std::memcpy(&d, &c.a, 8); return std::to_string(d); }
        StringView sv; std::memcpy(&sv, &c, sizeof sv);
        return std::string(kernels::utf8::sv_bytes(sv, st->agg_str_buf[0]), sv.length);
    }
    return "<none>";
}

// DuckDB: skip NULL ordering and NULL value; keep the first of ties.
std::string reference(const Col& v, const Col& o, Z g, bool max) {
    std::int64_t best = -1;
    for (Z r = 0; r < kRows; ++r) {
        if (r % kGroups != g || !o.ok[r] || !v.ok[r]) continue;
        if (best < 0) { best = static_cast<std::int64_t>(r); continue; }
        const Z b = static_cast<Z>(best);
        const bool better = o.t == BoltType::Utf8 ? (max ? o.s[r] > o.s[b] : o.s[r] < o.s[b])
                          : o.t == BoltType::Float64 ? (max ? o.d[r] > o.d[b] : o.d[r] < o.d[b])
                          : (max ? o.i[r] > o.i[b] : o.i[r] < o.i[b]);
        if (better) best = static_cast<std::int64_t>(r);
    }
    return best < 0 ? "<null>" : text(v, static_cast<Z>(best));
}

class Duck : public ::testing::TestWithParam<std::tuple<int, int, bool>> {};

TEST_P(Duck, MatchesReferenceOnePartitionAndMergedPartials) {
    const BoltType ts[3] = {BoltType::Int64, BoltType::Float64, BoltType::Utf8};
    const BoltType vt = ts[std::get<0>(GetParam())], ot = ts[std::get<1>(GetParam())];
    const bool max = std::get<2>(GetParam());
    const AggKind k = max ? AggKind::MaxBy : AggKind::MinBy;
    const Col v = make_col(vt, 11, 40), o = make_col(ot, 29, 9);   // 9 distinct orderings: ties everywhere
    Arena a{};
    GroupbyTypedState one{};
    ASSERT_TRUE(begin(&one, &a, vt, ot, k, kGbPickDuckBy));
    for (Z from = 0; from < kRows; from += 1000) ASSERT_TRUE(ingest(&one, &a, v, o, from, 1000));
    GroupbyTypedState part[4]{}, merged{};
    ASSERT_TRUE(begin(&merged, &a, vt, ot, k, kGbPickDuckBy));
    for (Z p = 0; p < 4; ++p) {
        ASSERT_TRUE(begin(&part[p], &a, vt, ot, k, kGbPickDuckBy));
        ASSERT_TRUE(ingest(&part[p], &a, v, o, p * 1500, 1500));
        ASSERT_TRUE(groupby_agg_multi_key_typed_merge(&merged, &part[p]));
    }
    for (Z g = 0; g < kGroups; ++g) {
        const std::string want = reference(v, o, g, max);
        EXPECT_EQ(cell(&one, static_cast<std::int64_t>(g), v), want) << "one state, group " << g;
        EXPECT_EQ(cell(&merged, static_cast<std::int64_t>(g), v), want) << "merged partials, group " << g;
    }
}

TEST_P(Duck, SparkFlagsKeepNullValueAndLaterTie) {
    const BoltType ts[3] = {BoltType::Int64, BoltType::Float64, BoltType::Utf8};
    const BoltType vt = ts[std::get<0>(GetParam())], ot = ts[std::get<1>(GetParam())];
    const AggKind k = std::get<2>(GetParam()) ? AggKind::MaxBy : AggKind::MinBy;
    const Col v = make_col(vt, 11, 40), o = make_col(ot, 29, 9);
    Arena a{};
    GroupbyTypedState st{};
    ASSERT_TRUE(begin(&st, &a, vt, ot, k, 0));
    ASSERT_TRUE(ingest(&st, &a, v, o, 0, kRows));
    bool differs = false;
    for (Z g = 0; g < kGroups; ++g) differs |= cell(&st, static_cast<std::int64_t>(g), v) != reference(v, o, g, std::get<2>(GetParam()));
    EXPECT_TRUE(differs) << "the unflagged lane must keep Spark's semantics";
}

INSTANTIATE_TEST_SUITE_P(Types, Duck,
    ::testing::Combine(::testing::Range(0, 3), ::testing::Range(0, 3), ::testing::Bool()));
}  // namespace
