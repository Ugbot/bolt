// test_bolt_constant_detect.cpp — MSEG K4: constant / all-null page detection
// against a naive reference, and the Constant page it emits (values and a
// bolt_wire round trip).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/kernels/bolt_page_detect.h"
#include "bolt/wire/bolt_wire.h"

using namespace bolt;
using namespace bolt::page;
using namespace bolt::wire;

namespace {

constexpr uint32_t kSeeds = 24;

bool is_str(BoltType t) { return t == BoltType::Utf8 || t == BoltType::Binary || t == BoltType::Symbol; }

struct Page {
    BoltType type = BoltType::Int64;
    uint32_t n = 0, w = 0;
    std::vector<uint8_t> data;        // n * w (strings: StringViews)
    std::vector<uint8_t> validity;    // empty = all valid; else (n+7)/8 + 8
    std::vector<char> overflow;       // long strings
    std::vector<std::string> strs;    // strings: the row values (garbage for nulls)

    bool valid(uint32_t r) const { return validity.empty() || ((validity[r >> 3] >> (r & 7)) & 1); }
    const uint8_t* vptr() const { return validity.empty() ? nullptr : validity.data(); }
};

// ---- reference -----------------------------------------------------------------

bool is_nan(BoltType t, const uint8_t* p) {
    if (t == BoltType::Float32) { float f; std::memcpy(&f, p, 4); return std::isnan(f); }
    if (t == BoltType::Float64) { double f; std::memcpy(&f, p, 8); return std::isnan(f); }
    return false;
}

PageKind ref_kind(const Page& pg) {
    uint32_t nulls = 0;
    int first = -1;
    bool same = true;
    for (uint32_t r = 0; r < pg.n; ++r) {
        if (!pg.valid(r)) { ++nulls; continue; }
        if (!is_str(pg.type) && is_nan(pg.type, &pg.data[size_t(r) * pg.w])) same = false;
        if (first < 0) { first = int(r); continue; }
        if (is_str(pg.type)) same = same && pg.strs[r] == pg.strs[size_t(first)];
        else same = same && std::memcmp(&pg.data[size_t(r) * pg.w], &pg.data[size_t(first) * pg.w], pg.w) == 0;
    }
    if (pg.n > 0 && nulls == pg.n) return PageKind::kAllNull;
    return first >= 0 && same ? PageKind::kConstant : PageKind::kFlat;
}

// ---- generators ------------------------------------------------------------------

struct Gen {
    std::mt19937_64 g;
    explicit Gen(uint64_t s) : g(s) {}
    uint64_t below(uint64_t k) { return g() % k; }
};

void fixed_pool(BoltType t, uint32_t w, Gen& r, std::vector<std::vector<uint8_t>>* pool) {
    pool->clear();
    auto put = [&](const void* p) {
        std::vector<uint8_t> v(w);
        std::memcpy(v.data(), p, w);
        for (const auto& e : *pool) if (e == v) return;   // narrow widths alias edges
        pool->push_back(v);
    };
    if (t == BoltType::Float32) {
        const float f[] = {0.0f, -0.0f, std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::quiet_NaN(),
                           1.5f, std::numeric_limits<float>::infinity()};
        for (const float& x : f) put(&x);
    } else if (t == BoltType::Float64) {
        const double f[] = {0.0, -0.0, std::numeric_limits<double>::quiet_NaN(), -std::numeric_limits<double>::quiet_NaN(),
                            1.5, -std::numeric_limits<double>::infinity()};
        for (const double& x : f) put(&x);
    } else {
        const uint64_t edge[] = {0, 1, ~0ull, 0x80ull << 56, 0x7Full};
        for (uint64_t e : edge) { uint8_t b[16]; std::memcpy(b, &e, 8); std::memcpy(b + 8, &e, 8); put(b); }
        for (int k = 0; k < 3; ++k) { uint8_t b[16]; for (auto& x : b) x = uint8_t(r.g()); put(b); }
    }
}

const char* kStrPool[] = {"", "a", "abc", "abcd", "abcde", "hello world!", "hello world!!",
                          "a long string of forty characters......", "a long string of forty characters.....!",
                          "!a long string of forty characters......"};
constexpr uint32_t kStrPoolN = sizeof(kStrPool) / sizeof(kStrPool[0]);

void put_str(Page* pg, uint32_t r, const std::string& s, Gen& g) {
    StringView sv;
    for (size_t b = 0; b < sizeof(sv); ++b) reinterpret_cast<uint8_t*>(&sv)[b] = uint8_t(g.g());  // garbage padding
    sv.length = uint32_t(s.size());
    std::memcpy(sv.prefix, s.data(), s.size() < 4 ? s.size() : 4);
    if (s.size() <= 12) {
        if (s.size() > 4) std::memcpy(sv.inline_data, s.data() + 4, s.size() - 4);
    } else {
        sv.ref.buf_idx = uint32_t(g.below(3));
        sv.ref.offset = uint32_t(pg->overflow.size());   // every row its own copy
        pg->overflow.insert(pg->overflow.end(), s.begin(), s.end());
    }
    std::memcpy(&pg->data[size_t(r) * 16], &sv, 16);
    pg->strs[r] = s;
}

// shape: 0 const all-valid, 1 const + nulls, 2 all null, 3 random, 4 const + one
// differing value at `pos`, 5 const + nulls whose slots hold other values.
Page make_page(BoltType t, uint32_t n, int shape, uint32_t pos, Gen& g) {
    Page pg;
    pg.type = t; pg.n = n; pg.w = is_str(t) ? 16 : uint32_t(type_size(t));
    pg.data.assign(size_t(n) * pg.w, 0);
    if (is_str(t)) pg.strs.assign(n, "");
    const bool nulls = shape == 1 || shape == 2 || shape == 5 || (shape >= 3 && g.below(2));
    if (nulls) {
        pg.validity.assign((n + 7) / 8 + 8, 0);
        for (uint32_t r = 0; r < n; ++r)
            if (shape != 2 && (g.below(3) != 0 || (shape == 4 && r == pos)))
                pg.validity[r >> 3] |= uint8_t(1u << (r & 7));
    }
    std::vector<std::vector<uint8_t>> pool;
    if (!is_str(t)) fixed_pool(t, pg.w, g, &pool);
    const uint32_t np = is_str(t) ? kStrPoolN : uint32_t(pool.size());
    const uint32_t base = uint32_t(g.below(np));
    uint32_t other = uint32_t(g.below(np - 1));
    if (other >= base) ++other;
    for (uint32_t r = 0; r < n; ++r) {
        uint32_t k = base;
        if (shape == 3) k = uint32_t(g.below(np));
        if (shape == 4 && r == pos) k = other;
        if (shape == 5 && !pg.valid(r)) k = uint32_t(g.below(np));
        if (is_str(t)) put_str(&pg, r, kStrPool[k], g);
        else std::memcpy(&pg.data[size_t(r) * pg.w], pool[k].data(), pg.w);
    }
    return pg;
}

// ---- detectors under test ----------------------------------------------------------

// Feeds the page in random chunks; returns the kind, fills value and dst.
PageKind run_detector(const Page& pg, Gen& g, uint8_t value[16], std::vector<uint8_t>* dst,
                      uint32_t* nulls) {
    dst->assign(pg.data.size() + 16, 0xAB);
    const bool k7 = !is_str(pg.type) && stats::stats_kind(pg.type) != stats::StatsKind::kNone;
    stats::StatsFusedState st7; FixedDetectState stf; SvDetectState sts;
    if (k7) EXPECT_EQ(stats::stats_fused_init(&st7, pg.type), stats::StatsStatus::kOk);
    else if (is_str(pg.type)) sv_detect_init(&sts);
    else EXPECT_EQ(fixed_detect_init(&stf, pg.w), DetectStatus::kOk);
    for (uint32_t at = 0; at < pg.n;) {
        const uint32_t c = std::min<uint32_t>(pg.n - at, 1 + uint32_t(g.below(g.below(2) ? 7 : 3000)));
        const uint8_t* s = pg.data.data() + size_t(at) * pg.w;
        uint8_t* d = dst->data() + size_t(at) * pg.w;
        if (k7) EXPECT_EQ(stats::stats_fused_update(&st7, s, pg.vptr(), at, c, d), stats::StatsStatus::kOk);
        else if (is_str(pg.type))
            EXPECT_EQ(sv_detect_update(&sts, reinterpret_cast<const StringView*>(s), pg.vptr(), at, c,
                                       reinterpret_cast<StringView*>(d), pg.overflow.data()), DetectStatus::kOk);
        else EXPECT_EQ(fixed_detect_update(&stf, s, pg.vptr(), at, c, d), DetectStatus::kOk);
        at += c;
    }
    if (k7) {
        const stats::FusedStats fs = stats::stats_fused_finish(st7);
        page_value_from_stats(fs, value);
        *nulls = fs.null_count;
        return page_kind_from_stats(fs);
    }
    if (is_str(pg.type)) {
        StringView v; const PageKind k = sv_detect_finish(sts, &v);
        std::memcpy(value, &v, 16); *nulls = sts.nulls; return k;
    }
    *nulls = stf.nulls;
    return fixed_detect_finish(stf, value);
}

std::string row_bytes(const Page& pg, uint32_t r) {
    if (is_str(pg.type)) return pg.strs[r];
    return std::string(reinterpret_cast<const char*>(&pg.data[size_t(r) * pg.w]), pg.w);
}

std::string const_bytes(const BoltColumn& c) {
    if (is_str(c.type)) {
        StringView sv; std::memcpy(&sv, c.inline_value, 16);
        if (sv.length <= 12) return std::string(sv.prefix, sv.length);
        return std::string(static_cast<const char*>(c.str_overflow_base) + sv.ref.offset, sv.length);
    }
    return std::string(reinterpret_cast<const char*>(c.inline_value), type_size(c.type));
}

void check_copy(const Page& pg, const std::vector<uint8_t>& dst) {
    for (uint32_t r = 0; r < pg.n; ++r) {
        const uint8_t* d = &dst[size_t(r) * pg.w];
        if (!pg.valid(r)) {
            for (uint32_t b = 0; b < pg.w; ++b) ASSERT_EQ(d[b], 0) << "null slot " << r;
        } else {
            ASSERT_EQ(std::memcmp(d, &pg.data[size_t(r) * pg.w], pg.w), 0) << "row " << r;
        }
    }
}

// The emitted Constant column reads back as the page, directly and after wire.
void check_emit(const Page& pg, PageKind k, const uint8_t value[16], uint32_t nulls) {
    std::vector<uint8_t> vbits = pg.validity;
    BoltColumn c = page_make_constant(pg.type, k, value, pg.n, vbits.empty() ? nullptr : vbits.data(), 0,
                                      nulls, const_cast<char*>(pg.overflow.data()));
    ASSERT_EQ(c.format, ColumnFormat::Constant);
    for (uint32_t r = 0; r < pg.n; ++r) {
        ASSERT_EQ(c.is_null(r), !pg.valid(r)) << r;
        if (pg.valid(r)) ASSERT_EQ(const_bytes(c), row_bytes(pg, r)) << r;
    }
    Arena a;
    BoltBatch b;
    BoltBatch::init_empty(&b);
    ASSERT_TRUE(BoltBatch::alloc_columns(&b, &a, 1));
    b.num_rows = pg.n;
    b.schema.num_fields = 1;
    std::memset(&b.schema.fields[0], 0, sizeof(BoltField));
    b.schema.fields[0].set_name("c");
    b.schema.fields[0].type = pg.type;
    b.schema.fields[0].nullable = c.validity != nullptr;
    b.columns[0][0] = c;
    b.columns[1][0] = c;
    const size_t sz = bolt_wire_size(&b);
    ASSERT_GT(sz, 0u);
    std::vector<uint8_t> raw(sz + 64);
    uint8_t* buf = raw.data() + ((64 - (reinterpret_cast<uintptr_t>(raw.data()) & 63)) & 63);
    ASSERT_EQ(bolt_wire_serialize(&b, buf, sz), sz);
    BoltBatch out;
    ASSERT_TRUE(bolt_wire_deserialize(buf, sz, &out, &a));
    const BoltColumn& o = out.columns[0][0];
    ASSERT_EQ(o.format, ColumnFormat::Constant);
    for (uint32_t r = 0; r < pg.n; ++r) {
        ASSERT_EQ(o.is_null(r), !pg.valid(r)) << "wire row " << r;
        if (pg.valid(r)) ASSERT_EQ(const_bytes(o), row_bytes(pg, r)) << "wire row " << r;
    }
}

void check_page(const Page& pg, Gen& g, const char* what) {
    SCOPED_TRACE(std::string(what) + " type=" + std::to_string(int(pg.type)) + " n=" + std::to_string(pg.n));
    uint8_t value[16]; uint32_t nulls = 0;
    std::vector<uint8_t> dst;
    const PageKind got = run_detector(pg, g, value, &dst, &nulls);
    ASSERT_EQ(got, ref_kind(pg));
    check_copy(pg, dst);
    if (got != PageKind::kFlat) check_emit(pg, got, value, nulls);
}

const BoltType kTypes[] = {BoltType::Int8, BoltType::Int16, BoltType::Int32, BoltType::Int64,
                           BoltType::UInt8, BoltType::UInt64, BoltType::Float32, BoltType::Float64,
                           BoltType::Date32, BoltType::Timestamp, BoltType::Decimal64,
                           BoltType::Decimal128, BoltType::Bool, BoltType::Float16, BoltType::UUID,
                           BoltType::IPv4, BoltType::FixedSizeBinary, BoltType::Utf8, BoltType::Binary};
const uint32_t kSizes[] = {1, 2, 7, 64, 1023, 1024, 1025, 4100};

}  // namespace

TEST(ConstantDetect, FuzzEveryTypeAndShape) {
    for (uint32_t seed = 0; seed < kSeeds; ++seed) {
        Gen g(0xC0457A47ull + seed);
        for (BoltType t : kTypes)
            for (uint32_t n : kSizes)
                for (int shape = 0; shape < 6; ++shape) {
                    const uint32_t pos = uint32_t(g.below(n));
                    check_page(make_page(t, n, shape, pos, g), g, "fuzz");
                }
    }
}

// A single differing value at the first, middle and last position (and, on a
// small page, every position) is never a Constant.
TEST(ConstantDetect, OneDifferingValueAtEveryPositionIsFlat) {
    Gen g(7);
    for (BoltType t : kTypes)
        for (uint32_t n : {2u, 33u, 1024u, 4100u}) {
            std::vector<uint32_t> at = {0, n / 2, n - 1};
            if (n <= 33) for (uint32_t p = 0; p < n; ++p) at.push_back(p);
            for (uint32_t p : at) {
                for (int with_nulls = 0; with_nulls < 2; ++with_nulls) {
                    Page pg = make_page(t, n, 4, p, g);
                    pg.validity.clear();
                    if (with_nulls) {   // every third row null, the differing row and one other valid
                        pg.validity.assign((n + 7) / 8 + 8, 0);
                        const uint32_t o = p == 0 ? 1 : 0;
                        for (uint32_t r = 0; r < n; ++r)
                            if (r % 3 != 2 || r == p || r == o) pg.validity[r >> 3] |= uint8_t(1u << (r & 7));
                    }
                    ASSERT_EQ(ref_kind(pg), PageKind::kFlat);
                    uint8_t value[16]; uint32_t nulls; std::vector<uint8_t> dst;
                    SCOPED_TRACE("type=" + std::to_string(int(t)) + " n=" + std::to_string(n) +
                                 " pos=" + std::to_string(p) + " nulls=" + std::to_string(with_nulls));
                    ASSERT_EQ(run_detector(pg, g, value, &dst, &nulls), PageKind::kFlat);
                }
            }
        }
}

TEST(ConstantDetect, FloatSpecials) {
    Gen g(11);
    auto page_of = [](const std::vector<double>& v) {
        Page pg; pg.type = BoltType::Float64; pg.n = uint32_t(v.size()); pg.w = 8;
        pg.data.resize(v.size() * 8); std::memcpy(pg.data.data(), v.data(), pg.data.size());
        return pg;
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    uint8_t value[16]; uint32_t nulls; std::vector<uint8_t> dst;
    EXPECT_EQ(run_detector(page_of({-0.0, -0.0, -0.0}), g, value, &dst, &nulls), PageKind::kConstant);
    EXPECT_EQ(run_detector(page_of({-0.0, -0.0, 0.0}), g, value, &dst, &nulls), PageKind::kFlat);
    EXPECT_EQ(run_detector(page_of({0.0, -0.0, 0.0}), g, value, &dst, &nulls), PageKind::kFlat);
    EXPECT_EQ(run_detector(page_of({nan, nan, nan}), g, value, &dst, &nulls), PageKind::kFlat);
    EXPECT_EQ(run_detector(page_of({1.0, 1.0, nan}), g, value, &dst, &nulls), PageKind::kFlat);
    std::vector<double> big(2048, -0.0);
    EXPECT_EQ(run_detector(page_of(big), g, value, &dst, &nulls), PageKind::kConstant);
    big[1500] = 0.0;
    EXPECT_EQ(run_detector(page_of(big), g, value, &dst, &nulls), PageKind::kFlat);
}

TEST(ConstantDetect, EmptyStringsAndLongStringsByBytes) {
    Gen g(13);
    auto page_of = [&](const std::vector<std::string>& v) {
        Page pg; pg.type = BoltType::Utf8; pg.n = uint32_t(v.size()); pg.w = 16;
        pg.data.assign(v.size() * 16, 0); pg.strs.assign(v.size(), "");
        for (uint32_t r = 0; r < pg.n; ++r) put_str(&pg, r, v[r], g);
        return pg;
    };
    uint8_t value[16]; uint32_t nulls; std::vector<uint8_t> dst;
    const std::string L(40, 'x');
    EXPECT_EQ(run_detector(page_of({"", "", ""}), g, value, &dst, &nulls), PageKind::kConstant);
    EXPECT_EQ(run_detector(page_of({"", "", "a"}), g, value, &dst, &nulls), PageKind::kFlat);
    EXPECT_EQ(run_detector(page_of({"a", "", ""}), g, value, &dst, &nulls), PageKind::kFlat);
    EXPECT_EQ(run_detector(page_of({L, L, L}), g, value, &dst, &nulls), PageKind::kConstant);
    EXPECT_EQ(run_detector(page_of({L, L, L.substr(0, 39) + "y"}), g, value, &dst, &nulls), PageKind::kFlat);
    EXPECT_EQ(run_detector(page_of({"abc", "abd"}), g, value, &dst, &nulls), PageKind::kFlat);
}

TEST(ConstantDetect, EmptyPageIsFlatAndBadWidthsRefused) {
    FixedDetectState st;
    EXPECT_EQ(fixed_detect_init(&st, 32), DetectStatus::kUnsupportedWidth);
    EXPECT_EQ(fixed_detect_init(&st, 3), DetectStatus::kUnsupportedWidth);
    ASSERT_EQ(fixed_detect_init(&st, 4), DetectStatus::kOk);
    uint8_t value[16];
    EXPECT_EQ(fixed_detect_finish(st, value), PageKind::kFlat);
    SvDetectState sv;
    sv_detect_init(&sv);
    StringView v;
    EXPECT_EQ(sv_detect_finish(sv, &v), PageKind::kFlat);
}
