// test_bolt_column_codec.cpp — column codec chains round-trip every stage and
// chain over fuzzed ints and strings; truncated input refuses.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/codec/bolt_column_codec.h"

using namespace bolt;
namespace C = bolt::codec;

namespace {

const char* const kIntChains[] = {
    "none", "delta", "for", "delta,for", "rle", "rle,for", "rle,delta,for", "delta,for,lz4",
    "for,zstd", "delta,zstdself", "zstd(9)", "lz4", "delta,rle,for,zstd",
};
const char* const kStrChains[] = {
    "none", "dict", "front", "template", "template(1)", "template(2)", "template(3)",
    "dict,for", "dict,front,for,lz4", "front,for,zstd", "template,dict,for,zstd",
    "template(2),dict,delta,for,zstdself", "template,front,rle,for,lz4", "zstd", "lz4",
    "template(3),dict,front,delta,for,zstd(1)",
};
const uint64_t kSizes[] = {0, 1, 2, 5, 1023, 1024, 1025, 3000};

struct Strs {
    std::vector<uint32_t> offs{0};
    std::vector<uint8_t> bytes;
    void add(const std::string& s) {
        bytes.insert(bytes.end(), s.begin(), s.end());
        offs.push_back(static_cast<uint32_t>(bytes.size()));
    }
    C::Stream stream() const { return C::strs_stream(offs.data(), bytes.data(), offs.size() - 1); }
};

std::vector<int64_t> gen_ints(std::mt19937_64& g, uint64_t n, int shape) {
    std::vector<int64_t> v(n);
    int64_t acc = static_cast<int64_t>(g() % 1000000);
    for (uint64_t i = 0; i < n; ++i) {
        switch (shape) {
            case 0: v[i] = static_cast<int64_t>(g()); break;
            case 1: v[i] = static_cast<int64_t>(g() % 7) - 3; break;
            case 2: acc += static_cast<int64_t>(g() % 20); v[i] = acc; break;
            case 3: v[i] = static_cast<int64_t>((i / 37) % 5); break;
            case 4: v[i] = (g() & 1) ? std::numeric_limits<int64_t>::min() : std::numeric_limits<int64_t>::max(); break;
            default: v[i] = 42; break;
        }
    }
    return v;
}

std::string gen_str(std::mt19937_64& g, int shape) {
    static const char* const kWords[] = {"GET", "POST", "/images/", "hm_bg.jpg", "HTTP/1.0", "error",
                                         "user=", "took", "ms", " ", "::", "a.b-c_d", "id-", "x9y"};
    static const char* const kNums[] = {"0", "7", "007", "200", "999999999999999999",
                                        "1000000000000000000", "12345678901234567890", "-5", "1.0", "0x1f"};
    std::string s;
    const int parts = static_cast<int>(g() % 9);
    for (int i = 0; i < parts; ++i) {
        const uint64_t r = g() % 10;
        if (shape == 1) { s.push_back(static_cast<char>(g() % 256)); continue; }
        if (shape == 2 && r < 3) { s.push_back(static_cast<char>(0x10 + g() % 3)); continue; }
        if (r < 4) s += kWords[g() % 14];
        else if (r < 7) s += kNums[g() % 10];
        else if (r < 8) s += std::to_string(g() % 100000);
        else s += " ";
    }
    return s;
}

Strs gen_strs(std::mt19937_64& g, uint64_t n, int shape) {
    Strs s;
    std::vector<std::string> rows(n);
    for (auto& r : rows) r = gen_str(g, shape);
    if (shape == 3) std::sort(rows.begin(), rows.end());
    if (shape == 4) for (auto& r : rows) r = rows[g() % (n < 4 ? n : 4)];
    for (const auto& r : rows) s.add(r);
    return s;
}

bool same(const C::Stream& a, const C::Stream& b) {
    if (a.n != b.n) return false;
    if (a.kind == C::Kind::kInts) return a.n == 0 || std::memcmp(a.ints, b.ints, a.n * 8) == 0;
    return std::memcmp(a.offs, b.offs, (a.n + 1) * 4) == 0 &&
           (a.nbytes == 0 || std::memcmp(a.bytes, b.bytes, a.nbytes) == 0);
}

void round_trip(const C::Stream& in, const char* chain_text, bool check_truncation) {
    C::Chain chain;
    ASSERT_TRUE(C::chain_parse(chain_text, &chain)) << chain_text;
    Arena a;
    C::Buf out; out.a = &a;
    ASSERT_TRUE(C::encode_column(in, chain, a, &out)) << chain_text;
    C::Stream back;
    ASSERT_TRUE(C::decode_column(out.p, out.n, a, &back)) << chain_text << " n=" << in.n;
    ASSERT_EQ(back.kind, in.kind) << chain_text;
    ASSERT_TRUE(same(in, back)) << chain_text << " n=" << in.n;
    if (!check_truncation) return;
    for (uint64_t cut = 0; cut < out.n; cut += 1 + out.n / 64) {
        C::Stream bad;
        ASSERT_FALSE(C::decode_column(out.p, cut, a, &bad)) << chain_text << " cut=" << cut;
    }
}

TEST(ColumnCodec, IntChainsRoundTrip) {
    std::mt19937_64 g(1);
    for (const char* chain : kIntChains)
        for (uint64_t n : kSizes)
            for (int shape = 0; shape < 6; ++shape) {
                const auto v = gen_ints(g, n, shape);
                round_trip(C::ints_stream(v.data(), n), chain, n == 1025);
            }
}

TEST(ColumnCodec, StringChainsRoundTrip) {
    std::mt19937_64 g(2);
    for (const char* chain : kStrChains)
        for (uint64_t n : kSizes)
            for (int shape = 0; shape < 5; ++shape) {
                const Strs s = gen_strs(g, n, shape);
                round_trip(s.stream(), chain, n == 5);
            }
}

TEST(ColumnCodec, TemplateSplitsRepeatedLinesIntoOneTemplate) {
    Strs s;
    for (int i = 0; i < 2000; ++i)
        s.add("GET /images/img" + std::to_string(i % 50) + ".gif HTTP/1.0 took " + std::to_string(i * 7) + " ms");
    Arena a;
    C::Chain plain, tpl;
    ASSERT_TRUE(C::chain_parse("none", &plain));
    ASSERT_TRUE(C::chain_parse("template,dict,for", &tpl));
    C::Buf p, t; p.a = &a; t.a = &a;
    ASSERT_TRUE(C::encode_column(s.stream(), plain, a, &p));
    ASSERT_TRUE(C::encode_column(s.stream(), tpl, a, &t));
    EXPECT_LT(t.n * 8, p.n);
    C::Stream back;
    ASSERT_TRUE(C::decode_column(t.p, t.n, a, &back));
    EXPECT_TRUE(same(s.stream(), back));
}

TEST(ColumnCodec, ABlockCodecAloneReachesStringAndIntPayloads) {
    Strs s; std::vector<int64_t> v;
    for (int i = 0; i < 4000; ++i) { s.add("the same line again and again"); v.push_back(i % 3); }
    for (const char* text : {"lz4", "zstd", "zstdself"}) {
        Arena a;
        C::Chain plain, blk;
        ASSERT_TRUE(C::chain_parse("none", &plain));
        ASSERT_TRUE(C::chain_parse(text, &blk));
        C::Buf p, b, pi, bi; p.a = &a; b.a = &a; pi.a = &a; bi.a = &a;
        ASSERT_TRUE(C::encode_column(s.stream(), plain, a, &p));
        ASSERT_TRUE(C::encode_column(s.stream(), blk, a, &b));
        ASSERT_TRUE(C::encode_column(C::ints_stream(v.data(), v.size()), plain, a, &pi));
        ASSERT_TRUE(C::encode_column(C::ints_stream(v.data(), v.size()), blk, a, &bi));
        EXPECT_LT(b.n * 10, p.n) << text;
        EXPECT_LT(bi.n * 10, pi.n) << text;
    }
}

TEST(ColumnCodec, ChainParseRefusesUnknownAndMalformed) {
    C::Chain c;
    EXPECT_TRUE(C::chain_parse("", &c)); EXPECT_EQ(c.n, 0u);
    EXPECT_TRUE(C::chain_parse("zstd(19),lz4", &c)); EXPECT_EQ(c.n, 2u); EXPECT_EQ(c.st[0].param, 19u);
    EXPECT_FALSE(C::chain_parse("gorilla", &c));
    EXPECT_FALSE(C::chain_parse("zstd(", &c));
    EXPECT_FALSE(C::chain_parse("zstd(x)", &c));
    EXPECT_FALSE(C::chain_parse("dict lz4", &c));
    EXPECT_FALSE(C::chain_parse("lz4,lz4,lz4,lz4,lz4,lz4,lz4,lz4,lz4", &c));
}

}  // namespace
