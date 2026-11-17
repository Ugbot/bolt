// test_bolt_kv_packed.cpp: kv_packed_get against literals and against a
// per-pair reference walk written here (not the kernel's search).

#include <gtest/gtest.h>

#include "bolt/kernels/bolt_kv_packed.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace kv = bolt::kernels::kv_packed;
using bolt::StringView;

namespace {
constexpr char KV = '\x1f', PR = '\x1e';

struct Col {
    std::vector<StringView> v;
    std::string bytes;
    void add(const std::string& s) {
        StringView r;
        std::memset(&r, 0, sizeof(r));
        r.length = static_cast<uint32_t>(s.size());
        if (s.size() <= 12) {
            std::memcpy(r.prefix, s.data(), s.size());   // prefix + inline_data are contiguous
        } else {
            std::memcpy(r.prefix, s.data(), 4);
            r.ref.offset = static_cast<uint32_t>(bytes.size());
            bytes += s;
        }
        v.push_back(r);
    }
};

std::string text(const StringView& s, const char* base) {
    return std::string(s.length <= 12 ? s.prefix : base + s.ref.offset, s.length);
}

std::string pairs(std::initializer_list<std::pair<const char*, const char*>> kv_) {
    std::string s;
    for (const auto& p : kv_) { s += p.first; s += KV; s += p.second; s += PR; }
    return s;
}

// The last pair named `key` of a set of whole pairs; "" otherwise.
std::string reference(const std::string& s, const std::string& key) {
    if (s.empty() || s.back() != PR) return "";
    std::string out;
    size_t pos = 0;
    while (pos < s.size()) {
        const size_t e = s.find(PR, pos);
        const std::string pair = s.substr(pos, e - pos);
        if (pair.size() > key.size() && pair.compare(0, key.size(), key) == 0 && pair[key.size()] == KV)
            out = pair.substr(key.size() + 1);
        pos = e + 1;
    }
    return out;
}
}  // namespace

TEST(KvPackedGet, FindsFirstMiddleLastAndAbsent) {
    Col c;
    c.add(pairs({{"__name__", "http_requests_total"}, {"job", "api"}, {"le", "0.5"}, {"span_name", "GET /a/long/route/name"}}));
    c.add(pairs({{"job", "x"}}));
    c.add("");
    c.add(pairs({{"le", "+Inf"}}));
    std::vector<StringView> out(c.v.size());
    EXPECT_EQ(kv::kv_packed_get(c.v.data(), c.bytes.data(), 4, "__name__", 8, KV, PR, out.data()), 1);
    EXPECT_EQ(text(out[0], c.bytes.data()), "http_requests_total");
    EXPECT_EQ(out[1].length, 0u);
    EXPECT_EQ(out[2].length, 0u);
    EXPECT_EQ(kv::kv_packed_get(c.v.data(), c.bytes.data(), 4, "le", 2, KV, PR, out.data()), 2);
    EXPECT_EQ(text(out[0], c.bytes.data()), "0.5");
    EXPECT_EQ(text(out[3], c.bytes.data()), "+Inf");
    EXPECT_EQ(kv::kv_packed_get(c.v.data(), c.bytes.data(), 4, "span_name", 9, KV, PR, out.data()), 1);
    EXPECT_EQ(text(out[0], c.bytes.data()), "GET /a/long/route/name");
    EXPECT_GT(out[0].length, 12u);                       // a view of the row's own bytes
    EXPECT_EQ(out[0].ref.offset, c.v[0].ref.offset + static_cast<uint32_t>(c.bytes.find("GET /a")));
    EXPECT_EQ(kv::kv_packed_get(c.v.data(), c.bytes.data(), 4, "job", 3, KV, PR, out.data()), 2);
    EXPECT_EQ(text(out[1], c.bytes.data()), "x");        // an inline row
}

TEST(KvPackedGet, AKeyIsAWholeKeyNotASuffixOrPrefix) {
    Col c;
    c.add(pairs({{"xle", "1"}, {"le_x", "2"}, {"host_name", "h"}, {"name", "n"}}));
    c.add(pairs({{"a", std::string("le\x1f" "9").c_str()}}));          // "le KV 9" inside a value: not a pair start
    std::vector<StringView> out(2);
    EXPECT_EQ(kv::kv_packed_get(c.v.data(), c.bytes.data(), 2, "le", 2, KV, PR, out.data()), 0);
    EXPECT_EQ(out[0].length, 0u);
    EXPECT_EQ(out[1].length, 0u);
    EXPECT_EQ(kv::kv_packed_get(c.v.data(), c.bytes.data(), 2, "name", 4, KV, PR, out.data()), 1);
    EXPECT_EQ(text(out[0], c.bytes.data()), "n");
}

TEST(KvPackedGet, EmptyValueRepeatsAndBrokenSets) {
    Col c;
    c.add(pairs({{"a", ""}, {"b", "1"}}));
    c.add(pairs({{"k", "first"}, {"z", "0"}, {"k", "second-and-longer-than-twelve"}}));
    c.add(std::string("k") + KV + "unterminated");                    // not whole pairs
    c.add(pairs({{"k", "v"}}) + "tail");
    std::vector<StringView> out(4);
    EXPECT_EQ(kv::kv_packed_get(c.v.data(), c.bytes.data(), 4, "a", 1, KV, PR, out.data()), 1);
    EXPECT_EQ(out[0].length, 0u);                                     // present and empty
    EXPECT_EQ(kv::kv_packed_get(c.v.data(), c.bytes.data(), 4, "k", 1, KV, PR, out.data()), 1);
    EXPECT_EQ(text(out[1], c.bytes.data()), "second-and-longer-than-twelve");
    EXPECT_EQ(out[2].length, 0u);
    EXPECT_EQ(out[3].length, 0u);
}

TEST(KvPackedGet, RefusesAKeyItCannotCarry) {
    Col c;
    c.add(pairs({{"a", "1"}}));
    StringView out;
    std::memset(&out, 0x5a, sizeof(out));
    const StringView before = out;
    const std::string big(kv::kKeyMaxBytes + 1u, 'k');
    EXPECT_EQ(kv::kv_packed_get(c.v.data(), nullptr, 1, "", 0, KV, PR, &out), -1);
    EXPECT_EQ(kv::kv_packed_get(c.v.data(), nullptr, 1, big.data(), static_cast<uint32_t>(big.size()), KV, PR, &out), -1);
    EXPECT_EQ(kv::kv_packed_get(c.v.data(), nullptr, 1, "a\x1f", 2, KV, PR, &out), -1);
    EXPECT_EQ(std::memcmp(&out, &before, sizeof(out)), 0);
    const std::string max(kv::kKeyMaxBytes, 'k');
    EXPECT_EQ(kv::kv_packed_get(c.v.data(), nullptr, 1, max.data(), static_cast<uint32_t>(max.size()), KV, PR, &out), 0);
}

TEST(KvPackedGet, AgreesWithAPairWalkOnGeneratedSets) {
    const char* names[] = {"a", "ab", "b", "job", "le", "le2", "service_name", "span_name", "x_le"};
    std::uint64_t st = 0x9e3779b97f4a7c15ull;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
    Col c;
    std::vector<std::string> rows;
    for (int r = 0; r < 4000; ++r) {
        std::string s;
        const int np = static_cast<int>(rnd() % 7);
        for (int k = 0; k < np; ++k) {
            s += names[rnd() % 9];
            s += KV;
            const int vl = static_cast<int>(rnd() % 20);
            for (int j = 0; j < vl; ++j) s += "le\x1f" "abc/-_ 0123"[rnd() % 14];   // values may hold KV and key text
            s += PR;
        }
        if (rnd() % 31 == 0) s += "junk";                              // a broken set
        rows.push_back(s);
        c.add(s);
    }
    std::vector<StringView> out(rows.size());
    for (const char* key : names) {
        const int64_t found = kv::kv_packed_get(c.v.data(), c.bytes.data(), static_cast<int64_t>(rows.size()), key,
                                                static_cast<uint32_t>(std::strlen(key)), KV, PR, out.data());
        EXPECT_GT(found, 0);
        for (size_t i = 0; i < rows.size(); ++i)
            ASSERT_EQ(text(out[i], c.bytes.data()), reference(rows[i], key)) << "row " << i << " key " << key;
    }
}
