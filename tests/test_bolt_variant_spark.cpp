// G2CHK-963: bolt::kernels::variantspark parse_json / to_json against real Spark 4.0.4 output
// (tests/data/spark_variant_vectors.tsv: value and metadata bytes from parse_json, text from to_json).
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "bolt/kernels/bolt_variant_spark.h"

namespace vs = bolt::kernels::variantspark;
namespace js = bolt::kernels::jsonspark;

namespace {

std::string unhex(const std::string& h) {
    std::string out;
    for (size_t i = 0; i + 1 < h.size(); i += 2) out.push_back(static_cast<char>(std::stoi(h.substr(i, 2), nullptr, 16)));
    return out;
}
std::string hex(const uint8_t* p, uint32_t n) {
    static const char d[] = "0123456789abcdef";
    std::string s;
    for (uint32_t i = 0; i < n; ++i) { s.push_back(d[p[i] >> 4]); s.push_back(d[p[i] & 15]); }
    return s;
}

struct Bufs {
    std::vector<uint8_t> key_bytes = std::vector<uint8_t>(1 << 16);
    std::vector<uint32_t> key_off = std::vector<uint32_t>(1025);
    std::vector<uint32_t> table = std::vector<uint32_t>(4096);
    std::vector<uint32_t> ent_id = std::vector<uint32_t>(4096), ent_off = std::vector<uint32_t>(4096);
    std::vector<uint32_t> fs = std::vector<uint32_t>(1024), fe = std::vector<uint32_t>(1024);
    std::vector<uint8_t> fo = std::vector<uint8_t>(1024);
    std::vector<uint8_t> value = std::vector<uint8_t>(1 << 20), meta = std::vector<uint8_t>(1 << 16);
    std::vector<uint8_t> text = std::vector<uint8_t>(1 << 16);
    vs::Workspace ws() {
        return vs::Workspace{key_bytes.data(), static_cast<uint32_t>(key_bytes.size()), key_off.data(), 1024,
                             table.data(), 4096, ent_id.data(), ent_off.data(), 4096, fs.data(), fe.data(), fo.data(), 1024};
    }
};

vs::Status parse(Bufs* b, const std::string& j, vs::Encoded* e) {
    vs::Workspace w = b->ws();
    *e = vs::Encoded{b->value.data(), static_cast<uint32_t>(b->value.size()), 0, b->meta.data(),
                     static_cast<uint32_t>(b->meta.size()), 0};
    return vs::parse_json(reinterpret_cast<const uint8_t*>(j.data()), static_cast<uint32_t>(j.size()), &w, e,
                          b->text.data(), static_cast<uint32_t>(b->text.size()));
}

}  // namespace

TEST(BoltVariantSpark, MatchesSparkVectors) {
    std::FILE* f = std::fopen(BOLT_TEST_DATA_DIR "/spark_variant_vectors.tsv", "rb");
    ASSERT_NE(f, nullptr);
    static char line[1 << 18];
    int cases = 0;
    Bufs b;
    while (std::fgets(line, sizeof line, f) != nullptr) {
        std::string l(line);
        while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
        if (l.empty() || l[0] == '#') continue;
        std::vector<std::string> c;
        size_t s = 0;
        for (size_t t; (t = l.find('\t', s)) != std::string::npos; s = t + 1) c.push_back(l.substr(s, t - s));
        c.push_back(l.substr(s));
        const std::string json = unhex(c[0]);
        vs::Encoded e{};
        const vs::Status st = parse(&b, json, &e);
        ++cases;
        if (c[1] == "ERR") {
            EXPECT_NE(st, vs::Status::Ok) << json;
            continue;
        }
        ASSERT_EQ(st, vs::Status::Ok) << json;
        EXPECT_EQ(hex(e.value, e.value_n), c[1]) << json;
        EXPECT_EQ(hex(e.metadata, e.metadata_n), c[2]) << json;
        if (c.size() > 3 && c[3] != "-") {
            char out[1 << 12];
            js::Out o{out, sizeof out, 0, false};
            ASSERT_EQ(vs::to_json(e.value, e.value_n, e.metadata, e.metadata_n, &o), vs::JsonStatus::Ok) << json;
            EXPECT_EQ(std::string(out, o.n), c[3]) << json;
        }
    }
    std::fclose(f);
    EXPECT_GE(cases, 50);
}

TEST(BoltVariantSpark, BudgetsRefuseNeverTruncate) {
    Bufs b;
    vs::Workspace w = b.ws();
    w.keys_cap = 1;
    const std::string j = "{\"a\":1,\"b\":2}";
    vs::Encoded e{b.value.data(), 1u << 20, 0, b.meta.data(), 1u << 16, 0};
    EXPECT_EQ(vs::parse_json(reinterpret_cast<const uint8_t*>(j.data()), static_cast<uint32_t>(j.size()), &w, &e,
                             b.text.data(), 1u << 16), vs::Status::Budget);
    w = b.ws();
    e = vs::Encoded{b.value.data(), 3, 0, b.meta.data(), 1u << 16, 0};
    EXPECT_EQ(vs::parse_json(reinterpret_cast<const uint8_t*>(j.data()), static_cast<uint32_t>(j.size()), &w, &e,
                             b.text.data(), 1u << 16), vs::Status::Budget);
}

TEST(BoltVariantSpark, LargeObjectAndArrayHeaders) {
    // 300 elements: is_large and 2-byte offsets, round-tripped through to_json.
    std::string j = "[";
    for (int i = 0; i < 300; ++i) j += (i ? "," : "") + std::to_string(i * 1000);
    j += "]";
    Bufs b;
    vs::Encoded e{};
    ASSERT_EQ(parse(&b, j, &e), vs::Status::Ok);
    EXPECT_EQ(e.value[0], static_cast<uint8_t>(((1u << 2 | 1u) << 2) | 3u));   // large, 2-byte offsets, array
    std::vector<char> out(1 << 16);
    js::Out o{out.data(), static_cast<uint32_t>(out.size()), 0, false};
    ASSERT_EQ(vs::to_json(e.value, e.value_n, e.metadata, e.metadata_n, &o), vs::JsonStatus::Ok);
    EXPECT_EQ(std::string(out.data(), o.n), j);
}
