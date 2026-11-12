// G2CHK-963: bolt::kernels::variantspark parse_json / to_json against real Spark 4.0.4 output
// (tests/data/spark_variant_vectors.tsv: value and metadata bytes from parse_json, text from to_json).
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <cstring>

#include "bolt/datetime/bolt_tz.h"
#include "bolt/kernels/bolt_variant_spark.h"
#include "bolt/kernels/bolt_variant_spark_schema.h"

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

TEST(BoltVariantSpark, CastVectorsToJsonInZone) {
    // cast(x AS VARIANT) bytes and their to_json from Spark 4.0.4 in America/New_York.
    bolt::tz::ZoneId ny = bolt::tz::kInvalidZone;
    ASSERT_TRUE(bolt::tz::zone_parse("America/New_York", 16, &ny));
    std::FILE* f = std::fopen(BOLT_TEST_DATA_DIR "/spark_variant_cast_vectors.tsv", "rb");
    ASSERT_NE(f, nullptr);
    static char line[1 << 12];
    int cases = 0;
    while (std::fgets(line, sizeof line, f) != nullptr) {
        std::string l(line);
        while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
        if (l.empty() || l[0] == '#') continue;
        const size_t a = l.find('\t'), b = l.find('\t', a + 1);
        const std::string v = unhex(l.substr(0, a)), m = unhex(l.substr(a + 1, b - a - 1)), want = l.substr(b + 1);
        char out[256];
        js::Out o{out, sizeof out, 0, false};
        ASSERT_EQ(vs::to_json(reinterpret_cast<const uint8_t*>(v.data()), static_cast<uint32_t>(v.size()),
                              reinterpret_cast<const uint8_t*>(m.data()), static_cast<uint32_t>(m.size()), &o, ny),
                  vs::JsonStatus::Ok) << l;
        EXPECT_EQ(std::string(out, o.n), want) << l;
        ++cases;
    }
    std::fclose(f);
    EXPECT_GE(cases, 17);
}

TEST(BoltVariantSpark, EncodersMatchSparkCasts) {
    uint8_t b[64];
    EXPECT_EQ(hex(b, vs::encode_int(b, sizeof b, 5, 1)), "0c05");
    EXPECT_EQ(hex(b, vs::encode_int(b, sizeof b, 300, 2)), "102c01");
    EXPECT_EQ(hex(b, vs::encode_int(b, sizeof b, 70000, 4)), "1470110100");
    EXPECT_EQ(hex(b, vs::encode_int(b, sizeof b, 5000000000LL, 8)), "1800f2052a01000000");
    EXPECT_EQ(hex(b, vs::encode_float(b, sizeof b, 1.5f)), "380000c03f");
    EXPECT_EQ(hex(b, vs::encode_double(b, sizeof b, 1e20)), "1c408cb5781daf1544");
    EXPECT_EQ(hex(b, vs::encode_bool(b, sizeof b, true)), "04");
    EXPECT_EQ(hex(b, vs::encode_temporal(b, sizeof b, vs::kDate, 19782)), "2c464d0000");
    EXPECT_EQ(hex(b, vs::encode_temporal(b, sizeof b, vs::kTimestamp, 1704182645678000LL)), "30b05f3af1f10d0600");
    const uint8_t bin[2] = {0x01, 0xff};
    EXPECT_EQ(hex(b, vs::encode_binary(b, sizeof b, bin, 2)), "3c0200000001ff");
    EXPECT_EQ(hex(b, vs::encode_decimal(b, sizeof b, 0, 150, 5, 2)), "200296000000");
    const char* h = "h\xc3\xa9llo";
    EXPECT_EQ(hex(b, vs::encode_string(b, sizeof b, reinterpret_cast<const uint8_t*>(h), 6)), "1968c3a96c6c6f");
    EXPECT_EQ(vs::encode_int(b, 1, 5, 1), 0u);   // too small: refused, not truncated
}

TEST(BoltVariantSpark, GetPathAndCastsMatchSpark) {
    Bufs bf;
    vs::Encoded e{};
    auto get = [&](const char* json, const char* path, const uint8_t** v, uint32_t* n) {
        EXPECT_EQ(parse(&bf, json, &e), vs::Status::Ok) << json;
        return vs::get_path(e.value, e.value_n, e.metadata, e.metadata_n, path, static_cast<uint32_t>(std::strlen(path)), v, n);
    };
    const uint8_t* v = nullptr;
    uint32_t n = 0;
    int64_t x = 0;
    double d = 0;
    bool t = false;
    // Spark 4.0.4: variant_get(parse_json('{"a":{"b":[10,20]}}'), '$.a.b[1]', 'int') = 20
    ASSERT_EQ(get("{\"a\":{\"b\":[10,20]}}", "$.a.b[1]", &v, &n), vs::GetStatus::Found);
    EXPECT_EQ(vs::to_long(v, n, &x), vs::CastStatus::Ok);
    EXPECT_EQ(x, 20);
    ASSERT_EQ(get("{\"a\":{\"b\":[10,20]}}", "$['a'][\"b\"][0]", &v, &n), vs::GetStatus::Found);
    EXPECT_EQ(vs::to_long(v, n, &x), vs::CastStatus::Ok);
    EXPECT_EQ(x, 10);
    // '$.a' of {"a":1.5} as double 1.5, as int 1 (truncated)
    ASSERT_EQ(get("{\"a\":1.5}", "$.a", &v, &n), vs::GetStatus::Found);
    EXPECT_EQ(vs::to_double(v, n, &d), vs::CastStatus::Ok);
    EXPECT_EQ(d, 1.5);
    EXPECT_EQ(vs::to_long(v, n, &x), vs::CastStatus::Ok);
    EXPECT_EQ(x, 1);
    // "12" as int 12; "x" as int invalid (variant_get errors, try_variant_get NULL)
    ASSERT_EQ(get("{\"a\":\"12\"}", "$.a", &v, &n), vs::GetStatus::Found);
    EXPECT_EQ(vs::to_long(v, n, &x), vs::CastStatus::Ok);
    EXPECT_EQ(x, 12);
    ASSERT_EQ(get("{\"a\":\"x\"}", "$.a", &v, &n), vs::GetStatus::Found);
    EXPECT_EQ(vs::to_long(v, n, &x), vs::CastStatus::Invalid);
    // a missing key is NULL; a bad path is an error
    EXPECT_EQ(get("{\"a\":2}", "$.b", &v, &n), vs::GetStatus::Missing);
    EXPECT_EQ(get("{\"a\":2}", "a", &v, &n), vs::GetStatus::BadPath);
    EXPECT_FALSE(vs::path_valid("$.", 2));
    EXPECT_TRUE(vs::path_valid("$.a[0]['b c']", 13));
    // parse_json('1') as boolean true
    ASSERT_EQ(get("1", "$", &v, &n), vs::GetStatus::Found);
    EXPECT_EQ(vs::to_bool(v, n, &t), vs::CastStatus::Ok);
    EXPECT_TRUE(t);
    // is_variant_null(parse_json('null'))
    ASSERT_EQ(get("null", "$", &v, &n), vs::GetStatus::Found);
    EXPECT_TRUE(vs::is_null(v, n));
    // sub-variant '$.a' of {"a":[1,2]} to_json [1,2]
    ASSERT_EQ(get("{\"a\":[1,2]}", "$.a", &v, &n), vs::GetStatus::Found);
    char out[64];
    js::Out o{out, sizeof out, 0, false};
    ASSERT_EQ(vs::to_json(v, n, e.metadata, e.metadata_n, &o), vs::JsonStatus::Ok);
    EXPECT_EQ(std::string(out, o.n), "[1,2]");
}

// schema_of_variant: expected text from real Spark 4.0.4 (schema_of_variant(parse_json(j)), probe c_sov.py).
TEST(BoltVariantSpark, SchemaOfMatchesSpark) {
    const std::pair<const char*, const char*> cases[] = {
        {R"({"a":1,"b":[1,2]})", "OBJECT<a: BIGINT, b: ARRAY<BIGINT>>"},
        {R"([1,"a",null,true,[2,{"x":1}]])", "ARRAY<VARIANT>"},
        {"1.5", "DECIMAL(2,1)"}, {"null", "VOID"}, {R"("x")", "STRING"},
        {R"({"z":{"y":{"x":"deep"}},"a":null})", "OBJECT<a: VOID, z: OBJECT<y: OBJECT<x: STRING>>>"},
        {"[]", "ARRAY<VOID>"}, {"{}", "OBJECT<>"}, {"[1, 2.5]", "ARRAY<DECIMAL(21,1)>"}, {"[1, 1e20]", "ARRAY<DOUBLE>"},
        {"[1.5, 1e300]", "ARRAY<DOUBLE>"}, {R"([1, "s"])", "ARRAY<VARIANT>"},
        {R"([{"a":1},{"b":"x"}])", "ARRAY<OBJECT<a: BIGINT, b: STRING>>"}, {R"([{"a":1},{"a":2.25}])", "ARRAY<OBJECT<a: DECIMAL(22,2)>>"},
        {R"([{"a":1},{"a":"s"}])", "ARRAY<OBJECT<a: VARIANT>>"}, {"[[1],[2.5]]", "ARRAY<ARRAY<DECIMAL(21,1)>>"},
        {"[null, null]", "ARRAY<VOID>"}, {"[true, 1]", "ARRAY<VARIANT>"}, {R"({"A":1,"a":2})", "OBJECT<A: BIGINT, a: BIGINT>"},
        {R"({"a b":1,"c`d":2,"_e":3,"1f":4})", "OBJECT<`1f`: BIGINT, _e: BIGINT, `a b`: BIGINT, `c``d`: BIGINT>"},
        {"0.001", "DECIMAL(3,3)"}, {"123456789012345678901234567890", "DECIMAL(30,0)"}, {"-0.0", "DECIMAL(1,0)"},
        {"1e-5", "DOUBLE"}, {"[0.5, 0.25]", "ARRAY<DECIMAL(2,2)>"}, {"9223372036854775807", "BIGINT"},
        {R"([{"a":[1]},{"a":[null]},{"a":[2.5]}])", "ARRAY<OBJECT<a: ARRAY<DECIMAL(21,1)>>>"}, {"[[], [1]]", "ARRAY<ARRAY<BIGINT>>"},
        {R"([{}, {"a":1}])", "ARRAY<OBJECT<a: BIGINT>>"}, {"[1, 0.1234567890123456789012345678]", "ARRAY<DOUBLE>"},
        {"[[1],{\"a\":1}]", "ARRAY<VARIANT>"},
    };
    Bufs b;
    std::vector<vs::SchemaNode> nodes(4096);
    char out[512];
    for (const auto& [json, want] : cases) {
        vs::Encoded e{};
        ASSERT_EQ(parse(&b, json, &e), vs::Status::Ok) << json;
        vs::SchemaPool pool{nodes.data(), static_cast<uint32_t>(nodes.size()), 0};
        js::Out o{out, sizeof out, 0, false};
        ASSERT_EQ(vs::schema_of(e.value, e.value_n, e.metadata, e.metadata_n, &pool, &o), vs::SchemaStatus::Ok) << json;
        EXPECT_EQ(std::string(out, o.n), want) << json;
    }
    vs::Encoded e{};
    ASSERT_EQ(parse(&b, R"([{"a":1,"b":2},{"c":3}])", &e), vs::Status::Ok);
    vs::SchemaPool tiny{nodes.data(), 3, 0};
    js::Out o{out, sizeof out, 0, false};
    EXPECT_EQ(vs::schema_of(e.value, e.value_n, e.metadata, e.metadata_n, &tiny, &o), vs::SchemaStatus::Budget);
}

// CAST(x AS VARIANT) widths as Spark's VariantBuilder writes them (probe c_tvw.py bytes).
TEST(BoltVariantSpark, NarrowestWidthsMatchSpark) {
    uint8_t b[24];
    EXPECT_EQ(hex(b, vs::encode_int(b, sizeof b, 5, vs::long_width(5))), "0c05");
    EXPECT_EQ(hex(b, vs::encode_int(b, sizeof b, 300, vs::long_width(300))), "102c01");
    EXPECT_EQ(vs::long_width(70000), 4u);
    EXPECT_EQ(vs::long_width(5000000000LL), 8u);
    EXPECT_EQ(vs::long_width(-129), 2u);
    EXPECT_EQ(hex(b, vs::encode_decimal(b, sizeof b, 0, 125, vs::decimal_value_digits(0, 125), 2)), "20027d000000");
    EXPECT_EQ(hex(b, vs::encode_decimal(b, sizeof b, 0, 1, vs::decimal_value_digits(0, 1), 0)), "200001000000");
    EXPECT_EQ(hex(b, vs::encode_decimal(b, sizeof b, 0, 12345678901ull, vs::decimal_value_digits(0, 12345678901ull), 0)),
              "2400351cdcdf02000000");
    EXPECT_EQ(vs::decimal_value_digits(-1, ~0ull), 1u);            // -1
    EXPECT_EQ(vs::decimal_value_digits(0, 0), 1u);
}
