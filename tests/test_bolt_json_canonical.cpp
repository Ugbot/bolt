#include "bolt/parse/bolt_json_canonical.h"
#include "bolt/parse/bolt_json_writer.h"
#include "bolt/bolt_arena.h"
#include <gtest/gtest.h>
#include <cstring>
#include <string>
namespace j = bolt::parse::json;
namespace {
j::CanonicalLimits limits{1u << 20u,1u << 16u,64u};
std::string canonical(bolt::Arena& a, const std::string& input) {
    const uint8_t* out = nullptr; uint32_t n = 0u;
    const auto status = j::canonicalize(reinterpret_cast<const uint8_t*>(input.data()),static_cast<uint32_t>(input.size()),&a,limits,&out,&n);
    EXPECT_EQ(status,j::CanonicalResult::Ok); return out ? std::string(reinterpret_cast<const char*>(out),static_cast<size_t>(n)) : std::string{};
}
TEST(JsonCanonical, SortsDecodedKeysNormalizesEscapesAndRetainsArrayOrder) {
    bolt::Arena arena;
    const std::string a = R"JSON({ "z": [3,1,2], "\u0061": {"b":"\u00e9", "a":"x\/y"} })JSON";
    const std::string b = "{\"a\":{\"a\":\"x/y\",\"b\":\"\xc3\xa9\"},\"z\":[3,1,2]}";
    const std::string expected = R"JSON({"a":{"a":"x/y","b":"\u00e9"},"z":[3,1,2]})JSON";
    EXPECT_EQ(canonical(arena,a),expected); EXPECT_EQ(canonical(arena,b),expected); EXPECT_EQ(canonical(arena,expected),expected);
    EXPECT_NE(canonical(arena,R"({"z":[1,2,3]})"),canonical(arena,R"({"z":[3,2,1]})"));
}
TEST(JsonCanonical, NeverConvertsNumericSpellingOrWideValuesThroughFloatingPoint) {
    bolt::Arena arena; const std::string input = R"({"z":1e9999999999999999999,"d":1.00000000000000000000000000000000000001,"id":9223372036854775807,"a":-0})";
    const std::string expected = R"({"a":-0,"d":1.00000000000000000000000000000000000001,"id":9223372036854775807,"z":1e9999999999999999999})";
    EXPECT_EQ(canonical(arena,input),expected); EXPECT_NE(canonical(arena,"1.0"),canonical(arena,"1e0"));
}
TEST(JsonCanonical, RejectsDecodedDuplicateKeysAtEveryDepth) {
    bolt::Arena arena;
    for (const std::string& input : {R"({"a":1,"\u0061":2})",R"([{"x":0,"x":1}])",R"({"a":{"\u0000":1,"\u0000":2}})"}) {
        const uint8_t sentinel = 9u; const uint8_t* out = &sentinel; uint32_t n = 777u;
        EXPECT_EQ(j::canonicalize(reinterpret_cast<const uint8_t*>(input.data()),static_cast<uint32_t>(input.size()),&arena,limits,&out,&n),j::CanonicalResult::Invalid);
        EXPECT_EQ(out,&sentinel); EXPECT_EQ(n,777u);
    }
}
TEST(JsonCanonical, RejectsBadSyntaxUnicodeAndUtf8WithoutChangingOutputs) {
    bolt::Arena arena;
    for (const std::string& input : {std::string("{\"x\":}"),std::string("true false"),std::string(R"("\ud800")"),std::string("\"\xc0\xaf\""),std::string("{\"\xff\":0}")}) {
        const uint8_t* out = nullptr; uint32_t n = 777u;
        EXPECT_EQ(j::canonicalize(reinterpret_cast<const uint8_t*>(input.data()),static_cast<uint32_t>(input.size()),&arena,limits,&out,&n),j::CanonicalResult::Invalid) << input;
        EXPECT_EQ(out,nullptr); EXPECT_EQ(n,777u);
    }
}
TEST(JsonCanonical, EnforcesInputOutputNodeAndDepthBudgets) {
    bolt::Arena arena; const std::string unicode = "\"\xc3\xa9\"";
    const uint8_t* out = nullptr; uint32_t n = 777u;
    EXPECT_EQ(j::canonicalize(reinterpret_cast<const uint8_t*>(unicode.data()),4u,&arena,{4u,1u,0u},&out,&n),j::CanonicalResult::Limit);
    const std::string array = "[1,2]";
    EXPECT_EQ(j::canonicalize(reinterpret_cast<const uint8_t*>(array.data()),5u,&arena,{4u,10u,4u},&out,&n),j::CanonicalResult::Limit);
    EXPECT_EQ(j::canonicalize(reinterpret_cast<const uint8_t*>(array.data()),5u,&arena,{10u,2u,4u},&out,&n),j::CanonicalResult::Limit);
    EXPECT_EQ(j::canonicalize(reinterpret_cast<const uint8_t*>(array.data()),5u,&arena,{10u,3u,0u},&out,&n),j::CanonicalResult::Limit);
    EXPECT_EQ(out,nullptr); EXPECT_EQ(n,777u);
}
TEST(JsonCanonical, OutputHasIndependentArenaCustodyAndEmbeddedNulKey) {
    bolt::Arena arena; std::string input = R"({"\u0000":1,"":2,"q":"before"})"; const uint8_t* out = nullptr; uint32_t n = 0u;
    ASSERT_EQ(j::canonicalize(reinterpret_cast<const uint8_t*>(input.data()),static_cast<uint32_t>(input.size()),&arena,limits,&out,&n),j::CanonicalResult::Ok);
    input.assign(input.size(),'x'); EXPECT_EQ(std::string(reinterpret_cast<const char*>(out),static_cast<size_t>(n)),R"({"":2,"\u0000":1,"q":"before"})");
}
TEST(JsonCanonical, ExistingWriterModesRemainSourceFaithfulAndMeasured) {
    bolt::Arena arena; const std::string input = R"({"z":1.50,"a":"\u0061"})"; j::StructuralIndex index{}; j::JsonValue* root = nullptr;
    ASSERT_TRUE(j::build_index(reinterpret_cast<const uint8_t*>(input.data()),static_cast<int32_t>(input.size()),&arena,&index)); ASSERT_TRUE(j::value_from_index(&index,&arena,&root));
    const uint8_t* out = nullptr; int32_t n = 0; ASSERT_TRUE(j::serialize(root,&arena,&out,&n)); EXPECT_EQ(std::string(reinterpret_cast<const char*>(out),static_cast<size_t>(n)),input);
    j::EmitOptions opts{}; uint32_t measured = 0u; ASSERT_TRUE(j::measure_serialized(root,opts,&measured)); ASSERT_TRUE(j::serialize(root,&arena,opts,&out,&n)); EXPECT_EQ(measured,static_cast<uint32_t>(n));
}
} // namespace
