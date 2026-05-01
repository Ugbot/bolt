// test_bolt_parse_json.cpp — coverage for bolt::parse::json (Layer 1.3).
//
// Skip-aware JSON parser tests. STL is allowed in tests; production code
// stays Tiger-Style.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_types.h"
#include "bolt/parse/bolt_json.h"

using bolt::Arena;
using bolt::ArenaConfig;
using bolt::StringView;
using bolt::parse::json::Iterator;
using bolt::parse::json::PathFilter;
using bolt::parse::json::StructuralIndex;
using bolt::parse::json::Token;
using bolt::parse::json::TokenType;

namespace {

Arena make_arena() {
    ArenaConfig cfg{};
    cfg.initial_block_size = 64u * 1024u;
    cfg.max_block_size     = 4u * 1024u * 1024u;
    cfg.alignment          = 8u;
    return Arena(cfg);
}

bool build(Arena& a, const std::string& s, StructuralIndex* out) {
    return bolt::parse::json::build_index(
        reinterpret_cast<const uint8_t*>(s.data()),
        static_cast<int32_t>(s.size()), &a, out);
}

}  // namespace

TEST(BoltParseJson, EmptyDocument) {
    Arena a = make_arena();
    StructuralIndex idx;
    EXPECT_TRUE(build(a, "", &idx));
    ASSERT_EQ(idx.token_count, 1);
    EXPECT_EQ(idx.tokens[0].type, TokenType::End);

    EXPECT_TRUE(build(a, "   ", &idx));
    ASSERT_EQ(idx.token_count, 1);
    EXPECT_EQ(idx.tokens[0].type, TokenType::End);

    EXPECT_TRUE(build(a, "null", &idx));
    ASSERT_EQ(idx.token_count, 2);
    EXPECT_EQ(idx.tokens[0].type, TokenType::Null);
    EXPECT_EQ(idx.tokens[1].type, TokenType::End);
}

TEST(BoltParseJson, DeepNesting) {
    Arena a = make_arena();
    // 50 levels deep — within the cap of 64.
    std::string s(50, '[');
    s.append(50, ']');
    StructuralIndex idx;
    EXPECT_TRUE(build(a, s, &idx));
    int32_t begins = 0, ends = 0;
    for (int32_t i = 0; i < idx.token_count; ++i) {
        if (idx.tokens[i].type == TokenType::BeginArray) ++begins;
        if (idx.tokens[i].type == TokenType::EndArray)   ++ends;
    }
    EXPECT_EQ(begins, 50);
    EXPECT_EQ(ends, 50);

    // 80 levels — exceeds cap, must reject.
    std::string deep(80, '[');
    deep.append(80, ']');
    StructuralIndex idx2;
    EXPECT_FALSE(build(a, deep, &idx2));
}

TEST(BoltParseJson, EscapedStrings) {
    Arena a = make_arena();
    StructuralIndex idx;
    // {"k": "\"\\né"} — note the source bytes for é are the 6 ASCII
    // chars literally; UTF-8 decode happens lazily in the consumer.
    std::string s = "{\"k\": \"\\\"\\n\\u00e9\"}";
    ASSERT_TRUE(build(a, s, &idx));
    ASSERT_GE(idx.token_count, 5);
    EXPECT_EQ(idx.tokens[0].type, TokenType::BeginObject);
    EXPECT_EQ(idx.tokens[1].type, TokenType::Key);
    EXPECT_EQ(idx.tokens[2].type, TokenType::String);
    // Slice covers the source bytes between the quotes.
    const Token& str = idx.tokens[2];
    std::string slice(reinterpret_cast<const char*>(idx.src) + str.start, str.length);
    EXPECT_EQ(slice, "\\\"\\n\\u00e9");
}

TEST(BoltParseJson, UnicodeUtf8Validated) {
    Arena a = make_arena();
    StructuralIndex idx;
    // Valid UTF-8: U+00E9 LATIN SMALL LETTER E WITH ACUTE = 0xC3 0xA9.
    std::string ok;
    ok.push_back('"');
    ok.push_back(static_cast<char>(0xC3));
    ok.push_back(static_cast<char>(0xA9));
    ok.push_back('"');
    EXPECT_TRUE(build(a, ok, &idx));

    // Invalid UTF-8: overlong NUL via 0xC0 0x80.
    std::string bad;
    bad.push_back('"');
    bad.push_back(static_cast<char>(0xC0));
    bad.push_back(static_cast<char>(0x80));
    bad.push_back('"');
    StructuralIndex idx2;
    EXPECT_FALSE(build(a, bad, &idx2));
}

TEST(BoltParseJson, NumbersLazyMaterialise) {
    Arena a = make_arena();
    StructuralIndex idx;
    ASSERT_TRUE(build(a, "[42, -17, 3.14, 1.0e3]", &idx));
    Iterator it;
    ASSERT_TRUE(bolt::parse::json::iter_init(&idx, &it));
    EXPECT_EQ(bolt::parse::json::iter_peek(&it), TokenType::BeginArray);
    bolt::parse::json::iter_advance(&it);

    int64_t i64 = 0;
    ASSERT_TRUE(bolt::parse::json::iter_int64(&it, &i64));
    EXPECT_EQ(i64, 42);
    bolt::parse::json::iter_advance(&it);

    ASSERT_TRUE(bolt::parse::json::iter_int64(&it, &i64));
    EXPECT_EQ(i64, -17);
    bolt::parse::json::iter_advance(&it);

    double f = 0.0;
    ASSERT_TRUE(bolt::parse::json::iter_float64(&it, &f));
    EXPECT_DOUBLE_EQ(f, 3.14);
    bolt::parse::json::iter_advance(&it);

    ASSERT_TRUE(bolt::parse::json::iter_float64(&it, &f));
    EXPECT_DOUBLE_EQ(f, 1.0e3);
}

TEST(BoltParseJson, SkipToCloseEqualsSequentialAdvance) {
    Arena a = make_arena();
    StructuralIndex idx;
    const std::string src = "{\"a\":1,\"b\":[1,2,{\"x\":[true,false,null,\"s\"]}],\"c\":3}";
    ASSERT_TRUE(build(a, src, &idx));

    // Find the BeginArray for "b".
    int32_t b_begin = -1;
    for (int32_t i = 0; i < idx.token_count; ++i) {
        if (idx.tokens[i].type == TokenType::Key) {
            const Token& k = idx.tokens[i];
            if (k.length == 1 && idx.src[k.start] == 'b') {
                b_begin = i + 1;
                break;
            }
        }
    }
    ASSERT_GE(b_begin, 0);
    ASSERT_EQ(idx.tokens[b_begin].type, TokenType::BeginArray);

    // Skip-based.
    Iterator skip_it{ &idx, b_begin, 0 };
    ASSERT_TRUE(bolt::parse::json::iter_skip_to_close(&skip_it));

    // Advance-based: walk until depth returns to zero.
    Iterator adv_it{ &idx, b_begin, 0 };
    int32_t depth = 0;
    while (adv_it.cursor < idx.token_count) {
        TokenType t = idx.tokens[adv_it.cursor].type;
        if (t == TokenType::BeginObject || t == TokenType::BeginArray) ++depth;
        else if (t == TokenType::EndObject || t == TokenType::EndArray) {
            --depth;
            if (depth == 0) { ++adv_it.cursor; break; }
        }
        ++adv_it.cursor;
    }
    EXPECT_EQ(skip_it.cursor, adv_it.cursor);
}

TEST(BoltParseJson, PathFilterSubtreesSkipped) {
    Arena a = make_arena();
    const std::string doc =
        "{"
            "\"wanted\":{\"x\":1,\"y\":[10,20,30]},"
            "\"unwanted\":{\"a\":1,\"b\":[1,2,3,4,5,6,7,8,9,10]},"
            "\"also\":[{\"k\":\"v\"},{\"k\":\"v2\"}],"
            "\"trash\":[1,2,3,4,5,6,7,8,9,10]"
        "}";
    StructuralIndex full;
    ASSERT_TRUE(build(a, doc, &full));

    const char* paths[]    = { "/wanted", "/also/0" };
    const int32_t lens[]   = { 7, 7 };
    PathFilter pf{};
    ASSERT_TRUE(bolt::parse::json::compile_paths(paths, lens, 2, &a, &pf));

    StructuralIndex filt;
    ASSERT_TRUE(bolt::parse::json::build_index_filtered(
        reinterpret_cast<const uint8_t*>(doc.data()),
        static_cast<int32_t>(doc.size()), &pf, &a, &filt));

    EXPECT_LT(filt.token_count, full.token_count);

    // Verify "wanted" subtree is fully present (BeginObject/EndObject pair
    // for that subobject and all its leaves).
    int32_t wanted_keys = 0;
    int32_t unwanted_keys = 0;
    int32_t trash_keys = 0;
    for (int32_t i = 0; i < filt.token_count; ++i) {
        if (filt.tokens[i].type == TokenType::Key) {
            const Token& k = filt.tokens[i];
            std::string s(reinterpret_cast<const char*>(filt.src) + k.start, k.length);
            if (s == "wanted") ++wanted_keys;
            else if (s == "unwanted") ++unwanted_keys;
            else if (s == "trash") ++trash_keys;
        }
    }
    EXPECT_EQ(wanted_keys, 1);
    EXPECT_EQ(unwanted_keys, 0);
    EXPECT_EQ(trash_keys, 0);

    // Print delta for the report. (Test asserts the relationship; the bench
    // run captures the absolute numbers.)
    ::testing::Test::RecordProperty(
        "filtered_token_count", std::to_string(filt.token_count));
    ::testing::Test::RecordProperty(
        "full_token_count", std::to_string(full.token_count));
}

TEST(BoltParseJson, MalformedJsonRejected) {
    Arena a = make_arena();
    StructuralIndex idx;
    EXPECT_FALSE(build(a, "{\"a\":1", &idx));        // unterminated object
    EXPECT_FALSE(build(a, "[1,2,3", &idx));          // unterminated array
    EXPECT_FALSE(build(a, "{\"a\":}", &idx));        // missing value
    EXPECT_FALSE(build(a, "\"unterm", &idx));        // unterminated string
    EXPECT_FALSE(build(a, "[1,]", &idx));            // trailing comma
    EXPECT_FALSE(build(a, "{]}", &idx));             // bad bracket pair
    EXPECT_FALSE(build(a, "tru", &idx));             // truncated keyword
    EXPECT_FALSE(build(a, "1.", &idx));              // bad number
}

TEST(BoltParseJson, FilterTokenDelta) {
    // Standalone larger-document test that records the exact token-count
    // delta the report cares about.
    Arena a = make_arena();
    std::string doc = "{";
    doc += "\"wanted\":{\"x\":1,\"y\":2},";
    doc += "\"also\":[{\"keep\":true},{\"keep\":false},{\"keep\":null}],";
    doc += "\"big\":[";
    for (int i = 0; i < 64; ++i) {
        if (i) doc += ",";
        doc += "{\"id\":";
        doc += std::to_string(i);
        doc += ",\"name\":\"name_";
        doc += std::to_string(i);
        doc += "\",\"tags\":[\"a\",\"b\",\"c\"]}";
    }
    doc += "]}";

    StructuralIndex full;
    ASSERT_TRUE(build(a, doc, &full));

    const char* paths[] = { "/wanted", "/also/0" };
    const int32_t lens[] = { 7, 7 };
    PathFilter pf{};
    ASSERT_TRUE(bolt::parse::json::compile_paths(paths, lens, 2, &a, &pf));

    StructuralIndex filt;
    ASSERT_TRUE(bolt::parse::json::build_index_filtered(
        reinterpret_cast<const uint8_t*>(doc.data()),
        static_cast<int32_t>(doc.size()), &pf, &a, &filt));

    std::printf("[BoltParseJson] full_tokens=%d filtered_tokens=%d delta=%d\n",
                full.token_count, filt.token_count,
                full.token_count - filt.token_count);
    EXPECT_LT(filt.token_count, full.token_count);
}
