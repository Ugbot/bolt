// G2GRAPH-453 — bolt::parse::yaml::to_json.
#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "bolt/bolt_arena.h"
#include "bolt/parse/bolt_json_writer.h"
#include "bolt/parse/bolt_yaml.h"

namespace y = bolt::parse::yaml;

namespace {

constexpr y::YamlLimits kLim{1u << 20, 64u, 1u << 20};

struct Out {
    y::YamlCode code;
    std::string json;
    std::string what;
};

Out run(const char* src, y::YamlLimits lim = kLim) {
    bolt::Arena arena;
    const char* out = nullptr;
    uint32_t n = 0;
    y::YamlError err{};
    const y::YamlCode c = y::to_json(src, static_cast<uint32_t>(std::strlen(src)), &arena, lim, &out, &n, &err);
    return Out{c, c == y::YamlCode::Ok ? std::string(out, n) : std::string(), err.what ? err.what : ""};
}

void expect_json(const char* src, const char* want) {
    const Out o = run(src);
    EXPECT_EQ(o.code, y::YamlCode::Ok) << src << " -> " << o.what;
    EXPECT_EQ(o.json, want) << src;
}

void expect_refused(const char* src, y::YamlCode code, const char* what) {
    const Out o = run(src);
    EXPECT_EQ(o.code, code) << src;
    EXPECT_EQ(o.what, what) << src;
}

}  // namespace

TEST(BoltYaml, Scalars) {
    expect_json("", "null");
    expect_json("a: 1\nb: -2\nc: 0x1F\nd: 0o17\ne: 1.5\nf: 2e3\ng: .5\n",
                R"({"a":1,"b":-2,"c":31,"d":15,"e":1.5,"f":2e3,"g":0.5})");
    expect_json("a: true\nb: False\nc: ~\nd: null\ne:\nf: yes\n",
                R"({"a":true,"b":false,"c":null,"d":null,"e":null,"f":"yes"})");
    expect_json("d: 2026-12-31\nt: 2026-06-30T12:00:00Z\nv: \"2.1\"\n",
                R"({"d":"2026-12-31","t":"2026-06-30T12:00:00Z","v":"2.1"})");
    expect_json("u: https://x.example/a:b#frag\nh: human:jsmith@acme\n",
                R"({"u":"https://x.example/a:b#frag","h":"human:jsmith@acme"})");
    expect_json("q: 'it''s'\nd: \"tab\\there \\u00e9 \\\"q\\\"\"\n",
                "{\"q\":\"it's\",\"d\":\"tab\\there \\u00e9 \\\"q\\\"\"}");  // JSON escapes stay zero-copy
    expect_json("a: x # comment\n# whole line\nb: y\n", R"({"a":"x","b":"y"})");
}

TEST(BoltYaml, BlockCollections) {
    expect_json("tags:\n  - growth\n  - engagement\n", R"({"tags":["growth","engagement"]})");
    expect_json("tags:\n- a\n- b\nnext: 1\n", R"({"tags":["a","b"],"next":1})");
    expect_json("author:\n  - type: Person\n    id: https://a/p\n    name: Jordan\n  - id: x\n",
                R"({"author":[{"type":"Person","id":"https://a/p","name":"Jordan"},{"id":"x"}]})");
    expect_json("- - 1\n  - 2\n- 3\n", R"([[1,2],3])");
    expect_json("a:\n  b:\n    c: 1\n  d: 2\ne: 3\n", R"({"a":{"b":{"c":1},"d":2},"e":3})");
    expect_json("a:\nb: 1\n", R"({"a":null,"b":1})");
    expect_json("---\na: 1\n...\n", R"({"a":1})");
}

TEST(BoltYaml, FlowCollections) {
    expect_json("tags: [growth, engagement, north-star]\n", R"({"tags":["growth","engagement","north-star"]})");
    expect_json("g: { by: \"human:j\", at: 2026-06-30T12:00:00Z }\n",
                R"({"g":{"by":"human:j","at":"2026-06-30T12:00:00Z"}})");
    expect_json("s:\n  - { id: iml, resource: https://x.example/p/a, title: \"W, 2025\", author: human:a, n: 3 }\n",
                R"({"s":[{"id":"iml","resource":"https://x.example/p/a","title":"W, 2025","author":"human:a","n":3}]})");
    expect_json("a: [1, [2, 3], {k: v}, []]\n", R"({"a":[1,[2,3],{"k":"v"},[]]})");
    expect_json("a: [1,\n  2,\n  3]\n", R"({"a":[1,2,3]})");
    expect_json("a: {x, y: 1}\n", R"({"a":{"x":null,"y":1}})");
    expect_json("{\"a\": 1, \"b\": [true]}\n", R"({"a":1,"b":[true]})");
}

TEST(BoltYaml, MultiLineScalars) {
    expect_json("d: one\n  two\n\n  three\nn: 1\n", "{\"d\":\"one two\\nthree\",\"n\":1}");
    expect_json("l: |\n  a\n   b\n\n  c\nn: 1\n", "{\"l\":\"a\\n b\\n\\nc\\n\",\"n\":1}");
    expect_json("f: >\n  a\n  b\n\n  c\n", "{\"f\":\"a b\\nc\\n\"}");
    expect_json("s: |-\n  x\n\n", "{\"s\":\"x\"}");
    expect_json("k: |+\n  x\n\n", "{\"k\":\"x\\n\\n\"}");
    expect_json("q: \"a\n  b\"\n", "{\"q\":\"a b\"}");
}

TEST(BoltYaml, RefusalsAreNamed) {
    expect_refused("a: &x 1\nb: *x\n", y::YamlCode::Unsupported, "anchors and aliases");
    expect_refused("a: !!str 1\n", y::YamlCode::Unsupported, "tags");
    expect_refused("%YAML 1.2\n---\na: 1\n", y::YamlCode::Unsupported, "directives");
    expect_refused("? a\n: b\n", y::YamlCode::Unsupported, "complex keys");
    expect_refused("a: 1\n---\nb: 2\n", y::YamlCode::Unsupported, "more than one document");
    expect_refused("a:\n\tb: 1\n", y::YamlCode::Unsupported, "tab indentation");
    expect_refused("a: .inf\n", y::YamlCode::Unsupported, "non-finite float");
    expect_refused("a: 99999999999999999999\n", y::YamlCode::Unsupported, "integer beyond int64");
    expect_refused("a: b: c\n", y::YamlCode::Syntax, "mapping value not allowed here");
    expect_refused("a: [1, 2\n", y::YamlCode::Syntax, "unterminated flow sequence");
    expect_refused("a:\n  b: 1\n c: 2\n", y::YamlCode::Syntax, "bad indentation");
}

TEST(BoltYaml, BudgetsRefuseByName) {
    EXPECT_EQ(run("a: [[[[1]]]]\n", y::YamlLimits{1u << 20, 3u, 1000u}).what, "depth");
    EXPECT_EQ(run("a: [1, 2, 3, 4]\n", y::YamlLimits{1u << 20, 8u, 3u}).what, "nodes");
    EXPECT_EQ(run("a: 1\n", y::YamlLimits{2u, 8u, 100u}).what, "input bytes");
    EXPECT_EQ(run("a: \"\\x01\\x01\\x01\\x01\"\n", y::YamlLimits{24u, 8u, 100u}).what, "output bytes");
}

// ---- the tape (fionn architecture) ---------------------------------------------

namespace {

namespace bj = bolt::parse::json;

struct Tape {
    y::YamlCode code;
    bj::StructuralIndex idx;
    std::string what;
};

Tape tape(bolt::Arena* arena, const char* src, const bj::PathFilter* filter = nullptr) {
    Tape t{};
    y::YamlError err{};
    t.code = y::build_index(src, static_cast<uint32_t>(std::strlen(src)), arena, kLim, filter, &t.idx, &err);
    t.what = err.what ? err.what : "";
    return t;
}

std::string token_text(const bj::StructuralIndex& idx, int32_t i) {
    const bj::Token& t = idx.tokens[i];
    return std::string(reinterpret_cast<const char*>(bj::token_bytes(&idx, t)), static_cast<size_t>(t.length));
}

}  // namespace

TEST(BoltYamlTape, SameTokensAsJsonAndZeroCopy) {
    bolt::Arena arena;
    const char* src = "title: Weekly Active Users\nn: 42\nf: 1.5\nok: true\nq: \"a \\\"b\\\"\"\nh: 0x1F\ne:\n";
    const Tape t = tape(&arena, src);
    ASSERT_EQ(t.code, y::YamlCode::Ok) << t.what;
    const bj::TokenType want[] = {bj::TokenType::BeginObject, bj::TokenType::Key, bj::TokenType::String,
                                  bj::TokenType::Key, bj::TokenType::Int64, bj::TokenType::Key,
                                  bj::TokenType::Float64, bj::TokenType::Key, bj::TokenType::BoolTrue,
                                  bj::TokenType::Key, bj::TokenType::String, bj::TokenType::Key,
                                  bj::TokenType::Int64, bj::TokenType::Key, bj::TokenType::Null,
                                  bj::TokenType::EndObject, bj::TokenType::End};
    ASSERT_EQ(t.idx.token_count, 17);
    for (int i = 0; i < 17; ++i) EXPECT_EQ(t.idx.tokens[i].type, want[i]) << i;
    // Plain, number, keyword and JSON-compatible quoted scalars are source slices.
    for (int i : {1, 2, 4, 6, 8, 10}) EXPECT_EQ(t.idx.tokens[i].flags & bj::kTokenAux, 0) << i;
    EXPECT_EQ(token_text(t.idx, 2), "Weekly Active Users");
    EXPECT_EQ(token_text(t.idx, 10), "a \\\"b\\\"");
    // 0x1F is no JSON number: materialised once into aux; `e:` reads the aux null.
    EXPECT_NE(t.idx.tokens[12].flags & bj::kTokenAux, 0);
    EXPECT_EQ(token_text(t.idx, 12), "31");
    EXPECT_EQ(token_text(t.idx, 14), "null");
    bj::Iterator it{&t.idx, 4, 0};
    int64_t v = 0;
    ASSERT_TRUE(bj::iter_int64(&it, &v));
    EXPECT_EQ(v, 42);
    it.cursor = 12;
    ASSERT_TRUE(bj::iter_int64(&it, &v));
    EXPECT_EQ(v, 31);
}

TEST(BoltYamlTape, SkipToCloseAndValueTree) {
    bolt::Arena arena;
    const Tape t = tape(&arena, "a:\n  - {x: 1, y: [2, 3]}\n  - z\nb: |\n  l1\n  l2\n");
    ASSERT_EQ(t.code, y::YamlCode::Ok) << t.what;
    bj::Iterator it{&t.idx, 2, 0};                                // the value of a
    ASSERT_EQ(bj::iter_peek(&it), bj::TokenType::BeginArray);
    ASSERT_TRUE(bj::iter_skip_to_close(&it));
    EXPECT_EQ(t.idx.tokens[it.cursor].type, bj::TokenType::Key);   // b
    EXPECT_NE(t.idx.tokens[it.cursor + 1].flags & bj::kTokenAux, 0);   // block scalar: materialised
    bj::JsonValue* root = nullptr;
    ASSERT_TRUE(bj::value_from_index(&t.idx, &arena, &root));
    const uint8_t* out = nullptr;
    int32_t n = 0;
    ASSERT_TRUE(bj::serialize(root, &arena, &out, &n));
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(out), static_cast<size_t>(n)),
              R"({"a":[{"x":1,"y":[2,3]},"z"],"b":"l1\nl2\n"})");
}

TEST(BoltYamlTape, FilteredBuildEmitsOnlyInterestPaths) {
    bolt::Arena arena;
    const char* src = "base_iri: https://x.example/\ntitle: T\nbig:\n  - {a: 1, b: [1, 2, 3]}\n  - {a: 2}\n"
                      "also:\n  - keep: 1\n    drop: 2\n";
    const Tape all = tape(&arena, src);
    ASSERT_EQ(all.code, y::YamlCode::Ok);
    const char* paths[] = {"/base_iri", "/also/0/keep"};
    const int32_t lens[] = {9, 12};
    bj::PathFilter f{};
    ASSERT_TRUE(bj::compile_paths(paths, lens, 2, &arena, &f));
    const Tape some = tape(&arena, src, &f);
    ASSERT_EQ(some.code, y::YamlCode::Ok) << some.what;
    EXPECT_LT(some.idx.token_count, all.idx.token_count / 2);
    bj::JsonValue* root = nullptr;
    ASSERT_TRUE(bj::value_from_index(&some.idx, &arena, &root));
    const uint8_t* out = nullptr;
    int32_t n = 0;
    ASSERT_TRUE(bj::serialize(root, &arena, &out, &n));
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(out), static_cast<size_t>(n)),
              R"({"base_iri":"https://x.example/","also":[{"keep":1}]})");
    // A filtered build still validates what it skips.
    EXPECT_EQ(tape(&arena, "base_iri: x\nbig: &a 1\n", &f).what, "anchors and aliases");
}

TEST(BoltYamlTape, LinesWithCrLfAndLongIndents) {
    bolt::Arena arena;
    const Tape t = tape(&arena, "a:\r\n                    b:\r\n                      c: 1\r\nd: 2\r\n");
    ASSERT_EQ(t.code, y::YamlCode::Ok) << t.what;
    bj::JsonValue* root = nullptr;
    ASSERT_TRUE(bj::value_from_index(&t.idx, &arena, &root));
    const uint8_t* out = nullptr;
    int32_t n = 0;
    ASSERT_TRUE(bj::serialize(root, &arena, &out, &n));
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(out), static_cast<size_t>(n)), R"({"a":{"b":{"c":1}},"d":2})");
}
