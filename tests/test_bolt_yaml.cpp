// G2GRAPH-453 — bolt::parse::yaml::to_json.
#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "bolt/bolt_arena.h"
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
                R"({"a":1,"b":-2,"c":31,"d":15,"e":1.5,"f":2000.0,"g":0.5})");
    expect_json("a: true\nb: False\nc: ~\nd: null\ne:\nf: yes\n",
                R"({"a":true,"b":false,"c":null,"d":null,"e":null,"f":"yes"})");
    expect_json("d: 2026-12-31\nt: 2026-06-30T12:00:00Z\nv: \"2.1\"\n",
                R"({"d":"2026-12-31","t":"2026-06-30T12:00:00Z","v":"2.1"})");
    expect_json("u: https://x.example/a:b#frag\nh: human:jsmith@acme\n",
                R"({"u":"https://x.example/a:b#frag","h":"human:jsmith@acme"})");
    expect_json("q: 'it''s'\nd: \"tab\\there \\u00e9 \\\"q\\\"\"\n",
                "{\"q\":\"it's\",\"d\":\"tab\\there \xc3\xa9 \\\"q\\\"\"}");
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
