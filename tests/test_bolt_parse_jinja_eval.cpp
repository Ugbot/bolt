// test_bolt_parse_jinja_eval.cpp — coverage for bolt::parse::jinja's
// tree-walking evaluator (BLLM-73). STL is allowed in tests; production
// code stays Tiger-Style.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "bolt/bolt_arena.h"
#include "bolt/parse/bolt_jinja.h"
#include "bolt/parse/bolt_jinja_ast.h"
#include "bolt/parse/bolt_jinja_eval.h"
#include "bolt/parse/bolt_jinja_stmt.h"
#include "bolt/parse/bolt_jinja_value.h"

using bolt::Arena;
using bolt::ArenaConfig;
using bolt::parse::jinja::EvalLimits;
using bolt::parse::jinja::Output;
using bolt::parse::jinja::Scope;
using bolt::parse::jinja::scope_bind;
using bolt::parse::jinja::scope_init;
using bolt::parse::jinja::Stmt;
using bolt::parse::jinja::StmtLimits;
using bolt::parse::jinja::TokenTape;
using bolt::parse::jinja::Value;
using bolt::parse::jinja::value_list;
using bolt::parse::jinja::value_string;

namespace {

Arena make_arena() {
    ArenaConfig cfg{};
    cfg.initial_block_size = 4u * 1024u * 1024u;
    cfg.max_block_size     = 32u * 1024u * 1024u;
    cfg.alignment           = 8u;
    return Arena(cfg);
}

std::string out_str(const Output& out) {
    return std::string(out.data, static_cast<size_t>(out.len));
}

// `src` must be a named local in the caller's scope -- everything downstream
// (TokenTape, Stmt tree, rendered Output) is zero-copy against its buffer.
bool render(Arena& a, const std::string& src, Scope* scope, Output* out,
            const EvalLimits* limits = nullptr) {
    TokenTape tape;
    if (!bolt::parse::jinja::tokenize(
            reinterpret_cast<const uint8_t*>(src.data()),
            static_cast<int32_t>(src.size()), nullptr, &a, &tape)) {
        return false;
    }
    int32_t pos = 0;
    Stmt* doc = bolt::parse::jinja::parse_document(&tape, &pos, nullptr, &a);
    if (doc == nullptr && pos != 0) return false;  // parse failure (empty doc is legit)
    return bolt::parse::jinja::render_document(doc, scope, tape.src, limits, &a, out);
}

}  // namespace

TEST(BoltParseJinjaEval, PlainTextAndVariable) {
    Arena a = make_arena();
    Scope scope;
    scope_init(&scope, nullptr);
    scope_bind(&scope, "name", 4, value_string(&a, "Alice", 5));

    const std::string src = "hi {{ name }}!";
    Output out{};
    ASSERT_TRUE(render(a, src, &scope, &out));
    EXPECT_EQ(out_str(out), "hi Alice!");
}

TEST(BoltParseJinjaEval, UndefinedVariableRendersAsNone) {
    Arena a = make_arena();
    Scope scope;
    scope_init(&scope, nullptr);
    const std::string src = "{{ missing }}";
    Output out{};
    ASSERT_TRUE(render(a, src, &scope, &out));
    EXPECT_EQ(out_str(out), "None");
}

TEST(BoltParseJinjaEval, IfElse) {
    Arena a = make_arena();
    Scope scope;
    scope_init(&scope, nullptr);
    scope_bind(&scope, "x", 1, bolt::parse::jinja::value_bool(&a, true));
    const std::string src = "{% if x %}yes{% else %}no{% endif %}";
    Output out{};
    ASSERT_TRUE(render(a, src, &scope, &out));
    EXPECT_EQ(out_str(out), "yes");
}

TEST(BoltParseJinjaEval, ForLoopWithLoopIndex) {
    Arena a = make_arena();
    Scope scope;
    scope_init(&scope, nullptr);
    Value* items[3] = {
        value_string(&a, "a", 1),
        value_string(&a, "b", 1),
        value_string(&a, "c", 1),
    };
    scope_bind(&scope, "xs", 2, value_list(&a, items, 3));

    const std::string src =
        "{% for x in xs %}{{ loop.index }}:{{ x }} {% endfor %}";
    Output out{};
    ASSERT_TRUE(render(a, src, &scope, &out));
    EXPECT_EQ(out_str(out), "1:a 2:b 3:c ");
}

TEST(BoltParseJinjaEval, SetStatement) {
    Arena a = make_arena();
    Scope scope;
    scope_init(&scope, nullptr);
    const std::string src = "{% set x = 1 + 2 %}{{ x }}";
    Output out{};
    ASSERT_TRUE(render(a, src, &scope, &out));
    EXPECT_EQ(out_str(out), "3");
}

TEST(BoltParseJinjaEval, MacroCallWithDefaultParam) {
    Arena a = make_arena();
    Scope scope;
    scope_init(&scope, nullptr);
    const std::string src =
        "{% macro greet(name, greeting='hi') %}{{ greeting }} "
        "{{ name }}{% endmacro %}{{ greet('Bob') }}|{{ greet('Sam', 'yo') "
        "}}";
    Output out{};
    ASSERT_TRUE(render(a, src, &scope, &out));
    EXPECT_EQ(out_str(out), "hi Bob|yo Sam");
}

TEST(BoltParseJinjaEval, ConcatOperator) {
    Arena a = make_arena();
    Scope scope;
    scope_init(&scope, nullptr);
    const std::string src = "{{ 'a' ~ 1 ~ 'b' ~ true }}";
    Output out{};
    ASSERT_TRUE(render(a, src, &scope, &out));
    EXPECT_EQ(out_str(out), "a1bTrue");
}

TEST(BoltParseJinjaEval, LengthTrimJoinFilters) {
    Arena a = make_arena();
    Scope scope;
    scope_init(&scope, nullptr);
    Value* items[2] = {value_string(&a, "x", 1), value_string(&a, "y", 1)};
    scope_bind(&scope, "xs", 2, value_list(&a, items, 2));
    const std::string src =
        "{{ xs | length }}-{{ '  hi  ' | trim }}-{{ xs | join(', ') }}";
    Output out{};
    ASSERT_TRUE(render(a, src, &scope, &out));
    EXPECT_EQ(out_str(out), "2-hi-x, y");
}

TEST(BoltParseJinjaEval, IsDefinedTest) {
    Arena a = make_arena();
    Scope scope;
    scope_init(&scope, nullptr);
    scope_bind(&scope, "x", 1, value_string(&a, "v", 1));
    const std::string src =
        "{% if x is defined %}yes{% else %}no{% endif %}"
        "{% if y is defined %}yes{% else %}no{% endif %}";
    Output out{};
    ASSERT_TRUE(render(a, src, &scope, &out));
    EXPECT_EQ(out_str(out), "yesno");
}

TEST(BoltParseJinjaEval, LoopIterationCapFails) {
    Arena a = make_arena();
    Scope scope;
    scope_init(&scope, nullptr);
    Value* items[5];
    for (int i = 0; i < 5; ++i) items[i] = bolt::parse::jinja::value_int(&a, i);
    scope_bind(&scope, "xs", 2, value_list(&a, items, 5));
    const std::string src = "{% for x in xs %}{{ x }}{% endfor %}";
    EvalLimits limits;
    limits.max_loop_iterations = 3;
    Output out{};
    EXPECT_FALSE(render(a, src, &scope, &out, &limits));
}

TEST(BoltParseJinjaEval, OutputByteCapFails) {
    Arena a = make_arena();
    Scope scope;
    scope_init(&scope, nullptr);
    const std::string src = "0123456789";
    EvalLimits limits;
    limits.max_output_bytes = 5;
    Output out{};
    EXPECT_FALSE(render(a, src, &scope, &out, &limits));
}

TEST(BoltParseJinjaEval, MacroRecursionDepthCapFails) {
    Arena a = make_arena();
    Scope scope;
    scope_init(&scope, nullptr);
    // Unconditionally recursive macro -- must be stopped by the depth cap,
    // not run away / stack-overflow the process.
    const std::string src =
        "{% macro loop_forever(n) %}{{ loop_forever(n) }}{% endmacro "
        "%}{{ loop_forever(0) }}";
    EvalLimits limits;
    limits.max_macro_depth = 20;
    Output out{};
    EXPECT_FALSE(render(a, src, &scope, &out, &limits));
}

// End-to-end render of the COMPLETE, real render_item_list() macro from
// Qwen3-Coder-30B-A3B-Instruct's embedded chat_template (same fragment
// BLLM-73's statement-parser test already proved parses). Hand-computed
// expected output: the macro's own {%- -%} markers strip all whitespace
// AROUND the "meat" of its body down to "\n<required>[`a`, `b`]</required>",
// but two spots are genuinely untrimmed (no `-` on either side) under
// default (non-trim_blocks) Jinja2 semantics: the newline between the
// macro's inner `{%- endif %}` and its own `{% endmacro %}` (part of the
// macro's OWN rendered body -- hence the leading double newline below),
// and the newline in THIS TEST's outer document between `{% endmacro %}`
// and the `{{ render_item_list(...) }}` call (hence the trailing
// newline). Whether boltllm's real chat-template renderer should instead
// default to trim_blocks=True (as HF's own chat-template environment may)
// is a BLLM-76 wiring decision, not resolved here -- this test locks in
// what THIS evaluator's default (vanilla Jinja2) semantics actually
// produce, verified byte-exact, not just "it didn't crash."
TEST(BoltParseJinjaEval, RealQwen3CoderRenderItemListMacroRendersExactly) {
    Arena a = make_arena();
    Scope scope;
    scope_init(&scope, nullptr);

    const std::string macro_src =
        "{% macro render_item_list(item_list, tag_name='required') %}\n"
        "    {%- if item_list is defined and item_list is iterable and "
        "item_list | length > 0 %}\n"
        "        {%- if tag_name %}{{- '\\n<' ~ tag_name ~ '>' -}}{% endif "
        "%}\n"
        "            {{- '[' }}\n"
        "                {%- for item in item_list -%}\n"
        "                    {%- if loop.index > 1 %}{{- \", \"}}{% endif "
        "-%}\n"
        "                    {%- if item is string -%}\n"
        "                        {{ \"`\" ~ item ~ \"`\" }}\n"
        "                    {%- else -%}\n"
        "                        {{ item }}\n"
        "                    {%- endif -%}\n"
        "                {%- endfor -%}\n"
        "            {{- ']' }}\n"
        "        {%- if tag_name %}{{- '</' ~ tag_name ~ '>' -}}{% endif "
        "%}\n"
        "    {%- endif %}\n"
        "{% endmacro %}\n"
        "{{ render_item_list(item_list, 'required') }}";

    Value* items[2] = {value_string(&a, "a", 1), value_string(&a, "b", 1)};
    scope_bind(&scope, "item_list", 9, value_list(&a, items, 2));

    Output out{};
    ASSERT_TRUE(render(a, macro_src, &scope, &out));
    EXPECT_EQ(out_str(out), "\n\n<required>[`a`, `b`]</required>\n");
}
