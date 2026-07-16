// test_bolt_parse_jinja_stmt.cpp — coverage for bolt::parse::jinja's
// statement parser (BLLM-73). STL is allowed in tests; production code
// stays Tiger-Style.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "bolt/bolt_arena.h"
#include "bolt/parse/bolt_jinja.h"
#include "bolt/parse/bolt_jinja_ast.h"
#include "bolt/parse/bolt_jinja_stmt.h"

using bolt::Arena;
using bolt::ArenaConfig;
using bolt::parse::jinja::Node;
using bolt::parse::jinja::NodeKind;
using bolt::parse::jinja::Stmt;
using bolt::parse::jinja::StmtKind;
using bolt::parse::jinja::StmtLimits;
using bolt::parse::jinja::TokenTape;

namespace {

Arena make_arena() {
    ArenaConfig cfg{};
    cfg.initial_block_size = 2u * 1024u * 1024u;
    cfg.max_block_size     = 16u * 1024u * 1024u;
    cfg.alignment           = 8u;
    return Arena(cfg);
}

std::string text_of(const TokenTape& tape, const Stmt* s) {
    return std::string(reinterpret_cast<const char*>(tape.src) + s->text_start,
                        static_cast<size_t>(s->text_len));
}

std::string name_of(const TokenTape& tape, int32_t start, int32_t len) {
    return std::string(reinterpret_cast<const char*>(tape.src) + start,
                        static_cast<size_t>(len));
}

// `src` must be a named local in the caller's scope -- tokenize()/the
// returned Stmt tree are zero-copy and alias `src`'s buffer.
Stmt* parse(Arena& a, const std::string& src, TokenTape* tape,
            const StmtLimits* limits = nullptr) {
    if (!bolt::parse::jinja::tokenize(
            reinterpret_cast<const uint8_t*>(src.data()),
            static_cast<int32_t>(src.size()), nullptr, &a, tape)) {
        return nullptr;
    }
    int32_t pos = 0;
    return bolt::parse::jinja::parse_document(tape, &pos, limits, &a);
}

}  // namespace

TEST(BoltParseJinjaStmt, PlainTextOnly) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src = "hello world";
    Stmt* doc = parse(a, src, &tape);
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->kind, StmtKind::Text);
    EXPECT_EQ(text_of(tape, doc), "hello world");
    EXPECT_EQ(doc->next, nullptr);
}

TEST(BoltParseJinjaStmt, ExprStatement) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src = "hi {{ name }}!";
    Stmt* doc = parse(a, src, &tape);
    ASSERT_NE(doc, nullptr);
    ASSERT_EQ(doc->kind, StmtKind::Text);
    EXPECT_EQ(text_of(tape, doc), "hi ");
    Stmt* expr_stmt = doc->next;
    ASSERT_NE(expr_stmt, nullptr);
    ASSERT_EQ(expr_stmt->kind, StmtKind::Expr);
    ASSERT_EQ(expr_stmt->expr->kind, NodeKind::Ident);
    Stmt* tail = expr_stmt->next;
    ASSERT_NE(tail, nullptr);
    EXPECT_EQ(tail->kind, StmtKind::Text);
    EXPECT_EQ(text_of(tape, tail), "!");
}

TEST(BoltParseJinjaStmt, IfElse) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src = "{% if x %}A{% else %}B{% endif %}";
    Stmt* doc = parse(a, src, &tape);
    ASSERT_NE(doc, nullptr);
    ASSERT_EQ(doc->kind, StmtKind::If);
    ASSERT_NE(doc->then_body, nullptr);
    EXPECT_EQ(text_of(tape, doc->then_body), "A");
    ASSERT_NE(doc->else_body, nullptr);
    EXPECT_EQ(doc->else_body->kind, StmtKind::Text);
    EXPECT_EQ(text_of(tape, doc->else_body), "B");
    EXPECT_EQ(doc->next, nullptr);
}

TEST(BoltParseJinjaStmt, IfWithoutElse) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src = "{% if x %}A{% endif %}";
    Stmt* doc = parse(a, src, &tape);
    ASSERT_NE(doc, nullptr);
    ASSERT_EQ(doc->kind, StmtKind::If);
    EXPECT_EQ(doc->else_body, nullptr);
}

TEST(BoltParseJinjaStmt, IfElifElse) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src =
        "{% if a %}A{% elif b %}B{% else %}C{% endif %}";
    Stmt* doc = parse(a, src, &tape);
    ASSERT_NE(doc, nullptr);
    ASSERT_EQ(doc->kind, StmtKind::If);
    EXPECT_EQ(text_of(tape, doc->then_body), "A");
    // elif is represented as a nested If in else_body.
    Stmt* elif_node = doc->else_body;
    ASSERT_NE(elif_node, nullptr);
    ASSERT_EQ(elif_node->kind, StmtKind::If);
    EXPECT_EQ(text_of(tape, elif_node->then_body), "B");
    ASSERT_NE(elif_node->else_body, nullptr);
    EXPECT_EQ(elif_node->else_body->kind, StmtKind::Text);
    EXPECT_EQ(text_of(tape, elif_node->else_body), "C");
}

TEST(BoltParseJinjaStmt, ForLoopSingleVar) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src = "{% for item in items %}[{{ item }}]{% endfor %}";
    Stmt* doc = parse(a, src, &tape);
    ASSERT_NE(doc, nullptr);
    ASSERT_EQ(doc->kind, StmtKind::For);
    EXPECT_EQ(name_of(tape, doc->var1_start, doc->var1_len), "item");
    EXPECT_EQ(doc->var2_len, 0);
    ASSERT_EQ(doc->iterable->kind, NodeKind::Ident);
    EXPECT_EQ(name_of(tape, doc->iterable->name_start, doc->iterable->name_len),
              "items");
    ASSERT_NE(doc->body, nullptr);
    EXPECT_EQ(text_of(tape, doc->body), "[");
}

TEST(BoltParseJinjaStmt, ForLoopTwoVars) {
    // The real Qwen3-Coder template's dict-iteration shape:
    // {% for param_name, param_fields in tool.parameters.properties|items %}
    Arena a = make_arena();
    TokenTape tape;
    const std::string src =
        "{% for k, v in d|items %}{{ k }}{% endfor %}";
    Stmt* doc = parse(a, src, &tape);
    ASSERT_NE(doc, nullptr);
    ASSERT_EQ(doc->kind, StmtKind::For);
    EXPECT_EQ(name_of(tape, doc->var1_start, doc->var1_len), "k");
    ASSERT_GT(doc->var2_len, 0);
    EXPECT_EQ(name_of(tape, doc->var2_start, doc->var2_len), "v");
    EXPECT_EQ(doc->iterable->kind, NodeKind::Filter);
}

TEST(BoltParseJinjaStmt, SetStatement) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src = "{% set x = 1 + 2 %}";
    Stmt* doc = parse(a, src, &tape);
    ASSERT_NE(doc, nullptr);
    ASSERT_EQ(doc->kind, StmtKind::Set);
    EXPECT_EQ(name_of(tape, doc->set_name_start, doc->set_name_len), "x");
    ASSERT_NE(doc->set_value, nullptr);
    EXPECT_EQ(doc->set_value->kind, NodeKind::Binary);
}

TEST(BoltParseJinjaStmt, MacroDefWithDefaultParam) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src =
        "{% macro greet(name, greeting='hi') %}{{ greeting }} {{ name "
        "}}{% endmacro %}";
    Stmt* doc = parse(a, src, &tape);
    ASSERT_NE(doc, nullptr);
    ASSERT_EQ(doc->kind, StmtKind::MacroDef);
    EXPECT_EQ(name_of(tape, doc->macro_name_start, doc->macro_name_len),
              "greet");
    ASSERT_EQ(doc->param_count, 2);
    EXPECT_EQ(name_of(tape, doc->params[0].name_start, doc->params[0].name_len),
              "name");
    EXPECT_EQ(doc->params[0].default_value, nullptr);
    EXPECT_EQ(name_of(tape, doc->params[1].name_start, doc->params[1].name_len),
              "greeting");
    ASSERT_NE(doc->params[1].default_value, nullptr);
    EXPECT_EQ(doc->params[1].default_value->kind, NodeKind::StringLit);
    ASSERT_NE(doc->macro_body, nullptr);
}

TEST(BoltParseJinjaStmt, WhitespaceTrimNarrowsTextSpans) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src = "{%- if x -%}   y   {%- endif %}";
    Stmt* doc = parse(a, src, &tape);
    ASSERT_NE(doc, nullptr);
    ASSERT_EQ(doc->kind, StmtKind::If);
    // "   y   " with both sides trim-marked -> just "y".
    ASSERT_NE(doc->then_body, nullptr);
    EXPECT_EQ(text_of(tape, doc->then_body), "y");
}

TEST(BoltParseJinjaStmt, CommentProducesNoStatement) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src = "a{# comment #}b";
    Stmt* doc = parse(a, src, &tape);
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(text_of(tape, doc), "a");
    ASSERT_NE(doc->next, nullptr);
    EXPECT_EQ(text_of(tape, doc->next), "b");
    EXPECT_EQ(doc->next->next, nullptr);
}

TEST(BoltParseJinjaStmt, DanglingEndifFails) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src = "{% endif %}";
    Stmt* doc = parse(a, src, &tape);
    EXPECT_EQ(doc, nullptr);
}

TEST(BoltParseJinjaStmt, UnterminatedIfFails) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src = "{% if x %}unclosed";
    Stmt* doc = parse(a, src, &tape);
    EXPECT_EQ(doc, nullptr);
}

TEST(BoltParseJinjaStmt, UnterminatedForFails) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src = "{% for x in y %}body";
    Stmt* doc = parse(a, src, &tape);
    EXPECT_EQ(doc, nullptr);
}

TEST(BoltParseJinjaStmt, MacroMissingEndmacroFails) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src = "{% macro f() %}body";
    Stmt* doc = parse(a, src, &tape);
    EXPECT_EQ(doc, nullptr);
}

TEST(BoltParseJinjaStmt, StmtDepthCapFails) {
    Arena a = make_arena();
    TokenTape tape;
    std::string src;
    for (int i = 0; i < 100; ++i) src += "{% if x %}";
    src += "y";
    for (int i = 0; i < 100; ++i) src += "{% endif %}";
    StmtLimits limits;
    limits.max_stmt_depth = 10;
    Stmt* doc = parse(a, src, &tape, &limits);
    EXPECT_EQ(doc, nullptr);
}

// The complete, real render_item_list() macro from Qwen3-Coder-30B-A3B-
// Instruct's embedded tokenizer.chat_template (extracted via `dd` from the
// GGUF header, 2026-07-16) -- a fully self-contained, balanced fragment
// (starts at {% macro %}, ends at the matching {% endmacro %}), so it
// parses as a complete document. Proves the statement parser survives the
// real macro/whitespace-control/nested-if/for/filter syntax, not just
// synthetic examples.
TEST(BoltParseJinjaStmt, RealQwen3CoderRenderItemListMacroParses) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src =
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
        "{% endmacro %}\n";
    Stmt* doc = parse(a, src, &tape);
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->kind, StmtKind::MacroDef);
    EXPECT_EQ(name_of(tape, doc->macro_name_start, doc->macro_name_len),
              "render_item_list");
    ASSERT_EQ(doc->param_count, 2);
    ASSERT_NE(doc->params[1].default_value, nullptr);
    EXPECT_EQ(doc->params[1].default_value->kind, NodeKind::StringLit);
}

// Hand-assembled but faithful to gpt-oss-120b's actual macro shape (default
// param, elif chain, nested `is`/`is not` tests, recursive macro call) --
// the real dump wasn't captured to a clean endmacro boundary, so this
// mirrors its exact syntax patterns as a complete, parseable document.
TEST(BoltParseJinjaStmt, GptOssStyleRecursiveMacroParses) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src =
        "{%- macro render_typescript_type(param_spec, required_params, "
        "is_nullable=false) -%}\n"
        "    {%- if param_spec.type == \"array\" -%}\n"
        "        {%- if param_spec['items'] -%}\n"
        "            {%- if param_spec['items']['type'] == \"string\" "
        "-%}\n"
        "                {{- \"string[]\" }}\n"
        "            {%- else -%}\n"
        "                {%- set inner_type = "
        "render_typescript_type(param_spec['items'], required_params) "
        "-%}\n"
        "                {{- inner_type }}\n"
        "            {%- endif -%}\n"
        "        {%- else -%}\n"
        "            {{- \"any[]\" }}\n"
        "        {%- endif -%}\n"
        "    {%- elif param_spec.type is defined and param_spec.type is "
        "not string -%}\n"
        "        {{- param_spec.type }}\n"
        "    {%- endif -%}\n"
        "{%- endmacro -%}\n";
    Stmt* doc = parse(a, src, &tape);
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->kind, StmtKind::MacroDef);
    ASSERT_EQ(doc->param_count, 3);
    ASSERT_NE(doc->params[2].default_value, nullptr);
    EXPECT_EQ(doc->params[2].default_value->kind, NodeKind::BoolLit);
}
