// test_bolt_parse_jinja.cpp — coverage for bolt::parse::jinja's lexer
// (BLLM-71). STL is allowed in tests; production code stays Tiger-Style.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "bolt/bolt_arena.h"
#include "bolt/parse/bolt_jinja.h"

using bolt::Arena;
using bolt::ArenaConfig;
using bolt::parse::jinja::Limits;
using bolt::parse::jinja::Token;
using bolt::parse::jinja::TokenTape;
using bolt::parse::jinja::TokenType;

namespace {

Arena make_arena() {
    ArenaConfig cfg{};
    cfg.initial_block_size = 1u * 1024u * 1024u;
    cfg.max_block_size     = 8u * 1024u * 1024u;
    cfg.alignment           = 8u;
    return Arena(cfg);
}

// NOTE: `s` must be a named local in the caller's scope (not a temporary
// bound to this reference) -- tokenize() is zero-copy, so `out->src`
// aliases `s.data()` and must stay valid for as long as `out` is examined.
bool tok(Arena& a, const std::string& s, TokenTape* out,
         const Limits* limits = nullptr) {
    return bolt::parse::jinja::tokenize(
        reinterpret_cast<const uint8_t*>(s.data()),
        static_cast<int32_t>(s.size()), limits, &a, out);
}

std::string slice(const TokenTape& tape, const Token& t) {
    return std::string(reinterpret_cast<const char*>(tape.src) + t.start,
                        static_cast<size_t>(t.length));
}

}  // namespace

TEST(BoltParseJinja, PlainTextOnly) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src = "hello world";
    ASSERT_TRUE(tok(a, src, &tape));
    ASSERT_EQ(tape.token_count, 2);
    EXPECT_EQ(tape.tokens[0].type, TokenType::Text);
    EXPECT_EQ(slice(tape, tape.tokens[0]), "hello world");
    EXPECT_EQ(tape.tokens[1].type, TokenType::End);
}

TEST(BoltParseJinja, EmptySource) {
    Arena a = make_arena();
    TokenTape tape;
    ASSERT_TRUE(tok(a, "", &tape));
    ASSERT_EQ(tape.token_count, 1);
    EXPECT_EQ(tape.tokens[0].type, TokenType::End);
}

TEST(BoltParseJinja, SimpleExpression) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src = "hi {{ name }}!";
    ASSERT_TRUE(tok(a, src, &tape));
    // Text("hi ") ExprStart Identifier(name) ExprEnd Text("!") End
    ASSERT_EQ(tape.token_count, 6);
    EXPECT_EQ(tape.tokens[0].type, TokenType::Text);
    EXPECT_EQ(slice(tape, tape.tokens[0]), "hi ");
    EXPECT_EQ(tape.tokens[1].type, TokenType::ExprStart);
    EXPECT_FALSE(tape.tokens[1].trim_before);
    EXPECT_EQ(tape.tokens[2].type, TokenType::Identifier);
    EXPECT_EQ(slice(tape, tape.tokens[2]), "name");
    EXPECT_EQ(tape.tokens[3].type, TokenType::ExprEnd);
    EXPECT_EQ(tape.tokens[4].type, TokenType::Text);
    EXPECT_EQ(slice(tape, tape.tokens[4]), "!");
    EXPECT_EQ(tape.tokens[5].type, TokenType::End);
}

TEST(BoltParseJinja, WhitespaceTrimMarkers) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src = "{%- if x -%}y{% endif %}";
    ASSERT_TRUE(tok(a, src, &tape));
    EXPECT_EQ(tape.tokens[0].type, TokenType::StmtStart);
    EXPECT_TRUE(tape.tokens[0].trim_before);
    ASSERT_EQ(tape.tokens[1].type, TokenType::KwIf);
    ASSERT_EQ(tape.tokens[2].type, TokenType::Identifier);
    EXPECT_EQ(slice(tape, tape.tokens[2]), "x");
    ASSERT_EQ(tape.tokens[3].type, TokenType::StmtEnd);
    // `-%}` trims whitespace AFTER this tag -- that's trim_after, not
    // trim_before (trim_before is reserved for an opening `{%-` marker).
    EXPECT_TRUE(tape.tokens[3].trim_after);
    EXPECT_FALSE(tape.tokens[3].trim_before);
}

TEST(BoltParseJinja, StringAndNumberLiterals) {
    Arena a = make_arena();
    TokenTape tape;
    ASSERT_TRUE(tok(a, "{{ 'a\\'b' }}{{ 12 }}{{ 3.5 }}", &tape));
    EXPECT_EQ(tape.tokens[1].type, TokenType::StringLiteral);
    int32_t idx = 4;  // Text? none here since back-to-back tags -> no Text tokens between
    // Find IntLiteral / FloatLiteral by scanning.
    bool saw_int = false, saw_float = false;
    for (int32_t i = 0; i < tape.token_count; ++i) {
        if (tape.tokens[i].type == TokenType::IntLiteral) saw_int = true;
        if (tape.tokens[i].type == TokenType::FloatLiteral) saw_float = true;
    }
    EXPECT_TRUE(saw_int);
    EXPECT_TRUE(saw_float);
    (void)idx;
}

TEST(BoltParseJinja, OperatorsAndFilters) {
    Arena a = make_arena();
    TokenTape tape;
    ASSERT_TRUE(tok(a, "{{ a ~ b | trim | length > 0 }}", &tape));
    bool saw_tilde = false, saw_pipe = false, saw_gt = false;
    for (int32_t i = 0; i < tape.token_count; ++i) {
        if (tape.tokens[i].type == TokenType::Tilde) saw_tilde = true;
        if (tape.tokens[i].type == TokenType::Pipe) saw_pipe = true;
        if (tape.tokens[i].type == TokenType::Gt) saw_gt = true;
    }
    EXPECT_TRUE(saw_tilde);
    EXPECT_TRUE(saw_pipe);
    EXPECT_TRUE(saw_gt);
}

TEST(BoltParseJinja, CommentIsSkippedNotTokenized) {
    Arena a = make_arena();
    TokenTape tape;
    const std::string src = "a{# this has {{ junk }} inside #}b";
    ASSERT_TRUE(tok(a, src, &tape));
    // Text("a") CommentStart CommentEnd Text("b") End -- the comment body
    // (including the {{ junk }} inside it) must NOT be tokenized.
    ASSERT_EQ(tape.token_count, 5);
    EXPECT_EQ(tape.tokens[0].type, TokenType::Text);
    EXPECT_EQ(tape.tokens[1].type, TokenType::CommentStart);
    EXPECT_EQ(tape.tokens[2].type, TokenType::CommentEnd);
    EXPECT_EQ(tape.tokens[3].type, TokenType::Text);
    EXPECT_EQ(slice(tape, tape.tokens[3]), "b");
}

TEST(BoltParseJinja, MacroAndForKeywords) {
    Arena a = make_arena();
    TokenTape tape;
    ASSERT_TRUE(tok(a,
        "{% macro f(x, y=1) %}{% for i in items %}{{ i }}{% endfor %}"
        "{% endmacro %}",
        &tape));
    bool saw_macro = false, saw_for = false, saw_endfor = false,
         saw_endmacro = false, saw_in = false;
    for (int32_t i = 0; i < tape.token_count; ++i) {
        switch (tape.tokens[i].type) {
            case TokenType::KwMacro: saw_macro = true; break;
            case TokenType::KwFor: saw_for = true; break;
            case TokenType::KwEndfor: saw_endfor = true; break;
            case TokenType::KwEndmacro: saw_endmacro = true; break;
            case TokenType::KwIn: saw_in = true; break;
            default: break;
        }
    }
    EXPECT_TRUE(saw_macro);
    EXPECT_TRUE(saw_for);
    EXPECT_TRUE(saw_endfor);
    EXPECT_TRUE(saw_endmacro);
    EXPECT_TRUE(saw_in);
}

TEST(BoltParseJinja, UnterminatedExpressionFails) {
    Arena a = make_arena();
    TokenTape tape;
    EXPECT_FALSE(tok(a, "{{ name", &tape));
}

TEST(BoltParseJinja, UnterminatedStringFails) {
    Arena a = make_arena();
    TokenTape tape;
    EXPECT_FALSE(tok(a, "{{ 'unterminated }}", &tape));
}

TEST(BoltParseJinja, UnbalancedEndifFails) {
    Arena a = make_arena();
    TokenTape tape;
    EXPECT_FALSE(tok(a, "{% endif %}", &tape));
}

TEST(BoltParseJinja, UnbalancedIfAtEofFails) {
    Arena a = make_arena();
    TokenTape tape;
    EXPECT_FALSE(tok(a, "{% if x %}unclosed", &tape));
}

TEST(BoltParseJinja, SourceTooLargeFails) {
    Arena a = make_arena();
    TokenTape tape;
    Limits limits;
    limits.max_source_bytes = 4;
    EXPECT_FALSE(tok(a, "way too long", &tape, &limits));
}

TEST(BoltParseJinja, TokenCapFails) {
    Arena a = make_arena();
    TokenTape tape;
    Limits limits;
    limits.max_tokens = 2;  // not enough for even one real token + End
    EXPECT_FALSE(tok(a, "{{ name }}", &tape, &limits));
}

TEST(BoltParseJinja, NestingCapFails) {
    Arena a = make_arena();
    TokenTape tape;
    Limits limits;
    limits.max_nesting = 1;
    EXPECT_FALSE(tok(a,
        "{% if a %}{% if b %}x{% endif %}{% endif %}", &tape, &limits));
}

// Real snippet lifted from Qwen3-Coder-30B-A3B-Instruct's embedded
// tokenizer.chat_template (extracted via `dd` from the GGUF header,
// 2026-07-16) -- proves the lexer survives real-world macro/whitespace-
// control/filter/test syntax, not just synthetic examples.
TEST(BoltParseJinja, RealQwen3CoderTemplateSnippetTokenizes) {
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
        "                {%- endfor -%}\n"
        "            {{- ']' }}\n"
        "    {%- endif %}\n"
        "{% endmacro %}\n";
    EXPECT_TRUE(tok(a, src, &tape));
}

// Real snippet from gpt-oss-120b's embedded chat_template -- recursive
// macro call + negated `is not` tests.
TEST(BoltParseJinja, RealGptOssTemplateSnippetTokenizes) {
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
        "            {%- endif -%}\n"
        "        {%- endif -%}\n"
        "    {%- elif param_spec.type is defined and param_spec.type is "
        "iterable and param_spec.type is not string and param_spec.type "
        "is not mapping and param_spec.type[0] is defined -%}\n"
        "        {{- param_spec.type[0] }}\n"
        "    {%- endif -%}\n"
        "{%- endmacro -%}\n";
    EXPECT_TRUE(tok(a, src, &tape));
}
