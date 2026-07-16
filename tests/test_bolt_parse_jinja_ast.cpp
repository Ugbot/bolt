// test_bolt_parse_jinja_ast.cpp — coverage for bolt::parse::jinja's
// expression parser (BLLM-72). STL is allowed in tests; production code
// stays Tiger-Style.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "bolt/bolt_arena.h"
#include "bolt/parse/bolt_jinja.h"
#include "bolt/parse/bolt_jinja_ast.h"

using bolt::Arena;
using bolt::ArenaConfig;
using bolt::parse::jinja::BinOp;
using bolt::parse::jinja::Node;
using bolt::parse::jinja::NodeKind;
using bolt::parse::jinja::ParseLimits;
using bolt::parse::jinja::TokenTape;
using bolt::parse::jinja::UnOp;

namespace {

Arena make_arena() {
    ArenaConfig cfg{};
    cfg.initial_block_size = 1u * 1024u * 1024u;
    cfg.max_block_size     = 8u * 1024u * 1024u;
    cfg.alignment           = 8u;
    return Arena(cfg);
}

std::string name_of(const TokenTape& tape, const Node* n) {
    return std::string(reinterpret_cast<const char*>(tape.src) + n->name_start,
                        static_cast<size_t>(n->name_len));
}

// Wraps `expr` in a minimal {{ }} expression tag (the lexer only
// tokenizes identifiers/numbers/operators INSIDE a tag span -- bare
// top-level text is just one opaque Text token), tokenizes it into
// `*tape`, and parses one expression starting right after ExprStart.
//
// `wrapped_storage` must be a named local in the CALLER's scope (not a
// temporary) -- tokenize()/the returned Node tree are zero-copy, so they
// alias `wrapped_storage`'s buffer and it must outlive them.
Node* parse(Arena& a, const std::string& expr, std::string* wrapped_storage,
            TokenTape* tape) {
    *wrapped_storage = "{{ " + expr + " }}";
    if (!bolt::parse::jinja::tokenize(
            reinterpret_cast<const uint8_t*>(wrapped_storage->data()),
            static_cast<int32_t>(wrapped_storage->size()), nullptr, &a,
            tape)) {
        return nullptr;
    }
    int32_t pos = 1;  // skip the ExprStart token
    return bolt::parse::jinja::parse_expression(tape, &pos, nullptr, &a);
}

}  // namespace

TEST(BoltParseJinjaAst, IntLiteral) {
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "42", &wrapped, &tape);
    ASSERT_NE(n, nullptr);
    EXPECT_EQ(n->kind, NodeKind::IntLit);
    EXPECT_EQ(n->int_value, 42);
}

TEST(BoltParseJinjaAst, Identifier) {
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "name", &wrapped, &tape);
    ASSERT_NE(n, nullptr);
    EXPECT_EQ(n->kind, NodeKind::Ident);
    EXPECT_EQ(name_of(tape, n), "name");
}

TEST(BoltParseJinjaAst, BoolAndNoneLiterals) {
    Arena a = make_arena();
    TokenTape tape1, tape2, tape3;
    std::string w1, w2, w3;
    Node* nt = parse(a, "true", &w1, &tape1);
    Node* nf = parse(a, "false", &w2, &tape2);
    Node* nn = parse(a, "none", &w3, &tape3);
    ASSERT_NE(nt, nullptr);
    ASSERT_NE(nf, nullptr);
    ASSERT_NE(nn, nullptr);
    EXPECT_EQ(nt->kind, NodeKind::BoolLit);
    EXPECT_EQ(nt->int_value, 1);
    EXPECT_EQ(nf->kind, NodeKind::BoolLit);
    EXPECT_EQ(nf->int_value, 0);
    EXPECT_EQ(nn->kind, NodeKind::NullLit);
}

TEST(BoltParseJinjaAst, AttributeChain) {
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "a.b.c", &wrapped, &tape);
    ASSERT_NE(n, nullptr);
    ASSERT_EQ(n->kind, NodeKind::Attr);
    EXPECT_EQ(name_of(tape, n), "c");
    Node* b = n->operand[0];
    ASSERT_NE(b, nullptr);
    ASSERT_EQ(b->kind, NodeKind::Attr);
    EXPECT_EQ(name_of(tape, b), "b");
    Node* base = b->operand[0];
    ASSERT_NE(base, nullptr);
    EXPECT_EQ(base->kind, NodeKind::Ident);
    EXPECT_EQ(name_of(tape, base), "a");
}

TEST(BoltParseJinjaAst, IndexAccess) {
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "messages[0]", &wrapped, &tape);
    ASSERT_NE(n, nullptr);
    ASSERT_EQ(n->kind, NodeKind::Index);
    ASSERT_EQ(n->operand[0]->kind, NodeKind::Ident);
    ASSERT_EQ(n->operand[1]->kind, NodeKind::IntLit);
    EXPECT_EQ(n->operand[1]->int_value, 0);
}

TEST(BoltParseJinjaAst, SliceFromIndex) {
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "messages[1:]", &wrapped, &tape);
    ASSERT_NE(n, nullptr);
    ASSERT_EQ(n->kind, NodeKind::Slice);
    EXPECT_TRUE(n->has_lo);
    EXPECT_FALSE(n->has_hi);
    ASSERT_NE(n->operand[1], nullptr);
    EXPECT_EQ(n->operand[1]->int_value, 1);
    EXPECT_EQ(n->operand[2], nullptr);
}

TEST(BoltParseJinjaAst, DictSubscript) {
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "messages[0][\"role\"]", &wrapped, &tape);
    ASSERT_NE(n, nullptr);
    ASSERT_EQ(n->kind, NodeKind::Index);
    ASSERT_EQ(n->operand[1]->kind, NodeKind::StringLit);
    EXPECT_EQ(name_of(tape, n->operand[1]), "role");
}

TEST(BoltParseJinjaAst, FilterNoArgs) {
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "x | trim", &wrapped, &tape);
    ASSERT_NE(n, nullptr);
    ASSERT_EQ(n->kind, NodeKind::Filter);
    EXPECT_EQ(name_of(tape, n), "trim");
    EXPECT_EQ(n->arg_count, 0);
    ASSERT_EQ(n->operand[0]->kind, NodeKind::Ident);
}

TEST(BoltParseJinjaAst, FilterWithArgsChained) {
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "tool.parameters.required|join(\", \")|trim", &wrapped,
                     &tape);
    ASSERT_NE(n, nullptr);
    ASSERT_EQ(n->kind, NodeKind::Filter);
    EXPECT_EQ(name_of(tape, n), "trim");
    Node* join_node = n->operand[0];
    ASSERT_NE(join_node, nullptr);
    ASSERT_EQ(join_node->kind, NodeKind::Filter);
    EXPECT_EQ(name_of(tape, join_node), "join");
    ASSERT_EQ(join_node->arg_count, 1);
    EXPECT_EQ(join_node->args[0]->kind, NodeKind::StringLit);
}

TEST(BoltParseJinjaAst, TestDefined) {
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "x is defined", &wrapped, &tape);
    ASSERT_NE(n, nullptr);
    ASSERT_EQ(n->kind, NodeKind::Test);
    EXPECT_FALSE(n->negated);
    EXPECT_EQ(name_of(tape, n), "defined");
}

TEST(BoltParseJinjaAst, TestNegated) {
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "x is not string", &wrapped, &tape);
    ASSERT_NE(n, nullptr);
    ASSERT_EQ(n->kind, NodeKind::Test);
    EXPECT_TRUE(n->negated);
    EXPECT_EQ(name_of(tape, n), "string");
}

TEST(BoltParseJinjaAst, CallWithArgs) {
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "render_item_list(param_fields.enum, 'enum')",
                     &wrapped, &tape);
    ASSERT_NE(n, nullptr);
    ASSERT_EQ(n->kind, NodeKind::Call);
    ASSERT_EQ(n->operand[0]->kind, NodeKind::Ident);
    EXPECT_EQ(name_of(tape, n->operand[0]), "render_item_list");
    ASSERT_EQ(n->arg_count, 2);
    EXPECT_EQ(n->args[0]->kind, NodeKind::Attr);
    EXPECT_EQ(n->args[1]->kind, NodeKind::StringLit);
}

TEST(BoltParseJinjaAst, ArithmeticPrecedence) {
    // a + b * c  =>  Add(a, Mul(b, c))
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "a + b * c", &wrapped, &tape);
    ASSERT_NE(n, nullptr);
    ASSERT_EQ(n->kind, NodeKind::Binary);
    EXPECT_EQ(n->bin_op, BinOp::Add);
    EXPECT_EQ(n->operand[0]->kind, NodeKind::Ident);
    ASSERT_EQ(n->operand[1]->kind, NodeKind::Binary);
    EXPECT_EQ(n->operand[1]->bin_op, BinOp::Mul);
}

TEST(BoltParseJinjaAst, FilterBindsTighterThanCompare) {
    // item_list | length > 0  =>  Gt(Filter(item_list, length), 0)
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "item_list | length > 0", &wrapped, &tape);
    ASSERT_NE(n, nullptr);
    ASSERT_EQ(n->kind, NodeKind::Binary);
    EXPECT_EQ(n->bin_op, BinOp::Gt);
    ASSERT_EQ(n->operand[0]->kind, NodeKind::Filter);
    EXPECT_EQ(name_of(tape, n->operand[0]), "length");
    EXPECT_EQ(n->operand[1]->kind, NodeKind::IntLit);
}

TEST(BoltParseJinjaAst, AndOrNotPrecedence) {
    // a and not b or c  =>  Or(And(a, Not(b)), c)
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "a and not b or c", &wrapped, &tape);
    ASSERT_NE(n, nullptr);
    ASSERT_EQ(n->kind, NodeKind::Binary);
    EXPECT_EQ(n->bin_op, BinOp::Or);
    Node* lhs = n->operand[0];
    ASSERT_EQ(lhs->kind, NodeKind::Binary);
    EXPECT_EQ(lhs->bin_op, BinOp::And);
    Node* not_b = lhs->operand[1];
    ASSERT_EQ(not_b->kind, NodeKind::Unary);
    EXPECT_EQ(not_b->un_op, UnOp::Not);
}

TEST(BoltParseJinjaAst, NotInOperator) {
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "json_key not in handled_keys", &wrapped, &tape);
    ASSERT_NE(n, nullptr);
    ASSERT_EQ(n->kind, NodeKind::Binary);
    EXPECT_EQ(n->bin_op, BinOp::NotIn);
}

TEST(BoltParseJinjaAst, ConcatIsLeftAssociative) {
    // a ~ b ~ c  =>  Concat(Concat(a, b), c)
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "a ~ b ~ c", &wrapped, &tape);
    ASSERT_NE(n, nullptr);
    ASSERT_EQ(n->kind, NodeKind::Binary);
    EXPECT_EQ(n->bin_op, BinOp::Concat);
    Node* lhs = n->operand[0];
    ASSERT_EQ(lhs->kind, NodeKind::Binary);
    EXPECT_EQ(lhs->bin_op, BinOp::Concat);
}

TEST(BoltParseJinjaAst, ParenthesesOverridePrecedence) {
    // (a + b) * c  =>  Mul(Add(a, b), c)
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "(a + b) * c", &wrapped, &tape);
    ASSERT_NE(n, nullptr);
    ASSERT_EQ(n->kind, NodeKind::Binary);
    EXPECT_EQ(n->bin_op, BinOp::Mul);
    ASSERT_EQ(n->operand[0]->kind, NodeKind::Binary);
    EXPECT_EQ(n->operand[0]->bin_op, BinOp::Add);
}

TEST(BoltParseJinjaAst, UnmatchedParenFails) {
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a, "(a + b", &wrapped, &tape);
    EXPECT_EQ(n, nullptr);
}

TEST(BoltParseJinjaAst, DepthCapFails) {
    Arena a = make_arena();
    TokenTape tape;
    std::string expr;
    for (int i = 0; i < 200; ++i) expr += "(";
    expr += "a";
    for (int i = 0; i < 200; ++i) expr += ")";
    std::string wrapped;
    Node* n = parse(a, expr, &wrapped, &tape);
    EXPECT_EQ(n, nullptr);
}

// Real (whitespace-collapsed) fragments from the two grounding templates
// (Qwen3-Coder-30B-A3B, gpt-oss-120b) -- proves the parser handles the
// actual expression shapes, chained tests/filters included, not just
// synthetic examples.
TEST(BoltParseJinjaAst, RealQwen3CoderCompoundCondition) {
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a,
        "item_list is defined and item_list is iterable and "
        "item_list | length > 0",
        &wrapped, &tape);
    EXPECT_NE(n, nullptr);
}

TEST(BoltParseJinjaAst, RealGptOssCompoundCondition) {
    Arena a = make_arena();
    TokenTape tape;
    std::string wrapped;
    Node* n = parse(a,
        "param_spec.type is defined and param_spec.type is iterable and "
        "param_spec.type is not string and param_spec.type is not mapping "
        "and param_spec.type[0] is defined",
        &wrapped, &tape);
    EXPECT_NE(n, nullptr);
}
