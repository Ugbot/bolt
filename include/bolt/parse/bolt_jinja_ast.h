#pragma once
// ---------------------------------------------------------------------------
// bolt::parse::jinja — expression parser / AST (BLLM-72), built on the
// lexer's TokenTape (bolt_jinja.h, BLLM-71). Statement parsing (if/for/set/
// macro) and evaluation are follow-on layers (BLLM-73) that walk this AST.
//
// Grammar (subset grounded in the two real templates this project targets,
// see BLLM-70's description), lowest to highest precedence:
//   expr       := ternary_expr
//   ternary_expr := or_expr ( 'if' or_expr 'else' ternary_expr )?
//   or_expr    := and_expr ( 'or' and_expr )*
//   and_expr   := not_expr ( 'and' not_expr )*
//   not_expr   := 'not' not_expr | compare_expr
//   compare_expr := concat_expr ( ('=='|'!='|'<'|'>'|'<='|'>='|'in'|'not' 'in') concat_expr )?
//   concat_expr := additive_expr ( '~' additive_expr )*
//   additive_expr := mul_expr ( ('+'|'-') mul_expr )*
//   mul_expr   := unary_expr ( ('*'|'/'|'%') unary_expr )*
//   unary_expr := '-' unary_expr | postfix_expr
//   postfix_expr := primary ( '.' ident
//                            | '[' expr (':' expr?)? ']'
//                            | '(' args ')'
//                            | '|' ident ('(' args ')')?
//                            | 'is' 'not'? ident ('(' args ')')?
//                            )*
//   primary    := NUMBER | STRING | 'true' | 'false' | 'none' | ident
//               | '(' or_expr ')' | '[' (or_expr (',' or_expr)*)? ']'
//
// Conventions: arena-only allocation, noexcept, no std::*. AST nodes are
// POD; string/identifier content is zero-copy (byte spans into the same
// source buffer the TokenTape aliases) -- escape decoding for string
// literals is deferred to the evaluator (BLLM-73), not done here.
// ---------------------------------------------------------------------------

#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/parse/bolt_jinja.h"

namespace bolt {
namespace parse {
namespace jinja {

// Max positional arguments for a Call / Filter / Test node. Generous for
// the grounding templates (the widest real call, render_item_list, takes
// 2 args) with headroom; a hard cap, not a soft one -- exceeding it fails
// the parse (part of BLLM-70's sandboxing).
static constexpr int32_t kMaxArgs = 6;

enum class NodeKind : uint8_t {
    NullLit = 0, BoolLit, IntLit, FloatLit, StringLit,
    Ident,
    Attr,     // operand[0].name             (dotted attribute access)
    Index,    // operand[0][operand[1]]      (subscript)
    Slice,    // operand[0][operand[1]:operand[2]]  (either bound may be null)
    Call,     // operand[0](args...)         (operand[0] is the callee, e.g. Ident or Attr)
    Filter,   // operand[0] | name(args...)
    Test,     // operand[0] is [not] name(args...)
    Unary,    // op operand[0]
    Binary,   // operand[0] op operand[1]
    ListLit,  // [args...]  (a list literal, e.g. ['a', 'b'] or [])
    Ternary,  // operand[0] if operand[1] else operand[2]  (Jinja/Python conditional expression)
};

enum class BinOp : uint8_t {
    Or, And, Eq, Ne, Lt, Gt, Le, Ge, In, NotIn, Concat, Add, Sub, Mul, Div, Mod,
};
enum class UnOp : uint8_t { Not, Neg };

struct Node {
    NodeKind kind;
    union {
        BinOp bin_op;   // valid when kind == Binary
        UnOp  un_op;    // valid when kind == Unary
    };
    bool     negated;    // Test only: `is not` vs `is`
    uint8_t  _pad;
    int32_t  int_value;    // IntLit value, or BoolLit (0/1)
    double   float_value;  // FloatLit value
    // Ident: variable name. Attr/Filter/Test: the field/filter/test name.
    // StringLit: raw content BETWEEN the quotes (escapes not yet decoded).
    int32_t  name_start;
    int32_t  name_len;
    Node*    operand[3];           // meaning depends on `kind` (see enum comments)
    Node*    args[kMaxArgs];       // Call/Filter/Test extra arguments
    int32_t  arg_count;
    // Keyword-argument names for Call args, e.g. namespace(browser=false) --
    // arg_name_len[i] == 0 means args[i] is positional. Filter/Test args are
    // always positional in the grounding templates (never populated there).
    int32_t  arg_name_start[kMaxArgs];
    int32_t  arg_name_len[kMaxArgs];
    bool     has_lo;               // Slice only: operand[1] (lo) present
    bool     has_hi;                // Slice only: operand[2] (hi) present
};

struct ParseLimits {
    int32_t max_nodes = 16384;
    int32_t max_expr_depth = 64;   // recursion depth cap -- BLLM-70 sandbox bound
};

// Parses ONE expression (the grammar above) starting at `*pos` (an index
// into `tape->tokens`), stopping at the first token the grammar can't
// consume (e.g. a StmtEnd/ExprEnd, or an unrelated statement keyword).
// Advances `*pos` past the consumed tokens on success. Returns nullptr on
// syntax error, node-count cap exceeded, or expr-depth cap exceeded.
Node* parse_expression(const TokenTape* tape, int32_t* pos,
                        const ParseLimits* limits, Arena* arena) noexcept;

}  // namespace jinja
}  // namespace parse
}  // namespace bolt
