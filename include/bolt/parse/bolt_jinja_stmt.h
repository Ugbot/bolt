#pragma once
// ---------------------------------------------------------------------------
// bolt::parse::jinja — statement parser (BLLM-73), built on the expression
// parser's AST (bolt_jinja_ast.h, BLLM-72) and the lexer's TokenTape
// (bolt_jinja.h, BLLM-71). Evaluation lives in bolt_jinja_eval.h.
//
// Grammar (subset grounded in the two real templates this project targets,
// see BLLM-70's description):
//   block   := (Text | '{{' expr '}}' | '{#' ... '#}' | stmt)*
//   stmt    := if_stmt | for_stmt | set_stmt | macro_stmt
//   if_stmt := '{%' 'if' expr '%}' block
//              ( '{%' 'elif' expr '%}' block )*
//              ( '{%' 'else' '%}' block )?
//              '{%' 'endif' '%}'
//   for_stmt := '{%' 'for' ident (',' ident)? 'in' expr '%}' block
//               '{%' 'endfor' '%}'
//   set_stmt := '{%' 'set' ident '=' expr '%}'
//   macro_stmt := '{%' 'macro' ident '(' params ')' '%}' block
//                 '{%' 'endmacro' '%}'
//   params  := (ident ('=' expr)? (',' ident ('=' expr)?)*)?
//
// Whitespace control ({%- / -%} etc.) is resolved HERE: a Text statement's
// span is narrowed (leading/trailing bytes dropped) when the adjacent tag
// carried a trim marker, per the lexer's Token::trim_before/trim_after.
//
// Conventions: arena-only allocation, noexcept, no std::*. Statement lists
// are intrusive singly-linked lists (Stmt::next) -- Tiger Style has no
// std::vector, and the list length is unbounded-but-capped only by the
// overall node-count limit, not a fixed array.
// ---------------------------------------------------------------------------

#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/parse/bolt_jinja.h"
#include "bolt/parse/bolt_jinja_ast.h"

namespace bolt {
namespace parse {
namespace jinja {

enum class StmtKind : uint8_t {
    Text = 0,   // raw output span [text_start, text_start+text_len)
    Expr = 1,   // {{ expr }} -- output the stringified value of `expr`
    If = 2,     // cond / then_body / else_body (else_body may itself be a
                // single synthetic If statement representing an elif, or a
                // plain statement list for a real else, or null for neither)
    For = 3,    // var1 (+ optional var2) bound from iterating `iterable`, over `body`
    Set = 4,    // set_name = set_value
    MacroDef = 5,  // defines `macro_name` with `params`/`macro_body`, callable via Call nodes
};

struct MacroParam {
    int32_t name_start;
    int32_t name_len;
    Node*   default_value;  // nullable: no default
};

struct Stmt {
    StmtKind kind;

    // Text
    int32_t text_start;
    int32_t text_len;

    // Expr
    Node* expr;

    // If
    Node* cond;
    Stmt* then_body;
    Stmt* else_body;

    // For
    int32_t var1_start, var1_len;
    int32_t var2_start, var2_len;  // var2_len == 0 => single-variable form
    Node*   iterable;
    Stmt*   body;

    // Set: `{% set name = expr %}`, or `{% set name.attr = expr %}` (a
    // single-level attribute-mutation target, e.g. a namespace() object --
    // set_attr_len == 0 means the plain-variable form; > 0 means `name`
    // must already resolve to a mutable Dict/namespace object and `attr`
    // is mutated in place (see value_dict_set, bolt_jinja_value.h).
    int32_t set_name_start, set_name_len;
    int32_t set_attr_start, set_attr_len;
    Node*   set_value;

    // MacroDef
    int32_t macro_name_start, macro_name_len;
    static constexpr int32_t kMaxParams = 8;
    MacroParam params[kMaxParams];
    int32_t    param_count;
    Stmt*      macro_body;

    Stmt* next;  // next sibling in the enclosing statement list
};

struct StmtLimits {
    int32_t max_nodes = 65536;       // shared node-count cap (Stmt + Node allocations)
    int32_t max_expr_depth = 64;     // forwarded to parse_expression (BLLM-72)
    int32_t max_stmt_depth = 64;     // if/for/macro nesting depth -- BLLM-70 sandbox bound
};

// Parses a full template document (the whole TokenTape, from `pos` -- pass
// 0 to start at the beginning -- to TokenType::End) into a statement list.
// Returns nullptr on any syntax error (unmatched if/for/macro, malformed
// params, node-count cap exceeded, or statement-nesting depth exceeded).
Stmt* parse_document(const TokenTape* tape, int32_t* pos,
                     const StmtLimits* limits, Arena* arena) noexcept;

}  // namespace jinja
}  // namespace parse
}  // namespace bolt
