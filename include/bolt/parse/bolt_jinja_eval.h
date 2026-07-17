#pragma once
// ---------------------------------------------------------------------------
// bolt::parse::jinja — tree-walking evaluator (BLLM-73), the last layer:
// walks the statement AST (bolt_jinja_stmt.h) + expression AST
// (bolt_jinja_ast.h) against a Scope of bound variables (bolt_jinja_value.h)
// and renders template output.
//
// Sandbox bounds enforced here (the runtime half of BLLM-70/75, alongside
// the parse-time Limits in the lexer/parsers): a loop-iteration ceiling
// (shared across ALL loops in one render, not per-loop), a macro
// call-depth ceiling (recursion bound -- macros like gpt-oss's
// render_typescript_type call themselves), and a total-output-bytes
// ceiling (shared across the top-level render AND every nested macro-body
// render, since a macro invocation's rendered body becomes a string Value
// that still counts against the same budget).
//
// Conventions: arena-only allocation, noexcept, no std::*.
// ---------------------------------------------------------------------------

#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/parse/bolt_jinja_stmt.h"
#include "bolt/parse/bolt_jinja_value.h"

namespace bolt {
namespace parse {
namespace jinja {

struct EvalLimits {
    int32_t max_loop_iterations = 100000;
    int32_t max_output_bytes    = 4 * 1024 * 1024;
    int32_t max_macro_depth     = 64;
};

struct Output {
    char*   data;
    int32_t len;
    int32_t cap;
};

// Evaluates one expression against `scope`/`globals`. `source` is the
// template's original byte buffer (the same one the TokenTape aliased) --
// needed to materialize Ident/Attr/Filter/Test names and decode
// StringLit content. Returns nullptr only on a sandbox-limit violation or
// allocation failure; anything else that can't be resolved (unbound
// identifier, unsupported call, missing dict key) evaluates to an
// Undefined Value rather than failing the whole render.
Value* eval_expression(const Node* expr, Scope* scope, Scope* globals,
                       const uint8_t* source, const EvalLimits* limits,
                       Arena* arena, int32_t* loop_iterations_used,
                       int32_t* macro_depth, int32_t* output_bytes_used,
                       bool* failed) noexcept;

// Renders `doc` (from parse_document) into `out`, which must already be
// zero-initialized ({nullptr, 0, 0}) -- it grows via `arena` as needed.
// `scope` is both the initial variable scope AND the scope macro
// definitions bind into (top-level MacroDef statements make their macro
// callable by anything rendered after them, including from nested
// scopes, since macro lookup walks the parent chain up to this scope).
// Returns false on a sandbox-limit violation or allocation failure.
bool render_document(const Stmt* doc, Scope* scope, const uint8_t* source,
                     const EvalLimits* limits, Arena* arena,
                     Output* out) noexcept;

}  // namespace jinja
}  // namespace parse
}  // namespace bolt
