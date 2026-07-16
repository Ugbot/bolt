#pragma once
// ---------------------------------------------------------------------------
// bolt::parse::jinja — lexer for a sandboxed Jinja2 SUBSET used to render
// LLM chat templates (the HF `tokenizer.chat_template` GGUF/tokenizer_config
// KV). boltllm BLLM-70/71.
//
// Scope: this header is the LEXER only. It turns template source into a flat
// token tape: alternating Text runs and tag spans ({{ expr }}, {% stmt %},
// {# comment #}), with the tag interiors further tokenized into identifiers/
// literals/operators/keywords. The parser (AST) and evaluator are follow-on
// layers (BLLM-72/73) built on top of this tape — they are not in this file.
//
// This is a bounded SUBSET of Jinja2, grounded in the two real embedded
// templates found on-box (Qwen3-Coder-30B-A3B's and gpt-oss-120b's
// `tokenizer.chat_template`), not general Jinja/Python:
//   - macros with default params, {%- -%} whitespace control, `is`
//     tests, `~`/`+` concat, slicing, dict `|items`, filters, `loop.index`.
//   - NO arbitrary function calls, no imports, no exec, no file/network
//     access reachable from template source — sandboxing is enforced by
//     the evaluator (BLLM-73/75) never exposing such primitives, not by
//     this lexer.
//
// Conventions (per extern/bolt/CLAUDE.md Tiger Style, matching bolt_json.h):
//   - All allocation through bolt::Arena. No std::*.
//   - noexcept everywhere. Failure -> false / TokenType::Error or End.
//   - Strings are zero-copy slices (byte offset + length into the caller's
//     source buffer); the source must outlive the returned TokenTape.
//   - Hard caps on source size / token count / nesting depth (Limits) --
//     the sandbox's structural half. Exceeding a cap fails tokenize().
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"

namespace bolt {
namespace parse {
namespace jinja {

enum class TokenType : uint8_t {
    // Structural — delimit template regions.
    Text         = 0,   // raw output text outside any {{ }} / {% %} / {# #}
    ExprStart    = 1,   // {{  (or {{- , see Token::trim_before)
    ExprEnd      = 2,   // }}  (or -}} , see Token::trim_after)
    StmtStart    = 3,   // {%
    StmtEnd      = 4,   // %}
    CommentStart = 5,   // {#  (comment body is scanned but not tokenized)
    CommentEnd   = 6,   // #}

    // Inside an expression/statement span.
    Identifier   = 7,
    IntLiteral   = 8,
    FloatLiteral = 9,
    StringLiteral = 10,

    Dot = 11, Comma = 12, Colon = 13, Pipe = 14, Tilde = 15,
    LParen = 16, RParen = 17, LBracket = 18, RBracket = 19,
    Plus = 20, Minus = 21, Star = 22, Slash = 23, Percent = 24,
    Eq = 25, EqEq = 26, NotEq = 27, Lt = 28, Gt = 29, LtEq = 30, GtEq = 31,

    // Keywords (subset actually used by the two grounding templates).
    KwIf = 32, KwElif = 33, KwElse = 34, KwEndif = 35,
    KwFor = 36, KwEndfor = 37, KwIn = 38,
    KwSet = 39, KwMacro = 40, KwEndmacro = 41,
    KwAnd = 42, KwOr = 43, KwNot = 44, KwIs = 45,

    Error = 254,
    End   = 255,
};

// Compact tape entry: points back into the caller's source buffer.
// trim_before/trim_after carry the `-` whitespace-control markers seen on
// tag delimiters (`{%-`, `-%}`, ...) — the lexer records them, the renderer
// (BLLM-73) is the layer that actually strips surrounding whitespace.
struct Token {
    TokenType type;
    bool      trim_before;
    bool      trim_after;
    uint8_t   _pad;
    int32_t   start;    // byte offset into source
    int32_t   length;   // byte span; 0 for zero-width structural tokens
};
static_assert(sizeof(Token) == 12, "Token is 12 bytes");
static_assert(alignof(Token) == 4, "Token aligns on 4 bytes");

// Hard caps — the structural half of BLLM-70's sandboxing requirement.
// Exceeding any of these fails tokenize() outright (no partial tape).
struct Limits {
    int32_t max_source_bytes = 256 * 1024;  // 256KB template source cap
    int32_t max_tokens       = 65536;
    int32_t max_nesting      = 64;          // {% if/for/macro %} nesting depth
};

struct TokenTape {
    const uint8_t* src;
    int32_t        src_len;
    Token*         tokens;
    int32_t        token_count;
};

// Tokenize `src` into `out`, honoring `limits` (nullptr -> defaults).
// Returns false on: source exceeds max_source_bytes, token count exceeds
// max_tokens, unterminated tag ({{ / {% / {# with no matching close),
// unterminated string literal, or arena exhaustion. `out->tokens` is
// arena-allocated; `out->src` aliases the caller's buffer (zero-copy) and
// must outlive `out`.
bool tokenize(const uint8_t* src, int32_t src_len, const Limits* limits,
              Arena* arena, TokenTape* out) noexcept;

}  // namespace jinja
}  // namespace parse
}  // namespace bolt
