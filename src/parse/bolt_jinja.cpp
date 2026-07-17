// bolt_jinja.cpp — lexer implementation for bolt::parse::jinja (BLLM-71).
//
// Single-pass scanner: outer loop finds Text runs and tag delimiters
// ({{ }}, {% %}, {# #}); inside a tag span, a second scanning mode
// tokenizes identifiers/literals/operators/keywords until the matching
// close delimiter (optionally whitespace-trimmed, `-%}` etc.) is found.
//
// Tiger Style: no exceptions, no std::*, arena-only allocation, noexcept.

#include "bolt/parse/bolt_jinja.h"

#include <cassert>
#include <cstring>

namespace bolt {
namespace parse {
namespace jinja {

namespace {

struct Scanner {
    const uint8_t* src;
    int32_t        len;
    int32_t        pos;
    Token*         tokens;
    int32_t        token_cap;
    int32_t        token_count;
    int32_t        nesting;
    const Limits*  limits;
    bool           failed;
};

bool push(Scanner* s, TokenType type, int32_t start, int32_t length,
          bool trim_before, bool trim_after) noexcept {
    assert(s != nullptr);
    assert(length >= 0);
    if (s->token_count >= s->token_cap) {
        s->failed = true;
        return false;
    }
    Token& t = s->tokens[s->token_count++];
    t.type = type;
    t.trim_before = trim_before;
    t.trim_after = trim_after;
    t._pad = 0;
    t.start = start;
    t.length = length;
    return true;
}

bool is_ident_start(uint8_t c) noexcept {
    return (c == '_') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
bool is_ident_char(uint8_t c) noexcept {
    return is_ident_start(c) || (c >= '0' && c <= '9');
}
bool is_digit(uint8_t c) noexcept { return c >= '0' && c <= '9'; }
bool is_space(uint8_t c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

struct Keyword {
    const char* text;
    int32_t     len;
    TokenType   type;
};

// Subset actually used by the two grounding templates (Qwen3-Coder,
// gpt-oss) plus the obvious siblings (endif/endfor/endmacro) they imply.
constexpr Keyword kKeywords[] = {
    {"if", 2, TokenType::KwIf},
    {"elif", 4, TokenType::KwElif},
    {"else", 4, TokenType::KwElse},
    {"endif", 5, TokenType::KwEndif},
    {"for", 3, TokenType::KwFor},
    {"endfor", 6, TokenType::KwEndfor},
    {"in", 2, TokenType::KwIn},
    {"set", 3, TokenType::KwSet},
    {"macro", 5, TokenType::KwMacro},
    {"endmacro", 8, TokenType::KwEndmacro},
    {"and", 3, TokenType::KwAnd},
    {"or", 2, TokenType::KwOr},
    {"not", 3, TokenType::KwNot},
    {"is", 2, TokenType::KwIs},
};
constexpr int32_t kKeywordCount =
    static_cast<int32_t>(sizeof(kKeywords) / sizeof(kKeywords[0]));

TokenType keyword_lookup(const char* p, int32_t len) noexcept {
    for (int32_t i = 0; i < kKeywordCount; ++i) {
        if (kKeywords[i].len == len &&
            memcmp(p, kKeywords[i].text, static_cast<size_t>(len)) == 0) {
            return kKeywords[i].type;
        }
    }
    return TokenType::Identifier;
}

// Scans one comment body ({# ... #} / {# ... -#}) starting at s->pos
// (immediately after the opening delimiter). Emits the CommentEnd token
// and advances s->pos past it. Does not tokenize the comment interior.
bool scan_comment(Scanner* s) noexcept {
    int32_t scan = s->pos;
    int32_t close_pos = -1;
    bool inner_trim = false;
    while (scan + 1 < s->len) {
        if (s->src[scan] == '#' && s->src[scan + 1] == '}') {
            close_pos = scan;
            break;
        }
        if (s->src[scan] == '-' && scan + 2 < s->len &&
            s->src[scan + 1] == '#' && s->src[scan + 2] == '}') {
            close_pos = scan + 1;
            inner_trim = true;
            break;
        }
        scan++;
    }
    if (close_pos < 0) {
        s->failed = true;  // unterminated comment
        return false;
    }
    int32_t end_start = inner_trim ? close_pos - 1 : close_pos;
    s->pos = end_start;
    s->pos += inner_trim ? 3 : 2;
    // `-#}` trims whitespace AFTER this tag (the following Text token),
    // so the marker belongs in trim_after -- trim_before is reserved for
    // an opening tag's `{#-` (trims the PRECEDING Text token).
    return push(s, TokenType::CommentEnd, end_start, s->pos - end_start,
                false, inner_trim);
}

// Scans one {{ expr }} or {% stmt %} span starting at s->pos (immediately
// after the opening delimiter), emitting inner tokens then the matching
// End token. `close0` is '}' for expressions, '%' for statements.
bool scan_tag_body(Scanner* s, TokenType end_type, uint8_t close0) noexcept {
    const int32_t inner_first = s->token_count;
    for (;;) {
        while (s->pos < s->len && is_space(s->src[s->pos])) s->pos++;
        if (s->pos >= s->len) {
            s->failed = true;  // unterminated {{ / {%
            return false;
        }

        const int32_t maybe = s->pos;
        const bool trimmed_close =
            s->src[maybe] == '-' && maybe + 2 < s->len &&
            s->src[maybe + 1] == close0 && s->src[maybe + 2] == '}';
        const bool plain_close = !trimmed_close && s->src[maybe] == close0 &&
                                  maybe + 1 < s->len &&
                                  s->src[maybe + 1] == '}';

        if (trimmed_close || plain_close) {
            const int32_t end_start = s->pos;
            s->pos += trimmed_close ? 3 : 2;
            // `-%}`/`-}}` trims whitespace AFTER this tag (the following
            // Text token) -- belongs in trim_after, mirroring CommentEnd.
            if (!push(s, end_type, end_start, s->pos - end_start,
                      false, trimmed_close)) {
                return false;
            }
            break;
        }

        const uint8_t c = s->src[s->pos];
        if (is_ident_start(c)) {
            const int32_t st = s->pos++;
            while (s->pos < s->len && is_ident_char(s->src[s->pos])) s->pos++;
            const int32_t length = s->pos - st;
            const TokenType tt = keyword_lookup(
                reinterpret_cast<const char*>(s->src + st), length);
            if (!push(s, tt, st, length, false, false)) return false;
        } else if (is_digit(c)) {
            const int32_t st = s->pos;
            bool is_float = false;
            while (s->pos < s->len && is_digit(s->src[s->pos])) s->pos++;
            if (s->pos < s->len && s->src[s->pos] == '.' &&
                s->pos + 1 < s->len && is_digit(s->src[s->pos + 1])) {
                is_float = true;
                s->pos++;
                while (s->pos < s->len && is_digit(s->src[s->pos])) s->pos++;
            }
            if (!push(s, is_float ? TokenType::FloatLiteral
                                   : TokenType::IntLiteral,
                      st, s->pos - st, false, false)) {
                return false;
            }
        } else if (c == '\'' || c == '"') {
            const uint8_t quote = c;
            const int32_t st = s->pos++;
            while (s->pos < s->len && s->src[s->pos] != quote) {
                if (s->src[s->pos] == '\\' && s->pos + 1 < s->len) s->pos++;
                s->pos++;
            }
            if (s->pos >= s->len) {
                s->failed = true;  // unterminated string literal
                return false;
            }
            s->pos++;  // consume closing quote
            if (!push(s, TokenType::StringLiteral, st, s->pos - st, false,
                      false)) {
                return false;
            }
        } else {
            const int32_t st = s->pos;
            TokenType tt;
            if (c == '=' && s->pos + 1 < s->len && s->src[s->pos + 1] == '=') {
                tt = TokenType::EqEq;
                s->pos += 2;
            } else if (c == '!' && s->pos + 1 < s->len &&
                       s->src[s->pos + 1] == '=') {
                tt = TokenType::NotEq;
                s->pos += 2;
            } else if (c == '<' && s->pos + 1 < s->len &&
                       s->src[s->pos + 1] == '=') {
                tt = TokenType::LtEq;
                s->pos += 2;
            } else if (c == '>' && s->pos + 1 < s->len &&
                       s->src[s->pos + 1] == '=') {
                tt = TokenType::GtEq;
                s->pos += 2;
            } else {
                switch (c) {
                    case '.': tt = TokenType::Dot; s->pos++; break;
                    case ',': tt = TokenType::Comma; s->pos++; break;
                    case ':': tt = TokenType::Colon; s->pos++; break;
                    case '|': tt = TokenType::Pipe; s->pos++; break;
                    case '~': tt = TokenType::Tilde; s->pos++; break;
                    case '(': tt = TokenType::LParen; s->pos++; break;
                    case ')': tt = TokenType::RParen; s->pos++; break;
                    case '[': tt = TokenType::LBracket; s->pos++; break;
                    case ']': tt = TokenType::RBracket; s->pos++; break;
                    case '+': tt = TokenType::Plus; s->pos++; break;
                    case '-': tt = TokenType::Minus; s->pos++; break;
                    case '*': tt = TokenType::Star; s->pos++; break;
                    case '/': tt = TokenType::Slash; s->pos++; break;
                    case '%': tt = TokenType::Percent; s->pos++; break;
                    case '=': tt = TokenType::Eq; s->pos++; break;
                    case '<': tt = TokenType::Lt; s->pos++; break;
                    case '>': tt = TokenType::Gt; s->pos++; break;
                    default:
                        s->failed = true;  // unexpected character
                        return false;
                }
            }
            if (!push(s, tt, st, s->pos - st, false, false)) return false;
        }
    }

    // Structural nesting depth (statements only). This is a coarse cap,
    // not full validation (matching if/elif/else pairs is the parser's
    // job, BLLM-73) -- it exists purely so a pathological/adversarial
    // template can't recurse the lexer's caller past a bound.
    if (close0 == '%' && s->token_count > inner_first + 1) {
        const TokenType first_tt = s->tokens[inner_first].type;
        if (first_tt == TokenType::KwIf || first_tt == TokenType::KwFor ||
            first_tt == TokenType::KwMacro) {
            s->nesting++;
            if (s->nesting > s->limits->max_nesting) {
                s->failed = true;
                return false;
            }
        } else if (first_tt == TokenType::KwEndif ||
                   first_tt == TokenType::KwEndfor ||
                   first_tt == TokenType::KwEndmacro) {
            s->nesting--;
            if (s->nesting < 0) {
                s->failed = true;  // unbalanced end tag
                return false;
            }
        }
    }
    return true;
}

}  // namespace

bool tokenize(const uint8_t* src, int32_t src_len, const Limits* limits_in,
              Arena* arena, TokenTape* out) noexcept {
    assert(src != nullptr || src_len == 0);
    assert(arena != nullptr);
    assert(out != nullptr);

    const Limits limits = limits_in ? *limits_in : Limits{};
    if (src_len < 0 || src_len > limits.max_source_bytes) return false;
    if (limits.max_tokens <= 0 || limits.max_nesting < 0) return false;

    Token* tokens = arena->allocate_array<Token>(
        static_cast<size_t>(limits.max_tokens));
    if (tokens == nullptr) return false;

    Scanner s{};
    s.src = src;
    s.len = src_len;
    s.pos = 0;
    s.tokens = tokens;
    s.token_cap = limits.max_tokens;
    s.token_count = 0;
    s.nesting = 0;
    s.limits = &limits;
    s.failed = false;

    while (s.pos < s.len) {
        const int32_t text_start = s.pos;
        while (s.pos < s.len) {
            if (s.pos + 1 < s.len) {
                const uint8_t c0 = s.src[s.pos];
                const uint8_t c1 = s.src[s.pos + 1];
                if (c0 == '{' && (c1 == '{' || c1 == '%' || c1 == '#')) break;
            }
            s.pos++;
        }
        if (s.pos > text_start) {
            if (!push(&s, TokenType::Text, text_start, s.pos - text_start,
                      false, false)) {
                return false;
            }
        }
        if (s.pos >= s.len) break;

        const uint8_t kind = s.src[s.pos + 1];
        const int32_t tag_start = s.pos;
        s.pos += 2;
        bool trim_before = false;
        if (s.pos < s.len && s.src[s.pos] == '-') {
            trim_before = true;
            s.pos++;
        }

        const TokenType start_type = (kind == '{') ? TokenType::ExprStart
                                    : (kind == '%') ? TokenType::StmtStart
                                                    : TokenType::CommentStart;
        if (!push(&s, start_type, tag_start, s.pos - tag_start, trim_before,
                  false)) {
            return false;
        }

        if (kind == '#') {
            if (!scan_comment(&s)) return false;
            continue;
        }

        const TokenType end_type =
            (kind == '{') ? TokenType::ExprEnd : TokenType::StmtEnd;
        const uint8_t close0 = (kind == '{') ? '}' : '%';
        if (!scan_tag_body(&s, end_type, close0)) return false;
    }

    if (s.nesting != 0) return false;  // unbalanced if/for/macro at EOF
    if (!push(&s, TokenType::End, s.len, 0, false, false)) return false;

    out->src = src;
    out->src_len = src_len;
    out->tokens = tokens;
    out->token_count = s.token_count;
    return !s.failed;
}

}  // namespace jinja
}  // namespace parse
}  // namespace bolt
