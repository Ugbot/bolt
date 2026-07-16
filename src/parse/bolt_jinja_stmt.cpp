// bolt_jinja_stmt.cpp — statement parser for bolt::parse::jinja (BLLM-73).
//
// Recursive descent over the lexer's TokenTape, building an intrusive
// linked list of Stmt nodes per block. Tiger Style: no exceptions, no
// std::*, arena-only allocation, noexcept.

#include "bolt/parse/bolt_jinja_stmt.h"

#include <cassert>
#include <cstring>

namespace bolt {
namespace parse {
namespace jinja {

namespace {

enum StopFlag : uint8_t {
    kStopEndif    = 1u << 0,
    kStopElif     = 1u << 1,
    kStopElse     = 1u << 2,
    kStopEndfor   = 1u << 3,
    kStopEndmacro = 1u << 4,
};

struct Parser {
    const TokenTape*   tape;
    int32_t            pos;
    const StmtLimits*  limits;
    Arena*             arena;
    int32_t            node_count;
    int32_t            depth;
    bool               failed;
};

TokenType peek(Parser* p) noexcept { return p->tape->tokens[p->pos].type; }
const Token& peek_tok(Parser* p) noexcept { return p->tape->tokens[p->pos]; }

bool is_end_tag(TokenType t) noexcept {
    return t == TokenType::ExprEnd || t == TokenType::StmtEnd ||
           t == TokenType::CommentEnd;
}
bool is_start_tag(TokenType t) noexcept {
    return t == TokenType::ExprStart || t == TokenType::StmtStart ||
           t == TokenType::CommentStart;
}
bool is_space(uint8_t c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool enter_stmt(Parser* p) noexcept {
    p->depth++;
    if (p->depth > p->limits->max_stmt_depth) {
        p->failed = true;
        return false;
    }
    return true;
}
void leave_stmt(Parser* p) noexcept { p->depth--; }

Stmt* alloc_stmt(Parser* p) noexcept {
    if (p->node_count >= p->limits->max_nodes) {
        p->failed = true;
        return nullptr;
    }
    Stmt* s = static_cast<Stmt*>(p->arena->allocate(sizeof(Stmt), alignof(Stmt)));
    if (s == nullptr) {
        p->failed = true;
        return nullptr;
    }
    p->node_count++;
    memset(s, 0, sizeof(Stmt));
    return s;
}

// Consumes the current token if it matches `t`; fails otherwise.
bool expect(Parser* p, TokenType t) noexcept {
    if (peek(p) != t) {
        p->failed = true;
        return false;
    }
    p->pos++;
    return true;
}

// Narrows [start,len) by dropping leading/trailing ASCII whitespace,
// honoring the trim markers on the tags adjacent to this Text token
// (idx is this Text token's own index in the tape).
void trim_text_span(const TokenTape* tape, int32_t idx, int32_t* out_start,
                     int32_t* out_len) noexcept {
    const Token& t = tape->tokens[idx];
    int32_t start = t.start;
    int32_t len = t.length;

    bool left_trim = false;
    if (idx > 0) {
        const Token& prev = tape->tokens[idx - 1];
        if (is_end_tag(prev.type) && prev.trim_after) left_trim = true;
    }
    bool right_trim = false;
    if (idx + 1 < tape->token_count) {
        const Token& next = tape->tokens[idx + 1];
        if (is_start_tag(next.type) && next.trim_before) right_trim = true;
    }

    if (left_trim) {
        while (len > 0 && is_space(tape->src[start])) { start++; len--; }
    }
    if (right_trim) {
        while (len > 0 && is_space(tape->src[start + len - 1])) { len--; }
    }
    *out_start = start;
    *out_len = len;
}

// Expression-parser limits derived from StmtLimits -- shares the node/depth
// caps rather than exposing a second set of tunables to callers.
ParseLimits expr_limits_from(const StmtLimits* sl) noexcept {
    ParseLimits pl;
    pl.max_nodes = sl->max_nodes;
    pl.max_expr_depth = sl->max_expr_depth;
    return pl;
}

// Parses one expression at the current position, sharing this statement
// parser's node-count/depth caps rather than falling back to
// parse_expression's own independent defaults.
Node* parse_expr_here(Parser* p) noexcept {
    const ParseLimits el = expr_limits_from(p->limits);
    return parse_expression(p->tape, &p->pos, &el, p->arena);
}

Stmt* parse_block(Parser* p, uint8_t stop_mask) noexcept;

// Parses `{% if COND %} then_body [{% elif COND %} body]* [{% else %} body]
// {% endif %}` starting with `p->pos` at the `if`/`elif` keyword token
// itself (StmtStart already consumed by the caller). Builds the elif chain
// as nested If statements in else_body, matching the grammar comment in
// bolt_jinja_stmt.h.
Stmt* parse_if_from_keyword(Parser* p) noexcept {
    if (!enter_stmt(p)) return nullptr;
    p->pos++;  // consume 'if' or 'elif'
    Node* cond = parse_expr_here(p);
    if (cond == nullptr) { p->failed = true; leave_stmt(p); return nullptr; }
    if (!expect(p, TokenType::StmtEnd)) { leave_stmt(p); return nullptr; }

    Stmt* then_body = parse_block(
        p, kStopEndif | kStopElif | kStopElse);
    if (p->failed) { leave_stmt(p); return nullptr; }

    Stmt* node = alloc_stmt(p);
    if (node == nullptr) { leave_stmt(p); return nullptr; }
    node->kind = StmtKind::If;
    node->cond = cond;
    node->then_body = then_body;
    node->else_body = nullptr;

    // p->pos is at a StmtStart whose keyword is elif/else/endif.
    if (!expect(p, TokenType::StmtStart)) { leave_stmt(p); return nullptr; }
    if (peek(p) == TokenType::KwElif) {
        node->else_body = parse_if_from_keyword(p);
        if (node->else_body == nullptr) { leave_stmt(p); return nullptr; }
    } else if (peek(p) == TokenType::KwElse) {
        p->pos++;  // consume 'else'
        if (!expect(p, TokenType::StmtEnd)) { leave_stmt(p); return nullptr; }
        node->else_body = parse_block(p, kStopEndif);
        if (p->failed) { leave_stmt(p); return nullptr; }
        if (!expect(p, TokenType::StmtStart) ||
            !expect(p, TokenType::KwEndif) ||
            !expect(p, TokenType::StmtEnd)) {
            leave_stmt(p);
            return nullptr;
        }
    } else if (peek(p) == TokenType::KwEndif) {
        p->pos++;
        if (!expect(p, TokenType::StmtEnd)) { leave_stmt(p); return nullptr; }
    } else {
        p->failed = true;
        leave_stmt(p);
        return nullptr;
    }

    leave_stmt(p);
    return node;
}

Stmt* parse_for(Parser* p) noexcept {
    if (!enter_stmt(p)) return nullptr;
    p->pos++;  // consume 'for'
    if (peek(p) != TokenType::Identifier) { p->failed = true; leave_stmt(p); return nullptr; }
    const Token& v1 = peek_tok(p);
    p->pos++;

    int32_t v2_start = 0, v2_len = 0;
    if (peek(p) == TokenType::Comma) {
        p->pos++;
        if (peek(p) != TokenType::Identifier) { p->failed = true; leave_stmt(p); return nullptr; }
        const Token& v2 = peek_tok(p);
        v2_start = v2.start;
        v2_len = v2.length;
        p->pos++;
    }

    if (!expect(p, TokenType::KwIn)) { leave_stmt(p); return nullptr; }
    Node* iterable = parse_expr_here(p);
    if (iterable == nullptr) { p->failed = true; leave_stmt(p); return nullptr; }
    if (!expect(p, TokenType::StmtEnd)) { leave_stmt(p); return nullptr; }

    Stmt* body = parse_block(p, kStopEndfor);
    if (p->failed) { leave_stmt(p); return nullptr; }
    if (!expect(p, TokenType::StmtStart) || !expect(p, TokenType::KwEndfor) ||
        !expect(p, TokenType::StmtEnd)) {
        leave_stmt(p);
        return nullptr;
    }

    Stmt* node = alloc_stmt(p);
    if (node == nullptr) { leave_stmt(p); return nullptr; }
    node->kind = StmtKind::For;
    node->var1_start = v1.start;
    node->var1_len = v1.length;
    node->var2_start = v2_start;
    node->var2_len = v2_len;
    node->iterable = iterable;
    node->body = body;
    leave_stmt(p);
    return node;
}

Stmt* parse_set(Parser* p) noexcept {
    if (!enter_stmt(p)) return nullptr;
    p->pos++;  // consume 'set'
    if (peek(p) != TokenType::Identifier) { p->failed = true; leave_stmt(p); return nullptr; }
    const Token& name = peek_tok(p);
    p->pos++;

    // Optional single-level attribute target: `set ns.attr = expr`
    // (namespace()-object mutation -- see bolt_jinja_stmt.h's Set comment).
    int32_t attr_start = 0, attr_len = 0;
    if (peek(p) == TokenType::Dot) {
        p->pos++;
        if (peek(p) != TokenType::Identifier) { p->failed = true; leave_stmt(p); return nullptr; }
        const Token& attr = peek_tok(p);
        attr_start = attr.start;
        attr_len = attr.length;
        p->pos++;
    }

    if (!expect(p, TokenType::Eq)) { leave_stmt(p); return nullptr; }
    Node* value = parse_expr_here(p);
    if (value == nullptr) { p->failed = true; leave_stmt(p); return nullptr; }
    if (!expect(p, TokenType::StmtEnd)) { leave_stmt(p); return nullptr; }

    Stmt* node = alloc_stmt(p);
    if (node == nullptr) { leave_stmt(p); return nullptr; }
    node->kind = StmtKind::Set;
    node->set_name_start = name.start;
    node->set_name_len = name.length;
    node->set_attr_start = attr_start;
    node->set_attr_len = attr_len;
    node->set_value = value;
    leave_stmt(p);
    return node;
}

Stmt* parse_macro(Parser* p) noexcept {
    if (!enter_stmt(p)) return nullptr;
    p->pos++;  // consume 'macro'
    if (peek(p) != TokenType::Identifier) { p->failed = true; leave_stmt(p); return nullptr; }
    const Token& name = peek_tok(p);
    p->pos++;
    if (!expect(p, TokenType::LParen)) { leave_stmt(p); return nullptr; }

    Stmt* node = alloc_stmt(p);
    if (node == nullptr) { leave_stmt(p); return nullptr; }
    node->kind = StmtKind::MacroDef;
    node->macro_name_start = name.start;
    node->macro_name_len = name.length;
    node->param_count = 0;

    if (peek(p) != TokenType::RParen) {
        for (;;) {
            if (node->param_count >= Stmt::kMaxParams) {
                p->failed = true;
                leave_stmt(p);
                return nullptr;
            }
            if (peek(p) != TokenType::Identifier) {
                p->failed = true;
                leave_stmt(p);
                return nullptr;
            }
            const Token& pname = peek_tok(p);
            p->pos++;
            MacroParam& mp = node->params[node->param_count++];
            mp.name_start = pname.start;
            mp.name_len = pname.length;
            mp.default_value = nullptr;
            if (peek(p) == TokenType::Eq) {
                p->pos++;
                mp.default_value =
                    parse_expr_here(p);
                if (mp.default_value == nullptr) {
                    p->failed = true;
                    leave_stmt(p);
                    return nullptr;
                }
            }
            if (peek(p) == TokenType::Comma) { p->pos++; continue; }
            break;
        }
    }
    if (!expect(p, TokenType::RParen)) { leave_stmt(p); return nullptr; }
    if (!expect(p, TokenType::StmtEnd)) { leave_stmt(p); return nullptr; }

    node->macro_body = parse_block(p, kStopEndmacro);
    if (p->failed) { leave_stmt(p); return nullptr; }
    if (!expect(p, TokenType::StmtStart) ||
        !expect(p, TokenType::KwEndmacro) ||
        !expect(p, TokenType::StmtEnd)) {
        leave_stmt(p);
        return nullptr;
    }

    leave_stmt(p);
    return node;
}

Stmt* parse_block(Parser* p, uint8_t stop_mask) noexcept {
    if (!enter_stmt(p)) return nullptr;
    Stmt* head = nullptr;
    Stmt* tail = nullptr;

    auto append = [&](Stmt* s) {
        if (s == nullptr) return;
        if (tail == nullptr) { head = tail = s; }
        else { tail->next = s; tail = s; }
    };

    while (!p->failed) {
        const TokenType t = peek(p);
        if (t == TokenType::End) break;

        if (t == TokenType::Text) {
            const int32_t idx = p->pos;
            int32_t start, len;
            trim_text_span(p->tape, idx, &start, &len);
            p->pos++;
            if (len > 0) {
                Stmt* s = alloc_stmt(p);
                if (s == nullptr) break;
                s->kind = StmtKind::Text;
                s->text_start = start;
                s->text_len = len;
                append(s);
            }
            continue;
        }
        if (t == TokenType::CommentStart) {
            p->pos++;
            if (!expect(p, TokenType::CommentEnd)) break;
            continue;
        }
        if (t == TokenType::ExprStart) {
            p->pos++;
            Node* e = parse_expr_here(p);
            if (e == nullptr) { p->failed = true; break; }
            if (!expect(p, TokenType::ExprEnd)) break;
            Stmt* s = alloc_stmt(p);
            if (s == nullptr) break;
            s->kind = StmtKind::Expr;
            s->expr = e;
            append(s);
            continue;
        }
        if (t == TokenType::StmtStart) {
            // Peek the keyword WITHOUT consuming StmtStart yet -- the
            // stop-set check needs to leave (StmtStart, keyword) untouched
            // for the caller (parse_if_from_keyword/parse_for/... or the
            // top-level parse_document) to consume itself.
            const TokenType kw = p->tape->tokens[p->pos + 1].type;
            const bool is_endif = kw == TokenType::KwEndif;
            const bool is_elif = kw == TokenType::KwElif;
            const bool is_else = kw == TokenType::KwElse;
            const bool is_endfor = kw == TokenType::KwEndfor;
            const bool is_endmacro = kw == TokenType::KwEndmacro;

            if ((is_endif && (stop_mask & kStopEndif)) ||
                (is_elif && (stop_mask & kStopElif)) ||
                (is_else && (stop_mask & kStopElse)) ||
                (is_endfor && (stop_mask & kStopEndfor)) ||
                (is_endmacro && (stop_mask & kStopEndmacro))) {
                break;  // leave p->pos at this StmtStart for the caller
            }

            p->pos++;  // consume StmtStart
            Stmt* s;
            switch (kw) {
                case TokenType::KwIf: s = parse_if_from_keyword(p); break;
                case TokenType::KwFor: s = parse_for(p); break;
                case TokenType::KwSet: s = parse_set(p); break;
                case TokenType::KwMacro: s = parse_macro(p); break;
                default:
                    p->failed = true;
                    s = nullptr;
                    break;
            }
            if (s == nullptr) break;
            append(s);
            continue;
        }

        // Unexpected token at block level (e.g. a stray ExprEnd/StmtEnd).
        p->failed = true;
        break;
    }

    leave_stmt(p);
    return p->failed ? nullptr : head;
}

}  // namespace

Stmt* parse_document(const TokenTape* tape, int32_t* pos,
                     const StmtLimits* limits_in, Arena* arena) noexcept {
    assert(tape != nullptr);
    assert(pos != nullptr);
    assert(arena != nullptr);

    const StmtLimits limits = limits_in ? *limits_in : StmtLimits{};
    if (limits.max_nodes <= 0 || limits.max_expr_depth <= 0 ||
        limits.max_stmt_depth <= 0) {
        return nullptr;
    }

    Parser p{};
    p.tape = tape;
    p.pos = *pos;
    p.limits = &limits;
    p.arena = arena;
    p.node_count = 0;
    p.depth = 0;
    p.failed = false;

    Stmt* doc = parse_block(&p, 0);  // top level: nothing stops it but End
    if (p.failed || peek(&p) != TokenType::End) return nullptr;

    *pos = p.pos;
    return doc;
}

}  // namespace jinja
}  // namespace parse
}  // namespace bolt
