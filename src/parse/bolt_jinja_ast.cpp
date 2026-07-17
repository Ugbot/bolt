// bolt_jinja_ast.cpp — expression parser for bolt::parse::jinja (BLLM-72).
//
// Precedence-climbing recursive descent over the lexer's TokenTape. See
// bolt_jinja_ast.h for the grammar. Tiger Style: no exceptions, no std::*,
// arena-only allocation, noexcept.

#include "bolt/parse/bolt_jinja_ast.h"

#include <cassert>
#include <cstring>

namespace bolt {
namespace parse {
namespace jinja {

namespace {

struct Parser {
    const TokenTape*   tape;
    int32_t            pos;
    const ParseLimits* limits;
    Arena*             arena;
    int32_t            node_count;
    int32_t            depth;
    bool               failed;
};

TokenType peek(Parser* p) noexcept {
    return p->tape->tokens[p->pos].type;
}

const Token& peek_tok(Parser* p) noexcept {
    return p->tape->tokens[p->pos];
}

bool at_end_of_expr(Parser* p) noexcept {
    const TokenType t = peek(p);
    return t == TokenType::ExprEnd || t == TokenType::StmtEnd ||
           t == TokenType::End;
}

Node* alloc_node(Parser* p) noexcept {
    if (p->node_count >= p->limits->max_nodes) {
        p->failed = true;
        return nullptr;
    }
    Node* n = static_cast<Node*>(p->arena->allocate(sizeof(Node), alignof(Node)));
    if (n == nullptr) {
        p->failed = true;
        return nullptr;
    }
    p->node_count++;
    memset(n, 0, sizeof(Node));
    return n;
}

// RAII-free depth guard: call enter_expr() at the top of every recursive
// parse_* function that can recurse into itself (directly or via the
// grammar chain), and leave_expr() before every return. This is BLLM-70's
// sandbox recursion-depth bound applied to the parser itself.
bool enter_expr(Parser* p) noexcept {
    p->depth++;
    if (p->depth > p->limits->max_expr_depth) {
        p->failed = true;
        return false;
    }
    return true;
}
void leave_expr(Parser* p) noexcept { p->depth--; }

Node* parse_or(Parser* p) noexcept;
Node* parse_ternary(Parser* p) noexcept;
bool parse_arg_list(Parser* p, TokenType close, Node** out, int32_t* out_count,
                    int32_t* name_starts, int32_t* name_lens) noexcept;

Node* parse_primary(Parser* p) noexcept {
    if (!enter_expr(p)) return nullptr;
    const Token& t = peek_tok(p);
    Node* n = nullptr;

    switch (t.type) {
        case TokenType::IntLiteral: {
            n = alloc_node(p);
            if (n == nullptr) break;
            n->kind = NodeKind::IntLit;
            int64_t v = 0;
            for (int32_t i = 0; i < t.length; ++i) {
                v = v * 10 + (p->tape->src[t.start + i] - '0');
            }
            n->int_value = static_cast<int32_t>(v);
            p->pos++;
            break;
        }
        case TokenType::FloatLiteral: {
            n = alloc_node(p);
            if (n == nullptr) break;
            n->kind = NodeKind::FloatLit;
            n->name_start = t.start;
            n->name_len = t.length;
            p->pos++;
            break;
        }
        case TokenType::StringLiteral: {
            n = alloc_node(p);
            if (n == nullptr) break;
            n->kind = NodeKind::StringLit;
            // Content between the quotes, escapes decoded at eval time.
            n->name_start = t.start + 1;
            n->name_len = t.length - 2;
            p->pos++;
            break;
        }
        case TokenType::Identifier: {
            const int32_t len = t.length;
            const char* txt = reinterpret_cast<const char*>(p->tape->src) + t.start;
            if (len == 4 && memcmp(txt, "true", 4) == 0) {
                n = alloc_node(p);
                if (n == nullptr) break;
                n->kind = NodeKind::BoolLit;
                n->int_value = 1;
            } else if (len == 5 && memcmp(txt, "false", 5) == 0) {
                n = alloc_node(p);
                if (n == nullptr) break;
                n->kind = NodeKind::BoolLit;
                n->int_value = 0;
            } else if (len == 4 && memcmp(txt, "none", 4) == 0) {
                n = alloc_node(p);
                if (n == nullptr) break;
                n->kind = NodeKind::NullLit;
            } else {
                n = alloc_node(p);
                if (n == nullptr) break;
                n->kind = NodeKind::Ident;
                n->name_start = t.start;
                n->name_len = t.length;
            }
            p->pos++;
            break;
        }
        case TokenType::LParen: {
            p->pos++;  // consume '('
            // parse_ternary (not parse_or): a parenthesized sub-expression
            // is the FULL grammar, e.g. gpt-oss's real template has
            // `(tool.content_type if tool.content_type is defined else
            // "json")` -- a ternary inside parens.
            n = parse_ternary(p);
            if (n != nullptr) {
                if (peek(p) != TokenType::RParen) {
                    p->failed = true;
                    n = nullptr;
                } else {
                    p->pos++;  // consume ')'
                }
            }
            break;
        }
        case TokenType::LBracket: {
            p->pos++;  // consume '['
            n = alloc_node(p);
            if (n == nullptr) break;
            n->kind = NodeKind::ListLit;
            if (!parse_arg_list(p, TokenType::RBracket, n->args, &n->arg_count, nullptr,
                               nullptr)) {
                n = nullptr;
                break;
            }
            break;
        }
        default:
            p->failed = true;
            break;
    }
    leave_expr(p);
    return n;
}

// Parses a comma-separated list of expressions into `out`/`out_count`
// (already having consumed the opening delimiter at the call site).
// Consumes up to and including the matching `close` delimiter (RParen for
// call/filter/test args, RBracket for a list literal). Each argument may
// optionally be `identifier '=' expr` (a keyword argument, e.g.
// `namespace(browser=false)`) -- recognized by a single '=' (not '==')
// immediately after a leading identifier, unambiguous since `==` tokenizes
// as one EqEq token. `name_starts`/`name_lens` may be nullptr (list
// literals have no keyword-arg concept); when non-null, name_lens[i] == 0
// means args[i] was positional.
bool parse_arg_list(Parser* p, TokenType close, Node** out, int32_t* out_count,
                    int32_t* name_starts, int32_t* name_lens) noexcept {
    int32_t count = 0;
    if (peek(p) == close) {
        p->pos++;
        *out_count = 0;
        return true;
    }
    for (;;) {
        if (count >= kMaxArgs) {
            p->failed = true;
            return false;
        }
        int32_t kw_start = 0, kw_len = 0;
        if (peek(p) == TokenType::Identifier &&
            p->tape->tokens[p->pos + 1].type == TokenType::Eq) {
            const Token& kw = peek_tok(p);
            kw_start = kw.start;
            kw_len = kw.length;
            p->pos += 2;  // consume identifier '='
        }
        Node* arg = parse_ternary(p);
        if (arg == nullptr) return false;
        out[count] = arg;
        if (name_starts != nullptr) name_starts[count] = kw_start;
        if (name_lens != nullptr) name_lens[count] = kw_len;
        count++;
        if (peek(p) == TokenType::Comma) {
            p->pos++;
            continue;
        }
        break;
    }
    if (peek(p) != close) {
        p->failed = true;
        return false;
    }
    p->pos++;
    *out_count = count;
    return true;
}

Node* parse_postfix(Parser* p) noexcept {
    if (!enter_expr(p)) return nullptr;
    Node* n = parse_primary(p);
    while (n != nullptr && !p->failed) {
        const TokenType t = peek(p);
        if (t == TokenType::Dot) {
            p->pos++;
            if (peek(p) != TokenType::Identifier) {
                p->failed = true;
                break;
            }
            const Token& field = peek_tok(p);
            Node* attr = alloc_node(p);
            if (attr == nullptr) break;
            attr->kind = NodeKind::Attr;
            attr->operand[0] = n;
            attr->name_start = field.start;
            attr->name_len = field.length;
            p->pos++;
            n = attr;
        } else if (t == TokenType::LBracket) {
            p->pos++;
            Node* lo = nullptr;
            Node* hi = nullptr;
            bool has_colon = false;
            if (peek(p) != TokenType::Colon) {
                lo = parse_or(p);
                if (lo == nullptr) { n = nullptr; break; }
            }
            if (peek(p) == TokenType::Colon) {
                has_colon = true;
                p->pos++;
                if (peek(p) != TokenType::RBracket) {
                    hi = parse_or(p);
                    if (hi == nullptr) { n = nullptr; break; }
                }
            }
            if (peek(p) != TokenType::RBracket) {
                p->failed = true;
                n = nullptr;
                break;
            }
            p->pos++;
            Node* idx = alloc_node(p);
            if (idx == nullptr) { n = nullptr; break; }
            if (has_colon) {
                idx->kind = NodeKind::Slice;
                idx->operand[0] = n;
                idx->operand[1] = lo;
                idx->operand[2] = hi;
                idx->has_lo = (lo != nullptr);
                idx->has_hi = (hi != nullptr);
            } else {
                idx->kind = NodeKind::Index;
                idx->operand[0] = n;
                idx->operand[1] = lo;
            }
            n = idx;
        } else if (t == TokenType::LParen) {
            p->pos++;
            Node* call = alloc_node(p);
            if (call == nullptr) { n = nullptr; break; }
            call->kind = NodeKind::Call;
            call->operand[0] = n;
            if (!parse_arg_list(p, TokenType::RParen, call->args, &call->arg_count,
                               call->arg_name_start, call->arg_name_len)) {
                n = nullptr;
                break;
            }
            n = call;
        } else if (t == TokenType::Pipe) {
            p->pos++;
            if (peek(p) != TokenType::Identifier) {
                p->failed = true;
                n = nullptr;
                break;
            }
            const Token& fname = peek_tok(p);
            p->pos++;
            Node* filt = alloc_node(p);
            if (filt == nullptr) { n = nullptr; break; }
            filt->kind = NodeKind::Filter;
            filt->operand[0] = n;
            filt->name_start = fname.start;
            filt->name_len = fname.length;
            if (peek(p) == TokenType::LParen) {
                p->pos++;
                if (!parse_arg_list(p, TokenType::RParen, filt->args, &filt->arg_count, nullptr,
                                   nullptr)) {
                    n = nullptr;
                    break;
                }
            }
            n = filt;
        } else if (t == TokenType::KwIs) {
            p->pos++;
            bool negated = false;
            if (peek(p) == TokenType::KwNot) {
                negated = true;
                p->pos++;
            }
            if (peek(p) != TokenType::Identifier) {
                p->failed = true;
                n = nullptr;
                break;
            }
            const Token& tname = peek_tok(p);
            p->pos++;
            Node* test = alloc_node(p);
            if (test == nullptr) { n = nullptr; break; }
            test->kind = NodeKind::Test;
            test->operand[0] = n;
            test->negated = negated;
            test->name_start = tname.start;
            test->name_len = tname.length;
            if (peek(p) == TokenType::LParen) {
                p->pos++;
                if (!parse_arg_list(p, TokenType::RParen, test->args, &test->arg_count, nullptr,
                                   nullptr)) {
                    n = nullptr;
                    break;
                }
            }
            n = test;
        } else {
            break;
        }
    }
    leave_expr(p);
    return p->failed ? nullptr : n;
}

Node* parse_unary(Parser* p) noexcept {
    if (!enter_expr(p)) return nullptr;
    Node* n;
    if (peek(p) == TokenType::Minus) {
        p->pos++;
        Node* operand = parse_unary(p);
        if (operand == nullptr) { n = nullptr; }
        else {
            n = alloc_node(p);
            if (n != nullptr) {
                n->kind = NodeKind::Unary;
                n->un_op = UnOp::Neg;
                n->operand[0] = operand;
            }
        }
    } else {
        n = parse_postfix(p);
    }
    leave_expr(p);
    return n;
}

Node* parse_mul(Parser* p) noexcept {
    if (!enter_expr(p)) return nullptr;
    Node* n = parse_unary(p);
    while (n != nullptr) {
        BinOp op;
        if (peek(p) == TokenType::Star) op = BinOp::Mul;
        else if (peek(p) == TokenType::Slash) op = BinOp::Div;
        else if (peek(p) == TokenType::Percent) op = BinOp::Mod;
        else break;
        p->pos++;
        Node* rhs = parse_unary(p);
        if (rhs == nullptr) { n = nullptr; break; }
        Node* bin = alloc_node(p);
        if (bin == nullptr) { n = nullptr; break; }
        bin->kind = NodeKind::Binary;
        bin->bin_op = op;
        bin->operand[0] = n;
        bin->operand[1] = rhs;
        n = bin;
    }
    leave_expr(p);
    return n;
}

Node* parse_additive(Parser* p) noexcept {
    if (!enter_expr(p)) return nullptr;
    Node* n = parse_mul(p);
    while (n != nullptr) {
        BinOp op;
        if (peek(p) == TokenType::Plus) op = BinOp::Add;
        else if (peek(p) == TokenType::Minus) op = BinOp::Sub;
        else break;
        p->pos++;
        Node* rhs = parse_mul(p);
        if (rhs == nullptr) { n = nullptr; break; }
        Node* bin = alloc_node(p);
        if (bin == nullptr) { n = nullptr; break; }
        bin->kind = NodeKind::Binary;
        bin->bin_op = op;
        bin->operand[0] = n;
        bin->operand[1] = rhs;
        n = bin;
    }
    leave_expr(p);
    return n;
}

Node* parse_concat(Parser* p) noexcept {
    if (!enter_expr(p)) return nullptr;
    Node* n = parse_additive(p);
    while (n != nullptr && peek(p) == TokenType::Tilde) {
        p->pos++;
        Node* rhs = parse_additive(p);
        if (rhs == nullptr) { n = nullptr; break; }
        Node* bin = alloc_node(p);
        if (bin == nullptr) { n = nullptr; break; }
        bin->kind = NodeKind::Binary;
        bin->bin_op = BinOp::Concat;
        bin->operand[0] = n;
        bin->operand[1] = rhs;
        n = bin;
    }
    leave_expr(p);
    return n;
}

Node* parse_compare(Parser* p) noexcept {
    if (!enter_expr(p)) return nullptr;
    Node* n = parse_concat(p);
    if (n == nullptr) { leave_expr(p); return nullptr; }

    BinOp op = BinOp::Eq;  // overwritten whenever matched == true
    bool matched = true;
    switch (peek(p)) {
        case TokenType::EqEq: op = BinOp::Eq; break;
        case TokenType::NotEq: op = BinOp::Ne; break;
        case TokenType::Lt: op = BinOp::Lt; break;
        case TokenType::Gt: op = BinOp::Gt; break;
        case TokenType::LtEq: op = BinOp::Le; break;
        case TokenType::GtEq: op = BinOp::Ge; break;
        case TokenType::KwIn: op = BinOp::In; break;
        case TokenType::KwNot: {
            // Only consume as `not in`; a bare `not` here is not part of
            // the compare grammar (it belongs to parse_not, one level up).
            if (p->tape->tokens[p->pos + 1].type == TokenType::KwIn) {
                op = BinOp::NotIn;
            } else {
                matched = false;
            }
            break;
        }
        default: matched = false; break;
    }
    if (!matched) { leave_expr(p); return n; }

    p->pos += (op == BinOp::NotIn) ? 2 : 1;
    Node* rhs = parse_concat(p);
    if (rhs == nullptr) { leave_expr(p); return nullptr; }
    Node* bin = alloc_node(p);
    if (bin == nullptr) { leave_expr(p); return nullptr; }
    bin->kind = NodeKind::Binary;
    bin->bin_op = op;
    bin->operand[0] = n;
    bin->operand[1] = rhs;
    leave_expr(p);
    return bin;
}

Node* parse_not(Parser* p) noexcept {
    if (!enter_expr(p)) return nullptr;
    Node* n;
    if (peek(p) == TokenType::KwNot) {
        p->pos++;
        Node* operand = parse_not(p);
        if (operand == nullptr) { n = nullptr; }
        else {
            n = alloc_node(p);
            if (n != nullptr) {
                n->kind = NodeKind::Unary;
                n->un_op = UnOp::Not;
                n->operand[0] = operand;
            }
        }
    } else {
        n = parse_compare(p);
    }
    leave_expr(p);
    return n;
}

Node* parse_and(Parser* p) noexcept {
    if (!enter_expr(p)) return nullptr;
    Node* n = parse_not(p);
    while (n != nullptr && peek(p) == TokenType::KwAnd) {
        p->pos++;
        Node* rhs = parse_not(p);
        if (rhs == nullptr) { n = nullptr; break; }
        Node* bin = alloc_node(p);
        if (bin == nullptr) { n = nullptr; break; }
        bin->kind = NodeKind::Binary;
        bin->bin_op = BinOp::And;
        bin->operand[0] = n;
        bin->operand[1] = rhs;
        n = bin;
    }
    leave_expr(p);
    return n;
}

Node* parse_or(Parser* p) noexcept {
    if (!enter_expr(p)) return nullptr;
    Node* n = parse_and(p);
    while (n != nullptr && peek(p) == TokenType::KwOr) {
        p->pos++;
        Node* rhs = parse_and(p);
        if (rhs == nullptr) { n = nullptr; break; }
        Node* bin = alloc_node(p);
        if (bin == nullptr) { n = nullptr; break; }
        bin->kind = NodeKind::Binary;
        bin->bin_op = BinOp::Or;
        bin->operand[0] = n;
        bin->operand[1] = rhs;
        n = bin;
    }
    leave_expr(p);
    return n;
}

// Jinja/Python's conditional expression, e.g.
// `args_value if args_value is string else args_value | string`.
// Lowest precedence of all -- wraps a full or_expr on each side, and the
// else-branch recurses into another ternary to allow right-associative
// chaining (`a if x else b if y else c`).
Node* parse_ternary(Parser* p) noexcept {
    if (!enter_expr(p)) return nullptr;
    Node* true_expr = parse_or(p);
    if (true_expr == nullptr) { leave_expr(p); return nullptr; }
    if (peek(p) != TokenType::KwIf) { leave_expr(p); return true_expr; }

    p->pos++;  // consume 'if'
    Node* cond = parse_or(p);
    if (cond == nullptr) { leave_expr(p); return nullptr; }
    if (peek(p) != TokenType::KwElse) { p->failed = true; leave_expr(p); return nullptr; }
    p->pos++;  // consume 'else'
    Node* false_expr = parse_ternary(p);
    if (false_expr == nullptr) { leave_expr(p); return nullptr; }

    Node* node = alloc_node(p);
    if (node == nullptr) { leave_expr(p); return nullptr; }
    node->kind = NodeKind::Ternary;
    node->operand[0] = true_expr;
    node->operand[1] = cond;
    node->operand[2] = false_expr;
    leave_expr(p);
    return node;
}

}  // namespace

Node* parse_expression(const TokenTape* tape, int32_t* pos,
                        const ParseLimits* limits_in, Arena* arena) noexcept {
    assert(tape != nullptr);
    assert(pos != nullptr);
    assert(arena != nullptr);

    const ParseLimits limits = limits_in ? *limits_in : ParseLimits{};
    if (limits.max_nodes <= 0 || limits.max_expr_depth <= 0) return nullptr;

    Parser p{};
    p.tape = tape;
    p.pos = *pos;
    p.limits = &limits;
    p.arena = arena;
    p.node_count = 0;
    p.depth = 0;
    p.failed = false;

    if (at_end_of_expr(&p)) return nullptr;  // nothing to parse

    Node* result = parse_ternary(&p);
    if (p.failed || result == nullptr) return nullptr;

    *pos = p.pos;
    return result;
}

}  // namespace jinja
}  // namespace parse
}  // namespace bolt
