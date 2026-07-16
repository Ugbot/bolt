// bolt_jinja_eval.cpp — tree-walking evaluator for bolt::parse::jinja
// (BLLM-73). Tiger Style: no exceptions, no std::*, arena-only allocation,
// noexcept.

#include "bolt/parse/bolt_jinja_eval.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace bolt {
namespace parse {
namespace jinja {

namespace {

struct Env {
    const uint8_t*    source;
    const EvalLimits* limits;
    Arena*            arena;
    Scope*            globals;
    int32_t           loop_iterations_used;
    int32_t           macro_depth;
    int32_t           output_bytes_used;
    bool              failed;
};

const char* name_ptr(const Env* env, int32_t start) noexcept {
    return reinterpret_cast<const char*>(env->source) + start;
}

Value* eval_node(const Node* n, Scope* scope, Env* env) noexcept;
bool exec_block(const Stmt* head, Scope* scope, Env* env, Output* out) noexcept;

bool output_append(Env* env, Output* out, const char* s, int32_t len) noexcept {
    if (len <= 0) return true;
    if (env->output_bytes_used + len > env->limits->max_output_bytes) {
        env->failed = true;
        return false;
    }
    if (out->len + len > out->cap) {
        int32_t new_cap = out->cap > 0 ? out->cap * 2 : 256;
        while (new_cap < out->len + len) new_cap *= 2;
        char* nbuf = static_cast<char*>(env->arena->allocate(static_cast<size_t>(new_cap), 1));
        if (nbuf == nullptr) { env->failed = true; return false; }
        if (out->len > 0) memcpy(nbuf, out->data, static_cast<size_t>(out->len));
        out->data = nbuf;
        out->cap = new_cap;
    }
    memcpy(out->data + out->len, s, static_cast<size_t>(len));
    out->len += len;
    env->output_bytes_used += len;
    return true;
}

Value* decode_string_literal(const Node* n, Env* env) noexcept {
    const char* raw = name_ptr(env, n->name_start);
    const int32_t raw_len = n->name_len;
    char* buf = static_cast<char*>(env->arena->allocate(
        static_cast<size_t>(raw_len > 0 ? raw_len : 1), 1));
    if (buf == nullptr) { env->failed = true; return nullptr; }
    int32_t out_len = 0;
    for (int32_t i = 0; i < raw_len; ++i) {
        char c = raw[i];
        if (c == '\\' && i + 1 < raw_len) {
            char next = raw[i + 1];
            char decoded;
            switch (next) {
                case 'n': decoded = '\n'; break;
                case 't': decoded = '\t'; break;
                case 'r': decoded = '\r'; break;
                case '\\': decoded = '\\'; break;
                case '\'': decoded = '\''; break;
                case '"': decoded = '"'; break;
                default: decoded = next; break;
            }
            buf[out_len++] = decoded;
            i += 1;  // always consume the escaped char (on top of the for-loop's ++i)
        } else {
            buf[out_len++] = c;
        }
    }
    return value_string(env->arena, buf, out_len);
}

double value_as_double(const Value* v) noexcept {
    if (v == nullptr) return 0.0;
    switch (v->kind) {
        case ValueKind::Int: return static_cast<double>(v->i);
        case ValueKind::Float: return v->f;
        case ValueKind::Bool: return v->b ? 1.0 : 0.0;
        default: return 0.0;
    }
}

bool value_is_float_ish(const Value* v) noexcept {
    return v != nullptr && v->kind == ValueKind::Float;
}

// Shared by BinOp::Concat (~) and the string-concatenation half of
// BinOp::Add (+ between strings -- see that case's own comment).
Value* concat_values(Env* env, const Value* lhs, const Value* rhs) noexcept {
    const char *ls, *rs; int32_t ll, rl;
    if (!value_to_string(env->arena, lhs, &ls, &ll) ||
        !value_to_string(env->arena, rhs, &rs, &rl)) {
        env->failed = true;
        return nullptr;
    }
    char* buf = static_cast<char*>(
        env->arena->allocate(static_cast<size_t>(ll + rl > 0 ? ll + rl : 1), 1));
    if (buf == nullptr) { env->failed = true; return nullptr; }
    if (ll > 0) memcpy(buf, ls, static_cast<size_t>(ll));
    if (rl > 0) memcpy(buf + ll, rs, static_cast<size_t>(rl));
    return value_string(env->arena, buf, ll + rl);
}

bool values_equal(const Value* a, const Value* b) noexcept {
    if (a == nullptr || b == nullptr) return a == b;
    if ((a->kind == ValueKind::Int || a->kind == ValueKind::Float ||
         a->kind == ValueKind::Bool) &&
        (b->kind == ValueKind::Int || b->kind == ValueKind::Float ||
         b->kind == ValueKind::Bool)) {
        return value_as_double(a) == value_as_double(b);
    }
    if (a->kind != b->kind) return false;
    switch (a->kind) {
        case ValueKind::Undefined:
        case ValueKind::Null: return true;
        case ValueKind::String:
            return a->str_len == b->str_len &&
                   memcmp(a->str, b->str, static_cast<size_t>(a->str_len)) == 0;
        default: return a == b;  // list/dict/macro: identity
    }
}

// Filters and tests use the raw identifier span (not necessarily
// null-terminated) -- this compares against a C string literal by length.
bool name_is(const uint8_t* source, int32_t start, int32_t len,
             const char* lit) noexcept {
    const size_t lit_len = strlen(lit);
    return static_cast<size_t>(len) == lit_len &&
           memcmp(source + start, lit, lit_len) == 0;
}

// Dict.items() -- shared by the `|items` filter and the `.items()`
// method-call postfix form (real templates use both spellings; gpt-oss's
// uses `tool.parameters.properties.items()`, Qwen3-Coder's uses
// `tool.parameters.properties|items`).
Value* dict_items(Env* env, Value* base) noexcept {
    if (base->kind != ValueKind::Dict) return value_list(env->arena, nullptr, 0);
    Value** pairs = env->arena->allocate_array<Value*>(static_cast<size_t>(base->entry_count));
    if (pairs == nullptr) { env->failed = true; return nullptr; }
    for (int32_t i = 0; i < base->entry_count; ++i) {
        const Value::Entry& e = base->entries[i];
        Value* k = value_string(env->arena, e.key, e.key_len);
        Value* kv[2] = {k, e.value};
        pairs[i] = value_list(env->arena, kv, 2);
        if (pairs[i] == nullptr) { env->failed = true; return nullptr; }
    }
    return value_list(env->arena, pairs, base->entry_count);
}

Value* eval_filter(const Node* n, Env* env, Scope* scope) noexcept {
    Value* base = eval_node(n->operand[0], scope, env);
    if (base == nullptr) return nullptr;
    const uint8_t* src = env->source;
    const int32_t ns = n->name_start, nl = n->name_len;

    if (name_is(src, ns, nl, "length")) {
        int32_t n_items = 0;
        if (base->kind == ValueKind::List) n_items = base->item_count;
        else if (base->kind == ValueKind::Dict) n_items = base->entry_count;
        else if (base->kind == ValueKind::String) n_items = base->str_len;
        return value_int(env->arena, n_items);
    }
    if (name_is(src, ns, nl, "trim")) {
        const char* s; int32_t sl;
        if (!value_to_string(env->arena, base, &s, &sl)) { env->failed = true; return nullptr; }
        int32_t lo = 0, hi = sl;
        while (lo < hi && (s[lo] == ' ' || s[lo] == '\t' || s[lo] == '\n' || s[lo] == '\r')) lo++;
        while (hi > lo && (s[hi - 1] == ' ' || s[hi - 1] == '\t' || s[hi - 1] == '\n' || s[hi - 1] == '\r')) hi--;
        return value_string(env->arena, s + lo, hi - lo);
    }
    if (name_is(src, ns, nl, "string") || name_is(src, ns, nl, "safe") ||
        name_is(src, ns, nl, "tojson")) {
        const char* s; int32_t sl;
        if (!value_to_string(env->arena, base, &s, &sl)) { env->failed = true; return nullptr; }
        if (name_is(src, ns, nl, "tojson") && base->kind == ValueKind::String) {
            // Minimal JSON-string quoting -- not a general serializer.
            char* buf = static_cast<char*>(env->arena->allocate(
                static_cast<size_t>(sl * 2 + 2), 1));
            if (buf == nullptr) { env->failed = true; return nullptr; }
            int32_t o = 0;
            buf[o++] = '"';
            for (int32_t i = 0; i < sl; ++i) {
                if (s[i] == '"' || s[i] == '\\') buf[o++] = '\\';
                buf[o++] = s[i];
            }
            buf[o++] = '"';
            return value_string(env->arena, buf, o);
        }
        return value_string(env->arena, s, sl);
    }
    if (name_is(src, ns, nl, "default")) {
        if (base->kind == ValueKind::Undefined && n->arg_count > 0) {
            return eval_node(n->args[0], scope, env);
        }
        return base;
    }
    if (name_is(src, ns, nl, "join")) {
        const char* sep = ""; int32_t sep_len = 0;
        if (n->arg_count > 0) {
            Value* sep_v = eval_node(n->args[0], scope, env);
            if (sep_v == nullptr) return nullptr;
            if (!value_to_string(env->arena, sep_v, &sep, &sep_len)) { env->failed = true; return nullptr; }
        }
        char* buf = static_cast<char*>(env->arena->allocate(256, 1));
        if (buf == nullptr) { env->failed = true; return nullptr; }
        int32_t len = 0, cap = 256;
        auto grow_append = [&](const char* s, int32_t sl) -> bool {
            if (len + sl > cap) {
                int32_t nc = cap * 2;
                while (nc < len + sl) nc *= 2;
                char* nb = static_cast<char*>(env->arena->allocate(static_cast<size_t>(nc), 1));
                if (nb == nullptr) return false;
                memcpy(nb, buf, static_cast<size_t>(len));
                buf = nb; cap = nc;
            }
            memcpy(buf + len, s, static_cast<size_t>(sl));
            len += sl;
            return true;
        };
        const int32_t n_items = (base->kind == ValueKind::List) ? base->item_count : 0;
        for (int32_t i = 0; i < n_items; ++i) {
            if (i > 0 && !grow_append(sep, sep_len)) { env->failed = true; return nullptr; }
            const char* s; int32_t sl;
            if (!value_to_string(env->arena, base->items[i], &s, &sl)) { env->failed = true; return nullptr; }
            if (!grow_append(s, sl)) { env->failed = true; return nullptr; }
        }
        return value_string(env->arena, buf, len);
    }
    if (name_is(src, ns, nl, "items")) return dict_items(env, base);
    // Unrecognized filter: pass the base value through unchanged rather
    // than failing the whole render (BLLM-74 fills out the builtin set).
    return base;
}

Value* eval_test(const Node* n, Env* env, Scope* scope) noexcept {
    Value* base = eval_node(n->operand[0], scope, env);
    if (base == nullptr) return nullptr;
    const uint8_t* src = env->source;
    const int32_t ns = n->name_start, nl = n->name_len;
    bool result = false;

    if (name_is(src, ns, nl, "defined")) {
        result = base->kind != ValueKind::Undefined;
    } else if (name_is(src, ns, nl, "iterable")) {
        result = base->kind == ValueKind::List || base->kind == ValueKind::Dict ||
                 base->kind == ValueKind::String;
    } else if (name_is(src, ns, nl, "string")) {
        result = base->kind == ValueKind::String;
    } else if (name_is(src, ns, nl, "mapping")) {
        result = base->kind == ValueKind::Dict;
    } else if (name_is(src, ns, nl, "none")) {
        result = base->kind == ValueKind::Null;
    } else if (name_is(src, ns, nl, "number")) {
        result = base->kind == ValueKind::Int || base->kind == ValueKind::Float;
    }
    if (n->negated) result = !result;
    return value_bool(env->arena, result);
}

Value* call_macro(Value* macro, const Node* call, Env* env, Scope* caller_scope) noexcept {
    if (macro == nullptr || macro->kind != ValueKind::Macro) {
        return value_undefined(env->arena);
    }
    env->macro_depth++;
    if (env->macro_depth > env->limits->max_macro_depth) {
        env->failed = true;
        env->macro_depth--;
        return nullptr;
    }

    Scope* macro_scope = static_cast<Scope*>(
        env->arena->allocate(sizeof(Scope), alignof(Scope)));
    if (macro_scope == nullptr) { env->failed = true; env->macro_depth--; return nullptr; }
    scope_init(macro_scope, macro->macro_closure);

    const Stmt* def = macro->macro_def;
    for (int32_t i = 0; i < def->param_count; ++i) {
        Value* arg_val;
        if (i < call->arg_count) {
            arg_val = eval_node(call->args[i], caller_scope, env);
        } else if (def->params[i].default_value != nullptr) {
            arg_val = eval_node(def->params[i].default_value, macro_scope, env);
        } else {
            arg_val = value_undefined(env->arena);
        }
        if (arg_val == nullptr) { env->macro_depth--; return nullptr; }
        scope_bind(macro_scope, name_ptr(env, def->params[i].name_start),
                  def->params[i].name_len, arg_val);
    }

    Output body_out{};
    if (!exec_block(def->macro_body, macro_scope, env, &body_out)) {
        env->macro_depth--;
        return nullptr;
    }
    env->macro_depth--;
    return value_string(env->arena, body_out.data, body_out.len);
}

// namespace(key=val, ...) -- the standard Jinja2 builtin for a mutable
// object that persists across for-loop iterations (each iteration gets a
// fresh scope, so plain `{% set %}` inside a loop body doesn't survive to
// the next iteration or past the loop; a namespace object's ATTRIBUTES can
// be mutated in place instead, via `{% set ns.attr = expr %}` --
// value_dict_set, bolt_jinja_value.h). Every arg must be a keyword arg
// (gpt-oss's real template only ever calls it that way); a stray
// positional arg is ignored rather than failing the render.
Value* builtin_namespace(const Node* call, Env* env, Scope* scope) noexcept {
    Value::Entry entries[kMaxArgs];
    int32_t count = 0;
    for (int32_t i = 0; i < call->arg_count; ++i) {
        if (call->arg_name_len[i] <= 0) continue;  // positional -- ignored
        Value* v = eval_node(call->args[i], scope, env);
        if (v == nullptr) return nullptr;
        entries[count].key = name_ptr(env, call->arg_name_start[i]);
        entries[count].key_len = call->arg_name_len[i];
        entries[count].value = v;
        count++;
    }
    return value_dict(env->arena, entries, count);
}

// strftime_now(format) -- real wall-clock date/time, formatted per a
// strftime-style format string (e.g. "%Y-%m-%d"). Genuinely
// non-deterministic by design (it IS "what is today's date" for the
// system-prompt text) -- unlike the rest of this evaluator, which is a
// pure function of its Value inputs.
Value* builtin_strftime_now(const Node* call, Env* env, Scope* scope) noexcept {
    char fmt_buf[64] = "%Y-%m-%d";
    if (call->arg_count > 0) {
        Value* fmt_val = eval_node(call->args[0], scope, env);
        if (fmt_val == nullptr) return nullptr;
        if (fmt_val->kind == ValueKind::String && fmt_val->str_len < 63) {
            memcpy(fmt_buf, fmt_val->str, static_cast<size_t>(fmt_val->str_len));
            fmt_buf[fmt_val->str_len] = '\0';
        }
    }
    std::time_t now = std::time(nullptr);
    std::tm tm_now{};
#if defined(_WIN32)
    localtime_s(&tm_now, &now);
#else
    localtime_r(&now, &tm_now);
#endif
    char out_buf[128];
    const size_t n = std::strftime(out_buf, sizeof(out_buf), fmt_buf, &tm_now);
    return value_string(env->arena, out_buf, static_cast<int32_t>(n));
}

Value* eval_call(const Node* n, Scope* scope, Env* env) noexcept {
    const Node* callee = n->operand[0];
    if (callee->kind == NodeKind::Ident) {
        if (name_is(env->source, callee->name_start, callee->name_len, "namespace")) {
            return builtin_namespace(n, env, scope);
        }
        if (name_is(env->source, callee->name_start, callee->name_len, "strftime_now")) {
            return builtin_strftime_now(n, env, scope);
        }
        if (name_is(env->source, callee->name_start, callee->name_len, "raise_exception")) {
            // A real Jinja template's own validation-error path (e.g. "tool
            // call id mismatch"). We don't have a message channel back to
            // the caller beyond the render failing outright, but failing
            // the render (rather than silently continuing) is the correct
            // behavior -- EngineGenerator falls back to the hardcoded
            // template on any render failure (BLLM-76).
            env->failed = true;
            return nullptr;
        }
        Value* callee_val = scope_lookup(scope, name_ptr(env, callee->name_start),
                                        callee->name_len);
        return call_macro(callee_val, n, env, scope);
    }
    if (callee->kind == NodeKind::Attr) {
        // Method-call postfix form, e.g. `d.items()` (Qwen3-Coder's
        // template uses the `|items` filter instead; gpt-oss's uses this
        // dotted-call form for the same operation) -- only `.items()` is
        // needed by the two grounding templates today.
        if (name_is(env->source, callee->name_start, callee->name_len, "items")) {
            Value* base = eval_node(callee->operand[0], scope, env);
            if (base == nullptr) return nullptr;
            return dict_items(env, base);
        }
    }
    return value_undefined(env->arena);
}

Value* eval_node(const Node* n, Scope* scope, Env* env) noexcept {
    if (env->failed) return nullptr;
    switch (n->kind) {
        case NodeKind::NullLit: return value_null(env->arena);
        case NodeKind::BoolLit: return value_bool(env->arena, n->int_value != 0);
        case NodeKind::IntLit: return value_int(env->arena, n->int_value);
        case NodeKind::FloatLit: {
            char buf[64];
            int32_t len = n->name_len < 63 ? n->name_len : 63;
            memcpy(buf, name_ptr(env, n->name_start), static_cast<size_t>(len));
            buf[len] = '\0';
            return value_float(env->arena, atof(buf));
        }
        case NodeKind::StringLit: return decode_string_literal(n, env);
        case NodeKind::Ident: {
            Value* v = scope_lookup(scope, name_ptr(env, n->name_start), n->name_len);
            return v != nullptr ? v : value_undefined(env->arena);
        }
        case NodeKind::Attr: {
            Value* base = eval_node(n->operand[0], scope, env);
            if (base == nullptr) return nullptr;
            const Value* v = value_dict_get(base, name_ptr(env, n->name_start), n->name_len);
            return v != nullptr ? const_cast<Value*>(v) : value_undefined(env->arena);
        }
        case NodeKind::Index: {
            Value* base = eval_node(n->operand[0], scope, env);
            Value* idx = eval_node(n->operand[1], scope, env);
            if (base == nullptr || idx == nullptr) return nullptr;
            if (base->kind == ValueKind::List && idx->kind == ValueKind::Int) {
                int64_t i = idx->i;
                if (i < 0) i += base->item_count;
                if (i >= 0 && i < base->item_count) return base->items[i];
            } else if (base->kind == ValueKind::Dict && idx->kind == ValueKind::String) {
                const Value* v = value_dict_get(base, idx->str, idx->str_len);
                if (v != nullptr) return const_cast<Value*>(v);
            }
            return value_undefined(env->arena);
        }
        case NodeKind::Slice: {
            Value* base = eval_node(n->operand[0], scope, env);
            if (base == nullptr) return nullptr;
            if (base->kind != ValueKind::List) return value_undefined(env->arena);
            int64_t lo = 0, hi = base->item_count;
            if (n->has_lo) {
                Value* lo_v = eval_node(n->operand[1], scope, env);
                if (lo_v == nullptr) return nullptr;
                lo = lo_v->i;
                if (lo < 0) lo += base->item_count;
            }
            if (n->has_hi) {
                Value* hi_v = eval_node(n->operand[2], scope, env);
                if (hi_v == nullptr) return nullptr;
                hi = hi_v->i;
                if (hi < 0) hi += base->item_count;
            }
            if (lo < 0) lo = 0;
            if (hi > base->item_count) hi = base->item_count;
            if (lo >= hi) return value_list(env->arena, nullptr, 0);
            return value_list(env->arena, base->items + lo, static_cast<int32_t>(hi - lo));
        }
        case NodeKind::Call: return eval_call(n, scope, env);
        case NodeKind::Filter: return eval_filter(n, env, scope);
        case NodeKind::Test: return eval_test(n, env, scope);
        case NodeKind::ListLit: {
            Value* items[kMaxArgs];
            for (int32_t i = 0; i < n->arg_count; ++i) {
                items[i] = eval_node(n->args[i], scope, env);
                if (items[i] == nullptr) return nullptr;
            }
            return value_list(env->arena, items, n->arg_count);
        }
        case NodeKind::Ternary: {
            Value* cond = eval_node(n->operand[1], scope, env);
            if (cond == nullptr) return nullptr;
            return eval_node(value_truthy(cond) ? n->operand[0] : n->operand[2], scope, env);
        }
        case NodeKind::Unary: {
            Value* operand = eval_node(n->operand[0], scope, env);
            if (operand == nullptr) return nullptr;
            if (n->un_op == UnOp::Not) return value_bool(env->arena, !value_truthy(operand));
            // Neg
            if (value_is_float_ish(operand)) return value_float(env->arena, -operand->f);
            return value_int(env->arena, -operand->i);
        }
        case NodeKind::Binary: {
            Value* lhs = eval_node(n->operand[0], scope, env);
            if (lhs == nullptr) return nullptr;
            if (n->bin_op == BinOp::Or) {
                if (value_truthy(lhs)) return lhs;
                return eval_node(n->operand[1], scope, env);
            }
            if (n->bin_op == BinOp::And) {
                if (!value_truthy(lhs)) return lhs;
                return eval_node(n->operand[1], scope, env);
            }
            Value* rhs = eval_node(n->operand[1], scope, env);
            if (rhs == nullptr) return nullptr;
            switch (n->bin_op) {
                case BinOp::Eq: return value_bool(env->arena, values_equal(lhs, rhs));
                case BinOp::Ne: return value_bool(env->arena, !values_equal(lhs, rhs));
                case BinOp::Lt: return value_bool(env->arena, value_as_double(lhs) < value_as_double(rhs));
                case BinOp::Gt: return value_bool(env->arena, value_as_double(lhs) > value_as_double(rhs));
                case BinOp::Le: return value_bool(env->arena, value_as_double(lhs) <= value_as_double(rhs));
                case BinOp::Ge: return value_bool(env->arena, value_as_double(lhs) >= value_as_double(rhs));
                case BinOp::In:
                case BinOp::NotIn: {
                    bool found = false;
                    if (rhs->kind == ValueKind::List) {
                        for (int32_t i = 0; i < rhs->item_count && !found; ++i) {
                            found = values_equal(lhs, rhs->items[i]);
                        }
                    } else if (rhs->kind == ValueKind::Dict && lhs->kind == ValueKind::String) {
                        found = value_dict_get(rhs, lhs->str, lhs->str_len) != nullptr;
                    }
                    return value_bool(env->arena, n->bin_op == BinOp::In ? found : !found);
                }
                case BinOp::Concat: return concat_values(env, lhs, rhs);
                case BinOp::Add:
                    // Real Jinja2/Python semantics: `+` between strings (or
                    // a list and a list) concatenates, exactly like `~` --
                    // it is NOT numeric-only. This was a real bug (BLLM-77
                    // caught it against the actual Qwen3-Coder template,
                    // which uses `message.role + '\n' + message.content +
                    // ...` for concatenation): two String Values both have
                    // a zeroed `.i` field, so treating `+` as numeric-only
                    // silently produced Int(0) instead of the concatenation.
                    if (lhs->kind == ValueKind::String || rhs->kind == ValueKind::String) {
                        return concat_values(env, lhs, rhs);
                    }
                    if (value_is_float_ish(lhs) || value_is_float_ish(rhs)) {
                        return value_float(env->arena, value_as_double(lhs) + value_as_double(rhs));
                    }
                    return value_int(env->arena, lhs->i + rhs->i);
                case BinOp::Sub:
                    if (value_is_float_ish(lhs) || value_is_float_ish(rhs)) {
                        return value_float(env->arena, value_as_double(lhs) - value_as_double(rhs));
                    }
                    return value_int(env->arena, lhs->i - rhs->i);
                case BinOp::Mul:
                    if (value_is_float_ish(lhs) || value_is_float_ish(rhs)) {
                        return value_float(env->arena, value_as_double(lhs) * value_as_double(rhs));
                    }
                    return value_int(env->arena, lhs->i * rhs->i);
                case BinOp::Div:
                    return value_float(env->arena, value_as_double(lhs) / value_as_double(rhs));
                case BinOp::Mod:
                    return value_int(env->arena, rhs->i != 0 ? lhs->i % rhs->i : 0);
                default: return value_undefined(env->arena);
            }
        }
    }
    return value_undefined(env->arena);
}

bool exec_stmt(const Stmt* s, Scope* scope, Env* env, Output* out) noexcept {
    if (env->failed) return false;
    switch (s->kind) {
        case StmtKind::Text:
            return output_append(env, out, name_ptr(env, s->text_start), s->text_len);
        case StmtKind::Expr: {
            Value* v = eval_node(s->expr, scope, env);
            if (v == nullptr) return false;
            const char* str; int32_t str_len;
            if (!value_to_string(env->arena, v, &str, &str_len)) { env->failed = true; return false; }
            return output_append(env, out, str, str_len);
        }
        case StmtKind::If: {
            Value* cond = eval_node(s->cond, scope, env);
            if (cond == nullptr) return false;
            if (value_truthy(cond)) return exec_block(s->then_body, scope, env, out);
            if (s->else_body != nullptr) return exec_block(s->else_body, scope, env, out);
            return true;
        }
        case StmtKind::For: {
            Value* iterable = eval_node(s->iterable, scope, env);
            if (iterable == nullptr) return false;
            const bool two_var = s->var2_len > 0;
            const int32_t n_items = (iterable->kind == ValueKind::List) ? iterable->item_count : 0;
            for (int32_t idx = 0; idx < n_items; ++idx) {
                env->loop_iterations_used++;
                if (env->loop_iterations_used > env->limits->max_loop_iterations) {
                    env->failed = true;
                    return false;
                }
                Scope loop_scope;
                scope_init(&loop_scope, scope);

                Value* elem = iterable->items[idx];
                if (two_var) {
                    if (elem->kind == ValueKind::List && elem->item_count == 2) {
                        scope_bind(&loop_scope, name_ptr(env, s->var1_start), s->var1_len, elem->items[0]);
                        scope_bind(&loop_scope, name_ptr(env, s->var2_start), s->var2_len, elem->items[1]);
                    } else {
                        scope_bind(&loop_scope, name_ptr(env, s->var1_start), s->var1_len, value_undefined(env->arena));
                        scope_bind(&loop_scope, name_ptr(env, s->var2_start), s->var2_len, value_undefined(env->arena));
                    }
                } else {
                    scope_bind(&loop_scope, name_ptr(env, s->var1_start), s->var1_len, elem);
                }

                Value* previtem = idx > 0 ? iterable->items[idx - 1] : value_undefined(env->arena);
                Value* nextitem =
                    idx < n_items - 1 ? iterable->items[idx + 1] : value_undefined(env->arena);
                // sizeof(lit)-1 (not a manual count) for the compile-time
                // string-literal lengths below -- avoids the off-by-one
                // miscounts a hand-counted length is prone to.
                Value::Entry loop_entries[6] = {
                    {"index", static_cast<int32_t>(sizeof("index") - 1), value_int(env->arena, idx + 1)},
                    {"index0", static_cast<int32_t>(sizeof("index0") - 1), value_int(env->arena, idx)},
                    {"first", static_cast<int32_t>(sizeof("first") - 1), value_bool(env->arena, idx == 0)},
                    {"last", static_cast<int32_t>(sizeof("last") - 1), value_bool(env->arena, idx == n_items - 1)},
                    {"previtem", static_cast<int32_t>(sizeof("previtem") - 1), previtem},
                    {"nextitem", static_cast<int32_t>(sizeof("nextitem") - 1), nextitem},
                };
                Value* loop_val = value_dict(env->arena, loop_entries, 6);
                if (loop_val == nullptr) { env->failed = true; return false; }
                scope_bind(&loop_scope, "loop", static_cast<int32_t>(sizeof("loop") - 1), loop_val);

                if (!exec_block(s->body, &loop_scope, env, out)) return false;
            }
            return true;
        }
        case StmtKind::Set: {
            Value* v = eval_node(s->set_value, scope, env);
            if (v == nullptr) return false;
            if (s->set_attr_len > 0) {
                // `set ns.attr = expr` -- ns must already be a namespace()
                // (Dict) object; mutate the existing entry in place rather
                // than rebinding `ns` itself (bolt_jinja_value.h's
                // value_dict_set doc comment explains why this doesn't
                // grow the entries array).
                Value* target = scope_lookup(scope, name_ptr(env, s->set_name_start),
                                             s->set_name_len);
                value_dict_set(target, name_ptr(env, s->set_attr_start), s->set_attr_len, v);
                return true;
            }
            scope_bind(scope, name_ptr(env, s->set_name_start), s->set_name_len, v);
            return true;
        }
        case StmtKind::MacroDef: {
            Value* m = value_macro(env->arena, s, scope);
            if (m == nullptr) { env->failed = true; return false; }
            scope_bind(scope, name_ptr(env, s->macro_name_start), s->macro_name_len, m);
            return true;
        }
    }
    return true;
}

bool exec_block(const Stmt* head, Scope* scope, Env* env, Output* out) noexcept {
    for (const Stmt* s = head; s != nullptr; s = s->next) {
        if (!exec_stmt(s, scope, env, out)) return false;
    }
    return true;
}

}  // namespace

Value* eval_expression(const Node* expr, Scope* scope, Scope* globals,
                       const uint8_t* source, const EvalLimits* limits_in,
                       Arena* arena, int32_t* loop_iterations_used,
                       int32_t* macro_depth, int32_t* output_bytes_used,
                       bool* failed) noexcept {
    assert(expr != nullptr);
    assert(scope != nullptr);
    assert(arena != nullptr);

    const EvalLimits limits = limits_in ? *limits_in : EvalLimits{};
    Env env{};
    env.source = source;
    env.limits = &limits;
    env.arena = arena;
    env.globals = globals;
    env.loop_iterations_used = loop_iterations_used ? *loop_iterations_used : 0;
    env.macro_depth = macro_depth ? *macro_depth : 0;
    env.output_bytes_used = output_bytes_used ? *output_bytes_used : 0;
    env.failed = false;

    Value* result = eval_node(expr, scope, &env);

    if (loop_iterations_used) *loop_iterations_used = env.loop_iterations_used;
    if (macro_depth) *macro_depth = env.macro_depth;
    if (output_bytes_used) *output_bytes_used = env.output_bytes_used;
    if (failed) *failed = env.failed;
    return env.failed ? nullptr : result;
}

bool render_document(const Stmt* doc, Scope* scope, const uint8_t* source,
                     const EvalLimits* limits_in, Arena* arena,
                     Output* out) noexcept {
    assert(scope != nullptr);
    assert(arena != nullptr);
    assert(out != nullptr);

    const EvalLimits limits = limits_in ? *limits_in : EvalLimits{};
    Env env{};
    env.source = source;
    env.limits = &limits;
    env.arena = arena;
    env.globals = scope;
    env.loop_iterations_used = 0;
    env.macro_depth = 0;
    env.output_bytes_used = 0;
    env.failed = false;

    if (doc == nullptr) return true;  // empty document -> empty output
    bool ok = exec_block(doc, scope, &env, out);
    return ok && !env.failed;
}

}  // namespace jinja
}  // namespace parse
}  // namespace bolt
