#pragma once
// ---------------------------------------------------------------------------
// bolt::parse::jinja — runtime Value type + variable Scope, used by the
// evaluator (bolt_jinja_eval.h, BLLM-73). Values are NOT tied to the
// template source buffer (unlike the lexer/parser's zero-copy Node/Stmt
// spans) -- a Value can be the result of concatenation, a filter, or an
// externally-supplied variable (e.g. the `messages`/`tools` request
// payload), so string content is always arena-COPIED, not source-aliased.
//
// Conventions: arena-only allocation, noexcept, no std::*.
// ---------------------------------------------------------------------------

#include <cstdint>

#include "bolt/bolt_arena.h"

namespace bolt {
namespace parse {
namespace jinja {

struct Stmt;  // fwd (bolt_jinja_stmt.h) -- Macro values reference a MacroDef

enum class ValueKind : uint8_t {
    Undefined = 0,  // Jinja's `is defined` test distinguishes this from Null
    Null,
    Bool,
    Int,
    Float,
    String,
    List,
    Dict,
    Macro,
};

struct Value {
    ValueKind kind;
    bool     b;
    int64_t  i;
    double   f;
    const char* str;      // arena-owned buffer (String only)
    int32_t     str_len;
    Value**  items;        // arena-owned array (List only)
    int32_t  item_count;

    struct Entry {
        const char* key;
        int32_t     key_len;
        Value*      value;
    };
    Entry*   entries;      // arena-owned array (Dict only)
    int32_t  entry_count;

    const Stmt* macro_def;      // Macro only
    struct Scope* macro_closure; // Macro only: the globals scope captured at def time
};

Value* value_undefined(Arena* arena) noexcept;
Value* value_null(Arena* arena) noexcept;
Value* value_bool(Arena* arena, bool v) noexcept;
Value* value_int(Arena* arena, int64_t v) noexcept;
Value* value_float(Arena* arena, double v) noexcept;
Value* value_string(Arena* arena, const char* s, int32_t len) noexcept;
Value* value_list(Arena* arena, Value** items, int32_t count) noexcept;
Value* value_dict(Arena* arena, const Value::Entry* entries, int32_t count) noexcept;
Value* value_macro(Arena* arena, const Stmt* macro_def, struct Scope* closure) noexcept;

// Jinja truthiness: Undefined/Null/false/0/0.0/""/[]/{} are falsy.
bool value_truthy(const Value* v) noexcept;

// Looks up `key` in a Dict value's entries (linear scan). Returns nullptr
// (not Undefined) if `v` isn't a Dict or the key isn't present -- callers
// distinguish "not a dict" from "key missing" if they need to.
const Value* value_dict_get(const Value* v, const char* key, int32_t key_len) noexcept;

// Mutates an EXISTING entry's value in place (does not grow the entries
// array -- Dicts are allocated at exactly their construction-time size).
// This is the mechanism behind `{% set ns.attr = expr %}` on a namespace()
// object (bolt_jinja_eval.cpp's Set-statement handling and its "namespace"
// builtin): namespace() pre-declares every attribute as a keyword arg, and
// `set ns.attr = ...` only ever mutates an already-declared one. Returns
// false if `v` isn't a Dict or `key` isn't an existing entry.
bool value_dict_set(Value* v, const char* key, int32_t key_len, Value* new_value) noexcept;

// Renders `v`'s Jinja str()-equivalent into an arena-owned buffer.
// Null -> "None", Bool -> "True"/"False", Int -> decimal, Float -> %g,
// String -> itself, List/Dict -> a minimal bracket/brace rendering
// (NOT guaranteed to match Python repr exactly -- sufficient for the
// concatenation/output use case, not a general serializer).
bool value_to_string(Arena* arena, const Value* v, const char** out,
                     int32_t* out_len) noexcept;

// ---------------------------------------------------------------------------
// Scope: a fixed-capacity binding list + parent chain. Macro calls bind a
// FRESH scope whose parent is the macro's closure (its defining scope),
// NOT the caller's scope -- matching real Jinja2 semantics (macros don't
// see the caller's locals, only the global/module scope they were
// defined in plus their own bound parameters).
// ---------------------------------------------------------------------------

static constexpr int32_t kMaxScopeBindings = 32;

struct Binding {
    const char* name;
    int32_t     name_len;
    Value*      value;
};

struct Scope {
    Binding bindings[kMaxScopeBindings];
    int32_t binding_count;
    Scope*  parent;
};

void scope_init(Scope* s, Scope* parent) noexcept;

// Binds `name` in `s` (overwriting an existing binding with the same name
// if present). Returns false if `s` is already at kMaxScopeBindings
// capacity and `name` is new -- a hard cap, not a soft one.
bool scope_bind(Scope* s, const char* name, int32_t name_len,
                Value* value) noexcept;

// Looks up `name` in `s`, then its parent chain. Returns nullptr if not
// found anywhere in the chain (callers treat that as Undefined).
Value* scope_lookup(const Scope* s, const char* name, int32_t name_len) noexcept;

}  // namespace jinja
}  // namespace parse
}  // namespace bolt
