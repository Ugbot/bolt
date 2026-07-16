// bolt_jinja_value.cpp — Value/Scope runtime for bolt::parse::jinja
// (BLLM-73). Tiger Style: no exceptions, no std::*, arena-only allocation.

#include "bolt/parse/bolt_jinja_value.h"

#include <cassert>
#include <cstdio>
#include <cstring>

namespace bolt {
namespace parse {
namespace jinja {

namespace {

Value* alloc_value(Arena* arena) noexcept {
    Value* v = static_cast<Value*>(arena->allocate(sizeof(Value), alignof(Value)));
    if (v == nullptr) return nullptr;
    memset(v, 0, sizeof(Value));
    return v;
}

char* copy_bytes(Arena* arena, const char* s, int32_t len) noexcept {
    if (len <= 0) return static_cast<char*>(arena->allocate(1, 1));
    char* buf = static_cast<char*>(arena->allocate(static_cast<size_t>(len), 1));
    if (buf != nullptr) memcpy(buf, s, static_cast<size_t>(len));
    return buf;
}

bool append_str(Arena* arena, char** buf, int32_t* len, int32_t* cap,
                const char* s, int32_t s_len) noexcept {
    if (*len + s_len > *cap) {
        int32_t new_cap = *cap * 2;
        while (new_cap < *len + s_len) new_cap *= 2;
        char* nbuf = static_cast<char*>(arena->allocate(static_cast<size_t>(new_cap), 1));
        if (nbuf == nullptr) return false;
        memcpy(nbuf, *buf, static_cast<size_t>(*len));
        *buf = nbuf;
        *cap = new_cap;
    }
    memcpy(*buf + *len, s, static_cast<size_t>(s_len));
    *len += s_len;
    return true;
}

}  // namespace

Value* value_undefined(Arena* arena) noexcept {
    Value* v = alloc_value(arena);
    if (v != nullptr) v->kind = ValueKind::Undefined;
    return v;
}

Value* value_null(Arena* arena) noexcept {
    Value* v = alloc_value(arena);
    if (v != nullptr) v->kind = ValueKind::Null;
    return v;
}

Value* value_bool(Arena* arena, bool b) noexcept {
    Value* v = alloc_value(arena);
    if (v != nullptr) { v->kind = ValueKind::Bool; v->b = b; }
    return v;
}

Value* value_int(Arena* arena, int64_t i) noexcept {
    Value* v = alloc_value(arena);
    if (v != nullptr) { v->kind = ValueKind::Int; v->i = i; }
    return v;
}

Value* value_float(Arena* arena, double f) noexcept {
    Value* v = alloc_value(arena);
    if (v != nullptr) { v->kind = ValueKind::Float; v->f = f; }
    return v;
}

Value* value_string(Arena* arena, const char* s, int32_t len) noexcept {
    Value* v = alloc_value(arena);
    if (v == nullptr) return nullptr;
    v->kind = ValueKind::String;
    v->str = copy_bytes(arena, s, len);
    v->str_len = len;
    if (v->str == nullptr) return nullptr;
    return v;
}

Value* value_list(Arena* arena, Value** items, int32_t count) noexcept {
    Value* v = alloc_value(arena);
    if (v == nullptr) return nullptr;
    v->kind = ValueKind::List;
    v->item_count = count;
    if (count > 0) {
        v->items = arena->allocate_array<Value*>(static_cast<size_t>(count));
        if (v->items == nullptr) return nullptr;
        memcpy(v->items, items, sizeof(Value*) * static_cast<size_t>(count));
    }
    return v;
}

Value* value_dict(Arena* arena, const Value::Entry* entries, int32_t count) noexcept {
    Value* v = alloc_value(arena);
    if (v == nullptr) return nullptr;
    v->kind = ValueKind::Dict;
    v->entry_count = count;
    if (count > 0) {
        v->entries = arena->allocate_array<Value::Entry>(static_cast<size_t>(count));
        if (v->entries == nullptr) return nullptr;
        memcpy(v->entries, entries,
               sizeof(Value::Entry) * static_cast<size_t>(count));
    }
    return v;
}

Value* value_macro(Arena* arena, const Stmt* macro_def, Scope* closure) noexcept {
    Value* v = alloc_value(arena);
    if (v == nullptr) return nullptr;
    v->kind = ValueKind::Macro;
    v->macro_def = macro_def;
    v->macro_closure = closure;
    return v;
}

bool value_truthy(const Value* v) noexcept {
    if (v == nullptr) return false;
    switch (v->kind) {
        case ValueKind::Undefined: return false;
        case ValueKind::Null: return false;
        case ValueKind::Bool: return v->b;
        case ValueKind::Int: return v->i != 0;
        case ValueKind::Float: return v->f != 0.0;
        case ValueKind::String: return v->str_len > 0;
        case ValueKind::List: return v->item_count > 0;
        case ValueKind::Dict: return v->entry_count > 0;
        case ValueKind::Macro: return true;
    }
    return false;
}

const Value* value_dict_get(const Value* v, const char* key,
                            int32_t key_len) noexcept {
    if (v == nullptr || v->kind != ValueKind::Dict) return nullptr;
    for (int32_t i = 0; i < v->entry_count; ++i) {
        const Value::Entry& e = v->entries[i];
        if (e.key_len == key_len && memcmp(e.key, key, static_cast<size_t>(key_len)) == 0) {
            return e.value;
        }
    }
    return nullptr;
}

bool value_dict_set(Value* v, const char* key, int32_t key_len, Value* new_value) noexcept {
    if (v == nullptr || v->kind != ValueKind::Dict) return false;
    for (int32_t i = 0; i < v->entry_count; ++i) {
        Value::Entry& e = v->entries[i];
        if (e.key_len == key_len && memcmp(e.key, key, static_cast<size_t>(key_len)) == 0) {
            e.value = new_value;
            return true;
        }
    }
    return false;
}

bool value_to_string(Arena* arena, const Value* v, const char** out,
                     int32_t* out_len) noexcept {
    assert(arena != nullptr);
    assert(v != nullptr);
    assert(out != nullptr);
    assert(out_len != nullptr);

    char stack_buf[64];
    switch (v->kind) {
        case ValueKind::Undefined:
        case ValueKind::Null: {
            const char* s = "None";
            *out = copy_bytes(arena, s, 4);
            *out_len = 4;
            return *out != nullptr;
        }
        case ValueKind::Bool: {
            const char* s = v->b ? "True" : "False";
            const int32_t n = v->b ? 4 : 5;
            *out = copy_bytes(arena, s, n);
            *out_len = n;
            return *out != nullptr;
        }
        case ValueKind::Int: {
            int n = snprintf(stack_buf, sizeof(stack_buf), "%lld",
                              static_cast<long long>(v->i));
            *out = copy_bytes(arena, stack_buf, n);
            *out_len = n;
            return *out != nullptr;
        }
        case ValueKind::Float: {
            int n = snprintf(stack_buf, sizeof(stack_buf), "%g", v->f);
            *out = copy_bytes(arena, stack_buf, n);
            *out_len = n;
            return *out != nullptr;
        }
        case ValueKind::String: {
            *out = v->str;
            *out_len = v->str_len;
            return true;
        }
        case ValueKind::List: {
            char* buf = static_cast<char*>(arena->allocate(64, 1));
            if (buf == nullptr) return false;
            int32_t len = 0, cap = 64;
            if (!append_str(arena, &buf, &len, &cap, "[", 1)) return false;
            for (int32_t i = 0; i < v->item_count; ++i) {
                if (i > 0 && !append_str(arena, &buf, &len, &cap, ", ", 2)) return false;
                const char* s; int32_t sl;
                if (!value_to_string(arena, v->items[i], &s, &sl)) return false;
                if (!append_str(arena, &buf, &len, &cap, s, sl)) return false;
            }
            if (!append_str(arena, &buf, &len, &cap, "]", 1)) return false;
            *out = buf;
            *out_len = len;
            return true;
        }
        case ValueKind::Dict: {
            char* buf = static_cast<char*>(arena->allocate(64, 1));
            if (buf == nullptr) return false;
            int32_t len = 0, cap = 64;
            if (!append_str(arena, &buf, &len, &cap, "{", 1)) return false;
            for (int32_t i = 0; i < v->entry_count; ++i) {
                if (i > 0 && !append_str(arena, &buf, &len, &cap, ", ", 2)) return false;
                const Value::Entry& e = v->entries[i];
                if (!append_str(arena, &buf, &len, &cap, e.key, e.key_len)) return false;
                if (!append_str(arena, &buf, &len, &cap, ": ", 2)) return false;
                const char* s; int32_t sl;
                if (!value_to_string(arena, e.value, &s, &sl)) return false;
                if (!append_str(arena, &buf, &len, &cap, s, sl)) return false;
            }
            if (!append_str(arena, &buf, &len, &cap, "}", 1)) return false;
            *out = buf;
            *out_len = len;
            return true;
        }
        case ValueKind::Macro: {
            const char* s = "<macro>";
            *out = copy_bytes(arena, s, 7);
            *out_len = 7;
            return *out != nullptr;
        }
    }
    return false;
}

void scope_init(Scope* s, Scope* parent) noexcept {
    assert(s != nullptr);
    s->binding_count = 0;
    s->parent = parent;
}

bool scope_bind(Scope* s, const char* name, int32_t name_len,
                Value* value) noexcept {
    assert(s != nullptr);
    for (int32_t i = 0; i < s->binding_count; ++i) {
        if (s->bindings[i].name_len == name_len &&
            memcmp(s->bindings[i].name, name, static_cast<size_t>(name_len)) == 0) {
            s->bindings[i].value = value;
            return true;
        }
    }
    if (s->binding_count >= kMaxScopeBindings) return false;
    Binding& b = s->bindings[s->binding_count++];
    b.name = name;
    b.name_len = name_len;
    b.value = value;
    return true;
}

Value* scope_lookup(const Scope* s, const char* name, int32_t name_len) noexcept {
    for (const Scope* cur = s; cur != nullptr; cur = cur->parent) {
        for (int32_t i = 0; i < cur->binding_count; ++i) {
            if (cur->bindings[i].name_len == name_len &&
                memcmp(cur->bindings[i].name, name, static_cast<size_t>(name_len)) == 0) {
                return cur->bindings[i].value;
            }
        }
    }
    return nullptr;
}

}  // namespace jinja
}  // namespace parse
}  // namespace bolt
