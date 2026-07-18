// ---------------------------------------------------------------------------
// bolt::parse::json — byte-faithful JSON writer / value builder impl.
//
// The write side of bolt::parse::json (see bolt_json_writer.h). Mirrors the
// reader's style: arena-only, noexcept, bounded, compile-time-dispatched
// sinks so one recursive walk serves both the length-count pass and the
// byte-write pass with zero runtime branching between them.
// ---------------------------------------------------------------------------
#include "bolt/parse/bolt_json_writer.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace bolt {
namespace parse {
namespace json {

namespace {

// ---------------------------------------------------------------------------
// Node allocation helpers.
// ---------------------------------------------------------------------------
JsonValue* alloc_value(Arena* arena, ValueKind kind) noexcept {
    assert(arena != nullptr);
    JsonValue* v = arena->allocate_array<JsonValue>(1);
    if (v == nullptr) return nullptr;
    v->kind  = kind;
    v->_pad[0] = 0; v->_pad[1] = 0; v->_pad[2] = 0; v->_pad[3] = 0;
    v->_pad[4] = 0; v->_pad[5] = 0; v->_pad[6] = 0;
    v->bytes = nullptr;
    v->len   = 0;
    v->count = 0;
    v->head  = nullptr;
    v->tail  = nullptr;
    return v;
}

// Append a pre-built node (key may be nullptr for arrays). The key bytes are
// referenced as-is — callers that need a copy do it before calling.
bool container_append(Arena* arena, JsonValue* container,
                      const char* key, int32_t key_len,
                      JsonValue* val) noexcept {
    assert(arena != nullptr);
    assert(container != nullptr);
    if (val == nullptr) return false;
    JsonNode* node = arena->allocate_array<JsonNode>(1);
    if (node == nullptr) return false;
    node->next    = nullptr;
    node->key     = key;
    node->key_len = key_len;
    node->_pad    = 0;
    node->value   = val;
    if (container->tail == nullptr) {
        container->head = node;
        container->tail = node;
    } else {
        container->tail->next = node;
        container->tail = node;
    }
    ++container->count;
    return true;
}

// Signed 64-bit -> decimal ASCII. Returns the byte count [1..20]. Handles
// INT64_MIN via unsigned magnitude. Writes into `out` (>= 20 bytes).
int32_t format_int64(char* out, int64_t v) noexcept {
    assert(out != nullptr);
    char tmp[20];
    int32_t n = 0;
    uint64_t mag = (v < 0) ? (~static_cast<uint64_t>(v) + 1u)
                           : static_cast<uint64_t>(v);
    do {
        tmp[n++] = static_cast<char>('0' + static_cast<int>(mag % 10u));
        mag /= 10u;
    } while (mag != 0u);
    int32_t w = 0;
    if (v < 0) out[w++] = '-';
    for (int32_t i = n - 1; i >= 0; --i) out[w++] = tmp[i];
    assert(w >= 1 && w <= 20);
    return w;
}

// ---------------------------------------------------------------------------
// Emit sinks. Compile-time dispatch: CountSink measures the exact output
// length, WriteSink fills the buffer. Both drive the SAME emit routines, so
// pass-1 length and pass-2 write can never disagree.
// ---------------------------------------------------------------------------
struct CountSink {
    int64_t n;
    BOLT_FORCE_INLINE void put(char) noexcept { ++n; }
    BOLT_FORCE_INLINE void put_bytes(const char*, int32_t k) noexcept { n += k; }
};

struct WriteSink {
    char* p;
    char* end;
    bool  ok;
    BOLT_FORCE_INLINE void put(char c) noexcept {
        if (p < end) { *p++ = c; } else { ok = false; }
    }
    BOLT_FORCE_INLINE void put_bytes(const char* b, int32_t k) noexcept {
        assert(k >= 0);
        if (k > 0 && p + k <= end) { memcpy(p, b, static_cast<size_t>(k)); p += k; }
        else if (k > 0) { ok = false; }
    }
};

// JSON string-escape one byte run into the sink. Both passes call this, so the
// counted and written lengths are identical by construction. Bytes >= 0x80 are
// passed through verbatim (raw UTF-8 is legal in JSON string content).
template <typename Sink>
BOLT_FORCE_INLINE void emit_escaped(Sink& s, const char* b, int32_t len) noexcept {
    assert(b != nullptr || len == 0);
    static const char kHex[] = "0123456789abcdef";
    for (int32_t i = 0; i < len; ++i) {
        const uint8_t c = static_cast<uint8_t>(b[i]);
        switch (c) {
            case '"':  s.put('\\'); s.put('"');  break;
            case '\\': s.put('\\'); s.put('\\'); break;
            case '\b': s.put('\\'); s.put('b');  break;
            case '\f': s.put('\\'); s.put('f');  break;
            case '\n': s.put('\\'); s.put('n');  break;
            case '\r': s.put('\\'); s.put('r');  break;
            case '\t': s.put('\\'); s.put('t');  break;
            default:
                if (c < 0x20) {
                    s.put('\\'); s.put('u'); s.put('0'); s.put('0');
                    s.put(kHex[(c >> 4) & 0xF]);
                    s.put(kHex[c & 0xF]);
                } else {
                    s.put(static_cast<char>(c));
                }
                break;
        }
    }
}

// Recursive value emit. `depth` is bounded by kJsonMaxDepth so a pathological
// (or maliciously deep) tree cannot blow the C stack.
template <typename Sink>
bool emit_value(const JsonValue* v, Sink& s, int32_t depth) noexcept {
    assert(v != nullptr);
    if (depth > kJsonMaxDepth) return false;
    switch (v->kind) {
        case ValueKind::Literal:
            s.put_bytes(v->bytes, v->len);
            return true;
        case ValueKind::StringVerbatim:
            s.put('"');
            s.put_bytes(v->bytes, v->len);
            s.put('"');
            return true;
        case ValueKind::StringEscaped:
            s.put('"');
            emit_escaped(s, v->bytes, v->len);
            s.put('"');
            return true;
        case ValueKind::Object: {
            s.put('{');
            bool first = true;
            for (const JsonNode* m = v->head; m != nullptr; m = m->next) {
                if (!first) s.put(',');
                first = false;
                s.put('"');
                s.put_bytes(m->key, m->key_len);
                s.put('"');
                s.put(':');
                if (!emit_value(m->value, s, depth + 1)) return false;
            }
            s.put('}');
            return true;
        }
        case ValueKind::Array: {
            s.put('[');
            bool first = true;
            for (const JsonNode* e = v->head; e != nullptr; e = e->next) {
                if (!first) s.put(',');
                first = false;
                if (!emit_value(e->value, s, depth + 1)) return false;
            }
            s.put(']');
            return true;
        }
    }
    return false;  // unreachable for a well-formed node
}

// ---------------------------------------------------------------------------
// Tape -> value tree. Walks the pre-order token stream with a shared cursor.
// ---------------------------------------------------------------------------
bool build_from_tape(const StructuralIndex* idx, int32_t* cur,
                     Arena* arena, int32_t depth, JsonValue** out) noexcept {
    assert(idx != nullptr);
    assert(cur != nullptr);
    if (depth > kJsonMaxDepth) return false;
    if (*cur < 0 || *cur >= idx->token_count) return false;

    const char* src = reinterpret_cast<const char*>(idx->src);
    const Token& t = idx->tokens[*cur];
    switch (t.type) {
        case TokenType::BeginObject: {
            ++*cur;
            JsonValue* obj = make_object(arena);
            if (obj == nullptr) return false;
            while (true) {
                if (*cur >= idx->token_count) return false;
                const Token& kt = idx->tokens[*cur];
                if (kt.type == TokenType::EndObject) { ++*cur; break; }
                if (kt.type != TokenType::Key) return false;
                const char* key = src + kt.start;
                const int32_t key_len = kt.length;
                ++*cur;
                JsonValue* val = nullptr;
                if (!build_from_tape(idx, cur, arena, depth + 1, &val)) return false;
                // Reference the source key bytes directly (zero-copy, verbatim).
                if (!container_append(arena, obj, key, key_len, val)) return false;
            }
            *out = obj;
            return true;
        }
        case TokenType::BeginArray: {
            ++*cur;
            JsonValue* arr = make_array(arena);
            if (arr == nullptr) return false;
            while (true) {
                if (*cur >= idx->token_count) return false;
                if (idx->tokens[*cur].type == TokenType::EndArray) { ++*cur; break; }
                JsonValue* val = nullptr;
                if (!build_from_tape(idx, cur, arena, depth + 1, &val)) return false;
                if (!container_append(arena, arr, nullptr, 0, val)) return false;
            }
            *out = arr;
            return true;
        }
        case TokenType::String: {
            JsonValue* v = make_string_raw(arena, src + t.start, t.length);
            if (v == nullptr) return false;
            ++*cur;
            *out = v;
            return true;
        }
        case TokenType::Int64:
        case TokenType::Float64:
        case TokenType::BoolTrue:
        case TokenType::BoolFalse:
        case TokenType::Null: {
            // Every scalar keyword/number token slices its exact source bytes
            // ("true"/"false"/"null"/the raw number) — emit them verbatim.
            JsonValue* v = make_literal(arena, src + t.start, t.length);
            if (v == nullptr) return false;
            ++*cur;
            *out = v;
            return true;
        }
        case TokenType::Key:
        case TokenType::EndObject:
        case TokenType::EndArray:
        case TokenType::End:
            return false;  // not a value start
    }
    return false;
}

}  // namespace

// ===========================================================================
// Constructors.
// ===========================================================================
JsonValue* make_null(Arena* arena) noexcept {
    JsonValue* v = alloc_value(arena, ValueKind::Literal);
    if (v == nullptr) return nullptr;
    v->bytes = "null";
    v->len   = 4;
    return v;
}

JsonValue* make_bool(Arena* arena, bool b) noexcept {
    JsonValue* v = alloc_value(arena, ValueKind::Literal);
    if (v == nullptr) return nullptr;
    v->bytes = b ? "true" : "false";
    v->len   = b ? 4 : 5;
    return v;
}

JsonValue* make_int(Arena* arena, int64_t x) noexcept {
    assert(arena != nullptr);
    char tmp[20];
    const int32_t n = format_int64(tmp, x);
    char* copy = static_cast<char*>(arena->copy_into(tmp, static_cast<size_t>(n), 1));
    if (copy == nullptr) return nullptr;
    return make_literal(arena, copy, n);
}

JsonValue* make_double(Arena* arena, double x) noexcept {
    assert(arena != nullptr);
    if (!std::isfinite(x)) return make_null(arena);  // NaN/Inf -> null (valid JSON)
    char tmp[32];
    const int n = std::snprintf(tmp, sizeof(tmp), "%.17g", x);
    if (n <= 0 || n >= static_cast<int>(sizeof(tmp))) return nullptr;
    char* copy = static_cast<char*>(arena->copy_into(tmp, static_cast<size_t>(n), 1));
    if (copy == nullptr) return nullptr;
    return make_literal(arena, copy, n);
}

JsonValue* make_string(Arena* arena, const char* bytes, int32_t len) noexcept {
    if (len < 0) return nullptr;
    JsonValue* v = alloc_value(arena, ValueKind::StringEscaped);
    if (v == nullptr) return nullptr;
    v->bytes = bytes;
    v->len   = len;
    return v;
}

JsonValue* make_string_raw(Arena* arena, const char* bytes, int32_t len) noexcept {
    if (len < 0) return nullptr;
    JsonValue* v = alloc_value(arena, ValueKind::StringVerbatim);
    if (v == nullptr) return nullptr;
    v->bytes = bytes;
    v->len   = len;
    return v;
}

JsonValue* make_literal(Arena* arena, const char* bytes, int32_t len) noexcept {
    if (len < 0) return nullptr;
    JsonValue* v = alloc_value(arena, ValueKind::Literal);
    if (v == nullptr) return nullptr;
    v->bytes = bytes;
    v->len   = len;
    return v;
}

JsonValue* make_object(Arena* arena) noexcept {
    return alloc_value(arena, ValueKind::Object);
}

JsonValue* make_array(Arena* arena) noexcept {
    return alloc_value(arena, ValueKind::Array);
}

// ===========================================================================
// Container mutation.
// ===========================================================================
bool object_add(Arena* arena, JsonValue* obj,
                const char* key, int32_t key_len, JsonValue* val) noexcept {
    if (arena == nullptr || obj == nullptr) return false;
    if (obj->kind != ValueKind::Object) return false;
    if (key_len < 0 || (key == nullptr && key_len != 0)) return false;
    char* key_copy = nullptr;
    if (key_len > 0) {
        key_copy = static_cast<char*>(
            arena->copy_into(key, static_cast<size_t>(key_len), 1));
        if (key_copy == nullptr) return false;
    }
    return container_append(arena, obj, key_copy, key_len, val);
}

bool array_add(Arena* arena, JsonValue* arr, JsonValue* val) noexcept {
    if (arena == nullptr || arr == nullptr) return false;
    if (arr->kind != ValueKind::Array) return false;
    return container_append(arena, arr, nullptr, 0, val);
}

JsonValue* object_get(const JsonValue* obj,
                      const char* key, int32_t key_len) noexcept {
    if (obj == nullptr || obj->kind != ValueKind::Object) return nullptr;
    if (key_len < 0 || (key == nullptr && key_len != 0)) return nullptr;
    for (const JsonNode* m = obj->head; m != nullptr; m = m->next) {
        if (m->key_len == key_len &&
            (key_len == 0 ||
             memcmp(m->key, key, static_cast<size_t>(key_len)) == 0)) {
            return m->value;
        }
    }
    return nullptr;
}

// ===========================================================================
// In-place scalar mutation.
// ===========================================================================
void set_string(JsonValue* v, const char* bytes, int32_t len) noexcept {
    assert(v != nullptr);
    assert(len >= 0);
    v->kind  = ValueKind::StringEscaped;
    v->bytes = bytes;
    v->len   = len;
}

void set_string_raw(JsonValue* v, const char* bytes, int32_t len) noexcept {
    assert(v != nullptr);
    assert(len >= 0);
    v->kind  = ValueKind::StringVerbatim;
    v->bytes = bytes;
    v->len   = len;
}

// ===========================================================================
// Tape -> value tree.
// ===========================================================================
bool value_from_index(const StructuralIndex* idx, Arena* arena,
                      JsonValue** out_root) noexcept {
    if (idx == nullptr || arena == nullptr || out_root == nullptr) return false;
    if (idx->tokens == nullptr || idx->token_count <= 0) return false;
    if (idx->src == nullptr) return false;
    int32_t cur = 0;
    JsonValue* root = nullptr;
    if (!build_from_tape(idx, &cur, arena, 0, &root)) return false;
    // The only token allowed to remain is the terminal End sentinel.
    if (cur >= idx->token_count) return false;
    if (idx->tokens[cur].type != TokenType::End) return false;
    *out_root = root;
    return true;
}

// ===========================================================================
// Serialise.
// ===========================================================================
bool serialize(const JsonValue* root, Arena* arena,
               const uint8_t** out_bytes, int32_t* out_len) noexcept {
    if (root == nullptr || arena == nullptr ||
        out_bytes == nullptr || out_len == nullptr) {
        return false;
    }
    // Pass 1: exact length.
    CountSink cs{0};
    if (!emit_value(root, cs, 0)) return false;
    if (cs.n < 0 || cs.n > INT32_MAX) return false;
    const int32_t total = static_cast<int32_t>(cs.n);

    // Single allocation (never zero-sized so the pointer is always valid).
    char* buf = arena->allocate_array<char>(total == 0 ? 1 : static_cast<size_t>(total));
    if (buf == nullptr) return false;

    // Pass 2: fill it. The length must match pass 1 exactly.
    WriteSink ws{buf, buf + total, true};
    if (!emit_value(root, ws, 0)) return false;
    assert(ws.ok && "write pass overran the counted length");
    assert(ws.p == buf + total && "write pass length != count pass length");
    if (!ws.ok || ws.p != buf + total) return false;

    *out_bytes = reinterpret_cast<const uint8_t*>(buf);
    *out_len   = total;
    return true;
}

}  // namespace json
}  // namespace parse
}  // namespace bolt
