#include "bolt/parse/bolt_json_canonical.h"
#include "bolt/parse/bolt_json_writer.h"
#include "bolt/kernels/bolt_json_path.h"
#include <algorithm>
#include <cassert>
#include <cstring>
namespace bolt::parse::json {
namespace {
using R = CanonicalResult;
struct State { Arena* arena; CanonicalLimits limits; uint32_t used = 0u; };
R decoded(State& s, const char* in, int32_t len, const char** out, int32_t* n) noexcept {
    assert(in && out && n); assert(len >= 0);
    auto* p = static_cast<char*>(s.arena->allocate(static_cast<size_t>(len) + 1u)); if (!p) return R::NoMemory;
    const kernels::json::JsonSpan span{in - 1,static_cast<uint32_t>(len) + 2u};
    const auto bytes = kernels::json::json_span_unquote(span,p,static_cast<uint32_t>(len)); if (bytes < 0) return R::Invalid;
    // The maintained writer's size pass validates decoded UTF-8 before emit.
    *out = p; *n = bytes; return R::Ok;
}
bool less(const JsonNode* a, const JsonNode* b) noexcept {
    assert(a && b); assert(a->key_len >= 0 && b->key_len >= 0);
    const auto n = std::min(a->key_len,b->key_len); const int order = std::memcmp(a->key,b->key,static_cast<size_t>(n));
    return order < 0 || (order == 0 && a->key_len < b->key_len);
}
R normalize(State& s, JsonValue* v, uint32_t depth) noexcept {
    assert(v); assert(s.used <= s.limits.nodes);
    if (depth > s.limits.depth || s.used == s.limits.nodes) return R::Limit;
    ++s.used;
    if (v->kind == ValueKind::StringVerbatim) {
        const char* p = nullptr; int32_t n = 0; const auto result = decoded(s,v->bytes,v->len,&p,&n); if (result != R::Ok) return result;
        v->kind = ValueKind::StringEscaped; v->bytes = p; v->len = n;
    }
    if (v->kind != ValueKind::Object && v->kind != ValueKind::Array) return R::Ok;
    if (v->count < 0 || static_cast<uint32_t>(v->count) > s.limits.nodes - s.used) return R::Limit;
    JsonNode** order = nullptr;
    if (v->kind == ValueKind::Object && v->count) {
        order = s.arena->allocate_array<JsonNode*>(static_cast<size_t>(v->count)); if (!order) return R::NoMemory;
    }
    auto* node = v->head;
    for (int32_t i = 0; i < v->count; ++i) {
        if (!node) return R::Invalid;
        if (order) {
            const char* key = nullptr; int32_t n = 0; const auto result = decoded(s,node->key,node->key_len,&key,&n); if (result != R::Ok) return result;
            node->key = key; node->key_len = n; order[i] = node;
        }
        const auto result = normalize(s,node->value,depth + 1u); if (result != R::Ok) return result;
        node = node->next;
    }
    if (node) return R::Invalid;
    if (order) {
        std::sort(order,order + v->count,less);
        for (int32_t i = 1; i < v->count; ++i) if (!less(order[i-1],order[i])) return R::Invalid;
        v->head = order[0]; v->tail = order[v->count - 1];
        for (int32_t i = 0; i < v->count; ++i) order[i]->next = i + 1 < v->count ? order[i+1] : nullptr;
    }
    return R::Ok;
}
}
CanonicalResult canonicalize(const uint8_t* input, uint32_t bytes, Arena* arena, CanonicalLimits limits,
    const uint8_t** output, uint32_t* output_bytes) noexcept {
    assert(kJsonMaxDepth > 0); assert(sizeof(uint32_t) == 4u);
    if (!input || !bytes || !arena || !output || !output_bytes || !limits.bytes || limits.bytes > INT32_MAX || !limits.nodes ||
        limits.depth > static_cast<uint32_t>(kJsonMaxDepth)) return R::Invalid;
    if (bytes > limits.bytes) return R::Limit;
    const SaxHandler syntax{}; if (!sax_parse(input,static_cast<int32_t>(bytes),&syntax)) return R::Invalid;
    StructuralIndex index{}; JsonValue* root = nullptr;
    if (!build_index(input,static_cast<int32_t>(bytes),arena,&index) || !value_from_index(&index,arena,&root)) return R::NoMemory;
    State state{arena,limits}; const auto result = normalize(state,root,0u); if (result != R::Ok) return result;
    const uint8_t* out = nullptr; int32_t size = 0; EmitOptions options{}; options.ensure_ascii = true;
    uint32_t measured = 0u; if (!measure_serialized(root,options,&measured)) return R::Invalid;
    if (measured > limits.bytes) return R::Limit;
    if (!serialize(root,arena,options,&out,&size)) return R::NoMemory;
    if (size < 0 || static_cast<uint32_t>(size) > limits.bytes) return R::Limit;
    *output = out; *output_bytes = static_cast<uint32_t>(size); return R::Ok;
}
} // namespace bolt::parse::json
