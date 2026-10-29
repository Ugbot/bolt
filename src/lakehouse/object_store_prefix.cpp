#include "bolt/lakehouse/object_store/prefix.h"

#include <cassert>
#include <cstring>

namespace bolt::lakehouse {
namespace {

bool relative_length(const char* key, bool empty, uint32_t* out) noexcept {
    assert(out != nullptr);
    assert(kOsMaxKey > 1);
    if (key == nullptr) return false;
    uint32_t start = 0;
    for (uint32_t n = 0; n < kOsMaxKey; ++n) {
        const auto ch = static_cast<unsigned char>(key[n]);
        if (ch < 32 || ch == 127 || ch == '\\') {
            if (ch != 0) return false;
        }
        if (ch != '/' && ch != 0) continue;
        const uint32_t length = n - start;
        if (length == 0 && !(ch == 0 && (n != 0 || empty))) return false;
        if ((length == 1 && key[start] == '.') ||
            (length == 2 && key[start] == '.' && key[start + 1] == '.')) return false;
        if (ch == 0) { *out = n; return n != 0 || empty; }
        start = n + 1;
    }
    return false;
}

PrefixObjectStore* state_of(void* context) noexcept {
    assert(context != nullptr);
    auto* state = static_cast<PrefixObjectStore*>(context);
    assert(state->prefix_length > 0 && state->prefix_length < kOsMaxKey);
    return state;
}

bool joined(PrefixObjectStore* state, const char* key, bool empty,
            char out[kOsMaxKey]) noexcept {
    assert(state != nullptr && out != nullptr);
    assert(state->prefix_length > 0 && state->prefix_length < kOsMaxKey);
    uint32_t n = 0;
    if (!relative_length(key, empty, &n) || n >= kOsMaxKey - state->prefix_length)
        return false;
    std::memcpy(out, state->prefix, state->prefix_length);
    std::memcpy(out + state->prefix_length, key, n + 1u);
    return true;
}

int get(void* context, const char* key, Arena* arena,
        const uint8_t** data, uint64_t* length) noexcept {
    assert(arena != nullptr && data != nullptr);
    assert(length != nullptr);
    *data = nullptr;
    *length = 0;
    auto* state = state_of(context);
    char full[kOsMaxKey]{};
    if (!joined(state, key, true, full)) return kOsBadArg;
    return os_get(&state->origin, full, arena, data, length);
}

int put(void* context, const char* key, const uint8_t* data, uint64_t length) noexcept {
    assert(data != nullptr || length == 0);
    assert(context != nullptr);
    auto* state = state_of(context);
    char full[kOsMaxKey]{};
    if (!joined(state, key, true, full)) return kOsBadArg;
    return os_put(&state->origin, full, data, length);
}

int create(void* context, const char* key, const uint8_t* data, uint64_t length) noexcept {
    assert(data != nullptr || length == 0);
    assert(context != nullptr);
    auto* state = state_of(context);
    char full[kOsMaxKey]{};
    if (!joined(state, key, true, full)) return kOsBadArg;
    return os_put_if_absent(&state->origin, full, data, length);
}

int create_versioned(void* context, const char* key, const uint8_t* data, uint64_t length, ObjectVersion* out) noexcept {
    assert(data != nullptr || length == 0);
    assert(context != nullptr && out != nullptr);
    auto* state = state_of(context);
    char full[kOsMaxKey]{};
    *out = {};
    if (!joined(state, key, true, full)) return kOsBadArg;
    return os_put_if_absent_versioned(&state->origin, full, data, length, out);
}

int erase(void* context, const char* key) noexcept {
    assert(context != nullptr);
    assert(key != nullptr);
    auto* state = state_of(context);
    char full[kOsMaxKey]{};
    if (!joined(state, key, true, full)) return kOsBadArg;
    return os_delete(&state->origin, full);
}

int head(void* context, const char* key, ObjectMeta* meta) noexcept {
    assert(context != nullptr);
    assert(meta != nullptr);
    *meta = {};
    auto* state = state_of(context);
    char full[kOsMaxKey]{};
    if (!joined(state, key, true, full)) return kOsBadArg;
    return os_head(&state->origin, full, meta);
}

int list(void* context, const char* prefix, ObjectEntry* entries,
         uint32_t capacity, uint32_t* count) noexcept {
    assert(count != nullptr);
    assert(entries != nullptr || capacity == 0);
    *count = 0;
    auto* state = state_of(context);
    char full[kOsMaxKey]{};
    if (!joined(state, prefix, true, full)) return kOsBadArg;
    const size_t full_length = std::strlen(full);
    uint32_t got = 0;
    ObjectEntry unused{};
    const int status = os_list(&state->origin, full, entries == nullptr ? &unused : entries, capacity, &got);
    if (status != kOsOk && status != kOsBufferSmall) return status;
    if (got > capacity || (status == kOsBufferSmall && got != capacity)) return kOsIoError;
    for (uint32_t i = 0; i < got; ++i) {
        uint32_t length = 0;
        if (!relative_length(entries[i].key, false, &length) || length < full_length ||
            std::memcmp(entries[i].key, full, full_length) != 0)
            return kOsIoError;
    }
    for (uint32_t i = 0; i < got; ++i) {
        // The first bounded pass established NUL termination and membership.
        const size_t length = std::strlen(entries[i].key) - state->prefix_length;
        std::memmove(entries[i].key, entries[i].key + state->prefix_length, length + 1u);
    }
    *count = got;
    return status;
}

int range(void* context, const char* key, uint64_t offset, uint64_t length,
          uint8_t* dst, uint64_t* got) noexcept {
    assert(got != nullptr);
    assert(dst != nullptr || length == 0);
    auto* state = state_of(context);
    detail::ObjectSpan input{};
    const detail::ObjectSpan outputs[]{{dst, length > SIZE_MAX ? 0u : static_cast<size_t>(length)}, {got, sizeof(*got)}};
    const detail::ObjectSpan protected_span{state, sizeof(*state)};
    if (length > SIZE_MAX || !detail::object_key_span(key, kOsMaxKey, &input) ||
        !detail::object_io_admit(&input, 1, outputs, 2, nullptr, &protected_span, 1)) return kOsBadArg;
    char full[kOsMaxKey]{};
    if (!joined(state, key, true, full)) { *got = 0; return kOsBadArg; }
    return os_get_range(&state->origin, full, offset, length, dst, got);
}

int version(void* context, const char* key, Arena* scratch, ObjectVersion* out) noexcept {
    assert(scratch != nullptr);
    assert(out != nullptr);
    auto* state = state_of(context);
    detail::ObjectSpan input{};
    const detail::ObjectSpan output{out, sizeof(*out)}, protected_span{state, sizeof(*state)};
    if (!detail::object_key_span(key, kOsMaxKey, &input) ||
        !detail::object_io_admit(&input, 1, &output, 1, scratch, &protected_span, 1)) return kOsBadArg;
    char full[kOsMaxKey]{};
    if (!joined(state, key, true, full)) { *out = {}; return kOsBadArg; }
    return os_head_versioned(&state->origin, full, scratch, out);
}

int pinned_range(void* context, const char* key, const ObjectVersion* expected,
                 uint64_t offset, uint32_t length, Arena* scratch,
                 uint8_t* dst, uint32_t* got) noexcept {
    assert(scratch != nullptr && expected != nullptr);
    assert(got != nullptr && (dst != nullptr || length == 0));
    auto* state = state_of(context);
    detail::ObjectSpan inputs[2]{{}, {expected, sizeof(*expected)}};
    const detail::ObjectSpan outputs[]{{dst, length > kOsVersionedRangeMaxBytes ? 0u : length}, {got, sizeof(*got)}};
    const detail::ObjectSpan protected_span{state, sizeof(*state)};
    if (!detail::object_key_span(key, kOsMaxKey, &inputs[0]) ||
        !detail::object_io_admit(inputs, 2, outputs, 2, scratch, &protected_span, 1)) return kOsBadArg;
    char full[kOsMaxKey]{};
    if (!joined(state, key, true, full)) { *got = 0; return kOsBadArg; }
    return os_get_range_if_match(&state->origin, full, expected, offset, length, scratch, dst, got);
}

int pinned_stream(void* context, const char* key, const ObjectVersion* expected, uint64_t offset, uint64_t length,
                  const ObjectRangeSink* sink, uint64_t* delivered) noexcept {
    assert(expected != nullptr && sink != nullptr);
    assert(delivered != nullptr);
    auto* state = state_of(context);
    *delivered = 0;
    char full[kOsMaxKey]{};
    if (!joined(state, key, true, full)) return kOsBadArg;
    return os_get_range_if_match_stream(&state->origin, full, expected, offset, length, sink, delivered);
}
}  // namespace

bool prefix_object_store_init(PrefixObjectStore* state, const ObjectStore* origin,
                              const char* prefix, ObjectStore* out) noexcept {
    assert(state != nullptr && origin != nullptr);
    assert(out != nullptr);
    if (out == &state->origin || origin->vt == nullptr || origin->vt == &state->vtable || origin->impl == state ||
        origin->vt->get_object == nullptr || origin->vt->put_object == nullptr ||
        origin->vt->list_prefix == nullptr || origin->vt->delete_object == nullptr ||
        origin->vt->head_object == nullptr) return false;
    uint32_t length = 0;
    if (!relative_length(prefix, false, &length)) return false;
    const bool slash = prefix[length - 1] == '/';
    const uint32_t normalized = length + (slash ? 0u : 1u);
    const auto* parent = origin->vt->get_object == get
        ? static_cast<const PrefixObjectStore*>(origin->impl) : nullptr;
    if (origin->vt->get_object == get && parent == nullptr) return false;
    if (parent != nullptr && (parent->origin.vt == nullptr ||
        parent->origin.vt->get_object == get || parent->prefix_length == 0)) return false;
    const uint32_t inherited = parent == nullptr ? 0u : parent->prefix_length;
    if (inherited > kOsMaxKey - 2u || normalized > kOsMaxKey - 2u - inherited) return false;
    char copied_prefix[kOsMaxKey]{};
    if (parent != nullptr) std::memcpy(copied_prefix, parent->prefix, inherited);
    std::memcpy(copied_prefix + inherited, prefix, length);
    length += inherited;
    // All admission is complete. No remote effects or partially bound output.
    const ObjectStore copy = parent == nullptr ? *origin : parent->origin;
    state->origin = copy;
    std::memcpy(state->prefix, copied_prefix, length);
    if (!slash) state->prefix[length++] = '/';
    state->prefix[length] = 0;
    state->prefix_length = length;
    state->vtable = {get, put, list, erase, head,
        copy.vt->put_if_absent == nullptr ? nullptr : create,
        copy.vt->get_range == nullptr ? nullptr : range,
        copy.vt->head_versioned == nullptr ? nullptr : version,
        copy.vt->get_range_if_match == nullptr ? nullptr : pinned_range,
        copy.vt->put_if_absent == nullptr ? nullptr : create_versioned,
        copy.vt->get_range_if_match_stream == nullptr ? nullptr : pinned_stream};
    out->vt = &state->vtable;
    out->impl = state;
    assert(state->prefix_length > 0 && out->impl == state);
    return true;
}
}  // namespace bolt::lakehouse
