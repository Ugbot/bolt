#include "bolt/lakehouse/object_store/pinned.h"

#include <cassert>
#include <cstring>

namespace bolt::lakehouse {
namespace {
using detail::ObjectSpan;
extern const ObjectStoreVT kPinnedVT;

bool canonical_key(const char* key) noexcept {
    assert(key != nullptr);
    assert(kOsMaxKey > 1);
    uint32_t component = 0;
    for (uint32_t i = 0; i < kOsMaxKey; ++i) {
        const auto c = static_cast<unsigned char>(key[i]);
        if (c == 0 || c == '/') {
            if (component == 0 || (component == 1 && key[i - 1] == '.') ||
                (component == 2 && key[i - 1] == '.' && key[i - 2] == '.')) return false;
            if (c == 0) return true;
            component = 0;
        } else {
            if (c < 32 || c == 127 || c == '\\') return false;
            ++component;
        }
    }
    return false;
}

bool ready(PinnedObjectStore* s) noexcept {
    assert(s != nullptr);
    assert(s == nullptr || !s->initialized || s->owner_thread == std::this_thread::get_id());
    return s != nullptr && s->initialized && s->io != nullptr &&
        s->owner_thread == std::this_thread::get_id() && !s->io->growth_enabled() &&
        !s->io->concurrent() && s->origin.vt != nullptr &&
        s->origin.vt->get_range_if_match != nullptr;
}

uint32_t protected_spans(PinnedObjectStore* s, ObjectSpan* spans) noexcept {
    assert(s != nullptr);
    assert(spans != nullptr);
    spans[0] = {s, sizeof(*s)};
    spans[1] = {s->entries, static_cast<size_t>(s->count) * sizeof(PinnedObject)};
    spans[2] = {s->origin.vt, sizeof(ObjectStoreVT)};
    spans[3] = {&kPinnedVT, sizeof(kPinnedVT)};
    return 4;
}

bool workspace_admit(PinnedObjectStore* s, Arena* arena, bool allow_io) noexcept {
    assert(s != nullptr);
    assert(s->io != nullptr);
    const ObjectSpan object{arena, sizeof(*arena)};
    if (!detail::object_span_valid(object)) return false;
    if (arena == s->io) return allow_io;
    ObjectSpan spans[4]{};
    const auto n = protected_spans(s, spans);
    // Reject every object alias before inspecting the proposed Arena at all.
    // Only the already-admitted IO Arena may be dereferenced in this phase.
    for (uint32_t i = 0; i < n; ++i)
        if (detail::object_spans_overlap(object, spans[i])) return false;
    if (detail::object_spans_overlap(object, {s->io, sizeof(*s->io)}) ||
        s->io->overlaps_backing(object.data, object.size)) return false;
    // The proposed Arena object is now disjoint from all protected storage.
    for (uint32_t i = 0; i < n; ++i)
        if (arena->overlaps_backing(spans[i].data, spans[i].size)) return false;
    return !arena->overlaps_backing(s->io, sizeof(*s->io));
}

bool arena_object_disjoint(Arena* arena, const ObjectSpan* inputs, uint32_t ni,
                           const ObjectSpan* outputs, uint32_t no) noexcept {
    assert(ni <= 4);
    assert(no <= 2);
    const ObjectSpan object{arena, sizeof(*arena)};
    return detail::object_span_valid(object) &&
        detail::object_io_admit(inputs, ni, outputs, no, nullptr, &object, 1);
}

bool admit(PinnedObjectStore* s, const char* key, const ObjectVersion* expected,
           const ObjectSpan* outputs, uint32_t no, Arena* scratch) noexcept {
    assert(s != nullptr);
    assert(no <= 2);
    if (!ready(s)) return false;
    ObjectSpan inputs[2]{{}, {expected, expected == nullptr ? 0u : sizeof(*expected)}};
    if (!detail::object_key_span(key, kOsMaxKey, &inputs[0]) ||
        !detail::object_io_admit(inputs, 2, outputs, no, s->io, nullptr, 0)) return false;
    ObjectSpan protected_values[4]{};
    const uint32_t np = protected_spans(s, protected_values);
    if (!detail::object_io_admit(nullptr, 0, outputs, no, nullptr, protected_values, np)) return false;
    if (scratch != nullptr && (!arena_object_disjoint(scratch, inputs, 2, outputs, no) ||
        !workspace_admit(s, scratch, true) ||
        !detail::object_io_admit(inputs, 2, outputs, no, scratch, nullptr, 0))) return false;
    return true;
}

const PinnedObject* lookup(PinnedObjectStore* s, const char* key) noexcept {
    assert(s != nullptr && key != nullptr);
    assert(s->count <= kOsPinnedMaxEntries);
    uint32_t lo = 0, hi = s->count;
    for (uint32_t step = 0; step < 17 && lo < hi; ++step) {
        const uint32_t mid = lo + (hi - lo) / 2;
        const int cmp = std::strcmp(key, s->entries[mid].key);
        if (cmp == 0) return &s->entries[mid];
        if (cmp < 0) hi = mid; else lo = mid + 1;
    }
    return nullptr;
}

bool same_version(const ObjectVersion* a, const ObjectVersion* b) noexcept {
    assert(a != nullptr);
    assert(b != nullptr);
    return os_object_version_valid(a) && a->size == b->size && std::strcmp(a->etag, b->etag) == 0;
}

int read_chunk(PinnedObjectStore* s, const PinnedObject* pin, uint64_t offset,
               uint32_t length, uint8_t* dst) noexcept {
    assert(s != nullptr && pin != nullptr);
    assert(length > 0 && length <= s->limits.request_bytes);
    s->io->reset();
    uint32_t got = 0;
    const int rc = os_get_range_if_match(&s->origin, pin->key, &pin->version,
                                        offset, length, s->io, dst, &got);
    if (rc != kOsOk) return rc;
    return got == length ? kOsOk : kOsIoError;
}

int range_impl(PinnedObjectStore* s, const char* key, const ObjectVersion* expected,
               uint64_t offset, uint64_t length, uint8_t* dst) noexcept {
    assert(s != nullptr && key != nullptr);
    assert(length <= SIZE_MAX);
    const auto* pin = lookup(s, key);
    if (pin == nullptr) return kOsNotFound;
    if (expected != nullptr && !os_object_version_valid(expected)) return kOsBadArg;
    if (expected != nullptr && !same_version(expected, &pin->version)) return kOsPreconditionFailed;
    if (length > s->limits.request_bytes) return kOsBufferSmall;
    if (offset > pin->version.size || length > pin->version.size - offset) return kOsBadArg;
    if (length == 0) return kOsOk;
    return read_chunk(s, pin, offset, static_cast<uint32_t>(length), dst);
}

int pinned_range(void* impl, const char* key, uint64_t offset, uint64_t length,
                 uint8_t* dst, uint64_t* out_length) noexcept {
    assert(impl != nullptr);
    assert(out_length != nullptr);
    auto* s = static_cast<PinnedObjectStore*>(impl);
    const ObjectSpan outputs[]{{dst, length <= SIZE_MAX ? static_cast<size_t>(length) : SIZE_MAX},
                               {out_length, sizeof(*out_length)}};
    if (length > SIZE_MAX || !admit(s, key, nullptr, outputs, 2, nullptr)) return kOsBadArg;
    *out_length = 0;
    const int rc = range_impl(s, key, nullptr, offset, length, dst);
    if (rc == kOsOk) *out_length = length;
    return rc;
}

int pinned_conditional(void* impl, const char* key, const ObjectVersion* expected,
                       uint64_t offset, uint32_t length, Arena* scratch,
                       uint8_t* dst, uint32_t* out_length) noexcept {
    assert(impl != nullptr);
    assert(out_length != nullptr);
    auto* s = static_cast<PinnedObjectStore*>(impl);
    const ObjectSpan outputs[]{{dst, length}, {out_length, sizeof(*out_length)}};
    if (expected == nullptr || scratch == nullptr || !admit(s, key, expected, outputs, 2, scratch))
        return kOsBadArg;
    *out_length = 0;
    const int rc = range_impl(s, key, expected, offset, length, dst);
    if (rc == kOsOk) *out_length = length;
    return rc;
}

// G2ICE-381: one streamed request for a whole range of a pinned object (no request_bytes cap: the sink's buffer bounds memory).
int pinned_stream(void* impl, const char* key, const ObjectVersion* expected, uint64_t offset, uint64_t length,
                  const ObjectRangeSink* sink, uint64_t* delivered) noexcept {
    assert(impl != nullptr);
    assert(sink != nullptr && delivered != nullptr);
    auto* s = static_cast<PinnedObjectStore*>(impl);
    const ObjectSpan outputs[]{{sink->buf, sink->buf_cap}, {delivered, sizeof(*delivered)}};
    if (expected == nullptr || !admit(s, key, expected, outputs, 2, nullptr)) return kOsBadArg;
    *delivered = 0;
    const auto* pin = lookup(s, key);
    if (pin == nullptr) return kOsNotFound;
    if (!same_version(expected, &pin->version)) return kOsPreconditionFailed;
    if (offset > pin->version.size || length > pin->version.size - offset) return kOsBadArg;
    if (length == 0) return kOsOk;
    return os_get_range_if_match_stream(&s->origin, pin->key, &pin->version, offset, length, sink, delivered);
}

int pinned_head(void* impl, const char* key, ObjectMeta* out) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr);
    auto* s = static_cast<PinnedObjectStore*>(impl);
    const ObjectSpan output{out, sizeof(*out)};
    if (!admit(s, key, nullptr, &output, 1, nullptr)) return kOsBadArg;
    *out = {};
    const auto* pin = lookup(s, key);
    if (pin == nullptr) return kOsNotFound;
    out->exists = true;
    out->size = pin->version.size;
    return kOsOk;
}

int pinned_version(void* impl, const char* key, Arena* scratch, ObjectVersion* out) noexcept {
    assert(impl != nullptr);
    assert(out != nullptr);
    auto* s = static_cast<PinnedObjectStore*>(impl);
    const ObjectSpan output{out, sizeof(*out)};
    if (scratch == nullptr || !admit(s, key, nullptr, &output, 1, scratch)) return kOsBadArg;
    *out = {};
    const auto* pin = lookup(s, key);
    if (pin == nullptr) return kOsNotFound;
    *out = pin->version;
    return kOsOk;
}

int pinned_get(void* impl, const char* key, Arena* arena,
               const uint8_t** out_data, uint64_t* out_length) noexcept {
    assert(impl != nullptr);
    assert(out_data != nullptr && out_length != nullptr);
    auto* s = static_cast<PinnedObjectStore*>(impl);
    const ObjectSpan outputs[]{{out_data, sizeof(*out_data)}, {out_length, sizeof(*out_length)}};
    if (!admit(s, key, nullptr, outputs, 2, arena) || !workspace_admit(s, arena, false)) return kOsBadArg;
    const auto* pin = lookup(s, key);
    if (pin == nullptr) return kOsNotFound;
    const uint64_t size = pin->version.size;
    if (size > s->limits.max_get_bytes || size > SIZE_MAX) return kOsBufferSmall;
    uint8_t* data = size == 0 ? nullptr : static_cast<uint8_t*>(arena->allocate(static_cast<size_t>(size), 1));
    if (size != 0 && data == nullptr) return kOsBufferSmall;
    uint64_t offset = 0;
    const uint64_t max_chunks = (s->limits.max_get_bytes + s->limits.request_bytes - 1) / s->limits.request_bytes;
    for (uint64_t chunk = 0; chunk < max_chunks && offset < size; ++chunk) {
        const auto length = static_cast<uint32_t>(size - offset < s->limits.request_bytes ?
                                                size - offset : s->limits.request_bytes);
        const int rc = read_chunk(s, pin, offset, length, data + static_cast<size_t>(offset));
        if (rc != kOsOk) return rc;
        offset += length;
    }
    assert(offset == size);
    *out_data = data;
    *out_length = size;
    return kOsOk;
}

int unsupported_put(void* impl, const char*, const uint8_t*, uint64_t) noexcept {
    assert(impl != nullptr);
    assert(ready(static_cast<PinnedObjectStore*>(impl)));
    return kOsNotImplemented;
}
int unsupported_list(void* impl, const char*, ObjectEntry*, uint32_t, uint32_t*) noexcept {
    assert(impl != nullptr);
    assert(ready(static_cast<PinnedObjectStore*>(impl)));
    return kOsNotImplemented;
}
int unsupported_delete(void* impl, const char*) noexcept {
    assert(impl != nullptr);
    assert(ready(static_cast<PinnedObjectStore*>(impl)));
    return kOsNotImplemented;
}

const ObjectStoreVT kPinnedVT{pinned_get, unsupported_put, unsupported_list, unsupported_delete,
                             pinned_head, unsupported_put, pinned_range, pinned_version, pinned_conditional,
                             nullptr, nullptr, pinned_stream};

bool init_admit(PinnedObjectStore* state, const ObjectStore* origin, const char* ns,
                const PinnedObject* entries, uint32_t count,
                const PinnedObjectStoreLimits* limits, Arena* io, ObjectStore* out) noexcept {
    assert(count <= kOsPinnedMaxEntries);
    assert(sizeof(PinnedObject) <= SIZE_MAX / kOsPinnedMaxEntries);
    if (state == nullptr || origin == nullptr || limits == nullptr || io == nullptr || out == nullptr) return false;
    ObjectSpan inputs[4]{{origin, sizeof(*origin)}, {},
                        {entries, static_cast<size_t>(count) * sizeof(*entries)}, {limits, sizeof(*limits)}};
    const ObjectSpan outputs[]{{state, sizeof(*state)}, {out, sizeof(*out)}};
    if (!detail::object_key_span(ns, kOsMaxKey, &inputs[1]) || inputs[1].size < 2 ||
        !arena_object_disjoint(io, inputs, 4, outputs, 2)) return false;
    if (origin->vt == nullptr) return false;
    const ObjectSpan vt{origin->vt, sizeof(*origin->vt)};
    return arena_object_disjoint(io, &vt, 1, outputs, 2) &&
        detail::object_io_admit(inputs, 4, outputs, 2, io, nullptr, 0) &&
        detail::object_io_admit(&vt, 1, outputs, 2, io, nullptr, 0) &&
        !io->overlaps_backing(io, sizeof(*io));
}
}  // namespace

int pinned_object_store_init(PinnedObjectStore* state, const ObjectStore* origin,
                             const char* namespace_uri, const PinnedObject* entries,
                             uint32_t count, const PinnedObjectStoreLimits* limits,
                             Arena* io, ObjectStore* out) noexcept {
    assert(kOsPinnedMaxRequestBytes <= kOsVersionedRangeMaxBytes);
    assert(kOsPinnedMaxGetBytes <= SIZE_MAX);
    if (count > kOsPinnedMaxEntries || !init_admit(state, origin, namespace_uri, entries, count, limits, io, out))
        return kOsBadArg;
    if (origin->vt->get_range_if_match == nullptr) return kOsNotImplemented;
    if (limits->max_entries == 0 || limits->max_entries > kOsPinnedMaxEntries || count > limits->max_entries ||
        limits->max_get_bytes == 0 || limits->max_get_bytes > kOsPinnedMaxGetBytes ||
        limits->request_bytes == 0 || limits->request_bytes > kOsPinnedMaxRequestBytes ||
        io->growth_enabled() || io->concurrent() || io->total_allocated() != 0) return kOsBadArg;
    for (uint32_t i = 0; i < count; ++i) {
        if (!canonical_key(entries[i].key) || !os_object_version_valid(&entries[i].version) ||
            (i != 0 && std::strcmp(entries[i - 1].key, entries[i].key) >= 0)) return kOsBadArg;
    }
    const bool enough = io->allocate(limits->request_bytes, 1) != nullptr;
    io->reset();
    if (!enough) return kOsBufferSmall;
    state->origin = *origin;
    state->entries = entries;
    state->count = count;
    const size_t ns_len = std::strlen(namespace_uri);
    std::memset(state->namespace_uri, 0, sizeof(state->namespace_uri));
    std::memcpy(state->namespace_uri, namespace_uri, ns_len);
    state->limits = *limits;
    state->io = io;
    state->owner_thread = std::this_thread::get_id();
    state->initialized = true;
    *out = {&kPinnedVT, state};
    return kOsOk;
}

int pinned_object_store_view(PinnedObjectStore* state, ObjectStore* out) noexcept {
    assert(state != nullptr);
    assert(out != nullptr);
    if (state == nullptr || out == nullptr || !ready(state)) return kOsBadArg;
    ObjectSpan protected_values[4]{};
    const auto np = protected_spans(state, protected_values);
    const ObjectSpan output{out, sizeof(*out)};
    if (!detail::object_io_admit(nullptr, 0, &output, 1, state->io, protected_values, np)) return kOsBadArg;
    *out = {&kPinnedVT, state};
    return kOsOk;
}
}  // namespace bolt::lakehouse
