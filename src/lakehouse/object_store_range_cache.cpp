#include "bolt/lakehouse/object_store/range_cache.h"
#include <cassert>
#include <cstring>
#include <new>
#include <thread>

namespace bolt::lakehouse {
namespace {
struct RangePage {
    char key[kOsMaxKey]{};
    ObjectVersion version{};
    uint64_t offset = 0, age = 0;
    uint32_t length = 0;
    uint8_t* bytes = nullptr;
    bool valid = false;
};
void add(uint64_t* value, uint64_t amount = 1) noexcept {
    assert(value != nullptr);
    *value = amount > UINT64_MAX - *value ? UINT64_MAX : *value + amount;
    assert(*value >= amount);
}
bool same_version(const ObjectVersion& a, const ObjectVersion& b) noexcept {
    assert(os_object_version_valid(&a));
    assert(os_object_version_valid(&b));
    return a.size == b.size && std::strcmp(a.etag, b.etag) == 0;
}
}  // namespace

struct ObjectRangeCache {
    ObjectStore origin{};
    Arena* io = nullptr;
    Arena* backing = nullptr;
    uint8_t* page_bytes = nullptr;
    RangePage* pages = nullptr;
    uint32_t count = 0;
    uint64_t clock = 0;
    ObjectRangeCacheStats stats{};
    std::thread::id owner;
};

namespace {
void confined(const ObjectRangeCache* c) noexcept {
    assert(c != nullptr && c->io != nullptr);
    assert(c->owner == std::this_thread::get_id());
}
void invalidate(ObjectRangeCache* c, const char* key) noexcept {
    confined(c);
    assert(key != nullptr);
    for (uint32_t i = 0; i < c->count; ++i) {
        auto& page = c->pages[i];
        if (page.valid && std::strcmp(page.key, key) == 0) {
            page.valid = false;
            add(&c->stats.invalidated_pages);
        }
    }
}
int read_error(ObjectRangeCache* c, const char* key, int status) noexcept {
    confined(c);
    assert(status != kOsOk);
    invalidate(c, key);
    add(&c->stats.failures);
    if (status == kOsPreconditionFailed) add(&c->stats.preconditions);
    return status;
}
int read_head(ObjectRangeCache* c, const char* key, ObjectVersion* out) noexcept {
    confined(c);
    assert(out != nullptr);
    *out = {};
    if (c->io->overlaps_backing(key, std::strlen(key) + 1u) ||
        c->io->overlaps_backing(out, sizeof(*out))) return kOsBadArg;
    c->io->reset();
    add(&c->stats.head_calls);
    const int rc = os_head_versioned(&c->origin, key, c->io, out);
    if (rc != kOsOk) { *out = {}; return rc; }
    if (!os_object_version_valid(out)) { *out = {}; return kOsIoError; }
    // Do not retain obsolete generations of a key even when the new extent matches.
    for (uint32_t i = 0; i < c->count; ++i) {
        auto& page = c->pages[i];
        if (page.valid && std::strcmp(page.key, key) == 0 && !same_version(page.version, *out)) {
            page.valid = false;
            add(&c->stats.invalidated_pages);
        }
    }
    return kOsOk;
}
uint64_t next_age(ObjectRangeCache* c) noexcept {
    confined(c);
    if (c->clock == UINT64_MAX) {
        for (uint32_t i = 0; i < c->count; ++i) c->pages[i].age = 0;
        c->clock = 0;
    }
    ++c->clock;
    assert(c->clock > 0);
    return c->clock;
}
int range_from_origin(ObjectRangeCache* c, const char* key, const ObjectVersion& version,
                       uint64_t offset, uint32_t length, uint8_t* dst) noexcept {
    confined(c);
    assert(length > 0 && length <= kObjectRangePageBytes);
    c->io->reset();
    add(&c->stats.range_calls);
    uint32_t received = 0;
    const int rc = os_get_range_if_match(&c->origin, key, &version, offset, length, c->io, dst, &received);
    if (rc != kOsOk) return rc;
    if (received != length) return kOsIoError;
    add(&c->stats.origin_bytes, received);
    return kOsOk;
}
int cached_page(ObjectRangeCache* c, const char* key, const ObjectVersion& version,
                 uint64_t offset, RangePage** out) noexcept {
    confined(c);
    assert(out != nullptr && c->count > 0 && offset < version.size);
    RangePage* victim = nullptr;
    for (uint32_t i = 0; i < c->count; ++i) {
        auto& page = c->pages[i];
        if (page.valid && page.offset == offset && std::strcmp(page.key, key) == 0 &&
            same_version(page.version, version)) {
            page.age = next_age(c);
            add(&c->stats.page_hits);
            *out = &page;
            return kOsOk;
        }
        if (victim == nullptr || (!page.valid && victim->valid) ||
            (page.valid == victim->valid && page.age < victim->age)) victim = &page;
    }
    assert(victim != nullptr);
    add(&c->stats.page_misses);
    if (victim->valid) add(&c->stats.evictions);
    victim->valid = false;
    const uint64_t remaining = version.size - offset;
    const auto length = static_cast<uint32_t>(remaining < kObjectRangePageBytes ? remaining : kObjectRangePageBytes);
    const int rc = range_from_origin(c, key, version, offset, length, victim->bytes);
    if (rc != kOsOk) return rc;
    std::memcpy(victim->key, key, std::strlen(key) + 1u);
    victim->version = version;
    victim->offset = offset;
    victim->length = length;
    victim->age = next_age(c);
    victim->valid = true;
    add(&c->stats.fills);
    *out = victim;
    return kOsOk;
}
int read_pages(ObjectRangeCache* c, const char* key, const ObjectVersion& version,
                uint64_t offset, uint64_t length, uint8_t* dst) noexcept {
    confined(c);
    assert(length <= kObjectRangeMaxReadBytes);
    uint64_t copied = 0;
    for (uint32_t step = 0; copied < length && step <= kObjectRangeMaxPages; ++step) {
        const uint64_t position = offset + copied;
        if (c->count == 0) {
            const uint64_t remaining = length - copied;
            const auto want = static_cast<uint32_t>(remaining < kObjectRangePageBytes ? remaining : kObjectRangePageBytes);
            const int rc = range_from_origin(c, key, version, position, want, dst + copied);
            if (rc != kOsOk) return rc;
            copied += want;
        } else {
            const uint64_t page_offset = position - position % kObjectRangePageBytes;
            RangePage* page = nullptr;
            const int rc = cached_page(c, key, version, page_offset, &page);
            if (rc != kOsOk) return rc;
            const uint64_t within = position - page_offset;
            uint64_t want = page->length - within;
            if (want > length - copied) want = length - copied;
            std::memcpy(dst + copied, page->bytes + within, static_cast<size_t>(want));
            copied += want;
        }
    }
    assert(copied <= length);
    return copied == length ? kOsOk : kOsBufferSmall;
}
bool admit(ObjectRangeCache* c, const detail::ObjectSpan* inputs, uint32_t ni,
             const detail::ObjectSpan* outputs, uint32_t no) noexcept {
    confined(c);
    assert(c->count <= kObjectRangeMaxPages);
    const detail::ObjectSpan protected_spans[]{
        {c, sizeof(*c)}, {c->pages, static_cast<size_t>(c->count) * sizeof(RangePage)},
        {c->page_bytes, static_cast<size_t>(c->count) * kObjectRangePageBytes},
        {c->backing, sizeof(Arena)}, {c->origin.vt, sizeof(ObjectStoreVT)}};
    return detail::object_io_admit(inputs, ni, outputs, no, c->io, protected_spans, 5);
}
int read_admitted(ObjectRangeCache* c, const char* key, const ObjectVersion* expected,
                    uint64_t offset, uint64_t length, uint8_t* dst, uint64_t* out,
                    bool legacy_eof) noexcept {
    confined(c);
    assert(out != nullptr && key != nullptr);
    if (key[0] == 0 || (expected != nullptr && !os_object_version_valid(expected))) return kOsBadArg;
    if (length > kObjectRangeMaxReadBytes) return kOsBufferSmall;
    if ((!legacy_eof && length > UINT64_MAX - offset) || (expected != nullptr &&
        (offset > expected->size || length > expected->size - offset))) return kOsBadArg;
    add(&c->stats.reads);
    add(&c->stats.requested_bytes, length);
    ObjectVersion version{};
    int rc = read_head(c, key, &version);
    if (rc != kOsOk) return read_error(c, key, rc);
    if (expected != nullptr && !same_version(*expected, version)) return read_error(c, key, kOsPreconditionFailed);
    if (legacy_eof) {
        if (offset >= version.size) return kOsOk;
        if (length > version.size - offset) length = version.size - offset;
    } else if (offset > version.size || length > version.size - offset) return kOsBadArg;
    rc = read_pages(c, key, version, offset, length, dst);
    if (rc != kOsOk) return read_error(c, key, rc);
    *out = length;
    return kOsOk;
}
template <typename Count>
int read_impl(ObjectRangeCache* c, const char* key, const ObjectVersion* expected,
               uint64_t offset, uint64_t length, uint8_t* dst, Count* out,
               bool legacy_eof, bool conditional = false) noexcept {
    confined(c);
    assert(out != nullptr && (dst != nullptr || length == 0));
    detail::ObjectSpan inputs[2]{{}, {expected, expected == nullptr ? 0u : sizeof(*expected)}};
    const uint64_t limit = conditional ? kOsVersionedRangeMaxBytes : kObjectRangeMaxReadBytes;
    const detail::ObjectSpan outputs[]{{dst, length > limit ? 0u : static_cast<size_t>(length)}, {out, sizeof(*out)}};
    if (length > SIZE_MAX || !detail::object_key_span(key, kOsMaxKey, &inputs[0]) ||
        !admit(c, inputs, 2, outputs, 2)) return kOsBadArg;
    char stable_key[kOsMaxKey]{};
    std::memcpy(stable_key, key, inputs[0].size);
    ObjectVersion stable_expected{};
    if (expected != nullptr) stable_expected = *expected;
    *out = 0;
    if (conditional && (expected == nullptr || length > kOsVersionedRangeMaxBytes)) return kOsBadArg;
    uint64_t got = 0;
    const int rc = read_admitted(c, stable_key, expected == nullptr ? nullptr : &stable_expected,
                                 offset, length, dst, &got, legacy_eof);
    *out = static_cast<Count>(got);
    return rc;
}

ObjectRangeCache* cast(void* p) noexcept { return static_cast<ObjectRangeCache*>(p); }
int adapter_get(void* p, const char* key, Arena* arena, const uint8_t** data, uint64_t* length) noexcept {
    auto* c = cast(p); confined(c);
    return os_get(&c->origin, key, arena, data, length); // legacy whole-object forwarding, never advertised as cached
}
int write_result(ObjectRangeCache* c, int rc, uint64_t bytes) noexcept {
    confined(c);
    assert(rc >= kOsOk);
    if (rc == kOsOk) add(&c->stats.write_ack_bytes, bytes);
    else if (rc == kOsWriteUncertain) add(&c->stats.write_uncertain);
    else add(&c->stats.write_failures);
    return rc;
}
int adapter_put(void* p, const char* key, const uint8_t* data, uint64_t length) noexcept {
    auto* c = cast(p); confined(c);
    invalidate(c, key);
    add(&c->stats.write_attempts); add(&c->stats.write_attempted_bytes, length);
    return write_result(c, os_put(&c->origin, key, data, length), length);
}
int adapter_create(void* p, const char* key, const uint8_t* data, uint64_t length) noexcept {
    auto* c = cast(p); confined(c);
    invalidate(c, key);
    add(&c->stats.write_attempts); add(&c->stats.write_attempted_bytes, length);
    return write_result(c, os_put_if_absent(&c->origin, key, data, length), length);
}
int adapter_list(void* p, const char* prefix, ObjectEntry* entries, uint32_t capacity, uint32_t* count) noexcept {
    auto* c = cast(p); confined(c);
    return os_list(&c->origin, prefix, entries, capacity, count);
}
int adapter_delete(void* p, const char* key) noexcept {
    auto* c = cast(p); confined(c);
    invalidate(c, key);
    add(&c->stats.delete_attempts); add(&c->stats.write_attempts);
    return write_result(c, os_delete(&c->origin, key), 0);
}
int adapter_head(void* p, const char* key, ObjectMeta* out) noexcept {
    auto* c = cast(p); confined(c);
    return os_head(&c->origin, key, out);
}
int adapter_range(void* p, const char* key, uint64_t offset, uint64_t length,
                    uint8_t* dst, uint64_t* out) noexcept {
    return read_impl(cast(p), key, nullptr, offset, length, dst, out, true);
}
int adapter_version(void* p, const char* key, Arena*, ObjectVersion* out) noexcept {
    return object_range_cache_head(cast(p), key, out);
}
int adapter_conditional(void* p, const char* key, const ObjectVersion* expected,
                         uint64_t offset, uint32_t length, Arena*, uint8_t* dst, uint32_t* out) noexcept {
    return read_impl(cast(p), key, expected, offset, length, dst, out, false, true);
}
const ObjectStoreVT kCacheVt{adapter_get, adapter_put, adapter_list, adapter_delete, adapter_head,
                            adapter_create, adapter_range, adapter_version, adapter_conditional};
}  // namespace

int object_range_cache_init(ObjectRangeCache** out, ObjectStore* origin, Arena* backing,
                            Arena* io, uint32_t pages, ObjectStore* adapter) noexcept {
    assert(out != nullptr && adapter != nullptr);
    assert(origin != nullptr && backing != nullptr && io != nullptr);
    const detail::ObjectSpan input{origin, sizeof(*origin)};
    const detail::ObjectSpan outputs[]{{out, sizeof(*out)}, {adapter, sizeof(*adapter)}};
    const detail::ObjectSpan state{backing, sizeof(Arena)};
    if (!detail::object_io_admit(&input, 1, outputs, 2, io, &state, 1)) return kOsBadArg;
    const ObjectStore source = *origin;
    const detail::ObjectSpan protected_spans[]{state,
        {source.vt, source.vt == nullptr ? 0u : sizeof(ObjectStoreVT)}};
    if (!detail::object_io_admit(&input, 1, outputs, 2, io, protected_spans, 2)) return kOsBadArg;
    *out = nullptr; *adapter = {};
    if (source.vt == nullptr || backing == io || pages > kObjectRangeMaxPages || io->total_allocated() != 0 ||
        io->total_reserved() != kObjectRangePageBytes) return kOsBadArg;
    if (io->overlaps_backing(source.vt, sizeof(ObjectStoreVT)) ||
        (source.impl != nullptr && io->overlaps_backing(source.impl, 1))) return kOsBadArg;
    if (source.vt->head_versioned == nullptr || source.vt->get_range_if_match == nullptr) return kOsNotImplemented;
    io->set_growth_enabled(false);
    if (io->allocate(kObjectRangePageBytes, 1) == nullptr) return kOsBufferSmall;
    io->reset();
    void* memory = backing->allocate(sizeof(ObjectRangeCache), alignof(ObjectRangeCache));
    if (memory == nullptr) return kOsBufferSmall;
    auto* c = new (memory) ObjectRangeCache{};
    c->origin = source; c->io = io; c->backing = backing; c->count = pages; c->owner = std::this_thread::get_id();
    if (pages != 0) {
        c->pages = backing->allocate_array<RangePage>(pages);
        auto* bytes = static_cast<uint8_t*>(backing->allocate(static_cast<size_t>(pages) * kObjectRangePageBytes, 64));
        c->page_bytes = bytes;
        if (c->pages == nullptr || bytes == nullptr) return kOsBufferSmall;
        for (uint32_t i = 0; i < pages; ++i) {
            new (&c->pages[i]) RangePage{};
            c->pages[i].bytes = bytes + static_cast<size_t>(i) * kObjectRangePageBytes;
        }
    }
    *adapter = ObjectStore{&kCacheVt, c}; *out = c;
    assert(c->count <= kObjectRangeMaxPages);
    return kOsOk;
}
int object_range_cache_head(ObjectRangeCache* c, const char* key, ObjectVersion* out) noexcept {
    confined(c);
    assert(out != nullptr);
    detail::ObjectSpan key_span{};
    const detail::ObjectSpan output{out, sizeof(*out)};
    if (!detail::object_key_span(key, kOsMaxKey, &key_span) ||
        !admit(c, &key_span, 1, &output, 1)) return kOsBadArg;
    char stable_key[kOsMaxKey]{};
    std::memcpy(stable_key, key, key_span.size);
    *out = {};
    if (stable_key[0] == 0) return kOsBadArg;
    ObjectVersion version{};
    const int rc = read_head(c, stable_key, &version);
    *out = version;
    return rc == kOsOk ? rc : read_error(c, stable_key, rc);
}
int object_range_cache_read(ObjectRangeCache* c, const char* key, const ObjectVersion* expected,
                            uint64_t offset, uint64_t length, uint8_t* dst, uint64_t* out) noexcept {
    return read_impl(c, key, expected, offset, length, dst, out, false);
}
void object_range_cache_stats(const ObjectRangeCache* c, ObjectRangeCacheStats* out) noexcept {
    confined(c);
    assert(out != nullptr);
    const detail::ObjectSpan output{out, sizeof(*out)};
    if (!admit(const_cast<ObjectRangeCache*>(c), nullptr, 0, &output, 1)) return;
    *out = c->stats;
}
}  // namespace bolt::lakehouse
