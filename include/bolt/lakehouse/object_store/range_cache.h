// Thread-confined, validator-bound object ranges; explicit startup ownership.
#pragma once
#include "bolt/lakehouse/object_store.h"

namespace bolt::lakehouse {
inline constexpr uint32_t kObjectRangePageBytes = 1u << 20;
inline constexpr uint32_t kObjectRangeMaxPages = 64;
inline constexpr uint64_t kObjectRangeMaxReadBytes = uint64_t{kObjectRangePageBytes} * kObjectRangeMaxPages;
struct ObjectRangeCache;
struct ObjectRangeCacheStats {
    uint64_t reads = 0, head_calls = 0, range_calls = 0;
    uint64_t requested_bytes = 0, origin_bytes = 0;
    uint64_t page_hits = 0, page_misses = 0, fills = 0, evictions = 0;
    uint64_t invalidated_pages = 0, preconditions = 0, failures = 0;
    uint64_t write_attempts = 0, write_attempted_bytes = 0, write_ack_bytes = 0;
    uint64_t write_uncertain = 0, write_failures = 0, delete_attempts = 0;
};
// Startup only. Both arenas are exclusive and borrowed until the adapter dies.
// backing holds the opaque handle and 0..64 MiB page storage; never reset it.
// io is separate, empty and constructed with exactly 1 MiB reserved backing; init
// freezes it. Only this module resets it; input/output spans must not alias it.
// origin and its immutable implementation/config must outlive this instance.
// All calls, including adapter VT operations, occur on the init thread.
// No retries. Maintained HTTP retains its separately bounded wire allocation.
int object_range_cache_init(ObjectRangeCache** out, ObjectStore* origin,
                            Arena* backing, Arena* io, uint32_t pages,
                            ObjectStore* adapter) noexcept;
int object_range_cache_head(ObjectRangeCache* cache, const char* key,
                            ObjectVersion* out) noexcept;
// Fresh HEAD per logical read. Pinned expected descriptor cannot silently change.
// Exact success <=64 MiB; at most 65 intersected pages. On failure *out_len=0,
// destination may contain partial bytes and caller MUST ignore them. Invalid aliases
// return BadArg before I/O without modifying outputs. Writable spans must be
// disjoint from key/expected/each other and all owned cache/I/O storage.
int object_range_cache_read(ObjectRangeCache* cache, const char* key,
                            const ObjectVersion* expected, uint64_t offset,
                            uint64_t length, uint8_t* dst, uint64_t* out_len) noexcept;
// Invalid output aliases leave the statistics destination unchanged.
void object_range_cache_stats(const ObjectRangeCache* cache,
                              ObjectRangeCacheStats* out) noexcept;
}  // namespace bolt::lakehouse
