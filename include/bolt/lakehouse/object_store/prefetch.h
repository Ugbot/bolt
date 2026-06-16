// bolt/lakehouse/object_store/prefetch.h — async object prefetcher.
//
// W7. One background worker thread reads requests off an SPSC channel, calls
// `os_get` against a caller-provided `ObjectStore*`, and pushes results
// (slot_id + arena-backed buffer) onto an SPSC result channel. The producer
// thread submits and polls; nothing is allocated on the hot path beyond the
// fetched buffer.
//
// Notes on shape vs the spec:
//   - The ObjectStore vtable in W1 has no `read_range`; only `get_object`.
//     We pass `range_lo`/`range_hi` through the PrefetchRequest for future
//     use (when the vtable grows), but today we fetch the full object and
//     slice in the producer if needed. This keeps the W7 surface stable.
//   - Default depth 16, max 64 — bounded.
//
// Tiger Style: PODs in/out, bounded everything, no heap on the hot path
// (the worker thread itself is the one allowed std::thread).

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt {
namespace lakehouse {
namespace prefetch {

static constexpr uint32_t kPrefetchDefaultDepth = 16u;
static constexpr uint32_t kPrefetchMaxDepth     = 64u;
static constexpr uint32_t kPrefetchUriCap       = 1024u;

struct PrefetchRequest {
    char     uri[kPrefetchUriCap];     // object key (relative to the store)
    int64_t  range_lo;                 // -1 = whole object
    int64_t  range_hi;                 // -1 = whole object
    uint32_t slot_id;                  // caller-supplied tag, echoed in result
    uint8_t  _pad[4];
};

struct PrefetchResult {
    uint8_t*  buf;       // arena-allocated; lifetime = arena
    uint32_t  len;
    uint32_t  slot_id;
    int32_t   status;    // kOs* code, kOsOk on success
    bool      ok;
    uint8_t   _pad[3];
};

struct Prefetcher;     // opaque

// Open. Returns false on bad arg or oversize depth. `arena` is used for
// both the Prefetcher struct AND the fetched buffers — sized for the
// caller's expected working set.
bool prefetcher_open(Prefetcher** out, Arena* arena, ObjectStore* store,
                     uint32_t depth) noexcept;

// Submit (non-blocking). Returns false if the request ring is full.
bool prefetcher_submit(Prefetcher* p, const PrefetchRequest* req) noexcept;

// Poll for one result, waiting up to `timeout_ms`. Returns true if a result
// was written into *out; false on timeout / empty.
bool prefetcher_poll(Prefetcher* p, PrefetchResult* out,
                     uint32_t timeout_ms) noexcept;

// Close + join the worker. Safe to call on null.
void prefetcher_close(Prefetcher* p) noexcept;

}  // namespace prefetch
}  // namespace lakehouse
}  // namespace bolt
