// A finite, caller-admitted directory of immutable object versions.
#pragma once

#include <thread>
#include "bolt/lakehouse/object_store.h"

namespace bolt::lakehouse {

inline constexpr uint32_t kOsPinnedMaxEntries = 65536u;
inline constexpr uint64_t kOsPinnedMaxGetBytes = uint64_t{1} << 30;
inline constexpr uint32_t kOsPinnedMaxRequestBytes = 1u << 20;

struct PinnedObject {
    char key[kOsMaxKey];
    ObjectVersion version;
};

struct PinnedObjectStoreLimits {
    uint32_t max_entries;
    uint64_t max_get_bytes;
    uint32_t request_bytes;
};

// Startup-owned, noncopyable after successful init. Keep every field and the
// borrowed directory immutable until all views retire. Each worker owns a
// separate state and exclusive IO arena; immutable directories may be shared.
// The namespace is an opaque logical identity, not an origin configuration.
struct PinnedObjectStore {
    ObjectStore origin;
    const PinnedObject* entries;
    uint32_t count;
    char namespace_uri[kOsMaxKey];
    PinnedObjectStoreLimits limits;
    Arena* io;
    std::thread::id owner_thread;
    bool initialized;
};

// No discovery, allocation of backing, or origin IO. Entries must be strictly
// sorted by canonical raw relative key and carry copied strong versions. The
// frozen, nonconcurrent IO arena must be empty and exclusively borrowed. After
// all validation, startup probes request_bytes and resets IO; an admitted
// BufferSmall failure may change IO bookkeeping, never state/out. All other
// rejected arguments leave state/out and IO unchanged. The opaque origin impl
// extent is caller-excluded: the adapter cannot inspect arbitrary backend state.
int pinned_object_store_init(PinnedObjectStore* state, const ObjectStore* origin,
                             const char* namespace_uri, const PinnedObject* entries,
                             uint32_t count, const PinnedObjectStoreLimits* limits,
                             Arena* io, ObjectStore* out) noexcept;

// Return another borrowed view. No copying, ownership transfer, allocation, or IO.
int pinned_object_store_view(PinnedObjectStore* state, ObjectStore* out) noexcept;

// Reads use only admitted versions and origin.get_range_if_match. HEAD is local.
// Full GET allocates at most max_get_bytes in the caller arena, using exact
// chunks <= request_bytes; only successful completion publishes data/length.
// Conditional GET refuses a differing supplied size/tag before origin IO.
// Missing keys return NotFound. Positive ranges must be wholly in the pinned
// extent and no larger than request_bytes. Short origin success is IoError.
// Invalid writable aliases preserve outputs and arenas; admitted ranged failure
// sets output length to zero and may leave dst partially written. Full GET may
// consume caller arena space on admitted failure. Only the separate state.io
// arena is reset, once per conditional request. Passed scratch is alias-checked
// but never reset by this adapter. LIST, PUT, DELETE and create are unsupported.

}  // namespace bolt::lakehouse
