// A table-relative view of an existing object store. No transport or allocation.
#pragma once

#include "bolt/lakehouse/object_store.h"

namespace bolt::lakehouse {

// Initialise at the final address; do not copy/move an active wrapper. The
// origin handle is copied; its backend and vtable remain borrowed and immutable.
// They and this state must outlive every use of the returned ObjectStore.
// Concurrent use has the same guarantees as the origin; this wrapper is immutable.
// Nested prefix wrappers are flattened at init, avoiding recursive dispatch.
struct PrefixObjectStore {
    ObjectStore origin{};
    ObjectStoreVT vtable{};
    char prefix[kOsMaxKey]{};
    uint32_t prefix_length = 0;
};

// Raw relative keys, not URIs: reject leading slash, backslash, control bytes,
// empty interior components and . or .. components. An optional trailing slash
// on prefix is accepted. Empty prefix is refused. All joined keys must fit
// kOsMaxKey including NUL; no truncation or percent decoding is performed.
// The empty key addresses the exact configured prefix object (an S3 root
// directory marker); it is not the empty origin key. This keeps LIST complete.
// LIST accepts an empty relative prefix and strips the configured root from
// returned keys. It preserves kOsBufferSmall/count semantics. A malformed or
// out-of-root origin listing fails with zero returned count.
// Optional operations remain optional; conditional creation and uncertainty
// are forwarded without emulation/retry. Failure leaves state and out unchanged.
bool prefix_object_store_init(PrefixObjectStore* state, const ObjectStore* origin,
                              const char* prefix, ObjectStore* out) noexcept;

}  // namespace bolt::lakehouse
