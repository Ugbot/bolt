// Exact canonical S3 namespace mapping. No normalization, rebasing or I/O.
#pragma once
#include "bolt/lakehouse/object_store.h"

namespace bolt::lakehouse {
enum class S3UriStatus : uint8_t { Ok, BadArg, Unsupported, OutsideRoot, Capacity };

// Bounded by kOsMaxKey, including the terminator. The canonical subset accepts
// s3://bucket[/path] whose path IS the object key byte for byte (G2ICE-342: as
// Iceberg-Java's S3FileIO and pyiceberg read it; '%' is a literal byte, never an
// escape): no control bytes, backslash, '?' or '#', trailing slash or empty/dot
// components. A bucket-only namespace root is allowed.
S3UriStatus s3_uri_validate(const char* uri) noexcept;

// Map one canonical absolute URI under the exact configured namespace root to
// its relative object key, the path's bytes; never strips another
// authority or probes a table-relative fallback. A URI equal to root has no
// object key and is refused. Root paths must end at a component boundary.
// G2ICE-300: the URI may spell its scheme s3a:// or s3n:// (Hadoop's names for the same namespace); the root may not.
S3UriStatus s3_uri_object_key(const char* root, const char* uri,
                            char* out, uint32_t capacity) noexcept;

// The inverse: exact canonical root + '/' + the key, literally. Keys are nonempty,
// relative, with no empty/dot components, backslash, '?', '#' or control bytes.
// Both mapping functions leave out unchanged on failure, including aliases.
// Output spans (capacity 1..kOsMaxKey) must be disjoint from both input strings.
S3UriStatus s3_uri_for_object_key(const char* root, const char* key,
                                char* out, uint32_t capacity) noexcept;
} // namespace bolt::lakehouse
