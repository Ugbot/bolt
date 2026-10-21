// Exact canonical S3 namespace mapping. No normalization, rebasing or I/O.
#pragma once
#include "bolt/lakehouse/object_store.h"

namespace bolt::lakehouse {
enum class S3UriStatus : uint8_t { Ok, BadArg, Unsupported, OutsideRoot, Capacity };

// Bounded by kOsMaxKey, including the terminator. The canonical subset accepts
// s3://bucket[/encoded/path], uppercase escapes for reserved/non-ASCII bytes,
// no query/fragment, trailing slash, empty/dot components or escaped slash.
// Unreserved bytes must be literal. A bucket-only namespace root is allowed.
S3UriStatus s3_uri_validate(const char* uri) noexcept;

// Map one canonical absolute URI under the exact configured namespace root to
// its raw relative object key. Decodes exactly once; never strips another
// authority or probes a table-relative fallback. A URI equal to root has no
// object key and is refused. Root paths must end at a component boundary.
S3UriStatus s3_uri_object_key(const char* root, const char* uri,
                            char* out, uint32_t capacity) noexcept;

// The inverse: exact canonical root + '/' + maintained S3 encoding of raw key.
// Raw keys are nonempty, relative, with no empty/dot components, backslash or
// control bytes. No percent decoding occurs here; a raw '%' becomes "%25".
// Both mapping functions leave out unchanged on failure, including aliases.
// Output spans (capacity 1..kOsMaxKey) must be disjoint from both input strings.
S3UriStatus s3_uri_for_object_key(const char* root, const char* key,
                                char* out, uint32_t capacity) noexcept;
} // namespace bolt::lakehouse
