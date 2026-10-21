// Exact immutable metadata-object load; no catalog/path discovery or mutation.
#pragma once

#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt::lakehouse::iceberg {
inline constexpr uint32_t kMetadataLoadMaxRequestBytes = 1u << 20;

struct MetadataObjectIdentity {
    // Canonical root of the SUPPLIED ObjectStore's relative key namespace.
    // Independent from table_location: metadata can live outside the table.
    char store_location[kIcebergMaxLocation]{};
    char table_location[kIcebergMaxLocation]{};
    char metadata_uri[kIcebergMaxLocation]{};
    char table_uuid[kIcebergMaxUuid]{}; // required catalog-pinned expected UUID
    char object_key[kOsMaxKey]{}; // raw relative key, never URI-decoded again
    ObjectVersion version{};    // caller-pinned strong validator and exact extent
};
struct MetadataLoadLimits {
    uint64_t max_bytes = 0; // explicit nonzero metadata JSON budget
    uint32_t request_bytes = kMetadataLoadMaxRequestBytes;
};
struct MetadataLoadResult {
    MetadataObjectIdentity object{}; // copied, no arena/transport spans
    char sha256[65]{};               // complete exact JSON bytes, lowercase hex
};
enum class MetadataLoadStatus : uint8_t {
    Ok, BadArg, NotFound, Changed, IoError, Capacity, DecodeFailed, Unsupported,
    IdentityMismatch
};

// S3 canonical URI subset only: s3://bucket[/canonical-encoded path], no
// trailing slash/query/fragment/dot/empty components. Other schemes explicitly
// return Unsupported. Construct metadata_uri as store_location + '/' + the
// maintained S3 percent-encoding of raw object_key; no rebasing/normalization.
// Parsed metadata.location/UUID must equal table_location/table_uuid exactly;
// otherwise IdentityMismatch, without publishing either output.
// The existing metadata parser/SAX expose raw JSON identity strings/keys:
// ALL escaped top-level keys and escapes in selected location/UUID values are
// refused with DecodeFailed, not normalized or silently truncated. Escapes in
// other values or nested keys remain subject to the existing metadata parser.
// No HEAD/LIST/hint/probe, repin, retry, alternate URI, or fallback operation.
// Requires get_range_if_match; every request is <=request_bytes and <=1MiB.
// Input size must also fit the existing metadata parser (<1GiB) before casts.
//
// Arenas are distinct, exclusively owned, frozen (growth disabled) before the
// call and must outlive returned Metadata. They supply explicit finite backing;
// metadata_arena retains exact bytes, parser scratch and all decoded arrays.
// io_arena is reset before EVERY provider request and must not back retained
// metadata. Caller allocates arena backing at startup; exhaustion may
// consume scratch. Direct byte-budget/allocation refusal returns Capacity;
// DecodeFailed includes refusal within the existing bool metadata parser.
// On ANY failure both outputs stay unchanged. Invalid overlapping descriptors,
// arena objects/backing, known store/VT metadata are refused before I/O/writes.
// The opaque provider implementation's full private extent is unknown: caller
// must exclude it from both arenas, writable outputs and input descriptors and
// keep provider/VT immutable for this synchronous call. No ownership transfer.
MetadataLoadStatus metadata_load_exact(ObjectStore* store,
    const MetadataObjectIdentity* identity, const MetadataLoadLimits* limits,
    Arena* metadata_arena, Arena* io_arena, Metadata* out,
    MetadataLoadResult* result) noexcept;

// Validate an exact locally retained copy through the SAME URI, original JSON
// identity, parser and complete digest contract. No store access or discovery.
// The caller establishes that these bytes belong to the supplied version (for
// example, a durable binding with a full SHA); this function does not verify an
// ETag against an origin. It copies the bytes into the exclusively owned frozen
// metadata arena. Extent must equal identity.version.size and fit max_bytes.
// Inputs, arena object/backing and both writable outputs must be disjoint;
// inputs may alias each other. Failure leaves both outputs unchanged, while
// admitted scratch allocations may remain consumed. Parser allocation refusal
// can still be DecodeFailed, as for metadata_load_exact.
MetadataLoadStatus metadata_validate_exact(const MetadataObjectIdentity* identity,
    const MetadataLoadLimits* limits, const uint8_t* bytes, uint64_t length,
    Arena* metadata_arena, Metadata* out, MetadataLoadResult* result) noexcept;
} // namespace bolt::lakehouse::iceberg
