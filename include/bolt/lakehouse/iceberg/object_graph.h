// Admit one complete immutable Iceberg snapshot object graph.
#pragma once

#include "bolt/lakehouse/format.h"
#include "bolt/lakehouse/iceberg/metadata_io.h"
#include "bolt/lakehouse/object_store/pinned.h"

namespace bolt::lakehouse::iceberg {
struct GraphAdmissionRequest {
    MetadataObjectIdentity object{};
    MetadataLoadLimits metadata_limits{};
    const uint8_t* metadata_json = nullptr;
    uint64_t metadata_length = 0;
    char sha256[65]{};
    SnapshotSelection selection = SnapshotSelection::Empty;
    int64_t snapshot_id = 0;
};
struct GraphAdmissionLimits {
    uint32_t max_objects = 0; // 1..65536, including retained metadata
    uint32_t max_manifests = 0; // 1..65535, manifest-list visits
    uint32_t max_entries = 0; // positive total manifest-entry visits
    uint64_t max_metadata_object_bytes = 0; // positive, <=1GiB
    uint64_t max_total_metadata_bytes = 0; // retained JSON + unique list/manifests
    uint32_t request_bytes = 0; // 1..1MiB, conditional metadata requests
};
struct GraphAdmissionWork {
    Arena* metadata = nullptr;
    Arena* list = nullptr;
    Arena* manifest = nullptr;
    Arena* io = nullptr;
    Arena* index = nullptr;
    PinnedObject* objects = nullptr;
    uint32_t objects_capacity = 0;
};
struct GraphAdmissionResult {
    MetadataLoadResult metadata{}; // copied identity and complete retained SHA
    SnapshotSelection selection = SnapshotSelection::Empty;
    int64_t snapshot_id = 0;
    const PinnedObject* objects = nullptr; // borrowed from work.objects
    uint32_t object_count = 0;
    uint64_t total_metadata_bytes = 0;
};
enum class GraphAdmissionStatus : uint8_t {
    Ok, BadArg, NotFound, Changed, IoError, Capacity, DecodeFailed, Unsupported, Conflict
};

// No metadata origin access/discovery. Validates exact retained JSON, identity,
// extent and full lowercase SHA, then selects literal signed Exact ID, or
// Empty (ID zero and absent current snapshot). Legacy and ambiguous IDs refuse.
// Pins the metadata object and every selected list/manifest/live data/position
// delete/equality delete. Deleted entries never cause file IO. Every other pin
// uses one strong HEAD_VERSIONED; metadata bodies use bounded If-Match ranges.
// No data/delete bodies, whole GET, LIST, retry, repin, mutation or relocation.
// References map once against object.store_location, including outside-table
// objects within that root. Roles and declared exact extents must agree.
//
// All arenas are distinct, exclusively owned, nonconcurrent and frozen with
// finite startup backing. Descriptor objects, backing, request/limits/store/VT,
// retained bytes, objects and result must be disjoint (read-only inputs may
// alias each other). Caller excludes the opaque origin private extent, whose
// size is unknown. Admission checks raw Arena-object aliases before inspecting
// proposed Arena methods. Invalid aliases do not reset/write or perform IO.
// On admitted calls all five arenas reset initially; manifest resets between
// manifests, IO resets for each provider request. Metadata/list/index scratch
// may remain consumed. Existing arena contents must therefore be retired.
//
// Failure leaves *result unchanged; admitted scratch/directory may change and
// cannot back an older live result. Success publishes strictly sorted unique
// copied key/version entries usable with pinned_object_store_init. The caller
// keeps the borrowed directory immutable and alive until its consumers retire.
// Result metadata identity/digest are copied, own no scratch/provider pointers,
// and survive all arena resets; the directory remains caller owned. No durable
// publication or catalog attachment occurs here. Parser-internal allocation
// refusals can be DecodeFailed where existing bool parsers cannot distinguish.
GraphAdmissionStatus iceberg_admit_object_graph(ObjectStore* store,
    const GraphAdmissionRequest* request, const GraphAdmissionLimits* limits,
    GraphAdmissionWork* work, GraphAdmissionResult* result) noexcept;
} // namespace bolt::lakehouse::iceberg
