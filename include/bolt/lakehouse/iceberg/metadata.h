// bolt/lakehouse/iceberg/metadata.h — table-level metadata.json POD.

#pragma once

#include <cstdint>

#include "bolt/lakehouse/iceberg/partition.h"
#include "bolt/lakehouse/iceberg/snapshot.h"
#include "bolt/lakehouse/iceberg/sort_order.h"

namespace bolt { class Arena; class Budget; }

namespace bolt {
namespace lakehouse {
namespace iceberg {

static constexpr uint32_t kIcebergMaxLocation   = 1024u;
static constexpr uint32_t kIcebergMaxUuid       = 64u;
static constexpr uint32_t kIcebergMaxTypeName   = 32u;
// Oldest metadata-log entries are dropped past this, like Java's
// write.metadata.previous-versions-max. Snapshots, schemas, schema fields and
// the snapshot-log have no count cap: they grow in Metadata::arena, charged to
// Metadata::budget, and only table_expire_snapshots shrinks them.
static constexpr uint32_t kIcebergMaxMetadataLog = 64u;

struct SchemaField {
    int32_t  id;
    bool     required;
    uint8_t  _pad[3];
    char     name[kIcebergMaxFieldName];
    char     type[kIcebergMaxTypeName];
};

// `fields` lives in an arena; a Schema copied by value shares it. Use
// schema_copy for an independent copy.
struct Schema {
    int32_t      schema_id;
    uint32_t     n_fields;
    uint32_t     cap_fields;
    uint32_t     _pad;
    SchemaField* fields;
};

// One `snapshot-log` entry: `snapshot_id` became current at `timestamp_ms`.
struct SnapshotLogEntry {
    int64_t timestamp_ms;
    int64_t snapshot_id;
};

// One `metadata-log` entry: a superseded metadata file and its
// last-updated-ms.
struct MetadataLogEntry {
    int64_t timestamp_ms;
    char    metadata_file[kIcebergMaxManifestPath];
};

struct Metadata {
    int32_t   format_version;
    int32_t   current_schema_id;
    int32_t   current_spec_id;
    int32_t   default_sort_order_id;
    int64_t   last_sequence_number;
    int64_t   last_updated_ms;
    int64_t   current_snapshot_id;
    // Retention policy (table properties history.expire.*); 0 = unset.
    int64_t   retention_max_age_ms;
    int32_t   retention_min_snapshots;
    int32_t   _pad0;
    char      table_uuid[kIcebergMaxUuid];
    char      location[kIcebergMaxLocation];

    // Growth arena and (optional) byte budget for the arrays below. Set by
    // metadata_init / metadata_parse; the arena must outlive the Metadata.
    Arena*        arena;
    Budget*       budget;

    uint32_t      n_snapshots;
    uint32_t      cap_snapshots;
    Snapshot*     snapshots;

    uint32_t      n_schemas;
    uint32_t      cap_schemas;
    Schema*       schemas;

    uint32_t      n_specs;
    uint32_t      _pad3;
    PartitionSpec specs[kIcebergMaxSpecs];

    uint32_t      n_sort_orders;
    uint32_t      _pad4;
    SortOrder     sort_orders[kIcebergMaxSortOrders];

    uint32_t         n_snapshot_log;
    uint32_t         cap_snapshot_log;
    SnapshotLogEntry* snapshot_log;
    uint32_t         n_metadata_log;
    uint32_t         _pad5;
    MetadataLogEntry metadata_log[kIcebergMaxMetadataLog];
};

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt

namespace bolt { class Arena; }

namespace bolt {
namespace lakehouse {
namespace iceberg {

// Zero `m` and bind its growth arena/budget (budget may be nullptr).
void metadata_init(Metadata* m, Arena* arena, Budget* budget) noexcept;

// `scratch` becomes the Metadata's growth arena, so it must outlive `out`.
// `budget` (optional) is charged for every array the parse grows; past it the
// parse fails with ResourceExhausted set.
bool metadata_parse(const uint8_t* src, uint32_t len, Arena* scratch,
                    Metadata* out) noexcept;
bool metadata_parse_budget(const uint8_t* src, uint32_t len, Arena* scratch,
                           Budget* budget, Metadata* out) noexcept;
const Schema* metadata_current_schema(const Metadata* m) noexcept;

// Growth. Each returns false (ResourceExhausted set) when the arena or budget
// refuses; the Metadata is unchanged.
bool metadata_push_snapshot(Metadata* m, const Snapshot* s) noexcept;
Schema* metadata_push_schema(Metadata* m) noexcept;   // zeroed, fields empty
bool schema_reserve(Schema* s, Arena* a, Budget* b, uint32_t n) noexcept;
bool schema_push_field(Schema* s, Arena* a, Budget* b,
                       const SchemaField* f) noexcept;
// Deep copy: dst gets its own field array.
bool schema_copy(Schema* dst, const Schema* src, Arena* a, Budget* b) noexcept;

// snapshot-log grows; metadata-log drops its oldest entry when full.
bool metadata_snapshot_log_push(Metadata* m, int64_t timestamp_ms,
                                int64_t snapshot_id) noexcept;
void metadata_metadata_log_push(Metadata* m,
                                const MetadataLogEntry* e) noexcept;

// Serialize a Metadata POD to Iceberg's `metadata.json`, allocated in `a`.
// The inverse of `metadata_parse`. Previously defined but never declared in a
// header, so the only way to reach it was an extern declaration at the call
// site -- which is why a caller wanting to serve metadata over REST (where the
// spec carries it INLINE in LoadTableResult, never as a fetched file) had no
// supported way to produce it.
//
// `last-column-id` is derived from the schemas. Emitting fails rather than
// falling back if a partition transform cannot be represented.
bool metadata_json_emit_plain(const Metadata* m, Arena* a,
                              const uint8_t** out, uint64_t* out_len) noexcept;

// Derive the RFC 4122 UUID this writer assigns to a table at `loc`. Stable for
// a given location (the same table always gets the same id) and distinct
// between locations. `cap` must be >= 37.
void table_uuid_from_location(const char* loc, char* dst,
                              uint32_t cap) noexcept;
const PartitionSpec* metadata_spec(const Metadata* m, int32_t spec_id) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
