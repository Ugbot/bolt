// bolt/lakehouse/iceberg/metadata.h — table-level metadata.json POD.

#pragma once

#include <cstdint>

#include "bolt/lakehouse/iceberg/partition.h"
#include "bolt/lakehouse/iceberg/snapshot.h"
#include "bolt/lakehouse/iceberg/sort_order.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

static constexpr uint32_t kIcebergMaxSchemas    = 8u;
static constexpr uint32_t kIcebergMaxFieldsPerSchema = 256u;
static constexpr uint32_t kIcebergMaxLocation   = 1024u;
static constexpr uint32_t kIcebergMaxUuid       = 64u;
static constexpr uint32_t kIcebergMaxTypeName   = 32u;

struct SchemaField {
    int32_t  id;
    bool     required;
    uint8_t  _pad[3];
    char     name[kIcebergMaxFieldName];
    char     type[kIcebergMaxTypeName];
};

struct Schema {
    int32_t      schema_id;
    uint32_t     n_fields;
    SchemaField  fields[kIcebergMaxFieldsPerSchema];
};

struct Metadata {
    int32_t   format_version;
    int32_t   current_schema_id;
    int32_t   current_spec_id;
    int32_t   default_sort_order_id;
    int64_t   last_sequence_number;
    int64_t   last_updated_ms;
    int64_t   current_snapshot_id;
    char      table_uuid[kIcebergMaxUuid];
    char      location[kIcebergMaxLocation];

    uint32_t      n_snapshots;
    uint32_t      _pad;
    Snapshot      snapshots[kIcebergMaxSnapshots];

    uint32_t      n_schemas;
    uint32_t      _pad2;
    Schema        schemas[kIcebergMaxSchemas];

    uint32_t      n_specs;
    uint32_t      _pad3;
    PartitionSpec specs[kIcebergMaxSpecs];

    uint32_t      n_sort_orders;
    uint32_t      _pad4;
    SortOrder     sort_orders[kIcebergMaxSortOrders];
};

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt

namespace bolt { class Arena; }

namespace bolt {
namespace lakehouse {
namespace iceberg {

bool metadata_parse(const uint8_t* src, uint32_t len, Arena* scratch,
                    Metadata* out) noexcept;
const Schema* metadata_current_schema(const Metadata* m) noexcept;
const PartitionSpec* metadata_spec(const Metadata* m, int32_t spec_id) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
