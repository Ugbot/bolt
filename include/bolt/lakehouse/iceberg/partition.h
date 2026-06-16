// bolt/lakehouse/iceberg/partition.h — Iceberg partition spec POD.
//
// One PartitionSpec per spec_id; metadata.json typically carries the current
// spec + historical specs (Iceberg supports spec evolution). Tiger Style.

#pragma once

#include <cstdint>

#include "bolt/lakehouse/format.h"
#include "bolt/lakehouse/iceberg/transform.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

static constexpr uint32_t kIcebergMaxFieldName = 64u;
static constexpr uint32_t kIcebergMaxFieldsPerSpec = 32u;
static constexpr uint32_t kIcebergMaxSpecs = 16u;

struct PartitionField {
    int32_t   source_id;     // refers to schema field id
    int32_t   field_id;       // partition-side field id (>=1000 in v2)
    Transform transform;
    char      name[kIcebergMaxFieldName];
};

struct PartitionSpec {
    int32_t        spec_id;
    uint32_t       n_fields;
    PartitionField fields[kIcebergMaxFieldsPerSpec];
};

struct DataFileRef;
struct Schema;

// Partition-equality pruning. Conservative: returns true on any predicate
// we can't evaluate. Identity transform only at W4.
bool partition_passes(const DataFileRef* f, const PartitionSpec* spec,
                      const Schema* schema,
                      const ::bolt::lakehouse::Predicate* preds,
                      uint32_t n_preds) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
