// bolt/lakehouse/delta/column_mapping.h — physical/logical column map.

#pragma once

#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/delta/log.h"
#include "bolt/lakehouse/format.h"

namespace bolt {
namespace lakehouse {
namespace delta {

static constexpr uint32_t kDeltaMaxMapCols = 64u;

struct ColumnMapEntry {
    char    logical[kLakeMaxColName];
    char    physical[kLakeMaxColName];
    int64_t field_id;
};

struct ColumnMap {
    ColumnMappingMode mode;
    uint8_t           _pad[7];
    ColumnMapEntry    entries[kDeltaMaxMapCols];
    uint32_t          n_entries;
    uint32_t          _pad2;
};

bool delta_column_map_build(const DeltaMetadata* md, Arena* scratch,
                            ColumnMap* out) noexcept;

const char* delta_column_map_logical(const ColumnMap* m,
                                     const char* physical) noexcept;
const char* delta_column_map_physical(const ColumnMap* m,
                                      const char* logical) noexcept;

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
