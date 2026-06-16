// bolt/lakehouse/iceberg/sort_order.h — Iceberg sort-order POD (read-side stub).
//
// W4 parses but does not act on sort orders. Carrying the metadata so writers
// (W5) can preserve it round-trip. Tiger Style.

#pragma once

#include <cstdint>

namespace bolt {
namespace lakehouse {
namespace iceberg {

static constexpr uint32_t kIcebergMaxSortFields = 16u;
static constexpr uint32_t kIcebergMaxSortOrders = 8u;

enum class SortDirection : uint8_t { kAsc = 0, kDesc = 1 };
enum class NullOrder     : uint8_t { kFirst = 0, kLast = 1 };

struct SortField {
    int32_t       source_id;
    int32_t       transform_param;
    uint8_t       transform_kind;
    SortDirection direction;
    NullOrder     null_order;
    uint8_t       _pad;
};

struct SortOrder {
    int32_t   order_id;
    uint32_t  n_fields;
    SortField fields[kIcebergMaxSortFields];
};

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
