// Internal to the parquet reader TUs: one SchemaElement as the footer
// parser reads it, shared with the schema-tree builder so both decode the
// element identically.
#pragma once

#include <cstdint>

#include "bolt/ingest/bolt_parquet_meta.h"

namespace bolt {
namespace ingest {
namespace parquet {

struct PqRawElem {
    PqColumn col;          // name, types, field id, optional flag
    int32_t  num_children; // > 0 => group
    uint8_t  has_type;
    uint8_t  repeated;
};

bool pq_parse_schema_elem(TcCursor* c, PqRawElem* out) noexcept;

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
