#pragma once
// Appendix D JSON is a wire boundary. Materialized values are owned Bolt columns.
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/bolt_column.h"

namespace bolt::lakehouse::iceberg {
enum class JsonValueStatus : uint8_t { Ok, Invalid, Unsupported, Capacity };
struct JsonValueLimits {
    uint32_t json_bytes = limits::kDefaultJsonMax;
    uint32_t encoded_bytes = 16u << 20;
    uint32_t values = 1u << 20;
};
// Decode one value to a one-row column. Children, validity, offsets and payloads
// are arena-owned, independent of source JSON/schema buffers. Requiredness,
// IDs, exact decimal/time semantics and duplicate keys are checked. Failure
// preserves *out; scratch may be consumed. Finite caller arenas bound allocation.
JsonValueStatus json_value_decode(const SchemaField&,const uint8_t* json,uint32_t size,
    Arena*,BoltColumn* out,const JsonValueLimits& = {}) noexcept;
// Geometry/geography values use WKT JSON strings and owned ISO WKB Binary
// columns. Schema-default rules separately prohibit nonnull geo defaults.
// Encode one typed row to arena-owned JSON (field-ID struct keys; parallel map
// keys/values arrays). No Arrow runtime or named-field result JSON is involved.
// Outputs publish together only on success; caller retains the source column.
JsonValueStatus json_value_encode(const SchemaField&,const BoltColumn&,int64_t row,
    Arena*,const uint8_t** json,uint32_t* size,const JsonValueLimits& = {}) noexcept;
} // namespace bolt::lakehouse::iceberg
