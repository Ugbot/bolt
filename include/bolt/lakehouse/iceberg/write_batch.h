#pragma once
#include "bolt/lakehouse/iceberg/json_value.h"

namespace bolt::lakehouse::iceberg {
// Bind named top-level input columns to the current schema. Omitted fields use
// write-default, otherwise null; explicit values/nulls never use a default.
// Missing required fields and unknown/duplicate input names refuse. Recursive
// defaults use the owned Appendix D codec. Supplied column buffers are borrowed
// and must remain alive through commit; new descriptors/defaults are arena-owned.
// This is default application over trusted typed columns, not raw buffer admission.
// Unknown values must be null; physical file projection omits them. Empty
// structs and schemas/groups with no representable Parquet leaf refuse as
// Unsupported. Reader boolean integer/byte lanes normalize to validated Bool.
// Failure preserves *out; scratch may be consumed.
JsonValueStatus write_batch_conform(const Schema&,const BoltBatch&,Arena*,BoltBatch** out) noexcept;
} // namespace bolt::lakehouse::iceberg
