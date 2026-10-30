#pragma once
#include "bolt/lakehouse/iceberg/json_value.h"

namespace bolt::lakehouse::iceberg {
enum class DefaultValueMode : uint8_t { Initial, Write };
// Materialize an effective schema default. A stored struct default is null or
// {}; a present struct obtains missing children from the selected child defaults.
// Supplied nested values in collection defaults retain explicit values/nulls.
// Prohibited non-null unknown/variant/geo defaults refuse. Output is an owned
// one-row column and publishes only on success; scratch may be consumed.
JsonValueStatus default_value_decode(const SchemaField&,DefaultValueMode,Arena*,BoltColumn*,const JsonValueLimits& = {}) noexcept;
// Validate declarations, including both modes and nested field defaults. A null
// declaration means unset, so it may occur on required fields; consuming that
// default for an omitted required value still refuses. Schema evolution must
// separately require non-null initial/write defaults for new required fields.
JsonValueStatus schema_defaults_validate(const Schema&,Arena*,const JsonValueLimits& = {}) noexcept;
// Rebind only explicitly stored JSON field-ID keys to an otherwise identical
// destination shape. Missing children stay missing, so later child-default
// changes remain effective. Does not bake resolved defaults into metadata.
JsonValueStatus default_value_rebind(const SchemaField& source,const SchemaField& destination,
    DefaultValueMode,Arena*,SchemaDefault* out,const JsonValueLimits& = {}) noexcept;
} // namespace bolt::lakehouse::iceberg
