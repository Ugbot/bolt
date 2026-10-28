// bolt/ingest/bolt_parquet_variant.h — read a Parquet VARIANT column, shredded or not (G2ICE-310).
//
// A VARIANT group is { required binary metadata; optional binary value; optional <T> typed_value } (Parquet
// VariantEncoding.md / VariantShredding.md). An unshredded writer emits {metadata, value}; a shredding writer (Iceberg-Java
// with write.parquet.shred-variants) moves part or all of each value into typed_value:
//   - a primitive typed_value (INT8..64, FLOAT, DOUBLE, DECIMAL, DATE, TIME(MICROS), TIMESTAMP(MICROS | NANOS), STRING,
//     BINARY, BOOLEAN, UUID) holds the whole value; `value` is null;
//   - an object typed_value is a group of field groups {value, typed_value}, one per shredded key; the residual `value`
//     (when present) is an object of the keys that were not shredded; a field with both null is absent;
//   - an array typed_value is a LIST of element groups {value, typed_value}.
// The reader rebuilds every row's value in the variant binary encoding: objects list their fields in key order with the
// field ids of the row's metadata dictionary, short strings below 64 bytes, decimal width by precision (4 / 8 / 16 bytes).
// The rebuilt bytes are a valid encoding of the same variant, not necessarily the writer's original bytes.
#pragma once

#include <cstdint>

#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_nested.h"

namespace bolt {

class Arena;
struct BoltColumn;

namespace ingest {
namespace parquet {

// Bytes one rebuilt variant column may hold (every row's value); past it the read fails (a budget, not a silent cut).
inline constexpr uint64_t kPqVariantValueBytes = uint64_t{1} << 30;

// Is field `f` (an index into tree->fields) a VARIANT group this reader takes: a struct whose first child is a binary
// leaf `metadata` and whose other children are a binary leaf `value` and / or a `typed_value` field.
bool pq_variant_group_ok(const PqSchemaTree* tree, int32_t f) noexcept;

// Is that group shredded (has a typed_value child)?
bool pq_variant_group_shredded(const PqSchemaTree* tree, int32_t f) noexcept;

// Top-level field `top_field` of `row_group` as BoltColumn::make_variant({metadata, value}) with the group's nullability,
// typed_value folded back into `value`. False on a shape outside the spec, a typed_value type the variant encoding has no
// primitive for, a residual value that is not an object beside an object typed_value, a key absent from the row's
// metadata dictionary, or more than kPqVariantValueBytes of values.
bool parquet_read_variant_column(const uint8_t* buf, uint64_t len, const PqMeta* meta, const PqSchemaTree* tree,
                                 uint32_t row_group, uint32_t top_field, Arena* arena, BoltColumn* out,
                                 int64_t* out_rows) noexcept;

// parquet_read_nested_column_tree with every VARIANT-annotated group inside the field (a struct field, a list element, a map value
// at any depth) read as above: each such group's column is make_variant({metadata, value}). A field with no variant group reads as
// parquet_read_nested_column_tree does.
bool parquet_read_nested_column_variants(const uint8_t* buf, uint64_t len, const PqMeta* meta, const PqSchemaTree* tree,
                                         uint32_t row_group, uint32_t top_field, Arena* arena, BoltColumn* out,
                                         int64_t* out_rows) noexcept;

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
