// bolt/lakehouse/iceberg/partition_values.h — typed Iceberg partition values (G2ICE-292).
//
// A partition field's value has the transform's RESULT type over its source column's type (spec: "Partition
// Transforms": identity/truncate/void keep the source type, bucket/year/month/hour give int, day gives date). This
// header is that type model, the per-row computation of a tuple value from a BoltColumn, the value ordering used for
// manifest-list partition summaries, Iceberg's binary single-value encoding of a value (spec: "Binary single-value
// serialization") and the Avro type a manifest declares for it (spec: "Avro" type mapping).

#pragma once

#include <cstdint>

#include "bolt/bolt_column.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/partition.h"
#include "bolt/lakehouse/iceberg/transform.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

enum class IcebergType : uint8_t {
    kUnknown = 0, kBoolean, kInt, kLong, kFloat, kDouble, kDate, kTime, kTimestamp, kTimestampTz,
    kTimestampNs, kTimestampTzNs, kString, kBinary, kUuid, kFixed, kDecimal,
    kGeometry, kGeography,   // G2ICE-327: WKB values; no identity/bucket/truncate/temporal transform, no partition value
};

struct IcebergTypeInfo {
    IcebergType type;
    uint8_t     precision;   // decimal
    uint8_t     scale;       // decimal
    uint8_t     _pad;
    uint32_t    fixed_len;   // fixed[L]; the Avro fixed width of a decimal; 16 for uuid
};

// Parse a primitive Iceberg type name: "int", "decimal(9,2)", "fixed[16]", "timestamptz_ns", ... kUnknown (false)
// for anything else, including nested types.
bool iceberg_type_parse(const char* name, IcebergTypeInfo* out) noexcept;

// Bytes of the Avro fixed holding a decimal of precision P (Java TypeUtil.decimalRequiredBytes); 0 past 38.
uint32_t decimal_required_bytes(uint32_t precision) noexcept;

// The transform's result type over `src`. False when the spec does not define the transform for that source type,
// or the transform is unknown -- a writer must not commit such a spec.
bool partition_result_type(Transform t, const IcebergTypeInfo& src, IcebergTypeInfo* out) noexcept;

// The source and result types of every field of `spec`, resolved against `schema` (a field whose source column is
// not in it, or whose transform is not defined for it, fails). `n` = spec->n_fields entries each.
bool partition_spec_types(const PartitionSpec* spec, const Schema* schema, IcebergTypeInfo* src,
                          IcebergTypeInfo* result) noexcept;

// Field `f`'s value for row `row` of column `c` (a column of source type `src`). Supported column layouts: Flat
// fixed-width (Bool, Int8..Int64, Date32, Timestamp, Float32/64, Decimal64/128, UUID, FixedSizeBinary) and
// Utf8/Binary (StringView or VarBinary). False on a type/column pairing it cannot read, or a result wider than a
// PartitionValue holds (kLakeMaxValBytes - 1 bytes); nothing is guessed.
bool partition_value_of(const PartitionField& f, const IcebergTypeInfo& src, const BoltColumn& c, int64_t row,
                        PartitionValue* out) noexcept;

// Total order of two non-null values of result type `t` (Iceberg's comparators: unsigned bytes for string/binary/
// fixed, numeric for decimal, -0.0 < 0.0 and NaN excluded by the caller for floats). <0, 0, >0. `uuid_signed`
// compares uuids as Java's UUID.compareTo (two signed longs) instead of unsigned bytes.
int partition_value_compare(const IcebergTypeInfo& t, const PartitionValue& a, const PartitionValue& b,
                            bool uuid_signed) noexcept;

// Equality of two values (both null = equal): the grouping key of a partition tuple.
bool partition_value_equal(const PartitionValue& a, const PartitionValue& b) noexcept;

// Binary single-value serialization of a non-null value of type `t` into out[0..cap), *n bytes (0 is a legal
// length: an empty string/binary). Exact typed lanes, integer widths, time range,
// decimal precision and UUID/fixed lengths are required. False on unsupported,
// partial or invalid values/capacity; *n becomes zero and bytes remain unchanged.
bool partition_value_encode(const IcebergTypeInfo& t, const PartitionValue& v, uint8_t* out, uint32_t cap,
                            uint32_t* n) noexcept;

// The manifest-list partition summaries (spec: "Manifest Lists", field_summary) of a manifest holding `files` under
// `spec` (result types `types`): per field contains_null, contains_nan and the lower/upper bound over every entry.
// Sets e->n_partitions to 0 (the manifest's partitions are then written null, which only forfeits pruning) when the
// spec has more fields than a ManifestListEntry holds, a value is partial, a bound is wider than a summary holds, or
// a uuid field's bounds differ between unsigned and Java's signed UUID order.
void manifest_partition_summaries(const DataFileRef* files, uint32_t n, const PartitionSpec* spec,
                                  const IcebergTypeInfo* types, ManifestListEntry* e) noexcept;

// The inverse, for reading partition summary bounds. Exact declared widths and
// values are checked; failure preserves *out. Nonzero boolean bytes mean true.
// Geometry/geography values are opaque WKB; their stats bounds use another format.
bool partition_value_decode(const IcebergTypeInfo& t, const uint8_t* b, uint32_t len, PartitionValue* out) noexcept;

// The Avro type JSON of a partition field of result type `t` (`{"type":"int","logicalType":"date"}`, a uniquely
// named fixed for uuid/fixed/decimal). Returns the length written (NUL-terminated), 0 if `cap` is too small.
uint32_t partition_avro_type_json(const IcebergTypeInfo& t, int32_t field_id, char* out, uint32_t cap) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
