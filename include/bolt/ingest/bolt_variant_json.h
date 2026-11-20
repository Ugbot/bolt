// bolt/ingest/bolt_variant_json.h — a VARIANT value (Parquet VariantEncoding.md: metadata + value) as JSON text, in the
// form Spark 4's to_json(variant) writes it (G2ICE-382), so SQL over an Iceberg variant column shows what Spark shows.
//
//   null / true / false; ints as integers; a decimal with its trailing fraction zeros stripped, never an exponent
//   (BigDecimal.stripTrailingZeros().toPlainString()); double / float as Java's Double.toString / Float.toString
//   (shortest round-trip digits, "1.5", "1.0E10", "1.0E-4"), NaN / Infinity / -Infinity quoted; string with Jackson's
//   escapes (\" \\ \b \f \n \r \t, other controls \u00XX uppercase hex, everything else raw UTF-8); date "YYYY-MM-DD";
//   timestamp "YYYY-MM-DD HH:MM:SS[.f]+00:00" (UTC), timestamp_ntz without the offset, fraction trimmed (micros and nanos);
//   time "HH:MM:SS[.f]"; binary as base64; uuid as its 8-4-4-4-12 lowercase text; objects in their stored (key) order,
//   arrays in order.
#pragma once

#include <cstdint>

namespace bolt {

class Arena;
struct BoltColumn;

namespace ingest {
namespace variant {

// Nesting a value may have (objects and arrays): past it the value is refused, never cut.
inline constexpr uint32_t kVariantJsonMaxDepth = 64;
// JSON text one rendered column may hold (every row): past it the conversion fails (a budget, not a silent cut).
inline constexpr uint64_t kVariantJsonColumnBytes = uint64_t{1} << 31;

// Render one variant into out[0, cap). False when the encoding is malformed, nests past kVariantJsonMaxDepth, names a
// dictionary id the metadata lacks, or the text does not fit cap (*out_len then holds no meaning).
bool variant_to_json(const uint8_t* metadata, uint32_t metadata_len, const uint8_t* value, uint32_t value_len, char* out,
                     uint32_t cap, uint32_t* out_len) noexcept;

// An upper bound on the JSON text of a value of value_len bytes (strings escape to 6 bytes per byte, base64, numbers).
inline uint64_t variant_json_bound(uint32_t value_len) noexcept { return uint64_t{value_len} * 8u + 64u; }

// A BoltColumn::make_variant column (rows rows; its null rows stay null) as a Utf8 StringView column of JSON texts, all
// storage from arena. False on a malformed row or past kVariantJsonColumnBytes.
bool variant_column_to_json(const BoltColumn& variant, int64_t rows, Arena* arena, BoltColumn* out) noexcept;

}  // namespace variant
}  // namespace ingest
}  // namespace bolt
