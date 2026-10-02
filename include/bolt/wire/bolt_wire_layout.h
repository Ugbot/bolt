// bolt_wire_layout.h — compile-time guards on every byte the wire format
// (and therefore MSEG pages, WAL frames and spill files) persists (MSEG B5).
//
// Each value here is on disk. Changing one breaks every file already written,
// so the build must fail first and the change must be a format version bump.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/bolt_zonemap.h"

namespace bolt {
namespace wire {
namespace layout {

// Schema entry: name[64] | type u8 | format u8 | nullable u8 | scale u8 |
// fixed_size u32  = 72 bytes. bolt_wire.h writes these at literal offsets.
inline constexpr size_t kSchemaNameBytes   = 64;
inline constexpr size_t kSchemaTypeOff     = 64;
inline constexpr size_t kSchemaFormatOff   = 65;
inline constexpr size_t kSchemaNullableOff = 66;
inline constexpr size_t kSchemaScaleOff    = 67;
inline constexpr size_t kSchemaFixedOff    = 68;
inline constexpr size_t kSchemaEntryBytes  = 72;
static_assert(kMaxFieldName + 1 == kSchemaNameBytes, "wire schema name width");
static_assert(sizeof(BoltField::name) == kSchemaNameBytes, "BoltField::name width");
static_assert(kSchemaFixedOff + sizeof(uint32_t) == kSchemaEntryBytes,
              "schema entry tail");

// Descriptor: 3 x (off u64, len u64) | format u8 | pad = 56 bytes.
inline constexpr size_t kDescFormatOff = 48;
inline constexpr size_t kDescBytes     = 56;
static_assert(3 * 2 * sizeof(uint64_t) == kDescFormatOff, "descriptor buffers");

// Header: magic[4] version u32 flags u32 num_rows i64 num_cols u32
// schema_off u32 data_off u32 = 32 bytes.
inline constexpr size_t kHeaderBytes = 32;
static_assert(4 + 4 + 4 + 8 + 4 + 4 + 4 == kHeaderBytes, "header fields");

// Persisted enum bytes: the type and format bytes of every schema entry and
// descriptor. Renumbering any of these silently re-types old files.
static_assert(sizeof(BoltType) == 1 && sizeof(ColumnFormat) == 1 &&
              sizeof(BoltLogical) == 1, "persisted enums are one byte");
static_assert(static_cast<uint8_t>(BoltType::Bool) == 1 &&
              static_cast<uint8_t>(BoltType::Int8) == 2 &&
              static_cast<uint8_t>(BoltType::Int64) == 5 &&
              static_cast<uint8_t>(BoltType::UInt64) == 9 &&
              static_cast<uint8_t>(BoltType::Float16) == 10 &&
              static_cast<uint8_t>(BoltType::Float64) == 12 &&
              static_cast<uint8_t>(BoltType::Date32) == 16 &&
              static_cast<uint8_t>(BoltType::Date64) == 17 &&
              static_cast<uint8_t>(BoltType::Timestamp) == 18 &&
              static_cast<uint8_t>(BoltType::Duration) == 19 &&
              static_cast<uint8_t>(BoltType::Utf8) == 25 &&
              static_cast<uint8_t>(BoltType::Binary) == 26 &&
              static_cast<uint8_t>(BoltType::List) == 30 &&
              static_cast<uint8_t>(BoltType::Struct) == 31 &&
              static_cast<uint8_t>(BoltType::Map) == 32 &&
              static_cast<uint8_t>(BoltType::FixedSizeBinary) == 41 &&
              static_cast<uint8_t>(BoltType::Decimal128) == 42 &&
              static_cast<uint8_t>(BoltType::Decimal256) == 43 &&
              static_cast<uint8_t>(BoltType::Decimal64) == 44 &&
              static_cast<uint8_t>(BoltType::UUID) == 50 &&
              static_cast<uint8_t>(BoltType::IPv4) == 51 &&
              static_cast<uint8_t>(BoltType::Embedding) == 52 &&
              static_cast<uint8_t>(BoltType::Symbol) == 53 &&
              static_cast<uint8_t>(BoltType::EmbeddingI8) == 56,
              "BoltType byte values are persisted");
static_assert(static_cast<uint8_t>(ColumnFormat::Flat) == 0 &&
              static_cast<uint8_t>(ColumnFormat::Constant) == 1 &&
              static_cast<uint8_t>(ColumnFormat::Dictionary) == 2 &&
              static_cast<uint8_t>(ColumnFormat::Sequence) == 3 &&
              static_cast<uint8_t>(ColumnFormat::View) == 4 &&
              static_cast<uint8_t>(ColumnFormat::RLE) == 5 &&
              static_cast<uint8_t>(ColumnFormat::BitPacked) == 6 &&
              static_cast<uint8_t>(ColumnFormat::FrameOfRef) == 7 &&
              static_cast<uint8_t>(ColumnFormat::VarBinary) == 8 &&
              static_cast<uint8_t>(ColumnFormat::Nested) == 9 &&
              static_cast<uint8_t>(ColumnFormat::DeltaFOR) == 10,
              "ColumnFormat byte values are persisted (MSEG PageEntry.format)");

// Fixed-width row strides that sit in b1 buffers on disk.
static_assert(type_size(BoltType::Bool) == 1, "Bool is byte-packed on disk");
static_assert(type_size(BoltType::Date32) == 4 && type_size(BoltType::Date64) == 8 &&
              type_size(BoltType::Timestamp) == 8 &&
              type_size(BoltType::Duration) == 8 &&
              type_size(BoltType::Decimal64) == 8 &&
              type_size(BoltType::Decimal128) == 16 &&
              type_size(BoltType::Decimal256) == 32 &&
              type_size(BoltType::UUID) == 16 && type_size(BoltType::IPv4) == 4 &&
              type_size(BoltType::FixedSizeBinary) == 16 &&
              type_size(BoltType::Float16) == 2 && type_size(BoltType::Utf8) == 16,
              "fixed-width strides are persisted");

// StringView is the Utf8 b1 row on disk: length u32 | prefix[4] |
// {inline[8] | buf_idx u32, offset u32}; offsets are relative to the page's
// own overflow buffer (b2), so the bytes are relocatable.
static_assert(sizeof(StringView) == 16 && alignof(StringView) == 4, "StringView size");
static_assert(offsetof(StringView, length) == 0 && offsetof(StringView, prefix) == 4,
              "StringView head");
static_assert(std::is_trivially_copyable_v<StringView>, "StringView is raw bytes");

// ZoneMap is persisted verbatim in frame trailers and MSEG stats.
static_assert(sizeof(ZoneMap) == 32 && std::is_trivially_copyable_v<ZoneMap>,
              "ZoneMap is raw bytes");

}  // namespace layout
}  // namespace wire
}  // namespace bolt
