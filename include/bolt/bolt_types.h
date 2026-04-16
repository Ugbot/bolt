// bolt_types.h — Self-contained type system for Bolt
//
// RULES: No exceptions. No RTTI. No smart pointers. No std::string in structs.
// Fixed-size field names. Compile-time size table. Zero heap in the type system.
// Arrow C Data Interface compatible (ArrowSchema/ArrowArray export).

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cassert>

#include "bolt/bolt_port.h"

namespace bolt {

// ============================================================================
// Type Enum — values match arrow::Type::type for common types
// ============================================================================

enum class BoltType : uint8_t {
    NA        = 0,
    Bool      = 1,
    Int8      = 2,  Int16  = 3,  Int32  = 4,  Int64  = 5,
    UInt8     = 6,  UInt16 = 7,  UInt32 = 8,  UInt64 = 9,
    Float16   = 10, Float32= 11, Float64= 12,
    Date32    = 16, Date64 = 17, Timestamp = 18, Duration = 19,
    Utf8      = 25, Binary = 26,
    List      = 30, Struct = 31, Map    = 32,
    Dictionary= 40,
    FixedSizeBinary = 41, Decimal128 = 42, Decimal256 = 43,
    // Bolt extensions
    UUID      = 50, IPv4   = 51, Embedding = 52, Symbol = 53,
    NUM_TYPES = 64,
};

// Compile-time byte-size table. 0 = variable width.
inline constexpr uint8_t kTypeSize[64] = {
    0,                       // NA
    1,                       // Bool (byte-packed for SIMD, not bit-packed)
    1, 2, 4, 8,             // Int8..Int64
    1, 2, 4, 8,             // UInt8..UInt64
    2, 4, 8,                 // Float16..Float64
    0, 0, 0,                 // 13-15 reserved
    4, 8, 8, 8,             // Date32, Date64, Timestamp, Duration
    0, 0, 0, 0, 0,          // 20-24 reserved
    16, 0,                   // Utf8 (16-byte view), Binary
    0, 0, 0,                 // 27-29 reserved
    0, 0, 0,                 // List, Struct, Map
    0, 0, 0, 0, 0, 0, 0,   // 33-39 reserved
    0, 0, 16, 32,           // Dictionary, FixedSizeBinary, Decimal128, Decimal256
    0, 0, 0, 0, 0, 0,       // 44-49 reserved
    16, 4, 0, 0,             // UUID, IPv4, Embedding, Symbol
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 54-63 reserved
};

inline constexpr size_t type_size(BoltType t) noexcept {
    return kTypeSize[static_cast<uint8_t>(t)];
}
inline constexpr bool is_fixed_width(BoltType t) noexcept { return type_size(t) > 0; }
inline constexpr bool is_numeric(BoltType t) noexcept {
    auto v = static_cast<uint8_t>(t); return v >= 2 && v <= 12;
}
inline constexpr bool is_integer(BoltType t) noexcept {
    auto v = static_cast<uint8_t>(t); return v >= 2 && v <= 9;
}
inline constexpr bool is_floating(BoltType t) noexcept {
    return t == BoltType::Float16 || t == BoltType::Float32 || t == BoltType::Float64;
}
inline constexpr bool is_temporal(BoltType t) noexcept {
    auto v = static_cast<uint8_t>(t); return v >= 16 && v <= 19;
}
inline constexpr bool is_string(BoltType t) noexcept {
    return t == BoltType::Utf8 || t == BoltType::Symbol;
}

inline const char* type_name(BoltType t) noexcept {
    switch (t) {
        case BoltType::NA:       return "null";
        case BoltType::Bool:     return "bool";
        case BoltType::Int8:     return "int8";
        case BoltType::Int16:    return "int16";
        case BoltType::Int32:    return "int32";
        case BoltType::Int64:    return "int64";
        case BoltType::UInt8:    return "uint8";
        case BoltType::UInt16:   return "uint16";
        case BoltType::UInt32:   return "uint32";
        case BoltType::UInt64:   return "uint64";
        case BoltType::Float16:  return "float16";
        case BoltType::Float32:  return "float32";
        case BoltType::Float64:  return "float64";
        case BoltType::Date32:   return "date32";
        case BoltType::Date64:   return "date64";
        case BoltType::Timestamp:return "timestamp[us]";
        case BoltType::Duration: return "duration[us]";
        case BoltType::Utf8:     return "utf8";
        case BoltType::Binary:   return "binary";
        case BoltType::List:     return "list";
        case BoltType::Struct:   return "struct";
        case BoltType::Map:      return "map";
        case BoltType::Dictionary:return "dictionary";
        case BoltType::FixedSizeBinary: return "fixed_binary";
        case BoltType::Decimal128:return "decimal128";
        case BoltType::Decimal256:return "decimal256";
        case BoltType::UUID:     return "uuid";
        case BoltType::IPv4:     return "ipv4";
        case BoltType::Embedding:return "embedding";
        case BoltType::Symbol:   return "symbol";
        default:                 return "unknown";
    }
}

/// Arrow format string for C Data Interface export.
/// See https://arrow.apache.org/docs/format/CDataInterface.html
inline const char* arrow_format_string(BoltType t) noexcept {
    switch (t) {
        case BoltType::NA:       return "n";
        case BoltType::Bool:     return "b";
        case BoltType::Int8:     return "c";
        case BoltType::Int16:    return "s";
        case BoltType::Int32:    return "i";
        case BoltType::Int64:    return "l";
        case BoltType::UInt8:    return "C";
        case BoltType::UInt16:   return "S";
        case BoltType::UInt32:   return "I";
        case BoltType::UInt64:   return "L";
        case BoltType::Float16:  return "e";
        case BoltType::Float32:  return "f";
        case BoltType::Float64:  return "g";
        case BoltType::Date32:   return "tdD";
        case BoltType::Date64:   return "tdm";
        case BoltType::Timestamp:return "tsu:";   // microseconds, no tz
        case BoltType::Duration: return "tDu";
        case BoltType::Utf8:     return "vu";      // utf8_view (Arrow StringView)
        case BoltType::Binary:   return "vz";      // binary_view
        case BoltType::Decimal128:return "d:38,10"; // default precision/scale
        case BoltType::UUID:     return "w:16";     // fixed-size binary 16
        case BoltType::IPv4:     return "I";        // stored as uint32
        default:                 return "";
    }
}

// ============================================================================
// X-Macro Type Lists
// ============================================================================

#define BOLT_NUMERIC_TYPES \
    X(Int8,    int8_t,   BoltType::Int8)   \
    X(Int16,   int16_t,  BoltType::Int16)  \
    X(Int32,   int32_t,  BoltType::Int32)  \
    X(Int64,   int64_t,  BoltType::Int64)  \
    X(UInt8,   uint8_t,  BoltType::UInt8)  \
    X(UInt16,  uint16_t, BoltType::UInt16) \
    X(UInt32,  uint32_t, BoltType::UInt32) \
    X(UInt64,  uint64_t, BoltType::UInt64) \
    X(Float32, float,    BoltType::Float32)\
    X(Float64, double,   BoltType::Float64)

#define BOLT_INTEGER_TYPES \
    X(Int8,    int8_t,   BoltType::Int8)   \
    X(Int16,   int16_t,  BoltType::Int16)  \
    X(Int32,   int32_t,  BoltType::Int32)  \
    X(Int64,   int64_t,  BoltType::Int64)  \
    X(UInt8,   uint8_t,  BoltType::UInt8)  \
    X(UInt16,  uint16_t, BoltType::UInt16) \
    X(UInt32,  uint32_t, BoltType::UInt32) \
    X(UInt64,  uint64_t, BoltType::UInt64)

// ============================================================================
// German-Style String View (16 bytes, Arrow StringView compatible)
// ============================================================================

struct StringView {
    uint32_t length;
    char prefix[4];
    union {
        char inline_data[8];   // length <= 12: remaining after prefix
        struct {
            uint32_t buf_idx;  // index into data buffer array
            uint32_t offset;   // byte offset within buffer
        } ref;
    };

    bool is_inline()   const noexcept { return length <= 12; }
    bool is_empty()    const noexcept { return length == 0; }

    /// Fast equality: check length + 4-byte prefix first. Resolves
    /// >99% of distinct-string comparisons without pointer chase.
    static int cmp_prefix(const StringView& a, const StringView& b) noexcept {
        if (a.length != b.length) return (a.length < b.length) ? -1 : 1;
        return memcmp(a.prefix, b.prefix, 4);
    }

    /// Full inline equality (only valid when both are inline)
    bool eq_inline(const StringView& o) const noexcept {
        if (length != o.length) return false;
        // Compare all 12 inline bytes at once (prefix + inline_data)
        // Safe because we always have 16 bytes allocated
        return memcmp(prefix, o.prefix, (length < 12) ? length : 12) == 0;
    }

    /// Create inline StringView from C string
    static StringView from_cstr(const char* s) noexcept {
        StringView v;
        v.length = 0;
        memset(&v, 0, sizeof(v));
        if (!s) return v;
        size_t len = strlen(s);
        v.length = static_cast<uint32_t>(len);
        if (len <= 12) {
            memcpy(v.prefix, s, (len < 4) ? len : 4);
            if (len > 4) memcpy(v.inline_data, s + 4, len - 4);
        } else {
            memcpy(v.prefix, s, 4);
            // Caller must set ref.buf_idx and ref.offset
        }
        return v;
    }
};

static_assert(sizeof(StringView) == 16, "StringView must be exactly 16 bytes");
static_assert(alignof(StringView) == 4, "StringView must be 4-byte aligned");

// ============================================================================
// Field + Schema (fixed-size, no heap)
// ============================================================================

static constexpr uint32_t kMaxFieldName = 63;  // 63 chars + null terminator

/// A single field. Fixed-size (80 bytes). No heap allocation.
struct BoltField {
    char     name[kMaxFieldName + 1];  // Null-terminated, 64 bytes
    BoltType type;
    bool     nullable;
    BoltType dict_key_type;    // For Dictionary/Symbol
    BoltType dict_value_type;  // For Dictionary/Symbol
    uint32_t fixed_size;       // For FixedSizeBinary, Embedding dimension
    uint8_t  _pad[6];

    void set_name(const char* n) noexcept {
        size_t len = strlen(n);
        if (len > kMaxFieldName) len = kMaxFieldName;
        memcpy(name, n, len);
        name[len] = '\0';
    }
};

static constexpr uint32_t kMaxSchemaFields = 256;

/// Schema: fixed-capacity array of fields. No heap. No exceptions.
struct BoltSchema {
    BoltField fields[kMaxSchemaFields];
    uint32_t  num_fields;

    BoltSchema() noexcept : num_fields(0) {
        memset(fields, 0, sizeof(fields));
    }

    int find_field(const char* name) const noexcept {
        for (uint32_t i = 0; i < num_fields; ++i) {
            if (strcmp(fields[i].name, name) == 0) return static_cast<int>(i);
        }
        return -1;
    }

    bool add_field(const char* name, BoltType type, bool nullable = true) noexcept {
        if (num_fields >= kMaxSchemaFields) return false;
        BoltField& f = fields[num_fields++];
        memset(&f, 0, sizeof(f));
        f.set_name(name);
        f.type = type;
        f.nullable = nullable;
        f.dict_key_type = BoltType::UInt8;
        f.dict_value_type = BoltType::Utf8;
        return true;
    }

    const BoltField& field(int i) const noexcept {
        assert(i >= 0 && i < (int)num_fields);
        return fields[i];
    }

    bool equals(const BoltSchema& other) const noexcept {
        if (num_fields != other.num_fields) return false;
        for (uint32_t i = 0; i < num_fields; ++i) {
            if (strcmp(fields[i].name, other.fields[i].name) != 0) return false;
            if (fields[i].type != other.fields[i].type) return false;
        }
        return true;
    }
};

// ============================================================================
// Arrow C Data Interface (ABI-level, zero dependency)
// ============================================================================
// These structs are defined by the Arrow spec. We define them ourselves so
// we can export BoltColumn/BoltBatch to any Arrow consumer (Polars, DuckDB,
// Pandas) without linking libarrow. Just fill the struct and hand it over.
//
// Spec: https://arrow.apache.org/docs/format/CDataInterface.html

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

struct ArrowSchema {
    const char*    format;
    const char*    name;
    const char*    metadata;
    int64_t        flags;
    int64_t        n_children;
    ArrowSchema**  children;
    ArrowSchema*   dictionary;
    void (*release)(ArrowSchema*);
    void*          private_data;
};

struct ArrowArray {
    int64_t        length;
    int64_t        null_count;
    int64_t        offset;
    int64_t        n_buffers;
    int64_t        n_children;
    const void**   buffers;
    ArrowArray**   children;
    ArrowArray*    dictionary;
    void (*release)(ArrowArray*);
    void*          private_data;
};

// Arrow PyCapsule interface names
#define ARROW_SCHEMA_CAPSULE_NAME "arrow_schema"
#define ARROW_ARRAY_CAPSULE_NAME  "arrow_array"

#endif  // ARROW_C_DATA_INTERFACE

}  // namespace bolt
