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
    // Nested Struct{metadata, value} carrying a self-describing value --
    // parquet's VARIANT logical type. Zero-width like the other nested types:
    // the data lives in the child columns.
    Variant   = 33,
    Dictionary= 40,
    FixedSizeBinary = 41, Decimal128 = 42, Decimal256 = 43,
    // W-DEC: exact decimal with an int64 MANTISSA (scale lives on the
    // column, same as Decimal128). Internal physical refinement chosen
    // when precision <= 18 — mantissa math is plain int64 math, so every
    // SIMD int kernel applies. Code that doesn't know this type fails
    // SAFE (no match -> per-row/Decimal128 fallback paths).
    Decimal64 = 44,
    // Bolt extensions
    UUID      = 50, IPv4   = 51, Embedding = 52, Symbol = 53,
    // Wave 9.4 z.5 — multi-precision Embedding variants. Same wire shape
    // as `Embedding` (kTypeSize sentinel 0 → runtime-dynamic stride =
    // dim * bytes_per_elt). Element type traits live in
    // `bolt/kernels/bolt_vector_traits.h`. Allocated at 54/55/56 because
    // slot 53 is already taken by `Symbol` — renumbering Symbol would
    // shift a wire-persisted enum byte and break old payloads.
    EmbeddingF16 = 54, EmbeddingU8 = 55, EmbeddingI8 = 56,
    NUM_TYPES = 64,
};

// Compile-time byte-size table. 0 = variable width.
// Logical refinement of a column's PHYSICAL type.
//
// This is Arrow's model, not a departure from it: pyarrow reports a JSON
// column as `extension<arrow.json>` over string storage, and parquet
// annotates a BYTE_ARRAY with a JSON logical type. The bytes are a string;
// what changes is how a consumer should interpret them.
//
// Deliberately NOT a new BoltType. A Json BoltType would be StringView-shaped
// and yet fail every one of the ~70 `type == BoltType::Utf8` tests across
// bolt, chukonu and marbledb -- silently, since those are equality tests and
// not exhaustive switches. Carrying the refinement beside the type instead
// means a JSON column IS a string column to everything that has not been
// taught otherwise, which is the behaviour that degrades safely.
//
// Lives in a byte that was already padding on BoltColumn, so nothing grows.
enum class BoltLogical : uint8_t {
    None    = 0,
    Json    = 1,   // Utf8 storage; parquet JSON / Arrow arrow.json
    Bson    = 2,   // Binary storage; parquet BSON
    Uuid    = 3,   // FixedSizeBinary(16) storage
    // Struct{metadata: Binary, value: Binary} storage, i.e. a
    // ColumnFormat::Nested column of BoltType::Variant. The parquet VARIANT
    // logical type and Spark's variant binary encoding.
    Variant = 4,
};

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
    0,                       // Variant (nested: data is in the children)
    0, 0, 0, 0, 0, 0,       // 34-39 reserved
    0, 0, 16, 32,           // Dictionary, FixedSizeBinary, Decimal128, Decimal256
    8,                       // Decimal64 (int64 mantissa; scale on the column)
    0, 0, 0, 0, 0,          // 45-49 reserved
    16, 4, 0, 0,             // UUID, IPv4, Embedding, Symbol
    0, 0, 0,                 // EmbeddingF16, EmbeddingU8, EmbeddingI8 (runtime stride)
    0, 0, 0, 0, 0, 0, 0,    // 57-63 reserved
};

inline constexpr size_t type_size(BoltType t) noexcept {
    return kTypeSize[static_cast<uint8_t>(t)];
}
inline constexpr bool is_fixed_width(BoltType t) noexcept { return type_size(t) > 0; }
inline constexpr bool is_numeric(BoltType t) noexcept {
    auto v = static_cast<uint8_t>(t); return v >= 2 && v <= 12;
}
inline constexpr bool is_vector(BoltType t) noexcept {
    return t == BoltType::Embedding ||
           t == BoltType::EmbeddingF16 ||
           t == BoltType::EmbeddingU8 ||
           t == BoltType::EmbeddingI8;
}

// Vector / Embedding columns carry a runtime-dynamic stride. For the
// legacy f32 `Embedding` type the row size is `dim * 4`; the Wave 9.4
// z.5 multi-precision variants use 2 bytes (f16) or 1 byte (u8/i8) per
// element. `kTypeSize[]` stays 0 for all four (the "dynamic" sentinel
// that `is_fixed_width` already recognises) — callers must consult the
// dimension explicitly via this helper.
BOLT_FORCE_INLINE size_t
embedding_stride(uint32_t dim) noexcept {
    return static_cast<size_t>(dim) * sizeof(float);
}

// Per-element-precision stride. Returns 0 for any non-vector type so
// callers can branch on a single non-zero result. Header-only kernel:
// branchless on the common case (cold switch on a compile-time-constant
// `t` collapses to a single multiply at -O2).
BOLT_FORCE_INLINE size_t
embedding_stride_for_type(BoltType t, uint32_t dim) noexcept {
    switch (t) {
        case BoltType::Embedding:    return static_cast<size_t>(dim) * 4u;
        case BoltType::EmbeddingF16: return static_cast<size_t>(dim) * 2u;
        case BoltType::EmbeddingU8:  return static_cast<size_t>(dim) * 1u;
        case BoltType::EmbeddingI8:  return static_cast<size_t>(dim) * 1u;
        default:                     return 0u;
    }
}

// Element-width helper for the multi-precision vector family. Returns 0
// for non-vector types. Used by `vector_dim()` to decode the runtime
// stride stored in `BoltColumn::type_size_bytes`.
BOLT_FORCE_INLINE constexpr uint32_t
embedding_element_bytes(BoltType t) noexcept {
    return (t == BoltType::Embedding)    ? 4u :
           (t == BoltType::EmbeddingF16) ? 2u :
           (t == BoltType::EmbeddingU8)  ? 1u :
           (t == BoltType::EmbeddingI8)  ? 1u : 0u;
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
        case BoltType::EmbeddingF16: return "embedding_f16";
        case BoltType::EmbeddingU8:  return "embedding_u8";
        case BoltType::EmbeddingI8:  return "embedding_i8";
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
        // Embedding family: no canonical Arrow short name (would be a
        // fixed-size-list child schema, which the C Data Interface
        // export builds out separately). Returning "" keeps every
        // Embedding variant consistent with the existing `Embedding`
        // entry; consumers that need Arrow round-trip resolve through
        // the full ArrowSchema child-schema path, not this short name.
        // TODO(wave-9.4 z.6): emit "+w:<dim>:e"/"+w:<dim>:C"/"+w:<dim>:c"
        // when the FFI surface grows fixed-size-list export.
        case BoltType::Embedding:    return "";
        case BoltType::EmbeddingF16: return "";
        case BoltType::EmbeddingU8:  return "";
        case BoltType::EmbeddingI8:  return "";
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

    /// Byte-lexicographic prefilter on (length, prefix[0..4]).
    ///
    /// Returns the *byte-lex* sign if the comparison is decidable from just
    /// the 4-byte prefix; returns 0 ("undecidable") when both views agree on
    /// `min(length, 4)` prefix bytes — caller must fall back to a full byte
    /// compare on the remaining tail. This is the cheap branch-predicted
    /// screen that resolves >99% of distinct-string comparisons in TPC-H
    /// without a pointer chase.
    ///
    /// NOTE: This used to length-first-compare ("short < long regardless of
    /// content"), which is correct only for equality, NOT for MIN/MAX. The
    /// prior semantics broke byte-lex ordering for any caller that read the
    /// sign (e.g. typed GROUP BY MIN/MAX). Tiger Style: the cheap shortcut
    /// must agree with the slow path on sign.
    static int cmp_prefix(const StringView& a, const StringView& b) noexcept {
        const uint32_t na = (a.length < 4u) ? a.length : 4u;
        const uint32_t nb = (b.length < 4u) ? b.length : 4u;
        const uint32_t n  = (na < nb) ? na : nb;
        if (n > 0) {
            const int c = memcmp(a.prefix, b.prefix, n);
            if (c != 0) return (c < 0) ? -1 : 1;
        }
        // First `n` bytes are equal. If one view is shorter than `n` we
        // would have stopped earlier; the only way to land here is na==nb
        // (same number of prefix bytes considered). Lengths still differ →
        // we cannot decide the sign without the tail. Return 0 to signal
        // "fall back to a full byte compare on the spilled tail".
        return 0;
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

/// A single field — the "better Arrow" column descriptor. Fixed-size (80 bytes),
/// self-describing. No heap allocation. Carries type + nullability + the rich
/// kinds: Dictionary/Symbol key+value types, Embedding dimension / FixedSizeBinary
/// width (`fixed_size`), and decimal scale. Text/Keyword/Document map to
/// Utf8/Dictionary/Binary at this layer; per-column index attachments (BM25 /
/// PDX-vector) live on the storage-schema authority (marbledb ColumnDef), not
/// here — this descriptor is what travels with an in-memory batch.
struct BoltField {
    char     name[kMaxFieldName + 1];  // Null-terminated, 64 bytes
    BoltType type;
    bool     nullable;
    BoltType dict_key_type;    // For Dictionary/Symbol
    BoltType dict_value_type;  // For Dictionary/Symbol
    uint32_t fixed_size;       // For FixedSizeBinary, Embedding dimension
    uint8_t  decimal_scale;    // For Decimal64/128 (G2FEAT-47: rich schema)
    uint8_t  _pad[5];

    void set_name(const char* n) noexcept {
        size_t len = strlen(n);
        if (len > kMaxFieldName) len = kMaxFieldName;
        memcpy(name, n, len);
        name[len] = '\0';
    }
};

// One honest assertion ceiling for column/field counts across bolt (G2FEAT-47).
// Used ONLY in asserts/rejects — NEVER to size an array or a bitmask. Schemas and
// batches are arena-allocated right-sized to their real column count; this bound
// only stops a pathological / corrupt count. (Replaces the old fixed
// `kMaxSchemaFields`/`kMaxBatchColumns` inline-array sizes.)
static constexpr uint32_t kMaxColumns = 4096;
// Deprecated back-compat aliases — existing call sites still name these. They are
// now assertion ceilings, not allocation sizes. New code should use kMaxColumns.
static constexpr uint32_t kMaxSchemaFields = kMaxColumns;

/// Schema: a dynamically-sized, arena-backed list of fields (G2FEAT-47). No heap
/// of its own, no exceptions. `fields` points at `cap` arena-allocated BoltField
/// slots (nullptr until `set_storage`). A batch-embedded schema gets its storage
/// from `BoltBatch::alloc_columns` (sized to the batch's column count); a
/// standalone schema calls `set_storage` with a caller-allocated buffer.
/// `sizeof(BoltSchema)` is 16 bytes (was ~20 KB) — batches no longer embed a
/// fixed 256-field array. `add_field` on a schema with no storage cleanly returns
/// false (num_fields 0 >= cap 0) — never dereferences null.
struct BoltSchema {
    BoltField* fields;       // arena-owned, `cap` entries (nullptr until set_storage)
    uint32_t   num_fields;
    uint32_t   cap;

    BoltSchema() noexcept : fields(nullptr), num_fields(0), cap(0) {}

    // Attach arena-backed field storage (buf holds `capacity` BoltField). Resets
    // num_fields to 0. Callers with an Arena: allocate `capacity` BoltField and
    // pass the buffer (bolt_types.h stays Arena-free by design).
    void set_storage(BoltField* buf, uint32_t capacity) noexcept {
        fields = buf;
        cap = capacity;
        num_fields = 0;
    }

    int find_field(const char* name) const noexcept {
        for (uint32_t i = 0; i < num_fields; ++i) {
            if (strcmp(fields[i].name, name) == 0) return static_cast<int>(i);
        }
        return -1;
    }

    bool add_field(const char* name, BoltType type, bool nullable = true) noexcept {
        if (num_fields >= cap) return false;   // storage bound (was kMaxSchemaFields)
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
