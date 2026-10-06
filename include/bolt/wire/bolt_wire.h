// bolt_wire.h — Self-describing flat serialization for BoltBatch.
//
// RULES: No exceptions. No RTTI. No smart pointers. No std::string. No std::vector.
// All functions noexcept. Fixed caps, >=2 asserts per function, functions <=70 lines.
//
// Scope:
//   - Format::Flat columns: numeric types in BOLT_NUMERIC_TYPES plus Bool
//     (byte-packed — 1 byte/row, matching kTypeSize[Bool] and every other
//     fixed-width column layout; NOT bit-packed) plus, since v4,
//     Date32/Timestamp/Decimal128/Decimal64 (fixed-width, decimal scale
//     carried in the schema entry's byte 67).
//   - Format::VarBinary columns: BoltType::Utf8 / Binary / Symbol — payload
//     is `(offsets[len+1], raw bytes)`. Per-row slice = pool[off[i]..off[i+1]).
//     Added at version 2 (Pass-Wiring 2026-05-01) to carry the unified-
//     memtable Field::* paths (kText / kKeyword / kJson / kStored) through
//     the WAL kBatch path.
//   - v5 (MSEG B3) adds, written only when a batch needs them (a batch of
//     v4 pairs is byte-identical to what older bolt wrote):
//       Constant   b1 = the 16-byte inline value, b2 = a long string's bytes;
//                  descriptor flag all_null (b0 omitted), u32 = value width
//       Sequence   b1 = {offset i64, step i64}; u32 = value width
//       RLE        b1 = values[runs], b2 = int32 run_ends[runs]
//       Dictionary b1 = keys (u8/u16/u32, width in descriptor byte 50),
//                  b2 = the values as a one-column wire sub-blob
//       Nested     b1 = the children as a wire sub-blob (List/Map: the
//                  element column; Struct: the fields), b2 = int32 element
//                  offsets[len+1] for List/Map; u32 = child count
//       View       written as Flat (validity_offset applied)
//       Flat Binary/Symbol (StringView + overflow, as Utf8), Date64,
//       Duration, UUID, IPv4, FixedSizeBinary, Decimal256, and the
//       BoltLogical tag (schema byte 66 bits 1..4).
//     Sub-blobs recurse to at most kWireMaxNestDepth levels. Descriptor bytes
//     49 (flags), 50 (u8 param) and 52..55 (u32 param) were zero padding in
//     v1..v4.
//   - BitPacked / FrameOfRef / DeltaFOR have no wire id yet (B3 remainder,
//     after their layout is fixed by B1) and must be materialized first.
//   - Readers treat the blob as untrusted: every span is bounds-checked
//     without overflow and (in an aligned blob, which every writer emits)
//     64 B aligned; rows <= kWireMaxRows; a Flat b1 is exactly rows x
//     stride; RLE run ends strictly increase to rows; Dictionary codes are
//     below the dictionary length; List/Map offsets are monotone; a long
//     Constant string carries exactly its bytes. VarBinary offsets are bounded and monotone before allocation. Flat
//     StringView rows are NOT walked: the frame CRC guards those bytes.
//   - Little-endian target only (x86 / ARM64). Flag bit 0 records this.
//   - Deserialize copies each column buffer into the caller-provided Arena.
//     Zero-copy-over-mmap is a future goal; the 64-byte alignment in the blob
//     already makes every buffer pointer-aligned for future mmap paths.
//
// Binary layout:
//   +------------------------------------------------+ offset 0
//   |  magic[4]       = "BOLT"                       |
//   |  version        = u32 (3 — Flat Utf8 support)  |
//   |  flags          = u32 (bit0=LE, bit1=aligned64)|
//   |  num_rows       = i64                          |
//   |  num_cols       = u32                          |
//   |  schema_offset  = u32                          |
//   |  data_offset    = u32                          |
//   |  header_pad     -> total 32 bytes              |
//   +------------------------------------------------+
//   | Schema block (num_cols * kWireSchemaEntrySize) |
//   |   per col: { name[64], type u8, format u8,     |
//   |              nullable u8, decimal_scale u8,    |
//   |              fixed_size u32 -> 72 bytes }      |
//   +------------------------------------------------+
//   | Column descriptors (num_cols * kWireDescSize)  |
//   |   per col: { b0_off u64, b0_len u64,           |
//   |              b1_off u64, b1_len u64,           |
//   |              b2_off u64, b2_len u64,           |
//   |              format u8, pad -> 56 bytes }      |
//   +------------------------------------------------+
//   | Data region (64-byte aligned per buffer)       |
//   |   b0 = validity bitmap (may be empty)          |
//   |   Flat (non-Utf8):                             |
//   |     b1 = primitive data array                  |
//   |     b2 = (unused; len 0)                       |
//   |   Flat + Utf8 (v3+): StringView is a fixed 16-  |
//   |   byte type, so it rides b1 like any other Flat |
//   |   primitive; b2 carries the spilled overflow    |
//   |   buffer (str_overflow_base) any >12-byte row's |
//   |   ref.offset resolves against — same b1/b2      |
//   |   split VarBinary uses below, b1 holding fixed- |
//   |   width views instead of an offsets array:      |
//   |     b1 = StringView[num_rows] (16 B/row)        |
//   |     b2 = spilled bytes actually referenced      |
//   |          (walked per-row; see flat_utf8_sizes)  |
//   |   VarBinary:                                   |
//   |     b1 = offsets array (int32 × (rows + 1))    |
//   |     b2 = payload bytes (offsets[rows] long)    |
//   +------------------------------------------------+

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/wire/bolt_wire_layout.h"
#include "bolt/wire/bolt_wire_limits.h"

namespace bolt {
namespace wire {

// ===========================================================================
// Constants
// ===========================================================================

inline constexpr uint32_t kWireMagic    = 0x544C4F42u;  // 'BOLT' LE
// Version written for a batch of v4 pairs (byte-identical to pre-B3 bolt).
inline constexpr uint32_t kWireVersion  = 4u;  // bumped: Date32/Timestamp/Decimal128/Decimal64
// B3: written when a batch carries a v5 pair (Constant, Sequence, Dictionary,
// RLE, Nested, Flat Binary/Symbol, Date64/Duration/UUID/IPv4/FixedSizeBinary/
// Decimal256, a BoltLogical tag). Readers accept 1..kWireVersionLatest.
inline constexpr uint32_t kWireVersionEncoded = 5u;
inline constexpr uint32_t kWireVersionLatest     = 5u;
inline constexpr uint32_t kWireFlagLE   = 1u << 0;
inline constexpr uint32_t kWireFlagAln  = 1u << 1;

inline constexpr size_t kWireHeaderSize      = 32;
inline constexpr size_t kWireSchemaEntrySize = 72;
inline constexpr size_t kWireDescSize        = 56;
inline constexpr size_t kWireAlign           = 64;

// A frame carries up to bolt::kMaxColumns columns (num_cols is a u32 in the
// header; the serializer keeps no per-column scratch, so width costs nothing).
inline constexpr uint32_t kWireMaxCols = kMaxColumns;

static_assert(kWireHeaderSize == layout::kHeaderBytes, "wire header size");
static_assert(kWireSchemaEntrySize == layout::kSchemaEntryBytes, "schema entry size");
static_assert(kWireDescSize == layout::kDescBytes, "descriptor size");
static_assert(kWireAlign == 64 && kWireDescSize % 8 == 0, "wire alignment");

// ===========================================================================
// Internal helpers
// ===========================================================================

namespace detail {

BOLT_FORCE_INLINE size_t align_up(size_t x, size_t a) noexcept {
    assert(a > 0);
    assert((a & (a - 1)) == 0);
    return (x + a - 1) & ~(a - 1);
}

// G2ICE-141: zero [buf+off+written, buf+off+aligned) — the 64-byte alignment
// pad after a data span's real payload and, when `written == 0` but the span
// has nonzero length, the whole span whose source pointer was absent (e.g. a
// zero-row VarBinary column with no dict_child: its 4-byte offsets span is
// never memcpy'd, and the old whole-buffer memset silently supplied
// offsets[0] == 0). Together with the metadata-region memset in
// bolt_wire_serialize this reproduces byte-for-byte what the old
// `memset(buf, 0, total)` guaranteed — deterministic, non-leaking wire
// bytes — without re-zeroing the multi-MB payload the memcpys fully
// overwrite (measured ~29% of the synchronous ingest hot path).
BOLT_FORCE_INLINE void zero_wire_gap(uint8_t* buf, size_t off,
                                     size_t written, size_t aligned) noexcept {
    assert(buf != nullptr);
    assert(written <= aligned);
    if (aligned > written) memset(buf + off + written, 0, aligned - written);
}

// Fixed-width Flat types (a real kTypeSize row stride), v4 set.
BOLT_FORCE_INLINE bool is_fixed_v4(BoltType t) noexcept {
    if (t == BoltType::Bool || t == BoltType::Date32 || t == BoltType::Timestamp ||
        t == BoltType::Decimal128 || t == BoltType::Decimal64) return true;
    const auto v = static_cast<uint8_t>(t);
    return v >= static_cast<uint8_t>(BoltType::Int8) &&
           v <= static_cast<uint8_t>(BoltType::Float64);   // includes Float16
}

// B3 (v5): the remaining fixed-width types.
BOLT_FORCE_INLINE bool is_fixed_v5(BoltType t) noexcept {
    return t == BoltType::Date64 || t == BoltType::Duration ||
           t == BoltType::UUID || t == BoltType::IPv4 || t == BoltType::FixedSizeBinary ||
           t == BoltType::Decimal256;
}

// StringView-shaped string types: b1 = StringView[rows], b2 = overflow.
BOLT_FORCE_INLINE bool is_sv_string(BoltType t) noexcept {
    return t == BoltType::Utf8 || t == BoltType::Binary || t == BoltType::Symbol;
}

BOLT_FORCE_INLINE bool is_supported_type(BoltType t) noexcept {
    return is_fixed_v4(t) || is_fixed_v5(t) || is_sv_string(t) || is_vector(t);
}

// The lowest wire version that carries (type, format), or 0 when the wire
// format has no encoding for it. v4 is what every pre-B3 writer emitted;
// v5 (B3) adds the hot column kinds and the missing fixed-width types.
// A batch is written with the highest version any of its columns needs, so
// a batch of v4 pairs stays byte-identical to what older bolt wrote.
BOLT_FORCE_INLINE uint32_t pair_version(BoltType t, ColumnFormat f) noexcept {
    switch (f) {
        case ColumnFormat::Flat:
        case ColumnFormat::View:      // written as Flat
            if (t == BoltType::Utf8 || is_fixed_v4(t) || is_vector(t)) return 4;
            if (t == BoltType::Binary || t == BoltType::Symbol || is_fixed_v5(t)) return 5;
            return 0;
        case ColumnFormat::VarBinary:
            return is_sv_string(t) ? 4u : 0u;
        case ColumnFormat::Constant:
            return (is_fixed_v4(t) || is_fixed_v5(t) || is_sv_string(t)) &&
                   t != BoltType::Decimal256 ? 5u : 0u;
        case ColumnFormat::Sequence:
            return is_integer(t) || t == BoltType::Date32 || t == BoltType::Date64 ||
                   t == BoltType::Timestamp || t == BoltType::Duration ? 5u : 0u;
        case ColumnFormat::Dictionary:
            return is_fixed_v4(t) || is_fixed_v5(t) || is_sv_string(t) ? 5u : 0u;
        case ColumnFormat::RLE:
            return is_fixed_v4(t) || is_fixed_v5(t) ? 5u : 0u;
        case ColumnFormat::Nested:
            return t == BoltType::List || t == BoltType::Map || t == BoltType::Struct ? 5u : 0u;
        default:
            return 0;
    }
}

// True if (`type`, `format`) is a legal pair for the wire format.
BOLT_FORCE_INLINE bool is_supported_format_pair(BoltType t, ColumnFormat f) noexcept {
    return pair_version(t, f) != 0;
}

// Byte length of the data buffer (buffer1) for a Flat column of the given
// type and row count. Utf8's b1 (the fixed 16-byte StringView row array)
// falls through to the generic `type_size(t) * n` path below like any
// other fixed-width type — its SEPARATE b2 (spilled overflow bytes) is
// sized by `flat_utf8_sizes()`, not here. Embedding columns must use the
// 3-arg overload (their stride is runtime-dynamic = dim * 4); the 2-arg
// form returns 0 for Embedding.
BOLT_FORCE_INLINE size_t data_buffer_size(BoltType t, int64_t n) noexcept {
    assert(n >= 0);
    // G2ICE-144: Bool is BYTE-packed everywhere else that stores/consumes
    // it -- kTypeSize[Bool]==1 ("byte-packed for SIMD, not bit-packed"),
    // BoltColumn::make_flat_alloc, marbledb's own storage, and bolt's own
    // Arrow export/import (which explicitly pack_bool/unpack_bool AT the
    // Arrow boundary because Arrow itself is bit-packed and BoltColumn is
    // not). Bit-packing here was a local-only bug, not a second convention
    // any other consumer relied on: falls through to the generic
    // type_size(t)*n path, same as every other fixed-width type.
    size_t tsz = type_size(t);
    return tsz * static_cast<size_t>(n);
}

// Sizes the (b1, b2) pair for a Flat StringView column (Utf8; v5 Binary/Symbol): b1 is the fixed
// StringView[length] row array (16 bytes/row, same as any other Flat
// type); b2 is the spilled-overflow bytes ACTUALLY referenced by a valid
// (non-null) row with length > 12 — there is no stored "bytes used" field
// on BoltColumn, so this walks rows via the shared helper
// `bolt::detail::utf8_overflow_span` (bolt_column.h). `*out_min_off` is the
// LOWEST `ref.offset` referenced by any row in `c` — 0 for a whole (row-0-
// based) column, but potentially large for a SLICE of a bigger column (a
// chunk of a windowed/chunked wire-serialize): those rows' offsets are
// still absolute into the ORIGINAL overflow buffer. The caller must copy
// `str_overflow_base[*out_min_off, *out_min_off + *out_b2)` — NOT
// `[0, *out_b2)` — and rebase each written row's `ref.offset` by
// `-*out_min_off`, or a sliced column's b2 silently balloons to include
// every earlier row's spilled bytes too (G2FEAT-308/311's finding).
// Returns b2 == 0 when every row is inline or str_overflow_base is null
// (nothing to spill) — the caller decides whether a non-zero b2 with a
// null str_overflow_base is a malformed-column error.
BOLT_FORCE_INLINE void flat_utf8_sizes(const BoltColumn& c,
                                       size_t* out_b1, size_t* out_b2,
                                       size_t* out_min_off) noexcept {
    assert(out_b1 != nullptr && out_b2 != nullptr && out_min_off != nullptr);
    assert(is_sv_string(c.type));
    assert(c.format == ColumnFormat::Flat || c.format == ColumnFormat::View);
    *out_b1 = static_cast<size_t>(c.length) * sizeof(StringView);
    *out_b2 = 0;
    *out_min_off = 0;
    if (c.length > 0 && c.data != nullptr && c.str_overflow_base != nullptr) {
        const auto* rows = static_cast<const StringView*>(c.data);
        bolt::detail::utf8_overflow_span(rows, c.length, c.validity,
                                         c.validity_offset, out_min_off, out_b2);
    }
}

// 3-arg overload — supplies the per-row stride directly so vector
// Embedding columns (whose `type_size_bytes` is `dim * bytes_per_elt`)
// reuse the same caller shape as scalar Flat columns. Falls through to
// the 2-arg form for non-Embedding types. Wave 9.4 z.5: covers the
// f16 / u8 / i8 multi-precision variants — each carries a different
// stride but the row-product math is identical.
BOLT_FORCE_INLINE size_t data_buffer_size(BoltType t, int64_t n,
                                           uint16_t type_size_bytes) noexcept {
    assert(n >= 0);
    if (t == BoltType::Embedding ||
        t == BoltType::EmbeddingF16 ||
        t == BoltType::EmbeddingU8 ||
        t == BoltType::EmbeddingI8) {
        return static_cast<size_t>(type_size_bytes) * static_cast<size_t>(n);
    }
    return data_buffer_size(t, n);
}

BOLT_FORCE_INLINE size_t validity_bytes(int64_t n) noexcept {
    assert(n >= 0);
    return static_cast<size_t>((n + 7) / 8);
}

// Copy `n` validity bits starting at bit `bit_off` of `src` into `dst` at bit
// 0 (B4: a sliced column's bitmap starts at `validity_offset`, not bit 0).
// Bits past `n` in the last byte are written as zero, so the wire bytes are a
// function of the logical nulls only (stable CRCs and dedup).
inline void copy_validity_bits(uint8_t* BOLT_RESTRICT dst,
                               const uint8_t* BOLT_RESTRICT src,
                               int64_t bit_off, int64_t n) noexcept {
    assert(dst != nullptr && src != nullptr);
    assert(bit_off >= 0 && n >= 0);
    const size_t nbytes = validity_bytes(n);
    if (nbytes == 0) return;
    const uint8_t* s = src + (bit_off >> 3);
    const unsigned sh = static_cast<unsigned>(bit_off & 7);
    if (sh == 0) {
        memcpy(dst, s, nbytes);
    } else {
        // Byte i of the output takes the high (8 - sh) bits of s[i] and the
        // low sh bits of s[i + 1]; s[i + 1] exists only while it holds a bit
        // below n.
        const size_t src_bytes = validity_bytes(n + static_cast<int64_t>(sh));
        for (size_t i = 0; i < nbytes; ++i) {        // bounded: nbytes
            unsigned v = static_cast<unsigned>(s[i]) >> sh;
            if (i + 1 < src_bytes) v |= static_cast<unsigned>(s[i + 1]) << (8 - sh);
            dst[i] = static_cast<uint8_t>(v);
        }
    }
    const unsigned tail = static_cast<unsigned>(n & 7);
    if (tail != 0) dst[nbytes - 1] &= static_cast<uint8_t>((1u << tail) - 1u);
}

BOLT_FORCE_INLINE void write_u32_le(uint8_t* p, uint32_t v) noexcept {
    assert(p != nullptr);
    memcpy(p, &v, sizeof(v));
}
BOLT_FORCE_INLINE void write_u64_le(uint8_t* p, uint64_t v) noexcept {
    assert(p != nullptr);
    memcpy(p, &v, sizeof(v));
}
BOLT_FORCE_INLINE void write_i64_le(uint8_t* p, int64_t v) noexcept {
    assert(p != nullptr);
    memcpy(p, &v, sizeof(v));
}

BOLT_FORCE_INLINE uint32_t read_u32_le(const uint8_t* p) noexcept {
    assert(p != nullptr);
    uint32_t v; memcpy(&v, p, sizeof(v)); return v;
}
BOLT_FORCE_INLINE uint64_t read_u64_le(const uint8_t* p) noexcept {
    assert(p != nullptr);
    uint64_t v; memcpy(&v, p, sizeof(v)); return v;
}
BOLT_FORCE_INLINE int64_t read_i64_le(const uint8_t* p) noexcept {
    assert(p != nullptr);
    int64_t v; memcpy(&v, p, sizeof(v)); return v;
}

// Descriptor tail (bytes 49..55 were zero padding through v4).
inline constexpr size_t  kDescFlagsOff   = 49;   // u8 kDescFlag*
inline constexpr size_t  kDescParam8Off  = 50;   // u8: Dictionary key width
inline constexpr size_t  kDescParam32Off = 52;   // u32: Constant/Sequence stride, Nested children
inline constexpr uint8_t kDescFlagAllNull = 1u;  // Constant with every row null (b0 omitted)

// Schema entry byte 66: bit 0 nullable; v5 bits 1..4 = BoltLogical.
inline constexpr uint8_t kSchemaLogicalShift = 1;
inline constexpr uint8_t kSchemaLogicalMask  = 0x1Eu;

// One column's buffer plan.
struct ColWire {
    size_t   b0, b1, b2;
    size_t   moff;        // Flat string: lowest spilled offset (rebase)
    uint32_t version;     // wire version this column needs (incl. children)
    uint32_t p32;         // descriptor u32 parameter
    uint8_t  flags;       // kDescFlag*
    uint8_t  p8;          // descriptor u8 parameter
};

inline size_t cols_wire_size(const BoltColumn* cols, uint32_t n, uint32_t depth,
                             uint32_t* version) noexcept;
inline size_t cols_wire_write(const BoltColumn* cols, const BoltField* fields, uint32_t n,
                              int64_t rows, uint8_t* buf, size_t cap,
                              uint32_t depth) noexcept;

// True if a valid row of a StringView column is longer than 12 bytes (needs
// str_overflow_base). Walked only when the base is absent.
inline bool sv_rows_spill(const BoltColumn& c) noexcept {
    assert(is_sv_string(c.type));
    assert(c.data != nullptr || c.length == 0);
    const auto* rows = static_cast<const StringView*>(c.data);
    for (int64_t r = 0; r < c.length; ++r)            // bounded: length
        if (rows[r].length > 12u && !c.is_null(r)) return true;
    return false;
}

inline bool constant_all_null(const BoltColumn& c) noexcept {
    assert(c.format == ColumnFormat::Constant);
    assert(c.length >= 0);
    if (c.validity == nullptr || c.length == 0) return false;
    for (int64_t r = 0; r < c.length; ++r)            // bounded: length
        if (!c.is_null(r)) return false;
    return true;
}

// Children of a Nested column: the element (List/Map) or the fields (Struct).
inline bool nested_children(const BoltColumn& c, const BoltColumn** kids, uint32_t* n,
                            int64_t* rows) noexcept {
    assert(c.format == ColumnFormat::Nested);
    assert(kids != nullptr && n != nullptr && rows != nullptr);
    if (c.data == nullptr || c.seq_offset <= 0 ||
        c.seq_offset > static_cast<int64_t>(kWireMaxCols)) return false;
    *kids = static_cast<const BoltColumn*>(c.data);
    *n = static_cast<uint32_t>(c.seq_offset);
    if (c.type == BoltType::Struct) {
        *rows = c.length;
        for (uint32_t i = 0; i < *n; ++i)             // bounded: n
            if ((*kids)[i].length != c.length) return false;
        return true;
    }
    if (*n != 1 || c.dict_child == nullptr || c.dict_child->data == nullptr) return false;
    const int32_t* offs = static_cast<const int32_t*>(c.dict_child->data);
    if (offs[0] != 0 || offs[c.length] != (*kids)[0].length) return false;
    *rows = (*kids)[0].length;
    return true;
}

// Plan the encoded (non-Flat, non-VarBinary) kinds.
inline bool column_wire_plan_encoded(const BoltColumn& c, uint32_t depth, ColWire* w) noexcept {
    assert(w != nullptr);
    assert(depth <= kWireMaxNestDepth);
    const size_t tsz = type_size(c.type);
    switch (c.format) {
        case ColumnFormat::Constant: {
            w->b1 = sizeof(c.inline_value);
            w->p32 = c.type_size_bytes;
            if (constant_all_null(c)) { w->b0 = 0; w->flags = kDescFlagAllNull; }
            if (is_sv_string(c.type)) {
                StringView sv; memcpy(&sv, c.inline_value, sizeof(sv));
                if (sv.length > 12u) {
                    if (c.str_overflow_base == nullptr) return false;
                    w->b2 = sv.length;
                    w->moff = sv.ref.offset;
                }
            }
            return c.type_size_bytes <= sizeof(c.inline_value);
        }
        case ColumnFormat::Sequence:
            w->b0 = 0;
            w->b1 = 2 * sizeof(int64_t);
            w->p32 = c.type_size_bytes;
            return true;
        case ColumnFormat::RLE: {
            if (c.length > 0 && (c.dict_child == nullptr || c.dict_child->data == nullptr ||
                                 c.data == nullptr)) return false;
            const int64_t runs = c.length > 0 ? c.dict_child->length : 0;
            w->b1 = static_cast<size_t>(runs) * tsz;
            w->b2 = static_cast<size_t>(runs) * sizeof(int32_t);
            return runs >= 0 && tsz != 0;
        }
        case ColumnFormat::Dictionary: {
            const uint16_t kw = c.type_size_bytes;
            if ((kw != 1 && kw != 2 && kw != 4) || c.dict_child == nullptr) return false;
            if (c.length > 0 && c.data == nullptr) return false;
            w->b1 = static_cast<size_t>(c.length) * kw;
            w->p8 = static_cast<uint8_t>(kw);
            uint32_t v = 0;
            w->b2 = cols_wire_size(c.dict_child, 1, depth + 1, &v);
            return w->b2 != 0;
        }
        case ColumnFormat::Nested: {
            const BoltColumn* kids = nullptr;
            uint32_t n = 0;
            int64_t rows = 0;
            if (!nested_children(c, &kids, &n, &rows)) return false;
            uint32_t v = 0;
            w->b1 = cols_wire_size(kids, n, depth + 1, &v);
            w->b2 = c.type == BoltType::Struct ? 0u
                    : static_cast<size_t>(c.length + 1) * sizeof(int32_t);
            w->p32 = n;
            return w->b1 != 0;
        }
        default:
            return false;
    }
}

// Plan one column's buffers. False on an unsupported or malformed column.
inline bool column_wire_plan(const BoltColumn& c, uint32_t depth, ColWire* w) noexcept {
    assert(w != nullptr);
    assert(c.length >= 0);
    memset(w, 0, sizeof(*w));
    w->version = pair_version(c.type, c.format);
    if (w->version == 0 || depth > kWireMaxNestDepth) return false;
    if (c.logical != BoltLogical::None) w->version = 5;
    w->b0 = c.validity ? validity_bytes(c.length) : 0;
    if (c.format == ColumnFormat::VarBinary) {
        w->b1 = static_cast<size_t>(c.length + 1) * sizeof(int32_t);
        if (c.length > 0 && c.dict_child != nullptr && c.dict_child->data != nullptr) {
            const int32_t payload = static_cast<const int32_t*>(c.dict_child->data)[c.length];
            if (payload < 0) return false;
            w->b2 = static_cast<size_t>(payload);
        } else if (c.length != 0) {
            return false;
        }
        return true;
    }
    if (c.format == ColumnFormat::Flat || c.format == ColumnFormat::View) {
        if (c.length > 0 && c.data == nullptr) return false;
        if (is_sv_string(c.type)) {
            flat_utf8_sizes(c, &w->b1, &w->b2, &w->moff);
            return c.str_overflow_base != nullptr || !sv_rows_spill(c);
        }
        w->b1 = data_buffer_size(c.type, c.length, c.type_size_bytes);
        return true;
    }
    return column_wire_plan_encoded(c, depth, w);
}

// Header + schema + descriptor bytes, aligned: where the data region starts.
BOLT_FORCE_INLINE size_t wire_data_off(uint32_t n) noexcept {
    assert(n <= kWireMaxCols);
    return align_up(kWireHeaderSize + static_cast<size_t>(n) * (kWireSchemaEntrySize +
                    kWireDescSize), kWireAlign);
}

// Exact bytes for the column set, or 0. *version = highest version needed.
inline size_t cols_wire_size(const BoltColumn* cols, uint32_t n, uint32_t depth,
                             uint32_t* version) noexcept {
    assert(version != nullptr);
    assert(cols != nullptr || n == 0);
    if (n > kWireMaxCols || depth > kWireMaxNestDepth) return 0;
    size_t data_bytes = 0;
    *version = kWireVersion;
    for (uint32_t i = 0; i < n; ++i) {               // bounded: n <= kWireMaxCols
        ColWire w;
        if (!column_wire_plan(cols[i], depth, &w)) return 0;
        if (w.version > *version) *version = w.version;
        data_bytes += align_up(w.b0, kWireAlign) + align_up(w.b1, kWireAlign) +
                      align_up(w.b2, kWireAlign);
    }
    return wire_data_off(n) + data_bytes;
}

// Field entry for a column that has no BoltField (a nested child).
inline void child_field(const BoltColumn& c, BoltField* f) noexcept {
    assert(f != nullptr);
    assert(c.length >= 0);
    memset(f, 0, sizeof(*f));
    f->type = c.type;
    f->nullable = c.validity != nullptr;
    if (is_vector(c.type)) {
        const size_t one = embedding_stride_for_type(c.type, 1);
        f->fixed_size = one ? static_cast<uint32_t>(c.type_size_bytes / one) : 0u;
    } else if (c.type == BoltType::FixedSizeBinary) {
        f->fixed_size = c.fixed_width;
    }
}

// Copy a Flat StringView column's rows, rebasing spilled offsets by -moff.
inline void copy_sv_rows(const BoltColumn& c, size_t moff, uint8_t* dst_buf) noexcept {
    assert(dst_buf != nullptr);
    assert(c.data != nullptr || c.length == 0);
    const auto* src_rows = static_cast<const StringView*>(c.data);
    auto* dst_rows = reinterpret_cast<StringView*>(dst_buf);
    for (int64_t r = 0; r < c.length; ++r) {         // bounded: length
        StringView v = src_rows[r];
        if (v.length > 12u)
            v.ref.offset = static_cast<uint32_t>(static_cast<size_t>(v.ref.offset) - moff);
        dst_rows[r] = v;
    }
}

// Copy the encoded kinds' buffers; returns false when a child set fails.
inline bool column_wire_copy_encoded(const BoltColumn& c, const ColWire& w, uint8_t* buf,
                                     const size_t off[3], size_t wr[3],
                                     uint32_t depth) noexcept {
    assert(buf != nullptr && wr != nullptr);
    assert(depth <= kWireMaxNestDepth);
    if (c.format == ColumnFormat::Constant) {
        StringView sv;
        memcpy(buf + off[1], c.inline_value, sizeof(c.inline_value));
        if (w.b2 > 0) {
            memcpy(&sv, c.inline_value, sizeof(sv));
            sv.ref.buf_idx = 0;
            sv.ref.offset = 0;
            memcpy(buf + off[1], &sv, sizeof(sv));
            memcpy(buf + off[2], static_cast<const uint8_t*>(c.str_overflow_base) + w.moff, w.b2);
        }
    } else if (c.format == ColumnFormat::Sequence) {
        detail::write_i64_le(buf + off[1], c.seq_offset);
        detail::write_i64_le(buf + off[1] + 8, c.seq_step);
    } else if (c.format == ColumnFormat::RLE) {
        if (w.b1) memcpy(buf + off[1], c.data, w.b1);
        if (w.b2) memcpy(buf + off[2], c.dict_child->data, w.b2);
    } else if (c.format == ColumnFormat::Dictionary) {
        if (w.b1) memcpy(buf + off[1], c.data, w.b1);
        if (cols_wire_write(c.dict_child, nullptr, 1, c.dict_child->length, buf + off[2],
                            w.b2, depth + 1) != w.b2) return false;
    } else {
        const BoltColumn* kids = nullptr;
        uint32_t n = 0;
        int64_t rows = 0;
        if (!nested_children(c, &kids, &n, &rows)) return false;
        if (cols_wire_write(kids, nullptr, n, rows, buf + off[1], w.b1, depth + 1) != w.b1)
            return false;
        if (w.b2) memcpy(buf + off[2], c.dict_child->data, w.b2);
    }
    wr[1] = w.b1;
    wr[2] = w.b2;
    return true;
}

// Copy one column's buffers to off[0..2]; wr[] = bytes written per span.
inline bool column_wire_copy(const BoltColumn& c, const ColWire& w, uint8_t* buf,
                             const size_t off[3], size_t wr[3], uint32_t depth) noexcept {
    assert(buf != nullptr && wr != nullptr);
    assert(off != nullptr);
    wr[0] = wr[1] = wr[2] = 0;
    if (w.b0 && c.validity) {
        copy_validity_bits(buf + off[0], c.validity, c.validity_offset, c.length);
        wr[0] = w.b0;
    }
    if (c.format == ColumnFormat::VarBinary) {
        if (w.b1 > 0 && c.dict_child != nullptr && c.dict_child->data != nullptr) {
            memcpy(buf + off[1], c.dict_child->data, w.b1);
            wr[1] = w.b1;
        }
        if (w.b2 > 0 && c.data != nullptr) { memcpy(buf + off[2], c.data, w.b2); wr[2] = w.b2; }
        return true;
    }
    if (c.format == ColumnFormat::Flat || c.format == ColumnFormat::View) {
        if (is_sv_string(c.type)) {
            // G2FEAT-308/311: `c` may be a SLICE; b2 is [moff, moff + b2) of
            // the original overflow buffer and every spilled row is rebased.
            if (w.b1 && c.data) { copy_sv_rows(c, w.moff, buf + off[1]); wr[1] = w.b1; }
            if (w.b2 && c.str_overflow_base) {
                memcpy(buf + off[2], static_cast<const uint8_t*>(c.str_overflow_base) + w.moff,
                       w.b2);
                wr[2] = w.b2;
            }
        } else if (w.b1 && c.data) {
            memcpy(buf + off[1], c.data, w.b1);
            wr[1] = w.b1;
        }
        return true;
    }
    return column_wire_copy_encoded(c, w, buf, off, wr, depth);
}

// Header + one schema entry.
inline void wire_write_entry(uint8_t* e, const BoltField& f, const BoltColumn& c) noexcept {
    assert(e != nullptr);
    assert(static_cast<uint8_t>(c.logical) < 16);
    memcpy(e, f.name, kMaxFieldName + 1);
    e[layout::kSchemaTypeOff] = static_cast<uint8_t>(f.type);
    const ColumnFormat fm = c.format == ColumnFormat::View ? ColumnFormat::Flat : c.format;
    e[layout::kSchemaFormatOff] = static_cast<uint8_t>(fm);
    e[layout::kSchemaNullableOff] = static_cast<uint8_t>(
        (f.nullable ? 1u : 0u) |
        (static_cast<uint32_t>(c.logical) << kSchemaLogicalShift));
    // G2ICE-143: byte 67 = decimal scale (0 for every non-decimal column).
    e[layout::kSchemaScaleOff] = c.decimal_scale;
    // Embedding dim / FixedSizeBinary width; 0 for every other type.
    write_u32_le(e + layout::kSchemaFixedOff, f.fixed_size);
}

inline void wire_write_header(uint8_t* buf, uint32_t version, int64_t rows, uint32_t n,
                              uint32_t data_off) noexcept {
    assert(buf != nullptr);
    assert(data_off % kWireAlign == 0);
    memcpy(buf + 0, "BOLT", 4);
    write_u32_le(buf + 4,  version);
    write_u32_le(buf + 8,  kWireFlagLE | kWireFlagAln);
    write_i64_le(buf + 12, rows);
    write_u32_le(buf + 20, n);
    write_u32_le(buf + 24, static_cast<uint32_t>(kWireHeaderSize));
    write_u32_le(buf + 28, data_off);
}

inline void wire_write_desc(uint8_t* d, const size_t off[3], const ColWire& w,
                            ColumnFormat fm) noexcept {
    assert(d != nullptr);
    assert(fm != ColumnFormat::View);
    write_u64_le(d +  0, off[0]); write_u64_le(d +  8, w.b0);
    write_u64_le(d + 16, off[1]); write_u64_le(d + 24, w.b1);
    write_u64_le(d + 32, off[2]); write_u64_le(d + 40, w.b2);
    d[layout::kDescFormatOff] = static_cast<uint8_t>(fm);
    d[kDescFlagsOff] = w.flags;
    d[kDescParam8Off] = w.p8;
    if (w.p32 != 0 && fm != ColumnFormat::Flat && fm != ColumnFormat::VarBinary)
        write_u32_le(d + kDescParam32Off, w.p32);
}

// Serialize a column set into buf[0, cap). `fields` may be null (nested
// children: synthesized entries). Returns bytes written or 0.
inline size_t cols_wire_write(const BoltColumn* cols, const BoltField* fields, uint32_t n,
                              int64_t rows, uint8_t* buf, size_t cap,
                              uint32_t depth) noexcept {
    assert(buf != nullptr || cap == 0);
    assert(cols != nullptr || n == 0);
    uint32_t version = 0;
    const size_t total = cols_wire_size(cols, n, depth, &version);
    if (total == 0 || total > cap || buf == nullptr) return 0;
    const size_t schema_off = kWireHeaderSize;
    const size_t desc_off = schema_off + static_cast<size_t>(n) * kWireSchemaEntrySize;
    const uint32_t data_off = static_cast<uint32_t>(wire_data_off(n));
    // G2ICE-141: zero ONLY the metadata region; every data byte is copied or
    // explicitly zeroed below.
    memset(buf, 0, data_off);
    wire_write_header(buf, version, rows, n, data_off);
    size_t cursor = data_off;
    for (uint32_t i = 0; i < n; ++i) {               // bounded: n <= kWireMaxCols
        const BoltColumn& c = cols[i];
        BoltField cf;
        if (fields == nullptr) child_field(c, &cf);
        wire_write_entry(buf + schema_off + i * kWireSchemaEntrySize,
                         fields ? fields[i] : cf, c);
        ColWire w;
        if (!column_wire_plan(c, depth, &w)) return 0;
        const size_t aln[3] = {align_up(w.b0, kWireAlign), align_up(w.b1, kWireAlign),
                               align_up(w.b2, kWireAlign)};
        const size_t off[3] = {cursor, cursor + aln[0], cursor + aln[0] + aln[1]};
        cursor += aln[0] + aln[1] + aln[2];
        assert(cursor <= total);
        const ColumnFormat fm = c.format == ColumnFormat::View ? ColumnFormat::Flat : c.format;
        wire_write_desc(buf + desc_off + i * kWireDescSize, off, w, fm);
        size_t wr[3];
        if (!column_wire_copy(c, w, buf, off, wr, depth)) return 0;
        for (int k = 0; k < 3; ++k) zero_wire_gap(buf, off[k], wr[k], aln[k]);
    }
    return total;
}

}  // namespace detail

// ===========================================================================
// Public C++ API
// ===========================================================================

namespace detail {
BOLT_FORCE_INLINE const BoltColumn* batch_cols(const BoltBatch* b) noexcept {
    assert(b != nullptr);
    return b->num_cols ? &b->col(0) : nullptr;
}
}  // namespace detail

/// Exact byte size the serialized form needs, or 0 if batch is not
/// serializable (unsupported column format / type, too many columns).
inline size_t bolt_wire_size(const BoltBatch* b) noexcept {
    assert(b != nullptr);
    assert(b->num_rows >= 0);
    if (b->num_cols > kWireMaxCols) return 0;
    uint32_t version = 0;
    return detail::cols_wire_size(detail::batch_cols(b), b->num_cols, 0, &version);
}

/// Serialize into out_buf. Returns bytes written, or 0 on failure.
inline size_t bolt_wire_serialize(const BoltBatch* b,
                                  void* out_buf,
                                  size_t buf_capacity) noexcept {
    assert(b != nullptr);
    assert(out_buf != nullptr || buf_capacity == 0);
    if (out_buf == nullptr) return 0;
    if (b->num_cols > kWireMaxCols) return 0;
    return detail::cols_wire_write(detail::batch_cols(b), b->schema.fields, b->num_cols,
                                   b->num_rows, static_cast<uint8_t*>(out_buf),
                                   buf_capacity, 0);
}

namespace detail {

// Materialize a [p+off, off+len) wire span into a BoltColumn buffer pointer.
//   kView == false  → copy into `arena` (owning, aligned) — bolt_wire_deserialize.
//   kView == true   → ALIAS the source bytes in place (zero-copy) — bolt_wire_view.
// The view variant returns a mutable pointer into a const buffer: the const_cast
// is sound ONLY because every view consumer treats the column as read-only.
// Buffers are 64 B aligned relative to the blob start, so a blob placed on a
// 64 B boundary (a frame in a mapped file) yields aligned typed arrays.
// `len == 0` ⇒ nullptr.
template <bool kView>
BOLT_FORCE_INLINE void* wire_span(const uint8_t* p, uint64_t off, uint64_t len,
                                  Arena* arena) noexcept {
    if (len == 0u) return nullptr;
    if (kView) return const_cast<void*>(static_cast<const void*>(p + off));
    return arena->copy_into(p + off, len, kWireAlign);
}

// One parsed descriptor.
struct WireDesc {
    uint64_t     o[3], l[3];
    ColumnFormat fm;
    uint8_t      flags, p8;
    uint32_t     p32;
};

template <bool kView>
inline bool parse_cols(const uint8_t* p, size_t len, BoltBatch* out, Arena* arena,
                       uint32_t depth) noexcept;

// Every dictionary code is below the dictionary length (a branch-free max).
inline bool dict_codes_in_range(const uint8_t* keys, uint32_t kw, int64_t rows,
                                int64_t dict_len) noexcept {
    assert(kw == 1 || kw == 2 || kw == 4);
    assert(keys != nullptr || rows == 0);
    uint32_t mx = 0;
    for (int64_t r = 0; r < rows; ++r) {              // bounded: rows
        uint32_t k = 0;
        memcpy(&k, keys + static_cast<size_t>(r) * kw, kw);
        mx = k > mx ? k : mx;
    }
    return rows == 0 || static_cast<int64_t>(mx) < dict_len;
}

// Dictionary values (one-column sub-blob in b2) and Nested children
// (sub-blob in b1, element offsets in b2 for List/Map).
template <bool kView>
inline bool build_child_set(const uint8_t* p, const WireDesc& d, const BoltField& f,
                            int64_t rows, Arena* arena, uint32_t depth, BoltColumn* c) noexcept {
    assert(c != nullptr && arena != nullptr);
    assert(d.fm == ColumnFormat::Dictionary || d.fm == ColumnFormat::Nested);
    BoltBatch sub;
    if (d.fm == ColumnFormat::Dictionary) {
        const uint32_t kw = d.p8;
        if ((kw != 1 && kw != 2 && kw != 4) || d.l[1] != static_cast<uint64_t>(rows) * kw)
            return false;
        if (!parse_cols<kView>(p + d.o[2], d.l[2], &sub, arena, depth + 1) || sub.num_cols != 1)
            return false;
        c->data = wire_span<kView>(p, d.o[1], d.l[1], arena);
        c->type_size_bytes = static_cast<uint16_t>(kw);
        c->dict_child = &sub.columns[0][0];
        if (sub.columns[0][0].type != f.type) return false;
        return dict_codes_in_range(static_cast<const uint8_t*>(c->data), kw, rows,
                                   sub.num_rows);
    }
    if (!parse_cols<kView>(p + d.o[1], d.l[1], &sub, arena, depth + 1) ||
        sub.num_cols != d.p32 || sub.num_cols == 0) return false;
    c->data = sub.columns[0];
    c->seq_offset = sub.num_cols;
    c->type_size_bytes = 0;
    if (f.type == BoltType::Struct) return d.l[2] == 0 && sub.num_rows == rows;
    if (sub.num_cols != 1 || d.l[2] != static_cast<uint64_t>(rows + 1) * sizeof(int32_t))
        return false;
    auto* offs = static_cast<int32_t*>(wire_span<kView>(p, d.o[2], d.l[2], arena));
    if (offs == nullptr || offs[0] != 0 || offs[rows] != sub.num_rows) return false;
    for (int64_t r = 0; r < rows; ++r)               // bounded: rows
        if (offs[r + 1] < offs[r]) return false;
    BoltColumn* oc = arena->allocate_array<BoltColumn>(1);
    if (oc == nullptr) return false;
    *oc = BoltColumn::make_flat(offs, nullptr, rows + 1, BoltType::Int32);
    c->dict_child = oc;
    return true;
}

// Rebuild the encoded kinds (v5).
template <bool kView>
inline bool build_encoded(const uint8_t* p, const WireDesc& d, const BoltField& f,
                          int64_t rows, Arena* arena, uint32_t depth, BoltColumn* c) noexcept {
    assert(c != nullptr && arena != nullptr);
    assert(depth <= kWireMaxNestDepth);
    if (d.fm == ColumnFormat::Constant) {
        if (d.l[1] != sizeof(c->inline_value) || d.p32 > sizeof(c->inline_value)) return false;
        memcpy(c->inline_value, p + d.o[1], sizeof(c->inline_value));
        c->type_size_bytes = static_cast<uint16_t>(d.p32);
        // A long string constant carries exactly its bytes in b2, at offset 0.
        StringView sv;
        memcpy(&sv, c->inline_value, sizeof(sv));
        const bool spilled = is_sv_string(f.type) && sv.length > 12u;
        if (spilled ? (d.l[2] != sv.length || sv.ref.offset != 0) : d.l[2] != 0) return false;
        if (d.l[2]) c->str_overflow_base = wire_span<kView>(p, d.o[2], d.l[2], arena);
        if (d.flags & kDescFlagAllNull) {
            c->validity = static_cast<uint8_t*>(arena->allocate(validity_bytes(rows) + 1, 64));
            if (c->validity == nullptr) return false;
            memset(c->validity, 0, validity_bytes(rows) + 1);
            c->stats.all_valid = false;
        }
        return true;
    }
    if (d.fm == ColumnFormat::Sequence) {
        if (d.l[1] != 16 || d.l[0] != 0) return false;
        c->seq_offset = read_i64_le(p + d.o[1]);
        c->seq_step = read_i64_le(p + d.o[1] + 8);
        c->type_size_bytes = static_cast<uint16_t>(d.p32);
        return true;
    }
    if (d.fm == ColumnFormat::RLE) {
        const size_t tsz = type_size(f.type);
        const uint64_t runs = d.l[2] / sizeof(int32_t);
        if (d.l[2] % sizeof(int32_t) != 0 || d.l[1] != runs * tsz) return false;
        if ((runs == 0) != (rows == 0)) return false;
        c->data = wire_span<kView>(p, d.o[1], d.l[1], arena);
        auto* ends = static_cast<int32_t*>(wire_span<kView>(p, d.o[2], d.l[2], arena));
        if (runs != 0 && (ends == nullptr || ends[runs - 1] != rows)) return false;
        int64_t prev = 0;
        for (uint64_t i = 0; i < runs; ++i) {         // bounded: runs
            if (ends[i] <= prev) return false;        // strictly increasing, > 0
            prev = ends[i];
        }
        BoltColumn* rc = arena->allocate_array<BoltColumn>(1);
        if (rc == nullptr) return false;
        *rc = BoltColumn::make_flat(ends, nullptr, static_cast<int64_t>(runs), BoltType::Int32);
        c->dict_child = rc;
        return true;
    }
    return build_child_set<kView>(p, d, f, rows, arena, depth, c);
}

// Rebuild one column from its schema entry and descriptor.
template <bool kView>
inline bool build_column(const uint8_t* p, const WireDesc& d, const BoltField& f,
                         BoltLogical lg, int64_t rows, Arena* arena, uint32_t depth,
                         BoltColumn* out) noexcept {
    assert(out != nullptr);
    assert(rows >= 0);
    uint8_t* validity = nullptr;
    if (d.l[0]) {
        if (d.l[0] < validity_bytes(rows)) return false;
        validity = static_cast<uint8_t*>(wire_span<kView>(p, d.o[0], d.l[0], arena));
        if (validity == nullptr) return false;
    }
    if (d.fm == ColumnFormat::VarBinary) {
        // G2ICE-76: offsets and payload alias `buf` in view mode; only the
        // child descriptor is allocated.
        int32_t* offs = static_cast<int32_t*>(wire_span<kView>(p, d.o[1], d.l[1], arena));
        uint8_t* data = static_cast<uint8_t*>(wire_span<kView>(p, d.o[2], d.l[2], arena));
        if ((d.l[1] && offs == nullptr) || (d.l[2] && data == nullptr)) return false;
        *out = BoltColumn::make_var_binary(data, validity, offs, rows, f.type, arena);
        out->logical = lg;
        return out->format == ColumnFormat::VarBinary;
    }
    BoltColumn c = BoltColumn::make_empty();
    c.length = rows;
    c.format = d.fm;
    c.type = f.type;
    c.arena = arena;      // descriptors live here; data aliases buf in a view
    c.validity = validity;
    c.stats.all_valid = (validity == nullptr);
    c.decimal_scale = f.decimal_scale;
    c.logical = lg;
    if (d.fm == ColumnFormat::Flat) {
        if (is_vector(f.type)) {
            // Embedding stride = dim * bytes_per_elt (Wave 9.4 z.5).
            const size_t stride = embedding_stride_for_type(f.type, f.fixed_size);
            if (stride == 0u || stride > UINT16_MAX) return false;
            c.type_size_bytes = static_cast<uint16_t>(stride);
        } else {
            c.type_size_bytes = static_cast<uint16_t>(
                is_sv_string(f.type) ? sizeof(StringView) : type_size(f.type));
        }
        if (f.type == BoltType::FixedSizeBinary) {
            if (f.fixed_size > type_size(BoltType::FixedSizeBinary)) return false;
            c.fixed_width = static_cast<uint8_t>(f.fixed_size);
        }
        // b1 holds exactly rows x stride bytes (rows <= kWireMaxRows).
        if (d.l[1] != static_cast<uint64_t>(rows) * c.type_size_bytes) return false;
        if (d.l[1]) {
            c.data = wire_span<kView>(p, d.o[1], d.l[1], arena);
            if (!c.data) return false;
        }
        // Flat strings (v3+): offsets were rebased at serialize, so b2 is the
        // overflow buffer every spilled ref.offset resolves against.
        if (is_sv_string(f.type) && d.l[2]) {
            c.str_overflow_base = wire_span<kView>(p, d.o[2], d.l[2], arena);
            if (!c.str_overflow_base) return false;
        }
        *out = c;
        return true;
    }
    if (!build_encoded<kView>(p, d, f, rows, arena, depth, &c)) return false;
    *out = c;
    return true;
}

// Read and bounds-check descriptor i.
inline bool read_desc(const uint8_t* dp, size_t buf_len, bool aligned, WireDesc* d) noexcept {
    assert(dp != nullptr && d != nullptr);
    assert(buf_len > 0);
    for (int k = 0; k < 3; ++k) {
        d->o[k] = read_u64_le(dp + 16 * k);
        d->l[k] = read_u64_le(dp + 16 * k + 8);
        if (d->o[k] > buf_len || d->l[k] > buf_len - d->o[k]) return false;
        // Typed spans (int32 offsets / run ends, keys) are read in place.
        if (aligned && (d->o[k] & (kWireAlign - 1)) != 0) return false;
    }
    d->fm = static_cast<ColumnFormat>(dp[layout::kDescFormatOff]);
    d->flags = dp[kDescFlagsOff];
    d->p8 = dp[kDescParam8Off];
    d->p32 = read_u32_le(dp + kDescParam32Off);
    return true;
}

// Read schema entry i into f; validates the (type, format) pair.
inline bool read_entry(const uint8_t* e, uint32_t version, BoltField* f,
                       ColumnFormat* fm, BoltLogical* lg) noexcept {
    assert(e != nullptr && f != nullptr && fm != nullptr && lg != nullptr);
    assert(version >= 1 && version <= kWireVersionLatest);
    memset(f, 0, sizeof(*f));
    memcpy(f->name, e, kMaxFieldName);
    f->name[kMaxFieldName] = '\0';
    f->type     = static_cast<BoltType>(e[layout::kSchemaTypeOff]);
    const uint8_t nb = e[layout::kSchemaNullableOff];
    f->nullable = (nb & 1u) != 0 || (version < 5 && nb != 0);
    *lg = version >= 5 ? static_cast<BoltLogical>((nb & kSchemaLogicalMask) >>
                                                  kSchemaLogicalShift)
                       : BoltLogical::None;
    f->decimal_scale = e[layout::kSchemaScaleOff];   // G2ICE-143
    f->fixed_size = read_u32_le(e + layout::kSchemaFixedOff);
    *fm = static_cast<ColumnFormat>(e[layout::kSchemaFormatOff]);
    const uint32_t need = pair_version(f->type, *fm);
    if (need == 0 || need > version || *fm == ColumnFormat::View) return false;
    // Embedding columns require a non-zero dim (the stride).
    return !(is_vector(f->type) && f->fixed_size == 0u);
}

// Validate the header; fills the counts.
inline bool read_header(const uint8_t* p, size_t len, uint32_t* version, int64_t* rows,
                        uint32_t* n, uint32_t* data_off, bool* aligned) noexcept {
    assert(version != nullptr && rows != nullptr);
    assert(n != nullptr && data_off != nullptr);
    if (p == nullptr || len < kWireHeaderSize) return false;
    if (memcmp(p, "BOLT", 4) != 0) return false;
    // v1..v4 only widened the legal (type, format) pairs; v5 (B3) adds the
    // encoded kinds and the descriptor tail. A reader refuses a pair newer
    // than the header's version.
    *version = read_u32_le(p + 4);
    if (*version < 1u || *version > kWireVersionLatest) return false;
    const uint32_t flags = read_u32_le(p + 8);
    if (!(flags & kWireFlagLE)) return false;
    *aligned = (flags & kWireFlagAln) != 0;
    if (*version >= kWireVersionEncoded && !*aligned) return false;
    *rows = read_i64_le(p + 12);
    *n = read_u32_le(p + 20);
    *data_off = read_u32_le(p + 28);
    if (*n > kWireMaxCols || *rows < 0 || *rows > kWireMaxRows) return false;
    if (read_u32_le(p + 24) != kWireHeaderSize) return false;
    if ((*data_off & (kWireAlign - 1)) != 0 || *data_off > len) return false;
    return kWireHeaderSize + static_cast<size_t>(*n) * (kWireSchemaEntrySize + kWireDescSize)
           <= len;
}

// Validate row-dependent storage before any ownership allocation. Compressed
// columns can precede a malformed Flat column; building in descriptor order
// would otherwise expand its null bitmap before noticing the impossible rows.
inline bool row_storage_valid(const uint8_t* p, const WireDesc& d,
                              const BoltField& f, int64_t rows) noexcept {
    assert(p != nullptr);
    assert(rows >= 0 && rows <= kWireMaxRows);
    if (d.l[0] && d.l[0] < validity_bytes(rows)) return false;
    if (d.fm == ColumnFormat::Flat) {
        const size_t stride = is_vector(f.type) ? embedding_stride_for_type(f.type, f.fixed_size)
            : (is_sv_string(f.type) ? sizeof(StringView) : type_size(f.type));
        return stride != 0 && stride <= UINT16_MAX &&
               d.l[1] == static_cast<uint64_t>(rows) * stride;
    }
    if (d.fm == ColumnFormat::VarBinary) {
        if (d.l[1] != static_cast<uint64_t>(rows + 1) * sizeof(int32_t)) return false;
        int32_t previous = 0;
        for (int64_t r = 0; r <= rows; ++r) { // bounded by the validated offsets span
            int32_t offset = 0;
            memcpy(&offset, p + d.o[1] + static_cast<size_t>(r) * sizeof(offset), sizeof(offset));
            if (offset < previous || static_cast<uint64_t>(offset) > d.l[2]) return false;
            previous = offset;
        }
        return static_cast<uint64_t>(previous) == d.l[2];
    }
    if (d.fm == ColumnFormat::Dictionary)
        return (d.p8 == 1 || d.p8 == 2 || d.p8 == 4) &&
               d.l[1] == static_cast<uint64_t>(rows) * d.p8;
    if (d.fm == ColumnFormat::Nested && f.type != BoltType::Struct)
        return d.l[2] == static_cast<uint64_t>(rows + 1) * sizeof(int32_t);
    return true;
}

inline bool preflight_cols(const uint8_t* p, size_t len, uint32_t depth,
                           uint64_t* null_bytes_remaining) noexcept {
    assert(null_bytes_remaining != nullptr);
    assert(depth <= kWireMaxNestDepth + 1);
    uint32_t version = 0, n = 0, data_off = 0;
    int64_t rows = 0;
    bool aligned = false;
    if (depth > kWireMaxNestDepth ||
        !read_header(p, len, &version, &rows, &n, &data_off, &aligned)) return false;
    const size_t desc_off = kWireHeaderSize + static_cast<size_t>(n) * kWireSchemaEntrySize;
    assert(desc_off <= len);
    for (uint32_t i = 0; i < n; ++i) {
        BoltField f;
        ColumnFormat fm;
        BoltLogical lg;
        WireDesc d;
        if (!read_entry(p + kWireHeaderSize + i * kWireSchemaEntrySize, version, &f, &fm, &lg) ||
            !read_desc(p + desc_off + i * kWireDescSize, len, aligned, &d) || d.fm != fm ||
            !row_storage_valid(p, d, f, rows)) return false;
        if (fm == ColumnFormat::Constant && (d.flags & kDescFlagAllNull)) {
            const uint64_t bytes = validity_bytes(rows) + 1;
            if (bytes > *null_bytes_remaining) return false;
            *null_bytes_remaining -= bytes;
        }
        if (fm == ColumnFormat::Nested || fm == ColumnFormat::Dictionary) {
            const uint32_t child = fm == ColumnFormat::Nested ? 1 : 2;
            if (!preflight_cols(p + d.o[child], d.l[child], depth + 1, null_bytes_remaining)) return false;
        }
    }
    return true;
}

template <bool kView>
inline bool parse_cols(const uint8_t* p, size_t len, BoltBatch* out, Arena* arena,
                       uint32_t depth) noexcept {
    assert(out != nullptr);
    assert(depth <= kWireMaxNestDepth + 1);
    uint32_t version = 0, n = 0, data_off = 0;
    int64_t rows = 0;
    bool aligned = false;
    uint64_t null_bytes_remaining = kWireMaxNullBitmapBytes;
    if (depth > kWireMaxNestDepth ||
        !read_header(p, len, &version, &rows, &n, &data_off, &aligned) ||
        (depth == 0 && !preflight_cols(p, len, depth, &null_bytes_remaining)))
        return false;
    BoltBatch::init_empty(out);
    // G2FEAT-47: right-size the column arrays (sets num_cols + arena).
    if (!BoltBatch::alloc_columns(out, arena, n)) return false;
    out->num_rows = rows;
    out->schema.num_fields = n;
    const size_t desc_off = kWireHeaderSize + static_cast<size_t>(n) * kWireSchemaEntrySize;
    for (uint32_t i = 0; i < n; ++i) {               // bounded: n <= kWireMaxCols
        BoltField& f = out->schema.fields[i];
        ColumnFormat fm;
        BoltLogical lg;
        if (!read_entry(p + kWireHeaderSize + i * kWireSchemaEntrySize, version, &f, &fm, &lg))
            return false;
        WireDesc d;
        if (!read_desc(p + desc_off + i * kWireDescSize, len, aligned, &d) || d.fm != fm)
            return false;
        BoltColumn c;
        if (!build_column<kView>(p, d, f, lg, rows, arena, depth, &c)) return false;
        out->columns[0][i] = c;
        out->columns[1][i] = c;
        if (fm == ColumnFormat::Constant) {      // data is a self-pointer
            out->columns[0][i].data = out->columns[0][i].inline_value;
            out->columns[1][i].data = out->columns[1][i].inline_value;
        }
    }
    return true;
}

}  // namespace detail

/// Shared parse for bolt_wire_deserialize (kView=false, copies into `arena`) and
/// bolt_wire_view (kView=true, zero-copy alias into `buf`). Returns false on any
/// validation error.
template <bool kView>
inline bool bolt_wire_parse(const void* buf, size_t buf_len,
                            BoltBatch* out, Arena* arena) noexcept {
    assert(out != nullptr);
    assert(kView || arena != nullptr);
    return detail::parse_cols<kView>(static_cast<const uint8_t*>(buf), buf_len, out, arena, 0);
}


/// Parse a wire blob, copying every column buffer into `arena`. The result owns
/// its data and outlives `buf`. Returns false on any validation error.
inline bool bolt_wire_deserialize(const void* buf, size_t buf_len,
                                  BoltBatch* out, Arena* arena) noexcept {
    return bolt_wire_parse<false>(buf, buf_len, out, arena);
}

/// Zero-copy parse: `out`'s column buffers ALIAS into `buf` (no allocation, no
/// arena). The result is READ-ONLY and valid only while `buf` stays alive and
/// unmodified — consumers must copy out before `buf` is recycled, and must
/// memcpy rather than assume typed alignment. Halves the apply-path memory
/// traffic vs deserialize+apply (one copy into the sink instead of two). Returns
/// false on any validation error. See MarbleDB's mt_apply_entry kBatch path.
// G2FEAT-47: `arena` is required now — dynamic BoltBatch.columns[2] need arena
// storage for the (small) per-column DESCRIPTOR array even in view mode. The
// column DATA still aliases `buf` zero-copy; only the descriptors are allocated
// (same arena/lifetime the deserialize fallback uses). Passing nullptr with a
// non-empty batch asserts in alloc_columns.
inline bool bolt_wire_view(const void* buf, size_t buf_len,
                           BoltBatch* out, Arena* arena) noexcept {
    return bolt_wire_parse<true>(buf, buf_len, out, arena);
}

}  // namespace wire
}  // namespace bolt

// ===========================================================================
// extern "C" surface (FFI / lakehouse interop). Thin wrappers.
// ===========================================================================

BOLT_C_API size_t bolt_wire_size_c(const void* batch) noexcept;
BOLT_C_API size_t bolt_wire_serialize_c(const void* batch,
                                        void* out_buf,
                                        size_t cap) noexcept;
BOLT_C_API int    bolt_wire_deserialize_c(const void* buf, size_t len,
                                          void* out_batch, void* arena) noexcept;

// Header-only inline definitions for the C shim (safe: header is single-TU
// per consumer; BOLT_C_API gives external linkage but these are marked inline
// so the ODR is satisfied when included in multiple TUs).
inline size_t bolt_wire_size_c(const void* batch) noexcept {
    assert(batch != nullptr);
    return bolt::wire::bolt_wire_size(
        static_cast<const bolt::BoltBatch*>(batch));
}
inline size_t bolt_wire_serialize_c(const void* batch,
                                    void* out_buf,
                                    size_t cap) noexcept {
    assert(batch != nullptr);
    assert(out_buf != nullptr || cap == 0);
    return bolt::wire::bolt_wire_serialize(
        static_cast<const bolt::BoltBatch*>(batch), out_buf, cap);
}
inline int bolt_wire_deserialize_c(const void* buf, size_t len,
                                   void* out_batch, void* arena) noexcept {
    assert(out_batch != nullptr);
    assert(arena != nullptr);
    return bolt::wire::bolt_wire_deserialize(
        buf, len,
        static_cast<bolt::BoltBatch*>(out_batch),
        static_cast<bolt::Arena*>(arena)) ? 1 : 0;
}
