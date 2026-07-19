// bolt_wire.h — Self-describing flat serialization for BoltBatch.
//
// RULES: No exceptions. No RTTI. No smart pointers. No std::string. No std::vector.
// All functions noexcept. Fixed caps, >=2 asserts per function, functions <=70 lines.
//
// Scope:
//   - Format::Flat columns: numeric types in BOLT_NUMERIC_TYPES plus Bool.
//   - Format::VarBinary columns: BoltType::Utf8 / Binary / Symbol — payload
//     is `(offsets[len+1], raw bytes)`. Per-row slice = pool[off[i]..off[i+1]).
//     Added at version 2 (Pass-Wiring 2026-05-01) to carry the unified-
//     memtable Field::* paths (kText / kKeyword / kJson / kStored) through
//     the WAL kBatch path.
//   - Other formats (Constant / Dictionary / Sequence / View / RLE /
//     BitPacked / FrameOfRef) must be materialized by the caller before
//     serialize — otherwise serialize returns 0.
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
//   |              nullable u8, pad -> 72 bytes }    |
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

namespace bolt {
namespace wire {

// ===========================================================================
// Constants
// ===========================================================================

inline constexpr uint32_t kWireMagic    = 0x544C4F42u;  // 'BOLT' LE
inline constexpr uint32_t kWireVersion  = 3u;            // bumped: Flat Utf8 support
inline constexpr uint32_t kWireFlagLE   = 1u << 0;
inline constexpr uint32_t kWireFlagAln  = 1u << 1;

inline constexpr size_t kWireHeaderSize      = 32;
inline constexpr size_t kWireSchemaEntrySize = 72;
inline constexpr size_t kWireDescSize        = 56;
inline constexpr size_t kWireAlign           = 64;

// Hard cap mirrors kMaxBatchColumns so we never underestimate bounds.
// G2FEAT-47: the wire codec keeps the historical 256-col cap (see
// kMaxFixedColumns) — decoupled from the in-memory kMaxBatchColumns (1024) so
// the fixed serialize scratch + WireStream framing stay compact.
inline constexpr uint32_t kWireMaxCols = kMaxFixedColumns;

// ===========================================================================
// Internal helpers
// ===========================================================================

namespace detail {

BOLT_FORCE_INLINE size_t align_up(size_t x, size_t a) noexcept {
    assert(a > 0);
    assert((a & (a - 1)) == 0);
    return (x + a - 1) & ~(a - 1);
}

BOLT_FORCE_INLINE bool is_supported_type(BoltType t) noexcept {
    if (t == BoltType::Bool)         return true;
    if (t == BoltType::Utf8)         return true;
    if (t == BoltType::Binary)       return true;
    if (t == BoltType::Symbol)       return true;
    if (t == BoltType::Embedding)    return true;
    if (t == BoltType::EmbeddingF16) return true;
    if (t == BoltType::EmbeddingU8)  return true;
    if (t == BoltType::EmbeddingI8)  return true;
    auto v = static_cast<uint8_t>(t);
    return v >= static_cast<uint8_t>(BoltType::Int8)
        && v <= static_cast<uint8_t>(BoltType::Float64);
}

// True if (`type`, `format`) is a legal pair for the wire format. v3:
//   Format::Flat       + numeric / Bool / Embedding{,F16,U8,I8} / Utf8
//   Format::VarBinary  + Utf8 / Binary / Symbol
//
// Flat + Utf8 was excluded through v2 — `bolt_wire_size()` returned 0 for
// any batch carrying a Flat-format Utf8 column (StringView row array +
// separate str_overflow_base spill buffer), which meant `marbledb::put()`
// (a direct, unconditional bolt_wire_size/serialize caller) failed cleanly
// for every Utf8 column real Parquet ingestion decodes — parquet_read
// always produces Flat StringViews, never VarBinary. Fixed at v3: see
// `flat_utf8_sizes` below for the b1(StringView array)/b2(spilled bytes)
// split, mirroring VarBinary's existing b1(offsets)/b2(payload) shape.
// Binary/Symbol stay VarBinary-only (unchanged) — out of this fix's scope.
BOLT_FORCE_INLINE bool is_supported_format_pair(BoltType t,
                                                 ColumnFormat f) noexcept {
    if (f == ColumnFormat::Flat) {
        return is_supported_type(t)
            && t != BoltType::Binary && t != BoltType::Symbol;
    }
    if (f == ColumnFormat::VarBinary) {
        return t == BoltType::Utf8 || t == BoltType::Binary
            || t == BoltType::Symbol;
    }
    return false;
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
    if (t == BoltType::Bool) return static_cast<size_t>((n + 7) / 8);
    size_t tsz = type_size(t);
    return tsz * static_cast<size_t>(n);
}

// Sizes the (b1, b2) pair for a Flat-format Utf8 column: b1 is the fixed
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
    assert(c.type == BoltType::Utf8 && c.format == ColumnFormat::Flat);
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

// Compute the total buffer-region size (summed, each aligned to 64) and
// report per-column buffer lengths. Returns 0 on unsupported column.
inline size_t collect_column_sizes(const BoltBatch* b,
                                   size_t* out_b0_len,
                                   size_t* out_b1_len,
                                   size_t* out_b2_len) noexcept {
    assert(b != nullptr);
    assert(out_b0_len != nullptr);

    size_t total = 0;
    const uint32_t n = b->num_cols;
    for (uint32_t i = 0; i < n && i < kWireMaxCols; ++i) {
        const BoltColumn& c = b->col(i);
        if (!is_supported_format_pair(c.type, c.format)) return 0;

        const size_t v_len = c.validity ? validity_bytes(c.length) : 0;
        size_t d_len = 0;
        size_t s_len = 0;

        if (c.format == ColumnFormat::VarBinary) {
            // b1 = (length+1) Int32 offsets; b2 = `offsets[length]` payload bytes.
            d_len = static_cast<size_t>(c.length + 1) * sizeof(int32_t);
            if (c.length > 0 && c.dict_child != nullptr &&
                c.dict_child->data != nullptr) {
                const int32_t* offs =
                    static_cast<const int32_t*>(c.dict_child->data);
                const int32_t total_payload = offs[c.length];
                if (total_payload < 0) return 0;
                s_len = static_cast<size_t>(total_payload);
            } else if (c.length == 0) {
                s_len = 0;
            } else {
                return 0;            // length > 0 but no offsets
            }
        } else if (c.format == ColumnFormat::Flat && c.type == BoltType::Utf8) {
            size_t min_off_unused = 0;
            flat_utf8_sizes(c, &d_len, &s_len, &min_off_unused);
            if (s_len > 0 && c.str_overflow_base == nullptr) return 0;  // malformed
        } else {
            // Flat numeric / Bool / Embedding. Embedding uses the 3-arg
            // overload so the runtime-dynamic stride (dim * 4) is taken
            // from `c.type_size_bytes` rather than the 0 sentinel in
            // `kTypeSize[]`.
            d_len = data_buffer_size(c.type, c.length, c.type_size_bytes);
        }

        out_b0_len[i] = v_len;
        out_b1_len[i] = d_len;
        out_b2_len[i] = s_len;

        total += detail::align_up(v_len, kWireAlign);
        total += detail::align_up(d_len, kWireAlign);
        total += detail::align_up(s_len, kWireAlign);
    }
    return total;
}

}  // namespace detail

// ===========================================================================
// Public C++ API
// ===========================================================================

/// Exact byte size the serialized form needs, or 0 if batch is not
/// serializable (unsupported column format / type, too many columns).
inline size_t bolt_wire_size(const BoltBatch* b) noexcept {
    assert(b != nullptr);
    assert(b->num_cols <= kWireMaxCols);

    if (b->num_cols > kWireMaxCols) return 0;

    size_t b0[kWireMaxCols], b1[kWireMaxCols], b2[kWireMaxCols];
    memset(b0, 0, sizeof(b0));
    memset(b1, 0, sizeof(b1));
    memset(b2, 0, sizeof(b2));

    const size_t data_bytes = detail::collect_column_sizes(b, b0, b1, b2);
    if (data_bytes == 0 && b->num_cols > 0) {
        // Only "no supported columns" is an error here. A legitimately
        // empty (zero-row, zero-byte) set of columns would still produce
        // data_bytes == 0 but should succeed — so additionally verify
        // that every column is supported.
        for (uint32_t i = 0; i < b->num_cols; ++i) {
            const BoltColumn& c = b->col(i);
            if (!detail::is_supported_format_pair(c.type, c.format)) return 0;
        }
    }

    size_t off = kWireHeaderSize;
    off += static_cast<size_t>(b->num_cols) * kWireSchemaEntrySize;
    off += static_cast<size_t>(b->num_cols) * kWireDescSize;
    off = detail::align_up(off, kWireAlign);
    off += data_bytes;
    return off;
}

/// Serialize into out_buf. Returns bytes written, or 0 on failure.
inline size_t bolt_wire_serialize(const BoltBatch* b,
                                  void* out_buf,
                                  size_t buf_capacity) noexcept {
    assert(b != nullptr);
    assert(out_buf != nullptr || buf_capacity == 0);

    if (out_buf == nullptr) return 0;
    if (b->num_cols > kWireMaxCols) return 0;

    size_t b0[kWireMaxCols], b1[kWireMaxCols], b2[kWireMaxCols];
    size_t utf8_min_off[kWireMaxCols];   // only meaningful for Flat+Utf8 cols
    memset(b0, 0, sizeof(b0));
    memset(b1, 0, sizeof(b1));
    memset(b2, 0, sizeof(b2));
    memset(utf8_min_off, 0, sizeof(utf8_min_off));

    // Collect sizes + validate supported formats.
    for (uint32_t i = 0; i < b->num_cols; ++i) {
        const BoltColumn& c = b->col(i);
        if (!detail::is_supported_format_pair(c.type, c.format)) return 0;
        b0[i] = c.validity ? detail::validity_bytes(c.length) : 0;
        if (c.format == ColumnFormat::VarBinary) {
            b1[i] = static_cast<size_t>(c.length + 1) * sizeof(int32_t);
            if (c.length > 0 && c.dict_child != nullptr &&
                c.dict_child->data != nullptr) {
                const int32_t* offs =
                    static_cast<const int32_t*>(c.dict_child->data);
                const int32_t payload = offs[c.length];
                if (payload < 0) return 0;
                b2[i] = static_cast<size_t>(payload);
            } else if (c.length == 0) {
                b2[i] = 0;
            } else {
                return 0;
            }
        } else if (c.format == ColumnFormat::Flat && c.type == BoltType::Utf8) {
            detail::flat_utf8_sizes(c, &b1[i], &b2[i], &utf8_min_off[i]);
            if (b2[i] > 0 && c.str_overflow_base == nullptr) return 0;  // malformed
        } else {
            b1[i] = detail::data_buffer_size(c.type, c.length,
                                              c.type_size_bytes);
            b2[i] = 0;
        }
    }

    const size_t total = bolt_wire_size(b);
    if (total == 0 || total > buf_capacity) return 0;

    uint8_t* buf = static_cast<uint8_t*>(out_buf);
    memset(buf, 0, total);

    // --- Header ---
    memcpy(buf + 0, "BOLT", 4);
    detail::write_u32_le(buf + 4,  kWireVersion);
    detail::write_u32_le(buf + 8,  kWireFlagLE | kWireFlagAln);
    detail::write_i64_le(buf + 12, b->num_rows);
    detail::write_u32_le(buf + 20, b->num_cols);
    const uint32_t schema_off = static_cast<uint32_t>(kWireHeaderSize);
    const uint32_t desc_off   = schema_off +
        static_cast<uint32_t>(b->num_cols) * kWireSchemaEntrySize;
    const uint32_t data_off_u = static_cast<uint32_t>(
        detail::align_up(desc_off + b->num_cols * kWireDescSize, kWireAlign));
    detail::write_u32_le(buf + 24, schema_off);
    detail::write_u32_le(buf + 28, data_off_u);

    // --- Schema entries ---
    for (uint32_t i = 0; i < b->num_cols; ++i) {
        uint8_t* e = buf + schema_off + i * kWireSchemaEntrySize;
        const BoltField& f = b->schema.fields[i];
        memcpy(e, f.name, kMaxFieldName + 1);
        e[64] = static_cast<uint8_t>(f.type);
        e[65] = static_cast<uint8_t>(b->col(i).format);
        e[66] = f.nullable ? 1u : 0u;
        e[67] = 0u;                                // reserved
        // bytes 68..71: fixed_size (Embedding dim or FixedSizeBinary
        // width). Zero for all other types — old readers see zeroes
        // and ignore the field; new readers consult it for vector
        // round-trip.
        detail::write_u32_le(e + 68, f.fixed_size);
    }

    // --- Column descriptors + data copy ---
    size_t cursor = data_off_u;
    for (uint32_t i = 0; i < b->num_cols; ++i) {
        uint8_t* d = buf + desc_off + i * kWireDescSize;
        const BoltColumn& c = b->col(i);

        const size_t off0 = cursor;                      cursor += detail::align_up(b0[i], kWireAlign);
        const size_t off1 = cursor;                      cursor += detail::align_up(b1[i], kWireAlign);
        const size_t off2 = cursor;                      cursor += detail::align_up(b2[i], kWireAlign);
        assert(cursor <= total);

        detail::write_u64_le(d +  0, off0); detail::write_u64_le(d +  8, b0[i]);
        detail::write_u64_le(d + 16, off1); detail::write_u64_le(d + 24, b1[i]);
        detail::write_u64_le(d + 32, off2); detail::write_u64_le(d + 40, b2[i]);
        d[48] = static_cast<uint8_t>(c.format);

        if (b0[i] && c.validity) memcpy(buf + off0, c.validity, b0[i]);
        if (c.format == ColumnFormat::VarBinary) {
            // b1 = offsets array; b2 = payload bytes.
            if (b1[i] > 0 && c.dict_child != nullptr &&
                c.dict_child->data != nullptr) {
                memcpy(buf + off1, c.dict_child->data, b1[i]);
            }
            if (b2[i] > 0 && c.data != nullptr) {
                memcpy(buf + off2, c.data, b2[i]);
            }
        } else if (c.format == ColumnFormat::Flat && c.type == BoltType::Utf8) {
            // b1 = StringView row array; b2 = spilled bytes. G2FEAT-308/311:
            // `c` may be a SLICE of a larger column (a chunked/windowed
            // batch), so spilled rows' `ref.offset` can be far from 0 —
            // b2 was sized as [utf8_min_off[i], utf8_min_off[i]+b2[i]) by
            // flat_utf8_sizes, NOT [0, b2[i]). Copy that span (not a [0,..)
            // prefix) and REBASE each spilled row's offset by -utf8_min_off[i]
            // so it resolves correctly against the copied span; inline rows
            // (length <= 12, no ref.offset) pass through untouched. When
            // utf8_min_off[i] == 0 (the common whole-column case) every
            // rebased offset equals the original — no behavior change there.
            if (b1[i] && c.data) {
                const auto* src_rows = static_cast<const StringView*>(c.data);
                auto* dst_rows = reinterpret_cast<StringView*>(buf + off1);
                const size_t moff = utf8_min_off[i];
                for (int64_t r = 0; r < c.length; ++r) {
                    StringView v = src_rows[r];
                    if (v.length > 12u) {
                        v.ref.offset = static_cast<uint32_t>(
                            static_cast<size_t>(v.ref.offset) - moff);
                    }
                    dst_rows[r] = v;
                }
            }
            if (b2[i] && c.str_overflow_base) {
                memcpy(buf + off2,
                      static_cast<const uint8_t*>(c.str_overflow_base) + utf8_min_off[i],
                      b2[i]);
            }
        } else {
            if (b1[i] && c.data) memcpy(buf + off1, c.data, b1[i]);
        }
    }

    return total;
}

namespace detail {

// Materialize a [p+off, off+len) wire span into a BoltColumn buffer pointer.
//   kView == false  → copy into `arena` (owning, aligned) — bolt_wire_deserialize.
//   kView == true   → ALIAS the source bytes in place (zero-copy) — bolt_wire_view.
// The view variant returns a mutable pointer into a const buffer: the const_cast
// is sound ONLY because every view consumer treats the column as read-only and
// copies OUT of it (e.g. MarbleDB's memtable apply memcpy's into the zone). The
// aliased bytes carry the source's alignment, NOT kWireAlign — consumers must
// memcpy rather than reinterpret as a typed array. `len == 0` ⇒ nullptr.
template <bool kView>
BOLT_FORCE_INLINE void* wire_span(const uint8_t* p, uint64_t off, uint64_t len,
                                  Arena* arena) noexcept {
    if (len == 0u) return nullptr;
    if (kView) return const_cast<void*>(static_cast<const void*>(p + off));
    return arena->copy_into(p + off, len, kWireAlign);
}

}  // namespace detail

/// Shared parse for bolt_wire_deserialize (kView=false, copies into `arena`) and
/// bolt_wire_view (kView=true, zero-copy alias into `buf`; `arena` may be null).
/// Returns false on any validation error.
template <bool kView>
inline bool bolt_wire_parse(const void* buf, size_t buf_len,
                            BoltBatch* out, Arena* arena) noexcept {
    assert(out != nullptr);
    assert(kView || arena != nullptr);

    if (buf == nullptr || buf_len < kWireHeaderSize) return false;
    const uint8_t* p = static_cast<const uint8_t*>(buf);

    // --- Header validation ---
    if (memcmp(p, "BOLT", 4) != 0) return false;
    const uint32_t version = detail::read_u32_le(p + 4);
    // v1/v2/v3 are wire-compatible at the header / descriptor level; each
    // bump only widens which (type, format) pairs are legal, never the
    // on-disk shape. v2 added VarBinary; v3 adds Flat Utf8 (StringView
    // row array in b1 + spilled overflow in b2). Readers accept all three
    // — an old reader compiled against a lower kWireVersion simply fails
    // `is_supported_format_pair` for the newer pair it doesn't know about,
    // which is the intentional break for that column's writer.
    if (version != 1u && version != 2u && version != 3u) return false;
    const uint32_t flags = detail::read_u32_le(p + 8);
    if (!(flags & kWireFlagLE)) return false;
    const int64_t  num_rows     = detail::read_i64_le(p + 12);
    const uint32_t num_cols     = detail::read_u32_le(p + 20);
    const uint32_t schema_off   = detail::read_u32_le(p + 24);
    const uint32_t data_off     = detail::read_u32_le(p + 28);
    if (num_cols > kWireMaxCols)            return false;
    if (num_rows < 0)                       return false;
    if (schema_off != kWireHeaderSize)      return false;
    if ((data_off & (kWireAlign - 1)) != 0) return false;

    const size_t desc_off = static_cast<size_t>(schema_off) +
        static_cast<size_t>(num_cols) * kWireSchemaEntrySize;
    if (desc_off + static_cast<size_t>(num_cols) * kWireDescSize > buf_len) return false;
    if (data_off > buf_len) return false;

    BoltBatch::init_empty(out);
    // G2FEAT-47: right-size the column arrays to the wire's num_cols before the
    // per-column loop writes columns[epoch][i]. Sets num_cols + arena.
    if (!BoltBatch::alloc_columns(out, arena, num_cols)) return false;
    out->num_rows = num_rows;
    out->schema.num_fields = num_cols;

    // --- Per-column parse ---
    for (uint32_t i = 0; i < num_cols; ++i) {
        const uint8_t* e = p + schema_off + i * kWireSchemaEntrySize;
        BoltField& f = out->schema.fields[i];
        memset(&f, 0, sizeof(f));
        memcpy(f.name, e, kMaxFieldName);
        f.name[kMaxFieldName] = '\0';
        f.type     = static_cast<BoltType>(e[64]);
        f.nullable = (e[66] != 0);
        // bytes 68..71 carry fixed_size (Embedding dim / FixedSizeBinary
        // width). Old writers wrote zero into the trailing pad, so
        // pre-vector schemas decode as fixed_size = 0 — matches the
        // default-initialised BoltField.
        f.fixed_size = detail::read_u32_le(e + 68);
        const ColumnFormat schema_fmt =
            static_cast<ColumnFormat>(e[65]);
        if (!detail::is_supported_format_pair(f.type, schema_fmt)) return false;
        // Embedding columns require a non-zero dim; reject corrupt /
        // pre-vector payloads that label a column Embedding without
        // carrying the stride. Same for the multi-precision variants.
        if (is_vector(f.type) && f.fixed_size == 0u) return false;

        const uint8_t* d = p + desc_off + i * kWireDescSize;
        const uint64_t o0 = detail::read_u64_le(d +  0);
        const uint64_t l0 = detail::read_u64_le(d +  8);
        const uint64_t o1 = detail::read_u64_le(d + 16);
        const uint64_t l1 = detail::read_u64_le(d + 24);
        const uint64_t o2 = detail::read_u64_le(d + 32);
        const uint64_t l2 = detail::read_u64_le(d + 40);
        const ColumnFormat fm = static_cast<ColumnFormat>(d[48]);
        if (fm != schema_fmt) return false;
        if (!detail::is_supported_format_pair(f.type, fm)) return false;
        if (o0 + l0 > buf_len || o1 + l1 > buf_len ||
            o2 + l2 > buf_len) return false;

        // Validity is shared across both formats.
        uint8_t* validity = nullptr;
        bool all_valid = true;
        if (l0) {
            void* v = detail::wire_span<kView>(p, o0, l0, arena);
            if (!v) return false;
            validity = static_cast<uint8_t*>(v);
            all_valid = false;
        }

        if (fm == ColumnFormat::Flat) {
            BoltColumn c = BoltColumn::make_empty();
            c.length  = num_rows;
            c.format  = ColumnFormat::Flat;
            c.type    = f.type;
            // Embedding columns store a runtime stride = dim *
            // bytes_per_elt (4 for f32 / 2 for f16 / 1 for u8|i8);
            // everything else falls back to the static `kTypeSize[]`
            // lookup. Wave 9.4 z.5: use `embedding_stride_for_type` so
            // every precision variant is decoded uniformly.
            if (is_vector(f.type)) {
                const size_t stride =
                    embedding_stride_for_type(f.type, f.fixed_size);
                if (stride == 0u || stride > UINT16_MAX) return false;
                c.type_size_bytes = static_cast<uint16_t>(stride);
            } else {
                c.type_size_bytes = static_cast<uint16_t>(type_size(f.type));
            }
            c.arena   = arena;   // nullptr for a view (non-owning, read-only)
            c.validity = validity;
            c.stats.all_valid = all_valid;
            if (l1) {
                c.data = detail::wire_span<kView>(p, o1, l1, arena);
                if (!c.data) return false;
            }
            // Flat Utf8 (v3+): b2 is the spilled-overflow byte buffer any
            // >12-byte row's StringView::ref.offset resolves against.
            // No re-basing needed HERE: `bolt_wire_serialize` already
            // rebased every row's `ref.offset` (subtracting the source
            // column's minimum spilled offset — see `flat_utf8_sizes` /
            // G2FEAT-308/311) so offsets are relative to THIS b2 span's
            // start, wherever in the ORIGINAL buffer that span came from —
            // just point str_overflow_base at the resolved span.
            if (f.type == BoltType::Utf8 && l2) {
                c.str_overflow_base = detail::wire_span<kView>(p, o2, l2, arena);
                if (!c.str_overflow_base) return false;
            }
            out->columns[0][i] = c;
            out->columns[1][i] = c;
        } else {
            // VarBinary needs make_var_binary, which allocates a child offset
            // column from `arena`. A zero-copy view has no arena, so it declines
            // var columns — the caller (e.g. mt_apply_entry) falls back to the
            // copying deserialize path. Fixed-width (Flat) tables — the P6 target
            // and the common ingest case — view fully zero-copy above.
            if (kView) return false;
            // VarBinary: rebuild via make_var_binary over the offsets + data
            // buffers (arena-copied for deserialize; aliased for a view).
            int32_t* offs = nullptr;
            uint8_t* data = nullptr;
            if (l1) {
                offs = static_cast<int32_t*>(detail::wire_span<kView>(p, o1, l1, arena));
                if (!offs) return false;
            }
            if (l2) {
                data = static_cast<uint8_t*>(detail::wire_span<kView>(p, o2, l2, arena));
                if (!data) return false;
            }
            // num_rows == 0 → make_var_binary with a nullptr offsets is OK
            // because the inner check skips offsets/data when total_rows == 0.
            BoltColumn c = BoltColumn::make_var_binary(
                data, validity, offs, num_rows, f.type, arena);
            if (c.format != ColumnFormat::VarBinary) {
                // make_var_binary rejected the input — that's a corrupt
                // wire payload (e.g. offsets[len] > 0 but data == nullptr).
                return false;
            }
            out->columns[0][i] = c;
            out->columns[1][i] = c;
        }
    }
    return true;
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
