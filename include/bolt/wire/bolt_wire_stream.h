// bolt_wire_stream.h — Append-mode writer for the Bolt wire format.
//
// Purpose: `bolt::wire::bolt_wire_serialize` (bolt_wire.h) requires the whole
// BoltBatch materialised in memory and copies each column buffer into the
// output blob. MarbleDB compaction builds each output column from many input
// runs and would rather stream bytes directly into the output buffer —
// without first gathering every row into one BoltBatch.
//
// This header provides a three-call sequence that produces a byte-identical
// Bolt wire file (as consumed by `bolt::wire::bolt_wire_deserialize`) while
// writing each column's data buffers in place:
//
//     wire_stream_begin_file(s, out, cap, schema, ncols);
//     for (col c in schema) wire_stream_append_column(s, col, raw, raw_len);
//     size_t written = wire_stream_finalize(s);
//
// Tiger Style: POD + free functions, noexcept, no smart pointers, no heap,
// ≥2 asserts per function, functions ≤70 lines.
//
// Layout recap (matches bolt_wire.h exactly — no new wire bits):
//   - 32-byte header at offset 0
//   - schema block (num_cols * kWireSchemaEntrySize)
//   - descriptor block (num_cols * kWireDescSize)
//   - 64-byte-aligned data region: per column, (b0 validity, b1 data, b2 utf8)
//     each aligned up to 64 bytes.
//
// Streaming contract:
//   - `begin_file` writes the header + schema block immediately, reserves
//     the descriptor block zeroed-out, and aligns the write cursor to the
//     64-byte data-region boundary.
//   - Each `append_column` writes {validity?, data} into the current cursor,
//     records the absolute offsets + lengths it landed at, and advances the
//     cursor (with per-buffer 64-byte alignment). Cols are written in the
//     schema order; out-of-order appends are rejected.
//   - `finalize` patches the descriptor block and writes `data_offset` into
//     the header. Returns the total byte count (== final cursor).
//
// Utf8 limitation:
//   Phase 1 `bolt::wire::bolt_wire_deserialize` rejects Utf8 (offsets +
//   string blob reconstruction are not yet metadata-complete on a Flat
//   BoltColumn). This writer matches — Utf8 columns cause `begin_file` /
//   `append_column` to sticky-flag the stream with `ok = false`.
//   TODO(bolt_wire): once Utf8 serialisation lands upstream, extend
//   `wire_stream_append_column` to accept a (offsets, data) pair and write
//   both buffers per column.
//
// Non-Flat formats (Constant, Sequence, Dictionary, View, RLE, BitPacked)
// are likewise unsupported here — they're unsupported by `bolt_wire.h`
// itself. Callers must materialise to Flat first.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/wire/bolt_wire.h"

namespace bolt {

/// Streaming writer state. Caller owns `out` — the stream never reallocates.
struct WireStream {
    uint8_t* out;
    size_t   capacity;
    size_t   pos;            // current write cursor (also total bytes when finalised)

    uint32_t num_cols;       // set at begin_file
    uint32_t cols_written;   // append progress
    int64_t  num_rows;       // captured from first column's length (or 0 if none)
    bool     num_rows_set;   // have we seen a column to pin num_rows?
    bool     ok;             // sticky error flag

    // Fixed layout anchors (computed once at begin_file).
    uint32_t schema_off;     // == kWireHeaderSize
    uint32_t desc_off;       // schema_off + num_cols * kWireSchemaEntrySize
    uint32_t data_off;       // align_up(desc_off + num_cols * kWireDescSize, 64)

    // Per-column absolute offsets + lengths captured during append.
    // Indexed by column index; filled into the descriptor block at finalize.
    // G2FEAT-47: sized to the wire cap (kMaxFixedColumns=256), not the raised
    // in-memory kMaxBatchColumns — keeps WireStream inside its 16 KB budget and
    // matches the kWireMaxCols guards below.
    uint64_t col_b0_off[wire::kWireMaxCols];
    uint64_t col_b0_len[wire::kWireMaxCols];
    uint64_t col_b1_off[wire::kWireMaxCols];
    uint64_t col_b1_len[wire::kWireMaxCols];
    uint64_t col_b2_off[wire::kWireMaxCols];
    uint64_t col_b2_len[wire::kWireMaxCols];
    uint8_t  col_format[wire::kWireMaxCols];  // always Flat in Phase 1
};

// WireStream is ~12.5 KB at kWireMaxCols=kMaxFixedColumns=256 (NOT
// kMaxBatchColumns=1024 — the wire format deliberately stays on the
// compact 256-column cap; see the G2FEAT-47 note on col_b0_off[] etc.
// above). Pass it by pointer always. This guard makes an accidental
// by-value parameter or return a hard error.
static_assert(sizeof(WireStream) <= 16u * 1024u,
              "WireStream layout blew past 16 KB — audit kWireMaxCols");
static_assert(alignof(WireStream) >= alignof(uint64_t),
              "WireStream must be 8-byte aligned for u64 offset arrays");

namespace wire_stream_detail {

BOLT_FORCE_INLINE size_t align_up_64(size_t x) noexcept {
    return (x + wire::kWireAlign - 1u) & ~static_cast<size_t>(wire::kWireAlign - 1u);
}

BOLT_FORCE_INLINE bool type_is_supported(BoltType t) noexcept {
    // Mirror wire::detail::is_supported_type but also reject Utf8 (Phase 1
    // limitation — see header doc).
    if (t == BoltType::Utf8) return false;
    if (t == BoltType::Bool) return true;
    const auto v = static_cast<uint8_t>(t);
    return v >= static_cast<uint8_t>(BoltType::Int8)
        && v <= static_cast<uint8_t>(BoltType::Float64);
}

// Delegate to the canonical helper in bolt_wire.h so the layout stays in
// lock-step with the non-streaming serializer (no divergence opportunity).
BOLT_FORCE_INLINE size_t validity_bytes(int64_t n) noexcept {
    return wire::detail::validity_bytes(n);
}

}  // namespace wire_stream_detail

/// Begin a multi-column streaming file. `out` must be writable for at least
/// `capacity` bytes and caller-owned. `schema_fields` points to `num_columns`
/// `BoltField` records (same type BoltBatch uses — field name + type +
/// nullable). Returns false on bad arguments and leaves `s->ok = false`.
inline bool wire_stream_begin_file(WireStream* BOLT_RESTRICT s,
                                   uint8_t* BOLT_RESTRICT out,
                                   size_t capacity,
                                   const BoltField* BOLT_RESTRICT schema_fields,
                                   uint32_t num_columns) noexcept {
    assert(s != nullptr);
    assert(out != nullptr || capacity == 0);
    assert(num_columns <= wire::kWireMaxCols);
    if (s == nullptr)                          return false;
    if (out == nullptr)                        return false;
    if (num_columns > wire::kWireMaxCols)      return false;
    if (schema_fields == nullptr && num_columns > 0) return false;

    memset(s, 0, sizeof(*s));
    s->out      = out;
    s->capacity = capacity;
    s->ok       = true;
    s->num_cols = num_columns;

    // Validate every field's type up front so later appends have a clean
    // contract to rely on.
    for (uint32_t i = 0; i < num_columns; ++i) {
        if (!wire_stream_detail::type_is_supported(schema_fields[i].type)) {
            s->ok = false;
            return false;
        }
    }

    s->schema_off = static_cast<uint32_t>(wire::kWireHeaderSize);
    s->desc_off   = s->schema_off + num_columns * wire::kWireSchemaEntrySize;
    s->data_off   = static_cast<uint32_t>(
        wire_stream_detail::align_up_64(
            s->desc_off + num_columns * wire::kWireDescSize));

    if (s->data_off > capacity) { s->ok = false; return false; }

    // Zero header + schema + descriptor regions (descriptors patched later).
    memset(out, 0, s->data_off);

    // Header: magic, version, flags, num_rows placeholder (patched in
    // finalize from num_rows pinned by the first column append), num_cols,
    // schema_off, data_off.
    memcpy(out + 0, "BOLT", 4);
    wire::detail::write_u32_le(out +  4, wire::kWireVersion);
    wire::detail::write_u32_le(out +  8, wire::kWireFlagLE | wire::kWireFlagAln);
    wire::detail::write_i64_le(out + 12, 0);  // num_rows patched in finalize
    wire::detail::write_u32_le(out + 20, num_columns);
    wire::detail::write_u32_le(out + 24, s->schema_off);
    wire::detail::write_u32_le(out + 28, s->data_off);

    // Schema block: one 72-byte entry per column.
    for (uint32_t i = 0; i < num_columns; ++i) {
        uint8_t* e = out + s->schema_off + i * wire::kWireSchemaEntrySize;
        const BoltField& f = schema_fields[i];
        memcpy(e, f.name, kMaxFieldName + 1);
        e[64] = static_cast<uint8_t>(f.type);
        e[65] = static_cast<uint8_t>(ColumnFormat::Flat);  // Phase 1 is Flat-only
        e[66] = f.nullable ? 1u : 0u;
        // bytes 67..71 left zero
    }

    s->pos = s->data_off;  // data cursor starts at data region
    return true;
}

/// Append one Flat column's data. `col` provides length + type + validity
/// pointer (validity may be null when the column is all-valid). `raw_data`
/// is `raw_len` bytes of the primitive payload; its length MUST match
/// `type_size(col->type) * col->length` for fixed-width types, or the
/// caller-declared Utf8 layout once Utf8 is supported.
///
/// Columns are appended in schema order. Out-of-order or duplicate appends
/// set the sticky error flag and return false.
inline bool wire_stream_append_column(WireStream* BOLT_RESTRICT s,
                                      const BoltColumn* BOLT_RESTRICT col,
                                      const void* BOLT_RESTRICT raw_data,
                                      size_t raw_len) noexcept {
    assert(s != nullptr);
    assert(col != nullptr);
    assert(raw_data != nullptr || raw_len == 0);
    if (s == nullptr || col == nullptr)                 return false;
    if (!s->ok)                                         return false;
    if (s->cols_written >= s->num_cols)                 { s->ok = false; return false; }
    if (col->format != ColumnFormat::Flat)              { s->ok = false; return false; }
    if (!wire_stream_detail::type_is_supported(col->type)) { s->ok = false; return false; }
    if (col->length < 0)                                { s->ok = false; return false; }
    const size_t expect = wire::detail::data_buffer_size(col->type, col->length);
    if (raw_data == nullptr && raw_len != 0)            { s->ok = false; return false; }
    if (raw_len != expect)                              { s->ok = false; return false; }

    // Pin num_rows against the first column; subsequent columns must agree.
    if (!s->num_rows_set) {
        s->num_rows = col->length;
        s->num_rows_set = true;
    } else if (col->length != s->num_rows) {
        s->ok = false;
        return false;
    }

    const uint32_t i = s->cols_written;
    const size_t v_len = col->validity ? wire_stream_detail::validity_bytes(col->length) : 0;
    const size_t d_len = raw_len;
    const size_t s_len = 0;  // Utf8 not yet supported (see header doc)

    // Reserve the three aligned slots, recording absolute offsets.
    size_t cursor = s->pos;
    const size_t off0 = cursor; cursor += wire_stream_detail::align_up_64(v_len);
    const size_t off1 = cursor; cursor += wire_stream_detail::align_up_64(d_len);
    const size_t off2 = cursor; cursor += wire_stream_detail::align_up_64(s_len);
    if (cursor > s->capacity) { s->ok = false; return false; }

    // Zero alignment padding so the buffer stays reproducible (matches the
    // non-streaming serializer, which memsets the full buffer first).
    if (v_len > 0) {
        memcpy(s->out + off0, col->validity, v_len);
        const size_t pad = wire_stream_detail::align_up_64(v_len) - v_len;
        if (pad > 0) memset(s->out + off0 + v_len, 0, pad);
    }
    if (d_len > 0) {
        memcpy(s->out + off1, raw_data, d_len);
        const size_t pad = wire_stream_detail::align_up_64(d_len) - d_len;
        if (pad > 0) memset(s->out + off1 + d_len, 0, pad);
    }
    // s_len == 0 for now; when Utf8 lands, a third buffer copy goes here.

    s->col_b0_off[i] = off0; s->col_b0_len[i] = v_len;
    s->col_b1_off[i] = off1; s->col_b1_len[i] = d_len;
    s->col_b2_off[i] = off2; s->col_b2_len[i] = s_len;
    s->col_format[i] = static_cast<uint8_t>(ColumnFormat::Flat);

    s->pos = cursor;
    ++s->cols_written;
    return true;
}

/// Patch descriptor block + header num_rows. Returns total bytes written,
/// or 0 on error (sticky `ok=false` or partial column count).
inline size_t wire_stream_finalize(WireStream* BOLT_RESTRICT s) noexcept {
    assert(s != nullptr);
    assert(s->out != nullptr || s->capacity == 0);
    assert(s->cols_written <= s->num_cols);
    if (s == nullptr || !s->ok)                 return 0;
    if (s->cols_written != s->num_cols)         { s->ok = false; return 0; }

    // Patch num_rows in the header (offset 12).
    wire::detail::write_i64_le(s->out + 12,
                               s->num_rows_set ? s->num_rows : 0);

    // Write the descriptor block in one pass.
    for (uint32_t i = 0; i < s->num_cols; ++i) {
        uint8_t* d = s->out + s->desc_off + i * wire::kWireDescSize;
        wire::detail::write_u64_le(d +  0, s->col_b0_off[i]);
        wire::detail::write_u64_le(d +  8, s->col_b0_len[i]);
        wire::detail::write_u64_le(d + 16, s->col_b1_off[i]);
        wire::detail::write_u64_le(d + 24, s->col_b1_len[i]);
        wire::detail::write_u64_le(d + 32, s->col_b2_off[i]);
        wire::detail::write_u64_le(d + 40, s->col_b2_len[i]);
        d[48] = s->col_format[i];
        // bytes 49..55 already zeroed by begin_file's memset.
    }
    return s->pos;
}

}  // namespace bolt
