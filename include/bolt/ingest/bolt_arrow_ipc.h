// bolt_arrow_ipc.h — Arrow IPC STREAM writer (G2ARROW-10, widened G2ARROW-20).
//
// Writes the Arrow IPC streaming format — an encapsulated Schema message
// followed by RecordBatch messages and the end-of-stream marker — such
// that `pyarrow.ipc.open_stream()` reads the bytes and every value
// matches. The flatbuffer message headers are encoded FROM THE SPEC by a
// minimal bottom-up builder in bolt_arrow_ipc.cpp (the same from-spec
// discipline as the thrift-compact parquet page-index decoder): no
// flatbuffers library dependency, no code generation.
//
// TYPE MAPPING — deliberately the SAME per-value resolution the Arrow C
// Data export (bolt/bolt_arrow.h) uses: fixed-width columns hand their
// Flat payload through verbatim (per-type width via bolt::type_size —
// Int64/Float64/Date32/Decimal128 differ only in byte width), Utf8/Binary
// both resolve every row through bolt::arrow::detail::var_at (inline
// StringView / spilled str_overflow_base / VarBinary offsets all covered
// — var_at does not care whether the logical type is Utf8 or Binary, only
// about the physical layout), Bool is bit-packed from bolt's byte-packed
// storage the same way bolt_arrow.h's pack_bool does, Decimal128's 16-byte
// payload is ALREADY the little-endian two's-complement layout Arrow
// wants (bolt stores it that way natively), and the validity bitmap is
// bolt's Arrow-shape LSB-first bitmap. One mapping, two transports — or
// they drift.
//
// HONEST SCOPE (fail closed at open, never a misencoded stream):
//   supported column types: Int64, Float64, Utf8, Bool, Date32, Binary,
//   Decimal128.
//   Everything else — Decimal64 (bolt_arrow.h's own C-Data export has no
//   Arrow format string for it either; extending only this writer would
//   invent a mapping the C-Data export doesn't share, breaking the "one
//   mapping, two transports" rule above) and nested (List/Struct/Map) —
//   is rejected by arrow_ipc_open() with `false`. Nested types need a
//   genuinely different open() shape (the schema is written once at
//   open(), before any batch or child column is known, so a child
//   field's own type must be describable at open() time too) — tracked
//   as a follow-up rather than half-built here.
//   Decimal128 additionally requires `decimal_scales[c]` at open() — the
//   schema's Decimal type table carries (precision, scale) and both must
//   be known before any batch is written, so there is no later point to
//   source it from; arrow_ipc_open() fails closed if a Decimal128 column
//   is declared without one, or with scale > 38. Precision is always 38
//   (the Decimal128 maximum — bolt does not track a narrower precision
//   per column, matching bolt_arrow.h's C-Data export).
//   Column formats: Flat/View for fixed-width; Flat/View/VarBinary for
//   Utf8/Binary (exactly what var_at can resolve). Dictionary/RLE/
//   Constant must be materialized by the caller first.
//
// Tiger Style: caller-owned writer struct (~40 KB — arena/heap it, a
// stack BoltBatch-sized local is this repo's documented stack-overflow
// trap), bounded columns, no allocation, no exceptions; errors are
// `false` returns and the writer latches failed state.

#pragma once

#include <cstdint>
#include <cstdio>

#include "bolt/bolt_types.h"

namespace bolt { struct BoltBatch; struct BoltColumn; }

namespace bolt::ingest {

// Bounded surface. 64 columns matches the Arrow C-Data export cap.
inline constexpr std::uint16_t kIpcMaxCols   = 64;
inline constexpr std::uint32_t kIpcNameCap   = 64;   // per-column name bytes
inline constexpr std::uint32_t kIpcFbCap     = 1u << 15;  // flatbuffer scratch

struct ArrowIpcWriter {
    std::FILE*    f;                       // borrowed; caller closes
    std::uint16_t n_cols;
    std::uint16_t open;                    // 1 between open() and close()
    std::uint16_t failed;                  // latched on any write error
    std::uint16_t _pad;
    std::uint16_t col_types[kIpcMaxCols];  // BoltType per column
    std::uint8_t  decimal_scale[kIpcMaxCols]; // Decimal128 scale; 0 otherwise
    char          names[kIpcMaxCols][kIpcNameCap];
    std::uint8_t  fb[kIpcFbCap];           // flatbuffer build scratch
};

/// Begin a stream: validate the schema (fail closed on any unsupported
/// type) and write the Schema message to `f`. `types` are BoltType
/// values; `names[i]` may be nullptr (a "cN" name is synthesized —
/// pyarrow requires non-null field names). `decimal_scales[i]` (0..38)
/// supplies the Decimal128 scale for any column whose type is
/// BoltType::Decimal128; it may be nullptr only when no column is
/// Decimal128 (arrow_ipc_open fails closed otherwise, since the schema's
/// Decimal type table must carry a scale and open() is the only point
/// that knows it before any batch is written).
bool arrow_ipc_open(ArrowIpcWriter* w, std::FILE* f,
                    const BoltType* types, const char* const* names,
                    std::uint16_t n_cols,
                    const std::uint8_t* decimal_scales = nullptr) noexcept;

/// Append one RecordBatch message from the batch's read-epoch columns.
/// Column count and types must match the open schema; row values are
/// resolved with the C-Data mapping (see header comment). Returns false
/// (and latches failure) on any mismatch or I/O error.
bool arrow_ipc_write_batch(ArrowIpcWriter* w,
                           const BoltBatch* batch) noexcept;

/// Write the end-of-stream marker and flush. The FILE* stays open
/// (borrowed). Returns false if the stream previously failed or the
/// final write fails; the writer is closed either way.
bool arrow_ipc_close(ArrowIpcWriter* w) noexcept;

}  // namespace bolt::ingest
