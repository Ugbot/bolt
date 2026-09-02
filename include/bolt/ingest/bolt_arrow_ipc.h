// bolt_arrow_ipc.h — Arrow IPC STREAM writer (G2ARROW-10).
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
// Flat payload through verbatim, Utf8 resolves every row through
// bolt::arrow::detail::var_at (inline StringView / spilled
// str_overflow_base / VarBinary offsets all covered), and the validity
// bitmap is bolt's Arrow-shape LSB-first bitmap. One mapping, two
// transports — or they drift.
//
// HONEST SCOPE (fail closed at open, never a misencoded stream):
//   supported column types: Int64, Float64, Utf8.
//   Everything else — Bool (needs bit-packing), Date32, Decimal,
//   Binary, nested — is rejected by arrow_ipc_open() with `false`.
//   Column formats: Flat/View for fixed-width; Flat/View/VarBinary for
//   Utf8 (exactly what var_at can resolve). Dictionary/RLE/Constant
//   must be materialized by the caller first.
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
    char          names[kIpcMaxCols][kIpcNameCap];
    std::uint8_t  fb[kIpcFbCap];           // flatbuffer build scratch
};

/// Begin a stream: validate the schema (fail closed on any unsupported
/// type) and write the Schema message to `f`. `types` are BoltType
/// values; `names[i]` may be nullptr (a "cN" name is synthesized —
/// pyarrow requires non-null field names).
bool arrow_ipc_open(ArrowIpcWriter* w, std::FILE* f,
                    const BoltType* types, const char* const* names,
                    std::uint16_t n_cols) noexcept;

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
