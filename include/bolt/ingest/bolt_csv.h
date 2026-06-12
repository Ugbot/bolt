#pragma once
// ---------------------------------------------------------------------------
// bolt::ingest — CSV loader into BoltBatch columns.
//
// Compiled library. Delegates numeric field parsing to bolt::parse and uses
// the SWAR helpers in bolt_port.h for delimiter / newline scanning. No heap
// outside the caller-supplied Arena. No exceptions.
// ---------------------------------------------------------------------------
#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"

#include <cstddef>
#include <cstdint>

namespace bolt {
namespace ingest {

inline constexpr uint32_t kCsvMaxCols = kMaxBatchColumns;

struct CsvSchema {
    BoltType col_types[kCsvMaxCols];
    // Per-column scale. Only meaningful for Decimal128 columns: the parser
    // multiplies parsed values up to (or truncates down to) this scale.
    // Range: [0..38]. Ignored for all other types. Zero-init friendly.
    uint8_t  col_scales[kCsvMaxCols];
    uint32_t num_cols;
    char     delimiter;   // typically ';' or ','
    bool     has_header;  // skip first line if true
    // If true, Decimal128 fields with MORE fractional digits than the
    // declared scale cause parse_csv to fail. If false (default), extra
    // fractional digits are silently truncated (DuckDB CSV default).
    bool     strict_decimal_scale;
};

// Parse a contiguous CSV buffer into columns. The buffer is owned by the
// caller; column data is allocated from `arena`. Returns false on:
//   - schema mismatch (mid-row column count != schema.num_cols)
//   - arena OOM
//   - unsupported BoltType in schema
// Supported types: Int32 (int10ths interpretation), Int64, Date32
// (ISO YYYY-MM-DD), Decimal128 (per-column scale), Utf8.
//
// Inner loop uses bolt_swar_find_byte_u64 for delimiter / line scanning
// and bolt::parse::parse_int10th for numeric fields. No floats.
bool parse_csv(const char* BOLT_RESTRICT buf, size_t buf_len,
               const CsvSchema& schema,
               Arena*           arena,
               BoltBatch*       out_batch) noexcept;

}  // namespace ingest

struct Scheduler;   // bolt_scheduler.h — fwd-declared to keep this header light

namespace ingest {

// W3 — parallel CSV parse over line-aligned ~1 MiB chunks: parallel SWAR
// newline counts -> prefix-summed absolute row indices -> parallel per-
// chunk parsing straight into the final column buffers (disjoint row
// ranges; one shared Utf8 overflow buffer windowed per chunk by source
// bytes, so StringView offsets stay global and windows never collide).
// Byte-identical output to parse_csv. Falls back to the serial parser
// when `sched` is null, the input is small (< 4 MiB), or any row-count
// accounting cross-check fails (defensive).
bool parse_csv_parallel(const char* BOLT_RESTRICT buf, size_t buf_len,
                        const CsvSchema& schema,
                        Arena*           arena,
                        Scheduler*       sched,
                        BoltBatch*       out_batch) noexcept;

}  // namespace ingest
}  // namespace bolt

extern "C" {
// FFI surface — the lakehouse library is the natural consumer.
int bolt_ingest_parse_csv_c(const char* buf, size_t buf_len,
                            const void* schema_ptr,
                            void* arena_ptr,
                            void* batch_ptr);
}
