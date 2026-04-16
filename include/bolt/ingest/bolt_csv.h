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
    uint32_t num_cols;
    char     delimiter;   // typically ';' or ','
    bool     has_header;  // skip first line if true
};

// Parse a contiguous CSV buffer into columns. The buffer is owned by the
// caller; column data is allocated from `arena`. Returns false on:
//   - schema mismatch (mid-row column count != schema.num_cols)
//   - arena OOM
//   - unsupported BoltType in schema
// Supported types in v0: Int32 (int10ths interpretation), Int64, Utf8.
//
// Inner loop uses bolt_swar_find_byte_u64 for delimiter / line scanning
// and bolt::parse::parse_int10th for numeric fields. No floats.
bool parse_csv(const char* BOLT_RESTRICT buf, size_t buf_len,
               const CsvSchema& schema,
               Arena*           arena,
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
