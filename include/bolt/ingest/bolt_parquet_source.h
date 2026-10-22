// Parquet over an exact, version-pinned byte source. No transport dependency.
#pragma once
#include "bolt/ingest/bolt_parquet_source_limits.h"
#include <cstdint>
#include "bolt/ingest/bolt_parquet_meta.h"

namespace bolt { class Arena; struct BoltColumn; }
namespace bolt::ingest::parquet {

enum class ParquetSourceStatus : uint8_t {
    Ok, BadArg, IoError, Changed, Capacity, DecodeFailed
};

// Borrowed for the call. read_at fills EXACTLY length bytes or returns an
// error, including Changed when an immutable object/version validator fails.
// The adapter owns validation of transport ranges and short reads. A source
// must identify the same immutable bytes across ALL calls; never repin/retry
// another version after Changed. size is the pinned object's full extent.
struct ParquetByteSource {
    uint64_t size = 0;
    void* context = nullptr;
    ParquetSourceStatus (*read_at)(void*, uint64_t offset, uint32_t length,
                                   uint8_t* dst) noexcept = nullptr;
};

// Caller-owned reusable storage, disjoint from metadata/output arenas. Holds
// only a footer OR one compressed column chunk, never the complete file.
// capacity is a caller-selected finite budget, not a limit on file size.
// Each callback length is <=request_bytes, nonzero and <=1MiB.
struct ParquetSourceWorkspace {
    uint8_t* bytes = nullptr;
    uint64_t capacity = 0;
    uint32_t request_bytes = kParquetSourceMaxReadBytes;
};

// Metadata (including all strings/statistics) is copied into metadata_arena;
// it never borrows workspace bytes. On failure *out is unchanged. The arena
// may consume scratch on refusal; its lifetime/budget belong to the caller.
ParquetSourceStatus parquet_read_meta_source(const ParquetByteSource* source,
    ParquetSourceWorkspace* workspace, Arena* metadata_arena, PqMeta* out) noexcept;

// Same pinned transport, staging, capacity and output contract. Additionally
// requires the maintained exact flat footer/schema decoder before publication;
// nested/repeated shapes and ambiguous logical metadata are refused. No extra
// origin requests; validation consumes the already fetched exact footer bytes.
ParquetSourceStatus parquet_read_meta_source_flat_strict(const ParquetByteSource* source,
    ParquetSourceWorkspace* workspace, Arena* metadata_arena, PqMeta* out) noexcept;

// Flat-column projection (caller-deduplicated indices), with the same supported types/encodings as the
// contiguous row-group decoder. Fetches each selected compressed chunk once,
// in bounded requests; metadata and output live outside the reused workspace.
// On failure out_cols[0..n_idx) is unchanged. Admission/alias refusal leaves
// *out_rows unchanged; once admitted, a failed operation sets it to zero. Output
// allocations belong to output_arena, including any scratch consumed on error.
// Capacity, Changed and IoError propagate distinctly. DecodeFailed covers
// malformed/unsupported pages or a failure within the existing decoder.
// Workspace/descriptors, input metadata/indices and writable descriptors must
// not overlap; overflowed or overlapping spans are refused before I/O.
ParquetSourceStatus parquet_read_row_group_cols_source(const ParquetByteSource* source,
    ParquetSourceWorkspace* workspace, const PqMeta* meta, uint32_t row_group,
    const uint16_t* col_idx, uint32_t n_idx, Arena* output_arena,
    BoltColumn* out_cols, int64_t* out_rows) noexcept;
}  // namespace bolt::ingest::parquet
