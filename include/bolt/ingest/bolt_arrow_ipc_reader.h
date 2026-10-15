#pragma once
// Owned Arrow IPC reader, format pin apache-arrow-21.0.0. No Arrow or
// FlatBuffers runtime dependency. This is a bounded flat-type subset.
#include <cstddef>
#include <cstdint>
#include "bolt/bolt_types.h"
#include "bolt/io/bolt_file_io.h"
namespace bolt { struct BoltBatch; }
namespace bolt::ingest {
struct IpcReadString { const char* data = nullptr; std::uint32_t size = 0; };
struct IpcReadKeyValue { IpcReadString key, value; };
// Exact Arrow logical metadata accompanies the physical Bolt column. In
// particular Time/Timestamp/Duration use raw Int32/Int64 lanes, never an
// implicit conversion to Bolt's microsecond Timestamp/Duration convention.
struct IpcReadField {
    IpcReadString name, timezone;
    const IpcReadKeyValue* metadata = nullptr;
    std::uint32_t metadata_count = 0;
    std::uint32_t first_child = 0, child_count = 0; // reserved tree shape
    std::int32_t precision = 0, scale = 0, byte_width = 0;
    std::int16_t unit = 0; // Arrow DateUnit / TimeUnit according to arrow_type
    std::uint8_t arrow_type = 0;
    BoltType physical_type = BoltType::NA;
    bool nullable = false;
};
struct IpcReadSchema {
    const IpcReadField* fields = nullptr;
    std::uint32_t field_count = 0;
    const IpcReadKeyValue* metadata = nullptr;
    std::uint32_t metadata_count = 0;
    const IpcReadKeyValue* file_metadata = nullptr;
    std::uint32_t file_metadata_count = 0;
};
struct IpcReadLimits {
    std::uint32_t max_fields = 256;
    std::uint32_t max_key_values = 1024;
    std::uint32_t max_metadata_bytes = 1024 * 1024;
    std::uint32_t max_batch_rows = 65536;
    std::uint32_t max_file_batches = 1048576;
    std::uint64_t max_body_bytes = 64ull * 1024 * 1024;
    std::uint64_t output_bytes = 64ull * 1024 * 1024;
};
enum class IpcReadStatus : std::uint8_t {
    Ok, End, InvalidArgument, WorkspaceSmall, Truncated, BadInput,
    Unsupported, Limit, IoError, Closed
};
// Zero initialize; lexical ownership, not copyable ownership. Do not copy an
// open reader. No concurrent calls. Borrowed input must remain immutable and
// alive until close; mapped files must not be concurrently changed/truncated.
struct ArrowIpcReader {
    void* state = nullptr;
    io::MappedFile mapping{};
    bool owns_mapping = false;
};
inline constexpr std::size_t kIpcReaderAlignment = 64;
// Checked exact workspace bound, including reusable output and descriptors.
// A zero limit is allowed for row/field/key-value/file-batch counts.
IpcReadStatus arrow_ipc_reader_workspace_size(const IpcReadLimits&, std::uint64_t& bytes) noexcept;
// Startup only. Refusal preserves reader; workspace may be scratch-written
// after alias/size admission. The input/workspace/limits/reader objects must be disjoint.
IpcReadStatus arrow_ipc_reader_open(const void* input, std::uint64_t input_bytes,
    const IpcReadLimits&, void* workspace, std::uint64_t workspace_bytes,
    ArrowIpcReader&) noexcept;
IpcReadStatus arrow_ipc_reader_open_file(const char* path, const IpcReadLimits&,
    void* workspace, std::uint64_t workspace_bytes, ArrowIpcReader&) noexcept;
// Returned schema borrows input+workspace; valid until close. Exact signed
// decimal scale is retained, but this subset admits Decimal128 only with
// precision 1..38 and scale 0..38; other widths/scales return Unsupported.
IpcReadStatus arrow_ipc_reader_schema(const ArrowIpcReader&, IpcReadSchema& out) noexcept;
// No allocation, I/O or arena growth. Entire next batch validates before any
// previous payload is overwritten. On failure/End: out, prior batch payload,
// and batch cursor unchanged. On Ok: out is reader-owned, read-only, valid
// until next successful next or close. It may be copied before then.
// End accepts optional EOS or clean span exhaustion. Extra bytes after EOS
// are BadInput. Nested/dictionary/compressed/view/large-offset arrays and
// big-endian bodies are explicitly Unsupported. V4/V5 metadata admitted.
IpcReadStatus arrow_ipc_reader_next(ArrowIpcReader&, const BoltBatch*& out) noexcept;
void arrow_ipc_reader_close(ArrowIpcReader&) noexcept;
} // namespace bolt::ingest
