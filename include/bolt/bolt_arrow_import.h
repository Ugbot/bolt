#pragma once
// Cold Arrow C Data copy ingress. No Arrow runtime or Python dependency.
// Foreign structs/buffers must satisfy the C ABI's readable-memory contract;
// it provides no buffer byte lengths. All computed extents/offsets are checked.
#include "bolt/interop/arrow_c.h"
#include "bolt/bolt_types.h"
#include <cstdint>
namespace bolt { class Arena; struct BoltBatch; }
namespace bolt::arrow {
enum class ImportError : std::uint8_t { Ok, Invalid, Unsupported, Limit, Allocation };
struct ImportLimits {
    // max_rows <= INT32_MAX (owned selection/variable-offset representation).
    // max_bytes accounts for transient copied chunks, final copy, metadata
    // and per-allocation alignment slack; large offsets compact to Int32.
    // This bounds allocations, not the arena's reserved backing capacity.
    std::uint32_t max_batches=1024, max_rows=1048576, max_columns=kMaxColumns;
    std::uint64_t max_bytes=256ull*1024*1024;
};
struct ImportField {
    BoltField field{};
    std::uint8_t precision=0, unit=0; // unit: 1=s,2=ms,3=us,4=ns; no rescaling
    std::uint8_t temporal_kind=0; // 1=timestamp,2=duration; non-us physical Int64
    bool large_offsets=false;
};
struct ImportChunk;
// Both arenas must have default_alignment() <= 64; larger alignments refuse
// explicitly because the admitted byte charge includes 64-byte allocation slack.
// Lexical scratch owner. Begin/append borrow foreign roots only for the call;
// the caller releases those roots. Append failure poisons state. Output is
// published only by finish; the caller must additionally require clean EOF.
struct ImportState {
    ImportLimits limits{}; Arena* scratch=nullptr; ImportField* fields=nullptr;
    ImportChunk* head=nullptr; ImportChunk* tail=nullptr;
    std::uint64_t rows=0, charged_bytes=0;
    std::uint32_t columns=0, batches=0;
    ImportError error=ImportError::Invalid;
    bool finished=false;
};
// Supports flat primitive struct fields: exact integers/Float32/64, Bool,
// Utf8/Binary (32/64 offsets), Date32/64, timezone-free Timestamp/Duration,
// Decimal128 precision1..38/scale0..precision. Dictionary/nested/extension,
// timezone and unsupported decimal shapes refuse explicitly.
ImportError import_begin(const ArrowSchema*, const ImportLimits&, Arena*, ImportState&) noexcept;
ImportError import_append(ImportState&, const ArrowArray*) noexcept;
// Concatenates owned chunks once into destination; empty typed state is valid.
// Output unchanged on error. Scratch and destination must be distinct arenas.
ImportError import_finish(ImportState&, Arena* destination, BoltBatch& output) noexcept;
const char* import_error_name(ImportError) noexcept;
} // namespace bolt::arrow
