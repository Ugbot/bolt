// Cold, bounded immutable Arrow snapshot and allocation-free C ABI leases.
// No engine, Python, or libarrow dependency. CPU flat struct batches only.
#pragma once
#include <cstdint>
#include "bolt/interop/arrow_c.h"
namespace bolt { class Arena; }
namespace bolt::arrow {
struct ArrowOwner;
inline constexpr std::uint32_t kOwnerMaxFields = 4096;
inline constexpr std::uint32_t kOwnerMaxLeases = 64;
struct OwnerLimits {
    std::uint64_t bytes = 128 * 1024 * 1024;
    std::uint32_t fields = 32;
    std::uint32_t leases = 8;
};
enum class OwnerStatus : std::uint8_t { Ok, Invalid, Capacity, Unsupported };
// Cold allocation only. bytes includes owner, fixed arena, and lease storage.
// fields <= 4096, leases <= 64. Failure leaves *out unchanged.
OwnerStatus owner_create(const OwnerLimits&, ArrowOwner** out) noexcept;
// Trusted builder: all payload buffers must be allocated in this arena. Never
// retain/mutate this pointer after publish. Growth is disabled from creation.
Arena* owner_build_arena(ArrowOwner*) noexcept;
// Publish once; flat struct root + leaf columns, no dictionaries or metadata.
// Formats/names and descriptor topology are copied; full buffer spans must fit
// arena backing. Admits bool, integers, floats, decimal128, dates, timestamps,
// durations, binary/UTF-8 and null leaves. Root validity is unsupported.
// The trusted builder owns allocation extents, alignment and payload semantics.
// Failure does not publish (build arena scratch may have been consumed).
OwnerStatus owner_publish(ArrowOwner*, const ArrowSchema&, const ArrowArray&) noexcept;
// Fresh independent descriptors, immutable shared buffers. Output shells must
// be disjoint, outside owner backing, and contain no unreleased prior export.
// Outputs unchanged on refusal. A moved child retains the snapshot separately.
OwnerStatus owner_export_schema(ArrowOwner*, ArrowSchema*) noexcept;
OwnerStatus owner_export_array(ArrowOwner*, ArrowSchema*, ArrowArray*) noexcept;
// One materialized batch, including an empty batch, followed by EOF. CStream
// callbacks are caller-serialized; returned schemas/arrays outlive the stream.
// Lease exhaustion returns ENOMEM without advancing the stream. No callbacks
// allocate, acquire the GIL, or access the originating query/session.
OwnerStatus owner_export_stream(ArrowOwner*, ArrowArrayStream*) noexcept;
void owner_release(ArrowOwner*) noexcept;
std::int64_t owner_rows(const ArrowOwner*) noexcept;
std::uint32_t owner_fields(const ArrowOwner*) noexcept;
const char* owner_field_name(const ArrowOwner*, std::uint32_t) noexcept;
} // namespace bolt::arrow
