// bolt_io_limits.h — tuning constants and caps of bolt::io: the 3-way CRC32C
// lanes and the aligned page writer (MSEG K9). Alignment units are the wire
// container's (bolt_wire_limits.h); this file holds only what is io-specific.
// Override per build with -DBOLT_IO_STAGING_BYTES=... etc.

#pragma once

#include <cstddef>
#include <cstdint>

#include "bolt/wire/bolt_wire_limits.h"

#ifndef BOLT_IO_CRC32C_LANE_BYTES
#define BOLT_IO_CRC32C_LANE_BYTES 1024u
#endif
#ifndef BOLT_IO_STAGING_BYTES
#define BOLT_IO_STAGING_BYTES (4u << 20)
#endif

namespace bolt {
namespace io {

// One lane of the interleaved CRC: a buffer is consumed in blocks of three
// lanes whose CRCs run in parallel and are recombined by a table shift.
inline constexpr size_t kCrc32cLaneBytes = BOLT_IO_CRC32C_LANE_BYTES;
// Below this the serial loop is as fast (one block of three lanes).
inline constexpr size_t kCrc32c3WayMinBytes = 3 * kCrc32cLaneBytes;

// Aligned writer staging buffer (kind B: fixed at open). Must be a multiple
// of the stripe unit so a flush never splits a stripe-aligned write.
inline constexpr uint32_t kAlignedWriterStagingDefault = BOLT_IO_STAGING_BYTES;
inline constexpr uint32_t kAlignedWriterStagingMin = wire::kFrameStripeAlign;
inline constexpr uint32_t kAlignedWriterStagingMax = 256u << 20;
// EINTR / short-write retries per flush before the write is reported failed.
inline constexpr uint32_t kAlignedWriterMaxWriteRetries = 1024;
// One page appended in one call (a page is a chunk; chunks are <= a stripe
// region of a segment, far below this).
inline constexpr uint64_t kAlignedWriterMaxPageBytes = uint64_t(1) << 32;

static_assert(kCrc32cLaneBytes % 8 == 0 && kCrc32cLaneBytes >= 64, "lane is whole u64 words");
static_assert(kAlignedWriterStagingDefault % wire::kFrameStripeAlign == 0,
              "BOLT_IO_STAGING_BYTES is a multiple of the stripe unit");
static_assert(kAlignedWriterStagingDefault >= kAlignedWriterStagingMin &&
              kAlignedWriterStagingDefault <= kAlignedWriterStagingMax,
              "BOLT_IO_STAGING_BYTES range");

}  // namespace io
}  // namespace bolt
