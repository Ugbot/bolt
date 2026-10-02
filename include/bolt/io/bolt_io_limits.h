// bolt_io_limits.h — tuning constants and caps of bolt::io: the 3-way CRC32C
// lanes, the aligned page writer (MSEG K9) and file reads (MSEG B6). Alignment units are the wire
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
#ifndef BOLT_IO_COALESCE_MAX_REQUEST
#define BOLT_IO_COALESCE_MAX_REQUEST (8u << 20)
#endif
#ifndef BOLT_IO_COALESCE_MAX_GAP
#define BOLT_IO_COALESCE_MAX_GAP (64u << 10)
#endif
#ifndef BOLT_IO_READ_POOL_MAX_WORKERS
#define BOLT_IO_READ_POOL_MAX_WORKERS 256u
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

// --- file reads (bolt_file_io.h) -------------------------------------------
// EINTR / short-read retries per pread before it is reported failed.
inline constexpr uint32_t kFileReadMaxRetries = 1024;
// Largest single OS read call; longer reads loop (Windows ReadFile is DWORD).
inline constexpr uint64_t kFileReadMaxCallBytes = uint64_t(1) << 30;
// Largest pread one call accepts (kind A: an extent, never a whole dataset).
inline constexpr uint64_t kFileReadMaxBytes = uint64_t(1) << 40;
// F_RDADVISE takes an int count; longer hints are issued in these pieces.
inline constexpr uint64_t kFileAdviseMaxCallBytes = uint64_t(1) << 30;
// range_coalesce defaults (kind C, per call): object GET / cold read requests
// are at most this long, and inputs this close are merged into one request.
inline constexpr uint64_t kCoalesceMaxRequestDefault = BOLT_IO_COALESCE_MAX_REQUEST;
inline constexpr uint64_t kCoalesceMaxGapDefault = BOLT_IO_COALESCE_MAX_GAP;
// Read pool worker threads (kind B: fixed at create; 0 = run inline).
inline constexpr uint32_t kReadPoolMaxWorkers = BOLT_IO_READ_POOL_MAX_WORKERS;
// mincore results gathered per call (a stack array of this many bytes).
inline constexpr uint64_t kMincoreChunkPages = 4096;
// Requests in one read batch (kind A: the request index is a uint32).
inline constexpr uint32_t kReadBatchMaxRequests = uint32_t(1) << 24;

static_assert(kCoalesceMaxRequestDefault >= wire::kFrameStripeAlign,
              "BOLT_IO_COALESCE_MAX_REQUEST is at least one stripe");
static_assert(kCoalesceMaxGapDefault < kCoalesceMaxRequestDefault, "coalesce gap < request");
static_assert(kReadPoolMaxWorkers >= 1 && kReadPoolMaxWorkers <= 4096, "read pool range");
static_assert(kFileReadMaxCallBytes <= kFileReadMaxBytes, "call <= read cap");

static_assert(kCrc32cLaneBytes % 8 == 0 && kCrc32cLaneBytes >= 64, "lane is whole u64 words");
static_assert(kAlignedWriterStagingDefault % wire::kFrameStripeAlign == 0,
              "BOLT_IO_STAGING_BYTES is a multiple of the stripe unit");
static_assert(kAlignedWriterStagingDefault >= kAlignedWriterStagingMin &&
              kAlignedWriterStagingDefault <= kAlignedWriterStagingMax,
              "BOLT_IO_STAGING_BYTES range");

}  // namespace io
}  // namespace bolt
