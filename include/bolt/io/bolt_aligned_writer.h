// bolt_aligned_writer.h — sequential file writer that places pages on the
// recorded alignment units (MSEG K9, layout decision §6.1.1): 64 B buffers,
// align_unit (chunk) starts, 64 KiB stripe starts. Every page's trailing
// padding to its unit is zero-filled and covered by its CRC32C.
//
// Bytes are staged in an io-aligned buffer and written in whole io units at
// io-aligned offsets, so the file can be opened for direct I/O: O_DIRECT on
// Linux, F_NOCACHE on macOS, FILE_FLAG_NO_BUFFERING on Windows. When the
// platform or file system refuses it the writer falls back to buffered I/O
// (AlignedWriter::direct reports which). finish() writes the tail padded to
// the io unit, truncates the file to the logical length and optionally syncs.
//
// Unlike the rest of bolt::io this does file I/O; it lives in its own
// library (bolt::io_file) so bolt::io stays pure compute.

#pragma once

#include <cstddef>
#include <cstdint>

#include "bolt/io/bolt_crc32c.h"
#include "bolt/io/bolt_io_limits.h"

namespace bolt {
namespace io {

enum class AlignedWriteStatus : uint8_t {
    kOk = 0,
    kBadUnits,      // a unit out of range or buf <= chunk <= stripe violated
    kBadStaging,    // staging size out of range or not a stripe multiple
    kOpenFailed,
    kNoMemory,
    kIoError,
    kTooLarge,      // page past kAlignedWriterMaxPageBytes
    kClosed,
};

enum class AlignUnit : uint8_t { kBuf = 0, kChunk = 1, kStripe = 2 };

struct AlignedWriterConfig {
    uint8_t  buf_log2, chunk_log2, stripe_log2, io_log2;
    uint8_t  direct_io;      // 1: try direct I/O, fall back to buffered
    uint8_t  _pad[3];
    uint32_t staging_bytes;  // multiple of the stripe unit
};

/// This build's units (bolt_wire_limits.h), direct I/O requested.
AlignedWriterConfig aligned_writer_config_default() noexcept;

struct AlignedPage {
    uint64_t off;         // multiple of the page's start unit
    uint64_t len;         // payload bytes
    uint64_t padded_len;  // len rounded up to the unit; padding is zero
    uint32_t crc32c;      // page_crc32c over the padded bytes
    uint32_t _pad;
};

struct AlignedWriter {
    intptr_t  handle;       // fd, or a Windows HANDLE; -1 when closed
    uint8_t*  staging;      // io-aligned, staging_cap bytes
    uint64_t  staging_cap;
    uint64_t  staged;       // bytes in staging, starting at file offset flushed
    uint64_t  flushed;      // file bytes written (multiple of staging_cap)
    uint64_t  bytes_io;     // bytes handed to the OS, incl. tail io padding
    uint32_t  unit[3];      // buf, chunk, stripe in bytes
    uint32_t  io_unit;
    uint8_t   direct;       // 1 if the file is open for direct I/O
    uint8_t   _pad[7];
};

inline uint64_t aligned_writer_pos(const AlignedWriter& w) noexcept {
    return w.flushed + w.staged;
}

AlignedWriteStatus aligned_writer_open(const char* path, const AlignedWriterConfig& cfg,
                                       AlignedWriter* w) noexcept;

/// Append raw bytes at the current position (no alignment).
AlignedWriteStatus aligned_writer_append(AlignedWriter* w, const void* data,
                                         size_t len) noexcept;

/// Zero-fill to the next multiple of `unit`.
AlignedWriteStatus aligned_writer_pad(AlignedWriter* w, AlignUnit unit) noexcept;

/// Pad to `unit`, write the page, zero-fill its tail to `unit`, and report
/// its offset, padded length and CRC32C (seeded with `seed`).
AlignedWriteStatus aligned_writer_page(AlignedWriter* w, const void* page, size_t len,
                                       AlignUnit unit, uint32_t seed,
                                       AlignedPage* out) noexcept;

/// Flush, truncate to the logical length, close. `out_size` may be null.
AlignedWriteStatus aligned_writer_finish(AlignedWriter* w, bool sync,
                                         uint64_t* out_size) noexcept;

/// Close without flushing (error paths). Idempotent.
void aligned_writer_abort(AlignedWriter* w) noexcept;

}  // namespace io
}  // namespace bolt
