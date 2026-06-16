// bolt/ingest/bolt_zstd.h — Zstandard codec wrapper.
//
// Thin shim over libzstd, opt-in via CMake `BOLT_WITH_ZSTD` (find_package).
// When zstd is not found at configure time, `BOLT_HAS_ZSTD` is undefined and
// every entry point returns the "unavailable" error (nonzero); callers detect
// support with `zstd_available()`.
//
// Return convention (matches bolt I/O code): int 0 = ok, nonzero = error.
// No allocation, no exceptions, bounded — Tiger Style.

#pragma once

#include <cstddef>
#include <cstdint>

#include "bolt/ingest/bolt_codec.h"

namespace bolt {
namespace ingest {

// True iff this build links libzstd.
bool zstd_available() noexcept;

// Upper bound on the compressed size of `src_len` bytes. 0 on overflow.
uint64_t zstd_max_compressed_len(uint64_t src_len) noexcept;

// Compress src[0..src_len) into dst[0..*dst_len). On entry *dst_len is the
// capacity; on success it is set to the bytes written. `level` clamps to the
// library's valid range. Returns 0 on success.
int zstd_compress(const uint8_t* src, uint64_t src_len,
                  uint8_t* dst, uint64_t* dst_len, int level) noexcept;

// Decompress src[0..src_len) into dst. `dst_len` is the exact expected output
// size (caller knows it). Returns 0 on success and an exact fill.
int zstd_decompress(const uint8_t* src, uint64_t src_len,
                    uint8_t* dst, uint64_t dst_len) noexcept;

}  // namespace ingest
}  // namespace bolt
