// bolt/ingest/bolt_lz4.h — LZ4 block codec wrapper.
//
// Thin shim over liblz4 (block format, not the frame format), opt-in via
// CMake `BOLT_WITH_LZ4` (find_package). When lz4 is not found, `BOLT_HAS_LZ4`
// is undefined and entry points return the "unavailable" error; callers detect
// support with `lz4_available()`.
//
// Return convention: int 0 = ok, nonzero = error (see kCodec* in bolt_zstd.h).

#pragma once

#include <cstddef>
#include <cstdint>

#include "bolt/ingest/bolt_codec.h"

namespace bolt {
namespace ingest {

// True iff this build links liblz4.
bool lz4_available() noexcept;

// Upper bound on the compressed size of `src_len` bytes. 0 on overflow.
uint64_t lz4_max_compressed_len(uint64_t src_len) noexcept;

// Compress src into dst (LZ4 block format). *dst_len is the capacity on entry,
// the bytes written on success. `level` is accepted for signature symmetry but
// the default block compressor ignores it. Returns 0 on success.
int lz4_compress(const uint8_t* src, uint64_t src_len,
                 uint8_t* dst, uint64_t* dst_len, int level) noexcept;

// Decompress an LZ4 block into dst. `dst_len` is the exact expected output
// size. Returns 0 on success and an exact fill.
int lz4_decompress(const uint8_t* src, uint64_t src_len,
                   uint8_t* dst, uint64_t dst_len) noexcept;

}  // namespace ingest
}  // namespace bolt
