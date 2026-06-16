// bolt/ingest/bolt_gzip.h — zlib (deflate/gzip) codec wrapper.
//
// Thin shim over zlib, opt-in via CMake `BOLT_WITH_ZLIB` (find_package).
// When zlib is not found, `BOLT_HAS_ZLIB` is undefined and entry points
// return the "unavailable" error; callers detect support with
// `gzip_available()`.
//
// `gzip_*` operate on the raw zlib/deflate stream (the wbits-15 zlib wrapper),
// which is what Avro's "deflate" codec and Parquet's GZIP codec consume.
//
// Return convention: int 0 = ok, nonzero = error (see kCodec* in bolt_zstd.h).

#pragma once

#include <cstddef>
#include <cstdint>

#include "bolt/ingest/bolt_codec.h"

namespace bolt {
namespace ingest {

// True iff this build links zlib.
bool gzip_available() noexcept;

// Upper bound on the deflated size of `src_len` bytes. 0 on overflow.
uint64_t gzip_max_compressed_len(uint64_t src_len) noexcept;

// Deflate src into dst (raw deflate stream — no gzip header). *dst_len is the
// capacity on entry, the bytes written on success. Returns 0 on success.
int gzip_compress(const uint8_t* src, uint64_t src_len,
                  uint8_t* dst, uint64_t* dst_len, int level) noexcept;

// Inflate a raw deflate stream into dst. `dst_len` is the exact expected
// output size. Returns 0 on success and an exact fill.
int gzip_decompress(const uint8_t* src, uint64_t src_len,
                    uint8_t* dst, uint64_t dst_len) noexcept;

}  // namespace ingest
}  // namespace bolt
