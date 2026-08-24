// bolt/ingest/bolt_lz4_raw.h — self-contained LZ4 BLOCK codec.
//
// Why this exists, when bolt_lz4.h is already here: that header is a shim over
// liblz4 behind `find_package(lz4)`, and liblz4 is not present on a default
// build. So parquet's LZ4_RAW codec -- which Hadoop-era writers still emit --
// could not be READ at all unless the machine happened to have liblz4, and
// could never be written. That is the same situation the self-contained zstd
// decoder (bolt_zstd_dec.h) was written to fix, and the same rule applies:
// "a reader that needs a find_package to open a real table is not a reader".
//
// This is the LZ4 BLOCK format (LZ4_compress_default / LZ4_decompress_safe),
// which is exactly what parquet's LZ4_RAW carries -- no frame header, no
// checksum, no block size prefix. It is NOT the LZ4 FRAME format, and it is
// not parquet's deprecated LZ4 codec (which wrapped a Hadoop framing).
//
// Format, from the LZ4 block specification:
//
//   block    := sequence*
//   sequence := token literal_len_ext* literals [offset match_len_ext*]
//   token    := high nibble = literal length (15 = "more follows")
//               low  nibble = match length - 4 (15 = "more follows")
//   *_ext    := 0xFF repeated, then a final byte < 0xFF, all summed
//   offset   := 2 bytes little-endian, distance BACK from the output cursor,
//               1..65535. A match may overlap the output cursor, which is how
//               LZ4 expresses runs.
//
// Two end-of-block rules the spec imposes and a decoder must not assume away:
// the last sequence carries literals only (no offset follows), and the final
// 5 bytes of a block are always literals. A compressor that violates either
// produces a block liblz4 refuses.
//
// Tiger Style: noexcept, no allocation, no exceptions, bounded loops, every
// read and write bounds-checked against the caller's buffers. Hostile input
// returns false; it never reads or writes out of bounds.

#pragma once

#include <cstdint>

namespace bolt {
namespace ingest {

// Worst-case compressed size for `src_len` incompressible bytes: every byte a
// literal, plus one token per 255 literals, plus the final token. Matches
// liblz4's LZ4_compressBound so a caller sizing a buffer with either agrees.
inline constexpr uint64_t lz4_raw_bound(uint64_t src_len) noexcept {
    return src_len + (src_len / 255u) + 16u;
}

// Compress `src[0..src_len)` into `dst[0..dst_cap)`. On success writes the
// compressed length to *out_len and returns true. Returns false if the output
// does not fit -- never a partial block.
//
// `src_len` is capped at 2 GiB: LZ4 offsets are 16-bit and lengths are
// accumulated in 32-bit arithmetic, and parquet pages are far smaller.
bool lz4_raw_compress(const uint8_t* src, uint64_t src_len,
                      uint8_t* dst, uint64_t dst_cap,
                      uint64_t* out_len) noexcept;

// Decompress `src[0..src_len)` into `dst[0..dst_cap)`, expecting EXACTLY
// `dst_cap` bytes of output (parquet always knows a page's uncompressed size,
// and a block that produces fewer or more bytes than promised is corrupt).
// Returns false on any malformed input rather than producing partial output.
bool lz4_raw_decompress(const uint8_t* src, uint64_t src_len,
                        uint8_t* dst, uint64_t dst_cap) noexcept;

}  // namespace ingest
}  // namespace bolt
