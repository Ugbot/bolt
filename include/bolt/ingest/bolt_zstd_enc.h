// bolt/ingest/bolt_zstd_enc.h — self-contained ZSTD COMPRESSOR (RFC 8878).
//
// bolt has decoded zstd without a dependency since bolt_zstd_dec.cpp -- which
// exists because zstd is pyiceberg's default parquet codec and a reader that
// needs a find_package to open a real table is not a reader. This is the
// other half, and it is the last codec bolt could read and not write.
//
// It is also the only one with no small correct version. zstd cannot express
// a single match without its sequences section, and that section is FSE
// entropy coding: three interleaved states, written in reverse sequence order
// into a bitstream the decoder reads backwards. There is no "literals only"
// shortcut that still compresses, the way snappy and LZ4 have.
//
// WHAT THIS EMITS
//
//   * One frame, Single_Segment (so the window is the content size and no
//     Window_Descriptor is written), no dictionary, no content checksum.
//   * Blocks of at most 128 KiB, each either Raw or Compressed.
//   * Compressed blocks use RAW literals (no Huffman) and PREDEFINED FSE
//     tables for all three sequence symbols, so no table description is
//     transmitted.
//
// The two deliberate omissions -- Huffman-coded literals and custom FSE
// tables -- are what separate this from libzstd's ratio, not from its
// correctness. Both are pure additions to the same frame structure, and both
// cost ratio only. Every zstd decoder reads what this produces.
//
// Sequence codes are derived from the DECODER's own kLLBase/kLLBits and
// kMLBase/kMLBits tables rather than transcribed from a second copy in the
// spec. That is deliberate: a mis-transcribed table is exactly the bug class
// already recorded in bolt_zstd_dec.cpp (the predefined match-length
// distribution was wrong by two entries and decoded a 100000-byte block to
// 38 bytes). Deriving keeps encoder and decoder consistent by construction.
//
// Tiger Style: noexcept, no allocation (the caller supplies scratch), no
// exceptions, bounded loops, all output bounds-checked.

#pragma once

#include <cstdint>

namespace bolt {
namespace ingest {

// Caller-supplied scratch: the match-finder table and the per-block sequence
// buffers. ~1.6 MB. One instance is reusable for any number of calls.
struct ZstdEncState;

// Bytes of scratch the caller must provide, and its alignment.
std::uint64_t zstd_enc_state_size() noexcept;
ZstdEncState* zstd_enc_state_init(void* mem, std::uint64_t len) noexcept;

// Upper bound on the compressed size: every block stored raw, plus block
// headers and the frame header.
inline constexpr std::uint64_t zstd_enc_bound(std::uint64_t n) noexcept {
    return n + 3u * ((n >> 17) + 1u) + 32u;
}

// Compress `src` into a complete single-frame zstd stream. Returns false if
// the output does not fit -- never a partial frame.
bool zstd_compress_self(const std::uint8_t* src, std::uint64_t src_len,
                        std::uint8_t* dst, std::uint64_t dst_cap,
                        std::uint64_t* out_len, ZstdEncState* st) noexcept;

}  // namespace ingest
}  // namespace bolt
