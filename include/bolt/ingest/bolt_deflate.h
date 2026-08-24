// bolt/ingest/bolt_deflate.h — self-contained DEFLATE (RFC 1951) COMPRESSOR
// and GZIP (RFC 1952) container writer.
//
// bolt already INFLATES without a dependency (bolt_inflate.h, which is what
// lets the parquet reader open a GZIP-compressed file on a default build).
// The other direction had nothing: bolt_gzip.h is a shim behind
// `find_package(ZLIB)`, so bolt could read GZIP parquet and never write it.
// This closes that, under the same rule the inflate and zstd decoders were
// written to: no find_package for a format the ecosystem routinely uses.
//
// WHAT THIS EMITS, and what it deliberately does not.
//
// Blocks are either STORED (BTYPE=00) or FIXED-Huffman (BTYPE=01). There is
// no dynamic-Huffman (BTYPE=10) path. That is a real and bounded cost --
// dynamic Huffman typically buys another 5-15% on text -- taken because the
// encoder side of it is a canonical-code builder plus a code-length code,
// roughly tripling the code for a fraction that does not change whether a
// file is readable. Every inflater accepts fixed blocks; this is a
// compression-ratio choice, never a compatibility one.
//
// A block that fixed-Huffman would not shrink is emitted STORED instead, so
// incompressible input costs 5 bytes per 65535-byte block rather than
// expanding by ~12% the way a naive fixed-Huffman-only encoder does.
//
// Tiger Style: noexcept, no allocation (the caller supplies every buffer and
// the 128 KiB match-finder state), no exceptions, bounded loops, all output
// bounds-checked. A caller whose destination is too small gets false, never a
// partial stream.

#pragma once

#include <cstdint>

namespace bolt {
namespace ingest {

// Scratch the match finder needs, supplied by the caller so the compressor
// allocates nothing. One instance can be reused for any number of
// calls; deflate_raw_compress resets what it needs. 256 KiB.
struct DeflateState {
    // FULL positions, not 16-bit ones. A uint16 cannot address past 65535,
    // and truncating silently corrupts every hash chain once the input grows
    // beyond that -- which costs compression ratio, not correctness, and so
    // does not show up in a round-trip test.
    std::uint32_t head[1u << 15];   // hash -> most recent position
    std::uint32_t prev[1u << 15];   // position -> previous position, same hash
};

// Upper bound on the compressed size. Worst case is all-STORED: 5 bytes of
// block header per 65535 bytes, plus the final empty block and byte padding.
inline constexpr std::uint64_t deflate_bound(std::uint64_t n) noexcept {
    return n + 5u * ((n / 65535u) + 1u) + 8u;
}

// GZIP adds a 10-byte header and an 8-byte trailer (CRC32 + ISIZE).
inline constexpr std::uint64_t gzip_bound(std::uint64_t n) noexcept {
    return deflate_bound(n) + 18u;
}

// CRC-32 (RFC 1952 / IEEE 802.3), the one gzip's trailer carries. Exposed
// because it is the only checksum bolt owns and callers verifying a gzip
// stream need the same one.
std::uint32_t crc32_ieee(const std::uint8_t* data, std::uint64_t len,
                         std::uint32_t seed) noexcept;

// Compress into a RAW deflate stream (no zlib or gzip wrapper) -- the exact
// input bolt_inflate.h's inflate_raw consumes.
bool deflate_raw_compress(const std::uint8_t* src, std::uint64_t src_len,
                          std::uint8_t* dst, std::uint64_t dst_cap,
                          std::uint64_t* out_len, DeflateState* st) noexcept;

// Compress into a complete GZIP container: 10-byte header, deflate stream,
// CRC32 of the UNCOMPRESSED bytes, then the uncompressed length mod 2^32.
// This is what parquet's GZIP codec carries.
bool gzip_compress(const std::uint8_t* src, std::uint64_t src_len,
                   std::uint8_t* dst, std::uint64_t dst_cap,
                   std::uint64_t* out_len, DeflateState* st) noexcept;

}  // namespace ingest
}  // namespace bolt
