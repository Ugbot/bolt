// bolt/ingest/bolt_inflate.h — self-contained raw DEFLATE (RFC 1951) decoder.
//
// WHY THIS EXISTS, given bolt_gzip.h already wraps zlib: two reasons, both
// hard requirements of the Avro/Iceberg read path.
//
//   1. zlib is OPTIONAL (`BOLT_WITH_ZLIB`, find_package). A default bolt build
//      does not link it, so `gzip_decompress` returns "unavailable". Real
//      Iceberg manifests are deflate-compressed — Iceberg's
//      `write.avro.compression-codec` defaults to gzip/deflate, and pyiceberg
//      and Iceberg-Java both emit `"avro.codec":"deflate"` — so a reader that
//      needs a find_package to open a real table is not a reader.
//   2. `gzip_decompress` requires the EXACT decompressed length up front. An
//      Avro OCF block carries the object COUNT and the COMPRESSED size, never
//      the raw size. There is nothing to pass.
//
// So this is the general-case entry point: raw DEFLATE in, bounded buffer out,
// an explicit "your buffer was too small" signal so the caller can retry into
// a larger one. Zero dependencies, no allocation, no exceptions — inflate is
// ~200 lines of table-driven bit twiddling and owning it removes a build-time
// dependency from the critical path (see bolt CLAUDE.md, "own it all the way
// down"). Correctness model is `puff.c`'s (zlib's own reference decoder):
// canonical Huffman decoded bit-at-a-time from per-length counts. Manifests are
// kilobytes, so clarity and bounds-safety beat a table-driven fast path here.
//
// SECURITY POSTURE: every byte read is bounds-checked against `src_len` and
// every byte written against `dst_cap`; a malformed or hostile stream returns
// an error code, never asserts and never reads or writes out of bounds.

#pragma once

#include <cstdint>

namespace bolt {
namespace ingest {

enum : int {
    kInflateOk       = 0,
    kInflateBadInput = 1,   // malformed stream (bad block type, bad codes, ...)
    kInflateTruncSrc = 2,   // ran off the end of the input
    kInflateNeedMore = 3,   // dst_cap too small — retry with a bigger buffer
};

// Inflate a RAW deflate stream (no zlib wrapper, no gzip header) from
// src[0..src_len) into dst[0..dst_cap).
//
// On kInflateOk, *out_len is the number of bytes produced. On kInflateNeedMore
// *out_len is the EXACT number of bytes the stream produces, so a caller
// allocates that and retries once — no doubling search. (Decoding never reads
// back its own output to decide a length, so the count stays exact after the
// writes stop; only the BYTES are lost.)
//
// `dst` may be null when dst_cap == 0, which makes this a pure size probe:
// it returns kInflateNeedMore with the required size in *out_len.
int inflate_raw(const uint8_t* src, uint64_t src_len,
                uint8_t* dst, uint64_t dst_cap, uint64_t* out_len) noexcept;

}  // namespace ingest
}  // namespace bolt
