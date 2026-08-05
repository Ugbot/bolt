// bolt/ingest/bolt_zstd_dec.h — self-contained Zstandard (RFC 8878) DECODER.
//
// WHY THIS EXISTS, given bolt_zstd.h already wraps libzstd: the same two
// reasons bolt_inflate.h exists next to bolt_gzip.h, and one more.
//
//   1. libzstd is OPTIONAL (`BOLT_WITH_ZSTD`, find_package). A default bolt
//      build does not link it — on the development box it is not installed at
//      all (`zstd_DIR-NOTFOUND`). And zstd is *pyiceberg's DEFAULT* data-file
//      codec, so "read a real Iceberg table" through the wrapper would mean a
//      reader that needs a find_package to open the most common table on
//      earth. A reader that needs a find_package to open a real table is not
//      a reader.
//   2. It is the parquet page codec generally, not just Iceberg — ClickBench
//      and most modern parquet in the wild is zstd.
//   3. Owning it removes a build-time dependency from the critical path (bolt
//      CLAUDE.md, "own it all the way down").
//
// This is a DECODER ONLY. Compression stays in bolt_zstd.h's libzstd wrapper
// (we never need to *write* zstd to read a table). Decoding is materially more
// code than inflate — FSE and Huffman entropy stages, sequence decoding with
// repeat offsets — so this file is bigger than bolt_inflate.cpp by design.
//
// COVERAGE (measured against what pyarrow/pyiceberg actually emit, not
// guessed — see tests/data/make_zstd_vectors.py):
//   frames    : single-segment and windowed, with/without content checksum,
//               with/without Frame_Content_Size, multi-block, multi-frame
//   blocks    : Raw, RLE, Compressed
//   literals  : Raw, RLE, Compressed (Huffman), Treeless (reuse prior tree),
//               1-stream and 4-stream
//   sequences : Predefined, RLE, FSE_Compressed and Repeat modes for each of
//               the literal-length / offset / match-length symbols
//   offsets   : full repeat-offset machinery (rep[3], the litLength==0 shift)
//
// REFUSED, LOUDLY (returns kZstdUnsupported, never a wrong answer):
//   - dictionaries (Dictionary_ID != 0). No parquet/Iceberg writer emits them;
//     supporting them means a dictionary-loading API this call site has no way
//     to feed. A dictionary frame is detected and rejected, not mis-decoded.
//
// The content checksum is VERIFIED when present (xxhash64), so a corrupt page
// is caught rather than silently returning wrong bytes.
//
// SECURITY POSTURE: this is a parser over untrusted bytes. Every length and
// offset read from the stream is bounds-checked against the buffer before use;
// a malformed or hostile stream returns an error code. It never throws, never
// asserts on bad input (asserts guard programmer errors only), never allocates,
// and never reads or writes out of bounds.

#pragma once

#include <cstdint>

namespace bolt {
namespace ingest {

enum : int {
    kZstdOk          = 0,
    kZstdBadInput    = 1,   // malformed stream (bad magic, bad table, ...)
    kZstdTruncSrc    = 2,   // ran off the end of the input
    kZstdNeedMore    = 3,   // dst_cap too small to hold the output
    kZstdUnsupported = 4,   // valid zstd we deliberately refuse (dictionary)
    kZstdScratchSmall= 5,   // scratch buffer smaller than zstd_scratch_size()
    kZstdChecksum    = 6,   // content checksum mismatch
};

// Bytes of scratch `zstd_decode_raw` needs. Allocate once (arena/heap — it is
// ~150 KB, far too large for a stack frame) and reuse across calls.
uint64_t zstd_scratch_size() noexcept;

// Decode a zstd stream (one or more concatenated frames) from src[0..src_len)
// into dst[0..dst_cap).
//
// On kZstdOk, *out_len is the number of bytes produced.
//
// Unlike inflate_raw there is no exact-size-on-overflow probe: zstd's window
// makes a size-only pass nearly as expensive as decoding, and every caller we
// have (parquet pages) already knows the exact uncompressed size from the page
// header. Too small a `dst_cap` returns kZstdNeedMore.
//
// `scratch` must be at least zstd_scratch_size() bytes and 8-byte aligned.
int zstd_decode_raw(const uint8_t* src, uint64_t src_len,
                    uint8_t* dst, uint64_t dst_cap, uint64_t* out_len,
                    void* scratch, uint64_t scratch_len) noexcept;

}  // namespace ingest
}  // namespace bolt
