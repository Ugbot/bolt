// bolt/ingest/bolt_zstd.h — THE Zstandard interface. Every zstd call in bolt, marbledb, chukonu,
// boltapi and Gestalt2 goes through this header; nothing else includes <zstd.h> or bolt's own
// encoder / decoder headers (private, src/ingest/; scripts/zstd_iface_lint.py enforces it).
//
// The implementation behind each direction is chosen at BUILD time by CMake cache options:
//   BOLT_ZSTD_ENCODER = reference | bolt     (default reference: libzstd)
//   BOLT_ZSTD_DECODER = reference | bolt     (default reference)
// `reference` without libzstd is a configure error unless BOLT_ZSTD_ALLOW_FALLBACK=ON (then bolt's
// own). The streaming decoder (bolt_zstd_stream.h) is bolt's own in every build.
// zstd_*_name() report the choice.
//
// Two return conventions, both int, 0 = ok: one-shot calls return kCodec* (bolt_codec.h);
// zstd_decode returns kZstd* below. No exceptions. The one-shot calls keep one context per thread;
// zstd_decode uses only the caller's scratch.

#pragma once

#include <cstddef>
#include <cstdint>

#include "bolt/ingest/bolt_codec.h"

namespace bolt {
namespace ingest {

enum : int {
    kZstdOk          = 0,
    kZstdBadInput    = 1,   // malformed stream (bad magic, bad table, ...)
    kZstdTruncSrc    = 2,   // ran off the end of the input
    kZstdNeedMore    = 3,   // dst_cap too small to hold the output
    kZstdUnsupported = 4,   // valid zstd that is refused (dictionary)
    kZstdScratchSmall= 5,   // scratch smaller than zstd_decode_scratch_size() or misaligned
    kZstdChecksum    = 6,   // content checksum mismatch
};

inline constexpr int kZstdDefaultLevel = 3;

// True iff this build can both compress and decompress (always, when bolt::ingest is built).
bool zstd_available() noexcept;

// Build-time selection: "reference" (libzstd) or "bolt". Static strings.
const char* zstd_encoder_name() noexcept;
const char* zstd_decoder_name() noexcept;
const char* zstd_stream_decoder_name() noexcept;

// Upper bound on the compressed size of `src_len` bytes for the selected encoder. 0 on overflow.
uint64_t zstd_max_compressed_len(uint64_t src_len) noexcept;

// Compress src[0..src_len) into dst[0..*dst_len). On entry *dst_len is the capacity; on success
// the bytes written. `level` is clamped by the reference encoder and ignored by bolt's own.
int zstd_compress(const uint8_t* src, uint64_t src_len,
                  uint8_t* dst, uint64_t* dst_len, int level) noexcept;

// Decompress src[0..src_len) into dst. `dst_len` is the exact expected output size.
int zstd_decompress(const uint8_t* src, uint64_t src_len,
                    uint8_t* dst, uint64_t dst_len) noexcept;

// Caller-memory decode of all frames in src into dst[0..dst_cap); *out_len = bytes produced.
// `scratch` is 8-byte aligned, at least zstd_decode_scratch_size() bytes, and reusable across
// calls. No allocation. Returns kZstd*.
uint64_t zstd_decode_scratch_size() noexcept;
int zstd_decode(const uint8_t* src, uint64_t src_len,
                uint8_t* dst, uint64_t dst_cap, uint64_t* out_len,
                void* scratch, uint64_t scratch_len) noexcept;

}  // namespace ingest
}  // namespace bolt
