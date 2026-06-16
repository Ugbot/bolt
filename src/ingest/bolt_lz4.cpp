// bolt/ingest/bolt_lz4.cpp — liblz4 block wrapper. Compiles with or without
// lz4; absent BOLT_HAS_LZ4 every entry point returns kCodecUnavailable.

#include "bolt/ingest/bolt_lz4.h"

#include <cassert>

#ifdef BOLT_HAS_LZ4
#include <lz4.h>
#endif

namespace bolt {
namespace ingest {

bool lz4_available() noexcept {
#ifdef BOLT_HAS_LZ4
    return true;
#else
    return false;
#endif
}

uint64_t lz4_max_compressed_len(uint64_t src_len) noexcept {
#ifdef BOLT_HAS_LZ4
    if (src_len > static_cast<uint64_t>(LZ4_MAX_INPUT_SIZE)) return 0;
    const int bound = LZ4_compressBound(static_cast<int>(src_len));
    if (bound <= 0) return 0;
    return static_cast<uint64_t>(bound);
#else
    // LZ4 worst case: src_len + src_len/255 + 16.
    if (src_len > (UINT64_MAX - 16u)) return 0;
    return src_len + src_len / 255u + 16u;
#endif
}

int lz4_compress(const uint8_t* src, uint64_t src_len,
                 uint8_t* dst, uint64_t* dst_len, int level) noexcept {
    assert(dst_len != nullptr);
    assert(src != nullptr || src_len == 0);
    (void)level;
#ifdef BOLT_HAS_LZ4
    if (dst_len == nullptr) return kCodecBadArg;
    if (dst == nullptr || *dst_len == 0) return kCodecBufferSmall;
    if (src_len > static_cast<uint64_t>(LZ4_MAX_INPUT_SIZE)) return kCodecBadArg;
    const int r = LZ4_compress_default(
        reinterpret_cast<const char*>(src), reinterpret_cast<char*>(dst),
        static_cast<int>(src_len), static_cast<int>(*dst_len));
    if (r <= 0) return kCodecBufferSmall;
    *dst_len = static_cast<uint64_t>(r);
    return kCodecOk;
#else
    (void)src; (void)src_len; (void)dst; (void)dst_len;
    return kCodecUnavailable;
#endif
}

int lz4_decompress(const uint8_t* src, uint64_t src_len,
                   uint8_t* dst, uint64_t dst_len) noexcept {
    assert(src != nullptr || src_len == 0);
    assert(dst != nullptr || dst_len == 0);
#ifdef BOLT_HAS_LZ4
    const int r = LZ4_decompress_safe(
        reinterpret_cast<const char*>(src), reinterpret_cast<char*>(dst),
        static_cast<int>(src_len), static_cast<int>(dst_len));
    if (r < 0) return kCodecCorrupt;
    if (static_cast<uint64_t>(r) != dst_len) return kCodecCorrupt;
    return kCodecOk;
#else
    (void)src; (void)src_len; (void)dst; (void)dst_len;
    return kCodecUnavailable;
#endif
}

}  // namespace ingest
}  // namespace bolt
