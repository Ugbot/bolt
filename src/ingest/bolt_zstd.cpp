// bolt/ingest/bolt_zstd.cpp — libzstd wrapper. Compiles with or without zstd;
// when BOLT_HAS_ZSTD is undefined every entry point returns kCodecUnavailable.

#include "bolt/ingest/bolt_zstd.h"

#include <cassert>

#ifdef BOLT_HAS_ZSTD
#include <zstd.h>
#endif

namespace bolt {
namespace ingest {

bool zstd_available() noexcept {
#ifdef BOLT_HAS_ZSTD
    return true;
#else
    return false;
#endif
}

uint64_t zstd_max_compressed_len(uint64_t src_len) noexcept {
#ifdef BOLT_HAS_ZSTD
    const size_t bound = ZSTD_compressBound(static_cast<size_t>(src_len));
    assert(bound >= src_len || src_len == 0);
    if (ZSTD_isError(bound)) return 0;
    return static_cast<uint64_t>(bound);
#else
    // Conservative bound even without the lib so callers can size buffers.
    if (src_len > (UINT64_MAX - 512u)) return 0;
    return src_len + src_len / 2u + 512u;
#endif
}

int zstd_compress(const uint8_t* src, uint64_t src_len,
                  uint8_t* dst, uint64_t* dst_len, int level) noexcept {
    assert(dst_len != nullptr);
    assert(src != nullptr || src_len == 0);
#ifdef BOLT_HAS_ZSTD
    if (dst_len == nullptr) return kCodecBadArg;
    if (dst == nullptr || *dst_len == 0) return kCodecBufferSmall;
    const size_t r = ZSTD_compress(dst, static_cast<size_t>(*dst_len),
                                   src, static_cast<size_t>(src_len), level);
    if (ZSTD_isError(r)) return kCodecCorrupt;
    *dst_len = static_cast<uint64_t>(r);
    return kCodecOk;
#else
    (void)src; (void)src_len; (void)dst; (void)dst_len; (void)level;
    return kCodecUnavailable;
#endif
}

int zstd_decompress(const uint8_t* src, uint64_t src_len,
                    uint8_t* dst, uint64_t dst_len) noexcept {
    assert(src != nullptr || src_len == 0);
    assert(dst != nullptr || dst_len == 0);
#ifdef BOLT_HAS_ZSTD
    const size_t r = ZSTD_decompress(dst, static_cast<size_t>(dst_len),
                                     src, static_cast<size_t>(src_len));
    if (ZSTD_isError(r)) return kCodecCorrupt;
    if (static_cast<uint64_t>(r) != dst_len) return kCodecCorrupt;
    return kCodecOk;
#else
    (void)src; (void)src_len; (void)dst; (void)dst_len;
    return kCodecUnavailable;
#endif
}

}  // namespace ingest
}  // namespace bolt
