// bolt/ingest/bolt_gzip.cpp — zlib wrapper (raw deflate stream). Compiles with
// or without zlib; absent BOLT_HAS_ZLIB every entry point is kCodecUnavailable.

#include "bolt/ingest/bolt_gzip.h"

#include <cassert>

#ifdef BOLT_HAS_ZLIB
#include <zlib.h>
#endif

namespace bolt {
namespace ingest {

bool gzip_available() noexcept {
#ifdef BOLT_HAS_ZLIB
    return true;
#else
    return false;
#endif
}

uint64_t gzip_max_compressed_len(uint64_t src_len) noexcept {
#ifdef BOLT_HAS_ZLIB
    const uLong bound = compressBound(static_cast<uLong>(src_len));
    assert(bound >= src_len || src_len == 0);
    return static_cast<uint64_t>(bound);
#else
    if (src_len > (UINT64_MAX - 64u)) return 0;
    return src_len + src_len / 1000u + 64u;
#endif
}

int gzip_compress(const uint8_t* src, uint64_t src_len,
                  uint8_t* dst, uint64_t* dst_len, int level) noexcept {
    assert(dst_len != nullptr);
    assert(src != nullptr || src_len == 0);
#ifdef BOLT_HAS_ZLIB
    if (dst_len == nullptr) return kCodecBadArg;
    if (dst == nullptr || *dst_len == 0) return kCodecBufferSmall;
    uLongf out_len = static_cast<uLongf>(*dst_len);
    const int r = compress2(dst, &out_len, src, static_cast<uLong>(src_len),
                            level);
    if (r == Z_BUF_ERROR) return kCodecBufferSmall;
    if (r != Z_OK) return kCodecCorrupt;
    *dst_len = static_cast<uint64_t>(out_len);
    return kCodecOk;
#else
    (void)src; (void)src_len; (void)dst; (void)dst_len; (void)level;
    return kCodecUnavailable;
#endif
}

int gzip_decompress(const uint8_t* src, uint64_t src_len,
                    uint8_t* dst, uint64_t dst_len) noexcept {
    assert(src != nullptr || src_len == 0);
    assert(dst != nullptr || dst_len == 0);
#ifdef BOLT_HAS_ZLIB
    uLongf out_len = static_cast<uLongf>(dst_len);
    const int r = uncompress(dst, &out_len, src, static_cast<uLong>(src_len));
    if (r == Z_BUF_ERROR) return kCodecBufferSmall;
    if (r != Z_OK) return kCodecCorrupt;
    if (static_cast<uint64_t>(out_len) != dst_len) return kCodecCorrupt;
    return kCodecOk;
#else
    (void)src; (void)src_len; (void)dst; (void)dst_len;
    return kCodecUnavailable;
#endif
}

}  // namespace ingest
}  // namespace bolt
