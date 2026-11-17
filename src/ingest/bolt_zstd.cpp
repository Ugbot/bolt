// bolt/ingest/bolt_zstd.cpp — libzstd wrapper. Compiles with or without zstd;
// when BOLT_HAS_ZSTD is undefined every entry point returns kCodecUnavailable.

#include "bolt/ingest/bolt_zstd.h"

#include <cassert>

#ifdef BOLT_HAS_ZSTD
#include <zstd.h>
#endif

// One switch per direction puts bolt's own implementation behind this wrapper instead of libzstd
// (decision 2026-11-17: the reference library everywhere until bolt's own is better; measured in
// benchmarks/bench_zstd_decoders.cpp). Both default off.
#if defined(BOLT_ZSTD_DECODE_SELF) || defined(BOLT_ZSTD_ENCODE_SELF)
#include <cstdlib>
#include "bolt/ingest/bolt_zstd_dec.h"
#include "bolt/ingest/bolt_zstd_enc.h"
#endif

namespace bolt {
namespace ingest {

bool zstd_available() noexcept {
#if defined(BOLT_HAS_ZSTD) || (defined(BOLT_ZSTD_DECODE_SELF) && defined(BOLT_ZSTD_ENCODE_SELF))
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
#if defined(BOLT_ZSTD_ENCODE_SELF)
    (void)level;
    if (dst_len == nullptr) return kCodecBadArg;
    static thread_local ZstdEncState* st = nullptr;   // one state per thread, kept for the process
    if (st == nullptr) {
        void* mem = std::aligned_alloc(64, (zstd_enc_state_size() + 63u) & ~uint64_t{63});
        st = mem != nullptr ? zstd_enc_state_init(mem, zstd_enc_state_size()) : nullptr;
        if (st == nullptr) return kCodecBufferSmall;
    }
    uint64_t out = 0;
    if (!zstd_compress_self(src, src_len, dst, *dst_len, &out, st)) return kCodecBufferSmall;
    *dst_len = out;
    return kCodecOk;
#elif defined(BOLT_HAS_ZSTD)
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
#if defined(BOLT_ZSTD_DECODE_SELF)
    static thread_local void* scratch = nullptr;      // one scratch per thread, kept for the process
    if (scratch == nullptr) scratch = std::aligned_alloc(64, (zstd_scratch_size() + 63u) & ~uint64_t{63});
    if (scratch == nullptr) return kCodecBufferSmall;
    uint64_t got = 0;
    if (zstd_decode_raw(src, src_len, dst, dst_len, &got, scratch, zstd_scratch_size()) != kZstdOk || got != dst_len)
        return kCodecCorrupt;
    return kCodecOk;
#elif defined(BOLT_HAS_ZSTD)
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
