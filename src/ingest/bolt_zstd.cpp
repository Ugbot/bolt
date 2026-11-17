// bolt/ingest/bolt_zstd.cpp — the one zstd interface: both backends and the build-time selection.
// BOLT_ZSTD_ENCODER_REFERENCE / BOLT_ZSTD_DECODER_REFERENCE (0 | 1) come from the CMake options
// BOLT_ZSTD_ENCODER / BOLT_ZSTD_DECODER; BOLT_HAS_ZSTD says libzstd is linked.

#include "bolt/ingest/bolt_zstd.h"

#include <cassert>
#include <cstdlib>

#include "bolt_zstd_backends.h"
#include "bolt_zstd_dec.h"
#include "bolt_zstd_enc.h"

#ifdef BOLT_HAS_ZSTD
#define ZSTD_STATIC_LINKING_ONLY 1   // ZSTD_initStaticDCtx: decode in caller memory
#include <zstd.h>
#include <zstd_errors.h>
#endif

#ifndef BOLT_ZSTD_ENCODER_REFERENCE
#define BOLT_ZSTD_ENCODER_REFERENCE 0
#endif
#ifndef BOLT_ZSTD_DECODER_REFERENCE
#define BOLT_ZSTD_DECODER_REFERENCE 0
#endif
#if (BOLT_ZSTD_ENCODER_REFERENCE || BOLT_ZSTD_DECODER_REFERENCE) && !defined(BOLT_HAS_ZSTD)
#error "reference zstd selected but libzstd is not linked (see BOLT_ZSTD_ENCODER / BOLT_ZSTD_DECODER)"
#endif

namespace bolt {
namespace ingest {
namespace zstd_backend {

namespace {
// One block per thread, kept for the thread's life (worker threads are pooled).
void* tls_block(void** slot, uint64_t bytes) noexcept {
    assert(slot != nullptr);
    assert(bytes > 0);
    if (*slot == nullptr) *slot = std::aligned_alloc(64, (bytes + 63u) & ~uint64_t{63});
    return *slot;
}
}  // namespace

uint64_t self_bound(uint64_t src_len) noexcept {
    if (src_len > (UINT64_MAX >> 1)) return 0;
    return zstd_enc_bound(src_len);
}

int self_compress(const uint8_t* src, uint64_t src_len, uint8_t* dst, uint64_t* dst_len, int level) noexcept {
    assert(dst_len != nullptr);
    assert(src != nullptr || src_len == 0);
    (void)level;
    if (dst_len == nullptr) return kCodecBadArg;
    if (dst == nullptr || *dst_len == 0) return kCodecBufferSmall;
    static thread_local void* mem = nullptr;
    static thread_local ZstdEncState* st = nullptr;
    if (st == nullptr) {
        void* m = tls_block(&mem, zstd_enc_state_size());
        st = m != nullptr ? zstd_enc_state_init(m, zstd_enc_state_size()) : nullptr;
        if (st == nullptr) return kCodecBufferSmall;
    }
    uint64_t out = 0;
    if (!zstd_compress_self(src, src_len, dst, *dst_len, &out, st)) return kCodecBufferSmall;
    assert(out <= *dst_len);
    *dst_len = out;
    return kCodecOk;
}

uint64_t self_scratch_size() noexcept { return zstd_scratch_size(); }

int self_decode(const uint8_t* src, uint64_t src_len, uint8_t* dst, uint64_t dst_cap, uint64_t* out_len,
                void* scratch, uint64_t scratch_len) noexcept {
    assert(out_len != nullptr);
    assert(scratch != nullptr);
    return zstd_decode_raw(src, src_len, dst, dst_cap, out_len, scratch, scratch_len);
}

#ifdef BOLT_HAS_ZSTD

bool ref_built() noexcept { return true; }

uint64_t ref_bound(uint64_t src_len) noexcept {
    const size_t bound = ZSTD_compressBound(static_cast<size_t>(src_len));
    if (ZSTD_isError(bound)) return 0;
    assert(bound >= src_len);
    return static_cast<uint64_t>(bound);
}

int ref_compress(const uint8_t* src, uint64_t src_len, uint8_t* dst, uint64_t* dst_len, int level) noexcept {
    assert(dst_len != nullptr);
    assert(src != nullptr || src_len == 0);
    if (dst_len == nullptr) return kCodecBadArg;
    if (dst == nullptr || *dst_len == 0) return kCodecBufferSmall;
    static thread_local ZSTD_CCtx* cctx = nullptr;
    if (cctx == nullptr) cctx = ZSTD_createCCtx();
    if (cctx == nullptr) return kCodecBufferSmall;
    const size_t r = ZSTD_compressCCtx(cctx, dst, static_cast<size_t>(*dst_len), src, static_cast<size_t>(src_len), level);
    if (ZSTD_isError(r)) return ZSTD_getErrorCode(r) == ZSTD_error_dstSize_tooSmall ? kCodecBufferSmall : kCodecCorrupt;
    assert(r <= *dst_len);
    *dst_len = static_cast<uint64_t>(r);
    return kCodecOk;
}

uint64_t ref_scratch_size() noexcept {
    const uint64_t n = (static_cast<uint64_t>(ZSTD_estimateDCtxSize()) + 7u) & ~uint64_t{7};
    assert(n >= 8);
    assert((n & 7u) == 0);
    return n;
}

int ref_decode(const uint8_t* src, uint64_t src_len, uint8_t* dst, uint64_t dst_cap, uint64_t* out_len,
               void* scratch, uint64_t scratch_len) noexcept {
    assert(out_len != nullptr);
    assert(src != nullptr || src_len == 0);
    if (out_len == nullptr) return kZstdBadInput;
    *out_len = 0;
    if (scratch == nullptr || scratch_len < ref_scratch_size() || (reinterpret_cast<uintptr_t>(scratch) & 7u) != 0)
        return kZstdScratchSmall;
    if (src_len == 0) return kZstdTruncSrc;
    ZSTD_DCtx* d = ZSTD_initStaticDCtx(scratch, static_cast<size_t>(scratch_len));
    if (d == nullptr) return kZstdScratchSmall;
    uint8_t none = 0;   // libzstd rejects a null dst even for an empty frame
    const size_t r = ZSTD_decompressDCtx(d, dst != nullptr ? dst : &none, static_cast<size_t>(dst != nullptr ? dst_cap : 0),
                                         src, static_cast<size_t>(src_len));
    if (!ZSTD_isError(r)) { assert(r <= dst_cap); *out_len = static_cast<uint64_t>(r); return kZstdOk; }
    switch (ZSTD_getErrorCode(r)) {
        case ZSTD_error_dstSize_tooSmall:      return kZstdNeedMore;
        case ZSTD_error_srcSize_wrong:         return kZstdTruncSrc;
        case ZSTD_error_checksum_wrong:        return kZstdChecksum;
        case ZSTD_error_dictionary_wrong:
        case ZSTD_error_dictionary_corrupted:
        case ZSTD_error_frameParameter_unsupported:
        case ZSTD_error_frameParameter_windowTooLarge: return kZstdUnsupported;
        default:                               return kZstdBadInput;
    }
}

#else   // no libzstd: the reference entries exist so the interface links, and refuse.

bool ref_built() noexcept { return false; }
uint64_t ref_bound(uint64_t) noexcept { return 0; }
int ref_compress(const uint8_t*, uint64_t, uint8_t*, uint64_t*, int) noexcept { return kCodecUnavailable; }
uint64_t ref_scratch_size() noexcept { return 0; }
int ref_decode(const uint8_t*, uint64_t, uint8_t*, uint64_t, uint64_t* out_len, void*, uint64_t) noexcept {
    if (out_len != nullptr) *out_len = 0;
    return kZstdUnsupported;
}

#endif

}  // namespace zstd_backend

bool zstd_available() noexcept { return true; }

const char* zstd_encoder_name() noexcept { return BOLT_ZSTD_ENCODER_REFERENCE ? "reference" : "bolt"; }
const char* zstd_decoder_name() noexcept { return BOLT_ZSTD_DECODER_REFERENCE ? "reference" : "bolt"; }
const char* zstd_stream_decoder_name() noexcept { return "bolt"; }

uint64_t zstd_max_compressed_len(uint64_t src_len) noexcept {
#if BOLT_ZSTD_ENCODER_REFERENCE
    return zstd_backend::ref_bound(src_len);
#else
    return zstd_backend::self_bound(src_len);
#endif
}

int zstd_compress(const uint8_t* src, uint64_t src_len, uint8_t* dst, uint64_t* dst_len, int level) noexcept {
    assert(dst_len != nullptr);
    assert(src != nullptr || src_len == 0);
#if BOLT_ZSTD_ENCODER_REFERENCE
    return zstd_backend::ref_compress(src, src_len, dst, dst_len, level);
#else
    return zstd_backend::self_compress(src, src_len, dst, dst_len, level);
#endif
}

uint64_t zstd_decode_scratch_size() noexcept {
#if BOLT_ZSTD_DECODER_REFERENCE
    return zstd_backend::ref_scratch_size();
#else
    return zstd_backend::self_scratch_size();
#endif
}

int zstd_decode(const uint8_t* src, uint64_t src_len, uint8_t* dst, uint64_t dst_cap, uint64_t* out_len,
                void* scratch, uint64_t scratch_len) noexcept {
    assert(out_len != nullptr);
    assert(src != nullptr || src_len == 0);
    if (out_len == nullptr) return kZstdBadInput;
    if (scratch == nullptr || scratch_len < zstd_decode_scratch_size()) { *out_len = 0; return kZstdScratchSmall; }
#if BOLT_ZSTD_DECODER_REFERENCE
    return zstd_backend::ref_decode(src, src_len, dst, dst_cap, out_len, scratch, scratch_len);
#else
    return zstd_backend::self_decode(src, src_len, dst, dst_cap, out_len, scratch, scratch_len);
#endif
}

int zstd_decompress(const uint8_t* src, uint64_t src_len, uint8_t* dst, uint64_t dst_len) noexcept {
    assert(src != nullptr || src_len == 0);
    assert(dst != nullptr || dst_len == 0);
    static thread_local void* scratch = nullptr;   // one decode context per thread
    if (scratch == nullptr) scratch = std::aligned_alloc(64, (zstd_decode_scratch_size() + 63u) & ~uint64_t{63});
    if (scratch == nullptr) return kCodecBufferSmall;
    uint64_t got = 0;
    if (zstd_decode(src, src_len, dst, dst_len, &got, scratch, zstd_decode_scratch_size()) != kZstdOk) return kCodecCorrupt;
    return got == dst_len ? kCodecOk : kCodecCorrupt;
}

}  // namespace ingest
}  // namespace bolt
