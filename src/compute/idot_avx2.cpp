// idot_avx2.cpp -- AVX2 tier for bolt::compute's IDOT GEMV (BLLM-14).
// #if BOLT_ARCH_X86 (empty TU on non-x86, so macos-arm and any ARM consumer
// links a scalar-only path with no dispatch reference to these symbols --
// idot_dispatch.cpp only names idot_i8a_*_avx2 inside its own BOLT_ARCH_X86
// block). Compiled at /arch:AVX2 (MSVC) or -mavx2 -mfma (GCC/Clang).
//
// Ported verbatim (numerics-preserving) from boltllm::quant::IdotAvx2.cpp.
// The dot_row_* helpers take raw pointers and carry over UNCHANGED; only the
// gemv wrappers switch QTensor -> bolt::Tensor (weight.fmt/cols/rows ->
// weight.dtype/cols()/rows(), kInt4Offset -> detail::kInt4DequantOffset).
//
// Int8 sign trick: VPMADDUBSW takes one unsigned + one signed byte operand,
// but both act_q and the weight row are signed int8, so we use the standard
// AVX2 idiom abs_a = |a|, signed_b = sign(a)*b, prod16 = maddubs(abs_a,
// signed_b) == a*b per element. For a,b in [-127,127] a pairwise int16 sum is
// <= 32258, inside int16 range (no saturation); the a==b==INT8_MIN corner
// lands exactly on the int16 boundary and is documented-accepted (identical
// analysis to the boltllm source this ports from and to production int8 AVX2
// kernels generally, e.g. ggml/llama.cpp q8_0 paths).

#include "bolt/bolt_port.h"

#if BOLT_ARCH_X86

#include "idot_kernels_internal.h"

#include "bolt/bolt_compute.h"  // detail::kInt4DequantOffset

#include <cassert>
#include <cstdint>

#include <immintrin.h>

namespace bolt::compute::detail {

namespace {

// One 32-byte lane of the sign-trick dot product: 8 int32 partial sums, NOT
// yet horizontally reduced, so callers accumulate many chunks into a running
// __m256i and do a single horizontal reduce per row.
inline __m256i dot32_i8i8_partial(__m256i a, __m256i b) noexcept {
    const __m256i abs_a = _mm256_sign_epi8(a, a);
    const __m256i signed_b = _mm256_sign_epi8(b, a);
    const __m256i prod16 = _mm256_maddubs_epi16(abs_a, signed_b);
    return _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));
}

inline int32_t hsum_epi32_avx2(__m256i v) noexcept {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    const __m128i shuf1 = _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1));
    s = _mm_add_epi32(s, shuf1);
    const __m128i shuf2 = _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2));
    s = _mm_add_epi32(s, shuf2);
    return _mm_cvtsi128_si32(s);
}

// Exact int64 dot of two int8 rows, cols wide. SIMD over 32-element chunks,
// scalar tail (< 32).
int64_t dot_row_int8_avx2(const int8_t* act_q, const int8_t* wrow, int32_t cols) noexcept {
    int32_t i = 0;
    __m256i acc = _mm256_setzero_si256();
    for (; i + 32 <= cols; i += 32) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act_q + i));
        const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wrow + i));
        acc = _mm256_add_epi32(acc, dot32_i8i8_partial(a, w));
    }
    int64_t total = static_cast<int64_t>(hsum_epi32_avx2(acc));
    for (; i < cols; ++i) {
        total += static_cast<int64_t>(act_q[i]) * static_cast<int64_t>(wrow[i]);
    }
    return total;
}

// Int4 in-register nibble unpack (low-nibble-first, offset 8). Given 32
// packed bytes (= 64 int4 elements), unpacks to two __m256i of 32 signed int8
// each in NATURAL element order. See the boltllm source's comment for the
// per-byte srli_epi16+mask derivation and the permute2x128 lane-recombine.
inline void unpack_int4_chunk_to_i8(const uint8_t* packed32, __m256i& out_first32,
                                    __m256i& out_second32) noexcept {
    const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed32));
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i lo = _mm256_and_si256(packed, low_mask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
    const __m256i offset8 = _mm256_set1_epi8(static_cast<char>(kInt4DequantOffset));
    const __m256i lo_s = _mm256_sub_epi8(lo, offset8);
    const __m256i hi_s = _mm256_sub_epi8(hi, offset8);
    const __m256i part_a = _mm256_unpacklo_epi8(lo_s, hi_s);
    const __m256i part_b = _mm256_unpackhi_epi8(lo_s, hi_s);
    out_first32 = _mm256_permute2x128_si256(part_a, part_b, 0x20);
    out_second32 = _mm256_permute2x128_si256(part_a, part_b, 0x31);
}

// Exact int64 dot of int8 activation against an int4-packed weight row, cols
// wide. SIMD over 64-element chunks (32 packed bytes), scalar nibble tail.
int64_t dot_row_int4_avx2(const int8_t* act_q, const uint8_t* wrow_packed, int32_t cols) noexcept {
    int32_t i = 0;
    __m256i acc = _mm256_setzero_si256();
    for (; i + 64 <= cols; i += 64) {
        __m256i w0;
        __m256i w1;
        unpack_int4_chunk_to_i8(wrow_packed + (i >> 1), w0, w1);
        const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act_q + i));
        const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act_q + i + 32));
        acc = _mm256_add_epi32(acc, dot32_i8i8_partial(a0, w0));
        acc = _mm256_add_epi32(acc, dot32_i8i8_partial(a1, w1));
    }
    int64_t total = static_cast<int64_t>(hsum_epi32_avx2(acc));
    for (; i + 1 < cols; i += 2) {
        const uint8_t byte = wrow_packed[i >> 1];
        const int32_t lo_v = static_cast<int32_t>(byte & 0x0F) - kInt4DequantOffset;
        const int32_t hi_v = static_cast<int32_t>(byte >> 4) - kInt4DequantOffset;
        total += static_cast<int64_t>(act_q[i]) * lo_v + static_cast<int64_t>(act_q[i + 1]) * hi_v;
    }
    if (i < cols) {
        const uint8_t byte = wrow_packed[i >> 1];
        const int32_t lo_v = static_cast<int32_t>(byte & 0x0F) - kInt4DequantOffset;
        total += static_cast<int64_t>(act_q[i]) * lo_v;
    }
    return total;
}

}  // namespace

void idot_i8a_int8_avx2(const int8_t* act_q, float act_scale, const Tensor& weight, float* out,
                        int32_t rows) noexcept {
    assert(act_q != nullptr && out != nullptr);
    assert(weight.dtype == DType::Int8);
    assert(weight.cols() > 0);
    assert(rows > 0 && rows <= weight.rows());
    assert(act_scale > 0.0f);

    const int32_t cols = weight.cols();
    for (int32_t r = 0; r < rows; ++r) {
        const int8_t* wrow = reinterpret_cast<const int8_t*>(weight.row_ptr(r));
        const int64_t acc = dot_row_int8_avx2(act_q, wrow, cols);
        out[r] = static_cast<float>(acc) * act_scale * weight.scale(r);
    }
}

void idot_i8a_int4_avx2(const int8_t* act_q, float act_scale, const Tensor& weight, float* out,
                        int32_t rows) noexcept {
    assert(act_q != nullptr && out != nullptr);
    assert(weight.dtype == DType::Int4);
    assert(weight.cols() > 0);
    assert(rows > 0 && rows <= weight.rows());
    assert(act_scale > 0.0f);

    const int32_t cols = weight.cols();
    for (int32_t r = 0; r < rows; ++r) {
        const uint8_t* wrow = weight.row_ptr(r);
        const int64_t acc = dot_row_int4_avx2(act_q, wrow, cols);
        out[r] = static_cast<float>(acc) * act_scale * weight.scale(r);
    }
}

}  // namespace bolt::compute::detail

#endif  // BOLT_ARCH_X86
