// idot_avx512vnni.cpp -- AVX-512VNNI tier for bolt::compute's IDOT GEMV
// (BLLM-14). #if BOLT_ARCH_X86 (empty TU on non-x86). Compiled at
// /arch:AVX512 (MSVC) or -mavx2 -mfma -mavx512f -mavx512bw -mavx512vnni
// (GCC/Clang -- bolt's AVX512 SIMD tier omits -mavx512vnni/-mfma, so the
// CMake target adds them per-source; see CMakeLists.txt).
//
// Ported verbatim (numerics-preserving) from boltllm::quant::IdotAvx512Vnni.cpp.
// dot_row_* helpers carry over unchanged (raw pointers); gemv wrappers switch
// QTensor -> bolt::Tensor. VPDPBUSD (`_mm512_dpbusd_epi32`) folds
// multiply + 4-wide horizontal add + int32 accumulate into one instruction;
// the same sign trick as the AVX2 tier (abs_a, sign(a)*b) adapts it to
// signed*signed int8. Accumulation is int32 (no int16 saturation stage), so
// even the INT8_MIN corner introduces no overflow risk for realistic column
// counts (a lane would need ~2.1M columns to overflow int32). See the boltllm
// source for the full saturation/overflow derivation.

#include "bolt/bolt_port.h"

#if BOLT_ARCH_X86

#include "idot_kernels_internal.h"

#include "bolt/bolt_compute.h"  // detail::kInt4DequantOffset

#include <cassert>
#include <cstdint>

#include <immintrin.h>

namespace bolt::compute::detail {

namespace {

// One 64-byte lane: VPDPBUSD folds multiply + 4-wide sum + accumulate, so
// unlike AVX2 there is no separate "not-yet-reduced" return -- this directly
// updates and returns the running __m512i accumulator (16 int32 lanes).
inline __m512i dot64_i8i8_dpbusd_accum(__m512i acc, __m512i a, __m512i b) noexcept {
    const __m512i abs_a = _mm512_abs_epi8(a);
    const __mmask64 neg_mask = _mm512_movepi8_mask(a);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i signed_b = _mm512_mask_sub_epi8(b, neg_mask, zero, b);  // sign(a)*b
    return _mm512_dpbusd_epi32(acc, abs_a, signed_b);
}

int64_t dot_row_int8_avx512vnni(const int8_t* act_q, const int8_t* wrow, int32_t cols) noexcept {
    int32_t i = 0;
    __m512i acc = _mm512_setzero_si512();
    for (; i + 64 <= cols; i += 64) {
        const __m512i a = _mm512_loadu_si512(reinterpret_cast<const void*>(act_q + i));
        const __m512i w = _mm512_loadu_si512(reinterpret_cast<const void*>(wrow + i));
        acc = dot64_i8i8_dpbusd_accum(acc, a, w);
    }
    int64_t total = static_cast<int64_t>(_mm512_reduce_add_epi32(acc));
    for (; i < cols; ++i) {
        total += static_cast<int64_t>(act_q[i]) * static_cast<int64_t>(wrow[i]);
    }
    return total;
}

// Int4 nibble unpack, 512-bit / 4x-128-bit-lane. Given 64 packed bytes (= 128
// int4 elements), unpacks to two __m512i of 64 signed int8 each in natural
// order. Uses permutex2var_epi64 for the per-lane a/b/a/b interleave (the
// AVX2 permute2x128 analogue can't reproduce it in one call). See boltllm
// source for the qword-index derivation.
inline void unpack_int4_chunk_to_i8_512(const uint8_t* packed64, __m512i& out_first64,
                                        __m512i& out_second64) noexcept {
    const __m512i packed = _mm512_loadu_si512(reinterpret_cast<const void*>(packed64));
    const __m512i low_mask = _mm512_set1_epi8(0x0F);
    const __m512i lo = _mm512_and_si512(packed, low_mask);
    const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(packed, 4), low_mask);
    const __m512i offset8 = _mm512_set1_epi8(static_cast<char>(kInt4DequantOffset));
    const __m512i lo_s = _mm512_sub_epi8(lo, offset8);
    const __m512i hi_s = _mm512_sub_epi8(hi, offset8);
    const __m512i part_a = _mm512_unpacklo_epi8(lo_s, hi_s);
    const __m512i part_b = _mm512_unpackhi_epi8(lo_s, hi_s);
    const __m512i idx_low = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    const __m512i idx_high = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
    out_first64 = _mm512_permutex2var_epi64(part_a, idx_low, part_b);
    out_second64 = _mm512_permutex2var_epi64(part_a, idx_high, part_b);
}

int64_t dot_row_int4_avx512vnni(const int8_t* act_q, const uint8_t* wrow_packed,
                                int32_t cols) noexcept {
    int32_t i = 0;
    __m512i acc = _mm512_setzero_si512();
    for (; i + 128 <= cols; i += 128) {
        __m512i w0;
        __m512i w1;
        unpack_int4_chunk_to_i8_512(wrow_packed + (i >> 1), w0, w1);
        const __m512i a0 = _mm512_loadu_si512(reinterpret_cast<const void*>(act_q + i));
        const __m512i a1 = _mm512_loadu_si512(reinterpret_cast<const void*>(act_q + i + 64));
        acc = dot64_i8i8_dpbusd_accum(acc, a0, w0);
        acc = dot64_i8i8_dpbusd_accum(acc, a1, w1);
    }
    int64_t total = static_cast<int64_t>(_mm512_reduce_add_epi32(acc));
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

void idot_i8a_int8_avx512vnni(const int8_t* act_q, float act_scale, const Tensor& weight,
                              float* out, int32_t rows) noexcept {
    assert(act_q != nullptr && out != nullptr);
    assert(weight.dtype == DType::Int8);
    assert(weight.cols() > 0);
    assert(rows > 0 && rows <= weight.rows());
    assert(act_scale > 0.0f);

    const int32_t cols = weight.cols();
    for (int32_t r = 0; r < rows; ++r) {
        const int8_t* wrow = reinterpret_cast<const int8_t*>(weight.row_ptr(r));
        const int64_t acc = dot_row_int8_avx512vnni(act_q, wrow, cols);
        out[r] = static_cast<float>(acc) * act_scale * weight.scale(r);
    }
}

void idot_i8a_int4_avx512vnni(const int8_t* act_q, float act_scale, const Tensor& weight,
                              float* out, int32_t rows) noexcept {
    assert(act_q != nullptr && out != nullptr);
    assert(weight.dtype == DType::Int4);
    assert(weight.cols() > 0);
    assert(rows > 0 && rows <= weight.rows());
    assert(act_scale > 0.0f);

    const int32_t cols = weight.cols();
    for (int32_t r = 0; r < rows; ++r) {
        const uint8_t* wrow = weight.row_ptr(r);
        const int64_t acc = dot_row_int4_avx512vnni(act_q, wrow, cols);
        out[r] = static_cast<float>(acc) * act_scale * weight.scale(r);
    }
}

}  // namespace bolt::compute::detail

#endif  // BOLT_ARCH_X86
