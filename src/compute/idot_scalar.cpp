// idot_scalar.cpp -- scalar reference tier for bolt::compute's IDOT GEMV
// (BLLM-14). Compiled at BASELINE ISA (no /arch flag) so it is a genuinely
// safe runtime fallback on any CPU. Ported verbatim (numerics-preserving)
// from boltllm::quant::IdotDispatch.cpp's scalar kernels, retargeted from
// QTensor onto bolt::Tensor -- the ONLY change is weight.fmt/cols/rows ->
// weight.dtype/cols()/rows() and kInt4Offset -> detail::kInt4DequantOffset;
// the load-bearing integer accumulation and the final
// `float(acc) * act_scale * weight.scale(r)` expression are byte-identical,
// so this agrees bit-for-bit with the AVX2/VNNI tiers (exact int64 addition
// is associative regardless of SIMD grouping).

#include "idot_kernels_internal.h"

#include "bolt/bolt_compute.h"  // detail::kInt4DequantOffset

#include <cassert>
#include <cstdint>

namespace bolt::compute::detail {

void idot_i8a_int8_scalar(const int8_t* act_q, float act_scale, const Tensor& weight, float* out,
                          int32_t rows) noexcept {
    assert(act_q != nullptr && out != nullptr);
    assert(weight.dtype == DType::Int8);
    assert(weight.cols() > 0);
    assert(rows > 0 && rows <= weight.rows());
    assert(act_scale > 0.0f);

    const int32_t cols = weight.cols();
    for (int32_t r = 0; r < rows; ++r) {
        const int8_t* wrow = reinterpret_cast<const int8_t*>(weight.row_ptr(r));
        int64_t acc = 0;
        for (int32_t i = 0; i < cols; ++i) {
            acc += static_cast<int64_t>(act_q[i]) * static_cast<int64_t>(wrow[i]);
        }
        out[r] = static_cast<float>(acc) * act_scale * weight.scale(r);
    }
}

void idot_i8a_int4_scalar(const int8_t* act_q, float act_scale, const Tensor& weight, float* out,
                          int32_t rows) noexcept {
    assert(act_q != nullptr && out != nullptr);
    assert(weight.dtype == DType::Int4);
    assert(weight.cols() > 0);
    assert(rows > 0 && rows <= weight.rows());
    assert(act_scale > 0.0f);

    const int32_t cols = weight.cols();
    for (int32_t r = 0; r < rows; ++r) {
        const uint8_t* wrow = weight.row_ptr(r);
        int64_t acc = 0;
        int32_t i = 0;
        for (; i + 1 < cols; i += 2) {
            const uint8_t byte = wrow[i >> 1];
            const int32_t lo_v = static_cast<int32_t>(byte & 0x0F) - kInt4DequantOffset;
            const int32_t hi_v = static_cast<int32_t>(byte >> 4) - kInt4DequantOffset;
            acc += static_cast<int64_t>(act_q[i]) * lo_v + static_cast<int64_t>(act_q[i + 1]) * hi_v;
        }
        if (i < cols) {
            const uint8_t byte = wrow[i >> 1];
            const int32_t lo_v = static_cast<int32_t>(byte & 0x0F) - kInt4DequantOffset;
            acc += static_cast<int64_t>(act_q[i]) * lo_v;
        }
        out[r] = static_cast<float>(acc) * act_scale * weight.scale(r);
    }
}

}  // namespace bolt::compute::detail
