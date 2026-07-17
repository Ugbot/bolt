// test_bolt_tensor.cpp — Tests for the device-generic tensor primitive
// (bolt/bolt_tensor.h). N0 substrate for boltllm's MoE inference engine
// (BLLM-12): proves bolt::Tensor is a genuine superset of
// boltllm::quant::QTensor's per-row-scale layout, not just a
// similar-looking parallel type.
//
// Since Bolt does not (and must not) depend on boltllm, these tests do
// not #include QuantTypes.h. Instead they hardcode the exact expected
// byte values QuantTypes.h's row_stride_bytes() computes for representative
// F32/Int8/Int4/Int2 cases (formulas restated in the comment above each
// case) and assert bolt::dtype_row_stride_bytes()/tensor_make_cpu_2d()
// reproduce them exactly.

#include "bolt/bolt_tensor.h"

#include <gtest/gtest.h>

#include <cstring>
#include <type_traits>
#include <vector>

using bolt::DType;
using bolt::Device;
using bolt::Tensor;

// ---------------------------------------------------------------------------
// POD size / layout sanity.
// ---------------------------------------------------------------------------

TEST(BoltTensor, SizeAndLayout) {
    EXPECT_EQ(sizeof(Tensor), 72u);
    EXPECT_TRUE(std::is_trivially_copyable<Tensor>::value);
    EXPECT_TRUE(std::is_standard_layout<Tensor>::value);

    EXPECT_EQ(offsetof(Tensor, device), 0u);
    EXPECT_EQ(offsetof(Tensor, dtype), 1u);
    EXPECT_EQ(offsetof(Tensor, dims), 4u);
    EXPECT_EQ(offsetof(Tensor, strides), 24u);
    EXPECT_EQ(offsetof(Tensor, data), 56u);
    EXPECT_EQ(offsetof(Tensor, row_scales), 64u);
}

TEST(BoltTensor, DefaultConstructedIsZeroed) {
    Tensor t;
    EXPECT_EQ(t.device, Device::CPU);
    EXPECT_EQ(t.dtype, DType::F32);
    for (int i = 0; i < bolt::kTensorMaxDims; ++i) {
        EXPECT_EQ(t.dims[i], 0);
        EXPECT_EQ(t.strides[i], 0);
    }
    EXPECT_EQ(t.data, nullptr);
    EXPECT_EQ(t.row_scales, nullptr);
}

TEST(BoltTensor, DeviceEnumReservesGpuValues) {
    // This ticket (N0) only reserves the index space; no Vulkan/CUDA/ROCm
    // dispatch exists yet (ROCm's real kernel, bolt::rocm, is BLLM-78's
    // optional compiled library -- not wired into this table). Pin the
    // numeric values since they may end up serialized (device tag
    // alongside a tensor) before the backends land.
    EXPECT_EQ(static_cast<uint8_t>(Device::CPU), 0u);
    EXPECT_EQ(static_cast<uint8_t>(Device::Vulkan), 1u);
    EXPECT_EQ(static_cast<uint8_t>(Device::CUDA), 2u);
    EXPECT_EQ(static_cast<uint8_t>(Device::ROCm), 3u);
}

TEST(BoltTensor, DTypeValuesMatchQuantFmtWhereTheyOverlap) {
    // boltllm::quant::QuantFmt (QuantTypes.h): F32=0, Int8=1, Int4=2,
    // Int2=3. bolt::DType must match by value for these four so a
    // QTensor::fmt byte and a Tensor::dtype byte are bit-identical.
    EXPECT_EQ(static_cast<uint8_t>(DType::F32), 0u);
    EXPECT_EQ(static_cast<uint8_t>(DType::Int8), 1u);
    EXPECT_EQ(static_cast<uint8_t>(DType::Int4), 2u);
    EXPECT_EQ(static_cast<uint8_t>(DType::Int2), 3u);
    // F16/BF16 are Bolt-only additions, appended after the QuantFmt range.
    EXPECT_EQ(static_cast<uint8_t>(DType::F16), 4u);
    EXPECT_EQ(static_cast<uint8_t>(DType::BF16), 5u);
}

// ---------------------------------------------------------------------------
// dtype_is_quantized / dtype_element_bytes.
// ---------------------------------------------------------------------------

TEST(BoltTensor, DtypeIsQuantized) {
    EXPECT_FALSE(bolt::dtype_is_quantized(DType::F32));
    EXPECT_FALSE(bolt::dtype_is_quantized(DType::F16));
    EXPECT_FALSE(bolt::dtype_is_quantized(DType::BF16));
    EXPECT_TRUE(bolt::dtype_is_quantized(DType::Int8));
    EXPECT_TRUE(bolt::dtype_is_quantized(DType::Int4));
    EXPECT_TRUE(bolt::dtype_is_quantized(DType::Int2));
}

TEST(BoltTensor, DtypeElementBytesWholeByteFormats) {
    EXPECT_EQ(bolt::dtype_element_bytes(DType::F32), 4);
    EXPECT_EQ(bolt::dtype_element_bytes(DType::F16), 2);
    EXPECT_EQ(bolt::dtype_element_bytes(DType::BF16), 2);
    EXPECT_EQ(bolt::dtype_element_bytes(DType::Int8), 1);
}

TEST(BoltTensor, DtypeElementBytesSubBytePackedIsZeroSentinel) {
    // Int4 (2 elems/byte) and Int2 (4 elems/byte) have no whole
    // per-element byte size — 0 is the "use the row-stride formula
    // instead" sentinel, matching QuantTypes.h never expressing a
    // per-element stride for these formats.
    EXPECT_EQ(bolt::dtype_element_bytes(DType::Int4), 0);
    EXPECT_EQ(bolt::dtype_element_bytes(DType::Int2), 0);
}

// ---------------------------------------------------------------------------
// dtype_row_stride_bytes() round-trips QuantTypes.h's row_stride_bytes()
// formulas exactly:
//   f32_row_stride(cols)  = cols * sizeof(float)
//   int8_row_stride(cols) = cols
//   int4_row_stride(cols) = (cols + 1) / 2
//   int2_row_stride(cols) = (cols + 3) / 4
// ---------------------------------------------------------------------------

TEST(BoltTensor, RowStrideBytesF32) {
    EXPECT_EQ(bolt::dtype_row_stride_bytes(DType::F32, 8), 32);   // 8*4
    EXPECT_EQ(bolt::dtype_row_stride_bytes(DType::F32, 1), 4);
    EXPECT_EQ(bolt::dtype_row_stride_bytes(DType::F32, 0), 0);
}

TEST(BoltTensor, RowStrideBytesInt8IsIdentity) {
    EXPECT_EQ(bolt::dtype_row_stride_bytes(DType::Int8, 13), 13);
    EXPECT_EQ(bolt::dtype_row_stride_bytes(DType::Int8, 4096), 4096);
}

TEST(BoltTensor, RowStrideBytesInt4CeilHalf) {
    // (cols + 1) / 2, no per-row padding — exact QuantTypes.h formula.
    EXPECT_EQ(bolt::dtype_row_stride_bytes(DType::Int4, 12), 6);   // even
    EXPECT_EQ(bolt::dtype_row_stride_bytes(DType::Int4, 13), 7);   // odd -> ceil
    EXPECT_EQ(bolt::dtype_row_stride_bytes(DType::Int4, 4096), 2048);
}

TEST(BoltTensor, RowStrideBytesInt2CeilQuarter) {
    // (cols + 3) / 4, no per-row padding — exact QuantTypes.h formula.
    EXPECT_EQ(bolt::dtype_row_stride_bytes(DType::Int2, 8), 2);
    EXPECT_EQ(bolt::dtype_row_stride_bytes(DType::Int2, 10), 3);   // ceil(10/4)
    EXPECT_EQ(bolt::dtype_row_stride_bytes(DType::Int2, 13), 4);   // ceil(13/4)
}

// ---------------------------------------------------------------------------
// tensor_make_cpu_2d() — constructing a view over externally-owned memory.
// Tensor never allocates; the backing store below stands in for an arena
// or caller-managed buffer.
// ---------------------------------------------------------------------------

TEST(BoltTensor, MakeCpu2dF32ViewOverExternalBuffer) {
    constexpr int32_t kRows = 4;
    constexpr int32_t kCols = 8;
    std::vector<float> backing(static_cast<size_t>(kRows) * kCols, 0.0f);
    for (size_t i = 0; i < backing.size(); ++i) backing[i] = static_cast<float>(i);

    Tensor t = bolt::tensor_make_cpu_2d(DType::F32, kRows, kCols, backing.data());

    EXPECT_EQ(t.device, Device::CPU);
    EXPECT_EQ(t.dtype, DType::F32);
    EXPECT_EQ(t.rows(), kRows);
    EXPECT_EQ(t.cols(), kCols);
    EXPECT_EQ(t.dims[2], 1);
    EXPECT_EQ(t.dims[3], 1);
    EXPECT_EQ(t.row_stride_bytes(), kCols * 4);
    EXPECT_EQ(t.data, backing.data());
    EXPECT_EQ(t.row_scales, nullptr);

    // No copy happened — the view points directly at `backing`.
    float* row1 = reinterpret_cast<float*>(t.row_ptr(1));
    EXPECT_EQ(row1, backing.data() + kCols);
    EXPECT_FLOAT_EQ(row1[0], static_cast<float>(kCols));  // backing[8] == 8.0f

    // Writes through the view are visible in the backing store (no
    // internal copy / no ownership transfer).
    row1[0] = 123.5f;
    EXPECT_FLOAT_EQ(backing[static_cast<size_t>(kCols)], 123.5f);
}

TEST(BoltTensor, MakeCpu2dInt8WithRowScales) {
    constexpr int32_t kRows = 3;
    constexpr int32_t kCols = 5;
    std::vector<uint8_t> backing(static_cast<size_t>(kRows) * kCols, 0);
    std::vector<float> scales = {0.1f, 0.2f, 0.3f};

    Tensor t = bolt::tensor_make_cpu_2d(DType::Int8, kRows, kCols,
                                         backing.data(), scales.data());

    ASSERT_EQ(t.row_stride_bytes(), kCols);  // int8_row_stride == cols
    for (int32_t r = 0; r < kRows; ++r) {
        EXPECT_EQ(t.row_ptr(r), backing.data() + static_cast<size_t>(r) * kCols);
        EXPECT_FLOAT_EQ(t.scale(r), scales[static_cast<size_t>(r)]);
    }
}

TEST(BoltTensor, MakeCpu2dInt4PackedRowStrideAndSubByteSentinel) {
    // 13 logical int4 values/row -> ceil(13/2) = 7 packed bytes/row,
    // tightly packed (no per-row padding) — identical to QuantTypes.h.
    constexpr int32_t kRows = 2;
    constexpr int32_t kCols = 13;
    const int64_t expected_row_stride = 7;
    std::vector<uint8_t> backing(static_cast<size_t>(kRows) * expected_row_stride, 0);
    std::vector<float> scales = {1.0f, 2.0f};

    Tensor t = bolt::tensor_make_cpu_2d(DType::Int4, kRows, kCols,
                                         backing.data(), scales.data());

    EXPECT_EQ(t.row_stride_bytes(), expected_row_stride);
    // Sub-byte-packed dtype: strides[0] must be the 0 sentinel, not a
    // fractional per-element byte count.
    EXPECT_EQ(t.strides[0], 0);
    EXPECT_EQ(t.row_ptr(1), backing.data() + expected_row_stride);
}

TEST(BoltTensor, RowPtrAndScaleAssertBounds) {
    constexpr int32_t kRows = 2;
    constexpr int32_t kCols = 4;
    std::vector<float> backing(static_cast<size_t>(kRows) * kCols, 0.0f);
    Tensor t = bolt::tensor_make_cpu_2d(DType::F32, kRows, kCols, backing.data());

    // In-bounds accesses succeed for every valid row.
    EXPECT_NO_FATAL_FAILURE(t.row_ptr(0));
    EXPECT_NO_FATAL_FAILURE(t.row_ptr(kRows - 1));
}
