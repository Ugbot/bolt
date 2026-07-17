// test_bolt_vulkan.cpp -- coverage for bolt::vulkan's device/queue/memory
// scaffolding (BLLM-46). Runs against whatever REAL Vulkan device this
// machine has (this box: AMD Radeon 8060S integrated GPU) -- self-skips
// (not a hard failure) if no Vulkan-capable device is present, matching
// this module's own "runtime-detected, CPU fallback always works"
// contract. STL is allowed in tests; production code stays Tiger-Style.

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "bolt/bolt_vulkan.h"

using bolt::vulkan::BufferHandle;
using bolt::vulkan::Context;
using bolt::vulkan::MatmulPipeline;
using bolt::vulkan::WeightDType;

TEST(BoltVulkan, AvailableDoesNotCrashRegardlessOfHardware) {
    // Must be callable (and return SOME bool) whether or not a device
    // exists -- this is the one assertion that should hold on ANY machine,
    // Vulkan-capable or not.
    const bool result = bolt::vulkan::available();
    EXPECT_TRUE(result == true || result == false);  // trivially true; documents intent
}

TEST(BoltVulkan, ContextCreateSucceedsOnThisRealDevice) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    EXPECT_TRUE(ctx.is_valid());
    EXPECT_GT(std::strlen(ctx.device_name()), 0u);
    EXPECT_GT(ctx.device_local_heap_bytes(), 0u);
    ctx.destroy();
    EXPECT_FALSE(ctx.is_valid());
}

TEST(BoltVulkan, DestroyIsSafeOnNeverCreatedContext) {
    Context ctx;
    ctx.destroy();  // must not crash
    EXPECT_FALSE(ctx.is_valid());
}

TEST(BoltVulkan, DestructorRunsCleanlyWithoutExplicitDestroy) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    {
        Context ctx;
        ASSERT_TRUE(ctx.create());
    }  // ~Context() must clean up without leaking/crashing
    SUCCEED();
}

TEST(BoltVulkan, AllocateAndMapHostVisibleBuffer) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());

    BufferHandle handle;
    ASSERT_TRUE(ctx.allocate_buffer(4096, /*prefer_host_visible=*/true, &handle));
    EXPECT_TRUE(handle.host_visible);
    EXPECT_EQ(handle.size_bytes, 4096u);

    void* ptr = ctx.map_buffer(handle);
    ASSERT_NE(ptr, nullptr);
    std::memset(ptr, 0xAB, 4096);
    // Re-read through the same mapping -- proves the map is a real,
    // writable view over the allocation, not a stub returning garbage.
    EXPECT_EQ(static_cast<unsigned char*>(ptr)[0], 0xAB);
    EXPECT_EQ(static_cast<unsigned char*>(ptr)[4095], 0xAB);
    ctx.unmap_buffer(handle);

    ctx.free_buffer(&handle);
    EXPECT_EQ(handle.buffer, nullptr);
    EXPECT_EQ(handle.memory, nullptr);
}

TEST(BoltVulkan, AllocateDeviceLocalBufferSucceeds) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());

    BufferHandle handle;
    // On this box's integrated GPU, DEVICE_LOCAL-preferring allocation may
    // still land on a host-visible-capable heap (UMA) -- the point of this
    // test is just that allocation succeeds and produces a usable handle,
    // not that host_visible is false.
    ASSERT_TRUE(ctx.allocate_buffer(1024 * 1024, /*prefer_host_visible=*/false, &handle));
    EXPECT_NE(handle.buffer, nullptr);
    EXPECT_NE(handle.memory, nullptr);
    ctx.free_buffer(&handle);
}

TEST(BoltVulkan, FreeBufferIsSafeOnDefaultConstructedHandle) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    BufferHandle handle;  // never allocated
    ctx.free_buffer(&handle);  // must not crash
    SUCCEED();
}

TEST(BoltVulkan, AllocateBufferFailsCleanlyOnInvalidContext) {
    Context ctx;  // create() never called
    BufferHandle handle;
    EXPECT_FALSE(ctx.allocate_buffer(1024, true, &handle));
    EXPECT_EQ(handle.buffer, nullptr);
}

// ---------------------------------------------------------------------------
// BLLM-47: matmul+dequant compute pipeline. Runs against this box's real
// AMD Radeon 8060S -- these are genuine hardware-validated results, not
// just structural/compile checks.
// ---------------------------------------------------------------------------

namespace {

BufferHandle make_and_fill(Context& ctx, const void* data, uint64_t size_bytes) {
    BufferHandle h;
    if (!ctx.allocate_buffer(size_bytes, /*prefer_host_visible=*/true, &h)) return BufferHandle{};
    void* ptr = ctx.map_buffer(h);
    if (ptr == nullptr) {
        ctx.free_buffer(&h);
        return BufferHandle{};
    }
    std::memcpy(ptr, data, size_bytes);
    ctx.unmap_buffer(h);
    return h;
}

}  // namespace

TEST(BoltVulkan, CreateAndDestroyMatmulPipelineForEachDtype) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());

    for (WeightDType dtype :
         {WeightDType::F32, WeightDType::Int8, WeightDType::Int4, WeightDType::Int2}) {
        MatmulPipeline pipeline;
        ASSERT_TRUE(ctx.create_matmul_pipeline(dtype, &pipeline));
        EXPECT_TRUE(pipeline.is_valid());
        ctx.destroy_matmul_pipeline(&pipeline);
        EXPECT_FALSE(pipeline.is_valid());
    }
}

TEST(BoltVulkan, MatmulDequantF32MatchesHandComputedDotProduct) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    MatmulPipeline pipeline;
    ASSERT_TRUE(ctx.create_matmul_pipeline(WeightDType::F32, &pipeline));

    // 2 rows x 3 cols: row0=[1,2,3], row1=[4,5,6]; activation=[1,1,1] ->
    // out = [6, 15].
    const std::vector<float> weight = {1, 2, 3, 4, 5, 6};
    const float activation[3] = {1.0f, 1.0f, 1.0f};

    BufferHandle weight_buf = make_and_fill(ctx, weight.data(), weight.size() * sizeof(float));
    BufferHandle act_buf = make_and_fill(ctx, activation, sizeof(activation));
    BufferHandle out_buf;
    ASSERT_TRUE(ctx.allocate_buffer(2 * sizeof(float), true, &out_buf));
    ASSERT_NE(weight_buf.buffer, nullptr);
    ASSERT_NE(act_buf.buffer, nullptr);

    ASSERT_TRUE(ctx.matmul_dequant(pipeline, weight_buf, BufferHandle{}, act_buf, out_buf,
                                    /*out_rows=*/2, /*cols=*/3));

    float* out_ptr = static_cast<float*>(ctx.map_buffer(out_buf));
    ASSERT_NE(out_ptr, nullptr);
    EXPECT_FLOAT_EQ(out_ptr[0], 6.0f);
    EXPECT_FLOAT_EQ(out_ptr[1], 15.0f);
    ctx.unmap_buffer(out_buf);

    ctx.free_buffer(&weight_buf);
    ctx.free_buffer(&act_buf);
    ctx.free_buffer(&out_buf);
    ctx.destroy_matmul_pipeline(&pipeline);
}

TEST(BoltVulkan, MatmulDequantInt8MatchesScaledDotProduct) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    MatmulPipeline pipeline;
    ASSERT_TRUE(ctx.create_matmul_pipeline(WeightDType::Int8, &pipeline));

    // 1 row x 4 cols, int8 weights all 10, scale 0.5, activation
    // [1, 2, 1, 0.5] -> raw dot = 10*(1+2+1+0.5) = 45 -> scaled = 22.5.
    const std::vector<int8_t> weight = {10, 10, 10, 10};
    const float scale = 0.5f;
    const float activation[4] = {1.0f, 2.0f, 1.0f, 0.5f};
    const float expected = 10.0f * (1.0f + 2.0f + 1.0f + 0.5f) * scale;

    BufferHandle weight_buf = make_and_fill(ctx, weight.data(), weight.size() * sizeof(int8_t));
    BufferHandle scale_buf = make_and_fill(ctx, &scale, sizeof(float));
    BufferHandle act_buf = make_and_fill(ctx, activation, sizeof(activation));
    BufferHandle out_buf;
    ASSERT_TRUE(ctx.allocate_buffer(sizeof(float), true, &out_buf));

    ASSERT_TRUE(ctx.matmul_dequant(pipeline, weight_buf, scale_buf, act_buf, out_buf,
                                    /*out_rows=*/1, /*cols=*/4));

    float* out_ptr = static_cast<float*>(ctx.map_buffer(out_buf));
    ASSERT_NE(out_ptr, nullptr);
    EXPECT_NEAR(out_ptr[0], expected, 1e-3f);
    ctx.unmap_buffer(out_buf);

    ctx.free_buffer(&weight_buf);
    ctx.free_buffer(&scale_buf);
    ctx.free_buffer(&act_buf);
    ctx.free_buffer(&out_buf);
    ctx.destroy_matmul_pipeline(&pipeline);
}

TEST(BoltVulkan, MatmulDequantInt4MatchesScaledDotProduct) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    MatmulPipeline pipeline;
    ASSERT_TRUE(ctx.create_matmul_pipeline(WeightDType::Int4, &pipeline));

    // 1 row x 4 cols. Nibbles (low-nibble-first packing), codes 10,6,12,8
    // -> dequant values (code-8): 2,-2,4,0. Packed: byte0 = lo=10(0xA)
    // hi=6(0x6) -> 0x6A; byte1 = lo=12(0xC) hi=8(0x8) -> 0x8C.
    const std::vector<uint8_t> weight_packed = {0x6A, 0x8C};
    const float scale = 0.5f;
    const float activation[4] = {1.0f, 2.0f, 1.0f, 0.5f};
    // raw = 2*1 + (-2)*2 + 4*1 + 0*0.5 = 2 - 4 + 4 + 0 = 2 -> scaled = 1.0
    const float expected = 1.0f;

    BufferHandle weight_buf =
        make_and_fill(ctx, weight_packed.data(), weight_packed.size() * sizeof(uint8_t));
    BufferHandle scale_buf = make_and_fill(ctx, &scale, sizeof(float));
    BufferHandle act_buf = make_and_fill(ctx, activation, sizeof(activation));
    BufferHandle out_buf;
    ASSERT_TRUE(ctx.allocate_buffer(sizeof(float), true, &out_buf));

    ASSERT_TRUE(ctx.matmul_dequant(pipeline, weight_buf, scale_buf, act_buf, out_buf,
                                    /*out_rows=*/1, /*cols=*/4));

    float* out_ptr = static_cast<float*>(ctx.map_buffer(out_buf));
    ASSERT_NE(out_ptr, nullptr);
    EXPECT_NEAR(out_ptr[0], expected, 1e-3f);
    ctx.unmap_buffer(out_buf);

    ctx.free_buffer(&weight_buf);
    ctx.free_buffer(&scale_buf);
    ctx.free_buffer(&act_buf);
    ctx.free_buffer(&out_buf);
    ctx.destroy_matmul_pipeline(&pipeline);
}

TEST(BoltVulkan, MatmulDequantInt2MatchesScaledDotProduct) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    MatmulPipeline pipeline;
    ASSERT_TRUE(ctx.create_matmul_pipeline(WeightDType::Int2, &pipeline));

    // 1 row x 4 cols. Codes (lowest-bits-first packing) 3,0,2,1 -> dequant
    // values (code-2): 1,-2,0,-1. Packed into one byte:
    // bits[1:0]=3, bits[3:2]=0, bits[5:4]=2, bits[7:6]=1 -> 0x63.
    const std::vector<uint8_t> weight_packed = {0x63};
    const float scale = 0.25f;
    const float activation[4] = {1.0f, 2.0f, 1.0f, 0.5f};
    // raw = 1*1 + (-2)*2 + 0*1 + (-1)*0.5 = 1 - 4 + 0 - 0.5 = -3.5 ->
    // scaled = -0.875
    const float expected = -0.875f;

    BufferHandle weight_buf =
        make_and_fill(ctx, weight_packed.data(), weight_packed.size() * sizeof(uint8_t));
    BufferHandle scale_buf = make_and_fill(ctx, &scale, sizeof(float));
    BufferHandle act_buf = make_and_fill(ctx, activation, sizeof(activation));
    BufferHandle out_buf;
    ASSERT_TRUE(ctx.allocate_buffer(sizeof(float), true, &out_buf));

    ASSERT_TRUE(ctx.matmul_dequant(pipeline, weight_buf, scale_buf, act_buf, out_buf,
                                    /*out_rows=*/1, /*cols=*/4));

    float* out_ptr = static_cast<float*>(ctx.map_buffer(out_buf));
    ASSERT_NE(out_ptr, nullptr);
    EXPECT_NEAR(out_ptr[0], expected, 1e-3f);
    ctx.unmap_buffer(out_buf);

    ctx.free_buffer(&weight_buf);
    ctx.free_buffer(&scale_buf);
    ctx.free_buffer(&act_buf);
    ctx.free_buffer(&out_buf);
    ctx.destroy_matmul_pipeline(&pipeline);
}

// Regression test for a real bug found 2026-07-17 (via boltllm's
// test_gpu_context.cpp): the original Int8/Int4/Int2 shaders assumed each
// row was padded to a whole 4-byte word boundary, but QuantTypes.h's
// host-side packing is TIGHT (no per-row padding) -- row 0 always happened
// to read correctly (offset 0 either way), but every row after it was
// silently corrupted whenever the row's byte stride wasn't itself a
// multiple of 4 (true whenever `cols` isn't a multiple of 4 for Int8, or
// the packed byte-per-row count isn't a multiple of 4 for Int4/Int2). ALL
// of the single-row tests above (out_rows=1) could not have caught this --
// this test deliberately uses multiple rows AND deliberately
// non-4-byte-aligned row strides.
TEST(BoltVulkan, MatmulDequantMultiRowMisalignedStrideRegression) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());

    // Int8: cols=5 -> row stride 5 bytes, not a multiple of 4.
    {
        MatmulPipeline pipeline;
        ASSERT_TRUE(ctx.create_matmul_pipeline(WeightDType::Int8, &pipeline));
        const int32_t rows = 4, cols = 5;
        std::vector<int8_t> weight(static_cast<size_t>(rows * cols));
        std::vector<float> scale(static_cast<size_t>(rows));
        std::vector<float> activation(static_cast<size_t>(cols));
        for (int32_t i = 0; i < cols; ++i) activation[static_cast<size_t>(i)] = 0.5f + static_cast<float>(i);
        std::vector<float> expected(static_cast<size_t>(rows));
        for (int32_t r = 0; r < rows; ++r) {
            scale[static_cast<size_t>(r)] = 0.5f + 0.25f * static_cast<float>(r);
            float acc = 0.0f;
            for (int32_t c = 0; c < cols; ++c) {
                const int8_t v = static_cast<int8_t>((r * 7 + c * 3) % 21 - 10);
                weight[static_cast<size_t>(r * cols + c)] = v;
                acc += activation[static_cast<size_t>(c)] * static_cast<float>(v);
            }
            expected[static_cast<size_t>(r)] = acc * scale[static_cast<size_t>(r)];
        }
        BufferHandle weight_buf = make_and_fill(ctx, weight.data(), weight.size() * sizeof(int8_t));
        BufferHandle scale_buf = make_and_fill(ctx, scale.data(), scale.size() * sizeof(float));
        BufferHandle act_buf = make_and_fill(ctx, activation.data(), activation.size() * sizeof(float));
        BufferHandle out_buf;
        ASSERT_TRUE(ctx.allocate_buffer(static_cast<uint64_t>(rows) * sizeof(float), true, &out_buf));
        ASSERT_TRUE(ctx.matmul_dequant(pipeline, weight_buf, scale_buf, act_buf, out_buf, rows, cols));
        float* out_ptr = static_cast<float*>(ctx.map_buffer(out_buf));
        ASSERT_NE(out_ptr, nullptr);
        for (int32_t r = 0; r < rows; ++r) {
            EXPECT_NEAR(out_ptr[r], expected[static_cast<size_t>(r)], 1e-2f)
                << "Int8 row " << r << " (regression: row-stride-misalignment corruption)";
        }
        ctx.unmap_buffer(out_buf);
        ctx.free_buffer(&weight_buf);
        ctx.free_buffer(&scale_buf);
        ctx.free_buffer(&act_buf);
        ctx.free_buffer(&out_buf);
        ctx.destroy_matmul_pipeline(&pipeline);
    }

    // Int2: cols=9 -> bytes_per_row = ceil(9/4) = 3, not a multiple of 4.
    {
        MatmulPipeline pipeline;
        ASSERT_TRUE(ctx.create_matmul_pipeline(WeightDType::Int2, &pipeline));
        const int32_t rows = 4, cols = 9;
        const size_t bytes_per_row = static_cast<size_t>((cols + 3) / 4);
        std::vector<uint8_t> packed(static_cast<size_t>(rows) * bytes_per_row, 0);
        std::vector<float> scale(static_cast<size_t>(rows));
        std::vector<float> activation(static_cast<size_t>(cols));
        for (int32_t i = 0; i < cols; ++i) activation[static_cast<size_t>(i)] = 0.5f + static_cast<float>(i) * 0.25f;
        std::vector<float> expected(static_cast<size_t>(rows), 0.0f);
        for (int32_t r = 0; r < rows; ++r) {
            scale[static_cast<size_t>(r)] = 0.25f + 0.1f * static_cast<float>(r);
            float acc = 0.0f;
            for (int32_t c = 0; c < cols; ++c) {
                const uint32_t code = static_cast<uint32_t>((r * 3 + c * 5) % 4);  // 0..3
                const int32_t v = static_cast<int32_t>(code) - 2;  // kInt2Offset
                const size_t byte_idx = static_cast<size_t>(r) * bytes_per_row + static_cast<size_t>(c >> 2);
                const uint32_t shift = static_cast<uint32_t>((c & 3) * 2);
                packed[byte_idx] = static_cast<uint8_t>(packed[byte_idx] | (code << shift));
                acc += activation[static_cast<size_t>(c)] * static_cast<float>(v);
            }
            expected[static_cast<size_t>(r)] = acc * scale[static_cast<size_t>(r)];
        }
        BufferHandle weight_buf = make_and_fill(ctx, packed.data(), packed.size());
        BufferHandle scale_buf = make_and_fill(ctx, scale.data(), scale.size() * sizeof(float));
        BufferHandle act_buf = make_and_fill(ctx, activation.data(), activation.size() * sizeof(float));
        BufferHandle out_buf;
        ASSERT_TRUE(ctx.allocate_buffer(static_cast<uint64_t>(rows) * sizeof(float), true, &out_buf));
        ASSERT_TRUE(ctx.matmul_dequant(pipeline, weight_buf, scale_buf, act_buf, out_buf, rows, cols));
        float* out_ptr = static_cast<float*>(ctx.map_buffer(out_buf));
        ASSERT_NE(out_ptr, nullptr);
        for (int32_t r = 0; r < rows; ++r) {
            EXPECT_NEAR(out_ptr[r], expected[static_cast<size_t>(r)], 1e-2f)
                << "Int2 row " << r << " (regression: row-stride-misalignment corruption)";
        }
        ctx.unmap_buffer(out_buf);
        ctx.free_buffer(&weight_buf);
        ctx.free_buffer(&scale_buf);
        ctx.free_buffer(&act_buf);
        ctx.free_buffer(&out_buf);
        ctx.destroy_matmul_pipeline(&pipeline);
    }
}

TEST(BoltVulkan, MatmulDequantFailsCleanlyOnInvalidPipeline) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    MatmulPipeline pipeline;  // never created
    BufferHandle weight, activation, out;
    ASSERT_TRUE(ctx.allocate_buffer(16, true, &weight));
    ASSERT_TRUE(ctx.allocate_buffer(16, true, &activation));
    ASSERT_TRUE(ctx.allocate_buffer(16, true, &out));
    EXPECT_FALSE(ctx.matmul_dequant(pipeline, weight, BufferHandle{}, activation, out, 1, 4));
    ctx.free_buffer(&weight);
    ctx.free_buffer(&activation);
    ctx.free_buffer(&out);
}

TEST(BoltVulkan, MatmulDequantRequiresScaleForQuantizedDtypes) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    MatmulPipeline pipeline;
    ASSERT_TRUE(ctx.create_matmul_pipeline(WeightDType::Int8, &pipeline));
    BufferHandle weight, activation, out;
    ASSERT_TRUE(ctx.allocate_buffer(16, true, &weight));
    ASSERT_TRUE(ctx.allocate_buffer(16, true, &activation));
    ASSERT_TRUE(ctx.allocate_buffer(sizeof(float), true, &out));
    // scale == default-constructed (buffer == nullptr) for a quantized
    // dtype must fail, not crash.
    EXPECT_FALSE(ctx.matmul_dequant(pipeline, weight, BufferHandle{}, activation, out, 1, 4));
    ctx.free_buffer(&weight);
    ctx.free_buffer(&activation);
    ctx.free_buffer(&out);
    ctx.destroy_matmul_pipeline(&pipeline);
}
