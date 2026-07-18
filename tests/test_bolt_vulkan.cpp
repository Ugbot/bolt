// test_bolt_vulkan.cpp -- coverage for bolt::vulkan's device/queue/memory
// scaffolding (BLLM-46). Runs against whatever REAL Vulkan device this
// machine has (this box: AMD Radeon 8060S integrated GPU) -- self-skips
// (not a hard failure) if no Vulkan-capable device is present, matching
// this module's own "runtime-detected, CPU fallback always works"
// contract. STL is allowed in tests; production code stays Tiger-Style.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
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

// BLLM-83: on this box's real integrated/UMA GPU (AMD Radeon 8060S), a
// direct vkGetPhysicalDeviceMemoryProperties dump (2026-07-17) confirmed 4
// of 16 memory types are simultaneously HOST_VISIBLE|HOST_COHERENT|
// DEVICE_LOCAL, backed by a genuinely separate ~68GB DEVICE_LOCAL heap
// distinct from a ~34GB host-only heap -- but the PRE-FIX find_memory_type
// order would have silently landed a prefer_host_visible=true allocation
// on the slower non-device-local heap purely because it sorted first by
// type index, with no error to reveal it. This test locks in the fix:
// a host-visible allocation on this hardware must ALSO be device_local.
TEST(BoltVulkan, PreferHostVisibleLandsOnDeviceLocalHeapOnUma) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());

    BufferHandle handle;
    ASSERT_TRUE(ctx.allocate_buffer(4096, /*prefer_host_visible=*/true, &handle));
    EXPECT_TRUE(handle.host_visible);
    // NOTE: this assertion is hardware-fact-dependent (true on this box's
    // real UMA GPU as of 2026-07-17) rather than a universal Vulkan
    // guarantee -- a genuinely discrete GPU with no coincident heap would
    // legitimately fail this and that would be correct, not a regression.
    EXPECT_TRUE(handle.device_local)
        << "expected a coincident HOST_VISIBLE+DEVICE_LOCAL heap on this UMA GPU -- "
           "if this fails on different hardware, the memory-type selection preference "
           "is still correct, this specific expectation just doesn't hold there";

    ctx.free_buffer(&handle);
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

// BLLM-176: on-device RMSNorm matches the CPU reference (f64-accumulated
// sum-of-squares -> inv-rms -> scale). n=300 spans multiple subgroups, so the
// shader's cross-subgroup reduction combine is exercised.
TEST(BoltVulkan, RmsnormMatchesCpuReference) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    MatmulPipeline pipeline;
    ASSERT_TRUE(ctx.create_rmsnorm_pipeline(&pipeline));

    const int n = 300;
    std::vector<float> x(static_cast<size_t>(n)), w(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        x[static_cast<size_t>(i)] = static_cast<float>((i % 7) - 3) * 0.5f;
        w[static_cast<size_t>(i)] = 1.0f + static_cast<float>(i % 3) * 0.25f;
    }
    const float eps = 1e-5f;

    BufferHandle xb = make_and_fill(ctx, x.data(), x.size() * sizeof(float));
    BufferHandle wb = make_and_fill(ctx, w.data(), w.size() * sizeof(float));
    BufferHandle yb;
    ASSERT_TRUE(ctx.allocate_buffer(x.size() * sizeof(float), true, &yb));
    ASSERT_NE(xb.buffer, nullptr);
    ASSERT_NE(wb.buffer, nullptr);

    ASSERT_TRUE(ctx.rmsnorm(pipeline, xb, wb, yb, n, eps));

    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += static_cast<double>(x[static_cast<size_t>(i)]) *
                                       static_cast<double>(x[static_cast<size_t>(i)]);
    const float inv = static_cast<float>(1.0 / std::sqrt(ss / n + eps));

    float* y = static_cast<float*>(ctx.map_buffer(yb));
    ASSERT_NE(y, nullptr);
    for (int i = 0; i < n; ++i) {
        const float want = x[static_cast<size_t>(i)] * inv * w[static_cast<size_t>(i)];
        EXPECT_NEAR(y[i], want, 1e-4f) << "i=" << i;
    }
    ctx.unmap_buffer(yb);
    ctx.free_buffer(&xb);
    ctx.free_buffer(&wb);
    ctx.free_buffer(&yb);
    ctx.destroy_matmul_pipeline(&pipeline);
}

// BLLM-179: on-device SwiGLU (op 1) and add (op 0) match CPU references.
TEST(BoltVulkan, ElementwiseMatchesCpuReference) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    MatmulPipeline pipeline;
    ASSERT_TRUE(ctx.create_elementwise_pipeline(&pipeline));

    const int n = 500;  // not a multiple of 256 -> exercises the OOB guard
    std::vector<float> a(static_cast<size_t>(n)), b(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        a[static_cast<size_t>(i)] = static_cast<float>((i % 11) - 5) * 0.3f;
        b[static_cast<size_t>(i)] = static_cast<float>((i % 5) - 2) * 0.7f;
    }
    BufferHandle ab = make_and_fill(ctx, a.data(), a.size() * sizeof(float));
    BufferHandle bb = make_and_fill(ctx, b.data(), b.size() * sizeof(float));
    BufferHandle ob;
    ASSERT_TRUE(ctx.allocate_buffer(a.size() * sizeof(float), true, &ob));

    // op 1: SwiGLU -> silu(a)*b.
    ASSERT_TRUE(ctx.elementwise(pipeline, ab, bb, ob, n, /*op=*/1));
    float* o = static_cast<float*>(ctx.map_buffer(ob));
    ASSERT_NE(o, nullptr);
    for (int i = 0; i < n; ++i) {
        const float x = a[static_cast<size_t>(i)];
        const float want = (x / (1.0f + std::exp(-x))) * b[static_cast<size_t>(i)];
        EXPECT_NEAR(o[i], want, 1e-4f) << "swiglu i=" << i;
    }
    ctx.unmap_buffer(ob);

    // op 0: add -> a+b.
    ASSERT_TRUE(ctx.elementwise(pipeline, ab, bb, ob, n, /*op=*/0));
    o = static_cast<float*>(ctx.map_buffer(ob));
    ASSERT_NE(o, nullptr);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(o[i], a[static_cast<size_t>(i)] + b[static_cast<size_t>(i)], 1e-5f)
            << "add i=" << i;
    }
    ctx.unmap_buffer(ob);

    ctx.free_buffer(&ab);
    ctx.free_buffer(&bb);
    ctx.free_buffer(&ob);
    ctx.destroy_matmul_pipeline(&pipeline);
}

// BLLM-177: on-device RoPE matches the CPU rope_half_split reference per head.
TEST(BoltVulkan, RopeMatchesCpuReference) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    MatmulPipeline pipeline;
    ASSERT_TRUE(ctx.create_rope_pipeline(&pipeline));

    const int head_dim = 64, num_heads = 4, position = 7;
    const float theta = 10000.0f;
    const int n = head_dim * num_heads;
    std::vector<float> x(static_cast<size_t>(n)), ref(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        x[static_cast<size_t>(i)] = static_cast<float>((i % 13) - 6) * 0.2f;
        ref[static_cast<size_t>(i)] = x[static_cast<size_t>(i)];
    }
    // CPU rope_half_split per head.
    const int half = head_dim / 2;
    for (int h = 0; h < num_heads; ++h) {
        float* v = ref.data() + static_cast<size_t>(h) * head_dim;
        for (int j = 0; j < half; ++j) {
            const float inv = std::pow(theta, -2.0f * static_cast<float>(j) / head_dim);
            const float ang = static_cast<float>(position) * inv;
            const float cs = std::cos(ang), sn = std::sin(ang);
            const float a = v[j], b = v[j + half];
            v[j] = a * cs - b * sn;
            v[j + half] = b * cs + a * sn;
        }
    }

    BufferHandle xb = make_and_fill(ctx, x.data(), x.size() * sizeof(float));
    ASSERT_TRUE(ctx.rope(pipeline, xb, head_dim, num_heads, position, theta));

    float* g = static_cast<float*>(ctx.map_buffer(xb));
    ASSERT_NE(g, nullptr);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(g[i], ref[static_cast<size_t>(i)], 1e-4f) << "i=" << i;
    }
    ctx.unmap_buffer(xb);
    ctx.free_buffer(&xb);
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

// ---------------------------------------------------------------------------
// BLLM-87: batched int8-activation matmul -- validated against the ALREADY-
// verified single-token int8-activation kernel (BLLM-81's own test above),
// rather than a fresh CPU reference: for a batch of N tokens, the batched
// kernel's output for token t must match the single-token kernel called
// individually on token t's own activation, since both compute the exact
// same int32-accumulation dot product against the exact same resident
// weight -- only the dispatch shape differs.
// ---------------------------------------------------------------------------

namespace {

// Quantizes `raw[rows][cols]` (row-major) to Int8 with a simple per-row
// max-abs scale -- a self-contained test helper (bolt has no project
// dependency on boltllm::quant's own quantize_rows_int8, and shouldn't
// gain one just for a test). Exact rounding convention doesn't matter here:
// this test only compares the batched kernel against the single-token
// kernel using the SAME quantized bytes, not against an external oracle.
void quantize_int8_rows(const std::vector<float>& raw, int32_t rows, int32_t cols,
                         std::vector<int8_t>& q_out, std::vector<float>& scale_out) {
    q_out.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    scale_out.resize(static_cast<size_t>(rows));
    for (int32_t r = 0; r < rows; ++r) {
        const float* row = raw.data() + static_cast<size_t>(r) * cols;
        float max_abs = 1e-8f;
        for (int32_t c = 0; c < cols; ++c) max_abs = std::max(max_abs, std::fabs(row[c]));
        const float scale = max_abs / 127.0f;
        scale_out[static_cast<size_t>(r)] = scale;
        int8_t* qrow = q_out.data() + static_cast<size_t>(r) * cols;
        for (int32_t c = 0; c < cols; ++c) {
            int v = static_cast<int>(row[c] / scale + (row[c] >= 0.0f ? 0.5f : -0.5f));
            v = std::max(-127, std::min(127, v));
            qrow[c] = static_cast<int8_t>(v);
        }
    }
}

void run_batched_int8act_case(WeightDType dtype, int32_t rows, int32_t cols, int32_t n_tokens) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    // Weight: [rows, cols], quantized per-row.
    std::vector<float> raw_weight(static_cast<size_t>(rows) * cols);
    for (auto& v : raw_weight) v = dist(rng);
    std::vector<int8_t> weight_q8;
    std::vector<float> weight_scale;
    quantize_int8_rows(raw_weight, rows, cols, weight_q8, weight_scale);

    std::vector<uint8_t> weight_packed;
    if (dtype == WeightDType::Int8) {
        weight_packed.assign(reinterpret_cast<uint8_t*>(weight_q8.data()),
                              reinterpret_cast<uint8_t*>(weight_q8.data()) + weight_q8.size());
    } else {
        // Int4: pack two int8-range-clamped-to-[-8,7] values per byte,
        // low-nibble-first, offset 8 -- matches matmul_dequant_int8act_int4's
        // own convention. Requantize from the SAME raw weight at 4-bit
        // range so weight_scale still reflects the actual clamp range used.
        weight_q8.clear();
        weight_scale.clear();
        weight_q8.resize(static_cast<size_t>(rows) * cols);
        weight_scale.resize(static_cast<size_t>(rows));
        weight_packed.assign(static_cast<size_t>(rows) * ((cols + 1) / 2), 0);
        for (int32_t r = 0; r < rows; ++r) {
            const float* row = raw_weight.data() + static_cast<size_t>(r) * cols;
            float max_abs = 1e-8f;
            for (int32_t c = 0; c < cols; ++c) max_abs = std::max(max_abs, std::fabs(row[c]));
            const float scale = max_abs / 7.0f;
            weight_scale[static_cast<size_t>(r)] = scale;
            for (int32_t c = 0; c < cols; ++c) {
                int v = static_cast<int>(row[c] / scale + (row[c] >= 0.0f ? 0.5f : -0.5f));
                v = std::max(-8, std::min(7, v)) + 8;  // offset-8 nibble range [0,15]
                const size_t byte_idx = static_cast<size_t>(r) * ((cols + 1) / 2) + static_cast<size_t>(c) / 2;
                if (c % 2 == 0) {
                    weight_packed[byte_idx] = static_cast<uint8_t>(v);
                } else {
                    weight_packed[byte_idx] |= static_cast<uint8_t>(v << 4);
                }
            }
        }
    }

    // n_tokens activations, each [cols], quantized independently (own scale).
    std::vector<std::vector<int8_t>> act_q(static_cast<size_t>(n_tokens));
    std::vector<float> act_scale(static_cast<size_t>(n_tokens));
    for (int32_t t = 0; t < n_tokens; ++t) {
        std::vector<float> raw_act(static_cast<size_t>(cols));
        for (auto& v : raw_act) v = dist(rng);
        std::vector<int8_t> q;
        std::vector<float> s;
        quantize_int8_rows(raw_act, 1, cols, q, s);
        act_q[static_cast<size_t>(t)] = std::move(q);
        act_scale[static_cast<size_t>(t)] = s[0];
    }

    Context ctx;
    ASSERT_TRUE(ctx.create());

    // ---- Reference: single-token kernel, called once per token.
    MatmulPipeline single_pipeline;
    ASSERT_TRUE(ctx.create_int8_activation_pipeline(dtype, &single_pipeline));
    BufferHandle weight_buf = make_and_fill(ctx, weight_packed.data(), weight_packed.size());
    BufferHandle scale_buf = make_and_fill(ctx, weight_scale.data(), weight_scale.size() * sizeof(float));

    std::vector<float> expected(static_cast<size_t>(n_tokens) * rows);
    for (int32_t t = 0; t < n_tokens; ++t) {
        BufferHandle act_buf =
            make_and_fill(ctx, act_q[static_cast<size_t>(t)].data(), static_cast<size_t>(cols));
        BufferHandle out_buf;
        ASSERT_TRUE(ctx.allocate_buffer(static_cast<uint64_t>(rows) * sizeof(float), true, &out_buf));
        ASSERT_TRUE(ctx.matmul_dequant_int8_activation(single_pipeline, weight_buf, scale_buf, act_buf,
                                                        act_scale[static_cast<size_t>(t)], out_buf, rows,
                                                        cols));
        float* out_ptr = static_cast<float*>(ctx.map_buffer(out_buf));
        ASSERT_NE(out_ptr, nullptr);
        std::memcpy(expected.data() + static_cast<size_t>(t) * rows, out_ptr, static_cast<size_t>(rows) * sizeof(float));
        ctx.unmap_buffer(out_buf);
        ctx.free_buffer(&act_buf);
        ctx.free_buffer(&out_buf);
    }
    ctx.destroy_matmul_pipeline(&single_pipeline);

    // ---- Batched kernel: ONE dispatch for all n_tokens.
    MatmulPipeline batched_pipeline;
    ASSERT_TRUE(ctx.create_int8_activation_batched_pipeline(dtype, &batched_pipeline));

    std::vector<int8_t> act_q_flat(static_cast<size_t>(n_tokens) * cols);
    for (int32_t t = 0; t < n_tokens; ++t) {
        std::memcpy(act_q_flat.data() + static_cast<size_t>(t) * cols, act_q[static_cast<size_t>(t)].data(),
                    static_cast<size_t>(cols));
    }
    BufferHandle act_flat_buf = make_and_fill(ctx, act_q_flat.data(), act_q_flat.size());
    BufferHandle act_scale_buf = make_and_fill(ctx, act_scale.data(), act_scale.size() * sizeof(float));
    BufferHandle out_flat_buf;
    ASSERT_TRUE(ctx.allocate_buffer(static_cast<uint64_t>(n_tokens) * rows * sizeof(float), true,
                                    &out_flat_buf));

    ASSERT_TRUE(ctx.matmul_dequant_int8_activation_batched(batched_pipeline, weight_buf, scale_buf,
                                                            act_flat_buf, act_scale_buf, out_flat_buf,
                                                            rows, cols, n_tokens));

    float* out_flat_ptr = static_cast<float*>(ctx.map_buffer(out_flat_buf));
    ASSERT_NE(out_flat_ptr, nullptr);
    for (int32_t t = 0; t < n_tokens; ++t) {
        for (int32_t r = 0; r < rows; ++r) {
            const size_t idx = static_cast<size_t>(t) * rows + r;
            const float diff = std::fabs(out_flat_ptr[idx] - expected[idx]);
            const float tol = 1e-3f * (std::fabs(expected[idx]) + 1.0f);
            EXPECT_LE(diff, tol) << "token=" << t << " row=" << r << " got=" << out_flat_ptr[idx]
                                  << " expected=" << expected[idx];
        }
    }
    ctx.unmap_buffer(out_flat_buf);

    ctx.free_buffer(&weight_buf);
    ctx.free_buffer(&scale_buf);
    ctx.free_buffer(&act_flat_buf);
    ctx.free_buffer(&act_scale_buf);
    ctx.free_buffer(&out_flat_buf);
    ctx.destroy_int8_activation_batched_pipeline(&batched_pipeline);
}

}  // namespace

TEST(BoltVulkan, BatchedInt8ActivationMatchesSingleTokenKernelInt8Weight) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    run_batched_int8act_case(WeightDType::Int8, /*rows=*/16, /*cols=*/128, /*n_tokens=*/5);
    run_batched_int8act_case(WeightDType::Int8, /*rows=*/384, /*cols=*/1024, /*n_tokens=*/12);
}

TEST(BoltVulkan, BatchedInt8ActivationMatchesSingleTokenKernelInt4Weight) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    run_batched_int8act_case(WeightDType::Int4, /*rows=*/16, /*cols=*/130, /*n_tokens=*/5);
    run_batched_int8act_case(WeightDType::Int4, /*rows=*/384, /*cols=*/1024, /*n_tokens=*/12);
}

TEST(BoltVulkan, BatchedInt8ActivationSingleTokenBatchWorks) {
    // n_tokens=1 is a valid, degenerate batch -- must behave identically to
    // the single-token kernel path.
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    run_batched_int8act_case(WeightDType::Int8, /*rows=*/8, /*cols=*/64, /*n_tokens=*/1);
}
