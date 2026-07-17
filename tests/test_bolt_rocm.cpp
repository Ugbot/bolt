// test_bolt_rocm.cpp -- coverage for bolt::rocm's matmul+dequant kernel
// core (BLLM-78). Only built when BOLT_BUILD_ROCM is ON (a real hipcc was
// found) -- this dev box has no ROCm/HIP SDK for native Windows, so this
// file is not part of the build here (see bolt_rocm.h's status note).
// Self-skips (not a hard failure) if bolt::rocm::available() is false,
// matching test_bolt_vulkan.cpp's own contract. STL is allowed in tests;
// production code stays Tiger-Style.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "bolt/bolt_rocm.h"

using bolt::rocm::Context;
using bolt::rocm::DeviceBuffer;
using bolt::rocm::WeightDType;

TEST(BoltRocm, AvailableDoesNotCrashRegardlessOfHardware) {
    const bool result = bolt::rocm::available();
    EXPECT_TRUE(result == true || result == false);  // trivially true; documents intent
}

TEST(BoltRocm, ContextCreateSucceedsOnThisRealDevice) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    EXPECT_TRUE(ctx.is_valid());
    EXPECT_GT(std::strlen(ctx.device_name()), 0u);
    EXPECT_GT(ctx.device_local_heap_bytes(), 0u);
    ctx.destroy();
    EXPECT_FALSE(ctx.is_valid());
}

TEST(BoltRocm, DestroyIsSafeOnNeverCreatedContext) {
    Context ctx;
    ctx.destroy();  // must not crash
    EXPECT_FALSE(ctx.is_valid());
}

TEST(BoltRocm, AllocateFreeAndRoundTripCopy) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());

    DeviceBuffer buf;
    ASSERT_TRUE(ctx.allocate(4 * sizeof(float), &buf));
    EXPECT_EQ(buf.size_bytes, 4u * sizeof(float));

    const float host_in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_TRUE(ctx.copy_to_device(host_in, &buf, sizeof(host_in)));

    float host_out[4] = {};
    ASSERT_TRUE(ctx.copy_to_host(buf, host_out, sizeof(host_out)));
    for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(host_out[i], host_in[i]);

    ctx.free(&buf);
    EXPECT_EQ(buf.ptr, nullptr);
}

TEST(BoltRocm, MatmulDequantF32MatchesHandComputedDotProduct) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());

    // 2 output rows x 3 cols. Row 0: [1,2,3], row 1: [4,5,6]. Activation:
    // [1,1,1] -> out = [6, 15] (plain sums).
    const std::vector<float> weight = {1, 2, 3, 4, 5, 6};
    const float activation[3] = {1.0f, 1.0f, 1.0f};

    DeviceBuffer weight_buf, act_buf, out_buf;
    ASSERT_TRUE(ctx.allocate(weight.size() * sizeof(float), &weight_buf));
    ASSERT_TRUE(ctx.allocate(sizeof(activation), &act_buf));
    ASSERT_TRUE(ctx.allocate(2 * sizeof(float), &out_buf));

    ASSERT_TRUE(ctx.copy_to_device(weight.data(), &weight_buf, weight.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(activation, &act_buf, sizeof(activation)));

    ASSERT_TRUE(bolt::rocm::matmul_dequant(ctx, weight_buf.ptr, WeightDType::F32, nullptr,
                                            static_cast<const float*>(act_buf.ptr),
                                            static_cast<float*>(out_buf.ptr),
                                            /*out_rows=*/2, /*cols=*/3));

    float out[2] = {};
    ASSERT_TRUE(ctx.copy_to_host(out_buf, out, sizeof(out)));
    EXPECT_FLOAT_EQ(out[0], 6.0f);
    EXPECT_FLOAT_EQ(out[1], 15.0f);

    ctx.free(&weight_buf);
    ctx.free(&act_buf);
    ctx.free(&out_buf);
}

TEST(BoltRocm, MatmulDequantInt8MatchesScaledDotProduct) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());

    // 1 row x 4 cols, int8 weights all 10, scale 0.5, activation
    // [1, 2, 1, 0.5] -> raw dot = 10*(1+2+1+0.5) = 45 -> scaled = 22.5.
    const std::vector<int8_t> weight = {10, 10, 10, 10};
    const float scale = 0.5f;
    const float activation[4] = {1.0f, 2.0f, 1.0f, 0.5f};
    const float expected_raw = 10.0f * (1.0f + 2.0f + 1.0f + 0.5f);
    const float expected = expected_raw * scale;

    DeviceBuffer weight_buf, scale_buf, act_buf, out_buf;
    ASSERT_TRUE(ctx.allocate(weight.size() * sizeof(int8_t), &weight_buf));
    ASSERT_TRUE(ctx.allocate(sizeof(float), &scale_buf));
    ASSERT_TRUE(ctx.allocate(sizeof(activation), &act_buf));
    ASSERT_TRUE(ctx.allocate(sizeof(float), &out_buf));

    ASSERT_TRUE(ctx.copy_to_device(weight.data(), &weight_buf, weight.size() * sizeof(int8_t)));
    ASSERT_TRUE(ctx.copy_to_device(&scale, &scale_buf, sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(activation, &act_buf, sizeof(activation)));

    ASSERT_TRUE(bolt::rocm::matmul_dequant(ctx, weight_buf.ptr, WeightDType::Int8,
                                            static_cast<const float*>(scale_buf.ptr),
                                            static_cast<const float*>(act_buf.ptr),
                                            static_cast<float*>(out_buf.ptr),
                                            /*out_rows=*/1, /*cols=*/4));

    float out = 0.0f;
    ASSERT_TRUE(ctx.copy_to_host(out_buf, &out, sizeof(out)));
    EXPECT_NEAR(out, expected, 1e-4f);

    ctx.free(&weight_buf);
    ctx.free(&scale_buf);
    ctx.free(&act_buf);
    ctx.free(&out_buf);
}

TEST(BoltRocm, MatmulDequantFailsCleanlyOnInvalidContext) {
    Context ctx;  // create() never called
    const float activation[2] = {1.0f, 1.0f};
    float out[1] = {0};
    const float weight[2] = {1.0f, 1.0f};
    EXPECT_FALSE(bolt::rocm::matmul_dequant(ctx, weight, WeightDType::F32, nullptr, activation, out,
                                             1, 2));
}

TEST(BoltRocm, MatmulDequantRequiresScaleForQuantizedDtypes) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    const float activation[2] = {1.0f, 1.0f};
    float out[1] = {0};
    const int8_t weight[2] = {1, 1};
    // scale_device == nullptr for a quantized dtype must fail, not crash.
    EXPECT_FALSE(bolt::rocm::matmul_dequant(ctx, weight, WeightDType::Int8, nullptr, activation, out,
                                             1, 2));
}
