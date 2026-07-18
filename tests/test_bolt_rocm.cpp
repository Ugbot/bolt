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

// BLLM-184: on-device HIP RMSNorm matches the CPU reference.
TEST(BoltRocm, RmsnormMatchesCpuReference) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    const int n = 300;
    std::vector<float> x(static_cast<size_t>(n)), w(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        x[static_cast<size_t>(i)] = static_cast<float>((i % 7) - 3) * 0.5f;
        w[static_cast<size_t>(i)] = 1.0f + static_cast<float>(i % 3) * 0.25f;
    }
    const float eps = 1e-5f;
    DeviceBuffer xb, wb, yb;
    ASSERT_TRUE(ctx.allocate(x.size() * sizeof(float), &xb));
    ASSERT_TRUE(ctx.allocate(w.size() * sizeof(float), &wb));
    ASSERT_TRUE(ctx.allocate(x.size() * sizeof(float), &yb));
    ASSERT_TRUE(ctx.copy_to_device(x.data(), &xb, x.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(w.data(), &wb, w.size() * sizeof(float)));
    ASSERT_TRUE(bolt::rocm::rmsnorm(ctx, static_cast<const float*>(xb.ptr),
                                    static_cast<const float*>(wb.ptr),
                                    static_cast<float*>(yb.ptr), n, eps));
    std::vector<float> y(static_cast<size_t>(n));
    ASSERT_TRUE(ctx.copy_to_host(yb, y.data(), y.size() * sizeof(float)));
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += static_cast<double>(x[static_cast<size_t>(i)]) *
                                       static_cast<double>(x[static_cast<size_t>(i)]);
    const float inv = static_cast<float>(1.0 / std::sqrt(ss / n + eps));
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(y[static_cast<size_t>(i)],
                    x[static_cast<size_t>(i)] * inv * w[static_cast<size_t>(i)], 1e-4f)
            << "i=" << i;
    }
    ctx.free(&xb);
    ctx.free(&wb);
    ctx.free(&yb);
}

// BLLM-185: on-device HIP RoPE matches rope_half_split per head.
TEST(BoltRocm, RopeMatchesCpuReference) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    const int head_dim = 64, num_heads = 4, position = 7;
    const float theta = 10000.0f;
    const int n = head_dim * num_heads, half = head_dim / 2;
    std::vector<float> x(static_cast<size_t>(n)), ref(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        x[static_cast<size_t>(i)] = static_cast<float>((i % 13) - 6) * 0.2f;
        ref[static_cast<size_t>(i)] = x[static_cast<size_t>(i)];
    }
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
    DeviceBuffer xb;
    ASSERT_TRUE(ctx.allocate(x.size() * sizeof(float), &xb));
    ASSERT_TRUE(ctx.copy_to_device(x.data(), &xb, x.size() * sizeof(float)));
    ASSERT_TRUE(bolt::rocm::rope(ctx, static_cast<float*>(xb.ptr), head_dim, num_heads, position,
                                 theta));
    std::vector<float> g(static_cast<size_t>(n));
    ASSERT_TRUE(ctx.copy_to_host(xb, g.data(), g.size() * sizeof(float)));
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(g[static_cast<size_t>(i)], ref[static_cast<size_t>(i)], 1e-4f) << "i=" << i;
    }
    ctx.free(&xb);
}

// BLLM-186: on-device HIP elementwise (SwiGLU op1 + add op0) matches CPU.
TEST(BoltRocm, ElementwiseMatchesCpuReference) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    const int n = 500;
    std::vector<float> a(static_cast<size_t>(n)), b(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        a[static_cast<size_t>(i)] = static_cast<float>((i % 11) - 5) * 0.3f;
        b[static_cast<size_t>(i)] = static_cast<float>((i % 5) - 2) * 0.7f;
    }
    DeviceBuffer ab, bb, ob;
    ASSERT_TRUE(ctx.allocate(a.size() * sizeof(float), &ab));
    ASSERT_TRUE(ctx.allocate(b.size() * sizeof(float), &bb));
    ASSERT_TRUE(ctx.allocate(a.size() * sizeof(float), &ob));
    ASSERT_TRUE(ctx.copy_to_device(a.data(), &ab, a.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(b.data(), &bb, b.size() * sizeof(float)));
    std::vector<float> o(static_cast<size_t>(n));

    ASSERT_TRUE(bolt::rocm::elementwise(ctx, static_cast<const float*>(ab.ptr),
                                        static_cast<const float*>(bb.ptr),
                                        static_cast<float*>(ob.ptr), n, /*op=*/1));
    ASSERT_TRUE(ctx.copy_to_host(ob, o.data(), o.size() * sizeof(float)));
    for (int i = 0; i < n; ++i) {
        const float x = a[static_cast<size_t>(i)];
        EXPECT_NEAR(o[static_cast<size_t>(i)], (x / (1.0f + std::exp(-x))) * b[static_cast<size_t>(i)],
                    1e-4f)
            << "swiglu i=" << i;
    }

    ASSERT_TRUE(bolt::rocm::elementwise(ctx, static_cast<const float*>(ab.ptr),
                                        static_cast<const float*>(bb.ptr),
                                        static_cast<float*>(ob.ptr), n, /*op=*/0));
    ASSERT_TRUE(ctx.copy_to_host(ob, o.data(), o.size() * sizeof(float)));
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(o[static_cast<size_t>(i)],
                    a[static_cast<size_t>(i)] + b[static_cast<size_t>(i)], 1e-5f)
            << "add i=" << i;
    }
    ctx.free(&ab);
    ctx.free(&bb);
    ctx.free(&ob);
}

// BLLM-187: on-device HIP decode-step attention (GQA) matches a two-pass CPU
// softmax reference (mathematically identical to the online-softmax kernel).
TEST(BoltRocm, AttentionMatchesCpuReference) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    const int num_heads = 4, num_kv_heads = 2, head_dim = 8, seq_len = 5;
    const int group_size = num_heads / num_kv_heads;
    const int q_n = num_heads * head_dim;
    const int kv_n = seq_len * num_kv_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> q(static_cast<size_t>(q_n)), kc(static_cast<size_t>(kv_n)),
        vc(static_cast<size_t>(kv_n));
    for (int i = 0; i < q_n; ++i) q[static_cast<size_t>(i)] = static_cast<float>((i % 9) - 4) * 0.15f;
    for (int i = 0; i < kv_n; ++i) {
        kc[static_cast<size_t>(i)] = static_cast<float>((i % 7) - 3) * 0.12f;
        vc[static_cast<size_t>(i)] = static_cast<float>((i % 5) - 2) * 0.2f;
    }

    // CPU reference: per head, two-pass softmax over cached KV.
    std::vector<float> ref(static_cast<size_t>(q_n), 0.0f);
    for (int h = 0; h < num_heads; ++h) {
        const int kv_head = h / group_size;
        const float* qh = q.data() + static_cast<size_t>(h) * head_dim;
        std::vector<float> sc(static_cast<size_t>(seq_len));
        float mx = -1e30f;
        for (int t = 0; t < seq_len; ++t) {
            const float* kh =
                kc.data() + (static_cast<size_t>(t) * num_kv_heads + kv_head) * head_dim;
            float s = 0.0f;
            for (int d = 0; d < head_dim; ++d) s += qh[d] * kh[d];
            s *= scale;
            sc[static_cast<size_t>(t)] = s;
            mx = std::fmax(mx, s);
        }
        double denom = 0.0;
        for (int t = 0; t < seq_len; ++t) {
            sc[static_cast<size_t>(t)] = std::exp(sc[static_cast<size_t>(t)] - mx);
            denom += sc[static_cast<size_t>(t)];
        }
        float* rh = ref.data() + static_cast<size_t>(h) * head_dim;
        for (int t = 0; t < seq_len; ++t) {
            const float w = static_cast<float>(sc[static_cast<size_t>(t)] / denom);
            const float* vh =
                vc.data() + (static_cast<size_t>(t) * num_kv_heads + kv_head) * head_dim;
            for (int d = 0; d < head_dim; ++d) rh[d] += w * vh[d];
        }
    }

    DeviceBuffer qb, kb, vb, ob;
    ASSERT_TRUE(ctx.allocate(q.size() * sizeof(float), &qb));
    ASSERT_TRUE(ctx.allocate(kc.size() * sizeof(float), &kb));
    ASSERT_TRUE(ctx.allocate(vc.size() * sizeof(float), &vb));
    ASSERT_TRUE(ctx.allocate(q.size() * sizeof(float), &ob));
    ASSERT_TRUE(ctx.copy_to_device(q.data(), &qb, q.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(kc.data(), &kb, kc.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(vc.data(), &vb, vc.size() * sizeof(float)));
    ASSERT_TRUE(bolt::rocm::attention(ctx, static_cast<const float*>(qb.ptr),
                                      static_cast<const float*>(kb.ptr),
                                      static_cast<const float*>(vb.ptr),
                                      static_cast<float*>(ob.ptr), num_heads, num_kv_heads, head_dim,
                                      seq_len, scale));
    std::vector<float> o(static_cast<size_t>(q_n));
    ASSERT_TRUE(ctx.copy_to_host(ob, o.data(), o.size() * sizeof(float)));
    for (int i = 0; i < q_n; ++i) {
        EXPECT_NEAR(o[static_cast<size_t>(i)], ref[static_cast<size_t>(i)], 1e-4f) << "i=" << i;
    }
    ctx.free(&qb);
    ctx.free(&kb);
    ctx.free(&vb);
    ctx.free(&ob);
}
