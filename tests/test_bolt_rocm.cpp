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

// BLLM-200: attention with gpt-oss sinks + sliding-window start. CPU reference
// restricts the softmax to keys [t0, seq_len) and folds a per-head sink logit into
// the denominator only (no value contribution).
TEST(BoltRocm, AttentionSinksAndWindowMatchesCpuReference) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    const int num_heads = 4, num_kv_heads = 2, head_dim = 8, seq_len = 6;
    const int group_size = num_heads / num_kv_heads;
    const int q_n = num_heads * head_dim;
    const int kv_n = seq_len * num_kv_heads * head_dim;
    const int t0 = 2;  // sliding-window: attend keys [2, 6)
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> q(static_cast<size_t>(q_n)), kc(static_cast<size_t>(kv_n)),
        vc(static_cast<size_t>(kv_n)), sinks(static_cast<size_t>(num_heads));
    for (int i = 0; i < q_n; ++i) q[static_cast<size_t>(i)] = static_cast<float>((i % 9) - 4) * 0.15f;
    for (int i = 0; i < kv_n; ++i) {
        kc[static_cast<size_t>(i)] = static_cast<float>((i % 7) - 3) * 0.12f;
        vc[static_cast<size_t>(i)] = static_cast<float>((i % 5) - 2) * 0.2f;
    }
    for (int h = 0; h < num_heads; ++h) sinks[static_cast<size_t>(h)] = 0.1f + 0.3f * h;

    std::vector<float> ref(static_cast<size_t>(q_n), 0.0f);
    for (int h = 0; h < num_heads; ++h) {
        const int kv_head = h / group_size;
        const float* qh = q.data() + static_cast<size_t>(h) * head_dim;
        std::vector<float> sc(static_cast<size_t>(seq_len));
        float mx = sinks[static_cast<size_t>(h)];  // sink participates in the max
        for (int t = t0; t < seq_len; ++t) {
            const float* kh = kc.data() + (static_cast<size_t>(t) * num_kv_heads + kv_head) * head_dim;
            float s = 0.0f;
            for (int d = 0; d < head_dim; ++d) s += qh[d] * kh[d];
            s *= scale;
            sc[static_cast<size_t>(t)] = s;
            mx = std::fmax(mx, s);
        }
        double denom = std::exp(sinks[static_cast<size_t>(h)] - mx);  // sink -> denom only
        for (int t = t0; t < seq_len; ++t) {
            sc[static_cast<size_t>(t)] = std::exp(sc[static_cast<size_t>(t)] - mx);
            denom += sc[static_cast<size_t>(t)];
        }
        float* rh = ref.data() + static_cast<size_t>(h) * head_dim;
        for (int t = t0; t < seq_len; ++t) {
            const float w = static_cast<float>(sc[static_cast<size_t>(t)] / denom);
            const float* vh = vc.data() + (static_cast<size_t>(t) * num_kv_heads + kv_head) * head_dim;
            for (int d = 0; d < head_dim; ++d) rh[d] += w * vh[d];
        }
    }

    DeviceBuffer qb, kb, vb, ob, sb;
    ASSERT_TRUE(ctx.allocate(q.size() * sizeof(float), &qb));
    ASSERT_TRUE(ctx.allocate(kc.size() * sizeof(float), &kb));
    ASSERT_TRUE(ctx.allocate(vc.size() * sizeof(float), &vb));
    ASSERT_TRUE(ctx.allocate(q.size() * sizeof(float), &ob));
    ASSERT_TRUE(ctx.allocate(sinks.size() * sizeof(float), &sb));
    ASSERT_TRUE(ctx.copy_to_device(q.data(), &qb, q.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(kc.data(), &kb, kc.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(vc.data(), &vb, vc.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(sinks.data(), &sb, sinks.size() * sizeof(float)));
    ASSERT_TRUE(bolt::rocm::attention(
        ctx, static_cast<const float*>(qb.ptr), static_cast<const float*>(kb.ptr),
        static_cast<const float*>(vb.ptr), static_cast<float*>(ob.ptr), num_heads, num_kv_heads,
        head_dim, seq_len, scale, static_cast<const float*>(sb.ptr), t0));
    std::vector<float> o(static_cast<size_t>(q_n));
    ASSERT_TRUE(ctx.copy_to_host(ob, o.data(), o.size() * sizeof(float)));
    for (int i = 0; i < q_n; ++i) {
        EXPECT_NEAR(o[static_cast<size_t>(i)], ref[static_cast<size_t>(i)], 1e-4f) << "i=" << i;
    }
    ctx.free(&qb);
    ctx.free(&kb);
    ctx.free(&vb);
    ctx.free(&ob);
    ctx.free(&sb);
}

// BLLM-200: gpt-oss expert activation (swiglu_oai) matches a CPU reference,
// including the asymmetric gate clamp, symmetric up clamp, alpha slope, +1 offset,
// and optional per-column biases.
TEST(BoltRocm, SwigluOaiMatchesCpuReference) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    const int n = 257;  // odd, to exercise the grid-stride tail
    const float alpha = 1.702f, limit = 7.0f;
    std::vector<float> gate(static_cast<size_t>(n)), up(static_cast<size_t>(n)),
        gb(static_cast<size_t>(n)), ub(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        gate[static_cast<size_t>(i)] = static_cast<float>((i % 23) - 11) * 0.9f;  // spans past +/-7
        up[static_cast<size_t>(i)] = static_cast<float>((i % 19) - 9) * 1.1f;
        gb[static_cast<size_t>(i)] = static_cast<float>((i % 5) - 2) * 0.05f;
        ub[static_cast<size_t>(i)] = static_cast<float>((i % 3) - 1) * 0.07f;
    }
    std::vector<float> ref(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        float g = gate[static_cast<size_t>(i)] + gb[static_cast<size_t>(i)];
        float u = up[static_cast<size_t>(i)] + ub[static_cast<size_t>(i)];
        g = std::fmin(g, limit);
        u = std::fmax(-limit, std::fmin(u, limit));
        const float glu = g / (1.0f + std::exp(-alpha * g));
        ref[static_cast<size_t>(i)] = glu * (u + 1.0f);
    }

    DeviceBuffer gd, ud, gbd, ubd, od;
    ASSERT_TRUE(ctx.allocate(gate.size() * sizeof(float), &gd));
    ASSERT_TRUE(ctx.allocate(up.size() * sizeof(float), &ud));
    ASSERT_TRUE(ctx.allocate(gb.size() * sizeof(float), &gbd));
    ASSERT_TRUE(ctx.allocate(ub.size() * sizeof(float), &ubd));
    ASSERT_TRUE(ctx.allocate(gate.size() * sizeof(float), &od));
    ASSERT_TRUE(ctx.copy_to_device(gate.data(), &gd, gate.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(up.data(), &ud, up.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(gb.data(), &gbd, gb.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(ub.data(), &ubd, ub.size() * sizeof(float)));
    ASSERT_TRUE(bolt::rocm::swiglu_oai(
        ctx, static_cast<const float*>(gd.ptr), static_cast<const float*>(ud.ptr),
        static_cast<const float*>(gbd.ptr), static_cast<const float*>(ubd.ptr),
        static_cast<float*>(od.ptr), n, alpha, limit));
    std::vector<float> o(static_cast<size_t>(n));
    ASSERT_TRUE(ctx.copy_to_host(od, o.data(), o.size() * sizeof(float)));
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(o[static_cast<size_t>(i)], ref[static_cast<size_t>(i)], 1e-4f) << "i=" << i;
    }
    ctx.free(&gd);
    ctx.free(&ud);
    ctx.free(&gbd);
    ctx.free(&ubd);
    ctx.free(&od);
}

// BLLM-200: NeoX + YaRN RoPE matches the CPU reference (the exact math boltllm's
// TransformerCore::rope_neox_yarn uses for gpt-oss).
TEST(BoltRocm, RopeNeoxYarnMatchesCpuReference) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    const int num_heads = 3, head_dim = 64, pos = 37;
    const float base = 150000.0f, freq_scale = 1.0f / 32.0f, ext_factor = 1.0f, attn_factor = 1.0f;
    const float beta_fast = 32.0f, beta_slow = 1.0f;
    const int orig_ctx = 4096;
    const int n = num_heads * head_dim;
    std::vector<float> x(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) x[static_cast<size_t>(i)] = static_cast<float>((i % 13) - 6) * 0.11f;

    // CPU reference (mirrors rope_neox_yarn + the kernel).
    std::vector<float> ref = x;
    const int half = head_dim / 2;
    const double twopi = 6.283185307179586;
    auto corr = [&](float b) {
        return static_cast<float>(head_dim * std::log(orig_ctx / (b * twopi)) / (2.0 * std::log(base)) *
                                  0.5);
    };
    float low = std::floor(corr(beta_fast)), high = std::ceil(corr(beta_slow));
    low = std::fmax(0.0f, low);
    high = std::fmin(static_cast<float>(half - 1), high);
    const float mscale = attn_factor * (1.0f + 0.1f * std::log(1.0f / freq_scale));
    for (int h = 0; h < num_heads; ++h) {
        float* xh = ref.data() + static_cast<size_t>(h) * head_dim;
        for (int j = 0; j < half; ++j) {
            const float freq = std::pow(base, -2.0f * j / static_cast<float>(head_dim));
            const float te = static_cast<float>(pos) * freq;
            const float ti = freq_scale * te;
            const float ramp = 1.0f - std::fmin(1.0f, std::fmax(0.0f, (j - low) / std::fmax(0.001f, high - low)));
            const float mix = ramp * ext_factor;
            const float theta = ti * (1.0f - mix) + te * mix;
            const float cs = std::cos(theta) * mscale, sn = std::sin(theta) * mscale;
            const float a = xh[j], b = xh[j + half];
            xh[j] = a * cs - b * sn;
            xh[j + half] = b * cs + a * sn;
        }
    }

    DeviceBuffer xb;
    ASSERT_TRUE(ctx.allocate(x.size() * sizeof(float), &xb));
    ASSERT_TRUE(ctx.copy_to_device(x.data(), &xb, x.size() * sizeof(float)));
    ASSERT_TRUE(bolt::rocm::rope_neox_yarn(ctx, static_cast<float*>(xb.ptr), head_dim, num_heads, pos,
                                           base, true, freq_scale, ext_factor, attn_factor, beta_fast,
                                           beta_slow, orig_ctx));
    std::vector<float> o(static_cast<size_t>(n));
    ASSERT_TRUE(ctx.copy_to_host(xb, o.data(), o.size() * sizeof(float)));
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(o[static_cast<size_t>(i)], ref[static_cast<size_t>(i)], 1e-4f) << "i=" << i;
    }
    ctx.free(&xb);
}

// BLLM-200: axpy_bias -- out[i] += scale*(x[i]+bias[i]).
TEST(BoltRocm, AxpyBiasMatchesCpuReference) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    const int n = 300;
    const float scale = 0.37f;
    std::vector<float> x(static_cast<size_t>(n)), bias(static_cast<size_t>(n)), out(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        x[static_cast<size_t>(i)] = static_cast<float>((i % 11) - 5) * 0.3f;
        bias[static_cast<size_t>(i)] = static_cast<float>((i % 7) - 3) * 0.2f;
        out[static_cast<size_t>(i)] = static_cast<float>((i % 5) - 2) * 0.5f;  // residual on entry
    }
    std::vector<float> ref(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        ref[static_cast<size_t>(i)] = out[static_cast<size_t>(i)] + scale * (x[static_cast<size_t>(i)] + bias[static_cast<size_t>(i)]);

    DeviceBuffer xb, bb, ob;
    ASSERT_TRUE(ctx.allocate(x.size() * sizeof(float), &xb));
    ASSERT_TRUE(ctx.allocate(bias.size() * sizeof(float), &bb));
    ASSERT_TRUE(ctx.allocate(out.size() * sizeof(float), &ob));
    ASSERT_TRUE(ctx.copy_to_device(x.data(), &xb, x.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(bias.data(), &bb, bias.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(out.data(), &ob, out.size() * sizeof(float)));
    ASSERT_TRUE(bolt::rocm::axpy_bias(ctx, static_cast<const float*>(xb.ptr),
                                      static_cast<const float*>(bb.ptr), scale,
                                      static_cast<float*>(ob.ptr), n));
    std::vector<float> o(static_cast<size_t>(n));
    ASSERT_TRUE(ctx.copy_to_host(ob, o.data(), o.size() * sizeof(float)));
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(o[static_cast<size_t>(i)], ref[static_cast<size_t>(i)], 1e-4f) << "i=" << i;
    ctx.free(&xb);
    ctx.free(&bb);
    ctx.free(&ob);
}

// BLLM-200: GPU MXFP4 matvec matches a CPU dequant-in-dot reference.
TEST(BoltRocm, MatmulMxfp4MatchesCpuReference) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    const int rows = 200, cols = 64;  // 2 MXFP4 blocks/row
    const int nb = cols / 32, row_bytes = nb * 17;
    static const float kMx[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                  0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    std::vector<uint8_t> w(static_cast<size_t>(rows) * row_bytes, 0);
    std::vector<float> x(static_cast<size_t>(cols)), ref(static_cast<size_t>(rows), 0.0f);
    for (int c = 0; c < cols; ++c) x[static_cast<size_t>(c)] = static_cast<float>((c % 9) - 4) * 0.2f;
    for (int r = 0; r < rows; ++r) {
        uint8_t* wr = w.data() + static_cast<size_t>(r) * row_bytes;
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            uint8_t* blk = wr + b * 17;
            const int e = 120 + (r % 12);
            blk[0] = static_cast<uint8_t>(e);
            const float scale = std::ldexp(1.0f, e - 127);
            for (int j = 0; j < 16; ++j) {
                const uint8_t lo = static_cast<uint8_t>((r + b + j) % 16);
                const uint8_t hi = static_cast<uint8_t>((r + 2 * b + j + 3) % 16);
                blk[1 + j] = static_cast<uint8_t>(lo | (hi << 4));
                acc += scale * (kMx[lo] * x[static_cast<size_t>(b) * 32 + j] +
                                kMx[hi] * x[static_cast<size_t>(b) * 32 + j + 16]);
            }
        }
        ref[static_cast<size_t>(r)] = acc;
    }
    DeviceBuffer wb, xb, ob;
    ASSERT_TRUE(ctx.allocate(w.size(), &wb));
    ASSERT_TRUE(ctx.allocate(x.size() * sizeof(float), &xb));
    ASSERT_TRUE(ctx.allocate(static_cast<uint64_t>(rows) * sizeof(float), &ob));
    ASSERT_TRUE(ctx.copy_to_device(w.data(), &wb, w.size()));
    ASSERT_TRUE(ctx.copy_to_device(x.data(), &xb, x.size() * sizeof(float)));
    ASSERT_TRUE(bolt::rocm::matmul_mxfp4(ctx, wb.ptr, static_cast<const float*>(xb.ptr),
                                         static_cast<float*>(ob.ptr), rows, cols));
    std::vector<float> out(static_cast<size_t>(rows));
    ASSERT_TRUE(ctx.copy_to_host(ob, out.data(), out.size() * sizeof(float)));
    for (int r = 0; r < rows; ++r)
        EXPECT_NEAR(out[static_cast<size_t>(r)], ref[static_cast<size_t>(r)], 1e-3f) << "r=" << r;
    ctx.free(&wb); ctx.free(&xb); ctx.free(&ob);
}

// BLLM-200 diagnostic: measure the iGPU's raw read bandwidth ceiling. Prints GB/s;
// compare to the ~105 GB/s the gpt-oss GEMVs achieve to decide whether HIP itself
// is the wall (kernels already at ceiling) or the kernels leave bandwidth unused.
TEST(BoltRocm, RawReadBandwidthProbe) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    double gbps = 0.0;
    ASSERT_TRUE(bolt::rocm::bandwidth_probe(ctx, 512ull * 1024 * 1024, 30, &gbps));
    std::printf("[BANDWIDTH] raw device read = %.1f GB/s (peak spec ~256 GB/s)\n", gbps);
    EXPECT_GT(gbps, 1.0);
}

// BLLM-200: the bandwidth-optimized aligned MXFP4 matvec matches the same CPU
// reference after repacking the 17-byte blocks into split codes/scales arrays.
TEST(BoltRocm, MatmulMxfp4AlignedMatchesCpuReference) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    const int rows = 200, cols = 128;  // 4 MXFP4 blocks/row
    const int nb = cols / 32, row_bytes = nb * 17;
    static const float kMx[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                  0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    std::vector<uint8_t> w(static_cast<size_t>(rows) * row_bytes, 0);
    std::vector<float> x(static_cast<size_t>(cols)), ref(static_cast<size_t>(rows), 0.0f);
    for (int c = 0; c < cols; ++c) x[static_cast<size_t>(c)] = static_cast<float>((c % 9) - 4) * 0.2f;
    for (int r = 0; r < rows; ++r) {
        uint8_t* wr = w.data() + static_cast<size_t>(r) * row_bytes;
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            uint8_t* blk = wr + b * 17;
            const int e = 120 + (r % 12);
            blk[0] = static_cast<uint8_t>(e);
            const float scale = std::ldexp(1.0f, e - 127);
            for (int j = 0; j < 16; ++j) {
                const uint8_t lo = static_cast<uint8_t>((r + b + j) % 16);
                const uint8_t hi = static_cast<uint8_t>((r + 2 * b + j + 3) % 16);
                blk[1 + j] = static_cast<uint8_t>(lo | (hi << 4));
                acc += scale * (kMx[lo] * x[static_cast<size_t>(b) * 32 + j] +
                                kMx[hi] * x[static_cast<size_t>(b) * 32 + j + 16]);
            }
        }
        ref[static_cast<size_t>(r)] = acc;
    }
    // Repack 17-byte blocks -> codes[rows*nb*16] + scales[rows*nb] (the upload-time
    // transform TransformerCore::rocm_setup_ does for the experts).
    std::vector<uint8_t> codes(static_cast<size_t>(rows) * nb * 16), scales(static_cast<size_t>(rows) * nb);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = w.data() + static_cast<size_t>(r) * row_bytes + b * 17;
            scales[static_cast<size_t>(r) * nb + b] = blk[0];
            std::memcpy(codes.data() + (static_cast<size_t>(r) * nb + b) * 16, blk + 1, 16);
        }

    DeviceBuffer cb, sb, xb, ob;
    ASSERT_TRUE(ctx.allocate(codes.size(), &cb));
    ASSERT_TRUE(ctx.allocate(scales.size(), &sb));
    ASSERT_TRUE(ctx.allocate(x.size() * sizeof(float), &xb));
    ASSERT_TRUE(ctx.allocate(static_cast<uint64_t>(rows) * sizeof(float), &ob));
    ASSERT_TRUE(ctx.copy_to_device(codes.data(), &cb, codes.size()));
    ASSERT_TRUE(ctx.copy_to_device(scales.data(), &sb, scales.size()));
    ASSERT_TRUE(ctx.copy_to_device(x.data(), &xb, x.size() * sizeof(float)));
    ASSERT_TRUE(bolt::rocm::matmul_mxfp4_aligned(ctx, cb.ptr, sb.ptr, static_cast<const float*>(xb.ptr),
                                                 static_cast<float*>(ob.ptr), rows, cols));
    std::vector<float> out(static_cast<size_t>(rows));
    ASSERT_TRUE(ctx.copy_to_host(ob, out.data(), out.size() * sizeof(float)));
    for (int r = 0; r < rows; ++r)
        EXPECT_NEAR(out[static_cast<size_t>(r)], ref[static_cast<size_t>(r)], 1e-3f) << "r=" << r;
    ctx.free(&cb); ctx.free(&sb); ctx.free(&xb); ctx.free(&ob);
}

// BLLM-188: a FULL pre-norm transformer layer (Qwen2/Llama-style, GQA, no
// biases, SwiGLU FFN) computed entirely on-device -- activations stay resident
// across the whole layer (only the input hidden is uploaded once and the final
// hidden read back once). Every step is a bolt::rocm op chained over device
// buffers: rmsnorm -> Q/K/V matmul -> RoPE -> attention -> O matmul -> residual
// add -> rmsnorm -> gate/up matmul -> SwiGLU -> down matmul -> residual add.
// This is the coherence gate for the ROCm resident-graph forward (BLLM-181):
// if the whole chain matches a CPU reference, the on-device graph is correct.
TEST(BoltRocm, ResidentLayerForwardMatchesCpuReference) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());

    const int H = 32, num_heads = 4, num_kv_heads = 2, head_dim = 8, ffn = 64;
    const int q_dim = num_heads * head_dim;   // 32
    const int kv_dim = num_kv_heads * head_dim;  // 16
    const int S = 4;                          // KV cache length; query is slot S-1
    const int group_size = num_heads / num_kv_heads;
    const float eps = 1e-5f, theta = 10000.0f;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto gen = [](int i, int salt) {
        return static_cast<float>(((i * 131 + salt * 17) % 19) - 9) * 0.05f;
    };

    // ---- Host tensors (deterministic) ----
    std::vector<float> x(static_cast<size_t>(H)), wn1(static_cast<size_t>(H)),
        wn2(static_cast<size_t>(H));
    for (int i = 0; i < H; ++i) {
        x[static_cast<size_t>(i)] = gen(i, 1);
        wn1[static_cast<size_t>(i)] = 0.5f + static_cast<float>(i % 4) * 0.1f;
        wn2[static_cast<size_t>(i)] = 0.7f + static_cast<float>(i % 3) * 0.1f;
    }
    auto mk = [&](int rows, int cols, int salt) {
        std::vector<float> w(static_cast<size_t>(rows) * cols);
        for (int i = 0; i < rows * cols; ++i) w[static_cast<size_t>(i)] = gen(i, salt);
        return w;
    };
    const std::vector<float> Wq = mk(q_dim, H, 2), Wk = mk(kv_dim, H, 3), Wv = mk(kv_dim, H, 4),
                             Wo = mk(H, q_dim, 5), Wg = mk(ffn, H, 6), Wu = mk(ffn, H, 7),
                             Wd = mk(H, ffn, 8);
    // Prior KV cache rows [0, S-1) are fixed synthetic (both paths read identical bytes).
    std::vector<float> kc(static_cast<size_t>(S) * kv_dim), vc(static_cast<size_t>(S) * kv_dim);
    for (int i = 0; i < (S - 1) * kv_dim; ++i) {
        kc[static_cast<size_t>(i)] = gen(i, 9);
        vc[static_cast<size_t>(i)] = gen(i, 10);
    }

    // ---- CPU reference ----
    auto rms = [&](const std::vector<float>& in, const std::vector<float>& w) {
        double ss = 0.0;
        for (int i = 0; i < H; ++i) ss += static_cast<double>(in[static_cast<size_t>(i)]) *
                                          in[static_cast<size_t>(i)];
        const float inv = static_cast<float>(1.0 / std::sqrt(ss / H + eps));
        std::vector<float> o(static_cast<size_t>(H));
        for (int i = 0; i < H; ++i)
            o[static_cast<size_t>(i)] = in[static_cast<size_t>(i)] * inv * w[static_cast<size_t>(i)];
        return o;
    };
    auto mv = [](const std::vector<float>& w, const std::vector<float>& a, int rows, int cols) {
        std::vector<float> o(static_cast<size_t>(rows), 0.0f);
        for (int r = 0; r < rows; ++r) {
            float s = 0.0f;
            for (int c = 0; c < cols; ++c)
                s += w[static_cast<size_t>(r) * cols + c] * a[static_cast<size_t>(c)];
            o[static_cast<size_t>(r)] = s;
        }
        return o;
    };
    auto rope = [&](std::vector<float>& t, int heads) {
        const int hd2 = head_dim / 2;
        for (int h = 0; h < heads; ++h) {
            float* v = t.data() + static_cast<size_t>(h) * head_dim;
            for (int j = 0; j < hd2; ++j) {
                const float inv = std::pow(theta, -2.0f * static_cast<float>(j) / head_dim);
                const float ang = static_cast<float>(S - 1) * inv;
                const float cs = std::cos(ang), sn = std::sin(ang);
                const float a = v[j], b = v[j + hd2];
                v[j] = a * cs - b * sn;
                v[j + hd2] = b * cs + a * sn;
            }
        }
    };
    std::vector<float> ref_x = x;
    {
        std::vector<float> n1 = rms(ref_x, wn1);
        std::vector<float> q = mv(Wq, n1, q_dim, H), k = mv(Wk, n1, kv_dim, H),
                           v = mv(Wv, n1, kv_dim, H);
        rope(q, num_heads);
        rope(k, num_kv_heads);
        for (int i = 0; i < kv_dim; ++i) {
            kc[static_cast<size_t>(S - 1) * kv_dim + i] = k[static_cast<size_t>(i)];
            vc[static_cast<size_t>(S - 1) * kv_dim + i] = v[static_cast<size_t>(i)];
        }
        std::vector<float> cxt(static_cast<size_t>(q_dim), 0.0f);
        for (int h = 0; h < num_heads; ++h) {
            const int kvh = h / group_size;
            std::vector<float> sc(static_cast<size_t>(S));
            float mx = -1e30f;
            for (int t = 0; t < S; ++t) {
                float s = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                    s += q[static_cast<size_t>(h) * head_dim + d] *
                         kc[(static_cast<size_t>(t) * num_kv_heads + kvh) * head_dim + d];
                s *= scale;
                sc[static_cast<size_t>(t)] = s;
                mx = std::fmax(mx, s);
            }
            double den = 0.0;
            for (int t = 0; t < S; ++t) {
                sc[static_cast<size_t>(t)] = std::exp(sc[static_cast<size_t>(t)] - mx);
                den += sc[static_cast<size_t>(t)];
            }
            for (int t = 0; t < S; ++t) {
                const float wgt = static_cast<float>(sc[static_cast<size_t>(t)] / den);
                for (int d = 0; d < head_dim; ++d)
                    cxt[static_cast<size_t>(h) * head_dim + d] +=
                        wgt * vc[(static_cast<size_t>(t) * num_kv_heads + kvh) * head_dim + d];
            }
        }
        std::vector<float> ao = mv(Wo, cxt, H, q_dim);
        for (int i = 0; i < H; ++i) ref_x[static_cast<size_t>(i)] += ao[static_cast<size_t>(i)];
        std::vector<float> n2 = rms(ref_x, wn2);
        std::vector<float> g = mv(Wg, n2, ffn, H), u = mv(Wu, n2, ffn, H);
        std::vector<float> act(static_cast<size_t>(ffn));
        for (int i = 0; i < ffn; ++i) {
            const float gv = g[static_cast<size_t>(i)];
            act[static_cast<size_t>(i)] = (gv / (1.0f + std::exp(-gv))) * u[static_cast<size_t>(i)];
        }
        std::vector<float> dn = mv(Wd, act, H, ffn);
        for (int i = 0; i < H; ++i) ref_x[static_cast<size_t>(i)] += dn[static_cast<size_t>(i)];
    }

    // ---- On-device resident forward ----
    auto up = [&](const std::vector<float>& h, DeviceBuffer* b) {
        ASSERT_TRUE(ctx.allocate(h.size() * sizeof(float), b));
        ASSERT_TRUE(ctx.copy_to_device(h.data(), b, h.size() * sizeof(float)));
    };
    DeviceBuffer dx, dn1, dn2, dWq, dWk, dWv, dWo, dWg, dWu, dWd, dkc, dvc;
    DeviceBuffer dq, dcxt, dao, dg, du, dact, ddn;
    up(x, &dx);
    up(wn1, &dn1);
    up(wn2, &dn2);
    up(Wq, &dWq);
    up(Wk, &dWk);
    up(Wv, &dWv);
    up(Wo, &dWo);
    up(Wg, &dWg);
    up(Wu, &dWu);
    up(Wd, &dWd);
    up(kc, &dkc);  // prior rows live; current-token slot overwritten on-device below
    up(vc, &dvc);
    DeviceBuffer dnorm;
    ASSERT_TRUE(ctx.allocate(static_cast<uint64_t>(H) * sizeof(float), &dnorm));
    ASSERT_TRUE(ctx.allocate(static_cast<uint64_t>(q_dim) * sizeof(float), &dq));
    ASSERT_TRUE(ctx.allocate(static_cast<uint64_t>(q_dim) * sizeof(float), &dcxt));
    ASSERT_TRUE(ctx.allocate(static_cast<uint64_t>(H) * sizeof(float), &dao));
    ASSERT_TRUE(ctx.allocate(static_cast<uint64_t>(ffn) * sizeof(float), &dg));
    ASSERT_TRUE(ctx.allocate(static_cast<uint64_t>(ffn) * sizeof(float), &du));
    ASSERT_TRUE(ctx.allocate(static_cast<uint64_t>(ffn) * sizeof(float), &dact));
    ASSERT_TRUE(ctx.allocate(static_cast<uint64_t>(H) * sizeof(float), &ddn));

    float* pkc = static_cast<float*>(dkc.ptr);
    float* pvc = static_cast<float*>(dvc.ptr);
    float* k_slot = pkc + static_cast<size_t>(S - 1) * kv_dim;
    float* v_slot = pvc + static_cast<size_t>(S - 1) * kv_dim;

    // Attention sublayer.
    ASSERT_TRUE(bolt::rocm::rmsnorm(ctx, static_cast<const float*>(dx.ptr),
                                    static_cast<const float*>(dn1.ptr),
                                    static_cast<float*>(dnorm.ptr), H, eps));
    // FUSED QKV (BLLM-189): one launch for q/k/v; k/v write into the cache slots.
    ASSERT_TRUE(bolt::rocm::fused_qkv(ctx, dWq.ptr, dWk.ptr, dWv.ptr, WeightDType::F32, nullptr,
                                      nullptr, nullptr, static_cast<const float*>(dnorm.ptr),
                                      static_cast<float*>(dq.ptr), k_slot, v_slot, q_dim, kv_dim, H));
    ASSERT_TRUE(bolt::rocm::rope(ctx, static_cast<float*>(dq.ptr), head_dim, num_heads, S - 1, theta));
    ASSERT_TRUE(bolt::rocm::rope(ctx, k_slot, head_dim, num_kv_heads, S - 1, theta));
    ASSERT_TRUE(bolt::rocm::attention(ctx, static_cast<const float*>(dq.ptr), pkc, pvc,
                                      static_cast<float*>(dcxt.ptr), num_heads, num_kv_heads,
                                      head_dim, S, scale));
    // O-proj with FUSED residual add (BLLM-189): dx += Wo·ctx.
    ASSERT_TRUE(bolt::rocm::matmul_dequant_residual(ctx, dWo.ptr, WeightDType::F32, nullptr,
                                                    static_cast<const float*>(dcxt.ptr),
                                                    static_cast<float*>(dx.ptr), H, q_dim));
    // FFN sublayer.
    ASSERT_TRUE(bolt::rocm::rmsnorm(ctx, static_cast<const float*>(dx.ptr),
                                    static_cast<const float*>(dn2.ptr),
                                    static_cast<float*>(dnorm.ptr), H, eps));
    // FUSED gate+up+SwiGLU (BLLM-189): 3 ops -> 1.
    ASSERT_TRUE(bolt::rocm::fused_swiglu(ctx, dWg.ptr, dWu.ptr, WeightDType::F32, nullptr, nullptr,
                                         static_cast<const float*>(dnorm.ptr),
                                         static_cast<float*>(dact.ptr), ffn, H));
    // down-proj with FUSED residual add: dx += Wd·act.
    ASSERT_TRUE(bolt::rocm::matmul_dequant_residual(ctx, dWd.ptr, WeightDType::F32, nullptr,
                                                    static_cast<const float*>(dact.ptr),
                                                    static_cast<float*>(dx.ptr), H, ffn));

    std::vector<float> out(static_cast<size_t>(H));
    ASSERT_TRUE(ctx.copy_to_host(dx, out.data(), out.size() * sizeof(float)));
    for (int i = 0; i < H; ++i) {
        EXPECT_NEAR(out[static_cast<size_t>(i)], ref_x[static_cast<size_t>(i)], 1e-3f) << "i=" << i;
    }
}

// BLLM-189: fused gate+up+SwiGLU (F32) matches the unfused gate-matmul +
// up-matmul + SwiGLU reference.
TEST(BoltRocm, FusedSwigluF32MatchesCpuReference) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    const int rows = 300, cols = 128;  // ffn rows x hidden cols
    std::vector<float> wg(static_cast<size_t>(rows) * cols), wu(static_cast<size_t>(rows) * cols),
        x(static_cast<size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        wg[static_cast<size_t>(i)] = static_cast<float>((i % 7) - 3) * 0.1f;
        wu[static_cast<size_t>(i)] = static_cast<float>((i % 5) - 2) * 0.15f;
    }
    for (int c = 0; c < cols; ++c) x[static_cast<size_t>(c)] = static_cast<float>((c % 9) - 4) * 0.2f;

    std::vector<float> ref(static_cast<size_t>(rows));
    for (int r = 0; r < rows; ++r) {
        float g = 0.0f, u = 0.0f;
        for (int c = 0; c < cols; ++c) {
            g += x[static_cast<size_t>(c)] * wg[static_cast<size_t>(r) * cols + c];
            u += x[static_cast<size_t>(c)] * wu[static_cast<size_t>(r) * cols + c];
        }
        ref[static_cast<size_t>(r)] = (g / (1.0f + std::exp(-g))) * u;
    }

    DeviceBuffer dg, du, dx, da;
    ASSERT_TRUE(ctx.allocate(wg.size() * sizeof(float), &dg));
    ASSERT_TRUE(ctx.allocate(wu.size() * sizeof(float), &du));
    ASSERT_TRUE(ctx.allocate(x.size() * sizeof(float), &dx));
    ASSERT_TRUE(ctx.allocate(static_cast<uint64_t>(rows) * sizeof(float), &da));
    ASSERT_TRUE(ctx.copy_to_device(wg.data(), &dg, wg.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(wu.data(), &du, wu.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(x.data(), &dx, x.size() * sizeof(float)));
    ASSERT_TRUE(bolt::rocm::fused_swiglu(ctx, dg.ptr, du.ptr, WeightDType::F32, nullptr, nullptr,
                                         static_cast<const float*>(dx.ptr),
                                         static_cast<float*>(da.ptr), rows, cols));
    std::vector<float> act(static_cast<size_t>(rows));
    ASSERT_TRUE(ctx.copy_to_host(da, act.data(), act.size() * sizeof(float)));
    for (int r = 0; r < rows; ++r)
        EXPECT_NEAR(act[static_cast<size_t>(r)], ref[static_cast<size_t>(r)], 1e-4f) << "r=" << r;
    ctx.free(&dg); ctx.free(&du); ctx.free(&dx); ctx.free(&da);
}

// BLLM-189: fused gate+up+SwiGLU (Int8 weights + per-row scales).
TEST(BoltRocm, FusedSwigluInt8MatchesCpuReference) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    const int rows = 256, cols = 96;
    std::vector<int8_t> wg(static_cast<size_t>(rows) * cols), wu(static_cast<size_t>(rows) * cols);
    std::vector<float> gs(static_cast<size_t>(rows)), us(static_cast<size_t>(rows)),
        x(static_cast<size_t>(cols));
    for (int i = 0; i < rows * cols; ++i) {
        wg[static_cast<size_t>(i)] = static_cast<int8_t>((i % 15) - 7);
        wu[static_cast<size_t>(i)] = static_cast<int8_t>((i % 11) - 5);
    }
    for (int r = 0; r < rows; ++r) {
        gs[static_cast<size_t>(r)] = 0.01f + static_cast<float>(r % 4) * 0.003f;
        us[static_cast<size_t>(r)] = 0.02f + static_cast<float>(r % 3) * 0.004f;
    }
    for (int c = 0; c < cols; ++c) x[static_cast<size_t>(c)] = static_cast<float>((c % 9) - 4) * 0.2f;

    std::vector<float> ref(static_cast<size_t>(rows));
    for (int r = 0; r < rows; ++r) {
        float g = 0.0f, u = 0.0f;
        for (int c = 0; c < cols; ++c) {
            g += x[static_cast<size_t>(c)] * static_cast<float>(wg[static_cast<size_t>(r) * cols + c]);
            u += x[static_cast<size_t>(c)] * static_cast<float>(wu[static_cast<size_t>(r) * cols + c]);
        }
        g *= gs[static_cast<size_t>(r)];
        u *= us[static_cast<size_t>(r)];
        ref[static_cast<size_t>(r)] = (g / (1.0f + std::exp(-g))) * u;
    }

    DeviceBuffer dg, du, dgs, dus, dx, da;
    ASSERT_TRUE(ctx.allocate(wg.size(), &dg));
    ASSERT_TRUE(ctx.allocate(wu.size(), &du));
    ASSERT_TRUE(ctx.allocate(gs.size() * sizeof(float), &dgs));
    ASSERT_TRUE(ctx.allocate(us.size() * sizeof(float), &dus));
    ASSERT_TRUE(ctx.allocate(x.size() * sizeof(float), &dx));
    ASSERT_TRUE(ctx.allocate(static_cast<uint64_t>(rows) * sizeof(float), &da));
    ASSERT_TRUE(ctx.copy_to_device(wg.data(), &dg, wg.size()));
    ASSERT_TRUE(ctx.copy_to_device(wu.data(), &du, wu.size()));
    ASSERT_TRUE(ctx.copy_to_device(gs.data(), &dgs, gs.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(us.data(), &dus, us.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(x.data(), &dx, x.size() * sizeof(float)));
    ASSERT_TRUE(bolt::rocm::fused_swiglu(ctx, dg.ptr, du.ptr, WeightDType::Int8,
                                         static_cast<const float*>(dgs.ptr),
                                         static_cast<const float*>(dus.ptr),
                                         static_cast<const float*>(dx.ptr),
                                         static_cast<float*>(da.ptr), rows, cols));
    std::vector<float> act(static_cast<size_t>(rows));
    ASSERT_TRUE(ctx.copy_to_host(da, act.data(), act.size() * sizeof(float)));
    for (int r = 0; r < rows; ++r)
        EXPECT_NEAR(act[static_cast<size_t>(r)], ref[static_cast<size_t>(r)], 1e-4f) << "r=" << r;
    ctx.free(&dg); ctx.free(&du); ctx.free(&dgs); ctx.free(&dus); ctx.free(&dx); ctx.free(&da);
}

// BLLM-189: fused gate+up+SwiGLU (Int4 weights, nibble-packed low-first, offset 8).
TEST(BoltRocm, FusedSwigluInt4MatchesCpuReference) {
    if (!bolt::rocm::available()) {
        GTEST_SKIP() << "No ROCm-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    const int rows = 256, cols = 96;                 // cols even -> exact nibble packing
    const int packed = (cols + 1) / 2;
    std::vector<uint8_t> wg(static_cast<size_t>(rows) * packed),
        wu(static_cast<size_t>(rows) * packed);
    std::vector<float> gs(static_cast<size_t>(rows)), us(static_cast<size_t>(rows)),
        x(static_cast<size_t>(cols));
    // Nibble codes in [0,15]; value = code - 8.
    auto code = [](int i) { return static_cast<uint8_t>(i % 16); };
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const uint8_t nib = code(r + c);
            const size_t bi = static_cast<size_t>(r) * packed + (c >> 1);
            if (c & 1) { wg[bi] = static_cast<uint8_t>((wg[bi] & 0x0F) | (nib << 4)); }
            else { wg[bi] = static_cast<uint8_t>((wg[bi] & 0xF0) | nib); }
            const uint8_t nib2 = code(r + 2 * c + 1);
            if (c & 1) { wu[bi] = static_cast<uint8_t>((wu[bi] & 0x0F) | (nib2 << 4)); }
            else { wu[bi] = static_cast<uint8_t>((wu[bi] & 0xF0) | nib2); }
        }
        gs[static_cast<size_t>(r)] = 0.01f + static_cast<float>(r % 4) * 0.003f;
        us[static_cast<size_t>(r)] = 0.02f + static_cast<float>(r % 3) * 0.004f;
    }
    for (int c = 0; c < cols; ++c) x[static_cast<size_t>(c)] = static_cast<float>((c % 9) - 4) * 0.2f;

    std::vector<float> ref(static_cast<size_t>(rows));
    for (int r = 0; r < rows; ++r) {
        float g = 0.0f, u = 0.0f;
        for (int c = 0; c < cols; ++c) {
            g += x[static_cast<size_t>(c)] * static_cast<float>(static_cast<int>(code(r + c)) - 8);
            u += x[static_cast<size_t>(c)] * static_cast<float>(static_cast<int>(code(r + 2 * c + 1)) - 8);
        }
        g *= gs[static_cast<size_t>(r)];
        u *= us[static_cast<size_t>(r)];
        ref[static_cast<size_t>(r)] = (g / (1.0f + std::exp(-g))) * u;
    }

    DeviceBuffer dg, du, dgs, dus, dx, da;
    ASSERT_TRUE(ctx.allocate(wg.size(), &dg));
    ASSERT_TRUE(ctx.allocate(wu.size(), &du));
    ASSERT_TRUE(ctx.allocate(gs.size() * sizeof(float), &dgs));
    ASSERT_TRUE(ctx.allocate(us.size() * sizeof(float), &dus));
    ASSERT_TRUE(ctx.allocate(x.size() * sizeof(float), &dx));
    ASSERT_TRUE(ctx.allocate(static_cast<uint64_t>(rows) * sizeof(float), &da));
    ASSERT_TRUE(ctx.copy_to_device(wg.data(), &dg, wg.size()));
    ASSERT_TRUE(ctx.copy_to_device(wu.data(), &du, wu.size()));
    ASSERT_TRUE(ctx.copy_to_device(gs.data(), &dgs, gs.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(us.data(), &dus, us.size() * sizeof(float)));
    ASSERT_TRUE(ctx.copy_to_device(x.data(), &dx, x.size() * sizeof(float)));
    ASSERT_TRUE(bolt::rocm::fused_swiglu(ctx, dg.ptr, du.ptr, WeightDType::Int4,
                                         static_cast<const float*>(dgs.ptr),
                                         static_cast<const float*>(dus.ptr),
                                         static_cast<const float*>(dx.ptr),
                                         static_cast<float*>(da.ptr), rows, cols));
    std::vector<float> act(static_cast<size_t>(rows));
    ASSERT_TRUE(ctx.copy_to_host(da, act.data(), act.size() * sizeof(float)));
    for (int r = 0; r < rows; ++r)
        EXPECT_NEAR(act[static_cast<size_t>(r)], ref[static_cast<size_t>(r)], 1e-4f) << "r=" << r;
    ctx.free(&dg); ctx.free(&du); ctx.free(&dgs); ctx.free(&dus); ctx.free(&dx); ctx.free(&da);
}
