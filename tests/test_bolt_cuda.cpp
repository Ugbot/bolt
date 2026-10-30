// test_bolt_cuda.cpp -- coverage for bolt::cuda's matmul+dequant kernel
// core (BLLM-182). Only built when BOLT_BUILD_CUDA is ON (a real nvcc was
// found) -- this dev box has no NVIDIA GPU / CUDA toolkit, so this
// file is not part of the build here (see bolt_cuda.h's status note).
// Self-skips (not a hard failure) if bolt::cuda::available() is false,
// matching test_bolt_vulkan.cpp's own contract. STL is allowed in tests;
// production code stays Tiger-Style.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "bolt/bolt_cuda.h"

using bolt::cuda::Context;
using bolt::cuda::DeviceBuffer;
using bolt::cuda::WeightDType;

TEST(BoltCuda, AvailableDoesNotCrashRegardlessOfHardware) {
    const bool result = bolt::cuda::available();
    EXPECT_TRUE(result == true || result == false);  // trivially true; documents intent
}

TEST(BoltCuda, ContextCreateSucceedsOnThisRealDevice) {
    if (!bolt::cuda::available()) {
        GTEST_SKIP() << "No CUDA-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    EXPECT_TRUE(ctx.is_valid());
    EXPECT_GT(std::strlen(ctx.device_name()), 0u);
    EXPECT_GT(ctx.device_local_heap_bytes(), 0u);
    ctx.destroy();
    EXPECT_FALSE(ctx.is_valid());
}

TEST(BoltCuda, DestroyIsSafeOnNeverCreatedContext) {
    Context ctx;
    ctx.destroy();  // must not crash
    EXPECT_FALSE(ctx.is_valid());
}

TEST(BoltCuda, AllocateFreeAndRoundTripCopy) {
    if (!bolt::cuda::available()) {
        GTEST_SKIP() << "No CUDA-capable device on this machine";
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

TEST(BoltCuda, MatmulDequantF32MatchesHandComputedDotProduct) {
    if (!bolt::cuda::available()) {
        GTEST_SKIP() << "No CUDA-capable device on this machine";
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

    ASSERT_TRUE(bolt::cuda::matmul_dequant(ctx, weight_buf.ptr, WeightDType::F32, nullptr,
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

TEST(BoltCuda, MatmulDequantInt8MatchesScaledDotProduct) {
    if (!bolt::cuda::available()) {
        GTEST_SKIP() << "No CUDA-capable device on this machine";
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

    ASSERT_TRUE(bolt::cuda::matmul_dequant(ctx, weight_buf.ptr, WeightDType::Int8,
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

TEST(BoltCuda, MatmulDequantFailsCleanlyOnInvalidContext) {
    Context ctx;  // create() never called
    const float activation[2] = {1.0f, 1.0f};
    float out[1] = {0};
    const float weight[2] = {1.0f, 1.0f};
    EXPECT_FALSE(bolt::cuda::matmul_dequant(ctx, weight, WeightDType::F32, nullptr, activation, out,
                                             1, 2));
}

TEST(BoltCuda, MatmulDequantRequiresScaleForQuantizedDtypes) {
    if (!bolt::cuda::available()) {
        GTEST_SKIP() << "No CUDA-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    const float activation[2] = {1.0f, 1.0f};
    float out[1] = {0};
    const int8_t weight[2] = {1, 1};
    // scale_device == nullptr for a quantized dtype must fail, not crash.
    EXPECT_FALSE(bolt::cuda::matmul_dequant(ctx, weight, WeightDType::Int8, nullptr, activation, out,
                                             1, 2));
}

// BLLM-184: on-device CUDA RMSNorm matches the CPU reference.
TEST(BoltCuda, RmsnormMatchesCpuReference) {
    if (!bolt::cuda::available()) {
        GTEST_SKIP() << "No CUDA-capable device on this machine";
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
    ASSERT_TRUE(bolt::cuda::rmsnorm(ctx, static_cast<const float*>(xb.ptr),
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

// BLLM-185: on-device CUDA RoPE matches rope_half_split per head.
TEST(BoltCuda, RopeMatchesCpuReference) {
    if (!bolt::cuda::available()) {
        GTEST_SKIP() << "No CUDA-capable device on this machine";
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
    ASSERT_TRUE(bolt::cuda::rope(ctx, static_cast<float*>(xb.ptr), head_dim, num_heads, position,
                                 theta));
    std::vector<float> g(static_cast<size_t>(n));
    ASSERT_TRUE(ctx.copy_to_host(xb, g.data(), g.size() * sizeof(float)));
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(g[static_cast<size_t>(i)], ref[static_cast<size_t>(i)], 1e-4f) << "i=" << i;
    }
    ctx.free(&xb);
}

// BLLM-186: on-device CUDA elementwise (SwiGLU op1 + add op0) matches CPU.
TEST(BoltCuda, ElementwiseMatchesCpuReference) {
    if (!bolt::cuda::available()) {
        GTEST_SKIP() << "No CUDA-capable device on this machine";
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

    ASSERT_TRUE(bolt::cuda::elementwise(ctx, static_cast<const float*>(ab.ptr),
                                        static_cast<const float*>(bb.ptr),
                                        static_cast<float*>(ob.ptr), n, /*op=*/1));
    ASSERT_TRUE(ctx.copy_to_host(ob, o.data(), o.size() * sizeof(float)));
    for (int i = 0; i < n; ++i) {
        const float x = a[static_cast<size_t>(i)];
        EXPECT_NEAR(o[static_cast<size_t>(i)], (x / (1.0f + std::exp(-x))) * b[static_cast<size_t>(i)],
                    1e-4f)
            << "swiglu i=" << i;
    }

    ASSERT_TRUE(bolt::cuda::elementwise(ctx, static_cast<const float*>(ab.ptr),
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

// BLLM-187: on-device CUDA decode-step attention (GQA) matches a two-pass CPU
// softmax reference (mathematically identical to the online-softmax kernel).
TEST(BoltCuda, AttentionMatchesCpuReference) {
    if (!bolt::cuda::available()) {
        GTEST_SKIP() << "No CUDA-capable device on this machine";
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
    ASSERT_TRUE(bolt::cuda::attention(ctx, static_cast<const float*>(qb.ptr),
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

// BLLM-188: a FULL pre-norm transformer layer (Qwen2/Llama-style, GQA, no
// biases, SwiGLU FFN) computed entirely on-device -- activations stay resident
// across the whole layer (only the input hidden is uploaded once and the final
// hidden read back once). Every step is a bolt::cuda op chained over device
// buffers: rmsnorm -> Q/K/V matmul -> RoPE -> attention -> O matmul -> residual
// add -> rmsnorm -> gate/up matmul -> SwiGLU -> down matmul -> residual add.
// This is the coherence gate for the CUDA resident-graph forward (BLLM-181):
// if the whole chain matches a CPU reference, the on-device graph is correct.
TEST(BoltCuda, ResidentLayerForwardMatchesCpuReference) {
    if (!bolt::cuda::available()) {
        GTEST_SKIP() << "No CUDA-capable device on this machine";
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
    ASSERT_TRUE(bolt::cuda::rmsnorm(ctx, static_cast<const float*>(dx.ptr),
                                    static_cast<const float*>(dn1.ptr),
                                    static_cast<float*>(dnorm.ptr), H, eps));
    ASSERT_TRUE(bolt::cuda::matmul_dequant(ctx, dWq.ptr, WeightDType::F32, nullptr,
                                           static_cast<const float*>(dnorm.ptr),
                                           static_cast<float*>(dq.ptr), q_dim, H));
    ASSERT_TRUE(bolt::cuda::matmul_dequant(ctx, dWk.ptr, WeightDType::F32, nullptr,
                                           static_cast<const float*>(dnorm.ptr), k_slot, kv_dim, H));
    ASSERT_TRUE(bolt::cuda::matmul_dequant(ctx, dWv.ptr, WeightDType::F32, nullptr,
                                           static_cast<const float*>(dnorm.ptr), v_slot, kv_dim, H));
    ASSERT_TRUE(bolt::cuda::rope(ctx, static_cast<float*>(dq.ptr), head_dim, num_heads, S - 1, theta));
    ASSERT_TRUE(bolt::cuda::rope(ctx, k_slot, head_dim, num_kv_heads, S - 1, theta));
    ASSERT_TRUE(bolt::cuda::attention(ctx, static_cast<const float*>(dq.ptr), pkc, pvc,
                                      static_cast<float*>(dcxt.ptr), num_heads, num_kv_heads,
                                      head_dim, S, scale));
    ASSERT_TRUE(bolt::cuda::matmul_dequant(ctx, dWo.ptr, WeightDType::F32, nullptr,
                                           static_cast<const float*>(dcxt.ptr),
                                           static_cast<float*>(dao.ptr), H, q_dim));
    ASSERT_TRUE(bolt::cuda::elementwise(ctx, static_cast<const float*>(dx.ptr),
                                        static_cast<const float*>(dao.ptr),
                                        static_cast<float*>(dx.ptr), H, /*add=*/0));
    // FFN sublayer.
    ASSERT_TRUE(bolt::cuda::rmsnorm(ctx, static_cast<const float*>(dx.ptr),
                                    static_cast<const float*>(dn2.ptr),
                                    static_cast<float*>(dnorm.ptr), H, eps));
    ASSERT_TRUE(bolt::cuda::matmul_dequant(ctx, dWg.ptr, WeightDType::F32, nullptr,
                                           static_cast<const float*>(dnorm.ptr),
                                           static_cast<float*>(dg.ptr), ffn, H));
    ASSERT_TRUE(bolt::cuda::matmul_dequant(ctx, dWu.ptr, WeightDType::F32, nullptr,
                                           static_cast<const float*>(dnorm.ptr),
                                           static_cast<float*>(du.ptr), ffn, H));
    ASSERT_TRUE(bolt::cuda::elementwise(ctx, static_cast<const float*>(dg.ptr),
                                        static_cast<const float*>(du.ptr),
                                        static_cast<float*>(dact.ptr), ffn, /*swiglu=*/1));
    ASSERT_TRUE(bolt::cuda::matmul_dequant(ctx, dWd.ptr, WeightDType::F32, nullptr,
                                           static_cast<const float*>(dact.ptr),
                                           static_cast<float*>(ddn.ptr), H, ffn));
    ASSERT_TRUE(bolt::cuda::elementwise(ctx, static_cast<const float*>(dx.ptr),
                                        static_cast<const float*>(ddn.ptr),
                                        static_cast<float*>(dx.ptr), H, /*add=*/0));

    std::vector<float> out(static_cast<size_t>(H));
    ASSERT_TRUE(ctx.copy_to_host(dx, out.data(), out.size() * sizeof(float)));
    for (int i = 0; i < H; ++i) {
        EXPECT_NEAR(out[static_cast<size_t>(i)], ref_x[static_cast<size_t>(i)], 1e-3f) << "i=" << i;
    }
}
