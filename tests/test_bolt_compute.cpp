// test_bolt_compute.cpp — Tests for the device-generic compute dispatch
// table (bolt/bolt_compute.h, BLLM-13), layered on bolt_tensor.h's
// bolt::Tensor/bolt::Device (BLLM-12).
//
// Each CPU-tier op is checked against either a hand-computed known-answer
// value (matmul's quantized paths, RoPE's small-dim cases) or an
// independently-written naive reference (rmsnorm/softmax/swiglu — written
// fresh here, not shared code with bolt_compute.h, so an algorithmic bug in
// one is unlikely to be mirrored in the other). The dispatch table itself is
// checked to resolve Device::CPU to the real implementation and
// Device::Vulkan/Device::CUDA/Device::ROCm to populated-but-hard-failing
// stubs (ROCm's real kernel, bolt::rocm::matmul_dequant -- BLLM-78 -- is a
// separate optional compiled library, not wired into this table yet).

#include "bolt/bolt_compute.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

using bolt::Device;
using bolt::DType;
using bolt::Tensor;

namespace {

// ---------------------------------------------------------------------------
// Independent naive references (deliberately not sharing code with
// bolt_compute.h's own implementations).
// ---------------------------------------------------------------------------

std::vector<float> naive_rmsnorm(const std::vector<float>& x, const std::vector<float>& w, float eps) {
    double sum_sq = 0.0;
    for (float xi : x) sum_sq += static_cast<double>(xi) * static_cast<double>(xi);
    const double inv_rms = 1.0 / std::sqrt(sum_sq / static_cast<double>(x.size()) + static_cast<double>(eps));
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = static_cast<float>(static_cast<double>(x[i]) * inv_rms * static_cast<double>(w[i]));
    }
    return y;
}

std::vector<float> naive_softmax(const std::vector<float>& x) {
    double max_val = x[0];
    for (float v : x) max_val = std::max(max_val, static_cast<double>(v));
    std::vector<double> exps(x.size());
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        exps[i] = std::exp(static_cast<double>(x[i]) - max_val);
        sum += exps[i];
    }
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = static_cast<float>(exps[i] / sum);
    return out;
}

float naive_sigmoid(double x) { return static_cast<float>(1.0 / (1.0 + std::exp(-x))); }
float naive_silu(double x) { return static_cast<float>(x * (1.0 / (1.0 + std::exp(-x)))); }

// ---------------------------------------------------------------------------
// matmul — F32 direct.
// ---------------------------------------------------------------------------

TEST(BoltCompute, MatmulF32KnownDotProducts) {
    // weight = [[1,2,3],[4,5,6]], activation = [1,0,-1].
    // row0 . act = 1 - 3 = -2 ; row1 . act = 4 - 6 = -2.
    std::vector<float> weight_data = {1, 2, 3, 4, 5, 6};
    Tensor weight = bolt::tensor_make_cpu_2d(DType::F32, /*rows=*/2, /*cols=*/3, weight_data.data());
    const float activation[3] = {1.0f, 0.0f, -1.0f};
    float out[2] = {0, 0};

    bolt::compute::matmul(Device::CPU, weight, activation, out, 2);

    EXPECT_FLOAT_EQ(out[0], -2.0f);
    EXPECT_FLOAT_EQ(out[1], -2.0f);
}

TEST(BoltCompute, MatmulF32PartialRows) {
    std::vector<float> weight_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Tensor weight = bolt::tensor_make_cpu_2d(DType::F32, 3, 3, weight_data.data());
    const float activation[3] = {1.0f, 1.0f, 1.0f};
    float out[2] = {0, 0};

    // Only compute the first 2 of 3 rows.
    bolt::compute::matmul(Device::CPU, weight, activation, out, 2);

    EXPECT_FLOAT_EQ(out[0], 6.0f);   // 1+2+3
    EXPECT_FLOAT_EQ(out[1], 15.0f);  // 4+5+6
}

// ---------------------------------------------------------------------------
// matmul — Int8/Int4/Int2 weight, dequant-on-the-fly against float activation.
// Packed bytes constructed by hand per QuantTypes.h's convention (restated
// in bolt_compute.h's banner): Int8 plain two's complement; Int4 low-nibble=
// even element + offset 8; Int2 2-bit code at bit offset 2k + offset 2.
// ---------------------------------------------------------------------------

TEST(BoltCompute, MatmulInt8DequantMatchesManualCalc) {
    // q = [10, -5, 3], scale = 0.1 -> dequant = [1.0, -0.5, 0.3].
    // activation = [2, 4, -1].
    // raw int dot = 10*2 + (-5)*4 + 3*(-1) = 20 - 20 - 3 = -3; * scale = -0.3.
    std::vector<int8_t> packed = {10, -5, 3};
    std::vector<float> scales = {0.1f};
    Tensor weight = bolt::tensor_make_cpu_2d(DType::Int8, 1, 3,
                                              reinterpret_cast<uint8_t*>(packed.data()), scales.data());
    const float activation[3] = {2.0f, 4.0f, -1.0f};
    float out[1] = {0};

    bolt::compute::matmul(Device::CPU, weight, activation, out, 1);

    EXPECT_NEAR(out[0], -0.3f, 1e-5f);
}

TEST(BoltCompute, MatmulInt4DequantMatchesManualCalcOddCols) {
    // cols=3 (odd -> exercises the tail branch). Logical values v=[3,-2,5],
    // scale=0.5 -> dequant = [1.5, -1.0, 2.5]. activation = [2,-1,4].
    // raw int dot = 2*3 + (-1)*(-2) + 4*5 = 6 + 2 + 20 = 28; * scale = 14.0.
    // byte0 (elements 0,1): lo=(3+8)=11, hi=(-2+8)=6 -> 11 | (6<<4) = 0x6B.
    // byte1 (element 2, tail): lo=(5+8)=13, hi unused (filler 0) -> 0x0D.
    std::vector<uint8_t> packed = {0x6B, 0x0D};
    std::vector<float> scales = {0.5f};
    Tensor weight = bolt::tensor_make_cpu_2d(DType::Int4, 1, 3, packed.data(), scales.data());
    const float activation[3] = {2.0f, -1.0f, 4.0f};
    float out[1] = {0};

    bolt::compute::matmul(Device::CPU, weight, activation, out, 1);

    EXPECT_NEAR(out[0], 14.0f, 1e-4f);
}

TEST(BoltCompute, MatmulInt2DequantMatchesManualCalcWithTailByte) {
    // cols=5 (spills into a 2nd byte). Logical values v=[1,-2,0,-1,1],
    // scale=0.25 -> dequant = [0.25,-0.5,0,-0.25,0.25].
    // activation = [4,2,-3,1,-2].
    // raw int dot = 4*1 + 2*(-2) + (-3)*0 + 1*(-1) + (-2)*1 = 4-4+0-1-2 = -3;
    // * scale = -0.75.
    // codes (v+2): [3,0,2,1,3].
    // byte0 (elems 0..3): 3 | (0<<2) | (2<<4) | (1<<6) = 0x63.
    // byte1 (elem 4, tail): 3 (rest filler 0) = 0x03.
    std::vector<uint8_t> packed = {0x63, 0x03};
    std::vector<float> scales = {0.25f};
    Tensor weight = bolt::tensor_make_cpu_2d(DType::Int2, 1, 5, packed.data(), scales.data());
    const float activation[5] = {4.0f, 2.0f, -3.0f, 1.0f, -2.0f};
    float out[1] = {0};

    bolt::compute::matmul(Device::CPU, weight, activation, out, 1);

    EXPECT_NEAR(out[0], -0.75f, 1e-4f);
}

// ---------------------------------------------------------------------------
// rmsnorm
// ---------------------------------------------------------------------------

TEST(BoltCompute, RmsnormMatchesNaiveReference) {
    std::vector<float> x = {3.0f, 4.0f, -2.0f, 1.5f};
    std::vector<float> w = {1.0f, 2.0f, 0.5f, 1.0f};
    const float eps = 1e-5f;
    std::vector<float> expected = naive_rmsnorm(x, w, eps);

    std::vector<float> y(x.size());
    bolt::compute::rmsnorm(Device::CPU, x.data(), w.data(), y.data(), static_cast<int32_t>(x.size()), eps);

    for (size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(y[i], expected[i], 1e-6f) << "i=" << i;
}

// ---------------------------------------------------------------------------
// rope_interleave — hand-computed known-answer cases (independent of
// boltllm::moe::MoeMath.h's own test suite, computed fresh from the pinned
// formula in bolt_compute.h's banner).
// ---------------------------------------------------------------------------

TEST(BoltCompute, RopeInterleaveKnownAnswerDim2) {
    // dim=2 (half=1), theta=10, position=1, x=[1,0].
    // inv_freq_0 = theta^0 = 1 -> angle = 1.
    // r0 = 1*cos(1) - 0*sin(1) = cos(1); r1 = 0*cos(1) + 1*sin(1) = sin(1).
    std::vector<float> x = {1.0f, 0.0f};
    bolt::compute::rope_interleave(Device::CPU, x.data(), 2, 1, 10.0f);
    EXPECT_NEAR(x[0], std::cos(1.0), 1e-5);
    EXPECT_NEAR(x[1], std::sin(1.0), 1e-5);
}

TEST(BoltCompute, RopeInterleavePositionZeroDeinterleavesOnly) {
    // At position 0, angle=0 for every pair -> pure de-interleave: values
    // unchanged, positions reshuffled (2j,2j+1) -> (j, half+j).
    std::vector<float> in = {10.0f, -3.0f, 2.5f, 7.0f};
    std::vector<float> x = in;
    bolt::compute::rope_interleave(Device::CPU, x.data(), 4, /*position=*/0, 8.0e6f);
    EXPECT_FLOAT_EQ(x[0], in[0]);
    EXPECT_FLOAT_EQ(x[1], in[2]);
    EXPECT_FLOAT_EQ(x[2], in[1]);
    EXPECT_FLOAT_EQ(x[3], in[3]);
}

TEST(BoltCompute, RopeInterleaveNormPreservation) {
    std::vector<float> x = {1.0f, 2.0f, -3.0f, 0.5f, 4.0f, -1.5f};
    const int half = 3;
    std::vector<float> in = x;
    bolt::compute::rope_interleave(Device::CPU, x.data(), 6, 12345, 8.0e6f);
    for (int j = 0; j < half; ++j) {
        const double norm_before = static_cast<double>(in[2 * j]) * in[2 * j] +
                                    static_cast<double>(in[2 * j + 1]) * in[2 * j + 1];
        const double norm_after = static_cast<double>(x[j]) * x[j] +
                                   static_cast<double>(x[half + j]) * x[half + j];
        EXPECT_NEAR(norm_after, norm_before, 1e-3 * (norm_before + 1.0)) << "j=" << j;
    }
}

// ---------------------------------------------------------------------------
// rope_half_split — the other convention (GPT-NeoX/Llama "rotate_half"
// style). See bolt_compute.h's banner for the HF transformers citation this
// implements.
// ---------------------------------------------------------------------------

TEST(BoltCompute, RopeHalfSplitPositionZeroIsIdentity) {
    // Unlike rope_interleave, half-split's pairing (j, half+j) already IS
    // the deposit layout, so position 0 (angle=0) is a true no-op, not a
    // permutation.
    std::vector<float> in = {10.0f, -3.0f, 2.5f, 7.0f};
    std::vector<float> x = in;
    bolt::compute::rope_half_split(Device::CPU, x.data(), 4, /*position=*/0, 8.0e6f);
    for (size_t i = 0; i < in.size(); ++i) EXPECT_FLOAT_EQ(x[i], in[i]) << "i=" << i;
}

TEST(BoltCompute, RopeHalfSplitKnownAnswerDim4) {
    // dim=4 (half=2), theta=10000, position=1, x=[1,2,3,4].
    // Pairing is (x[0],x[2]) and (x[1],x[3]) (NOT adjacent).
    // j=0: inv_freq=theta^0=1 -> angle=1.
    //   a=x[0]=1,b=x[2]=3 -> out[0]=cos(1)-3*sin(1); out[2]=3*cos(1)+sin(1).
    // j=1: inv_freq=theta^-0.5=0.01 -> angle=0.01.
    //   a=x[1]=2,b=x[3]=4 -> out[1]=2*cos(0.01)-4*sin(0.01);
    //                        out[3]=4*cos(0.01)+2*sin(0.01).
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    bolt::compute::rope_half_split(Device::CPU, x.data(), 4, 1, 10000.0f);

    const double angle0 = 1.0;
    const double angle1 = 0.01;
    EXPECT_NEAR(x[0], std::cos(angle0) - 3.0 * std::sin(angle0), 1e-5);
    EXPECT_NEAR(x[2], 3.0 * std::cos(angle0) + std::sin(angle0), 1e-5);
    EXPECT_NEAR(x[1], 2.0 * std::cos(angle1) - 4.0 * std::sin(angle1), 1e-5);
    EXPECT_NEAR(x[3], 4.0 * std::cos(angle1) + 2.0 * std::sin(angle1), 1e-5);
}

TEST(BoltCompute, RopeHalfSplitNormPreservation) {
    std::vector<float> x = {1.0f, 2.0f, -3.0f, 0.5f, 4.0f, -1.5f};
    const int half = 3;
    std::vector<float> in = x;
    bolt::compute::rope_half_split(Device::CPU, x.data(), 6, 999, 8.0e6f);
    for (int j = 0; j < half; ++j) {
        const double norm_before = static_cast<double>(in[j]) * in[j] +
                                    static_cast<double>(in[half + j]) * in[half + j];
        const double norm_after = static_cast<double>(x[j]) * x[j] +
                                   static_cast<double>(x[half + j]) * x[half + j];
        EXPECT_NEAR(norm_after, norm_before, 1e-3 * (norm_before + 1.0)) << "j=" << j;
    }
}

TEST(BoltCompute, RopeInterleaveAndHalfSplitAreGenuinelyDifferent) {
    // Same input/position/theta, two different conventions -> different
    // results (guards against an accidental copy-paste making one call the
    // other).
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = a;
    bolt::compute::rope_interleave(Device::CPU, a.data(), 4, 5, 10000.0f);
    bolt::compute::rope_half_split(Device::CPU, b.data(), 4, 5, 10000.0f);
    bool any_diff = false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > 1e-4f) any_diff = true;
    }
    EXPECT_TRUE(any_diff);
}

// ---------------------------------------------------------------------------
// softmax_inplace
// ---------------------------------------------------------------------------

TEST(BoltCompute, SoftmaxMatchesNaiveReference) {
    std::vector<float> x = {2.0f, -1.0f, 0.5f, 3.0f, -3.0f};
    std::vector<float> expected = naive_softmax(x);

    bolt::compute::softmax_inplace(Device::CPU, x.data(), static_cast<int32_t>(x.size()));

    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(x[i], expected[i], 1e-6f) << "i=" << i;
        sum += x[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-4f);
}

TEST(BoltCompute, SoftmaxNumericallyHostileInputStaysFinite) {
    std::vector<float> x = {1000.0f, 999.0f, -1000.0f};
    bolt::compute::softmax_inplace(Device::CPU, x.data(), 3);
    for (float v : x) {
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_GE(v, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// swiglu_row
// ---------------------------------------------------------------------------

TEST(BoltCompute, SwigluRowMatchesNaiveReference) {
    std::vector<float> gate = {-2.0f, 0.0f, 1.5f, 4.0f};
    std::vector<float> up = {3.0f, -1.0f, 2.0f, 0.5f};
    std::vector<float> out(gate.size());

    bolt::compute::swiglu_row(Device::CPU, gate.data(), up.data(), out.data(),
                               static_cast<int32_t>(gate.size()));

    for (size_t i = 0; i < gate.size(); ++i) {
        const float expected = naive_silu(gate[i]) * up[i];
        EXPECT_NEAR(out[i], expected, 1e-6f) << "i=" << i;
    }
}

// ---------------------------------------------------------------------------
// sample_topk_topp_temp — net-new capability. `uniform_draw` is an explicit
// parameter (not an owned RNG), so these tests can pin exact cumulative-
// probability boundaries deterministically instead of relying on
// statistical sampling.
// ---------------------------------------------------------------------------

TEST(BoltCompute, SampleGreedyTopK1AlwaysReturnsArgmax) {
    const float logits[5] = {1.0f, 5.0f, 3.0f, -2.0f, 4.0f};
    bolt::compute::SampleParams params;
    params.top_k = 1;
    // Regardless of the uniform draw, top_k=1 collapses to a single
    // candidate -- the argmax (index 1).
    for (float u : {0.0f, 0.3f, 0.999999f}) {
        EXPECT_EQ(bolt::compute::sample_topk_topp_temp(Device::CPU, logits, 5, params, u), 1);
    }
}

TEST(BoltCompute, SampleTopKRestrictsCandidateSetToKLargest) {
    const float logits[6] = {1.0f, 5.0f, 3.0f, -2.0f, 4.0f, 0.0f};
    bolt::compute::SampleParams params;
    params.top_k = 3;  // candidates: index 1 (5.0), 4 (4.0), 2 (3.0).
    for (float u = 0.0f; u < 1.0f; u += 0.05f) {
        const int32_t idx = bolt::compute::sample_topk_topp_temp(Device::CPU, logits, 6, params, u);
        EXPECT_TRUE(idx == 1 || idx == 4 || idx == 2) << "idx=" << idx << " u=" << u;
    }
}

TEST(BoltCompute, SampleTopPNucleusExactBoundary) {
    // Construct logits whose post-softmax probabilities are exactly
    // known: use log(p) as the logit so softmax reproduces p exactly
    // (softmax is invariant to a constant shift, and these already sum
    // to 1, so softmax(log(p)) == p to floating-point precision).
    // p = [0.5, 0.3, 0.2] for indices [0,1,2] (already descending).
    const float logits[3] = {std::log(0.5f), std::log(0.3f), std::log(0.2f)};
    bolt::compute::SampleParams params;
    params.top_p = 0.7f;  // cumulative 0.5 < 0.7 <= 0.8 (0.5+0.3) -> keep {0,1}.

    // Renormalized nucleus over {0,1}: p0'=0.5/0.8=0.625, p1'=0.3/0.8=0.375.
    // u below 0.625 -> index 0; u above -> index 1. Index 2 must never be
    // returned (it's outside the nucleus).
    EXPECT_EQ(bolt::compute::sample_topk_topp_temp(Device::CPU, logits, 3, params, 0.0f), 0);
    EXPECT_EQ(bolt::compute::sample_topk_topp_temp(Device::CPU, logits, 3, params, 0.5f), 0);
    EXPECT_EQ(bolt::compute::sample_topk_topp_temp(Device::CPU, logits, 3, params, 0.7f), 1);
    EXPECT_EQ(bolt::compute::sample_topk_topp_temp(Device::CPU, logits, 3, params, 0.999f), 1);
}

TEST(BoltCompute, SampleTopPFullMassNeverExcludesLast) {
    const float logits[3] = {std::log(0.5f), std::log(0.3f), std::log(0.2f)};
    bolt::compute::SampleParams params;
    params.top_p = 1.0f;  // disabled -- all 3 remain eligible.
    // A draw landing in the last slice must return index 2.
    EXPECT_EQ(bolt::compute::sample_topk_topp_temp(Device::CPU, logits, 3, params, 0.85f), 2);
}

TEST(BoltCompute, SampleTemperatureChangesDistributionShape) {
    // Two very different logits: with temperature=1 the top candidate
    // dominates almost all probability mass; with a large temperature the
    // distribution flattens toward uniform, shifting where a fixed uniform
    // draw lands.
    const float logits[2] = {0.0f, 10.0f};

    bolt::compute::SampleParams sharp;
    sharp.temperature = 1.0f;
    // softmax([0,10]) ~= [4.5e-5, ~1.0] -- index 1 owns virtually all mass.
    EXPECT_EQ(bolt::compute::sample_topk_topp_temp(Device::CPU, logits, 2, sharp, 0.00001f), 1);

    bolt::compute::SampleParams flat;
    flat.temperature = 1000.0f;
    // Candidates are ranked [index1, index0] (by ORIGINAL logit, descending)
    // before temperature scaling. Scaled logits become [0.01, 0.0], giving
    // softmax probabilities [0.50250 (index1), 0.49750 (index0)] -- nearly
    // uniform. Inverse-CDF walks candidates in that (index1-first) order, so
    // a draw past the ~0.5025 cumulative point is needed to land on index0 --
    // impossible in the sharp (temperature=1) case where index1 owns
    // ~0.99995 of the mass, but reachable once flattened.
    EXPECT_EQ(bolt::compute::sample_topk_topp_temp(Device::CPU, logits, 2, flat, 0.6f), 0);
    EXPECT_EQ(bolt::compute::sample_topk_topp_temp(Device::CPU, logits, 2, sharp, 0.6f), 1);
}

TEST(BoltCompute, SampleTiesBrokenByLowestIndex) {
    const float logits[4] = {2.0f, 2.0f, 2.0f, -5.0f};
    bolt::compute::SampleParams params;
    params.top_k = 1;  // forces a single winner among the tied maxima.
    EXPECT_EQ(bolt::compute::sample_topk_topp_temp(Device::CPU, logits, 4, params, 0.5f), 0);
}

// ---------------------------------------------------------------------------
// min_p (BLLM-101).
// ---------------------------------------------------------------------------

TEST(BoltCompute, SampleMinPDisabledByDefaultIsByteForByteUnaffected) {
    // min_p == 0.0f (the default) must reproduce the EXACT same result as
    // before this feature existed -- reusing SampleTopPNucleusExactBoundary's
    // own fixture/expectations verbatim.
    const float logits[3] = {std::log(0.5f), std::log(0.3f), std::log(0.2f)};
    bolt::compute::SampleParams params;
    params.top_p = 0.7f;
    EXPECT_EQ(bolt::compute::sample_topk_topp_temp(Device::CPU, logits, 3, params, 0.0f), 0);
    EXPECT_EQ(bolt::compute::sample_topk_topp_temp(Device::CPU, logits, 3, params, 0.7f), 1);
}

TEST(BoltCompute, SampleMinPExcludesLowProbabilityTail) {
    // p = [0.5, 0.2, 0.2, 0.1] (already descending, exact via log(p) trick).
    const float logits[4] = {std::log(0.5f), std::log(0.2f), std::log(0.2f), std::log(0.1f)};
    bolt::compute::SampleParams params;
    params.min_p = 0.5f;  // threshold = 0.5 * 0.5 = 0.25 -- 0.2 < 0.25, only index 0 clears it.
    // With only index 0 surviving min_p, the nucleus is a single candidate:
    // every draw must return index 0.
    for (float u : {0.0f, 0.5f, 0.999f}) {
        EXPECT_EQ(bolt::compute::sample_topk_topp_temp(Device::CPU, logits, 4, params, u), 0);
    }
}

TEST(BoltCompute, SampleMinPAlwaysKeepsAtLeastTopCandidate) {
    // An extreme min_p (1.0) would threshold out everything except a
    // perfect tie with the max -- must still keep at least index 0.
    const float logits[3] = {std::log(0.5f), std::log(0.3f), std::log(0.2f)};
    bolt::compute::SampleParams params;
    params.min_p = 1.0f;
    EXPECT_EQ(bolt::compute::sample_topk_topp_temp(Device::CPU, logits, 3, params, 0.999f), 0);
}

TEST(BoltCompute, SampleMinPComposesWithTopP) {
    // p = [0.4, 0.35, 0.15, 0.10]. min_p=0.3 -> threshold 0.12 -> survivors
    // {0,1,2} (0.15 >= 0.12, 0.10 < 0.12 excluded). top_p=0.7 over the
    // RENORMALIZED-view cumulative (still using the original probs, matching
    // the implementation's own cumulative-over-unnormalized-within-set
    // logic): cumulative 0.4, 0.75 (>=0.7) -> nucleus {0,1} only, excluding
    // index 2 even though min_p alone would have kept it.
    const float logits[4] = {std::log(0.4f), std::log(0.35f), std::log(0.15f), std::log(0.10f)};
    bolt::compute::SampleParams params;
    params.min_p = 0.3f;
    params.top_p = 0.7f;
    for (float u : {0.0f, 0.99f}) {
        const int32_t idx = bolt::compute::sample_topk_topp_temp(Device::CPU, logits, 4, params, u);
        EXPECT_TRUE(idx == 0 || idx == 1) << "idx=" << idx;
    }
}

// ---------------------------------------------------------------------------
// apply_repetition_penalties / apply_logit_bias (BLLM-101).
// ---------------------------------------------------------------------------

TEST(BoltCompute, RepetitionPenaltyDividesPositiveLogitOncePerDistinctId) {
    float logits[3] = {4.0f, -4.0f, 1.0f};
    const int32_t recent[3] = {0, 0, 1};  // id 0 appears twice -- must apply only once.
    bolt::compute::apply_repetition_penalties(logits, 3, recent, 3, /*repeat_penalty=*/2.0f,
                                                /*presence_penalty=*/0.0f, /*frequency_penalty=*/0.0f);
    EXPECT_FLOAT_EQ(logits[0], 2.0f);   // 4.0 / 2.0 (once, not twice -> would be 1.0)
    EXPECT_FLOAT_EQ(logits[1], -8.0f);  // negative logit: multiplied, not divided
    EXPECT_FLOAT_EQ(logits[2], 1.0f);   // untouched (not in recent)
}

TEST(BoltCompute, PresencePenaltyAppliesOncePerDistinctId) {
    float logits[2] = {5.0f, 5.0f};
    const int32_t recent[3] = {0, 0, 0};  // id 0 occurs 3x -- presence is still ONE subtraction.
    bolt::compute::apply_repetition_penalties(logits, 2, recent, 3, /*repeat_penalty=*/1.0f,
                                                /*presence_penalty=*/1.5f, /*frequency_penalty=*/0.0f);
    EXPECT_FLOAT_EQ(logits[0], 3.5f);  // 5.0 - 1.5, not 5.0 - 4.5
    EXPECT_FLOAT_EQ(logits[1], 5.0f);
}

TEST(BoltCompute, FrequencyPenaltyScalesWithOccurrenceCount) {
    float logits[2] = {10.0f, 10.0f};
    const int32_t recent[3] = {0, 0, 1};  // id 0 occurs twice, id 1 once.
    bolt::compute::apply_repetition_penalties(logits, 2, recent, 3, /*repeat_penalty=*/1.0f,
                                                /*presence_penalty=*/0.0f, /*frequency_penalty=*/1.0f);
    EXPECT_FLOAT_EQ(logits[0], 8.0f);  // 10.0 - 2*1.0
    EXPECT_FLOAT_EQ(logits[1], 9.0f);  // 10.0 - 1*1.0
}

TEST(BoltCompute, RepetitionPenaltiesAllDisabledIsNoOp) {
    float logits[2] = {3.0f, -3.0f};
    const int32_t recent[2] = {0, 1};
    bolt::compute::apply_repetition_penalties(logits, 2, recent, 2, 1.0f, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(logits[0], 3.0f);
    EXPECT_FLOAT_EQ(logits[1], -3.0f);
}

TEST(BoltCompute, RepetitionPenaltiesSkipOutOfRangeIds) {
    float logits[2] = {3.0f, 3.0f};
    const int32_t recent[2] = {5, -1};  // both out of [0, 2) -- must not crash or write OOB.
    bolt::compute::apply_repetition_penalties(logits, 2, recent, 2, 2.0f, 1.0f, 1.0f);
    EXPECT_FLOAT_EQ(logits[0], 3.0f);
    EXPECT_FLOAT_EQ(logits[1], 3.0f);
}

TEST(BoltCompute, LogitBiasAddsToTargetIdsOnly) {
    float logits[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const int32_t ids[2] = {1, 3};
    const float vals[2] = {5.0f, -100.0f};
    bolt::compute::apply_logit_bias(logits, 4, ids, vals, 2);
    EXPECT_FLOAT_EQ(logits[0], 1.0f);
    EXPECT_FLOAT_EQ(logits[1], 6.0f);
    EXPECT_FLOAT_EQ(logits[2], 1.0f);
    EXPECT_FLOAT_EQ(logits[3], -99.0f);
}

TEST(BoltCompute, LogitBiasSkipsOutOfRangeIds) {
    float logits[2] = {1.0f, 1.0f};
    const int32_t ids[2] = {2, -1};
    const float vals[2] = {100.0f, 100.0f};
    bolt::compute::apply_logit_bias(logits, 2, ids, vals, 2);
    EXPECT_FLOAT_EQ(logits[0], 1.0f);
    EXPECT_FLOAT_EQ(logits[1], 1.0f);
}

// ---------------------------------------------------------------------------
// Dispatch table structure — CPU resolves to a real implementation;
// Vulkan/CUDA resolve to populated (non-null), hard-failing stubs.
// ---------------------------------------------------------------------------

TEST(BoltCompute, DispatchTableCpuEntriesAreNonNull) {
    const auto& t = bolt::compute::table_for(Device::CPU);
    EXPECT_NE(t.matmul, nullptr);
    EXPECT_NE(t.rmsnorm, nullptr);
    EXPECT_NE(t.rope_interleave, nullptr);
    EXPECT_NE(t.rope_half_split, nullptr);
    EXPECT_NE(t.softmax_inplace, nullptr);
    EXPECT_NE(t.swiglu_row, nullptr);
    EXPECT_NE(t.sample_topk_topp_temp, nullptr);
}

TEST(BoltCompute, DispatchTableVulkanAndCudaSlotsArePopulatedNotSilentNoop) {
    // "Reserved but not implemented" means the slot exists (non-null
    // function pointer -- calling through the table never dereferences a
    // null pointer) but the function itself hard-fails when invoked (see
    // the death tests below), not a quiet no-op.
    for (Device d : {Device::Vulkan, Device::CUDA, Device::ROCm}) {
        const auto& t = bolt::compute::table_for(d);
        EXPECT_NE(t.matmul, nullptr);
        EXPECT_NE(t.rmsnorm, nullptr);
        EXPECT_NE(t.rope_interleave, nullptr);
        EXPECT_NE(t.rope_half_split, nullptr);
        EXPECT_NE(t.softmax_inplace, nullptr);
        EXPECT_NE(t.swiglu_row, nullptr);
        EXPECT_NE(t.sample_topk_topp_temp, nullptr);
    }
}

TEST(BoltCompute, DispatchTableIsStableAcrossCalls) {
    // Magic-static: repeated calls must return the SAME table instance
    // (same function pointers), not rebuild it.
    const auto& t1 = bolt::compute::table_for(Device::CPU);
    const auto& t2 = bolt::compute::table_for(Device::CPU);
    EXPECT_EQ(&t1, &t2);
    EXPECT_EQ(t1.matmul, t2.matmul);
}

// ---------------------------------------------------------------------------
// Vulkan/CUDA entry points must fail loudly (abort), never silently no-op
// or silently fall back to CPU semantics.
// ---------------------------------------------------------------------------

TEST(BoltComputeDeathTest, MatmulVulkanAborts) {
    std::vector<float> weight_data = {1, 2};
    Tensor weight = bolt::tensor_make_cpu_2d(DType::F32, 1, 2, weight_data.data());
    const float activation[2] = {1.0f, 1.0f};
    float out[1] = {0};
    EXPECT_DEATH(
        { bolt::compute::matmul(Device::Vulkan, weight, activation, out, 1); },
        "Vulkan");
}

TEST(BoltComputeDeathTest, RmsnormCudaAborts) {
    const float x[2] = {1.0f, 2.0f};
    const float w[2] = {1.0f, 1.0f};
    float y[2];
    EXPECT_DEATH({ bolt::compute::rmsnorm(Device::CUDA, x, w, y, 2, 1e-5f); }, "CUDA");
}

TEST(BoltComputeDeathTest, SoftmaxVulkanAborts) {
    float x[2] = {1.0f, 2.0f};
    EXPECT_DEATH({ bolt::compute::softmax_inplace(Device::Vulkan, x, 2); }, "Vulkan");
}

TEST(BoltComputeDeathTest, SampleTopkTopPTempCudaAborts) {
    const float logits[2] = {1.0f, 2.0f};
    bolt::compute::SampleParams params;
    EXPECT_DEATH(
        { bolt::compute::sample_topk_topp_temp(Device::CUDA, logits, 2, params, 0.5f); },
        "CUDA");
}

TEST(BoltComputeDeathTest, MatmulRocmAborts) {
    std::vector<float> weight_data = {1, 2};
    Tensor weight = bolt::tensor_make_cpu_2d(DType::F32, 1, 2, weight_data.data());
    const float activation[2] = {1.0f, 1.0f};
    float out[1] = {0};
    EXPECT_DEATH(
        { bolt::compute::matmul(Device::ROCm, weight, activation, out, 1); },
        "ROCm");
}

TEST(BoltComputeDeathTest, SwigluRocmAborts) {
    const float gate[2] = {1.0f, 2.0f};
    const float up[2] = {1.0f, 1.0f};
    float out[2];
    EXPECT_DEATH({ bolt::compute::swiglu_row(Device::ROCm, gate, up, out, 2); }, "ROCm");
}

}  // namespace
