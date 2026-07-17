// bench_compute_idot.cpp -- the PERF TRIPWIRE for bolt::compute's IDOT GEMV
// (BLLM-14). The functional tests (test_bolt_idot) assert cross-tier NUMERIC
// EQUALITY, which stays green even if the fast path silently degrades to
// scalar (a lost /arch flag, a broken CPUID probe). This benchmark is the
// only thing that catches that: it times the dispatched kernel (whatever tier
// the CPUID probe selected) against the scalar reference and, WHEN a SIMD
// tier is actually available on this machine, ASSERTS a conservative GFLOPS-
// ratio floor. A silent collapse to scalar makes the ratio ~1x and FAILS.
//
// Portable across bolt's CI lanes: on scalar-only hardware (e.g. macos-arm,
// or an x86 runner with no AVX2) there is no fast tier to compare against, so
// it reports the scalar throughput and PASSES (nothing to enforce). Returns
// non-zero only on a real regression: SIMD claimed available but not actually
// faster than scalar by the floor margin.
//
// Floor = 5x. Deliberately conservative: measured AVX-512VNNI is ~19-21x and
// AVX2-only is ~8-10x over scalar (boltllm BLLM-16), so 5x cleanly separates
// "a SIMD tier is doing real work" from "silently running scalar" (~1x)
// without being tight enough to flake on a slower/older x86 CI runner or an
// AVX2-only lane.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "bolt/bolt_tensor.h"
#include "bolt/compute_idot.h"

#include "../src/compute/idot_kernels_internal.h"

using bolt::DType;
using bolt::Tensor;
using bolt::tensor_make_cpu_2d;
namespace bc = bolt::compute;

namespace {

constexpr double kFloorRatio = 5.0;

void quantize_int8_rows(const float* w, int32_t rows, int32_t cols, int8_t* q_out,
                        float* scale_out) {
    for (int32_t r = 0; r < rows; ++r) {
        bc::quantize_act_row_int8(w + static_cast<size_t>(r) * cols, cols,
                                  q_out + static_cast<size_t>(r) * cols, &scale_out[r]);
    }
}

// Returns the dispatched/scalar GFLOPS ratio for one Int8 shape.
double bench_shape(int32_t rows, int32_t cols, int32_t iters) {
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> vdist(-2.0f, 2.0f);
    std::vector<float> w(static_cast<size_t>(rows) * cols);
    for (auto& x : w) x = vdist(rng);
    std::vector<int8_t> wq(w.size());
    std::vector<float> wscale(static_cast<size_t>(rows));
    quantize_int8_rows(w.data(), rows, cols, wq.data(), wscale.data());
    std::vector<float> act(static_cast<size_t>(cols));
    for (auto& x : act) x = vdist(rng);
    std::vector<int8_t> actq(static_cast<size_t>(cols));
    float act_scale = 0.0f;
    bc::quantize_act_row_int8(act.data(), cols, actq.data(), &act_scale);

    Tensor t = tensor_make_cpu_2d(DType::Int8, rows, cols, wq.data(), wscale.data());
    std::vector<float> out(static_cast<size_t>(rows));

    // Dispatched (fast) path.
    bc::matmul_i8a_int8(actq.data(), act_scale, t, out.data(), rows);  // warm
    auto d0 = std::chrono::steady_clock::now();
    for (int32_t it = 0; it < iters; ++it) bc::matmul_i8a_int8(actq.data(), act_scale, t, out.data(), rows);
    auto d1 = std::chrono::steady_clock::now();
    const double disp_s = std::chrono::duration<double>(d1 - d0).count();

    // Scalar reference (call the tier directly).
    bc::detail::idot_i8a_int8_scalar(actq.data(), act_scale, t, out.data(), rows);  // warm
    auto s0 = std::chrono::steady_clock::now();
    for (int32_t it = 0; it < iters; ++it)
        bc::detail::idot_i8a_int8_scalar(actq.data(), act_scale, t, out.data(), rows);
    auto s1 = std::chrono::steady_clock::now();
    const double scal_s = std::chrono::duration<double>(s1 - s0).count();

    const double flops = 2.0 * rows * cols * iters;
    const double disp_gflops = flops / disp_s / 1e9;
    const double scal_gflops = flops / scal_s / 1e9;
    const double ratio = scal_s / disp_s;
    std::printf("  rows=%d cols=%d: dispatched=%.1f GFLOPS  scalar=%.1f GFLOPS  ratio=%.2fx\n", rows,
                cols, disp_gflops, scal_gflops, ratio);
    return ratio;
}

}  // namespace

int main() {
    std::printf("bench_compute_idot: idot_avx2_available=%s idot_avx512vnni_available=%s\n",
                bc::idot_avx2_available() ? "true" : "false",
                bc::idot_avx512vnni_available() ? "true" : "false");

    // GLM expert-FFN-ish (2048x6144) + a square round number.
    const double r1 = bench_shape(2048, 6144, 40);
    const double r2 = bench_shape(4096, 4096, 40);
    const double worst = (r1 < r2) ? r1 : r2;

    if (!bc::idot_avx2_available()) {
        std::printf("bench_compute_idot: SCALAR-ONLY hardware -- no SIMD tier to enforce, PASS.\n");
        return 0;
    }
    if (worst < kFloorRatio) {
        std::fprintf(stderr,
                     "bench_compute_idot: FAIL -- a SIMD tier IS available but the dispatched "
                     "path is only %.2fx over scalar (floor %.1fx). The fast path likely silently "
                     "degraded to scalar (lost /arch flag or broken CPUID probe).\n",
                     worst, kFloorRatio);
        return 1;
    }
    std::printf("bench_compute_idot: PASS -- worst ratio %.2fx >= floor %.1fx.\n", worst, kFloorRatio);
    return 0;
}
