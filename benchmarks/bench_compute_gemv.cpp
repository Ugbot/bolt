// bench_compute_gemv.cpp -- the PERF TRIPWIRE for bolt::compute's f32/f16
// float-activation SIMD GEMV (perf pass). The functional test (test_bolt_gemv)
// asserts numeric agreement, which stays green even if the fast path silently
// degrades to scalar (a lost /arch flag, a broken CPUID probe). This benchmark
// is the only thing that catches that: it times the dispatched kernel (whatever
// tier the CPUID probe selected) against the scalar reference on real
// model-sized GEMV shapes and, WHEN a SIMD tier is available, ASSERTS a
// conservative GFLOPS-ratio floor. A silent collapse to scalar makes the ratio
// ~1x and FAILS.
//
// Portable across bolt's CI lanes: on scalar-only hardware there is no fast
// tier, so it reports scalar throughput and PASSES. Returns non-zero only on a
// real regression: SIMD claimed available but not faster than scalar by the
// floor margin.
//
// Floor = 2.5x. Deliberately conservative: f32 GEMV is MEMORY-bandwidth-bound
// (every weight read once), so unlike the compute-bound IDOT kernels (~8-21x)
// the win is the ~4x measured on real model shapes (67 MB attn / 235 MB FFN /
// 2 GB lm_head). 2.5x cleanly separates "a SIMD tier is doing real work" from
// "silently running scalar" (~1x) without flaking on a slower/older x86 runner.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "bolt/compute_gemv.h"

#include "../src/compute/gemv_kernels_internal.h"

namespace bc = bolt::compute;

namespace {

constexpr double kFloorRatio = 2.5;

// Returns the dispatched/scalar GFLOPS ratio for one f32 [rows, cols] GEMV.
double bench_shape(const char* label, int32_t rows, int32_t cols, int32_t iters) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> vdist(-1.0f, 1.0f);
    std::vector<float> w(static_cast<size_t>(rows) * cols);
    for (auto& x : w) x = vdist(rng);
    std::vector<float> act(static_cast<size_t>(cols));
    for (auto& x : act) x = vdist(rng);
    std::vector<float> out(static_cast<size_t>(rows));

    bc::matmul_f32(w.data(), act.data(), out.data(), rows, cols);  // warm
    auto d0 = std::chrono::steady_clock::now();
    for (int32_t it = 0; it < iters; ++it) bc::matmul_f32(w.data(), act.data(), out.data(), rows, cols);
    auto d1 = std::chrono::steady_clock::now();
    const double disp_s = std::chrono::duration<double>(d1 - d0).count();

    bc::detail::gemv_f32_scalar(w.data(), act.data(), out.data(), rows, cols);  // warm
    auto s0 = std::chrono::steady_clock::now();
    for (int32_t it = 0; it < iters; ++it)
        bc::detail::gemv_f32_scalar(w.data(), act.data(), out.data(), rows, cols);
    auto s1 = std::chrono::steady_clock::now();
    const double scal_s = std::chrono::duration<double>(s1 - s0).count();

    const double flops = 2.0 * rows * cols * iters;
    const double ratio = scal_s / disp_s;
    std::printf("  %-22s rows=%-6d cols=%-5d: dispatched=%.1f GFLOPS  scalar=%.1f GFLOPS  ratio=%.2fx\n",
                label, rows, cols, flops / disp_s / 1e9, flops / scal_s / 1e9, ratio);
    return ratio;
}

}  // namespace

int main() {
    std::printf("bench_compute_gemv: gemv_avx2_available=%s gemv_avx512_available=%s\n",
                bc::gemv_avx2_available() ? "true" : "false",
                bc::gemv_avx512_available() ? "true" : "false");

    const double r1 = bench_shape("attn-proj 4096x4096", 4096, 4096, 40);
    const double r2 = bench_shape("ffn-up 14336x4096", 14336, 4096, 20);
    const double r3 = bench_shape("lm_head 32000x4096", 32000, 4096, 8);
    double worst = r1;
    if (r2 < worst) worst = r2;
    if (r3 < worst) worst = r3;

    if (!bc::gemv_avx2_available()) {
        std::printf("bench_compute_gemv: SCALAR-ONLY hardware -- no SIMD tier to enforce, PASS.\n");
        return 0;
    }
    if (worst < kFloorRatio) {
        std::fprintf(stderr,
                     "bench_compute_gemv: FAIL -- a SIMD tier IS available but the dispatched path "
                     "is only %.2fx over scalar (floor %.1fx). The fast path likely silently "
                     "degraded to scalar (lost /arch flag or broken CPUID probe).\n",
                     worst, kFloorRatio);
        return 1;
    }
    std::printf("bench_compute_gemv: PASS -- worst ratio %.2fx >= floor %.1fx.\n", worst, kFloorRatio);
    return 0;
}
