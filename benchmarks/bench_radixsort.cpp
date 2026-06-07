// bench_radixsort.cpp — micro-benchmark: LSD radix sort vs the argsort merge
// sort on a pure-int64 key at n = 1e6, single thread.
//
// Reports ns/row for:
//   * radix_sort_i64           (DIRECT value sort)
//   * radix_sort_indirect_i64  (INDIRECT permutation sort)
//   * sort_indirect_i64_scratch (the existing merge argsort — baseline)
// plus std::stable_sort as a sanity reference. The whole point of the radix
// kernel is to beat the comparison sort's ~6–10 ns/row floor with a linear
// ~1–2 ns/row floor; this bench shows the gap.
//
// Inputs are deterministic (fixed seed) across the full int64 range so all 8
// radix byte-passes see variation. No allocation inside the timed loop —
// scratch buffers are sized once up front (Tiger Style).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "bolt/kernels/bolt_argsort.h"
#include "bolt/kernels/bolt_radixsort.h"

using namespace bolt::kernels;
using clk = std::chrono::steady_clock;

static std::vector<int64_t> mk_keys(uint64_t seed, size_t n, int64_t lo,
                                    int64_t hi) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> d(lo, hi);
    std::vector<int64_t> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

static void run(const char* label, const std::vector<int64_t>& base) {
    const size_t N = base.size();
    const int iters = 20;
    std::vector<int64_t> keys(N), scratch(N), key_scratch(N);
    std::vector<int32_t> perm(N), perm_scratch(N);
    volatile int64_t sink = 0;
    std::printf("[%s] n = %zu, iters = %d\n", label, N, iters);

    // --- DIRECT radix -------------------------------------------------------
    auto t0 = clk::now();
    for (int it = 0; it < iters; ++it) {
        keys = base;
        radix_sort_i64(keys.data(), scratch.data(), (int64_t)N);
        sink += keys[N / 2];
    }
    auto t1 = clk::now();
    double ns_direct =
        std::chrono::duration<double, std::nano>(t1 - t0).count() /
        (double(iters) * double(N));

    // --- INDIRECT radix -----------------------------------------------------
    t0 = clk::now();
    for (int it = 0; it < iters; ++it) {
        keys = base;
        radix_sort_indirect_i64(perm.data(), keys.data(), perm_scratch.data(),
                                key_scratch.data(), (int64_t)N);
        sink += perm[N / 2];
    }
    t1 = clk::now();
    double ns_indirect =
        std::chrono::duration<double, std::nano>(t1 - t0).count() /
        (double(iters) * double(N));

    // --- merge argsort (baseline) ------------------------------------------
    t0 = clk::now();
    for (int it = 0; it < iters; ++it) {
        sort_indirect_i64_scratch(perm.data(), base.data(),
                                  perm_scratch.data(), (int64_t)N, true);
        sink += perm[N / 2];
    }
    t1 = clk::now();
    double ns_merge =
        std::chrono::duration<double, std::nano>(t1 - t0).count() /
        (double(iters) * double(N));

    // --- std::stable_sort (reference) --------------------------------------
    t0 = clk::now();
    for (int it = 0; it < iters; ++it) {
        keys = base;
        std::stable_sort(keys.begin(), keys.end());
        sink += keys[N / 2];
    }
    t1 = clk::now();
    double ns_std =
        std::chrono::duration<double, std::nano>(t1 - t0).count() /
        (double(iters) * double(N));

    (void)sink;
    std::printf("  radix_sort_i64 (direct)        : %6.2f ns/row\n", ns_direct);
    std::printf("  radix_sort_indirect_i64        : %6.2f ns/row\n", ns_indirect);
    std::printf("  sort_indirect_i64 (merge base) : %6.2f ns/row\n", ns_merge);
    std::printf("  std::stable_sort (reference)   : %6.2f ns/row\n", ns_std);
    std::printf("  radix direct speedup vs merge  : %5.2fx\n",
                ns_merge / ns_direct);
}

int main() {
    const size_t N = 1'000'000;
    // Full-range: worst case for the indirect form (no byte-pass skippable).
    run("full-range int64", mk_keys(0x5EED, N, INT64_MIN, INT64_MAX));
    // Dense ids (≤ 24-bit): the triple-store index / ORDER-BY-on-keys common
    // case. High bytes are constant ⇒ byte-skip drops to ~3 scatter passes.
    run("dense 24-bit ids", mk_keys(0x1234, N, 0, (1 << 24) - 1));
    return 0;
}
