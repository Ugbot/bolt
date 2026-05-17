// bench_setop.cpp — micro-benchmark for two-input sorted-merge set ops.
//
// Reports ns/row of (na + nb). Inputs are random sorted int64 in a fixed
// value range to drive realistic overlap.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "bolt/kernels/bolt_setop.h"

using namespace bolt::kernels::setop;
using clk = std::chrono::steady_clock;

static std::vector<int64_t> mk_sorted(std::mt19937_64& rng, size_t n,
                                      int64_t lo, int64_t hi) {
    std::uniform_int_distribution<int64_t> d(lo, hi);
    std::vector<int64_t> v(n);
    for (auto& x : v) x = d(rng);
    std::sort(v.begin(), v.end());
    return v;
}

template <typename Fn>
static double bench(Fn fn, const std::vector<int64_t>& a,
                    const std::vector<int64_t>& b,
                    std::vector<int64_t>& out, int iters) {
    // Warm up.
    volatile int64_t sink = 0;
    for (int i = 0; i < 2; ++i) {
        sink += fn(a.data(), (int64_t)a.size(),
                   b.data(), (int64_t)b.size(), out.data());
    }
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) {
        sink += fn(a.data(), (int64_t)a.size(),
                   b.data(), (int64_t)b.size(), out.data());
    }
    auto t1 = clk::now();
    (void)sink;
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    double rows = double(iters) * double(a.size() + b.size());
    return ns / rows;
}

int main() {
    std::mt19937_64 rng(0xDEADBEEF);
    // 1M rows per side, 50% overlap (range ~= n)
    const size_t N = 1'000'000;
    auto a = mk_sorted(rng, N, 0, (int64_t)N);
    auto b = mk_sorted(rng, N, (int64_t)N / 2, (int64_t)N + (int64_t)N / 2);
    std::vector<int64_t> out(2 * N + 1);
    const int iters = 20;

    double ns;
    ns = bench(set_union_distinct_sorted<int64_t>, a, b, out, iters);
    std::printf("set_union_distinct  %.2f ns/row (N=%zu+%zu, iters=%d)\n",
                ns, N, N, iters);
    ns = bench(set_union_all_sorted<int64_t>,      a, b, out, iters);
    std::printf("set_union_all       %.2f ns/row\n", ns);
    ns = bench(set_intersect_sorted<int64_t>,      a, b, out, iters);
    std::printf("set_intersect       %.2f ns/row\n", ns);
    ns = bench(set_intersect_all_sorted<int64_t>,  a, b, out, iters);
    std::printf("set_intersect_all   %.2f ns/row\n", ns);
    ns = bench(set_except_sorted<int64_t>,         a, b, out, iters);
    std::printf("set_except          %.2f ns/row\n", ns);
    ns = bench(set_except_all_sorted<int64_t>,     a, b, out, iters);
    std::printf("set_except_all      %.2f ns/row\n", ns);
    return 0;
}
