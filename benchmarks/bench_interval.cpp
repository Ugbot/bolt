// bench_interval.cpp — microbenchmark for INTERVAL arithmetic kernels.
//
// Isolates each path: days-only add (vectorisable), months add (slow path),
// date-sub-date. Best-of-N across iterations. Reports ns/row.

#include "bolt/kernels/bolt_interval.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using Clock = std::chrono::high_resolution_clock;
using ns    = std::chrono::nanoseconds;

using bolt::kernels::interval::IntervalDate;
using bolt::kernels::interval::date32_add_interval;
using bolt::kernels::interval::date32_add_interval_scalar;
using bolt::kernels::interval::date32_sub_date32;

template <typename F>
static double best_of(int iters, int64_t n, F&& body) noexcept {
    double best = 1e30;
    for (int it = 0; it < iters; ++it) {
        auto t0 = Clock::now();
        body();
        auto t1 = Clock::now();
        double dt = std::chrono::duration_cast<ns>(t1 - t0).count();
        double per = dt / static_cast<double>(n);
        if (per < best) best = per;
    }
    return best;
}

int main() {
    constexpr int64_t N = 1 << 20;  // 1M rows
    constexpr int     ITERS = 50;

    auto* dates  = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
    auto* dates2 = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
    auto* out    = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
    auto* ivs    = static_cast<IntervalDate*>(std::malloc(sizeof(IntervalDate) * N));
    auto* iv_out = static_cast<IntervalDate*>(std::malloc(sizeof(IntervalDate) * N));

    const int32_t base = 10000;  // ~1997-05-19
    for (int64_t i = 0; i < N; ++i) {
        dates[i]  = base + static_cast<int32_t>(i % 1000);
        dates2[i] = base + static_cast<int32_t>((i + 7) % 1000);
        ivs[i]    = IntervalDate{0, static_cast<int32_t>((i % 7) - 3)};  // days-only
    }

    // 1) days-only scalar broadcast: `date - INTERVAL '90' DAY`.
    double t1 = best_of(ITERS, N, [&] {
        date32_add_interval_scalar(dates, IntervalDate{0, -90}, out, N);
    });
    std::printf("date32_add_interval_scalar (days-only)  : %.3f ns/row\n", t1);

    // 2) days-only per-row interval array.
    double t2 = best_of(ITERS, N, [&] {
        date32_add_interval(dates, ivs, out, N);
    });
    std::printf("date32_add_interval        (days-only)  : %.3f ns/row\n", t2);

    // 3) date - date.
    double t3 = best_of(ITERS, N, [&] {
        date32_sub_date32(dates, dates2, iv_out, N);
    });
    std::printf("date32_sub_date32                       : %.3f ns/row\n", t3);

    // 4) months scalar broadcast — slow path.
    double t4 = best_of(ITERS, N, [&] {
        date32_add_interval_scalar(dates, IntervalDate{-12, 0}, out, N);
    });
    std::printf("date32_add_interval_scalar (months path): %.3f ns/row\n", t4);

    // 5) months per-row, every row has months != 0 (slow path).
    for (int64_t i = 0; i < N; ++i) {
        ivs[i] = IntervalDate{static_cast<int32_t>((i % 25) - 12),
                              static_cast<int32_t>((i % 7))};
    }
    double t5 = best_of(ITERS, N, [&] {
        date32_add_interval(dates, ivs, out, N);
    });
    std::printf("date32_add_interval        (months path): %.3f ns/row\n", t5);

    std::free(dates); std::free(dates2); std::free(out);
    std::free(ivs);   std::free(iv_out);
    return 0;
}
