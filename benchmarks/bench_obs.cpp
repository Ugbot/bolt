// bench_obs — cost of each instrumentation call, one thread and many.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "bolt/obs/log.h"
#include "bolt/obs/metrics.h"
#include "bolt/obs/trace.h"

BOLT_LOG_MODULE("bench");

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kIters = 20000000;

template <class F>
double ns_per_op(std::uint32_t threads, std::uint64_t iters, F f) {
    std::vector<std::thread> ts;
    std::atomic<bool> go{false};
    const auto body = [&] {
        while (!go.load()) {}
        for (std::uint64_t i = 0; i < iters; ++i) f(i);
    };
    for (std::uint32_t t = 0; t < threads; ++t) ts.emplace_back(body);
    const auto t0 = Clock::now();
    go.store(true);
    for (auto& t : ts) t.join();
    return std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / static_cast<double>(iters);
}

template <class F>
void row(const char* name, std::uint64_t iters, F f) {
    const double one = ns_per_op(1, iters, f);
    const double eight = ns_per_op(8, iters, f);
    std::printf("%-34s %8.2f ns (1 thread)  %8.2f ns (8 threads, wall per op per thread)\n", name, one, eight);
}

}  // namespace

int main() {
    bolt::log::Config c;
    c.path = "/dev/null";
    c.spec = "info";
    if (!bolt::log::start(c)) return 1;
    volatile std::uint64_t sink = 0;
    row("empty loop", kIters, [&](std::uint64_t i) { sink = i; });
    row("log below level (debug at info)", kIters, [](std::uint64_t i) { BLOG_DEBUG("n=%llu", static_cast<unsigned long long>(i)); });
    {
        // Bursts inside the ring: the cost a caller pays while the writer keeps up.
        constexpr int kBurst = 128, kBursts = 2000;
        double total = 0;
        for (int b = 0; b < kBursts; ++b) {
            const auto t0 = Clock::now();
            for (int i = 0; i < kBurst; ++i) BLOG_INFO("n=%d s=%s", i, "abc");
            total += std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::printf("log enabled, into the ring        %8.2f ns\n", total / (kBurst * kBursts));
    }
    row("log enabled, ring full (dropped)", 2000000, [](std::uint64_t i) { BLOG_INFO("n=%llu s=%s", static_cast<unsigned long long>(i), "abc"); });
    const auto ctr = bolt::metrics::counter("bench_total", "x");
    row("counter.inc", kIters, [&](std::uint64_t) { ctr.inc(); });
    const auto hist = bolt::metrics::histogram("bench_seconds", "x", "", bolt::metrics::kLatency);
    row("histogram.observe", kIters, [&](std::uint64_t i) { hist.observe(i); });
    std::atomic<std::uint64_t> shared{0};
    row("shared atomic add (contrast)", kIters, [&](std::uint64_t) { shared.fetch_add(1, std::memory_order_relaxed); });
    bolt::metrics::set_enabled(false);
    row("counter.inc, run-time off", kIters, [&](std::uint64_t) { ctr.inc(); });
    row("histogram.observe, run-time off", kIters, [&](std::uint64_t i) { hist.observe(i); });
    bolt::metrics::set_enabled(true);
    row("span unsampled", kIters, [](std::uint64_t) {
        bolt::trace::Span s("x");
        asm volatile("" ::: "memory");
    });
    (void)bolt::trace::start(nullptr, nullptr);
    bolt::trace::set_rate_ppm(1000000);
    std::printf("span sampled %.1f ns (1 thread)\n", ns_per_op(1, 200000, [](std::uint64_t) {
        bolt::obs::ScopedTraceCtx ctx(bolt::obs::TraceCtx{1, 2, 0, 1});
        bolt::trace::Span s("x");
    }));
    bolt::trace::stop();
    const auto st = bolt::log::stats();
    bolt::log::stop();
    std::printf("log dropped=%llu written=%llu\n", static_cast<unsigned long long>(st.dropped), static_cast<unsigned long long>(st.records[2]));
    return 0;
}
