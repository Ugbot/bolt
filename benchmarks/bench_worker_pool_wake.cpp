// Submit-to-run latency and throughput of WorkerThreadPool, one JSON line per
// arm. A/B: build it against two include trees and interleave the runs.
//   bench_worker_pool_wake [rounds_scale [arm-door-workers]]
#include "bolt/api/core/worker_pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <coroutine>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {
using bolt::api::core::WorkerPoolConfig;
using bolt::api::core::WorkerThreadPool;
using Clock = std::chrono::steady_clock;

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

struct Fire {
    struct promise_type {
        Fire get_return_object() { return {std::coroutine_handle<promise_type>::from_promise(*this)}; }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() noexcept { std::terminate(); }
    };
    std::coroutine_handle<> handle;
};

Fire stamp(int64_t* at, std::atomic<uint64_t>* n) {
    *at = now_ns();
    n->fetch_add(1, std::memory_order_release);
    co_return;
}

void send(WorkerThreadPool& pool, bool coro, int64_t* at, std::atomic<uint64_t>* n) {
    if (coro) {
        pool.submit_wait(stamp(at, n).handle);
        return;
    }
    while (!pool.try_submit_blocking([at, n] {
        *at = now_ns();
        n->fetch_add(1, std::memory_order_release);
    })) std::this_thread::yield();
}

void report(const char* arm, bool coro, int workers, std::vector<int64_t>& lat, std::vector<int64_t>& sub, double per_s) {
    auto at = [](std::vector<int64_t>& v, double q) {
        const size_t i = std::min(v.size() - 1, static_cast<size_t>(q * static_cast<double>(v.size())));
        std::nth_element(v.begin(), v.begin() + static_cast<long>(i), v.end());
        return static_cast<double>(v[i]) / 1000.0;
    };
    std::printf("{\"arm\":\"%s\",\"door\":\"%s\",\"workers\":%d,\"n\":%zu,\"p50_us\":%.2f,\"p99_us\":%.2f,\"max_us\":%.2f,"
                "\"submit_p50_us\":%.3f,\"submit_p99_us\":%.3f,\"per_s\":%.0f}\n",
                arm, coro ? "coro" : "blocking", workers, lat.size(), at(lat, 0.5), at(lat, 0.99), at(lat, 1.0),
                at(sub, 0.5), at(sub, 0.99), per_s);
    std::fflush(stdout);
}

// gap_ns < 0: wait for the task, then idle -gap_ns (wake from idle).
// gap_ns > 0: open loop, one submit every gap_ns. 0: closed-loop flood.
void run(const char* arm, bool coro, int workers, size_t n, int64_t gap_ns) {
    WorkerThreadPool pool(WorkerPoolConfig{.num_workers = static_cast<size_t>(workers),
                                           .blocking_workers = static_cast<size_t>(workers)});
    pool.start();
    std::vector<int64_t> ran(n), due(n), sub(n);
    std::atomic<uint64_t> done{0};
    const int64_t t0 = now_ns();
    for (size_t i = 0; i < n; ++i) {
        if (gap_ns < 0) std::this_thread::sleep_for(std::chrono::nanoseconds(-gap_ns));
        if (gap_ns > 0) while (now_ns() < t0 + static_cast<int64_t>(i) * gap_ns) {}
        due[i] = now_ns();
        const int64_t s = due[i];
        send(pool, coro, &ran[i], &done);
        sub[i] = now_ns() - s;
        if (gap_ns < 0) while (done.load(std::memory_order_acquire) <= i) std::this_thread::yield();
    }
    while (done.load(std::memory_order_acquire) < n) std::this_thread::yield();
    const double per_s = static_cast<double>(n) * 1e9 / static_cast<double>(now_ns() - t0);
    pool.stop();
    std::vector<int64_t> lat(n);
    for (size_t i = 0; i < n; ++i) lat[i] = ran[i] - due[i];  // from the submit, not the schedule
    report(arm, coro, workers, lat, sub, per_s);
}
}  // namespace

int main(int argc, char** argv) {
    const size_t scale = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1u;
    const char* only = argc > 2 ? argv[2] : "";
    struct Arm { const char* name; size_t n; int64_t gap_ns; };
    const Arm arms[] = {{"idle_2ms", 500, -2'000'000}, {"steady_10k", 10'000, 100'000},
                        {"steady_100k", 50'000, 10'000}, {"flood", 300'000, 0}};
    for (const int workers : {1, 8, 64}) {
        for (const bool coro : {true, false}) {
            for (const Arm& a : arms) {
                char id[64];
                std::snprintf(id, sizeof id, "%s-%s-%d", a.name, coro ? "coro" : "blocking", workers);
                if (*only == 0 || std::strcmp(only, id) == 0) run(a.name, coro, workers, a.n * scale, a.gap_ns);
            }
        }
    }
    return 0;
}
