// WorkerThreadPool parking (G2PERF-99): no lost wakeups, no lost parallelism,
// stop races, and an idle pool that uses no CPU.
#include "bolt/api/core/worker_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <coroutine>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

namespace {

using bolt::api::core::WorkerPoolConfig;
using bolt::api::core::WorkerThreadPool;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

constexpr auto kHang = 60s;

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

Fire count_up(std::atomic<uint64_t>* n) {
    n->fetch_add(1, std::memory_order_acq_rel);
    co_return;
}

bool wait_for(const std::atomic<uint64_t>& n, uint64_t want) {
    const auto end = Clock::now() + kHang;
    while (n.load(std::memory_order_acquire) < want) {
        if (Clock::now() > end) return false;
        std::this_thread::yield();
    }
    return true;
}

enum class Door { Coro, TryBlocking, Blocking };

void send(WorkerThreadPool& pool, Door door, std::atomic<uint64_t>* ran) {
    if (door == Door::Coro) {
        pool.submit_wait(count_up(ran).handle);
    } else if (door == Door::TryBlocking) {
        while (!pool.try_submit_blocking([ran] { ran->fetch_add(1, std::memory_order_acq_rel); }))
            std::this_thread::yield();
    } else {
        (void)pool.submit_blocking([ran] { ran->fetch_add(1, std::memory_order_acq_rel); });
    }
}

// Each submitter waits for its own burst before the next, so one lost wakeup
// with every worker parked hangs it.
void bursts(WorkerThreadPool& pool, Door door, int submitters, int rounds, int max_burst) {
    std::atomic<bool> ok{true};
    std::vector<std::thread> threads;
    for (int s = 0; s < submitters; ++s) {
        threads.emplace_back([&, s] {
            std::mt19937 rng(static_cast<uint32_t>(s * 7919 + rounds));
            std::atomic<uint64_t> ran{0};
            uint64_t sent = 0;
            for (int r = 0; r < rounds && ok.load(); ++r) {
                const int n = 1 + static_cast<int>(rng() % static_cast<uint32_t>(max_burst));
                for (int i = 0; i < n; ++i) send(pool, door, &ran);
                sent += static_cast<uint64_t>(n);
                if (!wait_for(ran, sent)) ok.store(false);
                const uint32_t gap = rng() % 8u;
                if (gap == 1) std::this_thread::sleep_for(std::chrono::microseconds(rng() % 300u));
            }
        });
    }
    for (auto& t : threads) t.join();
    EXPECT_TRUE(ok.load()) << "a submitted task never ran";
}

class Park : public ::testing::TestWithParam<std::tuple<Door, int, int>> {};

TEST_P(Park, BurstsWithIdleGapsAllRun) {
    const auto [door, workers, submitters] = GetParam();
    WorkerThreadPool pool(WorkerPoolConfig{.num_workers = static_cast<size_t>(workers),
                                           .blocking_workers = static_cast<size_t>(workers)});
    pool.start();
    bursts(pool, door, submitters, 4000, 1);
    bursts(pool, door, submitters, 300, 16);
    pool.stop();
}

INSTANTIATE_TEST_SUITE_P(Doors, Park,
                         ::testing::Combine(::testing::Values(Door::Coro, Door::TryBlocking, Door::Blocking),
                                            ::testing::Values(1, 2, 8), ::testing::Values(1, 4)));

// W tasks that only finish once all W run at the same time: a wake spent on a
// worker that found nothing must not leave an item behind a busy worker.
TEST(WorkerPoolPark, EveryWorkerIsWokenWhenThereIsWorkForIt) {
    constexpr uint32_t kW = 4;
    WorkerThreadPool pool(WorkerPoolConfig{.num_workers = 1, .blocking_workers = kW});
    pool.start();
    for (int round = 0; round < 3000; ++round) {
        std::atomic<uint32_t> here{0};
        std::atomic<uint64_t> done{0};
        auto meet = [&] {
            here.fetch_add(1);
            const auto end = Clock::now() + kHang;
            while (here.load() < kW && Clock::now() < end) std::this_thread::yield();
            done.fetch_add(here.load() >= kW ? 1 : 1000);
        };
        std::thread a([&] { for (uint32_t i = 0; i < kW / 2; ++i) (void)pool.submit_blocking(meet); });
        for (uint32_t i = 0; i < kW / 2; ++i) (void)pool.submit_blocking(meet);
        a.join();
        ASSERT_TRUE(wait_for(done, kW));
        ASSERT_EQ(done.load(), kW) << "round " << round << ": a worker stayed parked beside a queued task";
    }
    pool.stop();
}

TEST(WorkerPoolPark, ProducersParkOnAFullQueueAndAllWorkRuns) {
    WorkerThreadPool pool(WorkerPoolConfig{.num_workers = 1, .blocking_workers = 1});
    std::atomic<uint64_t> blocking{0}, coro{0};
    constexpr uint64_t kB = 6000, kC = 20000;
    std::atomic<bool> gate{false};
    pool.start();
    (void)pool.submit_blocking([&] { while (!gate.load()) std::this_thread::sleep_for(1ms); });
    std::vector<std::thread> t;
    for (int i = 0; i < 4; ++i) {
        t.emplace_back([&] { for (uint64_t k = 0; k < kB / 4; ++k) send(pool, Door::Blocking, &blocking); });
        t.emplace_back([&] { for (uint64_t k = 0; k < kC / 4; ++k) send(pool, Door::Coro, &coro); });
    }
    std::this_thread::sleep_for(50ms);
    EXPECT_EQ(blocking.load(), 0u);
    gate.store(true);
    for (auto& x : t) x.join();
    EXPECT_TRUE(wait_for(blocking, kB));
    EXPECT_TRUE(wait_for(coro, kC));
    pool.stop();
}

TEST(WorkerPoolPark, StopRacesAdmissionWithoutLosingOrRepeatingWork) {
    for (int round = 0; round < 300; ++round) {
        WorkerThreadPool pool(WorkerPoolConfig{.num_workers = 2, .blocking_workers = 1u + static_cast<size_t>(round % 3)});
        pool.start();
        std::atomic<uint64_t> accepted{0}, ran{0};
        std::atomic<bool> go{false};
        std::vector<std::thread> t;
        for (int i = 0; i < 3; ++i) {
            t.emplace_back([&] {
                while (!go.load()) std::this_thread::yield();
                for (int k = 0; k < 200; ++k)
                    if (pool.try_submit_blocking([&] { ran.fetch_add(1); })) accepted.fetch_add(1);
            });
        }
        go.store(true);
        if (round % 2) std::this_thread::sleep_for(std::chrono::microseconds(round));
        pool.stop();
        for (auto& x : t) x.join();
        ASSERT_EQ(accepted.load(), ran.load());
    }
}

TEST(WorkerPoolPark, StopWakesAParkedPoolAndItRestarts) {
    WorkerThreadPool pool(WorkerPoolConfig{.num_workers = 4, .blocking_workers = 16});
    std::atomic<uint64_t> ran{0};
    uint64_t sent = 0;
    for (int round = 0; round < 50; ++round) {
        pool.start();
        if (round % 4 == 0) std::this_thread::sleep_for(2ms);
        send(pool, Door::Blocking, &ran);
        send(pool, Door::Coro, &ran);
        sent += 2;
        ASSERT_TRUE(wait_for(ran, sent));
        const auto t0 = Clock::now();
        pool.stop();
        ASSERT_LT(Clock::now() - t0, kHang);
    }
}

TEST(WorkerPoolPark, ZeroBlockingWorkersRefusesAndStops) {
    WorkerThreadPool pool(WorkerPoolConfig{.num_workers = 2, .blocking_workers = 0});
    pool.start();
    EXPECT_FALSE(pool.try_submit_blocking([] {}));
    std::atomic<uint64_t> ran{0};
    send(pool, Door::Coro, &ran);
    EXPECT_TRUE(wait_for(ran, 1));
    pool.stop();
    EXPECT_FALSE(pool.try_submit_blocking([] {}));
}

TEST(WorkerPoolPark, StopReleasesAProducerParkedOnAFullQueue) {
    WorkerThreadPool pool(WorkerPoolConfig{.num_workers = 1, .blocking_workers = 0});
    pool.start();
    std::atomic<bool> returned{false};
    std::thread t([&] {
        for (int i = 0; i < 1100; ++i) (void)pool.submit_blocking([] {});
        returned.store(true);
    });
    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(returned.load());
    pool.stop();
    t.join();
    EXPECT_TRUE(returned.load());
}

#if !defined(_WIN32)
struct Use { double cpu_s; long switches; };

Use use_now() {
    rusage r{};
    getrusage(RUSAGE_SELF, &r);
    return {static_cast<double>(r.ru_utime.tv_sec + r.ru_stime.tv_sec) +
                static_cast<double>(r.ru_utime.tv_usec + r.ru_stime.tv_usec) * 1e-6,
            r.ru_nvcsw + r.ru_nivcsw};
}

// gestaltd's shape: 8 workers, 64 blocking workers, nothing to do. The bounds
// are per second and hold on a loaded box; sleep-polling makes ~70k switches.
TEST(WorkerPoolPark, IdlePoolUsesNoCpu) {
    WorkerThreadPool pool(WorkerPoolConfig{.num_workers = 8, .blocking_workers = 64});
    pool.start();
    std::atomic<uint64_t> ran{0};
    for (int i = 0; i < 256; ++i) send(pool, i % 2 ? Door::Blocking : Door::Coro, &ran);
    ASSERT_TRUE(wait_for(ran, 256));
    std::this_thread::sleep_for(500ms);
    const Use a = use_now();
    std::this_thread::sleep_for(2s);
    const Use b = use_now();
    pool.stop();
    RecordProperty("cpu_ms_per_s", static_cast<int>((b.cpu_s - a.cpu_s) * 500));
    RecordProperty("switches_per_s", static_cast<int>((b.switches - a.switches) / 2));
    EXPECT_LT(b.cpu_s - a.cpu_s, 0.040) << "2 s idle";
    EXPECT_LT(b.switches - a.switches, 1000) << "2 s idle";
}
#endif

}  // namespace
