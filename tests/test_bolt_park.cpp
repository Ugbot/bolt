// test_bolt_park.cpp — bolt::park (G2CHK-1293) and the reactor park protocol
// (G2CHK-1270): no lost wakes, one kernel wake per park, deadlines, policy
// resolution, and an idle reactor that blocks instead of polling every 1 ms.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <coroutine>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

#include "bolt/bolt_park.h"
#include "test_bolt_park_stress.h"

#if !defined(_WIN32)
#include "bolt/api/core/worker_pool.h"
#include "bolt/api/net/io_dispatcher.h"
#endif

namespace {

namespace park = bolt::park;

TEST(Park, NoLostWakeSpinThenPark) {
    const auto r = park_stress::run(20000, park::WaitMode::kSpinThenPark);
    EXPECT_EQ(r.lost, 0u) << "max late " << r.max_late_ns << " ns";
}

TEST(Park, NoLostWakeParkOnly) {
    const auto r = park_stress::run(20000, park::WaitMode::kPark);
    EXPECT_EQ(r.lost, 0u) << "max late " << r.max_late_ns << " ns";
    EXPECT_GT(r.parks, 0u);
    // Only the CAS winner pays: never more kernel wakes than park cycles.
    EXPECT_LE(r.kernel_wakes, r.parks);
}

TEST(Park, ManyProducersPayOneWakePerPark) {
    park::Slot slot;
    std::atomic<uint64_t> value{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> producers;
    for (int t = 0; t < 3; ++t) {
        producers.emplace_back([&] {
            while (!stop.load(std::memory_order_relaxed)) {
                value.fetch_add(1, std::memory_order_release);
                (void)park::wake(&slot);
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        });
    }
    park::WaitPolicy p;
    p.mode = park::WaitMode::kPark;
    p.deadline_ms = 100;
    uint64_t seen = 0;
    for (int i = 0; i < 2000; ++i) {
        const uint64_t want = seen + 1;
        (void)park::wait(&slot, [&]() noexcept { return value.load(std::memory_order_acquire) >= want; }, 0, p);
        seen = value.load(std::memory_order_acquire);
    }
    stop = true;
    for (auto& t : producers) t.join();
    EXPECT_LE(slot.kernel_wakes.load(), slot.announces);
}

TEST(Park, DeadlineReturnsWithoutAPublish) {
    park::Slot slot;
    park::WaitPolicy p;
    p.mode = park::WaitMode::kSpinThenPark;
    p.spin_ns = 1000;
    const uint64_t start = park::now_ns();
    const auto res = park::wait(&slot, []() noexcept { return false; }, start + 20u * 1000u * 1000u, p);
    const uint64_t took = park::now_ns() - start;
    EXPECT_EQ(res, park::WaitResult::kDeadline);
    EXPECT_GE(took, 19u * 1000u * 1000u);
    EXPECT_LT(took, 2000u * 1000u * 1000u);
    EXPECT_EQ(slot.word.load(), park::kRunning);
}

TEST(Park, ProfileSelectsModes) {
    EXPECT_EQ(park::detail::profile_mode(park::WaitClass::kReactor, 0), park::WaitMode::kPark);
    EXPECT_EQ(park::detail::profile_mode(park::WaitClass::kReactor, 1), park::WaitMode::kSpinThenPark);
    EXPECT_EQ(park::detail::profile_mode(park::WaitClass::kReactor, 2), park::WaitMode::kBusy);
    EXPECT_EQ(park::detail::profile_mode(park::WaitClass::kPool, 2), park::WaitMode::kSpinThenPark);
    EXPECT_EQ(park::detail::profile_mode(park::WaitClass::kBackground, 2), park::WaitMode::kPark);
    const park::WaitPolicy r = park::policy(park::WaitClass::kReactor);
    EXPECT_EQ(r.spin_ns, 20000u);
    EXPECT_EQ(r.deadline_ms, 1000u);
}

TEST(Park, SpinnerCapAdmitsAndReleases) {
    ASSERT_TRUE(park::spin_enter());
    park::spin_leave();
    EXPECT_GE(park::detail::max_spinners(), 1u);
}

#if !defined(_WIN32)
namespace net = bolt::api::net;
namespace core = bolt::api::core;

struct detached {
    struct promise_type {
        detached get_return_object() noexcept { return {std::coroutine_handle<promise_type>::from_promise(*this)}; }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() noexcept { std::abort(); }
    };
    std::coroutine_handle<promise_type> h;
};

std::atomic<uint64_t> g_posted_at{0};
std::atomic<uint64_t> g_ran{0};
std::atomic<uint64_t> g_max_late{0};
detached note_run() {
    const uint64_t late = park::now_ns() - g_posted_at.load(std::memory_order_acquire);
    uint64_t prev = g_max_late.load();
    while (late > prev && !g_max_late.compare_exchange_weak(prev, late)) {}
    g_ran.fetch_add(1, std::memory_order_release);
    co_return;
}

// Every post to a parked reactor runs promptly: a lost wake would cost the
// whole park deadline (1 s).
TEST(ReactorPark, PostsToAParkedReactorAreNotLost) {
    core::WorkerThreadPool pool(core::WorkerPoolConfig{.num_workers = 1, .blocking_workers = 0});
    pool.start();
    net::IODispatcherConfig cfg;
    cfg.num_io_threads = 1;
    cfg.worker_pool = &pool;
    cfg.pin_io_threads = false;
    net::IODispatcher io(cfg);
    io.start();
    for (uint64_t i = 1; i <= 3000; ++i) {
        const uint32_t gap = static_cast<uint32_t>((i * 2654435761u) % 60u);
        std::this_thread::sleep_for(std::chrono::microseconds(gap));
        g_posted_at.store(park::now_ns(), std::memory_order_release);
        ASSERT_TRUE(io.post_to_io_thread(0, note_run().h));
        while (g_ran.load(std::memory_order_acquire) < i) std::this_thread::yield();
    }
    io.stop();
    pool.stop();
    EXPECT_LT(g_max_late.load(), 200u * 1000u * 1000u) << "a post waited out the park deadline";
}

// An idle reactor parks instead of polling every 1 ms (G2PERF-100).
TEST(ReactorPark, IdleReactorDoesNotTick) {
    core::WorkerThreadPool pool(core::WorkerPoolConfig{.num_workers = 1, .blocking_workers = 0});
    pool.start();
    net::IODispatcherConfig cfg;
    cfg.num_io_threads = 1;
    cfg.worker_pool = &pool;
    cfg.pin_io_threads = false;
    net::IODispatcher io(cfg);
    io.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const uint64_t before = io.engine_stats(0).polls;
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    const uint64_t polls = io.engine_stats(0).polls - before;
    io.stop();
    pool.stop();
    // 1 ms ticking would be ~400. Parked: the class deadline (1 s) bounds it to a few.
    EXPECT_LT(polls, 20u);
}
#endif

}  // namespace
