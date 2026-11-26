// test_bolt_reactor_timers.cpp — IODispatcher::sleep_for (G2CHK-1273): never
// early, promptly late-bounded, fired in deadline order, refused off an IO
// thread, and a reactor waiting on a timer parks instead of ticking.

#if !defined(_WIN32)

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <coroutine>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

#include "bolt/api/core/worker_pool.h"
#include "bolt/api/net/io_dispatcher.h"

namespace {

namespace net = bolt::api::net;
namespace core = bolt::api::core;
namespace park = bolt::park;

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

struct Reactor {
    core::WorkerThreadPool pool{core::WorkerPoolConfig{.num_workers = 1, .blocking_workers = 0}};
    std::unique_ptr<net::IODispatcher> io;
    Reactor() {
        pool.start();
        net::IODispatcherConfig cfg;
        cfg.num_io_threads = 1;
        cfg.worker_pool = &pool;
        cfg.pin_io_threads = false;
        io = std::make_unique<net::IODispatcher>(cfg);
        io->start();
    }
    ~Reactor() {
        io->stop();
        pool.stop();
    }
};

detached sleeper(net::IODispatcher* io, uint32_t n, uint64_t ns, std::vector<int64_t>* late, std::atomic<int>* done) {
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t deadline = park::now_ns() + ns;
        const bool fired = co_await io->sleep_until(deadline);
        late->push_back(static_cast<int64_t>(park::now_ns()) - static_cast<int64_t>(deadline));
        EXPECT_TRUE(fired);
    }
    done->store(1);
}

TEST(ReactorTimers, SleepIsNeverEarlyAndBoundedLate) {
    Reactor r;
    std::vector<int64_t> late;
    late.reserve(500);
    std::atomic<int> done{0};
    ASSERT_TRUE(r.io->post_to_io_thread(0, sleeper(r.io.get(), 500, 250000, &late, &done).h));
    while (done.load() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    ASSERT_EQ(late.size(), 500u);
    std::sort(late.begin(), late.end());
    EXPECT_GE(late.front(), 0) << "a timer fired early with zero slack";
    // The old reactor had no timer at all (1 ms poll tick); a 250 us sleep now
    // waits on a microsecond deadline. Loose bound for a loaded host.
    EXPECT_LT(late[late.size() / 2], 1000000) << "median lateness";
}

std::atomic<int> g_order_seq{0};
detached ordered(net::IODispatcher* io, uint64_t deadline, int* out) {
    (void)co_await io->sleep_until(deadline);
    *out = ++g_order_seq;
}

TEST(ReactorTimers, FireInDeadlineOrder) {
    Reactor r;
    constexpr int kN = 200;
    std::vector<int> order(kN, 0);
    std::vector<uint64_t> deadline(kN);
    const uint64_t base = park::now_ns() + 20u * 1000u * 1000u;
    for (int i = 0; i < kN; ++i) deadline[i] = base + uint64_t((i * 7919) % kN) * 100000u;
    for (int i = 0; i < kN; ++i) ASSERT_TRUE(r.io->post_to_io_thread(0, ordered(r.io.get(), deadline[i], &order[i]).h));
    while (g_order_seq.load() < kN) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    for (int a = 0; a < kN; ++a)
        for (int b = 0; b < kN; ++b)
            if (deadline[a] < deadline[b]) { ASSERT_LT(order[a], order[b]) << a << " vs " << b; }
}

detached off_thread(net::IODispatcher* io, std::atomic<int>* result) {
    const bool fired = co_await io->sleep_for(5u * 1000u * 1000u);
    result->store(fired ? 1 : 2);
}

TEST(ReactorTimers, SleepOffAnIoThreadIsRefused) {
    Reactor r;
    std::atomic<int> result{0};
    off_thread(r.io.get(), &result).h.resume();  // runs on this (non-IO) thread
    EXPECT_EQ(result.load(), 2);
}

TEST(ReactorTimers, ReactorWaitingOnATimerParks) {
    Reactor r;
    std::vector<int64_t> late;
    std::atomic<int> done{0};
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const uint64_t before = r.io->engine_stats(0).polls;
    ASSERT_TRUE(r.io->post_to_io_thread(0, sleeper(r.io.get(), 1, 300u * 1000u * 1000u, &late, &done).h));
    while (done.load() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    const uint64_t polls = r.io->engine_stats(0).polls - before;
    ASSERT_EQ(late.size(), 1u);
    EXPECT_GE(late[0], 0);
    EXPECT_LT(late[0], 50000000) << "300 ms sleep fired more than 50 ms late";
    EXPECT_LT(polls, 40u) << "the reactor ticked while waiting on its timer";
}

}  // namespace

#endif
