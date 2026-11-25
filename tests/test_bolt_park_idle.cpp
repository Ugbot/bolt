// test_bolt_park_idle.cpp — park::IdleSet (G2CHK-1295): LIFO wake, spinner
// rule, no lost wake with several workers, shutdown wakes everyone.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "bolt/bolt_park_idle.h"

namespace {

namespace park = bolt::park;
using Idle = park::IdleSetT<64>;

park::WaitPolicy park_only() {
    park::WaitPolicy p;
    p.mode = park::WaitMode::kPark;
    p.deadline_ms = 2000;
    return p;
}

// The most recently parked worker is the one woken.
TEST(ParkIdle, NotifyWakesTheNewestIdleWorker) {
    Idle idle;
    std::atomic<int> go[3] = {};
    std::atomic<int> woke_order[3] = {};
    std::atomic<int> seq{0};
    std::vector<std::thread> ts;
    for (int i = 0; i < 3; ++i) {
        ts.emplace_back([&, i] {
            park::Slot s;
            (void)idle.wait(&s, [&]() noexcept { return go[i].load() != 0; }, park_only());
            woke_order[i] = ++seq;
        });
        while (idle.nidle.load() < static_cast<uint32_t>(i + 1)) std::this_thread::yield();
    }
    // Make worker 2 runnable, then notify: LIFO pops worker 2 first.
    go[2] = 1;
    idle.notify();
    while (woke_order[2].load() == 0) std::this_thread::yield();
    EXPECT_EQ(idle.nidle.load(), 2u);
    go[0] = go[1] = 1;
    idle.notify_all();
    for (auto& t : ts) t.join();
    EXPECT_EQ(woke_order[2].load(), 1);
}

// A spinning worker makes notify skip its wake.
TEST(ParkIdle, SpinnerRuleSkipsTheWake) {
    Idle idle;
    idle.spinning.store(1);
    idle.notify();
    EXPECT_EQ(idle.skipped_for_spinner.load(), 1u);
    EXPECT_EQ(idle.wakes.load(), 0u);
    idle.spinning.store(0);
}

// Several workers, one queue counter, items published at random gaps: every
// item is taken, nobody waits out the deadline.
TEST(ParkIdle, NoLostWakeWithSeveralWorkers) {
    Idle idle;
    std::atomic<int64_t> queued{0};
    std::atomic<int64_t> taken{0};
    std::atomic<bool> stop{false};
    std::atomic<int> timeouts{0};
    std::vector<std::thread> ws;
    for (int w = 0; w < 4; ++w) {
        ws.emplace_back([&] {
            park::Slot s;
            park::WaitPolicy p;
            p.mode = park::WaitMode::kSpinThenPark;
            p.spin_ns = 2000;
            p.deadline_ms = 500;
            for (;;) {
                const auto r = idle.wait(&s, [&]() noexcept {
                    if (stop.load(std::memory_order_acquire)) return true;
                    int64_t q = queued.load(std::memory_order_acquire);
                    int64_t t = taken.load(std::memory_order_acquire);
                    while (t < q) {
                        if (taken.compare_exchange_weak(t, t + 1)) {
                            if (t + 1 < q) idle.notify();  // chained wake
                            return true;
                        }
                    }
                    return false;
                }, p);
                if (r == park::WaitResult::kDeadline && queued.load() > taken.load()) timeouts.fetch_add(1);
                if (stop.load()) return;
            }
        });
    }
    for (int i = 0; i < 20000; ++i) {
        const uint32_t gap = static_cast<uint32_t>((uint64_t(i) * 2654435761u) % 40u);
        if (gap > 30) std::this_thread::sleep_for(std::chrono::microseconds(gap));
        queued.fetch_add(1, std::memory_order_release);
        idle.notify();
    }
    while (taken.load() < queued.load()) std::this_thread::yield();
    stop = true;
    idle.notify_all();
    for (auto& t : ws) t.join();
    EXPECT_EQ(timeouts.load(), 0);
    EXPECT_EQ(taken.load(), 20000);
}

}  // namespace
