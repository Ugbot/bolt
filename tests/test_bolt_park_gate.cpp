// test_bolt_park_gate.cpp — park::Gate leases (G2CHK-1294): exact-target
// wakes, deadlines, capacity refusal, handoff, the lease registry and the
// stuck-waiter report.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "bolt/bolt_park_gate.h"

namespace {

namespace park = bolt::park;

park::WaitPolicy park_only(uint32_t deadline_ms = 2000) {
    park::WaitPolicy p;
    p.mode = park::WaitMode::kPark;
    p.deadline_ms = deadline_ms;
    return p;
}

TEST(ParkGate, ReadyWithoutWaitingWhenReached) {
    park::Gate g;
    g.advance(10);
    EXPECT_EQ(g.wait(5, 0, park_only()), park::GateWait::kReady);
    EXPECT_EQ(g.waiters(), 0u);
}

// An advance wakes only the waiters whose target it reached.
TEST(ParkGate, AdvanceWakesOnlySatisfiedWaiters) {
    park::Gate g;
    std::atomic<int> done[4] = {};
    std::vector<std::thread> ts;
    for (int i = 0; i < 4; ++i) {
        ts.emplace_back([&, i] {
            const auto r = g.wait(static_cast<uint64_t>(10 * (i + 1)), 0, park_only());
            done[i].store(r == park::GateWait::kReady ? 1 : 2);
        });
    }
    while (g.waiters() < 4) std::this_thread::yield();
    g.advance(20);
    for (int i = 0; i < 2; ++i) while (done[i].load() == 0) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_EQ(done[2].load(), 0);
    EXPECT_EQ(done[3].load(), 0);
    EXPECT_EQ(g.waiters(), 2u);
    EXPECT_EQ(g.woken, 2u);
    g.advance(40);
    for (auto& t : ts) t.join();
    for (int i = 0; i < 4; ++i) EXPECT_EQ(done[i].load(), 1) << i;
    EXPECT_EQ(g.woken, 4u);
}

TEST(ParkGate, DeadlineRemovesTheLease) {
    park::Gate g;
    const uint64_t start = park::now_ns();
    EXPECT_EQ(g.wait(1, start + 30u * 1000u * 1000u, park_only()), park::GateWait::kDeadline);
    EXPECT_GE(park::now_ns() - start, 29u * 1000u * 1000u);
    EXPECT_EQ(g.waiters(), 0u);
}

TEST(ParkGate, FullGateRefusesByName) {
    park::GateT<2> g;
    std::vector<std::thread> ts;
    for (int i = 0; i < 2; ++i) ts.emplace_back([&] { (void)g.wait(100, 0, park_only()); });
    while (g.waiters() < 2) std::this_thread::yield();
    EXPECT_EQ(g.wait(100, park::now_ns() + 1000000u, park_only()), park::GateWait::kExhausted);
    g.advance(100);
    for (auto& t : ts) t.join();
}

TEST(ParkGate, HandoffReachesTheWokenWaiter) {
    park::Gate g;
    uint64_t got = 0;
    std::thread t([&] { (void)g.wait(7, 0, park_only(), 3, &got); });
    while (g.waiters() < 1) std::this_thread::yield();
    g.advance(7, 0xabcdu);
    t.join();
    EXPECT_EQ(got, 0xabcdu);
}

// Registry: a parked waiter is listed with its target and site; the stuck
// report names it once.
TEST(ParkGate, RegistryListsAndReportsStuckWaiters) {
    park::Gate g;
    std::thread t([&] { (void)g.wait(55, 0, park_only(), 42); });
    while (g.waiters() < 1) std::this_thread::yield();
    int listed = 0;
    park::for_each_lease([&](const park::Lease& l, uint32_t) {
        if (l.gate == &g) { ++listed; EXPECT_EQ(l.target, 55u); EXPECT_EQ(l.site, 42u); }
    });
    EXPECT_EQ(listed, 1);
    const uint64_t far = park::now_ns() + 3600ull * 1000000000ull;
    int reported = 0;
    EXPECT_EQ(park::report_stuck(far, [&](const park::Lease&, uint32_t) { ++reported; }), 1u);
    EXPECT_EQ(park::report_stuck(far, [&](const park::Lease&, uint32_t) { ++reported; }), 0u);
    EXPECT_EQ(reported, 1);
    g.advance(55);
    t.join();
}

// Lost-wake stress through a gate: one waiter per target, the writer advances
// at random gaps; nobody may wait out the deadline.
TEST(ParkGate, NoLostWakeUnderStress) {
    park::Gate g;
    std::atomic<uint64_t> late{0};
    std::thread w([&] {
        park::WaitPolicy p;
        p.mode = park::WaitMode::kSpinThenPark;
        p.spin_ns = 2000;
        p.deadline_ms = 500;
        for (uint64_t t = 1; t <= 20000; ++t) {
            if (g.wait(t, 0, p) != park::GateWait::kReady) late.fetch_add(1);
        }
    });
    for (uint64_t v = 1; v <= 20000; ++v) {
        const uint32_t gap = static_cast<uint32_t>((v * 2654435761u) % 40u);
        if (gap > 30) std::this_thread::sleep_for(std::chrono::microseconds(gap));
        g.advance(v);
    }
    w.join();
    EXPECT_EQ(late.load(), 0u);
}

}  // namespace
