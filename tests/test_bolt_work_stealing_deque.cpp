// test_bolt_work_stealing_deque.cpp — basic + stress for Chase-Lev deque.

#include "bolt/bolt_work_stealing_deque.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {

struct Task { int id; };

TEST(WorkStealingDeque, OwnerPushPopLIFO) {
    bolt::WorkStealingDeque<Task, 16> dq;
    EXPECT_TRUE(dq.empty());

    Task a{1}, b{2}, c{3};
    ASSERT_TRUE(dq.push(&a));
    ASSERT_TRUE(dq.push(&b));
    ASSERT_TRUE(dq.push(&c));
    EXPECT_EQ(dq.approx_size(), 3u);

    // Owner pops from bottom — LIFO order.
    EXPECT_EQ(dq.pop(), &c);
    EXPECT_EQ(dq.pop(), &b);
    EXPECT_EQ(dq.pop(), &a);
    EXPECT_EQ(dq.pop(), nullptr);
    EXPECT_TRUE(dq.empty());
}

TEST(WorkStealingDeque, StealFromTopIsFIFO) {
    bolt::WorkStealingDeque<Task, 16> dq;
    Task a{1}, b{2}, c{3};
    ASSERT_TRUE(dq.push(&a));
    ASSERT_TRUE(dq.push(&b));
    ASSERT_TRUE(dq.push(&c));

    // Thief steals from top — FIFO order from owner's perspective.
    EXPECT_EQ(dq.steal(), &a);
    EXPECT_EQ(dq.steal(), &b);
    EXPECT_EQ(dq.steal(), &c);
    EXPECT_EQ(dq.steal(), nullptr);
}

TEST(WorkStealingDeque, FullReturnsFalse) {
    bolt::WorkStealingDeque<Task, 4> dq;
    Task tasks[4] = {{0},{1},{2},{3}};
    for (int i = 0; i < 4; ++i) ASSERT_TRUE(dq.push(&tasks[i]));
    Task extra{99};
    EXPECT_FALSE(dq.push(&extra));
}

TEST(WorkStealingDeque, ConcurrentStealStress) {
    constexpr int kCapacity = 4096;
    constexpr int kTasks    = 20000;
    constexpr int kThieves  = 4;

    bolt::WorkStealingDeque<Task, kCapacity> dq;

    std::vector<Task> backing(kTasks);
    for (int i = 0; i < kTasks; ++i) backing[i].id = i;

    std::atomic<int>   processed{0};
    std::atomic<bool>  done{false};
    std::vector<std::atomic<uint8_t>> seen(kTasks);
    for (int i = 0; i < kTasks; ++i) seen[i].store(0, std::memory_order_relaxed);
    std::atomic<int>   dup_count{0};

    auto consume = [&](Task* t) {
        if (t == nullptr) return;
        ASSERT_GE(t->id, 0);
        ASSERT_LT(t->id, kTasks);
        uint8_t prior = seen[t->id].exchange(1, std::memory_order_acq_rel);
        if (prior) dup_count.fetch_add(1, std::memory_order_relaxed);
        processed.fetch_add(1, std::memory_order_relaxed);
    };

    std::vector<std::thread> thieves;
    thieves.reserve(kThieves);
    for (int i = 0; i < kThieves; ++i) {
        thieves.emplace_back([&]{
            while (!done.load(std::memory_order_acquire) ||
                   !dq.empty()) {
                Task* t = dq.steal();
                if (t) consume(t);
                else   std::this_thread::yield();
            }
        });
    }

    // Owner: push everything, interleaving its own pops to exercise the
    // last-item race against thieves.
    std::thread owner([&]{
        int next_push = 0;
        while (next_push < kTasks) {
            // Push up to a small burst, then try a self-pop.
            for (int b = 0; b < 32 && next_push < kTasks; ++b) {
                if (dq.push(&backing[next_push])) {
                    ++next_push;
                } else {
                    break; // full; let thieves drain
                }
            }
            Task* mine = dq.pop();
            if (mine) consume(mine);
            else      std::this_thread::yield();
        }
        // Drain anything still owned.
        while (Task* mine = dq.pop()) consume(mine);
        done.store(true, std::memory_order_release);
    });

    owner.join();
    for (auto& th : thieves) th.join();

    EXPECT_EQ(processed.load(), kTasks);
    EXPECT_EQ(dup_count.load(), 0) << "Chase-Lev invariant violated: an item was returned twice";
    for (int i = 0; i < kTasks; ++i) {
        ASSERT_EQ(seen[i].load(), 1u) << "missing task " << i;
    }
}

} // namespace
