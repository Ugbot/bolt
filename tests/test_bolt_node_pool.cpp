// test_bolt_node_pool.cpp — basic + stress for NodePool.

#include "bolt/bolt_node_pool.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {

struct Node {
    int value;
    int pad;
};

TEST(NodePool, AllocAndFreeReturnsToFreelist) {
    bolt::NodePool<Node, 8> pool;
    EXPECT_EQ(pool.in_use(), 0u);
    EXPECT_FALSE(pool.exhausted());

    Node* a = pool.allocate();
    Node* b = pool.allocate();
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_NE(a, b);
    EXPECT_EQ(pool.in_use(), 2u);

    pool.deallocate(a);
    pool.deallocate(b);
    EXPECT_EQ(pool.in_use(), 0u);
}

TEST(NodePool, ExhaustionReturnsNullptr) {
    constexpr size_t N = 4;
    bolt::NodePool<Node, N> pool;
    Node* nodes[N] = {nullptr, nullptr, nullptr, nullptr};
    for (size_t i = 0; i < N; ++i) {
        nodes[i] = pool.allocate();
        ASSERT_NE(nodes[i], nullptr);
    }
    EXPECT_TRUE(pool.exhausted());
    EXPECT_EQ(pool.allocate(), nullptr);

    pool.deallocate(nodes[0]);
    EXPECT_FALSE(pool.exhausted());
    Node* again = pool.allocate();
    ASSERT_NE(again, nullptr);
}

TEST(NodePool, DataWritesSurviveAllocFreeCycle) {
    bolt::NodePool<Node, 16> pool;
    Node* p = pool.allocate();
    ASSERT_NE(p, nullptr);
    p->value = 42;
    pool.deallocate(p);

    // Reallocate; with a single thread the freelist is LIFO so we should
    // get the same slot back.
    Node* q = pool.allocate();
    ASSERT_EQ(q, p);
    q->value = 99;
    EXPECT_EQ(q->value, 99);
    pool.deallocate(q);
}

TEST(NodePool, ConcurrentAllocFreeStress) {
    constexpr size_t kPoolSize   = 1024;
    constexpr int    kThreads    = 8;
    constexpr int    kIterations = 5000;

    bolt::NodePool<Node, kPoolSize> pool;
    std::atomic<bool> go{false};
    std::atomic<int>  ready{0};
    std::atomic<int>  alloc_fails{0};

    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&, t]{
            ready.fetch_add(1, std::memory_order_relaxed);
            while (!go.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            for (int i = 0; i < kIterations; ++i) {
                Node* n = pool.allocate();
                if (n == nullptr) {
                    alloc_fails.fetch_add(1, std::memory_order_relaxed);
                    std::this_thread::yield();
                    continue;
                }
                n->value = t * 10000 + i;
                // Brief use, then return to the pool.
                ASSERT_EQ(n->value, t * 10000 + i);
                pool.deallocate(n);
            }
        });
    }

    while (ready.load() < kThreads) std::this_thread::yield();
    go.store(true, std::memory_order_release);
    for (auto& w : workers) w.join();

    // All allocations returned: pool back to zero.
    EXPECT_EQ(pool.in_use(), 0u);
    EXPECT_FALSE(pool.exhausted());
}

} // namespace
