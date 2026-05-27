// test_bolt_sharded_mpsc.cpp — basic + stress for ShardedMPSCQueue.

#include "bolt/bolt_sharded_mpsc.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {

TEST(ShardedMPSCQueue, BasicPushPopSingleShard) {
    bolt::ShardedMPSCQueue<int, 4, 16> q;
    EXPECT_TRUE(q.empty());

    ASSERT_TRUE(q.try_push(1, /*shard*/ 0));
    ASSERT_TRUE(q.try_push(2, /*shard*/ 0));
    ASSERT_TRUE(q.try_push(3, /*shard*/ 0));
    EXPECT_FALSE(q.empty());

    int out = 0;
    ASSERT_TRUE(q.try_pop(&out));
    EXPECT_EQ(out, 1);
    ASSERT_TRUE(q.try_pop(&out));
    EXPECT_EQ(out, 2);
    ASSERT_TRUE(q.try_pop(&out));
    EXPECT_EQ(out, 3);
    EXPECT_FALSE(q.try_pop(&out));
    EXPECT_TRUE(q.empty());
}

TEST(ShardedMPSCQueue, PushAcrossShardsRoundRobinDrain) {
    bolt::ShardedMPSCQueue<int, 4, 16> q;
    for (int s = 0; s < 4; ++s) {
        ASSERT_TRUE(q.try_push(s * 10 + 0, s));
        ASSERT_TRUE(q.try_push(s * 10 + 1, s));
    }
    std::vector<int> drained;
    q.drain([&](int v){ drained.push_back(v); });
    EXPECT_EQ(drained.size(), 8u);

    int sum = 0;
    for (int v : drained) sum += v;
    EXPECT_EQ(sum, 0 + 1 + 10 + 11 + 20 + 21 + 30 + 31);
}

TEST(ShardedMPSCQueue, ShardFullReturnsFalse) {
    bolt::ShardedMPSCQueue<int, 1, 4> q; // 1 shard, capacity 4
    for (int i = 0; i < 4; ++i) {
        int v = i;
        ASSERT_TRUE(q.try_push(std::move(v), 0));
    }
    int extra = 999;
    EXPECT_FALSE(q.try_push(std::move(extra), 0));
}

TEST(ShardedMPSCQueue, MultiProducerStress) {
    constexpr int kProducers     = 8;
    constexpr int kItemsPerThread = 4000;
    constexpr int kExpectedTotal  = kProducers * kItemsPerThread;

    // Heap-allocate: each shard slot is cache-line padded (64 B), and
    // 16 shards x 1024 slots ≈ 1 MiB — overflows the default Windows
    // thread stack if stored as a local.
    auto q_owned = std::make_unique<bolt::ShardedMPSCQueue<int, 16, 1024>>();
    auto& q = *q_owned;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};

    std::vector<std::thread> producers;
    producers.reserve(kProducers);
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p]{
            ready.fetch_add(1, std::memory_order_relaxed);
            while (!go.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            for (int i = 0; i < kItemsPerThread; ++i) {
                while (!q.try_push_local(p * kItemsPerThread + i)) {
                    std::this_thread::yield();
                }
            }
        });
    }

    while (ready.load(std::memory_order_acquire) < kProducers) {
        std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);

    // Single consumer drains until expected count reached.
    std::atomic<int> consumed{0};
    std::vector<uint8_t> seen(kExpectedTotal, 0);
    std::thread consumer([&]{
        int got = 0;
        int item = 0;
        while (got < kExpectedTotal) {
            if (q.try_pop(&item)) {
                ASSERT_GE(item, 0);
                ASSERT_LT(item, kExpectedTotal);
                ASSERT_EQ(seen[item], 0u) << "duplicate item " << item;
                seen[item] = 1;
                ++got;
            } else {
                std::this_thread::yield();
            }
        }
        consumed.store(got, std::memory_order_release);
    });

    for (auto& t : producers) t.join();
    consumer.join();

    EXPECT_EQ(consumed.load(), kExpectedTotal);
    for (int i = 0; i < kExpectedTotal; ++i) {
        ASSERT_EQ(seen[i], 1u) << "missing item " << i;
    }
    EXPECT_TRUE(q.empty());
}

} // namespace
