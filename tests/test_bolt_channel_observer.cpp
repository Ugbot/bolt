// test_bolt_channel_observer.cpp — G2CHK-200: a third thread peeking
// empty()/approx_size() while the producer and consumer run. Build with
// -fsanitize=thread to see the race this guards; without TSan it still checks
// that a concurrent approx_size() never reads a wrapped (negative) size.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>

#include "bolt/bolt_channel.h"

namespace {

constexpr std::uint64_t kItems = 200000;

TEST(BoltChannelObserver, SpscPeekFromThirdThread) {
    static bolt::SPSCChannel<std::uint64_t, 64> ch;
    std::atomic<bool> done{false};
    std::uint64_t     bad = 0;

    std::thread producer([&] {
        for (std::uint64_t i = 0; i < kItems; ++i) {
            while (!ch.try_push(std::uint64_t{i})) bolt::cpu_pause();
        }
    });
    std::thread consumer([&] {
        std::uint64_t v = 0;
        for (std::uint64_t i = 0; i < kItems; ++i) {
            while (!ch.try_pop(&v)) bolt::cpu_pause();
            ASSERT_EQ(v, i);
        }
        done.store(true, std::memory_order_release);
    });
    std::uint64_t peeks = 0;
    while (!done.load(std::memory_order_acquire) && peeks < (kItems * 64)) {
        if (ch.approx_size() > (std::uint64_t{1} << 32)) ++bad;
        (void)ch.empty();
        ++peeks;
    }
    producer.join();
    consumer.join();
    EXPECT_EQ(bad, 0u);
    EXPECT_TRUE(ch.empty());
    EXPECT_EQ(ch.approx_size(), 0u);
}

TEST(BoltChannelObserver, MpscSizeFromProducers) {
    static bolt::MPSCChannel<std::uint64_t, 64> ch;
    std::atomic<std::uint64_t> bad{0};
    constexpr std::uint64_t kPer = kItems / 4;

    auto produce = [&] {
        for (std::uint64_t i = 0; i < kPer; ++i) {
            while (!ch.try_push_nowait(std::uint64_t{i})) {
                if (ch.approx_size() > (std::uint64_t{1} << 32)) bad.fetch_add(1);
                bolt::cpu_pause();
            }
        }
    };
    std::thread p0(produce);
    std::thread p1(produce);
    std::uint64_t v = 0;
    for (std::uint64_t got = 0; got < 2 * kPer;) {
        if (ch.try_pop(&v)) ++got; else bolt::cpu_pause();
    }
    p0.join();
    p1.join();
    EXPECT_EQ(bad.load(), 0u);
    EXPECT_EQ(ch.approx_size(), 0u);
}

}  // namespace
