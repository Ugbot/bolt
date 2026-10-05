// test_bolt_disruptor.cpp — Disruptor publish order (G2CHK-656).
//
// publish() must advance `published` only in sequence order, whether the
// waiting producer spins, yields or parks. Producers here claim, write a
// payload, sometimes sleep before publishing (an off-CPU predecessor), then
// publish; a consumer checks every slot it sees as published is fully
// written and that `published` never skips a sequence.

#include <gtest/gtest.h>

#include "bolt/bolt_disruptor.h"

#include <atomic>
#include <chrono>
#include <ctime>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

struct Item {
    uint64_t seq;
    uint64_t check;
};

using Ring = bolt::Disruptor<Item, 1024>;

uint64_t mix(uint64_t s) { return s * 0x9E3779B97F4A7C15ull + 7u; }

void run(Ring* d, int producers, uint64_t per, uint32_t sleep_every) {
    d->init();
    bolt::Sequence consumed;
    const uint32_t cid = d->register_consumer(&consumed);
    ASSERT_EQ(cid, 0u);
    const uint64_t total = per * static_cast<uint64_t>(producers);
    std::atomic<uint64_t> bad{0};
    std::thread consumer([&] {
        uint64_t next = 0;
        while (next < total) {
            const uint64_t upto = d->wait_for(next);
            for (uint64_t s = next; s <= upto; ++s) {
                const Item& it = *d->slot(s);
                if (it.seq != s || it.check != mix(s)) bad.fetch_add(1);
            }
            d->mark_consumed(cid, upto);
            next = upto + 1;
        }
    });
    std::vector<std::thread> ps;
    for (int p = 0; p < producers; ++p) {
        ps.emplace_back([&, p] {
            for (uint64_t i = 0; i < per; ++i) {
                const uint64_t s = d->claim(1);
                Item* it = d->slot(s);
                it->seq = s;
                it->check = mix(s);
                if (sleep_every != 0u && (s + static_cast<uint64_t>(p)) % sleep_every == 0u)
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                d->publish(s);
            }
        });
    }
    for (auto& t : ps) t.join();
    consumer.join();
    EXPECT_EQ(bad.load(), 0u);
    EXPECT_EQ(d->published.load_acquire(), total);
}

TEST(BoltDisruptor, SoleProducerPublishesInOrder) {
    auto* d = new Ring;
    run(d, 1, 50000, 0);
    delete d;
}

TEST(BoltDisruptor, ManyProducersSpinPath) {
    auto* d = new Ring;
    run(d, 8, 20000, 0);
    delete d;
}

TEST(BoltDisruptor, ManyProducersParkOnSleepingPredecessor) {
    auto* d = new Ring;
    run(d, 8, 2000, 64);
    delete d;
}

TEST(BoltDisruptor, PublishRangeAfterSleepingPredecessor) {
    auto* d = new Ring;
    d->init();
    const uint64_t a = d->claim(1);
    const uint64_t b = d->claim(4);
    std::thread late([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        d->publish(a);
    });
    d->publish_range(b, b + 3);
    late.join();
    EXPECT_EQ(d->published.load_acquire(), 5u);
    delete d;
}

#if !defined(_WIN32)
// Producers waiting on an off-CPU predecessor park instead of spinning: four
// waiters behind a 300 ms predecessor burn well under one core-second.
// (Before G2CHK-656 each waiter spun the whole 300 ms: ~1.2 s of CPU.)
TEST(BoltDisruptor, WaitersBehindSleepingPredecessorUseNoCpu) {
    auto* d = new Ring;
    d->init();
    const uint64_t first = d->claim(1);
    std::vector<std::thread> ws;
    for (int i = 0; i < 4; ++i) {
        ws.emplace_back([d] { d->publish(d->claim(1)); });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const std::clock_t c0 = std::clock();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const double cpu_s = double(std::clock() - c0) / CLOCKS_PER_SEC;
    d->publish(first);
    for (auto& t : ws) t.join();
    EXPECT_EQ(d->published.load_acquire(), 5u);
    EXPECT_LT(cpu_s, 0.25) << "waiting producers spun";
    delete d;
}
#endif

}  // namespace
