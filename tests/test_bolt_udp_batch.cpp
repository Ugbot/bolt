// test_bolt_udp_batch.cpp — batched datagram I/O (G2CHK-1271): one call moves
// a batch, truncation and held slots, GSO/GRO on Linux, multicast loopback, and
// a reactor receive loop on IODispatcher::async_readable.

#if !defined(_WIN32)

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <coroutine>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "bolt/api/core/worker_pool.h"
#include "bolt/api/net/io_dispatcher.h"
#include "bolt/net/bolt_udp_batch.h"

namespace {

namespace net = bolt::net;

struct Slots {
    static constexpr std::uint32_t kStride = 2048 + 128;  // cache-set colouring
    std::vector<std::uint8_t> mem;
    std::vector<net::UdpSlot> slots;
    net::UdpBatch batch{};
    Slots(std::uint32_t n, std::uint32_t cap) : mem(std::size_t(n) * (cap + 128)), slots(n) {
        for (std::uint32_t i = 0; i < n; ++i) slots[i] = {mem.data() + std::size_t(i) * (cap + 128), cap, 0, {}, 0, 0, 0};
        batch = {slots.data(), n, 0};
    }
};

struct Pair {
    net::UdpEndpoint rx{}, tx{};
    Pair() {
        EXPECT_TRUE(net::udp_endpoint_open(&rx, 0, 0));
        EXPECT_TRUE(net::udp_endpoint_open(&tx, 0, 0));
        std::uint32_t got = 0;
        (void)net::udp_endpoint_set_recv_buffer(&rx, 4u << 20, &got);
    }
    ~Pair() {
        net::udp_endpoint_close(&rx);
        net::udp_endpoint_close(&tx);
    }
    net::UdpAddr to_rx() const { return {net::k_udp_loopback_ipv4, rx.local_port, 0}; }
};

std::int64_t drain(const net::UdpEndpoint* ep, Slots& s, std::uint32_t want, std::uint32_t* calls) {
    std::int64_t total = 0;
    for (int i = 0; i < 2000 && total < want; ++i) {
        const std::int64_t n = net::udp_recv_batch(ep, &s.batch, s.batch.cap);
        if (n < 0) return -1;
        if (n == 0) { std::this_thread::sleep_for(std::chrono::microseconds(200)); continue; }
        total += n;
        ++*calls;
    }
    return total;
}

TEST(UdpBatch, EmptySocketReturnsZero) {
    Pair p;
    Slots s(8, 1500);
    EXPECT_EQ(net::udp_recv_batch(&p.rx, &s.batch, 8), 0);
    EXPECT_EQ(s.batch.count, 0u);
}

TEST(UdpBatch, OneCallTakesManyQueuedDatagrams) {
    Pair p;
    for (std::uint32_t i = 0; i < 40; ++i) {
        std::uint8_t b[200];
        std::memset(b, int(i), sizeof(b));
        ASSERT_EQ(net::udp_send_to(&p.tx, p.to_rx(), b, 20 + i), std::int64_t(20 + i));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    Slots s(64, 1500);
    std::uint32_t calls = 0;
    ASSERT_EQ(drain(&p.rx, s, 40, &calls), 40);
    EXPECT_LE(calls, 2u) << "one batch call per queue, not one per datagram";
    // Datagrams arrive in order, with sizes and sources.
    if (calls == 1) {
        for (std::uint32_t i = 0; i < 40; ++i) {
            EXPECT_EQ(s.slots[i].len, 20 + i);
            EXPECT_EQ(s.slots[i].buf[0], std::uint8_t(i));
            EXPECT_NE(s.slots[i].flags & net::kUdpSlotPeerValid, 0);
            EXPECT_EQ(s.slots[i].peer.port, p.tx.local_port);
        }
    }
}

TEST(UdpBatch, OversizeDatagramIsFlaggedTruncated) {
    Pair p;
    std::uint8_t big[900] = {};
    ASSERT_EQ(net::udp_send_to(&p.tx, p.to_rx(), big, sizeof(big)), std::int64_t(sizeof(big)));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Slots s(4, 100);
    std::uint32_t calls = 0;
    ASSERT_EQ(drain(&p.rx, s, 1, &calls), 1);
    EXPECT_NE(s.slots[0].flags & net::kUdpSlotTruncated, 0);
    EXPECT_EQ(s.slots[0].len, 0u);
}

TEST(UdpBatch, HeldSlotsAreSkipped) {
    Pair p;
    for (int i = 0; i < 3; ++i) ASSERT_EQ(net::udp_send_to(&p.tx, p.to_rx(), "abc", 3), 3);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Slots s(4, 1500);
    s.slots[0].flags = net::kUdpSlotHeld;
    s.slots[2].flags = net::kUdpSlotHeld;
    std::uint32_t calls = 0;
    ASSERT_EQ(drain(&p.rx, s, 2, &calls), 2);
    EXPECT_EQ(s.slots[1].len, 3u);
    EXPECT_EQ(s.slots[3].len, 3u);
    EXPECT_NE(s.slots[0].flags & net::kUdpSlotHeld, 0);
    EXPECT_EQ(s.slots[0].len, 0u);
}

TEST(UdpBatch, SendBatchDeliversEveryDatagram) {
    Pair p;
    Slots out(32, 1500);
    for (std::uint32_t i = 0; i < 32; ++i) {
        std::memset(out.slots[i].buf, int(i), 64);
        out.slots[i].len = 64;
        out.slots[i].peer = p.to_rx();
    }
    net::UdpBatchStats st{};
    ASSERT_EQ(net::udp_send_batch(&p.tx, &out.batch, 0, 32, true, &st), 32);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    Slots in(64, 1500);
    std::uint32_t calls = 0;
    ASSERT_EQ(drain(&p.rx, in, 32, &calls), 32);
#if defined(__linux__)
    EXPECT_GE(st.gso_runs + st.gso_disabled, 1u) << "same-peer same-size run goes out as one GSO message";
#endif
}

#if defined(__linux__)
TEST(UdpBatch, GroCoalescesAndTheSlotIteratorSplits) {
    Pair p;
    if (!net::udp_endpoint_enable_gro(&p.rx)) GTEST_SKIP() << "no UDP_GRO";
    Slots out(10, 1500);
    for (std::uint32_t i = 0; i < 10; ++i) {
        std::memset(out.slots[i].buf, int(i), 500);
        out.slots[i].len = 500;
        out.slots[i].peer = p.to_rx();
    }
    net::UdpBatchStats st{};
    ASSERT_EQ(net::udp_send_batch(&p.tx, &out.batch, 0, 10, true, &st), 10);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    Slots in(4, 65536);
    std::uint32_t datagrams = 0;
    for (int tries = 0; tries < 50 && datagrams < 10; ++tries) {
        const std::int64_t n = net::udp_recv_batch(&p.rx, &in.batch, 4);
        for (std::int64_t k = 0; k < n; ++k) {
            auto it = net::udp_slot_datagrams(in.slots[k]);
            const std::uint8_t* d = nullptr;
            std::uint32_t len = 0;
            while (it.next(&d, &len)) { EXPECT_EQ(len, 500u); EXPECT_EQ(d[0], std::uint8_t(datagrams)); ++datagrams; }
        }
        if (n == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_EQ(datagrams, 10u);
}
#endif

TEST(UdpBatch, MulticastLoopbackReachesTheBatch) {
    net::UdpEndpoint rx{}, tx{};
    const std::uint32_t group = 0xEF0101F5u;  // 239.1.1.245
    ASSERT_TRUE(net::udp_endpoint_open_reuse(&rx, net::k_udp_any_ipv4, 0, 0));
    if (!net::udp_multicast_join(&rx, group, net::k_udp_loopback_ipv4)) {
        net::udp_endpoint_close(&rx);
        GTEST_SKIP() << "multicast join on loopback refused here";
    }
    ASSERT_TRUE(net::udp_endpoint_open(&tx, 0, 0));
    ASSERT_TRUE(net::udp_multicast_set_publisher(&tx, net::k_udp_loopback_ipv4, 1));
    ASSERT_TRUE(net::udp_multicast_set_loop(&tx, true));
    for (int i = 0; i < 5; ++i) (void)net::udp_send_to(&tx, {group, rx.local_port, 0}, "mc", 2);
    Slots s(8, 1500);
    std::uint32_t calls = 0;
    const std::int64_t got = drain(&rx, s, 5, &calls);
    net::udp_endpoint_close(&rx);
    net::udp_endpoint_close(&tx);
    if (got == 0) GTEST_SKIP() << "no multicast route on loopback here";
    EXPECT_EQ(got, 5);
}

// A reactor receive loop: batch until empty, then co_await async_readable.
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

detached rx_loop(bolt::api::net::IODispatcher* io, net::UdpEndpoint* ep, Slots* s, std::atomic<std::uint64_t>* got,
                 std::atomic<std::uint64_t>* waits, std::uint64_t want) {
    while (got->load() < want) {
        const std::int64_t n = net::udp_recv_batch(ep, &s->batch, s->batch.cap);
        if (n > 0) { got->fetch_add(std::uint64_t(n)); continue; }
        if (n < 0) break;
        waits->fetch_add(1);
        if (co_await io->async_readable(static_cast<int>(ep->sock)) < 0) break;
    }
}

TEST(UdpBatch, ReactorReceiveLoopOnAsyncReadable) {
    Pair p;
    bolt::api::core::WorkerThreadPool pool(bolt::api::core::WorkerPoolConfig{.num_workers = 1, .blocking_workers = 0});
    pool.start();
    bolt::api::net::IODispatcherConfig cfg;
    cfg.num_io_threads = 1;
    cfg.worker_pool = &pool;
    cfg.pin_io_threads = false;
    bolt::api::net::IODispatcher io(cfg);
    io.start();
    Slots s(64, 1500);
    std::atomic<std::uint64_t> got{0}, waits{0};
    constexpr std::uint64_t kWant = 5000;
    ASSERT_TRUE(io.post_to_io_thread(0, rx_loop(&io, &p.rx, &s, &got, &waits, kWant).h));
    for (std::uint64_t i = 0; i < kWant; ++i) {
        (void)net::udp_send_to(&p.tx, p.to_rx(), "x", 1);
        if (i % 50 == 0) std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (got.load() < kWant && std::chrono::steady_clock::now() < end) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_EQ(got.load(), kWant);
    EXPECT_LT(waits.load(), kWant) << "readiness waits, not one per datagram";
    io.stop();
    pool.stop();
}

}  // namespace

#endif
