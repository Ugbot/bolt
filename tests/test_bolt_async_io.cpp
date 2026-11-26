// test_bolt_async_io.cpp — readiness backends (kqueue / epoll) of
// bolt::api::core::async_io over real socketpairs: pooled ops (no per-op heap
// allocation in steady state) and close_async completing in-flight ops with -1
// (IOCP parity; a caller draining its in-flight count must not hang).

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__linux__)

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <sys/socket.h>
#include <fcntl.h>
#include <unistd.h>

#include "bolt/api/core/async_io.h"

namespace {
std::atomic<std::uint64_t> g_alloc_count{0};
}  // namespace

void* operator new(std::size_t n) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    void* p = std::malloc(n == 0 ? 1 : n);
    if (!p) std::abort();
    return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(n == 0 ? 1 : n);
}
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }

namespace {

namespace core = bolt::api::core;

std::unique_ptr<core::async_io> make_io() {
#if defined(__linux__)
    return std::make_unique<core::epoll_io>(core::async_io_config{});
#else
    return std::make_unique<core::kqueue_io>(core::async_io_config{});
#endif
}

struct Pair {
    int a = -1;
    int b = -1;
    Pair() {
        int fds[2] = {-1, -1};
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0) {
            a = fds[0];
            b = fds[1];
        }
    }
    ~Pair() {
        if (a >= 0) close(a);
        if (b >= 0) close(b);
    }
};

// Poll until `done` or the bound is hit; returns whether `done` became true.
bool poll_until(core::async_io& io, const int& done, int want) {
    for (int i = 0; i < 2000 && done < want; ++i) io.poll(1000);
    return done >= want;
}

TEST(AsyncIo, NothrowNewUsesCountedMallocAndMatchingDeleteFamily) {
    const auto before = g_alloc_count.load(std::memory_order_relaxed);
    void* first = ::operator new(17u, std::nothrow);
    const auto allocated = g_alloc_count.load(std::memory_order_relaxed);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(allocated, before + 1u);
    ::operator delete(first); // GTest's observed nothrow-new/ordinary-delete path
    void* second = ::operator new(0u, std::nothrow);
    ASSERT_NE(second, nullptr);
    ::operator delete(second, std::nothrow);
}

TEST(AsyncIo, CloseCompletesPendingReadWithError) {
    auto io = make_io();
    Pair p;
    ASSERT_GE(p.a, 0);
    char buf[16];
    int fired = 0;
    long long result = 0;
    ASSERT_EQ(io->read_async(p.a, buf, sizeof(buf),
                             [&](const core::io_event& ev) {
                                 ++fired;
                                 result = ev.result;
                             },
                             nullptr),
              0);
    ASSERT_EQ(io->close_async(p.a), 0);
    p.a = -1;
    EXPECT_EQ(fired, 1);
    EXPECT_EQ(result, -1);
    io->poll(1000);  // a stale readiness event must not fire it again
    EXPECT_EQ(fired, 1);
}

TEST(AsyncIo, ReadAndWriteOnOneFdCompleteIndependently) {
    auto io = make_io();
    io->poll(0);  // owner thread: the register-once path keeps both directions
    Pair p;
    ASSERT_GE(p.a, 0);
    char rbuf[8] = {};
    const char wbuf[4] = {'p', 'i', 'n', 'g'};
    int reads = 0;
    int writes = 0;
    ASSERT_EQ(io->read_async(p.a, rbuf, sizeof(rbuf),
                             [&](const core::io_event&) { ++reads; }, nullptr), 0);
    {
    // Both directions on one fd, on epoll as on kqueue (G2CHK-1269); the read
    // stays armed while the write completes.
    ASSERT_EQ(io->write_async(p.a, wbuf, sizeof(wbuf),
                              [&](const core::io_event& ev) {
                                  ++writes;
                                  EXPECT_EQ(ev.result, 4);
                              },
                              nullptr),
              0);
    ASSERT_TRUE(poll_until(*io, writes, 1));
    EXPECT_EQ(reads, 0);
    }
    ASSERT_EQ(write(p.b, "x", 1), 1);
    ASSERT_TRUE(poll_until(*io, reads, 1));
    EXPECT_EQ(rbuf[0], 'x');
}


// G2CHK-1269: on the polling thread, a read with data already queued finishes
// on the submit path: result returned inline, callback never called.
TEST(AsyncIo, ReadWithQueuedDataFinishesInline) {
    auto io = make_io();
    io->poll(0);  // this thread owns the engine
    Pair p;
    ASSERT_GE(p.a, 0);
    ASSERT_EQ(write(p.b, "abc", 3), 3);
    char buf[16];
    int fired = 0;
    ssize_t got = -2;
    const int rc = io->read_async(p.a, buf, sizeof(buf),
                                  [&](const core::io_event&) { ++fired; }, nullptr, &got);
    EXPECT_EQ(rc, 1);
    EXPECT_EQ(got, 3);
    EXPECT_EQ(fired, 0);
    EXPECT_EQ(std::memcmp(buf, "abc", 3), 0);
    io->poll(0);
    EXPECT_EQ(fired, 0);
    EXPECT_GE(io->get_stats().inline_completions, 1u);
}

// Without an inline slot the same finished read is delivered by the next poll.
TEST(AsyncIo, FinishedReadWithoutInlineSlotIsDeliveredByPoll) {
    auto io = make_io();
    io->poll(0);
    Pair p;
    ASSERT_GE(p.a, 0);
    ASSERT_EQ(write(p.b, "q", 1), 1);
    char buf[8];
    int fired = 0;
    long long result = 0;
    ASSERT_EQ(io->read_async(p.a, buf, sizeof(buf),
                             [&](const core::io_event& ev) { ++fired; result = ev.result; },
                             nullptr),
              0);
    EXPECT_EQ(fired, 0);  // never from inside the submit call
    ASSERT_TRUE(poll_until(*io, fired, 1));
    EXPECT_EQ(result, 1);
}

// A socket registers with the kernel once, not once per op: 500 echo round
// trips through one fd pair cost a bounded number of registrations.
TEST(AsyncIo, SteadyStateReadsAndWritesRegisterOnce) {
    auto io = make_io();
    io->poll(0);
    Pair p;
    ASSERT_GE(p.a, 0);
    char buf[8];
    int done = 0;
    const std::uint64_t before = io->get_stats().registrations;
    for (int i = 0; i < 500; ++i) {
        const int target = done + 1;
        ASSERT_GE(io->read_async(p.a, buf, sizeof(buf),
                                 [&](const core::io_event&) { ++done; }, nullptr), 0);
        ASSERT_EQ(write(p.b, "z", 1), 1);
        ASSERT_TRUE(poll_until(*io, done, target));
    }
    const std::uint64_t regs = io->get_stats().registrations - before;
    EXPECT_LE(regs, 2u) << "kernel registration per op";
}

// Data that arrived while nothing was armed (its edge already consumed) is
// still delivered: the next read tries before it waits.
TEST(AsyncIo, ReadinessConsumedWhileIdleIsNotLost) {
    auto io = make_io();
    io->poll(0);
    Pair p;
    ASSERT_GE(p.a, 0);
    char buf[8];
    int done = 0;
    ASSERT_GE(io->read_async(p.a, buf, 1, [&](const core::io_event&) { ++done; }, nullptr), 0);
    ASSERT_EQ(write(p.b, "1", 1), 1);
    ASSERT_TRUE(poll_until(*io, done, 1));
    ASSERT_EQ(write(p.b, "2", 1), 1);
    for (int i = 0; i < 5; ++i) io->poll(1000);  // edge fires with no op waiting
    ASSERT_GE(io->read_async(p.a, buf, 1, [&](const core::io_event&) { ++done; }, nullptr), 0);
    ASSERT_TRUE(poll_until(*io, done, 2));
    EXPECT_EQ(buf[0], '2');
}

// A closed fd's number reused by a new socket is registered again, not taken
// for the old registration (would park forever).
TEST(AsyncIo, ReusedFdNumberIsRegisteredAgain) {
    auto io = make_io();
    io->poll(0);
    int first = -1;
    {
        Pair p;
        ASSERT_GE(p.a, 0);
        char buf[4];
        int done = 0;
        ASSERT_GE(io->read_async(p.a, buf, sizeof(buf), [&](const core::io_event&) { ++done; }, nullptr), 0);
        ASSERT_EQ(write(p.b, "x", 1), 1);
        ASSERT_TRUE(poll_until(*io, done, 1));
        first = p.a;
        ASSERT_EQ(io->close_async(p.a), 0);
        p.a = -1;
    }
    Pair q;
    ASSERT_GE(q.a, 0);
    const int reused = q.a == first ? q.a : (q.b == first ? q.b : -1);
    if (reused < 0) GTEST_SKIP() << "kernel did not reuse the fd number";
    const int peer = reused == q.a ? q.b : q.a;
    char buf[4];
    int done = 0;
    ASSERT_GE(io->read_async(reused, buf, sizeof(buf), [&](const core::io_event&) { ++done; }, nullptr), 0);
    ASSERT_EQ(write(peer, "y", 1), 1);
    ASSERT_TRUE(poll_until(*io, done, 1));
    EXPECT_EQ(buf[0], 'y');
}

// Callers drain a socket with raw recv after a readiness callback, relying on
// bolt having made the fd O_NONBLOCK (boltapi UdpTransport does).
TEST(AsyncIo, FdIsNonBlockingAfterAnOp) {
    auto io = make_io();
    io->poll(0);
    Pair p;
    ASSERT_GE(p.a, 0);
    ASSERT_EQ(fcntl(p.a, F_GETFL, 0) & O_NONBLOCK, 0);
    char buf[4];
    int done = 0;
    ASSERT_GE(io->read_async(p.a, buf, sizeof(buf), [&](const core::io_event&) { ++done; }, nullptr), 0);
    EXPECT_NE(fcntl(p.a, F_GETFL, 0) & O_NONBLOCK, 0);
    ASSERT_EQ(write(p.b, "k", 1), 1);
    ASSERT_TRUE(poll_until(*io, done, 1));
}

// A datagram socket with several datagrams queued is drained by successive
// reads, not one datagram per readiness edge (the edge fires once).
TEST(AsyncIo, QueuedDatagramsAllArriveAfterOneEdge) {
    auto io = make_io();
    io->poll(0);
    int fds[2] = {-1, -1};
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_DGRAM, 0, fds), 0);
    char buf[16];
    int done = 0;
    sockaddr_storage src;
    socklen_t srclen = sizeof(src);
    // Arm first (nothing queued), then queue 8 datagrams behind one edge.
    ASSERT_EQ(io->recvfrom_async(fds[0], buf, sizeof(buf), reinterpret_cast<sockaddr*>(&src),
                                 &srclen, [&](const core::io_event&) { ++done; }, nullptr), 0);
    for (int i = 0; i < 8; ++i) ASSERT_EQ(send(fds[1], "d", 1, 0), 1);
    ASSERT_TRUE(poll_until(*io, done, 1));
    for (int i = 1; i < 8; ++i) {
        ssize_t got = -2;
        srclen = sizeof(src);
        const int rc = io->recvfrom_async(fds[0], buf, sizeof(buf), reinterpret_cast<sockaddr*>(&src),
                                          &srclen, [&](const core::io_event&) { ++done; }, nullptr, &got);
        if (rc == 1) { ++done; EXPECT_EQ(got, 1); continue; }
        ASSERT_EQ(rc, 0);
        ASSERT_TRUE(poll_until(*io, done, i + 1)) << "datagram " << i << " stranded behind a consumed edge";
    }
    EXPECT_EQ(done, 8);
    close(fds[0]);
    close(fds[1]);
}

// A write that fits completes on the submit path without arming.
TEST(AsyncIo, WriteThatFitsFinishesInline) {
    auto io = make_io();
    io->poll(0);
    Pair p;
    ASSERT_GE(p.a, 0);
    ssize_t got = -2;
    int fired = 0;
    const int rc = io->write_async(p.a, "hello", 5, [&](const core::io_event&) { ++fired; },
                                   nullptr, &got);
    EXPECT_EQ(rc, 1);
    EXPECT_EQ(got, 5);
    EXPECT_EQ(fired, 0);
    char buf[8];
    ASSERT_EQ(read(p.b, buf, sizeof(buf)), 5);
}

TEST(AsyncIo, SteadyStateReadWriteDoesNotAllocate) {
    auto io = make_io();
    Pair p;
    ASSERT_GE(p.a, 0);
    char buf[8];
    int done = 0;
    auto cycle = [&](int n) {
        for (int i = 0; i < n; ++i) {
            const int target = done + 2;
            ASSERT_EQ(io->write_async(p.b, "y", 1,
                                      [&](const core::io_event&) { ++done; }, nullptr), 0);
            ASSERT_EQ(io->read_async(p.a, buf, sizeof(buf),
                                     [&](const core::io_event&) { ++done; }, nullptr), 0);
            ASSERT_TRUE(poll_until(*io, done, target));
        }
    };
    cycle(64);  // warmup
    const std::uint64_t start = g_alloc_count.load(std::memory_order_relaxed);
    cycle(2000);
    const std::uint64_t delta = g_alloc_count.load(std::memory_order_relaxed) - start;
    EXPECT_EQ(delta, 0u) << "per-op heap allocation on the async_io submit path";
}

}  // namespace

#endif
