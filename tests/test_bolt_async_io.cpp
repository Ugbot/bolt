// test_bolt_async_io.cpp — readiness backends (kqueue / epoll) of
// bolt::api::core::async_io over real socketpairs: pooled ops (no per-op heap
// allocation in steady state) and close_async completing in-flight ops with -1
// (IOCP parity; a caller draining its in-flight count must not hang).

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__linux__)

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <sys/socket.h>
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
    Pair p;
    ASSERT_GE(p.a, 0);
    char rbuf[8] = {};
    const char wbuf[4] = {'p', 'i', 'n', 'g'};
    int reads = 0;
    int writes = 0;
    ASSERT_EQ(io->read_async(p.a, rbuf, sizeof(rbuf),
                             [&](const core::io_event&) { ++reads; }, nullptr), 0);
#if defined(__linux__)
    // epoll keeps one registration per fd; only kqueue arms both filters.
    (void)wbuf;
    (void)writes;
#else
    ASSERT_EQ(io->write_async(p.a, wbuf, sizeof(wbuf),
                              [&](const core::io_event& ev) {
                                  ++writes;
                                  EXPECT_EQ(ev.result, 4);
                              },
                              nullptr),
              0);
    ASSERT_TRUE(poll_until(*io, writes, 1));
    EXPECT_EQ(reads, 0);
#endif
    ASSERT_EQ(write(p.b, "x", 1), 1);
    ASSERT_TRUE(poll_until(*io, reads, 1));
    EXPECT_EQ(rbuf[0], 'x');
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
