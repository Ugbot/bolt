// test_bolt_event_loop.cpp — first direct test of the bolt::reactor event
// loop (kqueue on macOS/BSD, epoll on Linux) over REAL sockets.
//
// Written alongside the G2CHK-85 registry swap (std::unordered_map →
// SwissTableGrowable-backed FdRegistry in event_loop_{kqueue,epoll}.cpp):
// nothing previously exercised these loops, so this is both the regression
// gate for the swap and the missing baseline coverage — real socketpair I/O
// dispatch, modify_fd, remove_fd semantics, handler-adds-fd-mid-dispatch
// (the case the old unordered_map could rehash under the dispatcher's
// iterator), and a churn stress registering/unregistering hundreds of fds.

#if !defined(_WIN32)

#include <gtest/gtest.h>

#include <cerrno>
#include <cstring>
#include <sys/socket.h>
#include <unistd.h>

#include "bolt/api/net/event_loop.h"

using bolt::api::net::EventLoop;
using bolt::api::net::IOEvent;
using bolt::api::net::create_event_loop;

namespace {

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

}  // namespace

TEST(EventLoop, DispatchesReadOverRealSocket) {
    auto loop = create_event_loop();
    ASSERT_TRUE(loop);
    Pair p;
    ASSERT_GE(p.a, 0);
    ASSERT_GE(p.b, 0);
    EventLoop::set_nonblocking(p.a);

    int hits = 0;
    char buf[64] = {};
    int marker = 42;
    ASSERT_EQ(loop->add_fd(
                  p.a, IOEvent::READ,
                  [&](int fd, IOEvent ev, void* ud) {
                      EXPECT_EQ(fd, p.a);
                      EXPECT_TRUE(ev & IOEvent::READ);
                      EXPECT_EQ(*static_cast<int*>(ud), 42);
                      const ssize_t n = read(fd, buf, sizeof(buf));
                      EXPECT_EQ(n, 5);
                      ++hits;
                  },
                  &marker),
              0);

    ASSERT_EQ(write(p.b, "hello", 5), 5);
    int n = 0;
    for (int spin = 0; spin < 50 && hits == 0; ++spin) n = loop->poll(20);
    EXPECT_EQ(hits, 1);
    EXPECT_GE(n, 1);
    EXPECT_EQ(memcmp(buf, "hello", 5), 0);
    EXPECT_EQ(loop->remove_fd(p.a), 0);
}

TEST(EventLoop, RemoveFdStopsDispatchAndReportsAbsent) {
    auto loop = create_event_loop();
    ASSERT_TRUE(loop);
    Pair p;
    ASSERT_GE(p.a, 0);
    EventLoop::set_nonblocking(p.a);

    int hits = 0;
    ASSERT_EQ(loop->add_fd(p.a, IOEvent::READ,
                           [&](int, IOEvent, void*) { ++hits; }),
              0);
    ASSERT_EQ(loop->remove_fd(p.a), 0);
    EXPECT_EQ(loop->remove_fd(p.a), -1);  // double remove → ENOENT
    EXPECT_EQ(errno, ENOENT);
    EXPECT_EQ(loop->modify_fd(p.a, IOEvent::READ), -1);  // absent → ENOENT

    ASSERT_EQ(write(p.b, "x", 1), 1);
    (void)loop->poll(50);
    EXPECT_EQ(hits, 0);  // no dispatch after removal
}

// A handler that registers a NEW fd while the dispatcher is mid-poll. With
// the old std::unordered_map registry this could rehash the map underneath
// the iterator the dispatcher held; the segmented-pool registry keeps the
// dispatched entry's address stable by construction. Both fds must work.
TEST(EventLoop, HandlerAddsFdMidDispatch) {
    auto loop = create_event_loop();
    ASSERT_TRUE(loop);
    Pair p1, p2;
    ASSERT_GE(p1.a, 0);
    ASSERT_GE(p2.a, 0);
    EventLoop::set_nonblocking(p1.a);
    EventLoop::set_nonblocking(p2.a);

    int hits1 = 0, hits2 = 0;
    ASSERT_EQ(loop->add_fd(p1.a, IOEvent::READ,
                           [&](int fd, IOEvent, void*) {
                               char c;
                               (void)read(fd, &c, 1);
                               ++hits1;
                               // Register ANOTHER fd from inside dispatch.
                               EXPECT_EQ(loop->add_fd(
                                             p2.a, IOEvent::READ,
                                             [&](int fd2, IOEvent, void*) {
                                                 char d;
                                                 (void)read(fd2, &d, 1);
                                                 ++hits2;
                                             }),
                                         0);
                           }),
              0);

    ASSERT_EQ(write(p1.b, "a", 1), 1);
    for (int spin = 0; spin < 50 && hits1 == 0; ++spin) loop->poll(20);
    ASSERT_EQ(hits1, 1);

    ASSERT_EQ(write(p2.b, "b", 1), 1);
    for (int spin = 0; spin < 50 && hits2 == 0; ++spin) loop->poll(20);
    EXPECT_EQ(hits2, 1);
}

// Churn stress: register/dispatch/unregister hundreds of real fds, in waves,
// through one loop — the fd-registry workload (tombstone reuse + growth) on
// real sockets. Every byte written must be dispatched to the RIGHT handler.
TEST(EventLoop, FdChurnStressDispatchesToRightHandler) {
    auto loop = create_event_loop();
    ASSERT_TRUE(loop);

    constexpr int kWaves = 8;
    constexpr int kPairs = 64;  // per wave (bounded well under default ulimit)
    for (int wave = 0; wave < kWaves; ++wave) {
        Pair pairs[kPairs];
        int got[kPairs] = {};
        int registered = 0;
        for (int i = 0; i < kPairs; ++i) {
            ASSERT_GE(pairs[i].a, 0) << "socketpair failed (ulimit?)";
            EventLoop::set_nonblocking(pairs[i].a);
            ASSERT_EQ(loop->add_fd(
                          pairs[i].a, IOEvent::READ,
                          [&got, i](int fd, IOEvent, void*) {
                              char c;
                              const ssize_t n = read(fd, &c, 1);
                              ASSERT_EQ(n, 1);
                              got[i] = c;
                          }),
                      0)
                << "wave " << wave << " i " << i;
            ++registered;
        }
        ASSERT_EQ(registered, kPairs);

        for (int i = 0; i < kPairs; ++i) {
            const char c = static_cast<char>('A' + (i % 26));
            ASSERT_EQ(write(pairs[i].b, &c, 1), 1);
        }
        int have = 0;
        for (int spin = 0; spin < 200 && have < kPairs; ++spin) {
            (void)loop->poll(20);
            have = 0;
            for (int i = 0; i < kPairs; ++i) have += (got[i] != 0);
        }
        for (int i = 0; i < kPairs; ++i) {
            EXPECT_EQ(got[i], 'A' + (i % 26)) << "wave " << wave << " i " << i;
        }
        for (int i = 0; i < kPairs; ++i) {
            EXPECT_EQ(loop->remove_fd(pairs[i].a), 0);
        }
    }
}

#endif  // !_WIN32
