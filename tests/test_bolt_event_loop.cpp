// test_bolt_event_loop.cpp — first direct test of the bolt::reactor event
// loop (kqueue on macOS/BSD, epoll on Linux) over REAL sockets.
//
// Written alongside the G2CHK-85 registry swap (std::unordered_map →
// SwissTableGrowable-backed FdRegistry in event_loop_{kqueue,epoll}.cpp):
// nothing previously exercised these loops, so this is both the regression
// gate for the swap and the missing baseline coverage — real socketpair I/O
// dispatch, modify_fd, remove_fd semantics, handler-adds-fd-mid-dispatch
// (the case the old unordered_map could rehash under the dispatcher's
// iterator), callback self-removal/replacement lifetime, nested-poll refusal,
// and a churn stress registering/unregistering hundreds of fds.

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

struct LifetimeState {
    EventLoop* loop = nullptr;
    int fd = -1;
    int calls = 0;
    int replacement_calls = 0;
    bool active = false;
    bool destroyed_while_active = false;
    bool install_replacement = false;
};

struct RemovingHandler {
    LifetimeState* state = nullptr;

    void operator()(int fd, IOEvent, void*) const {
        // Copy everything needed before removal. A broken backend destroys
        // this callable inside remove_fd(); the remainder touches only the
        // external state and local values, never the expired closure object.
        LifetimeState* const external = state;
        EventLoop* const loop = external->loop;
        const int registered_fd = external->fd;
        const bool replace = external->install_replacement;
        external->active = true;
        char byte = 0;
        EXPECT_EQ(read(fd, &byte, 1), 1);
        EXPECT_EQ(loop->remove_fd(registered_fd), 0);
        if (replace) {
            EXPECT_EQ(loop->add_fd(
                          registered_fd, IOEvent::READ,
                          [external](int replacement_fd, IOEvent, void*) {
                              char replacement_byte = 0;
                              EXPECT_EQ(read(replacement_fd, &replacement_byte, 1), 1);
                              ++external->replacement_calls;
                          }),
                      0);
        }
        ++external->calls;
        external->active = false;
    }

    ~RemovingHandler() {
        if (state != nullptr && state->active) {
            state->destroyed_while_active = true;
        }
    }
};

void poll_until(EventLoop* loop, const int* count, int expected) {
    ASSERT_NE(loop, nullptr);
    ASSERT_NE(count, nullptr);
    for (int spin = 0; spin < 50 && *count < expected; ++spin) {
        ASSERT_GE(loop->poll(20), 0);
    }
}

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

TEST(EventLoop, SelfRemovalPinsActiveCallback) {
    auto loop = create_event_loop();
    ASSERT_TRUE(loop);
    Pair p;
    ASSERT_GE(p.a, 0);
    ASSERT_EQ(EventLoop::set_nonblocking(p.a), 0);

    LifetimeState state{};
    state.loop = loop.get();
    state.fd = p.a;
    ASSERT_EQ(loop->add_fd(p.a, IOEvent::READ, RemovingHandler{&state}), 0);
    ASSERT_EQ(write(p.b, "s", 1), 1);
    poll_until(loop.get(), &state.calls, 1);

    EXPECT_EQ(state.calls, 1);
    EXPECT_FALSE(state.active);
    EXPECT_FALSE(state.destroyed_while_active);
    EXPECT_EQ(loop->remove_fd(p.a), -1);
    EXPECT_EQ(errno, ENOENT);
}

TEST(EventLoop, SelfReplacementPinsOldCallbackAndRunsNewCallback) {
    auto loop = create_event_loop();
    ASSERT_TRUE(loop);
    Pair p;
    ASSERT_GE(p.a, 0);
    ASSERT_EQ(EventLoop::set_nonblocking(p.a), 0);

    LifetimeState state{};
    state.loop = loop.get();
    state.fd = p.a;
    state.install_replacement = true;
    ASSERT_EQ(loop->add_fd(p.a, IOEvent::READ, RemovingHandler{&state}), 0);
    ASSERT_EQ(write(p.b, "a", 1), 1);
    poll_until(loop.get(), &state.calls, 1);
    ASSERT_EQ(state.calls, 1);
    ASSERT_FALSE(state.destroyed_while_active);

    ASSERT_EQ(write(p.b, "b", 1), 1);
    poll_until(loop.get(), &state.replacement_calls, 1);
    EXPECT_EQ(state.replacement_calls, 1);
    EXPECT_FALSE(state.destroyed_while_active);
    EXPECT_EQ(loop->remove_fd(p.a), 0);
}

TEST(EventLoop, NestedPollIsRejectedBeforeSharedEventBufferReuse) {
    auto loop = create_event_loop();
    ASSERT_TRUE(loop);
    Pair p;
    ASSERT_GE(p.a, 0);
    ASSERT_EQ(EventLoop::set_nonblocking(p.a), 0);

    int calls = 0;
    int nested_result = 0;
    int nested_errno = 0;
    ASSERT_EQ(loop->add_fd(
                  p.a, IOEvent::READ,
                  [&](int fd, IOEvent, void*) {
                      char byte = 0;
                      EXPECT_EQ(read(fd, &byte, 1), 1);
                      errno = 0;
                      nested_result = loop->poll(0);
                      nested_errno = errno;
                      ++calls;
                  }),
              0);
    ASSERT_EQ(write(p.b, "n", 1), 1);
    poll_until(loop.get(), &calls, 1);

    EXPECT_EQ(calls, 1);
    EXPECT_EQ(nested_result, -1);
    EXPECT_EQ(nested_errno, EBUSY);
    EXPECT_EQ(loop->remove_fd(p.a), 0);
}

TEST(EventLoop, ReadyEntryFromOldRegistrationCannotDispatchReplacement) {
    auto loop = create_event_loop();
    ASSERT_TRUE(loop);
    Pair pairs[2];
    ASSERT_GE(pairs[0].a, 0);
    ASSERT_GE(pairs[1].a, 0);
    ASSERT_EQ(EventLoop::set_nonblocking(pairs[0].a), 0);
    ASSERT_EQ(EventLoop::set_nonblocking(pairs[1].a), 0);

    int original_calls[2] = {};
    int replacement_calls = 0;
    int replaced_index = -1;
    bool selected_first = false;
    for (int i = 0; i < 2; ++i) {
        ASSERT_EQ(loop->add_fd(
                      pairs[i].a, IOEvent::READ,
                      [&, i](int fd, IOEvent, void*) {
                          char byte = 0;
                          EXPECT_EQ(read(fd, &byte, 1), 1);
                          ++original_calls[i];
                          if (selected_first) return;
                          selected_first = true;
                          replaced_index = 1 - i;
                          const int replaced_fd = pairs[replaced_index].a;
                          EXPECT_EQ(loop->remove_fd(replaced_fd), 0);
                          EXPECT_EQ(loop->add_fd(
                                        replaced_fd, IOEvent::READ,
                                        [&](int new_fd, IOEvent, void*) {
                                            char preserved_byte = 0;
                                            EXPECT_EQ(read(new_fd, &preserved_byte, 1), 1);
                                            ++replacement_calls;
                                        }),
                                    0);
                      }),
                  0);
    }

    ASSERT_EQ(write(pairs[0].b, "0", 1), 1);
    ASSERT_EQ(write(pairs[1].b, "1", 1), 1);
    EXPECT_EQ(loop->poll(100), 2);
    ASSERT_TRUE(selected_first);
    ASSERT_GE(replaced_index, 0);
    EXPECT_EQ(original_calls[0] + original_calls[1], 1);
    EXPECT_EQ(replacement_calls, 0);

    poll_until(loop.get(), &replacement_calls, 1);
    EXPECT_EQ(replacement_calls, 1);
    EXPECT_EQ(loop->remove_fd(pairs[0].a), 0);
    EXPECT_EQ(loop->remove_fd(pairs[1].a), 0);
}

TEST(EventLoop, FixedEventBatchLeavesExcessReadinessForLaterPoll) {
    auto loop = create_event_loop();
    ASSERT_TRUE(loop);
    constexpr int kPairs = 300;
    constexpr int kMaxPolls = 10;
    Pair pairs[kPairs];
    int deliveries[kPairs] = {};

    for (int i = 0; i < kPairs; ++i) {
        if (pairs[i].a < 0 || pairs[i].b < 0) {
            GTEST_SKIP() << "process fd limit cannot create 300 socketpairs";
        }
    }
    for (int i = 0; i < kPairs; ++i) {
        ASSERT_EQ(EventLoop::set_nonblocking(pairs[i].a), 0);
        ASSERT_EQ(loop->add_fd(
                      pairs[i].a, IOEvent::READ,
                      [&, i](int fd, IOEvent, void*) {
                          char byte = 0;
                          const ssize_t read_n = read(fd, &byte, 1);
                          EXPECT_EQ(read_n, 1);
                          if (read_n == 1) ++deliveries[i];
                      }),
                  0);
        ASSERT_EQ(write(pairs[i].b, "x", 1), 1);
    }

    int delivered = 0;
    int polls = 0;
    for (; polls < kMaxPolls && delivered < kPairs; ++polls) {
        ASSERT_GE(loop->poll(20), 0);
        delivered = 0;
        for (int i = 0; i < kPairs; ++i) delivered += deliveries[i];
    }
    EXPECT_EQ(delivered, kPairs);
    EXPECT_GE(polls, 2);
    for (int i = 0; i < kPairs; ++i) {
        EXPECT_EQ(deliveries[i], 1) << "socketpair " << i;
        EXPECT_EQ(loop->remove_fd(pairs[i].a), 0);
    }
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
