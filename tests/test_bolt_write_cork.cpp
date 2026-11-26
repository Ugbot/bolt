// test_bolt_write_cork.cpp — IODispatcher::write_corked (G2CHK-1273): writes a
// reactor turn issues to one fd go out in one sendmsg, in order; a short write
// waits for writability without reordering; close fails queued writes.

#if !defined(_WIN32)

#include <gtest/gtest.h>

#include <atomic>
#include <coroutine>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "bolt/api/core/worker_pool.h"
#include "bolt/api/net/io_dispatcher.h"

namespace {

namespace net = bolt::api::net;
namespace core = bolt::api::core;

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

struct Reactor {
    core::WorkerThreadPool pool{core::WorkerPoolConfig{.num_workers = 1, .blocking_workers = 0}};
    std::unique_ptr<net::IODispatcher> io;
    Reactor() {
        pool.start();
        net::IODispatcherConfig cfg;
        cfg.num_io_threads = 1;
        cfg.worker_pool = &pool;
        cfg.pin_io_threads = false;
        io = std::make_unique<net::IODispatcher>(cfg);
        io->start();
    }
    ~Reactor() {
        io->stop();
        pool.stop();
    }
};

// Several coroutines resumed in one turn each cork a write to the same fd.
detached writer(net::IODispatcher* io, int fd, const std::string* msg, std::atomic<int>* ok) {
    const ssize_t n = co_await io->write_corked(fd, msg->data(), msg->size());
    if (n == static_cast<ssize_t>(msg->size())) ok->fetch_add(1);
}

detached batch_of_writers(net::IODispatcher* io, int fd, std::vector<std::string>* msgs, std::atomic<int>* ok) {
    for (auto& m : *msgs) writer(io, fd, &m, ok).h.resume();  // all corked in this turn
    co_return;
}

TEST(WriteCork, OneTurnOfWritesArrivesInOrder) {
    Reactor r;
    int sv[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
    std::vector<std::string> msgs;
    for (int i = 0; i < 20; ++i) msgs.push_back("msg" + std::to_string(i) + ";");
    std::atomic<int> ok{0};
    ASSERT_TRUE(r.io->post_to_io_thread(0, batch_of_writers(r.io.get(), sv[0], &msgs, &ok).h));
    std::string want;
    for (auto& m : msgs) want += m;
    std::string got;
    char buf[4096];
    while (got.size() < want.size()) {
        const ssize_t n = read(sv[1], buf, sizeof(buf));
        ASSERT_GT(n, 0);
        got.append(buf, static_cast<size_t>(n));
    }
    EXPECT_EQ(got, want);
    while (ok.load() < 20) std::this_thread::yield();
    close(sv[0]);
    close(sv[1]);
}

// More bytes than the socket buffer holds: the corked writers resume only once
// their bytes are out, and the stream stays in order.
TEST(WriteCork, ShortWriteWaitsWithoutReordering) {
    Reactor r;
    int sv[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
    int small = 4096;
    (void)setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &small, sizeof(small));
    std::vector<std::string> msgs;
    for (int i = 0; i < 8; ++i) msgs.push_back(std::string(64 * 1024, static_cast<char>('a' + i)));
    std::atomic<int> ok{0};
    ASSERT_TRUE(r.io->post_to_io_thread(0, batch_of_writers(r.io.get(), sv[0], &msgs, &ok).h));
    std::string got;
    std::vector<char> buf(65536);
    const size_t total = 8u * 64u * 1024u;
    while (got.size() < total) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        const ssize_t n = read(sv[1], buf.data(), buf.size());
        ASSERT_GT(n, 0);
        got.append(buf.data(), static_cast<size_t>(n));
    }
    for (int i = 0; i < 8; ++i) {
        ASSERT_EQ(got[static_cast<size_t>(i) * 65536u], static_cast<char>('a' + i)) << i;
        ASSERT_EQ(got[static_cast<size_t>(i) * 65536u + 65535u], static_cast<char>('a' + i)) << i;
    }
    while (ok.load() < 8) std::this_thread::yield();
    close(sv[0]);
    close(sv[1]);
}

TEST(WriteCork, OffAnIoThreadIsRefused) {
    Reactor r;
    int sv[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
    std::string m = "x";
    std::atomic<int> ok{0};
    writer(r.io.get(), sv[0], &m, &ok).h.resume();  // this thread is not an IO thread
    EXPECT_EQ(ok.load(), 0);
    close(sv[0]);
    close(sv[1]);
}

}  // namespace

#endif
