// A second CoroTcpListener (same or another process) must refuse a port that
// is already bound instead of joining the holder's SO_REUSEPORT group
// (G2LAUNCH-64). POSIX only: the cross-process case forks.

#if !defined(_WIN32)

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <thread>
#include <cerrno>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include "bolt/api/net/coro_tcp_listener.h"

using bolt::api::net::CoroTcpListener;
using bolt::api::net::CoroTcpListenerConfig;
using bolt::api::net::IODispatcher;

namespace {

bolt::api::core::coro_task<void> drop_conn(IODispatcher&, int fd) {
    close(fd);
    co_return;
}

uint16_t free_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    EXPECT_GE(fd, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = 0;
    EXPECT_EQ(bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    socklen_t len = sizeof(addr);
    EXPECT_EQ(getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);
    close(fd);
    return ntohs(addr.sin_port);
}

CoroTcpListenerConfig cfg_for(uint16_t port) {
    CoroTcpListenerConfig cfg;
    cfg.port = port;
    cfg.num_io_threads = 2;
    cfg.num_workers = 2;
    return cfg;
}

}  // namespace

// Runs first: the fork must happen before this process has any threads.
TEST(CoroListenerPortClash, AnotherProcessHoldingThePortIsRefused) {
    const uint16_t port = free_port();
    ASSERT_NE(port, 0);

    int ready[2] = {-1, -1};
    int release[2] = {-1, -1};
    ASSERT_EQ(pipe(ready), 0);
    ASSERT_EQ(pipe(release), 0);

    const pid_t child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        close(ready[0]);
        close(release[1]);
        CoroTcpListener holder(cfg_for(port), drop_conn);
        const char ok = holder.start_background() == 0 ? '1' : '0';
        (void)!write(ready[1], &ok, 1);
        char buf = 0;
        (void)!read(release[0], &buf, 1);  // parent closes -> EOF
        holder.stop();
        _exit(ok == '1' ? 0 : 1);
    }
    close(ready[1]);
    close(release[0]);

    char ok = 0;
    ASSERT_EQ(read(ready[0], &ok, 1), 1);
    ASSERT_EQ(ok, '1') << "child could not bind the free port";

    CoroTcpListener second(cfg_for(port), drop_conn);
    errno = 0;
    EXPECT_EQ(second.start_background(), -1);
    EXPECT_EQ(errno, EADDRINUSE);
    EXPECT_FALSE(second.is_running());

    close(release[1]);
    int status = 0;
    ASSERT_EQ(waitpid(child, &status, 0), child);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    close(ready[0]);
}

TEST(CoroListenerPortClash, SecondListenerInProcessIsRefusedAndPortFreesOnStop) {
    const uint16_t port = free_port();
    ASSERT_NE(port, 0);

    CoroTcpListener first(cfg_for(port), drop_conn);
    ASSERT_EQ(first.start_background(), 0);

    CoroTcpListener second(cfg_for(port), drop_conn);
    errno = 0;
    EXPECT_EQ(second.start_background(), -1);
    EXPECT_EQ(errno, EADDRINUSE);

    first.stop();
    CoroTcpListener third(cfg_for(port), drop_conn);
    EXPECT_EQ(third.start_background(), 0);
    third.stop();
}

namespace {
bolt::api::core::coro_task<void> report_owner(IODispatcher& io, int fd) {
    for (unsigned request = 0; request < 2; ++request) {
        char input = 0;
        if (co_await io.async_read(fd, &input, 1) != 1) break;
        const std::size_t owner = io.current_io_thread_index();
        const char response = owner < 4 ? static_cast<char>('0' + owner) : '?';
        if (co_await io.async_write(fd, &response, 1) != 1) break;
    }
    io.async_close(fd);
    co_return;
}

int connect_owner_client(uint16_t port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
#ifdef SO_NOSIGPIPE
    const int no_sigpipe = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe)) != 0) {
        close(fd);
        return -1;
    }
#endif
    timeval timeout{2, 0};
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0 ||
        connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

void check_natural_owners(std::size_t reactors) {
    const uint16_t port = free_port();
    ASSERT_NE(port, 0);
    auto config = cfg_for(port);
    config.num_io_threads = reactors;
    CoroTcpListener listener(config, report_owner);
    ASSERT_EQ(listener.start_background(), 0);
    std::array<unsigned, 4> owners{};
    for (unsigned connection = 0; connection < 64; ++connection) {
        const int fd = connect_owner_client(port);
        EXPECT_GE(fd, 0);
        if (fd < 0) continue;
        char first = '?', second = '?';
        const char request = 'x';
        const bool exact = send(fd, &request, 1, 0) == 1 && recv(fd, &first, 1, 0) == 1 &&
            send(fd, &request, 1, 0) == 1 && recv(fd, &second, 1, 0) == 1;
        EXPECT_TRUE(exact);
        EXPECT_EQ(first, second) << "same connection changed reactor";
        if (exact && first >= '0' && first < '0' + static_cast<int>(reactors))
            ++owners[static_cast<std::size_t>(first - '0')];
        else ADD_FAILURE() << "invalid natural owner " << first;
        close(fd);
    }
    for (std::size_t i = 0; i < reactors; ++i) {
        EXPECT_GT(owners[i], 0u) << "natural accepts never reached reactor " << i;
#ifdef __APPLE__
        EXPECT_EQ(owners[i], 64u / reactors) << "macOS round-robin owner " << i;
#endif
    }
    for (unsigned i = 0; i < 1000 && listener.get_stats().connections_active != 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_EQ(listener.get_stats().connections_active, 0u);
    listener.stop();
}
}  // namespace

TEST(CoroListenerFanout, NaturalConnectionsReachEveryReactorAndStayPinned) {
    for (std::size_t reactors : {std::size_t{1}, std::size_t{2}, std::size_t{4}}) {
        SCOPED_TRACE(reactors);
        check_natural_owners(reactors);
    }
}

#ifdef __APPLE__
namespace {
struct FanoutGate { std::atomic<bool> entered{false}, release{false}; };
bolt::api::core::detached_task hold_fanout_reactor(FanoutGate& gate) {
    gate.entered.store(true);
    while (!gate.release.load()) gate.release.wait(false);
    co_return;
}

bolt::api::core::detached_task mark_acceptor_turn(std::atomic<bool>& passed) {
    passed.store(true);
    co_return;
}

void expect_owner_zero(uint16_t port) {
    const int fd = connect_owner_client(port);
    EXPECT_GE(fd, 0);
    if (fd < 0) return;
    for (unsigned i = 0; i < 2; ++i) {
        const char request = 'x';
        char owner = '?';
        EXPECT_EQ(send(fd, &request, 1, 0), 1);
        EXPECT_EQ(recv(fd, &owner, 1, 0), 1);
        EXPECT_EQ(owner, '0');
    }
    close(fd);
}
}  // namespace

TEST(CoroListenerFanout, FullTargetInboxClosesRefusedFdAndKeepsAcceptorResponsive) {
    FanoutGate gate;  // published captures outlive listener/dispatcher join
    const uint16_t port = free_port();
    ASSERT_NE(port, 0);
    CoroTcpListener listener(cfg_for(port), report_owner);
    ASSERT_EQ(listener.start_background(), 0);
    auto* io = listener.dispatcher();
    expect_owner_zero(port);  // sequence 0; next fresh fd targets reactor 1
    auto hold = hold_fanout_reactor(gate);
    auto handle = hold.release();
    const bool posted = io->post_to_io_thread(1, handle);
    EXPECT_TRUE(posted);
    if (!posted) handle.destroy();
    for (unsigned i = 0; i < 1000 && !gate.entered.load(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_TRUE(gate.entered.load());
    for (size_t i = 0; i < 256; ++i)
        EXPECT_TRUE(io->try_post_to_io_thread(1, std::noop_coroutine()));
    const int refused = connect_owner_client(port);
    EXPECT_GE(refused, 0);
    if (refused >= 0) {
        char byte = 0;
        const ssize_t n = recv(refused, &byte, 1, 0);
        const int error = errno;
        EXPECT_TRUE(n == 0 || (n < 0 && error == ECONNRESET))
            << "refused fd must close, not remain queued until target progress";
        close(refused);
    }
    expect_owner_zero(port);  // sequence 2 progresses while target 1 is held
    EXPECT_GE(io->inbox_posts_rejected(), 1u);
    gate.release.store(true);  // no fatal assertions bypass release/join
    gate.release.notify_all();
    for (unsigned i = 0; i < 1000 && listener.get_stats().connections_active != 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_EQ(listener.get_stats().connections_active, 0u);
    listener.stop();
}
TEST(CoroListenerFanout, StopClosesQueuedFdBeforeStartingClientIo) {
    FanoutGate gate;
    std::atomic<bool> passed{false};  // captures outlive listener and I/O join
    const uint16_t port = free_port();
    ASSERT_NE(port, 0);
    CoroTcpListener listener(cfg_for(port), report_owner);
    ASSERT_EQ(listener.start_background(), 0);
    auto* io = listener.dispatcher();
    expect_owner_zero(port);
    for (unsigned i = 0; i < 1000 && listener.get_stats().connections_active != 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_EQ(listener.get_stats().connections_active, 0u);
    auto hold = hold_fanout_reactor(gate);
    auto hold_handle = hold.release();
    const bool held = io->post_to_io_thread(1, hold_handle);
    EXPECT_TRUE(held);
    if (!held) hold_handle.destroy();
    for (unsigned i = 0; i < 1000 && !gate.entered.load(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_TRUE(gate.entered.load());
    const int fd = connect_owner_client(port);  // sequence 1 queues on held reactor
    EXPECT_GE(fd, 0);
    for (unsigned i = 0; i < 1000 && listener.get_stats().connections_active != 1; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_EQ(listener.get_stats().connections_active, 1u);
    auto barrier = mark_acceptor_turn(passed);
    auto barrier_handle = barrier.release();
    const bool posted = io->post_to_io_thread(0, barrier_handle);
    EXPECT_TRUE(posted);
    if (!posted) barrier_handle.destroy();
    for (unsigned i = 0; i < 1000 && !passed.load(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_TRUE(passed.load());  // accept callback returned after queue publication
    EXPECT_EQ(io->inbox_posts_rejected(), 0u);
    std::thread stopper([&] { io->stop(); });
    for (unsigned i = 0; i < 1000 && io->is_running(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_FALSE(io->is_running());
    gate.release.store(true);  // always release target before joining stop
    gate.release.notify_all();
    stopper.join();
    EXPECT_EQ(listener.get_stats().connections_active, 0u);
    if (fd >= 0) {
        char byte = 0;
        const ssize_t n = recv(fd, &byte, 1, 0);
        const int error = errno;
        EXPECT_TRUE(n == 0 || (n < 0 && error == ECONNRESET));
        close(fd);
    }
    listener.stop();
}

#endif  // __APPLE__

#endif  // !_WIN32
