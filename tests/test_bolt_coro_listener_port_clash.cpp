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
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
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

bool actual_backend(CoroTcpListener& listener, const char** name) {
    *name = nullptr;
    auto* dispatcher = listener.dispatcher();
    if (dispatcher == nullptr) return false;
    auto* engine = dispatcher->io_engine();
    if (engine == nullptr) return false;
    *name = engine->backend_name();
    return *name != nullptr;
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
    const char* backend = nullptr;
    ASSERT_TRUE(actual_backend(first, &backend));
    SCOPED_TRACE(::testing::Message() << "stopped listener backend=" << backend);
    std::fprintf(stderr, "[listener-diag] stop/rebind backend=%s port=%u\n", backend, static_cast<unsigned>(port));

    CoroTcpListener second(cfg_for(port), drop_conn);
    errno = 0;
    const int second_result = second.start_background();
    const int second_error = errno;
    EXPECT_EQ(second_result, -1) << "backend=" << backend << " errno=" << second_error << " " << std::strerror(second_error);
    EXPECT_EQ(second_error, EADDRINUSE) << "backend=" << backend << " " << std::strerror(second_error);

    first.stop();
    CoroTcpListener third(cfg_for(port), drop_conn);
    errno = 0;
    const int third_result = third.start_background();
    const int third_error = errno;
    EXPECT_EQ(third_result, 0) << "stopped_backend=" << backend << " errno=" << third_error << " " << std::strerror(third_error);
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

using ClientClock = std::chrono::steady_clock;
using ClientDeadline = ClientClock::time_point;
// Test-fixture syscall seam: defaults are real OS calls. Scripted controls only
// introduce EINTR; readiness/peer state and byte transfer still require the OS.
struct ClientCalls {
    int (*poll_ready)(pollfd*, nfds_t, int) = ::poll;
    int (*connect_socket)(int, const sockaddr*, socklen_t) = ::connect;
    ssize_t (*send_one)(int, const void*, size_t, int) = ::send;
    ssize_t (*recv_one)(int, void*, size_t, int) = ::recv;
};
bool client_ready(int fd, short events, ClientDeadline deadline, const ClientCalls& calls = {}) {
    for (unsigned attempt = 0u; attempt < 1024u && ClientClock::now() < deadline; ++attempt) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - ClientClock::now()).count();
        pollfd p{fd, events, 0};
        const int result = calls.poll_ready(&p, 1u, static_cast<int>(left > 0 ? left : 1));
        if (result < 0 && errno == EINTR) continue;
        if (result < 0) return false;
        if (result == 0) break;
        if ((p.revents & POLLNVAL) != 0) { errno = EBADF; return false; }
        if ((p.revents & (events | POLLHUP | POLLERR)) != 0) return true;
    }
    errno = ETIMEDOUT; return false;
}
ssize_t client_one(int fd, char* byte, bool write, ClientDeadline deadline, const ClientCalls& calls = {}) {
#ifdef MSG_NOSIGNAL
    constexpr int send_flags = MSG_NOSIGNAL;
#else
    constexpr int send_flags = 0; // SO_NOSIGPIPE is set on the macOS client
#endif
    for (unsigned attempt = 0u; attempt < 1024u && ClientClock::now() < deadline; ++attempt) {
        const auto n = write ? calls.send_one(fd, byte, 1u, send_flags) : calls.recv_one(fd, byte, 1u, 0);
        if (n >= 0) return n; // exact1 or real EOF; never manufacture bytes
        const int error = errno;
        if (error == EINTR) continue;
        if (error != EAGAIN && error != EWOULDBLOCK) return -1;
        if (!client_ready(fd, write ? POLLOUT : POLLIN, deadline, calls)) return -1;
    }
    errno = ETIMEDOUT; return -1;
}
int client_failure(int fd, int error) { close(fd); errno = error; return -1; }
bool client_connected(int fd, uint16_t port, ClientDeadline deadline, const ClientCalls& calls) {
    for (unsigned attempt = 0u; attempt < 1024u && ClientClock::now() < deadline; ++attempt) {
        if (!client_ready(fd, POLLOUT, deadline, calls)) return false;
        int error = 0; socklen_t bytes = sizeof(error);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &bytes) != 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (error == EINPROGRESS || error == EALREADY) continue;
        if (error != 0) { errno = error; return false; }
        sockaddr_in peer{}; bytes = sizeof(peer);
        if (getpeername(fd, reinterpret_cast<sockaddr*>(&peer), &bytes) == 0) {
            if (peer.sin_family == AF_INET && peer.sin_addr.s_addr == htonl(INADDR_LOOPBACK) && peer.sin_port == htons(port)) return true;
            errno = EPROTO; return false;
        }
        if (errno != ENOTCONN && errno != EINTR) return false;
    }
    errno = ETIMEDOUT; return false;
}
int connect_owner_client(uint16_t port, ClientDeadline deadline, const ClientCalls& calls = {}) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
#ifdef SO_NOSIGPIPE
    const int no_sigpipe = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe)) != 0) {
        return client_failure(fd, errno);
    }
#endif
    timeval timeout{2, 0};
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) return client_failure(fd, errno);
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) return client_failure(fd, errno);
    const int result = calls.connect_socket(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)); // exactly once
    if (result != 0 && errno != EINTR && errno != EINPROGRESS && errno != EALREADY) return client_failure(fd, errno);
    if (!client_connected(fd, port, deadline, calls)) return client_failure(fd, errno);
    return fd;
}

unsigned interrupted_polls = 0u, interrupted_reads = 0u, interrupted_writes = 0u, connect_calls = 0u;
int interrupt_poll_once(pollfd* fds, nfds_t count, int timeout) {
    if (interrupted_polls++ == 0u) { errno = EINTR; return -1; }
    return ::poll(fds, count, timeout);
}
ssize_t interrupt_read_once(int fd, void* bytes, size_t count, int flags) {
    if (interrupted_reads++ == 0u) { errno = EINTR; return -1; }
    return ::recv(fd, bytes, count, flags);
}
ssize_t interrupt_write_once(int fd, const void* bytes, size_t count, int flags) {
    if (interrupted_writes++ == 0u) { errno = EINTR; return -1; }
    return ::send(fd, bytes, count, flags);
}
int interrupt_actual_connect(int fd, const sockaddr* peer, socklen_t bytes) {
    ++connect_calls;
    const int result = ::connect(fd, peer, bytes);
    if (result == 0 || errno == EINPROGRESS) { errno = EINTR; return -1; }
    return result;
}
struct ClientPair {
    int fd[2]{-1, -1};
    ClientPair() {
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, fd) != 0) return;
        for (int value : fd) {
            const int flags = fcntl(value, F_GETFL, 0);
            if (flags < 0 || fcntl(value, F_SETFL, flags | O_NONBLOCK) != 0) {
                close(fd[0]); close(fd[1]); fd[0] = fd[1] = -1; return;
            }
        }
    }
    ~ClientPair() { for (int value : fd) if (value >= 0) close(value); }
};

TEST(CoroListenerClient, InterruptedPollAndExactIoStillRequireRealBytesAndRespectDeadline) {
    ClientPair pair;
    ASSERT_GE(pair.fd[0], 0);
    interrupted_polls = interrupted_reads = interrupted_writes = 0u;
    ClientCalls calls;
    calls.poll_ready = interrupt_poll_once;
    calls.recv_one = interrupt_read_once;
    calls.send_one = interrupt_write_once;
    const auto deadline = ClientClock::now() + std::chrono::seconds(2);
    char byte = 'q';
    ASSERT_EQ(client_one(pair.fd[0], &byte, true, deadline, calls), 1);
    EXPECT_EQ(interrupted_writes, 2u);
    ASSERT_TRUE(client_ready(pair.fd[1], POLLIN, deadline, calls));
    EXPECT_EQ(interrupted_polls, 2u);
    byte = '?';
    ASSERT_EQ(client_one(pair.fd[1], &byte, false, deadline, calls), 1);
    EXPECT_EQ(interrupted_reads, 2u);
    EXPECT_EQ(byte, 'q');
    const auto expired = ClientClock::now() + std::chrono::milliseconds(2);
    EXPECT_EQ(client_one(pair.fd[1], &byte, false, expired), -1);
    EXPECT_EQ(errno, ETIMEDOUT);
    EXPECT_EQ(byte, 'q');
    close(pair.fd[0]); pair.fd[0] = -1;
    EXPECT_EQ(client_one(pair.fd[1], &byte, false, deadline), 0); // real EOF, never a byte
    EXPECT_EQ(client_one(-1, &byte, false, deadline), -1);
    EXPECT_EQ(errno, EBADF);
}

TEST(CoroListenerClient, InterruptedSingleConnectRequiresActualPeerWithoutReplacementClient) {
    const uint16_t port = free_port();
    ASSERT_NE(port, 0);
    auto config = cfg_for(port); config.num_io_threads = 1u;
    config.worker_config.num_workers = 1u; config.worker_config.blocking_workers = 0u;
    CoroTcpListener listener(config, drop_conn);
    ASSERT_EQ(listener.start_background(), 0);
    ClientCalls calls; calls.connect_socket = interrupt_actual_connect;
    connect_calls = 0u;
    const int fd = connect_owner_client(port, ClientClock::now() + std::chrono::seconds(2), calls);
    const int error = errno;
    if (fd >= 0) close(fd);
    EXPECT_GE(fd, 0) << error << " " << std::strerror(error);
    EXPECT_EQ(connect_calls, 1u);
    listener.stop();
}

void check_natural_owners(std::size_t reactors) {
    const uint16_t port = free_port();
    ASSERT_NE(port, 0);
    auto config = cfg_for(port);
    config.num_io_threads = reactors;
    CoroTcpListener listener(config, report_owner);
    ASSERT_EQ(listener.start_background(), 0);
    const char* backend = nullptr;
    ASSERT_TRUE(actual_backend(listener, &backend));
    SCOPED_TRACE(::testing::Message() << "backend=" << backend << " reactors=" << reactors);
    std::fprintf(stderr, "[listener-diag] natural fanout backend=%s reactors=%zu port=%u\n",
        backend, reactors, static_cast<unsigned>(port));
    std::array<unsigned, 4> owners{};
    for (unsigned connection = 0; connection < 64; ++connection) {
        const auto deadline = ClientClock::now() + std::chrono::seconds(2);
        errno = 0;
        const int fd = connect_owner_client(port, deadline);
        const int connect_error = errno;
        EXPECT_GE(fd, 0) << "backend=" << backend << " reactors=" << reactors << " connection=" << connection
            << " errno=" << connect_error << " " << std::strerror(connect_error);
        if (fd < 0) continue;
        char first = '?', second = '?';
        char request = 'x';
        const bool exact = client_one(fd, &request, true, deadline) == 1 && client_one(fd, &first, false, deadline) == 1 &&
            client_one(fd, &request, true, deadline) == 1 && client_one(fd, &second, false, deadline) == 1;
        const int io_error = errno;
        EXPECT_TRUE(exact) << "backend=" << backend << " errno=" << io_error << " " << std::strerror(io_error);
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

void expect_owner_zero(uint16_t port, const char* backend) {
    const auto deadline = ClientClock::now() + std::chrono::seconds(2);
    errno = 0;
    const int fd = connect_owner_client(port, deadline);
    const int connect_error = errno;
    EXPECT_GE(fd, 0) << "backend=" << backend << " errno=" << connect_error << " " << std::strerror(connect_error);
    if (fd < 0) return;
    for (unsigned i = 0; i < 2; ++i) {
        char request = 'x';
        char owner = '?';
        EXPECT_EQ(client_one(fd, &request, true, deadline), 1);
        EXPECT_EQ(client_one(fd, &owner, false, deadline), 1);
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
    const char* backend = nullptr;
    ASSERT_TRUE(actual_backend(listener, &backend));
    SCOPED_TRACE(::testing::Message() << "backend=" << backend);
    auto* io = listener.dispatcher();
    expect_owner_zero(port, backend);  // sequence 0; next fresh fd targets reactor 1
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
    errno = 0;
    const auto refused_deadline = ClientClock::now() + std::chrono::seconds(2);
    const int refused = connect_owner_client(port, refused_deadline);
    const int connect_error = errno;
    EXPECT_GE(refused, 0) << "backend=" << backend << " errno=" << connect_error << " " << std::strerror(connect_error);
    if (refused >= 0) {
        char byte = 0;
        const ssize_t n = client_one(refused, &byte, false, refused_deadline);
        const int error = errno;
        EXPECT_TRUE(n == 0 || (n < 0 && error == ECONNRESET))
            << "refused fd must close, not remain queued until target progress";
        close(refused);
    }
    expect_owner_zero(port, backend);  // sequence 2 progresses while target 1 is held
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
    const char* backend = nullptr;
    ASSERT_TRUE(actual_backend(listener, &backend));
    SCOPED_TRACE(::testing::Message() << "backend=" << backend);
    auto* io = listener.dispatcher();
    expect_owner_zero(port, backend);
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
    errno = 0;
    const int fd = connect_owner_client(port, ClientClock::now() + std::chrono::seconds(2)); // sequence1 queues
    const int connect_error = errno;
    EXPECT_GE(fd, 0) << "backend=" << backend << " errno=" << connect_error << " " << std::strerror(connect_error);
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
    // Retire the acceptor while its completion owner is alive. Reactor 1 stays
    // paused with the queued client; accept retirement does not drain handlers.
    using Retirement = bolt::api::core::AcceptRetirementStatus;
    const auto retirement_deadline = ClientClock::now() + std::chrono::seconds(2);
    Retirement retired = Retirement::Pending;
    for (unsigned attempt = 0u; attempt < 1000000u && ClientClock::now() < retirement_deadline; ++attempt) {
        retired = listener.try_retire_accepts(retirement_deadline);
        if (retired != Retirement::Pending) break;
        std::this_thread::yield();
    }
    if (retired != Retirement::Complete) {
        // Failed proof cannot strand the paused owner or destroy borrowed
        // handler state. Keep dispatcher progress alive until the closed peer's
        // queued/started handler really exits, then perform legacy owner cleanup.
        if (fd >= 0) close(fd);
        gate.release.store(true);
        gate.release.notify_all();
        while (listener.get_stats().connections_active != 0u) std::this_thread::yield();
        listener.stop();
    }
    ASSERT_EQ(retired, Retirement::Complete); // failure cleanup already released/joined owners
    EXPECT_EQ(listener.get_stats().connections_active, 1u);
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
        const ssize_t n = client_one(fd, &byte, false, ClientClock::now() + std::chrono::seconds(2));
        const int error = errno;
        EXPECT_TRUE(n == 0 || (n < 0 && error == ECONNRESET));
        close(fd);
    }
    listener.stop();
}

#endif  // __APPLE__

#endif  // !_WIN32
