// A second CoroTcpListener (same or another process) must refuse a port that
// is already bound instead of joining the holder's SO_REUSEPORT group
// (G2LAUNCH-64). POSIX only: the cross-process case forks.

#if !defined(_WIN32)

#include <gtest/gtest.h>

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

#endif  // !_WIN32
