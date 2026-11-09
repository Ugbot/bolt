// CoroTcpListener binds the configured host, not every interface: a listener
// for 127.0.0.1 must not be reachable on the machine's other addresses, and an
// unparseable host must fail start() instead of widening to INADDR_ANY.
// POSIX only (getifaddrs).

#if !defined(_WIN32)

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <string>

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
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    EXPECT_EQ(bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    socklen_t len = sizeof(addr);
    EXPECT_EQ(getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);
    close(fd);
    return ntohs(addr.sin_port);
}

CoroTcpListenerConfig cfg_for(const std::string& host, uint16_t port) {
    CoroTcpListenerConfig cfg;
    cfg.host = host;
    cfg.port = port;
    cfg.num_io_threads = 1;
    cfg.num_workers = 1;
    return cfg;
}

// First up, non-loopback IPv4 address of this machine, or false if none.
bool non_loopback_ipv4(in_addr* out) {
    ifaddrs* list = nullptr;
    if (getifaddrs(&list) != 0) return false;
    bool found = false;
    for (ifaddrs* it = list; it != nullptr && !found; it = it->ifa_next) {
        if (it->ifa_addr == nullptr || it->ifa_addr->sa_family != AF_INET) continue;
        if ((it->ifa_flags & IFF_UP) == 0 || (it->ifa_flags & IFF_LOOPBACK) != 0) continue;
        const auto* sin = reinterpret_cast<const sockaddr_in*>(it->ifa_addr);
        if ((ntohl(sin->sin_addr.s_addr) >> 24) == 127) continue;
        *out = sin->sin_addr;
        found = true;
    }
    freeifaddrs(list);
    return found;
}

// Blocking connect; returns 0 on success, else the errno.
int connect_errno(in_addr ip, uint16_t port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return errno;
    sockaddr_in peer{};
    peer.sin_family = AF_INET;
    peer.sin_addr = ip;
    peer.sin_port = htons(port);
    const int rc = connect(fd, reinterpret_cast<sockaddr*>(&peer), sizeof(peer));
    const int err = rc == 0 ? 0 : errno;
    close(fd);
    return err;
}

in_addr loopback() {
    in_addr a{};
    a.s_addr = htonl(INADDR_LOOPBACK);
    return a;
}

}  // namespace

TEST(CoroListenerBindHost, LoopbackHostIsNotReachableOnOtherInterfaces) {
    in_addr external{};
    if (!non_loopback_ipv4(&external)) GTEST_SKIP() << "no non-loopback IPv4 interface";

    const uint16_t port = free_port();
    CoroTcpListener listener(cfg_for("127.0.0.1", port), drop_conn);
    ASSERT_EQ(listener.start_background(), 0);

    EXPECT_EQ(connect_errno(loopback(), port), 0);
    EXPECT_EQ(connect_errno(external, port), ECONNREFUSED)
        << "listener for 127.0.0.1 accepted on " << inet_ntoa(external);

    // The address the listener does not own stays free for someone else.
    const int other = socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_GE(other, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr = external;
    addr.sin_port = htons(port);
    EXPECT_EQ(bind(other, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0)
        << strerror(errno);
    close(other);

    listener.stop();
}

TEST(CoroListenerBindHost, LocalhostMeansLoopback) {
    in_addr external{};
    const bool have_external = non_loopback_ipv4(&external);

    const uint16_t port = free_port();
    CoroTcpListener listener(cfg_for("localhost", port), drop_conn);
    ASSERT_EQ(listener.start_background(), 0);
    EXPECT_EQ(connect_errno(loopback(), port), 0);
    if (have_external) EXPECT_EQ(connect_errno(external, port), ECONNREFUSED);
    listener.stop();
}

TEST(CoroListenerBindHost, WildcardHostIsReachableOnOtherInterfaces) {
    in_addr external{};
    if (!non_loopback_ipv4(&external)) GTEST_SKIP() << "no non-loopback IPv4 interface";

    const uint16_t port = free_port();
    CoroTcpListener listener(cfg_for("0.0.0.0", port), drop_conn);
    ASSERT_EQ(listener.start_background(), 0);
    EXPECT_EQ(connect_errno(loopback(), port), 0);
    EXPECT_EQ(connect_errno(external, port), 0);
    listener.stop();
}

TEST(CoroListenerBindHost, UnparseableHostFailsInsteadOfBindingEverything) {
    for (const char* host : {"not-an-address", "::1", "127.0.0.1.5", "10.0.0"}) {
        const uint16_t port = free_port();
        CoroTcpListener listener(cfg_for(host, port), drop_conn);
        errno = 0;
        EXPECT_EQ(listener.start_background(), -1) << host;
        EXPECT_EQ(errno, EINVAL) << host;
        EXPECT_EQ(listener.accept_error(), EINVAL) << host;
        EXPECT_FALSE(listener.is_running()) << host;
        EXPECT_EQ(connect_errno(loopback(), port), ECONNREFUSED) << host;
    }
}

#endif  // !_WIN32
