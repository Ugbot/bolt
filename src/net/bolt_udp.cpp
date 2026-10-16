// bolt/net/bolt_udp.cpp — cross-platform UDP endpoint implementation. Winsock
// (Windows) / BSD sockets (POSIX). The platform headers are confined to this
// TU; the public header (bolt_udp.h) exposes only POD + an opaque socket handle.
//
// Moved up from chukonu::transport into bolt::net (see bolt_udp.h).
//
// Tiger Style: explicit error returns, no exceptions, no allocation, >=2
// asserts per fn.

#include "bolt/net/bolt_udp.h"

#include <cassert>
#include <cerrno>
#include <climits>
#include <cstring>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
using sock_t = SOCKET;
constexpr sock_t k_bad_sock = INVALID_SOCKET;
#else
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <sys/time.h>
#  include <sys/uio.h>
#  include <unistd.h>
using sock_t = int;
constexpr sock_t k_bad_sock = -1;
#endif

namespace bolt::net {
namespace {

// One-time Winsock init (no-op on POSIX). Idempotent; WSAStartup refcounts.
bool platform_init() noexcept {
#if defined(_WIN32)
    static bool s_done = false;
    if (s_done) return true;
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
    s_done = true;
    return true;
#else
    return true;
#endif
}

inline sock_t to_native(std::uint64_t h) noexcept {
    return static_cast<sock_t>(h);
}

inline void close_native(sock_t s) noexcept {
#if defined(_WIN32)
    ::closesocket(s);
#else
    ::close(s);
#endif
}

UdpReceiveResult receive_failure(std::int32_t error) noexcept {
    assert(error != 0);
    UdpReceiveStatus status = UdpReceiveStatus::Error;
#if defined(_WIN32)
    if (error == WSAEWOULDBLOCK || error == WSAETIMEDOUT) status = UdpReceiveStatus::Idle;
    if (error == WSAEINTR) status = UdpReceiveStatus::Interrupted;
    if (error == WSAEMSGSIZE) status = UdpReceiveStatus::Truncated;
#else
    if (error == EAGAIN || error == EWOULDBLOCK || error == ETIMEDOUT) {
        status = UdpReceiveStatus::Idle;
    }
    if (error == EINTR) status = UdpReceiveStatus::Interrupted;
#endif
    const UdpReceiveResult result{status, 0u, error};
    assert(result.bytes == 0u);
    return result;
}

// Validate the OS address and copied count before publishing any peer/bytes.
UdpReceiveResult receive_complete(std::uint64_t count, std::uint32_t cap,
                                  const sockaddr_in& from, std::uint32_t from_len,
                                  UdpAddr* src) noexcept {
    assert(cap > 0u);
    assert(cap <= static_cast<std::uint32_t>(INT_MAX));
    if (count > cap || from_len != sizeof(from) || from.sin_family != AF_INET) {
#if defined(_WIN32)
        return receive_failure(WSAEINVAL);
#else
        return receive_failure(EINVAL);
#endif
    }
    if (src != nullptr) {
        *src = {ntohl(from.sin_addr.s_addr), ntohs(from.sin_port), 0u};
    }
    return {UdpReceiveStatus::Complete, static_cast<std::uint32_t>(count), 0};
}

// Shared bind + buffer/timeout setup. `reuse` sets SO_REUSEADDR (+ SO_REUSEPORT
// on POSIX) before bind so multiple multicast subscribers can share a port.
// Writes the live socket + resolved port into *ep on success.
bool open_bound(UdpEndpoint* ep, std::uint32_t bind_ipv4, std::uint16_t port,
                std::uint32_t recv_timeout_ms, bool reuse) noexcept {
    assert(ep != nullptr);
    ep->sock = k_udp_invalid_sock;
    ep->local_port = 0;
    if (!platform_init()) return false;

    const sock_t s = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == k_bad_sock) return false;

    if (reuse) {
        int on = 1;
        ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR,
                     reinterpret_cast<const char*>(&on), sizeof(on));
#if defined(SO_REUSEPORT) && !defined(_WIN32)
        ::setsockopt(s, SOL_SOCKET, SO_REUSEPORT,
                     reinterpret_cast<const char*>(&on), sizeof(on));
#endif
    }

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(bind_ipv4);
    addr.sin_port        = htons(port);
    if (::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close_native(s);
        return false;
    }

    sockaddr_in bound;
    std::memset(&bound, 0, sizeof(bound));
#if defined(_WIN32)
    int blen = sizeof(bound);
#else
    socklen_t blen = sizeof(bound);
#endif
    if (::getsockname(s, reinterpret_cast<sockaddr*>(&bound), &blen) == 0) {
        ep->local_port = ntohs(bound.sin_port);
    }

    int rcvbuf = 8 * 1024 * 1024;
    ::setsockopt(s, SOL_SOCKET, SO_RCVBUF,
                 reinterpret_cast<const char*>(&rcvbuf), sizeof(rcvbuf));
    if (recv_timeout_ms > 0) {
#if defined(_WIN32)
        DWORD tv = recv_timeout_ms;
        ::setsockopt(s, SOL_SOCKET, SO_RCVTIMEO,
                     reinterpret_cast<const char*>(&tv), sizeof(tv));
#else
        timeval tv;
        tv.tv_sec  = static_cast<long>(recv_timeout_ms / 1000u);
        tv.tv_usec = static_cast<long>((recv_timeout_ms % 1000u) * 1000u);
        ::setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
    }
    ep->sock = static_cast<std::uint64_t>(s);
    assert(ep->sock != k_udp_invalid_sock);
    return true;
}

}  // namespace

bool udp_endpoint_open_at(UdpEndpoint* ep, std::uint32_t bind_ipv4,
                          std::uint16_t port,
                          std::uint32_t recv_timeout_ms) noexcept {
    assert(ep != nullptr);
    return open_bound(ep, bind_ipv4, port, recv_timeout_ms, /*reuse=*/false);
}

bool udp_endpoint_set_nonblocking(UdpEndpoint* ep, bool enabled) noexcept {
    assert(ep != nullptr);
    assert(sizeof(ep->sock) >= sizeof(sock_t));
    if (ep->sock == k_udp_invalid_sock) return false;
#if defined(_WIN32)
    u_long mode = enabled ? 1u : 0u;
    return ::ioctlsocket(to_native(ep->sock), FIONBIO, &mode) == 0;
#else
    const int flags = ::fcntl(to_native(ep->sock), F_GETFL, 0);
    if (flags < 0) return false;
    const int desired = enabled ? flags | O_NONBLOCK : flags & ~O_NONBLOCK;
    return ::fcntl(to_native(ep->sock), F_SETFL, desired) == 0;
#endif
}

bool udp_endpoint_open_reuse(UdpEndpoint* ep, std::uint32_t bind_ipv4,
                             std::uint16_t port,
                             std::uint32_t recv_timeout_ms) noexcept {
    assert(ep != nullptr);
    return open_bound(ep, bind_ipv4, port, recv_timeout_ms, /*reuse=*/true);
}

bool udp_endpoint_open(UdpEndpoint* ep, std::uint16_t port,
                       std::uint32_t recv_timeout_ms) noexcept {
    assert(ep != nullptr);
    return udp_endpoint_open_at(ep, k_udp_loopback_ipv4, port, recv_timeout_ms);
}

void udp_endpoint_close(UdpEndpoint* ep) noexcept {
    assert(ep != nullptr);
    if (ep->sock == k_udp_invalid_sock) return;
    const sock_t s = to_native(ep->sock);
#if defined(_WIN32)
    ::closesocket(s);
#else
    ::close(s);
#endif
    ep->sock = k_udp_invalid_sock;
}

std::int64_t udp_send_to(const UdpEndpoint* ep, UdpAddr dst,
                         const void* buf, std::uint32_t len) noexcept {
    assert(ep != nullptr && ep->sock != k_udp_invalid_sock);
    assert(buf != nullptr || len == 0);
    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(dst.ipv4);
    addr.sin_port        = htons(dst.port);
    const auto n = ::sendto(to_native(ep->sock),
                            reinterpret_cast<const char*>(buf),
                            static_cast<int>(len), 0,
                            reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    return (n < 0) ? -1 : static_cast<std::int64_t>(n);
}

std::int64_t udp_recv_from(const UdpEndpoint* ep, void* buf,
                           std::uint32_t cap, UdpAddr* src) noexcept {
    assert(ep != nullptr && ep->sock != k_udp_invalid_sock);
    assert(buf != nullptr && cap > 0);
    sockaddr_in from;
    std::memset(&from, 0, sizeof(from));
#if defined(_WIN32)
    int flen = sizeof(from);
#else
    socklen_t flen = sizeof(from);
#endif
    const auto n = ::recvfrom(to_native(ep->sock),
                              reinterpret_cast<char*>(buf),
                              static_cast<int>(cap), 0,
                              reinterpret_cast<sockaddr*>(&from), &flen);
    if (n < 0) return -1;
    if (src != nullptr) {
        src->ipv4 = ntohl(from.sin_addr.s_addr);
        src->port = ntohs(from.sin_port);
        src->_pad = 0;
    }
    return static_cast<std::int64_t>(n);
}

UdpReceiveResult udp_recv_from_checked(const UdpEndpoint* ep, void* buf,
                                      std::uint32_t cap, UdpAddr* src) noexcept {
#if defined(_WIN32)
    constexpr std::int32_t bad_endpoint = WSAENOTSOCK;
    constexpr std::int32_t bad_input = WSAEINVAL;
#else
    constexpr std::int32_t bad_endpoint = EBADF;
    constexpr std::int32_t bad_input = EINVAL;
#endif
    if (ep == nullptr || ep->sock == k_udp_invalid_sock) return receive_failure(bad_endpoint);
    if (buf == nullptr || cap == 0u || cap > static_cast<std::uint32_t>(INT_MAX)) {
        return receive_failure(bad_input);
    }
#if defined(_WIN32)
    if (static_cast<std::uint64_t>(to_native(ep->sock)) != ep->sock) {
        return receive_failure(bad_endpoint);
    }
#else
    if (ep->sock > static_cast<std::uint64_t>(INT_MAX)) return receive_failure(bad_endpoint);
#endif
    assert(cap > 0u);
    assert(cap <= static_cast<std::uint32_t>(INT_MAX));
    sockaddr_in from{};
#if defined(_WIN32)
    int from_len = sizeof(from);
    // recvfrom guarantees WSAEMSGSIZE and discarded UDP excess, without
    // WSARecvFrom's provider-dependent partial-message continuation semantics.
    const int count = ::recvfrom(to_native(ep->sock), static_cast<char*>(buf),
        static_cast<int>(cap), 0, reinterpret_cast<sockaddr*>(&from), &from_len);
    if (count == SOCKET_ERROR) return receive_failure(::WSAGetLastError());
    return receive_complete(static_cast<std::uint64_t>(count), cap, from,
                            static_cast<std::uint32_t>(from_len), src);
#else
    iovec buffer{buf, cap};
    msghdr message{};
    message.msg_name = &from;
    message.msg_namelen = sizeof(from);
    message.msg_iov = &buffer;
    message.msg_iovlen = 1;
    // Inspect output MSG_TRUNC; do not request Linux's original-length extension.
    const auto count = ::recvmsg(to_native(ep->sock), &message, 0);
    if (count < 0) return receive_failure(errno);
    if ((message.msg_flags & MSG_TRUNC) != 0) return {UdpReceiveStatus::Truncated, 0u, 0};
    return receive_complete(static_cast<std::uint64_t>(count), cap, from,
                            message.msg_namelen, src);
#endif
}

// --- Multicast -------------------------------------------------------------

bool udp_multicast_set_publisher(UdpEndpoint* ep, std::uint32_t iface_ipv4,
                                 std::uint8_t ttl) noexcept {
    assert(ep != nullptr && ep->sock != k_udp_invalid_sock);
    assert(ttl >= 1u);
    const sock_t s = to_native(ep->sock);
    in_addr ifaddr;
    std::memset(&ifaddr, 0, sizeof(ifaddr));
    ifaddr.s_addr = htonl(iface_ipv4);
    if (::setsockopt(s, IPPROTO_IP, IP_MULTICAST_IF,
                     reinterpret_cast<const char*>(&ifaddr),
                     sizeof(ifaddr)) != 0) {
        return false;
    }
    // TTL and LOOP are a single byte on POSIX, an int (DWORD) on Winsock.
#if defined(_WIN32)
    DWORD ttl_v = ttl;
    DWORD loop_v = 1;
#else
    unsigned char ttl_v = ttl;
    unsigned char loop_v = 1;
#endif
    if (::setsockopt(s, IPPROTO_IP, IP_MULTICAST_TTL,
                     reinterpret_cast<const char*>(&ttl_v),
                     sizeof(ttl_v)) != 0) {
        return false;
    }
    if (::setsockopt(s, IPPROTO_IP, IP_MULTICAST_LOOP,
                     reinterpret_cast<const char*>(&loop_v),
                     sizeof(loop_v)) != 0) {
        return false;
    }
    return true;
}

bool udp_multicast_join(UdpEndpoint* ep, std::uint32_t group_ipv4,
                        std::uint32_t iface_ipv4) noexcept {
    assert(ep != nullptr && ep->sock != k_udp_invalid_sock);
    assert(udp_is_multicast(group_ipv4));
    ip_mreq mreq;
    std::memset(&mreq, 0, sizeof(mreq));
    mreq.imr_multiaddr.s_addr = htonl(group_ipv4);
    mreq.imr_interface.s_addr = htonl(iface_ipv4);
    return ::setsockopt(to_native(ep->sock), IPPROTO_IP, IP_ADD_MEMBERSHIP,
                        reinterpret_cast<const char*>(&mreq),
                        sizeof(mreq)) == 0;
}

bool udp_multicast_leave(UdpEndpoint* ep, std::uint32_t group_ipv4,
                         std::uint32_t iface_ipv4) noexcept {
    assert(ep != nullptr && ep->sock != k_udp_invalid_sock);
    assert(udp_is_multicast(group_ipv4));
    ip_mreq mreq;
    std::memset(&mreq, 0, sizeof(mreq));
    mreq.imr_multiaddr.s_addr = htonl(group_ipv4);
    mreq.imr_interface.s_addr = htonl(iface_ipv4);
    return ::setsockopt(to_native(ep->sock), IPPROTO_IP, IP_DROP_MEMBERSHIP,
                        reinterpret_cast<const char*>(&mreq),
                        sizeof(mreq)) == 0;
}

}  // namespace bolt::net
