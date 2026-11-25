// bolt/net/bolt_udp_batch.cpp — batched datagram I/O (G2CHK-1271). Platform
// headers stay in this TU; see bolt_udp_batch.h for the contract.

#include "bolt/net/bolt_udp_batch.h"

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
using sock_t = SOCKET;
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <netinet/udp.h>
#  include <sys/socket.h>
#  include <sys/uio.h>
#  include <unistd.h>
using sock_t = int;
#endif

#if defined(__APPLE__)
// recvmsg_x / sendmsg_x: XNU batch calls (private SPI, the ones libuv links).
struct msghdr_x {
    void* msg_name;
    socklen_t msg_namelen;
    struct iovec* msg_iov;
    int msg_iovlen;
    void* msg_control;
    socklen_t msg_controllen;
    int msg_flags;
    size_t msg_datalen;
};
extern "C" ssize_t recvmsg_x(int s, const struct msghdr_x* msgp, unsigned int cnt, int flags);
extern "C" ssize_t sendmsg_x(int s, const struct msghdr_x* msgp, unsigned int cnt, int flags);
#endif

namespace bolt::net {
namespace {

using limits::kUdpBatchMax;

inline sock_t native(const UdpEndpoint* ep) noexcept { return static_cast<sock_t>(ep->sock); }

#if !defined(_WIN32)
inline bool would_block(int e) noexcept { return e == EAGAIN || e == EWOULDBLOCK || e == EINTR; }

inline sockaddr_in to_sockaddr(UdpAddr a) noexcept {
    sockaddr_in s{};
    s.sin_family = AF_INET;
    s.sin_port = htons(a.port);
    s.sin_addr.s_addr = htonl(a.ipv4);
    return s;
}

inline UdpAddr from_sockaddr(const sockaddr_in& s) noexcept {
    return UdpAddr{ntohl(s.sin_addr.s_addr), ntohs(s.sin_port), 0u};
}

// Indices of the non-held slots, up to max.
std::uint32_t free_slots(const UdpBatch* b, std::uint32_t max, std::uint32_t* idx) noexcept {
    assert(b != nullptr && idx != nullptr);
    std::uint32_t n = 0;
    for (std::uint32_t i = 0; i < b->cap && n < max; ++i) {
        if ((b->slots[i].flags & kUdpSlotHeld) == 0) idx[n++] = i;
    }
    assert(n <= max);
    return n;
}

void finish_slot(UdpSlot& s, ssize_t got, int msg_flags, const sockaddr_in& from,
                 std::uint16_t segment) noexcept {
    s.flags = static_cast<std::uint8_t>((s.flags & kUdpSlotHeld) | kUdpSlotPeerValid);
    s.peer = from_sockaddr(from);
    s.segment = segment;
    if ((msg_flags & MSG_TRUNC) != 0 || got < 0) {
        s.flags |= kUdpSlotTruncated;
        s.len = 0;
        return;
    }
    s.len = static_cast<std::uint32_t>(got);
    assert(s.len <= s.cap);
}
#endif

#if defined(__linux__)
std::uint16_t gro_segment(const msghdr& m, std::uint32_t* drops) noexcept {
    std::uint16_t seg = 0;
    for (cmsghdr* c = CMSG_FIRSTHDR(const_cast<msghdr*>(&m)); c != nullptr;
         c = CMSG_NXTHDR(const_cast<msghdr*>(&m), c)) {
#  if defined(UDP_GRO)
        if (c->cmsg_level == SOL_UDP && c->cmsg_type == UDP_GRO) {
            int v = 0;
            std::memcpy(&v, CMSG_DATA(c), sizeof(v));
            seg = static_cast<std::uint16_t>(v);
        }
#  endif
#  if defined(SO_RXQ_OVFL)
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SO_RXQ_OVFL && drops != nullptr) {
            std::uint32_t v = 0;
            std::memcpy(&v, CMSG_DATA(c), sizeof(v));
            *drops = v;
        }
#  endif
    }
    return seg;
}

std::int64_t recv_linux(const UdpEndpoint* ep, UdpBatch* b, std::uint32_t max,
                        UdpBatchStats* stats) noexcept {
    std::uint32_t idx[kUdpBatchMax];
    const std::uint32_t n = free_slots(b, max, idx);
    if (n == 0) return 0;
    mmsghdr h[kUdpBatchMax];
    iovec v[kUdpBatchMax];
    sockaddr_in from[kUdpBatchMax];
    alignas(cmsghdr) char ctrl[kUdpBatchMax][64];
    for (std::uint32_t i = 0; i < n; ++i) {
        UdpSlot& s = b->slots[idx[i]];
        v[i] = {s.buf, s.cap};
        std::memset(&h[i], 0, sizeof(h[i]));
        h[i].msg_hdr.msg_name = &from[i];
        h[i].msg_hdr.msg_namelen = sizeof(from[i]);
        h[i].msg_hdr.msg_iov = &v[i];
        h[i].msg_hdr.msg_iovlen = 1;
        h[i].msg_hdr.msg_control = ctrl[i];
        h[i].msg_hdr.msg_controllen = sizeof(ctrl[i]);  // reset every call (GRO / OVFL)
    }
    const int got = ::recvmmsg(native(ep), h, n, MSG_DONTWAIT, nullptr);
    if (got < 0) return would_block(errno) ? 0 : -1;
    std::uint32_t drops = 0;
    for (int i = 0; i < got; ++i) {
        const std::uint16_t seg = gro_segment(h[i].msg_hdr, &drops);
        finish_slot(b->slots[idx[i]], static_cast<ssize_t>(h[i].msg_len), h[i].msg_hdr.msg_flags,
                    from[i], seg);
    }
    if (stats != nullptr && drops != 0) stats->kernel_drops = drops;
    return got;
}

// Slots [i, end) that can go out as one GSO message: same peer, all but the
// last exactly `seg` bytes, the last <= seg, within the GSO bounds.
std::uint32_t gso_run(const UdpBatch* b, std::uint32_t i, std::uint32_t end) noexcept {
    const UdpSlot& head = b->slots[i];
    const std::uint32_t seg = head.len;
    std::uint32_t j = i + 1, bytes = seg;
    while (j < end && j - i < limits::kUdpGsoSegmentsMax) {
        const UdpSlot& s = b->slots[j];
        if (s.peer.ipv4 != head.peer.ipv4 || s.peer.port != head.peer.port || s.len > seg || s.len == 0) break;
        if (bytes + s.len > limits::kUdpGsoBytesMax) break;
        bytes += s.len;
        ++j;
        if (s.len < seg) break;  // a short segment ends the run
    }
    return j - i;
}

std::int64_t send_linux(const UdpEndpoint* ep, const UdpBatch* b, std::uint32_t first,
                        std::uint32_t n, bool gso, UdpBatchStats* stats) noexcept {
    mmsghdr h[kUdpBatchMax];
    iovec v[kUdpBatchMax];
    sockaddr_in to[kUdpBatchMax];
    alignas(cmsghdr) char ctrl[kUdpBatchMax][CMSG_SPACE(sizeof(std::uint16_t))];
    std::uint32_t run_len[kUdpBatchMax];
    const bool use_gso = gso && (stats == nullptr || stats->gso_disabled == 0);
    std::uint32_t m = 0;
    for (std::uint32_t i = first; i < first + n; ++m) {
        const std::uint32_t r = use_gso ? gso_run(b, i, first + n) : 1u;
        std::memset(&h[m], 0, sizeof(h[m]));
        to[m] = to_sockaddr(b->slots[i].peer);
        h[m].msg_hdr.msg_name = &to[m];
        h[m].msg_hdr.msg_namelen = sizeof(to[m]);
        h[m].msg_hdr.msg_iov = &v[i - first];
        h[m].msg_hdr.msg_iovlen = r;
        for (std::uint32_t k = 0; k < r; ++k) v[i - first + k] = {b->slots[i + k].buf, b->slots[i + k].len};
#  if defined(UDP_SEGMENT)
        if (r > 1) {
            h[m].msg_hdr.msg_control = ctrl[m];
            h[m].msg_hdr.msg_controllen = CMSG_SPACE(sizeof(std::uint16_t));
            cmsghdr* c = CMSG_FIRSTHDR(&h[m].msg_hdr);
            c->cmsg_level = SOL_UDP;
            c->cmsg_type = UDP_SEGMENT;
            c->cmsg_len = CMSG_LEN(sizeof(std::uint16_t));
            const std::uint16_t seg = static_cast<std::uint16_t>(b->slots[i].len);
            std::memcpy(CMSG_DATA(c), &seg, sizeof(seg));
            if (stats != nullptr) ++stats->gso_runs;
        }
#  endif
        run_len[m] = r;
        i += r;
    }
    int sent = ::sendmmsg(native(ep), h, m, MSG_DONTWAIT);
    if (sent < 0 && errno == EIO && use_gso && stats != nullptr) {
        stats->gso_disabled = 1;  // the kernel / NIC refused UDP_SEGMENT: resend as singles
        return send_linux(ep, b, first, n, false, stats);
    }
    if (sent < 0) return would_block(errno) ? 0 : -1;
    std::uint32_t datagrams = 0;
    for (int k = 0; k < sent; ++k) datagrams += run_len[k];
    if (sent > 0 && static_cast<std::uint32_t>(sent) < m) {  // resume once from the first unsent
        const int more = ::sendmmsg(native(ep), h + sent, m - static_cast<std::uint32_t>(sent), MSG_DONTWAIT);
        for (int k = 0; k < more; ++k) datagrams += run_len[sent + k];
    }
    return datagrams;
}
#endif

#if defined(__APPLE__)
std::int64_t recv_darwin(const UdpEndpoint* ep, UdpBatch* b, std::uint32_t max) noexcept {
    std::uint32_t idx[kUdpBatchMax];
    const std::uint32_t n = free_slots(b, max, idx);
    if (n == 0) return 0;
    // recvmsg_x does not report MSG_TRUNC per entry: give each datagram one
    // extra scratch byte, so more than cap bytes received means truncated.
    msghdr_x h[kUdpBatchMax];
    iovec v[kUdpBatchMax][2];
    char spill[kUdpBatchMax];
    sockaddr_in from[kUdpBatchMax];
    for (std::uint32_t i = 0; i < n; ++i) {
        UdpSlot& s = b->slots[idx[i]];
        v[i][0] = {s.buf, s.cap};
        v[i][1] = {&spill[i], 1};
        std::memset(&h[i], 0, sizeof(h[i]));
        h[i].msg_name = &from[i];
        h[i].msg_namelen = sizeof(from[i]);
        h[i].msg_iov = v[i];
        h[i].msg_iovlen = 2;
    }
    const ssize_t got = ::recvmsg_x(native(ep), h, n, MSG_DONTWAIT);
    if (got < 0) return would_block(errno) ? 0 : -1;
    for (ssize_t i = 0; i < got; ++i) {
        UdpSlot& s = b->slots[idx[i]];
        const bool over = h[i].msg_datalen > s.cap;
        finish_slot(s, static_cast<ssize_t>(h[i].msg_datalen), over ? MSG_TRUNC : h[i].msg_flags, from[i], 0);
    }
    return got;
}

std::int64_t send_darwin(const UdpEndpoint* ep, const UdpBatch* b, std::uint32_t first,
                         std::uint32_t n) noexcept {
    msghdr_x h[kUdpBatchMax];
    iovec v[kUdpBatchMax];
    sockaddr_in to[kUdpBatchMax];
    for (std::uint32_t i = 0; i < n; ++i) {
        const UdpSlot& s = b->slots[first + i];
        v[i] = {s.buf, s.len};
        to[i] = to_sockaddr(s.peer);
        std::memset(&h[i], 0, sizeof(h[i]));
        h[i].msg_name = &to[i];
        h[i].msg_namelen = sizeof(to[i]);
        h[i].msg_iov = &v[i];
        h[i].msg_iovlen = 1;
    }
    ssize_t sent = ::sendmsg_x(native(ep), h, n, MSG_DONTWAIT);
    if (sent < 0) return would_block(errno) ? 0 : -1;
    if (sent > 0 && static_cast<std::uint32_t>(sent) < n) {
        const ssize_t more = ::sendmsg_x(native(ep), h + sent, n - static_cast<std::uint32_t>(sent), MSG_DONTWAIT);
        if (more > 0) sent += more;
    }
    return sent;
}
#endif

#if !defined(__linux__) && !defined(__APPLE__)
// Portable: one syscall per datagram, until the socket is empty / full.
// Windows needs the endpoint in non-blocking mode (udp_endpoint_set_nonblocking).
std::int64_t recv_loop(const UdpEndpoint* ep, UdpBatch* b, std::uint32_t max) noexcept;
std::int64_t send_loop(const UdpEndpoint* ep, const UdpBatch* b, std::uint32_t first, std::uint32_t n) noexcept;
#endif

}  // namespace

std::int64_t udp_recv_batch(const UdpEndpoint* ep, UdpBatch* b, std::uint32_t max,
                            UdpBatchStats* stats) noexcept {
    assert(ep != nullptr && b != nullptr && b->slots != nullptr);
    assert(max <= b->cap && max <= kUdpBatchMax);
    b->count = 0;
    if (ep->sock == k_udp_invalid_sock || max == 0) return max == 0 ? 0 : -1;
#if defined(__linux__)
    const std::int64_t got = recv_linux(ep, b, max, stats);
#elif defined(__APPLE__)
    const std::int64_t got = recv_darwin(ep, b, max);
#else
    const std::int64_t got = recv_loop(ep, b, max);
#endif
    if (got > 0) b->count = static_cast<std::uint32_t>(got);
    if (stats != nullptr) {
        ++stats->recv_calls;
        if (got > 0) stats->datagrams_in += static_cast<std::uint64_t>(got);
    }
    assert(got <= static_cast<std::int64_t>(max));
    return got;
}

std::int64_t udp_send_batch(const UdpEndpoint* ep, const UdpBatch* b, std::uint32_t first,
                            std::uint32_t n, bool gso, UdpBatchStats* stats) noexcept {
    assert(ep != nullptr && b != nullptr && b->slots != nullptr);
    assert(n <= kUdpBatchMax && first + n <= b->cap);
    if (ep->sock == k_udp_invalid_sock) return -1;
    if (n == 0) return 0;
#if defined(__linux__)
    const std::int64_t sent = send_linux(ep, b, first, n, gso, stats);
#elif defined(__APPLE__)
    (void)gso;
    const std::int64_t sent = send_darwin(ep, b, first, n);
#else
    (void)gso;
    const std::int64_t sent = send_loop(ep, b, first, n);
#endif
    if (stats != nullptr) {
        ++stats->send_calls;
        if (sent > 0) stats->datagrams_out += static_cast<std::uint64_t>(sent);
    }
    return sent;
}

bool udp_endpoint_enable_gro(UdpEndpoint* ep) noexcept {
    assert(ep != nullptr);
#if defined(__linux__) && defined(UDP_GRO)
    const int one = 1;
    return ::setsockopt(native(ep), SOL_UDP, UDP_GRO, &one, sizeof(one)) == 0;
#else
    (void)ep;
    return false;
#endif
}

bool udp_endpoint_enable_drop_count(UdpEndpoint* ep) noexcept {
    assert(ep != nullptr);
#if defined(SO_RXQ_OVFL)
    const int one = 1;
    return ::setsockopt(native(ep), SOL_SOCKET, SO_RXQ_OVFL, &one, sizeof(one)) == 0;
#else
    (void)ep;
    return false;
#endif
}

bool udp_endpoint_set_send_buffer(UdpEndpoint* ep, std::uint32_t bytes, std::uint32_t* actual) noexcept {
    assert(ep != nullptr && actual != nullptr);
    assert(bytes > 0);
    const int want = bytes > static_cast<std::uint32_t>(INT_MAX) ? INT_MAX : static_cast<int>(bytes);
    if (::setsockopt(native(ep), SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char*>(&want), sizeof(want)) != 0) {
        return false;
    }
    int got = 0;
    socklen_t len = sizeof(got);
    if (::getsockopt(native(ep), SOL_SOCKET, SO_SNDBUF, reinterpret_cast<char*>(&got), &len) != 0) return false;
    *actual = static_cast<std::uint32_t>(got);
    return true;
}

bool udp_multicast_set_loop(UdpEndpoint* ep, bool loop) noexcept {
    assert(ep != nullptr);
    assert(ep->sock != k_udp_invalid_sock);
#if defined(_WIN32)
    const DWORD v = loop ? 1 : 0;
#else
    const unsigned char v = loop ? 1 : 0;
#endif
    return ::setsockopt(native(ep), IPPROTO_IP, IP_MULTICAST_LOOP, reinterpret_cast<const char*>(&v), sizeof(v)) == 0;
}

#if !defined(__linux__) && !defined(__APPLE__)
namespace {
std::int64_t recv_loop(const UdpEndpoint* ep, UdpBatch* b, std::uint32_t max) noexcept {
    std::int64_t got = 0;
    for (std::uint32_t i = 0; i < b->cap && static_cast<std::uint32_t>(got) < max; ++i) {
        UdpSlot& s = b->slots[i];
        if ((s.flags & kUdpSlotHeld) != 0) continue;
        UdpAddr from{};
        const UdpReceiveResult r = udp_recv_from_checked(ep, s.buf, s.cap, &from);
        if (r.status == UdpReceiveStatus::Idle || r.status == UdpReceiveStatus::Interrupted) break;
        if (r.status == UdpReceiveStatus::Error) return got > 0 ? got : -1;
        s.flags = static_cast<std::uint8_t>((s.flags & kUdpSlotHeld) | kUdpSlotPeerValid);
        s.segment = 0;
        s.peer = from;
        s.len = r.status == UdpReceiveStatus::Complete ? r.bytes : 0;
        if (r.status == UdpReceiveStatus::Truncated) s.flags |= kUdpSlotTruncated;
        ++got;
    }
    return got;
}

std::int64_t send_loop(const UdpEndpoint* ep, const UdpBatch* b, std::uint32_t first, std::uint32_t n) noexcept {
    std::int64_t sent = 0;
    for (std::uint32_t i = first; i < first + n; ++i) {
        const UdpSlot& s = b->slots[i];
        if (udp_send_to(ep, s.peer, s.buf, s.len) < 0) return sent > 0 ? sent : 0;
        ++sent;
    }
    return sent;
}
}  // namespace
#endif

}  // namespace bolt::net
