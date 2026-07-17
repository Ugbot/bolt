// bolt/net/bolt_udp.h — cross-platform UDP endpoint, the thin datagram
// primitive the transport frames ride on. Winsock on Windows, BSD sockets on
// POSIX; the platform headers live ONLY in bolt_udp.cpp so this header is safe
// to include anywhere (no winsock2.h / sys/socket.h in public headers).
//
// Moved up from chukonu::transport into bolt::net so BOTH the control plane
// (gestalt replication) and the compute plane (chukonu) can use one datagram
// primitive without depending on each other — the transport is a base-layer
// networking concern, not a compute-plane one. chukonu/transport/udp_endpoint.h
// re-exports these into chukonu::transport for source compatibility.
//
// The socket handle is carried as an opaque std::uint64_t (SOCKET is UINT_PTR
// on Windows, an int fd on POSIX — both fit).
//
// Tiger Style: POD endpoint, explicit error returns (no exceptions), fixed
// bounds, >=2 asserts per fn.

#pragma once

#include <cstdint>

namespace bolt::net {

// Sentinel for an unopened / closed socket handle.
constexpr std::uint64_t k_udp_invalid_sock = ~0ull;

// A host-order IPv4 address + UDP port. 127.0.0.1 == 0x7F000001.
struct UdpAddr {
    std::uint32_t ipv4;   // host byte order
    std::uint16_t port;   // host byte order
    std::uint16_t _pad;
};
static_assert(__is_trivially_copyable(UdpAddr), "UdpAddr must be POD");

// The loopback address constant (127.0.0.1).
constexpr std::uint32_t k_udp_loopback_ipv4 = 0x7F000001u;

// INADDR_ANY in host byte order — bind to every local interface. Used by
// cross-host endpoints that want to receive on any NIC; the kernel still
// chooses the source IP on send based on the route.
constexpr std::uint32_t k_udp_any_ipv4 = 0x00000000u;

// An open UDP endpoint. POD; opened in place via udp_endpoint_open.
struct UdpEndpoint {
    std::uint64_t sock;        // opaque platform socket handle
    std::uint16_t local_port;  // the bound port (resolved when opened with 0)
    std::uint16_t _pad[3];
};
static_assert(__is_trivially_copyable(UdpEndpoint), "UdpEndpoint must be POD");

// Open a UDP endpoint bound to `bind_ipv4`:`port`. `port == 0` requests an
// ephemeral port; the chosen port is written back to ep->local_port. A
// `recv_timeout_ms` of 0 means block forever; > 0 sets SO_RCVTIMEO so
// udp_recv_from returns -1 on timeout (lets a receiver loop poll for
// shutdown). Pass `k_udp_loopback_ipv4` for single-host, `k_udp_any_ipv4`
// to receive on any local interface, or any specific NIC address for
// pinned binding. Returns false on any socket/bind error (ep->sock left
// invalid).
//
// Preconditions: ep != nullptr.
bool udp_endpoint_open_at(UdpEndpoint* ep, std::uint32_t bind_ipv4,
                          std::uint16_t port,
                          std::uint32_t recv_timeout_ms) noexcept;

// Legacy convenience: bind to 127.0.0.1. Equivalent to
// udp_endpoint_open_at(ep, k_udp_loopback_ipv4, port, recv_timeout_ms).
bool udp_endpoint_open(UdpEndpoint* ep, std::uint16_t port,
                       std::uint32_t recv_timeout_ms) noexcept;

// Close the endpoint (idempotent; safe on an already-closed endpoint).
void udp_endpoint_close(UdpEndpoint* ep) noexcept;

// Send `len` bytes to `dst`. Returns bytes sent (== len on success), or -1
// on error. One datagram; `len` must be <= the negotiated MTU (caller's
// responsibility). Never blocks for a bounded send.
//
// Preconditions: ep != nullptr, ep->sock valid, buf != nullptr || len == 0.
std::int64_t udp_send_to(const UdpEndpoint* ep, UdpAddr dst,
                         const void* buf, std::uint32_t len) noexcept;

// Receive one datagram into `buf` (capacity `cap`). Blocks until a datagram
// arrives or the recv timeout elapses. Returns the datagram length (>= 0,
// truncated to `cap`), or -1 on timeout/error. `src` (if non-null) gets the
// sender's address.
//
// Preconditions: ep != nullptr, ep->sock valid, buf != nullptr, cap > 0.
std::int64_t udp_recv_from(const UdpEndpoint* ep, void* buf,
                           std::uint32_t cap, UdpAddr* src) noexcept;

// --- Multicast -------------------------------------------------------------
constexpr std::uint32_t k_udp_mcast_lo = 0xE0000000u;  // 224.0.0.0
constexpr std::uint32_t k_udp_mcast_hi = 0xEFFFFFFFu;  // 239.255.255.255

// True iff `ipv4` (host byte order) is a multicast group address.
inline bool udp_is_multicast(std::uint32_t ipv4) noexcept {
    return (ipv4 & 0xF0000000u) == 0xE0000000u;
}

// Configure `ep` as a multicast publisher (IP_MULTICAST_IF/TTL/LOOP).
// Preconditions: ep != nullptr, ep->sock valid, ttl >= 1.
bool udp_multicast_set_publisher(UdpEndpoint* ep, std::uint32_t iface_ipv4,
                                 std::uint8_t ttl) noexcept;

// Join `ep` to multicast `group_ipv4` on interface `iface_ipv4`.
// Preconditions: ep != nullptr, ep->sock valid, udp_is_multicast(group_ipv4).
bool udp_multicast_join(UdpEndpoint* ep, std::uint32_t group_ipv4,
                        std::uint32_t iface_ipv4) noexcept;

// Leave a previously-joined group (IP_DROP_MEMBERSHIP).
// Preconditions: ep != nullptr, ep->sock valid, udp_is_multicast(group_ipv4).
bool udp_multicast_leave(UdpEndpoint* ep, std::uint32_t group_ipv4,
                         std::uint32_t iface_ipv4) noexcept;

// Allow multiple sockets to bind the same multicast port (SO_REUSEADDR, and
// SO_REUSEPORT where it exists). Call BEFORE bind; opens a fresh socket with
// reuse set and binds it. Returns false on socket/bind error.
//
// Preconditions: ep != nullptr.
bool udp_endpoint_open_reuse(UdpEndpoint* ep, std::uint32_t bind_ipv4,
                             std::uint16_t port,
                             std::uint32_t recv_timeout_ms) noexcept;

}  // namespace bolt::net
