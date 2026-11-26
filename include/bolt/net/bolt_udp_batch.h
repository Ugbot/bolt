// bolt/net/bolt_udp_batch.h — batched datagram I/O on a UdpEndpoint (G2CHK-1271).
//
// The datagram layer every bolt::net transport builds on (chukonu's reliable
// UDP, VSR links, boltapi QUIC/WebRTC): one syscall moves a batch.
//   receive  Linux recvmmsg (+ UDP_GRO), Darwin recvmsg_x, else recvfrom loop
//   send     Linux sendmmsg with same-peer same-size runs coalesced into one
//            UDP_SEGMENT (GSO) message, Darwin sendmsg_x, else sendto loop
// Calls never block: an empty socket returns 0. Wait for readiness with the
// reactor (IODispatcher::async_readable) or udp_wait_readable, then call again.
//
// Slots are caller-owned and caller-placed: any buffer pointer works (a retain
// ring's sealed fragment, an arena slab). Receive buffers are writable and stay
// the caller's: protocols decrypt and reassemble in place. A slot marked
// kUdpSlotHeld is skipped by udp_recv_batch until the caller clears the flag
// (lend / release by index). Bytes only: frame parsing belongs above.
//
// Tiger Style: POD, no allocation, explicit returns, bounded by
// limits::kUdpBatchMax datagrams per call.
#pragma once

#include "bolt/net/bolt_udp.h"
#include "bolt/net/bolt_udp_limits.h"

#include <cassert>
#include <cstdint>

namespace bolt::net {

enum : std::uint8_t {
    kUdpSlotTruncated = 1u << 0,  // the datagram was larger than cap; contents unusable
    kUdpSlotPeerValid = 1u << 1,  // `peer` holds this datagram's source
    kUdpSlotHeld = 1u << 2,       // lent to the protocol; receive skips it until cleared
};

struct UdpSlot {
    std::uint8_t* buf;      // writable, caller-owned
    std::uint32_t cap;      // bytes buf can hold
    std::uint32_t len;      // receive: bytes received; send: bytes to send
    UdpAddr peer;           // receive: source; send: destination
    std::uint16_t segment;  // receive with GRO: datagram size inside buf (0 = one datagram)
    std::uint8_t flags;
    std::uint8_t _pad;
};
static_assert(__is_trivially_copyable(UdpSlot), "UdpSlot must be POD");

struct UdpBatch {
    UdpSlot* slots;
    std::uint32_t cap;    // slots available
    std::uint32_t count;  // receive: slots filled by the last call (in slot order, held ones skipped)
};
static_assert(__is_trivially_copyable(UdpBatch), "UdpBatch must be POD");

struct UdpBatchStats {
    std::uint64_t recv_calls;
    std::uint64_t datagrams_in;
    std::uint64_t send_calls;
    std::uint64_t datagrams_out;
    std::uint64_t gso_runs;
    std::uint32_t kernel_drops;  // SO_RXQ_OVFL running total (Linux), 0 elsewhere
    std::uint32_t gso_disabled;  // 1 once the kernel refused UDP_SEGMENT
};

// Receive up to `max` datagrams (<= b->cap, <= kUdpBatchMax) into the batch's
// non-held slots. Returns the number received (b->count), 0 when the socket is
// empty, -1 on error (errno / WSAGetLastError). Truncated datagrams count, with
// kUdpSlotTruncated set and len == 0.
std::int64_t udp_recv_batch(const UdpEndpoint* ep, UdpBatch* b, std::uint32_t max,
                            UdpBatchStats* stats = nullptr) noexcept;

// Send slots [first, first + n) (n <= kUdpBatchMax). Returns datagrams sent
// (the kernel's count, resumed once after a short count), 0 when the socket is
// full, -1 on error. `gso` lets same-peer same-size runs go out as one
// UDP_SEGMENT message on Linux; a kernel that refuses it turns GSO off in
// stats->gso_disabled and the run goes out as single datagrams.
std::int64_t udp_send_batch(const UdpEndpoint* ep, const UdpBatch* b, std::uint32_t first,
                            std::uint32_t n, bool gso, UdpBatchStats* stats = nullptr) noexcept;

// Linux UDP_GRO (coalesced receive: a slot then holds several datagrams of
// `segment` bytes, the last possibly shorter; slots should be >= 64 KiB).
// SO_RXQ_OVFL drop counting is turned on with it where available. False where
// unsupported (the endpoint keeps working one datagram per slot).
bool udp_endpoint_enable_gro(UdpEndpoint* ep) noexcept;
bool udp_endpoint_enable_drop_count(UdpEndpoint* ep) noexcept;

// SO_SNDBUF, as udp_endpoint_set_recv_buffer does for SO_RCVBUF.
bool udp_endpoint_set_send_buffer(UdpEndpoint* ep, std::uint32_t bytes,
                                  std::uint32_t* actual) noexcept;
bool udp_multicast_set_loop(UdpEndpoint* ep, bool loop) noexcept;

// The datagrams inside one received slot (one, or several with GRO).
struct UdpSlotDatagrams {
    const std::uint8_t* data;
    std::uint32_t len;
    std::uint32_t segment;
    std::uint32_t offset = 0;

    // Next datagram into (*p, *n); false when none is left.
    bool next(const std::uint8_t** p, std::uint32_t* n) noexcept {
        assert(p != nullptr && n != nullptr);
        if (offset >= len) return false;
        const std::uint32_t left = len - offset;
        const std::uint32_t take = segment == 0 || segment > left ? left : segment;
        *p = data + offset;
        *n = take;
        offset += take;
        assert(offset <= len);
        return true;
    }
};

inline UdpSlotDatagrams udp_slot_datagrams(const UdpSlot& s) noexcept {
    assert((s.flags & kUdpSlotTruncated) == 0 || s.len == 0);
    return UdpSlotDatagrams{s.buf, s.len, s.segment, 0};
}

}  // namespace bolt::net
