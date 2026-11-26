// bolt/net/bolt_udp_limits.h — compile-time bounds of the batched datagram
// layer (G2CHK-1271). Kind B: per-call scratch sized at compile time.
#pragma once

#include <cstdint>

namespace bolt::net::limits {

// Datagrams one udp_recv_batch / udp_send_batch call moves at most.
inline constexpr std::uint32_t kUdpBatchMax = 64;
// Segments one GSO (UDP_SEGMENT) message carries at most (kernel UDP_MAX_SEGMENTS).
inline constexpr std::uint32_t kUdpGsoSegmentsMax = 64;
// Payload bytes one GSO message carries at most (one IP datagram).
inline constexpr std::uint32_t kUdpGsoBytesMax = 65000;

static_assert(kUdpBatchMax > 0 && kUdpBatchMax <= 1024);
static_assert(kUdpGsoSegmentsMax > 0 && kUdpGsoSegmentsMax <= 128);

}  // namespace bolt::net::limits
