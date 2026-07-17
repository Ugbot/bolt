// bolt/net/bolt_transport_frame.h — the 32-byte datagram header that carries a
// (fragment of a) message across a UDP stream edge: message-oriented framing +
// app-layer fragmentation + CRC32C integrity.
//
// Moved up from chukonu::transport into bolt::net so the transport primitive is
// shared by the control plane (gestalt replication) and the compute plane
// (chukonu) without a cross-dependency. chukonu/transport/frame.h re-exports
// these into chukonu::transport for source compatibility.
//
// The first 16 bytes (magic / version / flags / stream_id) and the 32-byte size
// + CRC are the stable wire prefix; the back half carries message fragmentation
// (msg_seq + frag_index/frag_count).
//
// Tiger Style: POD, explicit widths + layout, no I/O, no allocation, no
// platform headers. >=2 asserts per fn.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/io/bolt_crc32c.h"

namespace bolt::net {

// "Chukonu Bolt 17" — fast reject for non-transport UDP traffic on shared
// ports. Any datagram whose first 4 bytes differ is dropped.
constexpr std::uint32_t k_frame_magic   = 0xC4D4B017u;
constexpr std::uint16_t k_frame_version = 1u;

// Default payload MTU: 1500-byte Ethernet - 20 IP - 8 UDP - 32 frame = 1440
// bytes of application payload per datagram. Bounded so the IP layer never
// fragments (poison for tail latency).
constexpr std::uint32_t k_frame_mtu_payload = 1440u;

// Frame flags. v1 uses DATA / FRAGMENT / LAST_FRAG / ORDER; the rest are
// reserved for the reliability + security increments.
namespace frame_flag {
constexpr std::uint16_t k_data      = 1u << 0;  // carries application payload
constexpr std::uint16_t k_fragment  = 1u << 8;  // a fragment of a larger message
constexpr std::uint16_t k_last_frag = 1u << 9;  // the last fragment of a message
// Ordering policy. Set => the receiver delivers this stream's messages strictly
// in msg_seq order; clear => the receiver delivers each message the instant it
// completes (no head-of-line blocking).
constexpr std::uint16_t k_order     = 1u << 10;
}  // namespace frame_flag

// 32-byte datagram header. Trivially copyable; memcpy'd to/from the wire in
// host byte order (v1 assumes homogeneous little-endian peers).
struct FrameHeader {
    std::uint32_t magic;        // k_frame_magic
    std::uint16_t version;      // k_frame_version
    std::uint16_t flags;        // frame_flag::*
    std::uint64_t stream_id;    // logical stream identity (fragment edge)
    std::uint32_t msg_seq;      // per-stream message id
    std::uint16_t frag_index;   // fragment index, 0 .. frag_count-1
    std::uint16_t frag_count;   // total fragments in this message (>= 1)
    std::uint32_t payload_len;  // bytes of THIS fragment's payload (<= MTU)
    std::uint32_t checksum;     // CRC32C of THIS fragment's payload
};
static_assert(sizeof(FrameHeader) == 32, "frame header must be 32 bytes");
static_assert(__is_trivially_copyable(FrameHeader), "FrameHeader must be POD");

// Initialise a DATA frame header for fragment `idx` of `count` (count >= 1).
// `payload`/`len` are this fragment's bytes; the CRC is computed over them.
// `order` sets the k_order flag (in-order delivery) when true.
inline void frame_init_data(FrameHeader* h, std::uint64_t stream_id,
                            std::uint32_t msg_seq, std::uint16_t idx,
                            std::uint16_t count, const void* payload,
                            std::uint32_t len, bool order) noexcept {
    assert(h != nullptr);
    assert(count >= 1 && idx < count);
    assert(len <= k_frame_mtu_payload);
    assert(payload != nullptr || len == 0);
    h->magic       = k_frame_magic;
    h->version     = k_frame_version;
    h->flags       = static_cast<std::uint16_t>(
        frame_flag::k_data |
        (count > 1 ? frame_flag::k_fragment : 0u) |
        (idx + 1 == count ? frame_flag::k_last_frag : 0u) |
        (order ? frame_flag::k_order : 0u));
    h->stream_id   = stream_id;
    h->msg_seq     = msg_seq;
    h->frag_index  = idx;
    h->frag_count  = count;
    h->payload_len = len;
    h->checksum    = (len == 0) ? 0u
                                : bolt::io::crc32c(payload, len, 0u);
}

// Initialise a single-datagram CTRL frame (no k_data flag). CTRL frames are
// never fragmented — every control POD fits in one MTU.
inline void frame_init_ctrl(FrameHeader* h, std::uint64_t stream_id,
                            std::uint32_t msg_seq, const void* payload,
                            std::uint32_t len) noexcept {
    assert(h != nullptr);
    assert(payload != nullptr && len > 0u && len <= k_frame_mtu_payload);
    h->magic       = k_frame_magic;
    h->version     = k_frame_version;
    h->flags       = 0u;                 // CTRL: k_data clear
    h->stream_id   = stream_id;
    h->msg_seq     = msg_seq;
    h->frag_index  = 0u;
    h->frag_count  = 1u;
    h->payload_len = len;
    h->checksum    = bolt::io::crc32c(payload, len, 0u);
}

// Validate a received header + its payload: magic, version, sane fragment
// counts, payload length bound, and CRC32C over the payload. Pure check.
inline bool frame_validate(const FrameHeader* h, const void* payload,
                           std::uint32_t avail) noexcept {
    assert(h != nullptr);
    assert(payload != nullptr || avail == 0);
    if (h->magic != k_frame_magic) return false;
    if (h->version != k_frame_version) return false;
    if (h->frag_count == 0 || h->frag_index >= h->frag_count) return false;
    if (h->payload_len > k_frame_mtu_payload) return false;
    if (h->payload_len > avail) return false;
    const std::uint32_t crc =
        (h->payload_len == 0) ? 0u
                              : bolt::io::crc32c(payload, h->payload_len, 0u);
    return crc == h->checksum;
}

}  // namespace bolt::net
