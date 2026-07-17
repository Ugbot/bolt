// bolt/net/wal_frag.h — DS4 WAL frame ⇄ fragment framing (moved up into bolt::net).
//
// A whole WAL frame (wal_frame.h; 37-byte header + payload) can exceed one UDP
// MTU, so it rides chukonu's transport fragment layer (chukonu/transport/
// frame.h): each datagram is a 32-byte FrameHeader + ≤1440 payload bytes, with
// a per-fragment CRC32C, sequenced by (stream_id, msg_seq, frag_index). This
// is the "use the chukonu transport" path — the SAME framing the mesh's
// DocReplicator uses, applied to raw WAL-frame bytes instead of a BoltBatch.
//
// Header-only + shared by the producer (WalReplicator, gestalt) and the
// consumer (chukonu_node receiver + WalReceiver). Single reassembly impl → no
// drift. Reassembly is single-threaded per receiver (one recv loop) → no locks.
//
// Tiger Style: caller-owned reassembly buffer (no per-message alloc), fixed
// slots, bounded fragment count, ≥2 asserts per non-trivial fn.
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/net/bolt_transport_frame.h"
#include "bolt/net/bolt_udp.h"
#include "bolt/net/wal_frame.h"

namespace bolt::net {

// Max fragments a kWalMaxFrameBytes frame can split into (ceil at MTU).
static constexpr uint32_t kWalFragMax =
    (kWalMaxFrameBytes + bolt::net::k_frame_mtu_payload - 1u) /
    bolt::net::k_frame_mtu_payload;

// Concurrent reassembly slots — enough for interleaved fragments from the
// DDL lane + all data lanes (kMaxApplyShards). One in-flight message per lane.
static constexpr uint32_t kWalReasmSlots = 8u;

// ── sender ──────────────────────────────────────────────────────────────────
// Fragment `frame[0..frame_len)` and send each fragment to `dst`. `order` sets
// the transport's in-order-delivery flag. Returns false on a null/oversized
// frame or any datagram send error. One datagram per fragment; a small stack
// buffer per datagram (no heap). Thread-safe on the socket (udp_send_to wraps
// a single sendto); callers must give each message a unique `msg_seq` per dst.
inline bool wal_frag_send(const bolt::net::UdpEndpoint* ep,
                          bolt::net::UdpAddr dst, uint64_t stream_id,
                          uint32_t msg_seq, const uint8_t* frame,
                          uint32_t frame_len, bool order) noexcept {
    assert(ep != nullptr);
    assert(frame != nullptr);
    if (ep == nullptr || frame == nullptr) return false;
    if (frame_len == 0u || frame_len > kWalMaxFrameBytes) return false;

    const uint32_t mtu   = bolt::net::k_frame_mtu_payload;
    const uint32_t count = (frame_len + mtu - 1u) / mtu;
    if (count == 0u || count > 0xFFFFu) return false;   // frag_count is u16

    uint8_t dg[sizeof(bolt::net::FrameHeader) + bolt::net::k_frame_mtu_payload];
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t off = i * mtu;
        const uint32_t len = (frame_len - off < mtu) ? (frame_len - off) : mtu;
        bolt::net::FrameHeader h{};
        bolt::net::frame_init_data(
            &h, stream_id, msg_seq, static_cast<uint16_t>(i),
            static_cast<uint16_t>(count), frame + off, len, order);
        std::memcpy(dg, &h, sizeof(h));
        std::memcpy(dg + sizeof(h), frame + off, len);
        if (bolt::net::udp_send_to(ep, dst, dg, sizeof(h) + len) < 0) {
            return false;
        }
    }
    return true;
}

// ── receiver reassembly ───────────────────────────────────────────────────
enum class WalReasmStatus : uint8_t {
    kNeedMore = 0,   // fragment absorbed; message not complete yet
    kComplete = 1,   // a full frame reassembled — *out valid until next ingest
    kIgnored  = 2,   // malformed/duplicate/corrupt datagram (idempotent skip)
    kError    = 3,   // no free slot, or oversized/parse failure on completion
};

struct WalReasmSlot {
    uint8_t  in_use;
    uint8_t  _pad[1];
    uint16_t frag_count;
    uint32_t msg_seq;
    uint32_t seen;        // distinct fragments placed
    uint32_t total_len;   // resolved when the last fragment arrives (0 = unknown)
    uint8_t  seen_bits[(kWalFragMax + 7u) / 8u];
};

// Reassembler over a caller-owned buffer split into kWalReasmSlots regions.
struct WalReasm {
    uint8_t*     buf;       // slot_cap * kWalReasmSlots bytes (caller-owned)
    uint32_t     slot_cap;  // >= kWalMaxFrameBytes per slot
    WalReasmSlot slots[kWalReasmSlots];
};

inline void wal_reasm_init(WalReasm* r, uint8_t* buf, uint32_t buf_len) noexcept {
    assert(r != nullptr);
    assert(buf != nullptr);
    std::memset(r, 0, sizeof(*r));
    r->buf      = buf;
    r->slot_cap = buf_len / kWalReasmSlots;
    assert(r->slot_cap >= kWalMaxFrameBytes);
}

// Absorb one datagram (32-byte FrameHeader + payload). On the datagram that
// completes a message, frame_parse the reassembled bytes into `out` (out.payload
// borrows the slot buffer — valid until this slot is reused) and free the slot.
inline WalReasmStatus wal_reasm_ingest(WalReasm* r, const uint8_t* dgram,
                                       uint32_t dlen, ParsedFrame* out) noexcept {
    assert(r != nullptr);
    assert(dgram != nullptr);
    namespace ct = bolt::net;
    if (dlen < sizeof(ct::FrameHeader)) return WalReasmStatus::kIgnored;

    ct::FrameHeader h{};
    std::memcpy(&h, dgram, sizeof(h));
    const uint8_t*  payload = dgram + sizeof(h);
    const uint32_t  avail   = dlen - static_cast<uint32_t>(sizeof(h));
    if (!ct::frame_validate(&h, payload, avail)) return WalReasmStatus::kIgnored;
    if (h.frag_count > kWalFragMax)              return WalReasmStatus::kError;

    // Find the slot for this msg_seq, or claim a free one.
    WalReasmSlot* s = nullptr;
    for (uint32_t i = 0; i < kWalReasmSlots; ++i) {
        if (r->slots[i].in_use && r->slots[i].msg_seq == h.msg_seq) { s = &r->slots[i]; break; }
    }
    if (s == nullptr) {
        for (uint32_t i = 0; i < kWalReasmSlots; ++i) {
            if (!r->slots[i].in_use) { s = &r->slots[i]; break; }
        }
        if (s == nullptr) return WalReasmStatus::kError;   // all slots busy
        std::memset(s, 0, sizeof(*s));
        s->in_use     = 1u;
        s->msg_seq    = h.msg_seq;
        s->frag_count = h.frag_count;
    }
    if (s->frag_count != h.frag_count) return WalReasmStatus::kError;

    const uint32_t mtu = ct::k_frame_mtu_payload;
    const uint32_t off = static_cast<uint32_t>(h.frag_index) * mtu;
    if (off + h.payload_len > r->slot_cap) return WalReasmStatus::kError;

    const uint32_t bit = h.frag_index;
    if ((s->seen_bits[bit >> 3] & (1u << (bit & 7u))) == 0u) {
        s->seen_bits[bit >> 3] |= static_cast<uint8_t>(1u << (bit & 7u));
        uint8_t* slot_buf = r->buf + (static_cast<uint32_t>(s - r->slots) * r->slot_cap);
        std::memcpy(slot_buf + off, payload, h.payload_len);
        s->seen++;
        if ((h.flags & ct::frame_flag::k_last_frag) != 0u || h.frag_count == 1u) {
            s->total_len = off + h.payload_len;
        }
    }

    if (s->seen != s->frag_count) return WalReasmStatus::kNeedMore;

    // Complete — parse + verify the reassembled frame, then free the slot.
    uint8_t* slot_buf = r->buf + (static_cast<uint32_t>(s - r->slots) * r->slot_cap);
    const bool ok = (out != nullptr) && frame_parse(slot_buf, s->total_len, *out);
    s->in_use = 0u;
    return ok ? WalReasmStatus::kComplete : WalReasmStatus::kError;
}

}  // namespace bolt::net
