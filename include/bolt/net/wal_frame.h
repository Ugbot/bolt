// bolt/net/wal_frame.h — DS4 shared WAL-replication wire format (moved up from
// gestalt::cluster into bolt::net so control + compute planes share one copy).
//
// The SINGLE source of truth for the replication frame + ACK layout, shared by
// the producer (WalReplicator, gestalt tree) and the consumer (WalReceiver in
// gestalt + the chukonu_node replica receiver). Dependency-free on purpose —
// only <cstdint>/<cstring>/<cassert> — so the chukonu node can include it with
// ZERO gestalt→chukonu build coupling and there is no second copy of the format
// to drift.
//
// Frame (little-endian). A per-frame checksum (FNV-1a-64 over everything after
// it) is the TigerBeetle end-to-end integrity invariant — it catches a
// serialize bug or a corrupt buffer that the transport's own CRC can't (the
// transport CRC covers only one hop; this covers producer→apply):
//   0   frame_len:u32   total bytes (header + payload), incl. this field
//   4   checksum:u64    FNV-1a-64 over buf[12 .. frame_len)
//   12  lsn:u64         PRIMARY WAL lsn (honored on apply → idempotent replay)
//   20  table_id:u32    marbledb tid (1..0xFFFF)
//   24  schema_fp:u64   schema fingerprint
//   32  kind:u8         marbledb::wal::EntryKind value
//   33  payload_len:u32
//   37  payload:bytes   bolt-wire EntryView payload
//   Header = 4+8+8+4+8+1+4 = 37 bytes (kFrameHeaderBytes).
//
// ACK (little-endian):
//   0  magic:u32 = 0xACE2ACE2
//   4  lsn:u64
//   12 applied:u8
//   13 _pad:u8×3
//   = 16 bytes (kAckFrameBytes).
//
// Tiger Style: bounded, no heap, no exceptions, ≥2 asserts per non-trivial fn.
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt::net {

// ── wire constants ─────────────────────────────────────────────────────────
static constexpr uint32_t kFrameHeaderBytes    = 37u;              // see layout
static constexpr uint32_t kAckMagic            = 0xACE2ACE2u;
static constexpr uint32_t kAckFrameBytes       = 16u;              // 4+8+1+3
static constexpr uint32_t kRecvMaxPayloadBytes = 64u * 1024u * 1024u;  // 64 MiB

// DS4 transport caps. A whole serialized WAL frame (header + payload) is
// bounded to kWalMaxFrameBytes for the fragment send/reassemble path; larger
// frames are dropped-and-counted loudly (a Phase-2 chunking follow-up), never
// silently truncated. kWalStreamId is the fixed logical stream every WAL frame
// rides (the destination UdpAddr distinguishes replicas).
static constexpr uint32_t kWalMaxFrameBytes = 4u * 1024u * 1024u;   // 4 MiB
static constexpr uint64_t kWalStreamId      = 0x57414C5F44533400ull; // "WAL_DS4\0"

// Field offsets (single source; used by serialize/parse below).
static constexpr uint32_t kOffFrameLen   = 0u;
static constexpr uint32_t kOffChecksum   = 4u;
static constexpr uint32_t kOffLsn        = 12u;
static constexpr uint32_t kOffTableId    = 20u;
static constexpr uint32_t kOffSchemaFp   = 24u;
static constexpr uint32_t kOffKind       = 32u;
static constexpr uint32_t kOffPayloadLen = 33u;
static constexpr uint32_t kChecksumCoverStart = kOffLsn;  // checksum covers [12, frame_len)

// ── ParsedFrame — result of frame_parse(); payload borrows the caller buffer ─
struct ParsedFrame {
    uint32_t    frame_len;
    uint64_t    lsn;
    uint32_t    table_id;
    uint64_t    schema_fp;
    uint8_t     kind;
    uint32_t    payload_len;
    const void* payload;      // points into the caller's buffer; not owned
};

// ── little-endian read/write helpers (no alignment assumed) ─────────────────
namespace wf_detail {
inline uint32_t rd32(const uint8_t* p) noexcept { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t rd64(const uint8_t* p) noexcept { uint64_t v; std::memcpy(&v, p, 8); return v; }
inline void     wr32(uint8_t* p, uint32_t v) noexcept { std::memcpy(p, &v, 4); }
inline void     wr64(uint8_t* p, uint64_t v) noexcept { std::memcpy(p, &v, 8); }
}  // namespace wf_detail

// ── frame_checksum — FNV-1a-64 over data[0..n). Non-cryptographic integrity;
// tamper-resistance is the transport's Noise layer, corruption detection is
// this. Bounded O(n); off the query hot path. ────────────────────────────────
inline uint64_t frame_checksum(const uint8_t* data, uint32_t n) noexcept {
    assert(data != nullptr || n == 0u);
    uint64_t h = 0xcbf29ce484222325ull;              // FNV offset basis
    for (uint32_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(data[i]);
        h *= 0x100000001b3ull;                       // FNV prime
    }
    return h;
}

// ── frame_serialize — write a frame into out[0..out_cap). Returns the frame
// length, or 0 if out is null / too small. ──────────────────────────────────
inline uint32_t frame_serialize(uint64_t lsn, uint32_t table_id, uint64_t schema_fp,
                                uint8_t kind, const void* payload, uint32_t payload_len,
                                uint8_t* out, uint32_t out_cap) noexcept {
    assert(payload != nullptr || payload_len == 0u);
    const uint32_t frame_len = kFrameHeaderBytes + payload_len;
    if (out == nullptr || frame_len < kFrameHeaderBytes || frame_len > out_cap) return 0u;

    wf_detail::wr32(out + kOffFrameLen, frame_len);
    // checksum written last, once the body is in place
    wf_detail::wr64(out + kOffLsn,        lsn);
    wf_detail::wr32(out + kOffTableId,    table_id);
    wf_detail::wr64(out + kOffSchemaFp,   schema_fp);
    out[kOffKind] = kind;
    wf_detail::wr32(out + kOffPayloadLen, payload_len);
    if (payload_len > 0u && payload != nullptr) {
        std::memcpy(out + kFrameHeaderBytes, payload, payload_len);
    }
    const uint64_t ck = frame_checksum(out + kChecksumCoverStart,
                                       frame_len - kChecksumCoverStart);
    wf_detail::wr64(out + kOffChecksum, ck);
    return frame_len;
}

// ── frame_parse — parse + verify a frame from buf[0..len). Returns true on a
// well-formed, checksum-valid frame; out.payload borrows buf. ────────────────
inline bool frame_parse(const uint8_t* buf, uint32_t len, ParsedFrame& out) noexcept {
    assert(buf != nullptr);
    if (buf == nullptr || len < kFrameHeaderBytes) return false;

    const uint32_t frame_len   = wf_detail::rd32(buf + kOffFrameLen);
    const uint64_t checksum    = wf_detail::rd64(buf + kOffChecksum);
    const uint64_t lsn         = wf_detail::rd64(buf + kOffLsn);
    const uint32_t table_id    = wf_detail::rd32(buf + kOffTableId);
    const uint64_t schema_fp   = wf_detail::rd64(buf + kOffSchemaFp);
    const uint8_t  kind        = buf[kOffKind];
    const uint32_t payload_len = wf_detail::rd32(buf + kOffPayloadLen);

    if (frame_len != kFrameHeaderBytes + payload_len) return false;
    if (frame_len > len)                              return false;
    if (payload_len > kRecvMaxPayloadBytes)           return false;
    if (table_id == 0u || table_id > 0xFFFFu)         return false;
    if (frame_checksum(buf + kChecksumCoverStart, frame_len - kChecksumCoverStart)
            != checksum)                              return false;

    out.frame_len   = frame_len;
    out.lsn         = lsn;
    out.table_id    = table_id;
    out.schema_fp   = schema_fp;
    out.kind        = kind;
    out.payload_len = payload_len;
    out.payload     = (payload_len > 0u) ? (buf + kFrameHeaderBytes) : nullptr;
    return true;
}

// ── frame_make_ack — write a 16-byte ACK into buf (caller: ≥ kAckFrameBytes) ─
inline void frame_make_ack(uint64_t lsn, bool applied, uint8_t* buf) noexcept {
    assert(buf != nullptr);
    wf_detail::wr32(buf + 0, kAckMagic);
    wf_detail::wr64(buf + 4, lsn);
    buf[12] = applied ? 1u : 0u;
    buf[13] = 0u;
    buf[14] = 0u;
    buf[15] = 0u;
}

}  // namespace bolt::net
