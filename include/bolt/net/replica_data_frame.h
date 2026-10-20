// Strict replica data-view envelope. This is syntax/integrity, NOT authority:
// receivers bind the decoded fence to trusted membership/view/source metadata.
// Legacy WAL frames and proof/control codecs remain unchanged. Strict peers
// must explicitly opt in; legacy apply must never receive this outer kind.
#pragma once
#include "bolt/net/wal_frame.h"
#include "bolt/wire/byte_reader.h"
#include <cstddef>
#include <type_traits>

namespace bolt::net {
inline constexpr std::uint8_t kReplicaDataOuterKind = 0xf1;
inline constexpr std::uint32_t kReplicaDataMagic = 0x31465752u; // LE "RWF1"
inline constexpr std::uint16_t kReplicaDataVersion = 1;
inline constexpr std::uint32_t kReplicaDataHeaderBytes = 32;
inline constexpr std::uint32_t kReplicaDataHasLane = 1u;
inline constexpr std::uint32_t kReplicaDataHasIncarnation = 2u;
inline constexpr std::uint8_t kReplicaDataMaxPrimaries = 8;
inline constexpr std::uint32_t kReplicaDataMaxPayload =
    kWalMaxFrameBytes - kFrameHeaderBytes - kReplicaDataHeaderBytes;
static_assert(kFrameHeaderBytes + kReplicaDataHeaderBytes < kWalMaxFrameBytes);
static_assert(kWalMaxFrameBytes <= kRecvMaxPayloadBytes);

// 32-byte LE metadata, immediately after the unchanged outer WAL header:
// 0 magic:u32; 4 version:u16; 6 header_bytes:u16; 8 data_view:u32;
// 12 membership:u16; 14 primary:u8; 15 original_kind:u8; 16 flags:u32;
// 20 source_lane:u32; 24 table_incarnation:u32; 28 reserved:u32=0.
// Lane 0 may name the global lane; the supplied flag distinguishes absence.
// Incarnation is supplied only when known and nonzero. Absence makes no claim.
struct ReplicaDataFence {
    std::uint32_t data_view = 0;
    std::uint16_t membership = 0;
    std::uint8_t primary = 0;
    std::uint32_t flags = 0;
    std::uint32_t source_lane = 0;
    std::uint32_t table_incarnation = 0;
};
namespace replica_data_detail {
template <class T> inline void write_le(std::uint8_t* bytes, T value) noexcept {
    static_assert(std::is_unsigned_v<T>);
    assert(bytes != nullptr); assert(sizeof(T) <= 8);
    for (std::uint32_t i = 0; i < sizeof(T); ++i)
        bytes[i] = static_cast<std::uint8_t>(value >> (i * 8));
}
inline bool valid_fence(const ReplicaDataFence& fence) noexcept {
    constexpr auto known = kReplicaDataHasLane | kReplicaDataHasIncarnation;
    if (fence.membership == 0 || fence.primary >= kReplicaDataMaxPrimaries ||
        (fence.flags & ~known) != 0) return false;
    const bool lane = (fence.flags & kReplicaDataHasLane) != 0;
    const bool incarnation = (fence.flags & kReplicaDataHasIncarnation) != 0;
    assert(known == 3u); assert(kReplicaDataMaxPrimaries > 0);
    return (lane || fence.source_lane == 0) &&
        (incarnation ? fence.table_incarnation != 0 : fence.table_incarnation == 0);
}
inline bool valid_source(const ParsedFrame& source) noexcept {
    // Stable Marble WAL wire classes 1..10. Transport/proof envelopes,
    // heartbeat and future unknown classes are never recursively unwrapped.
    assert(kWalTxnEnvelopeKind == 9u); assert(kReplicaDataMaxPayload < UINT32_MAX);
    return source.lsn != 0 && source.table_id != 0 && source.kind >= 1 && source.kind <= 10 &&
        source.payload_len <= kReplicaDataMaxPayload &&
        (source.payload != nullptr || source.payload_len == 0);
}
inline bool disjoint(const ParsedFrame& source, const std::uint8_t* out,
                     std::uint32_t length) noexcept {
    assert(out != nullptr); assert(length <= kWalMaxFrameBytes);
    const auto destination = reinterpret_cast<std::uintptr_t>(out);
    const auto input = reinterpret_cast<std::uintptr_t>(source.payload);
    if (destination > UINTPTR_MAX - length || input > UINTPTR_MAX - source.payload_len)
        return false;
    return source.payload_len == 0 || input + source.payload_len <= destination ||
        destination + length <= input;
}
inline void encode_header(const ParsedFrame& source, const ReplicaDataFence& fence,
                          std::uint8_t* out, std::uint32_t length) noexcept {
    assert(valid_source(source)); assert(valid_fence(fence));
    write_le(out + kOffFrameLen, length);
    write_le(out + kOffLsn, source.lsn); write_le(out + kOffTableId, source.table_id);
    write_le(out + kOffSchemaFp, source.schema_fp); write_le(out + kOffTraceId, source.trace_id);
    out[kOffKind] = kReplicaDataOuterKind;
    write_le(out + kOffPayloadLen, kReplicaDataHeaderBytes + source.payload_len);
    auto* meta = out + kFrameHeaderBytes;
    write_le(meta, kReplicaDataMagic); write_le(meta + 4, kReplicaDataVersion);
    write_le(meta + 6, static_cast<std::uint16_t>(kReplicaDataHeaderBytes));
    write_le(meta + 8, fence.data_view); write_le(meta + 12, fence.membership);
    meta[14] = fence.primary; meta[15] = source.kind;
    write_le(meta + 16, fence.flags); write_le(meta + 20, fence.source_lane);
    write_le(meta + 24, fence.table_incarnation); write_le(meta + 28, std::uint32_t{0});
}
inline bool decode_header(const std::uint8_t* bytes, std::uint32_t length,
                          ParsedFrame& source, ReplicaDataFence& fence) noexcept {
    if (bytes == nullptr || length < kFrameHeaderBytes + kReplicaDataHeaderBytes ||
        length > kWalMaxFrameBytes) return false;
    assert(bytes != nullptr); assert(length >= kFrameHeaderBytes);
    wire::ByteReader reader{bytes, length, 0};
    std::uint64_t checksum = 0;
    std::uint32_t outer_length = 0, payload = 0, magic = 0, reserved = 0;
    std::uint16_t version = 0, header = 0;
    std::uint8_t outer_kind = 0;
    if (!wire::read_le(reader, outer_length) || !wire::read_le(reader, checksum) ||
        !wire::read_le(reader, source.lsn) || !wire::read_le(reader, source.table_id) ||
        !wire::read_le(reader, source.schema_fp) || !wire::read_le(reader, source.trace_id) ||
        !wire::read_le(reader, outer_kind) || !wire::read_le(reader, payload) ||
        outer_kind != kReplicaDataOuterKind || outer_length != length ||
        payload != length - kFrameHeaderBytes) return false;
    if (!wire::read_le(reader, magic) || !wire::read_le(reader, version) ||
        !wire::read_le(reader, header) || !wire::read_le(reader, fence.data_view) ||
        !wire::read_le(reader, fence.membership) || !wire::read_le(reader, fence.primary) ||
        !wire::read_le(reader, source.kind) || !wire::read_le(reader, fence.flags) ||
        !wire::read_le(reader, fence.source_lane) || !wire::read_le(reader, fence.table_incarnation) ||
        !wire::read_le(reader, reserved) || magic != kReplicaDataMagic ||
        version != kReplicaDataVersion || header != kReplicaDataHeaderBytes || reserved != 0)
        return false;
    source.payload_len = payload - kReplicaDataHeaderBytes;
    source.frame_len = kFrameHeaderBytes + source.payload_len;
    source.payload = source.payload_len != 0 ? bytes + reader.offset : nullptr;
    return valid_source(source) && valid_fence(fence) &&
        frame_checksum(bytes + kChecksumCoverStart, length - kChecksumCoverStart) == checksum;
}
} // namespace replica_data_detail

// No allocation. Source frame_len is ignored: producers may construct source
// directly from EntryView. Caller supplies real storage and non-overlapping
// payload/output buffers. Invalid input/capacity/overlap leaves output unchanged.
// Total overhead is 45 + 32 bytes; the 4 MiB transport budget includes both.
inline std::uint32_t replica_data_encode(const ParsedFrame& source,
    const ReplicaDataFence& fence, std::uint8_t* out, std::uint32_t capacity) noexcept {
    if (out == nullptr || !replica_data_detail::valid_source(source) ||
        !replica_data_detail::valid_fence(fence)) return 0;
    const std::uint32_t length = kFrameHeaderBytes + kReplicaDataHeaderBytes + source.payload_len;
    if (capacity < length || !replica_data_detail::disjoint(source, out, length)) return 0;
    assert(length <= kWalMaxFrameBytes); assert(length >= kFrameHeaderBytes);
    replica_data_detail::encode_header(source, fence, out, length);
    if (source.payload_len != 0)
        std::memcpy(out + kFrameHeaderBytes + kReplicaDataHeaderBytes, source.payload, source.payload_len);
    replica_data_detail::write_le(out + kOffChecksum,
        frame_checksum(out + kChecksumCoverStart, length - kChecksumCoverStart));
    return length;
}

// Exact outer extent/checksum required. On success inner.payload borrows bytes;
// inner.frame_len describes its logical UNWRAPPED 45-byte-header frame. It does
// not describe serialized bytes in this buffer. Retain outer bytes separately
// for durable logging/retransmission; no raw inner frame/checksum is fabricated.
// Failure leaves BOTH destinations unchanged. Trusted fence authorization and
// any stronger source lane/incarnation requirements belong to the receiver.
inline bool replica_data_parse(const std::uint8_t* bytes, std::uint32_t length,
    ParsedFrame& inner, ReplicaDataFence& fence) noexcept {
    ParsedFrame candidate{};
    ReplicaDataFence candidate_fence{};
    if (!replica_data_detail::decode_header(bytes, length, candidate, candidate_fence)) return false;
    assert(candidate.frame_len == kFrameHeaderBytes + candidate.payload_len);
    assert(candidate.payload_len <= kReplicaDataMaxPayload);
    inner = candidate; fence = candidate_fence;
    return true;
}
} // namespace bolt::net
