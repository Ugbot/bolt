#include "bolt/net/replica_data_frame.h"
#include <gtest/gtest.h>
#include <array>
#include <vector>

namespace {
using namespace bolt::net;
constexpr std::uint8_t kPayload[]{9, 7, 5, 3, 1};
using Bytes = std::array<std::uint8_t, 256>;
ParsedFrame source_frame() {
    ParsedFrame source{};
    source.lsn = 0x0807060504030201ull; source.table_id = 0x12345678u;
    source.schema_fp = 0x1122334455667788ull; source.trace_id = 0x8877665544332211ull;
    source.kind = 2; source.payload = kPayload; source.payload_len = sizeof(kPayload);
    return source; // no prior raw serialization/frame_len needed
}
ReplicaDataFence source_fence() {
    ReplicaDataFence fence{};
    fence.data_view = 0x04030201u; fence.membership = 0x0605u; fence.primary = 7;
    fence.flags = kReplicaDataHasLane | kReplicaDataHasIncarnation;
    fence.source_lane = 0x0c0b0a09u; fence.table_incarnation = 0x100f0e0du;
    return fence;
}
void seal(Bytes& bytes, std::uint32_t length) {
    const auto checksum = frame_checksum(bytes.data() + kChecksumCoverStart,
                                         length - kChecksumCoverStart);
    for (unsigned i = 0; i < 8; ++i)
        bytes[kOffChecksum + i] = static_cast<std::uint8_t>(checksum >> (8 * i));
}
void expect_refused(const std::uint8_t* bytes, std::uint32_t length) {
    auto inner = source_frame(); inner.frame_len = 777;
    auto fence = source_fence();
    EXPECT_FALSE(replica_data_parse(bytes, length, inner, fence));
    EXPECT_EQ(inner.frame_len, 777u); EXPECT_EQ(inner.lsn, source_frame().lsn);
    EXPECT_EQ(inner.table_id, source_frame().table_id); EXPECT_EQ(inner.schema_fp, source_frame().schema_fp);
    EXPECT_EQ(inner.trace_id, source_frame().trace_id); EXPECT_EQ(inner.kind, 2);
    EXPECT_EQ(inner.payload, kPayload); EXPECT_EQ(inner.payload_len, sizeof(kPayload));
    EXPECT_EQ(fence.data_view, source_fence().data_view); EXPECT_EQ(fence.membership, source_fence().membership);
    EXPECT_EQ(fence.primary, 7); EXPECT_EQ(fence.flags, 3u);
    EXPECT_EQ(fence.source_lane, source_fence().source_lane);
    EXPECT_EQ(fence.table_incarnation, source_fence().table_incarnation);
}
TEST(ReplicaDataCodec, GoldenLittleEndianMetadataAndBorrowedInnerForEverySourceKind) {
    Bytes bytes{};
    constexpr std::uint8_t metadata[]{
        'R','W','F','1', 1,0,32,0, 1,2,3,4, 5,6,7,2,
        3,0,0,0, 9,10,11,12, 13,14,15,16, 0,0,0,0};
    const auto n = replica_data_encode(source_frame(), source_fence(), bytes.data(), static_cast<std::uint32_t>(bytes.size()));
    ASSERT_EQ(n, kFrameHeaderBytes + kReplicaDataHeaderBytes + sizeof(kPayload));
    EXPECT_EQ(std::memcmp(bytes.data() + kFrameHeaderBytes, metadata, sizeof(metadata)), 0);
    EXPECT_EQ(bytes[kOffLsn], 1); EXPECT_EQ(bytes[kOffLsn + 7], 8);
    for (std::uint8_t kind = 1; kind <= 10; ++kind) {
        auto source = source_frame(); source.kind = kind; source.frame_len = UINT32_MAX;
        ASSERT_EQ(replica_data_encode(source, source_fence(), bytes.data(), static_cast<std::uint32_t>(bytes.size())), n);
        ParsedFrame inner{}; ReplicaDataFence fence{};
        ASSERT_TRUE(replica_data_parse(bytes.data(), n, inner, fence));
        EXPECT_EQ(inner.kind, kind); EXPECT_EQ(inner.lsn, source.lsn);
        EXPECT_EQ(inner.table_id, source.table_id); EXPECT_EQ(inner.schema_fp, source.schema_fp);
        EXPECT_EQ(inner.trace_id, source.trace_id);
        EXPECT_EQ(inner.frame_len, kFrameHeaderBytes + sizeof(kPayload));
        EXPECT_EQ(inner.payload_len, sizeof(kPayload));
        EXPECT_EQ(inner.payload, bytes.data() + kFrameHeaderBytes + kReplicaDataHeaderBytes);
        EXPECT_EQ(std::memcmp(inner.payload, kPayload, sizeof(kPayload)), 0);
        EXPECT_EQ(fence.data_view, source_fence().data_view); EXPECT_EQ(fence.membership, source_fence().membership);
        EXPECT_EQ(fence.primary, 7); EXPECT_EQ(fence.flags, 3u);
        EXPECT_EQ(fence.source_lane, source_fence().source_lane);
        EXPECT_EQ(fence.table_incarnation, source_fence().table_incarnation);
    }
}
TEST(ReplicaDataCodec, InitialViewAndAbsentMetadataMakeNoInventedSourceClaim) {
    Bytes bytes{};
    for (std::uint32_t flags = 0; flags <= 3; ++flags) {
        auto source = source_frame(); source.kind = kWalTxnEnvelopeKind;
        source.payload = nullptr; source.payload_len = 0;
        ReplicaDataFence fence{}; fence.membership = 1; fence.flags = flags;
        fence.table_incarnation = (flags & kReplicaDataHasIncarnation) != 0 ? 9u : 0u;
        const auto n = replica_data_encode(source, fence, bytes.data(), static_cast<std::uint32_t>(bytes.size()));
        ASSERT_EQ(n, kFrameHeaderBytes + kReplicaDataHeaderBytes);
        ParsedFrame inner{}; ReplicaDataFence decoded{};
        ASSERT_TRUE(replica_data_parse(bytes.data(), n, inner, decoded));
        EXPECT_EQ(decoded.data_view, 0u); EXPECT_EQ(decoded.primary, 0);
        EXPECT_EQ(decoded.flags, flags); EXPECT_EQ(decoded.source_lane, 0u);
        EXPECT_EQ(decoded.table_incarnation, fence.table_incarnation);
        EXPECT_EQ(inner.frame_len, kFrameHeaderBytes); EXPECT_EQ(inner.payload, nullptr);
    }
}
TEST(ReplicaDataCodec, NormalFramesRemainReadableAndStrictDecoderRefusesOtherOuterClasses) {
    Bytes bytes{}; auto source = source_frame();
    const auto plain = frame_serialize(source.lsn, source.table_id, source.schema_fp,
        source.kind, source.payload, source.payload_len, bytes.data(), static_cast<std::uint32_t>(bytes.size()), source.trace_id);
    ParsedFrame decoded{}; ASSERT_TRUE(frame_parse(bytes.data(), plain, decoded));
    EXPECT_EQ(decoded.kind, source.kind); expect_refused(bytes.data(), plain);
    const auto n = replica_data_encode(source, source_fence(), bytes.data(), static_cast<std::uint32_t>(bytes.size()));
    ASSERT_NE(n, 0u);
    for (const std::uint8_t kind : {std::uint8_t{2}, std::uint8_t{0xf0}, std::uint8_t{0xfe}}) {
        bytes[kOffKind] = kind; seal(bytes, n); expect_refused(bytes.data(), n);
    }
}
TEST(ReplicaDataCodec, RechecksummedMalformedMetadataAndOriginsAreStillRefused) {
    Bytes good{};
    const auto n = replica_data_encode(source_frame(), source_fence(), good.data(), static_cast<std::uint32_t>(good.size()));
    ASSERT_NE(n, 0u);
    // Each mutation preserves exact integrity. Failure must be semantic validation.
    constexpr std::uint32_t offsets[]{0,1,2,3,4,5,6,7,16,17,18,19,28,29,30,31,12,14};
    for (const auto offset : offsets) {
        auto bytes = good;
        auto* meta = bytes.data() + kFrameHeaderBytes;
        if (offset == 12) { meta[12] = 0; meta[13] = 0; }
        else if (offset == 14) meta[14] = 8;
        else if (offset == 16) meta[16] = 7;
        else meta[offset] ^= 1;
        seal(bytes, n); expect_refused(bytes.data(), n);
    }
    for (const auto field : {kOffLsn, kOffTableId}) {
        auto bytes = good;
        std::memset(bytes.data() + field, 0, field == kOffLsn ? 8 : 4);
        seal(bytes, n); expect_refused(bytes.data(), n);
    }
    for (unsigned which = 0; which < 3; ++which) {
        auto bytes = good; auto* meta = bytes.data() + kFrameHeaderBytes;
        if (which == 0) meta[16] = kReplicaDataHasIncarnation; // missing lane flag, nonzero lane
        if (which == 1) meta[16] = kReplicaDataHasLane;        // missing incarnation flag
        if (which == 2) std::memset(meta + 24, 0, 4);         // present incarnation must be known
        seal(bytes, n); expect_refused(bytes.data(), n);
    }
}
TEST(ReplicaDataCodec, NestedProofHeartbeatAndUnknownSourceKindsAreRefusedBothWays) {
    Bytes good{}, refused{}; refused.fill(0xa5);
    const auto n = replica_data_encode(source_frame(), source_fence(), good.data(), static_cast<std::uint32_t>(good.size()));
    ASSERT_NE(n, 0u);
    for (const std::uint8_t kind : {std::uint8_t{0}, std::uint8_t{11}, std::uint8_t{0xf0},
                                  std::uint8_t{0xf1}, std::uint8_t{0xfe}, std::uint8_t{255}}) {
        auto source = source_frame(); source.kind = kind;
        EXPECT_EQ(replica_data_encode(source, source_fence(), refused.data(), static_cast<std::uint32_t>(refused.size())), 0u);
        auto bytes = good; bytes[kFrameHeaderBytes + 15] = kind;
        seal(bytes, n); expect_refused(bytes.data(), n);
    }
    for (const auto byte : refused) EXPECT_EQ(byte, 0xa5);
}
TEST(ReplicaDataCodec, ExactExtentsAndWrappedLengthsRefuseWithoutPartialDestinations) {
    Bytes bytes{};
    const auto n = replica_data_encode(source_frame(), source_fence(), bytes.data(), static_cast<std::uint32_t>(bytes.size()));
    ASSERT_NE(n, 0u); expect_refused(nullptr, n);
    for (std::uint32_t length = 0; length < n; ++length) expect_refused(bytes.data(), length);
    expect_refused(bytes.data(), n + 1);
    for (const auto field : {kOffFrameLen, kOffPayloadLen}) {
        auto broken = bytes;
        for (unsigned i = 0; i < 4; ++i) broken[field + i] = 0xff;
        seal(broken, n); expect_refused(broken.data(), n);
    }
    expect_refused(bytes.data(), UINT32_MAX); // exceeds cap before any body read
}
TEST(ReplicaDataCodec, EveryByteCorruptionIsDetected) {
    Bytes bytes{};
    const auto n = replica_data_encode(source_frame(), source_fence(), bytes.data(), static_cast<std::uint32_t>(bytes.size()));
    ASSERT_NE(n, 0u);
    for (std::uint32_t i = 0; i < n; ++i) {
        SCOPED_TRACE(i);
        bytes[i] ^= 1; expect_refused(bytes.data(), n); bytes[i] ^= 1;
    }
}
TEST(ReplicaDataCodec, InvalidEncodeCapacityPointerFenceAndOverlapLeaveBufferUntouched) {
    Bytes bytes{}; bytes.fill(0xa5);
    const auto source = source_frame(); const auto fence = source_fence();
    EXPECT_EQ(replica_data_encode(source, fence, nullptr, static_cast<std::uint32_t>(bytes.size())), 0u);
    EXPECT_EQ(replica_data_encode(source, fence, bytes.data(),
        kFrameHeaderBytes + kReplicaDataHeaderBytes + sizeof(kPayload) - 1), 0u);
    for (unsigned which = 0; which < 6; ++which) {
        auto bad = source;
        if (which == 0) bad.lsn = 0;
        if (which == 1) bad.table_id = 0;
        if (which == 2) bad.payload = nullptr;
        if (which == 3) bad.payload_len = UINT32_MAX;
        if (which == 4) bad.payload = bytes.data() + 1;
        if (which == 5) bad.payload = reinterpret_cast<const void*>(UINTPTR_MAX - 1);
        EXPECT_EQ(replica_data_encode(bad, fence, bytes.data(), static_cast<std::uint32_t>(bytes.size())), 0u);
    }
    for (unsigned which = 0; which < 6; ++which) {
        auto bad = fence;
        if (which == 0) bad.membership = 0;
        if (which == 1) bad.primary = 8;
        if (which == 2) bad.flags = 4;
        if (which == 3) bad.flags = kReplicaDataHasLane;
        if (which == 4) bad.flags = kReplicaDataHasIncarnation;
        if (which == 5) bad.table_incarnation = 0;
        EXPECT_EQ(replica_data_encode(source, bad, bytes.data(), static_cast<std::uint32_t>(bytes.size())), 0u);
    }
    for (const auto byte : bytes) EXPECT_EQ(byte, 0xa5);
}
TEST(ReplicaDataCodec, AdjacentNonoverlappingInputIsAllowedAndPayloadRemainsBorrowed) {
    Bytes bytes{}; bytes.fill(0xa5);
    auto source = source_frame();
    constexpr auto total = kFrameHeaderBytes + kReplicaDataHeaderBytes + sizeof(kPayload);
    source.payload = bytes.data() + total;
    const auto n = replica_data_encode(source, source_fence(), bytes.data(), static_cast<std::uint32_t>(bytes.size()));
    ASSERT_EQ(n, total);
    ParsedFrame inner{}; ReplicaDataFence fence{};
    ASSERT_TRUE(replica_data_parse(bytes.data(), n, inner, fence));
    const auto* payload = static_cast<const std::uint8_t*>(inner.payload);
    EXPECT_EQ(payload[0], 0xa5);
    bytes[kFrameHeaderBytes + kReplicaDataHeaderBytes] = 0x6b;
    EXPECT_EQ(payload[0], 0x6b); // borrowed view, no hidden allocation/copy on parse
}

TEST(ReplicaDataCodec, FullTransportBudgetIncludesBothHeadersAndRefusesOneByteOver) {
    std::vector<std::uint8_t> payload(kReplicaDataMaxPayload + 1, 0x5b);
    std::vector<std::uint8_t> bytes(kWalMaxFrameBytes + kReplicaDataHeaderBytes); // safe cap-error control
    auto source = source_frame(); source.payload = payload.data(); source.payload_len = kReplicaDataMaxPayload;
    ASSERT_EQ(replica_data_encode(source, source_fence(), bytes.data(), kWalMaxFrameBytes), kWalMaxFrameBytes);
    ParsedFrame inner{}; ReplicaDataFence fence{};
    ASSERT_TRUE(replica_data_parse(bytes.data(), kWalMaxFrameBytes, inner, fence));
    EXPECT_EQ(inner.payload_len, kReplicaDataMaxPayload);
    const auto* borrowed = static_cast<const std::uint8_t*>(inner.payload);
    EXPECT_EQ(borrowed[0], 0x5b); EXPECT_EQ(borrowed[kReplicaDataMaxPayload - 1], 0x5b);
    const auto before = frame_checksum(bytes.data(), static_cast<std::uint32_t>(bytes.size()));
    ++source.payload_len;
    EXPECT_EQ(replica_data_encode(source, source_fence(), bytes.data(), static_cast<std::uint32_t>(bytes.size())), 0u);
    EXPECT_EQ(frame_checksum(bytes.data(), static_cast<std::uint32_t>(bytes.size())), before);
}
} // namespace
