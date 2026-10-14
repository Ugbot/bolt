#include <gtest/gtest.h>
#include "bolt/net/replica_proof_frame.h"
#include <array>
#include <cstring>
#include <limits>
namespace {
using namespace bolt::net;
ReplicaProofFrame fixture(ReplicaProofKind kind = ReplicaProofKind::Begin) {
    ReplicaProofFrame f{};
    f.kind = kind; f.source_epoch[0] = 0xa1; f.schema[0] = 0xb2;
    f.assignment = 0x0102030405060708ull; f.manifest = 3; f.replay = 4;
    f.domain = 2; f.table = 7; f.incarnation = 9;
    f.base_lsn = 10; f.cut_lsn = 100;
    f.synced_lsn = kind == ReplicaProofKind::End ? 100 : 0;
    return f;
}
TEST(ReplicaProofCodec, IndependentLittleEndianHeaderAndBorrowedPayload) {
    auto f = fixture(ReplicaProofKind::Entry);
    constexpr std::uint8_t payload[]{0, 0xff, 0x80, 'A', 0};
    f.payload = {payload, sizeof(payload)}; f.ordinal = 1;
    std::array<std::uint8_t, 512> bytes{};
    const auto length = replica_proof_encode(f, bytes.data(), bytes.size());
    ASSERT_EQ(length, kReplicaProofHeaderBytes + sizeof(payload));
    const std::uint8_t prefix[]{0x47,0x52,0x50,0x32,2,0,2,0,1,1,0,0,0,0,0,0};
    EXPECT_EQ(std::memcmp(bytes.data(), prefix, sizeof(prefix)), 0); // 257 bytes.
    const std::uint8_t assignment[]{8,7,6,5,4,3,2,1};
    EXPECT_EQ(std::memcmp(bytes.data() + 48, assignment, sizeof(assignment)), 0);
    EXPECT_EQ(bytes[16], 0xa1); EXPECT_EQ(bytes[76], 7); EXPECT_EQ(bytes[80], 9);
    EXPECT_EQ(bytes[88], 10); EXPECT_EQ(bytes[96], 100); EXPECT_EQ(bytes[112], 1);
    EXPECT_EQ(bytes[192], 5); EXPECT_EQ(bytes[128], 0xb2);
    ReplicaProofFrame parsed{};
    ASSERT_TRUE(replica_proof_parse({bytes.data(), length}, parsed));
    EXPECT_EQ(parsed.assignment, f.assignment); EXPECT_EQ(parsed.table, 7u);
    EXPECT_EQ(parsed.payload.data, bytes.data() + kReplicaProofHeaderBytes);
    EXPECT_EQ(std::memcmp(parsed.payload.data, payload, sizeof(payload)), 0);
}
TEST(ReplicaProofCodec, EveryTruncationAndCorruptionPreservesDestination) {
    auto f = fixture(ReplicaProofKind::Entry);
    constexpr std::uint8_t payload[]{1,0,0xff}; f.payload = {payload, sizeof(payload)}; f.ordinal = 1;
    std::array<std::uint8_t, 512> bytes{};
    const auto length = replica_proof_encode(f, bytes.data(), bytes.size()); ASSERT_NE(length, 0u);
    ReplicaProofFrame out{}; out.assignment = 777;
    for (std::uint32_t n = 0; n < length; ++n) {
        EXPECT_FALSE(replica_proof_parse({bytes.data(), n}, out)); EXPECT_EQ(out.assignment, 777u);
    }
    EXPECT_FALSE(replica_proof_parse({nullptr, length}, out));
    EXPECT_FALSE(replica_proof_parse({bytes.data(), length + 1}, out));
    for (std::uint32_t i = 0; i < length; ++i) {
        bytes[i] ^= 1;
        EXPECT_FALSE(replica_proof_parse({bytes.data(), length}, out)) << i;
        EXPECT_EQ(out.assignment, 777u); bytes[i] ^= 1;
    }
}
TEST(ReplicaProofCodec, RefusalLeavesEncodingBufferUntouched) {
    std::array<std::uint8_t, 512> bytes{}; bytes.fill(0x6b);
    auto f = fixture();
    EXPECT_EQ(replica_proof_encode(f, bytes.data(), kReplicaProofHeaderBytes - 1), 0u);
    f.replay = 0; EXPECT_EQ(replica_proof_encode(f, bytes.data(), bytes.size()), 0u);
    f = fixture(); f.kind = static_cast<ReplicaProofKind>(99);
    EXPECT_EQ(replica_proof_encode(f, bytes.data(), bytes.size()), 0u);
    f = fixture(); f.base_lsn = 101; EXPECT_EQ(replica_proof_encode(f, bytes.data(), bytes.size()), 0u);
    f = fixture(); f.payload = {bytes.data(), kReplicaProofMaxPayload + 1};
    EXPECT_EQ(replica_proof_encode(f, bytes.data(), bytes.size()), 0u);
    f = fixture(ReplicaProofKind::End); f.count = f.ordinal = kReplicaProofMaxEntries + 1;
    EXPECT_EQ(replica_proof_encode(f, bytes.data(), bytes.size()), 0u);
    f = fixture(ReplicaProofKind::End); f.synced_lsn = 99;
    EXPECT_EQ(replica_proof_encode(f, bytes.data(), bytes.size()), 0u);
    for (const auto byte : bytes) EXPECT_EQ(byte, 0x6b);
}
TEST(ReplicaProofCodec, AckSeparatesReceivedDurableAndAppliedProgress) {
    auto f = fixture(ReplicaProofKind::Ack);
    f.ordinal = f.received_ordinal = 5; f.durable_ordinal = 4; f.applied_ordinal = 2;
    std::array<std::uint8_t, 512> bytes{};
    const auto n = replica_proof_encode(f, bytes.data(), bytes.size()); ASSERT_NE(n, 0u);
    ReplicaProofFrame out{}; ASSERT_TRUE(replica_proof_parse({bytes.data(), n}, out));
    EXPECT_EQ(out.received_ordinal, 5u); EXPECT_EQ(out.durable_ordinal, 4u); EXPECT_EQ(out.applied_ordinal, 2u);
    f.applied_ordinal = 5; EXPECT_EQ(replica_proof_encode(f, bytes.data(), bytes.size()), 0u);
    f.applied_ordinal = 2; f.durable_ordinal = 6;
    EXPECT_EQ(replica_proof_encode(f, bytes.data(), bytes.size()), 0u);
}
TEST(ReplicaProofCodec, UnusedPerKindFieldsHaveOneCanonicalSpelling) {
    std::array<std::uint8_t, 512> bytes{};
    for (const auto kind : {ReplicaProofKind::Begin, ReplicaProofKind::Entry, ReplicaProofKind::Decline}) {
        auto f = fixture(kind);
        constexpr std::uint8_t data[]{1};
        if (kind == ReplicaProofKind::Entry) { f.ordinal = 1; f.payload = {data, 1}; }
        if (kind == ReplicaProofKind::Decline) f.reason = ReplicaProofReason::Busy;
        ASSERT_NE(replica_proof_encode(f, bytes.data(), bytes.size()), 0u);
        f.count = 1; EXPECT_EQ(replica_proof_encode(f, bytes.data(), bytes.size()), 0u); f.count = 0;
        f.received_ordinal = 1; EXPECT_EQ(replica_proof_encode(f, bytes.data(), bytes.size()), 0u); f.received_ordinal = 0;
        f.synced_lsn = 100; EXPECT_EQ(replica_proof_encode(f, bytes.data(), bytes.size()), 0u);
    }
    auto ddl = fixture(); ddl.table = 0; ddl.domain = 0;
    EXPECT_EQ(replica_proof_encode(ddl, bytes.data(), bytes.size()), 0u);
    ddl.incarnation = 0; std::memset(ddl.schema, 0, 32);
    EXPECT_NE(replica_proof_encode(ddl, bytes.data(), bytes.size()), 0u);
    ddl.schema[0] = 1; EXPECT_EQ(replica_proof_encode(ddl, bytes.data(), bytes.size()), 0u);
    for (const auto kind : {ReplicaProofKind::Begin, ReplicaProofKind::Entry, ReplicaProofKind::End,
                           ReplicaProofKind::Decline, ReplicaProofKind::Ack}) {
        auto data = fixture(kind);
        constexpr std::uint8_t payload[]{1};
        if (kind == ReplicaProofKind::Entry) { data.ordinal = 1; data.payload = {payload, 1}; }
        if (kind == ReplicaProofKind::Decline) data.reason = ReplicaProofReason::Busy;
        const auto length = replica_proof_encode(data, bytes.data(), bytes.size()); ASSERT_NE(length, 0u);
        // Invalid relation with otherwise exact valid integrity: not a random
        // checksum failure. Every kind must use a data domain for table7.
        replica_proof_detail::write_le(bytes.data() + 72, std::uint32_t{0});
        replica_proof_detail::integrity(bytes.data(), 220,
            {bytes.data() + kReplicaProofHeaderBytes, data.payload.size}, bytes.data() + 220);
        ReplicaProofFrame destination{}; destination.assignment = 777;
        EXPECT_FALSE(replica_proof_parse({bytes.data(), length}, destination));
        EXPECT_EQ(destination.assignment, 777u);
        data.domain = 0; EXPECT_EQ(replica_proof_encode(data, bytes.data(), bytes.size()), 0u);
    }
}
TEST(ReplicaProofCodec, InitialChainBindsCompleteBegin) {
    std::array<std::uint8_t, 512> bytes{};
    auto f = fixture(); std::uint8_t first[32]{}, other[32]{};
    auto n = replica_proof_encode(f, bytes.data(), bytes.size()); ASSERT_NE(n, 0u);
    ASSERT_TRUE(replica_proof_seed({bytes.data(), n}, first));
    ++f.manifest; n = replica_proof_encode(f, bytes.data(), bytes.size()); ASSERT_NE(n, 0u);
    ASSERT_TRUE(replica_proof_seed({bytes.data(), n}, other));
    EXPECT_NE(std::memcmp(first, other, 32), 0);
    f = fixture(ReplicaProofKind::End); n = replica_proof_encode(f, bytes.data(), bytes.size()); ASSERT_NE(n, 0u);
    EXPECT_FALSE(replica_proof_seed({bytes.data(), n}, other));
}
TEST(ReplicaProofCodec, ChainBindsOrdinalAndDomain) {
    auto f = fixture(ReplicaProofKind::Entry);
    constexpr std::uint8_t first[]{10}, second[]{90};
    std::array<std::uint8_t, 512> bytes{};
    std::uint8_t chain0[32]{}, chain1[32]{}, chain2[32]{}, changed[32]{};
    f.ordinal = 1; f.payload = {first, 1};
    auto n = replica_proof_encode(f, bytes.data(), bytes.size()); ASSERT_NE(n, 0u);
    replica_proof_chain(chain0, {bytes.data(), n}, chain1);
    f.ordinal = 2; f.payload = {second, 1}; std::memcpy(f.chain, chain1, 32);
    n = replica_proof_encode(f, bytes.data(), bytes.size()); ASSERT_NE(n, 0u);
    replica_proof_chain(chain1, {bytes.data(), n}, chain2);
    EXPECT_NE(std::memcmp(chain1, chain2, 32), 0);
    f.domain = 3; n = replica_proof_encode(f, bytes.data(), bytes.size()); ASSERT_NE(n, 0u);
    replica_proof_chain(chain1, {bytes.data(), n}, changed);
    EXPECT_NE(std::memcmp(changed, chain2, 32), 0);
    f = fixture(ReplicaProofKind::End); f.count = f.ordinal = 2; std::memcpy(f.chain, chain2, 32);
    EXPECT_NE(replica_proof_encode(f, bytes.data(), bytes.size()), 0u);
    f = fixture(ReplicaProofKind::Decline); f.reason = ReplicaProofReason::HistoryUnavailable;
    EXPECT_NE(replica_proof_encode(f, bytes.data(), bytes.size()), 0u);
}
}  // namespace
