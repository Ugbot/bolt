// test_bolt_wal_frame_tid.cpp — G2CHK-485: replication frames carry u32 table
// ids (marbledb ids span u32); 0 stays invalid.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "bolt/net/wal_frame.h"

namespace wf = bolt::net;

namespace {
bool round_trip(uint32_t tid, wf::ParsedFrame* out) {
    const uint8_t payload[5] = {1, 2, 3, 4, 5};
    std::vector<uint8_t> buf(wf::kFrameHeaderBytes + sizeof(payload));
    const uint32_t n = wf::frame_serialize(77u, tid, 0xABCDu, 3u, payload,
                                           sizeof(payload), buf.data(),
                                           static_cast<uint32_t>(buf.size()), 9u);
    if (n == 0u) return false;
    return wf::frame_parse(buf.data(), n, *out);
}
}  // namespace

TEST(BoltWalFrameTid, TableIdsPast16BitsRoundTrip) {
    for (uint32_t tid : {1u, 0xFFFFu, 0x10000u, 70001u, 0xFFFFFF00u - 1u}) {
        wf::ParsedFrame pf{};
        ASSERT_TRUE(round_trip(tid, &pf)) << tid;
        EXPECT_EQ(pf.table_id, tid);
        EXPECT_EQ(pf.lsn, 77u);
        EXPECT_EQ(pf.payload_len, 5u);
    }
}

TEST(BoltWalFrameTid, TableIdZeroRefused) {
    wf::ParsedFrame pf{};
    EXPECT_FALSE(round_trip(0u, &pf));
}
TEST(BoltWalFrameTid, SourceTransactionKeepsRawZeroHeaderAndLegacyRefuses) {
    std::vector<std::uint8_t> bytes(wf::kFrameHeaderBytes + 1);
    const std::uint8_t payload = 0xff;
    ASSERT_EQ(wf::frame_serialize(91, 0, 0, wf::kWalTxnEnvelopeKind, &payload, 1,
        bytes.data(), static_cast<std::uint32_t>(bytes.size())), bytes.size());
    wf::ParsedFrame out{}; out.table_id = 777;
    EXPECT_FALSE(wf::frame_parse(bytes.data(), static_cast<std::uint32_t>(bytes.size()), out));
    EXPECT_EQ(out.table_id, 777u);
    ASSERT_TRUE(wf::frame_parse_source_wal(bytes.data(), static_cast<std::uint32_t>(bytes.size()), out));
    EXPECT_EQ(out.table_id, 0u); EXPECT_EQ(out.schema_fp, 0u); EXPECT_EQ(out.lsn, 91u);
    EXPECT_EQ(out.kind, wf::kWalTxnEnvelopeKind); EXPECT_EQ(*static_cast<const std::uint8_t*>(out.payload), payload);
    // A nonzero-table legacy-shaped envelope is syntactically allowed, but
    // source authority is decided by the bound payload validator elsewhere.
    ASSERT_NE(wf::frame_serialize(91, 7, 12, wf::kWalTxnEnvelopeKind, &payload, 1,
        bytes.data(), static_cast<std::uint32_t>(bytes.size())), 0u);
    EXPECT_TRUE(wf::frame_parse_source_wal(bytes.data(), static_cast<std::uint32_t>(bytes.size()), out));
}
TEST(BoltWalFrameTid, SourceParserRejectsWrongZeroShapeAndTrailing) {
    std::vector<std::uint8_t> bytes(wf::kFrameHeaderBytes + 2);
    const std::uint8_t payload = 1; wf::ParsedFrame out{}; out.table_id = 777;
    for (const auto kind : {std::uint8_t{1}, std::uint8_t{4}, std::uint8_t{99}}) {
        const auto n = wf::frame_serialize(1, 0, 0, kind, &payload, 1, bytes.data(), static_cast<std::uint32_t>(bytes.size()));
        ASSERT_NE(n, 0u); EXPECT_FALSE(wf::frame_parse_source_wal(bytes.data(), n, out)); EXPECT_EQ(out.table_id, 777u);
    }
    auto n = wf::frame_serialize(1, 0, 1, wf::kWalTxnEnvelopeKind, &payload, 1, bytes.data(), static_cast<std::uint32_t>(bytes.size()));
    EXPECT_FALSE(wf::frame_parse_source_wal(bytes.data(), n, out));
    n = wf::frame_serialize(1, 7, 1, 1, &payload, 1, bytes.data(), static_cast<std::uint32_t>(bytes.size()));
    EXPECT_FALSE(wf::frame_parse_source_wal(bytes.data(), n + 1, out));
    EXPECT_TRUE(wf::frame_parse(bytes.data(), n + 1, out)); // legacy contract unchanged
    EXPECT_FALSE(wf::frame_parse_source_wal(nullptr, n, out));
}
TEST(BoltWalFrameTid, WrappedLengthsAreRefusedBeforeChecksumRead) {
    std::vector<std::uint8_t> bytes(wf::kFrameHeaderBytes);
    ASSERT_NE(wf::frame_serialize(1, 7, 1, 1, nullptr, 0, bytes.data(), static_cast<std::uint32_t>(bytes.size())), 0u);
    wf::ParsedFrame out{}; out.table_id = 777;
    wf::wf_detail::wr32(bytes.data() + wf::kOffPayloadLen, UINT32_MAX - wf::kFrameHeaderBytes + 9);
    wf::wf_detail::wr32(bytes.data() + wf::kOffFrameLen, 8);
    EXPECT_FALSE(wf::frame_parse_source_wal(bytes.data(), static_cast<std::uint32_t>(bytes.size()), out));
    EXPECT_FALSE(wf::frame_parse(bytes.data(), static_cast<std::uint32_t>(bytes.size()), out)); EXPECT_EQ(out.table_id, 777u);
    wf::wf_detail::wr32(bytes.data() + wf::kOffPayloadLen, 0);
    EXPECT_FALSE(wf::frame_parse_source_wal(bytes.data(), static_cast<std::uint32_t>(bytes.size()), out));
}
