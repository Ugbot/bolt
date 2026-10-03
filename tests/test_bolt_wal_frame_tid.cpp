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
