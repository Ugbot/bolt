#include <gtest/gtest.h>
#include "bolt/wire/mold_udp64.h"
#include <array>

namespace {
namespace mold = bolt::wire::mold_udp64;
constexpr std::array<std::uint8_t, 25> kGolden{
    'S','E','S','S','I','O','N','0','0','1', 0,0,0,0,0,0,0,100, 0,2,
    0,1,0xaa, 0,0
};
constexpr mold::Limits kLimits{1500, 100};
TEST(BoltMoldUdp64, IndependentGoldenBigEndianAndEmptyMessage) {
    mold::Packet p{};
    ASSERT_EQ(mold::frame({kGolden.data(), kGolden.size()}, kLimits, p), mold::Status::Ok);
    EXPECT_EQ(p.first_sequence, 100u); EXPECT_EQ(p.message_count, 2u);
    EXPECT_EQ(p.session.size, 10u); EXPECT_EQ(p.session.data[9], '1');
    mold::Cursor c{}; mold::Message m{};
    ASSERT_EQ(mold::next(p, c, m), mold::Status::Ok);
    EXPECT_EQ(m.sequence, 100u); EXPECT_EQ(m.bytes.size, 1u); EXPECT_EQ(m.bytes.data[0], 0xaa);
    ASSERT_EQ(mold::next(p, c, m), mold::Status::Ok);
    EXPECT_EQ(m.sequence, 101u); EXPECT_EQ(m.bytes.size, 0u);
    EXPECT_EQ(mold::next(p, c, m), mold::Status::End);
}
TEST(BoltMoldUdp64, EveryTruncatedPrefixAtomicAndBudgetBounds) {
    mold::Packet p{}; p.first_sequence = 999;
    for (std::uint32_t n = 0; n < kGolden.size(); ++n) {
        EXPECT_NE(mold::frame({kGolden.data(), n}, kLimits, p), mold::Status::Ok);
        EXPECT_EQ(p.first_sequence, 999u);
    }
    EXPECT_EQ(mold::frame({kGolden.data(), kGolden.size()}, {1500, 1}, p), mold::Status::Capacity);
    EXPECT_EQ(mold::frame({kGolden.data(), kGolden.size()}, {20, 10}, p), mold::Status::Capacity);
    EXPECT_EQ(mold::frame({nullptr, 20}, kLimits, p), mold::Status::Invalid);
}
TEST(BoltMoldUdp64, HeartbeatEndTrailingBytesAndSequenceOverflow) {
    auto bytes = kGolden;
    bytes[19] = 0;
    mold::Packet p{};
    ASSERT_EQ(mold::frame({bytes.data(), 20}, kLimits, p), mold::Status::Ok);
    EXPECT_EQ(p.kind, mold::Kind::Heartbeat);
    EXPECT_EQ(mold::frame({bytes.data(), bytes.size()}, kLimits, p), mold::Status::Invalid);
    bytes[18] = 0xff; bytes[19] = 0xff;
    ASSERT_EQ(mold::frame({bytes.data(), 20}, kLimits, p), mold::Status::Ok);
    EXPECT_EQ(p.kind, mold::Kind::EndSession);
    bytes = kGolden;
    for (std::uint32_t i = 10; i < 18; ++i) bytes[i] = 0xff;
    EXPECT_EQ(mold::frame({bytes.data(), bytes.size()}, kLimits, p), mold::Status::Overflow);
    bytes[19] = 1;
    ASSERT_EQ(mold::frame({bytes.data(), 23}, kLimits, p), mold::Status::Ok);
    EXPECT_EQ(p.first_sequence, UINT64_MAX);
}
} // namespace
