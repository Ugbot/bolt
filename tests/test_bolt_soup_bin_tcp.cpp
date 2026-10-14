#include <gtest/gtest.h>
#include "bolt/wire/soup_bin_tcp.h"
#include <array>

namespace {
namespace soup = bolt::wire::soup_bin_tcp;
constexpr std::array<std::uint8_t, 36> kLoginAndHeartbeat{
    0,31,'A','S','E','S','S','I','O','N','0','0','1',
    ' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','1','2','3',
    0,1,'H'
};
TEST(BoltSoupBinTcp, PartialAndCoalescedFramesDoNotConfuseStreamWithPackets) {
    soup::Packet p{}; p.type = '?'; std::uint32_t consumed = 999;
    for (std::uint32_t n = 0; n < 33; ++n) {
        EXPECT_EQ(soup::frame({kLoginAndHeartbeat.data(), n}, 1024, soup::Direction::Server, p, consumed),
            soup::Status::NeedMore);
        EXPECT_EQ(consumed, 999u); EXPECT_EQ(p.type, '?');
    }
    ASSERT_EQ(soup::frame({kLoginAndHeartbeat.data(), kLoginAndHeartbeat.size()}, 1024,
        soup::Direction::Server, p, consumed), soup::Status::Ok);
    EXPECT_EQ(consumed, 33u); EXPECT_EQ(p.type, 'A');
    std::uint64_t sequence{};
    ASSERT_EQ(soup::sequence_number({p.payload.data + 10, 20}, sequence), soup::Status::Ok);
    EXPECT_EQ(sequence, 123u);
    ASSERT_EQ(soup::frame({kLoginAndHeartbeat.data() + consumed, 3}, 1024,
        soup::Direction::Server, p, consumed), soup::Status::Ok);
    EXPECT_EQ(p.type, 'H'); EXPECT_EQ(consumed, 3u);
}
TEST(BoltSoupBinTcp, DirectionUnknownLengthsAndBudgetsFailClosed) {
    soup::Packet p{}; std::uint32_t used = 777;
    EXPECT_EQ(soup::frame({kLoginAndHeartbeat.data(), 33}, 1024, soup::Direction::Client, p, used),
        soup::Status::Invalid);
    EXPECT_EQ(soup::frame({kLoginAndHeartbeat.data(), 33}, 32, soup::Direction::Server, p, used),
        soup::Status::Capacity);
    std::array<std::uint8_t, 4> b{0,1,'?',0};
    EXPECT_EQ(soup::frame({b.data(), 3}, 1024, soup::Direction::Server, p, used), soup::Status::Unsupported);
    b[1] = 0;
    EXPECT_EQ(soup::frame({b.data(), 3}, 1024, soup::Direction::Server, p, used), soup::Status::Invalid);
    b[1] = 2; b[2] = 'H';
    EXPECT_EQ(soup::frame({b.data(), 4}, 1024, soup::Direction::Server, p, used), soup::Status::Invalid);
    b[2] = 'J'; b[3] = 'X';
    EXPECT_EQ(soup::frame({b.data(), 4}, 1024, soup::Direction::Server, p, used), soup::Status::Invalid);
    EXPECT_EQ(used, 777u);
}
TEST(BoltSoupBinTcp, DecimalSequenceMaxOverflowAndPaddingAreExact) {
    constexpr std::array<std::uint8_t,20> max{
        '1','8','4','4','6','7','4','4','0','7','3','7','0','9','5','5','1','6','1','5'};
    std::uint64_t value = 123;
    EXPECT_EQ(soup::sequence_number({max.data(), max.size()}, value), soup::Status::Ok);
    EXPECT_EQ(value, UINT64_MAX);
    auto b = max; b[19] = '6';
    EXPECT_EQ(soup::sequence_number({b.data(), b.size()}, value), soup::Status::Overflow);
    EXPECT_EQ(value, UINT64_MAX);
    b = max; b[10] = ' ';
    EXPECT_EQ(soup::sequence_number({b.data(), b.size()}, value), soup::Status::Invalid);
    b.fill(' ');
    EXPECT_EQ(soup::sequence_number({b.data(), b.size()}, value), soup::Status::Invalid);
}
} // namespace
