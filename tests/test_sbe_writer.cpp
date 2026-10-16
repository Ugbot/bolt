#include <gtest/gtest.h>
#include "bolt/wire/sbe_writer.h"
#include <cstring>
#include <new>

namespace {
using namespace bolt::wire;
using namespace bolt::wire::sbe;
constexpr auto LE = ByteOrder::Little;
constexpr auto BE = ByteOrder::Big;
template<std::size_t N>
void expect_bytes(const std::uint8_t* actual, const std::uint8_t (&expected)[N]) {
    for (std::size_t i = 0; i < N; ++i) EXPECT_EQ(actual[i], expected[i]) << "byte " << i;
}
TEST(SbeWriter, LiteralLittleEndianScalarsAndFloatBits) {
    std::uint8_t bytes[24]{}; Writer w{bytes, sizeof(bytes), 0};
    EXPECT_EQ((write_scalar<LE, std::uint16_t>(w, 0x1234u)), Status::Ok);
    EXPECT_EQ((write_scalar<LE, std::int16_t>(w, -2)), Status::Ok);
    EXPECT_EQ((write_scalar<LE, std::uint32_t>(w, 0x89abcdefu)), Status::Ok);
    EXPECT_EQ((write_scalar<LE, float>(w, 1.0f)), Status::Ok);
    EXPECT_EQ((write_scalar<LE, double>(w, -2.0)), Status::Ok);
    const std::uint8_t expected[] = {0x34,0x12,0xfe,0xff,0xef,0xcd,0xab,0x89,
        0,0,0x80,0x3f, 0,0,0,0,0,0,0,0xc0};
    expect_bytes(bytes, expected); EXPECT_EQ(w.offset, 20u);
}
TEST(SbeWriter, LiteralBigEndianScalarsAndFloatBits) {
    std::uint8_t bytes[24]{}; Writer w{bytes, sizeof(bytes), 0};
    EXPECT_EQ((write_scalar<BE, std::uint16_t>(w, 0x1234u)), Status::Ok);
    EXPECT_EQ((write_scalar<BE, std::int16_t>(w, -2)), Status::Ok);
    EXPECT_EQ((write_scalar<BE, std::uint32_t>(w, 0x89abcdefu)), Status::Ok);
    EXPECT_EQ((write_scalar<BE, float>(w, 1.0f)), Status::Ok);
    EXPECT_EQ((write_scalar<BE, double>(w, -2.0)), Status::Ok);
    const std::uint8_t expected[] = {0x12,0x34,0xff,0xfe,0x89,0xab,0xcd,0xef,
        0x3f,0x80,0,0, 0xc0,0,0,0,0,0,0,0};
    expect_bytes(bytes, expected); EXPECT_EQ(w.offset, 20u);
}
TEST(SbeWriter, LiteralHeadersInBothOrders) {
    std::uint8_t bytes[16]{}; Writer w{bytes, sizeof(bytes), 0};
    const Header header{0x1234, 0x5678, 0x9abc, 0xdef0};
    EXPECT_EQ(write_header<LE>(w, header), Status::Ok);
    EXPECT_EQ(write_header<BE>(w, header), Status::Ok);
    const std::uint8_t expected[] = {0x34,0x12,0x78,0x56,0xbc,0x9a,0xf0,0xde,
        0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0};
    expect_bytes(bytes, expected); EXPECT_EQ(w.offset, 16u);
}
TEST(SbeWriter, FixedBlockVersionOffsetsAndNulls) {
    std::uint8_t bytes[12]; std::memset(bytes, 0xa5, sizeof(bytes));
    Writer w{bytes, sizeof(bytes), 1}; WriteBlock b{};
    ASSERT_EQ(open_write_block(w, 10, 2, b), Status::Ok);
    EXPECT_EQ((write_field<LE, std::uint16_t>(b, 1, 2, 0x1234u)), Status::Ok);
    EXPECT_EQ((write_optional<BE, std::int16_t>(b, 4, 1, false, 45, std::int16_t{-1})), Status::Ok);
    EXPECT_EQ((write_optional<LE, std::uint16_t>(b, 7, 1, true, 0x9876u, std::uint16_t{0xffff})), Status::Ok);
    EXPECT_EQ((write_field<LE, std::uint32_t>(b, 99, 3, 10u)), Status::Absent);
    const std::uint8_t expected[] = {0xa5,0,0x34,0x12,0,0xff,0xff,0,0x76,0x98,0,0xa5};
    expect_bytes(bytes, expected); EXPECT_EQ(w.offset, 11u);
}
TEST(SbeWriter, FloatNullPreservesDeclaredPayload) {
    std::uint8_t bytes[8]{}; WriteBlock b{bytes, sizeof(bytes), 1};
    const auto nan = std::bit_cast<float>(std::uint32_t{0x7fc00001});
    EXPECT_EQ((write_optional<LE, float>(b, 0, 0, false, 1.0f, nan)), Status::Ok);
    EXPECT_EQ((write_optional<BE, float>(b, 4, 0, false, 1.0f, nan)), Status::Ok);
    const std::uint8_t expected[] = {1,0,0xc0,0x7f,0x7f,0xc0,0,1};
    expect_bytes(bytes, expected);
    const auto other_nan = std::bit_cast<float>(std::uint32_t{0x7fc00002});
    EXPECT_EQ((write_optional<LE, float>(b, 0, 0, true, other_nan, nan)), Status::InvalidInput);
    expect_bytes(bytes, expected);
}
TEST(SbeWriter, LiteralPackedGroupDimensionsBothOrders) {
    std::uint8_t bytes[20]{}; Writer w{bytes, sizeof(bytes), 0};
    EXPECT_EQ((write_group_dimensions<LE, std::uint16_t, std::uint8_t, 5>(w, 4, 2, 1)), Status::Ok);
    const std::uint8_t little[] = {4,0,2}; expect_bytes(bytes, little);
    w.offset = 0;
    EXPECT_EQ((write_group_dimensions<BE, std::uint16_t, std::uint32_t, 5>(w, 3, 2, 1)), Status::Ok);
    const std::uint8_t big[] = {0,3,0,0,0,2}; expect_bytes(bytes, big);
    EXPECT_EQ(w.offset, 6u); EXPECT_EQ(bytes[6], 0u); // Entries are reserved by generated adapter.
}
TEST(SbeWriter, LiteralVariableDataBothOrdersAndEmbeddedNull) {
    std::uint8_t bytes[16]{}; Writer w{bytes, sizeof(bytes), 0};
    const std::uint8_t data[] = {'A',0,'Z'};
    EXPECT_EQ((write_data<LE, std::uint16_t, 3>(w, {data, 3})), Status::Ok);
    EXPECT_EQ((write_data<BE, std::uint16_t, 3>(w, {data, 3})), Status::Ok);
    EXPECT_EQ((write_data<LE, std::uint8_t, 0>(w, {})), Status::Ok);
    const std::uint8_t expected[] = {3,0,'A',0,'Z',0,3,'A',0,'Z',0};
    expect_bytes(bytes, expected); EXPECT_EQ(w.offset, 11u);
}
TEST(SbeWriter, FixedBytesAreExactAndVersioned) {
    std::uint8_t bytes[8]{}; WriteBlock b{bytes, sizeof(bytes), 1};
    const std::uint8_t data[] = {0x41,0,0xff};
    EXPECT_EQ(write_fixed_bytes(b, 2, {data, 3}, 1), Status::Ok);
    EXPECT_EQ(write_fixed_bytes(b, 7, {data, 3}, 2), Status::Absent);
    const std::uint8_t expected[] = {0,0,0x41,0,0xff,0,0,0}; expect_bytes(bytes, expected);
}
TEST(SbeWriter, ScalarNarrowingRefusalsAreAtomic) {
    std::uint8_t bytes[8]; std::memset(bytes, 0xa5, sizeof(bytes));
    Writer w{bytes, sizeof(bytes), 2}; std::uint8_t before[8]; std::memcpy(before, bytes, 8);
    EXPECT_EQ((write_scalar<LE, std::uint8_t>(w, 256u)), Status::LimitExceeded);
    EXPECT_EQ((write_scalar<LE, std::uint16_t>(w, -1)), Status::LimitExceeded);
    EXPECT_EQ((write_scalar<LE, std::int8_t>(w, 128u)), Status::LimitExceeded);
    EXPECT_EQ((write_scalar<LE, std::int8_t>(w, -129)), Status::LimitExceeded);
    EXPECT_EQ((write_scalar<LE, std::int64_t>(w, UINT64_MAX)), Status::LimitExceeded);
    EXPECT_EQ((write_scalar<LE, float>(w, 1.0)), Status::InvalidInput);
    EXPECT_EQ(w.offset, 2u); EXPECT_EQ(std::memcmp(bytes, before, 8), 0);
}
TEST(SbeWriter, SignedAndUnsignedLimitsAreRepresentable) {
    std::uint8_t bytes[16]{}; Writer w{bytes, sizeof(bytes), 0};
    EXPECT_EQ((write_scalar<LE, std::int64_t>(w, INT64_MIN)), Status::Ok);
    EXPECT_EQ((write_scalar<BE, std::uint64_t>(w, UINT64_MAX)), Status::Ok);
    const std::uint8_t expected[] = {0,0,0,0,0,0,0,0x80,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff};
    expect_bytes(bytes, expected);
}
TEST(SbeWriter, TruncationLeavesHeaderCursorBlockAndBytesUnchanged) {
    std::uint8_t bytes[8]; std::memset(bytes, 0x5a, 8); std::uint8_t before[8]; std::memcpy(before, bytes, 8);
    Writer w{bytes, 8, 1}; Header h{1,2,3,4}; WriteBlock b{nullptr, 0, 123};
    EXPECT_EQ(write_header<LE>(w, h), Status::Truncated);
    EXPECT_EQ(open_write_block(w, 8, 1, b), Status::Truncated);
    EXPECT_EQ((write_scalar<BE, std::uint64_t>(w, 99u)), Status::Truncated);
    EXPECT_EQ(w.offset, 1u); EXPECT_EQ(b.data, nullptr); EXPECT_EQ(b.acting_version, 123u);
    EXPECT_EQ(std::memcmp(bytes, before, 8), 0);
}
TEST(SbeWriter, GroupDimensionRefusalsAreAtomic) {
    std::uint8_t bytes[16]; std::memset(bytes, 0x5a, 16); std::uint8_t before[16]; std::memcpy(before, bytes, 16);
    Writer w{bytes, 16, 0};
    EXPECT_EQ((write_group_dimensions<LE, std::uint16_t, std::uint8_t, 300>(w, 0, 256, 1)), Status::LimitExceeded);
    EXPECT_EQ((write_group_dimensions<LE, std::uint8_t, std::uint8_t, 5>(w, 256, 1, 1)), Status::LimitExceeded);
    EXPECT_EQ((write_group_dimensions<LE, std::uint16_t, std::uint8_t, 5>(w, 1, 6, 1)), Status::LimitExceeded);
    EXPECT_EQ((write_group_dimensions<LE, std::uint16_t, std::uint8_t, 5>(w, 1, 1, 0)), Status::LimitExceeded);
    EXPECT_EQ((write_group_dimensions<LE, std::uint16_t, std::uint8_t, 5, 2>(w, 1, 1, 3)), Status::LimitExceeded);
    EXPECT_EQ((write_group_dimensions<LE, std::uint16_t, std::uint8_t, 5>(w, 5, 3, 1)), Status::Truncated);
    EXPECT_EQ(w.offset, 0u); EXPECT_EQ(std::memcmp(bytes, before, 16), 0);
}
TEST(SbeWriter, ShapeArgumentsRefuseFloatingBoolAndWideInputsAtomically) {
    std::uint8_t bytes[16]; std::memset(bytes, 0x5a, 16);
    std::uint8_t before[16]; std::memcpy(before, bytes, 16);
    Writer w{bytes, 16, 0}; WriteBlock block{nullptr, 0, 123};
    EXPECT_EQ((write_group_dimensions<LE, std::uint16_t, std::uint8_t, 5>(w, 1.9, 1, 1)), Status::InvalidInput);
    EXPECT_EQ((write_group_dimensions<LE, std::uint16_t, std::uint8_t, 5>(w, 1, 1.9, 1)), Status::InvalidInput);
    EXPECT_EQ((write_group_dimensions<LE, std::uint16_t, std::uint8_t, 5>(w, true, 1, 1)), Status::InvalidInput);
    EXPECT_EQ((write_group_dimensions<LE, std::uint16_t, std::uint8_t, 5>(w, 1, true, 1)), Status::InvalidInput);
    EXPECT_EQ((write_group_dimensions<LE, std::uint16_t, std::uint8_t, 5>(w, -1, 1, 1)), Status::LimitExceeded);
    EXPECT_EQ((write_group_dimensions<LE, std::uint16_t, std::uint8_t, 5>(w, 1, UINT64_MAX, 1)), Status::LimitExceeded);
    EXPECT_EQ((write_group_dimensions<LE, std::uint16_t, std::uint8_t, 5>(w, 1, 1, 1.9)), Status::InvalidInput);
    EXPECT_EQ((write_group_dimensions<LE, std::uint16_t, std::uint8_t, 5>(w, 1, 1, true)), Status::InvalidInput);
    EXPECT_EQ(open_write_block(w, std::uint64_t{UINT32_MAX} + 1, 1, block), Status::LimitExceeded);
    EXPECT_EQ(open_write_block(w, 1, std::uint64_t{UINT16_MAX} + 1, block), Status::LimitExceeded);
    EXPECT_EQ(open_write_block(w, 1.9, 1, block), Status::InvalidInput);
    EXPECT_EQ(open_write_block(w, 1, 1.9, block), Status::InvalidInput);
    EXPECT_EQ(open_write_block(w, true, 1, block), Status::InvalidInput);
    EXPECT_EQ(open_write_block(w, 1, true, block), Status::InvalidInput);
    EXPECT_EQ(open_write_block(w, -1, 1, block), Status::LimitExceeded);
    EXPECT_EQ(w.offset, 0u); EXPECT_EQ(block.data, nullptr); EXPECT_EQ(block.acting_version, 123u);
    EXPECT_EQ(std::memcmp(bytes, before, 16), 0);
}
TEST(SbeWriter, VariableDataRefusalsAreAtomic) {
    std::uint8_t bytes[8]; std::memset(bytes, 0x5a, 8); std::uint8_t before[8]; std::memcpy(before, bytes, 8);
    const std::uint8_t data[7]{}; Writer w{bytes, 8, 1};
    EXPECT_EQ((write_data<LE, std::uint16_t, 7>(w, {data, 7})), Status::Truncated);
    EXPECT_EQ((write_data<LE, std::uint8_t, 3>(w, {data, 4})), Status::LimitExceeded);
    EXPECT_EQ((write_data<LE, std::uint8_t, 7>(w, {nullptr, 1})), Status::InvalidInput);
    EXPECT_EQ((write_data<LE, std::uint8_t, 7>(w, {bytes + 4, 2})), Status::InvalidInput);
    const std::uint8_t long_data[256]{};
    EXPECT_EQ((write_data<LE, std::uint8_t, 256>(w, {long_data, 256})), Status::LimitExceeded);
    EXPECT_EQ(w.offset, 1u); EXPECT_EQ(std::memcmp(bytes, before, 8), 0);
}
TEST(SbeWriter, FixedFieldFailuresPreserveBytes) {
    std::uint8_t bytes[8]; std::memset(bytes, 0x5a, 8); std::uint8_t before[8]; std::memcpy(before, bytes, 8);
    WriteBlock b{bytes, 8, 1};
    EXPECT_EQ((write_field<LE, std::uint16_t>(b, UINT32_MAX, 0, 2u)), Status::Truncated);
    EXPECT_EQ((write_field<LE, std::uint16_t>(b, 7, 0, 2u)), Status::Truncated);
    EXPECT_EQ((write_field<LE, std::uint8_t>(b, 0, 0, 256u)), Status::LimitExceeded);
    EXPECT_EQ((write_optional<LE, std::uint8_t>(b, 0, 0, true, 255u, std::uint8_t{255})), Status::InvalidInput);
    EXPECT_EQ(write_fixed_bytes(b, 0, {bytes + 3, 2}, 0), Status::InvalidInput);
    EXPECT_EQ(std::memcmp(bytes, before, 8), 0);
}
TEST(SbeWriter, DescriptorAndHeaderAliasesAreRejected) {
    alignas(Writer) std::uint8_t bytes[80]{};
    auto* embedded = new(bytes) Writer{bytes, 80, 0};
    std::uint8_t before[80]; std::memcpy(before, bytes, 80);
    EXPECT_EQ((write_scalar<LE, std::uint16_t>(*embedded, 1u)), Status::InvalidInput);
    EXPECT_EQ(std::memcmp(bytes, before, 80), 0);
    Writer w{bytes, 80, 0};
    auto* out = new(bytes + 32) WriteBlock{nullptr, 0, 456};
    std::memcpy(before, bytes, 80);
    EXPECT_EQ(open_write_block(w, 8, 1, *out), Status::InvalidInput);
    EXPECT_EQ(w.offset, 0u); EXPECT_EQ(std::memcmp(bytes, before, 80), 0);
    auto* block = new(bytes + 32) WriteBlock{bytes, 80, 1};
    std::memcpy(before, bytes, 80);
    EXPECT_EQ((write_field<LE, std::uint16_t>(*block, 0, 0, 1u)), Status::InvalidInput);
    EXPECT_EQ(std::memcmp(bytes, before, 80), 0);
    auto* h = new(bytes + 64) Header{1,2,3,4};
    std::memcpy(before, bytes, 80);
    EXPECT_EQ(write_header<LE>(w, *h), Status::InvalidInput);
    EXPECT_EQ(w.offset, 0u); EXPECT_EQ(std::memcmp(bytes, before, 80), 0);
}
TEST(SbeWriter, InvalidCursorAndEmptyBlockAreExplicit) {
    Writer invalid{nullptr, 1, 0};
    EXPECT_EQ((write_scalar<LE, std::uint8_t>(invalid, 1u)), Status::InvalidInput);
    std::uint8_t bytes[2]{}; invalid = {bytes, 2, 3};
    EXPECT_EQ((write_scalar<LE, std::uint8_t>(invalid, 1u)), Status::InvalidInput);
    Writer empty{}; WriteBlock block{};
    EXPECT_EQ(open_write_block(empty, 0, 7, block), Status::Ok);
    EXPECT_EQ(block.data, nullptr); EXPECT_EQ(block.acting_version, 7u);
    EXPECT_EQ((write_scalar<LE, std::uint8_t>(empty, 1u)), Status::Truncated);
}
TEST(SbeWriter, FieldOffsetsVersionsAndGroupProductsCannotWrap) {
    std::uint8_t bytes[32]; std::memset(bytes,0x5a,sizeof(bytes));
    WriteBlock block{bytes,sizeof(bytes),2};
    const auto offset=UINT64_C(0x100000002);
    EXPECT_EQ((write_field<LE,std::uint16_t>(block,offset,0,7)),Status::LimitExceeded);
    EXPECT_EQ((write_field<LE,std::uint16_t>(block,0,65538,7)),Status::LimitExceeded);
    EXPECT_EQ((write_optional<LE,std::uint16_t>(block,offset,0,true,7,65535)),Status::LimitExceeded);
    const std::uint8_t input=3;
    EXPECT_EQ(write_fixed_bytes(block,offset,ByteView{&input,1},0),Status::LimitExceeded);
    Writer writer{bytes,sizeof(bytes),0};
    // Even representable wire dimensions must fit the actual output span.
    EXPECT_EQ((write_group_dimensions<LE,std::uint32_t,std::uint32_t,2>(writer,
        UINT32_C(0x80000000),2,1)),Status::Truncated);
    EXPECT_EQ(writer.offset,0U);
    for(auto byte:bytes) EXPECT_EQ(byte,0x5a);
}
} // namespace
