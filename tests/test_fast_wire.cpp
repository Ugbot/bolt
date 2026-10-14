#include "bolt/wire/fast.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
namespace {
namespace f = bolt::wire::fast;
using bolt::wire::ByteReader;
template<std::size_t N> ByteReader reader(const std::uint8_t (&bytes)[N]) { return {bytes,static_cast<std::uint32_t>(N),0}; }
TEST(FastWire, IndependentSignedAndUnsignedStopBitVectors) {
    const std::uint8_t u[]{0x01,0x00,0x80}; auto r = reader(u); f::Value<std::uint32_t> uv{};
    ASSERT_EQ(f::integer(r,false,uv),f::Result::Ok); EXPECT_EQ(uv.value,16384u); EXPECT_TRUE(uv.present);
    const std::uint8_t positive[]{0x00,0xc0}, negative[]{0x7f,0xbf}; f::Value<std::int32_t> sv{};
    r = reader(positive); ASSERT_EQ(f::integer(r,false,sv),f::Result::Ok); EXPECT_EQ(sv.value,64);
    r = reader(negative); ASSERT_EQ(f::integer(r,false,sv),f::Result::Ok); EXPECT_EQ(sv.value,-65);
}
TEST(FastWire, FullWidthNullableExtremaAreNotTruncatedTo64Bits) {
    const std::uint8_t umax[]{1,127,127,127,127,127,127,127,127,255};
    const std::uint8_t nullable_umax[]{2,0,0,0,0,0,0,0,0,128};
    f::Value<std::uint64_t> u{}; auto r = reader(umax);
    ASSERT_EQ(f::integer(r,false,u),f::Result::Ok); EXPECT_EQ(u.value,UINT64_MAX);
    r = reader(nullable_umax); ASSERT_EQ(f::integer(r,true,u),f::Result::Ok); EXPECT_EQ(u.value,UINT64_MAX);
    r = reader(nullable_umax); EXPECT_EQ(f::integer(r,false,u),f::Result::Overflow); EXPECT_EQ(r.offset,0u);
    const std::uint8_t smax[]{0,127,127,127,127,127,127,127,127,255};
    const std::uint8_t smin[]{127,0,0,0,0,0,0,0,0,128};
    const std::uint8_t nullable_smax[]{1,0,0,0,0,0,0,0,0,128};
    f::Value<std::int64_t> s{}; r = reader(smax);
    ASSERT_EQ(f::integer(r,false,s),f::Result::Ok); EXPECT_EQ(s.value,INT64_MAX);
    r = reader(smin); ASSERT_EQ(f::integer(r,true,s),f::Result::Ok); EXPECT_EQ(s.value,INT64_MIN);
    r = reader(nullable_smax); ASSERT_EQ(f::integer(r,true,s),f::Result::Ok); EXPECT_EQ(s.value,INT64_MAX);
    r = reader(nullable_smax); EXPECT_EQ(f::integer(r,false,s),f::Result::Overflow); EXPECT_EQ(r.offset,0u);
}
TEST(FastWire, NullableZeroNegativeAnd32BitMaximum) {
    const std::uint8_t b[]{0x80,0x81,0xff}; auto r = reader(b); f::Value<std::int32_t> v{};
    ASSERT_EQ(f::integer(r,true,v),f::Result::Ok); EXPECT_FALSE(v.present);
    ASSERT_EQ(f::integer(r,true,v),f::Result::Ok); EXPECT_TRUE(v.present); EXPECT_EQ(v.value,0);
    ASSERT_EQ(f::integer(r,true,v),f::Result::Ok); EXPECT_EQ(v.value,-1);
    const std::uint8_t max[]{0x10,0,0,0,0x80}; r = reader(max); f::Value<std::uint32_t> u{};
    ASSERT_EQ(f::integer(r,true,u),f::Result::Ok); EXPECT_EQ(u.value,UINT32_MAX);
    r = reader(max); EXPECT_EQ(f::integer(r,false,u),f::Result::Overflow); EXPECT_EQ(r.offset,0u);
}
TEST(FastWire, TruncationOverlongAndOverflowPreserveCursorAndDestination) {
    const std::uint8_t truncated[]{1,2}, overlong[]{0,128}, negative_overlong[]{127,255};
    f::Value<std::int64_t> v{19,true}; auto r = reader(truncated);
    EXPECT_EQ(f::integer(r,false,v),f::Result::Truncated); EXPECT_EQ(r.offset,0u); EXPECT_EQ(v.value,19);
    r = reader(overlong); EXPECT_EQ(f::integer(r,false,v),f::Result::Overlong); EXPECT_EQ(r.offset,0u);
    r = reader(negative_overlong); EXPECT_EQ(f::integer(r,false,v),f::Result::Overlong);
    const std::uint8_t overflow[]{3,0,0,0,0,0,0,0,0,128}; r = reader(overflow);
    f::Value<std::uint64_t> u{19,true}; EXPECT_EQ(f::integer(r,true,u),f::Result::Overflow); EXPECT_EQ(u.value,19u);
    const std::uint8_t unterminated[]{0,0,0,0,0,0,0,0,0,0}; r = reader(unterminated);
    EXPECT_EQ(f::integer(r,false,u),f::Result::Overflow); EXPECT_EQ(r.offset,0u);
    const std::uint8_t signed_overflow[]{1,0,0,0,0,0,0,0,0,129}; r = reader(signed_overflow);
    EXPECT_EQ(f::integer(r,true,v),f::Result::Overflow); EXPECT_EQ(v.value,19);
}
TEST(FastWire, PresenceHasImplicitZerosAndRejectsOverlongOrUnconsumedBits) {
    const std::uint8_t bytes[]{0x40,0x81}; auto r = reader(bytes); f::Presence p{};
    ASSERT_EQ(f::presence(r,p),f::Result::Ok); EXPECT_TRUE(p.next());
    for (auto i = 1u; i < 13; ++i) EXPECT_FALSE(p.next());
    EXPECT_FALSE(p.exhausted()); EXPECT_TRUE(p.next()); EXPECT_TRUE(p.exhausted());
    EXPECT_FALSE(p.next()); EXPECT_TRUE(p.exhausted());
    const std::uint8_t overlong[]{0x40,0x80}; r = reader(overlong);
    EXPECT_EQ(f::presence(r,p),f::Result::Overlong); EXPECT_EQ(r.offset,0u);
    const std::uint8_t unterminated[16]{}; r = reader(unterminated);
    EXPECT_EQ(f::presence(r,p),f::Result::Capacity); EXPECT_EQ(r.offset,0u);
}
TEST(FastWire, AsciiEmptyNullNulAndOverlongAreDistinct) {
    const std::uint8_t bytes[]{0x80,0,0x80,0,0,0x80,'X','E','U',std::uint8_t('R'|128)};
    auto r = reader(bytes); char out[4]{}; f::Value<std::uint32_t> value{};
    ASSERT_EQ(f::ascii(r,true,out,4,value),f::Result::Ok); EXPECT_FALSE(value.present);
    ASSERT_EQ(f::ascii(r,true,out,4,value),f::Result::Ok); EXPECT_TRUE(value.present); EXPECT_EQ(value.value,0u);
    ASSERT_EQ(f::ascii(r,true,out,4,value),f::Result::Ok); EXPECT_EQ(value.value,1u); EXPECT_EQ(out[0],0);
    ASSERT_EQ(f::ascii(r,false,out,4,value),f::Result::Ok); EXPECT_EQ(std::memcmp(out,"XEUR",4),0);
    const std::uint8_t overlong[]{0,0xc1}; r = reader(overlong); std::memset(out,0x55,4);
    EXPECT_EQ(f::ascii(r,true,out,4,value),f::Result::Overlong); EXPECT_EQ(r.offset,0u); EXPECT_EQ(out[0],0x55);
    const std::uint8_t long_text[]{'A','B',0xc3}; r = reader(long_text);
    EXPECT_EQ(f::ascii(r,false,out,2,value),f::Result::Capacity); EXPECT_EQ(r.offset,0u); EXPECT_EQ(out[0],0x55);
}
TEST(FastWire, CopyUndefinedEmptyAssignedAndResetStayDistinct) {
    const std::uint8_t absent[]{0x80}, explicit_null[]{0xc0,0x80}; auto r = reader(absent);
    f::Presence p{}; ASSERT_EQ(f::presence(r,p),f::Result::Ok);
    f::Cell<std::uint32_t> cell{}; f::Value<std::uint32_t> value{91,true};
    EXPECT_EQ(f::field(r,p,f::Operator::Copy,false,{},cell,value),f::Result::Undefined);
    EXPECT_EQ(cell.state,f::State::Undefined); EXPECT_EQ(p.used,0u); EXPECT_EQ(value.value,91u);
    ASSERT_EQ(f::field(r,p,f::Operator::Copy,true,{},cell,value),f::Result::Ok);
    EXPECT_EQ(cell.state,f::State::Empty); EXPECT_FALSE(value.present);
    f::reset(&cell,1); EXPECT_EQ(cell.state,f::State::Undefined);
    r = reader(absent); ASSERT_EQ(f::presence(r,p),f::Result::Ok);
    ASSERT_EQ(f::field(r,p,f::Operator::Copy,false,{7,true},cell,value),f::Result::Ok); EXPECT_EQ(value.value,7u);
    r = reader(explicit_null); ASSERT_EQ(f::presence(r,p),f::Result::Ok);
    ASSERT_EQ(f::field(r,p,f::Operator::Copy,true,{},cell,value),f::Result::Ok); EXPECT_EQ(cell.state,f::State::Empty);
}
TEST(FastWire, IncrementWrapsAtDeclaredWidthAndUsesInitialWithoutIncrement) {
    const std::uint8_t bytes[]{0x80}; auto r = reader(bytes); f::Presence p{}; ASSERT_EQ(f::presence(r,p),f::Result::Ok);
    f::Cell<std::uint32_t> cell{}; f::Value<std::uint32_t> value{};
    ASSERT_EQ(f::field(r,p,f::Operator::Increment,false,{UINT32_MAX,true},cell,value),f::Result::Ok);
    EXPECT_EQ(value.value,UINT32_MAX);
    ASSERT_EQ(f::field(r,p,f::Operator::Increment,false,{},cell,value),f::Result::Ok); EXPECT_EQ(value.value,0u);
    f::Cell<std::int64_t> signed_cell{INT64_MAX,f::State::Assigned}; f::Value<std::int64_t> signed_value{};
    ASSERT_EQ(f::field(r,p,f::Operator::Increment,false,{},signed_cell,signed_value),f::Result::Ok);
    EXPECT_EQ(signed_value.value,INT64_MIN);
}
TEST(FastWire, DefaultAndNullDeltaDoNotChangeDictionary) {
    const std::uint8_t bytes[]{0xc0,0x80}; auto r = reader(bytes); f::Presence p{}; ASSERT_EQ(f::presence(r,p),f::Result::Ok);
    f::Cell<std::uint32_t> cell{12,f::State::Assigned}; f::Value<std::uint32_t> value{};
    ASSERT_EQ(f::field(r,p,f::Operator::Default,true,{7,true},cell,value),f::Result::Ok);
    EXPECT_FALSE(value.present); EXPECT_EQ(cell.value,12u); EXPECT_EQ(cell.state,f::State::Assigned);
    r = reader(bytes); r.offset = 1;
    ASSERT_EQ(f::field(r,p,f::Operator::Delta,true,{},cell,value),f::Result::Ok);
    EXPECT_FALSE(value.present); EXPECT_EQ(cell.value,12u);
}
TEST(FastWire, IntegerDeltaBaseOverflowAndRefusalAreAtomic) {
    const std::uint8_t bytes[]{0xff}; auto r = reader(bytes); f::Presence p{};
    f::Cell<std::uint32_t> cell{}; f::Value<std::uint32_t> value{17,true};
    EXPECT_EQ(f::field(r,p,f::Operator::Delta,false,{},cell,value),f::Result::Overflow);
    EXPECT_EQ(r.offset,0u); EXPECT_EQ(cell.state,f::State::Undefined); EXPECT_EQ(value.value,17u);
    cell = {12,f::State::Assigned}; ASSERT_EQ(f::field(r,p,f::Operator::Delta,false,{},cell,value),f::Result::Ok);
    EXPECT_EQ(value.value,11u); EXPECT_EQ(cell.value,11u);
    cell = {0,f::State::Empty}; r = reader(bytes);
    EXPECT_EQ(f::field(r,p,f::Operator::Delta,false,{},cell,value),f::Result::Undefined); EXPECT_EQ(r.offset,0u);
}
TEST(FastWire, DecimalPreservesExactLayoutAndNullDeltaState) {
    const std::uint8_t initial[]{0xfe,0x09,0xd2}, next[]{0x80}, change[]{0x84,0x7f,0x9c};
    auto r = reader(initial); f::Cell<f::Decimal> cell{}; f::Value<f::Decimal> value{};
    ASSERT_EQ(f::decimal_delta(r,true,cell,value),f::Result::Ok);
    EXPECT_EQ(value.value.exponent,-2); EXPECT_EQ(value.value.mantissa,1234);
    r = reader(next); ASSERT_EQ(f::decimal_delta(r,true,cell,value),f::Result::Ok);
    EXPECT_FALSE(value.present); EXPECT_EQ(cell.value.exponent,-2); EXPECT_EQ(cell.value.mantissa,1234);
    r = reader(change); ASSERT_EQ(f::decimal_delta(r,true,cell,value),f::Result::Ok);
    EXPECT_EQ(value.value.exponent,1); EXPECT_EQ(value.value.mantissa,1134);
}
TEST(FastWire, DecimalExponentAndMantissaErrorsPreserveAllState) {
    const std::uint8_t exponent[]{0x00,0xc0,0x81}; auto r = reader(exponent);
    f::Cell<f::Decimal> cell{}; f::Value<f::Decimal> value{{19,1},true};
    EXPECT_EQ(f::decimal_delta(r,false,cell,value),f::Result::Overflow);
    EXPECT_EQ(r.offset,0u); EXPECT_EQ(cell.state,f::State::Undefined); EXPECT_EQ(value.value.mantissa,19);
    const std::uint8_t mantissa[]{0x80,0x81}; r = reader(mantissa); cell = {{INT64_MAX,0},f::State::Assigned};
    EXPECT_EQ(f::decimal_delta(r,false,cell,value),f::Result::Overflow); EXPECT_EQ(r.offset,0u);
    EXPECT_EQ(cell.value.mantissa,INT64_MAX); EXPECT_EQ(value.value.mantissa,19);
}
TEST(FastWire, ConstantsAndDecimalCopyHaveExplicitPresenceAndReset) {
    const std::uint8_t absent[]{0x80}; auto r = reader(absent); f::Presence p{};
    ASSERT_EQ(f::presence(r,p),f::Result::Ok); f::Cell<std::uint32_t> cell{}; f::Value<std::uint32_t> value{19,true};
    EXPECT_EQ(f::field(r,p,f::Operator::Constant,false,{},cell,value),f::Result::Undefined);
    EXPECT_EQ(p.used,0u); EXPECT_EQ(value.value,19u);
    ASSERT_EQ(f::field(r,p,f::Operator::Constant,true,{7,true},cell,value),f::Result::Ok); EXPECT_FALSE(value.present);
    f::Cell<f::Decimal> decimal{}; f::Value<f::Decimal> d{};
    ASSERT_EQ(f::decimal_field(r,p,f::Operator::Copy,true,{},decimal,d),f::Result::Ok);
    EXPECT_EQ(decimal.state,f::State::Empty); EXPECT_FALSE(d.present);
    f::reset(&decimal,1); EXPECT_EQ(decimal.state,f::State::Undefined);
    EXPECT_EQ(f::decimal_field(r,p,f::Operator::Copy,false,{},decimal,d),f::Result::Undefined);
    ASSERT_EQ(f::decimal_field(r,p,f::Operator::Copy,false,{{1234,-2},true},decimal,d),f::Result::Ok);
    EXPECT_EQ(d.value.mantissa,1234); EXPECT_EQ(d.value.exponent,-2);
}
TEST(FastWire, DeltaUsesOnlyPresentInitialAndRefusesUnsupportedDecimalInitialAtomically) {
    const std::uint8_t zero[]{0x80,0x80}; auto r = reader(zero); f::Presence p{};
    f::Cell<std::uint32_t> cell{}; f::Value<std::uint32_t> value{};
    ASSERT_EQ(f::field(r,p,f::Operator::Delta,false,{19,false},cell,value),f::Result::Ok);
    EXPECT_EQ(value.value,0u); EXPECT_EQ(cell.value,0u);
    cell = {}; r = reader(zero);
    ASSERT_EQ(f::field(r,p,f::Operator::Delta,false,{19,true},cell,value),f::Result::Ok);
    EXPECT_EQ(value.value,19u); EXPECT_EQ(cell.value,19u);
    f::Cell<f::Decimal> decimal{}; f::Value<f::Decimal> out{{9,1},true}; r = reader(zero);
    EXPECT_EQ(f::decimal_field(r,p,f::Operator::Delta,false,{{1234,-2},true},decimal,out),f::Result::Invalid);
    EXPECT_EQ(r.offset,0u); EXPECT_EQ(p.used,0u); EXPECT_EQ(decimal.state,f::State::Undefined);
    EXPECT_EQ(out.value.mantissa,9); EXPECT_EQ(out.value.exponent,1);
    ASSERT_EQ(f::decimal_field(r,p,f::Operator::Delta,false,{{1234,-2},false},decimal,out),f::Result::Ok);
    EXPECT_EQ(out.value.mantissa,0); EXPECT_EQ(out.value.exponent,0);
}
} // namespace
