#include "bolt/kernels/bolt_integer_filter.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <utility>

namespace {
using bolt::kernels::filter::CmpOp;
template<CmpOp Op, class T> bool oracle(T value, std::int64_t literal) {
    if constexpr (Op == CmpOp::Eq) return std::cmp_equal(value,literal);
    if constexpr (Op == CmpOp::Ne) return !std::cmp_equal(value,literal);
    if constexpr (Op == CmpOp::Lt) return std::cmp_less(value,literal);
    if constexpr (Op == CmpOp::Le) return !std::cmp_greater(value,literal);
    if constexpr (Op == CmpOp::Gt) return std::cmp_greater(value,literal);
    return !std::cmp_less(value,literal);
}
template<class T> std::array<T,7> values() {
    constexpr auto lo=std::numeric_limits<T>::min(), hi=std::numeric_limits<T>::max();
    if constexpr (std::is_signed_v<T>) return {lo,static_cast<T>(lo+1),-1,0,1,static_cast<T>(hi-1),hi};
    else return {0,1,127,static_cast<T>(hi/2),static_cast<T>(hi/2+1),static_cast<T>(hi-1),hi};
}
template<CmpOp Op,class T> void exercise() {
    const auto data=values<T>();
    const std::int32_t selection[]{6,0,4,2};
    constexpr std::int64_t literals[]{INT64_MIN,-1,0,1,127,128,255,256,32767,65535,65536,INT32_MAX,4294967295ll,INT64_MAX};
    for (const auto literal:literals) for (bool sparse:{false,true}) for (int bitmap=0;bitmap<3;++bitmap) {
        // An unaligned byte bitmap with a nonzero offset crossing a byte boundary.
        std::uint8_t backing[]{0,255,255}; auto* validity=backing+1;
        if (bitmap==2) { validity[0]&=static_cast<std::uint8_t>(~(1u<<4)); validity[1]&=static_cast<std::uint8_t>(~1u); }
        std::int32_t output[9]; for (auto& x:output) x=-99;
        const auto count=bolt::kernels::filter::integer_literal_filter<Op,T>(data.data(),7,
            sparse?selection:nullptr,sparse?4:0,literal,bitmap?validity:nullptr,3,output);
        std::int64_t expected=0;
        for (int i=0;i<(sparse?4:7);++i) {
            const auto row=sparse?selection[i]:i;
            if ((bitmap!=2 || (row!=1 && row!=5)) && oracle<Op>(data[row],literal)) {
                ASSERT_LT(expected,count); EXPECT_EQ(output[expected++],row);
            }
        }
        EXPECT_EQ(count,expected); EXPECT_EQ(output[7],-99); EXPECT_EQ(output[8],-99);
    }
    EXPECT_EQ((bolt::kernels::filter::integer_literal_filter<Op,T>(nullptr,0,nullptr,0,0,nullptr,0,nullptr)),0);
}
template<class T> void all_ops() {
    exercise<CmpOp::Eq,T>(); exercise<CmpOp::Ne,T>(); exercise<CmpOp::Lt,T>();
    exercise<CmpOp::Le,T>(); exercise<CmpOp::Gt,T>(); exercise<CmpOp::Ge,T>();
}
TEST(IntegerLiteralFilter, AllWidthsRangeNullOffsetAndSelection) {
    all_ops<std::int8_t>(); all_ops<std::int16_t>(); all_ops<std::int32_t>(); all_ops<std::int64_t>();
    all_ops<std::uint8_t>(); all_ops<std::uint16_t>(); all_ops<std::uint32_t>(); all_ops<std::uint64_t>();
}
} // namespace
