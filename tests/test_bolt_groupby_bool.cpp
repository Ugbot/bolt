#include "bolt/join/bolt_groupby.h"
#include <gtest/gtest.h>
#include <cstring>

namespace {
using namespace bolt;
struct GuardedBool {
    std::uint8_t before[8], values[8], after[8];
    GuardedBool() {
        std::memset(before,0xa5,sizeof(before)); std::memset(after,0x5a,sizeof(after));
        const std::uint8_t input[]={1,0,1,0,1,0,1,0}; std::memcpy(values,input,sizeof(values));
    }
    void check() const {
        for(auto byte:before) EXPECT_EQ(byte,0xa5);
        for(auto byte:after) EXPECT_EQ(byte,0x5a);
    }
};
void check_groups(const BoltColumn& keys,const BoltColumn& counts,std::uint32_t n) {
    ASSERT_EQ(keys.type,BoltType::Bool); ASSERT_EQ(keys.type_size_bytes,1);
    ASSERT_EQ(counts.type,BoltType::Int64); ASSERT_EQ(n,3u);
    std::int64_t totals[3]={-1,-1,-1};
    for(std::uint32_t i=0;i<n;++i) {
        const auto key=static_cast<const std::uint8_t*>(keys.data)[i];
        EXPECT_LE(key,1u); const auto group=keys.is_null(i)?2u:static_cast<unsigned>(key);
        ASSERT_EQ(totals[group],-1); totals[group]=static_cast<const std::int64_t*>(counts.data)[i];
    }
    EXPECT_EQ(totals[0],4); EXPECT_EQ(totals[1],2); EXPECT_EQ(totals[2],2);
}
TEST(BoltGroupbyBool, ByteWidthReadAndNullIdentityAreExact) {
    GuardedBool input; std::uint8_t validity=0xaf;
    auto key=BoltColumn::make_flat(input.values,&validity,8,BoltType::Bool);
    for(std::int64_t i=0;i<8;++i) {
        const auto cell=gb_detail::read_cell16(key,i);
        EXPECT_EQ(cell.a,input.values[i]); EXPECT_EQ(cell.b,0);
    }
    EXPECT_TRUE(gb_detail::gb_key_type_can_tag_null(BoltType::Bool));
    const auto null=gb_detail::read_key_cell16(key,4);
    EXPECT_TRUE(gb_detail::gb_cell_is_null_key(null,BoltType::Bool));
    input.check();
}
TEST(BoltGroupbyBool, OneShotFalseTrueNullLiteralCounts) {
    Arena arena; GuardedBool input; std::uint8_t validity=0xaf;
    auto key=BoltColumn::make_flat(input.values,&validity,8,BoltType::Bool);
    AggSpec spec{}; spec.kind=AggKind::CountStar;
    BoltColumn output[1]{},counts[1]{}; std::uint32_t n=0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key,1,nullptr,0,&spec,1,8,output,counts,&n,&arena,8));
    check_groups(output[0],counts[0],n); input.check();
}
TEST(BoltGroupbyBool, IncrementalSplitValidityOffsetLiteralCounts) {
    Arena arena; GuardedBool input; std::uint8_t validity=0xaf;
    auto key=BoltColumn::make_flat(input.values,&validity,8,BoltType::Bool);
    AggSpec spec{}; spec.kind=AggKind::CountStar; GroupbyTypedState state{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&state,&arena,&key,1,nullptr,0,&spec,1,8));
    auto first=key; first.length=4;
    groupby_agg_multi_key_typed_ingest(&state,&first,nullptr,nullptr,0,4);
    auto second=key; second.data=input.values+4; second.length=4; second.validity_offset=4;
    groupby_agg_multi_key_typed_ingest(&state,&second,nullptr,nullptr,0,4);
    ASSERT_FALSE(state.oom); ASSERT_FALSE(state.null_key);
    BoltColumn output[1]{},counts[1]{}; std::uint32_t n=0;
    ASSERT_TRUE(groupby_agg_multi_key_typed_finalize(&state,output,counts,&n));
    check_groups(output[0],counts[0],n); input.check();
}
} // namespace
