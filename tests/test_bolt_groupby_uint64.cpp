#include "bolt/join/bolt_groupby.h"
#include <gtest/gtest.h>
#include <cstring>

namespace {
using namespace bolt;
constexpr uint64_t values[]={0,INT64_MAX,UINT64_C(0x8000000000000000),UINT64_MAX,
                             0,INT64_MAX,UINT64_C(0x8000000000000000),UINT64_MAX};
struct Guarded {
    uint64_t before=UINT64_C(0xa5a5a5a5a5a5a5a5),data[8]{},after=UINT64_C(0x5a5a5a5a5a5a5a5a);
    Guarded() { std::memcpy(data,values,sizeof(data)); }
    void check() const {
        EXPECT_EQ(before,UINT64_C(0xa5a5a5a5a5a5a5a5)); EXPECT_EQ(after,UINT64_C(0x5a5a5a5a5a5a5a5a));
        EXPECT_EQ(std::memcmp(data,values,sizeof(data)),0);
    }
};
void check_groups(const BoltColumn& keys,const BoltColumn& counts,uint32_t n) {
    ASSERT_EQ(keys.type,BoltType::UInt64); ASSERT_EQ(keys.type_size_bytes,8);
    ASSERT_EQ(counts.type,BoltType::Int64); ASSERT_EQ(n,5u);
    int64_t totals[5]={-1,-1,-1,-1,-1};
    for(uint32_t i=0;i<n;++i) {
        unsigned index=4;
        if(!keys.is_null(i)) {
            const auto key=static_cast<const uint64_t*>(keys.data)[i];
            for(unsigned j=0;j<4;++j) if(values[j]==key) index=j;
            ASSERT_LT(index,4u);
        }
        ASSERT_EQ(totals[index],-1); totals[index]=static_cast<const int64_t*>(counts.data)[i];
    }
    EXPECT_EQ(totals[0],1); EXPECT_EQ(totals[1],2); EXPECT_EQ(totals[2],1);
    EXPECT_EQ(totals[3],2); EXPECT_EQ(totals[4],2);
}
TEST(BoltGroupbyUInt64, BitPreservingReadAndNullIdentity) {
    Guarded input; uint8_t validity=0xaf;
    auto key=BoltColumn::make_flat(input.data,&validity,8,BoltType::UInt64);
    for(int64_t i=0;i<8;++i) {
        const auto cell=gb_detail::read_cell16(key,i); uint64_t bits=0;
        std::memcpy(&bits,&cell.a,8); EXPECT_EQ(bits,values[i]); EXPECT_EQ(cell.b,0);
    }
    EXPECT_TRUE(gb_detail::gb_cell_is_null_key(gb_detail::read_key_cell16(key,4),BoltType::UInt64));
    EXPECT_FALSE(gb_detail::gb_cell_is_null_key(gb_detail::read_key_cell16(key,0),BoltType::UInt64));
    input.check();
}
TEST(BoltGroupbyUInt64, OneShotLiteralCounts) {
    Arena arena; Guarded input; uint8_t validity=0xaf;
    auto key=BoltColumn::make_flat(input.data,&validity,8,BoltType::UInt64);
    AggSpec spec{}; spec.kind=AggKind::CountStar; BoltColumn out[1]{},counts[1]{}; uint32_t n=0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key,1,nullptr,0,&spec,1,8,out,counts,&n,&arena,8));
    check_groups(out[0],counts[0],n); input.check();
}
TEST(BoltGroupbyUInt64, IncrementalAndPartialMergeLiteralCounts) {
    for(bool merge:{false,true}) {
        Arena arena; Guarded input; uint8_t validity=0xaf;
        auto key=BoltColumn::make_flat(input.data,&validity,8,BoltType::UInt64);
        AggSpec spec{}; spec.kind=AggKind::CountStar; GroupbyTypedState state{},partial{};
        ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&state,&arena,&key,1,nullptr,0,&spec,1,8));
        ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&partial,&arena,&key,1,nullptr,0,&spec,1,8));
        auto first=key; first.length=4;
        groupby_agg_multi_key_typed_ingest(&state,&first,nullptr,nullptr,0,4);
        auto second=key; second.data=input.data+4; second.length=4; second.validity_offset=4;
        groupby_agg_multi_key_typed_ingest(merge?&partial:&state,&second,nullptr,nullptr,0,4);
        if(merge) ASSERT_TRUE(groupby_agg_multi_key_typed_merge(&state,&partial));
        ASSERT_FALSE(state.oom); ASSERT_FALSE(state.null_key);
        BoltColumn out[1]{},counts[1]{}; uint32_t n=0;
        ASSERT_TRUE(groupby_agg_multi_key_typed_finalize(&state,out,counts,&n));
        check_groups(out[0],counts[0],n); input.check();
    }
}
} // namespace
