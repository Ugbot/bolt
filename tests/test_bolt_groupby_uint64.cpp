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
template<class T>
void check_narrow_keys(BoltType type,const T (&keys)[4]) {
    Arena arena; T data[4]; std::memcpy(data,keys,sizeof(data));
    auto key=BoltColumn::make_flat(data,nullptr,4,type);
    AggSpec spec{}; spec.kind=AggKind::CountStar; BoltColumn out[1]{},counts[1]{}; uint32_t n=0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key,1,nullptr,0,&spec,1,4,out,counts,&n,&arena,4));
    ASSERT_EQ(n,2u); ASSERT_EQ(out[0].type,type); ASSERT_EQ(out[0].type_size_bytes,sizeof(T));
    for(uint32_t i=0;i<n;++i) {
        const T k=static_cast<const T*>(out[0].data)[i];
        EXPECT_TRUE(k==keys[0] || k==keys[1]);
        EXPECT_EQ(static_cast<const int64_t*>(counts[0].data)[i],2);
    }
}
TEST(BoltGroupbyNarrow, KeysReadAtTheirOwnWidth) {
    constexpr uint8_t u8[4]={255,1,255,1}; check_narrow_keys(BoltType::UInt8,u8);
    constexpr uint16_t u16[4]={65535,1,65535,1}; check_narrow_keys(BoltType::UInt16,u16);
    constexpr uint32_t u32[4]={UINT32_MAX,1,UINT32_MAX,1}; check_narrow_keys(BoltType::UInt32,u32);
    constexpr int8_t i8[4]={-128,1,-128,1}; check_narrow_keys(BoltType::Int8,i8);
    constexpr int16_t i16[4]={-32768,1,-32768,1}; check_narrow_keys(BoltType::Int16,i16);
}
TEST(BoltGroupbyNarrow, PayloadWidensExactly) {
    Arena arena; int64_t keys[3]={1,1,1}; uint32_t values[3]={UINT32_MAX,UINT32_MAX,0};
    auto key=BoltColumn::make_flat(keys,nullptr,3,BoltType::Int64);
    auto payload=BoltColumn::make_flat(values,nullptr,3,BoltType::UInt32);
    AggSpec specs[3]{}; specs[0].kind=AggKind::Sum; specs[1].kind=AggKind::Max; specs[2].kind=AggKind::Min;
    BoltColumn out[1]{},aggs[3]{}; uint32_t n=0;
    ASSERT_TRUE(groupby_agg_multi_key_typed(&key,1,&payload,1,specs,3,3,out,aggs,&n,&arena,1));
    ASSERT_EQ(n,1u);
    EXPECT_EQ(static_cast<const int64_t*>(aggs[0].data)[0],INT64_C(8589934590));
    EXPECT_EQ(static_cast<const int64_t*>(aggs[1].data)[0],INT64_C(4294967295));
    EXPECT_EQ(static_cast<const int64_t*>(aggs[2].data)[0],0);
}
void uint64_aggregates(GroupbyTypedState* state,BoltColumn* aggs,uint32_t* n) {
    BoltColumn out[1]{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_finalize(state,out,aggs,n));
    ASSERT_EQ(*n,1u);
    ASSERT_EQ(aggs[0].type,BoltType::UInt64); ASSERT_EQ(aggs[1].type,BoltType::UInt64);
    ASSERT_EQ(aggs[2].type,BoltType::Decimal128); ASSERT_EQ(aggs[3].type,BoltType::Float64);
    EXPECT_EQ(static_cast<const uint64_t*>(aggs[0].data)[0],0u);
    EXPECT_EQ(static_cast<const uint64_t*>(aggs[1].data)[0],UINT64_MAX);
    // 2 * (0 + (2^63-1) + 2^63 + (2^64-1)) = 2^66 - 4: past 64 bits, never wrapped.
    const auto sum=static_cast<const kernels::decimal::Decimal128*>(aggs[2].data)[0];
    uint64_t lo=0; std::memcpy(&lo,&sum.lo,8);
    EXPECT_EQ(lo,UINT64_MAX-3u); EXPECT_EQ(sum.hi,3);
    EXPECT_DOUBLE_EQ(static_cast<const double*>(aggs[3].data)[0],9223372036854775808.0);
}
TEST(BoltGroupbyUInt64, MinMaxSumAvgAreUnsignedAndExact) {
    for(bool merge:{false,true}) {
        Arena arena; Guarded input; int64_t keys[8]={}; 
        auto key=BoltColumn::make_flat(keys,nullptr,8,BoltType::Int64);
        auto payload=BoltColumn::make_flat(input.data,nullptr,8,BoltType::UInt64);
        AggSpec specs[4]{}; specs[0].kind=AggKind::Min; specs[1].kind=AggKind::Max;
        specs[2].kind=AggKind::Sum; specs[3].kind=AggKind::Avg;
        GroupbyTypedState state{},partial{};
        ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&state,&arena,&key,1,&payload,1,specs,4,8));
        ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&partial,&arena,&key,1,&payload,1,specs,4,8));
        auto first_key=key; first_key.length=4; auto first=payload; first.length=4;
        groupby_agg_multi_key_typed_ingest(&state,&first_key,&first,nullptr,0,4);
        auto second=payload; second.data=input.data+4; second.length=4;
        groupby_agg_multi_key_typed_ingest(merge?&partial:&state,&first_key,&second,nullptr,0,4);
        if(merge) ASSERT_TRUE(groupby_agg_multi_key_typed_merge(&state,&partial));
        BoltColumn aggs[4]{}; uint32_t n=0;
        uint64_aggregates(&state,aggs,&n); input.check();
    }
}
} // namespace
