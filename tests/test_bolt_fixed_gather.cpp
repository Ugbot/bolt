#include "bolt/kernels/bolt_fixed_gather.h"
#include <gtest/gtest.h>
#include <cstring>
#include <limits>

namespace {
using bolt::kernels::fixed_gather_to;
TEST(FixedGather, SelectedWideLiteralNullableViewIgnoresCachedAllValid) {
    std::uint64_t values[]={0,UINT64_MAX,0x8000000000000000ull,73};
    std::uint8_t bits=0x28; // offset3: row0 valid,row1 null,row2 valid,row3 null.
    auto src=bolt::BoltColumn::make_flat(values,&bits,4,bolt::BoltType::UInt64);
    src.format=bolt::ColumnFormat::View;src.validity_offset=3;src.stats.all_valid=true;
    const std::int32_t selected[]={2,1,0,2};
    std::uint8_t storage[34];std::memset(storage,0xa5,sizeof(storage));
    std::uint8_t validity[3]={0xa5,0xa5,0xa5};std::uint32_t nulls=99;
    ASSERT_TRUE(fixed_gather_to(src,selected,4,4,storage+1,32,validity+1,1,&nulls));
    const std::uint64_t expected[]={values[2],values[1],values[0],values[2]};
    EXPECT_EQ(std::memcmp(storage+1,expected,sizeof(expected)),0);EXPECT_EQ(validity[1],13);
    EXPECT_EQ(nulls,1u);EXPECT_EQ(storage[0],0xa5);EXPECT_EQ(storage[33],0xa5);
    EXPECT_EQ(validity[0],0xa5);EXPECT_EQ(validity[2],0xa5);
}
TEST(FixedGather, ConstantActualBitmapAllNullAndMixedRows) {
    auto src=bolt::BoltColumn::make_constant<std::int64_t>(-73,4,bolt::BoltType::Int64);
    std::uint8_t input=0;src.validity=&input;src.stats.all_valid=true;
    std::int64_t out[4]{};std::uint8_t validity=255;std::uint32_t nulls=0;
    ASSERT_TRUE(fixed_gather_to(src,nullptr,4,4,out,sizeof(out),&validity,1,&nulls));
    for(auto x:out)EXPECT_EQ(x,-73);EXPECT_EQ(validity,0);EXPECT_EQ(nulls,4u);
    input=10;src.stats.null_count=4;
    const std::int32_t selected[]={3,1,0};
    ASSERT_TRUE(fixed_gather_to(src,selected,3,4,out,sizeof(out),&validity,1,&nulls));
    EXPECT_EQ(validity,3);EXPECT_EQ(nulls,1u);
}
TEST(FixedGather, ExactScalarWidthsBoolAndFloatBitPatterns) {
    const bolt::BoltType types[]={bolt::BoltType::Bool,bolt::BoltType::Int8,bolt::BoltType::UInt8,
        bolt::BoltType::Int16,bolt::BoltType::UInt16,bolt::BoltType::Int32,bolt::BoltType::UInt32,
        bolt::BoltType::Int64,bolt::BoltType::UInt64,bolt::BoltType::Float64};
    const std::uint64_t values[]={0x7ff8000000000042ull,0x8000000000000000ull};
    const std::int32_t selection[]={1,0};
    for(const auto type:types) {
        SCOPED_TRACE(static_cast<unsigned>(type));const auto width=bolt::type_size(type);
        std::uint8_t source[16]{};std::memcpy(source,values,width);std::memcpy(source+width,values+1,width);
        if(type==bolt::BoltType::Bool){source[0]=1;source[1]=0;}
        auto src=bolt::BoltColumn::make_flat(source,nullptr,2,type);
        std::uint8_t out[16]{};std::uint8_t validity=0;std::uint32_t nulls=99;
        ASSERT_TRUE(fixed_gather_to(src,selection,2,2,out,sizeof(out),&validity,1,&nulls));
        EXPECT_EQ(std::memcmp(out,source+width,width),0);EXPECT_EQ(std::memcmp(out+width,source,width),0);
        EXPECT_EQ(validity,3);EXPECT_EQ(nulls,0u);
    }
}
TEST(FixedGather, RefusalsPreserveAllOutputBytesAndCount) {
    std::uint64_t values[]={1,2,3};std::uint8_t bits=7;
    const auto base=bolt::BoltColumn::make_flat(values,&bits,3,bolt::BoltType::UInt64);
    for(unsigned mutation=0;mutation<12;++mutation) {
        SCOPED_TRACE(mutation);auto src=base;std::uint64_t out[3];std::memset(out,0xa5,sizeof(out));
        const std::uint64_t expected[3]={out[0],out[1],out[2]};std::uint8_t validity=0xa5;std::uint32_t nulls=99;
        std::int32_t selected[]={2,0,1};std::uint64_t bytes=sizeof(out),vb=1;std::int64_t rows=3,input=3;
        const std::int32_t* selection=selected;
        if(mutation==0)selected[2]=-1;if(mutation==1)selected[2]=3;
        if(mutation==2)bytes=23;if(mutation==3)vb=0;
        if(mutation==4)src.format=bolt::ColumnFormat::Sequence;
        if(mutation==5)src.type=bolt::BoltType::Float32;
        if(mutation==6)src.type_size_bytes=4;
        if(mutation==7)src.validity_offset=INT64_MAX;
        if(mutation==8){selection=nullptr;rows=2;}
        if(mutation==9)input=4;if(mutation==10)rows=-1;
        if(mutation==11)src.logical=bolt::BoltLogical::Variant;
        EXPECT_FALSE(fixed_gather_to(src,selection,rows,input,out,bytes,&validity,vb,&nulls));
        EXPECT_EQ(std::memcmp(out,expected,sizeof(out)),0);EXPECT_EQ(validity,0xa5);EXPECT_EQ(nulls,99u);
    }
}
TEST(FixedGather, AliasesAndOverflowRefuseAndZeroRowsAreExplicit) {
    std::uint64_t values[]={1,2};auto src=bolt::BoltColumn::make_flat(values,nullptr,2,bolt::BoltType::UInt64);
    std::uint8_t validity=0xa5;std::uint32_t nulls=99;std::uint64_t out[2]{};
    EXPECT_FALSE(fixed_gather_to(src,nullptr,2,2,values,sizeof(values),&validity,1,&nulls));
    EXPECT_EQ(values[0],1u);EXPECT_EQ(values[1],2u);EXPECT_EQ(nulls,99u);
    EXPECT_FALSE(fixed_gather_to(src,nullptr,2,2,out,sizeof(out),reinterpret_cast<std::uint8_t*>(out),1,&nulls));
    EXPECT_FALSE(fixed_gather_to(src,nullptr,2,2,out,sizeof(out),&validity,1,reinterpret_cast<std::uint32_t*>(out)));
    EXPECT_FALSE(fixed_gather_to(src,nullptr,INT64_MAX,INT64_MAX,out,sizeof(out),&validity,1,&nulls));
    ASSERT_TRUE(fixed_gather_to(src,nullptr,0,0,nullptr,0,nullptr,0,&nulls));EXPECT_EQ(nulls,0u);
}
} // namespace
