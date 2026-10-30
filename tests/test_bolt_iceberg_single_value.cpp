// Appendix D vectors and refusal controls; no production writer as an oracle.
#include <gtest/gtest.h>
#include "bolt/lakehouse/iceberg/partition_values.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {
using namespace bolt::lakehouse::iceberg;
IcebergTypeInfo type(const char* name) {
    IcebergTypeInfo out{};EXPECT_TRUE(iceberg_type_parse(name,&out));return out;
}
PartitionValue integer(int64_t x) {PartitionValue v{};v.is_int=true;v.i64=x;return v;}
PartitionValue bytes(const std::vector<uint8_t>& input) {
    PartitionValue value{};value.is_str=true;
    EXPECT_LT(input.size(),sizeof(value.str));value.str_len=static_cast<uint8_t>(input.size());
    if(!input.empty())std::memcpy(value.str,input.data(),input.size());return value;
}
std::vector<uint8_t> unhex(const std::string& text) {
    std::vector<uint8_t> out;
    for(size_t i=0;i+1<text.size();i+=2)out.push_back(static_cast<uint8_t>(std::stoul(text.substr(i,2),nullptr,16)));
    return out;
}
void refused_encoding(const IcebergTypeInfo& t,const PartitionValue& value,uint32_t cap=64) {
    std::array<uint8_t,64> buffer{};buffer.fill(0xA5);const auto before=buffer;uint32_t length=91;
    EXPECT_FALSE(partition_value_encode(t,value,buffer.data(),cap,&length));
    EXPECT_EQ(buffer,before);EXPECT_EQ(length,0u);
}
void refused_decoding(const IcebergTypeInfo& t,const uint8_t* input,uint32_t length) {
    auto value=integer(919);const auto before=value;
    EXPECT_FALSE(partition_value_decode(t,input,length,&value));
    EXPECT_EQ(std::memcmp(&value,&before,sizeof(value)),0);
}
TEST(IcebergSingleValue, IntegerWidthsEndpointsAndTypedLanes) {
    for(const char* name:{"int","date","long","timestamp","timestamptz","timestamp_ns","timestamptz_ns"}) {
        const auto t=type(name);const bool narrow=t.type==IcebergType::kInt || t.type==IcebergType::kDate;
        const uint32_t width=narrow?4u:8u;
        for(const int64_t x:{narrow?int64_t{INT32_MIN}:INT64_MIN,int64_t{-1},int64_t{0},narrow?int64_t{INT32_MAX}:INT64_MAX}) {
            const auto value=integer(x);std::array<uint8_t,16> encoded{};uint32_t count=0;
            ASSERT_TRUE(partition_value_encode(t,value,encoded.data(),encoded.size(),&count));ASSERT_EQ(count,width);
            for(uint32_t i=0;i<count;++i)EXPECT_EQ(encoded[i],static_cast<uint8_t>(uint64_t(x)>>(8u*i)));
            PartitionValue decoded{};ASSERT_TRUE(partition_value_decode(t,encoded.data(),count,&decoded));
            EXPECT_TRUE(decoded.is_int);EXPECT_EQ(decoded.i64,x);
            for(uint32_t bad=0;bad<=9;++bad)if(bad!=width)refused_decoding(t,encoded.data(),bad);
            refused_encoding(t,value,width-1u);
        }
        PartitionValue wrong{};wrong.is_float=true;wrong.f64=1;refused_encoding(t,wrong);
        if(narrow)refused_encoding(t,integer(int64_t{INT32_MAX}+1));
    }
}
TEST(IcebergSingleValue, BooleanNormalizationAndTimeRange) {
    const auto boolean=type("boolean"),time=type("time");
    for(const uint8_t b:{uint8_t{0},uint8_t{1},uint8_t{0x80},uint8_t{0xFF}}) {
        PartitionValue value{};ASSERT_TRUE(partition_value_decode(boolean,&b,1,&value));EXPECT_EQ(value.i64,b!=0);
        uint8_t encoded=99;uint32_t count=0;ASSERT_TRUE(partition_value_encode(boolean,value,&encoded,1,&count));
        EXPECT_EQ(encoded,b!=0);EXPECT_EQ(count,1u);
    }
    refused_encoding(boolean,integer(256));refused_encoding(boolean,integer(-1));
    refused_encoding(time,integer(-1));refused_encoding(time,integer(INT64_C(86400000000)));
    const uint8_t invalid[8]{0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff};refused_decoding(time,invalid,8);
}
TEST(IcebergSingleValue, FloatingBitsSignedZerosAndCapacity) {
    for(const char* name:{"float","double"})for(const double x:{-0.0,0.0,-1.25,1.5,
        std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        const auto t=type(name);const uint32_t width=t.type==IcebergType::kFloat?4u:8u;
        PartitionValue value{};value.is_float=true;value.f64=x;
        uint8_t encoded[8]{};uint32_t count=0;ASSERT_TRUE(partition_value_encode(t,value,encoded,8,&count));EXPECT_EQ(count,width);
        const uint64_t bits=width==4?std::bit_cast<uint32_t>(static_cast<float>(x)):std::bit_cast<uint64_t>(x);
        for(uint32_t i=0;i<count;++i)EXPECT_EQ(encoded[i],static_cast<uint8_t>(bits>>(8u*i)));
        PartitionValue decoded{};ASSERT_TRUE(partition_value_decode(t,encoded,count,&decoded));EXPECT_TRUE(decoded.is_float);
        if(std::isnan(x))EXPECT_TRUE(std::isnan(decoded.f64));else {EXPECT_EQ(decoded.f64,x);EXPECT_EQ(std::signbit(decoded.f64),std::signbit(x));}
        refused_encoding(t,value,width-1u);refused_encoding(t,integer(1));
    }
    PartitionValue too_large{};too_large.is_float=true;too_large.f64=std::numeric_limits<double>::max();
    refused_encoding(type("float"),too_large);
}
TEST(IcebergSingleValue, ByteTypesFixedLengthsAndEmbeddedNulls) {
    for(const char* name:{"string","binary","fixed[3]","uuid","geometry","geography"}) {
        const auto t=type(name);const uint32_t count=t.type==IcebergType::kUuid?16u:3u;
        std::vector<uint8_t> input(count,0xFF);input[1]=0;const auto value=bytes(input);
        uint8_t encoded[64]{};uint32_t size=0;ASSERT_TRUE(partition_value_encode(t,value,encoded,sizeof(encoded),&size));
        ASSERT_EQ(size,count);EXPECT_EQ(std::memcmp(encoded,input.data(),count),0);
        PartitionValue decoded{};ASSERT_TRUE(partition_value_decode(t,input.data(),count,&decoded));
        EXPECT_TRUE(decoded.is_str);EXPECT_EQ(partition_value_bytes_len(decoded),count);
        EXPECT_EQ(std::memcmp(decoded.str,input.data(),count),0);refused_encoding(t,value,count-1u);
        if(t.type==IcebergType::kUuid || t.type==IcebergType::kFixed)refused_decoding(t,input.data(),count-1u);
        refused_encoding(t,integer(1));
    }
    for(const char* name:{"string","binary"}) {
        const auto t=type(name);PartitionValue decoded{};ASSERT_TRUE(partition_value_decode(t,nullptr,0,&decoded));
        uint8_t guard=0xA5;uint32_t size=99;ASSERT_TRUE(partition_value_encode(t,decoded,&guard,0,&size));
        EXPECT_EQ(size,0u);EXPECT_EQ(guard,0xA5);
    }
    auto over=bytes({1});over.str_len=UINT8_MAX;refused_encoding(type("binary"),over);
    auto unterminated=bytes({1});unterminated.str_len=0;std::memset(unterminated.str,'x',sizeof(unterminated.str));
    refused_encoding(type("binary"),unterminated);
    std::array<uint8_t,64> too_wide{};refused_decoding(type("binary"),too_wide.data(),64);

}
TEST(IcebergSingleValue, DecimalMinimalSignEncodingAndDeclaredPrecision) {
    const auto t=type("decimal(3,2)");
    for(const char* hex:{"00","7f","0080","80","ff7f","03e7","fc19"}) {
        const auto input=unhex(hex);PartitionValue value{};ASSERT_TRUE(partition_value_decode(t,input.data(),static_cast<uint32_t>(input.size()),&value));
        uint8_t encoded[16]{};uint32_t size=0;ASSERT_TRUE(partition_value_encode(t,value,encoded,16,&size));
        ASSERT_EQ(size,input.size());EXPECT_EQ(std::memcmp(encoded,input.data(),size),0);
        refused_encoding(t,value,size-1u);
    }
    for(const char* hex:{"03e8","fc18","7fffffffffffffffffffffffffffffff"}) {
        const auto input=unhex(hex);refused_decoding(t,input.data(),static_cast<uint32_t>(input.size()));
        refused_encoding(t,bytes(input));
    }
    const uint8_t sign_extended[]{0xFF,0xFF,0x80};PartitionValue value{};
    ASSERT_TRUE(partition_value_decode(t,sign_extended,3,&value));
    uint8_t encoded[16]{};uint32_t size=0;ASSERT_TRUE(partition_value_encode(t,value,encoded,16,&size));
    EXPECT_EQ(size,1u);EXPECT_EQ(encoded[0],0x80);
}
TEST(IcebergSingleValue, UnsupportedAndPartialValuesDoNotPublish) {
    const auto value=integer(1);refused_encoding(IcebergTypeInfo{},value);
    const uint8_t input=1;refused_decoding(IcebergTypeInfo{},&input,1);
    for(const char* name:{"unknown","struct","list","map","variant"}) {
        IcebergTypeInfo t{};EXPECT_FALSE(iceberg_type_parse(name,&t));refused_encoding(t,value);refused_decoding(t,&input,1);
    }
    auto partial=value;partial.is_partial=true;refused_encoding(type("long"),partial);
    partial=value;partial.is_null=true;refused_encoding(type("long"),partial);
}
TEST(IcebergSingleValue, UuidOrderingSupportsBothUnsignedAndJavaSignedOrder) {
    std::vector<uint8_t> low(16,0),high(16,0);high[0]=0x80;
    const auto a=bytes(low),b=bytes(high);const auto t=type("uuid");
    EXPECT_LT(partition_value_compare(t,a,b,false),0);EXPECT_GT(partition_value_compare(t,a,b,true),0);
    high[0]=0;high[8]=0x80;const auto c=bytes(high);
    EXPECT_LT(partition_value_compare(t,a,c,false),0);EXPECT_GT(partition_value_compare(t,a,c,true),0);
}
TEST(IcebergSingleValue, ExternalOracle) {
    const char* path=std::getenv("BOLT_SINGLE_VALUE_INPUT");if(!path)GTEST_SKIP()<<"independent conversion fixture";
    std::ifstream input(path);ASSERT_TRUE(input.good());std::string line;uint32_t count=0;
    for(uint32_t guard=0;guard<4096 && std::getline(input,line);++guard) {
        const auto tab=line.find('\t');ASSERT_NE(tab,std::string::npos);
        const auto t=type(line.substr(0,tab).c_str());const auto data=unhex(line.substr(tab+1));
        PartitionValue decoded{};ASSERT_TRUE(partition_value_decode(t,data.data(),static_cast<uint32_t>(data.size()),&decoded))<<line;
        uint8_t encoded[64]{};uint32_t size=0;ASSERT_TRUE(partition_value_encode(t,decoded,encoded,sizeof(encoded),&size))<<line;
        ASSERT_EQ(size,data.size());if(size)EXPECT_EQ(std::memcmp(encoded,data.data(),size),0)<<line;
        ++count;
    }
    EXPECT_TRUE(input.eof());EXPECT_GE(count,100u);
}
} // namespace
