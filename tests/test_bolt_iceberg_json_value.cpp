#include "bolt/lakehouse/iceberg/json_value.h"
#include "bolt/kernels/bolt_nested_gather.h"
#include "bolt/bolt_nested_type.h"
#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

namespace {
using namespace bolt;
using namespace bolt::lakehouse::iceberg;
using S=JsonValueStatus;
SchemaField field(const char* type,int32_t id=1,bool required=false) {
    SchemaField f{};f.id=id;f.required=required;std::strcpy(f.type,type);std::strcpy(f.name,"value");return f;
}
S decode(const SchemaField& f,const std::string& json,Arena& arena,BoltColumn* out,const JsonValueLimits& limits={}) {
    return json_value_decode(f,reinterpret_cast<const uint8_t*>(json.data()),static_cast<uint32_t>(json.size()),&arena,out,limits);
}
std::string encode(const SchemaField& f,const BoltColumn& col,Arena& arena,int64_t row=0) {
    const uint8_t* bytes=nullptr;uint32_t size=0;
    EXPECT_EQ(json_value_encode(f,col,row,&arena,&bytes,&size),S::Ok);
    return bytes?std::string(reinterpret_cast<const char*>(bytes),size):"";
}
int64_t integer(const BoltColumn& c) {
    int64_t value=0;
    if(c.type_size_bytes==4) {int32_t x=0;std::memcpy(&x,c.data,4);return x;}
    std::memcpy(&value,c.data,c.type_size_bytes);return value;
}
std::string bytes(const BoltColumn& c,int64_t row=0) {
    const uint8_t* p=nullptr;int32_t n=0;c.utf8_at(row,&p,&n);return std::string(reinterpret_cast<const char*>(p),static_cast<size_t>(n));
}
TEST(IcebergJsonValue, ExactIntegerTemporalLanesAndEndpoints) {
    struct Case {const char* type;const char* json;BoltType lane;int64_t expected;};
    const Case cases[]{
        {"boolean","true",BoltType::Bool,1},{"int","-2147483648",BoltType::Int32,INT32_MIN},
        {"long","9223372036854775807",BoltType::Int64,INT64_MAX},
        {"long","-9223372036854775808",BoltType::Int64,INT64_MIN},
        {"date","\"1969-12-31\"",BoltType::Date32,-1},
        {"time","\"23:59:59.999999\"",BoltType::Duration,INT64_C(86399999999)},
        {"timestamp","\"1969-12-31T23:59:59.999999\"",BoltType::Timestamp,-1},
        {"timestamptz","\"1970-01-01T00:00:00+00:00\"",BoltType::Timestamp,0},
        {"timestamp_ns","\"1677-09-21T00:12:43.145224192\"",BoltType::Int64,INT64_MIN},
        {"timestamp_ns","\"2262-04-11T23:47:16.854775807\"",BoltType::Int64,INT64_MAX},
        {"timestamptz_ns","\"1969-12-31T23:59:59.999999999+00:00\"",BoltType::Int64,-1}};
    for(const auto& c:cases) {
        SCOPED_TRACE(c.type);Arena arena;BoltColumn out{};const auto f=field(c.type);
        ASSERT_EQ(decode(f,c.json,arena,&out),S::Ok);EXPECT_EQ(out.type,c.lane);EXPECT_EQ(out.length,1);
        EXPECT_FALSE(out.is_null(0));EXPECT_EQ(integer(out),c.expected);EXPECT_EQ(encode(f,out,arena),c.json);
    }
}
TEST(IcebergJsonValue, DecimalFloatAndOwnedBytes) {
    Arena arena;BoltColumn c{};
    auto f=field("decimal(30,2)");ASSERT_EQ(decode(f,"\"-12345678901234567890.12\"",arena,&c),S::Ok);
    int64_t lo=0,hi=0;std::memcpy(&lo,c.data,8);std::memcpy(&hi,static_cast<const uint8_t*>(c.data)+8,8);
    EXPECT_EQ(c.type,BoltType::Decimal128);EXPECT_EQ(c.decimal_scale,2);EXPECT_EQ(hi,-67);
    EXPECT_EQ(static_cast<uint64_t>(lo),UINT64_C(0x12edc4f427dfc5ec));
    EXPECT_EQ(encode(f,c,arena),"\"-12345678901234567890.12\"");
    f=field("decimal(9,2)");ASSERT_EQ(decode(f,"\"1E-2\"",arena,&c),S::Ok);EXPECT_EQ(integer(c),1);
    EXPECT_EQ(encode(f,c,arena),"\"0.01\"");
    f=field("float");ASSERT_EQ(decode(f,"0.1",arena,&c),S::Ok);
    EXPECT_EQ(*static_cast<const double*>(c.data),static_cast<double>(0.1f));
    f=field("double");ASSERT_EQ(decode(f,"-0.0",arena,&c),S::Ok);EXPECT_TRUE(std::signbit(*static_cast<const double*>(c.data)));
    f=field("string");std::string source="\"a\\\"b\\u00e9\\ud83d\\ude00\\u0000tail\"";
    ASSERT_EQ(decode(f,source,arena,&c),S::Ok);source.assign(source.size(),'x');
    EXPECT_EQ(bytes(c),std::string("a\"b\xc3\xa9\xf0\x9f\x98\x80\0tail",14));
    for(const char* type:{"binary","fixed[3]","fixed[17]","uuid"}) {
        f=field(type);const std::string text=std::string(type)=="uuid"?"\"f79c3e09-677c-4bbd-a479-3f349cb785e7\"":
            std::string(type)=="fixed[17]"?"\"000102030405060708090a0b0c0d0e0f10\"":"\"0a00ff\"";
        ASSERT_EQ(decode(f,text,arena,&c),S::Ok)<<type;EXPECT_EQ(encode(f,c,arena),text);
    }
    f=field("binary");ASSERT_EQ(decode(f,"\"\"",arena,&c),S::Ok);EXPECT_EQ(bytes(c),"");EXPECT_EQ(encode(f,c,arena),"\"\"");
}
TEST(IcebergJsonValue, NestedFieldIdsNullsMapValuesAndRepeatedRows) {
    SchemaField root=field("struct"),children[]{field("int",2,true),field("list",3),field("map",5)};
    SchemaField element=field("string",4),map_children[]{field("string",6,true),field("long",7)};
    children[1].children=&element;children[1].n_children=1;children[2].children=map_children;children[2].n_children=2;
    root.children=children;root.n_children=3;
    std::string source=R"({"5":{"keys":["a","b"],"values":[9223372036854775807,null]},"2":7,"3":[null,"","雪"]})";
    Arena arena;BoltColumn c{};ASSERT_EQ(decode(root,source,arena,&c),S::Ok);source.assign(source.size(),'x');
    ASSERT_EQ(c.child_count(),3);EXPECT_EQ(integer(*c.child_at(0)),7);
    const auto* list=c.child_at(1);ASSERT_EQ(list->list_element()->length,3);EXPECT_TRUE(list->list_element()->is_null(0));
    EXPECT_EQ(bytes(*list->list_element(),2),"雪");
    const auto* entries=c.child_at(2)->list_element();EXPECT_EQ(integer(*entries->child_at(1)),INT64_MAX);
    EXPECT_TRUE(entries->child_at(1)->is_null(1));
    const uint32_t selected[]{0,0,0};BoltColumn repeated{};ASSERT_TRUE(kernels::nested_gather(c,selected,3,&arena,&repeated));
    ASSERT_EQ(repeated.length,3);EXPECT_EQ(repeated.child_at(1)->list_offsets()[3],9);
    const auto canonical=encode(root,c,arena);EXPECT_EQ(encode(root,repeated,arena,2),canonical);
    EXPECT_EQ(canonical,R"({"2":7,"3":[null,"","雪"],"5":{"keys":["a","b"],"values":[9223372036854775807,null]}})");
    ASSERT_EQ(decode(root,"null",arena,&c),S::Ok);EXPECT_TRUE(c.is_null(0));EXPECT_TRUE(c.child_at(0)->is_null(0));
}
TEST(IcebergJsonValue, EmptyStructIsAValueAndRetainsTypeThroughGather) {
    Arena arena;const auto f=field("struct");BoltColumn c{};ASSERT_EQ(decode(f,"{}",arena,&c),S::Ok);
    EXPECT_EQ(c.type,BoltType::Struct);EXPECT_EQ(c.child_count(),0);EXPECT_FALSE(c.is_null(0));EXPECT_EQ(encode(f,c,arena),"{}");
    nested_row::TypeTree tree{};ASSERT_TRUE(nested_row::tree_from_column(c,&tree));EXPECT_TRUE(nested_row::validate_tree(tree));
    const uint32_t rows[]{0,0};BoltColumn repeat{};ASSERT_TRUE(kernels::nested_gather(c,rows,2,&arena,&repeat));
    EXPECT_EQ(repeat.length,2);EXPECT_EQ(encode(f,repeat,arena,1),"{}");
    ASSERT_EQ(decode(f,"null",arena,&c),S::Ok);EXPECT_TRUE(c.is_null(0));EXPECT_EQ(encode(f,c,arena),"null");
}
TEST(IcebergJsonValue, RefusalsAreTransactionalAndBudgeted) {
    struct Case {const char* type;const char* json;};
    const Case bad[]{ {"boolean","1"},{"int","2147483648"},{"long","9223372036854775808"},
        {"int","1.0"},{"float","3.5e38"},{"double","1e999"},{"decimal(3,2)","\"10.00\""},
        {"decimal(3,2)","\"1.0\""},{"date","\"2023-02-29\""},{"time","\"24:00:00\""},
        {"time","\"12:34.1\""},{"time","\"12:34:56.1234567\""},{"timestamp","\"1970-01-01T00:00:00Z\""},
        {"timestamptz","\"1970-01-01T00:00:00+01:00\""},{"timestamp_ns","\"2262-04-11T23:47:16.854775808\""},
        {"timestamp_ns","\"1677-09-21T00:12:43.145224191\""},{"uuid","\"bad\""},{"fixed[3]","\"00\""},
        {"binary","\"0g\""},{"string","\"\\q\""}};
    Arena arena;BoltColumn c{};c.length=919;const auto saved=c;
    for(const auto& v:bad) {EXPECT_EQ(decode(field(v.type),v.json,arena,&c),S::Invalid)<<v.type<<v.json;EXPECT_EQ(std::memcmp(&c,&saved,sizeof(c)),0);}
    EXPECT_EQ(decode(field("long",1,true),"null",arena,&c),S::Invalid);
    for(const char* type:{"unknown","variant","geometry","geography"}) {
        EXPECT_EQ(decode(field(type),"\"POINT (1 2)\"",arena,&c),S::Unsupported);
        EXPECT_EQ(decode(field(type),"null",arena,&c),S::Ok);
    }
    JsonValueLimits limits{};limits.json_bytes=2;EXPECT_EQ(decode(field("long"),"123",arena,&c,limits),S::Capacity);
    limits={};limits.encoded_bytes=2;EXPECT_EQ(decode(field("long"),"1",arena,&c,limits),S::Capacity);
    const auto f=field("long");ASSERT_EQ(decode(f,"1",arena,&c),S::Ok);
    const uint8_t* output=reinterpret_cast<const uint8_t*>(uintptr_t{1});uint32_t size=919;
    EXPECT_EQ(json_value_encode(f,c,1,&arena,&output,&size),S::Invalid);EXPECT_EQ(size,919u);EXPECT_EQ(output,reinterpret_cast<const uint8_t*>(uintptr_t{1}));
}
TEST(IcebergJsonValue, NestedRequirednessDuplicateAliasesAndSemanticMapKeys) {
    SchemaField root=field("struct"),child=field("long",2,true);root.children=&child;root.n_children=1;
    Arena arena;BoltColumn c{};
    for(const char* text:{"{}","{\"2\":null}","{\"3\":1}","{\"2\":1,\"\\u0032\":2}"})EXPECT_EQ(decode(root,text,arena,&c),S::Invalid)<<text;
    EXPECT_EQ(decode(root,"{\"\\u0032\":1}",arena,&c),S::Ok);
    root=field("map");SchemaField kv[]{field("string",2,true),field("long",3)};root.children=kv;root.n_children=2;
    for(const char* text:{R"({"keys":["a","\u0061"],"values":[1,2]})",R"({"keys":[null],"values":[1]})",
        R"({"keys":["a"],"values":[]})",R"({"keys":[],"keys":[]})"})EXPECT_EQ(decode(root,text,arena,&c),S::Invalid)<<text;
    EXPECT_EQ(decode(root,R"({"keys":[],"values":[]})",arena,&c),S::Ok);
    root.children[1].id=2;EXPECT_EQ(decode(root,R"({"keys":[],"values":[]})",arena,&c),S::Invalid);
}
TEST(IcebergJsonValue, ExternalTypedOracle) {
    const char* input=std::getenv("BOLT_JSON_VALUE_INPUT"),*output=std::getenv("BOLT_JSON_VALUE_OUTPUT");
    if(!input || !output)GTEST_SKIP()<<"independent typed fixture";
    std::ifstream file(input);std::string source((std::istreambuf_iterator<char>(file)),{});ASSERT_TRUE(file.good() || file.eof());
    Arena arena;Metadata metadata{};ASSERT_TRUE(metadata_parse(reinterpret_cast<const uint8_t*>(source.data()),static_cast<uint32_t>(source.size()),&arena,&metadata));
    source.assign(source.size(),'x');const auto* schema=metadata_current_schema(&metadata);ASSERT_NE(schema,nullptr);
    std::ofstream result(output);result<<"[";uint32_t count=0;
    for(uint32_t i=0;i<schema->n_fields;++i)for(const auto d:{schema->fields[i].initial_default_value,schema->fields[i].write_default_value}) {
        if(!d.json)continue;BoltColumn value{};ASSERT_EQ(json_value_decode(schema->fields[i],d.json,d.size,&arena,&value),S::Ok)<<i;
        auto copy=schema->fields[i];std::strcpy(copy.name,"renamed");const auto actual=encode(copy,value,arena);
        if(count++)result<<",";result<<actual;
    }
    result<<"]";ASSERT_TRUE(result.good());EXPECT_GE(count,40u);
}
} // namespace
