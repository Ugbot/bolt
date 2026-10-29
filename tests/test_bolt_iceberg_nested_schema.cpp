#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"
#include "bolt/ingest/bolt_parquet_nested.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/bolt_budget.h"
#include <gtest/gtest.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace {
namespace bi=bolt::lakehouse::iceberg;
namespace pq=bolt::ingest::parquet;
const char kFields[]=R"([{"id":1,"name":"st","required":false,"type":{"type":"struct","fields":[
{"id":2,"name":"a","required":true,"type":"int"},
{"id":3,"name":"xs","required":false,"type":{"type":"list","element-id":4,"element-required":false,"element":"string"}}]}},
{"id":5,"name":"mp","required":false,"type":{"type":"map","key-id":6,"key":"string","value-id":7,
"value-required":false,"value":{"type":"struct","fields":[{"id":8,"name":"v","required":false,"type":"long"}]}}},
{"id":9,"name":"ll","required":false,"type":{"type":"list","element-id":10,"element-required":false,
"element":{"type":"list","element-id":11,"element-required":false,"element":"fixed[4]"}}}])";
std::string document(const std::string& fields) {
    return std::string(R"({"format-version":2,"table-uuid":"12345678-1234-1234-1234-123456789abc",
"location":"s3://nested-bucket/table","last-updated-ms":1,"last-sequence-number":0,"current-snapshot-id":null,
"current-schema-id":0,"schemas":[{"type":"struct","schema-id":0,"fields":)")+fields+R"(}],
"default-spec-id":0,"partition-specs":[{"spec-id":0,"fields":[]}],"default-sort-order-id":0,
"sort-orders":[{"order-id":0,"fields":[]}],"snapshots":[]})";
}
bi::Metadata* parse(bolt::Arena& arena,const std::string& text) {
    auto* out=arena.allocate_array<bi::Metadata>(1);
    return out && bi::metadata_parse(reinterpret_cast<const uint8_t*>(text.data()),static_cast<uint32_t>(text.size()),&arena,out)?out:nullptr;
}
std::string emit(bolt::Arena& arena,const bi::Metadata& metadata) {
    const uint8_t* out=nullptr;uint64_t size=0;
    if (!bi::metadata_json_emit_plain(&metadata,&arena,&out,&size)) return {};
    return std::string(reinterpret_cast<const char*>(out),size);
}
TEST(IcebergNestedSchema, RetainsStructListMapIdsAndRequiredness) {
    bolt::Arena arena;auto* metadata=parse(arena,document(kFields));ASSERT_NE(metadata,nullptr);
    const auto& schema=metadata->schemas[0];ASSERT_EQ(schema.n_fields,3u);
    const auto& st=schema.fields[0];EXPECT_STREQ(st.type,"struct");ASSERT_EQ(st.n_children,2u);
    EXPECT_TRUE(st.children[0].required);EXPECT_EQ(st.children[0].id,2);
    const auto& xs=st.children[1];EXPECT_STREQ(xs.type,"list");ASSERT_EQ(xs.n_children,1u);
    EXPECT_EQ(xs.children[0].id,4);EXPECT_FALSE(xs.children[0].required);EXPECT_STREQ(xs.children[0].type,"string");
    const auto& map=schema.fields[1];EXPECT_STREQ(map.type,"map");ASSERT_EQ(map.n_children,2u);
    EXPECT_TRUE(map.children[0].required);EXPECT_EQ(map.children[1].id,7);
    EXPECT_EQ(map.children[1].children[0].id,8);
    int32_t max=0;ASSERT_TRUE(bi::schema_max_field_id(&schema,&max));EXPECT_EQ(max,11);
}
TEST(IcebergNestedSchema, CopyHasIndependentChildrenAfterSourceReset) {
    bolt::Arena source,target;auto* metadata=parse(source,document(kFields));ASSERT_NE(metadata,nullptr);
    bi::Schema copied{};ASSERT_TRUE(bi::schema_copy(&copied,&metadata->schemas[0],&target,nullptr));
    ASSERT_NE(copied.fields[0].children,metadata->schemas[0].fields[0].children);
    source.reset();EXPECT_STREQ(copied.fields[1].children[1].children[0].name,"v");
    EXPECT_EQ(copied.fields[2].children[0].children[0].id,11);
    EXPECT_TRUE(bi::schema_validate(&copied,&target,nullptr));
}
TEST(IcebergNestedSchema, EmittedTypesRoundTripAndMaximumIncludesElementIds) {
    bolt::Arena arena;auto* original=parse(arena,document(kFields));ASSERT_NE(original,nullptr);
    const auto text=emit(arena,*original);ASSERT_FALSE(text.empty());
    EXPECT_NE(text.find("\"last-column-id\":11"),std::string::npos);
    auto* round=parse(arena,text);ASSERT_NE(round,nullptr);
    EXPECT_STREQ(round->schemas[0].fields[2].children[0].children[0].type,"fixed[4]");
    EXPECT_EQ(round->schemas[0].fields[1].children[1].children[0].id,8);
    bi::NameMapEntry* names=nullptr;uint32_t count=0;
    ASSERT_TRUE(bi::metadata_name_mapping(round,&arena,&names,&count));ASSERT_EQ(count,11u);
    for (int32_t id=1;id<=11;++id) {
        uint32_t hits=0;for (uint32_t i=0;i<count;++i) hits+=names[i].field_id==id?1u:0u;
        EXPECT_EQ(hits,1u);
    }
}
TEST(IcebergNestedSchema, MalformedShapesAndGlobalDuplicateIdsRefuse) {
    for (const auto& change : {std::pair{"\"id\":8","\"id\":2"},
         std::pair{"\"element-id\":4","\"element-id\":-1"},
         std::pair{"\"element-required\":false,\"element\":\"string\"","\"element\":\"string\""},
         std::pair{"\"value-id\":7","\"value-id\":7.5"},std::pair{"\"type\":\"map\"","\"type\":\"bogus\""}}) {
        std::string fields=kFields;const auto at=fields.find(change.first);ASSERT_NE(at,std::string::npos);
        fields.replace(at,std::strlen(change.first),change.second);bolt::Arena arena;
        EXPECT_EQ(parse(arena,document(fields)),nullptr)<<fields;
    }
}
TEST(IcebergNestedSchema, AmbiguousNestedNameMappingRefuses) {
    constexpr char mapping[]=R"([{\"field-id\":1,\"names\":[\"a\"],\"fields\":[{\"field-id\":1,\"names\":[\"b\"]}]}])";
    bolt::Arena arena;auto* metadata=arena.allocate_array<bi::Metadata>(1);ASSERT_NE(metadata,nullptr);
    bi::metadata_init(metadata,&arena,nullptr);metadata->name_mapping_json=mapping;metadata->name_mapping_len=sizeof(mapping)-1u;
    bi::NameMapEntry* out=nullptr;uint32_t count=0;
    EXPECT_FALSE(bi::metadata_name_mapping(metadata,&arena,&out,&count));EXPECT_EQ(out,nullptr);EXPECT_EQ(count,0u);
}
TEST(IcebergNestedSchema, DepthCyclesAndBudgetRefuseWithoutPublishingCopy) {
    std::string boundary="\"long\"";
    for (uint32_t i=0;i<bi::kIcebergSchemaMaxDepth;++i)
        boundary="{\"type\":\"list\",\"element-id\":"+std::to_string(i+2u)+",\"element-required\":false,\"element\":"+boundary+"}";
    bolt::Arena exact;EXPECT_NE(parse(exact,document("[{\"id\":1,\"name\":\"x\",\"type\":"+boundary+"}]")),nullptr);
    std::string type="\"long\"";
    for (uint32_t i=0;i<bi::kIcebergSchemaMaxDepth+2u;++i)
        type="{\"type\":\"list\",\"element-id\":"+std::to_string(i+2u)+",\"element-required\":false,\"element\":"+type+"}";
    bolt::Arena arena;EXPECT_EQ(parse(arena,document("[{\"id\":1,\"name\":\"x\",\"type\":"+type+"}]")),nullptr);
    auto* good=parse(arena,document(kFields));ASSERT_NE(good,nullptr);
    bi::Schema target{};target.schema_id=777;bolt::Budget budget("nested_schema_test",1);
    EXPECT_FALSE(bi::schema_copy(&target,&good->schemas[0],&arena,&budget));EXPECT_EQ(target.schema_id,777);
    bi::SchemaField cycle{};cycle.id=1;std::strcpy(cycle.type,"list");cycle.children=&cycle;cycle.n_children=1;
    bi::Schema schema{};schema.fields=&cycle;schema.n_fields=1;int32_t max=777;
    EXPECT_FALSE(bi::schema_max_field_id(&schema,&max));EXPECT_EQ(max,777);
}
TEST(IcebergNestedSchema, AddColumnDoesNotReuseNestedIds) {
    const auto root=std::filesystem::temp_directory_path()/std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    std::filesystem::create_directories(root/"metadata");
    bolt::Arena arena;bolt::lakehouse::FilesystemObjectStore fs{};bolt::lakehouse::ObjectStore store{};
    ASSERT_TRUE(bolt::lakehouse::filesystem_object_store_init(&fs,root.string().c_str(),&store));
    const auto json=document(kFields);
    ASSERT_EQ(bolt::lakehouse::os_put(&store,"metadata/v1.metadata.json",reinterpret_cast<const uint8_t*>(json.data()),json.size()),0);
    bi::TableHandle* table=nullptr;ASSERT_TRUE(bi::table_open(&table,&arena,&store,root.string().c_str()));
    ASSERT_TRUE(bi::table_add_column(table,"added",bolt::BoltType::Int64,true));
    const auto* schema=bi::metadata_current_schema(bi::table_metadata(table));ASSERT_NE(schema,nullptr);
    ASSERT_EQ(schema->n_fields,4u);EXPECT_EQ(schema->fields[3].id,12);
    EXPECT_EQ(schema->fields[1].children[1].children[0].id,8);
    bi::table_close(table);std::filesystem::remove_all(root);
}
TEST(IcebergNestedSchema, ExternalOracle) {
    const char* input=std::getenv("BOLT_NESTED_SCHEMA_INPUT");const char* output=std::getenv("BOLT_NESTED_SCHEMA_OUTPUT");
    ASSERT_NE(input,nullptr);ASSERT_NE(output,nullptr);
    std::ifstream in(input);const std::string json{std::istreambuf_iterator<char>(in),{}};
    bolt::Arena arena;auto* metadata=parse(arena,json);ASSERT_NE(metadata,nullptr);
    const char* parquet=std::getenv("BOLT_NESTED_SCHEMA_PARQUET");ASSERT_NE(parquet,nullptr);
    std::ifstream file(parquet,std::ios::binary);const std::string bytes{std::istreambuf_iterator<char>(file),{}};
    auto* meta=arena.allocate_array<pq::PqMeta>(1);ASSERT_NE(meta,nullptr);pq::PqSchemaTree tree{};
    const auto* raw=reinterpret_cast<const uint8_t*>(bytes.data());
    ASSERT_TRUE(pq::parquet_read_meta(raw,bytes.size(),&arena,meta));
    ASSERT_TRUE(pq::parquet_read_schema_tree(raw,bytes.size(),&arena,&tree));
    bool matches=true;
    for (uint32_t i=0;i<metadata->schemas[0].n_fields;++i) {
        auto& field=metadata->schemas[0].fields[i];const auto index=pq::pq_field_find_id(&tree,-1,field.id);
        const bool matched=bi::schema_field_matches_parquet(field,meta,&tree,index);
        if (!std::getenv("BOLT_NESTED_SCHEMA_MUST_REFUSE")) EXPECT_TRUE(matched)<<field.name;
        matches=matched && matches;
        const auto id=field.id;field.id=123456;EXPECT_FALSE(bi::schema_field_matches_parquet(field,meta,&tree,index));field.id=id;
    }
    if (std::getenv("BOLT_NESTED_SCHEMA_MUST_REFUSE")) {EXPECT_FALSE(matches);return;}
    ASSERT_TRUE(matches);
    const auto encoded=emit(arena,*metadata);ASSERT_FALSE(encoded.empty());
    std::ofstream out(output,std::ios::binary);out<<encoded;ASSERT_TRUE(out.good());
}
}  // namespace
