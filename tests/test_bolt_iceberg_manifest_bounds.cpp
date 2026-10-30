#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/iceberg/json_value.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/statistics.h"
#include "bolt/lakehouse/object_store.h"
#include "bolt/parse/bolt_json_writer.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <cmath>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
namespace {
using namespace bolt;using namespace bolt::lakehouse;using namespace bolt::lakehouse::iceberg;
namespace j=bolt::parse::json;
void decode_value(Arena& arena,const SchemaField& field,const j::JsonValue* value,BoltColumn* out) {
    ASSERT_NE(value,nullptr);
    if(value->kind==j::ValueKind::Object && (!std::strncmp(field.type,"geometry",8) || !std::strncmp(field.type,"geography",9))) {
        const auto* hex=j::object_get(value,"wkb",3);ASSERT_NE(hex,nullptr);ASSERT_GE(hex->len,0);ASSERT_EQ(hex->len%2,0);
        const auto count=static_cast<uint32_t>(hex->len)/2u;auto* bytes=arena.allocate_array<uint8_t>(count+1u);
        auto* offsets=arena.allocate_array<int32_t>(2);ASSERT_NE(bytes,nullptr);ASSERT_NE(offsets,nullptr);
        for(uint32_t i=0;i<count;++i) {
            uint32_t v=0;const auto parsed=std::from_chars(hex->bytes+2u*i,hex->bytes+2u*i+2u,v,16);
            ASSERT_EQ(parsed.ec,std::errc{});ASSERT_EQ(parsed.ptr,hex->bytes+2u*i+2u);bytes[i]=static_cast<uint8_t>(v);
        }
        offsets[0]=0;offsets[1]=static_cast<int32_t>(count);*out=BoltColumn::make_var_binary(bytes,nullptr,offsets,1,BoltType::Binary,&arena);return;
    }
    const uint8_t* json=nullptr;int32_t size=0;
    ASSERT_TRUE(j::serialize(value,&arena,j::EmitOptions{},&json,&size));
    const bool floating=!std::strcmp(field.type,"float") || !std::strcmp(field.type,"double");
    const bool nan=size==5 && !std::memcmp(json,"\"NaN\"",5);
    const bool positive=size==10 && !std::memcmp(json,"\"Infinity\"",10);
    const bool negative=size==11 && !std::memcmp(json,"\"-Infinity\"",11);
    if(floating && (nan || positive || negative)) {
        auto* number=arena.allocate_array<double>(1);ASSERT_NE(number,nullptr);*number=nan?std::numeric_limits<double>::quiet_NaN():negative?-std::numeric_limits<double>::infinity():std::numeric_limits<double>::infinity();
        *out=BoltColumn::make_flat(number,nullptr,1,BoltType::Float64);return;
    }
    ASSERT_EQ(json_value_decode(field,json,static_cast<uint32_t>(size),&arena,out),JsonValueStatus::Ok)<<field.name;
}
void native_pruning(Arena& arena,const Schema& schema,const Metadata& metadata,const j::JsonValue* rows) {
    ASSERT_EQ(metadata.n_snapshots,1u);
    std::ifstream list_file(metadata.snapshots[0].manifest_list);
    std::string list((std::istreambuf_iterator<char>(list_file)),{});ManifestListEntry* manifests=nullptr;uint32_t n=0;
    ASSERT_TRUE(manifest_list_parse_avro_grow(reinterpret_cast<const uint8_t*>(list.data()),list.size(),&arena,1,&manifests,&n));ASSERT_EQ(n,1u);
    std::ifstream file(manifests[0].manifest_path);std::string bytes((std::istreambuf_iterator<char>(file)),{});
    DataFileRef* files=nullptr;ASSERT_TRUE(manifest_parse_avro_grow(reinterpret_cast<const uint8_t*>(bytes.data()),bytes.size(),&arena,0,1,&files,&n));ASSERT_EQ(n,1u);
    for(uint32_t i=0;i<schema.n_fields;++i) {
        const auto& field=schema.fields[i];const bool integer=!std::strcmp(field.type,"long") || !std::strcmp(field.type,"timestamp_ns") || !std::strcmp(field.type,"timestamptz_ns");
        const bool floating=!std::strcmp(field.type,"float") || !std::strcmp(field.type,"double");
        if(!integer && !floating)continue;const auto* row=rows->head;
        for(int32_t r=0;r<rows->count;++r) {
            char id[16];const auto length=std::snprintf(id,sizeof(id),"%d",field.id);const auto* raw=j::object_get(row->value,id,length);
            BoltColumn value{};decode_value(arena,field,raw,&value);ASSERT_FALSE(::testing::Test::HasFatalFailure());row=row->next;
            if(value.is_null(0))continue;
            Predicate predicate{};std::strcpy(predicate.column,field.name);predicate.op=PredicateOp::kEq;
            if(integer) {predicate.value.type=BoltType::Int64;std::memcpy(&predicate.value.i64,value.data,8);}
            else {predicate.value.type=BoltType::Float64;std::memcpy(&predicate.value.f64,value.data,8);if(std::isnan(predicate.value.f64))continue;}
            EXPECT_TRUE(stats_pass(&files[0],&schema,&predicate,1))<<field.name<<" row "<<r;
        }
        if(!std::strcmp(field.name,"all_null")) {
            Predicate predicate{};std::strcpy(predicate.column,field.name);predicate.op=PredicateOp::kIsNotNull;
            EXPECT_FALSE(stats_pass(&files[0],&schema,&predicate,1));
        }
    }
}
TEST(IcebergManifestBounds, ExternalRealWriterMatrix) {
    const char* input=std::getenv("BOLT_BOUND_INPUT"),*directory=std::getenv("BOLT_BOUND_DIR");
    if(!input || !directory)GTEST_SKIP()<<"independent manifest fixture";
    std::ifstream file(input);std::string source((std::istreambuf_iterator<char>(file)),{});ASSERT_LE(source.size(),1u<<20);
    Arena arena;j::StructuralIndex index{};j::JsonValue* root=nullptr;
    ASSERT_TRUE(j::build_index(reinterpret_cast<const uint8_t*>(source.data()),static_cast<int32_t>(source.size()),&arena,&index));
    ASSERT_TRUE(j::value_from_index(&index,&arena,&root));ASSERT_EQ(root->kind,j::ValueKind::Object);
    const auto* declared=j::object_get(root,"schema",6);const auto* rows=j::object_get(root,"rows",4);ASSERT_NE(declared,nullptr);ASSERT_NE(rows,nullptr);
    ASSERT_EQ(rows->kind,j::ValueKind::Array);ASSERT_GT(rows->count,0);ASSERT_LE(rows->count,32);
    const uint8_t* json=nullptr;int32_t size=0;ASSERT_TRUE(j::serialize(declared,&arena,j::EmitOptions{},&json,&size));
    Schema schema{};ASSERT_TRUE(schema_json_parse(json,static_cast<uint32_t>(size),&arena,&schema));ASSERT_LE(schema.n_fields,128u);
    const auto* version=j::object_get(root,"version",7);ASSERT_NE(version,nullptr);ASSERT_EQ(version->len,1);
    WriteOptions options{};write_options_init(&options);options.format_version=static_cast<uint8_t>(version->bytes[0]-'0');
    FilesystemObjectStore fs{};ObjectStore store{};ASSERT_TRUE(filesystem_object_store_init(&fs,directory,&store));
    TableHandle* table=nullptr;ASSERT_TRUE(table_create(&table,&arena,&store,directory,&schema,nullptr,nullptr,&options));
    AppendHandle* append=nullptr;ASSERT_TRUE(append_open(&append,table));const auto* row=rows->head;
    for(int32_t i=0;i<rows->count;++i) {
        ASSERT_NE(row,nullptr);auto* batch=arena.allocate_array<BoltBatch>(1);ASSERT_NE(batch,nullptr);BoltBatch::init_empty(batch);
        ASSERT_TRUE(BoltBatch::alloc_columns(batch,&arena,schema.n_fields));batch->num_rows=1;
        for(uint32_t c=0;c<schema.n_fields;++c) {
            const auto& field=schema.fields[c];char id[16];const auto n=std::snprintf(id,sizeof(id),"%d",field.id);ASSERT_GT(n,0);
            decode_value(arena,field,j::object_get(row->value,id,n),&batch->columns[0][c]);ASSERT_FALSE(::testing::Test::HasFatalFailure());
            ASSERT_TRUE(batch->schema.add_field(field.name,batch->col(c).type,!field.required));
        }
        ASSERT_TRUE(append_write(append,batch));row=row->next;
    }
    ASSERT_TRUE(append_commit(append));append_close(append);native_pruning(arena,schema,*table_metadata(table),rows);table_close(table);
}
} // namespace
