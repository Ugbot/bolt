#include "bolt/lakehouse/iceberg/write_batch.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

namespace {
using namespace bolt;
using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;
using S=JsonValueStatus;
SchemaField field(const char* name,const char* type,int32_t id,bool required=false) {
    SchemaField f{};f.id=id;f.required=required;std::strcpy(f.name,name);std::strcpy(f.type,type);return f;
}
void defaults(Arena& a,SchemaField& f,const char* initial,const char* write) {
    ASSERT_TRUE(schema_default_assign(&f.initial_default_value,reinterpret_cast<const uint8_t*>(initial),static_cast<uint32_t>(std::strlen(initial)),&a));
    ASSERT_TRUE(schema_default_assign(&f.write_default_value,reinterpret_cast<const uint8_t*>(write),static_cast<uint32_t>(std::strlen(write)),&a));
}
void input(Arena& a,BoltBatch& b,int64_t* ids,int64_t n) {
    BoltBatch::init_empty(&b);b.arena=&a;ASSERT_TRUE(BoltBatch::alloc_columns(&b,&a,1));b.num_cols=1;b.num_rows=n;
    ASSERT_TRUE(b.schema.add_field("id",BoltType::Int64,false));b.columns[0][0]=BoltColumn::make_flat(ids,nullptr,n,BoltType::Int64);
}
TEST(IcebergWriteDefaults, OmissionUsesWriteValueAndExplicitNullWins) {
    Arena a;SchemaField fields[]{field("id","long",1,true),field("n","long",2)};
    defaults(a,fields[1],"7","9");Schema schema{};schema.fields=fields;schema.n_fields=2;
    int64_t ids[]{1,2};BoltBatch b{};input(a,b,ids,2);BoltBatch* out=nullptr;
    ASSERT_EQ(write_batch_conform(schema,b,&a,&out),S::Ok);EXPECT_EQ(out->num_cols,2u);
    EXPECT_EQ(static_cast<const int64_t*>(out->col(1).data)[0],9);EXPECT_EQ(static_cast<const int64_t*>(out->col(1).data)[1],9);
    BoltBatch supplied{};BoltBatch::init_empty(&supplied);ASSERT_TRUE(BoltBatch::alloc_columns(&supplied,&a,2));supplied.num_cols=2;supplied.num_rows=2;
    ASSERT_TRUE(supplied.schema.add_field("n",BoltType::Int64,true));ASSERT_TRUE(supplied.schema.add_field("id",BoltType::Int64,false));
    uint8_t valid[]{2};int64_t values[]{99,11};supplied.columns[0][0]=BoltColumn::make_flat(values,valid,2,BoltType::Int64);supplied.columns[0][1]=b.col(0);
    ASSERT_EQ(write_batch_conform(schema,supplied,&a,&out),S::Ok);EXPECT_TRUE(out->col(1).is_null(0));EXPECT_EQ(static_cast<const int64_t*>(out->col(1).data)[1],11);
    fields[1].required=true;auto* before=out;EXPECT_EQ(write_batch_conform(schema,supplied,&a,&out),S::Invalid);EXPECT_EQ(out,before);
    fields[1].write_default_value={};EXPECT_EQ(write_batch_conform(schema,b,&a,&out),S::Invalid);EXPECT_EQ(out,before);
    defaults(a,fields[1],"7","\"bad\"");EXPECT_EQ(write_batch_conform(schema,b,&a,&out),S::Invalid);EXPECT_EQ(out,before);
    fields[1].required=false;defaults(a,fields[1],"7","null");ASSERT_EQ(write_batch_conform(schema,b,&a,&out),S::Ok);EXPECT_TRUE(out->col(1).is_null(0));
}
TEST(IcebergWriteDefaults, UnknownAndDuplicateNamesRefuseBeforePublication) {
    Arena a;SchemaField f=field("id","long",1,true);Schema schema{};schema.fields=&f;schema.n_fields=1;
    int64_t ids[]{1};BoltBatch b{};input(a,b,ids,1);std::strcpy(b.schema.fields[0].name,"other");
    BoltBatch* output=reinterpret_cast<BoltBatch*>(uintptr_t{1});
    EXPECT_EQ(write_batch_conform(schema,b,&a,&output),S::Invalid);EXPECT_EQ(output,reinterpret_cast<BoltBatch*>(uintptr_t{1}));
    SchemaField fields[]{field("id","long",1,true),field("extra","long",2)};schema.fields=fields;schema.n_fields=2;
    BoltBatch duplicate{};BoltBatch::init_empty(&duplicate);ASSERT_TRUE(BoltBatch::alloc_columns(&duplicate,&a,2));
    duplicate.num_cols=2;duplicate.num_rows=1;
    for(uint32_t i=0;i<2;++i) {ASSERT_TRUE(duplicate.schema.add_field("id",BoltType::Int64,false));duplicate.columns[0][i]=b.col(0);}
    EXPECT_EQ(write_batch_conform(schema,duplicate,&a,&output),S::Invalid);EXPECT_EQ(output,reinterpret_cast<BoltBatch*>(uintptr_t{1}));
}
TEST(IcebergWriteDefaults, PublicAppendMaterializesRecursiveDefaults) {
    const char* external=std::getenv("BOLT_WRITE_DEFAULT_DIR");
    const auto root=external?std::filesystem::path(external):std::filesystem::temp_directory_path()/"bolt_g2ice314_write_defaults";
    std::error_code ec;if(!external)std::filesystem::remove_all(root,ec);
    std::filesystem::create_directories(root,ec);ASSERT_FALSE(ec);
    Arena a;SchemaField fields[]{field("id","long",1,true),field("ns","timestamp_ns",2),field("st","struct",3),
        field("mp","map",7),field("dec","decimal(30,2)",10),field("text","string",11),
        field("f","float",12),field("small_dec","decimal(9,2)",13),field("fixed","fixed[3]",14),
        field("bool","boolean",15),field("time","time",16),field("uuid","uuid",17)};
    SchemaField children[]{field("n","long",4,true),field("xs","list",5)};auto element=field("element","string",6);
    children[1].children=&element;children[1].n_children=1;fields[2].children=children;fields[2].n_children=2;
    SchemaField kv[]{field("key","string",8,true),field("value","int",9)};fields[3].children=kv;fields[3].n_children=2;
    defaults(a,fields[1],"\"1969-12-31T23:59:59.999999999\"","\"2262-04-11T23:47:16.854775807\"");
    defaults(a,fields[2],R"({"4":1,"5":[]})",R"({"4":9223372036854775807,"5":[null,"","owned"]})");
    defaults(a,fields[3],R"({"keys":[],"values":[]})",R"({"keys":["a","b"],"values":[1,null]})");
    defaults(a,fields[4],"\"1.00\"","\"-12345678901234567890.12\"");defaults(a,fields[5],"\"initial\"",R"("write\"雪\n")");
    defaults(a,fields[6],"1.5","0.1");defaults(a,fields[7],"\"1.00\"","\"-12.34\"");
    defaults(a,fields[8],"\"000000\"","\"0a00ff\"");defaults(a,fields[9],"false","true");
    defaults(a,fields[10],"\"00:00:00\"","\"23:59:59.999999\"");
    defaults(a,fields[11],"\"00000000-0000-0000-0000-000000000000\"","\"f79c3e09-677c-4bbd-a479-3f349cb785e7\"");
    Schema schema{};schema.fields=fields;schema.n_fields=12;schema.cap_fields=12;
    FilesystemObjectStore fs{};ObjectStore store{};ASSERT_TRUE(filesystem_object_store_init(&fs,root.generic_string().c_str(),&store));
    WriteOptions opts{};write_options_init(&opts);opts.format_version=3;PartitionSpec spec{};bolt::lakehouse::iceberg::SortOrder sort{};TableHandle* table=nullptr;
    ASSERT_TRUE(table_create(&table,&a,&store,root.generic_string().c_str(),&schema,&spec,&sort,&opts));
    AppendHandle* append=nullptr;ASSERT_TRUE(append_open(&append,table));int64_t ids[]{1,2};BoltBatch batch{};input(a,batch,ids,2);
    ASSERT_TRUE(append_write(append,&batch));char path[kIcebergMaxPath]{};int64_t rows=0;
    ASSERT_TRUE(append_commit_ex(append,path,sizeof(path),&rows));EXPECT_EQ(rows,2);append_close(append);
    ASSERT_TRUE(append_open(&append,table));BoltBatch explicit_null{};BoltBatch::init_empty(&explicit_null);
    ASSERT_TRUE(BoltBatch::alloc_columns(&explicit_null,&a,2));explicit_null.num_cols=2;explicit_null.num_rows=1;
    int64_t third=3;explicit_null.columns[0][0]=BoltColumn::make_flat(&third,nullptr,1,BoltType::Int64);
    ASSERT_TRUE(explicit_null.schema.add_field("id",BoltType::Int64,false));ASSERT_TRUE(explicit_null.schema.add_field("st",BoltType::Struct,true));
    ASSERT_EQ(json_value_decode(fields[2],reinterpret_cast<const uint8_t*>("null"),4,&a,&explicit_null.columns[0][1]),S::Ok);
    ASSERT_TRUE(append_write(append,&explicit_null));ASSERT_TRUE(append_commit(append));append_close(append);table_close(table);
    ASSERT_TRUE(std::filesystem::is_regular_file(path));
    if(!external)std::filesystem::remove_all(root,ec);
}
} // namespace
