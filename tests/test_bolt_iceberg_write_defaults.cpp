#include "bolt/lakehouse/iceberg/write_batch.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/kernels/bolt_nested_gather.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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
    defaults(a,fields[2],"{}","{}");
    defaults(a,children[0],"1","9223372036854775807");defaults(a,children[1],"[]",R"([null,"","owned"])");
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
    ASSERT_TRUE(append_write(append,&explicit_null));ASSERT_TRUE(append_commit(append));append_close(append);
    Predicate pred{};std::strcpy(pred.column,"id");pred.op=PredicateOp::kEq;pred.value.type=BoltType::Int64;pred.value.i64=2;
    int64_t fourth=4;BoltBatch replacement{};input(a,replacement,&fourth,1);
    ASSERT_TRUE(table_overwrite(table,&pred,&replacement));const auto snapshots=table_metadata(table)->n_snapshots;
    ASSERT_TRUE(table_compact(table,64ll<<20));EXPECT_GT(table_metadata(table)->n_snapshots,snapshots);
    table_close(table);ASSERT_TRUE(table_open(&table,&a,&store,root.generic_string().c_str()));table_close(table);
    ASSERT_TRUE(std::filesystem::is_regular_file(path));
    if(!external)std::filesystem::remove_all(root,ec);
}
void check_unknown_rows(const std::filesystem::path& root,const ObjectStore& store,uint32_t expected,int64_t forbidden) {
    std::ifstream hint(root/"metadata/version-hint.text");uint64_t version=0;ASSERT_TRUE(hint>>version);
    std::ifstream file(root/"metadata"/("v"+std::to_string(version)+".metadata.json"));
    const std::string json((std::istreambuf_iterator<char>(file)),{});ASSERT_FALSE(json.empty());
    Arena a;TableHandle* table=nullptr;ASSERT_TRUE(iceberg_table_open_on_store(&table,&a,&store,"",
        reinterpret_cast<const uint8_t*>(json.data()),static_cast<uint32_t>(json.size())));
    ScanHandle* scan=nullptr;ReadOptions options{};read_options_init(&options);ASSERT_TRUE(iceberg_scan_open(&scan,table,&options));
    uint32_t seen=0;
    for(uint32_t step=0;step<32;++step) {
        BoltBatch batch{};bool eof=false;ASSERT_TRUE(iceberg_scan_next_batch(scan,&batch,&eof));if(eof)break;
        ASSERT_EQ(batch.num_cols,5u);
        for(int64_t row=0;row<batch.num_rows;++row) {
            EXPECT_NE(static_cast<const int64_t*>(batch.col(0).data)[row],forbidden);EXPECT_TRUE(batch.col(1).is_null(row));
            const auto& point=batch.col(2);ASSERT_EQ(point.child_count(),2);EXPECT_TRUE(point.child_at(1)->is_null(row));
            if(!point.is_null(row))EXPECT_EQ(static_cast<const int64_t*>(point.child_at(0)->data)[row],7);
            EXPECT_FALSE(batch.col(3).is_null(row));EXPECT_FALSE(batch.col(4).is_null(row));++seen;
        }
    }
    EXPECT_EQ(seen,expected)<<root.string();iceberg_scan_close(scan);iceberg_table_close(table);
}
TEST(IcebergWriteDefaults, UnknownOmissionOverwriteAndCompactionRetainLogicalNulls) {
    const char* external=std::getenv("BOLT_UNKNOWN_WRITE_DIR");
    const auto root=external?std::filesystem::path(external):std::filesystem::temp_directory_path()/"bolt_g2ice314_unknown_write";
    std::error_code ec;if(!external)std::filesystem::remove_all(root,ec);std::filesystem::create_directories(root,ec);ASSERT_FALSE(ec);
    Arena a;SchemaField fields[]{field("id","long",1,true),field("u","unknown",2),field("point","struct",3),
        field("f","float",6),field("d","decimal(9,2)",7)};
    SchemaField children[]{field("n","long",4),field("u","unknown",5)};fields[2].children=children;fields[2].n_children=2;
    defaults(a,fields[3],"1.5","0.1");defaults(a,fields[4],"\"1.00\"","\"-12.34\"");
    Schema schema{};schema.fields=fields;schema.n_fields=5;schema.cap_fields=5;
    FilesystemObjectStore fs{};ObjectStore store{};ASSERT_TRUE(filesystem_object_store_init(&fs,root.generic_string().c_str(),&store));
    WriteOptions opts{};write_options_init(&opts);opts.format_version=3;TableHandle* table=nullptr;
    ASSERT_TRUE(table_create(&table,&a,&store,root.generic_string().c_str(),&schema,nullptr,nullptr,&opts));
    BoltBatch batch{};BoltBatch::init_empty(&batch);ASSERT_TRUE(BoltBatch::alloc_columns(&batch,&a,2));batch.num_cols=2;batch.num_rows=2;
    int64_t ids[]{1,2};batch.columns[0][0]=BoltColumn::make_flat(ids,nullptr,2,BoltType::Int64);
    ASSERT_TRUE(batch.schema.add_field("id",BoltType::Int64,false));ASSERT_TRUE(batch.schema.add_field("point",BoltType::Struct,true));
    BoltColumn one{};ASSERT_EQ(json_value_decode(fields[2],reinterpret_cast<const uint8_t*>(R"({"4":7})"),7,&a,&one),S::Ok);
    const uint32_t selected[]{0,0};ASSERT_TRUE(kernels::nested_gather(one,selected,2,&a,&batch.columns[0][1]));
    AppendHandle* append=nullptr;ASSERT_TRUE(append_open(&append,table));ASSERT_TRUE(append_write(append,&batch));ASSERT_TRUE(append_commit(append));append_close(append);
    int64_t third=3;BoltBatch omitted{};input(a,omitted,&third,1);ASSERT_TRUE(append_open(&append,table));
    ASSERT_TRUE(append_write(append,&omitted));ASSERT_TRUE(append_commit(append));append_close(append);check_unknown_rows(root,store,3,-1);
    Predicate pred{};std::strcpy(pred.column,"id");pred.op=PredicateOp::kEq;pred.value.type=BoltType::Int64;pred.value.i64=2;
    int64_t fourth=4;BoltBatch replacement{};input(a,replacement,&fourth,1);
    BoltBatch invalid{};BoltBatch::init_empty(&invalid);ASSERT_TRUE(BoltBatch::alloc_columns(&invalid,&a,2));invalid.num_rows=1;
    ASSERT_TRUE(invalid.schema.add_field("id",BoltType::Int64,false));ASSERT_TRUE(invalid.schema.add_field("u",BoltType::Int64,true));
    invalid.columns[0][0]=replacement.col(0);invalid.columns[0][1]=replacement.col(0);
    const auto before=table_metadata(table)->current_snapshot_id;EXPECT_FALSE(table_overwrite(table,&pred,&invalid));
    EXPECT_EQ(table_metadata(table)->current_snapshot_id,before);check_unknown_rows(root,store,3,-1);
    ASSERT_TRUE(table_overwrite(table,&pred,&replacement));check_unknown_rows(root,store,3,2);
    const auto snapshots=table_metadata(table)->n_snapshots;ASSERT_TRUE(table_compact(table,64ll<<20));
    EXPECT_GT(table_metadata(table)->n_snapshots,snapshots);check_unknown_rows(root,store,3,2);
    table_close(table);ASSERT_TRUE(table_open(&table,&a,&store,root.generic_string().c_str()));table_close(table);
    check_unknown_rows(root,store,3,2);if(!external)std::filesystem::remove_all(root,ec);
}
TEST(IcebergWriteDefaults, NonNullUnknownAndEmptyStructRefuseBeforePublication) {
    Arena a;SchemaField fields[]{field("id","long",1,true),field("u","unknown",2)};Schema schema{};schema.fields=fields;schema.n_fields=2;
    int64_t id=1;BoltBatch supplied{};BoltBatch::init_empty(&supplied);ASSERT_TRUE(BoltBatch::alloc_columns(&supplied,&a,2));supplied.num_cols=2;supplied.num_rows=1;
    ASSERT_TRUE(supplied.schema.add_field("id",BoltType::Int64,false));ASSERT_TRUE(supplied.schema.add_field("u",BoltType::Int64,true));
    supplied.columns[0][0]=BoltColumn::make_flat(&id,nullptr,1,BoltType::Int64);supplied.columns[0][1]=supplied.col(0);
    BoltBatch* output=reinterpret_cast<BoltBatch*>(uintptr_t{1});EXPECT_EQ(write_batch_conform(schema,supplied,&a,&output),S::Invalid);
    EXPECT_EQ(output,reinterpret_cast<BoltBatch*>(uintptr_t{1}));
    fields[1]=field("u","struct",2);ASSERT_EQ(json_value_decode(fields[1],reinterpret_cast<const uint8_t*>("{}"),2,&a,&supplied.columns[0][1]),S::Ok);
    EXPECT_EQ(write_batch_conform(schema,supplied,&a,&output),S::Unsupported);EXPECT_EQ(output,reinterpret_cast<BoltBatch*>(uintptr_t{1}));
    fields[1]=field("u","struct",2);auto unknown=field("u","unknown",3);fields[1].children=&unknown;fields[1].n_children=1;
    ASSERT_EQ(json_value_decode(fields[1],reinterpret_cast<const uint8_t*>("{}"),2,&a,&supplied.columns[0][1]),S::Ok);
    EXPECT_EQ(write_batch_conform(schema,supplied,&a,&output),S::Unsupported);
    schema.fields=&unknown;schema.n_fields=1;unknown.name[0]='u';unknown.name[1]=0;
    BoltBatch only{};BoltBatch::init_empty(&only);ASSERT_TRUE(BoltBatch::alloc_columns(&only,&a,1));only.num_rows=1;
    ASSERT_TRUE(only.schema.add_field("u",BoltType::Binary,true));
    ASSERT_EQ(json_value_decode(unknown,reinterpret_cast<const uint8_t*>("null"),4,&a,&only.columns[0][0]),S::Ok);
    EXPECT_EQ(write_batch_conform(schema,only,&a,&output),S::Unsupported);EXPECT_EQ(output,reinterpret_cast<BoltBatch*>(uintptr_t{1}));
}
TEST(IcebergWriteDefaults, BooleanReadLanesNormalizeAndInvalidValuesKeepOutput) {
    Arena a;auto flag=field("flag","boolean",1);Schema schema{};schema.fields=&flag;schema.n_fields=1;
    BoltBatch source{};BoltBatch::init_empty(&source);ASSERT_TRUE(BoltBatch::alloc_columns(&source,&a,1));source.num_rows=2;
    ASSERT_TRUE(source.schema.add_field("flag",BoltType::Int64,true));int64_t values[]{0,1};
    source.columns[0][0]=BoltColumn::make_flat(values,nullptr,2,BoltType::Int64);BoltBatch* out=nullptr;
    ASSERT_EQ(write_batch_conform(schema,source,&a,&out),S::Ok);EXPECT_EQ(out->col(0).type,BoltType::Bool);EXPECT_EQ(out->col(0).type_size_bytes,1);
    EXPECT_EQ(static_cast<const uint8_t*>(out->col(0).data)[0],0);EXPECT_EQ(static_cast<const uint8_t*>(out->col(0).data)[1],1);
    auto* before=out;values[0]=2;EXPECT_EQ(write_batch_conform(schema,source,&a,&out),S::Invalid);EXPECT_EQ(out,before);
    uint8_t present[]{2};source.columns[0][0].validity=present;
    ASSERT_EQ(write_batch_conform(schema,source,&a,&out),S::Ok);EXPECT_TRUE(out->col(0).is_null(0));
    uint8_t bad[]{2,1};source.columns[0][0]=BoltColumn::make_flat(bad,nullptr,2,BoltType::Bool);
    before=out;EXPECT_EQ(write_batch_conform(schema,source,&a,&out),S::Invalid);EXPECT_EQ(out,before);
}
} // namespace
