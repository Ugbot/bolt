#include "bolt/ingest/bolt_arrow_ipc_reader.h"
#include "bolt/bolt_column.h"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>
namespace { std::atomic<std::uint64_t> allocations{0}; }
void* operator new(std::size_t n) {
    allocations.fetch_add(1,std::memory_order_relaxed);
    auto* p=std::malloc(n?n:1);if(!p)std::abort();return p;
}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}
namespace {
using namespace bolt;using namespace bolt::ingest;using Status=IpcReadStatus;
std::string path(const char* name){return std::string(BOLT_TEST_DATA_DIR)+"/arrow_ipc/"+name;}
std::vector<std::uint8_t> load(const char* name){
    auto* f=std::fopen(path(name).c_str(),"rb");if(!f)return {};
    std::fseek(f,0,SEEK_END);const auto n=std::ftell(f);std::rewind(f);
    std::vector<std::uint8_t> v(n>0?static_cast<std::size_t>(n):0);
    const auto read=std::fread(v.data(),1,v.size(),f);std::fclose(f);if(read!=v.size())v.clear();return v;
}
struct Work {
    ArrowIpcReader reader{};IpcReadLimits limits{};void* p=nullptr;std::uint64_t size=0;
    explicit Work(std::uint64_t output=65536){limits.max_fields=32;limits.max_key_values=64;
        limits.output_bytes=output;limits.max_batch_rows=16;
        if(arrow_ipc_reader_workspace_size(limits,size)==Status::Ok)p=bolt_aligned_alloc(64,static_cast<std::size_t>(size));}
    ~Work(){arrow_ipc_reader_close(reader);bolt_aligned_free(p);}
    Status open(const std::vector<std::uint8_t>& bytes){return arrow_ipc_reader_open(bytes.data(),bytes.size(),limits,p,size,reader);}
};
std::uint16_t u16(const std::vector<std::uint8_t>& v,std::size_t p){return static_cast<std::uint16_t>(v[p]|(std::uint16_t(v[p+1])<<8));}
std::uint32_t u32(const std::vector<std::uint8_t>& v,std::size_t p){std::uint32_t n=0;for(unsigned i=0;i<4;++i)n|=std::uint32_t(v[p+i])<<(8*i);return n;}
std::uint64_t u64(const std::vector<std::uint8_t>& v,std::size_t p){std::uint64_t n=0;for(unsigned i=0;i<8;++i)n|=std::uint64_t(v[p+i])<<(8*i);return n;}
void put32(std::vector<std::uint8_t>& v,std::size_t p,std::uint32_t n){for(unsigned i=0;i<4;++i)v[p+i]=static_cast<std::uint8_t>(n>>(8*i));}
void put64(std::vector<std::uint8_t>& v,std::size_t p,std::uint64_t n){for(unsigned i=0;i<8;++i)v[p+i]=static_cast<std::uint8_t>(n>>(8*i));}
std::size_t field(const std::vector<std::uint8_t>& v,std::size_t t,unsigned slot){
    const auto delta=static_cast<std::int32_t>(u32(v,t));const auto vt=static_cast<std::size_t>(static_cast<std::int64_t>(t)-delta);
    return 4+slot*2<u16(v,vt)?t+u16(v,vt+4+slot*2):0;
}
std::size_t ref(const std::vector<std::uint8_t>& v,std::size_t t,unsigned slot){const auto p=field(v,t,slot);return p?p+u32(v,p):0;}
std::size_t next_message(const std::vector<std::uint8_t>& v,std::size_t start){
    const auto prefix=u32(v,start)==UINT32_MAX?8u:4u;const auto n=u32(v,start+prefix-4);
    const auto meta=start+prefix,t=meta+u32(v,meta),body=field(v,t,3);
    return meta+n+(body?u64(v,body):0);
}
struct Positions {std::size_t start,meta,table,header,buffers,nodes,body;};
Positions positions(const std::vector<std::uint8_t>& v,std::size_t start){
    const auto prefix=u32(v,start)==UINT32_MAX?8u:4u;const auto meta=start+prefix;
    const auto table=meta+u32(v,meta),header=ref(v,table,2);
    return{start,meta,table,header,ref(v,header,2)+4,ref(v,header,1)+4,meta+u32(v,start+prefix-4)};
}
template<class T>T lane(const BoltColumn& c,unsigned row){T v{};std::memcpy(&v,static_cast<const std::uint8_t*>(c.data)+row*sizeof(T),sizeof(T));return v;}
void values(const BoltBatch& b){
    ASSERT_EQ(b.num_rows,3);ASSERT_EQ(b.num_cols,23u);
    for(unsigned i=0;i<8;i+=2){const auto& sign=b.col(i);const auto& unsign=b.col(i+1);
        EXPECT_EQ(sign.stats.null_count,1u);EXPECT_EQ(unsign.stats.null_count,1u);EXPECT_EQ(sign.validity[0]&7,5);
        const unsigned width=1u<<(i/2);std::uint64_t low=0,high=0,max=0;
        std::memcpy(&low,sign.data,width);std::memcpy(&high,static_cast<const std::uint8_t*>(sign.data)+2*width,width);
        std::memcpy(&max,static_cast<const std::uint8_t*>(unsign.data)+2*width,width);
        EXPECT_EQ(low,std::uint64_t{1}<<(width*8-1));EXPECT_EQ(high,(std::uint64_t{1}<<(width*8-1))-1);
        EXPECT_EQ(max,width==8?UINT64_MAX:(std::uint64_t{1}<<(width*8))-1);
    }
    EXPECT_EQ(b.col(7).type,BoltType::UInt64);EXPECT_EQ(lane<std::uint64_t>(b.col(7),2),UINT64_MAX);
    EXPECT_EQ(lane<std::uint16_t>(b.col(8),0),0x8000);EXPECT_EQ(lane<std::uint16_t>(b.col(8),1),0x7e55);
    EXPECT_EQ(lane<std::uint32_t>(b.col(9),0),0x80000000u);EXPECT_EQ(lane<std::uint64_t>(b.col(10),0),0x8000000000000000ull);
    EXPECT_EQ(lane<std::uint8_t>(b.col(11),0),1);EXPECT_EQ(lane<std::uint8_t>(b.col(11),2),0);
    const auto* text=static_cast<const StringView*>(b.col(12).data);EXPECT_EQ(text[0].length,5u);EXPECT_EQ(std::memcmp(text[0].prefix,"alpha",5),0);
    EXPECT_GT(text[2].length,12u);const auto* base=static_cast<const char*>(b.col(12).str_overflow_base);
    EXPECT_EQ(std::string(base+text[2].ref.offset,text[2].length),"a long string with snowman \xe2\x98\x83");
    const auto& binary=b.col(13);EXPECT_EQ(binary.format,ColumnFormat::VarBinary);
    const auto* offsets=static_cast<const std::int32_t*>(binary.dict_child->data);EXPECT_EQ(offsets[1]-offsets[0],2);EXPECT_EQ(offsets[3]-offsets[2],0);
    const auto* bytes=static_cast<const std::uint8_t*>(binary.data);EXPECT_EQ(bytes[0],0);EXPECT_EQ(bytes[1],255);
    constexpr std::uint8_t decimal0[]{0xeb,0xa0,0x27,0x3b,0x6e,0x85,0x86,0x92,0x14,0xca,0x02,0xfc,0xff,0xff,0xff,0xff};
    constexpr std::uint8_t decimal2[]{0xff,0xff,0xff,0xff,0x3f,0x22,0x8a,0x09,0x7a,0xc4,0x86,0x5a,0xa8,0x4c,0x3b,0x4b};
    EXPECT_EQ(b.col(14).type,BoltType::Decimal128);EXPECT_EQ(b.col(14).decimal_scale,9);EXPECT_EQ(std::memcmp(b.col(14).data,decimal0,16),0);
    EXPECT_EQ(std::memcmp(static_cast<const std::uint8_t*>(b.col(14).data)+32,decimal2,16),0);
    EXPECT_EQ(lane<std::int32_t>(b.col(15),0),-1);EXPECT_EQ(lane<std::int64_t>(b.col(16),0),-86400000);
    EXPECT_EQ(lane<std::int32_t>(b.col(17),2),86399);EXPECT_EQ(lane<std::int64_t>(b.col(18),2),86399999999999ll);
    EXPECT_EQ(b.col(19).type,BoltType::Int64);EXPECT_EQ(lane<std::int64_t>(b.col(19),0),-1);
    EXPECT_EQ(lane<std::int64_t>(b.col(20),2),123456789);EXPECT_EQ(lane<std::int64_t>(b.col(21),2),INT64_MAX);
    EXPECT_EQ(b.col(22).fixed_width,3);EXPECT_EQ(b.col(22).type_size_bytes,16);EXPECT_EQ(std::memcmp(b.col(22).data,"abc",3),0);
    const auto* fixed=static_cast<const std::uint8_t*>(b.col(22).data);for(unsigned i=3;i<16;++i)EXPECT_EQ(fixed[i],0);
}
TEST(ArrowIpcReader, IndependentReferenceStreamAndFileExactFlatValues){
    for(const auto* name:{"flat.arrows","flat.arrow","legacy.arrows","v4.arrows"}){SCOPED_TRACE(name);const auto bytes=load(name);ASSERT_FALSE(bytes.empty());Work w;ASSERT_NE(w.p,nullptr);
        ASSERT_EQ(w.open(bytes),Status::Ok);IpcReadSchema schema{};ASSERT_EQ(arrow_ipc_reader_schema(w.reader,schema),Status::Ok);
        ASSERT_EQ(schema.field_count,23u);EXPECT_EQ(schema.fields[14].precision,38);EXPECT_EQ(schema.fields[14].scale,9);
        EXPECT_EQ(schema.fields[19].unit,0);EXPECT_EQ(std::string(schema.fields[19].timezone.data,schema.fields[19].timezone.size),"America/New_York");
        EXPECT_EQ(schema.fields[21].unit,3);ASSERT_EQ(schema.metadata_count,1u);EXPECT_EQ(schema.metadata[0].value.size,14u);
        if(std::strcmp(name,"flat.arrow")==0){ASSERT_EQ(schema.file_metadata_count,1u);EXPECT_EQ(std::string(schema.file_metadata[0].value.data,schema.file_metadata[0].value.size),"owned");}
        const BoltBatch* batch=nullptr;ASSERT_EQ(arrow_ipc_reader_next(w.reader,batch),Status::Ok);values(*batch);
        ASSERT_EQ(arrow_ipc_reader_next(w.reader,batch),Status::Ok);EXPECT_EQ(batch->num_rows,0);
        ASSERT_EQ(arrow_ipc_reader_next(w.reader,batch),Status::Ok);values(*batch);
        const auto* saved=batch;EXPECT_EQ(arrow_ipc_reader_next(w.reader,batch),Status::End);EXPECT_EQ(batch,saved);values(*batch);
    }
}
TEST(ArrowIpcReader, OrdinaryOwnedMappedFileOpenerAndClose){
    Work w;ASSERT_EQ(arrow_ipc_reader_open_file(path("flat.arrow").c_str(),w.limits,w.p,w.size,w.reader),Status::Ok);
    EXPECT_TRUE(w.reader.owns_mapping);const BoltBatch* batch=nullptr;ASSERT_EQ(arrow_ipc_reader_next(w.reader,batch),Status::Ok);values(*batch);
    arrow_ipc_reader_close(w.reader);EXPECT_EQ(arrow_ipc_reader_next(w.reader,batch),Status::Closed);arrow_ipc_reader_close(w.reader);
    EXPECT_EQ(arrow_ipc_reader_open_file("missing-owned-arrow-reader-fixture",w.limits,w.p,w.size,w.reader),Status::IoError);
}
TEST(ArrowIpcReader, ExplicitUnsupportedSchemaAndCompressedBatch){
    for(const auto* name:{"nested.arrows","dictionary.arrows","negative_scale.arrows"}){SCOPED_TRACE(name);Work w;EXPECT_EQ(w.open(load(name)),Status::Unsupported);EXPECT_EQ(w.reader.state,nullptr);}
    auto bytes=load("compressed.arrows");Work w;ASSERT_EQ(w.open(bytes),Status::Ok);
    const BoltBatch* out=reinterpret_cast<const BoltBatch*>(std::uintptr_t{64});EXPECT_EQ(arrow_ipc_reader_next(w.reader,out),Status::Unsupported);EXPECT_EQ(reinterpret_cast<std::uintptr_t>(out),64u);
}
TEST(ArrowIpcReader, EmptySchemaAndEmptyBatchFile){
    auto bytes=load("empty.arrow");Work w;w.limits.max_fields=0;w.limits.max_key_values=0;w.limits.output_bytes=0;ASSERT_EQ(w.open(bytes),Status::Ok);const BoltBatch* out=nullptr;
    ASSERT_EQ(arrow_ipc_reader_next(w.reader,out),Status::Ok);EXPECT_EQ(out->num_rows,0);EXPECT_EQ(out->num_cols,0u);EXPECT_EQ(arrow_ipc_reader_next(w.reader,out),Status::End);
}
TEST(ArrowIpcReader, CleanBoundaryEofOptionalEosAndTrailingBytes){
    auto bytes=load("flat.arrows");ASSERT_GE(bytes.size(),8u);bytes.resize(bytes.size()-8);Work w;ASSERT_EQ(w.open(bytes),Status::Ok);
    const BoltBatch* out=nullptr;for(unsigned i=0;i<3;++i)ASSERT_EQ(arrow_ipc_reader_next(w.reader,out),Status::Ok);
    EXPECT_EQ(arrow_ipc_reader_next(w.reader,out),Status::End);
    bytes=load("flat.arrows");bytes.push_back(0);Work extra;ASSERT_EQ(extra.open(bytes),Status::Ok);
    for(unsigned i=0;i<3;++i)ASSERT_EQ(arrow_ipc_reader_next(extra.reader,out),Status::Ok);
    EXPECT_EQ(arrow_ipc_reader_next(extra.reader,out),Status::BadInput);
}
TEST(ArrowIpcReader, TruncationEveryByteNeverPublishesPartialBatch){
    const auto bytes=load("flat.arrows");const auto schema_end=next_message(bytes,0);const auto first_end=next_message(bytes,schema_end);
    for(std::size_t n=0;n<first_end;++n){Work w;auto status=arrow_ipc_reader_open(bytes.data(),n,w.limits,w.p,w.size,w.reader);
        const BoltBatch* out=nullptr;if(status==Status::Ok)status=arrow_ipc_reader_next(w.reader,out);
        EXPECT_TRUE(status==Status::Truncated||status==Status::End);EXPECT_EQ(out,nullptr);}
}
TEST(ArrowIpcReader, LimitsAndWorkspaceAdmissionAreExplicit){
    const auto bytes=load("flat.arrows");Work w;
    EXPECT_EQ(arrow_ipc_reader_open(bytes.data(),bytes.size(),w.limits,w.p,w.size-1,w.reader),Status::WorkspaceSmall);
    auto l=w.limits;l.max_fields=22;EXPECT_EQ(arrow_ipc_reader_open(bytes.data(),bytes.size(),l,w.p,w.size,w.reader),Status::Limit);
    l=w.limits;l.max_key_values=1;EXPECT_EQ(arrow_ipc_reader_open(bytes.data(),bytes.size(),l,w.p,w.size,w.reader),Status::Limit);
    l=w.limits;l.max_metadata_bytes=8;EXPECT_EQ(arrow_ipc_reader_open(bytes.data(),bytes.size(),l,w.p,w.size,w.reader),Status::Limit);
    l=w.limits;l.max_batch_rows=2;ASSERT_EQ(arrow_ipc_reader_open(bytes.data(),bytes.size(),l,w.p,w.size,w.reader),Status::Ok);
    const BoltBatch* out=nullptr;EXPECT_EQ(arrow_ipc_reader_next(w.reader,out),Status::Limit);EXPECT_EQ(out,nullptr);
    Work small(1);ASSERT_EQ(small.open(bytes),Status::Ok);EXPECT_EQ(arrow_ipc_reader_next(small.reader,out),Status::Limit);
    Work body;body.limits.max_body_bytes=8;ASSERT_EQ(body.open(bytes),Status::Ok);EXPECT_EQ(arrow_ipc_reader_next(body.reader,out),Status::Limit);
    Work file;file.limits.max_file_batches=2;EXPECT_EQ(file.open(load("flat.arrow")),Status::Limit);
    std::uint64_t need=123;l=w.limits;l.output_bytes=UINT64_MAX;EXPECT_EQ(arrow_ipc_reader_workspace_size(l,need),Status::InvalidArgument);EXPECT_EQ(need,123u);
}
TEST(ArrowIpcReader, RefusedNextPreservesPreviousBatchAndCursor){
    auto bytes=load("flat.arrows");const auto first=next_message(bytes,0),second=next_message(bytes,first);
    const auto p=positions(bytes,second);put64(bytes,p.nodes,4); // second empty batch has a contradictory node
    Work w;ASSERT_EQ(w.open(bytes),Status::Ok);const BoltBatch* out=nullptr;
    ASSERT_EQ(arrow_ipc_reader_next(w.reader,out),Status::Ok);values(*out);
    const auto* old=out;EXPECT_EQ(arrow_ipc_reader_next(w.reader,out),Status::BadInput);EXPECT_EQ(out,old);values(*out);
    EXPECT_EQ(arrow_ipc_reader_next(w.reader,out),Status::BadInput);values(*out); // no cursor or payload change
}
TEST(ArrowIpcReader, OffsetNullBodyAndUtf8CorruptionRefuseBeforeOutput){
    const auto original=load("flat.arrows");const auto start=next_message(original,0);const auto p=positions(original,start);
    for(unsigned mode=0;mode<5;++mode){auto bytes=original;
        if(mode==0)put64(bytes,p.buffers+8,UINT64_MAX); // signed buffer length
        if(mode==1)put64(bytes,p.buffers+16,1); // unaligned numeric buffer
        if(mode==2)put64(bytes,p.nodes+8,0); // null count contradicts bitmap
        // text is column12, preceding12 primitive columns each have2 buffers
        const auto off=u64(bytes,p.buffers+25*16),data=u64(bytes,p.buffers+26*16);
        if(mode==3)put32(bytes,p.body+off+4,UINT32_MAX);
        if(mode==4)bytes[p.body+data]=0xff;
        Work w;ASSERT_EQ(w.open(bytes),Status::Ok);const BoltBatch* out=nullptr;EXPECT_EQ(arrow_ipc_reader_next(w.reader,out),Status::BadInput);EXPECT_EQ(out,nullptr);
    }
}
TEST(ArrowIpcReader, FlatbufferRootVtableAndFooterIndexCorruption){
    auto original=load("flat.arrow");ASSERT_FALSE(original.empty());
    for(unsigned mode=0;mode<5;++mode){auto bytes=original;const auto n=u32(bytes,bytes.size()-10);
        const auto footer=bytes.size()-10-n;
        const auto table=footer+u32(bytes,footer),blocks=ref(bytes,table,3)+4;
        if(mode==0)put32(bytes,footer,UINT32_MAX);
        if(mode==1)put64(bytes,blocks,UINT64_MAX);
        if(mode==2)put32(bytes,blocks+8,8);
        if(mode==3)put64(bytes,blocks+16,UINT64_MAX);
        if(mode==4)bytes.back()='x';
        Work w;EXPECT_NE(w.open(bytes),Status::Ok);EXPECT_EQ(w.reader.state,nullptr);
    }
    auto stream=load("flat.arrows");put32(stream,8,UINT32_MAX);Work w;EXPECT_EQ(w.open(stream),Status::BadInput);
}
TEST(ArrowIpcReader, InputWorkspaceReaderAndOutputShellAliasAdmission){
    const auto bytes=load("flat.arrows");Work w;
    ASSERT_LT(bytes.size(),w.size);std::memcpy(w.p,bytes.data(),bytes.size());
    EXPECT_EQ(arrow_ipc_reader_open(w.p,bytes.size(),w.limits,w.p,w.size,w.reader),Status::InvalidArgument);
    auto* inside=new(w.p) ArrowIpcReader{};
    EXPECT_EQ(arrow_ipc_reader_open(bytes.data(),bytes.size(),w.limits,w.p,w.size,*inside),Status::InvalidArgument);
    inside->~ArrowIpcReader();
    auto* config=new(w.p) IpcReadLimits(w.limits);
    EXPECT_EQ(arrow_ipc_reader_open(bytes.data(),bytes.size(),*config,w.p,w.size,w.reader),Status::InvalidArgument);
    config->~IpcReadLimits();ASSERT_EQ(w.open(bytes),Status::Ok);
    auto** shell=static_cast<const BoltBatch**>(w.p); // refused before writing State
    EXPECT_EQ(arrow_ipc_reader_next(w.reader,*shell),Status::InvalidArgument);
    const BoltBatch* out=nullptr;ASSERT_EQ(arrow_ipc_reader_next(w.reader,out),Status::Ok);values(*out);
}
TEST(ArrowIpcReader, MetadataStringsRequireUtf8AndTablesRequireAlignment){
    const auto original=load("flat.arrows");ASSERT_FALSE(original.empty());
    const auto meta=std::size_t{8},message=meta+u32(original,meta);
    const auto schema=ref(original,message,2),fields=ref(original,schema,1)+4;
    const auto first=fields+u32(original,fields),name=ref(original,first,0);
    ASSERT_GT(u32(original,name),0u);
    auto bytes=original;bytes[name+4]=0xff;Work invalid_name;
    EXPECT_EQ(invalid_name.open(bytes),Status::BadInput);
    bytes=original;put32(bytes,meta,u32(bytes,meta)+1);Work unaligned_root;
    EXPECT_EQ(unaligned_root.open(bytes),Status::BadInput);
}
TEST(ArrowIpcReader, ReusableNextHasNoHeapAllocation){
    const auto bytes=load("flat.arrows");Work w;ASSERT_EQ(w.open(bytes),Status::Ok);
    const auto before=allocations.load(std::memory_order_relaxed);const BoltBatch* out=nullptr;
    const auto a=arrow_ipc_reader_next(w.reader,out),b=arrow_ipc_reader_next(w.reader,out),c=arrow_ipc_reader_next(w.reader,out),d=arrow_ipc_reader_next(w.reader,out);
    const auto after=allocations.load(std::memory_order_relaxed);
    EXPECT_EQ(a,Status::Ok);EXPECT_EQ(b,Status::Ok);EXPECT_EQ(c,Status::Ok);EXPECT_EQ(d,Status::End);EXPECT_EQ(before,after);values(*out);
}
} // namespace
