#include "bolt_arrow_ipc_reader_internal.h"
namespace bolt::ingest::ipc_read_detail {
Status message(const State& s,std::uint64_t offset,Message& out) noexcept {
    assert(s.input_bytes<=SIZE_MAX);assert(s.stream_end<=s.input_bytes);
    if(!range(offset,4,s.stream_end))return Status::Truncated;
    wire::ByteReader r{s.input+offset,4,0};std::uint32_t n=0;
    if(!wire::read_le(r,n))return Status::Truncated;
    std::uint32_t prefix=4;
    if(n==UINT32_MAX){prefix=8;if(!range(offset,8,s.stream_end))return Status::Truncated;
        r={s.input+offset+4,4,0};if(!wire::read_le(r,n))return Status::Truncated;}
    Message m{};m.next=offset+prefix;
    if(n==0){m.eos=true;out=m;return Status::Ok;}
    if(n>INT32_MAX)return Status::BadInput;
    if(n>s.limits.max_metadata_bytes)return Status::Limit;
    if(!range(offset+prefix,n,s.stream_end))return Status::Truncated;
    m.fb={s.input+offset+prefix,n};m.metadata=prefix+n;
    std::uint32_t root=0,header=0;Table t{};std::int64_t body=0;
    if(!m.fb.root(root)||!t.init(m.fb,root)||!t.scalar(0,std::int16_t{0},m.version)||
        !t.scalar(1,std::uint8_t{0},m.kind)||!t.ref(2,header)||!header||
        !m.header.init(m.fb,header)||!t.scalar(3,std::int64_t{0},body)||body<0)return Status::BadInput;
    if(m.version!=3&&m.version!=4)return Status::Unsupported;
    if(body%8 || m.metadata%8)return Status::BadInput;
    if(static_cast<std::uint64_t>(body)>s.limits.max_body_bytes)return Status::Limit;
    m.body=offset+m.metadata;
    if(!range(m.body,static_cast<std::uint64_t>(body),s.stream_end))return Status::Truncated;
    m.next=m.body+static_cast<std::uint64_t>(body);out=m;return Status::Ok;
}
static Status footer(State& s,std::uint32_t& schema_at) noexcept {
    if(s.input_bytes<18||std::memcmp(s.input+s.input_bytes-6,"ARROW1",6))return Status::BadInput;
    wire::ByteReader r{s.input+s.input_bytes-10,4,0};std::int32_t n=0;
    if(!wire::read_le(r,n)||n<=0)return Status::BadInput;
    if(static_cast<std::uint32_t>(n)>s.limits.max_metadata_bytes)return Status::Limit;
    if(static_cast<std::uint64_t>(n)>s.input_bytes-18)return Status::Truncated;
    s.stream_end=s.input_bytes-10-static_cast<std::uint32_t>(n);
    s.footer={s.input+s.stream_end,static_cast<std::uint32_t>(n)};
    std::uint32_t root=0,dictionaries=0,nd=0;Table f{};std::int16_t version=0;
    if(!s.footer.root(root)||!f.init(s.footer,root)||!f.scalar(0,std::int16_t{0},version)||
        !f.ref(1,schema_at)||!schema_at||!f.vector(2,24,dictionaries,nd)||
        !f.vector(3,24,s.file_blocks,s.block_count))return Status::BadInput;
    if(version!=s.version)return Status::BadInput;
    if(nd)return Status::Unsupported;
    if(s.block_count>s.limits.max_file_batches)return Status::Limit;
    std::uint32_t used=0;
    return key_values(f,4,s.compare_kv,s.limits.max_key_values,used,
        s.schema.file_metadata,s.schema.file_metadata_count);
}
static Status blocks(State& s) noexcept {
    std::uint64_t previous=s.cursor;
    for(std::uint32_t i=0;i<s.block_count;++i){
        const auto p=s.file_blocks+i*24;std::int64_t off=0,body=0;std::int32_t meta=0;
        if(!s.footer.get(p,off)||!s.footer.get(p+8,meta)||!s.footer.get(p+16,body)||
            off<0||meta<=0||body<0||off%8||meta%8||body%8)return Status::BadInput;
        if(static_cast<std::uint64_t>(off)!=previous)return Status::Unsupported;
        Message m{};auto st=message(s,static_cast<std::uint64_t>(off),m);if(st!=Status::Ok)return st;
        if(m.eos||m.kind!=3||m.version!=s.version||m.metadata!=static_cast<std::uint32_t>(meta)||
            m.next-m.body!=static_cast<std::uint64_t>(body))return Status::BadInput;
        previous=m.next;
    }
    if(previous==s.stream_end)return Status::Ok;
    Message eos{};auto st=message(s,previous,eos);if(st!=Status::Ok)return st;
    if(!eos.eos||eos.next!=s.stream_end)return Status::BadInput;
    assert(previous<=s.stream_end);assert(s.block_count<=s.limits.max_file_batches);return Status::Ok;
}
Status open_input(State& s) noexcept {
    assert(s.input!=nullptr || s.input_bytes==0);assert(s.stream_end==s.input_bytes);
    s.file=s.input_bytes>=6&&std::memcmp(s.input,"ARROW1",6)==0;
    const auto start=s.file?8u:0u;Message m{};
    auto st=message(s,start,m);if(st!=Status::Ok)return st;
    if(m.eos||m.kind!=1||m.body!=m.next)return Status::BadInput;
    s.version=m.version;s.cursor=m.next;
    st=parse_schema(s,m.header,false);if(st!=Status::Ok)return st;
    if(!s.file)return Status::Ok;
    std::uint32_t schema_at=0;st=footer(s,schema_at);if(st!=Status::Ok)return st;
    if(s.cursor>s.stream_end)return Status::BadInput;
    // Comparison scratch must not overwrite retained footer metadata.
    Table schema{};if(!schema.init(s.footer,schema_at))return Status::BadInput;
    st=parse_schema(s,schema,true);if(st!=Status::Ok)return st;
    // Re-read file metadata after the schema comparison reuses compare_kv.
    std::uint32_t root=0,used=0;Table f{};
    if(!s.footer.root(root)||!f.init(s.footer,root))return Status::BadInput;
    st=key_values(f,4,s.compare_kv,s.limits.max_key_values,used,
        s.schema.file_metadata,s.schema.file_metadata_count);if(st!=Status::Ok)return st;
    return blocks(s);
}
} // namespace bolt::ingest::ipc_read_detail
namespace bolt::ingest {
using namespace ipc_read_detail;
static bool add_part(std::uint64_t& n,std::uint64_t count,std::uint64_t width,std::uint64_t alignment) noexcept {
    if(n>UINT64_MAX-(alignment-1))return false;
    n=(n+alignment-1)&~(alignment-1);
    if(count>(UINT64_MAX-n)/width)return false;
    n+=count*width;return n<=SIZE_MAX;
}
IpcReadStatus arrow_ipc_reader_workspace_size(const IpcReadLimits& l,std::uint64_t& bytes) noexcept {
    if(l.max_fields>kMaxBatchColumns||l.max_metadata_bytes<8||l.max_metadata_bytes>INT32_MAX||
        l.max_batch_rows>INT32_MAX||l.output_bytes>SIZE_MAX)return Status::InvalidArgument;
    std::uint64_t n=0;
    if(!add_part(n,1,sizeof(State),64)||!add_part(n,2ull*l.max_fields,sizeof(IpcReadField),alignof(IpcReadField))||
        !add_part(n,2ull*l.max_key_values,sizeof(IpcReadKeyValue),alignof(IpcReadKeyValue))||
        !add_part(n,l.max_fields,sizeof(Layout),alignof(Layout))||
        !add_part(n,3ull*l.max_fields,sizeof(BoltColumn),alignof(BoltColumn))||
        !add_part(n,l.max_fields,sizeof(BoltField),alignof(BoltField))||
        !add_part(n,l.output_bytes,1,64))return Status::InvalidArgument;
    bytes=n;assert(n>=sizeof(State));assert(n<=SIZE_MAX);return Status::Ok;
}
static void* carve(std::uint8_t* base,std::uint64_t& at,std::uint64_t n,std::uint64_t alignment) noexcept {
    at=(at+alignment-1)&~(alignment-1);auto* p=base+at;at+=n;return p;
}
static void init_workspace(State& s,std::uint8_t* base,const IpcReadLimits& l) noexcept {
    std::uint64_t at=sizeof(State);s.limits=l;
    s.fields=static_cast<IpcReadField*>(carve(base,at,2ull*l.max_fields*sizeof(IpcReadField),alignof(IpcReadField)));
    s.compare_fields=s.fields+l.max_fields;
    s.kv=static_cast<IpcReadKeyValue*>(carve(base,at,2ull*l.max_key_values*sizeof(IpcReadKeyValue),alignof(IpcReadKeyValue)));
    s.compare_kv=s.kv+l.max_key_values;
    s.layout=static_cast<Layout*>(carve(base,at,l.max_fields*sizeof(Layout),alignof(Layout)));
    s.cols=static_cast<BoltColumn*>(carve(base,at,3ull*l.max_fields*sizeof(BoltColumn),alignof(BoltColumn)));
    s.other_cols=s.cols+l.max_fields;s.offset_cols=s.other_cols+l.max_fields;
    s.bolt_fields=static_cast<BoltField*>(carve(base,at,l.max_fields*sizeof(BoltField),alignof(BoltField)));
    s.output=static_cast<std::uint8_t*>(carve(base,at,l.output_bytes,64));
    assert(at==s.workspace_bytes);assert(s.output>=base+sizeof(State));
}
IpcReadStatus arrow_ipc_reader_open(const void* input,std::uint64_t size,const IpcReadLimits& l,
    void* work,std::uint64_t cap,ArrowIpcReader& out) noexcept {
    std::uint64_t need=0;auto st=arrow_ipc_reader_workspace_size(l,need);if(st!=Status::Ok)return st;
    if(std::endian::native!=std::endian::little)return Status::Unsupported;
    if(out.state||!work||(!input&&size)||size>SIZE_MAX||cap>SIZE_MAX||
        reinterpret_cast<std::uintptr_t>(work)%64||
        size>UINTPTR_MAX-reinterpret_cast<std::uintptr_t>(input)||
        cap>UINTPTR_MAX-reinterpret_cast<std::uintptr_t>(work))return Status::InvalidArgument;
    if(cap<need)return Status::WorkspaceSmall;
    if(overlap(&l,sizeof(l),work,cap)||overlap(input,size,work,cap)||overlap(&out,sizeof(out),work,cap)||overlap(&out,sizeof(out),input,size))return Status::InvalidArgument;
    auto* s=new(work) State{};s->workspace_bytes=need;s->workspace_capacity=cap;init_workspace(*s,static_cast<std::uint8_t*>(work),l);
    s->input=static_cast<const std::uint8_t*>(input);s->input_bytes=size;s->stream_end=size;
    st=open_input(*s);if(st!=Status::Ok)return st;
    out.state=s;out.owns_mapping=false;assert(s->cursor<=s->stream_end);return Status::Ok;
}
IpcReadStatus arrow_ipc_reader_open_file(const char* path,const IpcReadLimits& l,void* work,
    std::uint64_t cap,ArrowIpcReader& out) noexcept {
    if(!path||out.state)return Status::InvalidArgument;
    std::uint32_t n=0;for(;n<4096&&path[n];++n){}if(n==4096)return Status::InvalidArgument;
    io::MappedFile mapping{};
    if(io::mapped_file_open(path,&mapping)!=io::FileIoStatus::kOk)return Status::IoError;
    auto st=arrow_ipc_reader_open(mapping.base,mapping.size,l,work,cap,out);
    if(st!=Status::Ok){io::mapped_file_close(&mapping);return st;}
    out.mapping=mapping;out.owns_mapping=true;assert(out.state!=nullptr);assert(out.mapping.size==mapping.size);return Status::Ok;
}
IpcReadStatus arrow_ipc_reader_schema(const ArrowIpcReader& r,IpcReadSchema& out) noexcept {
    if(!r.state)return Status::Closed;
    const auto& s=*static_cast<const State*>(r.state);
    if(overlap(&out,sizeof(out),r.state,s.workspace_capacity)||overlap(&out,sizeof(out),s.input,s.input_bytes)||
        overlap(&out,sizeof(out),&r,sizeof(r)))return Status::InvalidArgument;
    out=s.schema;assert(out.field_count<=s.limits.max_fields);return Status::Ok;
}
IpcReadStatus arrow_ipc_reader_next(ArrowIpcReader& r,const BoltBatch*& out) noexcept {
    if(!r.state)return Status::Closed;
    auto& s=*static_cast<State*>(r.state);
    if(overlap(&out,sizeof(out),r.state,s.workspace_capacity)||overlap(&out,sizeof(out),s.input,s.input_bytes)||
        overlap(&out,sizeof(out),&r,sizeof(r)))return Status::InvalidArgument;
    if((s.file&&s.batch_index==s.block_count)||s.cursor==s.stream_end)return Status::End;
    Message m{};auto st=message(s,s.cursor,m);if(st!=Status::Ok)return st;
    if(m.eos)return m.next==s.stream_end?Status::End:Status::BadInput;
    if(m.version!=s.version)return Status::BadInput;
    if(m.kind!=3)return Status::Unsupported;
    if(s.batch_index==UINT64_MAX)return Status::Limit;
    std::int64_t rows=0;std::uint64_t bytes=0;st=preflight(s,m,rows,bytes);if(st!=Status::Ok)return st;
    publish(s,rows);s.cursor=m.next;++s.batch_index;out=&s.batch;
    assert(s.cursor<=s.stream_end);assert(bytes<=s.limits.output_bytes);return Status::Ok;
}
void arrow_ipc_reader_close(ArrowIpcReader& r) noexcept {
    if(r.owns_mapping)io::mapped_file_close(&r.mapping);
    r.state=nullptr;r.owns_mapping=false;assert(r.state==nullptr);assert(!r.owns_mapping);
}
} // namespace bolt::ingest
