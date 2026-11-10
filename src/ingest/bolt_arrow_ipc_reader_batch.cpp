#include "bolt_arrow_ipc_reader_internal.h"
#include "bolt/kernels/bolt_utf8_cp.h"
namespace bolt::ingest::ipc_read_detail {
struct Buffer { const std::uint8_t* p=nullptr;std::uint64_t n=0; };
static Status buffer(const State& s,const Message& m,std::uint32_t at,Buffer& out) noexcept {
    std::int64_t off=0,n=0;
    if(!m.fb.get(at,off)||!m.fb.get(at+8,n)||off<0||n<0||off%8||
        !range(static_cast<std::uint64_t>(off),static_cast<std::uint64_t>(n),m.next-m.body))return Status::BadInput;
    out={s.input+m.body+static_cast<std::uint64_t>(off),static_cast<std::uint64_t>(n)};
    assert(out.n<=m.next-m.body);assert(out.p>=s.input);return Status::Ok;
}
static bool reserve(std::uint64_t& used,std::uint64_t n,std::uint64_t cap) noexcept {
    if(used>UINT64_MAX-63)return false;
    const auto aligned=(used+63)&~std::uint64_t{63};
    if(!range(aligned,n,cap))return false;
    used=aligned+n;return true;
}
static bool valid(const Layout& l,std::uint64_t row) noexcept {
    return l.validity==nullptr || ((l.validity[row/8]>>(row%8))&1)!=0;
}
static Status validity(const IpcReadField& f,Buffer b,std::int64_t rows,std::int64_t nulls,Layout& l) noexcept {
    if(nulls<0||nulls>rows||(!f.nullable&&nulls))return Status::BadInput;
    const auto bytes=(static_cast<std::uint64_t>(rows)+7)/8;
    if((nulls&&b.n<bytes)||(b.n&&b.n<bytes))return Status::BadInput;
    l.validity=b.n?b.p:nullptr;l.nulls=nulls;
    std::int64_t count=0;
    if(b.n)for(std::int64_t i=0;i<rows;++i)count+=!valid(l,static_cast<std::uint64_t>(i));
    if(count!=nulls)return Status::BadInput;
    assert(nulls<=rows);assert(l.validity!=nullptr || nulls==0);return Status::Ok;
}
static Status offsets(Buffer b,Buffer values,std::int64_t rows,Layout& l) noexcept {
    const auto bytes=(static_cast<std::uint64_t>(rows)+1)*4;
    if(b.n<bytes)return Status::BadInput;
    std::int32_t prev=0;wire::ByteReader reader{b.p,4,0};
    if(!wire::read_le(reader,prev)||prev<0||static_cast<std::uint64_t>(prev)>values.n)return Status::BadInput;
    for(std::int64_t i=1;i<=rows;++i){std::int32_t n=0;
        reader={b.p+static_cast<std::uint64_t>(i)*4,4,0};
        if(!wire::read_le(reader,n)||n<prev||static_cast<std::uint64_t>(n)>values.n)return Status::BadInput;
        prev=n;}
    l.offsets=b.p;l.values=values.p;l.value_bytes=values.n;
    assert(l.offsets!=nullptr);assert(static_cast<std::uint64_t>(prev)<=values.n);return Status::Ok;
}
static bool utf8_values(const Layout& l,std::int64_t rows) noexcept {
    for(std::int64_t i=0;i<rows;++i){
        if(!valid(l,static_cast<std::uint64_t>(i)))continue;
        wire::ByteReader a{l.offsets+static_cast<std::uint64_t>(i)*4,4,0};
        wire::ByteReader b{l.offsets+static_cast<std::uint64_t>(i+1)*4,4,0};
        std::int32_t begin=0,end=0;
        if(!wire::read_le(a,begin)||!wire::read_le(b,end))return false;
        const auto size=static_cast<std::uint32_t>(end-begin);
        const auto* data=reinterpret_cast<const char*>(l.values+begin);
        for(std::uint32_t pos=0;pos<size;){std::uint32_t cp=0;
            pos+=kernels::utf8::utf8_cp_decode(data,size,pos,&cp);
            if(cp==kernels::utf8::kUtf8Invalid)return false;}
    }
    assert(rows>=0);assert(l.offsets!=nullptr);return true;
}
// Pre-order walk of one field node and its children: the FieldNode and
// buffer streams of a record batch are both depth-first.
struct Walk { const Message* m; std::uint32_t nodes, node, buf; std::uint64_t used; };
static Status list_offsets(Buffer b,std::int64_t rows,Layout& l,std::int64_t& children) noexcept {
    if(b.n<(static_cast<std::uint64_t>(rows)+1)*4)return Status::BadInput;
    std::int32_t prev=0;wire::ByteReader reader{b.p,4,0};
    if(!wire::read_le(reader,prev))return Status::BadInput;
    if(prev!=0)return Status::Unsupported;   // sliced lists: offsets not rebased
    for(std::int64_t i=1;i<=rows;++i){std::int32_t n=0;
        reader={b.p+static_cast<std::uint64_t>(i)*4,4,0};
        if(!wire::read_le(reader,n)||n<prev)return Status::BadInput;
        prev=n;}
    l.offsets=b.p;children=prev;
    assert(children>=0);assert(l.offsets!=nullptr);return Status::Ok;
}
static Status leaf(State& s,const Message& m,const IpcReadField& f,Layout& l,Walk& w,std::int64_t rows) noexcept {
    Buffer values{},off{};Status st=Status::Ok;
    if(f.arrow_type==4||f.arrow_type==5){
        st=buffer(s,m,w.buf,off);if(st!=Status::Ok)return st;w.buf+=16;
        st=buffer(s,m,w.buf,values);if(st!=Status::Ok)return st;w.buf+=16;
        st=offsets(off,values,rows,l);if(st!=Status::Ok)return st;
        if(f.arrow_type==5&&!utf8_values(l,rows))return Status::BadInput;
        if(!reserve(w.used,(static_cast<std::uint64_t>(rows)+1)*4,s.limits.output_bytes)||
            !reserve(w.used,values.n,s.limits.output_bytes)||
            (f.arrow_type==5&&!reserve(w.used,static_cast<std::uint64_t>(rows)*sizeof(StringView),s.limits.output_bytes)))return Status::Limit;
        return Status::Ok;
    }
    st=buffer(s,m,w.buf,values);if(st!=Status::Ok)return st;w.buf+=16;
    const auto required=f.arrow_type==6?(static_cast<std::uint64_t>(rows)+7)/8:
        static_cast<std::uint64_t>(rows)*static_cast<std::uint32_t>(f.byte_width);
    if(values.n<required)return Status::BadInput;
    l.values=values.p;l.value_bytes=required;
    const auto width=f.arrow_type==15?16u:static_cast<std::uint32_t>(f.byte_width);
    if(!reserve(w.used,static_cast<std::uint64_t>(rows)*width,s.limits.output_bytes))return Status::Limit;
    return Status::Ok;
}
static Status column(State& s,Walk& w,std::uint32_t index,std::int64_t rows,std::uint32_t depth) noexcept {
    assert(index<s.node_count);assert(depth<=kMaxIpcDepth);
    const Message& m=*w.m;const auto& f=s.fields[index];auto& l=s.layout[index];l={};
    if(w.node>=w.nodes)return Status::BadInput;
    std::int64_t length=0,nulls=0;const auto at=w.node;w.node+=16;
    if(!m.fb.get(at,length)||!m.fb.get(at+8,nulls))return Status::BadInput;
    if(length!=rows)return length>rows&&depth?Status::Unsupported:Status::BadInput;
    Buffer v{};auto st=buffer(s,m,w.buf,v);if(st!=Status::Ok)return st;w.buf+=16;
    st=validity(f,v,rows,nulls,l);if(st!=Status::Ok)return st;
    l.length=rows;
    if(!reserve(w.used,l.validity?(static_cast<std::uint64_t>(rows)+7)/8:0,s.limits.output_bytes))return Status::Limit;
    if(f.arrow_type==13){
        for(std::uint32_t i=0;i<f.child_count;++i){
            st=column(s,w,f.first_child+i,rows,depth+1);if(st!=Status::Ok)return st;}
        return Status::Ok;
    }
    if(f.arrow_type==12||f.arrow_type==17){
        Buffer off{};st=buffer(s,m,w.buf,off);if(st!=Status::Ok)return st;w.buf+=16;
        std::int64_t children=0;st=list_offsets(off,rows,l,children);if(st!=Status::Ok)return st;
        if(!reserve(w.used,(static_cast<std::uint64_t>(rows)+1)*4,s.limits.output_bytes))return Status::Limit;
        return column(s,w,f.first_child,children,depth+1);
    }
    return leaf(s,m,f,l,w,rows);
}
static std::uint32_t buffer_count(const IpcReadField& f) noexcept {
    return f.arrow_type==4||f.arrow_type==5||f.arrow_type==12||f.arrow_type==17?
        (f.arrow_type==4||f.arrow_type==5?3u:2u):f.arrow_type==13?1u:2u;
}
Status preflight(State& s,const Message& m,std::int64_t& rows,std::uint64_t& bytes) noexcept {
    assert(s.schema.field_count<=s.node_count);assert(s.node_count<=s.limits.max_fields);assert(m.kind==3);
    std::uint32_t nodes=0,nn=0,buffers=0,nb=0,compression=0,variadic=0,nv=0;
    if(!m.header.scalar(0,std::int64_t{0},rows)||rows<0||!m.header.vector(1,16,nodes,nn)||
        !m.header.vector(2,16,buffers,nb)||!m.header.ref(3,compression)||!m.header.vector(4,8,variadic,nv))return Status::BadInput;
    if(compression||nv)return Status::Unsupported;
    if(static_cast<std::uint64_t>(rows)>s.limits.max_batch_rows)return Status::Limit;
    std::uint32_t expected=0;
    for(std::uint32_t i=0;i<s.node_count;++i)expected+=buffer_count(s.fields[i]);
    if(nn!=s.node_count||nb!=expected)return Status::BadInput;
    Walk w{&m,nodes+16*nn,nodes,buffers,0};
    for(std::uint32_t i=0;i<s.schema.field_count;++i){auto st=column(s,w,i,rows,0);if(st!=Status::Ok)return st;}
    if(w.node!=w.nodes||w.buf!=buffers+16*nb)return Status::BadInput;
    bytes=w.used;assert(bytes<=s.limits.output_bytes);return Status::Ok;
}
static void* take(State& s,std::uint64_t& at,std::uint64_t n) noexcept {
    at=(at+63)&~std::uint64_t{63};void* p=s.output+at;at+=n;
    assert(at<=s.limits.output_bytes);assert(static_cast<std::uint8_t*>(p)>=s.output);return p;
}
static std::int32_t offset_at(const Layout& l,std::uint32_t i) noexcept {
    wire::ByteReader reader{l.offsets+static_cast<std::uint64_t>(i)*4,4,0};
    std::int32_t n=0;const bool ok=wire::read_le(reader,n);
    assert(ok);assert(n>=0);(void)ok;return n;
}
static void string_views(const Layout& l,std::int64_t rows,StringView* dst) noexcept {
    for(std::int64_t i=0;i<rows;++i){auto& v=dst[i];v={};if(!valid(l,static_cast<std::uint64_t>(i)))continue;
        const auto start=offset_at(l,static_cast<std::uint32_t>(i));
        const auto n=offset_at(l,static_cast<std::uint32_t>(i+1))-start;v.length=static_cast<std::uint32_t>(n);
        const auto* p=l.values+start;const auto prefix=n<4?n:4;
        if(prefix)std::memcpy(v.prefix,p,static_cast<std::size_t>(prefix));
        if(n<=12){if(n>4)std::memcpy(v.inline_data,p+4,static_cast<std::size_t>(n-4));}
        else{v.ref.buf_idx=0;v.ref.offset=static_cast<std::uint32_t>(start);}
    }
    assert(rows>=0);assert(dst!=nullptr);
}
static void publish_column(State& s,std::uint32_t i,std::int64_t rows,std::uint64_t& at,std::uint32_t depth) noexcept {
    assert(depth<=kMaxIpcDepth);
    const auto& f=s.fields[i];const auto& l=s.layout[i];const auto n=static_cast<std::uint64_t>(rows);
    auto* v=static_cast<std::uint8_t*>(take(s,at,l.validity?(n+7)/8:0));
    if(l.validity&&n)std::memcpy(v,l.validity,static_cast<std::size_t>((n+7)/8));
    auto& c=s.cols[i];
    if(f.arrow_type==12||f.arrow_type==13||f.arrow_type==17){
        c=BoltColumn::make_empty();
        c.format=ColumnFormat::Nested;c.type=f.physical_type;c.type_size_bytes=0;c.length=rows;
        c.data=&s.cols[f.first_child];c.seq_offset=f.child_count;c.validity=l.validity?v:nullptr;
        std::int64_t children=rows;
        if(f.arrow_type!=13){
            auto* off=static_cast<std::int32_t*>(take(s,at,(n+1)*4));
            std::memcpy(off,l.offsets,static_cast<std::size_t>((n+1)*4));
            children=off[n];
            s.offset_cols[i]=BoltColumn::make_flat(off,nullptr,rows+1,BoltType::Int32);c.dict_child=&s.offset_cols[i];
        }
        for(std::uint32_t k=0;k<f.child_count;++k)publish_column(s,f.first_child+k,children,at,depth+1);
    }else if(f.arrow_type==4||f.arrow_type==5){
        auto* off=static_cast<std::int32_t*>(take(s,at,(n+1)*4));
        std::memcpy(off,l.offsets,static_cast<std::size_t>((n+1)*4));
        auto* data=take(s,at,l.value_bytes);if(l.value_bytes)std::memcpy(data,l.values,static_cast<std::size_t>(l.value_bytes));
        if(f.arrow_type==4){c=BoltColumn::make_flat(data,l.validity?v:nullptr,rows,BoltType::Binary);
            c.format=ColumnFormat::VarBinary;c.type_size_bytes=0;
            s.offset_cols[i]=BoltColumn::make_flat(off,nullptr,rows+1,BoltType::Int32);c.dict_child=&s.offset_cols[i];}
        else{auto* views=static_cast<StringView*>(take(s,at,n*sizeof(StringView)));
            string_views(l,rows,views);c=BoltColumn::make_flat(views,l.validity?v:nullptr,rows,BoltType::Utf8);c.str_overflow_base=data;}
    }else{
        const auto width=f.arrow_type==15?16u:static_cast<std::uint32_t>(f.byte_width);
        auto* data=static_cast<std::uint8_t*>(take(s,at,n*width));
        if(f.arrow_type==6){for(std::uint64_t row=0;row<n;++row)data[row]=static_cast<std::uint8_t>((l.values[row/8]>>(row%8))&1);}
        else if(f.arrow_type==15){for(std::uint64_t row=0;row<n;++row){std::memset(data+row*16,0,16);
            std::memcpy(data+row*16,l.values+row*static_cast<std::uint32_t>(f.byte_width),static_cast<std::size_t>(f.byte_width));}}
        else if(n)std::memcpy(data,l.values,static_cast<std::size_t>(n*width));
        c=BoltColumn::make_flat(data,l.validity?v:nullptr,rows,f.physical_type);
        if(f.arrow_type==15)c.fixed_width=static_cast<std::uint8_t>(f.byte_width);
        if(f.arrow_type==7)c.decimal_scale=static_cast<std::uint8_t>(f.scale);
    }
    c.stats.null_count=static_cast<std::uint32_t>(l.nulls);c.stats.all_valid=l.nulls==0;assert(c.length==rows);assert(c.type==f.physical_type);
}
void publish(State& s,std::int64_t rows) noexcept {
    assert(rows>=0);assert(static_cast<std::uint64_t>(rows)<=s.limits.max_batch_rows);
    std::uint64_t at=0;
    for(std::uint32_t i=0;i<s.schema.field_count;++i){publish_column(s,i,rows,at,0);const auto& f=s.fields[i];
        auto& b=s.bolt_fields[i];b={};if(f.name.size)std::memcpy(b.name,f.name.data,f.name.size);
        b.type=f.physical_type;b.nullable=f.nullable;b.decimal_scale=static_cast<std::uint8_t>(f.scale);
        b.fixed_size=f.arrow_type==15?static_cast<std::uint32_t>(f.byte_width):0;}
    BoltBatch::init_empty(&s.batch);s.batch.columns[0]=s.cols;s.batch.columns[1]=s.other_cols;
    s.batch.num_cols=s.schema.field_count;s.batch.num_rows=rows;
    s.batch.schema.set_storage(s.bolt_fields,s.schema.field_count);s.batch.schema.num_fields=s.schema.field_count;
    assert(at<=s.limits.output_bytes);assert(s.batch.num_rows==rows);
}
} // namespace bolt::ingest::ipc_read_detail
