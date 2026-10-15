#include "bolt_arrow_ipc_reader_internal.h"
namespace bolt::ingest::ipc_read_detail {
Status key_values(Table t,std::uint32_t slot,IpcReadKeyValue* dst,std::uint32_t cap,
                  std::uint32_t& used,const IpcReadKeyValue*& out,std::uint32_t& count) noexcept {
    assert(used<=cap); assert(dst!=nullptr || cap==0);
    std::uint32_t p=0,n=0;if (!t.vector(slot,4,p,n)) return Status::BadInput;
    if (n>cap-used) return Status::Limit;
    out=n?dst+used:nullptr;count=n;
    for (std::uint32_t i=0;i<n;++i) {
        std::uint32_t a=0;Table kv{};
        if (!indirect(t.fb,p+4*i,a) || !kv.init(t.fb,a) ||
            !kv.string(0,dst[used+i].key) || !kv.string(1,dst[used+i].value)) return Status::BadInput;
    }
    used+=n;assert(used<=cap);return Status::Ok;
}
static Status type_int(Table t,IpcReadField& f) noexcept {
    std::int32_t bits=0;std::uint8_t sign=0;
    if (!t.scalar(0,0,bits) || !t.scalar(1,std::uint8_t{0},sign) || sign>1) return Status::BadInput;
    switch(bits) {
        case 8:f.physical_type=sign?BoltType::Int8:BoltType::UInt8;break;
        case 16:f.physical_type=sign?BoltType::Int16:BoltType::UInt16;break;
        case 32:f.physical_type=sign?BoltType::Int32:BoltType::UInt32;break;
        case 64:f.physical_type=sign?BoltType::Int64:BoltType::UInt64;break;
        default:return Status::BadInput;
    }
    f.byte_width=bits/8;assert(f.byte_width<=8);assert(f.byte_width>0);return Status::Ok;
}
static Status temporal(Table t,IpcReadField& f) noexcept {
    const std::int16_t def=(f.arrow_type==9||f.arrow_type==18)?1:0;
    if (!t.scalar(0,def,f.unit) || f.unit<0 || f.unit>3) return Status::BadInput;
    f.byte_width=8;f.physical_type=BoltType::Int64;
    if (f.arrow_type==9) {
        std::int32_t bits=0;if (!t.scalar(1,32,bits)) return Status::BadInput;
        if (bits!=(f.unit<2?32:64)) return Status::BadInput;
        f.byte_width=bits/8;if(bits==32) f.physical_type=BoltType::Int32;
    }
    if (f.arrow_type==10 && !t.string(1,f.timezone)) return Status::BadInput;
    assert(f.byte_width==4 || f.byte_width==8);assert(f.unit>=0);return Status::Ok;
}
static Status field_type(Table t,IpcReadField& f) noexcept {
    std::int16_t precision=0;std::int32_t width=0;
    switch(f.arrow_type) {
        case 2:return type_int(t,f);
        case 3:
            if (!t.scalar(0,std::int16_t{0},precision) || precision<0 || precision>2) return Status::BadInput;
            f.physical_type=precision==0?BoltType::Float16:precision==1?BoltType::Float32:BoltType::Float64;
            f.byte_width=2<<precision;break;
        case 4:f.physical_type=BoltType::Binary;break;
        case 5:f.physical_type=BoltType::Utf8;break;
        case 6:f.physical_type=BoltType::Bool;f.byte_width=1;break;
        case 7:
            if (!t.scalar(0,0,f.precision) || !t.scalar(1,0,f.scale) || !t.scalar(2,128,width)) return Status::BadInput;
            if (width!=128) return Status::Unsupported;
            if (f.precision<1 || f.precision>38) return Status::BadInput;
            if (f.scale<0 || f.scale>38) return Status::Unsupported;
            f.byte_width=16;f.physical_type=BoltType::Decimal128;break;
        case 8:
            if (!t.scalar(0,std::int16_t{1},f.unit) || f.unit<0 || f.unit>1) return Status::BadInput;
            f.byte_width=f.unit==0?4:8;f.physical_type=f.unit==0?BoltType::Date32:BoltType::Date64;break;
        case 9:case 10:case 18:return temporal(t,f);
        case 15:
            if (!t.scalar(0,0,f.byte_width) || f.byte_width<=0) return Status::BadInput;
            if (f.byte_width>16) return Status::Unsupported;
            f.physical_type=BoltType::FixedSizeBinary;break;
        default:return Status::Unsupported;
    }
    assert(f.physical_type!=BoltType::NA);assert(f.byte_width>=0);return Status::Ok;
}
static bool same_string(IpcReadString a,IpcReadString b) noexcept {
    return a.size==b.size && (a.size==0 || std::memcmp(a.data,b.data,a.size)==0);
}
static bool same_metadata(const IpcReadKeyValue* a,const IpcReadKeyValue* b,std::uint32_t n) noexcept {
    for(std::uint32_t i=0;i<n;++i)
        if(!same_string(a[i].key,b[i].key)||!same_string(a[i].value,b[i].value))return false;
    return true;
}
static bool same_field(const IpcReadField& a,const IpcReadField& b) noexcept {
    return same_string(a.name,b.name)&&same_string(a.timezone,b.timezone)&&a.arrow_type==b.arrow_type&&
        a.physical_type==b.physical_type&&a.nullable==b.nullable&&a.byte_width==b.byte_width&&
        a.precision==b.precision&&a.scale==b.scale&&a.unit==b.unit&&a.metadata_count==b.metadata_count&&
        same_metadata(a.metadata,b.metadata,a.metadata_count);
}
static Status parse_field(Table t,IpcReadField& f,IpcReadKeyValue* kv,std::uint32_t cap,std::uint32_t& used) noexcept {
    std::uint8_t nullable=0;std::uint32_t type=0,dict=0,children=0,count=0;Table ty{};
    f={};
    if (!t.string(0,f.name)||!t.scalar(1,std::uint8_t{0},nullable)||nullable>1||
        !t.scalar(2,std::uint8_t{0},f.arrow_type)||!t.ref(3,type)||!type||!ty.init(t.fb,type)||
        !t.ref(4,dict)||!t.vector(5,4,children,count)) return Status::BadInput;
    if(dict||count) return Status::Unsupported;
    if(f.name.size>kMaxFieldName || (f.name.size && std::memchr(f.name.data,0,f.name.size))) return Status::Unsupported;
    f.nullable=nullable!=0;auto st=field_type(ty,f);if(st!=Status::Ok)return st;
    return key_values(t,6,kv,cap,used,f.metadata,f.metadata_count);
}
Status parse_schema(State& s,Table schema,bool compare) noexcept {
    assert(s.fields!=nullptr || s.limits.max_fields==0);assert(s.limits.max_fields<=kMaxBatchColumns);
    std::int16_t endian=0;std::uint32_t fields=0,n=0,features=0,nfeatures=0;
    if(!schema.scalar(0,std::int16_t{0},endian)||!schema.vector(1,4,fields,n)||
        !schema.vector(3,8,features,nfeatures))return Status::BadInput;
    if(endian==1)return Status::Unsupported;
    if(endian!=0)return Status::BadInput;
    if(n>s.limits.max_fields)return Status::Limit;
    for(std::uint32_t i=0;i<nfeatures;++i){std::int64_t feature=0;
        if(!schema.fb.get(features+8*i,feature))return Status::BadInput;
        if(feature!=0)return Status::Unsupported;}
    auto* dst=compare?s.compare_fields:s.fields;auto* kv=compare?s.compare_kv:s.kv;
    std::uint32_t used=0;
    for(std::uint32_t i=0;i<n;++i){std::uint32_t p=0;Table field{};
        if(!indirect(schema.fb,fields+4*i,p)||!field.init(schema.fb,p))return Status::BadInput;
        auto st=parse_field(field,dst[i],kv,s.limits.max_key_values,used);if(st!=Status::Ok)return st;
        if(compare&&(i>=s.schema.field_count||!same_field(s.fields[i],dst[i])))return Status::BadInput;}
    const IpcReadKeyValue* meta=nullptr;std::uint32_t nmeta=0;
    auto st=key_values(schema,2,kv,s.limits.max_key_values,used,meta,nmeta);if(st!=Status::Ok)return st;
    if(compare){if(n!=s.schema.field_count||nmeta!=s.schema.metadata_count||
        !same_metadata(meta,s.schema.metadata,nmeta))return Status::BadInput;}
    else{s.schema.fields=dst;s.schema.field_count=n;s.schema.metadata=meta;s.schema.metadata_count=nmeta;}
    return Status::Ok;
}
} // namespace bolt::ingest::ipc_read_detail
