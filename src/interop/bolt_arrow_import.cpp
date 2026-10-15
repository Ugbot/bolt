#include "bolt/bolt_arrow_import.h"
#include "bolt/bolt_column.h"
#include <cassert>
#include <cstring>
#include <limits>
namespace bolt::arrow {
struct ImportChunk {
    ImportChunk* next=nullptr;
    BoltBatch batch{};
};
namespace {
using Error=ImportError;
bool charge(std::uint64_t& total,std::uint64_t bytes,std::uint64_t count=1) noexcept {
    assert(count>0); assert(total<=UINT64_MAX);
    if (bytes>UINT64_MAX-64 || count>UINT64_MAX/(bytes+64) || total>UINT64_MAX-count*(bytes+64)) return false;
    total+=count*(bytes+64); return true;
}
bool text_size(const char* s,std::uint32_t cap,std::uint32_t& n) noexcept {
    assert(cap>0); assert(s!=nullptr);
    for (n=0;n<cap;++n) if (!s[n]) return true;
    return false;
}
bool span(const void* p,std::uint64_t bytes) noexcept {
    if (!bytes) return true;
    return p!=nullptr && bytes<=SIZE_MAX && reinterpret_cast<std::uintptr_t>(p)<=UINTPTR_MAX-bytes;
}
Error field_type(const char* fmt,ImportField& f) noexcept {
    assert(fmt!=nullptr); assert(f.field.type==BoltType::NA);
    std::uint32_t n=0; if (!text_size(fmt,64,n)) return Error::Unsupported;
    if (n==1) {
        switch (fmt[0]) {
        case 'c':f.field.type=BoltType::Int8;break; case 's':f.field.type=BoltType::Int16;break;
        case 'i':f.field.type=BoltType::Int32;break; case 'l':f.field.type=BoltType::Int64;break;
        case 'C':f.field.type=BoltType::UInt8;break; case 'S':f.field.type=BoltType::UInt16;break;
        case 'I':f.field.type=BoltType::UInt32;break; case 'L':f.field.type=BoltType::UInt64;break;
        case 'f':f.field.type=BoltType::Float32;break; case 'g':f.field.type=BoltType::Float64;break;
        case 'b':f.field.type=BoltType::Bool;break;
        case 'u':case 'U':f.field.type=BoltType::Utf8;f.large_offsets=fmt[0]=='U';break;
        case 'z':case 'Z':f.field.type=BoltType::Binary;f.large_offsets=fmt[0]=='Z';break;
        default:return Error::Unsupported;
        }
        return Error::Ok;
    }
    if (std::strcmp(fmt,"tdD")==0) { f.field.type=BoltType::Date32;return Error::Ok; }
    if (std::strcmp(fmt,"tdm")==0) { f.field.type=BoltType::Date64;return Error::Ok; }
    const bool timestamp=n==4 && fmt[0]=='t' && fmt[1]=='s' && fmt[3]==':';
    const bool duration=n==3 && fmt[0]=='t' && fmt[1]=='D';
    if (timestamp || duration) {
        switch (fmt[2]) { case 's':f.unit=1;break; case 'm':f.unit=2;break;
        case 'u':f.unit=3;break; case 'n':f.unit=4;break; default:return Error::Unsupported; }
        f.temporal_kind=timestamp?1:2;
        f.field.type=f.unit!=3?BoltType::Int64:(timestamp?BoltType::Timestamp:BoltType::Duration);return Error::Ok;
    }
    if (n<5 || fmt[0]!='d' || fmt[1]!=':') return Error::Unsupported;
    std::uint32_t values[]{0,0,128}, index=0, digits=0;
    for (std::uint32_t i=2;i<n;++i) {
        if (fmt[i]==',') { if (!digits || ++index>2) return Error::Unsupported; values[index]=0;digits=0;continue; }
        if (fmt[i]<'0' || fmt[i]>'9' || values[index]>1000) return Error::Unsupported;
        values[index]=values[index]*10+static_cast<unsigned>(fmt[i]-'0'); ++digits;
    }
    if (!digits || index<1 || values[0]<1 || values[0]>38 || values[1]>values[0] || values[2]!=128) return Error::Unsupported;
    f.field.type=BoltType::Decimal128; f.precision=static_cast<std::uint8_t>(values[0]);
    f.field.decimal_scale=static_cast<std::uint8_t>(values[1]);return Error::Ok;
}
Error schema_field(const ArrowSchema* s,ImportField& f) noexcept {
    assert(f.field.type==BoltType::NA); assert(sizeof(f.field.name)==64);
    if (!s || !s->format || !s->release) return Error::Invalid;
    if (s->dictionary || s->n_children!=0 || s->metadata) return Error::Unsupported;
    std::uint32_t n=0;
    if (s->name && !text_size(s->name,64,n)) return Error::Unsupported;
    if (n) std::memcpy(f.field.name,s->name,n);
    f.field.nullable=(s->flags&2)!=0;
    return field_type(s->format,f);
}
bool variable(const ImportField& f) noexcept { return f.field.type==BoltType::Binary || f.field.type==BoltType::Utf8; }
bool valid_bit(const ArrowArray& a,std::uint64_t bit) noexcept {
    return !a.buffers[0] || (static_cast<const std::uint8_t*>(a.buffers[0])[bit>>3]&(1u<<(bit&7)))!=0;
}
std::int64_t offset_at(const ArrowArray& a,bool large,std::uint64_t i) noexcept {
    std::int64_t result=0;
    if (large) std::memcpy(&result,static_cast<const std::uint8_t*>(a.buffers[1])+i*8,8);
    else { std::int32_t small=0;std::memcpy(&small,static_cast<const std::uint8_t*>(a.buffers[1])+i*4,4);result=small; }
    return result;
}
struct ColumnPlan { std::uint64_t start=0,bytes=0,nulls=0; };
Error plan_column(const ImportField& f,const ArrowArray* a,std::uint64_t parent,
                  std::uint64_t rows,ColumnPlan& p,std::uint64_t& charge_bytes) noexcept {
    assert(rows<=INT32_MAX); assert(f.field.type!=BoltType::NA);
    if (!a || !a->release || a->length<0 || a->offset<0 || a->null_count< -1 || a->null_count>a->length) return Error::Invalid;
    if (a->dictionary || a->n_children!=0) return Error::Unsupported;
    if (parent>static_cast<std::uint64_t>(a->length) || rows>static_cast<std::uint64_t>(a->length)-parent ||
        static_cast<std::uint64_t>(a->offset)>static_cast<std::uint64_t>(INT64_MAX)-parent-rows) return Error::Invalid;
    p.start=static_cast<std::uint64_t>(a->offset)+parent;
    if (a->n_buffers!=(variable(f)?3:2) || !a->buffers) return Error::Invalid;
    if (!a->buffers[0] && a->null_count>0) return Error::Invalid;
    if (a->buffers[0] && !span(a->buffers[0],(p.start+rows+7)/8)) return Error::Invalid;
    for (std::uint64_t i=0;i<rows;++i) p.nulls+=valid_bit(*a,p.start+i)?0u:1u;
    if ((!f.field.nullable && p.nulls) || (a->null_count==0 && p.nulls)) return Error::Invalid;
    if (!charge(charge_bytes,(rows+7)/8,2)) return Error::Limit;
    if (!variable(f)) {
        const auto width=type_size(f.field.type);
        const auto input_bytes=f.field.type==BoltType::Bool ? (p.start+rows+7)/8 : (p.start+rows)*width;
        if ((width && p.start+rows>SIZE_MAX/width) || (rows && !span(a->buffers[1],input_bytes))) return Error::Invalid;
        p.bytes=rows*width;
        return charge(charge_bytes,p.bytes,2)?Error::Ok:Error::Limit;
    }
    const auto width=f.large_offsets?8u:4u;
    if (p.start+rows+1>SIZE_MAX/width || !span(a->buffers[1],(p.start+rows+1)*width)) return Error::Invalid;
    auto prior=offset_at(*a,f.large_offsets,p.start); if (prior<0) return Error::Invalid;
    for (std::uint64_t i=0;i<rows;++i) {
        const auto end=offset_at(*a,f.large_offsets,p.start+i+1);
        if (end<prior) return Error::Invalid;
        if (valid_bit(*a,p.start+i)) { if (static_cast<std::uint64_t>(end-prior)>INT32_MAX-p.bytes) return Error::Limit;p.bytes+=static_cast<std::uint64_t>(end-prior); }
        prior=end;
    }
    if (p.bytes && !span(a->buffers[2],static_cast<std::uint64_t>(prior))) return Error::Invalid;
    if (!charge(charge_bytes,p.bytes,2) || !charge(charge_bytes,(rows+1)*4,2) ||
        !charge(charge_bytes,sizeof(BoltColumn),2)) return Error::Limit;
    if (f.field.type==BoltType::Utf8 && !charge(charge_bytes,rows*sizeof(StringView),2)) return Error::Limit;
    return Error::Ok;
}
Error copy_column(const ImportField& f,const ArrowArray& a,const ColumnPlan& p,
                  std::uint64_t rows,Arena& arena,BoltColumn& out) noexcept {
    assert(rows<=INT32_MAX); assert(p.nulls<=rows);
    auto* bits=arena.allocate_array<std::uint8_t>(static_cast<std::size_t>((rows+7)/8+1));
    if (!bits) return Error::Allocation;
    std::memset(bits,0,static_cast<std::size_t>((rows+7)/8+1));
    for (std::uint64_t i=0;i<rows;++i) if (valid_bit(a,p.start+i)) bits[i>>3]|=static_cast<std::uint8_t>(1u<<(i&7));
    if (!variable(f)) {
        const auto width=type_size(f.field.type);
        auto* data=static_cast<std::uint8_t*>(arena.allocate(static_cast<std::size_t>(p.bytes+1),16));
        if (!data) return Error::Allocation;
        if (f.field.type==BoltType::Bool) {
            const auto* source=static_cast<const std::uint8_t*>(a.buffers[1]);
            for (std::uint64_t i=0;i<rows;++i) data[i]=valid_bit(a,p.start+i)?static_cast<std::uint8_t>((source[(p.start+i)>>3]>>((p.start+i)&7))&1):0;
        } else if (rows) std::memcpy(data,static_cast<const std::uint8_t*>(a.buffers[1])+p.start*width,static_cast<std::size_t>(p.bytes));
        out=BoltColumn::make_flat(data,p.nulls?bits:nullptr,static_cast<std::int64_t>(rows),f.field.type);
    } else {
        auto* data=static_cast<std::uint8_t*>(arena.allocate(static_cast<std::size_t>(p.bytes+1),1));
        auto* offsets=arena.allocate_array<std::int32_t>(static_cast<std::size_t>(rows+1));
        if (!data || !offsets) return Error::Allocation;
        std::uint64_t used=0;offsets[0]=0;
        for (std::uint64_t i=0;i<rows;++i) {
            const auto b=offset_at(a,f.large_offsets,p.start+i),e=offset_at(a,f.large_offsets,p.start+i+1);
            if (valid_bit(a,p.start+i) && e>b) { std::memcpy(data+used,static_cast<const std::uint8_t*>(a.buffers[2])+b,static_cast<std::size_t>(e-b));used+=static_cast<std::uint64_t>(e-b); }
            offsets[i+1]=static_cast<std::int32_t>(used);
        }
        out=f.field.type==BoltType::Utf8 ? (rows?BoltColumn::make_utf8_from_packed(data,p.nulls?bits:nullptr,offsets,static_cast<std::int64_t>(rows),&arena):
            BoltColumn::make_flat(data,nullptr,0,BoltType::Utf8)) :
            BoltColumn::make_var_binary(data,p.nulls?bits:nullptr,offsets,static_cast<std::int64_t>(rows),f.field.type,&arena);
        if (out.type==BoltType::NA) return Error::Allocation;
        out.stats.total_string_bytes=static_cast<std::uint32_t>(used);
    }
    out.decimal_scale=f.field.decimal_scale;out.stats.null_count=static_cast<std::uint32_t>(p.nulls);return Error::Ok;
}
} // namespace
const char* import_error_name(ImportError e) noexcept {
    switch (e) { case Error::Ok:return "Ok";case Error::Invalid:return "Invalid Arrow C Data";
    case Error::Unsupported:return "Unsupported Arrow shape/type";case Error::Limit:return "Arrow import budget exceeded";
    case Error::Allocation:return "Arrow import allocation refused"; } return "Invalid Arrow import error";
}
ImportError import_begin(const ArrowSchema* schema,const ImportLimits& limits,Arena* scratch,ImportState& out) noexcept {
    assert(scratch!=nullptr);assert(kMaxColumns<=UINT32_MAX);
    if (scratch->default_alignment()>64) return Error::Unsupported;
    if (!schema || !schema->release || !schema->format || std::strcmp(schema->format,"+s")!=0 ||
        schema->n_children<=0 || !schema->children || limits.max_rows>INT32_MAX || !limits.max_batches || !limits.max_bytes) return Error::Invalid;
    if (schema->dictionary || schema->metadata) return Error::Unsupported;
    if (static_cast<std::uint64_t>(schema->n_children)>limits.max_columns || schema->n_children>kMaxColumns) return Error::Limit;
    ImportState candidate{};candidate.limits=limits;candidate.scratch=scratch;candidate.columns=static_cast<std::uint32_t>(schema->n_children);
    const auto n=candidate.columns;
    if (!charge(candidate.charged_bytes,sizeof(ImportField)*n) || !charge(candidate.charged_bytes,sizeof(BoltBatch)) ||
        !charge(candidate.charged_bytes,sizeof(BoltColumn)*n,2) ||
        !charge(candidate.charged_bytes,sizeof(BoltField)*n)) return Error::Limit;
    if (candidate.charged_bytes>limits.max_bytes) return Error::Limit;
    candidate.fields=scratch->allocate_array<ImportField>(n);if (!candidate.fields) return Error::Allocation;
    for (std::uint32_t i=0;i<n;++i) {
        candidate.fields[i]={};const auto e=schema_field(schema->children[i],candidate.fields[i]);if (e!=Error::Ok) return e;
        // Finish also allocates an empty column when the stream has no
        // batches. Reserve its sentinel data/bitmap and variable metadata.
        if (!charge(candidate.charged_bytes,1,2) || (variable(candidate.fields[i]) &&
            (!charge(candidate.charged_bytes,4) || !charge(candidate.charged_bytes,sizeof(BoltColumn))))) return Error::Limit;
    }
    if (candidate.charged_bytes>limits.max_bytes) return Error::Limit;
    candidate.error=Error::Ok;out=candidate;return Error::Ok;
}
ImportError import_append(ImportState& state,const ArrowArray* array) noexcept {
    assert(state.scratch!=nullptr);assert(state.columns<=kMaxColumns);
    if (state.finished) return Error::Invalid;
    if (state.error!=Error::Ok) return state.error;
    auto fail=[&](Error e) noexcept { state.error=e;return e; };
    if (!array || !array->release || array->length<0 || array->offset<0 || array->null_count< -1 || array->null_count>array->length ||
        array->n_children!=state.columns || !array->children || array->n_buffers!=1 || !array->buffers) return fail(Error::Invalid);
    if (array->dictionary) return fail(Error::Unsupported);
    const auto rows=static_cast<std::uint64_t>(array->length), parent=static_cast<std::uint64_t>(array->offset);
    if (rows>state.limits.max_rows-state.rows || state.batches>=state.limits.max_batches) return fail(Error::Limit);
    if (parent>static_cast<std::uint64_t>(INT64_MAX)-rows || !span(array->buffers[0],array->buffers[0]?(parent+rows+7)/8:0)) return fail(Error::Invalid);
    if (!array->buffers[0] && array->null_count>0) return fail(Error::Invalid);
    for (std::uint64_t i=0;i<rows;++i) if (!valid_bit(*array,parent+i)) return fail(Error::Unsupported); // No nullable table rows.
    auto charged=state.charged_bytes;
    if (!charge(charged,sizeof(ImportChunk)) || !charge(charged,sizeof(BoltColumn)*state.columns,2) ||
        !charge(charged,sizeof(BoltField)*state.columns)) return fail(Error::Limit);
    if (!charge(charged,sizeof(ColumnPlan)*state.columns) || charged>state.limits.max_bytes) return fail(Error::Limit);
    auto* plans=state.scratch->allocate_array<ColumnPlan>(state.columns);if (!plans) return fail(Error::Allocation);
    for (std::uint32_t i=0;i<state.columns;++i) {
        plans[i]={};const auto e=plan_column(state.fields[i],array->children[i],parent,rows,plans[i],charged);
        if (e!=Error::Ok) return fail(e);
    }
    if (charged>state.limits.max_bytes) return fail(Error::Limit);
    auto* chunk=state.scratch->allocate_array<ImportChunk>(1);if (!chunk) return fail(Error::Allocation);
    chunk->next=nullptr;BoltBatch::init_empty(&chunk->batch);
    if (!BoltBatch::alloc_columns(&chunk->batch,state.scratch,state.columns)) return fail(Error::Allocation);
    chunk->batch.num_rows=static_cast<std::int64_t>(rows);
    for (std::uint32_t i=0;i<state.columns;++i) {
        const auto e=copy_column(state.fields[i],*array->children[i],plans[i],rows,*state.scratch,chunk->batch.columns[0][i]);
        if (e!=Error::Ok) return fail(e);
    }
    if (state.tail) state.tail->next=chunk;else state.head=chunk;
    state.tail=chunk;state.rows+=rows;++state.batches;state.charged_bytes=charged;return Error::Ok;
}
namespace {
Error finish_column(const ImportState& state,std::uint32_t index,Arena& arena,BoltColumn& out) noexcept {
    assert(index<state.columns);assert(state.rows<=INT32_MAX);
    const auto& field=state.fields[index];const auto rows=state.rows;
    auto* bits=arena.allocate_array<std::uint8_t>(static_cast<std::size_t>((rows+7)/8+1));
    if (!bits) return Error::Allocation;
    std::memset(bits,0,static_cast<std::size_t>((rows+7)/8+1));
    std::uint64_t nulls=0,bytes=0;
    auto* chunk=state.head;
    for (std::uint32_t b=0;b<state.batches;++b,chunk=chunk->next) {
        assert(chunk!=nullptr);const auto& col=chunk->batch.col(index);
        bytes+=variable(field)?col.stats.total_string_bytes:static_cast<std::uint64_t>(col.length)*type_size(field.field.type);
    }
    if (bytes>SIZE_MAX-1 || (variable(field) && bytes>INT32_MAX)) return Error::Limit;
    auto* data=static_cast<std::uint8_t*>(arena.allocate(static_cast<std::size_t>(bytes+1),16));
    auto* offsets=variable(field)?arena.allocate_array<std::int32_t>(static_cast<std::size_t>(rows+1)):nullptr;
    if (!data || (variable(field) && !offsets)) return Error::Allocation;
    std::uint64_t row=0,used=0;if (offsets) offsets[0]=0;chunk=state.head;
    for (std::uint32_t b=0;b<state.batches;++b,chunk=chunk->next) {
        const auto& col=chunk->batch.col(index);
        if (!variable(field) && col.length) std::memcpy(data+used,col.data,static_cast<std::size_t>(col.length)*type_size(field.field.type));
        for (std::int64_t r=0;r<col.length;++r,++row) {
            if (!col.is_null(r)) bits[row>>3]|=static_cast<std::uint8_t>(1u<<(row&7));else ++nulls;
            if (variable(field)) {
                const std::uint8_t* source=nullptr;std::int32_t length=0;
                if (!col.is_null(r)) col.utf8_at(r,&source,&length);
                if (length) std::memcpy(data+used,source,static_cast<std::size_t>(length));
                used+=static_cast<std::uint64_t>(length);offsets[row+1]=static_cast<std::int32_t>(used);
            }
        }
        if (!variable(field)) used+=static_cast<std::uint64_t>(col.length)*type_size(field.field.type);
    }
    assert(row==rows && used==bytes);
    out=variable(field) ? (field.field.type==BoltType::Utf8 ? (rows?BoltColumn::make_utf8_from_packed(data,nulls?bits:nullptr,offsets,static_cast<std::int64_t>(rows),&arena):
        BoltColumn::make_flat(data,nullptr,0,BoltType::Utf8)) :
        BoltColumn::make_var_binary(data,nulls?bits:nullptr,offsets,static_cast<std::int64_t>(rows),BoltType::Binary,&arena)) :
        BoltColumn::make_flat(data,nulls?bits:nullptr,static_cast<std::int64_t>(rows),field.field.type);
    if (out.type==BoltType::NA) return Error::Allocation;
    out.decimal_scale=field.field.decimal_scale;out.stats.null_count=static_cast<std::uint32_t>(nulls);
    if (variable(field)) out.stats.total_string_bytes=static_cast<std::uint32_t>(used);
    return Error::Ok;
}
} // namespace
ImportError import_finish(ImportState& state,Arena* arena,BoltBatch& output) noexcept {
    assert(arena!=nullptr);assert(state.scratch!=nullptr);
    if (state.error!=Error::Ok) return state.error;
    if (arena->default_alignment()>64) return Error::Unsupported;
    if (arena==state.scratch || state.finished) return Error::Invalid;
    BoltBatch candidate{};BoltBatch::init_empty(&candidate);
    if (!BoltBatch::alloc_columns(&candidate,arena,state.columns)) return Error::Allocation;
    candidate.num_rows=static_cast<std::int64_t>(state.rows);
    for (std::uint32_t i=0;i<state.columns;++i) {
        const auto e=finish_column(state,i,*arena,candidate.columns[0][i]);if (e!=Error::Ok) return e;
        candidate.columns[1][i]=candidate.columns[0][i];candidate.schema.fields[i]=state.fields[i].field;
    }
    candidate.schema.num_fields=state.columns;
    BoltBatch::init_empty(&output);output.columns[0]=candidate.columns[0];output.columns[1]=candidate.columns[1];
    output.num_rows=candidate.num_rows;output.num_cols=candidate.num_cols;output.schema=candidate.schema;output.arena=arena;
    state.finished=true;return Error::Ok;
}
} // namespace bolt::arrow
