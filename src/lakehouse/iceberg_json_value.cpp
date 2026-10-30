#include "bolt/lakehouse/iceberg/json_value.h"
#include "bolt/lakehouse/iceberg/default_value.h"
#include "bolt/lakehouse/iceberg/partition_values.h"
#include "bolt/bolt_nested_row.h"
#include "bolt/parse/bolt_json_writer.h"
#include "bolt/kernels/bolt_json_path.h"
#include "bolt/kernels/bolt_decimal.h"
#include "bolt/kernels/bolt_temporal.h"
#include "bolt/kernels/bolt_date.h"
#include "bolt/kernels/bolt_wkt.h"
#include "bolt/join/bolt_swiss.h"
#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace bolt::lakehouse::iceberg {
namespace {
namespace j=bolt::parse::json;
namespace nr=bolt::nested_row;
namespace dec=bolt::kernels::decimal;
using J=j::JsonValue;using S=JsonValueStatus;
struct State {
    Arena* arena;JsonValueLimits limits;S status=S::Invalid;uint32_t values=0;
    bool capacity() noexcept {status=S::Capacity;return false;}
    bool unsupported() noexcept {status=S::Unsupported;return false;}
    bool count() noexcept {if(values==limits.values)return capacity();++values;return true;}
};
bool limits_ok(const JsonValueLimits& limits) noexcept {
    return limits.json_bytes>0 && limits.json_bytes<=INT32_MAX && limits.encoded_bytes>0 &&
        limits.encoded_bytes<=INT32_MAX && limits.values>0 && limits.values<=INT32_MAX;
}
bool is_null(const J* value) noexcept {
    return !value || (value->kind==j::ValueKind::Literal && value->len==4 && !std::memcmp(value->bytes,"null",4));
}
bool raw_string(State& state,const J* value,const uint8_t** bytes,uint32_t* size) noexcept {
    assert(bytes && size);assert(state.arena);
    if(!value || (value->kind!=j::ValueKind::StringVerbatim && value->kind!=j::ValueKind::StringEscaped) || value->len<0)return false;
    if(value->kind==j::ValueKind::StringEscaped) {*bytes=reinterpret_cast<const uint8_t*>(value->bytes);*size=static_cast<uint32_t>(value->len);return true;}
    const auto n=static_cast<uint32_t>(value->len);
    auto* text=state.arena->allocate_array<char>(static_cast<size_t>(n)+3u);
    auto* decoded=state.arena->allocate_array<char>(static_cast<size_t>(n)+1u);
    if(!text || !decoded)return state.capacity();
    text[0]='"';if(n)std::memcpy(text+1,value->bytes,n);text[n+1u]='"';
    const auto got=kernels::json::json_span_unquote({text,n+2u},decoded,n+1u);
    if(got<0)return false;
    *bytes=reinterpret_cast<const uint8_t*>(decoded);*size=static_cast<uint32_t>(got);return true;
}
bool literal_i64(const J* value,int64_t* out) noexcept {
    assert(out);assert(sizeof(*out)==8);
    if(!value || value->kind!=j::ValueKind::Literal || value->len<=0)return false;
    const auto result=std::from_chars(value->bytes,value->bytes+value->len,*out);
    return result.ec==std::errc{} && result.ptr==value->bytes+value->len;
}
int hex_digit(uint8_t c) noexcept {
    return c>='0' && c<='9'?c-'0':c>='a' && c<='f'?c-'a'+10:c>='A' && c<='F'?c-'A'+10:-1;
}
bool hex_bytes(State& state,const uint8_t* text,uint32_t size,bool uuid,const uint8_t** out,uint32_t* length) noexcept {
    assert(out && length);assert(text || !size);
    if((uuid && size!=36) || (!uuid && size%2))return false;
    auto* bytes=state.arena->allocate_array<uint8_t>(static_cast<size_t>(size)/2u+1u);
    if(!bytes)return state.capacity();
    uint32_t at=0;
    for(uint32_t i=0;i<size;) {
        if(uuid && (i==8 || i==13 || i==18 || i==23)) {if(text[i++]!='-')return false;continue;}
        if(i+1u>=size)return false;
        const auto high=hex_digit(text[i]),low=hex_digit(text[i+1]);if(high<0 || low<0)return false;
        bytes[at++]=static_cast<uint8_t>((high<<4)|low);i+=2;
    }
    if(uuid && at!=16)return false;
    *out=bytes;*length=at;return true;
}
#include "iceberg_json_temporal.inc"
#include "iceberg_json_decimal.inc"
#include "iceberg_json_geo.inc"
#include "iceberg_json_scalar.inc"
#include "iceberg_json_decode.inc"
#include "iceberg_json_encode.inc"
#include "iceberg_json_defaults.inc"
#include "iceberg_json_rebind.inc"
#include "iceberg_json_initial.inc"
} // namespace

JsonValueStatus json_value_decode(const SchemaField& field,const uint8_t* json,uint32_t size,
    Arena* arena,BoltColumn* out,const JsonValueLimits& limits) noexcept {
    assert(arena);assert(out);
    if(!json || !size || !limits_ok(limits))return S::Invalid;
    if(size>limits.json_bytes)return S::Capacity;
    const j::SaxHandler syntax{};
    if(!j::sax_parse(json,static_cast<int32_t>(size),&syntax))return S::Invalid;
    State state{arena,limits};j::StructuralIndex index{};J* value=nullptr;
    if(!j::build_index(json,static_cast<int32_t>(size),arena,&index))return S::Capacity;
    if(index.token_count<2)return S::Invalid;
    if(!j::value_from_index(&index,arena,&value))return S::Capacity;
    auto* tree=arena->allocate_array<nr::TypeTree>(1);if(!tree)return S::Capacity;
    *tree={};tree->n=1;
    if(!type_tree(state,field,*tree,0,0))return state.status;
    Schema schema{};schema.fields=const_cast<SchemaField*>(&field);schema.n_fields=1;
    if(!schema_validate(&schema,arena,nullptr))return S::Invalid;
    RowWriter count{state,nullptr,0};if(!write_value(state,field,*tree,0,value,count,0))return state.status;
    auto* encoded=arena->allocate_array<uint8_t>(count.n+1u);if(!encoded)return S::Capacity;
    state.values=0;RowWriter writer{state,encoded,count.n};
    if(!write_value(state,field,*tree,0,value,writer,0) || writer.n!=count.n)return state.status;
    nr::Decoder decoder{};BoltColumn staged{};
    if(!nr::decoder_init(&decoder,tree,arena) || !nr::decoder_push(&decoder,encoded,writer.n) ||
        !nr::decoder_finish(decoder,&staged))return S::Capacity;
    *out=staged;return S::Ok;
}
JsonValueStatus json_value_encode(const SchemaField& field,const BoltColumn& column,int64_t row,
    Arena* arena,const uint8_t** json,uint32_t* size,const JsonValueLimits& limits) noexcept {
    assert(arena);assert(json && size);
    if(!limits_ok(limits) || row<0 || row>=column.length)return S::Invalid;
    State state{arena,limits};J* value=nullptr;
    auto* tree=arena->allocate_array<nr::TypeTree>(1);if(!tree)return S::Capacity;
    *tree={};tree->n=1;if(!type_tree(state,field,*tree,0,0))return state.status;
    Schema schema{};schema.fields=const_cast<SchemaField*>(&field);schema.n_fields=1;
    if(!schema_validate(&schema,arena,nullptr))return S::Invalid;
    if(!emit_value(state,field,column,row,0,&value))return state.status;
    const uint8_t* bytes=nullptr;int32_t length=0;
    if(!j::serialize(value,arena,j::EmitOptions{},&bytes,&length))return S::Capacity;
    if(length<0 || static_cast<uint32_t>(length)>limits.json_bytes)return S::Capacity;
    const j::SaxHandler syntax{};
    if(!j::sax_parse(bytes,length,&syntax))return S::Invalid;
    // Decode validates duplicate map keys and exact canonical leaf semantics.
    BoltColumn check{};const auto valid=json_value_decode(field,bytes,static_cast<uint32_t>(length),arena,&check,limits);
    if(valid!=S::Ok)return valid;
    *json=bytes;*size=static_cast<uint32_t>(length);return S::Ok;
}
JsonValueStatus default_value_decode(const SchemaField& field,DefaultValueMode mode,Arena* arena,
    BoltColumn* out,const JsonValueLimits& limits) noexcept {
    assert(arena && out);assert(kIcebergSchemaMaxDepth<=nr::kMaxDepth);
    if(!limits_ok(limits) || (mode!=DefaultValueMode::Initial && mode!=DefaultValueMode::Write))return S::Invalid;
    Schema schema{};schema.fields=const_cast<SchemaField*>(&field);schema.n_fields=1;
    if(!schema_validate(&schema,arena,nullptr))return S::Invalid;
    State state{arena,limits};
    auto* tree=arena->allocate_array<nr::TypeTree>(1);if(!tree)return S::Capacity;
    *tree={};tree->n=1;if(!type_tree(state,field,*tree,0,0))return state.status;
    DefaultState defaults{state,mode};J* value=nullptr;
    if(!effective_value(defaults,field,nullptr,true,0,&value))return state.status;
    const uint8_t* json=nullptr;int32_t size=0;
    if(!j::serialize(value,arena,j::EmitOptions{},&json,&size) || size<1)return S::Capacity;
    return json_value_decode(field,json,static_cast<uint32_t>(size),arena,out,limits);
}
JsonValueStatus schema_defaults_validate(const Schema& schema,Arena* arena,const JsonValueLimits& limits) noexcept {
    assert(arena);assert(kIcebergSchemaMaxDepth>0);
    if(!limits_ok(limits) || !schema_validate(&schema,arena,nullptr))return S::Invalid;
    State state{arena,limits};
    for(uint32_t i=0;i<schema.n_fields;++i)if(!validate_default_fields(state,schema.fields[i],0))return state.status;
    return S::Ok;
}

JsonValueStatus default_value_rebind(const SchemaField& source,const SchemaField& target,DefaultValueMode mode,
    Arena* arena,SchemaDefault* out,const JsonValueLimits& limits) noexcept {
    assert(arena && out);assert(nr::kMaxNodes>0);
    if(!limits_ok(limits) || (mode!=DefaultValueMode::Initial && mode!=DefaultValueMode::Write))return S::Invalid;
    State state{arena,limits};DefaultState defaults{state,mode};J* raw=nullptr;
    if(!default_json(defaults,source,&raw))return state.status;
    const auto declared=mode==DefaultValueMode::Initial?source.initial_default_value:source.write_default_value;
    if(!declared.json) {*out={};return S::Ok;}
    auto admitted=source;if(is_null(raw))admitted.required=false;
    BoltColumn checked{};const auto valid=default_value_decode(admitted,mode,arena,&checked,limits);
    if(valid!=S::Ok)return valid;
    Schema schema{};schema.fields=const_cast<SchemaField*>(&target);schema.n_fields=1;
    if(!schema_validate(&schema,arena,nullptr))return S::Invalid;
    auto* tree=arena->allocate_array<nr::TypeTree>(1);if(!tree)return S::Capacity;
    *tree={};tree->n=1;if(!type_tree(state,target,*tree,0,0))return state.status;
    if(!rebind_shape(source,target,0))return S::Invalid;
    J* rebound=nullptr;if(!rebind_value(state,source,target,raw,0,&rebound))return state.status;
    const uint8_t* json=nullptr;int32_t size=0;
    if(!j::serialize(rebound,arena,j::EmitOptions{},&json,&size) || size<1 || static_cast<uint32_t>(size)>limits.json_bytes)return S::Capacity;
    *out={json,static_cast<uint32_t>(size)};return S::Ok;
}

JsonValueStatus initial_default_equal(const SchemaField& source,const SchemaField& target,
    Arena* arena,bool* out,const JsonValueLimits& limits) noexcept {
    assert(arena && out);assert(nr::kMaxNodes>0);
    if(!limits_ok(limits) || source.id!=target.id)return S::Invalid;
    State state{arena,limits};const uint8_t* before=nullptr;const uint8_t* after=nullptr;int32_t a=0,b=0;
    if(!initial_canonical(state,source,target,false,&before,&a) || !initial_canonical(state,source,target,true,&after,&b))return state.status;
    *out=a==b && !std::memcmp(before,after,static_cast<size_t>(a));return S::Ok;
}
JsonValueStatus schema_initial_defaults_validate(const Schema& source,const Schema& target,
    Arena* arena,const JsonValueLimits& limits) noexcept {
    assert(arena);assert(nr::kMaxDepth>=kIcebergSchemaMaxDepth);
    auto status=schema_defaults_validate(source,arena,limits);if(status!=S::Ok)return status;
    status=schema_defaults_validate(target,arena,limits);if(status!=S::Ok)return status;
    State state{arena,limits};
    if(!initial_fields(state,source.fields,source.n_fields,target.fields,target.n_fields,0))return state.status;
    return S::Ok;
}
} // namespace bolt::lakehouse::iceberg
