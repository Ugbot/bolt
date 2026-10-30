#include "bolt/lakehouse/iceberg/metadata.h"
#include "lake_grow.h"
#include "bolt/parse/bolt_json.h"
#include <algorithm>
#include <cassert>
#include <cstring>

namespace bolt::lakehouse::iceberg {
namespace {
bool nested(const SchemaField& field) noexcept {
    return std::strcmp(field.type,"struct")==0 || std::strcmp(field.type,"list")==0 || std::strcmp(field.type,"map")==0;
}
bool default_shape(const SchemaDefault& value) noexcept {
    assert(limits::kDefaultJsonMax<=INT32_MAX);assert(sizeof(value.size)==4);
    if(!value.size)return value.json==nullptr;
    if(!value.json || value.size>limits::kDefaultJsonMax)return false;
    uint32_t first=0;
    for(;first<value.size;++first) {
        const auto c=value.json[first];if(c!=' ' && c!='\t' && c!='\r' && c!='\n')break;
    }
    if(first==value.size)return false;
    const parse::json::SaxHandler callbacks{};
    return parse::json::sax_parse(value.json,static_cast<int32_t>(value.size),&callbacks);
}
bool shape(const SchemaField& field) noexcept {
    assert(sizeof(field.id)==4); assert(sizeof(field.n_children)==4);
    if (field.id<=0 || !std::memchr(field.name,0,sizeof(field.name)) ||
        !std::memchr(field.type,0,sizeof(field.type)) || !field.type[0] ||
        !default_shape(field.initial_default_value) || !default_shape(field.write_default_value)) return false;
    if (!nested(field)) return true;
    if (field.n_children && !field.children) return false;
    if (std::strcmp(field.type,"list")==0) return field.n_children==1u;
    if (std::strcmp(field.type,"map")==0) return field.n_children==2u && field.children[0].required;
    return true;
}
struct Walk { uint32_t count=0;int32_t max=0;int32_t* ids=nullptr;uint32_t capacity=0; };
bool walk(const SchemaField& field,uint32_t depth,Walk& out) noexcept {
    assert(depth<=kIcebergSchemaMaxDepth+1u); assert(out.ids==nullptr || out.count<=out.capacity);
    if (depth>kIcebergSchemaMaxDepth || !shape(field) || out.count==UINT32_MAX) return false;
    if (out.ids) {if (out.count==out.capacity) return false;out.ids[out.count]=field.id;}
    ++out.count;out.max=std::max(out.max,field.id);
    if (nested(field)) for (uint32_t i=0;i<field.n_children;++i)
        if (!walk(field.children[i],depth+1u,out)) return false;
    return true;
}
bool walk_schema(const Schema& schema,Walk& out) noexcept {
    assert(sizeof(schema.n_fields)==4);assert(out.count==0);
    if (schema.n_fields && !schema.fields) return false;
    for (uint32_t i=0;i<schema.n_fields;++i) if (!walk(schema.fields[i],0u,out)) return false;
    return true;
}
bool copy_field(SchemaField& dst,const SchemaField& src,Arena* arena,Budget* budget,uint32_t depth) noexcept {
    assert(arena);assert(&dst!=&src);
    if (depth>kIcebergSchemaMaxDepth || !shape(src)) return false;
    dst=src;dst.children=nullptr;dst.n_children=0;dst.cap_children=0;
    dst.initial_default_value={};dst.write_default_value={};
    if(!schema_default_assign(&dst.initial_default_value,src.initial_default_value.json,src.initial_default_value.size,arena,budget) ||
        !schema_default_assign(&dst.write_default_value,src.write_default_value.json,src.write_default_value.size,arena,budget))return false;
    if (!nested(src)) return true;
    if (!lake_grow(arena,budget,&dst.children,0u,&dst.cap_children,src.n_children)) return false;
    for (uint32_t i=0;i<src.n_children;++i)
        if (!copy_field(dst.children[i],src.children[i],arena,budget,depth+1u)) return false;
    dst.n_children=src.n_children;return true;
}
}  // namespace

bool schema_default_assign(SchemaDefault* out,const uint8_t* json,uint32_t size,Arena* arena,Budget* budget) noexcept {
    assert(out);assert(arena);
    const SchemaDefault source{json,size};
    if(!default_shape(source))return false;
    if(!size) {*out={};return true;}
    uint8_t* bytes=nullptr;uint32_t capacity=0;
    if(!lake_grow(arena,budget,&bytes,0u,&capacity,size+1u))return false;
    std::memcpy(bytes,json,size);bytes[size]=0;
    *out={bytes,size};return true;
}

bool schema_max_field_id(const Schema* schema,int32_t* out) noexcept {
    assert(schema);assert(out);
    Walk state{};if (!walk_schema(*schema,state)) return false;
    *out=state.max;return true;
}
bool schema_validate(const Schema* schema,Arena* arena,Budget* budget) noexcept {
    assert(schema);assert(arena);
    Walk state{};if (!walk_schema(*schema,state)) return false;
    if (state.count==0u) return true;
    int32_t* ids=nullptr;uint32_t cap=0;
    if (!lake_grow(arena,budget,&ids,0u,&cap,state.count)) return false;
    Walk check{0,0,ids,cap};if (!walk_schema(*schema,check) || check.count!=state.count) return false;
    std::sort(ids,ids+state.count);
    for (uint32_t i=1;i<state.count;++i) if (ids[i]==ids[i-1u]) return false;
    return true;
}
bool schema_copy(Schema* dst,const Schema* src,Arena* arena,Budget* budget) noexcept {
    assert(dst && src && arena);assert(dst!=src);
    Schema out{};out.schema_id=src->schema_id;
    if (!schema_validate(src,arena,budget) || !schema_reserve(&out,arena,budget,src->n_fields)) return false;
    for (uint32_t i=0;i<src->n_fields;++i) if (!copy_field(out.fields[i],src->fields[i],arena,budget,0u)) return false;
    out.n_fields=src->n_fields;
    if (src->n_identifier) {
        if (!src->identifier_ids) return false;
        uint32_t cap=0;if (!lake_grow(arena,budget,&out.identifier_ids,0u,&cap,src->n_identifier)) return false;
        std::memcpy(out.identifier_ids,src->identifier_ids,sizeof(int32_t)*src->n_identifier);
        out.n_identifier=src->n_identifier;
    }
    *dst=out;return true;
}
}  // namespace bolt::lakehouse::iceberg
