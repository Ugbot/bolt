#include "bolt/lakehouse/iceberg/write_batch.h"
#include "bolt/kernels/bolt_nested_gather.h"
#include "bolt/join/bolt_swiss.h"
#include <cassert>
#include <cstring>
#include <cmath>
#include <limits>
#include "bolt/lakehouse/iceberg/partition_values.h"

namespace bolt::lakehouse::iceberg {
namespace {
using S=JsonValueStatus;
bool same_name(const char* a,const char* b) noexcept {
    assert(a && b);assert(kMaxFieldName>0);
    return std::strcmp(a,b)==0;
}
S omitted(const SchemaField& field,int64_t rows,Arena* arena,BoltColumn* out) noexcept {
    assert(arena && out);assert(rows>=0 && rows<=INT32_MAX);
    static constexpr uint8_t null[]{'n','u','l','l'};
    auto type=field;if(!rows)type.required=false;
    const auto value=field.write_default_value;
    if((value.json==nullptr)!=(value.size==0))return S::Invalid;
    BoltColumn one{};const auto status=json_value_decode(type,value.json?value.json:null,value.json?value.size:4,arena,&one);
    if(status!=S::Ok)return status;
    auto* selection=static_cast<uint32_t*>(arena->allocate_zeroed((static_cast<size_t>(rows)+1u)*sizeof(uint32_t)));
    if(!selection || !kernels::nested_gather(one,selection,static_cast<uint32_t>(rows),arena,out))return S::Capacity;
    return S::Ok;
}
S bind_names(const Schema& schema,const BoltBatch& input,Arena* arena,int32_t* mapping) noexcept {
    assert(arena && mapping);assert(schema.n_fields<=kMaxColumns);
    SwissTable names{};if(!SwissTable::create(&names,schema.n_fields,arena))return S::Capacity;
    auto* next=arena->allocate_array<int32_t>(schema.n_fields);if(!next)return S::Capacity;
    for(uint32_t i=0;i<schema.n_fields;++i) {
        const auto* name=schema.fields[i].name;
        if(!std::memchr(name,0,sizeof(schema.fields[i].name)) || !*name)return S::Invalid;
        const auto hash=static_cast<uint64_t>(hash_bytes(name,static_cast<uint32_t>(std::strlen(name))));
        const auto previous=names.find(hash);auto current=previous;
        for(uint32_t guard=0;current>=0 && guard<i;++guard) {
            if(static_cast<uint32_t>(current)>=i || same_name(name,schema.fields[current].name))return S::Invalid;
            current=next[current];
        }
        if(current>=0)return S::Invalid;
        mapping[i]=-1;next[i]=previous;if(!names.insert(hash,i))return S::Capacity;
    }
    for(uint32_t j=0;j<input.num_cols;++j) {
        const auto* name=input.schema.fields[j].name;
        if(!std::memchr(name,0,kMaxFieldName+1u) || !*name)return S::Invalid;
        auto current=names.find(static_cast<uint64_t>(hash_bytes(name,static_cast<uint32_t>(std::strlen(name)))));bool found=false;
        for(uint32_t guard=0;current>=0 && guard<schema.n_fields;++guard) {
            if(static_cast<uint32_t>(current)>=schema.n_fields)return S::Invalid;
            if(same_name(name,schema.fields[current].name)) {
                if(mapping[current]>=0)return S::Invalid;
                mapping[current]=static_cast<int32_t>(j);found=true;break;
            }
            current=next[current];
        }
        if(!found)return S::Invalid;
    }
    return S::Ok;
}
#include "iceberg_write_value.inc"
} // namespace

JsonValueStatus write_batch_conform(const Schema& schema,const BoltBatch& input,Arena* arena,BoltBatch** out) noexcept {
    assert(arena && out);assert(kMaxColumns<=INT32_MAX);
    if(!schema.n_fields || schema.n_fields>kMaxColumns || !schema.fields || input.num_cols>schema.n_fields ||
        input.schema.num_fields!=input.num_cols || (input.num_cols && (!input.schema.fields ||
        input.read_epoch>1 || !input.columns[input.read_epoch])) || input.num_rows<0 || input.num_rows>INT32_MAX)return S::Invalid;
    if(!schema_validate(&schema,arena,nullptr))return S::Invalid;
    auto* mapping=arena->allocate_array<int32_t>(schema.n_fields);if(!mapping)return S::Capacity;
    const auto bound=bind_names(schema,input,arena,mapping);if(bound!=S::Ok)return bound;
    auto* batch=arena->allocate_array<BoltBatch>(1);if(!batch)return S::Capacity;
    BoltBatch::init_empty(batch);batch->arena=arena;
    if(!BoltBatch::alloc_columns(batch,arena,schema.n_fields))return S::Capacity;
    batch->num_cols=schema.n_fields;batch->num_rows=input.num_rows;
    for(uint32_t i=0;i<schema.n_fields;++i) {
        const auto& field=schema.fields[i];BoltColumn value{};
        if(mapping[i]>=0) {
            value=input.col(static_cast<uint32_t>(mapping[i]));
            if(value.length!=input.num_rows)return S::Invalid;
            if(field.required)for(int64_t r=0;r<value.length;++r)if(value.is_null(r))return S::Invalid;
        } else {const auto status=omitted(field,input.num_rows,arena,&value);if(status!=S::Ok)return status;}
        const auto prepared=write_value_prepare(field,arena,&value,0);if(prepared!=S::Ok)return prepared;
        if(!batch->schema.add_field(field.name,value.type,!field.required))return S::Capacity;
        batch->schema.fields[i].decimal_scale=value.decimal_scale;
        batch->schema.fields[i].fixed_size=value.fixed_width;
        batch->columns[0][i]=value;batch->columns[1][i]=value;
    }
    *out=batch;return S::Ok;
}
} // namespace bolt::lakehouse::iceberg
