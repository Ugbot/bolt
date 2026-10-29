#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/partition_values.h"
#include "bolt/ingest/bolt_parquet_nested.h"
#include <cassert>
#include <cstring>

namespace bolt::lakehouse::iceberg {
namespace pq=bolt::ingest::parquet;
namespace {
bool temporal(const IcebergTypeInfo& type,const pq::PqColumn& column) noexcept {
    assert(sizeof(column.time_unit)==4);assert(sizeof(type.type)==1);
    const bool time=type.type==IcebergType::kTime;
    const bool nanos=type.type==IcebergType::kTimestampNs || type.type==IcebergType::kTimestampTzNs;
    const bool utc=type.type==IcebergType::kTimestampTz || type.type==IcebergType::kTimestampTzNs;
    return column.physical==pq::PqType::Int64 && column.logical==static_cast<int32_t>(time?pq::PqLogical::Time:pq::PqLogical::Timestamp) &&
        column.time_unit==(nanos?3:2) && (time || column.ts_utc==utc);
}
bool match(const SchemaField& field,const pq::PqMeta& metadata,const pq::PqSchemaTree& tree,int32_t index,uint32_t depth) noexcept {
    assert(depth<=kIcebergSchemaMaxDepth+1u);assert(tree.fields);
    if (depth>kIcebergSchemaMaxDepth || index<0 || static_cast<uint32_t>(index)>=tree.n_fields) return false;
    const auto& physical=tree.fields[index];
    if (!physical.has_field_id || physical.field_id!=field.id || !tree.nodes || physical.node<0 ||
        static_cast<uint32_t>(physical.node)>=tree.n_nodes) return false;
    // Assembled nullable includes an optional ancestor; requiredness belongs
    // to this node's own repetition, not to inherited definition levels.
    if (field.required && tree.nodes[physical.node].repetition==1u) return false;
    if (physical.kind==pq::PqFieldKind::Leaf)
        return physical.column>=0 && static_cast<uint32_t>(physical.column)<metadata.n_columns &&
            metadata.columns[physical.column].has_field_id && metadata.columns[physical.column].field_id==field.id &&
            schema_leaf_matches_parquet(field,metadata.columns[physical.column]);
    if (physical.kind!=pq::PqFieldKind::Struct && physical.kind!=pq::PqFieldKind::List && physical.kind!=pq::PqFieldKind::Map) return false;
    const char* kind=physical.kind==pq::PqFieldKind::Struct?"struct":physical.kind==pq::PqFieldKind::List?"list":"map";
    if (std::strcmp(field.type,kind)!=0 || (field.n_children && !field.children)) return false;
    if ((physical.kind==pq::PqFieldKind::List && field.n_children!=1u) ||
        (physical.kind==pq::PqFieldKind::Map && (field.n_children!=2u || !field.children[0].required))) return false;
    int32_t parent=index;
    if (physical.kind==pq::PqFieldKind::Map) {
        parent=physical.first_child;
        if (parent<0 || static_cast<uint32_t>(parent)>=tree.n_fields || tree.fields[parent].kind!=pq::PqFieldKind::Struct) return false;
    }
    for (uint32_t i=0;i<field.n_children;++i) {
        const auto child=pq::pq_field_find_id(&tree,parent,field.children[i].id);
        if (!match(field.children[i],metadata,tree,child,depth+1u)) return false;
    }
    return true;
}
}  // namespace
bool schema_leaf_matches_parquet(const SchemaField& field,const pq::PqColumn& column) noexcept {
    assert(field.id>0);assert(column.has_field_id);
    IcebergTypeInfo type{};if (!iceberg_type_parse(field.type,&type)) return false;
    const auto logical=static_cast<pq::PqLogical>(column.logical);
    const bool plain=logical==pq::PqLogical::None && column.converted<0;
    const bool integer=plain || (logical==pq::PqLogical::Int && column.int_signed &&
        column.int_bits==(column.physical==pq::PqType::Int64?64:32));
    using T=IcebergType;using P=pq::PqType;
    switch(type.type) {
        case T::kLong:return integer && column.physical==P::Int64;
        case T::kInt:return integer && column.physical==P::Int32;
        case T::kFloat:return plain && column.physical==P::Float;
        case T::kDouble:return plain && column.physical==P::Double;
        case T::kBoolean:return plain && column.physical==P::Boolean;
        case T::kString:return (logical==pq::PqLogical::String || (logical==pq::PqLogical::None && column.converted==0)) && column.physical==P::ByteArray;
        case T::kBinary:return plain && column.physical==P::ByteArray;
        case T::kDate:return logical==pq::PqLogical::Date && column.physical==P::Int32;
        case T::kTime:case T::kTimestamp:case T::kTimestampTz:case T::kTimestampNs:case T::kTimestampTzNs:return temporal(type,column);
        case T::kUuid:return logical==pq::PqLogical::Uuid && column.physical==P::FixedLenByteArray && column.type_length==16;
        case T::kFixed:return plain && column.physical==P::FixedLenByteArray && column.type_length>0 && static_cast<uint32_t>(column.type_length)==type.fixed_len;
        case T::kDecimal:return (logical==pq::PqLogical::Decimal || column.converted==5) && column.precision==type.precision && column.scale==type.scale &&
            ((column.physical==P::Int32 && type.precision<=9u) || (column.physical==P::Int64 && type.precision<=18u) ||
             column.physical==P::ByteArray || (column.physical==P::FixedLenByteArray && column.type_length>0 &&
               static_cast<uint32_t>(column.type_length)>=decimal_required_bytes(type.precision)));
        default:return false;
    }
}
bool schema_field_matches_parquet(const SchemaField& field,const pq::PqMeta* metadata,const pq::PqSchemaTree* tree,int32_t index) noexcept {
    assert(metadata);assert(tree);
    if (!tree->fields || tree->n_columns!=metadata->n_columns) return false;
    return match(field,*metadata,*tree,index,0u);
}
}  // namespace bolt::lakehouse::iceberg
