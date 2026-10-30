#pragma once
// Shared named schema for nested row values. Metadata is independent of buffers.
#include "bolt/bolt_nested_row.h"
#include "bolt/schema/exact_field.h"

namespace bolt::nested_row {
struct FieldMetadata {
    char name[64]{};
    int32_t field_id = 0;            // 0: no declared identity (synthetic map entry too)
    uint32_t fixed_length = 0;      // declared fixed binary width, including >16
    uint8_t logical_bool = 0;        // Int64 0/1 storage with declared BOOLEAN meaning
    uint8_t precision = 0;          // declared decimal precision, 0: unspecified
    schema::Temporal temporal = schema::Temporal::None;
    schema::Unit unit = schema::Unit::None;
    uint8_t nullable = 1;           // declared presence; storage presence is TypeNode::nullable
};
struct NamedType {
    TypeTree values{};
    FieldMetadata fields[kMaxNodes]{};
};
namespace type_detail {
inline bool decimal(BoltType type) noexcept {
    return type == BoltType::Decimal64 || type == BoltType::Decimal128 || type == BoltType::Decimal256;
}
inline bool leaf(const TypeNode& node) noexcept {
    assert(!is_group(node.type)); assert(kMaxNodes <= UINT16_MAX);
    if (node.n_children != 0 || node.first_child != 0 || node.nullable > 1 ||
        static_cast<uint8_t>(node.logical) > static_cast<uint8_t>(BoltLogical::Interval)) return false;
    if (node.type == BoltType::Utf8 || node.type == BoltType::Binary) return node.fixed_bytes == 0;
    if (node.type == BoltType::NA || node.type == BoltType::Embedding ||
        static_cast<uint8_t>(node.type) >= 64u || node.fixed_bytes == 0 ||
        node.fixed_bytes != bolt::type_size(node.type)) return false;
    if (node.type == BoltType::FixedSizeBinary && (node.fixed_width == 0 || node.fixed_width > 16)) return false;
    if (!decimal(node.type)) return node.scale == 0;
    const uint8_t cap = node.type == BoltType::Decimal64 ? 18 : node.type == BoltType::Decimal128 ? 38 : 76;
    return node.scale <= cap;
}
inline bool walk(const TypeTree& tree, uint32_t index, uint32_t depth,
                 uint8_t* visited, uint32_t* count) noexcept {
    assert(visited != nullptr && count != nullptr); assert(depth <= kMaxDepth + 1u);
    if (index >= tree.n || depth > kMaxDepth || visited[index]) return false;
    visited[index] = 1; ++*count;
    const auto& node = tree.nodes[index];
    if (!is_group(node.type)) return leaf(node);
    if (node.nullable > 1 || node.fixed_bytes != 0 || node.scale != 0 || node.fixed_width != 0 ||
        static_cast<uint8_t>(node.logical) > static_cast<uint8_t>(BoltLogical::Interval)) return false;
    if (node.n_children == 0) return node.type == BoltType::Struct && node.first_child <= tree.n;
    if (node.first_child >= tree.n || node.n_children > tree.n - node.first_child) return false;
    if ((node.type == BoltType::List || node.type == BoltType::Map) && node.n_children != 1) return false;
    if (node.type == BoltType::Map) {
        const auto& entry = tree.nodes[node.first_child];
        if (entry.type != BoltType::Struct || entry.n_children != 2) return false;
    }
    for (uint32_t i = 0; i < node.n_children; ++i)
        if (!walk(tree,node.first_child + i,depth + 1u,visited,count)) return false;
    return true;
}
}  // namespace type_detail

// Rejects cycles, shared/orphan nodes, impossible widths and malformed shapes.
inline bool validate_tree(const TypeTree& tree) noexcept {
    assert(kMaxNodes == 256); assert(kMaxDepth <= kMaxNodes);
    if (tree.n == 0 || tree.n > kMaxNodes) return false;
    uint8_t visited[kMaxNodes]{}; uint32_t count = 0;
    return type_detail::walk(tree,0,0,visited,&count) && count == tree.n;
}
inline bool set_field_name(NamedType* type, uint32_t index, const char* name, uint32_t length) noexcept {
    assert(type != nullptr); assert(name != nullptr || length == 0);
    if (index >= type->values.n || index >= kMaxNodes || length >= sizeof(FieldMetadata::name) ||
        (length != 0 && std::memchr(name,0,length) != nullptr)) return false;
    if (length != 0) std::memcpy(type->fields[index].name,name,length);
    type->fields[index].name[length] = 0;
    return true;
}
inline bool validate_named_type(const NamedType& type) noexcept {
    assert(kMaxNodes <= UINT16_MAX); assert(sizeof(FieldMetadata::name) == 64);
    if (!validate_tree(type.values)) return false;
    for (uint32_t i = 0; i < type.values.n; ++i) {
        const auto& field = type.fields[i]; const auto& node = type.values.nodes[i];
        if (!std::memchr(field.name,0,sizeof(field.name)) || field.field_id < 0 || field.nullable > 1 ||
            (field.nullable && !node.nullable) ||
            field.precision > 76 || field.logical_bool > 1 ||
            (field.logical_bool && node.type != BoltType::Int64 && node.type != BoltType::Bool) ||
            static_cast<uint8_t>(field.temporal) > static_cast<uint8_t>(schema::Temporal::TimestampUtc) ||
            static_cast<uint8_t>(field.unit) > static_cast<uint8_t>(schema::Unit::Day)) return false;
        if (field.precision && (!type_detail::decimal(node.type) || field.precision < node.scale ||
            field.precision > (node.type == BoltType::Decimal64 ? 18 : node.type == BoltType::Decimal128 ? 38 : 76))) return false;
        if (field.fixed_length > INT32_MAX || (field.fixed_length && node.type != BoltType::Binary &&
            node.type != BoltType::FixedSizeBinary) || (field.fixed_length && node.type == BoltType::FixedSizeBinary &&
            field.fixed_length != node.fixed_width)) return false;
        if (field.temporal == schema::Temporal::None ? field.unit != schema::Unit::None :
            field.temporal == schema::Temporal::Date ? field.unit != schema::Unit::Day :
            field.unit == schema::Unit::None || field.unit == schema::Unit::Day) return false;
        if (field.temporal != schema::Temporal::None && node.type != BoltType::Int32 && node.type != BoltType::Int64 &&
            node.type != BoltType::Date32 && node.type != BoltType::Timestamp && node.type != BoltType::Duration) return false;
        for (uint32_t j = 0; field.field_id != 0 && j < i; ++j)
            if (type.fields[j].field_id == field.field_id) return false;
        if (node.type == BoltType::Map) {
            const uint32_t entry = node.first_child;
            if (type.fields[entry].nullable || type.fields[type.values.nodes[entry].first_child].nullable) return false;
        }
        if (node.type != BoltType::Struct) continue;
        for (uint32_t j = 0; j < node.n_children; ++j) {
            const auto* name = type.fields[node.first_child + j].name;
            if (!name[0]) return false;
            for (uint32_t k = 0; k < j; ++k)
                if (!std::strcmp(name,type.fields[node.first_child + k].name)) return false;
        }
    }
    return true;
}
}  // namespace bolt::nested_row
