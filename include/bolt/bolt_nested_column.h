#pragma once
// Typed encoded columns share MarbleDB's existing nested-row bytes. The caller
// retains the separate immutable TypeTree/NamedType; no per-row schema copies.
#include "bolt/bolt_nested_type.h"
#include "bolt/wire/byte_reader.h"

namespace bolt::nested_row {
struct ColumnLimits {
    uint32_t rows = 1u << 20;
    uint32_t encoded_bytes = 64u << 20;
    uint32_t values = 16u << 20; // all decoded nodes, including children under null structs
};
namespace column_detail {
struct Check {
    const TypeTree* tree;
    const NamedType* named;
    uint64_t values = 0;
    uint32_t max_values;
};
inline bool count(Check& check) noexcept {
    if (check.values >= check.max_values) return false;
    ++check.values; return true;
}
inline bool null_children(Check& check, uint32_t index, uint32_t depth) noexcept {
    assert(index < check.tree->n); assert(depth <= kMaxDepth + 1u);
    if (depth > kMaxDepth || !count(check)) return false;
    const auto& node = check.tree->nodes[index];
    if (node.type == BoltType::Struct) for (uint32_t i = 0; i < node.n_children; ++i)
        if (!null_children(check,node.first_child+i,depth+1u)) return false;
    return true;
}
inline bool row(Check& check, wire::ByteReader& reader, uint32_t index, uint32_t depth) noexcept {
    assert(index < check.tree->n); assert(depth <= kMaxDepth + 1u);
    if (depth > kMaxDepth) return false;
    const auto& node = check.tree->nodes[index];
    if (node.nullable) {
        uint8_t present = 0;
        if (!wire::read_le(reader,present) || present > 1) return false;
        if (!present) return (check.named == nullptr || check.named->fields[index].nullable) &&
            null_children(check,index,depth);
    }
    if (!count(check)) return false;
    if (node.type == BoltType::Struct) {
        for (uint32_t i = 0; i < node.n_children; ++i)
            if (!row(check,reader,node.first_child+i,depth+1u)) return false;
        return true;
    }
    if (node.type == BoltType::List || node.type == BoltType::Map) {
        uint32_t length = 0;
        if (!wire::read_le(reader,length) || length > kMaxCount ||
            length > reader.size-reader.offset || length > check.max_values-check.values) return false;
        for (uint32_t i = 0; i < length; ++i)
            if (!row(check,reader,node.first_child,depth+1u)) return false;
        return true;
    }
    uint32_t bytes = node.fixed_bytes;
    if (!bytes && !wire::read_le(reader,bytes)) return false;
    if (check.named != nullptr && check.named->fields[index].fixed_length &&
        (node.type == BoltType::Binary || node.type == BoltType::FixedSizeBinary) &&
        (node.fixed_bytes ? node.fixed_width : bytes) != check.named->fields[index].fixed_length) return false;
    return wire::skip_bytes(reader,bytes);
}
inline bool limits_valid(const ColumnLimits& limits) noexcept {
    return limits.rows <= INT32_MAX && limits.encoded_bytes <= INT32_MAX &&
        limits.encoded_bytes > 0 && limits.values > 0 && limits.values <= INT32_MAX;
}
inline bool offsets_valid(const BoltColumn& column, const TypeTree& tree,
                          const ColumnLimits& limits, const int32_t** offsets) noexcept {
    assert(offsets != nullptr); assert(tree.n > 0);
    if (column.format != ColumnFormat::VarBinary || column.length < 0 ||
        static_cast<uint64_t>(column.length) > limits.rows || column.validity_offset < 0 ||
        column.validity_offset > INT64_MAX-column.length ||
        (column.type != tree.nodes[0].type && column.type != BoltType::Binary) ||
        column.dict_child == nullptr || column.dict_child->type != BoltType::Int32 ||
        column.dict_child->format != ColumnFormat::Flat || column.dict_child->type_size_bytes != sizeof(int32_t) ||
        column.dict_child->length != column.length+1 ||
        column.dict_child->data == nullptr) return false;
    const auto* spans = static_cast<const int32_t*>(column.dict_child->data);
    if (spans[0] != 0) return false;
    for (int64_t i = 0; i < column.length; ++i)
        if (spans[i] < 0 || spans[i+1] < spans[i] || static_cast<uint32_t>(spans[i+1]) > limits.encoded_bytes)
            return false;
    if (spans[column.length] != 0 && column.data == nullptr) return false;
    *offsets = spans; return true;
}
inline bool validate(const TypeTree& tree, const NamedType* named, const BoltColumn& column,
                     const ColumnLimits& limits, const int32_t** offsets) noexcept {
    assert(offsets != nullptr); assert(named == nullptr || &named->values == &tree);
    if (!limits_valid(limits) || !validate_tree(tree) || !is_group(tree.nodes[0].type) ||
        (named != nullptr && !validate_named_type(*named)) || !offsets_valid(column,tree,limits,offsets)) return false;
    Check check{&tree,named,0,limits.values};
    for (int64_t i = 0; i < column.length; ++i) {
        const auto length = static_cast<uint32_t>((*offsets)[i+1]-(*offsets)[i]);
        if (column.is_null(i)) {
            if (length || !tree.nodes[0].nullable || (named != nullptr && !named->fields[0].nullable) ||
                !null_children(check,0,0)) return false;
            continue;
        }
        if (!length) return false;
        wire::ByteReader reader{static_cast<const uint8_t*>(column.data)+(*offsets)[i],length,0};
        if (!row(check,reader,0,0) || reader.offset != length) return false;
    }
    return true;
}
inline bool spans(const BoltColumn& column, const int32_t** offsets) noexcept {
    assert(offsets != nullptr); assert(column.length >= 0);
    const auto* descriptor = column.dict_child;
    if (descriptor == nullptr || descriptor->type != BoltType::Int32 ||
        descriptor->type_size_bytes != sizeof(int32_t) || descriptor->format != ColumnFormat::Flat ||
        descriptor->length != column.length+1 || descriptor->data == nullptr) return false;
    const auto* values = static_cast<const int32_t*>(descriptor->data);
    if (values[0] < 0) return false;
    for (int64_t i = 0; i < column.length; ++i) if (values[i+1] < values[i]) return false;
    *offsets = values; return true;
}
inline bool shape(const BoltColumn& column, uint32_t depth, uint64_t& values,
                  const ColumnLimits& limits) noexcept {
    assert(depth <= kMaxDepth + 1u); assert(values <= limits.values);
    if (depth > kMaxDepth || column.length < 0 || static_cast<uint64_t>(column.length) > limits.values-values ||
        column.validity_offset < 0 || column.validity_offset > INT64_MAX-column.length) return false;
    values += static_cast<uint64_t>(column.length);
    if (column.is_nested()) {
        if (!is_group(column.type) || column.child_count() <= 0 || column.child_count() > kMaxNodes ||
            column.data == nullptr) return false;
        for (int64_t i = 0; i < column.child_count(); ++i) {
            const auto* child = column.child_at(i);
            if (child == nullptr || (column.type == BoltType::Struct && child->length != column.length) ||
                !shape(*child,depth+1u,values,limits)) return false;
        }
        if (column.type == BoltType::Struct) return true;
        const int32_t* offsets = nullptr;
        return column.child_count() == 1 && spans(column,&offsets) &&
            offsets[column.length] <= column.list_element()->length;
    }
    if (column.format != ColumnFormat::Flat && column.format != ColumnFormat::View &&
        column.format != ColumnFormat::VarBinary) return false;
    if (column.format != ColumnFormat::VarBinary && column.length && column.data == nullptr) return false;
    if (!is_var_leaf(column)) return column.format != ColumnFormat::VarBinary && column.type_size_bytes > 0;
    if (column.format == ColumnFormat::VarBinary) {
        const int32_t* offsets = nullptr;
        return spans(column,&offsets) && static_cast<uint32_t>(offsets[column.length]) <= limits.encoded_bytes &&
            (offsets[column.length] == 0 || column.data != nullptr);
    }
    if (column.type_size_bytes != sizeof(StringView)) return false;
    const auto* strings = static_cast<const StringView*>(column.data);
    for (int64_t i = 0; i < column.length; ++i) if (!column.is_null(i))
        if (strings[i].length > INT32_MAX || (strings[i].length > 12u &&
            (column.str_overflow_base == nullptr || strings[i].ref.buf_idx != 0))) return false;
    return true;
}
inline bool layout(const TypeTree& tree, const BoltColumn& column, const ColumnLimits& limits) noexcept {
    assert(tree.n > 0); assert(tree.n <= kMaxNodes);
    uint64_t values = 0;
    if (!shape(column,0,values,limits)) return false;
    TypeTree observed{};
    if (!tree_from_column(column,&observed) || observed.n != tree.n) return false;
    for (uint32_t i = 0; i < tree.n; ++i) {
        const auto& expected = tree.nodes[i]; const auto& actual = observed.nodes[i];
        if (expected.type != actual.type || expected.fixed_bytes != actual.fixed_bytes ||
            expected.first_child != actual.first_child || expected.n_children != actual.n_children ||
            expected.fixed_width != actual.fixed_width || expected.logical != actual.logical ||
            expected.scale != actual.scale) return false;
    }
    return true;
}
inline bool encode(const TypeTree& tree, const NamedType* named, const BoltColumn& source,
                   Arena* arena, const ColumnLimits& limits, BoltColumn* out) noexcept {
    assert(arena != nullptr && out != nullptr); assert(named == nullptr || &named->values == &tree);
    if (!limits_valid(limits) || !validate_tree(tree) || !is_group(tree.nodes[0].type) ||
        (named != nullptr && !validate_named_type(*named)) || source.length < 0 ||
        static_cast<uint64_t>(source.length) > limits.rows || source.validity_offset < 0 ||
        source.validity_offset > INT64_MAX-source.length || !layout(tree,source,limits)) return false;
    auto* offsets = arena->allocate_array<int32_t>(static_cast<size_t>(source.length)+1u);
    auto* validity = static_cast<uint8_t*>(arena->allocate_zeroed((static_cast<uint64_t>(source.length)+7u)/8u+1u));
    if (offsets == nullptr || validity == nullptr) return false;
    uint64_t total = 0; offsets[0] = 0;
    for (int64_t i = 0; i < source.length; ++i) {
        const auto need = source.is_null(i) ? 0u : encode_row(tree,source,i,nullptr,0);
        if ((!source.is_null(i) && !need) || need > limits.encoded_bytes-total) return false;
        total += need; offsets[i+1] = static_cast<int32_t>(total);
    }
    auto* bytes = static_cast<uint8_t*>(arena->allocate(total+1u));
    if (bytes == nullptr) return false;
    for (int64_t i = 0; i < source.length; ++i) if (!source.is_null(i)) {
        const auto need = static_cast<size_t>(offsets[i+1]-offsets[i]);
        if (encode_row(tree,source,i,bytes+offsets[i],need) != need) return false;
        validity[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
    }
    BoltColumn staged = BoltColumn::make_var_binary(bytes,validity,offsets,source.length,tree.nodes[0].type,arena);
    const int32_t* checked = nullptr;
    if (staged.format != ColumnFormat::VarBinary || !validate(tree,named,staged,limits,&checked)) return false;
    *out = staged; return true;
}
inline bool decode(const TypeTree& tree, const NamedType* named, const BoltColumn& source,
                   Arena* arena, const ColumnLimits& limits, BoltColumn* out) noexcept {
    assert(arena != nullptr && out != nullptr); assert(named == nullptr || &named->values == &tree);
    const int32_t* offsets = nullptr;
    if (!validate(tree,named,source,limits,&offsets)) return false;
    Decoder decoder{};
    if (!decoder_init(&decoder,&tree,arena)) return false;
    for (int64_t i = 0; i < source.length; ++i) {
        const auto* bytes = source.is_null(i) ? nullptr : static_cast<const uint8_t*>(source.data)+offsets[i];
        if (!decoder_push(&decoder,bytes,static_cast<size_t>(offsets[i+1]-offsets[i]))) return false;
    }
    BoltColumn staged{};
    if (!decoder_finish(decoder,&staged)) return false;
    *out = staged; return true;
}
}  // namespace column_detail

// Output descriptors publish only on success. Encoded bytes, validity and
// offsets are arena-owned, independent of source buffers. Decode first validates
// every row/count against explicit budgets, then rehydrates the existing codec.
// The caller retains the immutable schema; neither column embeds schema pointers.
inline bool encode_column(const TypeTree& tree, const BoltColumn& source, Arena* arena,
                          const ColumnLimits& limits, BoltColumn* out) noexcept {
    return column_detail::encode(tree,nullptr,source,arena,limits,out);
}
inline bool encode_column(const NamedType& type, const BoltColumn& source, Arena* arena,
                          const ColumnLimits& limits, BoltColumn* out) noexcept {
    return column_detail::encode(type.values,&type,source,arena,limits,out);
}
inline bool decode_column(const TypeTree& tree, const BoltColumn& source, Arena* arena,
                          const ColumnLimits& limits, BoltColumn* out) noexcept {
    return column_detail::decode(tree,nullptr,source,arena,limits,out);
}
inline bool decode_column(const NamedType& type, const BoltColumn& source, Arena* arena,
                          const ColumnLimits& limits, BoltColumn* out) noexcept {
    return column_detail::decode(type.values,&type,source,arena,limits,out);
}
inline bool validate_column(const NamedType& type, const BoltColumn& source,
                            const ColumnLimits& limits) noexcept {
    const int32_t* offsets = nullptr;
    return column_detail::validate(type.values,&type,source,limits,&offsets);
}
}  // namespace bolt::nested_row
