#pragma once
// Arena-backed row selection for decoded Flat/Nested column trees.
#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include <cassert>
#include <cstring>

namespace bolt::kernels {
namespace nested_gather_detail {
inline constexpr uint32_t kMaxDepth = 32;
inline constexpr uint32_t kMaxChildren = 65536;
inline constexpr uint32_t kMaxRows = INT32_MAX;

inline bool gather(const BoltColumn&, const uint32_t*, uint32_t, Arena*, uint32_t,
                   BoltColumn*) noexcept;
inline bool validity(const BoltColumn& src, const uint32_t* rows, uint32_t count,
                     Arena* arena, BoltColumn* out) noexcept {
    assert(arena != nullptr && out != nullptr);
    assert(rows != nullptr || count == 0);
    out->stats = ColumnStats::make_default();
    out->validity_offset = 0;
    out->validity = nullptr;
    if (src.validity == nullptr || count == 0) { out->stats.all_valid = true; return true; }
    const uint64_t bytes = (static_cast<uint64_t>(count) + 7u) / 8u;
    auto* bits = static_cast<uint8_t*>(arena->allocate_zeroed(bytes));
    if (bits == nullptr) return false;
    uint32_t nulls = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const bool present = !src.is_null(rows[i]);
        bits[i >> 3u] |= static_cast<uint8_t>(static_cast<uint8_t>(present) << (i & 7u));
        nulls += present ? 0u : 1u;
    }
    out->stats.all_valid = nulls == 0;
    out->stats.null_count = nulls;
    out->validity = nulls == 0 ? nullptr : bits;
    return true;
}
inline bool structure(const BoltColumn& src, const uint32_t* rows, uint32_t count,
                      Arena* arena, uint32_t depth, BoltColumn* out) noexcept {
    assert(arena != nullptr && out != nullptr);
    assert(depth <= kMaxDepth);
    const int64_t children = src.child_count();
    if (children < 0 || children > kMaxChildren || src.data == nullptr) return false;
    auto* columns = arena->allocate_array<BoltColumn>(static_cast<size_t>(children > 0 ? children : 1));
    if (columns == nullptr) return false;
    for (int64_t i = 0; i < children; ++i) {
        const auto* child = src.child_at(i);
        if (child == nullptr || child->length != src.length ||
            !gather(*child, rows, count, arena, depth + 1u, &columns[i])) return false;
    }
    out->data = columns;
    return true;
}
inline bool sequence(const BoltColumn& src, const uint32_t* rows, uint32_t count,
                     Arena* arena, uint32_t depth, BoltColumn* out) noexcept {
    assert(arena != nullptr && out != nullptr);
    assert(depth <= kMaxDepth);
    const auto* child = src.list_element();
    const auto* offsets = src.list_offsets();
    if (src.child_count() != 1 || child == nullptr || offsets == nullptr ||
        child->length < 0 || child->length > kMaxRows) return false;
    auto* selected_offsets = arena->allocate_array<int32_t>(static_cast<size_t>(count) + 1u);
    if (selected_offsets == nullptr) return false;
    uint64_t total = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t begin = offsets[rows[i]], end = offsets[rows[i] + 1u];
        if (begin < 0 || end < begin || end > child->length) return false;
        selected_offsets[i] = static_cast<int32_t>(total);
        total += static_cast<uint32_t>(end - begin);
        if (total > kMaxRows) return false;
    }
    selected_offsets[count] = static_cast<int32_t>(total);
    auto* elements = arena->allocate_array<uint32_t>(total + 1u);
    if (elements == nullptr) return false;
    uint32_t at = 0;
    for (uint32_t i = 0; i < count; ++i)
        for (int32_t j = offsets[rows[i]]; j < offsets[rows[i] + 1u]; ++j)
            elements[at++] = static_cast<uint32_t>(j);
    BoltColumn gathered{};
    if (!gather(*child, elements, at, arena, depth + 1u, &gathered)) return false;
    const auto selected = BoltColumn::make_list(&gathered, selected_offsets, count, out->validity, arena);
    if (selected.data == nullptr) return false;
    out->data = selected.data;
    out->dict_child = selected.dict_child;
    return true;
}
inline bool variable(const BoltColumn& src, const uint32_t* rows, uint32_t count,
                     Arena* arena, BoltColumn* out) noexcept {
    assert(arena != nullptr && out != nullptr); assert(rows != nullptr || count == 0);
    if (!src.dict_child || src.dict_child->type != BoltType::Int32 ||
        src.dict_child->format != ColumnFormat::Flat || src.dict_child->type_size_bytes != sizeof(int32_t) || !src.dict_child->data ||
        src.dict_child->length < src.length+1) return false;
    const auto* offsets = static_cast<const int32_t*>(src.dict_child->data);
    auto* selected = arena->allocate_array<int32_t>(static_cast<size_t>(count)+1u);
    if (!selected) return false;
    uint64_t bytes = 0; selected[0] = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (!src.is_null(rows[i])) {
            const auto begin = offsets[rows[i]], end = offsets[rows[i]+1u];
            if (begin < 0 || end < begin || (end != begin && !src.data)) return false;
            bytes += static_cast<uint32_t>(end-begin);
            if (bytes > INT32_MAX) return false;
        }
        selected[i+1u] = static_cast<int32_t>(bytes);
    }
    auto* data = arena->allocate_array<uint8_t>(bytes+1u); if (!data) return false;
    for (uint32_t i = 0; i < count; ++i) {
        const auto length = selected[i+1u]-selected[i];
        if (length) std::memcpy(data+selected[i],static_cast<const uint8_t*>(src.data)+offsets[rows[i]],static_cast<size_t>(length));
    }
    auto packed = BoltColumn::make_var_binary(data,out->validity,selected,count,src.type,arena);
    if (packed.format != ColumnFormat::VarBinary) return false;
    packed.stats = out->stats; packed.logical = src.logical; packed.decimal_scale = src.decimal_scale;
    *out = packed; return true;
}
inline bool gather(const BoltColumn& src, const uint32_t* rows, uint32_t count,
                   Arena* arena, uint32_t depth, BoltColumn* out) noexcept {
    assert(arena != nullptr && out != nullptr);
    assert(rows != nullptr || count == 0);
    if (depth > kMaxDepth || count > kMaxRows || src.length < 0 || src.length > kMaxRows ||
        src.validity_offset < 0 || src.validity_offset > INT64_MAX - src.length) return false;
    for (uint32_t i = 0; i < count; ++i)
        if (rows[i] >= static_cast<uint64_t>(src.length)) return false;
    *out = src;
    out->length = count;
    out->arena = arena;
    if (!validity(src, rows, count, arena, out)) return false;
    if (src.format == ColumnFormat::VarBinary) return variable(src,rows,count,arena,out);
    if (src.format == ColumnFormat::Nested) {
        if (src.type == BoltType::Struct) return structure(src, rows, count, arena, depth, out);
        if (src.type == BoltType::List || src.type == BoltType::Map)
            return sequence(src, rows, count, arena, depth, out);
        return false;
    }
    if (src.format != ColumnFormat::Flat || src.type_size_bytes == 0 ||
        (src.length != 0 && src.data == nullptr)) return false;
    const uint64_t bytes = static_cast<uint64_t>(count) * src.type_size_bytes;
    auto* values = static_cast<uint8_t*>(arena->allocate(bytes + 1u, 8));
    if (values == nullptr) return false;
    for (uint32_t i = 0; i < count; ++i)
        std::memcpy(values + static_cast<size_t>(i) * src.type_size_bytes,
            static_cast<const uint8_t*>(src.data) + static_cast<size_t>(rows[i]) * src.type_size_bytes,
            src.type_size_bytes);
    out->data = values;
    return true;
}
}  // namespace nested_gather_detail

// Supports Flat/VarBinary leaves and Nested struct/list/map. VarBinary bytes
// are packed into arena custody; unselected payloads are never interpreted.
// Arbitrary selection order and duplicates are preserved; zero rows is valid.
// Output publishes only on success. Arena may consume scratch on refusal.
// StringView spill bytes remain borrowed from src: retain its buffers until all
// consumers finish. Child arrays, validity, list offsets and flat slots are new.
// Source and selection must stay stable during the call. This is a trusted
// column-tree API, not raw untrusted-buffer admission.
inline bool nested_gather(const BoltColumn& src, const uint32_t* rows, uint32_t count,
                          Arena* arena, BoltColumn* out) noexcept {
    assert(arena != nullptr);
    assert(out != nullptr);
    if ((rows == nullptr && count != 0) || count > nested_gather_detail::kMaxRows) return false;
    BoltColumn staged{};
    if (!nested_gather_detail::gather(src, rows, count, arena, 0u, &staged)) return false;
    *out = staged;
    return true;
}
}  // namespace bolt::kernels
