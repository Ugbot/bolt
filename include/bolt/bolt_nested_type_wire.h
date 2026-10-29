#pragma once
// BNT1: portable bounded schema bytes for retained/transported nested values.
// Value rows keep their existing codec. No pointers, native padding or arenas
// cross this boundary. Version/unknown members refuse, never infer a schema.
#include "bolt/bolt_nested_type.h"
#include "bolt/wire/byte_reader.h"

namespace bolt::nested_row {
inline constexpr uint32_t kMaxNamedTypeBytes = 8u + kMaxNodes * (25u + 63u);
namespace type_wire_detail {
inline void number(Writer& writer, uint64_t value, uint32_t width) noexcept {
    assert(width > 0 && width <= 8); assert(writer.n <= kMaxNamedTypeBytes);
    uint8_t bytes[8]{};
    for (uint32_t i = 0; i < width; ++i) bytes[i] = static_cast<uint8_t>(value >> (8u*i));
    writer.put(bytes,width);
}
inline void node(Writer& writer, const TypeNode& value, const FieldMetadata& field) noexcept {
    assert(writer.n <= kMaxNamedTypeBytes); assert(std::strlen(field.name) < 64);
    number(writer,static_cast<uint8_t>(value.type),1); number(writer,static_cast<uint8_t>(value.logical),1);
    number(writer,value.nullable,1); number(writer,field.nullable,1);
    number(writer,value.first_child,2); number(writer,value.n_children,2); number(writer,value.fixed_bytes,2);
    number(writer,value.fixed_width,1); number(writer,value.scale,1);
    number(writer,static_cast<uint32_t>(field.field_id),4); number(writer,field.fixed_length,4);
    number(writer,field.precision,1); number(writer,static_cast<uint8_t>(field.temporal),1);
    number(writer,static_cast<uint8_t>(field.unit),1); number(writer,field.logical_bool,1);
    const auto length = static_cast<uint32_t>(std::strlen(field.name));
    number(writer,length,1); writer.put(field.name,length);
}
inline void encode(const NamedType& type, Writer& writer) noexcept {
    assert(type.values.n <= kMaxNodes); assert(type.values.n > 0);
    writer.put("BNT1",4); number(writer,type.values.n,2); number(writer,0,2);
    for (uint32_t i = 0; i < type.values.n; ++i) node(writer,type.values.nodes[i],type.fields[i]);
}
inline bool node(wire::ByteReader& reader, TypeNode& value, FieldMetadata& field) noexcept {
    assert(wire::byte_reader_valid(reader)); assert(reader.size <= kMaxNamedTypeBytes);
    uint8_t type = 0, logical = 0, temporal = 0, unit = 0, length = 0;
    if (!wire::read_le(reader,type) || !wire::read_le(reader,logical) ||
        !wire::read_le(reader,value.nullable) || !wire::read_le(reader,field.nullable) ||
        !wire::read_le(reader,value.first_child) || !wire::read_le(reader,value.n_children) ||
        !wire::read_le(reader,value.fixed_bytes) || !wire::read_le(reader,value.fixed_width) ||
        !wire::read_le(reader,value.scale) || !wire::read_le(reader,field.field_id) ||
        !wire::read_le(reader,field.fixed_length) || !wire::read_le(reader,field.precision) ||
        !wire::read_le(reader,temporal) || !wire::read_le(reader,unit) ||
        !wire::read_le(reader,field.logical_bool) || !wire::read_le(reader,length) || length >= 64) return false;
    wire::ByteView name{};
    if (!wire::read_bytes(reader,length,name) || (length && std::memchr(name.data,0,length))) return false;
    if (length) std::memcpy(field.name,name.data,length);
    field.name[length] = 0;
    value.type = static_cast<BoltType>(type); value.logical = static_cast<BoltLogical>(logical);
    field.temporal = static_cast<schema::Temporal>(temporal); field.unit = static_cast<schema::Unit>(unit);
    return true;
}
}  // namespace type_wire_detail

// Query exact byte count with named_type_wire_size; caller-owned schema, buffer
// and writeback descriptors must be disjoint. Failed encode leaves bytes and
// written unchanged. No schema identity may depend on native struct padding.
inline bool named_type_wire_size(const NamedType& type, uint32_t* size) noexcept {
    assert(size != nullptr); assert(kMaxNamedTypeBytes < UINT16_MAX);
    if (!validate_named_type(type)) return false;
    Writer writer{nullptr,0,0}; type_wire_detail::encode(type,writer);
    if (writer.failed || writer.n > kMaxNamedTypeBytes) return false;
    *size = static_cast<uint32_t>(writer.n); return true;
}
inline bool named_type_wire_encode(const NamedType& type, uint8_t* bytes,
    uint32_t capacity, uint32_t* written) noexcept {
    assert(written != nullptr); assert(kMaxNodes <= UINT16_MAX);
    uint32_t size = 0;
    if (!bytes || !named_type_wire_size(type,&size) || capacity < size) return false;
    Writer writer{bytes,capacity,0}; type_wire_detail::encode(type,writer);
    assert(writer.n == size); assert(!writer.failed);
    *written = size; return true;
}
inline bool named_type_wire_decode(const uint8_t* bytes, uint32_t size, NamedType* out) noexcept {
    assert(out != nullptr); assert(kMaxNodes <= UINT16_MAX);
    if (!bytes || size < 8 || size > kMaxNamedTypeBytes) return false;
    wire::ByteReader reader{bytes,size,0}; wire::ByteView magic{}; uint16_t count = 0, reserved = 0;
    if (!wire::read_bytes(reader,4,magic) || std::memcmp(magic.data,"BNT1",4) ||
        !wire::read_le(reader,count) || !wire::read_le(reader,reserved) || reserved || !count || count > kMaxNodes)
        return false;
    NamedType staged{}; staged.values.n = count;
    for (uint32_t i = 0; i < count; ++i)
        if (!type_wire_detail::node(reader,staged.values.nodes[i],staged.fields[i])) return false;
    if (reader.offset != size || !validate_named_type(staged)) return false;
    *out = staged; return true;
}
}  // namespace bolt::nested_row
