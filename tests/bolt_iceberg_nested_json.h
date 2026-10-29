#pragma once
// Oracle-only renderer: nested names come from the conformed Iceberg schema.
#include "bolt/bolt_column.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include <cstdio>
#include <string>

namespace nested_oracle {
using bolt::BoltColumn;
using bolt::BoltType;
using bolt::lakehouse::iceberg::SchemaField;
inline std::string quote(const uint8_t* text, size_t length) {
    std::string out = "\"";
    for (size_t i = 0; i < length; ++i) {
        const auto c = text[i];
        if (c < 32) { char escaped[7]; std::snprintf(escaped,sizeof(escaped),"\\u%04x",c); out += escaped; }
        else { if (c == '"' || c == '\\') out += '\\'; out += static_cast<char>(c); }
    }
    return out + "\"";
}
inline bool value(const SchemaField&, const BoltColumn&, int64_t, uint32_t, std::string*);
inline bool leaf(const SchemaField& field, const BoltColumn& col, int64_t row, std::string* out) {
    char text[64];
    if (!std::strcmp(field.type,"boolean")) {
        if (col.type_size_bytes != 8 && col.type_size_bytes != 1) return false;
        *out += (col.type_size_bytes == 8 ? static_cast<const int64_t*>(col.data)[row] != 0
                  : static_cast<const uint8_t*>(col.data)[row] != 0) ? "true" : "false";
    } else if (col.type == BoltType::Int32) *out += std::to_string(static_cast<const int32_t*>(col.data)[row]);
    else if (col.type == BoltType::Int64) *out += std::to_string(static_cast<const int64_t*>(col.data)[row]);
    else if (col.type == BoltType::Float64) {
        std::snprintf(text,sizeof(text),"%.17g",static_cast<const double*>(col.data)[row]); *out += text;
    } else if (col.type == BoltType::Utf8 || col.type == BoltType::Binary || col.type == BoltType::FixedSizeBinary) {
        const uint8_t* bytes = nullptr; int32_t length = 0;
        if (col.type == BoltType::FixedSizeBinary) {
            bytes = static_cast<const uint8_t*>(col.data) + row * col.type_size_bytes; length = col.fixed_width;
        } else col.utf8_at(row,&bytes,&length);
        if (length < 0 || (length && bytes == nullptr)) return false;
        if (col.type == BoltType::Utf8) *out += quote(bytes,static_cast<size_t>(length));
        else {
            std::string hex = "\\x";
            for (int32_t i = 0; i < length; ++i) {
                std::snprintf(text,sizeof(text),"%02x",bytes[i]); hex += text;
            }
            *out += quote(reinterpret_cast<const uint8_t*>(hex.data()),hex.size());
        }
    } else return false;
    return true;
}
inline bool value(const SchemaField& field, const BoltColumn& col, int64_t row,
                  uint32_t depth, std::string* out) {
    if (depth > 32 || row < 0 || row >= col.length) return false;
    if (col.is_null(row)) { *out += "null"; return true; }
    if (col.type == BoltType::Struct) {
        if (field.n_children != col.child_count()) return false;
        *out += "{";
        for (uint32_t i = 0; i < field.n_children; ++i) {
            const auto& child = field.children[i];
            *out += (i ? "," : "") + quote(reinterpret_cast<const uint8_t*>(child.name),std::strlen(child.name)) + ":";
            if (!value(child,*col.child_at(i),row,depth+1u,out)) return false;
        }
        *out += "}"; return true;
    }
    if (col.type != BoltType::List && col.type != BoltType::Map) return leaf(field,col,row,out);
    const auto* offsets = col.list_offsets(); const auto* element = col.list_element();
    if (offsets == nullptr || element == nullptr) return false;
    *out += "[";
    for (int32_t i = offsets[row]; i < offsets[row+1]; ++i) {
        if (i != offsets[row]) *out += ",";
        if (col.type == BoltType::List) {
            if (field.n_children != 1 || !value(field.children[0],*element,i,depth+1u,out)) return false;
        } else {
            if (field.n_children != 2 || element->child_count() != 2) return false;
            *out += "[";
            if (!value(field.children[0],*element->child_at(0),i,depth+1u,out)) return false;
            *out += ",";
            if (!value(field.children[1],*element->child_at(1),i,depth+1u,out)) return false;
            *out += "]";
        }
    }
    *out += "]"; return true;
}
}  // namespace nested_oracle
