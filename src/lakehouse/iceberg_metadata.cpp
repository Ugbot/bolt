// bolt/lakehouse/iceberg_metadata.cpp — parse Iceberg table metadata.json.
//
// v1 / v2 supported on read. We extract: format-version, table-uuid, location,
// last-updated-ms, current-snapshot-id, snapshots[], schemas[]
// (fields[{id,name,type,required}]), current-schema-id, partition-specs[] +
// current-spec-id. v3 extras: ignored. Tiger Style.

#include "bolt/lakehouse/iceberg/metadata.h"

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/parse/bolt_json.h"
#include "bolt/lakehouse/iceberg/transform.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace bj = bolt::parse::json;

namespace {

constexpr uint32_t kIterGuard = 4096u;

inline bool tok_eq(const bj::StructuralIndex* idx, int32_t cur,
                   const char* lit) noexcept {
    assert(idx != nullptr && lit != nullptr);
    if (cur < 0 || cur >= idx->token_count) return false;
    const bj::Token& t = idx->tokens[cur];
    if (t.type != bj::TokenType::Key && t.type != bj::TokenType::String)
        return false;
    const size_t n = std::strlen(lit);
    if (t.length != static_cast<int32_t>(n)) return false;
    return std::memcmp(idx->src + t.start, lit, n) == 0;
}

void copy_tok(const bj::StructuralIndex* idx, int32_t cur, char* dst,
              uint32_t cap) noexcept {
    assert(idx != nullptr && dst != nullptr && cap > 0u);
    const bj::Token& t = idx->tokens[cur];
    const uint32_t n = t.length > static_cast<int32_t>(cap - 1u)
                           ? cap - 1u
                           : static_cast<uint32_t>(t.length);
    if (n > 0) std::memcpy(dst, idx->src + t.start, n);
    dst[n] = '\0';
}

bool skip_value(bj::Iterator* it) noexcept {
    assert(it != nullptr);
    const bj::TokenType t = bj::iter_peek(it);
    if (t == bj::TokenType::BeginObject || t == bj::TokenType::BeginArray) {
        bj::iter_advance(it);
        return bj::iter_skip_to_close(it);
    }
    return bj::iter_advance(it);
}

bool read_int64(bj::Iterator* it, int64_t* out) noexcept {
    assert(it != nullptr && out != nullptr);
    const bj::TokenType t = bj::iter_peek(it);
    if (t == bj::TokenType::Int64) {
        bj::iter_int64(it, out); bj::iter_advance(it); return true;
    }
    if (t == bj::TokenType::Float64) {
        double d = 0.0; bj::iter_float64(it, &d);
        *out = static_cast<int64_t>(d); bj::iter_advance(it); return true;
    }
    skip_value(it); return false;
}

bool read_str(const bj::StructuralIndex* idx, bj::Iterator* it,
              char* out, uint32_t cap) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr && cap > 0u);
    const bj::TokenType t = bj::iter_peek(it);
    if (t != bj::TokenType::String) {
        skip_value(it); out[0] = '\0'; return false;
    }
    copy_tok(idx, it->cursor, out, cap);
    bj::iter_advance(it);
    return true;
}

bool parse_snapshot(const bj::StructuralIndex* idx, bj::Iterator* it,
                    Snapshot* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return false;
    bj::iter_advance(it);
    out->parent_snapshot_id = -1;
    out->sequence_number = 0;
    out->op = SnapshotOp::kUnknown;
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g++ < kIterGuard) {
        const int32_t key = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key, "snapshot-id")) {
            read_int64(it, &out->snapshot_id);
        } else if (tok_eq(idx, key, "parent-snapshot-id")) {
            if (bj::iter_peek(it) == bj::TokenType::Null) {
                bj::iter_advance(it); out->parent_snapshot_id = -1;
            } else {
                read_int64(it, &out->parent_snapshot_id);
            }
        } else if (tok_eq(idx, key, "timestamp-ms")) {
            read_int64(it, &out->timestamp_ms);
        } else if (tok_eq(idx, key, "sequence-number")) {
            read_int64(it, &out->sequence_number);
        } else if (tok_eq(idx, key, "manifest-list")) {
            read_str(idx, it, out->manifest_list, kIcebergMaxManifestPath);
        } else if (tok_eq(idx, key, "summary")) {
            if (bj::iter_peek(it) == bj::TokenType::BeginObject) {
                bj::iter_advance(it);
                uint32_t g2 = 0;
                while (bj::iter_peek(it) == bj::TokenType::Key &&
                       g2++ < kIterGuard) {
                    const int32_t sk = it->cursor;
                    bj::iter_advance(it);
                    if (tok_eq(idx, sk, "operation")) {
                        char op[32];
                        read_str(idx, it, op, sizeof(op));
                        if (std::strcmp(op, "append") == 0)
                            out->op = SnapshotOp::kAppend;
                        else if (std::strcmp(op, "replace") == 0)
                            out->op = SnapshotOp::kReplace;
                        else if (std::strcmp(op, "overwrite") == 0)
                            out->op = SnapshotOp::kOverwrite;
                        else if (std::strcmp(op, "delete") == 0)
                            out->op = SnapshotOp::kDelete;
                    } else {
                        skip_value(it);
                    }
                }
                if (bj::iter_peek(it) == bj::TokenType::EndObject)
                    bj::iter_advance(it);
            } else {
                skip_value(it);
            }
        } else {
            skip_value(it);
        }
    }
    if (bj::iter_peek(it) == bj::TokenType::EndObject) bj::iter_advance(it);
    return true;
}

bool parse_field(const bj::StructuralIndex* idx, bj::Iterator* it,
                 SchemaField* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return false;
    bj::iter_advance(it);
    std::memset(out, 0, sizeof(*out));
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g++ < kIterGuard) {
        const int32_t key = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key, "id")) {
            int64_t v = 0; read_int64(it, &v); out->id = static_cast<int32_t>(v);
        } else if (tok_eq(idx, key, "name")) {
            read_str(idx, it, out->name, kIcebergMaxFieldName);
        } else if (tok_eq(idx, key, "type")) {
            if (bj::iter_peek(it) == bj::TokenType::String) {
                read_str(idx, it, out->type, kIcebergMaxTypeName);
            } else {
                std::strcpy(out->type, "struct");
                skip_value(it);
            }
        } else if (tok_eq(idx, key, "required")) {
            const bj::TokenType t = bj::iter_peek(it);
            out->required = (t == bj::TokenType::BoolTrue);
            bj::iter_advance(it);
        } else {
            skip_value(it);
        }
    }
    if (bj::iter_peek(it) == bj::TokenType::EndObject) bj::iter_advance(it);
    return true;
}

bool parse_schema(const bj::StructuralIndex* idx, bj::Iterator* it,
                  Schema* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return false;
    bj::iter_advance(it);
    std::memset(out, 0, sizeof(*out));
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g++ < kIterGuard) {
        const int32_t key = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key, "schema-id")) {
            int64_t v = 0; read_int64(it, &v);
            out->schema_id = static_cast<int32_t>(v);
        } else if (tok_eq(idx, key, "fields")) {
            if (bj::iter_peek(it) != bj::TokenType::BeginArray) {
                skip_value(it); continue;
            }
            bj::iter_advance(it);
            uint32_t g2 = 0;
            while (bj::iter_peek(it) == bj::TokenType::BeginObject &&
                   g2++ < kIterGuard) {
                if (out->n_fields >= kIcebergMaxFieldsPerSchema) {
                    bj::iter_skip_to_close(it); break;
                }
                parse_field(idx, it, &out->fields[out->n_fields]);
                ++out->n_fields;
            }
            if (bj::iter_peek(it) == bj::TokenType::EndArray)
                bj::iter_advance(it);
        } else {
            skip_value(it);
        }
    }
    if (bj::iter_peek(it) == bj::TokenType::EndObject) bj::iter_advance(it);
    return true;
}

bool parse_pspec_field(const bj::StructuralIndex* idx, bj::Iterator* it,
                       PartitionField* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return false;
    bj::iter_advance(it);
    std::memset(out, 0, sizeof(*out));
    out->transform.kind = TransformKind::kUnknown;
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g++ < kIterGuard) {
        const int32_t key = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key, "source-id")) {
            int64_t v = 0; read_int64(it, &v);
            out->source_id = static_cast<int32_t>(v);
        } else if (tok_eq(idx, key, "field-id")) {
            int64_t v = 0; read_int64(it, &v);
            out->field_id = static_cast<int32_t>(v);
        } else if (tok_eq(idx, key, "name")) {
            read_str(idx, it, out->name, kIcebergMaxFieldName);
        } else if (tok_eq(idx, key, "transform")) {
            char buf[kIcebergMaxFieldName];
            read_str(idx, it, buf, sizeof(buf));
            out->transform = transform_parse(buf,
                static_cast<uint32_t>(std::strlen(buf)));
        } else {
            skip_value(it);
        }
    }
    if (bj::iter_peek(it) == bj::TokenType::EndObject) bj::iter_advance(it);
    return true;
}

bool parse_pspec(const bj::StructuralIndex* idx, bj::Iterator* it,
                 PartitionSpec* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return false;
    bj::iter_advance(it);
    std::memset(out, 0, sizeof(*out));
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g++ < kIterGuard) {
        const int32_t key = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key, "spec-id")) {
            int64_t v = 0; read_int64(it, &v);
            out->spec_id = static_cast<int32_t>(v);
        } else if (tok_eq(idx, key, "fields")) {
            if (bj::iter_peek(it) != bj::TokenType::BeginArray) {
                skip_value(it); continue;
            }
            bj::iter_advance(it);
            uint32_t g2 = 0;
            while (bj::iter_peek(it) == bj::TokenType::BeginObject &&
                   g2++ < kIterGuard) {
                if (out->n_fields >= kIcebergMaxFieldsPerSpec) {
                    bj::iter_skip_to_close(it); break;
                }
                parse_pspec_field(idx, it, &out->fields[out->n_fields]);
                ++out->n_fields;
            }
            if (bj::iter_peek(it) == bj::TokenType::EndArray)
                bj::iter_advance(it);
        } else {
            skip_value(it);
        }
    }
    if (bj::iter_peek(it) == bj::TokenType::EndObject) bj::iter_advance(it);
    return true;
}

}  // namespace

bool metadata_parse(const uint8_t* src, uint32_t len, Arena* scratch,
                    Metadata* out) noexcept {
    assert(src != nullptr && scratch != nullptr && out != nullptr);
    assert(len > 0u && len < (1u << 30));
    std::memset(out, 0, sizeof(*out));
    out->format_version = 0;
    out->current_snapshot_id = -1;
    out->current_schema_id = -1;
    out->current_spec_id = -1;
    out->default_sort_order_id = -1;
    bj::StructuralIndex idx{};
    if (!bj::build_index(src, static_cast<int32_t>(len), scratch, &idx))
        return false;
    bj::Iterator it{};
    if (!bj::iter_init(&idx, &it)) return false;
    if (bj::iter_peek(&it) != bj::TokenType::BeginObject) return false;
    bj::iter_advance(&it);
    uint32_t g = 0;
    while (bj::iter_peek(&it) == bj::TokenType::Key && g++ < kIterGuard) {
        const int32_t key = it.cursor;
        bj::iter_advance(&it);
        if (tok_eq(&idx, key, "format-version")) {
            int64_t v = 0; read_int64(&it, &v);
            out->format_version = static_cast<int32_t>(v);
        } else if (tok_eq(&idx, key, "table-uuid")) {
            read_str(&idx, &it, out->table_uuid, kIcebergMaxUuid);
        } else if (tok_eq(&idx, key, "location")) {
            read_str(&idx, &it, out->location, kIcebergMaxLocation);
        } else if (tok_eq(&idx, key, "last-updated-ms")) {
            read_int64(&it, &out->last_updated_ms);
        } else if (tok_eq(&idx, key, "last-sequence-number")) {
            read_int64(&it, &out->last_sequence_number);
        } else if (tok_eq(&idx, key, "current-snapshot-id")) {
            if (bj::iter_peek(&it) == bj::TokenType::Null) {
                bj::iter_advance(&it); out->current_snapshot_id = -1;
            } else {
                read_int64(&it, &out->current_snapshot_id);
            }
        } else if (tok_eq(&idx, key, "current-schema-id")) {
            int64_t v = 0; read_int64(&it, &v);
            out->current_schema_id = static_cast<int32_t>(v);
        } else if (tok_eq(&idx, key, "default-spec-id")) {
            int64_t v = 0; read_int64(&it, &v);
            out->current_spec_id = static_cast<int32_t>(v);
        } else if (tok_eq(&idx, key, "default-sort-order-id")) {
            int64_t v = 0; read_int64(&it, &v);
            out->default_sort_order_id = static_cast<int32_t>(v);
        } else if (tok_eq(&idx, key, "snapshots")) {
            if (bj::iter_peek(&it) != bj::TokenType::BeginArray) {
                skip_value(&it); continue;
            }
            bj::iter_advance(&it);
            uint32_t g2 = 0;
            while (bj::iter_peek(&it) == bj::TokenType::BeginObject &&
                   g2++ < kIterGuard) {
                if (out->n_snapshots >= kIcebergMaxSnapshots) {
                    bj::iter_skip_to_close(&it); break;
                }
                parse_snapshot(&idx, &it, &out->snapshots[out->n_snapshots]);
                ++out->n_snapshots;
            }
            if (bj::iter_peek(&it) == bj::TokenType::EndArray)
                bj::iter_advance(&it);
        } else if (tok_eq(&idx, key, "schemas")) {
            if (bj::iter_peek(&it) != bj::TokenType::BeginArray) {
                skip_value(&it); continue;
            }
            bj::iter_advance(&it);
            uint32_t g2 = 0;
            while (bj::iter_peek(&it) == bj::TokenType::BeginObject &&
                   g2++ < kIterGuard) {
                if (out->n_schemas >= kIcebergMaxSchemas) {
                    bj::iter_skip_to_close(&it); break;
                }
                parse_schema(&idx, &it, &out->schemas[out->n_schemas]);
                ++out->n_schemas;
            }
            if (bj::iter_peek(&it) == bj::TokenType::EndArray)
                bj::iter_advance(&it);
        } else if (tok_eq(&idx, key, "schema")) {
            if (bj::iter_peek(&it) == bj::TokenType::BeginObject) {
                if (out->n_schemas < kIcebergMaxSchemas) {
                    parse_schema(&idx, &it, &out->schemas[out->n_schemas]);
                    ++out->n_schemas;
                } else {
                    skip_value(&it);
                }
            } else {
                skip_value(&it);
            }
        } else if (tok_eq(&idx, key, "partition-specs")) {
            if (bj::iter_peek(&it) != bj::TokenType::BeginArray) {
                skip_value(&it); continue;
            }
            bj::iter_advance(&it);
            uint32_t g2 = 0;
            while (bj::iter_peek(&it) == bj::TokenType::BeginObject &&
                   g2++ < kIterGuard) {
                if (out->n_specs >= kIcebergMaxSpecs) {
                    bj::iter_skip_to_close(&it); break;
                }
                parse_pspec(&idx, &it, &out->specs[out->n_specs]);
                ++out->n_specs;
            }
            if (bj::iter_peek(&it) == bj::TokenType::EndArray)
                bj::iter_advance(&it);
        } else {
            skip_value(&it);
        }
    }
    if (out->current_schema_id < 0 && out->n_schemas > 0)
        out->current_schema_id = out->schemas[0].schema_id;
    if (out->current_spec_id < 0 && out->n_specs > 0)
        out->current_spec_id = out->specs[0].spec_id;
    return true;
}

const Schema* metadata_current_schema(const Metadata* m) noexcept {
    assert(m != nullptr);
    if (m->n_schemas == 0) return nullptr;
    for (uint32_t i = 0; i < m->n_schemas; ++i) {
        if (m->schemas[i].schema_id == m->current_schema_id)
            return &m->schemas[i];
    }
    return &m->schemas[0];
}

const PartitionSpec* metadata_spec(const Metadata* m, int32_t spec_id) noexcept {
    assert(m != nullptr);
    for (uint32_t i = 0; i < m->n_specs; ++i) {
        if (m->specs[i].spec_id == spec_id) return &m->specs[i];
    }
    if (m->n_specs == 0) return nullptr;
    return &m->specs[0];
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
