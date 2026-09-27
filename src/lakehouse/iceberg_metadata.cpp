// bolt/lakehouse/iceberg_metadata.cpp — parse Iceberg table metadata.json.
//
// v1 / v2 supported on read. We extract: format-version, table-uuid, location,
// last-updated-ms, current-snapshot-id, snapshots[], schemas[]
// (fields[{id,name,type,required}]), current-schema-id, partition-specs[] +
// current-spec-id, snapshot-log[], metadata-log[]. v3 extras: ignored.
// Tiger Style.

#include "bolt/lakehouse/iceberg/metadata.h"

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/parse/bolt_json.h"
#include "bolt/lakehouse/iceberg/transform.h"
#include "lake_grow.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace bj = bolt::parse::json;

namespace {

// Object keys per JSON object. Arrays are bounded by the token count instead:
// every element consumes at least one token, so no array is capped by a guard.
constexpr uint32_t kIterGuard = 4096u;

inline uint32_t array_guard(const bj::StructuralIndex* idx) noexcept {
    assert(idx != nullptr && idx->token_count >= 0);
    return static_cast<uint32_t>(idx->token_count) + 1u;
}

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

// Skip whatever value the iterator is sitting on.
//
// `bj::iter_skip_to_close` counts nesting depth FROM the token under the
// cursor, so it must be called ON the Begin token — and it already handles a
// scalar by advancing once. Advancing past Begin first (what every copy of this
// helper in src/lakehouse used to do) starts the walk at depth 0, so the
// container's OWN close drives depth to -1, the match never fires, and the
// cursor is left INSIDE the value. G2FEAT-125 found it the expensive way: a
// real pyiceberg metadata.json carries `"properties": {...}` ahead of
// `"snapshots"`, so skipping properties stranded the cursor on its inner key,
// the top-level `while (peek == Key)` loop saw EndObject and stopped, and the
// table parsed with ZERO snapshots and format-version 0 — silently, because
// every field before `properties` read back correctly.
bool skip_value(bj::Iterator* it) noexcept {
    assert(it != nullptr);
    assert(it->idx != nullptr);
    return bj::iter_skip_to_close(it);
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
    out->schema_id = 0;   // spec default when a snapshot omits it
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
        } else if (tok_eq(idx, key, "schema-id")) {
            // Read back so an open->modify->persist cycle preserves the schema
            // each historical snapshot was committed under, instead of
            // rewriting them all as the current one (G2ICE-50).
            int64_t v = 0; read_int64(it, &v);
            out->schema_id = static_cast<int32_t>(v);
        } else if (tok_eq(idx, key, "manifest-list")) {
            if (bj::iter_peek(it) == bj::TokenType::String &&
                idx->tokens[it->cursor].length >=
                    static_cast<int32_t>(kIcebergMaxManifestPath))
                return false;   // a cut path names a different object
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
            // A cut name can collide with a sibling: refuse the schema.
            if (bj::iter_peek(it) == bj::TokenType::String &&
                idx->tokens[it->cursor].length >=
                    static_cast<int32_t>(kIcebergMaxFieldName))
                return false;
            read_str(idx, it, out->name, kIcebergMaxFieldName);
        } else if (tok_eq(idx, key, "type")) {
            if (bj::iter_peek(it) == bj::TokenType::String) {
                if (idx->tokens[it->cursor].length >=
                    static_cast<int32_t>(kIcebergMaxTypeName))
                    return false;
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
                  Arena* a, Budget* b, Schema* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    assert(a != nullptr);
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
            const uint32_t guard = array_guard(idx);
            while (bj::iter_peek(it) == bj::TokenType::BeginObject &&
                   g2++ < guard) {
                SchemaField f;
                if (!parse_field(idx, it, &f)) return false;
                if (!schema_push_field(out, a, b, &f)) return false;
            }
            if (bj::iter_peek(it) != bj::TokenType::EndArray) return false;
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
                if (out->n_fields >= kIcebergMaxFieldsPerSpec) return false;
                parse_pspec_field(idx, it, &out->fields[out->n_fields]);
                ++out->n_fields;
            }
            if (bj::iter_peek(it) != bj::TokenType::EndArray) return false;
            bj::iter_advance(it);
        } else {
            skip_value(it);
        }
    }
    if (bj::iter_peek(it) == bj::TokenType::EndObject) bj::iter_advance(it);
    return true;
}

// {"timestamp-ms":T,"snapshot-id":S} or {"timestamp-ms":T,"metadata-file":F}.
bool parse_log_entry(const bj::StructuralIndex* idx, bj::Iterator* it,
                     int64_t* ts, int64_t* snap_id, char* file,
                     uint32_t file_cap) noexcept {
    assert(idx != nullptr && it != nullptr && ts != nullptr);
    assert((snap_id != nullptr) != (file != nullptr));
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) {
        skip_value(it); return false;
    }
    bj::iter_advance(it);
    bool have_ts = false, have_ref = false;
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g++ < kIterGuard) {
        const int32_t key = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key, "timestamp-ms")) {
            have_ts = read_int64(it, ts);
        } else if (snap_id != nullptr && tok_eq(idx, key, "snapshot-id")) {
            have_ref = read_int64(it, snap_id);
        } else if (file != nullptr && tok_eq(idx, key, "metadata-file")) {
            have_ref = read_str(idx, it, file, file_cap);
        } else {
            skip_value(it);
        }
    }
    if (bj::iter_peek(it) == bj::TokenType::EndObject) bj::iter_advance(it);
    return have_ts && have_ref;
}

// Property values are strings; a non-numeric retention value is refused
// rather than read as "no policy".
bool parse_i64_str(const char* v, int64_t* out) noexcept {
    assert(v != nullptr && out != nullptr);
    int64_t x = 0;
    uint32_t d = 0;
    for (; v[d] >= '0' && v[d] <= '9' && d < 18u; ++d) x = x * 10 + (v[d] - '0');
    if (d == 0u || v[d] != '\0') return false;
    *out = x;
    return true;
}

bool parse_properties(const bj::StructuralIndex* idx, bj::Iterator* it,
                      Metadata* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return skip_value(it);
    bj::iter_advance(it);
    const uint32_t guard = array_guard(idx);
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g++ < guard) {
        const int32_t key = it->cursor;
        bj::iter_advance(it);
        const bool age = tok_eq(idx, key, "history.expire.max-snapshot-age-ms");
        const bool keep = tok_eq(idx, key, "history.expire.min-snapshots-to-keep");
        if (!age && !keep) { skip_value(it); continue; }
        char v[32];
        if (!read_str(idx, it, v, sizeof(v))) return false;
        int64_t x = 0;
        if (!parse_i64_str(v, &x)) return false;
        if (age) out->retention_max_age_ms = x;
        else if (x > INT32_MAX) return false;
        else out->retention_min_snapshots = static_cast<int32_t>(x);
    }
    if (bj::iter_peek(it) != bj::TokenType::EndObject) return false;
    bj::iter_advance(it);
    return true;
}

bool parse_snapshot_log(const bj::StructuralIndex* idx, bj::Iterator* it,
                        Metadata* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginArray) {
        skip_value(it); return false;
    }
    bj::iter_advance(it);
    uint32_t g = 0;
    const uint32_t guard = array_guard(idx);
    while (bj::iter_peek(it) == bj::TokenType::BeginObject && g++ < guard) {
        int64_t ts = 0, sid = 0;
        if (parse_log_entry(idx, it, &ts, &sid, nullptr, 0u) &&
            !metadata_snapshot_log_push(out, ts, sid)) {
            return false;
        }
    }
    if (bj::iter_peek(it) != bj::TokenType::EndArray) return false;
    bj::iter_advance(it);
    assert(out->n_snapshot_log <= out->cap_snapshot_log);
    return true;
}

bool parse_metadata_log(const bj::StructuralIndex* idx, bj::Iterator* it,
                        Metadata* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginArray) {
        skip_value(it); return false;
    }
    bj::iter_advance(it);
    uint32_t g = 0;
    const uint32_t guard = array_guard(idx);
    while (bj::iter_peek(it) == bj::TokenType::BeginObject && g++ < guard) {
        MetadataLogEntry e{};
        if (parse_log_entry(idx, it, &e.timestamp_ms, nullptr,
                            e.metadata_file, sizeof(e.metadata_file))) {
            metadata_metadata_log_push(out, &e);
        }
    }
    if (bj::iter_peek(it) != bj::TokenType::EndArray) return false;
    bj::iter_advance(it);
    assert(out->n_metadata_log <= kIcebergMaxMetadataLog);
    return true;
}

}  // namespace

void metadata_init(Metadata* m, Arena* arena, Budget* budget) noexcept {
    assert(m != nullptr && arena != nullptr);
    std::memset(m, 0, sizeof(*m));
    m->arena = arena;
    m->budget = budget;
    m->current_snapshot_id = -1;
    assert(m->n_snapshots == 0 && m->snapshots == nullptr);
}

bool metadata_parse(const uint8_t* src, uint32_t len, Arena* scratch,
                    Metadata* out) noexcept {
    return metadata_parse_budget(src, len, scratch, nullptr, out);
}

bool metadata_parse_budget(const uint8_t* src, uint32_t len, Arena* scratch,
                           Budget* budget, Metadata* out) noexcept {
    assert(src != nullptr && scratch != nullptr && out != nullptr);
    assert(len > 0u && len < (1u << 30));
    metadata_init(out, scratch, budget);
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
    bool saw_snapshot_log = false;
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
            const uint32_t guard = array_guard(&idx);
            while (bj::iter_peek(&it) == bj::TokenType::BeginObject &&
                   g2++ < guard) {
                Snapshot snap;
                std::memset(&snap, 0, sizeof(snap));
                if (!parse_snapshot(&idx, &it, &snap)) return false;
                if (!metadata_push_snapshot(out, &snap)) return false;
            }
            if (bj::iter_peek(&it) != bj::TokenType::EndArray) return false;
            bj::iter_advance(&it);
        } else if (tok_eq(&idx, key, "schemas")) {
            if (bj::iter_peek(&it) != bj::TokenType::BeginArray) {
                skip_value(&it); continue;
            }
            bj::iter_advance(&it);
            uint32_t g2 = 0;
            const uint32_t guard = array_guard(&idx);
            while (bj::iter_peek(&it) == bj::TokenType::BeginObject &&
                   g2++ < guard) {
                Schema* sch = metadata_push_schema(out);
                if (sch == nullptr) return false;
                if (!parse_schema(&idx, &it, scratch, budget, sch)) return false;
            }
            if (bj::iter_peek(&it) != bj::TokenType::EndArray) return false;
            bj::iter_advance(&it);
        } else if (tok_eq(&idx, key, "schema")) {
            if (bj::iter_peek(&it) == bj::TokenType::BeginObject) {
                Schema* sch = metadata_push_schema(out);
                if (sch == nullptr) return false;
                if (!parse_schema(&idx, &it, scratch, budget, sch)) return false;
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
                if (out->n_specs >= kIcebergMaxSpecs) return false;
                if (!parse_pspec(&idx, &it, &out->specs[out->n_specs])) return false;
                ++out->n_specs;
            }
            if (bj::iter_peek(&it) != bj::TokenType::EndArray) return false;
            bj::iter_advance(&it);
        } else if (tok_eq(&idx, key, "properties")) {
            if (!parse_properties(&idx, &it, out)) return false;
        } else if (tok_eq(&idx, key, "snapshot-log")) {
            if (bj::iter_peek(&it) == bj::TokenType::BeginArray) {
                if (!parse_snapshot_log(&idx, &it, out)) return false;
                saw_snapshot_log = true;
            } else {
                skip_value(&it);
            }
        } else if (tok_eq(&idx, key, "metadata-log")) {
            if (bj::iter_peek(&it) == bj::TokenType::BeginArray) {
                if (!parse_metadata_log(&idx, &it, out)) return false;
            } else {
                skip_value(&it);
            }
        } else {
            skip_value(&it);
        }
    }
    // Tables bolt wrote before it emitted a snapshot-log: every bolt snapshot
    // became current when published, so commit order IS the log.
    if (!saw_snapshot_log) {
        for (uint32_t i = 0; i < out->n_snapshots; ++i) {      // bounded
            if (!metadata_snapshot_log_push(out, out->snapshots[i].timestamp_ms,
                                            out->snapshots[i].snapshot_id))
                return false;
        }
    }
    // A current snapshot the list does not carry would scan as an empty table.
    if (out->current_snapshot_id >= 0 &&
        snapshot_by_id(out, out->current_snapshot_id) == nullptr)
        return false;
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

bool metadata_push_snapshot(Metadata* m, const Snapshot* s) noexcept {
    assert(m != nullptr && s != nullptr);
    assert(m->arena != nullptr && "Metadata not bound to an arena");
    if (!lake_grow(m->arena, m->budget, &m->snapshots, m->n_snapshots,
                   &m->cap_snapshots, m->n_snapshots + 1u))
        return false;
    m->snapshots[m->n_snapshots++] = *s;
    assert(m->n_snapshots <= m->cap_snapshots);
    return true;
}

Schema* metadata_push_schema(Metadata* m) noexcept {
    assert(m != nullptr);
    assert(m->arena != nullptr && "Metadata not bound to an arena");
    if (!lake_grow(m->arena, m->budget, &m->schemas, m->n_schemas,
                   &m->cap_schemas, m->n_schemas + 1u))
        return nullptr;
    Schema* s = &m->schemas[m->n_schemas++];
    std::memset(s, 0, sizeof(*s));
    return s;
}

bool schema_reserve(Schema* s, Arena* a, Budget* b, uint32_t n) noexcept {
    assert(s != nullptr && a != nullptr);
    assert(s->n_fields <= s->cap_fields);
    return lake_grow(a, b, &s->fields, s->n_fields, &s->cap_fields, n);
}

bool schema_push_field(Schema* s, Arena* a, Budget* b,
                       const SchemaField* f) noexcept {
    assert(s != nullptr && f != nullptr);
    if (!schema_reserve(s, a, b, s->n_fields + 1u)) return false;
    s->fields[s->n_fields++] = *f;
    assert(s->n_fields <= s->cap_fields);
    return true;
}

bool schema_copy(Schema* dst, const Schema* src, Arena* a, Budget* b) noexcept {
    assert(dst != nullptr && src != nullptr && a != nullptr);
    assert(dst != src);
    Schema out;
    std::memset(&out, 0, sizeof(out));
    out.schema_id = src->schema_id;
    if (!schema_reserve(&out, a, b, src->n_fields)) return false;
    if (src->n_fields != 0u)
        std::memcpy(out.fields, src->fields, sizeof(SchemaField) * src->n_fields);
    out.n_fields = src->n_fields;
    *dst = out;
    return true;
}

bool metadata_snapshot_log_push(Metadata* m, int64_t timestamp_ms,
                                int64_t snapshot_id) noexcept {
    assert(m != nullptr);
    assert(m->arena != nullptr && "Metadata not bound to an arena");
    if (!lake_grow(m->arena, m->budget, &m->snapshot_log, m->n_snapshot_log,
                   &m->cap_snapshot_log, m->n_snapshot_log + 1u))
        return false;
    SnapshotLogEntry& e = m->snapshot_log[m->n_snapshot_log++];
    e.timestamp_ms = timestamp_ms;
    e.snapshot_id  = snapshot_id;
    return true;
}

void metadata_metadata_log_push(Metadata* m,
                                const MetadataLogEntry* e) noexcept {
    assert(m != nullptr && e != nullptr);
    assert(m->n_metadata_log <= kIcebergMaxMetadataLog);
    if (m->n_metadata_log == kIcebergMaxMetadataLog) {
        std::memmove(&m->metadata_log[0], &m->metadata_log[1],
                     sizeof(MetadataLogEntry) * (kIcebergMaxMetadataLog - 1u));
        --m->n_metadata_log;
    }
    MetadataLogEntry& dst = m->metadata_log[m->n_metadata_log++];
    dst.timestamp_ms = e->timestamp_ms;
    std::memcpy(dst.metadata_file, e->metadata_file, sizeof(dst.metadata_file));
    dst.metadata_file[sizeof(dst.metadata_file) - 1u] = '\0';
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
