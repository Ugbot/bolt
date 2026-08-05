// bolt/lakehouse/delta_column_mapping.cpp — physical/logical column map.

#include "bolt/lakehouse/delta/column_mapping.h"

#include <cstring>

#include "bolt/parse/bolt_json.h"

namespace bolt {
namespace lakehouse {
namespace delta {

namespace bj = bolt::parse::json;

namespace {

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
    assert(idx != nullptr && dst != nullptr);
    assert(cap > 0u);
    const bj::Token& t = idx->tokens[cur];
    const uint32_t n = t.length > static_cast<int32_t>(cap - 1u)
                           ? cap - 1u
                           : static_cast<uint32_t>(t.length);
    if (n > 0) std::memcpy(dst, idx->src + t.start, n);
    dst[n] = '\0';
}

// `bj::iter_skip_to_close` counts depth FROM the token under the cursor and
// already advances once for a scalar, so it must be called ON the Begin token.
// Advancing past Begin first leaves the cursor INSIDE the value and silently
// truncates the enclosing parse — see the long note in iceberg_metadata.cpp
// (G2FEAT-125), which is where that bug was finally caught.
bool skip_value(bj::Iterator* it) noexcept {
    assert(it != nullptr);
    assert(it->idx != nullptr);
    return bj::iter_skip_to_close(it);
}

bool parse_field_metadata(const bj::StructuralIndex* idx, bj::Iterator* it,
                          char* physical, uint32_t pcap,
                          int64_t* field_id) noexcept {
    assert(idx != nullptr && it != nullptr && physical != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return skip_value(it);
    bj::iter_advance(it);
    uint32_t g2 = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g2++ < 64u) {
        const int32_t ck = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, ck, "delta.columnMapping.physicalName") &&
            bj::iter_peek(it) == bj::TokenType::String) {
            copy_tok(idx, it->cursor, physical, pcap);
            bj::iter_advance(it);
        } else if (tok_eq(idx, ck, "delta.columnMapping.id") &&
                   bj::iter_peek(it) == bj::TokenType::Int64) {
            bj::iter_int64(it, field_id);
            bj::iter_advance(it);
        } else {
            skip_value(it);
        }
    }
    return bj::iter_advance(it);    // EndObject
}

bool parse_field(const bj::StructuralIndex* idx, bj::Iterator* it,
                 ColumnMap* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return skip_value(it);
    bj::iter_advance(it);
    char logical[kLakeMaxColName] = {0};
    char physical[kLakeMaxColName] = {0};
    int64_t field_id = -1;
    uint32_t guard = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && guard++ < 64u) {
        const int32_t key_cur = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key_cur, "name") &&
            bj::iter_peek(it) == bj::TokenType::String) {
            copy_tok(idx, it->cursor, logical, sizeof(logical));
            bj::iter_advance(it);
        } else if (tok_eq(idx, key_cur, "metadata")) {
            parse_field_metadata(idx, it, physical, sizeof(physical), &field_id);
        } else {
            skip_value(it);
        }
    }
    bj::iter_advance(it);   // EndObject field
    if (logical[0] == '\0') return true;
    if (out->n_entries >= kDeltaMaxMapCols) return false;
    ColumnMapEntry* e = &out->entries[out->n_entries++];
    std::memcpy(e->logical, logical, sizeof(e->logical));
    std::memcpy(e->physical,
                physical[0] == '\0' ? logical : physical,
                sizeof(e->physical));
    e->field_id = field_id;
    return true;
}

}  // namespace

bool delta_column_map_build(const DeltaMetadata* md, Arena* scratch,
                            ColumnMap* out) noexcept {
    assert(md != nullptr && scratch != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    out->mode = md->column_mapping;
    if (md->schema_len == 0) return true;
    bj::StructuralIndex idx{};
    if (!bj::build_index(reinterpret_cast<const uint8_t*>(md->schema_string),
                          static_cast<int32_t>(md->schema_len),
                          scratch, &idx)) {
        return true;
    }
    bj::Iterator it{};
    if (!bj::iter_init(&idx, &it)) return true;
    if (bj::iter_peek(&it) != bj::TokenType::BeginObject) return true;
    bj::iter_advance(&it);
    uint32_t guard = 0;
    while (bj::iter_peek(&it) == bj::TokenType::Key && guard++ < 64u) {
        const int32_t key_cur = it.cursor;
        bj::iter_advance(&it);
        if (tok_eq(&idx, key_cur, "fields") &&
            bj::iter_peek(&it) == bj::TokenType::BeginArray) {
            bj::iter_advance(&it);
            uint32_t g2 = 0;
            while (bj::iter_peek(&it) == bj::TokenType::BeginObject &&
                   g2++ < kDeltaMaxMapCols) {
                if (!parse_field(&idx, &it, out)) return false;
            }
            while (bj::iter_peek(&it) != bj::TokenType::EndArray &&
                   g2++ < kDeltaMaxMapCols * 2u) {
                if (!skip_value(&it)) break;
            }
            bj::iter_advance(&it);
        } else {
            skip_value(&it);
        }
    }
    return true;
}

const char* delta_column_map_logical(const ColumnMap* m,
                                     const char* physical) noexcept {
    assert(m != nullptr && physical != nullptr);
    for (uint32_t i = 0; i < m->n_entries; ++i) {
        if (std::strcmp(m->entries[i].physical, physical) == 0)
            return m->entries[i].logical;
    }
    return nullptr;
}

const char* delta_column_map_physical(const ColumnMap* m,
                                      const char* logical) noexcept {
    assert(m != nullptr && logical != nullptr);
    for (uint32_t i = 0; i < m->n_entries; ++i) {
        if (std::strcmp(m->entries[i].logical, logical) == 0)
            return m->entries[i].physical;
    }
    return nullptr;
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
