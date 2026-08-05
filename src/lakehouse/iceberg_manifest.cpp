// bolt/lakehouse/iceberg_manifest.cpp — JSON manifest + manifest-list parser.
// See manifest.h for the JSON schema. W4 stand-in for proper Avro reader.
// TODO(W5-avro-nested-reader). Tiger Style: bounded, noexcept.

#include "bolt/lakehouse/iceberg/manifest.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "bolt/parse/bolt_json.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace bj = bolt::parse::json;

namespace {

constexpr uint32_t kIterGuard = 8192u;

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
    if (bj::iter_peek(it) != bj::TokenType::String) {
        skip_value(it); out[0] = '\0'; return false;
    }
    copy_tok(idx, it->cursor, out, cap);
    bj::iter_advance(it);
    return true;
}

void read_stat_entry(const bj::StructuralIndex* idx, bj::Iterator* it,
                     ColumnStatEntry* slot, bool is_lower) noexcept {
    assert(idx != nullptr && it != nullptr && slot != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) {
        skip_value(it); return;
    }
    bj::iter_advance(it);
    char val[kLakeMaxValBytes]; val[0] = '\0';
    int32_t field_id = -1;
    bool have_val = false;
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g++ < kIterGuard) {
        const int32_t key = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key, "field-id") || tok_eq(idx, key, "field_id")) {
            int64_t v = 0; read_int64(it, &v); field_id = static_cast<int32_t>(v);
        } else if (tok_eq(idx, key, "value")) {
            const bj::TokenType t = bj::iter_peek(it);
            if (t == bj::TokenType::String) {
                read_str(idx, it, val, sizeof(val));
                have_val = true;
            } else if (t == bj::TokenType::Int64 ||
                       t == bj::TokenType::Float64) {
                int64_t iv = 0; read_int64(it, &iv);
                std::snprintf(val, sizeof(val), "%lld",
                              static_cast<long long>(iv));
                have_val = true;
            } else {
                skip_value(it);
            }
        } else {
            skip_value(it);
        }
    }
    if (bj::iter_peek(it) == bj::TokenType::EndObject) bj::iter_advance(it);
    slot->field_id = field_id;
    if (have_val) {
        if (is_lower) {
            std::strncpy(slot->lower, val, kLakeMaxValBytes - 1u);
            slot->lower[kLakeMaxValBytes - 1u] = '\0';
            slot->has_lower = true;
            // JSON bounds are text, so the length is just the stored string's.
            slot->lower_len =
                static_cast<uint8_t>(std::strlen(slot->lower));
        } else {
            std::strncpy(slot->upper, val, kLakeMaxValBytes - 1u);
            slot->upper[kLakeMaxValBytes - 1u] = '\0';
            slot->has_upper = true;
            slot->upper_len =
                static_cast<uint8_t>(std::strlen(slot->upper));
        }
    }
}

ColumnStatEntry* upsert_stat(FileStats* stats, int32_t field_id) noexcept {
    assert(stats != nullptr);
    for (uint32_t i = 0; i < stats->n_cols; ++i) {
        if (stats->cols[i].field_id == field_id) return &stats->cols[i];
    }
    if (stats->n_cols >= kIcebergMaxStatCols) return nullptr;
    ColumnStatEntry* slot = &stats->cols[stats->n_cols++];
    std::memset(slot, 0, sizeof(*slot));
    slot->field_id = field_id;
    return slot;
}

void read_stat_array(const bj::StructuralIndex* idx, bj::Iterator* it,
                     FileStats* stats, int kind) noexcept {
    assert(idx != nullptr && it != nullptr && stats != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginArray) { skip_value(it); return; }
    bj::iter_advance(it);
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::BeginObject &&
           g++ < kIterGuard) {
        if (kind < 2) {
            ColumnStatEntry tmp{}; tmp.field_id = -1;
            read_stat_entry(idx, it, &tmp, kind == 0);
            ColumnStatEntry* slot = upsert_stat(stats, tmp.field_id);
            if (slot != nullptr) {
                if (kind == 0 && tmp.has_lower) {
                    std::memcpy(slot->lower, tmp.lower, sizeof(slot->lower));
                    slot->has_lower = true;
                }
                if (kind == 1 && tmp.has_upper) {
                    std::memcpy(slot->upper, tmp.upper, sizeof(slot->upper));
                    slot->has_upper = true;
                }
            }
        } else {
            bj::iter_advance(it);
            int32_t fid = -1; int64_t nc = 0;
            uint32_t g2 = 0;
            while (bj::iter_peek(it) == bj::TokenType::Key &&
                   g2++ < kIterGuard) {
                const int32_t key = it->cursor;
                bj::iter_advance(it);
                if (tok_eq(idx, key, "field-id") ||
                    tok_eq(idx, key, "field_id")) {
                    int64_t v = 0; read_int64(it, &v);
                    fid = static_cast<int32_t>(v);
                } else if (tok_eq(idx, key, "value")) {
                    read_int64(it, &nc);
                } else {
                    skip_value(it);
                }
            }
            if (bj::iter_peek(it) == bj::TokenType::EndObject)
                bj::iter_advance(it);
            ColumnStatEntry* slot = upsert_stat(stats, fid);
            if (slot != nullptr) slot->null_count = nc;
        }
    }
    if (bj::iter_peek(it) == bj::TokenType::EndArray) bj::iter_advance(it);
}

void read_partition_array(const bj::StructuralIndex* idx, bj::Iterator* it,
                          DataFileRef* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginArray) {
        skip_value(it); return;
    }
    bj::iter_advance(it);
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::BeginObject &&
           g++ < kIterGuard) {
        bj::iter_advance(it);
        PartitionValue pv{};
        pv.field_id = -1; pv.is_null = true;
        uint32_t g2 = 0;
        while (bj::iter_peek(it) == bj::TokenType::Key && g2++ < kIterGuard) {
            const int32_t key = it->cursor;
            bj::iter_advance(it);
            if (tok_eq(idx, key, "field-id") ||
                tok_eq(idx, key, "field_id")) {
                int64_t v = 0; read_int64(it, &v);
                pv.field_id = static_cast<int32_t>(v);
            } else if (tok_eq(idx, key, "value")) {
                const bj::TokenType t = bj::iter_peek(it);
                if (t == bj::TokenType::Null) {
                    bj::iter_advance(it); pv.is_null = true;
                } else if (t == bj::TokenType::String) {
                    read_str(idx, it, pv.str, kLakeMaxValBytes);
                    pv.is_str = true; pv.is_null = false;
                } else if (t == bj::TokenType::Int64 ||
                           t == bj::TokenType::Float64) {
                    int64_t iv = 0; read_int64(it, &iv);
                    pv.i64 = iv; pv.is_int = true; pv.is_null = false;
                } else if (t == bj::TokenType::BoolTrue ||
                           t == bj::TokenType::BoolFalse) {
                    pv.i64 = (t == bj::TokenType::BoolTrue) ? 1 : 0;
                    pv.is_int = true; pv.is_null = false;
                    bj::iter_advance(it);
                } else {
                    skip_value(it);
                }
            } else {
                skip_value(it);
            }
        }
        if (bj::iter_peek(it) == bj::TokenType::EndObject)
            bj::iter_advance(it);
        if (out->n_partition < kIcebergMaxPartitionValues) {
            out->partition[out->n_partition++] = pv;
        }
    }
    if (bj::iter_peek(it) == bj::TokenType::EndArray) bj::iter_advance(it);
}

void read_eq_ids(const bj::StructuralIndex* /*idx*/, bj::Iterator* it,
                 DataFileRef* out) noexcept {
    assert(it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginArray) {
        skip_value(it); return;
    }
    bj::iter_advance(it);
    uint32_t g = 0;
    while (g++ < kIterGuard) {
        const bj::TokenType t = bj::iter_peek(it);
        if (t == bj::TokenType::EndArray) { bj::iter_advance(it); break; }
        if (t == bj::TokenType::Int64) {
            int64_t v = 0; bj::iter_int64(it, &v); bj::iter_advance(it);
            if (out->n_equality_ids < 8) {
                out->equality_ids[out->n_equality_ids++] =
                    static_cast<int32_t>(v);
            }
        } else {
            skip_value(it);
        }
    }
}

void parse_data_file(const bj::StructuralIndex* idx, bj::Iterator* it,
                     DataFileRef* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) {
        skip_value(it); return;
    }
    bj::iter_advance(it);
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g++ < kIterGuard) {
        const int32_t key = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key, "content")) {
            int64_t v = 0; read_int64(it, &v);
            switch (v) {
                case 0: out->content = FileContent::kData; break;
                case 1: out->content = FileContent::kEqualityDeletes; break;
                case 2: out->content = FileContent::kPositionDeletes; break;
                default: out->content = FileContent::kData; break;
            }
        } else if (tok_eq(idx, key, "file_path") ||
                   tok_eq(idx, key, "file-path")) {
            read_str(idx, it, out->file_path, kIcebergMaxPath);
        } else if (tok_eq(idx, key, "file_format") ||
                   tok_eq(idx, key, "file-format")) {
            char buf[16]; read_str(idx, it, buf, sizeof(buf));
            (void)buf;
        } else if (tok_eq(idx, key, "partition")) {
            read_partition_array(idx, it, out);
        } else if (tok_eq(idx, key, "record_count") ||
                   tok_eq(idx, key, "record-count")) {
            read_int64(it, &out->stats.record_count);
        } else if (tok_eq(idx, key, "file_size_in_bytes") ||
                   tok_eq(idx, key, "file-size-in-bytes")) {
            read_int64(it, &out->stats.file_size_in_bytes);
        } else if (tok_eq(idx, key, "lower_bounds") ||
                   tok_eq(idx, key, "lower-bounds")) {
            read_stat_array(idx, it, &out->stats, 0);
        } else if (tok_eq(idx, key, "upper_bounds") ||
                   tok_eq(idx, key, "upper-bounds")) {
            read_stat_array(idx, it, &out->stats, 1);
        } else if (tok_eq(idx, key, "null_value_counts") ||
                   tok_eq(idx, key, "null-value-counts")) {
            read_stat_array(idx, it, &out->stats, 2);
        } else if (tok_eq(idx, key, "equality_ids") ||
                   tok_eq(idx, key, "equality-ids")) {
            read_eq_ids(idx, it, out);
        } else {
            skip_value(it);
        }
    }
    if (bj::iter_peek(it) == bj::TokenType::EndObject) bj::iter_advance(it);
}

bool parse_manifest_entry(const bj::StructuralIndex* idx, bj::Iterator* it,
                          DataFileRef* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return false;
    bj::iter_advance(it);
    std::memset(out, 0, sizeof(*out));
    out->status = ManifestStatus::kAdded;
    out->content = FileContent::kData;
    out->partition_spec_id = -1;
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g++ < kIterGuard) {
        const int32_t key = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key, "status")) {
            int64_t v = 0; read_int64(it, &v);
            switch (v) {
                case 0: out->status = ManifestStatus::kExisting; break;
                case 1: out->status = ManifestStatus::kAdded; break;
                case 2: out->status = ManifestStatus::kDeleted; break;
                default: out->status = ManifestStatus::kUnknown; break;
            }
        } else if (tok_eq(idx, key, "snapshot_id") ||
                   tok_eq(idx, key, "snapshot-id")) {
            read_int64(it, &out->snapshot_id);
        } else if (tok_eq(idx, key, "data_file") ||
                   tok_eq(idx, key, "data-file")) {
            parse_data_file(idx, it, out);
        } else {
            skip_value(it);
        }
    }
    if (bj::iter_peek(it) == bj::TokenType::EndObject) bj::iter_advance(it);
    return true;
}

}  // namespace

bool manifest_list_parse_json(const uint8_t* src, uint32_t len, Arena* scratch,
                              ManifestListEntry* out, uint32_t cap,
                              uint32_t* out_n) noexcept {
    assert(src != nullptr && scratch != nullptr && out != nullptr);
    assert(out_n != nullptr && cap > 0u);
    *out_n = 0;
    bj::StructuralIndex idx{};
    if (!bj::build_index(src, static_cast<int32_t>(len), scratch, &idx))
        return false;
    bj::Iterator it{};
    if (!bj::iter_init(&idx, &it)) return false;
    if (bj::iter_peek(&it) != bj::TokenType::BeginArray) return false;
    bj::iter_advance(&it);
    uint32_t g = 0;
    while (bj::iter_peek(&it) == bj::TokenType::BeginObject &&
           g++ < kIterGuard) {
        if (*out_n >= cap) { bj::iter_skip_to_close(&it); break; }
        ManifestListEntry* slot = &out[*out_n];
        std::memset(slot, 0, sizeof(*slot));
        slot->partition_spec_id = -1;
        slot->content = FileContent::kData;
        bj::iter_advance(&it);
        uint32_t g2 = 0;
        while (bj::iter_peek(&it) == bj::TokenType::Key && g2++ < kIterGuard) {
            const int32_t key = it.cursor;
            bj::iter_advance(&it);
            if (tok_eq(&idx, key, "manifest_path") ||
                tok_eq(&idx, key, "manifest-path")) {
                read_str(&idx, &it, slot->manifest_path,
                         kIcebergMaxManifestPath);
            } else if (tok_eq(&idx, key, "manifest_length") ||
                       tok_eq(&idx, key, "manifest-length")) {
                read_int64(&it, &slot->manifest_length);
            } else if (tok_eq(&idx, key, "partition_spec_id") ||
                       tok_eq(&idx, key, "partition-spec-id")) {
                int64_t v = 0; read_int64(&it, &v);
                slot->partition_spec_id = static_cast<int32_t>(v);
            } else if (tok_eq(&idx, key, "content")) {
                int64_t v = 0; read_int64(&it, &v);
                slot->content = (v == 1) ? FileContent::kEqualityDeletes
                                          : FileContent::kData;
            } else if (tok_eq(&idx, key, "added_snapshot_id") ||
                       tok_eq(&idx, key, "added-snapshot-id")) {
                read_int64(&it, &slot->added_snapshot_id);
            } else if (tok_eq(&idx, key, "added_files_count") ||
                       tok_eq(&idx, key, "added-files-count")) {
                read_int64(&it, &slot->added_files_count);
            } else {
                skip_value(&it);
            }
        }
        if (bj::iter_peek(&it) == bj::TokenType::EndObject)
            bj::iter_advance(&it);
        ++*out_n;
    }
    return true;
}

bool manifest_parse_json(const uint8_t* src, uint32_t len, Arena* scratch,
                         int32_t default_spec_id,
                         DataFileRef* out, uint32_t cap,
                         uint32_t* out_n) noexcept {
    assert(src != nullptr && scratch != nullptr && out != nullptr);
    assert(out_n != nullptr && cap > 0u);
    *out_n = 0;
    bj::StructuralIndex idx{};
    if (!bj::build_index(src, static_cast<int32_t>(len), scratch, &idx))
        return false;
    bj::Iterator it{};
    if (!bj::iter_init(&idx, &it)) return false;
    if (bj::iter_peek(&it) != bj::TokenType::BeginArray) return false;
    bj::iter_advance(&it);
    uint32_t g = 0;
    while (bj::iter_peek(&it) == bj::TokenType::BeginObject &&
           g++ < kIterGuard) {
        if (*out_n >= cap) { bj::iter_skip_to_close(&it); break; }
        DataFileRef* slot = &out[*out_n];
        if (!parse_manifest_entry(&idx, &it, slot)) return false;
        if (slot->partition_spec_id < 0) slot->partition_spec_id = default_spec_id;
        ++*out_n;
    }
    return true;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
