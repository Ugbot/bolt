// bolt/lakehouse/iceberg_manifest.cpp — JSON manifest + manifest-list parser.
// See manifest.h for the JSON schema. W4 stand-in for proper Avro reader.
// TODO(W5-avro-nested-reader). Tiger Style: bounded, noexcept.

#include "bolt/lakehouse/iceberg/manifest.h"

#include <cassert>
#include <charconv>
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

// One {field-id, value} bound into its column's entry, the value kept whole in `a` (G2ICE-336). False = refused
// (iceberg_stat_columns / iceberg_bound_bytes).
bool read_stat_entry(const bj::StructuralIndex* idx, bj::Iterator* it, FileStats* stats, Arena* a,
                     bool is_lower) noexcept {
    assert(idx != nullptr && it != nullptr && stats != nullptr && a != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) { skip_value(it); return true; }
    bj::iter_advance(it);
    char num[32]; num[0] = '\0';
    const char* val = nullptr;
    uint32_t val_len = 0;
    int32_t field_id = -1;
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g++ < kIterGuard) {
        const int32_t key = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key, "field-id") || tok_eq(idx, key, "field_id")) {
            int64_t v = 0; read_int64(it, &v); field_id = static_cast<int32_t>(v);
            continue;
        }
        if (!tok_eq(idx, key, "value")) { skip_value(it); continue; }
        const bj::TokenType t = bj::iter_peek(it);
        if (t == bj::TokenType::String) {
            const bj::Token& tk = idx->tokens[it->cursor];
            val = reinterpret_cast<const char*>(idx->src + tk.start);
            val_len = static_cast<uint32_t>(tk.length);
            bj::iter_advance(it);
        } else if (t == bj::TokenType::Int64 || t == bj::TokenType::Float64) {
            int64_t iv = 0; read_int64(it, &iv);
            const int n = std::snprintf(num, sizeof(num), "%lld", static_cast<long long>(iv));
            val = num;
            val_len = static_cast<uint32_t>(n);
        } else {
            skip_value(it);
        }
    }
    if (bj::iter_peek(it) == bj::TokenType::EndObject) bj::iter_advance(it);
    if (val == nullptr) return true;
    ColumnStatEntry* slot = stats_col(stats, a, field_id);
    return slot != nullptr && stats_set_bound(slot, a, is_lower, val, val_len);
}

bool read_stat_array(const bj::StructuralIndex* idx, bj::Iterator* it,
                     FileStats* stats, Arena* a, int kind) noexcept {
    assert(idx != nullptr && it != nullptr && stats != nullptr && a != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginArray) { skip_value(it); return true; }
    bj::iter_advance(it);
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::BeginObject &&
           g++ < kIterGuard) {
        if (kind < 2) {
            if (!read_stat_entry(idx, it, stats, a, kind == 0)) return false;
            continue;
        }
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
        ColumnStatEntry* slot = stats_col(stats, a, fid);
        if (slot == nullptr) return false;
        slot->null_count = nc;
    }
    if (bj::iter_peek(it) == bj::TokenType::EndArray) bj::iter_advance(it);
    return true;
}

bool read_partition_array(const bj::StructuralIndex* idx, bj::Iterator* it,
                          DataFileRef* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginArray) {
        skip_value(it); return true;
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
                    if (idx->tokens[it->cursor].length >=
                        static_cast<int32_t>(kLakeMaxValBytes))
                        return false;
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
        // Dropping a partition value would prune on the wrong value.
        if (out->n_partition >= kIcebergMaxPartitionValues) return false;
        out->partition[out->n_partition++] = pv;
    }
    if (bj::iter_peek(it) != bj::TokenType::EndArray) return false;
    bj::iter_advance(it);
    assert(out->n_partition <= kIcebergMaxPartitionValues);
    return true;
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

bool read_file_content(const bj::StructuralIndex* idx, bj::Iterator* it,
    FileContent* out) noexcept {
    assert(idx!=nullptr && it!=nullptr && out!=nullptr); assert(it->idx==idx);
    if (bj::iter_peek(it)!=bj::TokenType::Int64) return false;
    const auto& token=idx->tokens[it->cursor];
    if (token.start<0 || token.length<=0 || token.length>idx->src_len ||
        token.start>idx->src_len-token.length) return false;
    const auto* first=reinterpret_cast<const char*>(idx->src+token.start);
    int64_t value=0;
    const auto decoded=std::from_chars(first,first+token.length,value);
    if (decoded.ec!=std::errc{} || decoded.ptr!=first+token.length || value<0 || value>2) return false;
    *out=static_cast<FileContent>(value);
    return bj::iter_advance(it);
}

bool parse_data_file(const bj::StructuralIndex* idx, bj::Iterator* it, Arena* a,
                     DataFileRef* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) {
        skip_value(it); return true;
    }
    bj::iter_advance(it);
    bool content_seen=false;
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g++ < kIterGuard) {
        const int32_t key = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key, "content")) {
            if (content_seen || !read_file_content(idx,it,&out->content)) return false;
            content_seen=true;
        } else if (tok_eq(idx, key, "file_path") ||
                   tok_eq(idx, key, "file-path")) {
            // A cut path names a different (or no) file.
            if (bj::iter_peek(it) == bj::TokenType::String &&
                idx->tokens[it->cursor].length >=
                    static_cast<int32_t>(kIcebergMaxPath))
                return false;
            read_str(idx, it, out->file_path, kIcebergMaxPath);
        } else if (tok_eq(idx, key, "file_format") ||
                   tok_eq(idx, key, "file-format")) {
            char buf[16]; read_str(idx, it, buf, sizeof(buf));
            (void)buf;
        } else if (tok_eq(idx, key, "partition")) {
            if (!read_partition_array(idx, it, out)) return false;
        } else if (tok_eq(idx, key, "record_count") ||
                   tok_eq(idx, key, "record-count")) {
            read_int64(it, &out->stats.record_count);
        } else if (tok_eq(idx, key, "file_size_in_bytes") ||
                   tok_eq(idx, key, "file-size-in-bytes")) {
            read_int64(it, &out->stats.file_size_in_bytes);
        } else if (tok_eq(idx, key, "lower_bounds") ||
                   tok_eq(idx, key, "lower-bounds")) {
            if (!read_stat_array(idx, it, &out->stats, a, 0)) return false;
        } else if (tok_eq(idx, key, "upper_bounds") ||
                   tok_eq(idx, key, "upper-bounds")) {
            if (!read_stat_array(idx, it, &out->stats, a, 1)) return false;
        } else if (tok_eq(idx, key, "null_value_counts") ||
                   tok_eq(idx, key, "null-value-counts")) {
            if (!read_stat_array(idx, it, &out->stats, a, 2)) return false;
        } else if (tok_eq(idx, key, "content_offset") ||
                   tok_eq(idx, key, "content-offset")) {
            if (bj::iter_peek(it) == bj::TokenType::Int64) { read_int64(it, &out->content_offset); out->is_dv = true; }
            else skip_value(it);
        } else if (tok_eq(idx, key, "content_size_in_bytes") ||
                   tok_eq(idx, key, "content-size-in-bytes")) {
            read_int64(it, &out->content_size);
        } else if (tok_eq(idx, key, "referenced_data_file") ||
                   tok_eq(idx, key, "referenced-data-file")) {
            if (bj::iter_peek(it) == bj::TokenType::String &&
                idx->tokens[it->cursor].length >= static_cast<int32_t>(kIcebergMaxPath))
                return false;
            read_str(idx, it, out->referenced_data_file, kIcebergMaxPath);
        } else if (tok_eq(idx, key, "equality_ids") ||
                   tok_eq(idx, key, "equality-ids")) {
            read_eq_ids(idx, it, out);
        } else {
            skip_value(it);
        }
    }
    if (bj::iter_peek(it) == bj::TokenType::EndObject) bj::iter_advance(it);
    return true;
}

bool parse_manifest_entry(const bj::StructuralIndex* idx, bj::Iterator* it,
                          Arena* a, DataFileRef* out) noexcept {
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
        } else if (tok_eq(idx, key, "sequence_number") || tok_eq(idx, key, "sequence-number")) {
            if (bj::iter_peek(it) == bj::TokenType::Int64) { read_int64(it, &out->sequence_number); out->has_sequence_number = true; }
            else skip_value(it);
        } else if (tok_eq(idx, key, "snapshot_id") ||
                   tok_eq(idx, key, "snapshot-id")) {
            read_int64(it, &out->snapshot_id);
        } else if (tok_eq(idx, key, "data_file") ||
                   tok_eq(idx, key, "data-file")) {
            if (!parse_data_file(idx, it, a, out)) return false;
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
        if (*out_n >= cap) return false;   // caller retries with more room
        ManifestListEntry* slot = &out[*out_n];
        std::memset(slot, 0, sizeof(*slot));
        slot->partition_spec_id = -1;
        slot->content = ManifestContent::kDataManifest;
        bj::iter_advance(&it);
        uint32_t g2 = 0;
        while (bj::iter_peek(&it) == bj::TokenType::Key && g2++ < kIterGuard) {
            const int32_t key = it.cursor;
            bj::iter_advance(&it);
            if (tok_eq(&idx, key, "manifest_path") ||
                tok_eq(&idx, key, "manifest-path")) {
                if (bj::iter_peek(&it) == bj::TokenType::String &&
                idx.tokens[it.cursor].length >=
                    static_cast<int32_t>(kIcebergMaxManifestPath))
                return false;
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
                slot->content = (v == 1) ? ManifestContent::kDeleteManifest
                                         : ManifestContent::kDataManifest;
            } else if (tok_eq(&idx, key, "sequence_number") || tok_eq(&idx, key, "sequence-number")) {
                read_int64(&it, &slot->sequence_number);
            } else if (tok_eq(&idx, key, "min_sequence_number") || tok_eq(&idx, key, "min-sequence-number")) {
                read_int64(&it, &slot->min_sequence_number);
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
    return bj::iter_peek(&it) == bj::TokenType::EndArray;
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
        if (*out_n >= cap) return false;   // caller retries with more room
        DataFileRef* slot = &out[*out_n];
        if (!parse_manifest_entry(&idx, &it, scratch, slot)) return false;
        if (slot->partition_spec_id < 0) slot->partition_spec_id = default_spec_id;
        ++*out_n;
    }
    return bj::iter_peek(&it) == bj::TokenType::EndArray;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
