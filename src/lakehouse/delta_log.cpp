// bolt/lakehouse/delta_log.cpp — _delta_log NDJSON parser via fionn.
// Tiger Style: PODs, fixed caps, ≥2 asserts/fn, ≤70 line fns, no exceptions,
// all alloc via Arena.

#include "bolt/lakehouse/delta/log.h"

#include <cstdio>
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

inline void copy_token(const bj::StructuralIndex* idx, int32_t cursor,
                       char* dst, uint32_t cap) noexcept {
    assert(idx != nullptr && dst != nullptr);
    assert(cap > 0u);
    const bj::Token& t = idx->tokens[cursor];
    const uint32_t n = t.length > static_cast<int32_t>(cap - 1u)
                           ? cap - 1u
                           : static_cast<uint32_t>(t.length);
    if (n > 0) std::memcpy(dst, idx->src + t.start, n);
    dst[n] = '\0';
}

void str_copy_cstr(char* dst, uint32_t cap, const char* src) noexcept {
    assert(dst != nullptr && cap > 0u);
    if (src == nullptr) { dst[0] = '\0'; return; }
    const size_t n = std::strlen(src);
    const size_t m = n > cap - 1u ? cap - 1u : n;
    if (m > 0) std::memcpy(dst, src, m);
    dst[m] = '\0';
}

bool iter_skip_value(bj::Iterator* it) noexcept {
    assert(it != nullptr);
    const bj::TokenType t = bj::iter_peek(it);
    if (t == bj::TokenType::BeginObject || t == bj::TokenType::BeginArray) {
        bj::iter_advance(it);
        return bj::iter_skip_to_close(it);
    }
    return bj::iter_advance(it);
}

bool eat_string(bj::Iterator* it, char* out, uint32_t cap) noexcept {
    assert(it != nullptr && out != nullptr);
    out[0] = '\0';
    if (bj::iter_peek(it) != bj::TokenType::String) {
        iter_skip_value(it);
        return false;
    }
    copy_token(it->idx, it->cursor, out, cap);
    return bj::iter_advance(it);
}

bool eat_int(bj::Iterator* it, int64_t* out) noexcept {
    assert(it != nullptr && out != nullptr);
    *out = 0;
    const bj::TokenType t = bj::iter_peek(it);
    if (t == bj::TokenType::Int64) {
        if (!bj::iter_int64(it, out)) return false;
        return bj::iter_advance(it);
    }
    if (t == bj::TokenType::Float64) {
        double d = 0.0;
        if (!bj::iter_float64(it, &d)) return false;
        *out = static_cast<int64_t>(d);
        return bj::iter_advance(it);
    }
    iter_skip_value(it);
    return false;
}

bool eat_bool(bj::Iterator* it, bool* out) noexcept {
    assert(it != nullptr && out != nullptr);
    const bj::TokenType t = bj::iter_peek(it);
    if (t == bj::TokenType::BoolTrue)  { *out = true;  return bj::iter_advance(it); }
    if (t == bj::TokenType::BoolFalse) { *out = false; return bj::iter_advance(it); }
    iter_skip_value(it);
    *out = false;
    return false;
}

// Copy the value at the cursor. String → JSON-unescape into `out` (so the
// caller can re-parse it as JSON, e.g. Delta's escaped `schemaString` and
// `stats`). Object/array → raw source bytes verbatim. Advances past the value.
uint32_t eat_raw(bj::Iterator* it, char* out, uint32_t cap) noexcept {
    assert(it != nullptr && out != nullptr);
    assert(cap > 0u);
    const bj::TokenType t = bj::iter_peek(it);
    if (t == bj::TokenType::String) {
        const bj::Token& tk = it->idx->tokens[it->cursor];
        const uint8_t* s = it->idx->src + tk.start;
        const int32_t n = tk.length;
        uint32_t w = 0;
        for (int32_t i = 0; i < n && w + 1u < cap; ++i) {
            uint8_t c = s[i];
            if (c == '\\' && i + 1 < n) {
                const uint8_t e = s[i + 1];
                ++i;
                switch (e) {
                    case '"':  out[w++] = '"';  break;
                    case '\\': out[w++] = '\\'; break;
                    case '/':  out[w++] = '/';  break;
                    case 'n':  out[w++] = '\n'; break;
                    case 'r':  out[w++] = '\r'; break;
                    case 't':  out[w++] = '\t'; break;
                    case 'b':  out[w++] = '\b'; break;
                    case 'f':  out[w++] = '\f'; break;
                    default:   out[w++] = static_cast<char>(e); break;
                }
            } else {
                out[w++] = static_cast<char>(c);
            }
        }
        out[w] = '\0';
        bj::iter_advance(it);
        return w;
    }
    if (t == bj::TokenType::BeginObject || t == bj::TokenType::BeginArray) {
        const int32_t start = it->idx->tokens[it->cursor].start;
        bj::iter_advance(it);
        if (!bj::iter_skip_to_close(it)) { out[0] = '\0'; return 0u; }
        const int32_t end_cursor = it->cursor - 1;
        if (end_cursor < 0 || end_cursor >= it->idx->token_count) {
            out[0] = '\0'; return 0u;
        }
        const int32_t end_off = it->idx->tokens[end_cursor].start + 1;
        if (end_off <= start) { out[0] = '\0'; return 0u; }
        const uint32_t span = static_cast<uint32_t>(end_off - start);
        const uint32_t n = span > cap - 1u ? cap - 1u : span;
        std::memcpy(out, it->idx->src + start, n);
        out[n] = '\0';
        return n;
    }
    iter_skip_value(it);
    out[0] = '\0';
    return 0u;
}

bool parse_partition_values(bj::Iterator* it, PartitionKV* out,
                            uint32_t cap, uint32_t* out_n) noexcept {
    assert(it != nullptr && out != nullptr && out_n != nullptr);
    *out_n = 0;
    if (bj::iter_peek(it) != bj::TokenType::BeginObject)
        return iter_skip_value(it);
    bj::iter_advance(it);
    uint32_t guard = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && guard++ < cap * 4u) {
        const int32_t key_cur = it->cursor;
        bj::iter_advance(it);
        if (*out_n >= cap) { iter_skip_value(it); continue; }
        PartitionKV* kv = &out[*out_n];
        copy_token(it->idx, key_cur, kv->key, kLakeMaxColName);
        kv->is_null = false;
        const bj::TokenType vt = bj::iter_peek(it);
        if (vt == bj::TokenType::Null) {
            kv->is_null = true;
            kv->value[0] = '\0';
            bj::iter_advance(it);
        } else if (vt == bj::TokenType::String) {
            copy_token(it->idx, it->cursor, kv->value, kLakeMaxValBytes);
            bj::iter_advance(it);
        } else if (vt == bj::TokenType::Int64) {
            int64_t iv = 0;
            bj::iter_int64(it, &iv);
            std::snprintf(kv->value, kLakeMaxValBytes, "%lld",
                          static_cast<long long>(iv));
            bj::iter_advance(it);
        } else {
            iter_skip_value(it);
            kv->value[0] = '\0';
        }
        ++(*out_n);
    }
    return bj::iter_peek(it) == bj::TokenType::EndObject &&
           bj::iter_advance(it);
}

bool parse_dv(bj::Iterator* it, DvDescriptor* out) noexcept {
    assert(it != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    out->offset = -1;
    if (bj::iter_peek(it) != bj::TokenType::BeginObject)
        return iter_skip_value(it);
    bj::iter_advance(it);
    uint32_t guard = 0;
    char storage[8] = {0};
    char path_or_inline[kDeltaMaxPath] = {0};
    while (bj::iter_peek(it) == bj::TokenType::Key && guard++ < 64u) {
        const int32_t key_cur = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(it->idx, key_cur, "storageType")) {
            eat_string(it, storage, sizeof(storage));
        } else if (tok_eq(it->idx, key_cur, "pathOrInlineDv")) {
            eat_string(it, path_or_inline, sizeof(path_or_inline));
        } else if (tok_eq(it->idx, key_cur, "sizeInBytes")) {
            int64_t v = 0; eat_int(it, &v); out->size_in_bytes = v;
        } else if (tok_eq(it->idx, key_cur, "cardinality")) {
            int64_t v = 0; eat_int(it, &v); out->cardinality = v;
        } else if (tok_eq(it->idx, key_cur, "offset")) {
            int64_t v = 0; eat_int(it, &v); out->offset = v;
        } else {
            iter_skip_value(it);
        }
    }
    if (storage[0] == 'u') {
        out->type = DvStorageType::kUuid;
        str_copy_cstr(out->uuid_or_path, sizeof(out->uuid_or_path), path_or_inline);
    } else if (storage[0] == 'p') {
        out->type = DvStorageType::kPath;
        str_copy_cstr(out->uuid_or_path, sizeof(out->uuid_or_path), path_or_inline);
    } else if (storage[0] == 'i') {
        out->type = DvStorageType::kInline;
        const size_t n = std::strlen(path_or_inline);
        out->inline_len = static_cast<uint32_t>(
            n > kDeltaMaxDvInline ? kDeltaMaxDvInline : n);
        std::memcpy(out->inline_bytes, path_or_inline, out->inline_len);
    } else {
        out->type = DvStorageType::kNone;
    }
    return bj::iter_peek(it) == bj::TokenType::EndObject &&
           bj::iter_advance(it);
}

bool parse_add(bj::Iterator* it, DeltaAdd* out) noexcept {
    assert(it != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return false;
    bj::iter_advance(it);
    uint32_t guard = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && guard++ < 64u) {
        const int32_t key_cur = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(it->idx, key_cur, "path")) {
            eat_string(it, out->path, sizeof(out->path));
        } else if (tok_eq(it->idx, key_cur, "size")) {
            eat_int(it, &out->size);
        } else if (tok_eq(it->idx, key_cur, "modificationTime")) {
            eat_int(it, &out->modification_time);
        } else if (tok_eq(it->idx, key_cur, "dataChange")) {
            eat_bool(it, &out->data_change);
        } else if (tok_eq(it->idx, key_cur, "partitionValues")) {
            parse_partition_values(it, out->partition_values,
                                    kDeltaMaxPartitions,
                                    &out->n_partition_values);
        } else if (tok_eq(it->idx, key_cur, "stats")) {
            out->stats_len = eat_raw(it, out->stats_json,
                                       sizeof(out->stats_json));
        } else if (tok_eq(it->idx, key_cur, "deletionVector")) {
            out->has_dv = true;
            parse_dv(it, &out->dv);
        } else {
            iter_skip_value(it);
        }
    }
    return bj::iter_peek(it) == bj::TokenType::EndObject &&
           bj::iter_advance(it);
}

bool parse_remove(bj::Iterator* it, DeltaRemove* out) noexcept {
    assert(it != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return false;
    bj::iter_advance(it);
    uint32_t guard = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && guard++ < 64u) {
        const int32_t key_cur = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(it->idx, key_cur, "path")) {
            eat_string(it, out->path, sizeof(out->path));
        } else if (tok_eq(it->idx, key_cur, "deletionTimestamp")) {
            eat_int(it, &out->deletion_timestamp);
        } else if (tok_eq(it->idx, key_cur, "dataChange")) {
            eat_bool(it, &out->data_change);
        } else {
            iter_skip_value(it);
        }
    }
    return bj::iter_peek(it) == bj::TokenType::EndObject &&
           bj::iter_advance(it);
}

bool parse_protocol(bj::Iterator* it, DeltaProtocol* out) noexcept {
    assert(it != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return false;
    bj::iter_advance(it);
    uint32_t guard = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && guard++ < 32u) {
        const int32_t key_cur = it->cursor;
        bj::iter_advance(it);
        int64_t v = 0;
        if (tok_eq(it->idx, key_cur, "minReaderVersion")) {
            eat_int(it, &v); out->min_reader_version = static_cast<int32_t>(v);
        } else if (tok_eq(it->idx, key_cur, "minWriterVersion")) {
            eat_int(it, &v); out->min_writer_version = static_cast<int32_t>(v);
        } else {
            iter_skip_value(it);
        }
    }
    return bj::iter_peek(it) == bj::TokenType::EndObject &&
           bj::iter_advance(it);
}

bool parse_metadata_configuration(bj::Iterator* it, ColumnMappingMode* mode) noexcept {
    assert(it != nullptr && mode != nullptr);
    char cm[16] = {0};
    if (bj::iter_peek(it) != bj::TokenType::BeginObject)
        return iter_skip_value(it);
    bj::iter_advance(it);
    uint32_t g2 = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g2++ < 64u) {
        const int32_t cur = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(it->idx, cur, "delta.columnMapping.mode")) {
            eat_string(it, cm, sizeof(cm));
        } else {
            iter_skip_value(it);
        }
    }
    bj::iter_advance(it);    // EndObject
    if (std::strcmp(cm, "name") == 0)    *mode = ColumnMappingMode::kName;
    else if (std::strcmp(cm, "id") == 0) *mode = ColumnMappingMode::kId;
    return true;
}

bool parse_metadata(bj::Iterator* it, DeltaMetadata* out) noexcept {
    assert(it != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    out->column_mapping = ColumnMappingMode::kNone;
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return false;
    bj::iter_advance(it);
    uint32_t guard = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && guard++ < 64u) {
        const int32_t key_cur = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(it->idx, key_cur, "id")) {
            eat_string(it, out->id, sizeof(out->id));
        } else if (tok_eq(it->idx, key_cur, "name")) {
            eat_string(it, out->name, sizeof(out->name));
        } else if (tok_eq(it->idx, key_cur, "schemaString")) {
            out->schema_len = eat_raw(it, out->schema_string,
                                       sizeof(out->schema_string));
        } else if (tok_eq(it->idx, key_cur, "partitionColumns") &&
                   bj::iter_peek(it) == bj::TokenType::BeginArray) {
            bj::iter_advance(it);
            uint32_t g2 = 0;
            while (bj::iter_peek(it) == bj::TokenType::String &&
                   out->n_partition_columns < kLakeMaxPartCols &&
                   g2++ < 64u) {
                copy_token(it->idx, it->cursor,
                           out->partition_columns[out->n_partition_columns],
                           kLakeMaxColName);
                ++out->n_partition_columns;
                bj::iter_advance(it);
            }
            while (bj::iter_peek(it) != bj::TokenType::EndArray && g2++ < 256u) {
                if (!iter_skip_value(it)) break;
            }
            bj::iter_advance(it);
        } else if (tok_eq(it->idx, key_cur, "configuration")) {
            parse_metadata_configuration(it, &out->column_mapping);
        } else if (tok_eq(it->idx, key_cur, "createdTime")) {
            eat_int(it, &out->created_time);
        } else {
            iter_skip_value(it);
        }
    }
    return bj::iter_peek(it) == bj::TokenType::EndObject &&
           bj::iter_advance(it);
}

bool parse_commit_info(bj::Iterator* it, DeltaCommitInfo* out) noexcept {
    assert(it != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return false;
    bj::iter_advance(it);
    uint32_t guard = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && guard++ < 64u) {
        const int32_t key_cur = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(it->idx, key_cur, "timestamp")) {
            int64_t v = 0;
            if (eat_int(it, &v)) { out->timestamp = v; out->has_timestamp = true; }
        } else if (tok_eq(it->idx, key_cur, "operation")) {
            eat_string(it, out->operation, sizeof(out->operation));
        } else {
            iter_skip_value(it);
        }
    }
    return bj::iter_peek(it) == bj::TokenType::EndObject &&
           bj::iter_advance(it);
}

ActionKind classify_token(const bj::StructuralIndex* idx, int32_t cursor) noexcept {
    assert(idx != nullptr);
    if (tok_eq(idx, cursor, "protocol"))       return ActionKind::kProtocol;
    if (tok_eq(idx, cursor, "metaData"))       return ActionKind::kMetadata;
    if (tok_eq(idx, cursor, "add"))            return ActionKind::kAdd;
    if (tok_eq(idx, cursor, "remove"))         return ActionKind::kRemove;
    if (tok_eq(idx, cursor, "commitInfo"))     return ActionKind::kCommitInfo;
    if (tok_eq(idx, cursor, "cdc"))            return ActionKind::kCdc;
    if (tok_eq(idx, cursor, "domainMetadata")) return ActionKind::kDomainMetadata;
    if (tok_eq(idx, cursor, "txn"))            return ActionKind::kTxn;
    return ActionKind::kUnknown;
}

struct RecordCtx {
    int64_t  version;
    void*    user_ctx;
    ActionFn cb;
    bool     stop;
};

bool on_record(void* raw_ctx, const bj::StructuralIndex* record,
               int64_t /*record_index*/) noexcept {
    assert(raw_ctx != nullptr && record != nullptr);
    RecordCtx* rc = static_cast<RecordCtx*>(raw_ctx);
    bj::Iterator it{};
    if (!bj::iter_init(record, &it)) { rc->stop = true; return false; }
    if (bj::iter_peek(&it) != bj::TokenType::BeginObject) return true;
    bj::iter_advance(&it);
    if (bj::iter_peek(&it) != bj::TokenType::Key) return true;
    const int32_t key_cur = it.cursor;
    bj::iter_advance(&it);
    DeltaAction a{};
    a.kind = classify_token(record, key_cur);
    a.version = rc->version;
    bool ok = true;
    switch (a.kind) {
        case ActionKind::kProtocol:   ok = parse_protocol(&it, &a.protocol); break;
        case ActionKind::kMetadata:   ok = parse_metadata(&it, &a.metadata); break;
        case ActionKind::kAdd:        ok = parse_add(&it, &a.add); break;
        case ActionKind::kRemove:     ok = parse_remove(&it, &a.rem); break;
        case ActionKind::kCommitInfo: ok = parse_commit_info(&it, &a.commit_info); break;
        default: iter_skip_value(&it); break;
    }
    if (!ok) return true;
    if (!rc->cb(rc->user_ctx, &a)) { rc->stop = true; return false; }
    return true;
}

}  // namespace

// `scratch` here is the PER-COMMIT tape arena: ndjson_for_each calls
// scratch->reset() between records. Callers (delta_log_walk_all) must NOT
// pass the snapshot's persistent arena — see the wrapper there.
bool delta_log_parse_commit(const uint8_t* src, uint64_t src_len,
                            int64_t version, Arena* scratch,
                            void* ctx, ActionFn cb) noexcept {
    assert(scratch != nullptr && cb != nullptr);
    assert(src != nullptr || src_len == 0);
    RecordCtx rc{};
    rc.version = version;
    rc.user_ctx = ctx;
    rc.cb = cb;
    rc.stop = false;
    bj::NdjsonStats st{};
    const bool ok = bj::ndjson_for_each(src, static_cast<int64_t>(src_len),
                                         /*ignore_errors=*/true, scratch,
                                         &rc, on_record, &st);
    return ok && !rc.stop;
}

namespace {

int64_t commit_version_for_key(const char* key,
                               const char* table_prefix) noexcept {
    assert(key != nullptr);
    const char* p = key;
    if (table_prefix != nullptr && *table_prefix != '\0') {
        const size_t pl = std::strlen(table_prefix);
        if (std::strncmp(key, table_prefix, pl) == 0) p = key + pl;
    }
    while (*p == '/' || *p == '\\') ++p;
    static const char kLog[] = "_delta_log/";
    if (std::strncmp(p, kLog, sizeof(kLog) - 1) != 0) return -1;
    p += sizeof(kLog) - 1;
    int64_t v = 0;
    int n = 0;
    while (*p >= '0' && *p <= '9' && n < 20) {
        v = v * 10 + (*p - '0');
        ++p; ++n;
    }
    if (n != 20) return -1;
    if (std::strcmp(p, ".json") != 0) return -1;
    return v;
}

bool join_rel(const char* prefix, const char* sub, char* out,
              uint32_t cap) noexcept {
    assert(out != nullptr && cap > 0u);
    const size_t pl = prefix != nullptr ? std::strlen(prefix) : 0u;
    const size_t sl = std::strlen(sub);
    if (pl + 1u + sl + 1u > cap) return false;
    uint32_t p = 0;
    if (pl > 0) { std::memcpy(out, prefix, pl); p = static_cast<uint32_t>(pl); }
    if (pl > 0 && prefix[pl - 1] != '/' && prefix[pl - 1] != '\\')
        out[p++] = '/';
    std::memcpy(out + p, sub, sl);
    p += static_cast<uint32_t>(sl);
    out[p] = '\0';
    return true;
}

void sort_versions(int64_t* arr, uint32_t* keys_idx, uint32_t n) noexcept {
    assert(arr != nullptr && keys_idx != nullptr);
    for (uint32_t i = 1; i < n; ++i) {
        const int64_t v = arr[i];
        const uint32_t ki = keys_idx[i];
        uint32_t j = i;
        while (j > 0 && arr[j - 1] > v) {
            arr[j] = arr[j - 1];
            keys_idx[j] = keys_idx[j - 1];
            --j;
        }
        arr[j] = v;
        keys_idx[j] = ki;
    }
}

}  // namespace

bool delta_log_walk_all(ObjectStore* os, const char* table_rel_prefix,
                        int64_t max_version, Arena* scratch,
                        void* ctx, ActionFn cb) noexcept {
    assert(os != nullptr && scratch != nullptr && cb != nullptr);
    char prefix[kDeltaMaxPath];
    if (!join_rel(table_rel_prefix, "_delta_log/", prefix, sizeof(prefix)))
        return false;
    ObjectEntry* entries = scratch->allocate_array<ObjectEntry>(kLakeMaxCommits);
    if (entries == nullptr) return false;
    uint32_t n_entries = 0;
    const int rc = os_list(os, prefix, entries, kLakeMaxCommits, &n_entries);
    if (rc != kOsOk) return false;
    int64_t* versions = scratch->allocate_array<int64_t>(kLakeMaxCommits);
    uint32_t* idxs    = scratch->allocate_array<uint32_t>(kLakeMaxCommits);
    if (versions == nullptr || idxs == nullptr) return false;
    uint32_t n = 0;
    for (uint32_t i = 0; i < n_entries && n < kLakeMaxCommits; ++i) {
        const int64_t v = commit_version_for_key(entries[i].key,
                                                  table_rel_prefix);
        if (v < 0) continue;
        if (max_version >= 0 && v > max_version) continue;
        versions[n] = v;
        idxs[n] = i;
        ++n;
    }
    sort_versions(versions, idxs, n);
    // ndjson_for_each resets the scratch arena between records; if we passed
    // the caller's `scratch` (which holds the snapshot's persistent bytes)
    // those bytes would vanish under us. Spin up a per-walk local arena for
    // the tape only. The body buffers go to the caller's arena so on_get's
    // bytes live across the parse.
    bolt::Arena tape_arena;
    for (uint32_t i = 0; i < n; ++i) {
        const ObjectEntry& e = entries[idxs[i]];
        const uint8_t* body = nullptr;
        uint64_t body_len = 0;
        if (os_get(os, e.key, scratch, &body, &body_len) != kOsOk)
            return false;
        if (!delta_log_parse_commit(body, body_len, versions[i],
                                     &tape_arena, ctx, cb)) {
            return false;
        }
    }
    return true;
}

namespace {

struct TsCtx { int64_t threshold; int64_t best; };

bool ts_visit(void* raw, const DeltaAction* a) noexcept {
    assert(raw != nullptr && a != nullptr);
    TsCtx* c = static_cast<TsCtx*>(raw);
    if (a->kind == ActionKind::kCommitInfo && a->commit_info.has_timestamp) {
        if (a->commit_info.timestamp <= c->threshold && a->version > c->best) {
            c->best = a->version;
        }
    }
    return true;
}

}  // namespace

bool delta_log_version_for_timestamp(ObjectStore* os,
                                     const char* table_rel_prefix,
                                     int64_t timestamp_ms, Arena* scratch,
                                     int64_t* out_version) noexcept {
    assert(os != nullptr && scratch != nullptr && out_version != nullptr);
    TsCtx c{};
    c.threshold = timestamp_ms;
    c.best = -1;
    const bool ok = delta_log_walk_all(os, table_rel_prefix, -1, scratch,
                                        &c, ts_visit);
    *out_version = c.best;
    return ok;
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
