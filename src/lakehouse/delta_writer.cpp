// bolt/lakehouse/delta_writer.cpp — Delta W3 core write path:
// schema JSON, Parquet-file emitter, commit-JSON append, conflict detection,
// AppendHandle, table_create.
//
// Tiger Style: PODs, ≥2 asserts/fn, ≤70 line fns, no exceptions, Arena allocs,
// commits atomic-via-conflict-check + os_put.

#include "bolt/lakehouse/delta/writer.h"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>

#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/lakehouse/delta/log.h"
#include "bolt/lakehouse/delta/snapshot.h"
#include "bolt/lakehouse/handle.h"
#include "delta_write_internal.h"

namespace bolt {
namespace lakehouse {
namespace delta {

namespace {

// ----------------------- string helpers (POD-safe) -------------------------

inline void str_copy_cstr(char* dst, uint32_t cap, const char* src) noexcept {
    assert(dst != nullptr && cap > 0u);
    if (src == nullptr) { dst[0] = '\0'; return; }
    const size_t n = std::strlen(src);
    const size_t m = n > cap - 1u ? cap - 1u : n;
    if (m > 0) std::memcpy(dst, src, m);
    dst[m] = '\0';
}

inline bool path_join(const char* a, const char* b, char* out,
                      uint32_t cap) noexcept {
    assert(a != nullptr && b != nullptr && out != nullptr);
    const size_t al = std::strlen(a);
    const size_t bl = std::strlen(b);
    if (al + 1u + bl + 1u > cap) return false;
    std::memcpy(out, a, al);
    uint32_t p = static_cast<uint32_t>(al);
    if (al > 0 && a[al - 1] != '/' && a[al - 1] != '\\') out[p++] = '/';
    std::memcpy(out + p, b, bl);
    p += static_cast<uint32_t>(bl);
    out[p] = '\0';
    return true;
}

inline uint64_t now_unix_ms() noexcept {
    return static_cast<uint64_t>(std::time(nullptr)) * 1000ull;
}

// counter-based file uuid; no crypto rng dep
inline void make_part_uuid(uint64_t seq, char out[32]) noexcept {
    assert(out != nullptr);
    static const char hex[] = "0123456789abcdef";
    const uint64_t t = now_unix_ms();
    uint64_t mix = t ^ (seq * 0x9E3779B97F4A7C15ull);
    for (uint32_t i = 0; i < 16; ++i) {
        out[i] = hex[(mix >> ((i & 0xF) * 4)) & 0xFu];
    }
    for (uint32_t i = 16; i < 31; ++i) {
        out[i] = hex[(seq >> ((i - 16) & 0xF) * 4) & 0xFu];
    }
    out[31] = '\0';
}

inline const char* compression_suffix(Compression c) noexcept {
    switch (c) {
        case Compression::kSnappy: return "snappy";
        case Compression::kGzip:   return "gzip";
        case Compression::kZstd:   return "zstd";
        case Compression::kLz4:    return "lz4";
        default:                   return "raw";
    }
}

inline uint8_t parquet_codec_byte(Compression c) noexcept {
    // ParquetWriteOpts: 0=none, 1=SNAPPY. Others rejected — fall back.
    return c == Compression::kSnappy ? uint8_t{1} : uint8_t{0};
}

// ---- type → Delta JSON type string -------------------------------------

const char* delta_type_string(BoltType t) noexcept {
    switch (t) {
        case BoltType::Int32:   return "integer";
        case BoltType::Int64:   return "long";
        case BoltType::Float32: return "float";
        case BoltType::Float64: return "double";
        case BoltType::Utf8:    return "string";
        case BoltType::Binary:  return "binary";
        case BoltType::Bool:    return "boolean";
        case BoltType::Date32:  return "date";
        case BoltType::Timestamp: return "timestamp";
        case BoltType::Int16:   return "short";
        case BoltType::Int8:    return "byte";
        default:                return "string";
    }
}

// Append a JSON-escaped literal into a buffer. Returns number of bytes added.
uint32_t json_escape_into(char* dst, uint32_t cap, uint32_t off,
                          const char* src) noexcept {
    assert(dst != nullptr && src != nullptr);
    uint32_t w = off;
    for (uint32_t i = 0; src[i] != '\0'; ++i) {
        if (w + 2u >= cap) break;
        const char c = src[i];
        if (c == '"' || c == '\\') { dst[w++] = '\\'; dst[w++] = c; }
        else if (c == '\n')        { dst[w++] = '\\'; dst[w++] = 'n'; }
        else if (c == '\r')        { dst[w++] = '\\'; dst[w++] = 'r'; }
        else if (c == '\t')        { dst[w++] = '\\'; dst[w++] = 't'; }
        else                       { dst[w++] = c; }
    }
    dst[w] = '\0';
    return w - off;
}

uint32_t schema_json_build(const BoltSchema* s, char* dst,
                           uint32_t cap) noexcept {
    assert(s != nullptr && dst != nullptr);
    assert(cap > 64u);
    uint32_t w = 0;
    const int n_written = std::snprintf(dst, cap,
        "{\"type\":\"struct\",\"fields\":[");
    if (n_written < 0) return 0;
    w = static_cast<uint32_t>(n_written);
    for (uint32_t i = 0; i < s->num_fields && w + 128u < cap; ++i) {
        const BoltField& f = s->fields[i];
        const int wn = std::snprintf(dst + w, cap - w,
            "%s{\"name\":\"%s\",\"type\":\"%s\",\"nullable\":%s,\"metadata\":{}}",
            i == 0 ? "" : ",",
            f.name,
            delta_type_string(f.type),
            f.nullable ? "true" : "false");
        if (wn < 0) break;
        w += static_cast<uint32_t>(wn);
    }
    if (w + 4u < cap) {
        dst[w++] = ']';
        dst[w++] = '}';
        dst[w]   = '\0';
    }
    return w;
}

// ----------- write a BoltBatch out as one Parquet file via the writer ------

bool parquet_write_batch_to_path(const char* abs_path, const BoltBatch* batch,
                                 const WriteOptions* wopts) noexcept {
    assert(abs_path != nullptr && batch != nullptr && wopts != nullptr);
    using namespace bolt::ingest::parquet;
    ParquetWriteOpts opts{};
    opts.n_columns = static_cast<uint32_t>(batch->num_cols);
    opts.row_group_target_bytes = 1u << 20;
    opts.compression = parquet_codec_byte(wopts->compression);
    opts.emit_statistics = wopts->emit_stats;
    // COLUMN-WIDTH NORMALIZATION (2026-08-03) — STACK BUFFER OVERFLOW FIX.
    // This loop was bounded by kMaxBatchColumns while writing into
    // `opts.columns`, which is `ParquetWriteColumn[kMaxFixedColumns]` (256) on
    // a STACK-LOCAL struct. The bound was correct while both constants were
    // 256; raising kMaxBatchColumns to 1024 (G2FEAT-47) silently turned this
    // into an out-of-bounds write for any 257..1024-column batch, corrupting
    // the adjacent fields of `opts` (n_columns, row_group_target_bytes,
    // compression, ...) and past the struct. parquet_write_open's own
    // `n_columns > kPwMaxColumns` guard runs AFTER this loop, so it could never
    // prevent the corruption — and may itself read already-corrupted fields.
    // Bound by the destination array's real size and fail closed instead;
    // n_columns keeps the TRUE width so parquet_write_open rejects rather than
    // silently writing a truncated schema.
    if (opts.n_columns > kMaxFixedColumns) return false;
    for (uint32_t i = 0; i < opts.n_columns; ++i) {
        const auto& f = batch->schema.field(static_cast<int>(i));
        str_copy_cstr(opts.columns[i].name, sizeof(opts.columns[i].name),
                      f.name);
        opts.columns[i].type     = f.type;
        opts.columns[i].nullable = f.nullable;
    }
    ParquetWriter* w = parquet_write_open(abs_path, &opts);
    if (w == nullptr) return false;
    if (!parquet_write_row_group(w, batch)) {
        parquet_write_close(w);
        return false;
    }
    return parquet_write_close(w);
}

// Read file size after write — exact size lands in Add action.
int64_t file_size_on_disk(const char* abs_path) noexcept {
    assert(abs_path != nullptr);
    std::FILE* f = std::fopen(abs_path, "rb");
    if (f == nullptr) return -1;
    if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); return -1; }
    const long sz = std::ftell(f);
    std::fclose(f);
    return sz < 0 ? -1 : static_cast<int64_t>(sz);
}

// Compose a min/max/null stats JSON from BoltBatch columns. Bounded.
uint32_t build_stats_json(const BoltBatch* b, char* dst,
                          uint32_t cap) noexcept {
    assert(b != nullptr && dst != nullptr);
    assert(cap > 64u);
    int w = std::snprintf(dst, cap, "{\"numRecords\":%lld,\"minValues\":{",
                          static_cast<long long>(b->num_rows));
    if (w < 0) return 0;
    uint32_t off = static_cast<uint32_t>(w);
    for (uint32_t c = 0; c < b->num_cols && off + 64u < cap; ++c) {
        const auto& f = b->schema.field(static_cast<int>(c));
        if (f.type != BoltType::Int64) continue;
        const int64_t* p = static_cast<const int64_t*>(
            b->columns[b->read_epoch][c].data);
        if (p == nullptr || b->num_rows == 0) continue;
        int64_t mn = p[0];
        for (int64_t i = 1; i < b->num_rows; ++i) if (p[i] < mn) mn = p[i];
        int wn = std::snprintf(dst + off, cap - off, "%s\"%s\":%lld",
                               c == 0 ? "" : ",", f.name,
                               static_cast<long long>(mn));
        if (wn < 0) break;
        off += static_cast<uint32_t>(wn);
    }
    int wn = std::snprintf(dst + off, cap - off, "},\"maxValues\":{");
    if (wn < 0) return off;
    off += static_cast<uint32_t>(wn);
    for (uint32_t c = 0; c < b->num_cols && off + 64u < cap; ++c) {
        const auto& f = b->schema.field(static_cast<int>(c));
        if (f.type != BoltType::Int64) continue;
        const int64_t* p = static_cast<const int64_t*>(
            b->columns[b->read_epoch][c].data);
        if (p == nullptr || b->num_rows == 0) continue;
        int64_t mx = p[0];
        for (int64_t i = 1; i < b->num_rows; ++i) if (p[i] > mx) mx = p[i];
        wn = std::snprintf(dst + off, cap - off, "%s\"%s\":%lld",
                           c == 0 ? "" : ",", f.name,
                           static_cast<long long>(mx));
        if (wn < 0) break;
        off += static_cast<uint32_t>(wn);
    }
    wn = std::snprintf(dst + off, cap - off, "}}");
    if (wn > 0) off += static_cast<uint32_t>(wn);
    return off;
}

// ----------- log key helpers -----------------------------------------------

bool log_key_for_version(const char* table_rel, int64_t version, char* out,
                         uint32_t cap) noexcept {
    assert(table_rel != nullptr && out != nullptr);
    return std::snprintf(out, cap, "%s/_delta_log/%020lld.json", table_rel,
                         static_cast<long long>(version)) > 0;
}

// Return highest visible commit version, or -1 if none.
int64_t latest_version(ObjectStore* os, const char* table_rel,
                       Arena* scratch) noexcept {
    assert(os != nullptr && table_rel != nullptr && scratch != nullptr);
    char prefix[kDeltaMaxPath];
    if (std::snprintf(prefix, sizeof(prefix), "%s/_delta_log/", table_rel) <= 0)
        return -1;
    ObjectEntry* entries = scratch->allocate_array<ObjectEntry>(kLakeMaxCommits);
    if (entries == nullptr) return -1;
    uint32_t n = 0;
    if (os_list(os, prefix, entries, kLakeMaxCommits, &n) != kOsOk) return -1;
    int64_t best = -1;
    for (uint32_t i = 0; i < n; ++i) {
        const char* k = entries[i].key;
        const char* slash = std::strrchr(k, '/');
        if (slash == nullptr) continue;
        ++slash;
        // expect 20 digits + ".json"
        int64_t v = 0;
        int d = 0;
        while (slash[d] >= '0' && slash[d] <= '9' && d < 20) {
            v = v * 10 + (slash[d] - '0');
            ++d;
        }
        if (d != 20) continue;
        if (std::strcmp(slash + 20, ".json") != 0) continue;
        if (v > best) best = v;
    }
    return best;
}

}  // namespace

// =============================================================================
// table_create — bootstrap a Delta table: catalog row + log version 0.
// =============================================================================

bool delta_table_create(TableHandle** out, Arena* arena, Catalog* cat,
                        const char* ns, const char* name,
                        const BoltSchema* schema,
                        const WriteOptions* opts) noexcept {
    assert(out != nullptr && arena != nullptr && cat != nullptr);
    assert(ns != nullptr && name != nullptr && schema != nullptr);
    if (cat_create_table(cat, ns, name, TableLayout::kDelta) != kCatOk &&
        cat_create_table(cat, ns, name, TableLayout::kDelta) != 0) {
        // ignore "already exists"; open below will still succeed
    }
    TableHandle* th = nullptr;
    if (!delta_table_open(&th, arena, cat, ns, name)) return false;

    WriteOptions wopts;
    if (opts != nullptr) wopts = *opts;
    else                 write_options_init(&wopts);

    // Build version-0 commit: protocol + metadata.
    Arena tape;
    char* body = tape.allocate_array<char>(kDeltaMaxSchemaBytes + 4096u);
    if (body == nullptr) return false;
    char schema_json[kDeltaMaxSchemaBytes];
    schema_json_build(schema, schema_json, sizeof(schema_json));
    char escaped[kDeltaMaxSchemaBytes];
    json_escape_into(escaped, sizeof(escaped), 0, schema_json);

    char parts[256] = "";
    uint32_t po = 0;
    for (uint32_t i = 0; i < wopts.n_partition_columns && po + 80u < sizeof(parts); ++i) {
        int n = std::snprintf(parts + po, sizeof(parts) - po, "%s\"%s\"",
                              i == 0 ? "" : ",", wopts.partition_columns[i]);
        if (n < 0) break;
        po += static_cast<uint32_t>(n);
    }

    const uint64_t ms = now_unix_ms();
    const int blen = std::snprintf(body, kDeltaMaxSchemaBytes + 4096u,
        "{\"protocol\":{\"minReaderVersion\":1,\"minWriterVersion\":2}}\n"
        "{\"metaData\":{\"id\":\"%s-%s\",\"name\":\"%s\",\"schemaString\":\"%s\","
        "\"partitionColumns\":[%s],\"configuration\":{},\"createdTime\":%llu}}\n"
        "{\"commitInfo\":{\"timestamp\":%llu,\"operation\":\"CREATE TABLE\"}}\n",
        ns, name, name, escaped, parts,
        static_cast<unsigned long long>(ms),
        static_cast<unsigned long long>(ms));
    if (blen <= 0) return false;

    char key[kDeltaMaxPath];
    if (!log_key_for_version(th->table_rel, 0, key, sizeof(key))) return false;

    // Atomicity: refuse if a v0 commit already lives there.
    ObjectMeta meta{};
    if (os_head(&th->os, key, &meta) == kOsOk && meta.exists) {
        *out = th;
        return true;  // table already created — idempotent
    }
    if (os_put(&th->os, key, reinterpret_cast<const uint8_t*>(body),
               static_cast<uint64_t>(blen)) != kOsOk) {
        return false;
    }
    *out = th;
    return true;
}

// =============================================================================
// AppendHandle
// =============================================================================

struct AppendHandle {
    TableHandle*  th;
    WriteOptions  wopts;
    int64_t       base_version;     // version observed at open()
    uint32_t      n_batches;
    char          parquet_paths[kWriteMaxBatches][kDeltaMaxPath];   // rel
    int64_t       sizes[kWriteMaxBatches];
    int64_t       row_counts[kWriteMaxBatches];
    char          stats_json[kWriteMaxBatches][kDeltaMaxStatsBytes];
    uint32_t      stats_len[kWriteMaxBatches];
};

bool delta_append_open(AppendHandle** out, TableHandle* th) noexcept {
    assert(out != nullptr && th != nullptr);
    AppendHandle* ah = th->arena->allocate_array<AppendHandle>(1);
    if (ah == nullptr) return false;
    std::memset(ah, 0, sizeof(*ah));
    ah->th = th;
    write_options_init(&ah->wopts);
    ah->base_version = latest_version(&th->os, th->table_rel, th->arena);
    *out = ah;
    return true;
}

bool delta_append_write(AppendHandle* ah, const BoltBatch* batch) noexcept {
    assert(ah != nullptr && batch != nullptr);
    if (ah->n_batches >= kWriteMaxBatches) return false;
    const uint32_t slot = ah->n_batches;
    char uuid[32];
    make_part_uuid(slot, uuid);
    char rel[kDeltaMaxPath];
    if (std::snprintf(rel, sizeof(rel), "data/part-%s.%s.parquet", uuid,
                      compression_suffix(ah->wopts.compression)) <= 0) {
        return false;
    }
    str_copy_cstr(ah->parquet_paths[slot], sizeof(ah->parquet_paths[slot]),
                  rel);
    // absolute path = root + table_rel + rel
    char tbl_rel_join[kDeltaMaxPath];
    if (!path_join(ah->th->table_rel, rel, tbl_rel_join, sizeof(tbl_rel_join)))
        return false;
    char abs_path[kDeltaMaxPath];
    if (!path_join(ah->th->fs_store.root, tbl_rel_join, abs_path,
                   sizeof(abs_path))) return false;
    // make parent dir
    char* last_slash = std::strrchr(abs_path, '/');
    if (last_slash != nullptr) {
        *last_slash = '\0';
        // mkdir -p via std::filesystem at the writer's outer edge.
        std::FILE* probe = std::fopen(abs_path, "rb");
        if (probe == nullptr) {
            // best effort directory create via system call replacement —
            // FilesystemObjectStore::put_object will already mkdir on put,
            // but parquet_write_open is a raw fopen and needs the dir live.
            // Use std::filesystem.
        } else { std::fclose(probe); }
        *last_slash = '/';
    }
    // Ensure parent dir exists.
    {
        char dir[kDeltaMaxPath];
        std::strncpy(dir, abs_path, sizeof(dir) - 1u);
        dir[sizeof(dir) - 1u] = '\0';
        char* sl = std::strrchr(dir, '/');
        if (sl != nullptr) {
            *sl = '\0';
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
        }
    }
    if (!parquet_write_batch_to_path(abs_path, batch, &ah->wopts)) return false;
    ah->sizes[slot]      = file_size_on_disk(abs_path);
    ah->row_counts[slot] = batch->num_rows;
    if (ah->wopts.emit_stats) {
        ah->stats_len[slot] = build_stats_json(batch, ah->stats_json[slot],
                                                sizeof(ah->stats_json[slot]));
    }
    ++ah->n_batches;
    return true;
}

namespace {

uint32_t append_add_json(char* dst, uint32_t cap, uint32_t off,
                         const char* path, int64_t size, uint64_t mtime,
                         const char* stats_json,
                         uint32_t stats_len) noexcept {
    assert(dst != nullptr);
    int wn = std::snprintf(dst + off, cap - off,
        "{\"add\":{\"path\":\"%s\",\"size\":%lld,\"modificationTime\":%llu,"
        "\"dataChange\":true,\"partitionValues\":{}",
        path, static_cast<long long>(size),
        static_cast<unsigned long long>(mtime));
    if (wn < 0) return off;
    off += static_cast<uint32_t>(wn);
    if (stats_len > 0 && off + stats_len + 32u < cap) {
        // embed stats as escaped JSON string
        wn = std::snprintf(dst + off, cap - off, ",\"stats\":\"");
        if (wn > 0) off += static_cast<uint32_t>(wn);
        off += json_escape_into(dst, cap, off, stats_json);
        if (off + 2u < cap) dst[off++] = '"';
    }
    if (off + 4u < cap) {
        dst[off++] = '}'; dst[off++] = '}'; dst[off++] = '\n'; dst[off] = '\0';
    }
    return off;
}

}  // namespace

bool delta_append_commit(AppendHandle* ah) noexcept {
    assert(ah != nullptr);
    if (ah->n_batches == 0) return true;
    // Conflict detection: re-check latest visible version.
    const int64_t now_latest = latest_version(&ah->th->os, ah->th->table_rel,
                                               ah->th->arena);
    if (now_latest != ah->base_version) return false;   // concurrent commit
    const int64_t target = ah->base_version + 1;

    Arena scratch;
    const uint32_t cap = 64u * 1024u + ah->n_batches * 8192u;
    char* body = scratch.allocate_array<char>(cap);
    if (body == nullptr) return false;
    uint32_t off = 0;
    const uint64_t ms = now_unix_ms();
    for (uint32_t i = 0; i < ah->n_batches; ++i) {
        off = append_add_json(body, cap, off, ah->parquet_paths[i],
                              ah->sizes[i], ms, ah->stats_json[i],
                              ah->stats_len[i]);
    }
    int wn = std::snprintf(body + off, cap - off,
        "{\"commitInfo\":{\"timestamp\":%llu,\"operation\":\"WRITE\"}}\n",
        static_cast<unsigned long long>(ms));
    if (wn > 0) off += static_cast<uint32_t>(wn);

    char key[kDeltaMaxPath];
    if (!log_key_for_version(ah->th->table_rel, target, key, sizeof(key)))
        return false;
    ObjectMeta meta{};
    if (os_head(&ah->th->os, key, &meta) == kOsOk && meta.exists) return false;
    if (os_put(&ah->th->os, key, reinterpret_cast<const uint8_t*>(body),
               static_cast<uint64_t>(off)) != kOsOk) return false;
    ah->base_version = target;
    ah->n_batches = 0;
    return true;
}

void delta_append_close(AppendHandle* /*ah*/) noexcept {}

// =============================================================================
// Re-usable commit helpers exposed to sibling write TUs via extern decls.
// =============================================================================

bool delta_writer_commit_raw(TableHandle* th, const char* body,
                             uint32_t body_len,
                             int64_t* inout_base_version) noexcept {
    assert(th != nullptr && body != nullptr && inout_base_version != nullptr);
    const int64_t now_latest = latest_version(&th->os, th->table_rel,
                                               th->arena);
    if (now_latest != *inout_base_version) return false;
    const int64_t target = *inout_base_version + 1;
    char key[kDeltaMaxPath];
    if (!log_key_for_version(th->table_rel, target, key, sizeof(key)))
        return false;
    ObjectMeta meta{};
    if (os_head(&th->os, key, &meta) == kOsOk && meta.exists) return false;
    if (os_put(&th->os, key, reinterpret_cast<const uint8_t*>(body),
               static_cast<uint64_t>(body_len)) != kOsOk) return false;
    *inout_base_version = target;
    return true;
}

int64_t delta_writer_latest_version(TableHandle* th) noexcept {
    assert(th != nullptr);
    return latest_version(&th->os, th->table_rel, th->arena);
}

uint64_t delta_writer_now_ms() noexcept { return now_unix_ms(); }

uint32_t delta_writer_escape_json(char* dst, uint32_t cap, uint32_t off,
                                   const char* src) noexcept {
    return json_escape_into(dst, cap, off, src);
}

bool delta_writer_log_key(const char* table_rel, int64_t version, char* out,
                           uint32_t cap) noexcept {
    return log_key_for_version(table_rel, version, out, cap);
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
