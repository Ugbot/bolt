// bolt/lakehouse/iceberg_scan.cpp — top-level Iceberg read path.
//
// metadata → snapshot → manifest-list → manifest → per-file Parquet decode.
// Opaque TableHandle / ScanHandle live in the `bolt::lakehouse::iceberg`
// sub-namespace so they don't collide with Delta's.
//
// W4 deviations:
//   - Manifests + manifest-list read as JSON (BOLT_ICEBERG_MANIFEST_JSON).
//     TODO(W5-avro-nested-reader).
//   - Position/equality delete files are in-memory sets only; Parquet-body
//     loaders TODO(W5-delete-load).
//
// Tiger Style: bounded, noexcept, fixed-size scratch.

#include "bolt/lakehouse/iceberg/scan.h"

#include <cassert>
#include <cstdio>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/lakehouse/iceberg/delete_file.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/partition.h"
#include "bolt/lakehouse/iceberg/snapshot.h"
#include "bolt/lakehouse/iceberg/statistics.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace pq = bolt::ingest::parquet;

namespace {

constexpr uint32_t kMaxNsName       = 128u;
constexpr uint32_t kMaxFsRoot       = 1024u;
constexpr uint32_t kMaxLiveFiles    = 4096u;
constexpr uint32_t kMaxPosDels      = 16384u;
constexpr uint32_t kMaxEqDels       = 4096u;

bool join_ns(const char* ns, const char* name, char* out,
             uint32_t cap) noexcept {
    assert(ns != nullptr && name != nullptr && out != nullptr);
    const size_t nl = std::strlen(ns);
    const size_t ml = std::strlen(name);
    if (nl + 1u + ml + 1u > cap) return false;
    std::memcpy(out, ns, nl);
    out[nl] = '/';
    std::memcpy(out + nl + 1u, name, ml);
    out[nl + 1u + ml] = '\0';
    return true;
}

bool bootstrap_object_store(Catalog* cat, const char* ns, const char* name,
                            FilesystemObjectStore* fs,
                            ObjectStore* os, char* root_out,
                            uint32_t root_cap) noexcept {
    assert(cat != nullptr && fs != nullptr && os != nullptr);
    char tp[kCatMaxPath];
    if (cat_table_path(cat, ns, name, tp, sizeof(tp)) != kCatOk) return false;
    char suffix[kMaxNsName * 2u + 4u];
    if (!join_ns(ns, name, suffix, sizeof(suffix))) return false;
    const size_t tl = std::strlen(tp);
    const size_t sl = std::strlen(suffix);
    if (tl <= sl + 1u) return false;
    if (std::strcmp(tp + tl - sl, suffix) != 0) return false;
    const size_t root_len = tl - sl - 1u;
    if (root_len + 1u > root_cap) return false;
    std::memcpy(root_out, tp, root_len);
    root_out[root_len] = '\0';
    return filesystem_object_store_init(fs, root_out, os);
}

bool path_join(const char* a, const char* b, char* out, uint32_t cap) noexcept {
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

const char* strip_root(const char* root, const char* path) noexcept {
    assert(root != nullptr && path != nullptr);
    const size_t rl = std::strlen(root);
    if (rl == 0u) return path;
    if (std::strncmp(path, root, rl) == 0) {
        const char* p = path + rl;
        while (*p == '/' || *p == '\\') ++p;
        return p;
    }
    return path;
}

bool read_version_hint(ObjectStore* os, const char* table_rel, Arena* scratch,
                       int64_t* out) noexcept {
    assert(os != nullptr && table_rel != nullptr && out != nullptr);
    char key[kCatMaxPath];
    if (!path_join(table_rel, "metadata/version-hint.text",
                   key, sizeof(key))) return false;
    const uint8_t* body = nullptr; uint64_t blen = 0;
    if (os_get(os, key, scratch, &body, &blen) != kOsOk) return false;
    int64_t v = 0; bool any = false;
    for (uint64_t i = 0; i < blen; ++i) {
        const char c = static_cast<char>(body[i]);
        if (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); any = true; }
        else if (any) break;
    }
    if (!any) return false;
    *out = v;
    return true;
}

bool find_latest_metadata(ObjectStore* os, const char* table_rel,
                          Arena* scratch, char* out_key,
                          uint32_t cap) noexcept {
    assert(os != nullptr && table_rel != nullptr && out_key != nullptr);
    int64_t hint = 0;
    if (read_version_hint(os, table_rel, scratch, &hint)) {
        char rel[64];
        const int n = std::snprintf(rel, sizeof(rel),
                                     "metadata/v%lld.metadata.json",
                                     static_cast<long long>(hint));
        if (n <= 0) return false;
        if (!path_join(table_rel, rel, out_key, cap)) return false;
        const uint8_t* body = nullptr; uint64_t blen = 0;
        if (os_get(os, out_key, scratch, &body, &blen) == kOsOk) return true;
    }
    char prefix[kCatMaxPath];
    if (!path_join(table_rel, "metadata/", prefix, sizeof(prefix))) return false;
    ObjectEntry* listing = scratch->allocate_array<ObjectEntry>(256u);
    if (listing == nullptr) return false;
    uint32_t nl = 0;
    if (os_list(os, prefix, listing, 256u, &nl) != kOsOk) return false;
    int64_t best = -1;
    char best_key[kCatMaxPath]; best_key[0] = '\0';
    for (uint32_t i = 0; i < nl; ++i) {
        const char* k = listing[i].key;
        const char* slash = std::strrchr(k, '/');
        const char* tail = slash ? slash + 1 : k;
        if (tail[0] != 'v') continue;
        int64_t v = 0; const char* p = tail + 1; bool any = false;
        while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); ++p; any = true; }
        if (!any) continue;
        if (std::strcmp(p, ".metadata.json") != 0) continue;
        if (v > best) {
            best = v;
            std::strncpy(best_key, k, cap - 1u);
            best_key[cap - 1u] = '\0';
        }
    }
    if (best < 0) return false;
    std::strncpy(out_key, best_key, cap - 1u);
    out_key[cap - 1u] = '\0';
    return true;
}

}  // namespace

struct TableHandle {
    Arena*                arena;
    Catalog*              catalog;
    char                  ns[kMaxNsName];
    char                  name[kMaxNsName];
    char                  table_rel[kMaxFsRoot];
    char                  fs_root[kMaxFsRoot];
    FilesystemObjectStore fs_store;
    ObjectStore           os;
    Metadata              meta;
    bool                  meta_loaded;
    uint8_t               _pad[7];
};

struct ScanHandle {
    TableHandle*  table;
    Arena*        scratch;
    ReadOptions   opts;
    Snapshot      snap;
    DataFileRef*  live_files;
    uint32_t      n_live;
    uint32_t      cur_file_idx;
    pq::PqMeta*   cur_meta;
    const uint8_t* cur_body;
    uint64_t      cur_body_len;
    uint32_t      cur_row_group;
    bool          cur_file_open;
    uint8_t       _pad[3];
    PositionDeleteSet    pos_dels;
    EqualityDeleteSetI64 eq_dels;
};

bool iceberg_table_open(TableHandle** out, Arena* arena, Catalog* catalog,
                        const char* namespace_, const char* name) noexcept {
    assert(out != nullptr && arena != nullptr);
    assert(catalog != nullptr && namespace_ != nullptr && name != nullptr);
    TableHandle* h = arena->allocate_array<TableHandle>(1);
    if (h == nullptr) return false;
    std::memset(h, 0, sizeof(*h));
    h->arena = arena;
    h->catalog = catalog;
    if (std::strlen(namespace_) + 1u > sizeof(h->ns)) return false;
    if (std::strlen(name) + 1u > sizeof(h->name)) return false;
    std::strncpy(h->ns, namespace_, sizeof(h->ns) - 1u);
    std::strncpy(h->name, name, sizeof(h->name) - 1u);
    if (!join_ns(namespace_, name, h->table_rel, sizeof(h->table_rel)))
        return false;
    if (!bootstrap_object_store(catalog, namespace_, name,
                                 &h->fs_store, &h->os,
                                 h->fs_root, sizeof(h->fs_root)))
        return false;
    char key[kCatMaxPath];
    if (!find_latest_metadata(&h->os, h->table_rel, arena, key, sizeof(key)))
        return false;
    const uint8_t* body = nullptr; uint64_t blen = 0;
    if (os_get(&h->os, key, arena, &body, &blen) != kOsOk) return false;
    if (!metadata_parse(body, static_cast<uint32_t>(blen), arena, &h->meta))
        return false;
    h->meta_loaded = true;
    *out = h;
    return true;
}

void iceberg_table_close(TableHandle* /*h*/) noexcept {}

namespace {

bool open_next_file(ScanHandle* s) noexcept {
    assert(s != nullptr);
    while (s->cur_file_idx < s->n_live) {
        const DataFileRef& f = s->live_files[s->cur_file_idx];
        const char* dot = std::strrchr(f.file_path, '.');
        if (dot != nullptr) {
            if (!(std::strcmp(dot, ".parquet") == 0 ||
                  std::strcmp(dot, ".PARQUET") == 0 ||
                  std::strcmp(dot, ".pq") == 0)) {
                ++s->cur_file_idx; continue;
            }
        }
        const char* rel = strip_root(s->table->fs_root, f.file_path);
        const uint8_t* body = nullptr; uint64_t blen = 0;
        if (os_get(&s->table->os, rel, s->scratch, &body, &blen) != kOsOk) {
            char key[kCatMaxPath];
            if (!path_join(s->table->table_rel, rel, key, sizeof(key))) {
                ++s->cur_file_idx; continue;
            }
            if (os_get(&s->table->os, key, s->scratch, &body, &blen) != kOsOk) {
                ++s->cur_file_idx; continue;
            }
        }
        pq::PqMeta* meta = s->scratch->allocate_array<pq::PqMeta>(1);
        if (meta == nullptr) return false;
        std::memset(meta, 0, sizeof(*meta));
        meta->chunks = s->scratch->allocate_array<pq::PqChunk>(
            pq::kPqMaxColumns * 16u);
        meta->chunks_cap = pq::kPqMaxColumns * 16u;
        if (!pq::parquet_read_meta(body, blen, s->scratch, meta)) {
            ++s->cur_file_idx; continue;
        }
        s->cur_meta = meta;
        s->cur_body = body;
        s->cur_body_len = blen;
        s->cur_row_group = 0;
        s->cur_file_open = true;
        return true;
    }
    return false;
}

}  // namespace

bool iceberg_scan_open(ScanHandle** out, TableHandle* h,
                       const ReadOptions* opts) noexcept {
    assert(out != nullptr && h != nullptr);
    if (!h->meta_loaded) return false;
    ScanHandle* s = h->arena->allocate_array<ScanHandle>(1);
    if (s == nullptr) return false;
    std::memset(s, 0, sizeof(*s));
    s->table = h;
    s->scratch = h->arena;
    if (opts != nullptr) s->opts = *opts; else read_options_init(&s->opts);
    if (!snapshot_resolve(&h->meta, s->opts.snapshot_id, s->opts.timestamp_ms,
                          &s->snap)) {
        s->live_files = h->arena->allocate_array<DataFileRef>(1u);
        s->n_live = 0;
        *out = s;
        return true;
    }
    ManifestListEntry* mlist =
        h->arena->allocate_array<ManifestListEntry>(kIcebergMaxManifestsPerList);
    if (mlist == nullptr) return false;
    uint32_t n_mlist = 0;
    {
        const char* mlist_path = s->snap.manifest_list;
        const char* mkey = strip_root(h->fs_root, mlist_path);
        const uint8_t* body = nullptr; uint64_t blen = 0;
        if (os_get(&h->os, mkey, h->arena, &body, &blen) != kOsOk) {
            char joined[kCatMaxPath];
            if (!(path_join(h->table_rel, mkey, joined, sizeof(joined)) &&
                  os_get(&h->os, joined, h->arena, &body, &blen) == kOsOk)) {
                return false;
            }
        }
        if (!manifest_list_parse_json(body, static_cast<uint32_t>(blen),
                                      h->arena, mlist,
                                      kIcebergMaxManifestsPerList, &n_mlist))
            return false;
    }
    s->live_files = h->arena->allocate_array<DataFileRef>(kMaxLiveFiles);
    if (s->live_files == nullptr) return false;
    PositionDeleteEntry* pd_buf =
        h->arena->allocate_array<PositionDeleteEntry>(kMaxPosDels);
    EqualityDeleteI64* ed_buf =
        h->arena->allocate_array<EqualityDeleteI64>(kMaxEqDels);
    if (pd_buf == nullptr || ed_buf == nullptr) return false;
    position_delete_set_init(&s->pos_dels, pd_buf, kMaxPosDels);
    equality_delete_set_init(&s->eq_dels, ed_buf, kMaxEqDels);

    const Schema* sch = metadata_current_schema(&h->meta);
    for (uint32_t mi = 0; mi < n_mlist; ++mi) {
        const ManifestListEntry& mle = mlist[mi];
        const char* mkey = strip_root(h->fs_root, mle.manifest_path);
        const uint8_t* body = nullptr; uint64_t blen = 0;
        if (os_get(&h->os, mkey, h->arena, &body, &blen) != kOsOk) {
            char joined[kCatMaxPath];
            if (!(path_join(h->table_rel, mkey, joined, sizeof(joined)) &&
                  os_get(&h->os, joined, h->arena, &body, &blen) == kOsOk)) {
                continue;
            }
        }
        DataFileRef* entries =
            h->arena->allocate_array<DataFileRef>(kIcebergMaxManifestEntries);
        if (entries == nullptr) return false;
        uint32_t n_entries = 0;
        if (!manifest_parse_json(body, static_cast<uint32_t>(blen), h->arena,
                                  mle.partition_spec_id, entries,
                                  kIcebergMaxManifestEntries, &n_entries))
            continue;
        const PartitionSpec* spec =
            metadata_spec(&h->meta, mle.partition_spec_id);
        for (uint32_t ei = 0; ei < n_entries; ++ei) {
            DataFileRef& e = entries[ei];
            if (e.status == ManifestStatus::kDeleted) continue;
            if (mle.content == FileContent::kEqualityDeletes ||
                e.content == FileContent::kEqualityDeletes ||
                e.content == FileContent::kPositionDeletes) {
                continue;   // TODO(W5-delete-load)
            }
            if (!partition_passes(&e, spec, sch,
                                  s->opts.predicates, s->opts.n_predicates))
                continue;
            if (!stats_pass(&e, sch,
                            s->opts.predicates, s->opts.n_predicates))
                continue;
            if (s->n_live >= kMaxLiveFiles) break;
            s->live_files[s->n_live++] = e;
        }
        if (s->n_live >= kMaxLiveFiles) break;
    }
    s->cur_file_idx = 0;
    s->cur_file_open = false;
    *out = s;
    return true;
}

bool iceberg_scan_next_batch(ScanHandle* s, BoltBatch* out,
                             bool* out_eof) noexcept {
    assert(s != nullptr && out != nullptr && out_eof != nullptr);
    *out_eof = false;
    BoltBatch::init_empty(out);
    out->arena = s->scratch;
    if (s->n_live == 0) { *out_eof = true; return true; }
    for (;;) {
        if (!s->cur_file_open) {
            if (!open_next_file(s)) { *out_eof = true; return true; }
        }
        if (s->cur_row_group >= s->cur_meta->n_row_groups ||
            s->cur_row_group >= kLakeMaxRowGroups) {
            s->cur_file_open = false;
            ++s->cur_file_idx;
            continue;
        }
        // G2FEAT-47: right-size columns[2] to this file's width before deref.
        if (!BoltBatch::alloc_columns(out, s->scratch, s->cur_meta->n_columns))
            return false;
        BoltColumn* cols = out->columns[out->read_epoch];
        int64_t rows = 0;
        if (!pq::parquet_read_row_group(s->cur_body, s->cur_body_len,
                                         s->cur_meta, s->cur_row_group,
                                         s->scratch, cols, &rows)) {
            s->cur_file_open = false;
            ++s->cur_file_idx;
            continue;
        }
        ++s->cur_row_group;
        out->num_rows = rows;
        out->num_cols = s->cur_meta->n_columns;
        for (uint32_t c = 0; c < s->cur_meta->n_columns; ++c) {
            const char* phys = s->cur_meta->columns[c].name;
            out->schema.add_field(phys, cols[c].type, true);
        }
        return true;
    }
}

void iceberg_scan_close(ScanHandle* /*s*/) noexcept {}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
