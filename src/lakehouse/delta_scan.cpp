// bolt/lakehouse/delta_scan.cpp — top-level Delta read path.

#include "bolt/lakehouse/handle.h"

#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/lakehouse/delta/column_mapping.h"
#include "bolt/lakehouse/delta/deletion_vector.h"
#include "bolt/lakehouse/delta/generated_column.h"
#include "bolt/lakehouse/delta/log.h"
#include "bolt/lakehouse/delta/snapshot.h"
#include "delta_write_internal.h"

namespace bolt {
namespace lakehouse {

namespace dl = bolt::lakehouse::delta;
namespace pq = bolt::ingest::parquet;

namespace {
constexpr uint32_t kMaxFsRootInTable = delta_writer_detail::kMaxFsRootInTable;
constexpr uint32_t kMaxNsName        = delta_writer_detail::kMaxNsName;
}  // namespace

struct ScanHandle {
    TableHandle*       table;
    Arena*             scratch;
    dl::Snapshot       snapshot;
    dl::ColumnMap      col_map;
    dl::GeneratedColumnSet gen_cols;
    ReadOptions        opts;
    uint32_t           cursor;
    uint32_t*          live_idx;
    uint32_t           n_live;
    pq::PqMeta*        cur_meta;
    const uint8_t*     cur_body;
    uint64_t           cur_body_len;
    uint32_t           cur_row_group;
    bool               cur_file_open;
    uint8_t            _pad[3];
};

namespace {

bool join_ns(const char* ns, const char* name, char* out, uint32_t cap) noexcept {
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
                            ObjectStore* os) noexcept {
    assert(cat != nullptr && fs != nullptr && os != nullptr);
    char tp[kCatMaxPath];
    if (cat_table_path(cat, ns, name, tp, sizeof(tp)) != kCatOk) return false;
    char suffix[kMaxNsName * 2u + 4u];
    if (!join_ns(ns, name, suffix, sizeof(suffix))) return false;
    const size_t tl = std::strlen(tp);
    const size_t sl = std::strlen(suffix);
    if (tl <= sl + 1u) return false;
    if (std::strcmp(tp + tl - sl, suffix) != 0) return false;
    char root[kCatMaxPath];
    const size_t root_len = tl - sl - 1u;
    if (root_len + 1u > sizeof(root)) return false;
    std::memcpy(root, tp, root_len);
    root[root_len] = '\0';
    return filesystem_object_store_init(fs, root, os);
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

}  // namespace

bool delta_table_open(TableHandle** out, Arena* arena, Catalog* catalog,
                      const char* namespace_, const char* name) noexcept {
    assert(out != nullptr && arena != nullptr);
    assert(catalog != nullptr && namespace_ != nullptr && name != nullptr);
    TableHandle* h = arena->allocate_array<TableHandle>(1);
    if (h == nullptr) return false;
    std::memset(h, 0, sizeof(*h));
    h->arena = arena;
    h->catalog = catalog;
    if (std::strlen(namespace_) + 1u > sizeof(h->namespace_)) return false;
    if (std::strlen(name) + 1u > sizeof(h->name)) return false;
    std::strncpy(h->namespace_, namespace_, sizeof(h->namespace_) - 1u);
    std::strncpy(h->name, name, sizeof(h->name) - 1u);
    if (!join_ns(namespace_, name, h->table_rel, sizeof(h->table_rel)))
        return false;
    if (!bootstrap_object_store(catalog, namespace_, name,
                                 &h->fs_store, &h->os))
        return false;
    h->os_ready = true;
    *out = h;
    return true;
}

void delta_table_close(TableHandle* /*h*/) noexcept {}

bool delta_scan_open(ScanHandle** out, TableHandle* h,
                     const ReadOptions* opts) noexcept {
    assert(out != nullptr && h != nullptr);
    ScanHandle* s = h->arena->allocate_array<ScanHandle>(1);
    if (s == nullptr) return false;
    std::memset(s, 0, sizeof(*s));
    s->table = h;
    s->scratch = h->arena;
    if (opts != nullptr) s->opts = *opts;
    else                 read_options_init(&s->opts);
    int64_t max_version = s->opts.snapshot_id;
    if (s->opts.timestamp_ms >= 0) {
        int64_t v = -1;
        if (!dl::delta_log_version_for_timestamp(&h->os, h->table_rel,
                                                  s->opts.timestamp_ms,
                                                  h->arena, &v)) return false;
        max_version = v;
    }
    if (!dl::delta_snapshot_build(&h->os, h->table_rel, max_version,
                                   h->arena, &s->snapshot))
        return false;
    if (s->snapshot.has_metadata) {
        dl::delta_column_map_build(&s->snapshot.metadata, h->arena, &s->col_map);
        dl::delta_generated_cols_build(&s->snapshot.metadata, h->arena,
                                        &s->gen_cols);
    }
    const uint32_t live_alloc = s->snapshot.n_files == 0 ? 1u : s->snapshot.n_files;
    s->live_idx = h->arena->allocate_array<uint32_t>(live_alloc);
    if (s->live_idx == nullptr) return false;
    s->n_live = 0;
    for (uint32_t i = 0; i < s->snapshot.n_files; ++i) {
        if (dl::delta_file_passes(&s->snapshot.files[i],
                                   s->opts.predicates,
                                   s->opts.n_predicates)) {
            s->live_idx[s->n_live++] = i;
        }
    }
    s->cursor = 0;
    s->cur_file_open = false;
    *out = s;
    return true;
}

namespace {

bool open_next_file(ScanHandle* s) noexcept {
    assert(s != nullptr);
    while (s->cursor < s->n_live) {
        const dl::LiveFile& f = s->snapshot.files[s->live_idx[s->cursor]];
        char key[dl::kDeltaMaxPath];
        if (!path_join(s->table->table_rel, f.path, key, sizeof(key))) {
            ++s->cursor; continue;
        }
        const uint8_t* body = nullptr;
        uint64_t blen = 0;
        if (os_get(&s->table->os, key, s->scratch, &body, &blen) != kOsOk) {
            ++s->cursor; continue;
        }
        pq::PqMeta* meta = s->scratch->allocate_array<pq::PqMeta>(1);
        if (meta == nullptr) return false;
        std::memset(meta, 0, sizeof(*meta));
        if (!pq::parquet_read_meta(body, blen, s->scratch, meta)) {
            ++s->cursor; continue;
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

bool delta_scan_next_batch(ScanHandle* s, BoltBatch* out,
                           bool* out_eof) noexcept {
    assert(s != nullptr && out != nullptr && out_eof != nullptr);
    *out_eof = false;
    BoltBatch::init_empty(out);
    out->arena = s->scratch;
    for (;;) {
        if (!s->cur_file_open) {
            if (!open_next_file(s)) {
                *out_eof = true;
                return true;
            }
        }
        if (s->cur_row_group >= s->cur_meta->n_row_groups ||
            s->cur_row_group >= kLakeMaxRowGroups) {
            s->cur_file_open = false;
            ++s->cursor;
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
            ++s->cursor;
            continue;
        }
        ++s->cur_row_group;
        out->num_rows = rows;
        out->num_cols = s->cur_meta->n_columns;
        for (uint32_t c = 0; c < s->cur_meta->n_columns; ++c) {
            const char* phys = s->cur_meta->columns[c].name;
            const char* logical = phys;
            if (s->col_map.mode != dl::ColumnMappingMode::kNone) {
                const char* m = dl::delta_column_map_logical(&s->col_map, phys);
                if (m != nullptr) logical = m;
            }
            out->schema.add_field(logical, cols[c].type, true);
        }
        return true;
    }
}

void delta_scan_close(ScanHandle* /*s*/) noexcept {}

}  // namespace lakehouse
}  // namespace bolt
