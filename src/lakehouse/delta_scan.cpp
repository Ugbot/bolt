// bolt/lakehouse/delta_scan.cpp — top-level Delta read path.

#include "bolt/lakehouse/handle.h"

#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_branchless.h"
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
    // G2ICE-80: the currently-open file's deletion vector (loaded once per
    // file-open) + how many of its rows this scan has already consumed —
    // the file-local row index a DV's bitmap is keyed on.
    bool               cur_has_dv;
    uint8_t            _pad[2];
    dl::DeletionVector cur_dv;
    uint64_t           cur_file_base_row;
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
        s->cur_file_base_row = 0;
        // G2ICE-80: a live file may carry a DV narrowing its visible rows
        // (e.g. attached by `delta_table_mark_deleted_via_dv` against a
        // cold-tier file with no physical rewrite). Load it once per file;
        // a load failure aborts the scan (never silently un-filters).
        s->cur_has_dv = f.has_dv;
        if (f.has_dv) {
            std::memset(&s->cur_dv, 0, sizeof(s->cur_dv));
            if (!dl::delta_dv_load(&s->table->os, s->table->table_rel, &f.dv,
                                   s->scratch, &s->cur_dv))
                return false;
        }
        return true;
    }
    return false;
}

// Gather dispatch (mirrors delta_delete.cpp's own copy — kept file-local so
// this read-only TU stays independent of the write TUs).
bool gather_pod_dispatch(const void* src, const int32_t* idx, int64_t n,
                         void* dst, uint32_t width) noexcept {
    assert(src != nullptr && dst != nullptr);
    assert(idx != nullptr || n == 0);
    switch (width) {
        case 1:  bolt::branchless::gather_pod<1>(src, idx, n, dst);  return true;
        case 2:  bolt::branchless::gather_pod<2>(src, idx, n, dst);  return true;
        case 4:  bolt::branchless::gather_pod<4>(src, idx, n, dst);  return true;
        case 8:  bolt::branchless::gather_pod<8>(src, idx, n, dst);  return true;
        case 16: bolt::branchless::gather_pod<16>(src, idx, n, dst); return true;
        case 32: bolt::branchless::gather_pod<32>(src, idx, n, dst); return true;
        default: return false;
    }
}

bool gather_column_for_dv(const BoltColumn& src, const int32_t* sel,
                          int64_t sel_n, Arena* arena,
                          BoltColumn* out) noexcept {
    assert(arena != nullptr && out != nullptr);
    assert(sel != nullptr || sel_n == 0);
    if (src.format != ColumnFormat::Flat) return false;
    const size_t tsz = bolt::type_size(src.type);
    if (tsz == 0) return false;
    BoltColumn c = BoltColumn::make_flat_alloc(sel_n, src.type, arena);
    if (sel_n > 0 && c.data == nullptr) return false;
    if (sel_n > 0 &&
        !gather_pod_dispatch(src.data, sel, sel_n, c.data,
                             static_cast<uint32_t>(tsz)))
        return false;
    if (src.validity != nullptr && !src.stats.all_valid && sel_n > 0) {
        const size_t vbytes = (static_cast<size_t>(sel_n) + 7u) / 8u;
        uint8_t* nval = static_cast<uint8_t*>(arena->allocate_zeroed(vbytes));
        if (nval == nullptr) return false;
        int64_t null_count = 0;
        for (int64_t i = 0; i < sel_n; ++i) {
            const int64_t srcbit = src.validity_offset + static_cast<int64_t>(sel[i]);
            const uint8_t bit = (src.validity[srcbit >> 3] >> (srcbit & 7)) & 1u;
            nval[i >> 3] |= static_cast<uint8_t>(bit << (i & 7));
            null_count += (1 - static_cast<int64_t>(bit));
        }
        c.validity = nval;
        c.validity_offset = 0;
        c.stats.null_count = static_cast<uint32_t>(null_count);
        c.stats.all_valid = (null_count == 0);
    }
    if (src.type == BoltType::Utf8) c.str_overflow_base = src.str_overflow_base;
    c.decimal_scale = src.decimal_scale;
    *out = c;
    return true;
}

// Filters `cols[0..n_cols)` (each with `rows` elements) down to the rows NOT
// covered by `dv`, keyed on `base_row + local_row_index`. Returns the new
// row count, or -1 on a column type this DV-gather path can't safely
// filter (caller must abort the scan rather than emit unfiltered rows).
int64_t apply_dv_to_row_group(const dl::DeletionVector* dv, uint64_t base_row,
                              BoltColumn* cols, uint32_t n_cols, int64_t rows,
                              Arena* arena) noexcept {
    assert(dv != nullptr && cols != nullptr);
    assert(arena != nullptr);
    if (!dv->present || rows <= 0) return rows;
    if (rows > 0x7FFFFFFFLL) return -1;   // bounded: keep[] indices are int32
    int32_t* keep = arena->allocate_array<int32_t>(static_cast<size_t>(rows));
    if (keep == nullptr) return -1;
    int64_t kept = 0;
    for (int64_t r = 0; r < rows; ++r) {   // bounded: rows
        if (!dl::delta_dv_contains(dv, base_row + static_cast<uint64_t>(r)))
            keep[kept++] = static_cast<int32_t>(r);
    }
    if (kept == rows) return rows;   // nothing in this row group is deleted
    for (uint32_t c = 0; c < n_cols; ++c) {   // bounded: n_cols
        BoltColumn gathered{};
        if (!gather_column_for_dv(cols[c], keep, kept, arena, &gathered))
            return -1;
        cols[c] = gathered;
    }
    return kept;
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
        const int64_t decoded_rows = rows;
        if (s->cur_has_dv) {
            rows = apply_dv_to_row_group(&s->cur_dv, s->cur_file_base_row,
                                         cols, s->cur_meta->n_columns, rows,
                                         s->scratch);
            if (rows < 0) return false;   // can't safely filter -- abort
        }
        s->cur_file_base_row += static_cast<uint64_t>(decoded_rows);
        ++s->cur_row_group;
        if (rows == 0) continue;   // every row in this group was deleted
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
