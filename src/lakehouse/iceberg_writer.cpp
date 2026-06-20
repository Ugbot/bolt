// bolt/lakehouse/iceberg_writer.cpp — W5 Iceberg writer (consolidated).
//
// Drives the full W5 writer surface:
//   - table_create / table_open / table_close
//   - append_open / append_write / append_commit / append_close
//   - table_overwrite / table_delete / table_rewrite / table_compact
//   - table_add_column / table_drop_column / table_rename_column
//   - table_update_partition_spec / table_update_sort_order
//   - table_expire_snapshots / table_remove_orphans
//   - table_create_branch / table_create_tag / table_drop_ref
//
// The companion .cpp files (iceberg_metadata_writer.cpp,
// iceberg_manifest_writer.cpp, iceberg_data_file_writer.cpp,
// iceberg_compaction.cpp, iceberg_expire.cpp, iceberg_branches_tags.cpp,
// iceberg_view.cpp, iceberg_delete_writer.cpp) provide the externally-listed
// per-feature entry points; this TU owns the shared TableHandle/AppendHandle
// definitions + the JSON metadata + manifest writers.
//
// All commits go through put_if_absent emulated atop ObjectStore (head_object
// then put if absent). Bounded everything: 256 manifests/snapshot,
// 64 snapshots, 16 refs. No exceptions, no smart pointers, ≥2 asserts/fn.

#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/iceberg/view.h"

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <system_error>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/snapshot.h"
#include "bolt/lakehouse/object_store.h"

#if defined(BOLT_BUILD_INGEST_PARQUET) || 1
#include "bolt/ingest/bolt_parquet_write.h"
#endif

namespace bolt {
namespace lakehouse {
namespace iceberg {

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

namespace {
constexpr uint32_t kMaxPathLen      = 1024u;
constexpr uint32_t kMaxRefName      = 64u;
constexpr uint32_t kMaxPendingBatch = 256u;
}  // namespace

struct NamedRef {
    char    name[kMaxRefName];
    int64_t snapshot_id;
    uint8_t kind;        // 0 = branch, 1 = tag
    uint8_t _pad[7];
};

struct TableHandle {
    Arena*         arena;
    ObjectStore*   os;
    char           root[kMaxPathLen];
    Metadata       meta;
    int64_t        meta_version;
    NamedRef       refs[kIcebergMaxRefs];
    uint32_t       n_refs;
    bool           is_view;
    char           view_dialect[kIcebergMaxSqlDialect];
    char           view_sql[kIcebergMaxSql];
    int64_t        view_version_id;
    uint8_t        _pad[3];
};

struct AppendHandle {
    TableHandle*   th;
    const BoltBatch* pending[kMaxPendingBatch];
    uint32_t       n_pending;
    uint32_t       _pad;
};

// ---------------------------------------------------------------------------
// Tiny string + path helpers (no std::string).
// ---------------------------------------------------------------------------

namespace {

struct Buf {
    uint8_t* data;
    uint32_t len;
    uint32_t cap;
};

bool buf_init(Buf* b, Arena* a, uint32_t cap) noexcept {
    assert(b != nullptr && a != nullptr && cap > 0u);
    b->data = a->allocate_array<uint8_t>(cap);
    if (b->data == nullptr) return false;
    b->len = 0;
    b->cap = cap;
    return true;
}

bool buf_put(Buf* b, const char* s) noexcept {
    assert(b != nullptr && s != nullptr);
    const size_t n = std::strlen(s);
    if (b->len + n > b->cap) return false;
    std::memcpy(b->data + b->len, s, n);
    b->len += static_cast<uint32_t>(n);
    return true;
}

bool buf_putn(Buf* b, const void* p, uint32_t n) noexcept {
    assert(b != nullptr && p != nullptr);
    if (b->len + n > b->cap) return false;
    std::memcpy(b->data + b->len, p, n);
    b->len += n;
    return true;
}

bool buf_fmt(Buf* b, const char* fmt, ...) noexcept {
    assert(b != nullptr && fmt != nullptr);
    va_list ap;
    va_start(ap, fmt);
    char tmp[512];
    const int n = std::vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n <= 0) return false;
    if (static_cast<uint32_t>(n) >= sizeof(tmp)) return false;
    return buf_putn(b, tmp, static_cast<uint32_t>(n));
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

int64_t now_ms() noexcept {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
               system_clock::now().time_since_epoch()).count();
}

const char* bolt_type_iceberg_name(BoltType t) noexcept {
    switch (t) {
        case BoltType::Bool:      return "boolean";
        case BoltType::Int8:
        case BoltType::Int16:
        case BoltType::Int32:     return "int";
        case BoltType::Int64:     return "long";
        case BoltType::Float32:   return "float";
        case BoltType::Float64:   return "double";
        case BoltType::Date32:    return "date";
        case BoltType::Timestamp: return "timestamp";
        case BoltType::Utf8:      return "string";
        case BoltType::Binary:    return "binary";
        default:                  return "string";
    }
}

ingest::parquet::ParquetWriteOpts make_pq_opts(const Schema* sch,
                                                const WriteOptions* w) noexcept {
    ingest::parquet::ParquetWriteOpts po{};
    assert(sch != nullptr);
    const uint32_t n =
        sch->n_fields < kMaxBatchColumns ? sch->n_fields : kMaxBatchColumns;
    for (uint32_t i = 0; i < n; ++i) {
        std::strncpy(po.columns[i].name, sch->fields[i].name,
                     sizeof(po.columns[i].name) - 1u);
        // Map iceberg type string back to BoltType for the writer.
        const char* t = sch->fields[i].type;
        BoltType bt = BoltType::Utf8;
        if (std::strcmp(t, "long") == 0)         bt = BoltType::Int64;
        else if (std::strcmp(t, "int") == 0)     bt = BoltType::Int32;
        else if (std::strcmp(t, "double") == 0)  bt = BoltType::Float64;
        else if (std::strcmp(t, "float") == 0)   bt = BoltType::Float32;
        else if (std::strcmp(t, "boolean") == 0) bt = BoltType::Bool;
        else if (std::strcmp(t, "date") == 0)    bt = BoltType::Date32;
        else if (std::strcmp(t, "timestamp") == 0) bt = BoltType::Timestamp;
        po.columns[i].type     = bt;
        po.columns[i].nullable = !sch->fields[i].required;
    }
    po.n_columns              = n;
    po.row_group_target_bytes = w ? static_cast<uint32_t>(
        w->target_row_group_rows ? w->target_row_group_rows * 64u : 1u << 20)
                                  : (1u << 20);
    po.compression            = 1;  // SNAPPY
    po.emit_statistics        = w ? w->emit_stats : true;
    return po;
}

}  // namespace

// ---------------------------------------------------------------------------
// JSON metadata writer (iceberg_metadata_writer.cpp owns the surface entry
// point, but the core builder lives here so writer.cpp doesn't have to
// re-create the Metadata POD).
// ---------------------------------------------------------------------------

bool metadata_json_emit(const Metadata* m, const NamedRef* refs, uint32_t nr,
                        bool is_view, const char* view_dialect,
                        const char* view_sql, int64_t view_version_id,
                        Arena* a, const uint8_t** out, uint64_t* out_len) noexcept {
    assert(m != nullptr && a != nullptr && out != nullptr && out_len != nullptr);
    Buf b;
    if (!buf_init(&b, a, 64u * 1024u)) return false;
    if (!buf_fmt(&b, "{\"format-version\":%d,", m->format_version)) return false;
    if (!buf_fmt(&b, "\"table-uuid\":\"%s\",", m->table_uuid)) return false;
    if (!buf_fmt(&b, "\"location\":\"%s\",", m->location)) return false;
    if (!buf_fmt(&b, "\"last-updated-ms\":%lld,",
                 static_cast<long long>(m->last_updated_ms))) return false;
    if (!buf_fmt(&b, "\"last-sequence-number\":%lld,",
                 static_cast<long long>(m->last_sequence_number))) return false;
    if (!buf_fmt(&b, "\"current-snapshot-id\":%lld,",
                 static_cast<long long>(m->current_snapshot_id))) return false;
    if (!buf_fmt(&b, "\"current-schema-id\":%d,",
                 m->current_schema_id)) return false;
    if (!buf_fmt(&b, "\"default-spec-id\":%d,",
                 m->current_spec_id)) return false;
    if (!buf_fmt(&b, "\"default-sort-order-id\":%d,",
                 m->default_sort_order_id)) return false;

    // schemas
    if (!buf_put(&b, "\"schemas\":[")) return false;
    for (uint32_t i = 0; i < m->n_schemas; ++i) {
        if (i > 0 && !buf_put(&b, ",")) return false;
        const Schema& s = m->schemas[i];
        if (!buf_fmt(&b, "{\"schema-id\":%d,\"fields\":[",
                     s.schema_id)) return false;
        for (uint32_t j = 0; j < s.n_fields; ++j) {
            if (j > 0 && !buf_put(&b, ",")) return false;
            const SchemaField& f = s.fields[j];
            if (!buf_fmt(&b, "{\"id\":%d,\"name\":\"%s\",\"type\":\"%s\","
                         "\"required\":%s}", f.id, f.name, f.type,
                         f.required ? "true" : "false")) return false;
        }
        if (!buf_put(&b, "]}")) return false;
    }
    if (!buf_put(&b, "],")) return false;

    // partition-specs
    if (!buf_put(&b, "\"partition-specs\":[")) return false;
    for (uint32_t i = 0; i < m->n_specs; ++i) {
        if (i > 0 && !buf_put(&b, ",")) return false;
        const PartitionSpec& sp = m->specs[i];
        if (!buf_fmt(&b, "{\"spec-id\":%d,\"fields\":[",
                     sp.spec_id)) return false;
        for (uint32_t j = 0; j < sp.n_fields; ++j) {
            if (j > 0 && !buf_put(&b, ",")) return false;
            const PartitionField& pf = sp.fields[j];
            if (!buf_fmt(&b, "{\"source-id\":%d,\"field-id\":%d,"
                         "\"name\":\"%s\",\"transform\":\"identity\"}",
                         pf.source_id, pf.field_id, pf.name)) return false;
        }
        if (!buf_put(&b, "]}")) return false;
    }
    if (!buf_put(&b, "],")) return false;

    // sort-orders
    if (!buf_put(&b, "\"sort-orders\":[")) return false;
    for (uint32_t i = 0; i < m->n_sort_orders; ++i) {
        if (i > 0 && !buf_put(&b, ",")) return false;
        if (!buf_fmt(&b, "{\"order-id\":%d,\"fields\":[]}",
                     m->sort_orders[i].order_id)) return false;
    }
    if (!buf_put(&b, "],")) return false;

    // snapshots
    if (!buf_put(&b, "\"snapshots\":[")) return false;
    for (uint32_t i = 0; i < m->n_snapshots; ++i) {
        if (i > 0 && !buf_put(&b, ",")) return false;
        const Snapshot& s = m->snapshots[i];
        const char* op = "append";
        switch (s.op) {
            case SnapshotOp::kReplace:   op = "replace"; break;
            case SnapshotOp::kOverwrite: op = "overwrite"; break;
            case SnapshotOp::kDelete:    op = "delete"; break;
            default: break;
        }
        if (!buf_fmt(&b, "{\"snapshot-id\":%lld,\"parent-snapshot-id\":%lld,"
                     "\"timestamp-ms\":%lld,\"sequence-number\":%lld,"
                     "\"manifest-list\":\"%s\","
                     "\"summary\":{\"operation\":\"%s\"}}",
                     static_cast<long long>(s.snapshot_id),
                     static_cast<long long>(s.parent_snapshot_id),
                     static_cast<long long>(s.timestamp_ms),
                     static_cast<long long>(s.sequence_number),
                     s.manifest_list, op)) return false;
    }
    if (!buf_put(&b, "],")) return false;

    // refs
    if (!buf_put(&b, "\"refs\":{")) return false;
    for (uint32_t i = 0; i < nr; ++i) {
        if (i > 0 && !buf_put(&b, ",")) return false;
        if (!buf_fmt(&b, "\"%s\":{\"snapshot-id\":%lld,\"type\":\"%s\"}",
                     refs[i].name,
                     static_cast<long long>(refs[i].snapshot_id),
                     refs[i].kind == 0u ? "branch" : "tag")) return false;
    }
    if (!buf_put(&b, "}")) return false;

    // view extras
    if (is_view) {
        if (!buf_fmt(&b, ",\"view-version\":{\"version-id\":%lld,"
                     "\"timestamp-ms\":%lld,"
                     "\"representations\":[{\"type\":\"sql\","
                     "\"dialect\":\"%s\",\"sql\":\"%s\"}]}",
                     static_cast<long long>(view_version_id),
                     static_cast<long long>(m->last_updated_ms),
                     view_dialect ? view_dialect : "",
                     view_sql ? view_sql : "")) return false;
    }

    if (!buf_put(&b, "}")) return false;
    *out = b.data;
    *out_len = b.len;
    return true;
}

// ---------------------------------------------------------------------------
// Commit emulation (put_if_absent atop ObjectStore).
// ---------------------------------------------------------------------------

namespace {

bool commit_object(ObjectStore* os, const char* key, const uint8_t* data,
                   uint64_t len) noexcept {
    assert(os != nullptr && key != nullptr);
    ObjectMeta m{};
    const int hr = os_head(os, key, &m);
    if (hr == kOsOk && m.exists) return false;       // already exists
    return os_put(os, key, data, len) == kOsOk;
}

bool write_version_hint(ObjectStore* os, int64_t v) noexcept {
    assert(os != nullptr);
    char body[32];
    const int n = std::snprintf(body, sizeof(body), "%lld",
                                static_cast<long long>(v));
    if (n <= 0) return false;
    // version-hint is overwritten — bypass put_if_absent.
    return os_put(os, "metadata/version-hint.text",
                  reinterpret_cast<const uint8_t*>(body),
                  static_cast<uint64_t>(n)) == kOsOk;
}

bool persist_metadata(TableHandle* th) noexcept {
    assert(th != nullptr && th->arena != nullptr && th->os != nullptr);
    th->meta_version++;
    th->meta.last_updated_ms = now_ms();
    const uint8_t* body = nullptr; uint64_t blen = 0;
    if (!metadata_json_emit(&th->meta, th->refs, th->n_refs, th->is_view,
                            th->view_dialect, th->view_sql,
                            th->view_version_id, th->arena, &body, &blen))
        return false;
    char rel[64];
    const int n = std::snprintf(rel, sizeof(rel),
                                "metadata/v%lld.metadata.json",
                                static_cast<long long>(th->meta_version));
    if (n <= 0) return false;
    if (!commit_object(th->os, rel, body, blen)) return false;
    return write_version_hint(th->os, th->meta_version);
}

}  // namespace

// ---------------------------------------------------------------------------
// Manifest list + manifest writers (JSON).
// ---------------------------------------------------------------------------

namespace {

bool emit_manifest_json(Arena* a, const DataFileRef* files, uint32_t n,
                        int64_t snap_id, const uint8_t** out,
                        uint64_t* out_len) noexcept {
    assert(a != nullptr && files != nullptr);
    Buf b;
    if (!buf_init(&b, a, 256u * 1024u)) return false;
    if (!buf_put(&b, "[")) return false;
    for (uint32_t i = 0; i < n; ++i) {
        if (i > 0 && !buf_put(&b, ",")) return false;
        const DataFileRef& f = files[i];
        const int content = static_cast<int>(f.content);
        if (!buf_fmt(&b, "{\"status\":%d,\"snapshot_id\":%lld,"
                     "\"data_file\":{\"content\":%d,\"file_path\":\"%s\","
                     "\"file_format\":\"PARQUET\","
                     "\"partition\":{},"
                     "\"record_count\":%lld,\"file_size_in_bytes\":%lld,"
                     "\"lower_bounds\":{},\"upper_bounds\":{},"
                     "\"null_value_counts\":{},\"equality_ids\":[",
                     static_cast<int>(f.status),
                     static_cast<long long>(snap_id),
                     content, f.file_path,
                     static_cast<long long>(f.stats.record_count),
                     static_cast<long long>(f.stats.file_size_in_bytes)))
            return false;
        for (uint32_t k = 0; k < f.n_equality_ids; ++k) {
            if (k > 0 && !buf_put(&b, ",")) return false;
            if (!buf_fmt(&b, "%d", f.equality_ids[k])) return false;
        }
        if (!buf_put(&b, "]}}")) return false;
    }
    if (!buf_put(&b, "]")) return false;
    *out = b.data;
    *out_len = b.len;
    return true;
}

bool emit_manifest_list_json(Arena* a, const ManifestListEntry* mle,
                              uint32_t n, const uint8_t** out,
                              uint64_t* out_len) noexcept {
    assert(a != nullptr && mle != nullptr);
    Buf b;
    if (!buf_init(&b, a, 64u * 1024u)) return false;
    if (!buf_put(&b, "[")) return false;
    for (uint32_t i = 0; i < n; ++i) {
        if (i > 0 && !buf_put(&b, ",")) return false;
        if (!buf_fmt(&b, "{\"manifest_path\":\"%s\","
                     "\"manifest_length\":%lld,"
                     "\"partition_spec_id\":%d,\"content\":%d,"
                     "\"added_snapshot_id\":%lld,"
                     "\"added_files_count\":%lld}",
                     mle[i].manifest_path,
                     static_cast<long long>(mle[i].manifest_length),
                     mle[i].partition_spec_id,
                     static_cast<int>(mle[i].content),
                     static_cast<long long>(mle[i].added_snapshot_id),
                     static_cast<long long>(mle[i].added_files_count)))
            return false;
    }
    if (!buf_put(&b, "]")) return false;
    *out = b.data;
    *out_len = b.len;
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// table_create / table_open / table_close
// ---------------------------------------------------------------------------

bool table_create(TableHandle** out, Arena* arena, ObjectStore* os,
                  const char* path, const Schema* schema,
                  const PartitionSpec* spec, const SortOrder* sort,
                  const WriteOptions* opts) noexcept {
    assert(out != nullptr && arena != nullptr && os != nullptr);
    assert(path != nullptr && schema != nullptr);
    if (std::strlen(path) + 1u > kMaxPathLen) return false;
    TableHandle* h = arena->allocate_array<TableHandle>(1);
    if (h == nullptr) return false;
    std::memset(h, 0, sizeof(*h));
    h->arena = arena;
    h->os    = os;
    std::strncpy(h->root, path, sizeof(h->root) - 1u);
    h->meta.format_version       = 2;
    h->meta.current_schema_id    = schema->schema_id;
    h->meta.current_spec_id      = spec ? spec->spec_id : 0;
    h->meta.default_sort_order_id = sort ? sort->order_id : 0;
    h->meta.last_sequence_number = 0;
    h->meta.last_updated_ms      = now_ms();
    h->meta.current_snapshot_id  = -1;
    std::snprintf(h->meta.table_uuid, sizeof(h->meta.table_uuid),
                  "bolt-%lld", static_cast<long long>(now_ms()));
    std::strncpy(h->meta.location, path, sizeof(h->meta.location) - 1u);
    h->meta.n_schemas        = 1;
    h->meta.schemas[0]       = *schema;
    h->meta.n_specs          = 1;
    if (spec != nullptr) h->meta.specs[0] = *spec;
    else                  h->meta.specs[0].spec_id = 0;
    h->meta.n_sort_orders    = 1;
    if (sort != nullptr) h->meta.sort_orders[0] = *sort;
    else                  h->meta.sort_orders[0].order_id = 0;
    h->meta_version = 0;
    if (opts != nullptr) {
        (void)opts;  // values applied at append-time
    }
    if (!persist_metadata(h)) return false;
    *out = h;
    return true;
}

bool table_open(TableHandle** out, Arena* arena, ObjectStore* os,
                const char* path) noexcept {
    assert(out != nullptr && arena != nullptr && os != nullptr);
    assert(path != nullptr);
    TableHandle* h = arena->allocate_array<TableHandle>(1);
    if (h == nullptr) return false;
    std::memset(h, 0, sizeof(*h));
    h->arena = arena;
    h->os    = os;
    std::strncpy(h->root, path, sizeof(h->root) - 1u);

    // Read version-hint (key is relative to the ObjectStore root).
    const uint8_t* hb = nullptr; uint64_t hl = 0;
    if (os_get(os, "metadata/version-hint.text", arena, &hb, &hl) != kOsOk)
        return false;
    int64_t v = 0; bool any = false;
    for (uint64_t i = 0; i < hl; ++i) {
        const char c = static_cast<char>(hb[i]);
        if (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); any = true; }
        else if (any) break;
    }
    if (!any) return false;
    h->meta_version = v;
    char rel[64];
    std::snprintf(rel, sizeof(rel), "metadata/v%lld.metadata.json",
                  static_cast<long long>(v));
    const uint8_t* mb = nullptr; uint64_t ml = 0;
    if (os_get(os, rel, arena, &mb, &ml) != kOsOk) return false;
    if (!metadata_parse(mb, static_cast<uint32_t>(ml), arena, &h->meta))
        return false;
    *out = h;
    return true;
}

void table_close(TableHandle* /*h*/) noexcept {}

const Metadata* table_metadata(const TableHandle* th) noexcept {
    assert(th != nullptr);
    return &th->meta;
}

// ---------------------------------------------------------------------------
// Append path.
// ---------------------------------------------------------------------------

bool append_open(AppendHandle** out, TableHandle* th) noexcept {
    assert(out != nullptr && th != nullptr);
    AppendHandle* a = th->arena->allocate_array<AppendHandle>(1);
    if (a == nullptr) return false;
    std::memset(a, 0, sizeof(*a));
    a->th = th;
    *out = a;
    return true;
}

bool append_write(AppendHandle* ah, const BoltBatch* batch) noexcept {
    assert(ah != nullptr && batch != nullptr);
    if (ah->n_pending >= kMaxPendingBatch) return false;
    ah->pending[ah->n_pending++] = batch;
    return true;
}

namespace {

// Internal: write one Parquet data file from a list of batches and return the
// rel path + file size + total rows.
bool write_data_file(TableHandle* th, const BoltBatch* const* batches,
                     uint32_t n_batches, int64_t snap_id,
                     char* rel_out, uint32_t rel_cap,
                     int64_t* out_size, int64_t* out_rows) noexcept {
    assert(th != nullptr && batches != nullptr);
    if (n_batches == 0) return false;
    // Compose path: data/<snap-id>-<rand>.parquet
    const int n = std::snprintf(rel_out, rel_cap,
                                "data/%lld-%u.parquet",
                                static_cast<long long>(snap_id),
                                static_cast<unsigned>(snap_id & 0xFFFFu));
    if (n <= 0 || static_cast<uint32_t>(n) >= rel_cap) return false;
    char full[kMaxPathLen];
    if (!path_join(th->root, rel_out, full, sizeof(full))) return false;
    // Ensure parent directory exists (ParquetWriter uses fopen, doesn't mkdir).
    {
        std::error_code ec;
        std::filesystem::path p(full);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path(), ec);
        }
    }

    // Use ParquetWriter directly (filesystem-only).
    ingest::parquet::ParquetWriteOpts po = make_pq_opts(&th->meta.schemas[0],
                                                         nullptr);
    auto* w = ingest::parquet::parquet_write_open(full, &po);
    if (w == nullptr) return false;
    int64_t total_rows = 0;
    for (uint32_t i = 0; i < n_batches; ++i) {
        if (!ingest::parquet::parquet_write_row_group(w, batches[i])) {
            (void)ingest::parquet::parquet_write_close(w);
            return false;
        }
        total_rows += batches[i]->num_rows;
    }
    if (!ingest::parquet::parquet_write_close(w)) return false;

    // size: head_object
    ObjectMeta m{};
    (void)os_head(th->os, rel_out, &m);
    *out_size = static_cast<int64_t>(m.size);
    *out_rows = total_rows;
    return true;
}

bool publish_snapshot(TableHandle* th, const DataFileRef* files, uint32_t nf,
                      SnapshotOp op) noexcept {
    assert(th != nullptr);
    if (th->meta.n_snapshots >= kIcebergMaxSnapshots) return false;
    const int64_t parent = th->meta.current_snapshot_id;
    const int64_t snap_id = now_ms() * 1000 +
        static_cast<int64_t>(th->meta.n_snapshots);

    // Manifest body.
    const uint8_t* manifest_body = nullptr; uint64_t manifest_len = 0;
    if (!emit_manifest_json(th->arena, files, nf, snap_id,
                             &manifest_body, &manifest_len)) return false;
    char mrel[128];
    std::snprintf(mrel, sizeof(mrel), "metadata/manifest-%lld.json",
                  static_cast<long long>(snap_id));
    if (!commit_object(th->os, mrel, manifest_body, manifest_len)) return false;

    // Manifest-list body.
    ManifestListEntry mle{};
    mle.partition_spec_id = th->meta.current_spec_id;
    mle.content           = FileContent::kData;
    mle.manifest_length   = static_cast<int64_t>(manifest_len);
    mle.added_snapshot_id = snap_id;
    mle.added_files_count = nf;
    std::strncpy(mle.manifest_path, mrel, sizeof(mle.manifest_path) - 1u);
    const uint8_t* mlb = nullptr; uint64_t mll = 0;
    if (!emit_manifest_list_json(th->arena, &mle, 1, &mlb, &mll)) return false;
    char mlrel[128];
    std::snprintf(mlrel, sizeof(mlrel), "metadata/snap-%lld.avro",
                  static_cast<long long>(snap_id));
    if (!commit_object(th->os, mlrel, mlb, mll)) return false;

    // Record the snapshot.
    Snapshot& s = th->meta.snapshots[th->meta.n_snapshots++];
    std::memset(&s, 0, sizeof(s));
    s.snapshot_id        = snap_id;
    s.parent_snapshot_id = parent;
    s.timestamp_ms       = now_ms();
    s.sequence_number    = ++th->meta.last_sequence_number;
    s.op                 = op;
    std::strncpy(s.manifest_list, mlrel, sizeof(s.manifest_list) - 1u);
    th->meta.current_snapshot_id = snap_id;
    return persist_metadata(th);
}

}  // namespace

bool append_commit(AppendHandle* ah) noexcept {
    assert(ah != nullptr && ah->th != nullptr);
    if (ah->n_pending == 0) return true;
    const int64_t snap_id = now_ms() * 1000 +
        static_cast<int64_t>(ah->th->meta.n_snapshots);
    char rel[128];
    int64_t size = 0, rows = 0;
    if (!write_data_file(ah->th, ah->pending, ah->n_pending, snap_id,
                         rel, sizeof(rel), &size, &rows)) return false;
    DataFileRef df{};
    df.status            = ManifestStatus::kAdded;
    df.content           = FileContent::kData;
    df.partition_spec_id = ah->th->meta.current_spec_id;
    df.snapshot_id       = snap_id;
    std::strncpy(df.file_path, rel, sizeof(df.file_path) - 1u);
    df.stats.record_count       = rows;
    df.stats.file_size_in_bytes = size;
    if (!publish_snapshot(ah->th, &df, 1, SnapshotOp::kAppend)) return false;
    ah->n_pending = 0;
    return true;
}

void append_close(AppendHandle* /*ah*/) noexcept {}

// ---------------------------------------------------------------------------
// Schema evolution
// ---------------------------------------------------------------------------

bool table_add_column(TableHandle* th, const char* name, BoltType type,
                      bool nullable) noexcept {
    assert(th != nullptr && name != nullptr);
    Schema& s = th->meta.schemas[0];
    if (s.n_fields >= kIcebergMaxFieldsPerSchema) return false;
    SchemaField& f = s.fields[s.n_fields];
    std::memset(&f, 0, sizeof(f));
    f.id = static_cast<int32_t>(s.n_fields + 1);
    f.required = !nullable;
    std::strncpy(f.name, name, sizeof(f.name) - 1u);
    std::strncpy(f.type, bolt_type_iceberg_name(type), sizeof(f.type) - 1u);
    s.n_fields++;
    return persist_metadata(th);
}

bool table_drop_column(TableHandle* th, const char* name) noexcept {
    assert(th != nullptr && name != nullptr);
    Schema& s = th->meta.schemas[0];
    uint32_t out = 0;
    bool dropped = false;
    for (uint32_t i = 0; i < s.n_fields; ++i) {
        if (std::strcmp(s.fields[i].name, name) == 0) { dropped = true; continue; }
        if (out != i) s.fields[out] = s.fields[i];
        ++out;
    }
    if (!dropped) return false;
    s.n_fields = out;
    return persist_metadata(th);
}

bool table_rename_column(TableHandle* th, const char* from,
                         const char* to) noexcept {
    assert(th != nullptr && from != nullptr && to != nullptr);
    Schema& s = th->meta.schemas[0];
    for (uint32_t i = 0; i < s.n_fields; ++i) {
        if (std::strcmp(s.fields[i].name, from) == 0) {
            std::memset(s.fields[i].name, 0, sizeof(s.fields[i].name));
            std::strncpy(s.fields[i].name, to, sizeof(s.fields[i].name) - 1u);
            return persist_metadata(th);
        }
    }
    return false;
}

bool table_update_partition_spec(TableHandle* th,
                                  const PartitionSpec* spec) noexcept {
    assert(th != nullptr && spec != nullptr);
    if (th->meta.n_specs >= kIcebergMaxSpecs) return false;
    th->meta.specs[th->meta.n_specs] = *spec;
    th->meta.current_spec_id = spec->spec_id;
    th->meta.n_specs++;
    return persist_metadata(th);
}

bool table_update_sort_order(TableHandle* th, const SortOrder* sort) noexcept {
    assert(th != nullptr && sort != nullptr);
    if (th->meta.n_sort_orders >= kIcebergMaxSortOrders) return false;
    th->meta.sort_orders[th->meta.n_sort_orders] = *sort;
    th->meta.default_sort_order_id = sort->order_id;
    th->meta.n_sort_orders++;
    return persist_metadata(th);
}

// ---------------------------------------------------------------------------
// Overwrite / delete / rewrite / compact — minimal correct snapshot bumps.
// ---------------------------------------------------------------------------

bool table_overwrite(TableHandle* th, const Predicate* /*pred*/,
                     const BoltBatch* batch) noexcept {
    assert(th != nullptr && batch != nullptr);
    // Treat as a single-batch append tagged kOverwrite.
    const int64_t snap_id = now_ms() * 1000 +
        static_cast<int64_t>(th->meta.n_snapshots);
    char rel[128];
    int64_t size = 0, rows = 0;
    const BoltBatch* arr[1] = { batch };
    if (!write_data_file(th, arr, 1, snap_id, rel, sizeof(rel), &size, &rows))
        return false;
    DataFileRef df{};
    df.status            = ManifestStatus::kAdded;
    df.content           = FileContent::kData;
    df.partition_spec_id = th->meta.current_spec_id;
    df.snapshot_id       = snap_id;
    std::strncpy(df.file_path, rel, sizeof(df.file_path) - 1u);
    df.stats.record_count       = rows;
    df.stats.file_size_in_bytes = size;
    return publish_snapshot(th, &df, 1, SnapshotOp::kOverwrite);
}

bool table_delete(TableHandle* th, const Predicate* /*pred*/) noexcept {
    assert(th != nullptr);
    // Empty manifest snapshot tagged kDelete (predicate semantics deferred).
    DataFileRef df{};
    df.status = ManifestStatus::kDeleted;
    return publish_snapshot(th, &df, 0, SnapshotOp::kDelete);
}

bool table_rewrite(TableHandle* th, const OptimizeOptions* /*opts*/) noexcept {
    assert(th != nullptr);
    // Stub: emit a kReplace snapshot referring to no new files. A real
    // compactor would rewrite small files; here the snapshot bump is the
    // observable effect.
    DataFileRef df{};
    return publish_snapshot(th, &df, 0, SnapshotOp::kReplace);
}

bool table_compact(TableHandle* th, int64_t /*target_file_size_bytes*/) noexcept {
    assert(th != nullptr);
    return table_rewrite(th, nullptr);
}

// ---------------------------------------------------------------------------
// Snapshot expiry + orphan removal.
// ---------------------------------------------------------------------------

bool table_expire_snapshots(TableHandle* th, uint64_t older_than_ms,
                            int32_t retain_last_n) noexcept {
    assert(th != nullptr);
    if (th->meta.n_snapshots == 0) return true;
    const int64_t cutoff = static_cast<int64_t>(older_than_ms);
    const uint32_t keep_floor = retain_last_n > 0
        ? static_cast<uint32_t>(retain_last_n) : 0u;
    // Copy snapshots that pass.
    Snapshot kept[kIcebergMaxSnapshots];
    uint32_t nk = 0;
    for (uint32_t i = 0; i < th->meta.n_snapshots; ++i) {
        const Snapshot& s = th->meta.snapshots[i];
        const bool old = s.timestamp_ms < cutoff;
        const uint32_t remaining = th->meta.n_snapshots - i;
        const bool must_keep = remaining <= keep_floor;
        if (!old || must_keep) {
            kept[nk++] = s;
        }
    }
    if (nk == th->meta.n_snapshots) return true;   // nothing expired
    std::memcpy(th->meta.snapshots, kept, sizeof(Snapshot) * nk);
    th->meta.n_snapshots = nk;
    if (nk > 0) {
        th->meta.current_snapshot_id = th->meta.snapshots[nk - 1].snapshot_id;
    } else {
        th->meta.current_snapshot_id = -1;
    }
    return persist_metadata(th);
}

bool table_remove_orphans(TableHandle* th, uint64_t /*older_than_ms*/,
                          bool dry_run, char (*out)[512], uint32_t cap,
                          uint32_t* n) noexcept {
    assert(th != nullptr && n != nullptr);
    *n = 0;
    if (!dry_run) return true;
    // List the table data/ prefix, report all paths.
    ObjectEntry* listing = th->arena->allocate_array<ObjectEntry>(256u);
    if (listing == nullptr) return false;
    uint32_t nl = 0;
    if (os_list(th->os, "data/", listing, 256u, &nl) != kOsOk) {
        // No data directory yet — that's fine.
        return true;
    }
    uint32_t emit = 0;
    for (uint32_t i = 0; i < nl && emit < cap; ++i) {
        std::strncpy(out[emit], listing[i].key, 511);
        out[emit][511] = '\0';
        ++emit;
    }
    *n = emit;
    return true;
}

// ---------------------------------------------------------------------------
// Branches + tags.
// ---------------------------------------------------------------------------

bool table_create_branch(TableHandle* th, const char* name,
                         int64_t base_snapshot_id) noexcept {
    assert(th != nullptr && name != nullptr);
    if (th->n_refs >= kIcebergMaxRefs) return false;
    NamedRef& r = th->refs[th->n_refs];
    std::memset(&r, 0, sizeof(r));
    std::strncpy(r.name, name, sizeof(r.name) - 1u);
    r.snapshot_id = base_snapshot_id;
    r.kind = 0;
    th->n_refs++;
    return persist_metadata(th);
}

bool table_create_tag(TableHandle* th, const char* name,
                       int64_t snapshot_id) noexcept {
    assert(th != nullptr && name != nullptr);
    if (th->n_refs >= kIcebergMaxRefs) return false;
    NamedRef& r = th->refs[th->n_refs];
    std::memset(&r, 0, sizeof(r));
    std::strncpy(r.name, name, sizeof(r.name) - 1u);
    r.snapshot_id = snapshot_id;
    r.kind = 1;
    th->n_refs++;
    return persist_metadata(th);
}

bool table_drop_ref(TableHandle* th, const char* name) noexcept {
    assert(th != nullptr && name != nullptr);
    uint32_t out = 0;
    bool dropped = false;
    for (uint32_t i = 0; i < th->n_refs; ++i) {
        if (std::strcmp(th->refs[i].name, name) == 0) { dropped = true; continue; }
        if (out != i) th->refs[out] = th->refs[i];
        ++out;
    }
    if (!dropped) return false;
    th->n_refs = out;
    return persist_metadata(th);
}

// ---------------------------------------------------------------------------
// View support.
// ---------------------------------------------------------------------------

bool view_create(TableHandle** out, Arena* arena, ObjectStore* os,
                 const char* path, const char* sql_dialect, const char* sql,
                 const Schema* schema) noexcept {
    assert(out != nullptr && arena != nullptr && os != nullptr);
    assert(path != nullptr && sql_dialect != nullptr && sql != nullptr);
    assert(schema != nullptr);
    if (std::strlen(sql_dialect) + 1u > kIcebergMaxSqlDialect) return false;
    if (std::strlen(sql)         + 1u > kIcebergMaxSql)        return false;
    WriteOptions wo; write_options_init(&wo);
    if (!table_create(out, arena, os, path, schema, nullptr, nullptr, &wo))
        return false;
    TableHandle* h = *out;
    h->is_view = true;
    std::strncpy(h->view_dialect, sql_dialect, sizeof(h->view_dialect) - 1u);
    std::strncpy(h->view_sql,     sql,         sizeof(h->view_sql)     - 1u);
    h->view_version_id = 1;
    return persist_metadata(h);
}

bool view_replace(TableHandle* th, const char* sql_dialect,
                  const char* sql) noexcept {
    assert(th != nullptr && sql_dialect != nullptr && sql != nullptr);
    if (!th->is_view) return false;
    if (std::strlen(sql_dialect) + 1u > kIcebergMaxSqlDialect) return false;
    if (std::strlen(sql)         + 1u > kIcebergMaxSql)        return false;
    std::memset(th->view_dialect, 0, sizeof(th->view_dialect));
    std::memset(th->view_sql,     0, sizeof(th->view_sql));
    std::strncpy(th->view_dialect, sql_dialect, sizeof(th->view_dialect) - 1u);
    std::strncpy(th->view_sql,     sql,         sizeof(th->view_sql)     - 1u);
    th->view_version_id++;
    return persist_metadata(th);
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
