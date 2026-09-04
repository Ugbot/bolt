// bolt/lakehouse/iceberg_scan.cpp — top-level Iceberg read path.
//
// metadata → snapshot → manifest-list → manifest → per-file Parquet decode.
// Opaque TableHandle / ScanHandle live in the `bolt::lakehouse::iceberg`
// sub-namespace so they don't collide with Delta's.
//
// G2FEAT-125 — this path now reads a REAL Iceberg table, not just a
// JSON-shaped stand-in. Three things a real table needs that the W4 path did
// not have, each proven against a pyiceberg-0.11.1 warehouse committed at
// tests/data/golden_iceberg_table (see test_bolt_iceberg_real_scan.cpp):
//
//   1. AVRO manifests. G2FEAT-76 landed the decoder but nothing called it.
//      `read_ref` sniffs the Avro object-container magic ("Obj\x01") and
//      dispatches to `manifest_{list_,}parse_avro`, falling back to the JSON
//      entry points for the W4 fixtures. Format detection is by CONTENT, not
//      by file extension or a build flag, so one table may mix them.
//
//   2. CATALOG-MANAGED metadata names. `find_latest_metadata` used to require
//      `version-hint.text` + `v<N>.metadata.json` — the Hadoop-catalog layout.
//      A pyiceberg SqlCatalog (and REST, Glue, Hive) writes NEITHER: there is
//      no version hint and the files are `<NNNNN>-<uuid>.metadata.json`. Both
//      namings are now recognised, ranked by version number.
//
//   3. RELOCATION. Every path Iceberg records — the snapshot's manifest list,
//      the manifest-list's manifest paths, the manifest's data-file paths — is
//      an ABSOLUTE URI naming wherever the writer stood. A table read anywhere
//      else (restored from backup, copied between buckets, or, here, checked
//      out of git) must rebase them. `read_ref` strips the metadata
//      `location` prefix and resolves the remainder under the table's actual
//      directory, then falls back to the older strip-root / join-table-rel
//      attempts so existing callers are unaffected.
//
// Still declined, loudly and never silently:
//   - Position/equality delete files are skipped, so a table carrying them
//     would over-report rows — `iceberg_scan_open` therefore FAILS on one
//     rather than returning a wrong answer. TODO(W5-delete-load).
//   - A live data file that cannot be fetched or decoded fails the scan. It
//     used to be skipped, which silently returned short.
//   - Parquet codecs outside Snappy/uncompressed (notably zstd, pyiceberg's
//     default) fail at decode via the rule above — never a short read.
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

// ---------------------------------------------------------------------------
// Relocation: an Iceberg path is an absolute URI naming where the WRITER
// stood, so reading the table anywhere else has to rebase it.
// ---------------------------------------------------------------------------

// Drop a leading "<scheme>://". `file:///C:/x` additionally leaves a spurious
// '/' before the drive letter; `file://C:/x` (what pyiceberg writes) does not.
// Returns a pointer INTO `p`, never null, never past the terminator.
const char* strip_scheme(const char* p) noexcept {
    assert(p != nullptr);
    const char* sep = std::strstr(p, "://");
    if (sep == nullptr) return p;
    const char* q = sep + 3;
    // "/C:/..." -> "C:/..." — only for a real drive letter, so a POSIX
    // "file:///tmp/x" keeps its leading slash.
    if (q[0] == '/' && q[1] != '\0' && q[2] == ':') ++q;
    assert(q >= p);
    return q;
}

// Path-equality-insensitive to separator flavour, so a '\'-joined root still
// matches a '/'-written URI.
bool path_prefix_len(const char* path, const char* prefix,
                     uint32_t* out_len) noexcept {
    assert(path != nullptr && prefix != nullptr && out_len != nullptr);
    uint32_t i = 0;
    for (; prefix[i] != '\0'; ++i) {          // bounded: prefix is NUL-terminated
        const char a = path[i] == '\\' ? '/' : path[i];
        const char b = prefix[i] == '\\' ? '/' : prefix[i];
        if (a == '\0' || a != b) return false;
    }
    // Must land on a boundary, so "/a/tablex" never matches prefix "/a/table".
    if (path[i] != '\0' && path[i] != '/' && path[i] != '\\') return false;
    *out_len = i;
    return true;
}

// Rebase `path` against the table's recorded `location` and produce an object
// key relative to the store root. False when the location does not cover it.
bool key_from_location(const char* location, const char* table_rel,
                       const char* path, char* out, uint32_t cap) noexcept {
    assert(location != nullptr && table_rel != nullptr);
    assert(path != nullptr && out != nullptr);
    if (location[0] == '\0') return false;
    const char* loc = strip_scheme(location);
    const char* p   = strip_scheme(path);
    uint32_t n = 0;
    if (!path_prefix_len(p, loc, &n)) return false;
    const char* rel = p + n;
    while (*rel == '/' || *rel == '\\') ++rel;
    if (*rel == '\0') return false;
    return path_join(table_rel, rel, out, cap);
}

// Read whatever an Iceberg path points at, trying each rebasing in turn:
// location-relative first (a relocated table), then the historical
// strip-root and join-table-rel attempts (unchanged for W4 callers).
bool read_ref(ObjectStore* os, const char* location, const char* fs_root,
              const char* table_rel, const char* path, Arena* scratch,
              const uint8_t** out_body, uint64_t* out_len) noexcept {
    assert(os != nullptr && path != nullptr);
    assert(out_body != nullptr && out_len != nullptr);
    char key[kCatMaxPath];
    if (key_from_location(location, table_rel, path, key, sizeof(key)) &&
        os_get(os, key, scratch, out_body, out_len) == kOsOk) {
        return true;
    }
    const char* stripped = strip_root(fs_root, path);
    if (os_get(os, stripped, scratch, out_body, out_len) == kOsOk) return true;
    if (path_join(table_rel, stripped, key, sizeof(key)) &&
        os_get(os, key, scratch, out_body, out_len) == kOsOk) {
        return true;
    }
    return false;
}

// Avro object-container magic. Format is detected from CONTENT so one table
// may hold both forms (and so no build flag decides how bytes are read).
bool is_avro_ocf(const uint8_t* b, uint64_t n) noexcept {
    return b != nullptr && n >= 4u &&
           b[0] == 'O' && b[1] == 'b' && b[2] == 'j' && b[3] == 1u;
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

// Version number encoded in a metadata file NAME, across both layouts a real
// warehouse uses:
//   Hadoop catalog     "v12.metadata.json"                    -> 12
//   catalog-managed    "00002-<uuid>.metadata.json"           -> 2
// The second is what pyiceberg's SqlCatalog, the Iceberg REST catalog, Glue
// and Hive all write, and the pre-G2FEAT-125 reader recognised neither it nor
// the absence of version-hint.text that goes with it.
bool metadata_version_of(const char* tail, int64_t* out) noexcept {
    assert(tail != nullptr && out != nullptr);
    const char* p = tail;
    if (*p == 'v') ++p;                     // Hadoop-catalog prefix, optional
    int64_t v = 0;
    uint32_t digits = 0;
    while (*p >= '0' && *p <= '9' && digits < 18u) {   // bounded: no overflow
        v = v * 10 + (*p - '0'); ++p; ++digits;
    }
    if (digits == 0u) return false;
    if (*p == '-') {                        // "<NNNNN>-<uuid>.metadata.json"
        if (p == tail) return false;        // a bare "-..." is not a version
        const char* dot = std::strstr(p, ".metadata.json");
        if (dot == nullptr || dot[14] != '\0') return false;
    } else if (std::strcmp(p, ".metadata.json") != 0) {
        return false;
    }
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
        int64_t v = 0;
        if (!metadata_version_of(tail, &v)) continue;
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

const Metadata* iceberg_table_metadata(const TableHandle* h) noexcept {
    assert(h != nullptr);
    if (h == nullptr || !h->meta_loaded) return nullptr;
    return &h->meta;
}

namespace {

// Open the next live data file. Every failure here means rows the table says
// exist that we cannot produce, so each one FAILS the scan (*out_err) rather
// than advancing past the file — the pre-G2FEAT-125 code skipped, which
// silently returned a short result. Returns false with *out_err false only
// when the file list is genuinely exhausted.
bool open_next_file(ScanHandle* s, bool* out_err) noexcept {
    assert(s != nullptr);
    assert(out_err != nullptr);
    *out_err = false;
    if (s->cur_file_idx >= s->n_live) return false;
    const DataFileRef& f = s->live_files[s->cur_file_idx];
    const char* dot = std::strrchr(f.file_path, '.');
    if (dot == nullptr ||
        !(std::strcmp(dot, ".parquet") == 0 ||
          std::strcmp(dot, ".PARQUET") == 0 ||
          std::strcmp(dot, ".pq") == 0)) {
        *out_err = true;                 // ORC/Avro data files are not read yet
        return false;
    }
    const uint8_t* body = nullptr; uint64_t blen = 0;
    if (!read_ref(&s->table->os, s->table->meta.location, s->table->fs_root,
                  s->table->table_rel, f.file_path, s->scratch,
                  &body, &blen)) {
        *out_err = true;
        return false;
    }
    pq::PqMeta* meta = s->scratch->allocate_array<pq::PqMeta>(1);
    if (meta == nullptr) { *out_err = true; return false; }
    std::memset(meta, 0, sizeof(*meta));
    meta->chunks = s->scratch->allocate_array<pq::PqChunk>(
        pq::kPqMaxColumns * 16u);
    meta->chunks_cap = pq::kPqMaxColumns * 16u;
    if (!pq::parquet_read_meta(body, blen, s->scratch, meta)) {
        *out_err = true;
        return false;
    }
    s->cur_meta = meta;
    s->cur_body = body;
    s->cur_body_len = blen;
    s->cur_row_group = 0;
    s->cur_file_open = true;
    return true;
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
        const uint8_t* body = nullptr; uint64_t blen = 0;
        if (!read_ref(&h->os, h->meta.location, h->fs_root, h->table_rel,
                      s->snap.manifest_list, h->arena, &body, &blen)) {
            return false;
        }
        // A real manifest list is Avro; the W4 fixtures are JSON. Dispatch on
        // the bytes, never on a build flag.
        const bool ok = is_avro_ocf(body, blen)
            ? manifest_list_parse_avro(body, blen, h->arena, mlist,
                                       kIcebergMaxManifestsPerList, &n_mlist)
            : manifest_list_parse_json(body, static_cast<uint32_t>(blen),
                                       h->arena, mlist,
                                       kIcebergMaxManifestsPerList, &n_mlist);
        if (!ok) return false;
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
        const uint8_t* body = nullptr; uint64_t blen = 0;
        // A manifest the list names but the store cannot produce is missing
        // data, not an empty manifest — fail rather than under-report.
        if (!read_ref(&h->os, h->meta.location, h->fs_root, h->table_rel,
                      mle.manifest_path, h->arena, &body, &blen)) {
            return false;
        }
        DataFileRef* entries =
            h->arena->allocate_array<DataFileRef>(kIcebergMaxManifestEntries);
        if (entries == nullptr) return false;
        uint32_t n_entries = 0;
        const bool parsed = is_avro_ocf(body, blen)
            ? manifest_parse_avro(body, blen, h->arena, mle.partition_spec_id,
                                  entries, kIcebergMaxManifestEntries,
                                  &n_entries)
            : manifest_parse_json(body, static_cast<uint32_t>(blen), h->arena,
                                  mle.partition_spec_id, entries,
                                  kIcebergMaxManifestEntries, &n_entries);
        if (!parsed) return false;
        const PartitionSpec* spec =
            metadata_spec(&h->meta, mle.partition_spec_id);
        for (uint32_t ei = 0; ei < n_entries; ++ei) {
            DataFileRef& e = entries[ei];
            if (e.status == ManifestStatus::kDeleted) continue;
            // Delete files are not applied yet, and skipping one would
            // over-report rows. Decline the scan instead. TODO(W5-delete-load)
            if (mle.content == ManifestContent::kDeleteManifest ||
                e.content == FileContent::kEqualityDeletes ||
                e.content == FileContent::kPositionDeletes) {
                return false;
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
    for (uint32_t guard = 0; guard <= kMaxLiveFiles; ++guard) {   // bounded
        if (!s->cur_file_open) {
            bool err = false;
            if (!open_next_file(s, &err)) {
                if (err) return false;              // unreadable live data file
                *out_eof = true;
                return true;
            }
        }
        assert(s->cur_meta != nullptr);
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
        // A row group the footer declares but the decoder cannot produce is
        // missing rows (an unsupported codec — zstd — or encoding lands here).
        // Fail; do not step over it.
        if (!pq::parquet_read_row_group(s->cur_body, s->cur_body_len,
                                         s->cur_meta, s->cur_row_group,
                                         s->scratch, cols, &rows)) {
            return false;
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
    // Unreachable: each iteration either returns or retires one live file, so
    // the guard cannot expire. Fail closed if the invariant ever breaks.
    assert(false && "iceberg scan advance did not make progress");
    return false;
}

void iceberg_scan_close(ScanHandle* /*s*/) noexcept {}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
