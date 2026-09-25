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
    // Rows already emitted from the CURRENT live file, across prior row
    // groups. Iceberg positional deletes name (file_path, pos) where `pos`
    // is the 0-based row index WITHIN THE DATA FILE, not within a row
    // group -- this is what turns a row-group-local index into that
    // absolute position. Reset to 0 by open_next_file.
    uint64_t      cur_file_row_base;
    PositionDeleteSet    pos_dels;
    EqualityDeleteSetI64 eq_dels;
};

namespace {

bool load_latest_metadata(TableHandle* h) noexcept {
    assert(h != nullptr && h->arena != nullptr);
    assert(h->os.vt != nullptr && h->table_rel[0] != '\0');
    char key[kCatMaxPath];
    if (!find_latest_metadata(&h->os, h->table_rel, h->arena, key, sizeof(key)))
        return false;
    const uint8_t* body = nullptr; uint64_t blen = 0;
    if (os_get(&h->os, key, h->arena, &body, &blen) != kOsOk) return false;
    if (!metadata_parse(body, static_cast<uint32_t>(blen), h->arena, &h->meta))
        return false;
    h->meta_loaded = true;
    return true;
}

}  // namespace

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
    if (!load_latest_metadata(h)) return false;
    *out = h;
    return true;
}

bool iceberg_table_open_store(TableHandle** out, Arena* arena,
                              const ObjectStore* store,
                              const char* table_rel) noexcept {
    assert(out != nullptr && arena != nullptr);
    assert(store != nullptr && table_rel != nullptr);
    if (store == nullptr || store->vt == nullptr || table_rel == nullptr)
        return false;
    const size_t tl = std::strlen(table_rel);
    if (tl == 0u || tl + 1u > kMaxFsRoot) return false;
    TableHandle* h = arena->allocate_array<TableHandle>(1);
    if (h == nullptr) return false;
    std::memset(h, 0, sizeof(*h));
    h->arena = arena;
    h->os = *store;
    std::memcpy(h->table_rel, table_rel, tl + 1u);
    if (!load_latest_metadata(h)) return false;
    assert(h->meta_loaded);
    *out = h;
    return true;
}

namespace {

// "s3://bucket/wh/ns/t" -> "wh/ns/t": object-store keys are bucket-relative.
bool key_prefix_from_location(const char* location, char* out,
                              uint32_t cap) noexcept {
    assert(location != nullptr && out != nullptr);
    assert(cap > 0u);
    const char* sep = std::strstr(location, "://");
    if (sep == nullptr) return false;
    const char* p = std::strchr(sep + 3, '/');
    if (p == nullptr) return false;
    while (*p == '/') ++p;
    size_t n = std::strlen(p);
    while (n > 0u && p[n - 1u] == '/') --n;
    if (n == 0u || n + 1u > cap) return false;
    std::memcpy(out, p, n);
    out[n] = '\0';
    return true;
}

}  // namespace

bool iceberg_table_open_on_store(TableHandle** out, Arena* arena,
                                 const ObjectStore* store,
                                 const char* table_key_prefix,
                                 const uint8_t* metadata_json,
                                 uint32_t metadata_len) noexcept {
    assert(out != nullptr && arena != nullptr);
    assert(store != nullptr && store->vt != nullptr);
    *out = nullptr;
    if (metadata_json == nullptr || metadata_len == 0u) return false;
    TableHandle* h = arena->allocate_array<TableHandle>(1);
    if (h == nullptr) return false;
    std::memset(h, 0, sizeof(*h));
    h->arena = arena;
    h->os = *store;
    if (!metadata_parse(metadata_json, metadata_len, arena, &h->meta))
        return false;
    if (table_key_prefix != nullptr) {
        const size_t n = std::strlen(table_key_prefix);
        if (n == 0u || n + 1u > sizeof(h->table_rel)) return false;
        std::memcpy(h->table_rel, table_key_prefix, n + 1u);
    } else if (!key_prefix_from_location(h->meta.location, h->table_rel,
                                         sizeof(h->table_rel))) {
        return false;
    }
    h->meta_loaded = true;
    assert(h->table_rel[0] != '\0' && h->fs_root[0] == '\0');
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
    s->cur_file_row_base = 0;
    return true;
}

// ---------------------------------------------------------------------------
// Position-delete loading + application (G2ICE-82).
//
// bolt's Iceberg WRITER (table_delete_positions) produces a real v2
// POSITIONAL delete file: a 2-column parquet ("file_path" Utf8, "pos"
// Int64) referenced from a delete manifest. Until this landed, this reader
// DECLINED every table that carried one (see the file banner) rather than
// silently over-reporting rows. This loads those files into `s->pos_dels`
// at scan_open and `iceberg_scan_next_batch` filters matching rows out of
// every row group it decodes -- so a table with real positional deletes
// (written by bolt itself, or by any other v2-conformant writer) reads
// correctly through bolt's own scan, not only through pyiceberg/DuckDB.
//
// Equality deletes remain declined: `equality_delete_set_contains` exists
// but nothing loads one, and no writer in this tree emits one either
// (table_delete_positions is POSITIONAL-only, deliberately -- see its
// banner). Declining is still the honest answer for that case.

// Read one delete file's rows into `s->pos_dels`. `df` must have
// content == kPositionDeletes. Returns false on any I/O/decode failure or
// on overflowing kMaxPosDels -- a delete file bolt cannot fully load must
// fail the scan, not load a PARTIAL delete set that would then under-delete.
bool load_position_delete_file(ScanHandle* s, const DataFileRef& df) noexcept {
    assert(s != nullptr);
    assert(df.content == FileContent::kPositionDeletes);

    const uint8_t* body = nullptr; uint64_t blen = 0;
    if (!read_ref(&s->table->os, s->table->meta.location, s->table->fs_root,
                  s->table->table_rel, df.file_path, s->scratch,
                  &body, &blen)) {
        return false;
    }
    pq::PqMeta* meta = s->scratch->allocate_array<pq::PqMeta>(1);
    if (meta == nullptr) return false;
    std::memset(meta, 0, sizeof(*meta));
    meta->chunks = s->scratch->allocate_array<pq::PqChunk>(pq::kPqMaxColumns * 16u);
    meta->chunks_cap = pq::kPqMaxColumns * 16u;
    if (!pq::parquet_read_meta(body, blen, s->scratch, meta)) return false;

    BoltColumn* cols = s->scratch->allocate_array<BoltColumn>(pq::kPqMaxColumns);
    if (cols == nullptr) return false;
    for (uint32_t rg = 0; rg < meta->n_row_groups && rg < kLakeMaxRowGroups; ++rg) {
        std::memset(cols, 0, sizeof(BoltColumn) * pq::kPqMaxColumns);
        int64_t rows = 0;
        if (!pq::parquet_read_row_group(body, blen, meta, rg, s->scratch,
                                        cols, &rows)) {
            return false;
        }
        int32_t path_ci = -1, pos_ci = -1;
        for (uint32_t c = 0; c < meta->n_columns && c < pq::kPqMaxColumns; ++c) {
            if (std::strcmp(meta->columns[c].name, "file_path") == 0) path_ci = static_cast<int32_t>(c);
            else if (std::strcmp(meta->columns[c].name, "pos") == 0) pos_ci = static_cast<int32_t>(c);
        }
        // A delete file without both well-known columns is not a delete
        // file bolt understands -- decline rather than skip it silently.
        if (path_ci < 0 || pos_ci < 0) return false;
        const BoltColumn& path_col = cols[path_ci];
        const BoltColumn& pos_col  = cols[pos_ci];
        if (pos_col.format != ColumnFormat::Flat ||
            pos_col.type_size_bytes != sizeof(int64_t)) {
            return false;
        }
        const int64_t* pos_vals = static_cast<const int64_t*>(pos_col.data);
        for (int64_t r = 0; r < rows; ++r) {
            if (s->pos_dels.n >= s->pos_dels.cap) return false;  // overflow: fail closed
            const uint8_t* pdata = nullptr; int32_t plen = 0;
            path_col.utf8_at(r, &pdata, &plen);
            if (plen <= 0 || static_cast<uint32_t>(plen) >= kIcebergMaxPath) return false;
            PositionDeleteEntry& e = s->pos_dels.entries[s->pos_dels.n];
            std::memcpy(e.file_path, pdata, static_cast<size_t>(plen));
            e.file_path[plen] = '\0';
            e.pos = pos_vals[r];
            ++s->pos_dels.n;
        }
    }
    return true;
}

// Insertion sort by (file_path, pos) -- position_delete_set_contains'
// binary-search branch (n > 32) requires this ordering. One-time cost at
// scan_open, bounded by kMaxPosDels; a real Iceberg delete file for a
// single seal/tier eviction is tiny (hundreds to low thousands of rows),
// so O(n^2) here is not the risk kMaxPosDels' size might suggest -- but
// bound the outer loop defensively anyway.
void sort_position_deletes(PositionDeleteSet* s) noexcept {
    assert(s != nullptr);
    for (uint32_t i = 1; i < s->n; ++i) {
        PositionDeleteEntry key = s->entries[i];
        uint32_t j = i;
        while (j > 0) {
            const PositionDeleteEntry& prev = s->entries[j - 1];
            const int c = std::strcmp(prev.file_path, key.file_path);
            const bool after = (c > 0) || (c == 0 && prev.pos > key.pos);
            if (!after) break;
            s->entries[j] = s->entries[j - 1];
            --j;
        }
        s->entries[j] = key;
    }
}

// Filter deleted rows out of one just-decoded row group, in place. `cols`
// holds `*rows` rows for `n_cols` columns, all ColumnFormat::Flat (what
// parquet_read_row_group always produces -- true for Utf8 too, which
// decodes as a 16-byte StringView "Flat" array referencing an unmoving
// spill buffer, per bolt/bolt_column.h's utf8_at banner). `base_row` is
// this row group's first row's position within the data FILE (Iceberg
// positions are file-relative, not row-group-relative).
//
// Returns false only when filtering was actually NEEDED (>=1 row in this
// group matched a delete) and at least one column's format/width this
// function cannot safely compact -- never a silent under-delete. A group
// with nothing to delete is untouched regardless of column shape, so a
// table's odd column types cost nothing unless a real delete lands in it.
bool apply_position_deletes_to_group(const PositionDeleteSet* pos_dels,
                                     const char* file_path, uint64_t base_row,
                                     BoltColumn* cols, uint32_t n_cols,
                                     int64_t* rows, Arena* scratch) noexcept {
    assert(pos_dels != nullptr && file_path != nullptr);
    assert(cols != nullptr && rows != nullptr && scratch != nullptr);
    if (pos_dels->n == 0 || *rows == 0) return true;  // nothing declared, or nothing to check

    auto* sel = scratch->allocate_array<uint32_t>(static_cast<uint64_t>(*rows));
    if (sel == nullptr) return false;
    uint32_t kept = 0;
    for (int64_t r = 0; r < *rows; ++r) {
        const int64_t file_pos = static_cast<int64_t>(base_row) + r;
        if (!position_delete_set_contains(pos_dels, file_path, file_pos)) {
            sel[kept++] = static_cast<uint32_t>(r);
        }
    }
    if (kept == static_cast<uint32_t>(*rows)) return true;  // nothing in THIS group matched

    for (uint32_t c = 0; c < n_cols; ++c) {
        BoltColumn& col = cols[c];
        if (col.format != ColumnFormat::Flat || col.type_size_bytes == 0) {
            std::fprintf(stderr,
                "[iceberg_scan] REFUSED: a positional delete matched a row in "
                "'%s', but column %u (format=%d, width=%u) cannot be safely "
                "compacted -- refusing rather than risk under-deleting.\n",
                file_path, c, static_cast<int>(col.format),
                static_cast<unsigned>(col.type_size_bytes));
            return false;
        }
        const size_t tsz = col.type_size_bytes;
        uint8_t* buf = static_cast<uint8_t*>(col.data);
        for (uint32_t i = 0; i < kept; ++i) {
            const uint32_t src = sel[i];
            if (src != i) std::memmove(buf + i * tsz, buf + static_cast<size_t>(src) * tsz, tsz);
        }
        if (col.validity != nullptr && !col.stats.all_valid) {
            uint8_t* nval = static_cast<uint8_t*>(
                scratch->allocate_zeroed((static_cast<size_t>(kept) + 7u) / 8u));
            if (nval == nullptr) return false;
            uint32_t null_count = 0;
            for (uint32_t i = 0; i < kept; ++i) {
                const int64_t srcbit = col.validity_offset + sel[i];
                const uint8_t bit =
                    (col.validity[srcbit >> 3] >> (srcbit & 7)) & 1u;
                nval[i >> 3] |= static_cast<uint8_t>(bit << (i & 7));
                null_count += (1u - static_cast<uint32_t>(bit));
            }
            col.validity = nval;
            col.validity_offset = 0;
            col.stats.null_count = null_count;
            col.stats.all_valid = (null_count == 0);
        }
        col.length = static_cast<int64_t>(kept);
    }
    *rows = static_cast<int64_t>(kept);
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
            // G2ICE-82/W5-delete-load: POSITION deletes are now loaded and
            // applied against every row group they cover (see
            // apply_position_deletes_to_group below) -- bolt's own
            // table_delete_positions() writes exactly this shape. EQUALITY
            // deletes still have no loader and no writer anywhere in this
            // tree, so skipping one would over-report rows: decline the
            // scan rather than silently under-delete.
            if (e.content == FileContent::kEqualityDeletes) {
                return false;
            }
            if (mle.content == ManifestContent::kDeleteManifest ||
                e.content == FileContent::kPositionDeletes) {
                if (!load_position_delete_file(s, e)) return false;
                continue;  // a delete file is never itself a live data file
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
    // position_delete_set_contains binary-searches once n > 32; entries were
    // appended file-by-file, not (file_path, pos)-ordered, across possibly
    // several delete files -- sort once, here, rather than per lookup.
    sort_position_deletes(&s->pos_dels);
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
        // Positions are counted against the FULL data file (deleted rows
        // included), so the next group's base must advance by the
        // pre-filter row count -- capture it before the filter can shrink
        // `rows`.
        const int64_t decoded_rows = rows;
        const DataFileRef& live_file = s->live_files[s->cur_file_idx];
        if (!apply_position_deletes_to_group(&s->pos_dels, live_file.file_path,
                                             s->cur_file_row_base, cols,
                                             s->cur_meta->n_columns, &rows,
                                             s->scratch)) {
            return false;
        }
        s->cur_file_row_base += static_cast<uint64_t>(decoded_rows);
        out->num_rows = rows;
        out->num_cols = s->cur_meta->n_columns;
        for (uint32_t c = 0; c < s->cur_meta->n_columns; ++c) {
            const char* phys = s->cur_meta->columns[c].name;
            out->schema.add_field(phys, cols[c].type, true);
        }
        if (rows == 0) continue;  // whole group deleted -- advance, don't emit empty
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
