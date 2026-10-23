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
// Positional deletes are loaded and applied across absolute file positions.
// Equality deletes remain refused. A live file or unsupported decode fails
// the scan, never a silent short result. Version-capable stores use bounded
// pinned Parquet sources without key fallback/retry; legacy stores retain the
// contiguous decoder. Codec support comes from the maintained Parquet reader.
//
// Tiger Style: bounded, noexcept, fixed-size scratch.

#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/iceberg/deletion_vector.h"
#include "bolt/lakehouse/iceberg/pinned_read.h"
#include "bolt/lakehouse/object_store/s3_uri.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <new>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_source.h"
#include "bolt/lakehouse/iceberg/delete_file.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/partition.h"
#include "bolt/lakehouse/iceberg/snapshot.h"
#include "bolt/lakehouse/iceberg/statistics.h"
#include "bolt/lakehouse/object_store.h"
#include "lake_grow.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace pq = bolt::ingest::parquet;

namespace {

constexpr uint32_t kMaxNsName       = 128u;
constexpr uint32_t kMaxFsRoot       = 1024u;

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
    // "file:/tmp/x" (one slash: what Hadoop's Path prints, and a valid URI for the same file as "/tmp/x").
    if (std::strncmp(p, "file:", 5) == 0 && p[5] == '/' && p[6] != '/') return p + 5;
    const char* sep = std::strstr(p, "://");
    if (sep == nullptr) return p;
    const char* q = sep + 3;
    // "/C:/..." -> "C:/..." — only for a real drive letter, so a POSIX
    // "file:///tmp/x" keeps its leading slash.
    if (q[0] == '/' && q[1] != '\0' && q[2] == ':') ++q;
    assert(q >= p);
    return q;
}

// Canonical form for comparing two references to the same object: no scheme (s3a:// == s3:// == s3n://, file:/x == /x), '/' separators.
void normalize_ref_path(char* p) noexcept {
    assert(p != nullptr);
    const char* q = strip_scheme(p);
    if (q != p) std::memmove(p, q, std::strlen(q) + 1u);
    for (char* c = p; *c != '\0'; ++c) if (*c == '\\') *c = '/';
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

int read_version_hint(ObjectStore* os, const char* table_rel, Arena* scratch,
                      int64_t* out) noexcept {
    assert(os != nullptr && table_rel != nullptr && out != nullptr);
    assert(scratch != nullptr);
    char key[kCatMaxPath];
    if (!path_join(table_rel, "metadata/version-hint.text",
                   key, sizeof(key))) return kOsBadArg;
    const uint8_t* body = nullptr; uint64_t blen = 0;
    const int rc = os_get(os, key, scratch, &body, &blen);
    if (rc != kOsOk) return rc;
    if (blen == 0 || blen > 32u) return kOsBadArg;
    int64_t v = 0; bool any = false;
    for (uint64_t i = 0; i < blen; ++i) {
        const char c = static_cast<char>(body[i]);
        if (c >= '0' && c <= '9') {
            const int64_t digit = c - '0';
            if (v > (INT64_MAX - digit) / 10) return kOsBadArg;
            v = v * 10 + digit;
            any = true;
        }
        else if (any) break;
    }
    if (!any) return kOsBadArg;
    *out = v;
    return kOsOk;
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
    assert(scratch != nullptr && cap > 0u);
    int64_t best = -1;
    char best_key[kCatMaxPath]{};
    int64_t hint = 0;
    const int hr = read_version_hint(os, table_rel, scratch, &hint);
    if (hr != kOsOk && hr != kOsNotFound && hr != kOsBadArg) return false;
    if (hr == kOsOk) {
        char rel[64];
        const int n = std::snprintf(rel, sizeof(rel),
                                     "metadata/v%lld.metadata.json",
                                     static_cast<long long>(hint));
        if (n <= 0 || static_cast<size_t>(n) >= sizeof(rel)) return false;
        if (!path_join(table_rel, rel, best_key, sizeof(best_key))) return false;
        const uint8_t* body = nullptr; uint64_t blen = 0;
        const int rc = os_get(os, best_key, scratch, &body, &blen);
        if (rc == kOsOk) best = hint;
        else if (rc != kOsNotFound) return false;
    }
    // The hint is advisory: its write can fail or lose a race AFTER metadata
    // commits. Rank the complete listing, including retained version gaps and
    // catalog-managed names. A failed listing is not an unchanged table.
    char prefix[kCatMaxPath];
    if (!path_join(table_rel, "metadata/", prefix, sizeof(prefix))) return false;
    ObjectEntry* listing = nullptr;
    uint32_t nl = 0;
    if (os_list_all(os, prefix, scratch, 256u, &listing, &nl) != kOsOk)
        return false;
    for (uint32_t i = 0; i < nl; ++i) {
        const char* k = listing[i].key;
        const char* slash = std::strrchr(k, '/');
        const char* tail = slash ? slash + 1 : k;
        int64_t v = 0;
        if (!metadata_version_of(tail, &v)) continue;
        if (v > best) {
            const size_t len = std::strlen(k);
            if (len >= sizeof(best_key)) return false;
            best = v;
            std::memcpy(best_key, k, len + 1u);
        }
    }
    if (best < 0) return false;
    const size_t len = std::strlen(best_key);
    if (len >= cap) return false;
    std::memcpy(out_key, best_key, len + 1u);
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
    bool                  strict_pinned;
    SnapshotSelection     selection;
    int64_t               selected_snapshot;
    uint32_t              parquet_metadata_bytes;
    uint32_t              manifest_bytes;
    uint32_t              request_bytes;
    Arena*                retained_arena;
    PinnedObjectStore*    pinned_source;
};

// iceberg_scan.cpp and iceberg_writer.cpp each define their OWN
// bolt::lakehouse::iceberg::TableHandle (the public headers only forward-declare
// it), which is an ODR violation. Arena::allocate_array<TableHandle> is a
// type-keyed template, so its instantiation is a shared COMDAT symbol: the MSVC
// linker kept the other TU's copy and the writer got 113,384 bytes for a
// 118,696-byte handle (CI 39887412144: the metadata buffer landed inside the
// handle and corrupted last_error/write_uncertain). Allocate through this
// internal-linkage helper so each TU uses its own sizeof. TODO(G2ICE-261):
// give the two handles distinct public types.
namespace {
TableHandle* new_table_handle(Arena* a) noexcept {
    assert(a != nullptr);
    return static_cast<TableHandle*>(a->allocate(sizeof(TableHandle), alignof(TableHandle)));
}
}  // namespace

// One live data file that passed pruning. The path lives in the manifest
// arena, which is reset when the scan moves to the next manifest.
struct FileTask {
    const char* path;
    int64_t     seq;        // data sequence number: the entry's own, else its manifest's (v1 tables: 0)
    int32_t     spec_id;
    uint64_t    part_hash;  // hash of the partition tuple, matched against equality deletes' partitions
};

constexpr uint32_t kMaxEqualityIds = 16;
constexpr uint32_t kMaxEqualitySets = 64;
struct EqSet {
    int64_t   seq;
    int32_t   spec_id;
    uint64_t  part_hash;
    bool      global;               // delete written under an unpartitioned spec
    uint8_t   _pad[3];
    uint32_t  n_ids;
    int32_t   ids[kMaxEqualityIds];
    int32_t   schema_idx[kMaxEqualityIds];   // emitted-column index of each equality field
    uint32_t  n_rows;
    uint32_t  cap_rows;
    uint32_t  n_slots;              // open-addressing table over rows, power of two (0 until built)
    uint32_t* slots;                // row index + 1, 0 = empty
    uint64_t* hashes;
    uint32_t* key_off;
    uint32_t* key_len;
    uint8_t*  key_bytes;            // concatenated keys
    uint64_t  key_used;
    uint64_t  key_cap;
};

// Declared here, defined in iceberg_eq_delete.inc (open_next_file runs before that include).
namespace { void select_equality_sets(ScanHandle* s, const FileTask& f) noexcept; }

struct PinnedParquet {
    ObjectStore* store;
    Arena* io;
    ObjectVersion version;
    char key[kCatMaxPath];
    pq::ParquetByteSource source;
};

// Manifests are evaluated one at a time as the scan reaches them (delete
// manifests all at open, since a delete applies to every data file). Only the
// current manifest's surviving files are held, in `man_arena`; metadata
// memory is O(manifest list + largest manifest + delete set), charged to
// `budget`, never O(table files).
struct ScanHandle {
    TableHandle*  table;
    Arena*        scratch;
    ReadOptions   opts;
    Snapshot      snap;
    Budget*       budget;
    ManifestListEntry* mlist;
    uint32_t      n_mlist;
    uint32_t      next_manifest;
    Arena*        man_arena;
    uint64_t      man_charged;
    FileTask*     tasks;
    uint32_t      n_tasks;
    uint32_t      tasks_cap;
    uint32_t      cur_file_idx;
    uint32_t      _pad0;
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
    // Footer metadata of the ONE open file, reset on each file advance so a
    // table's file count is not bounded by the table arena's block table.
    Arena*        meta_arena;
    PositionDeleteSet    pos_dels;
    EqSet*               eq_sets;      // loaded equality-delete files (G2ICE-282)
    uint32_t             n_eq_sets;
    uint32_t             eq_cap;
    uint32_t             n_cur_eq;     // sets that apply to the file being read
    uint32_t             cur_eq[kMaxEqualitySets];
    struct DvEntry { const char* path; bolt::ingest::RoaringBitmap64 bm; };
    DvEntry*             dvs;          // v3 deletion vectors, one per referenced data file
    uint32_t             n_dvs;
    uint32_t             dvs_cap;
    EqualityDeleteSetI64 eq_dels;
    IcebergScanStats     stats;
    PinnedParquet        pinned;
    pq::ParquetSourceWorkspace source_workspace;
    Arena* source_arena;
    Arena* transport_arena;
    bool ranged;
    bool failed;
    bool owns_transport;
    uint16_t source_indices[kMaxColumns];
    const Schema* selected_schema;
    uint32_t output_columns;
    // Non-pinned scans conform each file's columns to the table's CURRENT schema by Iceberg field id (spec: columns are identified by id,
    // never by name or position). use_map => decode_columns file columns are read, then mapped into output_columns schema columns.
    bool          use_map;
    uint32_t      decode_columns;
    const Schema* map_schema;
    int32_t*      colmap;     // per schema field: file column index, or -1 (absent in this file: read as null)
    NameMapEntry* nm_entries;   // the table's default name mapping, parsed once per scan
    uint32_t      nm_count;
    bool          nm_loaded;
    BoltType*     want_type;  // per schema field: the decoded type the Iceberg type implies (Invalid => not checked)
    // Identity of the row group behind the most recent iceberg_scan_next_batch result.
    IcebergScanGroup last_group;
    IcebergGroupProvider provider;  // optional cache-hit supplier (G2ICE-262)
};

namespace {

#include "iceberg_pinned_read.inc"
#include "iceberg_scan_source.inc"

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
    TableHandle* h = new_table_handle(arena);
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
    TableHandle* h = new_table_handle(arena);
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
    TableHandle* h = new_table_handle(arena);
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

bool iceberg_table_open_pinned(TableHandle** out, Arena* owner, Arena* metadata,
    PinnedObjectStore* source, const PinnedTableRead* read,
    const uint8_t* bytes, uint64_t length) noexcept {
    return open_pinned_table(out,owner,metadata,source,read,bytes,length);
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
bool load_next_manifest(ScanHandle* s, bool* out_err) noexcept;

bool open_next_file(ScanHandle* s, bool* out_err) noexcept {
    assert(s != nullptr);
    assert(out_err != nullptr);
    *out_err = false;
    // Bounded: each round consumes one manifest.
    while (s->cur_file_idx >= s->n_tasks) {
        if (s->next_manifest >= s->n_mlist) return false;
        if (!load_next_manifest(s, out_err)) return false;
    }
    const FileTask& f = s->tasks[s->cur_file_idx];
    const char* dot = std::strrchr(f.path, '.');
    if (dot == nullptr ||
        !(std::strcmp(dot, ".parquet") == 0 ||
          std::strcmp(dot, ".PARQUET") == 0 ||
          std::strcmp(dot, ".pq") == 0)) {
        *out_err = true;                 // ORC/Avro data files are not read yet
        return false;
    }
    const uint8_t* body = nullptr; uint64_t blen = 0;
    assert(s->meta_arena != nullptr);
    s->cur_meta = nullptr;
    s->meta_arena->reset();
    pq::PqMeta* meta = s->meta_arena->allocate_array<pq::PqMeta>(1);
    if (meta == nullptr) { *out_err = true; return false; }
    std::memset(meta, 0, sizeof(*meta));
    if (!open_parquet(s, f.path, s->meta_arena, meta, &s->pinned, &body, &blen)) {
        s->failed = true;
        *out_err = true;
        return false;
    }
    if (!bind_data_columns(s,meta)) { s->failed=true; *out_err=true; return false; }
    select_equality_sets(s, f);
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
    s->meta_arena->reset();
    pq::PqMeta* meta = s->meta_arena->allocate_array<pq::PqMeta>(1);
    if (meta == nullptr) return false;
    std::memset(meta, 0, sizeof(*meta));
    PinnedParquet pinned{};
    if (!open_parquet(s, df.file_path, s->meta_arena, meta, &pinned, &body, &blen)) return false;

    BoltColumn* cols = s->scratch->allocate_array<BoltColumn>(meta->n_columns);
    if (cols == nullptr) return false;
    for (uint32_t rg = 0; rg < meta->n_row_groups && rg < kLakeMaxRowGroups; ++rg) {
        std::memset(cols, 0, sizeof(BoltColumn) * meta->n_columns);
        int64_t rows = 0;
        if (!read_parquet_group(s, &pinned, body, blen, meta, rg, cols, &rows)) {
            return false;
        }
        int32_t path_ci = -1, pos_ci = -1;
        if (!delete_column_indices(s,meta,&path_ci,&pos_ci)) return false;
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
        if (rows < 0 ||
            static_cast<uint64_t>(s->pos_dels.n) + static_cast<uint64_t>(rows) >
                UINT32_MAX) {
            set_resource_exhausted(kLakeMetadataKnob,
                                   static_cast<uint64_t>(rows), UINT32_MAX);
            return false;
        }
        if (!lake_grow(s->scratch, s->budget, &s->pos_dels.entries,
                       s->pos_dels.n, &s->pos_dels.cap,
                       s->pos_dels.n + static_cast<uint32_t>(rows)))
            return false;   // budget refused: fail closed, never a partial set
        for (int64_t r = 0; r < rows; ++r) {
            assert(s->pos_dels.n < s->pos_dels.cap);
            const uint8_t* pdata = nullptr; int32_t plen = 0;
            path_col.utf8_at(r, &pdata, &plen);
            if (plen <= 0 || static_cast<uint32_t>(plen) >= kIcebergMaxPath) return false;
            if (s->table->strict_pinned && std::memchr(pdata,0,static_cast<size_t>(plen))!=nullptr)
                return false;
            PositionDeleteEntry& e = s->pos_dels.entries[s->pos_dels.n];
            std::memcpy(e.file_path, pdata, static_cast<size_t>(plen));
            e.file_path[plen] = '\0';
            e.pos = pos_vals[r];
            if (s->table->strict_pinned) {
                char key[kOsMaxKey];
                if (path_col.is_null(r) || pos_col.is_null(r) || e.pos<0 ||
                    s3_uri_object_key(s->table->fs_root,e.file_path,key,sizeof(key))!=S3UriStatus::Ok)
                    return false;
            }
            normalize_ref_path(e.file_path);                // s3a:// vs s3://, file:/x vs /x: the same object (after the pinned URI check above)
            ++s->pos_dels.n;
        }
    }
    return true;
}

// v3 deletion vector: read the Puffin blob [content_offset, +content_size) of df.file_path and keep its bitmap for the file named
// by referenced_data_file. Anything malformed, duplicated for one data file, or too large fails the scan (never under-deletes).
// The whole Puffin object is read (the transport has no ranged table read here); DVs are small next to the data they cover.
constexpr uint64_t kMaxDvObjectBytes = 256ull << 20;
bool load_deletion_vector(ScanHandle* s, const DataFileRef& df) noexcept {
    assert(s != nullptr && df.is_dv);
    if (df.referenced_data_file[0] == '\0' || df.content_offset < 0 || df.content_size <= 0) return false;
    char want_path[kIcebergMaxPath];
    std::memcpy(want_path, df.referenced_data_file, std::strlen(df.referenced_data_file) + 1u);
    normalize_ref_path(want_path);
    for (uint32_t i = 0; i < s->n_dvs; ++i)
        if (std::strcmp(s->dvs[i].path, want_path) == 0) return false;   // at most one DV per data file
    const uint8_t* body = nullptr; uint64_t blen = 0;
    s->meta_arena->reset();
    if (!table_read_ref(s->table, df.file_path, s->meta_arena, &body, &blen) || blen > kMaxDvObjectBytes) return false;
    const uint64_t off = static_cast<uint64_t>(df.content_offset), size = static_cast<uint64_t>(df.content_size);
    if (off > blen || size > blen - off) return false;
    if (!s->budget->try_reserve(size * 2u)) return false;
    ScanHandle::DvEntry e{};
    if (!bolt::lakehouse::iceberg::iceberg_dv_decode(body + off, size, s->scratch, &e.bm)) return false;
    const size_t plen = std::strlen(df.referenced_data_file);
    char* path = s->scratch->allocate_array<char>(plen + 1u);
    if (path == nullptr) return false;
    std::memcpy(path, df.referenced_data_file, plen + 1u);
    normalize_ref_path(path);
    e.path = path;
    if (!lake_grow(s->scratch, s->budget, &s->dvs, s->n_dvs, &s->dvs_cap, s->n_dvs + 1u)) return false;
    s->dvs[s->n_dvs++] = e;
    return true;
}

const bolt::ingest::RoaringBitmap64* dv_for(const ScanHandle* s, const char* file_path) noexcept {
    char norm[kIcebergMaxPath];
    const size_t n = std::strlen(file_path);
    if (n >= sizeof(norm)) return nullptr;
    std::memcpy(norm, file_path, n + 1u);
    normalize_ref_path(norm);
    for (uint32_t i = 0; i < s->n_dvs; ++i)
        if (std::strcmp(s->dvs[i].path, norm) == 0) return &s->dvs[i].bm;
    return nullptr;
}

// Order by (file_path, pos) -- position_delete_set_contains' binary-search
// branch (n > 32) requires it. Entries are 1 KiB, so sort indices (shell sort:
// no recursion, no allocation beyond two arrays) and permute once.
bool pos_del_less(const PositionDeleteEntry& a,
                  const PositionDeleteEntry& b) noexcept {
    const int c = std::strcmp(a.file_path, b.file_path);
    return c < 0 || (c == 0 && a.pos < b.pos);
}

bool sort_position_deletes(PositionDeleteSet* s, Arena* a, Budget* b) noexcept {
    assert(s != nullptr && a != nullptr);
    if (s->n < 2u) return true;
    uint32_t* idx = nullptr; uint32_t icap = 0;
    PositionDeleteEntry* out = nullptr; uint32_t ocap = 0;
    if (!lake_grow(a, b, &idx, 0u, &icap, s->n)) return false;
    if (!lake_grow(a, b, &out, 0u, &ocap, s->n)) return false;
    for (uint32_t i = 0; i < s->n; ++i) idx[i] = i;
    const PositionDeleteEntry* e = s->entries;
    for (uint32_t gap = s->n / 2u; gap > 0u; gap /= 2u) {     // bounded
        for (uint32_t i = gap; i < s->n; ++i) {
            const uint32_t x = idx[i];
            uint32_t j = i;
            while (j >= gap && pos_del_less(e[x], e[idx[j - gap]])) {
                idx[j] = idx[j - gap];
                j -= gap;
            }
            idx[j] = x;
        }
    }
    for (uint32_t i = 0; i < s->n; ++i) out[i] = e[idx[i]];
    s->entries = out;
    s->cap = ocap;
    assert(s->n <= s->cap);
    return true;
}

// Keep only rows `sel[0..kept)` of every column, in place. Fails (never under-deletes) on a column shape it cannot compact.
bool compact_selected(BoltColumn* cols, uint32_t n_cols, const uint32_t* sel, uint32_t kept, int64_t* rows,
                      Arena* scratch, const char* what) noexcept {
    assert(cols != nullptr && sel != nullptr && rows != nullptr && scratch != nullptr);
    for (uint32_t c = 0; c < n_cols; ++c) {
        BoltColumn& col = cols[c];
        if (col.format != ColumnFormat::Flat || col.type_size_bytes == 0) {
            std::fprintf(stderr,
                "[iceberg_scan] REFUSED: %s matched a row, but column %u (format=%d, width=%u) cannot be safely "
                "compacted -- refusing rather than risk under-deleting.\n",
                what, c, static_cast<int>(col.format),
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
                                     const bolt::ingest::RoaringBitmap64* dv,
                                     const char* file_path, uint64_t base_row,
                                     BoltColumn* cols, uint32_t n_cols,
                                     int64_t* rows, Arena* scratch) noexcept {
    assert(pos_dels != nullptr && file_path != nullptr);
    assert(cols != nullptr && rows != nullptr && scratch != nullptr);
    if ((pos_dels->n == 0 && dv == nullptr) || *rows == 0) return true;  // nothing declared, or nothing to check

    char norm_path[kIcebergMaxPath];
    if (std::strlen(file_path) >= sizeof(norm_path)) return false;
    std::memcpy(norm_path, file_path, std::strlen(file_path) + 1u);
    normalize_ref_path(norm_path);
    file_path = norm_path;
    auto* sel = scratch->allocate_array<uint32_t>(static_cast<uint64_t>(*rows));
    if (sel == nullptr) return false;
    uint32_t kept = 0;
    for (int64_t r = 0; r < *rows; ++r) {
        const int64_t file_pos = static_cast<int64_t>(base_row) + r;
        if (!position_delete_set_contains(pos_dels, file_path, file_pos) &&
            (dv == nullptr || !bolt::lakehouse::iceberg::iceberg_dv_contains(dv, static_cast<uint64_t>(file_pos)))) {
            sel[kept++] = static_cast<uint32_t>(r);
        }
    }
    if (kept == static_cast<uint32_t>(*rows)) return true;  // nothing in THIS group matched

    return compact_selected(cols, n_cols, sel, kept, rows, scratch, file_path);
}

}  // namespace

namespace {
#include "iceberg_eq_delete.inc"
}  // namespace

namespace {

// JSON manifests (the W4 fixtures) refuse past `cap` with `*n == cap`;
// retry with doubled room so no count is a ceiling.
constexpr uint32_t kParseGrowRounds = 32u;

bool parse_manifest_list_any(Arena* a, const uint8_t* body, uint64_t blen,
                             ManifestListEntry** out, uint32_t* n) noexcept {
    assert(a != nullptr && body != nullptr);
    assert(out != nullptr && n != nullptr);
    // A real manifest list is Avro; the W4 fixtures are JSON. Dispatch on the
    // bytes, never on a build flag.
    if (is_avro_ocf(body, blen))
        return manifest_list_parse_avro_grow(body, blen, a, 64u, out, n);
    uint32_t cap = 64u;
    for (uint32_t r = 0; r < kParseGrowRounds; ++r, cap *= 2u) {   // bounded
        ManifestListEntry* buf = a->allocate_array<ManifestListEntry>(cap);
        if (buf == nullptr) return false;
        *n = 0;
        if (manifest_list_parse_json(body, static_cast<uint32_t>(blen), a,
                                     buf, cap, n)) {
            *out = buf;
            return true;
        }
        if (*n < cap) return false;
    }
    return false;
}

// What an entry means to the scan, one rule for the data and delete passes.
enum class EntryUse : uint8_t { kSkip, kData, kPosDelete, kEqDelete, kRefuse };

EntryUse classify(const DataFileRef& e, ManifestContent mc) noexcept {
    if (e.status == ManifestStatus::kDeleted) return EntryUse::kSkip;
    if (e.content == FileContent::kEqualityDeletes) return EntryUse::kEqDelete;
    if (mc == ManifestContent::kDeleteManifest ||
        e.content == FileContent::kPositionDeletes)
        return EntryUse::kPosDelete;
    return EntryUse::kData;
}

struct VisitCtx {
    ScanHandle*          s;
    const PartitionSpec* spec;
    const Schema*        sch;
    ManifestContent      mc;
    bool                 deletes_pass;
    bool                 failed;
    uint8_t              _pad[5];
    int64_t              mle_seq;   // the manifest's sequence number, inherited by entries that carry none
};

bool push_task(ScanHandle* s, const char* path, int64_t seq, int32_t spec_id, uint64_t part_hash) noexcept {
    assert(s != nullptr && path != nullptr);
    const size_t len = std::strlen(path);
    const uint64_t before = s->tasks_cap;
    if (!lake_grow(s->man_arena, nullptr, &s->tasks, s->n_tasks, &s->tasks_cap,
                   s->n_tasks + 1u))
        return false;
    const uint64_t charge =
        (s->tasks_cap - before) * sizeof(FileTask) + len + 1u;
    if (!s->budget->try_reserve(charge)) return false;
    s->man_charged += charge;
    char* p = s->man_arena->allocate_array<char>(len + 1u);
    if (p == nullptr) return false;
    std::memcpy(p, path, len + 1u);
    FileTask& t = s->tasks[s->n_tasks++];
    t.path = p; t.seq = seq; t.spec_id = spec_id; t.part_hash = part_hash;
    ++s->stats.files_planned;
    return true;
}

bool visit_entry(void* c, const DataFileRef* e) noexcept {
    VisitCtx* v = static_cast<VisitCtx*>(c);
    assert(v != nullptr && e != nullptr);
    const EntryUse use = classify(*e, v->mc);
    if (use == EntryUse::kRefuse) { v->failed = true; return false; }
    if (use == EntryUse::kSkip) return true;
    if (v->deletes_pass) {
        if (use == EntryUse::kEqDelete) {
            const int64_t seq = e->has_sequence_number ? e->sequence_number : v->mle_seq;
            if (!load_equality_delete_file(v->s, *e, seq, v->spec)) { v->failed = true; return false; }
            return true;
        }
        if (use != EntryUse::kPosDelete) return true;
        if (e->is_dv ? !load_deletion_vector(v->s, *e) : !load_position_delete_file(v->s, *e)) { v->failed = true; return false; }
        return true;
    }
    // A delete file in a data manifest breaks the spec's layout and the delete
    // pass skipped it: refuse rather than under-delete.
    if (use == EntryUse::kEqDelete) { v->failed = true; return false; }       // a delete file in a data manifest: spec violation
    if (use == EntryUse::kPosDelete) {
        if (v->mc == ManifestContent::kDeleteManifest) return true;
        v->failed = true;
        return false;
    }
    const ScanHandle* s = v->s;
    ++v->s->stats.entries_visited;
    if (!partition_passes(e, v->spec, v->sch, s->opts.predicates,
                          s->opts.n_predicates))
        return true;
    if (!stats_pass(e, v->sch, s->opts.predicates, s->opts.n_predicates))
        return true;
    if (!push_task(v->s, e->file_path, e->has_sequence_number ? e->sequence_number : v->mle_seq, e->partition_spec_id, partition_hash(*e))) { v->failed = true; return false; }
    return true;
}

// Decode one manifest, feeding every entry to visit_entry. Scratch is the
// manifest arena; the caller resets it.
bool evaluate_manifest(ScanHandle* s, const ManifestListEntry& mle,
                       bool deletes_pass) noexcept {
    assert(s != nullptr && s->man_arena != nullptr);
    TableHandle* h = s->table;
    const uint8_t* body = nullptr; uint64_t blen = 0;
    // A manifest the list names but the store cannot produce is missing data,
    // not an empty manifest — fail rather than under-report.
    if (!table_read_ref(h, mle.manifest_path, s->man_arena, &body, &blen))
        return false;
    s->stats.metadata_bytes_read += blen;
    if (deletes_pass) ++s->stats.delete_manifests_read;
    else              ++s->stats.manifests_read;
    VisitCtx v{};
    v.s = s;
    v.spec = metadata_spec(&h->meta, mle.partition_spec_id);
    v.sch = h->strict_pinned ? s->selected_schema : metadata_current_schema(&h->meta);
    v.mc = mle.content;
    v.mle_seq = mle.sequence_number;
    v.deletes_pass = deletes_pass;
    uint32_t n = 0;
    if (is_avro_ocf(body, blen)) {
        const bool ok = manifest_visit_avro(body, blen, s->man_arena,
                                            mle.partition_spec_id, visit_entry,
                                            &v, &n);
        return ok && !v.failed;
    }
    uint32_t cap = 64u;
    for (uint32_t r = 0; r < kParseGrowRounds; ++r, cap *= 2u) {   // bounded
        DataFileRef* buf = s->man_arena->allocate_array<DataFileRef>(cap);
        if (buf == nullptr) return false;
        n = 0;
        if (manifest_parse_json(body, static_cast<uint32_t>(blen), s->man_arena,
                                mle.partition_spec_id, buf, cap, &n)) {
            for (uint32_t i = 0; i < n; ++i)                        // bounded
                if (!visit_entry(&v, &buf[i])) return false;
            return true;
        }
        if (n < cap) return false;
    }
    return false;
}

void reset_manifest_arena(ScanHandle* s) noexcept {
    assert(s != nullptr && s->man_arena != nullptr);
    s->man_arena->reset();
    if (s->man_charged != 0) s->budget->release(s->man_charged);
    s->man_charged = 0;
    s->tasks = nullptr;
    s->n_tasks = 0;
    s->tasks_cap = 0;
    s->cur_file_idx = 0;
}

// Advance to the next data manifest. The open file's body and footer live in
// other arenas, so resetting the manifest arena between files is safe.
bool load_next_manifest(ScanHandle* s, bool* out_err) noexcept {
    assert(s != nullptr && out_err != nullptr);
    assert(s->next_manifest < s->n_mlist);
    reset_manifest_arena(s);
    const ManifestListEntry& mle = s->mlist[s->next_manifest++];
    if (mle.content == ManifestContent::kDeleteManifest) return true;
    // A manifest whose partition summaries exclude the predicate holds no
    // file the scan could keep; skipping it is what makes planning cost
    // O(manifests touched) rather than O(table files).
    const TableHandle* h = s->table;
    if (!manifest_may_match(&mle, metadata_spec(&h->meta, mle.partition_spec_id),
                            h->strict_pinned ? s->selected_schema : metadata_current_schema(&h->meta),
                            s->opts.predicates, s->opts.n_predicates)) {
        ++s->stats.manifests_pruned;
        return true;
    }
    if (!evaluate_manifest(s, mle, false)) {
        *out_err = true;
        return false;
    }
    return true;
}

Arena* new_arena(Arena* in) noexcept {
    assert(in != nullptr);
    void* p = in->allocate(sizeof(Arena), alignof(Arena));
    return p == nullptr ? nullptr : new (p) Arena();
}

bool prepare_scan(ScanHandle* s) noexcept {
    assert(s != nullptr && s->table != nullptr);
    assert(s->budget != nullptr);
    TableHandle* h = s->table;
    if (s->opts.snapshot_selection == SnapshotSelection::Empty)
        return s->opts.snapshot_id == 0 && s->opts.timestamp_ms < 0 && !h->meta.current_snapshot_present;
    if (!snapshot_resolve_selected(&h->meta, s->opts.snapshot_id, s->opts.timestamp_ms,
                                  s->opts.snapshot_selection, &s->snap)) {
        // Only a table with no snapshot at all is legitimately empty.
        if (s->opts.snapshot_selection != SnapshotSelection::Legacy ||
            s->opts.snapshot_id >= 0 || s->opts.timestamp_ms >= 0 ||
            h->meta.current_snapshot_id >= 0)
            return false;
        return true;
    }
    if (h->strict_pinned && !select_pinned_schema(s)) return false;
    {
        const uint8_t* body = nullptr; uint64_t blen = 0;
        if (!table_read_ref(h, s->snap.manifest_list, h->arena, &body, &blen)) {
            return false;
        }
        if (!parse_manifest_list_any(h->arena, body, blen, &s->mlist,
                                     &s->n_mlist))
            return false;
        s->stats.metadata_bytes_read += blen;
        for (uint32_t mi = 0; mi < s->n_mlist; ++mi)               // bounded
            if (s->mlist[mi].content != ManifestContent::kDeleteManifest)
                ++s->stats.manifests_listed;
        if (!s->budget->try_reserve(sizeof(ManifestListEntry) *
                                    static_cast<uint64_t>(s->n_mlist)))
            return false;
    }
    // Delete manifests first: a delete applies to data files in any manifest,
    // including ones the scan has not reached yet.
    // The spec keeps delete files in delete manifests; one found in a data
    // manifest later fails the scan (visit_entry).
    for (uint32_t mi = 0; mi < s->n_mlist; ++mi) {                // bounded
        if (s->mlist[mi].content != ManifestContent::kDeleteManifest) continue;
        reset_manifest_arena(s);
        if (!evaluate_manifest(s, s->mlist[mi], true)) return false;
    }
    reset_manifest_arena(s);
    if (!sort_position_deletes(&s->pos_dels, h->arena, s->budget)) return false;
    s->next_manifest = 0;
    s->cur_file_idx = 0;
    s->cur_file_open = false;
    return true;
}

}  // namespace

bool iceberg_scan_open(ScanHandle** out, TableHandle* h,
                       const ReadOptions* opts) noexcept {
    return iceberg_scan_open_with_io(out, h, opts, nullptr);
}

bool iceberg_scan_open_with_io(ScanHandle** out, TableHandle* h,
    const ReadOptions* opts, const IcebergScanIO* io) noexcept {
    assert(out != nullptr && h != nullptr);
    assert(h->arena != nullptr);
    if (!scan_open_admit(out,h,opts,io)) return false;
    *out = nullptr;
    if (!h->meta_loaded) return false;
    ScanHandle* s = h->arena->allocate_array<ScanHandle>(1);
    if (s == nullptr) return false;
    std::memset(s, 0, sizeof(*s));
    s->table = h;
    s->scratch = h->arena;
    s->meta_arena = h->strict_pinned ? scan_fixed_arena(h->arena,h->parquet_metadata_bytes) : new_arena(h->arena);
    s->man_arena = h->strict_pinned ? scan_fixed_arena(h->arena,h->manifest_bytes) : new_arena(h->arena);
    if (!setup_scan_source(s, io)) { iceberg_scan_close(s); return false; }
    void* bp = h->arena->allocate(sizeof(Budget), alignof(Budget));
    EqualityDeleteI64* ed_buf = h->arena->allocate_array<EqualityDeleteI64>(1u);
    if (s->meta_arena == nullptr || s->man_arena == nullptr || bp == nullptr ||
        ed_buf == nullptr)
        { iceberg_scan_close(s); return false; }
    s->budget = new (bp) Budget(kLakeMetadataKnob, lake_metadata_budget_bytes());
    if (opts != nullptr) s->opts = *opts; else read_options_init(&s->opts);
    if (h->strict_pinned && !pinned_scan_options(s,opts)) { iceberg_scan_close(s); return false; }
    s->pos_dels.entries = nullptr;
    s->pos_dels.n = 0;
    s->pos_dels.cap = 0;
    equality_delete_set_init(&s->eq_dels, ed_buf, 1u);
    if (!prepare_scan(s)) { iceberg_scan_close(s); return false; }
    *out = s;
    return true;
}

bool iceberg_scan_next_batch(ScanHandle* s, BoltBatch* out,
                             bool* out_eof) noexcept {
    assert(s != nullptr && out != nullptr && out_eof != nullptr);
    assert(s->scratch != nullptr);
    const detail::ObjectSpan outputs[]{{out,sizeof(*out)}, {out_eof,sizeof(*out_eof)}};
    if (!scan_outputs_admit(s,outputs,2)) return false;
    *out_eof = false;
    BoltBatch::init_empty(out);
    out->arena = s->scratch;
    if (s->failed) return false;
    // Bounded: every iteration returns, decodes a row group, or retires a file.
    constexpr uint64_t kGuardMax = UINT64_C(1) << 48;
    for (uint64_t guard = 0; guard <= kGuardMax; ++guard) {
        if (!s->cur_file_open) {
            bool err = false;
            if (!open_next_file(s, &err)) {
                if (err) { s->failed=true; return false; } // unreadable live data file
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
        if (!BoltBatch::alloc_columns(out, s->scratch, s->output_columns))
            return false;
        BoltColumn* cols = out->columns[out->read_epoch];
        // With a schema map the file's own columns are decoded into `dec`, then conformed into `cols` (the table's current schema).
        BoltColumn* dec = cols;
        if (s->use_map) {
            dec = s->scratch->allocate_array<BoltColumn>(s->decode_columns);
            if (dec == nullptr) return false;
            std::memset(dec, 0, sizeof(BoltColumn) * s->decode_columns);
        }
        const uint32_t dec_columns = s->use_map ? s->decode_columns : s->output_columns;
        int64_t rows = 0;
        const uint32_t group_index = s->cur_row_group;
        const FileTask& live_file = s->tasks[s->cur_file_idx];
        // G2ICE-262: a caller may supply the group already decoded and already past positional deletes.
        int64_t decoded_rows = 0;
        bool supplied = false;
        if (s->provider.provide != nullptr) {
            int64_t emitted = 0;
            supplied = s->provider.provide(s->provider.ctx, live_file.path, group_index, s->scratch, cols,
                                           s->output_columns, &decoded_rows, &emitted);
            if (supplied) {
                if (emitted < 0 || decoded_rows < emitted) { s->failed = true; return false; }
                rows = emitted;
            }
        }
        if (!supplied) {
            // A row group the footer declares but the decoder cannot produce is
            // missing rows (an unsupported codec -- zstd -- or encoding lands here).
            // Fail; do not step over it.
            if (!read_parquet_group(s, &s->pinned, s->cur_body, s->cur_body_len,
                                    s->cur_meta, s->cur_row_group, dec, &rows, dec_columns)) {
                s->failed = true;
                return false;
            }
            decoded_rows = rows;
        }
        const uint64_t group_base = s->cur_file_row_base;
        ++s->cur_row_group;
        // Positions are counted against the FULL data file (deleted rows included), so the next group's base must
        // advance by the pre-filter row count -- `decoded_rows` was captured before the filter can shrink `rows`.
        if (!supplied && !apply_position_deletes_to_group(&s->pos_dels, dv_for(s, live_file.path), live_file.path,
                                                          s->cur_file_row_base, dec,
                                                          dec_columns, &rows,
                                                          s->scratch)) {
            return false;
        }
        if (!supplied && s->use_map && !conform_group(s, dec, cols, rows)) { s->failed = true; return false; }
        if (!supplied && !apply_equality_deletes(s, cols, s->output_columns, &rows)) { s->failed = true; return false; }
        s->cur_file_row_base += static_cast<uint64_t>(decoded_rows);
        s->last_group.file_path = live_file.path;
        s->last_group.row_group = group_index;
        s->last_group.file_row_base = group_base;
        s->last_group.decoded_rows = static_cast<uint64_t>(decoded_rows);
        s->last_group.emitted_rows = static_cast<uint64_t>(rows);
        out->num_rows = rows;
        out->num_cols = s->output_columns;
        for (uint32_t c = 0; c < s->output_columns; ++c) {
            const char* name = s->table->strict_pinned ? s->selected_schema->fields[c].name :
                s->use_map ? s->map_schema->fields[c].name : s->cur_meta->columns[c].name;
            if (!s->table->strict_pinned && !s->use_map && s->cur_meta->columns[c].name_truncated != 0u) return false;
            if (!out->schema.add_field(name, cols[c].type, true)) return false;
        }
        if (rows == 0) continue;  // whole group deleted -- advance, don't emit empty
        return true;
    }
    // Unreachable: each iteration either returns or retires one live file, so
    // the guard cannot expire. Fail closed if the invariant ever breaks.
    assert(false && "iceberg scan advance did not make progress");
    return false;
}

void iceberg_scan_set_group_provider(ScanHandle* s, const IcebergGroupProvider* provider) noexcept {
    assert(s != nullptr);
    if (s == nullptr) return;
    if (provider == nullptr) { s->provider = IcebergGroupProvider{}; return; }
    s->provider = *provider;
}

bool iceberg_scan_last_group(const ScanHandle* s, IcebergScanGroup* out) noexcept {
    assert(out != nullptr);
    if (s == nullptr || out == nullptr || s->last_group.file_path == nullptr) return false;
    *out = s->last_group;
    return true;
}

bool iceberg_scan_stats(const ScanHandle* s, IcebergScanStats* out) noexcept {
    assert(out != nullptr);
    if (s == nullptr || out == nullptr) return false;
    *out = s->stats;
    assert(out->manifests_read + out->manifests_pruned <= out->manifests_listed);
    return true;
}

bool iceberg_scan_next_file(ScanHandle* s, const char** out_path,
                            bool* out_eof) noexcept {
    assert(s != nullptr && out_path != nullptr && out_eof != nullptr);
    assert(!s->cur_file_open);
    const detail::ObjectSpan outputs[]{{out_path,sizeof(*out_path)}, {out_eof,sizeof(*out_eof)}};
    if (!scan_outputs_admit(s,outputs,2)) return false;
    *out_path = nullptr;
    *out_eof = false;
    if (s->failed) return false;
    // Bounded: each round consumes one manifest.
    while (s->cur_file_idx >= s->n_tasks) {
        if (s->next_manifest >= s->n_mlist) { *out_eof = true; return true; }
        bool err = false;
        if (!load_next_manifest(s, &err) && err) { s->failed=true; return false; }
    }
    *out_path = s->tasks[s->cur_file_idx++].path;
    return true;
}

void iceberg_scan_close(ScanHandle* s) noexcept {
    if (s == nullptr) return;
    s->cur_meta = nullptr;
    if (s->meta_arena != nullptr) {
        s->meta_arena->~Arena();
        s->meta_arena = nullptr;
    }
    if (s->man_arena != nullptr) {
        s->man_arena->~Arena();
        s->man_arena = nullptr;
    }
    if (s->source_arena != nullptr) {
        s->source_arena->~Arena();
        s->source_arena = nullptr;
    }
    if (s->owns_transport && s->transport_arena != nullptr) {
        s->transport_arena->~Arena();
    }
    s->transport_arena = nullptr;
    s->failed = true;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
