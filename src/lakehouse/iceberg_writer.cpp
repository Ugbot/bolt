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
// definitions + the JSON metadata writer. Manifests and manifest lists are
// AVRO (iceberg_manifest_writer.cpp) — that is the only form Iceberg defines,
// and the JSON ones this file used to emit were unreadable by every real
// reader.
//
// Every commit object goes through the store's atomic put_if_absent, and the
// table version (metadata/v<N>.metadata.json) is the compare-and-swap token:
// of two writers building on version N, exactly one creates v<N+1>; the other
// gets CommitError::kConflict (G2ICE-163). Bounded everything: 256 manifests/snapshot,
// 64 snapshots, 16 refs. No exceptions, no smart pointers, ≥2 asserts/fn.

#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/iceberg/metrics.h"
#include "bolt/lakehouse/iceberg/write_io.h"
#include "bolt/lakehouse/iceberg/view.h"
#include "bolt/lakehouse/iceberg/eq_key.h"
#include "bolt/lakehouse/iceberg/deletion_vector.h"
#include "bolt/lakehouse/iceberg/geo_type.h"

#include <atomic>
#include <cassert>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <new>
#include <system_error>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/crypto/sigv4.h"
#include "bolt/crypto/sha256.h"
#include "bolt/io/bolt_file_io.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/manifest_avro.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/partition_values.h"
#include "bolt/lakehouse/iceberg/snapshot.h"
#include "bolt/join/bolt_swiss_growable.h"
#include "bolt/lakehouse/object_store.h"
#include "iceberg_dml_read.h"

#if defined(BOLT_BUILD_INGEST_PARQUET) || 1
#include "bolt/ingest/bolt_parquet_write.h"
#include "lake_grow.h"
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

#include "iceberg_summary_types.inc"

struct NamedRef {
    char    name[kMaxRefName];
    int64_t snapshot_id;
    uint8_t kind;        // 0 = branch, 1 = tag
    uint8_t _pad[7];
};

struct FileWriteState {
    FileWriteIO config;
    StagedDataFile receipt;
    uint8_t* hash_workspace;
    bool has_receipt;
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
    CommitError    last_error;
    bool           write_uncertain;
    uint8_t        v1_maintenance;  // G2ICE-346: inside table_compact / table_rewrite / expiry, a v1 table is written (as v1)
    CommitCandidate candidate;
    FileWriteState* file_io;
    bool file_io_configurable;
    const uint8_t* prior_meta;      // G2ICE-279: the metadata document this handle last read or wrote (arena-owned)
    uint64_t       prior_meta_len;
    uint32_t       dirty_meta;      // what the pending commit changes beyond its snapshot (kDirty*)
    uint32_t       n_dropped_refs;
    char           dropped_refs[16][64];
    SumRec*        sums;            // summaries of the snapshots this handle committed (G2ICE-280)
    uint32_t       n_sums;
    uint32_t       cap_sums;
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

// Doubles in its arena; `cap` is only the first block. Past 4 GiB (uint32 len)
// an emit fails rather than truncating.
void apply_retention_policy(TableHandle* th) noexcept;

struct Buf {
    Arena*   arena;
    uint8_t* data;
    uint32_t len;
    uint32_t cap;
};

bool buf_init(Buf* b, Arena* a, uint32_t cap) noexcept {
    assert(b != nullptr && a != nullptr && cap > 0u);
    b->arena = a;
    b->data = a->allocate_array<uint8_t>(cap);
    if (b->data == nullptr) return false;
    b->len = 0;
    b->cap = cap;
    return true;
}

// Room for `n` more bytes plus a NUL.
bool buf_room(Buf* b, uint64_t n) noexcept {
    assert(b != nullptr && b->arena != nullptr);
    const uint64_t need = static_cast<uint64_t>(b->len) + n + 1u;
    if (need <= b->cap) return true;
    if (need > UINT32_MAX) return false;
    uint64_t want = static_cast<uint64_t>(b->cap) * 2u;
    if (want < need) want = need;
    if (want > UINT32_MAX) want = UINT32_MAX;
    uint8_t* next = b->arena->allocate_array<uint8_t>(static_cast<size_t>(want));
    if (next == nullptr) return false;
    std::memcpy(next, b->data, b->len);
    b->data = next;
    b->cap = static_cast<uint32_t>(want);
    assert(b->cap >= need);
    return true;
}

bool buf_put(Buf* b, const char* s) noexcept {
    assert(b != nullptr && s != nullptr);
    const size_t n = std::strlen(s);
    if (!buf_room(b, n)) return false;
    std::memcpy(b->data + b->len, s, n);
    b->len += static_cast<uint32_t>(n);
    return true;
}

bool buf_putn(Buf* b, const void* p, uint32_t n) noexcept {
    assert(b != nullptr && p != nullptr);
    if (!buf_room(b, n)) return false;
    std::memcpy(b->data + b->len, p, n);
    b->len += n;
    return true;
}

// Append `s` as a JSON string BODY (no surrounding quotes), escaping what
// RFC 8259 requires. Without this a table, column or location name containing
// a quote or backslash -- `he said "hi"`, a Windows path -- emitted JSON that
// no parser would accept, turning a naming choice into a corrupt table.
bool buf_put_jstr(Buf* b, const char* s) noexcept {
    assert(b != nullptr);
    assert(s != nullptr);
    for (const char* p = s; *p != '\0'; ++p) {   // bounded by NUL
        const unsigned char c = static_cast<unsigned char>(*p);
        const char* esc = nullptr;
        char ubuf[7];
        switch (c) {
            case '"':  esc = "\\\""; break;
            case '\\': esc = "\\\\"; break;
            case '\b': esc = "\\b";  break;
            case '\f': esc = "\\f";  break;
            case '\n': esc = "\\n";  break;
            case '\r': esc = "\\r";  break;
            case '\t': esc = "\\t";  break;
            default:
                if (c < 0x20u) {                  // other control chars
                    std::snprintf(ubuf, sizeof(ubuf), "\\u%04x", c);
                    esc = ubuf;
                }
                break;
        }
        if (esc != nullptr) {
            if (!buf_put(b, esc)) return false;
        } else {
            if (!buf_room(b, 1u)) return false;
            b->data[b->len++] = static_cast<uint8_t>(c);
        }
    }
    return true;
}

// `"key":"escaped value",` — the shape almost every metadata field wants.
bool buf_kv_str(Buf* b, const char* key, const char* val,
                bool trailing_comma) noexcept {
    assert(b != nullptr && key != nullptr);
    if (!buf_put(b, "\"")) return false;
    if (!buf_put(b, key)) return false;
    if (!buf_put(b, "\":\"")) return false;
    if (!buf_put_jstr(b, val != nullptr ? val : "")) return false;
    return buf_put(b, trailing_comma ? "\"," : "\"");
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

// `,"identifier-field-ids":[...]` when the schema names any (G2ICE-300), else nothing.
bool emit_identifier_ids(Buf* b, const Schema* s) noexcept {
    assert(b != nullptr && s != nullptr);
    if (s->n_identifier == 0u) return true;
    if (!buf_put(b, ",\"identifier-field-ids\":[")) return false;
    for (uint32_t k = 0; k < s->n_identifier; ++k) {                     // bounded: the identifier fields
        if (!buf_fmt(b, k == 0u ? "%d" : ",%d", s->identifier_ids[k])) return false;
    }
    return buf_put(b, "]");
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

// G2ICE-84 — every location Iceberg RECORDS is absolute; every key the
// ObjectStore is ADDRESSED with is relative to its root. Those are two
// different strings for the same byte range and this writer used to emit the
// store key in both places.
//
// The spec says a data file's `file_path`, a manifest-list's `manifest_path`
// and a snapshot's `manifest-list` are "full URI"s, and every real catalog
// writes them that way. bolt's own reader hid the bug: `iceberg_scan.cpp`'s
// `read_ref` tries three rebasings in turn (location-relative, strip-root,
// join-table-rel), so it resolved `metadata/snap-1.avro` perfectly well.
// pyiceberg resolves the string LITERALLY and opened it under the READER's
// working directory -- `/private/tmp/metadata/snap-...avro` for a table that
// lives in `/tmp/ice84_fix` -- so the table scanned from exactly one
// directory on earth and failed everywhere else.
//
// `th->root` is `meta.location`, set from the same argument at table_create,
// so joining against it produces a location consistent with what the metadata
// declares. Overflow is a hard failure: a TRUNCATED location names the wrong
// file, which is worse than not committing.
bool abs_location(const char* root, const char* rel, char* out,
                  uint32_t cap) noexcept {
    assert(root != nullptr && rel != nullptr);
    assert(out != nullptr && cap > 0u);
    return path_join(root, rel, out, cap);
}

// The inverse: the ObjectStore key for a location this writer recorded.
// Absolute since G2ICE-84, relative in anything written before it, so both
// shapes must resolve or a table stops being readable across the change.
const char* store_key_for(const char* root, const char* loc) noexcept {
    assert(root != nullptr && loc != nullptr);
    // A location another writer recorded may carry the file scheme (pyiceberg: file:///abs/path); the store root is a path.
    const char* path = loc;
    if (std::strncmp(path, "file://", 7) == 0) path += 7;
    else if (std::strncmp(path, "file:/", 6) == 0) path += 5;
    const size_t rl = std::strlen(root);
    if (rl > 0u && std::strncmp(path, root, rl) == 0 &&
        (path[rl] == '/' || path[rl] == '\\')) {
        return path + rl + 1u;
    }
    return loc;
}

int64_t now_ms() noexcept {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
               system_clock::now().time_since_epoch()).count();
}

// G2ICE-163: 64 bits that differ across calls, threads AND processes. Same
// convention as G2ICE-161's Delta part names (process seed + atomic nonce),
// with the pid read on every call rather than cached, so a fork()ed child —
// which inherits the parent's nonce value and any cached seed — still
// diverges. Finalised through splitmix64 so every output bit depends on all
// inputs.
std::atomic<uint64_t> g_unique_nonce{0};

uint64_t unique_u64() noexcept {
    static const uint64_t seed = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const uint64_t nonce = g_unique_nonce.fetch_add(1, std::memory_order_relaxed);
    const uint64_t ns = static_cast<uint64_t>(
        std::chrono::system_clock::now().time_since_epoch().count());
    uint64_t z = ns ^ seed ^
                 (static_cast<uint64_t>(static_cast<uint32_t>(bolt_getpid())) << 32) ^
                 (nonce * 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    assert(nonce != UINT64_MAX);
    return z;
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

// The Parquet column for the Iceberg types beyond the basic ones (G2ICE-292): binary, uuid, fixed[L], decimal(P,S),
// time. Strings and anything unlisted keep the Utf8 default; an unparseable fixed/decimal fails the write.
bool pq_iceberg_type(const char* t, ingest::parquet::ParquetWriteColumn* c, BoltType* bt) noexcept {
    assert(t != nullptr && c != nullptr && bt != nullptr);
    IcebergTypeInfo info{};
    if (!iceberg_type_parse(t, &info)) return true;                       // nested / unknown: the legacy default
    GeoTypeInfo g{};
    if (geo_type_parse(t, static_cast<uint32_t>(std::strlen(t)), &g)) {   // G2ICE-327: WKB with the GEOMETRY / GEOGRAPHY
        *bt = BoltType::Binary;                                           // logical type; the crs points into `t`, which
        c->geo = static_cast<uint8_t>(g.kind);                            // outlives the writer's open (it copies)
        c->geo_algorithm = g.algorithm;
        c->geo_crs = g.crs_len != 0u ? g.crs : kGeoDefaultCrs;
        c->geo_crs_len = g.crs_len != 0u ? g.crs_len : static_cast<uint32_t>(std::strlen(kGeoDefaultCrs));
        return true;
    }
    switch (info.type) {
        case IcebergType::kBinary: *bt = BoltType::Binary; return true;
        case IcebergType::kUuid:   *bt = BoltType::UUID; return true;
        case IcebergType::kFixed:  *bt = BoltType::Binary; c->fixed_len = info.fixed_len; return true;
        case IcebergType::kTime:   *bt = BoltType::Int64; c->time_micros = true; return true;
        case IcebergType::kDecimal:
            *bt = BoltType::Decimal128;
            c->precision = info.precision;
            c->scale = info.scale;
            return true;
        default: return true;
    }
}

// Fills `po` for `sch`. A schema wider than the inline descriptor array
// uses `wide` as its storage, so `wide` must outlive the writer open.
bool make_pq_opts(const Schema* sch, const WriteOptions* w,
                  ingest::parquet::ParquetWriteOpts* po,
                  std::vector<ingest::parquet::ParquetWriteColumn>* wide) noexcept {
    assert(sch != nullptr && po != nullptr);
    assert(wide != nullptr);
    *po = ingest::parquet::ParquetWriteOpts{};
    const uint32_t n = sch->n_fields;
    if (n > ingest::parquet::kPwInlineColumns) wide->resize(n);
    ingest::parquet::ParquetWriteColumn* cols =
        ingest::parquet::pw_opts_columns(po, n, wide->data());
    if (cols == nullptr) return false;
    for (uint32_t i = 0; i < n; ++i) {
        std::strncpy(cols[i].name, sch->fields[i].name, sizeof(cols[i].name) - 1u);
        // Map iceberg type string back to BoltType for the writer.
        const char* t = sch->fields[i].type;
        BoltType bt = BoltType::Utf8;
        if (std::strcmp(t, "long") == 0)         bt = BoltType::Int64;
        else if (std::strcmp(t, "int") == 0)     bt = BoltType::Int32;
        else if (std::strcmp(t, "double") == 0)  bt = BoltType::Float64;
        else if (std::strcmp(t, "float") == 0)   bt = BoltType::Float32;
        else if (std::strcmp(t, "boolean") == 0) bt = BoltType::Bool;
        else if (std::strcmp(t, "date") == 0)    bt = BoltType::Date32;
        else if (std::strcmp(t, "timestamp") == 0) { bt = BoltType::Timestamp; cols[i].ts_local = true; }
        else if (std::strcmp(t, "timestamptz") == 0) bt = BoltType::Timestamp;
        else if (!pq_iceberg_type(t, &cols[i], &bt)) return false;
        cols[i].type     = bt;
        cols[i].nullable = !sch->fields[i].required;
        cols[i].has_field_id = true;
        cols[i].field_id = sch->fields[i].id;
    }
    po->row_group_target_bytes = w ? static_cast<uint32_t>(
        w->target_row_group_rows ? w->target_row_group_rows * 64u : 1u << 20)
                                   : (1u << 20);
    po->compression            = 1;  // SNAPPY
    po->emit_statistics        = w ? w->emit_stats : true;
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// JSON metadata writer (iceberg_metadata_writer.cpp owns the surface entry
// point, but the core builder lives here so writer.cpp doesn't have to
// re-create the Metadata POD).
// ---------------------------------------------------------------------------

// Derive a spec-valid RFC 4122 UUID from the table location.
//
// Two separate bugs are fixed here. The old value was `bolt-<epoch_ms>`, which
// is not a UUID at all: pyiceberg parses `table-uuid` as one and REJECTS the
// metadata ("invalid character: found `o`"), so every table bolt wrote was
// unreadable by an external Iceberg engine.
//
// And it was TIME-based, so re-emitting a table's metadata minted a NEW
// identity each time. For a surface that regenerates metadata per request that
// is worse than wrong -- a client would see the table's identity change under
// it. Deriving from the location makes it stable for the same table forever,
// and distinct between tables.
//
// This is a name-based UUID in the spirit of v5 without pulling in SHA-1: a
// 128-bit value from four decorrelated 64-bit hashes of the location, with the
// version (4) and variant (RFC 4122) bits forced so the result is well-formed.
void table_uuid_from_location(const char* loc, char* dst,
                              uint32_t cap) noexcept {
    assert(loc != nullptr);
    assert(dst != nullptr);
    assert(cap >= 37u);                       // 36 chars + NUL
    if (dst == nullptr || cap < 37u) return;

    // FNV-1a over the location, four times with different offset bases so the
    // halves are not correlated.
    uint64_t acc[4] = {0xcbf29ce484222325ull, 0x9e3779b97f4a7c15ull,
                       0xff51afd7ed558ccdull, 0xc4ceb9fe1a85ec53ull};
    for (uint32_t k = 0; k < 4u; ++k) {                    // bounded
        uint64_t hv = acc[k];
        for (const char* p = (loc != nullptr ? loc : ""); *p != '\0'; ++p) {
            hv ^= static_cast<uint8_t>(*p);
            hv *= 0x100000001b3ull;
        }
        acc[k] = hv;
    }
    uint8_t u[16];
    for (uint32_t k = 0; k < 2u; ++k) {                    // bounded
        for (uint32_t i = 0; i < 8u; ++i) {
            u[k * 8u + i] = static_cast<uint8_t>((acc[k] >> (i * 8u)) & 0xFFu);
        }
    }
    u[6] = static_cast<uint8_t>((u[6] & 0x0Fu) | 0x40u);   // version 4
    u[8] = static_cast<uint8_t>((u[8] & 0x3Fu) | 0x80u);   // RFC 4122 variant

    static const char* kHex = "0123456789abcdef";
    uint32_t o = 0;
    for (uint32_t i = 0; i < 16u; ++i) {                   // bounded
        if (i == 4u || i == 6u || i == 8u || i == 10u) dst[o++] = '-';
        dst[o++] = kHex[(u[i] >> 4) & 0x0Fu];
        dst[o++] = kHex[u[i] & 0x0Fu];
    }
    dst[o] = '\0';
    assert(o == 36u);
}

namespace {

// G2ICE-84 — the body of the `schema.name-mapping.default` table property,
// returned NUL-terminated so the caller can hand it to `buf_kv_str` and get
// the outer JSON-string escaping for free (the mapping is itself JSON stored
// INSIDE a string, so it has to be escaped twice; doing that by hand is how
// a stray quote in a column name corrupts the metadata).
//
// New Iceberg data files carry their source schema IDs in physical Parquet
// SchemaElement field 9 via make_pq_opts. Keep this mapping for older files
// written without IDs. It is compatibility metadata, not the binding used
// for new files: physical IDs survive renames and dropped/re-added names.
//
// DERIVED from the schema on every emit rather than stored on Metadata, for
// the same reason `last-column-id` is: a mapping that can drift from the
// schema it describes projects the WRONG COLUMN, silently.
bool emit_name_mapping_field(Buf* out,const SchemaField& field,uint32_t depth) noexcept {
    assert(out);assert(field.id>0);
    if (depth>kIcebergSchemaMaxDepth || !buf_fmt(out,"{\"field-id\":%d,\"names\":[\"",field.id) ||
        !buf_put_jstr(out,field.name) || !buf_put(out,"\"]")) return false;
    const bool group=std::strcmp(field.type,"struct")==0 || std::strcmp(field.type,"list")==0 || std::strcmp(field.type,"map")==0;
    if (group) {
        if ((field.n_children && !field.children) || !buf_put(out,",\"fields\":[")) return false;
        for (uint32_t i=0;i<field.n_children;++i)
            if ((i && !buf_put(out,",")) || !emit_name_mapping_field(out,field.children[i],depth+1u)) return false;
        if (!buf_put(out,"]")) return false;
    }
    return buf_put(out,"}");
}
bool emit_name_mapping_json(Arena* a, const Schema* s,
                            const char** out) noexcept {
    assert(a != nullptr && out != nullptr);
    if (s == nullptr) return false;
    Buf b;
    // One extra byte is reserved below for the NUL, so cap is never reached
    // exactly by buf_put; a full buffer fails the emit rather than truncating.
    if (!buf_init(&b, a, 64u * 1024u)) return false;
    if (!buf_put(&b, "[")) return false;
    for (uint32_t i = 0; i < s->n_fields; ++i) {          // bounded
        if (i > 0 && !buf_put(&b, ",")) return false;
        if (!emit_name_mapping_field(&b,s->fields[i],0u)) return false;
    }
    if (!buf_put(&b, "]")) return false;
    if (b.len + 1u > b.cap) return false;
    b.data[b.len] = 0;
    *out = reinterpret_cast<const char*>(b.data);
    return true;
}

// snapshot-log, in timestamp order: Java rejects an unsorted log.
bool emit_snapshot_log(Buf* b, const Metadata* m) noexcept {
    assert(b != nullptr && m != nullptr);
    assert(m->n_snapshot_log <= m->cap_snapshot_log);
    // Insertion sort: the log is appended in commit order, so it is almost
    // always already sorted and this is linear.
    uint32_t* order = b->arena->allocate_array<uint32_t>(m->n_snapshot_log + 1u);
    if (order == nullptr) return false;
    for (uint32_t i = 0; i < m->n_snapshot_log; ++i) {          // bounded
        uint32_t j = i;
        while (j > 0 && m->snapshot_log[order[j - 1u]].timestamp_ms >
                            m->snapshot_log[i].timestamp_ms) {
            order[j] = order[j - 1u];
            --j;
        }
        order[j] = i;
    }
    if (!buf_put(b, "\"snapshot-log\":[")) return false;
    for (uint32_t i = 0; i < m->n_snapshot_log; ++i) {
        const SnapshotLogEntry& e = m->snapshot_log[order[i]];
        if (!buf_fmt(b, "%s{\"timestamp-ms\":%lld,\"snapshot-id\":%lld}",
                     i > 0 ? "," : "",
                     static_cast<long long>(e.timestamp_ms),
                     static_cast<long long>(e.snapshot_id))) return false;
    }
    return buf_put(b, "],");
}

// metadata-log, with `pending` (the file this emit supersedes) appended and
// the oldest entries dropped to stay within the cap.
bool emit_metadata_log(Buf* b, const Metadata* m,
                       const MetadataLogEntry* pending) noexcept {
    assert(b != nullptr && m != nullptr);
    assert(m->n_metadata_log <= kIcebergMaxMetadataLog);
    const uint32_t total = m->n_metadata_log + (pending != nullptr ? 1u : 0u);
    const uint32_t skip  = total > kIcebergMaxMetadataLog
                               ? total - kIcebergMaxMetadataLog : 0u;
    if (!buf_put(b, "\"metadata-log\":[")) return false;
    for (uint32_t i = skip; i < total; ++i) {
        const MetadataLogEntry& e = i < m->n_metadata_log ? m->metadata_log[i]
                                                          : *pending;
        if (!buf_fmt(b, "%s{\"timestamp-ms\":%lld,", i > skip ? "," : "",
                     static_cast<long long>(e.timestamp_ms))) return false;
        if (!buf_kv_str(b, "metadata-file", e.metadata_file, false))
            return false;
        if (!buf_put(b, "}")) return false;
    }
    return buf_put(b, "],");
}

}  // namespace

namespace {

const SnapSummary* find_summary(const SumRec* sums, uint32_t n, int64_t id) noexcept {
    assert(sums != nullptr || n == 0);
    assert(id != 0);
    for (uint32_t i = n; i > 0; --i)                       // bounded by n; newest first
        if (sums[i - 1u].snapshot_id == id) return &sums[i - 1u].s;
    return nullptr;
}

// `"summary":{"operation":...,<counters>}` — counters as decimal strings (spec: string map).
bool emit_summary(Buf* b, const char* op, const SnapSummary* s) noexcept {
    assert(b != nullptr && op != nullptr);
    assert(s == nullptr || s->changed_partitions >= 0);
    if (!buf_fmt(b, "\"summary\":{\"operation\":\"%s\"", op)) return false;
    if (s != nullptr) {
        for (uint32_t k = 0; k < kSumKeys; ++k) {          // bounded
            const bool total = k >= kSumFirstTotal;
            if ((total && !s->has_totals) || (!total && s->v[k] == 0)) continue;
            if (!buf_fmt(b, ",\"%s\":\"%lld\"", kSumNames[k], static_cast<long long>(s->v[k]))) return false;
        }
        if (!buf_fmt(b, ",\"changed-partition-count\":\"%lld\"",
                     static_cast<long long>(s->changed_partitions))) return false;
    }
    return buf_put(b, "}");
}

#include "iceberg_schema_emit.inc"
}  // namespace

bool metadata_json_emit_ex(const Metadata* m, const NamedRef* refs, uint32_t nr,
                           bool is_view, const char* view_dialect,
                           const char* view_sql, int64_t view_version_id,
                           const MetadataLogEntry* pending_prev, const SumRec* sums, uint32_t n_sums,
                           Arena* a, const uint8_t** out, uint64_t* out_len) noexcept;

bool metadata_json_emit(const Metadata* m, const NamedRef* refs, uint32_t nr,
                        bool is_view, const char* view_dialect,
                        const char* view_sql, int64_t view_version_id,
                        const MetadataLogEntry* pending_prev,
                        Arena* a, const uint8_t** out, uint64_t* out_len) noexcept {
    assert(m != nullptr && a != nullptr);
    return metadata_json_emit_ex(m, refs, nr, is_view, view_dialect, view_sql, view_version_id, pending_prev,
                                 nullptr, 0u, a, out, out_len);
}

bool metadata_json_emit_ex(const Metadata* m, const NamedRef* refs, uint32_t nr,
                           bool is_view, const char* view_dialect,
                           const char* view_sql, int64_t view_version_id,
                           const MetadataLogEntry* pending_prev, const SumRec* sums, uint32_t n_sums,
                           Arena* a, const uint8_t** out, uint64_t* out_len) noexcept {
    assert(m != nullptr && a != nullptr && out != nullptr && out_len != nullptr);
    Buf b;
    // First block; grows with the snapshot count.
    if (!buf_init(&b, a, 256u * 1024u)) return false;
    if (!buf_fmt(&b, "{\"format-version\":%d,", m->format_version)) return false;
    if (!buf_kv_str(&b, "table-uuid", m->table_uuid, true)) return false;
    if (!buf_kv_str(&b, "location", m->location, true)) return false;

    // `last-column-id` is REQUIRED by the spec, and pyiceberg rejects metadata
    // without it outright ("Field required"). It is DERIVED here rather than
    // stored on Metadata so it cannot drift from the schema it describes: it
    // is exactly the largest field id assigned so far.
    int32_t last_col_id = 0;
    for (uint32_t i = 0; i < m->n_schemas; ++i) {          // bounded
        const Schema& s = m->schemas[i];
        int32_t max=0;if (!schema_max_field_id(&s,&max)) return false;
        last_col_id=std::max(last_col_id,max);
    }
    if (!buf_fmt(&b, "\"last-column-id\":%d,", last_col_id)) return false;

    // `last-partition-id` is likewise REQUIRED by v2 and was likewise missing.
    // Derived, not stored, for the same reason: it is exactly the largest
    // partition field id assigned so far, and the spec starts that numbering
    // at 1000, so an unpartitioned table reports 999 ("none assigned yet")
    // rather than 0 — which would claim id 0 had been handed out.
    int32_t last_part_id = 999;
    for (uint32_t i = 0; i < m->n_specs; ++i) {            // bounded
        const PartitionSpec& sp = m->specs[i];
        for (uint32_t j = 0; j < sp.n_fields; ++j) {       // bounded
            if (sp.fields[j].field_id > last_part_id) {
                last_part_id = sp.fields[j].field_id;
            }
        }
    }
    if (!buf_fmt(&b, "\"last-partition-id\":%d,", last_part_id)) return false;
    if (!buf_fmt(&b, "\"last-updated-ms\":%lld,",
                 static_cast<long long>(m->last_updated_ms))) return false;
    if (m->format_version >= 2 && !buf_fmt(&b, "\"last-sequence-number\":%lld,",   // v1 has no sequence numbers
                 static_cast<long long>(m->last_sequence_number))) return false;
    if (m->format_version >= 3 && !buf_fmt(&b, "\"next-row-id\":%lld,", static_cast<long long>(m->next_row_id))) return false;
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
        if (!schema_validate(&s,a,m->budget)) return false;
        // An Iceberg schema IS a struct type, so it carries "type":"struct"
        // beside its fields. This was omitted here while `emit_schema_json`
        // (the manifest's copy of the very same structure) always had it —
        // two emitters of one thing, one of them wrong. bolt's own PARSER
        // never reads "type", so the round trip stayed green while DuckDB
        // 1.4.5 refused the table outright: "StructType required property
        // 'type' is missing".
        if (!buf_fmt(&b, "{\"type\":\"struct\",\"schema-id\":%d,\"fields\":[",
                     s.schema_id)) return false;
        for (uint32_t j = 0; j < s.n_fields; ++j) {
            if (j > 0 && !buf_put(&b, ",")) return false;
            const SchemaField& f = s.fields[j];
            if (!emit_schema_field(&b,f,0u)) return false;
        }
        if (!buf_put(&b, "]")) return false;
        if (!emit_identifier_ids(&b, &s)) return false;
        if (!buf_put(&b, "}")) return false;
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
            // The transform was HARDCODED "identity" here, silently rewriting
            // a bucketed or truncated spec into an identity one -- a reader
            // would then partition-prune against a transform the data was
            // never written with. An unrepresentable transform fails the emit
            // rather than being papered over with a default.
            char tname[kIcebergMaxTransformName];
            if (transform_format(pf.transform, tname, sizeof(tname)) == 0u) {
                return false;
            }
            if (!buf_fmt(&b, "{\"source-id\":%d,\"field-id\":%d,",
                         pf.source_id, pf.field_id)) return false;
            if (!buf_kv_str(&b, "name", pf.name, true)) return false;
            if (!buf_kv_str(&b, "transform", tname, false)) return false;
            if (!buf_put(&b, "}")) return false;
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

    // Retain name mapping for legacy field-id-less files (G2ICE-84).
    if (!buf_put(&b, "\"properties\":{")) return false;
    {
        const Schema* cur = metadata_current_schema(m);
        if (cur != nullptr) {
            const char* nm = nullptr;
            if (!emit_name_mapping_json(a, cur, &nm)) return false;
            if (!buf_kv_str(&b, "schema.name-mapping.default", nm,
                            false)) return false;
        }
        if (m->retention_max_age_ms > 0 &&
            !buf_fmt(&b, "%s\"history.expire.max-snapshot-age-ms\":\"%lld\"",
                     cur != nullptr ? "," : "",
                     static_cast<long long>(m->retention_max_age_ms)))
            return false;
        if (m->retention_min_snapshots > 0 &&
            !buf_fmt(&b, "%s\"history.expire.min-snapshots-to-keep\":\"%d\"",
                     (cur != nullptr || m->retention_max_age_ms > 0) ? "," : "",
                     m->retention_min_snapshots))
            return false;
    }
    if (!buf_put(&b, "},")) return false;

    if (!emit_snapshot_log(&b, m)) return false;
    if (!emit_metadata_log(&b, m, pending_prev)) return false;

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
        if (!buf_fmt(&b, "{\"snapshot-id\":%lld,",
                     static_cast<long long>(s.snapshot_id))) return false;
        // A root snapshot omits the optional field (G2ICE-177).
        if (s.parent_snapshot_id > 0 &&
            !buf_fmt(&b, "\"parent-snapshot-id\":%lld,",
                     static_cast<long long>(s.parent_snapshot_id))) return false;
        if (!buf_fmt(&b, "\"timestamp-ms\":%lld,", static_cast<long long>(s.timestamp_ms))) return false;
        if (m->format_version >= 2 &&
            !buf_fmt(&b, "\"sequence-number\":%lld,", static_cast<long long>(s.sequence_number))) return false;
        if (!buf_kv_str(&b, "manifest-list", s.manifest_list, true)) return false;
        if (!buf_fmt(&b, "\"schema-id\":%d,", s.schema_id)) return false;
        const SnapSummary* sum = find_summary(sums, n_sums, s.snapshot_id);
        if (sum != nullptr && sum->has_lineage &&                         // v3 row lineage of a snapshot this handle wrote
            !buf_fmt(&b, "\"first-row-id\":%lld,\"added-rows\":%lld,", static_cast<long long>(sum->first_row_id),
                     static_cast<long long>(sum->added_rows))) return false;
        if (!emit_summary(&b, op, sum)) return false;
        if (!buf_put(&b, "}")) return false;
    }
    if (!buf_put(&b, "],")) return false;

    // refs
    if (!buf_put(&b, "\"refs\":{")) return false;
    // main always names the current snapshot (spec: main must match current-snapshot-id).
    bool main_listed = false;
    for (uint32_t i = 0; i < nr; ++i) if (std::strcmp(refs[i].name, "main") == 0) main_listed = true;   // bounded
    const bool emit_main = !main_listed && !is_view && m->current_snapshot_id > 0;
    if (emit_main && !buf_fmt(&b, "\"main\":{\"snapshot-id\":%lld,\"type\":\"branch\"}",
                              static_cast<long long>(m->current_snapshot_id))) return false;
    for (uint32_t i = 0; i < nr; ++i) {
        if ((i > 0 || emit_main) && !buf_put(&b, ",")) return false;
        if (!buf_put(&b, "\"")) return false;
        if (!buf_put_jstr(&b, refs[i].name)) return false;
        if (!buf_fmt(&b, "\":{\"snapshot-id\":%lld,\"type\":\"%s\"}",
                     static_cast<long long>(refs[i].snapshot_id),
                     refs[i].kind == 0u ? "branch" : "tag")) return false;
    }
    if (!buf_put(&b, "}")) return false;

    // view extras
    if (is_view) {
        // View SQL is the single most likely field to contain a quote or a
        // newline, so it must go through the escaper.
        if (!buf_fmt(&b, ",\"view-version\":{\"version-id\":%lld,"
                     "\"timestamp-ms\":%lld,"
                     "\"representations\":[{\"type\":\"sql\",",
                     static_cast<long long>(view_version_id),
                     static_cast<long long>(m->last_updated_ms))) return false;
        if (!buf_kv_str(&b, "dialect", view_dialect, true)) return false;
        if (!buf_kv_str(&b, "sql", view_sql, false)) return false;
        if (!buf_put(&b, "}]}")) return false;
    }

    if (!buf_put(&b, "}")) return false;
    *out = b.data;
    *out_len = b.len;
    return true;
}

// ---------------------------------------------------------------------------
// Commit protocol: atomic put_if_absent + version CAS (G2ICE-163).
// ---------------------------------------------------------------------------

namespace {
#include "iceberg_write_io.inc"
#include "iceberg_metadata_merge.inc"
#include "iceberg_commit.inc"
}  // namespace

// ---------------------------------------------------------------------------
// Table schema JSON — the `schema` metadata key of a manifest OCF.
//
// The manifest and manifest-list BODIES are Avro (manifest_avro.h); the only
// JSON left on the write path is this one string, which Iceberg requires the
// manifest to carry so a reader can bind the data files' field ids. The JSON
// manifest emitters that used to live here are gone: what they produced was
// not a manifest, and nothing else called them.
// ---------------------------------------------------------------------------

namespace {

bool emit_schema_json(Arena* a, const Schema* s, const uint8_t** out,
                      uint64_t* out_len) noexcept {
    assert(a != nullptr && s != nullptr);
    assert(out != nullptr && out_len != nullptr);
    if (!schema_validate(s,a,nullptr)) return false;
    Buf b;
    if (!buf_init(&b, a, 64u * 1024u)) return false;
    if (!buf_fmt(&b, "{\"type\":\"struct\",\"schema-id\":%d,\"fields\":[",
                 s->schema_id)) return false;
    for (uint32_t i = 0; i < s->n_fields; ++i) {     // bounded by n_fields
        if (i > 0 && !buf_put(&b, ",")) return false;
        const SchemaField& f = s->fields[i];
        if (!emit_schema_field(&b,f,0u)) return false;
    }
    if (!buf_put(&b, "]")) return false;
    if (!emit_identifier_ids(&b, s)) return false;
    if (!buf_put(&b, "}")) return false;
    *out     = b.data;
    *out_len = b.len;
    return true;
}

}  // namespace

namespace {
#include "iceberg_partition_spec.inc"
}  // namespace

// ---------------------------------------------------------------------------
// table_create / table_open / table_close
// ---------------------------------------------------------------------------

bool table_create_ex(TableHandle** out, Arena* arena, ObjectStore* os,
                     const char* path, const Schema* schema,
                     const PartitionSpec* spec, const SortOrder* sort,
                     const WriteOptions* opts, CommitCandidate* out_uncertain) noexcept {
    assert(out != nullptr && arena != nullptr && os != nullptr);
    assert(path != nullptr && schema != nullptr);
    *out = nullptr;
    if (out_uncertain != nullptr) *out_uncertain = CommitCandidate{};
    if (std::strlen(path) + 1u > kMaxPathLen) return false;
    TableHandle* h = new_table_handle(arena);
    if (h == nullptr) return false;
    std::memset(h, 0, sizeof(*h));
    h->arena = arena;
    h->os    = os;
    std::strncpy(h->root, path, sizeof(h->root) - 1u);
    metadata_init(&h->meta, arena, nullptr);
    h->meta.format_version       = opts != nullptr && opts->format_version == 3u ? 3 : 2;
    h->meta.has_next_row_id      = h->meta.format_version >= 3;   // v3: row ids start at 0 (spec "Row Lineage")
    h->meta.next_row_id          = 0;
    h->meta.current_schema_id    = schema->schema_id;
    h->meta.current_spec_id      = spec ? spec->spec_id : 0;
    h->meta.default_sort_order_id = sort ? sort->order_id : 0;
    h->meta.last_sequence_number = 0;
    h->meta.last_updated_ms      = now_ms();
    h->meta.current_snapshot_id  = -1;
    table_uuid_from_location(path, h->meta.table_uuid,
                             sizeof(h->meta.table_uuid));
    std::strncpy(h->meta.location, path, sizeof(h->meta.location) - 1u);
    {
        Schema* s0 = metadata_push_schema(&h->meta);
        if (s0 == nullptr || !schema_copy(s0, schema, arena, nullptr))
            return false;
    }
    h->meta.n_specs = 0;
    PartitionSpec first{};
    int32_t equivalent = -1;
    if (spec != nullptr && !normalize_spec(&h->meta, schema, spec, &first, &equivalent)) return false;
    if (first.spec_id < 0) first.spec_id = 0;
    h->meta.n_specs          = 1;
    h->meta.specs[0]         = first;
    h->meta.current_spec_id  = first.spec_id;
    h->meta.n_sort_orders    = 1;
    if (sort != nullptr) h->meta.sort_orders[0] = *sort;
    else                  h->meta.sort_orders[0].order_id = 0;
    h->meta_version = 0;
    if (opts != nullptr) {
        (void)opts;  // values applied at append-time
    }
    if (!persist_metadata(h)) {
        if (h->write_uncertain && out_uncertain != nullptr) *out_uncertain = h->candidate;
        return false;
    }
    h->file_io_configurable = true;
    *out = h;
    return true;
}

bool table_create(TableHandle** out, Arena* arena, ObjectStore* os,
                  const char* path, const Schema* schema,
                  const PartitionSpec* spec, const SortOrder* sort,
                  const WriteOptions* opts) noexcept {
    assert(out != nullptr && arena != nullptr);
    assert(os != nullptr && path != nullptr);
    return table_create_ex(out, arena, os, path, schema, spec, sort, opts, nullptr);
}

bool table_open(TableHandle** out, Arena* arena, ObjectStore* os,
                const char* path) noexcept {
    assert(out != nullptr && arena != nullptr && os != nullptr);
    assert(path != nullptr);
    *out = nullptr;
    TableHandle* h = new_table_handle(arena);
    if (h == nullptr) return false;
    std::memset(h, 0, sizeof(*h));
    h->arena = arena;
    h->os    = os;
    std::strncpy(h->root, path, sizeof(h->root) - 1u);

    // The hint is a starting point only (G2ICE-163): it can lag behind a
    // concurrent writer or be caught mid-rewrite, so fall back to v1 and
    // probe forward to the real latest version either way.
    const uint8_t* hb = nullptr; uint64_t hl = 0;
    int64_t v = 0; bool any = false;
    if (os_get(os, "metadata/version-hint.text", arena, &hb, &hl) == kOsOk) {
        for (uint64_t i = 0; i < hl && i < 20u; ++i) {   // bounded: int64 digits
            const char c = static_cast<char>(hb[i]);
            if (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); any = true; }
            else if (any) break;
        }
    }
    if (!any || v < 1 || !version_exists(os, v)) {
        if (!version_exists(os, 1)) return false;
        v = 1;
    }
    v = probe_latest(os, v);
    h->meta_version = v;
    char rel[64];
    version_key(v, rel);
    const uint8_t* mb = nullptr; uint64_t ml = 0;
    if (os_get(os, rel, arena, &mb, &ml) != kOsOk) return false;
    if (!metadata_parse(mb, static_cast<uint32_t>(ml), arena, &h->meta))
        return false;
    if (!metadata_version_readable(&h->meta)) return false;
    h->prior_meta = mb;
    h->prior_meta_len = ml;
    h->file_io_configurable = true;
    *out = h;
    return true;
}

void table_close(TableHandle* /*h*/) noexcept {}

#include "iceberg_write_io_api.inc"

// ---------------------------------------------------------------------------
// Append path.
// ---------------------------------------------------------------------------

bool append_open(AppendHandle** out, TableHandle* th) noexcept {
    assert(out != nullptr && th != nullptr);
    if (!write_ready(th)) return false;
    AppendHandle* a = th->arena->allocate_array<AppendHandle>(1);
    if (a == nullptr) return false;
    std::memset(a, 0, sizeof(*a));
    a->th = th;
    *out = a;
    return true;
}

bool append_write(AppendHandle* ah, const BoltBatch* batch) noexcept {
    assert(ah != nullptr && batch != nullptr);
    if (!write_ready(ah->th)) return false;
    if (ah->n_pending >= kMaxPendingBatch) return false;
    ah->pending[ah->n_pending++] = batch;
    return true;
}

namespace {

// G2ICE-135: per-column data-file stats from the exact batches this commit serializes, never an upstream estimate.
// G2ICE-296: the full Iceberg metrics of the batches (value / null / NaN counts, truncate(16) bounds for every primitive
// type: NaN excluded, -0.0 before +0.0, strings / binary truncated with an incremented upper, decimals, uuids, fixed) via
// file_metrics_from_batches. Before, floats could carry a NaN bound and strings / decimals had none.
// G2ICE-336: the metrics live in the table handle's arena; a budget refusal (iceberg_stat_columns,
// iceberg_bound_bytes) fails the write rather than committing a file with metrics missing.
bool compute_file_stats(TableHandle* th, const BoltBatch* const* batches, uint32_t n_batches, FileStats* out) noexcept {
    assert(th != nullptr && batches != nullptr && out != nullptr);
    assert(n_batches > 0);
    const Schema* sch = metadata_current_schema(&th->meta);
    if (sch == nullptr) return true;                                     // no schema: record_count / size only
    return file_metrics_from_batches(batches, n_batches, sch, MetricsModeSpec{}, th->arena, out);
}

// Internal: write one Parquet data file from a list of batches and return the
// rel path + file size + total rows.
bool write_data_file(TableHandle* th, const BoltBatch* const* batches,
                     uint32_t n_batches, int64_t snap_id,
                     char* rel_out, uint32_t rel_cap,
                     int64_t* out_size, int64_t* out_rows) noexcept {
    assert(th != nullptr && batches != nullptr);
    if (!write_ready(th)) return false;
    th->last_error = CommitError::kFailed;
    if (n_batches == 0 || n_batches > kMaxPendingBatch) return false;
    int64_t total_rows = 0;
    for (uint32_t i = 0; i < n_batches; ++i) {
        if (batches[i] == nullptr || batches[i]->num_rows < 0 ||
            batches[i]->num_rows > INT64_MAX - total_rows) return false;
        total_rows += batches[i]->num_rows;
    }
    // data/<snap-id>-<unique>.parquet. The suffix used to be `snap_id & 0xFFFF`
    // — no entropy of its own, so a snapshot-id collision was also a data-file
    // collision (G2ICE-163). It is independent of the snapshot id now.
    const int n = std::snprintf(rel_out, rel_cap,
                                "data/%lld-%016llx.parquet",
                                static_cast<long long>(snap_id),
                                static_cast<unsigned long long>(unique_u64()));
    if (n <= 0 || static_cast<uint32_t>(n) >= rel_cap) return false;
    // The default writer is local; optional FileWriteIO stages and publishes.
    // Data is always written
    // under the CURRENT schema (G2ICE-89) -- `schemas[0]` is only "current"
    // before the table's first evolution; hardcoding it here made an append
    // committed after `table_add_column` serialize against the table's
    // original field layout instead of the one just evolved to.
    const Schema* write_sch = metadata_current_schema(&th->meta);
    if (write_sch == nullptr) return false;
    ingest::parquet::ParquetWriteOpts po{};
    std::vector<ingest::parquet::ParquetWriteColumn> wide;
    if (!make_pq_opts(write_sch, nullptr, &po, &wide)) return false;
    auto* w = data_file_begin(th, rel_out, snap_id, &po);
    if (w == nullptr) { th->last_error = CommitError::kFailed; return false; }
    for (uint32_t i = 0; i < n_batches; ++i) {
        if (!ingest::parquet::parquet_write_row_group(w, batches[i])) {
            data_file_abandon(th, w);
            th->last_error = CommitError::kFailed;
            return false;
        }
    }
    if (!data_file_finish(th, rel_out, w, out_size)) return false;
    *out_rows = total_rows;
    return true;
}

// The ONE place a snapshot id is minted. It used to be inlined at four sites,
// two of which run per commit: the caller stamped its DataFileRef from one
// now_ms() and publish_snapshot then derived the snapshot record's id from
// another. Across a millisecond boundary those disagree, and the manifest
// entry then names a snapshot that does not exist -- which is precisely the
// join an Iceberg reader makes for incremental and time-travel scans. It
// reproduced in ~9.5% of runs (38 of 400).
//
// G2ICE-163: it was `now_ms() * 1000 + n_snapshots` — identical for every
// handle opened on the same table version that commits in the same
// millisecond, and the id names the data file, manifest and manifest list. The
// losing writer's data file then overwrote the winner's already-committed one.
// Now a positive 63-bit value unique across threads and processes (Iceberg's
// own reference implementation uses a random UUID-derived long), re-drawn in
// the vanishing case it repeats an id already in this table's history.
int64_t mint_snapshot_id(const TableHandle* th) noexcept {
    assert(th != nullptr);
    for (uint32_t attempt = 0; attempt < 16u; ++attempt) {   // bounded
        const int64_t id = static_cast<int64_t>(unique_u64() & 0x7FFFFFFFFFFFFFFFull);
        if (id <= 0) continue;
        bool taken = false;
        for (uint32_t i = 0; i < th->meta.n_snapshots; ++i) {   // bounded
            if (th->meta.snapshots[i].snapshot_id == id) { taken = true; break; }
        }
        if (!taken) return id;
    }
    assert(false && "unique_u64 repeated 16 times");
    return 1;
}

#include "iceberg_change.inc"
#include "iceberg_partitioned.inc"
#include "iceberg_summary.inc"
#include "iceberg_change_commit.inc"

// One snapshot adding `files` (all data, or all delete files) on top of the current one. `snap_id` is minted by the
// caller so the files it already stamped and the snapshot record carry the SAME id; 0 mints one.
bool publish_snapshot(TableHandle* th, const DataFileRef* files, uint32_t nf,
                      SnapshotOp op, int64_t snap_id) noexcept {
    assert(th != nullptr);
    assert(th->arena != nullptr && th->os != nullptr);
    if (files == nullptr && nf != 0) return false;
    ChangeSet cs{};
    cs.op = op;
    cs.snap_id = snap_id != 0 ? snap_id : mint_snapshot_id(th);
    const bool dels = nf != 0 && files[0].content != FileContent::kData;
    cs.add_data = dels ? nullptr : files;
    cs.n_add_data = dels ? 0u : nf;
    cs.add_dels = dels ? files : nullptr;
    cs.n_add_dels = dels ? nf : 0u;
    return commit_change(th, &cs);
}

}  // namespace

bool append_commit_ex(AppendHandle* ah, char* out_path, uint32_t out_path_cap,
                      int64_t* out_rows) noexcept {
    assert(ah != nullptr && ah->th != nullptr);
    if (!write_ready(ah->th)) return false;
    ah->th->last_error = CommitError::kNone;
    if (ah->n_pending == 0) return true;
    // Don't write a data file for a commit that has already lost.
    if (!handle_is_current(ah->th)) return false;
    const int64_t snap_id = mint_snapshot_id(ah->th);
    const PartitionSpec* spec = metadata_spec(&ah->th->meta, ah->th->meta.current_spec_id);
    if (spec != nullptr && spec->n_fields != 0u)
        return append_partitioned(ah, spec, snap_id, out_path, out_path_cap, out_rows);
    char rel[128];
    int64_t size = 0, rows = 0;
    if (!write_data_file(ah->th, ah->pending, ah->n_pending, snap_id,
                         rel, sizeof(rel), &size, &rows)) {
        if (ah->th->last_error == CommitError::kNone) ah->th->last_error = CommitError::kFailed;
        return false;
    }
    DataFileRef df{};
    df.status            = ManifestStatus::kAdded;
    df.content           = FileContent::kData;
    df.partition_spec_id = ah->th->meta.current_spec_id;
    df.snapshot_id       = snap_id;
    // Absolute (G2ICE-84) — `rel` remains the ObjectStore key.
    if (!abs_location(ah->th->root, rel, df.file_path, sizeof(df.file_path))) {
        data_file_complete(ah->th, rel, false);
        return false;
    }
    df.stats.record_count       = rows;
    df.stats.file_size_in_bytes = size;
    // G2ICE-135 — real per-column metrics from the exact batches just serialized above.
    if (!compute_file_stats(ah->th, ah->pending, ah->n_pending, &df.stats)) {
        data_file_complete(ah->th, rel, false);
        return false;
    }
    if (!publish_snapshot(ah->th, &df, 1, SnapshotOp::kAppend, snap_id)) {
        // Only a definitive refusal proves this file was not published.
        data_file_complete(ah->th, rel, false);
        if (ah->th->last_error == CommitError::kNone)
            ah->th->last_error = CommitError::kFailed;
        return false;
    }
    data_file_complete(ah->th, rel, true);
    ah->n_pending = 0;
    if (out_path != nullptr && out_path_cap != 0) {
        std::strncpy(out_path, df.file_path, out_path_cap - 1u);
        out_path[out_path_cap - 1u] = '\0';
    }
    if (out_rows != nullptr) *out_rows = rows;
    return true;
}

bool append_commit(AppendHandle* ah) noexcept {
    return append_commit_ex(ah, nullptr, 0, nullptr);
}

void append_close(AppendHandle* /*ah*/) noexcept {}

// ---------------------------------------------------------------------------
// Positional deletes (Iceberg v2 merge-on-read)
// ---------------------------------------------------------------------------
//
// WHY POSITIONAL AND NOT EQUALITY. Equality deletes are the natural shape for
// a key-value tombstone -- you know the key, you do not know where the row
// landed -- and `delete_file.h` already models them. They are nonetheless the
// wrong choice HERE, for one decisive reason: pyiceberg, the external oracle
// this table is verified against, reads positional deletes and refuses
// equality deletes. A delete file no available reader applies is a delete that
// silently does not happen, which is the bug class this work exists to close.
//
// THE ORDERING HAZARD, AND HOW THIS SHAPE ANSWERS IT. A row may be deleted
// long after the file holding it was committed. A positional delete needs
// (file_path, pos), so a late delete must LOCATE the row in the already
// committed data -- it cannot be derived from the tombstone alone. That
// lookup is the caller's job and is deliberately not hidden in here: this
// function takes positions and commits them, and it makes no distinction
// between a position computed while the file was being written and one found
// by scanning it back a week later. Both produce the same delete file, in a
// snapshot whose sequence number is strictly greater than the data file's,
// which is precisely the condition a reader uses to decide the delete applies.

namespace {

// The Iceberg-defined schema of a positional delete file, spec section
// "Position Delete Files": field 2147483546 file_path, 2147483545 pos.
bool position_delete_schema(Arena* a, Schema* out) noexcept {
    assert(a != nullptr && out != nullptr);
    Schema s{};
    s.schema_id = 0;
    if (!schema_reserve(&s, a, nullptr, 2u)) return false;
    std::memset(s.fields, 0, sizeof(SchemaField) * 2u);
    s.n_fields  = 2;
    s.fields[0].id = 2147483546;
    s.fields[0].required = true;
    std::strncpy(s.fields[0].name, "file_path", sizeof(s.fields[0].name) - 1u);
    std::strncpy(s.fields[0].type, "string",    sizeof(s.fields[0].type) - 1u);
    s.fields[1].id = 2147483545;
    s.fields[1].required = true;
    std::strncpy(s.fields[1].name, "pos",  sizeof(s.fields[1].name) - 1u);
    std::strncpy(s.fields[1].type, "long", sizeof(s.fields[1].type) - 1u);
    *out = s;
    assert(out->n_fields == 2u);
    return true;
}

}  // namespace

#include "iceberg_dml.inc"
#include "iceberg_dml_partition.inc"
#include "iceberg_delete_row.inc"
#include "iceberg_delete_dv.inc"
#include "iceberg_dml_ops.inc"
#include "iceberg_delete_eq.inc"
#include "iceberg_delete_eq_ops.inc"
#include "iceberg_delete_dv_ops.inc"
#include "iceberg_compact.inc"

bool table_delete_positions(TableHandle* th, const PositionDeleteEntry* dels,
                            uint32_t n) noexcept {
    assert(th != nullptr);
    assert(dels != nullptr || n == 0);
    if (th == nullptr || th->arena == nullptr || th->os == nullptr) return false;
    if (!write_ready(th)) return false;
    th->last_error = CommitError::kFailed;
    if (dels == nullptr || n == 0 || n > kIcebergMaxManifestEntries || !handle_is_current(th)) return false;
    // A position delete is only meaningful against a real file, and a negative position names no row -- refuse
    // rather than emit a delete file a reader will silently ignore.
    Arena* a = th->arena;
    auto* paths = a->allocate_array<const char*>(n);
    auto* pos = a->allocate_array<int64_t>(n);
    if (paths == nullptr || pos == nullptr) return false;
    for (uint32_t i = 0; i < n; ++i) {
        size_t length = 0;
        if (!stage_text(dels[i].file_path, kIcebergMaxPath, &length) || length == 0 || dels[i].pos < 0) return false;
        paths[i] = dels[i].file_path;
        pos[i] = dels[i].pos;
    }
    // Sorted by file_path then pos (spec: "Position Delete Files"), then one delete file per partition (G2ICE-292).
    const char** sp = nullptr; int64_t* spos = nullptr;
    if (!sort_position_deletes(a, paths, pos, n, &sp, &spos)) return false;
    if (th->meta.format_version >= 3) return delete_positions_v3(th, sp, spos, n);   // G2ICE-283: deletion vectors
    LiveData live{};
    if (!live_data_build(th, &live)) return false;
    const int64_t snap_id = mint_snapshot_id(th);
    DataFileRef* dfs = nullptr; char (*rels)[128] = nullptr; uint32_t nd = 0;
    bool committed = write_grouped_deletes(th, snap_id, sp, spos, n, &live, &dfs, &rels, &nd);
    committed = committed && publish_snapshot(th, dfs, nd, SnapshotOp::kDelete, snap_id);
    for (uint32_t i = 0; i < nd; ++i) data_file_complete(th, rels[i], committed);   // bounded by nd
    return committed;
}

#include "iceberg_schema_evolution.inc"

#include "iceberg_expire.inc"

bool table_remove_orphans(TableHandle* th, uint64_t /*older_than_ms*/,
                          bool dry_run, char (*out)[512], uint32_t cap,
                          uint32_t* n) noexcept {
    assert(th != nullptr && n != nullptr);
    if (!dry_run && !write_ready(th)) return false;
    *n = 0;
    if (!dry_run) return true;
    // List the table data/ prefix, report all paths.
    ObjectEntry* listing = nullptr;
    uint32_t nl = 0;
    // A missing data/ lists empty; any other failure must not read as "none".
    if (os_list_all(th->os, "data/", th->arena, 256u, &listing, &nl) != kOsOk)
        return false;
    uint32_t emit = 0;
    for (uint32_t i = 0; i < nl && emit < cap; ++i) {
        std::strncpy(out[emit], listing[i].key, 511);
        out[emit][511] = '\0';
        ++emit;
    }
    *n = emit;
    return true;
}

#include "iceberg_refs_views.inc"

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
