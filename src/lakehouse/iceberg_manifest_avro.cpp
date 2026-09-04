// bolt/lakehouse/iceberg_manifest_avro.cpp — read REAL Iceberg manifests.
//
// An Iceberg manifest list and manifest are Avro object container files, not
// JSON. This translation unit is the parser the JSON path never had: it binds
// the flattened Avro schema BY FIELD NAME (so v1/v2 layouts, added fields and
// differing field order all work), reads scalars from the row callback, and
// reads the repeated per-column statistics from the per-ELEMENT callback.
//
// Iceberg encodes its int-keyed maps (column_sizes, value_counts,
// null_value_counts, lower_bounds, upper_bounds) as array<record<key,value>>
// because an Avro map may only key on string, so those arrive as element
// records of (int key, long|bytes value).
//
// Tiger Style: fixed caps, no allocation beyond the caller's scratch arena, no
// exceptions, every bound checked.

#include <cassert>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/ingest/bolt_avro.h"
#include "bolt/lakehouse/iceberg/manifest.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace ing = bolt::ingest;

namespace {

// Which repeated field an element belongs to. Resolved once, from the field
// table, so the element callback is a switch and not a string compare per row.
enum : uint8_t {
    kRepNone = 0, kRepLower, kRepUpper, kRepNullCounts, kRepValueCounts,
    kRepEqualityIds,
};

constexpr uint32_t kMaxPartCols = kIcebergMaxPartitionValues;

// Exact match against a flattened field name.
int32_t find_field(const ing::AvroHeader* h, const char* name) noexcept {
    assert(h != nullptr);
    assert(name != nullptr);
    for (uint32_t i = 0; i < h->n_fields; ++i) {        // bounded: n_fields
        if (std::strcmp(h->field[i].name, name) == 0) return static_cast<int32_t>(i);
    }
    return -1;
}

// First field whose name starts with `prefix`, from `from` onward.
int32_t find_prefixed(const ing::AvroHeader* h, const char* prefix,
                      uint32_t from) noexcept {
    assert(h != nullptr);
    assert(prefix != nullptr);
    const size_t pl = std::strlen(prefix);
    for (uint32_t i = from; i < h->n_fields; ++i) {     // bounded: n_fields
        if (std::strncmp(h->field[i].name, prefix, pl) == 0) {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

int64_t i64_or(const ing::AvroValue* v, int64_t dflt) noexcept {
    assert(v != nullptr);
    return v->is_null ? dflt : v->num.i64;
}

// Copy a string/bytes value into a fixed char buffer, NUL-terminated and
// truncated rather than overflowing. Returns the bytes copied.
uint32_t copy_str(char* dst, uint32_t cap, const ing::AvroValue* v) noexcept {
    assert(dst != nullptr);
    assert(cap > 0u);
    if (v == nullptr || v->is_null || v->bytes == nullptr) { dst[0] = '\0'; return 0u; }
    uint32_t n = v->bytes_len;
    if (n > cap - 1u) n = cap - 1u;
    std::memcpy(dst, v->bytes, n);
    dst[n] = '\0';
    return n;
}

// ---- manifest list --------------------------------------------------------

struct MlCtx {
    ManifestListEntry* out;
    uint32_t           cap;
    uint32_t           n;
    int32_t f_path, f_len, f_spec, f_content, f_snap, f_added;
    int32_t f_seq, f_minseq;
};

bool ml_row(void* c, const ing::AvroValue* vals, uint32_t n,
            int64_t) noexcept {
    MlCtx* s = static_cast<MlCtx*>(c);
    assert(s != nullptr);
    assert(vals != nullptr);
    if (s->n >= s->cap) return true;              // bounded: stop filling
    ManifestListEntry* e = &s->out[s->n];
    std::memset(e, 0, sizeof(*e));
    const auto at = [&](int32_t i) noexcept -> const ing::AvroValue* {
        return (i >= 0 && static_cast<uint32_t>(i) < n) ? &vals[i] : nullptr;
    };
    if (const ing::AvroValue* v = at(s->f_path)) {
        copy_str(e->manifest_path, kIcebergMaxManifestPath, v);
    }
    e->manifest_length   = i64_or(at(s->f_len), 0);
    e->added_snapshot_id = i64_or(at(s->f_snap), 0);
    e->added_files_count = i64_or(at(s->f_added), 0);
    e->partition_spec_id = static_cast<int32_t>(i64_or(at(s->f_spec), 0));
    // Carried through so a later commit that re-lists this manifest preserves
    // the sequence number it was committed at (see ManifestListEntry).
    e->sequence_number     = i64_or(at(s->f_seq), 0);
    e->min_sequence_number = i64_or(at(s->f_minseq), 0);
    // `content` is v2-only (0 = data, 1 = deletes); v1 manifests are all data.
    e->content = static_cast<ManifestContent>(i64_or(at(s->f_content), 0));
    ++s->n;
    return true;
}

// ---- manifest entries -----------------------------------------------------

struct DfCtx {
    DataFileRef* out;
    uint32_t     cap;
    uint32_t     n;
    int32_t      default_spec_id;

    DataFileRef  pending;      // accumulates elements + scalars for one row
    int64_t      cur_row;
    bool         have_pending;

    int32_t f_status, f_snap, f_content, f_path, f_records, f_size;
    // Partition struct leaves, in declared order.
    int32_t part_idx[kMaxPartCols];
    uint32_t n_part;
    // Container field index -> which repeated thing it is.
    uint8_t rep_of[ing::kAvroMaxFields];
};

void df_reset_pending(DfCtx* s, int64_t row) noexcept {
    assert(s != nullptr);
    std::memset(&s->pending, 0, sizeof(s->pending));
    s->pending.partition_spec_id = s->default_spec_id;
    // Provenance, so pruning knows how to read what this parser produces:
    // partition field_ids are struct ordinals, bounds are Iceberg's binary
    // single-value encoding. See DataFileRef in manifest.h.
    s->pending.partition_ordinal_ids = true;
    s->pending.binary_bounds         = true;
    s->cur_row      = row;
    s->have_pending = true;
}

// Find (or append) the per-column stat slot for `field_id`. Bounded by
// kIcebergMaxStatCols; past that, extra columns are dropped rather than
// overflowing — stats are a pruning aid, so losing some is degraded, not wrong.
ColumnStatEntry* stat_slot(FileStats* st, int32_t field_id) noexcept {
    assert(st != nullptr);
    for (uint32_t i = 0; i < st->n_cols; ++i) {         // bounded: n_cols
        if (st->cols[i].field_id == field_id) return &st->cols[i];
    }
    if (st->n_cols >= kIcebergMaxStatCols) return nullptr;
    ColumnStatEntry* c = &st->cols[st->n_cols++];
    std::memset(c, 0, sizeof(*c));
    c->field_id = field_id;
    return c;
}

bool df_elem(void* c, uint32_t field_index, int64_t row_index, int64_t,
             const ing::AvroValue* vals, uint32_t n_vals) noexcept {
    DfCtx* s = static_cast<DfCtx*>(c);
    assert(s != nullptr);
    assert(vals != nullptr);
    if (field_index >= ing::kAvroMaxFields) return true;
    const uint8_t rep = s->rep_of[field_index];
    if (rep == kRepNone) return true;                   // a field we don't want
    // Elements arrive before their row's row-callback, so start the row here.
    if (!s->have_pending || s->cur_row != row_index) df_reset_pending(s, row_index);

    if (rep == kRepEqualityIds) {                       // array<long>
        if (n_vals < 1u || vals[0].is_null) return true;
        if (s->pending.n_equality_ids >= 8u) return true;   // bounded
        s->pending.equality_ids[s->pending.n_equality_ids++] =
            static_cast<int32_t>(vals[0].num.i64);
        return true;
    }
    // The remaining three are array<record<key:int, value:...>>.
    if (n_vals < 2u || vals[0].is_null) return true;
    const int32_t fid = static_cast<int32_t>(vals[0].num.i64);
    ColumnStatEntry* col = stat_slot(&s->pending.stats, fid);
    if (col == nullptr) return true;
    const ing::AvroValue* v = &vals[1];
    if (rep == kRepNullCounts) {
        col->null_count = i64_or(v, 0);
    } else if (rep == kRepLower) {
        col->lower_len = static_cast<uint8_t>(
            copy_str(col->lower, kLakeMaxValBytes, v));
        col->has_lower = !v->is_null;
    } else if (rep == kRepUpper) {
        col->upper_len = static_cast<uint8_t>(
            copy_str(col->upper, kLakeMaxValBytes, v));
        col->has_upper = !v->is_null;
    }
    return true;
}

ManifestStatus status_from(int64_t v) noexcept {
    if (v == 0) return ManifestStatus::kExisting;
    if (v == 1) return ManifestStatus::kAdded;
    if (v == 2) return ManifestStatus::kDeleted;
    return ManifestStatus::kUnknown;
}

bool df_row(void* c, const ing::AvroValue* vals, uint32_t n,
            int64_t row_index) noexcept {
    DfCtx* s = static_cast<DfCtx*>(c);
    assert(s != nullptr);
    assert(vals != nullptr);
    if (!s->have_pending || s->cur_row != row_index) df_reset_pending(s, row_index);
    DataFileRef* e = &s->pending;
    const auto at = [&](int32_t i) noexcept -> const ing::AvroValue* {
        return (i >= 0 && static_cast<uint32_t>(i) < n) ? &vals[i] : nullptr;
    };
    e->status      = status_from(i64_or(at(s->f_status), 3));
    e->snapshot_id = i64_or(at(s->f_snap), 0);
    e->content     = static_cast<FileContent>(i64_or(at(s->f_content), 0));
    if (const ing::AvroValue* v = at(s->f_path)) {
        copy_str(e->file_path, kIcebergMaxPath, v);
    }
    e->stats.record_count       = i64_or(at(s->f_records), 0);
    e->stats.file_size_in_bytes = i64_or(at(s->f_size), 0);

    e->n_partition = 0;
    for (uint32_t i = 0; i < s->n_part; ++i) {          // bounded: kMaxPartCols
        const ing::AvroValue* v = at(s->part_idx[i]);
        if (v == nullptr) continue;
        PartitionValue* p = &e->partition[e->n_partition++];
        std::memset(p, 0, sizeof(*p));
        // Ordinal within the partition struct. The Avro schema's "field-id"
        // annotations are not modelled by the flattened field table, and the
        // partition struct's ORDER is the partition spec's order, so the
        // ordinal is the stable join key against a PartitionSpec.
        p->field_id = static_cast<int32_t>(i);
        p->is_null  = v->is_null;
        if (v->is_null) continue;
        if (v->type == ing::AvroType::kString || v->type == ing::AvroType::kBytes) {
            p->is_str = true;
            copy_str(p->str, kLakeMaxValBytes, v);
        } else if (v->type == ing::AvroType::kDouble ||
                   v->type == ing::AvroType::kFloat) {
            p->is_int = false;
            p->i64    = static_cast<int64_t>(v->num.f64);
        } else {
            p->is_int = true;
            p->i64    = v->num.i64;
        }
    }
    if (s->n < s->cap) s->out[s->n++] = *e;             // bounded: cap
    s->have_pending = false;
    return true;
}

// Bind a repeated field (and its two element descriptors) if present.
void bind_rep(DfCtx* s, const ing::AvroHeader* h, const char* name,
              uint8_t rep) noexcept {
    assert(s != nullptr);
    assert(h != nullptr);
    const int32_t i = find_field(h, name);
    if (i < 0) return;
    s->rep_of[static_cast<uint32_t>(i)] = rep;
}

}  // namespace

bool manifest_list_parse_avro(const uint8_t* src, uint64_t len, Arena* scratch,
                              ManifestListEntry* out, uint32_t cap,
                              uint32_t* out_n) noexcept {
    assert(scratch != nullptr);
    assert(out_n != nullptr);
    if (src == nullptr || scratch == nullptr || out == nullptr ||
        out_n == nullptr || cap == 0u) {
        return false;
    }
    *out_n = 0;
    ing::AvroHeader* h = scratch->allocate_array<ing::AvroHeader>(1);
    if (h == nullptr) return false;                 // AvroHeader is ~34 KB
    uint64_t body = 0;
    if (!ing::avro_read_header(src, len, scratch, h, &body)) return false;

    MlCtx s;
    std::memset(&s, 0, sizeof(s));
    s.out       = out;
    s.cap       = cap;
    s.f_path    = find_field(h, "manifest_path");
    s.f_len     = find_field(h, "manifest_length");
    s.f_spec    = find_field(h, "partition_spec_id");
    s.f_content = find_field(h, "content");
    s.f_snap    = find_field(h, "added_snapshot_id");
    s.f_added   = find_field(h, "added_files_count");
    s.f_seq     = find_field(h, "sequence_number");
    s.f_minseq  = find_field(h, "min_sequence_number");
    // Without a path there is nothing to point a scan at — that is not a
    // manifest list, so fail rather than emit path-less entries.
    if (s.f_path < 0) return false;

    int64_t rows = 0;
    if (!ing::avro_read(src, len, scratch, &s, ml_row, &rows)) return false;
    *out_n = s.n;
    return true;
}

bool manifest_parse_avro(const uint8_t* src, uint64_t len, Arena* scratch,
                         int32_t default_spec_id,
                         DataFileRef* out, uint32_t cap,
                         uint32_t* out_n) noexcept {
    assert(scratch != nullptr);
    assert(out_n != nullptr);
    if (src == nullptr || scratch == nullptr || out == nullptr ||
        out_n == nullptr || cap == 0u) {
        return false;
    }
    *out_n = 0;
    ing::AvroHeader* h = scratch->allocate_array<ing::AvroHeader>(1);
    if (h == nullptr) return false;
    uint64_t body = 0;
    if (!ing::avro_read_header(src, len, scratch, h, &body)) return false;

    // DfCtx embeds a DataFileRef and a 256-byte table; keep it off the stack.
    DfCtx* s = scratch->allocate_array<DfCtx>(1);
    if (s == nullptr) return false;
    std::memset(s, 0, sizeof(*s));
    s->out             = out;
    s->cap             = cap;
    s->default_spec_id = default_spec_id;
    s->f_status  = find_field(h, "status");
    s->f_snap    = find_field(h, "snapshot_id");
    s->f_content = find_field(h, "data_file.content");
    s->f_path    = find_field(h, "data_file.file_path");
    s->f_records = find_field(h, "data_file.record_count");
    s->f_size    = find_field(h, "data_file.file_size_in_bytes");
    if (s->f_path < 0) return false;                // not a manifest

    // Partition struct leaves flatten to "data_file.partition.<col>".
    uint32_t from = 0;
    for (uint32_t i = 0; i < kMaxPartCols; ++i) {   // bounded: kMaxPartCols
        const int32_t p = find_prefixed(h, "data_file.partition.", from);
        if (p < 0) break;
        s->part_idx[s->n_part++] = p;
        from = static_cast<uint32_t>(p) + 1u;
    }

    bind_rep(s, h, "data_file.lower_bounds",      kRepLower);
    bind_rep(s, h, "data_file.upper_bounds",      kRepUpper);
    bind_rep(s, h, "data_file.null_value_counts", kRepNullCounts);
    bind_rep(s, h, "data_file.value_counts",      kRepValueCounts);
    bind_rep(s, h, "data_file.equality_ids",      kRepEqualityIds);

    int64_t rows = 0;
    if (!ing::avro_read_ex(src, len, scratch, s, df_row, df_elem, &rows)) {
        return false;
    }
    *out_n = s->n;
    return true;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
