// bolt/lakehouse/iceberg_manifest_writer.cpp — W5 manifest + manifest-list
// emitters.
//
// The writers declared in manifest_avro.h — the mirror of manifest.h's
// `manifest_parse_avro` / `manifest_list_parse_avro`, and what the write path
// in iceberg_writer.cpp (`publish_snapshot`) commits. There is no JSON manifest
// writer any more: Iceberg defines manifests as Avro only, so the JSON ones
// that used to live in iceberg_writer.cpp produced files no real reader would
// open. The JSON manifest PARSERS stay — W4 fixtures are JSON, and the scan
// path sniffs the OCF magic to choose.
//
// The row encoding rides `bolt::ingest::avro_write_ex`. That is possible
// because of one property of the format: **an Avro record's fields are inline
// and positional**, so a FLAT field list encodes byte-identically to the
// nested `manifest_entry { ... data_file { ... partition { } } }` a reader
// expects. Only the SCHEMA has to describe the nesting; the bytes do not
// change. The optional repeated fields (`lower_bounds`, `split_offsets`, ...)
// are declared in that schema and written as a one-byte null union branch, so
// no array or map encoder is needed anywhere in this path.
//
// The two schemas are transcribed verbatim from manifests written by
// pyiceberg 0.11.1 (the same reference the READ path was verified against in
// tests/test_bolt_iceberg_real_avro.cpp), with only the per-field `doc`
// strings removed — they are documentation, carried by no reader. Every
// `field-id` is preserved, because that is how a reader binds columns.

#include "bolt/lakehouse/iceberg/manifest_avro.h"

#include "bolt/bolt_arena.h"
#include "bolt/ingest/bolt_avro.h"
#include "bolt/lakehouse/iceberg/delete_file.h"
#include "bolt/lakehouse/iceberg/statistics.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace ing = ::bolt::ingest;

namespace {

#include "iceberg_manifest_schemas.inc"

// Fixed, NOT random. Two synthesis runs over unchanged inputs must produce
// identical bytes — callers cache and compare manifests by content.
constexpr uint8_t kSync[ing::kAvroSyncLen] = {
    0x62, 0x6f, 0x6c, 0x74, 0x2d, 0x69, 0x63, 0x65,
    0x62, 0x65, 0x72, 0x67, 0x2d, 0x6d, 0x66, 0x00,
};

constexpr uint32_t kEntryFields = 19u;   // flattened manifest_entry
constexpr uint32_t kListFields  = 15u;   // flattened manifest_file

void set_field(ing::AvroField* f, const char* name, ing::AvroType t,
               bool nullable) noexcept {
    assert(f != nullptr);
    assert(name != nullptr);
    std::memset(f, 0, sizeof(*f));
    const size_t n = std::strlen(name);
    assert(n < ing::kAvroMaxName);
    std::memcpy(f->name, name, n);
    f->type     = t;
    f->nullable = nullable;
    // Iceberg declares every optional field as ["null", T]: null is branch 0.
    f->null_branch = 0u;
}

// The flattened wire order of `manifest_entry`. `data_file` and `partition`
// contribute no entry of their own — a record is its fields, inline — and an
// EMPTY partition record contributes nothing at all, which is why this list is
// valid only for an unpartitioned spec.
void build_entry_fields(ing::AvroField* f) noexcept {
    assert(f != nullptr);
    using T = ing::AvroType;
    uint32_t i = 0;
    set_field(&f[i++], "status",               T::kInt,    false);
    set_field(&f[i++], "snapshot_id",          T::kLong,   true);
    set_field(&f[i++], "sequence_number",      T::kLong,   true);
    set_field(&f[i++], "file_sequence_number", T::kLong,   true);
    set_field(&f[i++], "content",              T::kInt,    false);
    set_field(&f[i++], "file_path",            T::kString, false);
    set_field(&f[i++], "file_format",          T::kString, false);
    set_field(&f[i++], "record_count",         T::kLong,   false);
    set_field(&f[i++], "file_size_in_bytes",   T::kLong,   false);
    set_field(&f[i++], "column_sizes",         T::kArray,  true);
    set_field(&f[i++], "value_counts",         T::kArray,  true);
    set_field(&f[i++], "null_value_counts",    T::kArray,  true);
    set_field(&f[i++], "nan_value_counts",     T::kArray,  true);
    set_field(&f[i++], "lower_bounds",         T::kArray,  true);
    set_field(&f[i++], "upper_bounds",         T::kArray,  true);
    set_field(&f[i++], "key_metadata",         T::kBytes,  true);
    set_field(&f[i++], "split_offsets",        T::kArray,  true);
    set_field(&f[i++], "equality_ids",         T::kArray,  true);
    set_field(&f[i++], "sort_order_id",        T::kInt,    true);
    assert(i == kEntryFields);
}

void build_list_fields(ing::AvroField* f) noexcept {
    assert(f != nullptr);
    using T = ing::AvroType;
    uint32_t i = 0;
    set_field(&f[i++], "manifest_path",        T::kString, false);
    set_field(&f[i++], "manifest_length",      T::kLong,   false);
    set_field(&f[i++], "partition_spec_id",    T::kInt,    false);
    set_field(&f[i++], "content",              T::kInt,    false);
    set_field(&f[i++], "sequence_number",      T::kLong,   false);
    set_field(&f[i++], "min_sequence_number",  T::kLong,   false);
    set_field(&f[i++], "added_snapshot_id",    T::kLong,   false);
    set_field(&f[i++], "added_files_count",    T::kInt,    false);
    set_field(&f[i++], "existing_files_count", T::kInt,    false);
    set_field(&f[i++], "deleted_files_count",  T::kInt,    false);
    set_field(&f[i++], "added_rows_count",     T::kLong,   false);
    set_field(&f[i++], "existing_rows_count",  T::kLong,   false);
    set_field(&f[i++], "deleted_rows_count",   T::kLong,   false);
    set_field(&f[i++], "partitions",           T::kArray,  true);
    set_field(&f[i++], "key_metadata",         T::kBytes,  true);
    assert(i == kListFields);
}

void put_long(ing::AvroValue* v, int64_t x) noexcept {
    assert(v != nullptr);
    std::memset(v, 0, sizeof(*v));
    v->num.i64 = x;
}

void put_str(ing::AvroValue* v, const char* s) noexcept {
    assert(v != nullptr);
    assert(s != nullptr);
    std::memset(v, 0, sizeof(*v));
    v->bytes     = reinterpret_cast<const uint8_t*>(s);
    v->bytes_len = static_cast<uint32_t>(std::strlen(s));
}

void put_null(ing::AvroValue* v) noexcept {
    assert(v != nullptr);
    std::memset(v, 0, sizeof(*v));
    v->is_null = true;
}

// G2ICE-135 — a non-null kArray/kMap field: `bytes`/`bytes_len` carry the
// caller's pre-encoded Avro array body (see bolt_avro.h's updated contract
// for `avro_write_ex`). Mirrors put_null's zero-then-set shape so no stale
// field (the `num` union, `_pad*`) survives from the arena allocation.
void put_raw_container(ing::AvroValue* v, const uint8_t* bytes,
                       uint32_t bytes_len) noexcept {
    assert(v != nullptr);
    assert(bytes != nullptr || bytes_len == 0u);
    std::memset(v, 0, sizeof(*v));
    v->type      = ing::AvroType::kArray;
    v->is_null   = false;
    v->bytes     = bytes;
    v->bytes_len = bytes_len;
}

// Every data file this writer emits is parquet. MarbleDB-derived files are
// synthesized as parquet by construction; a caller with another format needs
// this to become a parameter rather than silently mislabelling the file.
constexpr const char* kFileFormat = "PARQUET";

// ---------------------------------------------------------------------------
// G2ICE-135 — hand-encode the three per-column stats maps this writer now has
// REAL data for (null_value_counts, lower_bounds, upper_bounds) as the raw
// Avro bytes `avro_write_ex` splices verbatim for a non-null container field
// (see bolt_avro.h's updated contract). Iceberg encodes an int-keyed map as
// `array<record<key:int, value:T>>` because Avro maps may only key on string
// (see the header comment at the top of this file), so this is exactly one
// count-prefixed block of (key, value) pairs followed by the zero-count
// terminator every array in this format ends with — mirroring
// bolt_avro.cpp's own write_long/enc_bytes primitives byte-for-byte, since
// that TU does not export them for reuse across the library boundary.
// ---------------------------------------------------------------------------

// Encode an Avro long (zig-zag varint). Returns bytes written (<= 10).
// Identical algorithm to bolt_avro.cpp's private `write_long` — the two
// cannot drift in a way that matters because both implement the same fixed
// Avro spec, and this file's own gtest round-trips through the real reader.
uint32_t enc_long_stat(int64_t v, uint8_t* dst) noexcept {
    assert(dst != nullptr);
    uint64_t u = (static_cast<uint64_t>(v) << 1) ^
                 static_cast<uint64_t>(v >> 63);
    uint32_t n = 0;
    do {                                            // bounded: <= 10 iters
        uint8_t b = static_cast<uint8_t>(u & 0x7Fu);
        u >>= 7;
        if (u != 0u) b |= 0x80u;
        dst[n++] = b;
    } while (u != 0u && n < 10u);
    return n;
}

// Length-prefixed Avro bytes.
uint32_t enc_bytes_stat(uint8_t* dst, const uint8_t* b, uint32_t len) noexcept {
    assert(dst != nullptr);
    uint32_t n = enc_long_stat(static_cast<int64_t>(len), dst);
    if (len > 0) std::memcpy(dst + n, b, len);
    return n + len;
}

// Bound on one encoded stat-map: count varint (<=10) + up to kIcebergMaxStatCols
// entries of (key varint <=5 + widest value <=10 for a long, or <=10+kLakeMaxValBytes
// for a bytes bound) + terminator varint (<=10). Generous, fixed, Tiger-Style.
constexpr uint32_t kStatMapBufCap = 16u +
    kIcebergMaxStatCols * (16u + 10u + kLakeMaxValBytes);

// null_value_counts (or value_counts, if ever needed): one entry per column
// this writer has a null_count for -- always present, because 0 is itself a
// real measured value a caller (e.g. G2FEAT-356's null-safety gate) must be
// able to trust as "certified zero nulls", not "unknown". Returns bytes
// written into `dst`, or 0 when there is nothing to encode (st->n_cols == 0),
// which the caller treats as "write null instead" — an empty map and an
// absent map are both spec-legal but an absent one costs fewer bytes and
// matches this writer's existing all-or-nothing null convention.
uint32_t encode_null_count_map(const FileStats* st, uint8_t* dst,
                               uint32_t cap) noexcept {
    assert(st != nullptr && dst != nullptr);
    if (st->n_cols == 0u) return 0u;
    assert(st->n_cols <= kIcebergMaxStatCols);
    uint32_t o = 0;
    if (o + 10u > cap) return 0u;
    o += enc_long_stat(static_cast<int64_t>(st->n_cols), dst + o);
    for (uint32_t i = 0; i < st->n_cols; ++i) {      // bounded: n_cols
        if (o + 20u > cap) return 0u;
        o += enc_long_stat(st->cols[i].field_id, dst + o);
        o += enc_long_stat(st->cols[i].null_count, dst + o);
    }
    if (o + 1u > cap) return 0u;
    o += enc_long_stat(0, dst + o);                  // terminator block
    return o;
}

// lower_bounds / upper_bounds: one entry per column that HAS that bound
// (has_lower / has_upper) -- a column with none is simply absent from the
// map, exactly what the reader's `find_stat` treats as "no information",
// never a fabricated value.
uint32_t encode_bound_map(const FileStats* st, bool want_lower, uint8_t* dst,
                          uint32_t cap) noexcept {
    assert(st != nullptr && dst != nullptr);
    if (st->n_cols == 0u) return 0u;
    assert(st->n_cols <= kIcebergMaxStatCols);
    uint32_t n_present = 0;
    for (uint32_t i = 0; i < st->n_cols; ++i) {
        if (want_lower ? st->cols[i].has_lower : st->cols[i].has_upper) {
            ++n_present;
        }
    }
    if (n_present == 0u) return 0u;
    uint32_t o = 0;
    if (o + 10u > cap) return 0u;
    o += enc_long_stat(static_cast<int64_t>(n_present), dst + o);
    for (uint32_t i = 0; i < st->n_cols; ++i) {      // bounded: n_cols
        const ColumnStatEntry& c = st->cols[i];
        const bool present = want_lower ? c.has_lower : c.has_upper;
        if (!present) continue;
        const char* bytes = want_lower ? c.lower : c.upper;
        const uint32_t blen = want_lower ? c.lower_len : c.upper_len;
        if (o + 15u + blen > cap) return 0u;
        o += enc_long_stat(c.field_id, dst + o);
        o += enc_bytes_stat(dst + o, reinterpret_cast<const uint8_t*>(bytes),
                            blen);
    }
    if (o + 1u > cap) return 0u;
    o += enc_long_stat(0, dst + o);                  // terminator block
    return o;
}

// Bound on the spliced partition text: per field, the JSON skeleton plus a
// name and two ints; per spec, the brackets.
constexpr uint32_t kPartFieldJsonCap =
    96u + kIcebergMaxFieldName + kIcebergMaxTransformName;
constexpr uint32_t kPartJsonCap = 8u + kIcebergMaxFieldsPerSpec * kPartFieldJsonCap;

// Avro names are [A-Za-z_][A-Za-z0-9_]*. A name outside that grammar would
// make the schema unparseable or, worse, bind a different field.
bool avro_name_ok(const char* n) noexcept {
    assert(n != nullptr);
    if (n[0] == '\0') return false;
    for (uint32_t i = 0; i < kIcebergMaxFieldName; ++i) {   // bounded
        const char c = n[i];
        if (c == '\0') return true;
        const bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                           c == '_';
        const bool digit = c >= '0' && c <= '9';
        if (!(alpha || (digit && i != 0u))) return false;
    }
    return false;   // unterminated within the field
}

// Advance `*o` by an snprintf result; false on error or overflow.
bool part_advance(int n, uint32_t cap, uint32_t* o) noexcept {
    assert(o != nullptr);
    assert(*o <= cap);
    if (n < 0 || static_cast<uint32_t>(n) >= cap - *o) return false;
    *o += static_cast<uint32_t>(n);
    return true;
}

// The `partition` record's fields (Avro) and the `partition-spec` metadata
// value (Iceberg JSON), both rendered from the one spec.
bool render_partition(const PartitionSpec* spec,
                      const PartitionAvroType* types, char* avro,
                      uint32_t* avro_len, char* meta,
                      uint32_t* meta_len) noexcept {
    assert(spec != nullptr && types != nullptr);
    assert(avro != nullptr && meta != nullptr);
    uint32_t a = 0;
    uint32_t m = 1;
    meta[0] = '[';
    for (uint32_t i = 0; i < spec->n_fields; ++i) {   // bounded: kMaxFieldsPerSpec
        const PartitionField& f = spec->fields[i];
        if (!avro_name_ok(f.name)) return false;
        char tf[kIcebergMaxTransformName];
        if (transform_format(f.transform, tf, sizeof(tf)) == 0u) return false;
        const char* t = types[i] == PartitionAvroType::kInt ? "int" : "long";
        const char* sep = i == 0u ? "" : ",";
        if (!part_advance(std::snprintf(avro + a, kPartJsonCap - a,
                                        "%s{\"name\":\"%s\",\"field-id\":%d,"
                                        "\"type\":[\"null\",\"%s\"],"
                                        "\"default\":null}",
                                        sep, f.name, f.field_id, t),
                          kPartJsonCap, &a)) {
            return false;
        }
        if (!part_advance(std::snprintf(meta + m, kPartJsonCap - m,
                                        "%s{\"name\":\"%s\",\"transform\":"
                                        "\"%s\",\"source-id\":%d,"
                                        "\"field-id\":%d}",
                                        sep, f.name, tf, f.source_id,
                                        f.field_id),
                          kPartJsonCap, &m)) {
            return false;
        }
    }
    if (m + 2u > kPartJsonCap) return false;
    meta[m++] = ']';
    meta[m]   = '\0';
    *avro_len = a;
    *meta_len = m;
    assert(m >= 2u);
    return true;
}

// kManifestEntrySchema with `avro[0..avro_len)` spliced into the empty
// `partition` record. The splice point is located, never assumed, so a schema
// edit that moves it fails the write instead of corrupting it.
bool splice_entry_schema(const char* avro, uint32_t avro_len, Arena* scratch,
                         const char** out, uint32_t* out_len) noexcept {
    assert(avro != nullptr && scratch != nullptr);
    assert(out != nullptr && out_len != nullptr);
    static const char kMark[] = "\"fields\":[],\"name\":\"r102\"";
    const char* hit = std::strstr(kManifestEntrySchema, kMark);
    if (hit == nullptr) return false;
    const uint32_t head =
        static_cast<uint32_t>(hit - kManifestEntrySchema) + 10u;   // past `"fields":[`
    const uint32_t base = static_cast<uint32_t>(sizeof(kManifestEntrySchema) - 1u);
    char* dst = scratch->allocate_array<char>(base + avro_len + 1u);
    if (dst == nullptr) return false;
    std::memcpy(dst, kManifestEntrySchema, head);
    std::memcpy(dst + head, avro, avro_len);
    std::memcpy(dst + head + avro_len, kManifestEntrySchema + head, base - head);
    dst[base + avro_len] = '\0';
    *out     = dst;
    *out_len = base + avro_len;
    return true;
}

}  // namespace

namespace {

// Shared body. `spec == nullptr` (or zero fields) is the unpartitioned form
// and encodes byte-identically to what this writer always produced.
bool write_manifest(const DataFileRef* files, uint32_t n_files,
                    int64_t snapshot_id, int64_t sequence_number,
                    const char* table_schema_json, uint32_t table_schema_len,
                    int32_t partition_spec_id, const PartitionSpec* spec,
                    const PartitionAvroType* result_types, Arena* scratch,
                    const uint8_t** out, uint64_t* out_len) noexcept {
    assert(scratch != nullptr);
    assert(out != nullptr && out_len != nullptr);
    if (files == nullptr && n_files != 0) return false;
    if (scratch == nullptr || out == nullptr || out_len == nullptr) return false;
    if (table_schema_json == nullptr || table_schema_len == 0) return false;
    if (n_files > kIcebergMaxManifestEntries) return false;
    const uint32_t np = spec == nullptr ? 0u : spec->n_fields;
    if (np > kIcebergMaxFieldsPerSpec || np > kIcebergMaxPartitionValues) return false;
    if (np != 0u && result_types == nullptr) return false;
    const uint32_t nf = kEntryFields + np;

    const char* entry_schema = kManifestEntrySchema;
    uint32_t    entry_schema_len =
        static_cast<uint32_t>(sizeof(kManifestEntrySchema) - 1u);
    const char* spec_json = "[]";
    uint32_t    spec_json_len = 2u;
    if (np != 0u) {
        char* avro = scratch->allocate_array<char>(kPartJsonCap);
        char* meta = scratch->allocate_array<char>(kPartJsonCap);
        if (avro == nullptr || meta == nullptr) return false;
        uint32_t avro_len = 0;
        if (!render_partition(spec, result_types, avro, &avro_len, meta,
                              &spec_json_len)) {
            return false;
        }
        spec_json = meta;
        if (!splice_entry_schema(avro, avro_len, scratch, &entry_schema,
                                 &entry_schema_len)) {
            return false;
        }
    }

    ing::AvroField* fields = scratch->allocate_array<ing::AvroField>(nf);
    if (fields == nullptr) return false;
    {
        // The partition tuple is inline after file_format (see the wire-order
        // note on build_entry_fields).
        ing::AvroField base[kEntryFields];
        build_entry_fields(base);
        std::memcpy(fields, base, 7u * sizeof(ing::AvroField));
        for (uint32_t p = 0; p < np; ++p) {            // bounded: np
            set_field(&fields[7u + p], spec->fields[p].name,
                      result_types[p] == PartitionAvroType::kInt
                          ? ing::AvroType::kInt : ing::AvroType::kLong,
                      true);
        }
        std::memcpy(fields + 7u + np, base + 7u,
                    (kEntryFields - 7u) * sizeof(ing::AvroField));
    }

    ing::AvroValue* rows =
        scratch->allocate_array<ing::AvroValue>(
            static_cast<uint64_t>(n_files) * nf + 1u);
    if (rows == nullptr) return false;

    // A manifest is homogeneous by spec: it holds data files or delete files,
    // never both, and its OCF `content` metadata says which. Deriving it from
    // the files (rather than taking it as a parameter) makes the two
    // impossible to disagree; a mixed set is refused rather than mislabelled,
    // because a reader trusts the label and would apply a delete file as data.
    const bool deletes_manifest =
        (n_files != 0 && files[0].content != FileContent::kData);
    for (uint32_t r = 1; r < n_files; ++r) {          // bounded by n_files
        const bool d = (files[r].content != FileContent::kData);
        if (d != deletes_manifest) return false;
    }

    uint64_t value_bytes = 0;
    for (uint32_t r = 0; r < n_files; ++r) {          // bounded by n_files
        const DataFileRef* d = &files[r];
        // A partitioned file needs partition-tuple fields this flat schema
        // does not declare. Writing it anyway yields a manifest that PARSES
        // and reports the wrong partition — fail instead.
        if (d->n_partition != np) return false;
        ing::AvroValue* v = rows + static_cast<uint64_t>(r) * nf;
        uint32_t i = 0;
        put_long(&v[i++], static_cast<int64_t>(d->status));
        put_long(&v[i++], d->snapshot_id != 0 ? d->snapshot_id : snapshot_id);
        put_long(&v[i++], sequence_number);
        put_long(&v[i++], sequence_number);
        put_long(&v[i++], static_cast<int64_t>(d->content));
        put_str(&v[i++], d->file_path);
        put_str(&v[i++], kFileFormat);
        for (uint32_t p = 0; p < np; ++p) {            // bounded: np
            const PartitionValue& pv = d->partition[p];
            if (pv.is_null) { put_null(&v[i++]); continue; }
            if (!pv.is_int || pv.is_str) return false;
            if (result_types[p] == PartitionAvroType::kInt &&
                (pv.i64 < INT32_MIN || pv.i64 > INT32_MAX)) {
                return false;
            }
            put_long(&v[i++], pv.i64);
        }
        put_long(&v[i++], d->stats.record_count);
        put_long(&v[i++], d->stats.file_size_in_bytes);
        // SEVEN fields here (was a flat put_null loop before G2ICE-135):
        // column_sizes, value_counts, null_value_counts, nan_value_counts,
        // lower_bounds, upper_bounds, key_metadata — exactly the six stats
        // maps plus key_metadata, matching build_entry_fields's declared
        // order. column_sizes/value_counts/nan_value_counts/key_metadata stay
        // null (this writer has no per-column compressed-byte-size or NaN
        // tracking, and value_counts has no downstream reader — see
        // compute_file_stats's header comment in iceberg_writer.cpp for the
        // scope decision). null_value_counts/lower_bounds/upper_bounds now
        // carry REAL data when `d->stats.n_cols > 0` (populated by
        // append_commit_ex / table_overwrite's compute_file_stats call).
        //
        // Historical note on the count (kept: still true and still the bug
        // this session's predecessor found): writing nine values into a
        // 19-wide row silently misaligned every following field and wrote one
        // element past `rows`'s `+ 1u` slack — only visible because this
        // repo's Release build keeps asserts live (G2ICE-4). The assert below
        // is the same tripwire.
        put_null(&v[i++]);                                   // column_sizes
        put_null(&v[i++]);                                   // value_counts
        {
            uint8_t* buf = scratch->allocate_array<uint8_t>(kStatMapBufCap);
            const uint32_t n = (buf == nullptr)
                ? 0u : encode_null_count_map(&d->stats, buf, kStatMapBufCap);
            if (n == 0u) put_null(&v[i++]);
            else         put_raw_container(&v[i++], buf, n);
        }
        put_null(&v[i++]);                                   // nan_value_counts
        {
            uint8_t* buf = scratch->allocate_array<uint8_t>(kStatMapBufCap);
            const uint32_t n = (buf == nullptr)
                ? 0u : encode_bound_map(&d->stats, true, buf, kStatMapBufCap);
            if (n == 0u) put_null(&v[i++]);
            else         put_raw_container(&v[i++], buf, n);
        }
        {
            uint8_t* buf = scratch->allocate_array<uint8_t>(kStatMapBufCap);
            const uint32_t n = (buf == nullptr)
                ? 0u : encode_bound_map(&d->stats, false, buf, kStatMapBufCap);
            if (n == 0u) put_null(&v[i++]);
            else         put_raw_container(&v[i++], buf, n);
        }
        put_null(&v[i++]);                                   // key_metadata
        put_null(&v[i++]);                                   // split_offsets
        put_null(&v[i++]);                                   // equality_ids
        put_null(&v[i++]);                                   // sort_order_id
        assert(i == nf);
        value_bytes += v[5].bytes_len + v[6].bytes_len +
                       v[11u + np].bytes_len + v[13u + np].bytes_len +
                       v[14u + np].bytes_len;
    }

    char spec_id_buf[16];
    const int spec_id_len =
        std::snprintf(spec_id_buf, sizeof(spec_id_buf), "%d", partition_spec_id);
    if (spec_id_len <= 0) return false;

    const ing::AvroMetaKV meta[] = {
        {"schema", reinterpret_cast<const uint8_t*>(table_schema_json),
         table_schema_len, 0u},
        {"partition-spec", reinterpret_cast<const uint8_t*>(spec_json),
         spec_json_len, 0u},
        {"partition-spec-id", reinterpret_cast<const uint8_t*>(spec_id_buf),
         static_cast<uint32_t>(spec_id_len), 0u},
        {"format-version", reinterpret_cast<const uint8_t*>("2"), 1u, 0u},
        {"content",
         reinterpret_cast<const uint8_t*>(deletes_manifest ? "deletes"
                                                           : "data"),
         deletes_manifest ? 7u : 4u, 0u},
    };
    constexpr uint32_t kNMeta = sizeof(meta) / sizeof(meta[0]);

    uint64_t meta_bytes = entry_schema_len + 1u + table_schema_len;
    for (uint32_t m = 0; m < kNMeta; ++m) meta_bytes += meta[m].val_len + 32u;

    uint64_t cap = ing::avro_write_ex_max_len(fields, nf, value_bytes,
                                              n_files, meta_bytes);
    uint8_t* dst = scratch->allocate_array<uint8_t>(cap);
    if (dst == nullptr) return false;

    uint64_t len = cap;
    if (!ing::avro_write_ex(fields, nf, rows, n_files, entry_schema,
                            entry_schema_len, meta, kNMeta, kSync, dst, &len)) {
        return false;
    }
    assert(len <= cap);
    *out     = dst;
    *out_len = len;
    return true;
}

}  // namespace

bool manifest_write_avro(const DataFileRef* files, uint32_t n_files,
                         int64_t snapshot_id, int64_t sequence_number,
                         const char* table_schema_json,
                         uint32_t table_schema_len,
                         int32_t partition_spec_id, Arena* scratch,
                         const uint8_t** out, uint64_t* out_len) noexcept {
    assert(scratch != nullptr);
    assert(out != nullptr && out_len != nullptr);
    return write_manifest(files, n_files, snapshot_id, sequence_number,
                          table_schema_json, table_schema_len,
                          partition_spec_id, nullptr, nullptr, scratch, out,
                          out_len);
}

bool manifest_write_avro_partitioned(const DataFileRef* files, uint32_t n_files,
                                     int64_t snapshot_id,
                                     int64_t sequence_number,
                                     const char* table_schema_json,
                                     uint32_t table_schema_len,
                                     const PartitionSpec* spec,
                                     const PartitionAvroType* result_types,
                                     Arena* scratch, const uint8_t** out,
                                     uint64_t* out_len) noexcept {
    assert(scratch != nullptr);
    assert(out != nullptr && out_len != nullptr);
    if (spec == nullptr) return false;
    return write_manifest(files, n_files, snapshot_id, sequence_number,
                          table_schema_json, table_schema_len, spec->spec_id,
                          spec, result_types, scratch, out, out_len);
}

bool manifest_list_write_avro(const ManifestListEntry* entries,
                              uint32_t n_entries, int64_t snapshot_id,
                              int64_t sequence_number, Arena* scratch,
                              const uint8_t** out, uint64_t* out_len) noexcept {
    assert(scratch != nullptr);
    assert(out != nullptr && out_len != nullptr);
    if (entries == nullptr && n_entries != 0) return false;
    if (scratch == nullptr || out == nullptr || out_len == nullptr) return false;
    if (n_entries > kIcebergMaxManifestsPerList) return false;

    ing::AvroField* fields = scratch->allocate_array<ing::AvroField>(kListFields);
    if (fields == nullptr) return false;
    build_list_fields(fields);

    ing::AvroValue* rows =
        scratch->allocate_array<ing::AvroValue>(
            static_cast<uint64_t>(n_entries) * kListFields + 1u);
    if (rows == nullptr) return false;

    uint64_t value_bytes = 0;
    for (uint32_t r = 0; r < n_entries; ++r) {        // bounded by n_entries
        const ManifestListEntry* e = &entries[r];
        ing::AvroValue* v = rows + static_cast<uint64_t>(r) * kListFields;
        uint32_t i = 0;
        put_str(&v[i++], e->manifest_path);
        put_long(&v[i++], e->manifest_length);
        put_long(&v[i++], static_cast<int64_t>(e->partition_spec_id));
        put_long(&v[i++], static_cast<int64_t>(e->content));
        // A carried-forward manifest keeps the sequence number it was
        // COMMITTED at; only a manifest this commit adds takes the current
        // one. Stamping every entry with `sequence_number` silently disarms
        // positional deletes — see ManifestListEntry::sequence_number.
        {
            const int64_t seq =
                e->sequence_number != 0 ? e->sequence_number : sequence_number;
            const int64_t mseq =
                e->min_sequence_number != 0 ? e->min_sequence_number : seq;
            put_long(&v[i++], seq);
            put_long(&v[i++], mseq);
        }
        put_long(&v[i++], e->added_snapshot_id != 0 ? e->added_snapshot_id
                                                    : snapshot_id);
        // G2ICE-49 — these five were hardcoded 0 regardless of what the
        // caller had computed: `ManifestListEntry` had no fields to hold
        // anything else, so there was nothing else TO write. The struct now
        // carries real per-manifest tallies (see its declaration); this
        // writer's job is only to place them, same as `added_files_count`
        // always did.
        put_long(&v[i++], e->added_files_count);
        put_long(&v[i++], e->existing_files_count);
        put_long(&v[i++], e->deleted_files_count);
        put_long(&v[i++], e->added_rows_count);
        put_long(&v[i++], e->existing_rows_count);
        put_long(&v[i++], e->deleted_rows_count);
        put_null(&v[i++]);                             // partitions
        put_null(&v[i++]);                             // key_metadata
        assert(i == kListFields);
        value_bytes += v[0].bytes_len;
    }

    const ing::AvroMetaKV meta[] = {
        {"format-version", reinterpret_cast<const uint8_t*>("2"), 1u, 0u},
        {"content", reinterpret_cast<const uint8_t*>("data"), 4u, 0u},
    };
    constexpr uint32_t kNMeta = sizeof(meta) / sizeof(meta[0]);

    uint64_t meta_bytes = sizeof(kManifestFileSchema);
    for (uint32_t m = 0; m < kNMeta; ++m) meta_bytes += meta[m].val_len + 32u;

    uint64_t cap = ing::avro_write_ex_max_len(fields, kListFields, value_bytes,
                                              n_entries, meta_bytes);
    uint8_t* dst = scratch->allocate_array<uint8_t>(cap);
    if (dst == nullptr) return false;

    uint64_t len = cap;
    if (!ing::avro_write_ex(fields, kListFields, rows, n_entries,
                            kManifestFileSchema,
                            static_cast<uint32_t>(sizeof(kManifestFileSchema) - 1u),
                            meta, kNMeta, kSync, dst, &len)) {
        return false;
    }
    assert(len <= cap);
    *out     = dst;
    *out_len = len;
    return true;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
