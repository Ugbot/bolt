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

// Every data file this writer emits is parquet. MarbleDB-derived files are
// synthesized as parquet by construction; a caller with another format needs
// this to become a parameter rather than silently mislabelling the file.
constexpr const char* kFileFormat = "PARQUET";

}  // namespace

bool manifest_write_avro(const DataFileRef* files, uint32_t n_files,
                         int64_t snapshot_id, int64_t sequence_number,
                         const char* table_schema_json,
                         uint32_t table_schema_len,
                         int32_t partition_spec_id, Arena* scratch,
                         const uint8_t** out, uint64_t* out_len) noexcept {
    assert(scratch != nullptr);
    assert(out != nullptr && out_len != nullptr);
    if (files == nullptr && n_files != 0) return false;
    if (scratch == nullptr || out == nullptr || out_len == nullptr) return false;
    if (table_schema_json == nullptr || table_schema_len == 0) return false;
    if (n_files > kIcebergMaxManifestEntries) return false;

    ing::AvroField* fields = scratch->allocate_array<ing::AvroField>(kEntryFields);
    if (fields == nullptr) return false;
    build_entry_fields(fields);

    ing::AvroValue* rows =
        scratch->allocate_array<ing::AvroValue>(
            static_cast<uint64_t>(n_files) * kEntryFields + 1u);
    if (rows == nullptr) return false;

    uint64_t value_bytes = 0;
    for (uint32_t r = 0; r < n_files; ++r) {          // bounded by n_files
        const DataFileRef* d = &files[r];
        // A partitioned file needs partition-tuple fields this flat schema
        // does not declare. Writing it anyway yields a manifest that PARSES
        // and reports the wrong partition — fail instead.
        if (d->n_partition != 0) return false;
        ing::AvroValue* v = rows + static_cast<uint64_t>(r) * kEntryFields;
        uint32_t i = 0;
        put_long(&v[i++], static_cast<int64_t>(d->status));
        put_long(&v[i++], d->snapshot_id != 0 ? d->snapshot_id : snapshot_id);
        put_long(&v[i++], sequence_number);
        put_long(&v[i++], sequence_number);
        put_long(&v[i++], static_cast<int64_t>(d->content));
        put_str(&v[i++], d->file_path);
        put_str(&v[i++], kFileFormat);
        put_long(&v[i++], d->stats.record_count);
        put_long(&v[i++], d->stats.file_size_in_bytes);
        // SEVEN, not nine: column_sizes, value_counts, null_value_counts,
        // nan_value_counts, lower_bounds, upper_bounds, key_metadata — the
        // six stats maps plus key_metadata, exactly as build_entry_fields
        // declares them. Writing nine put 21 values into a 19-wide row: each
        // row spilled two values into the next row's slot (masked, because the
        // next row promptly overwrote them) and the LAST row wrote one element
        // past `rows`, whose `+ 1u` slack was hiding a real heap overflow.
        //
        // Invisible in a stock CMake Release, which defines NDEBUG and
        // compiles the assert below away. THIS repo's Release does not
        // (`CMAKE_CXX_FLAGS_RELEASE=-O3 -g`), so the assert is live code and
        // fires — which is how registering these tests here (G2ICE-4) surfaced
        // it. The bytes on the wire were always correct, because the encoder
        // walks the 19 SCHEMA fields and never read the two extra values.
        for (uint32_t k = 0; k < 7; ++k) put_null(&v[i++]);  // 6 stats + key_md
        put_null(&v[i++]);                                   // split_offsets
        put_null(&v[i++]);                                   // equality_ids
        put_null(&v[i++]);                                   // sort_order_id
        assert(i == kEntryFields);
        value_bytes += v[5].bytes_len + v[6].bytes_len;
    }

    char spec_id_buf[16];
    const int spec_id_len =
        std::snprintf(spec_id_buf, sizeof(spec_id_buf), "%d", partition_spec_id);
    if (spec_id_len <= 0) return false;

    const ing::AvroMetaKV meta[] = {
        {"schema", reinterpret_cast<const uint8_t*>(table_schema_json),
         table_schema_len, 0u},
        {"partition-spec", reinterpret_cast<const uint8_t*>("[]"), 2u, 0u},
        {"partition-spec-id", reinterpret_cast<const uint8_t*>(spec_id_buf),
         static_cast<uint32_t>(spec_id_len), 0u},
        {"format-version", reinterpret_cast<const uint8_t*>("2"), 1u, 0u},
        {"content", reinterpret_cast<const uint8_t*>("data"), 4u, 0u},
    };
    constexpr uint32_t kNMeta = sizeof(meta) / sizeof(meta[0]);

    uint64_t meta_bytes = sizeof(kManifestEntrySchema) + table_schema_len;
    for (uint32_t m = 0; m < kNMeta; ++m) meta_bytes += meta[m].val_len + 32u;

    uint64_t cap = ing::avro_write_ex_max_len(fields, kEntryFields, value_bytes,
                                              n_files, meta_bytes);
    uint8_t* dst = scratch->allocate_array<uint8_t>(cap);
    if (dst == nullptr) return false;

    uint64_t len = cap;
    if (!ing::avro_write_ex(fields, kEntryFields, rows, n_files,
                            kManifestEntrySchema,
                            static_cast<uint32_t>(sizeof(kManifestEntrySchema) - 1u),
                            meta, kNMeta, kSync, dst, &len)) {
        return false;
    }
    assert(len <= cap);
    *out     = dst;
    *out_len = len;
    return true;
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
        put_long(&v[i++], sequence_number);
        put_long(&v[i++], sequence_number);
        put_long(&v[i++], e->added_snapshot_id != 0 ? e->added_snapshot_id
                                                    : snapshot_id);
        put_long(&v[i++], e->added_files_count);
        put_long(&v[i++], 0);                          // existing_files_count
        put_long(&v[i++], 0);                          // deleted_files_count
        put_long(&v[i++], 0);                          // added_rows_count
        put_long(&v[i++], 0);                          // existing_rows_count
        put_long(&v[i++], 0);                          // deleted_rows_count
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
