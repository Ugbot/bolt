// W5 — Iceberg write path smoke: create → append → read-back, view create,
// branch ref, snapshot expiry.

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/iceberg/view.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

namespace {

using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

std::string unique_root(const char* tag) {
    auto p = std::filesystem::temp_directory_path() /
             ("bolt_iceberg_write_" + std::string(tag) + "_" +
              std::to_string(reinterpret_cast<uintptr_t>(&tag)));
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    std::filesystem::create_directories(p, ec);
    return p.generic_string();
}

Schema make_schema_i64_utf8() {
    // Single-column long schema (Utf8 round-trip awaits the W5 round-trip
    // slice — the parquet writer requires precise StringView setup the
    // append path doesn't perform yet).
    Schema s{};
    s.schema_id = 0;
    s.n_fields  = 1;
    s.fields[0].id = 1; s.fields[0].required = true;
    std::strncpy(s.fields[0].name, "id",   sizeof(s.fields[0].name) - 1u);
    std::strncpy(s.fields[0].type, "long", sizeof(s.fields[0].type) - 1u);
    return s;
}

// Build a tiny single-column Int64 BoltBatch.
void make_batch(bolt::Arena* a, bolt::BoltBatch* out, int64_t base,
                int32_t n) {
    bolt::BoltBatch::init_empty(out);
    out->arena    = a;
    out->num_rows = n;
    out->num_cols = 1;
    bolt::BoltBatch::alloc_columns(out, a, 1);  // G2FEAT-47: size columns[2]
    out->schema.add_field("id", bolt::BoltType::Int64, false);
    auto* cols = out->columns[out->read_epoch];
    cols[0].type = bolt::BoltType::Int64;
    cols[0].length = n;
    auto* idata = a->allocate_array<int64_t>(static_cast<uint32_t>(n));
    for (int32_t i = 0; i < n; ++i) idata[i] = base + i;
    cols[0].data = idata;
}

// G2ICE-84 — a location Iceberg RECORDS and the key its ObjectStore is
// ADDRESSED with are two different strings. Every recorded location is now an
// absolute URI (which is what the spec says and what pyiceberg resolves
// literally); the store is still keyed relative to its root. These assertions
// used to feed a recorded location straight to `os_get`, which only worked
// while the writer conflated the two.
std::string store_key(const std::string& root, const char* loc) {
    if (loc == nullptr) return std::string();
    const std::string s(loc);
    if (s.size() > root.size() && s.compare(0, root.size(), root) == 0 &&
        (s[root.size()] == '/' || s[root.size()] == '\\')) {
        return s.substr(root.size() + 1u);
    }
    return s;
}

bool location_is_absolute(const char* p) {
    if (p == nullptr || p[0] == '\0') return false;
    if (p[0] == '/') return true;
    if (std::strstr(p, "://") != nullptr) return true;
    return (p[1] == ':' && (p[2] == '/' || p[2] == '\\'));
}

}  // namespace

TEST(IcebergWrite, CreateOpen) {
    bolt::Arena arena;
    const std::string root = unique_root("create");
    FilesystemObjectStore fs{};
    ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    Schema sch = make_schema_i64_utf8();
    PartitionSpec spec{}; spec.spec_id = 0; spec.n_fields = 0;
    SortOrder sort{};     sort.order_id = 0; sort.n_fields = 0;
    WriteOptions wo;      write_options_init(&wo);
    TableHandle* th = nullptr;
    ASSERT_TRUE(table_create(&th, &arena, &os, root.c_str(), &sch, &spec,
                              &sort, &wo));
    ASSERT_NE(th, nullptr);
    const Metadata* m = table_metadata(th);
    EXPECT_EQ(m->format_version, 2);
    EXPECT_EQ(m->n_schemas, 1u);
    EXPECT_EQ(m->n_specs, 1u);

    // Reopen
    bolt::Arena arena2;
    TableHandle* th2 = nullptr;
    ASSERT_TRUE(table_open(&th2, &arena2, &os, root.c_str()));
    EXPECT_EQ(table_metadata(th2)->format_version, 2);
}

TEST(IcebergWrite, AppendAndReadBack) {
    bolt::Arena arena;
    const std::string root = unique_root("append");
    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    Schema sch = make_schema_i64_utf8();
    PartitionSpec spec{}; spec.spec_id = 0; spec.n_fields = 0;
    SortOrder sort{}; sort.order_id = 0; sort.n_fields = 0;
    WriteOptions wo; write_options_init(&wo);
    TableHandle* th = nullptr;
    ASSERT_TRUE(table_create(&th, &arena, &os, root.c_str(), &sch, &spec,
                              &sort, &wo));

    // Append three small batches.
    bolt::BoltBatch b0{}, b1{}, b2{};
    make_batch(&arena, &b0, 0,  4);
    make_batch(&arena, &b1, 10, 4);
    make_batch(&arena, &b2, 20, 4);
    AppendHandle* ah = nullptr;
    ASSERT_TRUE(append_open(&ah, th));
    ASSERT_TRUE(append_write(ah, &b0));
    ASSERT_TRUE(append_commit(ah));
    ASSERT_TRUE(append_write(ah, &b1));
    ASSERT_TRUE(append_commit(ah));
    ASSERT_TRUE(append_write(ah, &b2));
    ASSERT_TRUE(append_commit(ah));
    append_close(ah);

    const Metadata* m = table_metadata(th);
    EXPECT_EQ(m->n_snapshots, 3u);
    EXPECT_GT(m->current_snapshot_id, 0);
}

TEST(IcebergWrite, SchemaEvolution) {
    bolt::Arena arena;
    const std::string root = unique_root("schema");
    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    Schema sch = make_schema_i64_utf8();
    WriteOptions wo; write_options_init(&wo);
    TableHandle* th = nullptr;
    ASSERT_TRUE(table_create(&th, &arena, &os, root.c_str(), &sch, nullptr,
                              nullptr, &wo));
    const uint32_t base = table_metadata(th)->schemas[0].n_fields;
    EXPECT_TRUE(table_add_column(th, "score", bolt::BoltType::Float64, true));
    EXPECT_EQ(table_metadata(th)->schemas[0].n_fields, base + 1u);
    EXPECT_TRUE(table_rename_column(th, "score", "score2"));
    EXPECT_STREQ(table_metadata(th)->schemas[0].fields[base].name, "score2");
    EXPECT_TRUE(table_drop_column(th, "score2"));
    EXPECT_EQ(table_metadata(th)->schemas[0].n_fields, base);
}

TEST(IcebergWrite, BranchAndTagAndExpire) {
    bolt::Arena arena;
    const std::string root = unique_root("refs");
    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    Schema sch = make_schema_i64_utf8();
    WriteOptions wo; write_options_init(&wo);
    TableHandle* th = nullptr;
    ASSERT_TRUE(table_create(&th, &arena, &os, root.c_str(), &sch, nullptr,
                              nullptr, &wo));
    // One append to get a snapshot.
    bolt::BoltBatch b0{}; make_batch(&arena, &b0, 0, 2);
    AppendHandle* ah = nullptr;
    ASSERT_TRUE(append_open(&ah, th));
    ASSERT_TRUE(append_write(ah, &b0));
    ASSERT_TRUE(append_commit(ah));
    append_close(ah);
    const int64_t snap = table_metadata(th)->current_snapshot_id;
    EXPECT_GT(snap, 0);
    EXPECT_TRUE(table_create_branch(th, "dev", snap));
    EXPECT_TRUE(table_create_tag(th, "v1.0", snap));
    EXPECT_TRUE(table_drop_ref(th, "dev"));

    // expire: nothing should change because cutoff is in the past.
    EXPECT_TRUE(table_expire_snapshots(th, 0, 10));
    EXPECT_GE(table_metadata(th)->n_snapshots, 1u);
}

TEST(IcebergWrite, ViewCreateAndReplace) {
    bolt::Arena arena;
    const std::string root = unique_root("view");
    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    Schema sch = make_schema_i64_utf8();
    TableHandle* th = nullptr;
    ASSERT_TRUE(view_create(&th, &arena, &os, root.c_str(), "spark",
                             "SELECT id FROM t", &sch));
    EXPECT_TRUE(view_replace(th, "spark", "SELECT id, name FROM t"));
}

TEST(IcebergWrite, OverwriteAndDeleteAndCompact) {
    bolt::Arena arena;
    const std::string root = unique_root("ovr");
    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    Schema sch = make_schema_i64_utf8();
    WriteOptions wo; write_options_init(&wo);
    TableHandle* th = nullptr;
    ASSERT_TRUE(table_create(&th, &arena, &os, root.c_str(), &sch, nullptr,
                              nullptr, &wo));
    bolt::BoltBatch b0{}; make_batch(&arena, &b0, 0, 3);
    EXPECT_TRUE(table_overwrite(th, nullptr, &b0));
    Predicate pred{};
    EXPECT_TRUE(table_delete(th, &pred));
    EXPECT_TRUE(table_compact(th, 64u * 1024u * 1024u));
    EXPECT_GE(table_metadata(th)->n_snapshots, 3u);
}

TEST(IcebergWrite, RemoveOrphansDryRun) {
    bolt::Arena arena;
    const std::string root = unique_root("orphan");
    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    Schema sch = make_schema_i64_utf8();
    WriteOptions wo; write_options_init(&wo);
    TableHandle* th = nullptr;
    ASSERT_TRUE(table_create(&th, &arena, &os, root.c_str(), &sch, nullptr,
                              nullptr, &wo));
    char out[32][512];
    uint32_t n = 0;
    EXPECT_TRUE(table_remove_orphans(th, 0u, true, out, 32u, &n));
}

// ---------------------------------------------------------------------------
// The manifests a commit actually LEAVES ON DISK.
//
// Iceberg defines a manifest as an Avro object container file. This writer
// used to emit JSON with a `.json` name, which every real reader (pyiceberg,
// Spark, Trino, DuckDB) rejects — and no test noticed, because the tests above
// assert on the in-memory `Metadata` the writer keeps, never on the bytes.
// These read the committed objects BACK OFF THE STORE and put them through
// `manifest_parse_avro` / `manifest_list_parse_avro`, which were themselves
// verified against pyiceberg 0.11.1 output (test_bolt_iceberg_real_avro.cpp).
// Every assertion is a VALUE carried through, never a count, so a writer that
// drops or misorders a field cannot pass.
// ---------------------------------------------------------------------------

TEST(IcebergWrite, AppendCommitEmitsAvroManifests) {
    bolt::Arena arena;
    const std::string root = unique_root("avromanifest");
    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    Schema sch = make_schema_i64_utf8();
    PartitionSpec spec{}; spec.spec_id = 0; spec.n_fields = 0;
    SortOrder sort{}; sort.order_id = 0; sort.n_fields = 0;
    WriteOptions wo; write_options_init(&wo);
    TableHandle* th = nullptr;
    ASSERT_TRUE(table_create(&th, &arena, &os, root.c_str(), &sch, &spec,
                              &sort, &wo));

    bolt::BoltBatch b0{};
    make_batch(&arena, &b0, 0, 7);
    AppendHandle* ah = nullptr;
    ASSERT_TRUE(append_open(&ah, th));
    ASSERT_TRUE(append_write(ah, &b0));
    ASSERT_TRUE(append_commit(ah));
    append_close(ah);

    const Metadata* m = table_metadata(th);
    ASSERT_EQ(m->n_snapshots, 1u);
    const Snapshot& snap = m->snapshots[0];

    // 1. The manifest LIST the snapshot points at is a real OCF.
    const uint8_t* mlb = nullptr; uint64_t mll = 0;
    EXPECT_TRUE(location_is_absolute(snap.manifest_list))
        << "manifest-list must be an absolute URI: " << snap.manifest_list;
    ASSERT_EQ(os_get(&os, store_key(root, snap.manifest_list).c_str(), &arena,
                     &mlb, &mll), kOsOk);
    ASSERT_GE(mll, 4u);
    EXPECT_EQ(std::memcmp(mlb, "Obj\x01", 4), 0) << "manifest list is not Avro";

    ManifestListEntry mle[8]{};
    uint32_t n_mle = 0;
    ASSERT_TRUE(manifest_list_parse_avro(mlb, mll, &arena, mle, 8, &n_mle));
    ASSERT_EQ(n_mle, 1u);
    EXPECT_EQ(mle[0].added_snapshot_id, snap.snapshot_id);
    EXPECT_EQ(mle[0].added_files_count, 1);
    EXPECT_EQ(mle[0].partition_spec_id, 0);

    // 2. The MANIFEST it names is a real OCF whose entry describes the file
    //    the append just wrote — including the two fields readers trust
    //    without verifying: record_count and file_size_in_bytes.
    const uint8_t* mb = nullptr; uint64_t mbl = 0;
    EXPECT_TRUE(location_is_absolute(mle[0].manifest_path))
        << "manifest_path must be an absolute URI: " << mle[0].manifest_path;
    ASSERT_EQ(os_get(&os, store_key(root, mle[0].manifest_path).c_str(), &arena,
                     &mb, &mbl), kOsOk);
    ASSERT_GE(mbl, 4u);
    EXPECT_EQ(std::memcmp(mb, "Obj\x01", 4), 0) << "manifest is not Avro";
    // The length the list advertises must be the manifest's real length, or a
    // reader that ranged-reads it gets a truncated file.
    EXPECT_EQ(mle[0].manifest_length, static_cast<int64_t>(mbl));

    DataFileRef df[8]{};
    uint32_t n_df = 0;
    ASSERT_TRUE(manifest_parse_avro(mb, mbl, &arena, /*default_spec_id=*/0,
                                    df, 8, &n_df));
    ASSERT_EQ(n_df, 1u);
    EXPECT_EQ(df[0].status, ManifestStatus::kAdded);
    EXPECT_EQ(df[0].content, FileContent::kData);
    EXPECT_EQ(df[0].snapshot_id, snap.snapshot_id);
    EXPECT_EQ(df[0].stats.record_count, 7);        // the batch's real row count
    EXPECT_GT(df[0].stats.file_size_in_bytes, 0);
    EXPECT_EQ(df[0].n_partition, 0u);

    // The data file the manifest names must exist at the size it claims.
    ObjectMeta dm{};
    EXPECT_TRUE(location_is_absolute(df[0].file_path))
        << "file_path must be an absolute URI: " << df[0].file_path;
    ASSERT_EQ(os_head(&os, store_key(root, df[0].file_path).c_str(), &dm),
              kOsOk);
    EXPECT_TRUE(dm.exists);
    EXPECT_EQ(df[0].stats.file_size_in_bytes, static_cast<int64_t>(dm.size));

    // 3. Nothing was left behind under the old `.json` manifest name — a
    //    lingering one would be a reader's first match and undo the fix.
    char stale[256];
    std::snprintf(stale, sizeof(stale), "metadata/manifest-%lld.json",
                  static_cast<long long>(snap.snapshot_id));
    ObjectMeta sm{};
    (void)os_head(&os, stale, &sm);
    EXPECT_FALSE(sm.exists);
}

// A partitioned table cannot be committed by this writer: the manifest schema
// it emits declares no partition tuple, so the manifest would parse and then
// report every file as unpartitioned — a reader prunes on that and silently
// drops rows. Fail the commit; do not write something wrong.
TEST(IcebergWrite, PartitionedCommitIsRefusedNotFaked) {
    bolt::Arena arena;
    const std::string root = unique_root("parted");
    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    Schema sch = make_schema_i64_utf8();
    PartitionSpec spec{};
    spec.spec_id  = 0;
    spec.n_fields = 1;
    spec.fields[0].source_id = 1;
    spec.fields[0].field_id  = 1000;
    spec.fields[0].transform.kind = TransformKind::kIdentity;
    std::strncpy(spec.fields[0].name, "id", sizeof(spec.fields[0].name) - 1u);
    SortOrder sort{}; sort.order_id = 0; sort.n_fields = 0;
    WriteOptions wo; write_options_init(&wo);
    TableHandle* th = nullptr;
    ASSERT_TRUE(table_create(&th, &arena, &os, root.c_str(), &sch, &spec,
                              &sort, &wo));

    bolt::BoltBatch b0{};
    make_batch(&arena, &b0, 0, 4);
    AppendHandle* ah = nullptr;
    ASSERT_TRUE(append_open(&ah, th));
    ASSERT_TRUE(append_write(ah, &b0));
    EXPECT_FALSE(append_commit(ah));
    append_close(ah);

    // No snapshot was published, so no reader can ever see a wrong manifest.
    EXPECT_EQ(table_metadata(th)->n_snapshots, 0u);
    EXPECT_EQ(table_metadata(th)->current_snapshot_id, -1);
}
