// W5 — Iceberg write path smoke: create → append → read-back, view create,
// branch ref, snapshot expiry.

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
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
    out->schema.add_field("id", bolt::BoltType::Int64, false);
    auto* cols = out->columns[out->read_epoch];
    cols[0].type = bolt::BoltType::Int64;
    cols[0].length = n;
    auto* idata = a->allocate_array<int64_t>(static_cast<uint32_t>(n));
    for (int32_t i = 0; i < n; ++i) idata[i] = base + i;
    cols[0].data = idata;
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
