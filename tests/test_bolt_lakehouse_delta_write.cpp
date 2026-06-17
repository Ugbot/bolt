// W3 — Delta write path: table create + append + concurrent-conflict +
// UPDATE/DELETE + OPTIMIZE + VACUUM + RESTORE smoke tests.

#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/delta/log.h"
#include "bolt/lakehouse/delta/optimize.h"
#include "bolt/lakehouse/delta/snapshot.h"
#include "bolt/lakehouse/delta/writer.h"
#include "bolt/lakehouse/handle.h"
#include "bolt/lakehouse/object_store.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"

namespace {

using namespace bolt::lakehouse;
namespace dl = bolt::lakehouse::delta;

std::string unique_root(const char* tag) {
    auto p = std::filesystem::temp_directory_path() /
             ("bolt_delta_w_" + std::string(tag) + "_" +
              std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
              "_" + std::to_string(reinterpret_cast<uintptr_t>(&tag)));
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    std::filesystem::create_directories(p, ec);
    return p.generic_string();
}

void fill_int_batch(bolt::Arena* arena, bolt::BoltBatch* batch,
                    std::int64_t rows, std::int64_t base) {
    bolt::BoltBatch::init_empty(batch);
    batch->num_cols = 1;
    batch->num_rows = rows;
    batch->schema.add_field("id", bolt::BoltType::Int64, false);
    bolt::BoltColumn& id = batch->columns[batch->read_epoch][0];
    id = bolt::BoltColumn::make_flat_alloc(rows, bolt::BoltType::Int64, arena);
    ASSERT_NE(id.data, nullptr);
    auto* ip = static_cast<std::int64_t*>(id.data);
    for (std::int64_t i = 0; i < rows; ++i) ip[i] = base + i;
}

TEST(BoltLakehouseDeltaWrite, AppendThreeBatchesRoundTrip) {
    const std::string root = unique_root("append");
    FilesystemCatalog fc; Catalog cat;
    ASSERT_TRUE(filesystem_catalog_init(&fc, root.c_str(), &cat));

    bolt::Arena arena;
    bolt::BoltSchema schema;
    schema.add_field("id", bolt::BoltType::Int64, false);

    dl::WriteOptions wopts;
    dl::write_options_init(&wopts);
    wopts.compression = Compression::kNone;

    TableHandle* th = nullptr;
    ASSERT_TRUE(dl::delta_table_create(&th, &arena, &cat, "sales", "orders",
                                       &schema, &wopts));

    dl::AppendHandle* ah = nullptr;
    ASSERT_TRUE(dl::delta_append_open(&ah, th));
    for (int i = 0; i < 3; ++i) {
        bolt::Arena ba;
        bolt::BoltBatch b{};
        fill_int_batch(&ba, &b, 4, static_cast<std::int64_t>(i * 4));
        ASSERT_TRUE(dl::delta_append_write(ah, &b));
    }
    ASSERT_TRUE(dl::delta_append_commit(ah));
    dl::delta_append_close(ah);

    // Snapshot should have 3 live files (independent ObjectStore against
    // the same fs root + table relative path).
    FilesystemObjectStore fs; ObjectStore os;
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    bolt::Arena s_ar;
    dl::Snapshot snap{};
    ASSERT_TRUE(dl::delta_snapshot_build(&os, "sales/orders", -1, &s_ar,
                                          &snap));
    EXPECT_EQ(snap.n_files, 3u);
    EXPECT_GE(snap.version, 1);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST(BoltLakehouseDeltaWrite, ConcurrentWriterSecondCommitFails) {
    const std::string root = unique_root("conflict");
    FilesystemCatalog fc; Catalog cat;
    ASSERT_TRUE(filesystem_catalog_init(&fc, root.c_str(), &cat));

    bolt::Arena arena;
    bolt::BoltSchema schema;
    schema.add_field("id", bolt::BoltType::Int64, false);

    TableHandle* th = nullptr;
    ASSERT_TRUE(dl::delta_table_create(&th, &arena, &cat, "ns", "t",
                                       &schema, nullptr));

    dl::AppendHandle* a1 = nullptr;
    dl::AppendHandle* a2 = nullptr;
    ASSERT_TRUE(dl::delta_append_open(&a1, th));
    ASSERT_TRUE(dl::delta_append_open(&a2, th));

    bolt::Arena ba1, ba2;
    bolt::BoltBatch b1{}, b2{};
    fill_int_batch(&ba1, &b1, 2, 0);
    fill_int_batch(&ba2, &b2, 2, 10);
    ASSERT_TRUE(dl::delta_append_write(a1, &b1));
    ASSERT_TRUE(dl::delta_append_write(a2, &b2));

    ASSERT_TRUE(dl::delta_append_commit(a1));
    EXPECT_FALSE(dl::delta_append_commit(a2));

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST(BoltLakehouseDeltaWrite, DeleteEmitsRemoveActions) {
    const std::string root = unique_root("del");
    FilesystemCatalog fc; Catalog cat;
    ASSERT_TRUE(filesystem_catalog_init(&fc, root.c_str(), &cat));

    bolt::Arena arena;
    bolt::BoltSchema schema;
    schema.add_field("id", bolt::BoltType::Int64, false);

    TableHandle* th = nullptr;
    ASSERT_TRUE(dl::delta_table_create(&th, &arena, &cat, "ns", "t",
                                       &schema, nullptr));
    dl::AppendHandle* ah = nullptr;
    ASSERT_TRUE(dl::delta_append_open(&ah, th));
    bolt::Arena ba;
    bolt::BoltBatch b{};
    fill_int_batch(&ba, &b, 4, 0);
    ASSERT_TRUE(dl::delta_append_write(ah, &b));
    ASSERT_TRUE(dl::delta_append_commit(ah));

    Predicate pred{};
    std::strncpy(pred.column, "id", sizeof(pred.column) - 1u);
    pred.op = PredicateOp::kEq;
    pred.value.type = bolt::BoltType::Int64;
    pred.value.i64 = 1;
    ASSERT_TRUE(dl::delta_table_delete(th, &pred));

    FilesystemObjectStore fs; ObjectStore os;
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    bolt::Arena s_ar;
    dl::Snapshot snap{};
    ASSERT_TRUE(dl::delta_snapshot_build(&os, "ns/t", -1, &s_ar, &snap));
    EXPECT_EQ(snap.n_files, 0u);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST(BoltLakehouseDeltaWrite, UpdateEmitsRemoveActions) {
    const std::string root = unique_root("upd");
    FilesystemCatalog fc; Catalog cat;
    ASSERT_TRUE(filesystem_catalog_init(&fc, root.c_str(), &cat));

    bolt::Arena arena;
    bolt::BoltSchema schema;
    schema.add_field("id", bolt::BoltType::Int64, false);

    TableHandle* th = nullptr;
    ASSERT_TRUE(dl::delta_table_create(&th, &arena, &cat, "ns", "t",
                                       &schema, nullptr));
    dl::AppendHandle* ah = nullptr;
    ASSERT_TRUE(dl::delta_append_open(&ah, th));
    bolt::Arena ba;
    bolt::BoltBatch b{};
    fill_int_batch(&ba, &b, 4, 0);
    ASSERT_TRUE(dl::delta_append_write(ah, &b));
    ASSERT_TRUE(dl::delta_append_commit(ah));

    Predicate pred{};
    std::strncpy(pred.column, "id", sizeof(pred.column) - 1u);
    pred.op = PredicateOp::kEq;
    pred.value.type = bolt::BoltType::Int64;
    pred.value.i64 = 2;
    dl::Assignment a{};
    std::strncpy(a.column, "id", sizeof(a.column) - 1u);
    a.type = bolt::BoltType::Int64;
    a.i64 = 99;
    ASSERT_TRUE(dl::delta_table_update(th, &pred, &a, 1));

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST(BoltLakehouseDeltaWrite, OptimizeBinPackCommitsAction) {
    const std::string root = unique_root("opt");
    FilesystemCatalog fc; Catalog cat;
    ASSERT_TRUE(filesystem_catalog_init(&fc, root.c_str(), &cat));

    bolt::Arena arena;
    bolt::BoltSchema schema;
    schema.add_field("id", bolt::BoltType::Int64, false);

    TableHandle* th = nullptr;
    ASSERT_TRUE(dl::delta_table_create(&th, &arena, &cat, "ns", "t",
                                       &schema, nullptr));
    dl::AppendHandle* ah = nullptr;
    ASSERT_TRUE(dl::delta_append_open(&ah, th));
    for (int i = 0; i < 10; ++i) {
        bolt::Arena ba;
        bolt::BoltBatch b{};
        fill_int_batch(&ba, &b, 1, i);
        ASSERT_TRUE(dl::delta_append_write(ah, &b));
    }
    ASSERT_TRUE(dl::delta_append_commit(ah));

    dl::OptimizeOptions opts;
    dl::optimize_options_init(&opts);
    EXPECT_TRUE(dl::delta_table_optimize(th, &opts));

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST(BoltLakehouseDeltaWrite, VacuumDryRunListsFiles) {
    const std::string root = unique_root("vac");
    FilesystemCatalog fc; Catalog cat;
    ASSERT_TRUE(filesystem_catalog_init(&fc, root.c_str(), &cat));

    bolt::Arena arena;
    bolt::BoltSchema schema;
    schema.add_field("id", bolt::BoltType::Int64, false);

    TableHandle* th = nullptr;
    ASSERT_TRUE(dl::delta_table_create(&th, &arena, &cat, "ns", "t",
                                       &schema, nullptr));
    dl::AppendHandle* ah = nullptr;
    ASSERT_TRUE(dl::delta_append_open(&ah, th));
    bolt::Arena ba;
    bolt::BoltBatch b{};
    fill_int_batch(&ba, &b, 2, 0);
    ASSERT_TRUE(dl::delta_append_write(ah, &b));
    ASSERT_TRUE(dl::delta_append_commit(ah));

    char out[dl::kVacuumMaxDeleted][dl::kVacuumPathBytes];
    uint32_t n = 0;
    // retention_hours=0 — anything not referenced is fair game; all referenced
    // files are skipped → list empty.
    EXPECT_TRUE(dl::delta_table_vacuum(th, 0, /*dry_run=*/true, out,
                                        dl::kVacuumMaxDeleted, &n));

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST(BoltLakehouseDeltaWrite, RestoreToEarlierVersion) {
    const std::string root = unique_root("rest");
    FilesystemCatalog fc; Catalog cat;
    ASSERT_TRUE(filesystem_catalog_init(&fc, root.c_str(), &cat));

    bolt::Arena arena;
    bolt::BoltSchema schema;
    schema.add_field("id", bolt::BoltType::Int64, false);

    TableHandle* th = nullptr;
    ASSERT_TRUE(dl::delta_table_create(&th, &arena, &cat, "ns", "t",
                                       &schema, nullptr));
    dl::AppendHandle* a1 = nullptr;
    ASSERT_TRUE(dl::delta_append_open(&a1, th));
    bolt::Arena ba;
    bolt::BoltBatch b{};
    fill_int_batch(&ba, &b, 2, 0);
    ASSERT_TRUE(dl::delta_append_write(a1, &b));
    ASSERT_TRUE(dl::delta_append_commit(a1));

    FilesystemObjectStore fs; ObjectStore os;
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    bolt::Arena s_ar;
    dl::Snapshot snap1{};
    ASSERT_TRUE(dl::delta_snapshot_build(&os, "ns/t", -1, &s_ar, &snap1));
    const int64_t target = snap1.version;

    // Add a second batch.
    dl::AppendHandle* a2 = nullptr;
    ASSERT_TRUE(dl::delta_append_open(&a2, th));
    bolt::Arena ba2;
    bolt::BoltBatch b2{};
    fill_int_batch(&ba2, &b2, 2, 100);
    ASSERT_TRUE(dl::delta_append_write(a2, &b2));
    ASSERT_TRUE(dl::delta_append_commit(a2));

    EXPECT_TRUE(dl::delta_table_restore(th, target));

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

}  // namespace
