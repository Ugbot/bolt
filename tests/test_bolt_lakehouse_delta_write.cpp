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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

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
    bolt::BoltBatch::alloc_columns(batch, arena, 1);  // G2FEAT-47: size columns[2]
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

// G2ICE-135 — real nullCount + min/max, read back through
// delta_snapshot_build's OWN stats parser (parse_stats_object in
// delta_snapshot.cpp) -- a genuinely separate code path from the writer, so
// this is a real round trip, not a self-consistency check. Before this fix,
// `build_stats_json` never emitted `nullCount` at all (the reader already
// parsed it, defaulting to -1 "unknown" on absence) and only ever bounded
// Int64 columns, silently skipping Float64 -- so a Delta reader had nothing
// trustworthy for null-safety gates or for Float64 file pruning, on any
// commit. `id` (Int64, nulls at rows 1 and 3) and `score` (Float64, nulls at
// row 2 only -- deliberately a DIFFERENT set, so a bug that shares one
// column's null_count with another cannot pass by accident) exercise both.
TEST(BoltLakehouseDeltaWrite, RealNullCountAndFloat64BoundsRoundTrip) {
    const std::string root = unique_root("stats");
    FilesystemCatalog fc; Catalog cat;
    ASSERT_TRUE(filesystem_catalog_init(&fc, root.c_str(), &cat));

    bolt::Arena arena;
    bolt::BoltSchema schema;
    schema.add_field("id",    bolt::BoltType::Int64,   true);
    schema.add_field("score", bolt::BoltType::Float64, true);

    dl::WriteOptions wopts;
    dl::write_options_init(&wopts);
    wopts.compression = Compression::kNone;

    TableHandle* th = nullptr;
    ASSERT_TRUE(dl::delta_table_create(&th, &arena, &cat, "sales", "nulls",
                                       &schema, &wopts));

    constexpr int64_t kN = 6;
    bolt::BoltBatch b{};
    b.arena    = &arena;
    b.num_rows = kN;
    b.num_cols = 2;
    bolt::BoltBatch::alloc_columns(&b, &arena, 2);
    b.schema.add_field("id",    bolt::BoltType::Int64,   true);
    b.schema.add_field("score", bolt::BoltType::Float64, true);
    auto* cols = b.columns[b.read_epoch];

    auto* idata = arena.allocate_array<int64_t>(kN);
    auto* ivalid = arena.allocate_array<uint8_t>(1);
    ivalid[0] = 0xFFu;
    for (int64_t i = 0; i < kN; ++i) idata[i] = 100 + i;
    ivalid[0] = static_cast<uint8_t>(ivalid[0] & ~(1u << 1));   // row 1 null
    ivalid[0] = static_cast<uint8_t>(ivalid[0] & ~(1u << 3));   // row 3 null
    cols[0].type = bolt::BoltType::Int64; cols[0].length = kN;
    cols[0].data = idata; cols[0].validity = ivalid;

    auto* sdata = arena.allocate_array<double>(kN);
    auto* svalid = arena.allocate_array<uint8_t>(1);
    svalid[0] = 0xFFu;
    for (int64_t i = 0; i < kN; ++i) sdata[i] = static_cast<double>(i) * 1.5;
    svalid[0] = static_cast<uint8_t>(svalid[0] & ~(1u << 2));   // row 2 null
    cols[1].type = bolt::BoltType::Float64; cols[1].length = kN;
    cols[1].data = sdata; cols[1].validity = svalid;

    dl::AppendHandle* ah = nullptr;
    ASSERT_TRUE(dl::delta_append_open(&ah, th));
    ASSERT_TRUE(dl::delta_append_write(ah, &b));
    ASSERT_TRUE(dl::delta_append_commit(ah));
    dl::delta_append_close(ah);

    FilesystemObjectStore fs; ObjectStore os;
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    bolt::Arena s_ar;
    dl::Snapshot snap{};
    ASSERT_TRUE(dl::delta_snapshot_build(&os, "sales/nulls", -1, &s_ar, &snap));
    ASSERT_EQ(snap.n_files, 1u);
    const dl::FileStats& st = snap.files[0].stats;
    ASSERT_TRUE(st.has_num_records);
    EXPECT_EQ(st.num_records, kN);
    ASSERT_GE(st.n_cols, 2u);

    int id_slot = -1, score_slot = -1;
    for (uint32_t i = 0; i < st.n_cols; ++i) {
        if (std::strcmp(st.col_names[i], "id") == 0) id_slot = static_cast<int>(i);
        if (std::strcmp(st.col_names[i], "score") == 0) score_slot = static_cast<int>(i);
    }
    ASSERT_GE(id_slot, 0);
    ASSERT_GE(score_slot, 0);

    // Real, distinct null counts -- not defaulted, not swapped.
    EXPECT_EQ(st.null_counts[id_slot], 2);
    EXPECT_EQ(st.null_counts[score_slot], 1);

    // id bounds: min/max over the 4 non-null rows (100,102,104,105) -> [100,105].
    EXPECT_STREQ(st.min_str[id_slot], "100");
    EXPECT_STREQ(st.max_str[id_slot], "105");

    // score bounds (Float64 -- the type this writer used to silently skip):
    // rows 0..5 give 0, 1.5, 3.0, 4.5, 6.0, 7.5; row 2 (3.0) is null and
    // excluded -> non-null set {0, 1.5, 4.5, 6.0, 7.5}, min 0, max 7.5.
    EXPECT_DOUBLE_EQ(std::strtod(st.min_str[score_slot], nullptr), 0.0);
    EXPECT_DOUBLE_EQ(std::strtod(st.max_str[score_slot], nullptr), 7.5);

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

// Reads every surviving row's "id" column back (via the real Delta scan
// path, not by peeking at internal state) and returns the sorted set of
// values still present.
std::vector<std::int64_t> scan_all_ids(bolt::Arena* arena, Catalog* cat,
                                       const char* ns, const char* name) {
    TableHandle* th = nullptr;
    if (!delta_table_open(&th, arena, cat, ns, name)) return {};
    ScanHandle* sh = nullptr;
    if (!delta_scan_open(&sh, th, /*opts=*/nullptr)) return {};
    std::vector<std::int64_t> out;
    for (;;) {
        bolt::BoltBatch batch{};
        bool eof = false;
        if (!delta_scan_next_batch(sh, &batch, &eof)) break;
        if (eof) break;
        const auto* ip = static_cast<const std::int64_t*>(
            batch.columns[batch.read_epoch][0].data);
        for (std::int64_t i = 0; i < batch.num_rows; ++i) out.push_back(ip[i]);
    }
    delta_scan_close(sh);
    delta_table_close(th);
    std::sort(out.begin(), out.end());
    return out;
}

// G2ICE-81: a DELETE predicate that matches exactly one row out of four in
// a SINGLE file used to remove the WHOLE file — silently losing the three
// non-matching rows. The fix rewrites the file keeping only the survivors.
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
    // Rewritten, not wholesale-removed: one live file remains, holding the
    // three survivors.
    EXPECT_EQ(snap.n_files, 1u);

    bolt::Arena r_ar;
    const auto ids = scan_all_ids(&r_ar, &cat, "ns", "t");
    const std::vector<std::int64_t> expected{0, 2, 3};
    EXPECT_EQ(ids, expected);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

// The other half of the same fix: when EVERY row in the file genuinely
// matches, the file legitimately disappears whole (no orphaned empty file).
TEST(BoltLakehouseDeltaWrite, DeleteAllRowsMatchRemovesWholeFile) {
    const std::string root = unique_root("del_all");
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
    pred.op = PredicateOp::kGe;
    pred.value.type = bolt::BoltType::Int64;
    pred.value.i64 = 0;  // every row (0..3) is >= 0
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

// The ticket's own acceptance shape: one file, 8 rows, delete 1 row, and
// prove — BY VALUE, via the real scan path — that the other 7 survive.
TEST(BoltLakehouseDeltaWrite, DeleteKeepsNonMatchingRowsByValue) {
    const std::string root = unique_root("del8");
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
    fill_int_batch(&ba, &b, 8, 0);  // ids 0..7, all in ONE parquet file
    ASSERT_TRUE(dl::delta_append_write(ah, &b));
    ASSERT_TRUE(dl::delta_append_commit(ah));

    bolt::Arena pre_ar;
    const auto before = scan_all_ids(&pre_ar, &cat, "ns", "t");
    ASSERT_EQ(before.size(), 8u);

    Predicate pred{};
    std::strncpy(pred.column, "id", sizeof(pred.column) - 1u);
    pred.op = PredicateOp::kEq;
    pred.value.type = bolt::BoltType::Int64;
    pred.value.i64 = 3;
    ASSERT_TRUE(dl::delta_table_delete(th, &pred));

    FilesystemObjectStore fs; ObjectStore os;
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    bolt::Arena s_ar;
    dl::Snapshot snap{};
    ASSERT_TRUE(dl::delta_snapshot_build(&os, "ns/t", -1, &s_ar, &snap));
    EXPECT_EQ(snap.n_files, 1u);

    bolt::Arena r_ar;
    const auto ids = scan_all_ids(&r_ar, &cat, "ns", "t");
    const std::vector<std::int64_t> expected{0, 1, 2, 4, 5, 6, 7};
    EXPECT_EQ(ids, expected) << "row 3 should be gone; rows 0,1,2,4,5,6,7 "
                                "must survive by VALUE";

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

// A predicate whose value falls INSIDE a file's [min,max] stats range but
// matches no actual row (the stats-based "may contain" prefilter is a sound
// over-approximation, not an exact test). The file must be left completely
// untouched -- no rewrite, same path, all rows intact -- not just "correct
// values after a pointless rewrite".
TEST(BoltLakehouseDeltaWrite, DeleteStatsFalsePositiveLeavesFileUntouched) {
    const std::string root = unique_root("del_fp");
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
    bolt::BoltBatch::init_empty(&b);
    b.num_cols = 1; b.num_rows = 4;
    bolt::BoltBatch::alloc_columns(&b, &ba, 1);
    b.schema.add_field("id", bolt::BoltType::Int64, false);
    bolt::BoltColumn& idc = b.columns[b.read_epoch][0];
    idc = bolt::BoltColumn::make_flat_alloc(4, bolt::BoltType::Int64, &ba);
    auto* ip = static_cast<std::int64_t*>(idc.data);
    ip[0] = 0; ip[1] = 2; ip[2] = 4; ip[3] = 6;  // 3 is a gap, not a value
    ASSERT_TRUE(dl::delta_append_write(ah, &b));
    ASSERT_TRUE(dl::delta_append_commit(ah));

    FilesystemObjectStore fs; ObjectStore os;
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    bolt::Arena s0_ar;
    dl::Snapshot snap0{};
    ASSERT_TRUE(dl::delta_snapshot_build(&os, "ns/t", -1, &s0_ar, &snap0));
    ASSERT_EQ(snap0.n_files, 1u);
    const std::string original_path = snap0.files[0].path;

    Predicate pred{};
    std::strncpy(pred.column, "id", sizeof(pred.column) - 1u);
    pred.op = PredicateOp::kEq;
    pred.value.type = bolt::BoltType::Int64;
    pred.value.i64 = 3;  // inside [0,6] but never actually stored
    ASSERT_TRUE(dl::delta_table_delete(th, &pred));

    bolt::Arena s1_ar;
    dl::Snapshot snap1{};
    ASSERT_TRUE(dl::delta_snapshot_build(&os, "ns/t", -1, &s1_ar, &snap1));
    ASSERT_EQ(snap1.n_files, 1u);
    EXPECT_STREQ(snap1.files[0].path, original_path.c_str())
        << "a stats false-positive must not trigger a rewrite";

    bolt::Arena r_ar;
    const auto ids = scan_all_ids(&r_ar, &cat, "ns", "t");
    const std::vector<std::int64_t> expected{0, 2, 4, 6};
    EXPECT_EQ(ids, expected);

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
