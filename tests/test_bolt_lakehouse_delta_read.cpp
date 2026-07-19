// W2 — Delta read path: snapshot replay over hand-written _delta_log + tiny
// Parquet data file; Remove-action filtering test.

#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/delta/log.h"
#include "bolt/lakehouse/delta/snapshot.h"
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
#include "bolt/ingest/bolt_parquet_write.h"

namespace {

using namespace bolt::lakehouse;

std::string unique_root(const char* tag) {
    auto p = std::filesystem::temp_directory_path() /
             ("bolt_delta_" + std::string(tag) + "_" +
              std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
              "_" + std::to_string(reinterpret_cast<uintptr_t>(&tag)));
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    std::filesystem::create_directories(p, ec);
    return p.generic_string();
}

void write_tiny_parquet(const std::string& path, std::int64_t rows) {
    bolt::Arena arena;
    bolt::BoltBatch* batch = arena.allocate_array<bolt::BoltBatch>(1);
    ASSERT_NE(batch, nullptr);
    bolt::BoltBatch::init_empty(batch);
    batch->num_cols = 1;
    batch->num_rows = rows;
    bolt::BoltBatch::alloc_columns(batch, &arena, 1);  // G2FEAT-47: size columns[2]
    batch->schema.add_field("id", bolt::BoltType::Int64, false);
    bolt::BoltColumn& id = batch->columns[batch->read_epoch][0];
    id = bolt::BoltColumn::make_flat_alloc(rows, bolt::BoltType::Int64, &arena);
    ASSERT_NE(id.data, nullptr);
    auto* ip = static_cast<std::int64_t*>(id.data);
    for (std::int64_t i = 0; i < rows; ++i) ip[i] = i;

    bolt::ingest::parquet::ParquetWriteOpts opts{};
    opts.n_columns = 1;
    opts.row_group_target_bytes = 1u << 16;
    opts.compression = 0;
    opts.emit_statistics = true;
    std::strncpy(opts.columns[0].name, "id", sizeof(opts.columns[0].name));
    opts.columns[0].type = bolt::BoltType::Int64;
    opts.columns[0].nullable = false;

    auto* w = bolt::ingest::parquet::parquet_write_open(path.c_str(), &opts);
    ASSERT_NE(w, nullptr);
    ASSERT_TRUE(bolt::ingest::parquet::parquet_write_row_group(w, batch));
    ASSERT_TRUE(bolt::ingest::parquet::parquet_write_close(w));
}

void write_commit(const std::string& dir, int version,
                  const std::string& body) {
    char fname[64];
    std::snprintf(fname, sizeof(fname), "%020d.json", version);
    std::filesystem::create_directories(dir);
    const auto p = std::filesystem::path(dir) / fname;
    std::FILE* f = std::fopen(p.generic_string().c_str(), "wb");
    ASSERT_NE(f, nullptr);
    std::fwrite(body.data(), 1, body.size(), f);
    std::fclose(f);
}

struct CountCtx { uint32_t n_protocol = 0; uint32_t n_meta = 0;
                  uint32_t n_add = 0;      uint32_t n_remove = 0; };

bool count_actions(void* raw, const bolt::lakehouse::delta::DeltaAction* a) noexcept {
    auto* c = static_cast<CountCtx*>(raw);
    using K = bolt::lakehouse::delta::ActionKind;
    if (a->kind == K::kProtocol) ++c->n_protocol;
    else if (a->kind == K::kMetadata) ++c->n_meta;
    else if (a->kind == K::kAdd) ++c->n_add;
    else if (a->kind == K::kRemove) ++c->n_remove;
    return true;
}

TEST(BoltLakehouseDeltaRead, SnapshotReplayAndScanRows) {
    const std::string root = unique_root("scan");
    const std::string ns = "sales";
    const std::string name = "orders";
    const std::string tbl = root + "/" + ns + "/" + name;
    const std::string logd = tbl + "/_delta_log";

    FilesystemCatalog fc; Catalog cat;
    ASSERT_TRUE(filesystem_catalog_init(&fc, root.c_str(), &cat));
    ASSERT_EQ(cat_create_table(&cat, ns.c_str(), name.c_str(),
                                TableLayout::kDelta), kCatOk);

    const std::string datad = tbl + "/data";
    std::filesystem::create_directories(datad);
    const std::string pq = datad + "/part-0.parquet";
    write_tiny_parquet(pq, /*rows=*/8);

    const std::string commit0 =
        std::string("{\"protocol\":{\"minReaderVersion\":1,\"minWriterVersion\":2}}\n") +
        "{\"metaData\":{\"id\":\"tbl-1\",\"name\":\"orders\",\"schemaString\":\""
            "{\\\"type\\\":\\\"struct\\\",\\\"fields\\\":["
            "{\\\"name\\\":\\\"id\\\",\\\"type\\\":\\\"long\\\",\\\"nullable\\\":false,\\\"metadata\\\":{}}"
            "]}\",\"partitionColumns\":[],\"configuration\":{},\"createdTime\":1700000000000}}\n" +
        "{\"add\":{\"path\":\"data/part-0.parquet\",\"size\":1234,\"modificationTime\":1700000001000,"
            "\"dataChange\":true,\"partitionValues\":{},"
            "\"stats\":\"{\\\"numRecords\\\":8,\\\"minValues\\\":{\\\"id\\\":0},\\\"maxValues\\\":{\\\"id\\\":7}}\"}}\n" +
        "{\"commitInfo\":{\"timestamp\":1700000002000,\"operation\":\"CREATE TABLE\"}}\n";
    write_commit(logd, 0, commit0);

    bolt::Arena arena;
    TableHandle* th = nullptr;
    ASSERT_TRUE(delta_table_open(&th, &arena, &cat, ns.c_str(), name.c_str()));
    ScanHandle* sh = nullptr;
    ASSERT_TRUE(delta_scan_open(&sh, th, /*opts=*/nullptr));

    bolt::BoltBatch out{};
    bool eof = false;
    ASSERT_TRUE(delta_scan_next_batch(sh, &out, &eof));
    EXPECT_FALSE(eof);
    EXPECT_EQ(out.num_rows, 8);
    EXPECT_EQ(out.num_cols, 1u);
    const auto* ip = static_cast<const std::int64_t*>(
        out.columns[out.read_epoch][0].data);
    ASSERT_NE(ip, nullptr);
    for (std::int64_t i = 0; i < 8; ++i) EXPECT_EQ(ip[i], i);

    bool eof2 = false;
    ASSERT_TRUE(delta_scan_next_batch(sh, &out, &eof2));
    EXPECT_TRUE(eof2);

    delta_scan_close(sh);
    delta_table_close(th);
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST(BoltLakehouseDeltaRead, RemoveActionFiltersFile) {
    const std::string root = unique_root("rem");
    const std::string ns = "sales";
    const std::string name = "orders";
    const std::string tbl = root + "/" + ns + "/" + name;
    const std::string logd = tbl + "/_delta_log";

    FilesystemCatalog fc; Catalog cat;
    ASSERT_TRUE(filesystem_catalog_init(&fc, root.c_str(), &cat));
    ASSERT_EQ(cat_create_table(&cat, ns.c_str(), name.c_str(),
                                TableLayout::kDelta), kCatOk);

    const std::string commit0 =
        std::string("{\"protocol\":{\"minReaderVersion\":1,\"minWriterVersion\":2}}\n") +
        "{\"metaData\":{\"id\":\"t\",\"name\":\"orders\",\"schemaString\":\""
            "{\\\"type\\\":\\\"struct\\\",\\\"fields\\\":[]}\","
            "\"partitionColumns\":[],\"configuration\":{},\"createdTime\":1}}\n" +
        "{\"add\":{\"path\":\"data/A.parquet\",\"size\":1,\"modificationTime\":1,\"dataChange\":true,\"partitionValues\":{}}}\n" +
        "{\"add\":{\"path\":\"data/B.parquet\",\"size\":1,\"modificationTime\":1,\"dataChange\":true,\"partitionValues\":{}}}\n";
    write_commit(logd, 0, commit0);
    const std::string commit1 =
        "{\"remove\":{\"path\":\"data/A.parquet\",\"deletionTimestamp\":2,\"dataChange\":true}}\n";
    write_commit(logd, 1, commit1);

    FilesystemObjectStore fs; ObjectStore os;
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));

    bolt::Arena arena;
    char rel[256];
    std::snprintf(rel, sizeof(rel), "%s/%s", ns.c_str(), name.c_str());

    CountCtx cc;
    ASSERT_TRUE(bolt::lakehouse::delta::delta_log_walk_all(
        &os, rel, -1, &arena, &cc, count_actions));
    EXPECT_EQ(cc.n_protocol, 1u);
    EXPECT_EQ(cc.n_meta, 1u);
    EXPECT_EQ(cc.n_add, 2u);
    EXPECT_EQ(cc.n_remove, 1u);

    bolt::lakehouse::delta::Snapshot snap{};
    ASSERT_TRUE(bolt::lakehouse::delta::delta_snapshot_build(
        &os, rel, -1, &arena, &snap));
    ASSERT_EQ(snap.n_files, 1u);
    EXPECT_STREQ(snap.files[0].path, "data/B.parquet");
    EXPECT_EQ(snap.version, 1);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

}  // namespace
