// Filesystem Catalog + ObjectStore — create namespace/table, list, commit
// metadata, drop. Uses a unique temp directory.

#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/object_store.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>

namespace {

using namespace bolt::lakehouse;

std::string temp_root(const char* tag) {
    auto p = std::filesystem::temp_directory_path() /
             ("bolt_lake_" + std::string(tag) + "_" +
              std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
              "_" + std::to_string(reinterpret_cast<uintptr_t>(&tag)));
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    return p.generic_string();
}

TEST(BoltLakehouseCatalog, CreateListDrop) {
    const std::string root = temp_root("cat");
    FilesystemCatalog fc;
    Catalog cat;
    ASSERT_TRUE(filesystem_catalog_init(&fc, root.c_str(), &cat));

    // Empty catalog: no namespaces.
    CatalogName ns_out[16];
    uint32_t n = 99;
    ASSERT_EQ(cat_list_namespaces(&cat, ns_out, 16, &n), kCatOk);
    EXPECT_EQ(n, 0u);

    // Create two tables in namespace "sales" (creates the namespace).
    ASSERT_EQ(cat_create_table(&cat, "sales", "orders", TableLayout::kDelta),
              kCatOk);
    ASSERT_EQ(cat_create_table(&cat, "sales", "items", TableLayout::kIceberg),
              kCatOk);
    // Duplicate create rejected.
    EXPECT_EQ(cat_create_table(&cat, "sales", "orders", TableLayout::kDelta),
              kCatExists);

    // Namespace now appears.
    ASSERT_EQ(cat_list_namespaces(&cat, ns_out, 16, &n), kCatOk);
    ASSERT_EQ(n, 1u);
    EXPECT_STREQ(ns_out[0].name, "sales");

    // Two tables listed.
    CatalogName tbl_out[16];
    uint32_t tn = 0;
    ASSERT_EQ(cat_list_tables(&cat, "sales", tbl_out, 16, &tn), kCatOk);
    EXPECT_EQ(tn, 2u);

    // Resolve a table path and confirm the Delta log dir exists.
    char path[kCatMaxPath];
    ASSERT_EQ(cat_table_path(&cat, "sales", "orders", path, sizeof(path)),
              kCatOk);
    EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(path) /
                                        "_delta_log"));

    // Commit a metadata file.
    const char* body = "{\"commitInfo\":{}}";
    ASSERT_EQ(cat_commit_metadata(&cat, "sales", "orders",
                                  "_delta_log/00000000000000000000.json",
                                  reinterpret_cast<const uint8_t*>(body),
                                  std::strlen(body)),
              kCatOk);
    EXPECT_TRUE(std::filesystem::exists(
        std::filesystem::path(path) / "_delta_log" /
        "00000000000000000000.json"));

    // Drop one table; the other remains.
    ASSERT_EQ(cat_drop_table(&cat, "sales", "orders"), kCatOk);
    EXPECT_EQ(cat_table_path(&cat, "sales", "orders", path, sizeof(path)),
              kCatNotFound);
    ASSERT_EQ(cat_list_tables(&cat, "sales", tbl_out, 16, &tn), kCatOk);
    EXPECT_EQ(tn, 1u);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST(BoltLakehouseObjectStore, FilesystemPutGetListDelete) {
    const std::string root = temp_root("os");
    std::error_code ec;
    std::filesystem::create_directories(root, ec);

    FilesystemObjectStore fs;
    ObjectStore os;
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));

    const char* payload = "hello object store";
    ASSERT_EQ(os_put(&os, "a/b/obj.txt",
                     reinterpret_cast<const uint8_t*>(payload),
                     std::strlen(payload)), kOsOk);

    ObjectMeta meta;
    ASSERT_EQ(os_head(&os, "a/b/obj.txt", &meta), kOsOk);
    EXPECT_TRUE(meta.exists);
    EXPECT_EQ(meta.size, std::strlen(payload));

    bolt::Arena arena;
    const uint8_t* data = nullptr;
    uint64_t len = 0;
    ASSERT_EQ(os_get(&os, "a/b/obj.txt", &arena, &data, &len), kOsOk);
    ASSERT_EQ(len, std::strlen(payload));
    EXPECT_EQ(0, std::memcmp(data, payload, len));

    ObjectEntry entries[16];
    uint32_t en = 0;
    ASSERT_EQ(os_list(&os, "a/", entries, 16, &en), kOsOk);
    ASSERT_EQ(en, 1u);
    EXPECT_STREQ(entries[0].key, "a/b/obj.txt");

    ASSERT_EQ(os_delete(&os, "a/b/obj.txt"), kOsOk);
    EXPECT_EQ(os_get(&os, "a/b/obj.txt", &arena, &data, &len), kOsNotFound);

    std::filesystem::remove_all(root, ec);
}

TEST(BoltLakehouseObjectStore, S3StubsNotImplemented) {
    S3ObjectStore s3;
    ObjectStore os;
    ASSERT_TRUE(s3_object_store_init(&s3, "my-bucket", "us-east-1",
                                     "AKID", "SECRET", &os));
    ObjectMeta meta;
    EXPECT_EQ(os_head(&os, "k", &meta), kOsNotImplemented);
    EXPECT_EQ(os_delete(&os, "k"), kOsNotImplemented);
}

}  // namespace
