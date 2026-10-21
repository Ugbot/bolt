// W4 — Iceberg read path: metadata.json + JSON-form manifest list/manifest
// parser smoke. Transforms unit tests. Empty-table scan returns 0 rows clean.
//
// A full Parquet round-trip lands in W5 alongside Avro-nested manifest read.

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/iceberg/snapshot.h"
#include "bolt/lakehouse/iceberg/transform.h"
#include "bolt/lakehouse/object_store.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

std::string unique_root(const char* tag) {
    auto p = std::filesystem::temp_directory_path() /
             ("bolt_iceberg_" + std::string(tag) + "_" +
              std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
              "_" + std::to_string(reinterpret_cast<uintptr_t>(&tag)));
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    std::filesystem::create_directories(p, ec);
    return p.generic_string();
}

void write_file(const std::string& path, const std::string& body) {
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());
    std::ofstream f(path, std::ios::binary);
    f.write(body.data(), static_cast<std::streamsize>(body.size()));
}

}  // namespace

TEST(IcebergTransform, Murmur3Spec) {
    const char* s = "iceberg";
    const uint32_t h = murmur3_32(s, 7u, 0u);
    EXPECT_NE(h, 0u);
    const int32_t b = bucket_str(s, 7u, 16);
    EXPECT_GE(b, 0);
    EXPECT_LT(b, 16);
}

TEST(IcebergTransform, Identity) {
    Transform t = transform_parse("identity", 8u);
    EXPECT_EQ(t.kind, TransformKind::kIdentity);
    t = transform_parse("bucket[16]", 10u);
    EXPECT_EQ(t.kind, TransformKind::kBucket);
    EXPECT_EQ(t.param, 16);
    t = transform_parse("truncate[10]", 12u);
    EXPECT_EQ(t.kind, TransformKind::kTruncate);
    EXPECT_EQ(t.param, 10);
    t = transform_parse("not-a-thing", 11u);
    EXPECT_EQ(t.kind, TransformKind::kUnknown);
}

TEST(IcebergTransform, TruncateI64) {
    EXPECT_EQ(truncate_i64(7, 5), 5);
    EXPECT_EQ(truncate_i64(0, 5), 0);
    EXPECT_EQ(truncate_i64(-3, 5), -5);
    EXPECT_EQ(truncate_i64(10, 5), 10);
}

TEST(IcebergMetadata, ParseV2) {
    bolt::Arena arena;
    const char* body = R"({
        "format-version": 2,
        "table-uuid": "abc-123",
        "location": "/tmp/iceberg/t1",
        "last-updated-ms": 1700000000000,
        "last-sequence-number": 5,
        "current-snapshot-id": 100,
        "current-schema-id": 0,
        "default-spec-id": 0,
        "schemas": [
            {"schema-id": 0, "fields": [
                {"id": 1, "name": "id", "type": "long", "required": true},
                {"id": 2, "name": "name", "type": "string", "required": false}
            ]}
        ],
        "partition-specs": [
            {"spec-id": 0, "fields": [
                {"source-id": 1, "field-id": 1000, "name": "id_bucket",
                 "transform": "bucket[8]"}
            ]}
        ],
        "snapshots": [
            {"snapshot-id": 100, "timestamp-ms": 1700000000000,
             "manifest-list": "metadata/snap-100.json",
             "summary": {"operation": "append"}}
        ]
    })";
    Metadata m{};
    ASSERT_TRUE(metadata_parse(reinterpret_cast<const uint8_t*>(body),
                                static_cast<uint32_t>(std::strlen(body)),
                                &arena, &m));
    EXPECT_EQ(m.format_version, 2);
    EXPECT_EQ(m.current_snapshot_id, 100);
    EXPECT_EQ(m.n_snapshots, 1u);
    ASSERT_EQ(m.n_schemas, 1u);
    EXPECT_EQ(m.schemas[0].n_fields, 2u);
    EXPECT_STREQ(m.schemas[0].fields[0].name, "id");
    EXPECT_STREQ(m.schemas[0].fields[1].name, "name");
    ASSERT_EQ(m.n_specs, 1u);
    EXPECT_EQ(m.specs[0].n_fields, 1u);
    EXPECT_EQ(m.specs[0].fields[0].transform.kind, TransformKind::kBucket);
    EXPECT_EQ(m.specs[0].fields[0].transform.param, 8);
    Snapshot s{};
    ASSERT_TRUE(snapshot_resolve(&m, -1, -1, &s));
    EXPECT_EQ(s.snapshot_id, 100);
}

TEST(IcebergSnapshot, ExactSelectionPreservesAllSignedIds) {
    Snapshot snapshots[4]{};
    const int64_t ids[]{INT64_MIN, -1, 0, INT64_MAX};
    for (uint32_t i = 0; i < 4; ++i) snapshots[i].snapshot_id = ids[i];
    Metadata metadata{};
    metadata.snapshots = snapshots;
    metadata.n_snapshots = metadata.cap_snapshots = 4;
    metadata.current_snapshot_id = INT64_MAX;
    metadata.current_snapshot_present = true;
    Snapshot selected{};
    for (const auto id : ids) {
        ASSERT_TRUE(snapshot_resolve_selected(&metadata, id, -1,
                                              SnapshotSelection::Exact, &selected));
        EXPECT_EQ(selected.snapshot_id, id);
    }
    ASSERT_TRUE(snapshot_resolve_selected(&metadata, -1, -1,
                                          SnapshotSelection::Legacy, &selected));
    EXPECT_EQ(selected.snapshot_id, INT64_MAX);
}

TEST(IcebergSnapshot, InvalidSelectionPreservesOutput) {
    Snapshot snapshot{};
    snapshot.snapshot_id = -1;
    Metadata metadata{};
    metadata.snapshots = &snapshot;
    metadata.n_snapshots = metadata.cap_snapshots = 1;
    Snapshot selected{};
    selected.snapshot_id = 987;
    EXPECT_FALSE(snapshot_resolve_selected(&metadata, -2, -1,
                                           SnapshotSelection::Exact, &selected));
    EXPECT_FALSE(snapshot_resolve_selected(&metadata, -1, 0,
                                           SnapshotSelection::Exact, &selected));
    EXPECT_FALSE(snapshot_resolve_selected(&metadata, 0, -1,
                                           SnapshotSelection::Empty, &selected));
    EXPECT_FALSE(snapshot_resolve_selected(&metadata, -1, -1,
                                           static_cast<SnapshotSelection>(255), &selected));
    EXPECT_EQ(selected.snapshot_id, 987);
    ReadOptions options{};
    read_options_init(&options);
    EXPECT_EQ(options.snapshot_selection, SnapshotSelection::Legacy);
}

TEST(IcebergManifest, ParseListAndManifestJson) {
    bolt::Arena arena;
    const char* mlist = R"([
        {"manifest_path": "metadata/m1.json",
         "manifest_length": 0,
         "partition_spec_id": 0,
         "content": 0,
         "added_snapshot_id": 100,
         "added_files_count": 1}
    ])";
    ManifestListEntry entries[4];
    uint32_t n = 0;
    ASSERT_TRUE(manifest_list_parse_json(
        reinterpret_cast<const uint8_t*>(mlist),
        static_cast<uint32_t>(std::strlen(mlist)),
        &arena, entries, 4u, &n));
    ASSERT_EQ(n, 1u);
    EXPECT_STREQ(entries[0].manifest_path, "metadata/m1.json");

    const char* manifest = R"([
        {"status": 1, "snapshot_id": 100,
         "data_file": {
             "content": 0,
             "file_path": "data/file-0.parquet",
             "file_format": "PARQUET",
             "partition": [{"field_id": 1000, "value": 3}],
             "record_count": 10,
             "file_size_in_bytes": 1024,
             "lower_bounds": [{"field_id": 1, "value": "0"}],
             "upper_bounds": [{"field_id": 1, "value": "9"}],
             "null_value_counts": [{"field_id": 1, "value": 0}]
         }}
    ])";
    DataFileRef files[4];
    uint32_t nf = 0;
    ASSERT_TRUE(manifest_parse_json(
        reinterpret_cast<const uint8_t*>(manifest),
        static_cast<uint32_t>(std::strlen(manifest)),
        &arena, 0, files, 4u, &nf));
    ASSERT_EQ(nf, 1u);
    EXPECT_EQ(files[0].status, ManifestStatus::kAdded);
    EXPECT_EQ(files[0].content, FileContent::kData);
    EXPECT_STREQ(files[0].file_path, "data/file-0.parquet");
    EXPECT_EQ(files[0].stats.record_count, 10);
    EXPECT_EQ(files[0].n_partition, 1u);
    EXPECT_TRUE(files[0].partition[0].is_int);
    EXPECT_EQ(files[0].partition[0].i64, 3);
    EXPECT_GE(files[0].stats.n_cols, 1u);
}

TEST(IcebergManifest, JsonFileContentMatchesSpecAndOwnedEnum) {
    for (const int content : {0,1,2,3,-1}) {
        bolt::Arena arena;
        const std::string json="[{\"status\":1,\"snapshot_id\":1,\"data_file\":{\"content\":"+
            std::to_string(content)+",\"file_path\":\"data.parquet\",\"record_count\":1}}]";
        DataFileRef file{}; uint32_t count=0;
        const bool ok=manifest_parse_json(reinterpret_cast<const uint8_t*>(json.data()),
            static_cast<uint32_t>(json.size()),&arena,0,&file,1,&count);
        if (content<0 || content>2) { EXPECT_FALSE(ok); continue; }
        ASSERT_TRUE(ok); ASSERT_EQ(count,1u);
        EXPECT_EQ(file.content,static_cast<FileContent>(content));
    }
}

TEST(IcebergManifest, JsonFileContentNeverTruncatesOrUsesLastDuplicate) {
    for (const char* value : {"0.5","null","\"1\"","9223372036854775808","1,\"content\":0"}) {
        bolt::Arena arena;
        const std::string json=std::string("[{\"status\":1,\"data_file\":{\"content\":")+value+
            ",\"file_path\":\"data.parquet\"}}]";
        DataFileRef file{}; uint32_t count=0;
        EXPECT_FALSE(manifest_parse_json(reinterpret_cast<const uint8_t*>(json.data()),
            static_cast<uint32_t>(json.size()),&arena,0,&file,1,&count));
    }
}

TEST(IcebergScan, EmptyTableNoSnapshot) {
    const std::string root = unique_root("empty");
    const std::string ns = "default";
    const std::string nm = "t1";
    const std::string tbl_dir = root + "/" + ns + "/" + nm;
    const std::string meta = R"({
        "format-version": 2,
        "table-uuid": "00000000-0000-0000-0000-000000000000",
        "location": ")" + tbl_dir + R"(",
        "current-snapshot-id": -1,
        "schemas": [{"schema-id":0,"fields":[]}],
        "partition-specs": [{"spec-id":0,"fields":[]}]
    })";
    write_file(tbl_dir + "/metadata/v1.metadata.json", meta);
    write_file(tbl_dir + "/metadata/version-hint.text", "1");

    bolt::Arena arena;
    FilesystemCatalog fs{};
    Catalog cat{};
    ASSERT_TRUE(filesystem_catalog_init(&fs, root.c_str(), &cat));
    // Directory is already laid out; create_table is idempotent here — accept
    // either kCatOk (fresh) or kCatExists (we pre-staged the layout).
    const int rc = cat_create_table(&cat, ns.c_str(), nm.c_str(),
                                    TableLayout::kIceberg);
    ASSERT_TRUE(rc == kCatOk || rc == kCatExists) << "rc=" << rc;

    TableHandle* h = nullptr;
    ASSERT_TRUE(iceberg_table_open(&h, &arena, &cat, ns.c_str(), nm.c_str()));
    ScanHandle* s = nullptr;
    ASSERT_TRUE(iceberg_scan_open(&s, h, nullptr));
    bolt::BoltBatch out{};
    bool eof = false;
    ASSERT_TRUE(iceberg_scan_next_batch(s, &out, &eof));
    EXPECT_TRUE(eof);
    iceberg_scan_close(s);
    iceberg_table_close(h);
}
