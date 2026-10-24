// G2ICE-276: format-version 3 manifests (first_row_id, referenced_data_file, content_offset / content_size_in_bytes,
// PUFFIN), v3 manifest lists (first_row_id) and Puffin deletion-vector files, read back by bolt's own readers.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/iceberg/deletion_vector.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/manifest_avro.h"
#include "bolt/lakehouse/iceberg/metadata.h"

using namespace bolt::lakehouse::iceberg;

namespace {
const char kSchema[] = "{\"type\":\"struct\",\"schema-id\":0,\"fields\":[{\"id\":1,\"name\":\"id\",\"required\":true,\"type\":\"long\"}]}";

DataFileRef data_file(const char* path, int64_t rows, bool frid, int64_t first) {
    DataFileRef d;
    std::memset(&d, 0, sizeof(d));
    d.status = ManifestStatus::kAdded;
    d.content = FileContent::kData;
    std::snprintf(d.file_path, sizeof(d.file_path), "%s", path);
    d.stats.record_count = rows;
    d.stats.file_size_in_bytes = 100;
    d.has_first_row_id = frid;
    d.first_row_id = first;
    return d;
}
}  // namespace

TEST(IcebergV3Write, ManifestCarriesRowLineageAndDeletionVectors) {
    bolt::Arena a;
    DataFileRef f[2] = {data_file("s3://b/t/data/a.parquet", 5, true, 40), data_file("s3://b/t/data/b.parquet", 3, false, 0)};
    const uint8_t* m = nullptr; uint64_t ml = 0;
    ASSERT_TRUE(manifest_write_avro_fv(3, f, 2, 7, 7, kSchema, sizeof(kSchema) - 1, 0, nullptr, nullptr, &a, &m, &ml));
    DataFileRef back[4];
    uint32_t n = 0;
    ASSERT_TRUE(manifest_parse_avro(m, ml, &a, 0, back, 4, &n));
    ASSERT_EQ(n, 2u);
    EXPECT_TRUE(back[0].has_first_row_id);
    EXPECT_EQ(back[0].first_row_id, 40);
    EXPECT_FALSE(back[1].has_first_row_id) << "null: inherited from the manifest's first_row_id";
    EXPECT_FALSE(back[0].is_dv);
    EXPECT_NE(std::string(reinterpret_cast<const char*>(m), ml).find(std::string("format-version\x02" "3")), std::string::npos)
        << "the OCF metadata says format-version 3";

    DataFileRef dv;
    std::memset(&dv, 0, sizeof(dv));
    dv.status = ManifestStatus::kAdded;
    dv.content = FileContent::kPositionDeletes;
    std::snprintf(dv.file_path, sizeof(dv.file_path), "%s", "s3://b/t/data/d.puffin");
    std::snprintf(dv.referenced_data_file, sizeof(dv.referenced_data_file), "%s", "s3://b/t/data/a.parquet");
    dv.is_dv = true; dv.content_offset = 4; dv.content_size = 42;
    dv.stats.record_count = 2; dv.stats.file_size_in_bytes = 300;
    dv.has_first_row_id = true; dv.first_row_id = 99;                   // never written for a delete file
    ASSERT_TRUE(manifest_write_avro_fv(3, &dv, 1, 8, 8, kSchema, sizeof(kSchema) - 1, 0, nullptr, nullptr, &a, &m, &ml));
    ASSERT_TRUE(manifest_parse_avro(m, ml, &a, 0, back, 4, &n));
    ASSERT_EQ(n, 1u);
    EXPECT_TRUE(back[0].is_dv);
    EXPECT_EQ(back[0].content_offset, 4);
    EXPECT_EQ(back[0].content_size, 42);
    EXPECT_STREQ(back[0].referenced_data_file, "s3://b/t/data/a.parquet");
    EXPECT_FALSE(back[0].has_first_row_id);
    EXPECT_NE(std::string(reinterpret_cast<const char*>(m), ml).find("PUFFIN"), std::string::npos);
    EXPECT_FALSE(manifest_write_avro_fv(4, &dv, 1, 8, 8, kSchema, sizeof(kSchema) - 1, 0, nullptr, nullptr, &a, &m, &ml));
}

TEST(IcebergV3Write, ManifestListCarriesFirstRowIdForDataManifestsOnly) {
    bolt::Arena a;
    ManifestListEntry e[2];
    std::memset(e, 0, sizeof(e));
    std::snprintf(e[0].manifest_path, sizeof(e[0].manifest_path), "%s", "s3://b/t/metadata/m0.avro");
    e[0].content = ManifestContent::kDataManifest;
    e[0].has_first_row_id = true; e[0].first_row_id = 1000;
    std::snprintf(e[1].manifest_path, sizeof(e[1].manifest_path), "%s", "s3://b/t/metadata/m1.avro");
    e[1].content = ManifestContent::kDeleteManifest;
    e[1].has_first_row_id = true; e[1].first_row_id = 5;
    const uint8_t* l = nullptr; uint64_t ll = 0;
    ASSERT_TRUE(manifest_list_write_avro_fv(3, e, 2, 9, 9, &a, &l, &ll));
    ManifestListEntry back[2];
    std::memset(back, 0, sizeof(back));
    uint32_t n = 0;
    ASSERT_TRUE(manifest_list_parse_avro(l, ll, &a, back, 2, &n));
    ASSERT_EQ(n, 2u);
    EXPECT_TRUE(back[0].has_first_row_id);
    EXPECT_EQ(back[0].first_row_id, 1000);
    EXPECT_FALSE(back[1].has_first_row_id);
    ASSERT_TRUE(manifest_list_write_avro_fv(2, e, 2, 9, 9, &a, &l, &ll));
    std::memset(back, 0, sizeof(back));
    ASSERT_TRUE(manifest_list_parse_avro(l, ll, &a, back, 2, &n));
    EXPECT_FALSE(back[0].has_first_row_id) << "a v2 list has no first_row_id";
}

TEST(IcebergV3Write, PuffinFileHoldsEachVectorAtItsPlacement) {
    bolt::Arena a;
    const uint32_t p0[] = {0, 3, 70000};
    const uint32_t p1[] = {1};
    const DvBlobIn in[2] = {{p0, 3, "s3://b/t/data/a.parquet"}, {p1, 1, "s3://b/t/data/\"q\".parquet"}};
    DvBlobPlacement pl[2];
    const uint8_t* f = nullptr; uint64_t fl = 0;
    ASSERT_TRUE(puffin_write_deletion_vectors(in, 2, "gestalt", &a, &f, &fl, pl));
    ASSERT_GE(fl, 16u);
    EXPECT_EQ(std::memcmp(f, "PFA1", 4), 0);
    EXPECT_EQ(std::memcmp(f + fl - 4, "PFA1", 4), 0);
    EXPECT_EQ(pl[0].offset, 4);
    EXPECT_EQ(pl[1].offset, pl[0].offset + pl[0].length);
    const std::string all(reinterpret_cast<const char*>(f), fl);
    EXPECT_NE(all.find("\"deletion-vector-v1\""), std::string::npos);
    EXPECT_NE(all.find("\\\"q\\\""), std::string::npos) << "the path is JSON-escaped";
    uint32_t plen = 0;
    for (int b = 0; b < 4; ++b) plen |= static_cast<uint32_t>(f[fl - 12 + b]) << (8 * b);
    EXPECT_EQ(std::memcmp(f + fl - 12 - plen - 4, "PFA1", 4), 0) << "footer magic precedes the payload";
    for (int i = 0; i < 2; ++i) {
        bolt::ingest::RoaringBitmap64 dv{};
        ASSERT_TRUE(iceberg_dv_decode(f + pl[i].offset, static_cast<uint64_t>(pl[i].length), &a, &dv));
        uint64_t pos[8], n = 0;
        ASSERT_TRUE(iceberg_dv_positions(&dv, &a, pos, 8, &n));
        ASSERT_EQ(n, in[i].n);
        for (uint64_t k = 0; k < n; ++k) EXPECT_EQ(pos[k], in[i].positions[k]);
    }
}

TEST(IcebergV3Metadata, ParsesNextRowIdAndInitialDefaults) {
    const std::string j = "{\"format-version\":3,\"table-uuid\":\"u\",\"location\":\"s3://b/t\",\"last-updated-ms\":1,\"next-row-id\":42,"
        "\"last-column-id\":6,\"current-schema-id\":0,\"schemas\":[{\"schema-id\":0,\"type\":\"struct\",\"fields\":["
        "{\"id\":1,\"name\":\"a\",\"required\":false,\"type\":\"long\",\"initial-default\":7},"
        "{\"id\":2,\"name\":\"b\",\"required\":false,\"type\":\"string\",\"initial-default\":\"x y\"},"
        "{\"id\":3,\"name\":\"c\",\"required\":false,\"type\":\"boolean\",\"initial-default\":true},"
        "{\"id\":4,\"name\":\"d\",\"required\":false,\"type\":\"long\",\"initial-default\":null},"
        "{\"id\":5,\"name\":\"e\",\"required\":false,\"type\":{\"type\":\"struct\",\"fields\":[]},\"initial-default\":{}},"
        "{\"id\":6,\"name\":\"f\",\"required\":false,\"type\":\"long\"}]}],"
        "\"default-spec-id\":0,\"partition-specs\":[{\"spec-id\":0,\"fields\":[]}],\"snapshots\":[]}";
    bolt::Arena a;
    auto* m = a.allocate_array<Metadata>(1);
    std::memset(m, 0, sizeof(*m));
    ASSERT_TRUE(metadata_parse(reinterpret_cast<const uint8_t*>(j.data()), static_cast<uint32_t>(j.size()), &a, m));
    EXPECT_TRUE(m->has_next_row_id);
    EXPECT_EQ(m->next_row_id, 42);
    const SchemaField* f = m->schemas[0].fields;
    ASSERT_EQ(m->schemas[0].n_fields, 6u);
    EXPECT_EQ(f[0].initial_default_kind, DefaultKind::kScalar);
    EXPECT_STREQ(f[0].initial_default, "7");
    EXPECT_EQ(f[1].initial_default_kind, DefaultKind::kString);
    EXPECT_STREQ(f[1].initial_default, "x y");
    EXPECT_EQ(f[2].initial_default_kind, DefaultKind::kScalar);
    EXPECT_STREQ(f[2].initial_default, "true");
    EXPECT_EQ(f[3].initial_default_kind, DefaultKind::kNone);
    EXPECT_EQ(f[4].initial_default_kind, DefaultKind::kUnsupported) << "a struct default is not filled: the scan refuses it";
    EXPECT_EQ(f[5].initial_default_kind, DefaultKind::kNone);
}
