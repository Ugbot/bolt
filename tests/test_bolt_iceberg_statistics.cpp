// G2ICE-450: table metadata keeps `statistics` / `partition-statistics`; a
// Puffin footer's blobs parse; theta sketches serialized by Apache
// DataSketches (headers below are datasketches 5.x output) give their estimate.
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/puffin.h"
#include "bolt/bolt_arena.h"
#include <gtest/gtest.h>
#include <cstring>
#include <string>
#include <vector>
namespace b = bolt::lakehouse::iceberg;
namespace {
const std::string schema = R"("current-schema-id":0,"schemas":[{"type":"struct","schema-id":0,"fields":[{"id":1,"name":"a","required":true,"type":"long"},{"id":2,"name":"b","required":false,"type":"string"}]}])";
const std::string stats = R"("statistics":[{"snapshot-id":7,"statistics-path":"s3://bk/t/metadata/7.stats","file-size-in-bytes":413,"file-footer-size-in-bytes":299,"blob-metadata":[{"type":"apache-datasketches-theta-v1","snapshot-id":7,"sequence-number":3,"fields":[1],"properties":{"ndv":"1234"}},{"type":"apache-datasketches-theta-v1","snapshot-id":7,"sequence-number":3,"fields":[2]},{"type":"custom-v1","snapshot-id":7,"sequence-number":3,"fields":[1,2],"properties":{"x":"y"}}]}],"partition-statistics":[{"snapshot-id":7,"statistics-path":"s3://bk/t/metadata/p7.parquet","file-size-in-bytes":99}])";
b::Metadata* parse(bolt::Arena& arena, const std::string& input) {
    auto* out = arena.allocate_array<b::Metadata>(1u);
    if (out == nullptr) return nullptr;
    b::metadata_init(out, &arena, nullptr);
    return b::metadata_parse(reinterpret_cast<const uint8_t*>(input.data()), static_cast<uint32_t>(input.size()), &arena, out) ? out : nullptr;
}
std::vector<uint8_t> hex(const char* h) {
    std::vector<uint8_t> v;
    for (size_t i = 0; h[i] && h[i + 1]; i += 2) v.push_back(static_cast<uint8_t>(std::stoul(std::string(h + i, 2), nullptr, 16)));
    return v;
}
}  // namespace

TEST(IcebergStatistics, MetadataKeepsStatisticsAndPartitionStatistics) {
    bolt::Arena arena;
    const auto* m = parse(arena, "{\"format-version\":2," + schema + "," + stats + "}");
    ASSERT_NE(m, nullptr);
    ASSERT_EQ(m->n_statistics, 1u);
    const b::StatisticsFile* sf = b::metadata_statistics_for(m, 7);
    ASSERT_NE(sf, nullptr);
    EXPECT_EQ(b::metadata_statistics_for(m, 8), nullptr);
    EXPECT_STREQ(sf->path, "s3://bk/t/metadata/7.stats");
    EXPECT_EQ(sf->file_size, 413); EXPECT_EQ(sf->footer_size, 299);
    ASSERT_EQ(sf->n_blobs, 3u);
    EXPECT_STREQ(sf->blobs[0].type, b::kThetaBlobType);
    ASSERT_EQ(sf->blobs[0].n_fields, 1u); EXPECT_EQ(sf->blobs[0].fields[0], 1);
    EXPECT_EQ(sf->blobs[0].ndv, 1234); EXPECT_EQ(sf->blobs[0].sequence_number, 3);
    EXPECT_EQ(sf->blobs[1].ndv, -1);
    ASSERT_EQ(sf->blobs[2].n_fields, 2u); EXPECT_EQ(sf->blobs[2].fields[1], 2);
    ASSERT_EQ(m->n_partition_statistics, 1u);
    EXPECT_EQ(m->partition_statistics[0].snapshot_id, 7);
    EXPECT_EQ(m->partition_statistics[0].file_size, 99);
}

TEST(IcebergStatistics, AStatisticsFileWithoutAPathIsRefused) {
    bolt::Arena arena;
    EXPECT_EQ(parse(arena, "{\"format-version\":2," + schema + R"(,"statistics":[{"snapshot-id":7,"file-size-in-bytes":1,"file-footer-size-in-bytes":1,"blob-metadata":[]}]})"), nullptr);
}

TEST(IcebergStatistics, ThetaSketchEstimates) {
    double est = -1.0;
    const auto exact = hex("02030300001acc930500000000000000");                         // 5 distinct, exact mode
    ASSERT_TRUE(b::theta_sketch_estimate(exact.data(), exact.size(), &est)); EXPECT_DOUBLE_EQ(est, 5.0);
    const auto v3 = hex("03030300001acc93bd10000000000000b45c206511599a05");         // 100000 updates, lg_k 12
    ASSERT_TRUE(b::theta_sketch_estimate(v3.data(), v3.size(), &est)); EXPECT_NEAR(est, 97891.786, 0.01);
    const auto v4 = hex("02040332021acc93b45c206511599a05bd10");                     // the same, compressed (v4)
    ASSERT_TRUE(b::theta_sketch_estimate(v4.data(), v4.size(), &est)); EXPECT_NEAR(est, 97891.786, 0.01);
    const auto quick = hex("02030200001acc93");                                       // an update sketch: refused
    EXPECT_FALSE(b::theta_sketch_estimate(quick.data(), quick.size(), &est));
}

TEST(IcebergStatistics, PuffinFooterBlobs) {
    const std::string footer = R"({"blobs":[{"type":"apache-datasketches-theta-v1","fields":[3],"snapshot-id":9,"sequence-number":1,"offset":4,"length":24,"properties":{"ndv":"42"}}],"properties":{"created-by":"test"}})";
    std::vector<uint8_t> f = {'P', 'F', 'A', '1'};
    const auto blob = hex("03030300001acc93bd10000000000000b45c206511599a05");
    f.insert(f.end(), blob.begin(), blob.end());
    f.insert(f.end(), {'P', 'F', 'A', '1'});
    f.insert(f.end(), footer.begin(), footer.end());
    const uint32_t n = static_cast<uint32_t>(footer.size());
    for (int i = 0; i < 4; ++i) f.push_back(static_cast<uint8_t>(n >> (8 * i)));
    f.insert(f.end(), {0, 0, 0, 0, 'P', 'F', 'A', '1'});
    bolt::Arena arena;
    b::BlobMetadata* blobs = nullptr; uint32_t count = 0;
    ASSERT_TRUE(b::puffin_file_blobs(f.data(), f.size(), &arena, &blobs, &count));
    ASSERT_EQ(count, 1u);
    EXPECT_EQ(blobs[0].offset, 4); EXPECT_EQ(blobs[0].length, 24); EXPECT_EQ(blobs[0].ndv, 42);
    EXPECT_EQ(blobs[0].fields[0], 3); EXPECT_EQ(blobs[0].snapshot_id, 9);
    f[f.size() - 8u] = 1u;                                                             // compressed footer: refused
    EXPECT_FALSE(b::puffin_file_blobs(f.data(), f.size(), &arena, &blobs, &count));
}
