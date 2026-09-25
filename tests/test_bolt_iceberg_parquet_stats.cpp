// test_bolt_iceberg_parquet_stats.cpp — G2ICE-43: Iceberg data-file stats
// folded from a parquet footer.
//
// Every expectation is a value the test computes from the rows it wrote, not
// from the footer, so a fold that picks one row group's bound instead of the
// file's, drops a sign bit, or trusts an all-null chunk fails on a number.
// The last case runs the stats through the manifest writer and back through
// the pyiceberg-verified reader so the bytes a client sees are checked too.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/manifest_avro.h"
#include "bolt/lakehouse/iceberg/statistics.h"

namespace ice = bolt::lakehouse::iceberg;
namespace pq  = bolt::ingest::parquet;

namespace {

struct Rg {
    std::vector<int64_t>     id;
    std::vector<double>      px;      // NaN allowed
    std::vector<bool>        px_null;
    std::vector<std::string> sym;
    std::vector<bool>        sym_null;
};

uint8_t* bitmap(bolt::Arena* a, const std::vector<bool>& nulls) {
    const size_t n = nulls.size();
    auto* bm = static_cast<uint8_t*>(a->allocate((n + 7u) / 8u, 8u));
    std::memset(bm, 0, (n + 7u) / 8u);
    for (size_t i = 0; i < n; ++i) {
        if (!nulls[i]) bm[i >> 3] = static_cast<uint8_t>(bm[i >> 3] | (1u << (i & 7u)));
    }
    return bm;
}

void build(bolt::Arena* a, const Rg& rg, bolt::BoltBatch* b) {
    const int64_t n = static_cast<int64_t>(rg.id.size());
    bolt::BoltBatch::init_empty(b);
    b->num_cols = 4;
    b->num_rows = n;
    bolt::BoltBatch::alloc_columns(b, a, 4);
    b->schema.add_field("id", bolt::BoltType::Int64, false);
    b->schema.add_field("px", bolt::BoltType::Float64, true);
    b->schema.add_field("sym", bolt::BoltType::Utf8, true);
    b->schema.add_field("gone", bolt::BoltType::Int64, true);

    bolt::BoltColumn& id = b->columns[b->read_epoch][0];
    id = bolt::BoltColumn::make_flat_alloc(n, bolt::BoltType::Int64, a);
    std::memcpy(id.data, rg.id.data(), static_cast<size_t>(n) * 8u);

    bolt::BoltColumn& px = b->columns[b->read_epoch][1];
    px = bolt::BoltColumn::make_flat_alloc(n, bolt::BoltType::Float64, a);
    std::memcpy(px.data, rg.px.data(), static_cast<size_t>(n) * 8u);
    px.validity = bitmap(a, rg.px_null);
    px.stats.all_valid = false;

    bolt::BoltColumn& sy = b->columns[b->read_epoch][2];
    sy = bolt::BoltColumn::make_empty();
    sy.length = n;
    sy.format = bolt::ColumnFormat::Flat;
    sy.type = bolt::BoltType::Utf8;
    sy.type_size_bytes = sizeof(bolt::StringView);
    auto* svs = static_cast<bolt::StringView*>(
        a->allocate(static_cast<size_t>(n) * sizeof(bolt::StringView),
                    alignof(bolt::StringView)));
    std::memset(svs, 0, static_cast<size_t>(n) * sizeof(bolt::StringView));
    for (int64_t i = 0; i < n; ++i) {
        const std::string& s = rg.sym[static_cast<size_t>(i)];
        EXPECT_LE(s.size(), 12u);
        svs[i].length = static_cast<uint32_t>(s.size());
        std::memcpy(&svs[i].prefix[0], s.data(), s.size());
    }
    sy.data = svs;
    sy.validity = bitmap(a, rg.sym_null);
    sy.stats.all_valid = false;

    bolt::BoltColumn& g = b->columns[b->read_epoch][3];
    g = bolt::BoltColumn::make_flat_alloc(n, bolt::BoltType::Int64, a);
    g.validity = bitmap(a, std::vector<bool>(static_cast<size_t>(n), true));
    g.stats.all_valid = false;
}

pq::ParquetWriteOpts opts() {
    pq::ParquetWriteOpts o{};
    o.n_columns = 4;
    o.compression = 0;
    o.emit_statistics = true;
    const char* names[4] = {"id", "px", "sym", "gone"};
    const bolt::BoltType types[4] = {bolt::BoltType::Int64, bolt::BoltType::Float64,
                                     bolt::BoltType::Utf8, bolt::BoltType::Int64};
    for (uint32_t i = 0; i < 4; ++i) {
        std::snprintf(o.columns[i].name, sizeof(o.columns[i].name), "%s", names[i]);
        o.columns[i].type = types[i];
        o.columns[i].nullable = i != 0u;
    }
    return o;
}

// Three row groups whose extremes live in DIFFERENT groups.
std::vector<Rg> fixture() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<Rg> v(3);
    v[0] = {{5, 9, 7}, {1.5, nan, 0.0}, {false, false, false},
            {"mm", "zz", "x"}, {false, false, true}};
    v[1] = {{-40, 3, 12}, {-2.25, 8.0, 0.0}, {false, false, true},
            {"b", "c", "q"}, {true, false, false}};
    v[2] = {{100, 0, 1}, {0.0, 0.0, 0.0}, {true, true, true},
            {"aa", "zzz", "m"}, {false, false, false}};
    return v;
}

bool write_file(bolt::Arena* a, const uint8_t** out, uint64_t* out_len) {
    pq::ParquetWriteOpts o = opts();
    pq::ParquetWriter* w = pq::parquet_write_open_mem(&o);
    if (w == nullptr) return false;
    for (const Rg& rg : fixture()) {
        auto* b = a->allocate_array<bolt::BoltBatch>(1);
        build(a, rg, b);
        if (!pq::parquet_write_row_group(w, b)) return false;
    }
    return pq::parquet_write_close_mem(w, a, out, out_len);
}

template <typename T>
T as(const char* b) { T v; std::memcpy(&v, b, sizeof(T)); return v; }

const ice::ColumnStatEntry* find(const ice::FileStats& s, int32_t fid) {
    for (uint32_t i = 0; i < s.n_cols; ++i) {
        if (s.cols[i].field_id == fid) return &s.cols[i];
    }
    return nullptr;
}

void fold(bolt::Arena* a, ice::FileStats* st) {
    const uint8_t* buf = nullptr;
    uint64_t len = 0;
    ASSERT_TRUE(write_file(a, &buf, &len));
    auto* meta = a->allocate_array<pq::PqMeta>(1);
    ASSERT_NE(meta, nullptr);
    ASSERT_TRUE(pq::parquet_read_meta(buf, len, a, meta));
    ASSERT_EQ(meta->n_row_groups, 3u);
    const int32_t fids[4] = {1, 2, 3, 4};
    std::memset(st, 0, sizeof(*st));
    ASSERT_TRUE(ice::file_stats_from_parquet_meta(meta, fids, 4, st));
}

}  // namespace

TEST(IcebergParquetStats, FoldsBoundsAndNullsAcrossRowGroups) {
    bolt::Arena a;
    ice::FileStats st;
    fold(&a, &st);
    ASSERT_EQ(st.n_cols, 4u);

    const ice::ColumnStatEntry* id = find(st, 1);
    ASSERT_NE(id, nullptr);
    EXPECT_EQ(id->null_count, 0);
    ASSERT_TRUE(id->has_lower && id->has_upper);
    ASSERT_EQ(id->lower_len, 8u);
    EXPECT_EQ(as<int64_t>(id->lower), -40);
    EXPECT_EQ(as<int64_t>(id->upper), 100);

    // NaN is excluded from both bounds.
    const ice::ColumnStatEntry* px = find(st, 2);
    ASSERT_NE(px, nullptr);
    EXPECT_EQ(px->null_count, 4);
    ASSERT_TRUE(px->has_lower && px->has_upper);
    EXPECT_EQ(as<double>(px->lower), -2.25);
    EXPECT_EQ(as<double>(px->upper), 8.0);

    const ice::ColumnStatEntry* sy = find(st, 3);
    ASSERT_NE(sy, nullptr);
    EXPECT_EQ(sy->null_count, 2);
    ASSERT_TRUE(sy->has_lower && sy->has_upper);
    EXPECT_EQ(std::string(sy->lower, sy->lower_len), "aa");
    EXPECT_EQ(std::string(sy->upper, sy->upper_len), "zzz");

    // All-null: a counted column with nothing to bound.
    const ice::ColumnStatEntry* g = find(st, 4);
    ASSERT_NE(g, nullptr);
    EXPECT_EQ(g->null_count, 9);
    EXPECT_FALSE(g->has_lower);
    EXPECT_FALSE(g->has_upper);
}

TEST(IcebergParquetStats, SkipsUnmappedColumnsAndRejectsNull) {
    bolt::Arena a;
    const uint8_t* buf = nullptr;
    uint64_t len = 0;
    ASSERT_TRUE(write_file(&a, &buf, &len));
    auto* meta = a.allocate_array<pq::PqMeta>(1);
    ASSERT_TRUE(pq::parquet_read_meta(buf, len, &a, meta));
    const int32_t fids[4] = {0, 7, -1, 0};
    ice::FileStats st{};
    ASSERT_TRUE(ice::file_stats_from_parquet_meta(meta, fids, 4, &st));
    ASSERT_EQ(st.n_cols, 1u);
    EXPECT_EQ(st.cols[0].field_id, 7);
    EXPECT_FALSE(ice::file_stats_from_parquet_meta(nullptr, fids, 4, &st));
    EXPECT_FALSE(ice::file_stats_from_parquet_meta(meta, nullptr, 4, &st));
}

// The stats survive the manifest writer and the pyiceberg-verified reader.
TEST(IcebergParquetStats, ManifestCarriesFoldedBounds) {
    bolt::Arena a;
    ice::DataFileRef f{};
    f.status  = ice::ManifestStatus::kAdded;
    f.content = ice::FileContent::kData;
    f.snapshot_id = 9;
    std::snprintf(f.file_path, sizeof(f.file_path), "s3://wh/d/t/data/r.parquet");
    fold(&a, &f.stats);
    f.stats.record_count = 9;
    f.stats.file_size_in_bytes = 1234;
    static const char kSchema[] =
        "{\"type\":\"struct\",\"schema-id\":0,\"fields\":["
        "{\"id\":1,\"name\":\"id\",\"required\":false,\"type\":\"long\"},"
        "{\"id\":2,\"name\":\"px\",\"required\":false,\"type\":\"double\"},"
        "{\"id\":3,\"name\":\"sym\",\"required\":false,\"type\":\"string\"},"
        "{\"id\":4,\"name\":\"gone\",\"required\":false,\"type\":\"long\"}]}";
    const uint8_t* mbuf = nullptr;
    uint64_t mlen = 0;
    ASSERT_TRUE(ice::manifest_write_avro(&f, 1, 9, 1, kSchema, sizeof(kSchema) - 1u,
                                         0, &a, &mbuf, &mlen));
    ice::DataFileRef out[2]{};
    uint32_t n = 0;
    ASSERT_TRUE(ice::manifest_parse_avro(mbuf, mlen, &a, 0, out, 2, &n));
    ASSERT_EQ(n, 1u);
    ASSERT_TRUE(out[0].binary_bounds);
    const ice::ColumnStatEntry* id = find(out[0].stats, 1);
    ASSERT_NE(id, nullptr);
    ASSERT_TRUE(id->has_lower && id->has_upper);
    EXPECT_EQ(as<int64_t>(id->lower), -40);
    EXPECT_EQ(as<int64_t>(id->upper), 100);
    const ice::ColumnStatEntry* sy = find(out[0].stats, 3);
    ASSERT_NE(sy, nullptr);
    EXPECT_EQ(sy->null_count, 2);
    EXPECT_EQ(std::string(sy->upper, sy->upper_len), "zzz");
    const ice::ColumnStatEntry* g = find(out[0].stats, 4);
    ASSERT_NE(g, nullptr);
    EXPECT_EQ(g->null_count, 9);
    EXPECT_FALSE(g->has_lower);
}
