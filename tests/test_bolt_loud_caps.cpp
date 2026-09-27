// L1a loud sweep: every cap below used to drop, clamp or overrun silently.
// Each test builds the overflowing input and asserts either the full answer
// (where the cap was removed) or an explicit refusal (where it remains).

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_ebr.h"
#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/join/bolt_hashjoin.h"
#include "bolt/join/bolt_joinkernel.h"
#include "bolt/kernels/bolt_kmeans.h"
#include "bolt/kernels/fintech/sma.h"
#include "bolt/kernels/fintech/sorted_ring.h"
#include "bolt/kernels/fintech/state.h"
#include "bolt/kernels/fintech/throttle_check.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/delta/log.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/object_store.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace bolt::lakehouse;

std::string unique_root(const char* tag) {
    auto p = std::filesystem::temp_directory_path() /
             ("bolt_loud_caps_" + std::string(tag) + "_" +
              std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
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

std::vector<uint8_t> read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
}

const uint8_t* u8(const std::string& s) {
    return reinterpret_cast<const uint8_t*>(s.data());
}

std::string snapshots_json(uint32_t n) {
    std::string s = "[";
    for (uint32_t i = 0; i < n; ++i) {
        if (i) s += ",";
        s += "{\"snapshot-id\":" + std::to_string(1000 + i) +
             ",\"timestamp-ms\":" + std::to_string(1 + i) +
             ",\"manifest-list\":\"metadata/snap.json\"}";
    }
    return s + "]";
}

std::string metadata_json(const std::string& location, uint32_t n_snaps,
                          int64_t current) {
    return R"({"format-version":2,"table-uuid":"u","location":")" + location +
           R"(","current-snapshot-id":)" + std::to_string(current) +
           R"(,"schemas":[{"schema-id":0,"fields":[{"id":1,"name":"id","required":true,"type":"long"}]}],)"
           R"("partition-specs":[{"spec-id":0,"fields":[]}],"snapshots":)" +
           snapshots_json(n_snaps) + "}";
}

}  // namespace

// ---- Iceberg metadata ------------------------------------------------------

// Past the snapshot array the current snapshot used to be dropped and the
// scan returned an EMPTY table with success.
TEST(LoudIcebergMetadata, SnapshotsPastCapAreRefusedNotDropped) {
    using namespace bolt::lakehouse::iceberg;
    bolt::Arena arena;
    Metadata* m = arena.allocate_array<Metadata>(1);
    ASSERT_NE(m, nullptr);
    const std::string ok = metadata_json("/t", kIcebergMaxSnapshots,
                                         1000 + kIcebergMaxSnapshots - 1);
    ASSERT_TRUE(metadata_parse(u8(ok), static_cast<uint32_t>(ok.size()),
                               &arena, m));
    EXPECT_EQ(m->n_snapshots, kIcebergMaxSnapshots);
    const std::string over = metadata_json("/t", kIcebergMaxSnapshots + 6,
                                           1000 + kIcebergMaxSnapshots + 5);
    EXPECT_FALSE(metadata_parse(u8(over), static_cast<uint32_t>(over.size()),
                                &arena, m));
}

TEST(LoudIcebergMetadata, CurrentSnapshotMissingFromListIsRefused) {
    using namespace bolt::lakehouse::iceberg;
    bolt::Arena arena;
    Metadata* m = arena.allocate_array<Metadata>(1);
    const std::string j = metadata_json("/t", 3, 424242);
    EXPECT_FALSE(metadata_parse(u8(j), static_cast<uint32_t>(j.size()),
                                &arena, m));
}

TEST(LoudIcebergMetadata, OverlongFieldNameIsRefusedNotCut) {
    using namespace bolt::lakehouse::iceberg;
    bolt::Arena arena;
    Metadata* m = arena.allocate_array<Metadata>(1);
    const std::string longname(kIcebergMaxFieldName + 5, 'q');
    std::string j = metadata_json("/t", 1, 1000);
    const std::string from = "\"name\":\"id\"";
    j.replace(j.find(from), from.size(), "\"name\":\"" + longname + "\"");
    EXPECT_FALSE(metadata_parse(u8(j), static_cast<uint32_t>(j.size()),
                                &arena, m));
}

// ---- Iceberg manifests -----------------------------------------------------

TEST(LoudIcebergManifest, JsonEntriesPastCapReportNeedForRoom) {
    using namespace bolt::lakehouse::iceberg;
    bolt::Arena arena;
    std::string j = "[";
    for (int i = 0; i < 5; ++i) {
        if (i) j += ",";
        j += R"({"status":1,"data_file":{"file_path":"data/f.parquet","record_count":1}})";
    }
    j += "]";
    DataFileRef* out = arena.allocate_array<DataFileRef>(5);
    uint32_t n = 0;
    EXPECT_FALSE(manifest_parse_json(u8(j), static_cast<uint32_t>(j.size()),
                                     &arena, 0, out, 3, &n));
    EXPECT_EQ(n, 3u);
    EXPECT_TRUE(manifest_parse_json(u8(j), static_cast<uint32_t>(j.size()),
                                    &arena, 0, out, 5, &n));
    EXPECT_EQ(n, 5u);
}

TEST(LoudIcebergManifest, PartitionValuesPastCapAreRefused) {
    using namespace bolt::lakehouse::iceberg;
    bolt::Arena arena;
    std::string parts;
    for (uint32_t i = 0; i < kIcebergMaxPartitionValues + 1; ++i) {
        if (i) parts += ",";
        parts += R"({"field-id":)" + std::to_string(1000 + i) + R"(,"value":)" +
                 std::to_string(i) + "}";
    }
    const std::string j = R"([{"status":1,"data_file":{"file_path":"data/f.parquet","partition":[)" +
                          parts + R"(]}}])";
    DataFileRef* out = arena.allocate_array<DataFileRef>(1);
    uint32_t n = 0;
    EXPECT_FALSE(manifest_parse_json(u8(j), static_cast<uint32_t>(j.size()),
                                     &arena, 0, out, 1, &n));
}

TEST(LoudIcebergManifest, OverlongPathIsRefusedAndCutBoundIsDropped) {
    using namespace bolt::lakehouse::iceberg;
    bolt::Arena arena;
    DataFileRef* out = arena.allocate_array<DataFileRef>(1);
    uint32_t n = 0;
    const std::string longpath(kIcebergMaxPath + 10, 'p');
    const std::string j1 = R"([{"status":1,"data_file":{"file_path":")" +
                           longpath + R"("}}])";
    EXPECT_FALSE(manifest_parse_json(u8(j1), static_cast<uint32_t>(j1.size()),
                                     &arena, 0, out, 1, &n));
    const std::string big(kLakeMaxValBytes + 20, 'z');
    const std::string j2 =
        R"([{"status":1,"data_file":{"file_path":"data/f.parquet","upper_bounds":[{"field-id":1,"value":")" +
        big + R"("}],"lower_bounds":[{"field-id":1,"value":"a"}]}}])";
    ASSERT_TRUE(manifest_parse_json(u8(j2), static_cast<uint32_t>(j2.size()),
                                    &arena, 0, out, 1, &n));
    ASSERT_EQ(n, 1u);
    bool saw = false;
    for (uint32_t c = 0; c < out[0].stats.n_cols; ++c) {
        if (out[0].stats.cols[c].field_id != 1) continue;
        saw = true;
        EXPECT_FALSE(out[0].stats.cols[c].has_upper);   // a prefix is no bound
        EXPECT_TRUE(out[0].stats.cols[c].has_lower);
    }
    EXPECT_TRUE(saw);
}

// A real scan past the old per-manifest (4,096 entries), per-list (256
// manifests) and live-file (4,096) caps. Every entry names the same one-row
// file, so the row count is the live-file count. Before: 4,096 rows, true.
TEST(LoudIcebergScan, ReadsEveryLiveFilePastTheOldCaps) {
    using namespace bolt::lakehouse::iceberg;
    namespace pqw = bolt::ingest::parquet;
    const std::string root = unique_root("scan");
    const std::string tbl = root + "/default/wide";
    {
        bolt::Arena a;
        bolt::BoltBatch b{};
        bolt::BoltBatch::init_empty(&b);
        b.num_cols = 1;
        b.num_rows = 1;
        ASSERT_TRUE(bolt::BoltBatch::alloc_columns(&b, &a, 1));
        b.schema.add_field("id", bolt::BoltType::Int64, false);
        bolt::BoltColumn& c = b.columns[b.read_epoch][0];
        c = bolt::BoltColumn::make_flat_alloc(1, bolt::BoltType::Int64, &a);
        static_cast<int64_t*>(c.data)[0] = 7;
        pqw::ParquetWriteOpts o{};
        o.n_columns = 1;
        std::strncpy(o.columns[0].name, "id", sizeof(o.columns[0].name));
        o.columns[0].type = bolt::BoltType::Int64;
        std::filesystem::create_directories(tbl + "/data");
        const std::string f = tbl + "/data/f.parquet";
        pqw::ParquetWriter* w = pqw::parquet_write_open(f.c_str(), &o);
        ASSERT_NE(w, nullptr);
        ASSERT_TRUE(pqw::parquet_write_row_group(w, &b));
        ASSERT_TRUE(pqw::parquet_write_close(w));
    }
    constexpr uint32_t kBig = 5000;       // entries in manifest 0
    constexpr uint32_t kManifests = 300;  // manifests in the list
    const std::string entry =
        R"({"status":1,"data_file":{"file_path":"data/f.parquet","record_count":1}})";
    std::string big = "[";
    for (uint32_t i = 0; i < kBig; ++i) { if (i) big += ","; big += entry; }
    big += "]";
    write_file(tbl + "/metadata/m0.json", big);
    write_file(tbl + "/metadata/m1.json", "[" + entry + "]");
    std::string list = "[";
    for (uint32_t i = 0; i < kManifests; ++i) {
        if (i) list += ",";
        list += std::string(R"({"manifest_path":"metadata/)") +
                (i == 0 ? "m0" : "m1") + R"(.json","partition_spec_id":0})";
    }
    list += "]";
    write_file(tbl + "/metadata/snap.json", list);
    write_file(tbl + "/metadata/v1.metadata.json", metadata_json(tbl, 1, 1000));
    write_file(tbl + "/metadata/version-hint.text", "1");

    bolt::Arena arena;
    FilesystemCatalog fs{};
    Catalog cat{};
    ASSERT_TRUE(filesystem_catalog_init(&fs, root.c_str(), &cat));
    const int rc = cat_create_table(&cat, "default", "wide",
                                    TableLayout::kIceberg);
    ASSERT_TRUE(rc == kCatOk || rc == kCatExists);
    TableHandle* h = nullptr;
    ASSERT_TRUE(iceberg_table_open(&h, &arena, &cat, "default", "wide"));
    ScanHandle* s = nullptr;
    ASSERT_TRUE(iceberg_scan_open(&s, h, nullptr));
    int64_t rows = 0;
    for (uint32_t guard = 0; guard < 2 * (kBig + kManifests); ++guard) {
        bolt::BoltBatch out{};
        bool eof = false;
        ASSERT_TRUE(iceberg_scan_next_batch(s, &out, &eof)) << "after rows=" << rows;
        if (eof) break;
        rows += out.num_rows;
    }
    EXPECT_EQ(rows, static_cast<int64_t>(kBig + kManifests - 1));
    iceberg_scan_close(s);
    iceberg_table_close(h);
}

// ---- Object-store listings + Delta log --------------------------------------

TEST(LoudObjectStore, ListingPastCapIsBufferSmallAndListAllGrows) {
    const std::string root = unique_root("list");
    for (int i = 0; i < 600; ++i) {
        write_file(root + "/p/k" + std::to_string(i), "x");
    }
    FilesystemObjectStore fs{};
    ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    bolt::Arena arena;
    ObjectEntry* buf = arena.allocate_array<ObjectEntry>(512);
    uint32_t n = 0;
    EXPECT_EQ(os_list(&os, "p/", buf, 512, &n), kOsBufferSmall);
    EXPECT_EQ(n, 512u);
    ObjectEntry* all = nullptr;
    ASSERT_EQ(os_list_all(&os, "p/", &arena, 64, &all, &n), kOsOk);
    EXPECT_EQ(n, 600u);
}

namespace {
bool count_adds(void* ctx, const delta::DeltaAction* a) noexcept {
    if (a->kind == delta::ActionKind::kAdd) ++*static_cast<uint32_t*>(ctx);
    return true;
}
}  // namespace

// 600 commits past kLakeMaxCommits (512): the walk used to see 512 and
// report the table without the rest.
TEST(LoudDeltaLog, WalkSeesEveryCommitPastTheListingCap) {
    const std::string root = unique_root("delta");
    constexpr uint32_t kCommits = 600;
    for (uint32_t v = 0; v < kCommits; ++v) {
        char name[64];
        std::snprintf(name, sizeof(name), "%020u.json", v);
        write_file(root + "/t/_delta_log/" + name,
                   R"({"add":{"path":"f)" + std::to_string(v) +
                   R"(.parquet","size":1,"modificationTime":0,"dataChange":true,"partitionValues":{}}})"
                   "\n");
    }
    FilesystemObjectStore fs{};
    ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    bolt::Arena arena;
    uint32_t adds = 0;
    ASSERT_TRUE(delta::delta_log_walk_all(&os, "t", -1, &arena, &adds,
                                          count_adds));
    EXPECT_EQ(adds, kCommits);
}

TEST(LoudDeltaLog, PartitionValuesPastCapAreRefused) {
    std::string pv;
    for (uint32_t i = 0; i < delta::kDeltaMaxPartitions + 1; ++i) {
        if (i) pv += ",";
        pv += "\"c" + std::to_string(i) + "\":\"v\"";
    }
    const std::string line =
        R"({"add":{"path":"f.parquet","size":1,"modificationTime":0,"dataChange":true,"partitionValues":{)" +
        pv + "}}}\n";
    bolt::Arena arena;
    uint32_t adds = 0;
    EXPECT_FALSE(delta::delta_log_parse_commit(u8(line), line.size(), 0,
                                               &arena, &adds, count_adds));
    EXPECT_EQ(adds, 0u);
}

// ---- Parquet names ----------------------------------------------------------

// Two 70+-byte names sharing a 63-byte prefix: cut, they were the SAME name.
TEST(LoudParquetNames, OverlongColumnNamesAreRefusedWhenBoundByName) {
    namespace pq = bolt::ingest::parquet;
    const auto buf = read_all(std::string(BOLT_TEST_DATA_DIR) +
                              "/golden_long_column_names.parquet");
    ASSERT_FALSE(buf.empty());
    bolt::Arena arena;
    pq::PqMeta* m = arena.allocate_array<pq::PqMeta>(1);
    ASSERT_TRUE(pq::parquet_read_meta(buf.data(), buf.size(), &arena, m));
    ASSERT_EQ(m->n_columns, 3u);
    EXPECT_EQ(m->columns[0].name_truncated, 1u);
    EXPECT_EQ(m->columns[1].name_truncated, 1u);
    EXPECT_EQ(m->columns[2].name_truncated, 0u);
    bolt::BoltSchema sch;
    bolt::BoltField* f = arena.allocate_array<bolt::BoltField>(3);
    sch.set_storage(f, 3);
    EXPECT_FALSE(pq::parquet_schema_from_meta(m, &sch, false));
    bolt::BoltBatch b{};
    EXPECT_FALSE(pq::parquet_read_file(buf.data(), buf.size(), &arena, &b));
}

// ---- Join / kernels ---------------------------------------------------------

TEST(LoudJoin, KeyCountPastCapIsRefusedInRelease) {
    bolt::Arena arena;
    const uint8_t over = bolt::kHJMaxKeys + 1;
    std::vector<bolt::BoltColumn> cols(over);
    std::vector<int64_t> v(4, 1);
    for (auto& c : cols) {
        c = bolt::BoltColumn::make_flat_alloc(4, bolt::BoltType::Int64, &arena);
    }
    bolt::JoinBuildTyped* jb = arena.allocate_array<bolt::JoinBuildTyped>(1);
    ASSERT_NE(jb, nullptr);
    EXPECT_FALSE(bolt::jk_build_core(cols.data(), over, 4, &arena, false, jb));
    bolt::HashJoinBuildCfgChained cfg{};
    cfg.n_keys = over;
    cfg.build_rows = 4;
    const uint64_t* keys[16] = {};
    for (auto& k : keys) k = reinterpret_cast<const uint64_t*>(v.data());
    bolt::HashJoinBuildChained out{};
    EXPECT_FALSE(bolt::hash_join_build_chained(keys, cfg, &arena, &out));
}

TEST(LoudFintech, WindowPastRingCapIsRefusedAndMemorySafe) {
    using namespace bolt::fintech;
    bolt::Arena arena;
    auto* sma = arena.allocate_array<SMAState<16>>(1);
    EXPECT_FALSE(sma->init(0, 0, 17));
    EXPECT_FALSE(sma->init(0, 0, 0));
    EXPECT_TRUE(sma->init(0, 0, 16));
    auto* ring = arena.allocate_array<RollingRing<double, 8>>(1);
    EXPECT_FALSE(ring->init(9));
#ifdef NDEBUG
    // Debug asserts on a push into a refused ring; Release must stay in bounds.
    for (int i = 0; i < 100; ++i) ring->push(1.0);   // stays inside ring[0]
    EXPECT_EQ(ring->size(), 0u);
#endif
    auto* sr = arena.allocate_array<SortedRing<double, 8>>(1);
    EXPECT_FALSE(sr->init(9));
    for (int i = 0; i < 100; ++i) sr->push(1.0);
    EXPECT_EQ(sr->size(), 0u);
    EXPECT_EQ(make_throttle_check_state<8>(&arena, 0, 1, 10, 8), nullptr);
}

// More in-window events than the ring holds: the flag must still be exact.
TEST(LoudFintech, ThrottleRingFullStaysExact) {
    using namespace bolt::fintech;
    bolt::Arena arena;
    constexpr uint32_t kCap = 4;
    auto* st = make_throttle_check_state<kCap>(&arena, 0, 1, /*window=*/100,
                                               /*max_count=*/2);
    ASSERT_NE(st, nullptr);
    std::vector<int64_t> ts;
    for (int64_t t = 0; t < 40; ++t) ts.push_back(t * 3);   // dense bursts
    for (int64_t t = 0; t < 10; ++t) ts.push_back(1000 + t * 200);  // sparse
    std::vector<int64_t> out(ts.size());
    execute_throttle_check(st, ts.data(), static_cast<int64_t>(ts.size()),
                           out.data());
    for (size_t i = 0; i < ts.size(); ++i) {
        int64_t live = 0;
        for (size_t j = 0; j <= i; ++j) live += (ts[j] >= ts[i] - 100) ? 1 : 0;
        EXPECT_EQ(out[i], live <= 2 ? 1 : 0) << "row " << i;
    }
}

// Dimensions and centroid counts past the old 4,096 stack arrays.
TEST(LoudKmeans, AssignHasNoDimOrKCeiling) {
    constexpr size_t D = 5000, N = 3, K = 1500;
    std::vector<float> slab(D * N), cents(K * D);
    for (size_t i = 0; i < slab.size(); ++i)
        slab[i] = static_cast<float>((i * 7919u) % 97u) * 0.01f;
    for (size_t i = 0; i < cents.size(); ++i)
        cents[i] = static_cast<float>((i * 104729u) % 89u) * 0.01f;
    std::vector<uint32_t> got(N);
    bolt::kmeans_assign_f32_l2(slab.data(), D, N, N, cents.data(), K,
                               got.data());
    for (size_t i = 0; i < N; ++i) {
        double best = std::numeric_limits<double>::max();
        std::vector<double> dist(K);
        for (size_t k = 0; k < K; ++k) {
            double acc = 0;
            for (size_t d = 0; d < D; ++d) {
                const double diff = static_cast<double>(slab[d * N + i]) -
                                    static_cast<double>(cents[k * D + d]);
                acc += diff * diff;
            }
            dist[k] = acc;
            if (acc < best) best = acc;
        }
        ASSERT_LT(got[i], K);
        // float vs double summation may break a near-tie either way.
        EXPECT_LE(dist[got[i]], best * (1.0 + 1e-4)) << "vector " << i;
    }
}

TEST(LoudEbr, ShardCountPastCapIsRefused) {
    auto* e = new bolt::Ebr();
    EXPECT_FALSE(bolt::ebr_init(e, bolt::kEbrMaxShards + 1));
    EXPECT_EQ(e->num_shards, 0u);
    bolt::ebr_destroy(e);
    EXPECT_TRUE(bolt::ebr_init(e, 4));
    bolt::ebr_destroy(e);
    delete e;
}
