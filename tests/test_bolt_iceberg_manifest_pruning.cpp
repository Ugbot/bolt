// test_bolt_iceberg_manifest_pruning.cpp — G2ICE-234.
//
// golden_iceberg_day_table/ was written by pyiceberg 0.10.0
// (data/make_iceberg_day_table.py): day(ts) + identity(tenant), two appends.
//     append #1: 2024-03-01 t7 (2 rows), 2024-03-02 t8 (1 row)
//     append #2: 2024-05-09 t9 (1 row),  2024-05-10 t7 (1 row)
//
// Two defects this pins:
//  1. pyiceberg declares the day value as ["null",{"type":"int",
//     "logicalType":"date"}]; the Avro flattener split that union into two
//     fields, so every partition value after it was misread and identity
//     pruning dropped live files (tenant == 9 planned ZERO files).
//  2. the manifest list's partition summaries were never read, so every
//     selective plan decoded every manifest: O(table files).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/lakehouse/format.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/partition.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/object_store.h"

namespace {

using namespace bolt;
using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

constexpr int64_t kMar01 = INT64_C(1709251200000000);   // 2024-03-01T00:00Z
constexpr int64_t kDayUs = INT64_C(86400000000);

Predicate pred(const char* col, PredicateOp op, int64_t v) {
    Predicate p;
    std::memset(&p, 0, sizeof(p));
    std::strncpy(p.column, col, kLakeMaxColName - 1);
    p.op = op;
    p.value.type = BoltType::Int64;
    p.value.i64 = v;
    return p;
}

struct Plan {
    std::vector<std::string> files;
    IcebergScanStats         st{};
    int64_t                  rows = -1;
};

Plan plan(std::vector<Predicate> preds, bool read_rows = false) {
    Plan out;
    Arena arena;
    FilesystemObjectStore fs;
    ObjectStore os;
    const std::string root =
        std::string(BOLT_TEST_DATA_DIR) + "/golden_iceberg_day_table/db";
    EXPECT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));
    TableHandle* th = nullptr;
    EXPECT_TRUE(iceberg_table_open_store(&th, &arena, &os, "ev"));
    if (th == nullptr) return out;
    ReadOptions ro;
    read_options_init(&ro);
    for (const Predicate& p : preds) ro.predicates[ro.n_predicates++] = p;
    ScanHandle* sh = nullptr;
    EXPECT_TRUE(iceberg_scan_open(&sh, th, &ro));
    if (sh == nullptr) return out;
    if (read_rows) {
        out.rows = 0;
        for (int guard = 0; guard < 1000; ++guard) {
            BoltBatch b;
            bool eof = false;
            EXPECT_TRUE(iceberg_scan_next_batch(sh, &b, &eof));
            if (eof) break;
            out.rows += b.num_rows;
        }
    } else {
        for (int guard = 0; guard < 1000; ++guard) {
            const char* path = nullptr;
            bool eof = false;
            EXPECT_TRUE(iceberg_scan_next_file(sh, &path, &eof));
            if (eof || path == nullptr) break;
            out.files.emplace_back(path);
        }
    }
    iceberg_scan_stats(sh, &out.st);
    iceberg_scan_close(sh);
    return out;
}

bool has(const Plan& p, const char* frag) {
    for (const auto& f : p.files)
        if (f.find(frag) != std::string::npos) return true;
    return false;
}

TEST(IcebergManifestPruning, FullPlanSeesEveryFile) {
    const Plan p = plan({});
    EXPECT_EQ(p.files.size(), 4u);
    EXPECT_EQ(p.st.manifests_listed, 2u);
    EXPECT_EQ(p.st.manifests_read, 2u);
    EXPECT_EQ(p.st.manifests_pruned, 0u);
    EXPECT_GT(p.st.metadata_bytes_read, 0u);
}

// Defect 1: the partition value AFTER a logical-type union was misread.
TEST(IcebergManifestPruning, IdentityPartitionAfterDateUnionIsRead) {
    const Plan p = plan({pred("tenant", PredicateOp::kEq, 9)});
    ASSERT_EQ(p.files.size(), 1u);
    EXPECT_TRUE(has(p, "ts_day=2024-05-09/tenant=9/"));
    const Plan q = plan({pred("tenant", PredicateOp::kEq, 7)});
    EXPECT_EQ(q.files.size(), 2u);
    EXPECT_TRUE(has(q, "ts_day=2024-03-01/tenant=7/"));
    EXPECT_TRUE(has(q, "ts_day=2024-05-10/tenant=7/"));
    EXPECT_EQ(plan({pred("tenant", PredicateOp::kEq, 7)}, true).rows, 3);
}

// Defect 2: a one-day predicate reads one manifest, not both.
TEST(IcebergManifestPruning, DayRangeSkipsManifestOnSummaries) {
    const Plan p = plan({pred("ts", PredicateOp::kGe, kMar01),
                         pred("ts", PredicateOp::kLt, kMar01 + kDayUs)});
    ASSERT_EQ(p.files.size(), 1u);
    EXPECT_TRUE(has(p, "ts_day=2024-03-01/tenant=7/"));
    EXPECT_EQ(p.st.manifests_pruned, 1u);
    EXPECT_EQ(p.st.manifests_read, 1u);
    EXPECT_EQ(p.st.manifests_read + p.st.manifests_pruned,
              p.st.manifests_listed);
}

// Pruning must never drop a live file: every boundary of the day transform.
TEST(IcebergManifestPruning, SummaryBoundariesKeepEveryLiveFile) {
    const int64_t may10_end = kMar01 + INT64_C(71) * kDayUs;  // 2024-05-11
    // ts < first instant of 05-11 still includes 05-10.
    EXPECT_TRUE(has(plan({pred("ts", PredicateOp::kLt, may10_end)}),
                    "ts_day=2024-05-10/"));
    // ts > last micro of 03-01 excludes nothing from 03-02 on.
    const Plan g = plan({pred("ts", PredicateOp::kGt, kMar01 + kDayUs - 1)});
    EXPECT_TRUE(has(g, "ts_day=2024-03-02/"));
    EXPECT_FALSE(has(g, "ts_day=2024-03-01/"));
    // tenant 9 lives only in append #2; append #1's tenant range is [7,8].
    const Plan t9 = plan({pred("tenant", PredicateOp::kEq, 9)});
    ASSERT_EQ(t9.files.size(), 1u);
    EXPECT_EQ(t9.st.manifests_pruned, 1u);
    // tenant 8 is inside BOTH ranges, so neither manifest may be skipped.
    const Plan t8 = plan({pred("tenant", PredicateOp::kEq, 8)});
    ASSERT_EQ(t8.files.size(), 1u);
    EXPECT_EQ(t8.st.manifests_pruned, 0u);
    // Nothing matches: both manifests pruned, no file planned.
    const Plan none = plan({pred("tenant", PredicateOp::kGt, 9)});
    EXPECT_TRUE(none.files.empty());
    EXPECT_EQ(none.st.manifests_read, 0u);
}

TEST(IcebergManifestPruning, MissingSummariesNeverPrune) {
    ManifestListEntry e;
    std::memset(&e, 0, sizeof(e));
    PartitionSpec spec;
    std::memset(&spec, 0, sizeof(spec));
    spec.n_fields = 1;
    spec.fields[0].source_id = 1;
    spec.fields[0].transform.kind = TransformKind::kDay;
    SchemaField f;
    std::memset(&f, 0, sizeof(f));
    f.id = 1;
    std::strcpy(f.name, "ts");
    std::strcpy(f.type, "timestamp");
    Schema sch;
    std::memset(&sch, 0, sizeof(sch));
    sch.n_fields = 1;
    sch.fields = &f;
    const Predicate p = pred("ts", PredicateOp::kEq, -1);
    EXPECT_TRUE(manifest_may_match(&e, &spec, &sch, &p, 1));   // no summary
    e.n_partitions = 1;
    e.partitions[0].has_lower = e.partitions[0].has_upper = true;
    const int32_t lo = -1, hi = -1;                            // 1969-12-31
    std::memcpy(e.partitions[0].lower, &lo, 4);
    std::memcpy(e.partitions[0].upper, &hi, 4);
    e.partitions[0].lower_len = e.partitions[0].upper_len = 4;
    // -1 us is 1969-12-31: floor division, not truncation toward zero.
    EXPECT_TRUE(manifest_may_match(&e, &spec, &sch, &p, 1));
    const Predicate q = pred("ts", PredicateOp::kEq, 0);
    EXPECT_FALSE(manifest_may_match(&e, &spec, &sch, &q, 1));
    e.partitions[0].has_upper = false;                         // half a bound
    EXPECT_TRUE(manifest_may_match(&e, &spec, &sch, &q, 1));
}

}  // namespace
