// W8 — selection vectors, column prune, predicate pushdown, DV inline
// filter, partition prune, parallel scan pool, SIMD dispatch. Tests use
// in-memory fixtures: no Parquet/IO surface needed.

#include "bolt/bolt_arena.h"
#include "bolt/bolt_types.h"
#include "bolt/lakehouse/parallel_scan.h"
#include "bolt/lakehouse/scan_optimizer.h"

using bolt::BoltType;

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace bolt::lakehouse;

// -----------------------------------------------------------------------
// Selection-vector primitives.
// -----------------------------------------------------------------------

TEST(W8Sel, IdentityFillsAscending) {
    uint32_t sel[16] = {0};
    sel_identity(sel, 16);
    for (uint32_t i = 0; i < 16; ++i) ASSERT_EQ(sel[i], i);
}

TEST(W8Sel, MarkFilterDropsTombstones) {
    uint32_t in[8]  = {0,1,2,3,4,5,6,7};
    uint8_t  mark[8] = {0,1,0,1,0,0,1,0};
    uint32_t out[8] = {0};
    const uint32_t n = sel_filter_mark(in, 8, mark, out);
    ASSERT_EQ(n, 5u);
    const uint32_t expected[] = {0,2,4,5,7};
    for (uint32_t i = 0; i < 5; ++i) ASSERT_EQ(out[i], expected[i]);
}

TEST(W8Sel, MarkFilterInPlace) {
    uint32_t buf[6] = {10,11,12,13,14,15};
    uint8_t  mark[6] = {1,0,0,1,1,0};
    const uint32_t n = sel_filter_mark(buf, 6, mark, buf);
    ASSERT_EQ(n, 3u);
    ASSERT_EQ(buf[0], 11u);
    ASSERT_EQ(buf[1], 12u);
    ASSERT_EQ(buf[2], 15u);
}

// -----------------------------------------------------------------------
// Column pruning.
// -----------------------------------------------------------------------

TEST(W8ColPrune, ProjectionKeepsListedColsInSchemaOrder) {
    ReadOptions opts{};
    read_options_init(&opts);
    std::strncpy(opts.projection[0], "a", kLakeMaxColName);
    std::strncpy(opts.projection[1], "c", kLakeMaxColName);
    opts.n_projection = 2;
    const char* names[] = {"a", "b", "c", "d"};
    ColumnPruneResult plan{};
    ASSERT_TRUE(column_prune_plan(&opts, names, 4, &plan));
    ASSERT_FALSE(plan.all_columns);
    ASSERT_EQ(plan.n_keep, 2u);
    ASSERT_EQ(plan.keep[0], 0u);   // "a"
    ASSERT_EQ(plan.keep[1], 2u);   // "c"
}

TEST(W8ColPrune, EmptyProjectionKeepsAll) {
    ReadOptions opts{};
    read_options_init(&opts);
    const char* names[] = {"x", "y"};
    ColumnPruneResult plan{};
    ASSERT_TRUE(column_prune_plan(&opts, names, 2, &plan));
    ASSERT_TRUE(plan.all_columns);
    ASSERT_EQ(plan.n_keep, 2u);
}

// G2FEAT-47-class real-data width: ClickBench `hits` is 105 real columns.
// Before this fix kLakeMaxProjection was 64 and this call returned false
// outright (n_src=105 > 64), even though no projection was requested —
// this is the discriminating repro for the fix.
TEST(W8ColPrune, WideSchema105ColumnsAllColumnsSurvives) {
    static char name_storage[105][kLakeMaxColName];
    static const char* names[105];
    for (uint32_t i = 0; i < 105; ++i) {
        std::snprintf(name_storage[i], kLakeMaxColName, "col_%u", i);
        names[i] = name_storage[i];
    }
    ReadOptions opts{};
    read_options_init(&opts);   // no projection -> "all columns"
    ColumnPruneResult plan{};
    ASSERT_TRUE(column_prune_plan(&opts, names, 105, &plan));
    ASSERT_TRUE(plan.all_columns);
    ASSERT_EQ(plan.n_keep, 105u);
    for (uint32_t i = 0; i < 105; ++i) ASSERT_EQ(plan.keep[i], i);
}

// A 90-column projected SELECT (the documented ClickBench Q30 shape):
// project a strict subset out of a 105-column source schema, still under
// the raised 128 cap.
TEST(W8ColPrune, WideSchema90ColumnProjectionExact) {
    static char name_storage[105][kLakeMaxColName];
    static const char* names[105];
    for (uint32_t i = 0; i < 105; ++i) {
        std::snprintf(name_storage[i], kLakeMaxColName, "col_%u", i);
        names[i] = name_storage[i];
    }
    ReadOptions opts{};
    read_options_init(&opts);
    for (uint32_t i = 0; i < 90; ++i) {
        std::strncpy(opts.projection[i], name_storage[i], kLakeMaxColName);
    }
    opts.n_projection = 90;
    ColumnPruneResult plan{};
    ASSERT_TRUE(column_prune_plan(&opts, names, 105, &plan));
    ASSERT_FALSE(plan.all_columns);
    ASSERT_EQ(plan.n_keep, 90u);
    for (uint32_t i = 0; i < 90; ++i) ASSERT_EQ(plan.keep[i], i);
}

// A width just over the raised cap (129 > 128) must reject loudly, never
// truncate silently.
TEST(W8ColPrune, JustOverRaisedCapRejectsLoudly) {
    static char name_storage[129][kLakeMaxColName];
    static const char* names[129];
    for (uint32_t i = 0; i < 129; ++i) {
        std::snprintf(name_storage[i], kLakeMaxColName, "col_%u", i);
        names[i] = name_storage[i];
    }
    ReadOptions opts{};
    read_options_init(&opts);
    ColumnPruneResult plan{};
    ASSERT_FALSE(column_prune_plan(&opts, names, 129, &plan));
}

// Exactly at the raised cap (128) must succeed.
TEST(W8ColPrune, ExactlyAtRaisedCapSucceeds) {
    static char name_storage[128][kLakeMaxColName];
    static const char* names[128];
    for (uint32_t i = 0; i < 128; ++i) {
        std::snprintf(name_storage[i], kLakeMaxColName, "col_%u", i);
        names[i] = name_storage[i];
    }
    ReadOptions opts{};
    read_options_init(&opts);
    ColumnPruneResult plan{};
    ASSERT_TRUE(column_prune_plan(&opts, names, 128, &plan));
    ASSERT_EQ(plan.n_keep, 128u);
}

// -----------------------------------------------------------------------
// Predicate pushdown — stats level.
// -----------------------------------------------------------------------

TEST(W8Pushdown, StatsSkipWhenRangeBelowEqValue) {
    RowGroupStats s{};
    s.has_minmax = true;
    s.n_cols = 1;
    s.min_i64[0] = 100;
    s.max_i64[0] = 200;
    s.num_rows = 1000;
    Predicate p{};
    std::strncpy(p.column, "x", kLakeMaxColName);
    p.op = PredicateOp::kEq;
    p.value.type = BoltType::Int64;
    p.value.i64  = 50;          // outside [100,200]
    const char* names[] = {"x"};
    ASSERT_EQ(predicate_pushdown_stats(&s, names, &p, 1),
              PushdownVerdict::kSkip);
}

TEST(W8Pushdown, StatsAllPassWhenSingletonEq) {
    RowGroupStats s{};
    s.has_minmax = true;
    s.n_cols = 1;
    s.min_i64[0] = 7;
    s.max_i64[0] = 7;
    s.num_rows = 1;
    Predicate p{};
    std::strncpy(p.column, "x", kLakeMaxColName);
    p.op = PredicateOp::kEq;
    p.value.type = BoltType::Int64;
    p.value.i64  = 7;
    const char* names[] = {"x"};
    ASSERT_EQ(predicate_pushdown_stats(&s, names, &p, 1),
              PushdownVerdict::kAllPass);
}

TEST(W8Pushdown, StatsIndeterminateForOverlap) {
    RowGroupStats s{};
    s.has_minmax = true;
    s.n_cols = 1;
    s.min_i64[0] = 0;
    s.max_i64[0] = 100;
    Predicate p{};
    std::strncpy(p.column, "x", kLakeMaxColName);
    p.op = PredicateOp::kLt;
    p.value.type = BoltType::Int64;
    p.value.i64  = 50;
    const char* names[] = {"x"};
    ASSERT_EQ(predicate_pushdown_stats(&s, names, &p, 1),
              PushdownVerdict::kIndeterminate);
}

// -----------------------------------------------------------------------
// Partition prune helper (centralised).
// -----------------------------------------------------------------------

TEST(W8PartPrune, KeepsOnlyMatchingFiles) {
    PartitionEntry parts_a[1] = {{"region", "us", false, {0}}};
    PartitionEntry parts_b[1] = {{"region", "eu", false, {0}}};
    PartitionEntry parts_c[1] = {{"region", "us", false, {0}}};
    PartitionedFile files[3];
    files[0] = {parts_a, 1, 0};
    files[1] = {parts_b, 1, 0};
    files[2] = {parts_c, 1, 0};
    Predicate p{};
    std::strncpy(p.column, "region", kLakeMaxColName);
    p.op = PredicateOp::kEq;
    p.value.type = BoltType::Utf8;
    std::strncpy(p.value.str, "us", kLakeMaxValBytes);
    p.value.str_len = 2;
    uint32_t keep[4] = {0};
    uint32_t n = 4;
    ASSERT_TRUE(partition_prune(files, 3, &p, 1, keep, &n));
    ASSERT_EQ(n, 2u);
    ASSERT_EQ(keep[0], 0u);
    ASSERT_EQ(keep[1], 2u);
}

// -----------------------------------------------------------------------
// SIMD scan dispatch.
// -----------------------------------------------------------------------

// Defined in simd_scan.cpp.
namespace bolt { namespace lakehouse {
uint32_t simd_scan_filter(
    const void* data, uint32_t n, BoltType data_type,
    PredicateOp op, int64_t scalar_i64, double scalar_f64,
    uint32_t* out_sel) noexcept;
}}

TEST(W8Simd, Int64GreaterThan) {
    int64_t data[8] = {1, 5, 7, 2, 9, 3, 8, 4};
    uint32_t sel[8] = {0};
    const uint32_t n = bolt::lakehouse::simd_scan_filter(
        data, 8, BoltType::Int64, PredicateOp::kGt, 4, 0.0, sel);
    ASSERT_EQ(n, 4u);  // 5,7,9,8
    const uint32_t expected[] = {1, 2, 4, 6};
    for (uint32_t i = 0; i < 4; ++i) ASSERT_EQ(sel[i], expected[i]);
}

TEST(W8Simd, Float64Eq) {
    double data[4] = {1.5, 2.5, 1.5, 3.5};
    uint32_t sel[4] = {0};
    const uint32_t n = bolt::lakehouse::simd_scan_filter(
        data, 4, BoltType::Float64, PredicateOp::kEq, 0, 1.5, sel);
    ASSERT_EQ(n, 2u);
    ASSERT_EQ(sel[0], 0u);
    ASSERT_EQ(sel[1], 2u);
}

TEST(W8Simd, UnsupportedTypeFallsBackToIdentity) {
    uint8_t data[4] = {0,0,0,0};
    uint32_t sel[4] = {0};
    const uint32_t n = bolt::lakehouse::simd_scan_filter(
        data, 4, BoltType::Utf8, PredicateOp::kEq, 0, 0.0, sel);
    ASSERT_EQ(n, 4u);
    for (uint32_t i = 0; i < 4; ++i) ASSERT_EQ(sel[i], i);
}

// -----------------------------------------------------------------------
// Parallel-scan pool — config clamp + submit/poll lifecycle.
// -----------------------------------------------------------------------

TEST(W8Parallel, ConfigClampsToCaps) {
    ParallelScanConfig cfg{};
    cfg.parallelism = 1024;
    cfg.lookahead   = 1024;
    cfg.inflight_max = 1024;
    cfg.use_prefetch = false;
    ASSERT_TRUE(parallel_scan_config_clamp(&cfg));
    ASSERT_EQ(cfg.parallelism, kScanMaxParallelism);
    ASSERT_EQ(cfg.lookahead,   kScanMaxLookahead);
    ASSERT_EQ(cfg.inflight_max, kScanMaxInflight);
}

TEST(W8Parallel, OpenSubmitPollClose) {
    bolt::Arena arena;
    ParallelScanConfig cfg{};
    parallel_scan_config_init(&cfg);
    cfg.use_prefetch = false;
    cfg.parallelism  = 4;
    ParallelScanPool* pool = nullptr;
    ASSERT_TRUE(parallel_scan_open(&pool, &arena, nullptr, &cfg));
    ASSERT_NE(pool, nullptr);
    // Submit 16 tasks.
    for (uint32_t i = 0; i < 16; ++i) {
        RowGroupTask t{};
        t.file_path = "test/file.parquet";
        t.row_group = i;
        t.slot_id   = i;
        ASSERT_TRUE(parallel_scan_submit(pool, &t));
    }
    parallel_scan_finish(pool);
    // Drain morsels (the pool produces one empty Morsel per task today;
    // real decode wiring is a follow-up — the lifecycle + counters are
    // what's under test here).
    uint32_t got = 0;
    bool eof = false;
    while (!eof && got < 16) {
        Morsel m{};
        if (parallel_scan_poll(pool, &m, 1000, &eof)) ++got;
    }
    ASSERT_EQ(got, 16u);
    ParallelScanStats stats{};
    parallel_scan_stats(pool, &stats);
    ASSERT_EQ(stats.tasks_submitted, 16u);
    ASSERT_EQ(stats.morsels_produced, 16u);
    parallel_scan_close(pool);
    arena.reset();
}
