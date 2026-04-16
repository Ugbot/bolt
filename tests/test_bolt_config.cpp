// test_bolt_config.cpp — verifies bolt_config.h override mechanism.
//
// NOTE ABOUT SCOPE: we only override values that do NOT affect struct
// layout. Overriding `BOLT_MAX_WORKERS`, `BOLT_TASK_RING_SIZE`, or
// `BOLT_SCHEDULER_POOL_CAPACITY` per-TU is an ODR violation because
// those constants size arrays inside `bolt::Scheduler` / `bolt::TaskRing`.
// Layout-affecting overrides must be applied project-wide — via a
// compiler flag (`-DBOLT_MAX_WORKERS=128`) or a shared
// `bolt/bolt_user_config.h`. This test deliberately avoids them.
//
// What we DO test per-TU:
//   - grain_bytes presets (read as values, not array sizes).
//   - dispatch_batch presets.
//   - parallel break-even thresholds (checked at kernel entry).

#define BOLT_LATENCY_GRAIN_BYTES       (16u * 1024u)
#define BOLT_BALANCED_GRAIN_BYTES      (128u * 1024u)
#define BOLT_THROUGHPUT_GRAIN_BYTES    (2u * 1024u * 1024u)
#define BOLT_THROUGHPUT_DISPATCH_BATCH 8u
#define BOLT_PARALLEL_MIN_ROWS_FILTER  1
#define BOLT_PARALLEL_MIN_ROWS_SUM     1
#define BOLT_PARALLEL_MIN_ROWS_GROUPBY 1

#include <gtest/gtest.h>

#include "bolt/bolt.h"

TEST(BoltConfig, GrainByteOverrides) {
    EXPECT_EQ(bolt::config::kLatencyGrainBytes,    16u * 1024u);
    EXPECT_EQ(bolt::config::kBalancedGrainBytes,   128u * 1024u);
    EXPECT_EQ(bolt::config::kThroughputGrainBytes, 2u * 1024u * 1024u);
}

TEST(BoltConfig, DispatchBatchOverride) {
    EXPECT_EQ(bolt::config::kThroughputDispatchBatch, 8u);
}

TEST(BoltConfig, ParallelThresholdsStreaming) {
    EXPECT_EQ(bolt::config::kParallelMinRowsFilter,  1);
    EXPECT_EQ(bolt::config::kParallelMinRowsSum,     1);
    EXPECT_EQ(bolt::config::kParallelMinRowsGroupby, 1);
}

TEST(BoltConfig, LatencyProfilePicksOverride) {
    auto c = bolt::SchedulerConfig::with_profile(bolt::TuneProfile::Latency);
    EXPECT_EQ(c.grain_bytes, 16u * 1024u);
}

TEST(BoltConfig, ThroughputProfilePicksOverride) {
    auto c = bolt::SchedulerConfig::with_profile(bolt::TuneProfile::Throughput);
    EXPECT_EQ(c.grain_bytes,    2u * 1024u * 1024u);
    EXPECT_EQ(c.dispatch_batch, 8u);
}

// Streaming-shape sanity: with MIN_ROWS_FILTER=1 the parallel wrapper
// dispatches even on tiny inputs. Result must still be correct.
TEST(BoltConfig, StreamingThresholdsAllowParallelOnSmallBatches) {
    const uint32_t hw = ::bolt_get_hardware_concurrency();
    if (hw < 2) { GTEST_SKIP() << "single-core host"; }

    bolt::Scheduler       sched;
    bolt::SchedulerConfig cfg;
    cfg.num_workers = 2;
    ASSERT_TRUE(sched.init(cfg));

    bolt::Arena arena;
    constexpr int N = 1024;
    int32_t data[N];
    int32_t out_par[N], out_ser[N];
    for (int i = 0; i < N; ++i) data[i] = i;

    int64_t c_par = bolt::kernels::parallel_filter_gt<int32_t>(
        &sched, data, N, 500, out_par, &arena);
    int64_t c_ser = bolt::kernels::filter_gt<int32_t>(data, N, 500, out_ser);

    EXPECT_EQ(c_par, c_ser);
    for (int64_t i = 0; i < c_par; ++i) EXPECT_EQ(out_par[i], out_ser[i]);

    sched.shutdown();
}

// Baseline is reachable without overrides: sanity-check that the default
// constants are still exposed (prevents accidental removal).
TEST(BoltConfig, DefaultsReachable) {
    static_assert(bolt::config::kMaxWorkers > 0);
    static_assert(bolt::config::kTaskRingSize > 0);
    static_assert(bolt::config::kSchedulerPoolCapacity > 0);
    SUCCEED();
}
