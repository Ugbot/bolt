// test_bolt_scheduler.cpp — Basic Scheduler init/shutdown/submit tests.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_scheduler.h"

namespace {

TEST(BoltScheduler, InitShutdownCycle) {
    bolt::Scheduler sched{};
    ASSERT_TRUE(sched.init(4, bolt::SpinPolicy::SpinYield));
    EXPECT_EQ(sched.thread_count(), 4u);
    sched.shutdown();
    EXPECT_EQ(sched.thread_count(), 0u);
}

struct RangeSumCtx {
    std::atomic<uint64_t> sum;
};

static void range_sum_fn(void* user_data, uint32_t start, uint32_t end,
                         uint32_t /*thread_id*/) noexcept {
    auto* ctx = static_cast<RangeSumCtx*>(user_data);
    uint64_t local = 0;
    for (uint32_t i = start; i < end; ++i) local += i;
    ctx->sum.fetch_add(local, std::memory_order_relaxed);
}

TEST(BoltScheduler, SubmitRangeSumsIndices) {
    bolt::Scheduler sched{};
    ASSERT_TRUE(sched.init(4));

    constexpr uint32_t kCount = 10000;
    RangeSumCtx ctx;
    ctx.sum.store(0, std::memory_order_relaxed);

    sched.submit_range(range_sum_fn, &ctx, kCount, /*grain=*/512);
    sched.wait_all();

    const uint64_t expected = static_cast<uint64_t>(kCount) * (kCount - 1) / 2;
    EXPECT_EQ(ctx.sum.load(std::memory_order_acquire), expected);

    sched.shutdown();
}

TEST(BoltScheduler, ReinitAfterShutdown) {
    bolt::Scheduler sched{};
    ASSERT_TRUE(sched.init(2));
    sched.shutdown();

    ASSERT_TRUE(sched.init(3));
    EXPECT_EQ(sched.thread_count(), 3u);

    RangeSumCtx ctx;
    ctx.sum.store(0, std::memory_order_relaxed);
    sched.submit_range(range_sum_fn, &ctx, 1000, 128);
    sched.wait_all();
    EXPECT_EQ(ctx.sum.load(std::memory_order_acquire),
              static_cast<uint64_t>(1000) * 999 / 2);

    sched.shutdown();
}

TEST(BoltScheduler, InitWithConfigBasic) {
    bolt::SchedulerConfig cfg;
    cfg.num_workers = 2;
    bolt::Scheduler s{};
    ASSERT_TRUE(s.init(cfg));
    EXPECT_EQ(s.thread_count(), 2u);
    s.shutdown();
}

TEST(BoltScheduler, WithProfileLatencyPresets) {
    auto cfg = bolt::SchedulerConfig::with_profile(bolt::TuneProfile::Latency);
    EXPECT_EQ(cfg.spin, bolt::SpinPolicy::BusySpin);
    EXPECT_TRUE(cfg.pin_workers);
    EXPECT_EQ(cfg.grain_bytes, 64u * 1024u);
}

TEST(BoltScheduler, WithProfileThroughputPresets) {
    auto cfg = bolt::SchedulerConfig::with_profile(bolt::TuneProfile::Throughput);
    EXPECT_EQ(cfg.grain_bytes, 1024u * 1024u);
    EXPECT_EQ(cfg.dispatch_batch, 4u);
    EXPECT_FALSE(cfg.prefer_p_cores);
}

TEST(BoltScheduler, PinnedWorkersReportCpus) {
    bolt::SchedulerConfig cfg;
    cfg.num_workers = 2;
    cfg.pin_workers = true;
    bolt::Scheduler s{};
    ASSERT_TRUE(s.init(cfg));

    // Give the worker threads a moment to apply pinning before reading.
    // Pinning is set by the worker as its first action before entering the
    // loop; submit a trivial task and wait_all() to synchronise.
    auto noop = [](void*, uint32_t, uint32_t, uint32_t) noexcept {};
    uint32_t dummy = 0;
    s.submit_range(noop, &dummy, 1, 1);
    s.wait_all();

    EXPECT_NE(s.worker_cpu(0), UINT32_MAX);
    if (::bolt_get_hardware_concurrency() >= 2) {
        EXPECT_NE(s.worker_cpu(0), s.worker_cpu(1));
    }
    s.shutdown();
}

TEST(BoltScheduler, UnpinnedReportsSentinel) {
    bolt::SchedulerConfig cfg;
    cfg.num_workers = 3;
    cfg.pin_workers = false;
    bolt::Scheduler s{};
    ASSERT_TRUE(s.init(cfg));
    for (uint32_t i = 0; i < s.thread_count(); ++i) {
        EXPECT_EQ(s.worker_cpu(i), UINT32_MAX);
    }
    s.shutdown();
}

TEST(BoltScheduler, GrainRowsHelper) {
    EXPECT_EQ(bolt::bolt_grain_rows(256u * 1024u, 4),   65536u);
    EXPECT_EQ(bolt::bolt_grain_rows(256u * 1024u, 8),   32768u);
    EXPECT_EQ(bolt::bolt_grain_rows(256u * 1024u, 128), 2048u);
}

}  // namespace
