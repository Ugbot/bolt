// test_bolt_scheduler.cpp — Basic Scheduler init/shutdown/submit tests.

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

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

// Whether this platform can actually pin a thread.
//
// Not every platform can. Darwin's THREAD_AFFINITY_POLICY is documented as a
// hint, and on arm64 it is not implemented at all -- thread_policy_set
// returns KERN_NOT_SUPPORTED, so bolt_pin_current_thread returns false and
// the scheduler correctly records UINT32_MAX for that worker ("honest
// diagnostics: record the failure", scheduler_worker_entry). There is nothing
// wrong for this test to catch there, and asserting otherwise made it fail
// permanently on every Apple Silicon machine.
//
// Probed on a THROWAWAY thread: pinning the test's own thread would leave it
// pinned for every test that runs afterwards in this binary.
bool platform_can_pin() {
    std::atomic<bool> ok{false};
    std::thread t([&ok] { ok.store(bolt_pin_current_thread(0)); });
    t.join();
    return ok.load();
}

TEST(BoltScheduler, PinnedWorkersReportCpus) {
    bolt::SchedulerConfig cfg;
    cfg.num_workers = 2;
    cfg.pin_workers = true;
    if (!platform_can_pin()) {
        GTEST_SKIP() << "thread pinning is unsupported on this platform "
                        "(Darwin/arm64 does not implement "
                        "THREAD_AFFINITY_POLICY); the scheduler reports the "
                        "sentinel, which is correct";
    }
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

// E2 — adaptive morsel sizing (opt-in feedback). Default path is
// unaffected; `recommended_grain_bytes` returns the static grain with
// zero samples, and grows / shrinks toward the 1-10 ms target band as
// observations accumulate.
TEST(BoltScheduler, AdaptiveGrainStartsAtStatic) {
    bolt::Scheduler s{};
    bolt::SchedulerConfig cfg{};
    cfg.num_workers = 1;
    ASSERT_TRUE(s.init(cfg));
    EXPECT_EQ(s.recommended_grain_bytes(8), s.grain_bytes());
    s.shutdown();
}

TEST(BoltScheduler, AdaptiveGrainShrinksWhenSlow) {
    bolt::Scheduler s{};
    bolt::SchedulerConfig cfg{};
    cfg.num_workers = 1;
    ASSERT_TRUE(s.init(cfg));
    // 256 KB / 8-byte rows = 32 768 rows per morsel. At 100 ns/row that's
    // 3.2 ms — inside the 1-10 ms target band (no adjustment).
    // At 1000 ns/row that's 32 ms — above 10 ms, so recommend halving.
    s.record_morsel_ns_per_row(1000.0);
    uint32_t rec = s.recommended_grain_bytes(8);
    EXPECT_LT(rec, s.grain_bytes());
    EXPECT_GE(rec, s.grain_bytes() / 4u);
    s.shutdown();
}

TEST(BoltScheduler, AdaptiveGrainGrowsWhenFast) {
    bolt::Scheduler s{};
    bolt::SchedulerConfig cfg{};
    cfg.num_workers = 1;
    ASSERT_TRUE(s.init(cfg));
    // 0.1 ns/row → 32 768 rows × 0.1 ns = 3.3 µs per morsel; below 1 ms,
    // so recommend doubling.
    s.record_morsel_ns_per_row(0.1);
    uint32_t rec = s.recommended_grain_bytes(8);
    EXPECT_GT(rec, s.grain_bytes());
    EXPECT_LE(rec, s.grain_bytes() * 4u);
    s.shutdown();
}

TEST(BoltScheduler, AdaptiveGrainIgnoresZeroObservation) {
    bolt::Scheduler s{};
    bolt::SchedulerConfig cfg{};
    cfg.num_workers = 1;
    ASSERT_TRUE(s.init(cfg));
    s.record_morsel_ns_per_row(0.0);           // ignored — guard against NaN / zero
    EXPECT_EQ(s.recommended_grain_bytes(8), s.grain_bytes());
    s.shutdown();
}

}  // namespace
