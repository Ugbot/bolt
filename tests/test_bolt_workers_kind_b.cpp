// L8: worker count and topology are kind B — sized at init from runtime
// config, validated against a generous ceiling, refused (never clamped) past it.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "bolt/bolt_budget.h"
#include "bolt/bolt_ebr.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/bolt_topology.h"

namespace {

struct TidSeen {
    std::atomic<uint32_t> hits[bolt::kMaxWorkers];
    std::atomic<uint32_t> bad;
};

void mark_tid(void* user, uint32_t start, uint32_t end, uint32_t tid) noexcept {
    auto* s = static_cast<TidSeen*>(user);
    if (tid >= bolt::kMaxWorkers) { s->bad.fetch_add(1); return; }
    s->hits[tid].fetch_add(end - start);
}

}  // namespace

TEST(WorkersKindB, CeilingIsGenerous) {
    static_assert(bolt::kMaxWorkers >= 1024u, "workers ceiling");
    static_assert(bolt::kEbrMaxShards == bolt::kMaxWorkers, "EBR derives");
    EXPECT_GE(bolt::kTopologyMaxCpus, 4096u);
}

// 96 workers used to be silently clamped to 64.
TEST(WorkersKindB, NinetySixWorkersAllRun) {
    auto* sched = new bolt::Scheduler();
    bolt::SchedulerConfig cfg{};
    cfg.num_workers = 96;
    ASSERT_TRUE(sched->init(cfg));
    EXPECT_EQ(sched->thread_count(), 96u);
    auto* seen = new TidSeen();
    for (auto& h : seen->hits) h.store(0);
    seen->bad.store(0);
    constexpr uint32_t kRows = 96u * 4096u;
    sched->submit_range(&mark_tid, seen, kRows, 64);
    sched->wait_all();
    uint64_t total = 0;
    for (uint32_t i = 0; i < bolt::kMaxWorkers; ++i) {
        if (i >= 96u) EXPECT_EQ(seen->hits[i].load(), 0u);
        total += seen->hits[i].load();
    }
    EXPECT_EQ(total, kRows);
    EXPECT_EQ(seen->bad.load(), 0u);
    sched->shutdown();
    delete seen;
    delete sched;
}

TEST(WorkersKindB, ExplicitCountPastCeilingIsRefused) {
    auto* sched = new bolt::Scheduler();
    EXPECT_FALSE(sched->init(bolt::kMaxWorkers + 1u));
    EXPECT_EQ(sched->thread_count(), 0u);
    sched->shutdown();   // safe after a refused init
    delete sched;
}

TEST(WorkersKindB, ShutdownThenReinit) {
    auto* sched = new bolt::Scheduler();
    ASSERT_TRUE(sched->init(3u));
    sched->shutdown();
    ASSERT_TRUE(sched->init(70u));
    EXPECT_EQ(sched->thread_count(), 70u);
    sched->shutdown();
    delete sched;
}

// BOLT_WORKERS resolves once per process, so each case runs in a fresh child.
TEST(WorkersKindBDeathTest, InvalidBoltWorkersIsAStartupError) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    ::setenv("BOLT_WORKERS", "2000", 1);
    EXPECT_EXIT({
        bolt::Scheduler* s = new bolt::Scheduler();
        const bool ok = s->init(0u);
        std::exit(ok ? 0 : 3);
    }, ::testing::ExitedWithCode(3), "BOLT_WORKERS=2000 is invalid");
    ::unsetenv("BOLT_WORKERS");
}

TEST(WorkersKindBDeathTest, BoltWorkersSizesAutoPools) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    ::setenv("BOLT_WORKERS", "5", 1);
    EXPECT_EXIT({
        bolt::Scheduler* s = new bolt::Scheduler();
        const bool ok = s->init(0u) && s->thread_count() == 5u &&
                        bolt::bolt_auto_workers() == 5u;
        s->shutdown();
        std::exit(ok ? 7 : 1);
    }, ::testing::ExitedWithCode(7), "");
    ::unsetenv("BOLT_WORKERS");
}

TEST(WorkersKindB, EbrShardsAreSizedAtInit) {
    auto* e = new bolt::Ebr();
    ASSERT_TRUE(bolt::ebr_init(e, 96u));
    EXPECT_EQ(e->num_shards, 96u);
    int* obj = new int(7);
    bolt::ebr_enter(e, 95u);
    ASSERT_TRUE(bolt::ebr_retire(e, 95u, obj, [](void* p) { delete static_cast<int*>(p); }));
    bolt::ebr_exit(e, 95u);
    bolt::ebr_destroy(e);
    EXPECT_EQ(e->shards, nullptr);
    EXPECT_FALSE(bolt::ebr_init(e, bolt::kEbrMaxShards + 1u));
    bolt::ebr_destroy(e);
    delete e;
}

TEST(WorkersKindB, TopologyMapsThisMachine) {
    bolt::CpuTopology t{};
    ASSERT_TRUE(bolt::bolt_detect_topology(&t));
    EXPECT_GE(t.logical_cpus, 1u);
    EXPECT_EQ(t.unmapped_cpus, 0u);
    EXPECT_LE(sizeof(bolt::CpuTopology), 16u * 1024u);
}
