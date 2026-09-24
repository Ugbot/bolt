// test_bolt_scheduler_multi_producer.cpp — G2ICE-189: bolt::Scheduler must run
// every task exactly once when several threads submit at the same time.
//
// Real callers do this: marbledb's maintenance pool had three producers
// (G2ICE-187) and chukonu's shared warm pool is borrowed by concurrent
// top-level operators. Against the single-producer ring two producers could
// store one slot (a task silently lost) or move `head` backwards, after which
// every later submit wedges in submit_wait. Both show up here as a census
// miss or as the watchdog firing; the watchdog exits the process because a
// wedged producer cannot be joined.

#include <bolt/bolt_scheduler.h>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kProducers        = 4;
constexpr std::uint32_t kPerProducer      = 50000;
constexpr std::uint32_t kTotal            = kProducers * kPerProducer;
constexpr std::uint32_t kWorkers          = 4;
constexpr std::uint32_t kWatchdogSeconds  = 60;

std::atomic<std::uint32_t> g_runs[kTotal];
std::atomic<std::uint64_t> g_executed{0};

void census_task(void* arg) noexcept {
    const std::uintptr_t idx = reinterpret_cast<std::uintptr_t>(arg);
    assert(idx < kTotal);
    g_runs[idx].fetch_add(1, std::memory_order_relaxed);
    g_executed.fetch_add(1, std::memory_order_release);
}

struct RangeCtx {
    std::uint32_t base;
};

void census_range(void* user, std::uint32_t start, std::uint32_t end,
                  std::uint32_t) noexcept {
    const RangeCtx* c = static_cast<const RangeCtx*>(user);
    assert(c != nullptr);
    assert(end > start);
    for (std::uint32_t i = start; i < end; ++i) {
        census_task(reinterpret_cast<void*>(
            static_cast<std::uintptr_t>(c->base + i)));
    }
}

void reset_census() {
    for (std::uint32_t i = 0; i < kTotal; ++i) {
        g_runs[i].store(0, std::memory_order_relaxed);
    }
    g_executed.store(0, std::memory_order_relaxed);
}

// Waits for kTotal executions; on timeout reports and exits (a wedged producer
// cannot be joined, so returning would hang the test binary instead).
void wait_or_die(const char* what) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(kWatchdogSeconds);
    while (g_executed.load(std::memory_order_acquire) < kTotal) {
        if (std::chrono::steady_clock::now() > deadline) {
            std::fprintf(stderr,
                "[G2ICE-189] %s: WEDGED — %llu of %u tasks executed after %us\n",
                what,
                static_cast<unsigned long long>(g_executed.load()),
                kTotal, kWatchdogSeconds);
            std::fflush(stderr);
            std::_Exit(3);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void expect_exactly_once(const char* what) {
    std::uint32_t never = 0, dup = 0;
    for (std::uint32_t i = 0; i < kTotal; ++i) {
        const std::uint32_t k = g_runs[i].load(std::memory_order_relaxed);
        if (k == 0) ++never;
        else if (k > 1) ++dup;
    }
    EXPECT_EQ(never, 0u) << what << ": tasks never run";
    EXPECT_EQ(dup, 0u) << what << ": tasks run more than once";
}

TEST(BoltSchedulerMultiProducer, ConcurrentSubmitRunsEveryTaskOnce) {
    static bolt::Scheduler sched;
    ASSERT_TRUE(sched.init(kWorkers));
    reset_census();

    std::atomic<bool> go{false};
    std::vector<std::thread> producers;
    for (std::uint32_t p = 0; p < kProducers; ++p) {
        producers.emplace_back([p, &go] {
            while (!go.load(std::memory_order_acquire)) bolt::cpu_pause();
            for (std::uint32_t i = 0; i < kPerProducer; ++i) {
                const std::uintptr_t idx = std::uintptr_t{p} * kPerProducer + i;
                sched.submit(&census_task, reinterpret_cast<void*>(idx));
            }
        });
    }
    go.store(true, std::memory_order_release);
    wait_or_die("submit");
    for (auto& t : producers) t.join();
    // No wait_all(): it counts completions only for range/column tasks.
    expect_exactly_once("submit");
    EXPECT_EQ(sched.stats.tasks_submitted.load(), std::uint64_t{kTotal});
    sched.shutdown();
}

TEST(BoltSchedulerMultiProducer, ConcurrentSubmitRangeRunsEveryTaskOnce) {
    static bolt::Scheduler sched;
    ASSERT_TRUE(sched.init(kWorkers));
    reset_census();

    RangeCtx ctx[kProducers];
    for (std::uint32_t p = 0; p < kProducers; ++p) ctx[p].base = p * kPerProducer;

    std::atomic<bool> go{false};
    std::vector<std::thread> producers;
    for (std::uint32_t p = 0; p < kProducers; ++p) {
        producers.emplace_back([p, &go, &ctx] {
            while (!go.load(std::memory_order_acquire)) bolt::cpu_pause();
            // grain 1: one pool acquire + one ring slot per index, so the
            // payload pool's pop path is contended as hard as the ring.
            sched.submit_range(&census_range, &ctx[p], kPerProducer, 1u);
        });
    }
    go.store(true, std::memory_order_release);
    wait_or_die("submit_range");
    for (auto& t : producers) t.join();
    sched.wait_all();
    expect_exactly_once("submit_range");
    sched.shutdown();
}

}  // namespace
