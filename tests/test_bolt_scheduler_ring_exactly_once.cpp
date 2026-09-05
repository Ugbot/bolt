// test_bolt_scheduler_ring_exactly_once.cpp — the exactly-once contract of
// bolt::Scheduler's SPMC TaskRing, pinned.
//
// WHY THIS EXISTS
// ---------------
// The ring promises that every submitted task runs exactly once. Every caller
// that fans per-item work out over workers depends on it, and chukonu's
// morsel-parallel hash-join probe replay depends on it *silently*: a morsel
// that is never probed contributes no rows, a morsel probed twice contributes
// its rows twice, and the operator has no way to notice either.
//
// THE MEASURED DEFECT (W4-L1). `try_claim_and_execute` used to bump `tail`
// (the claim) and only THEN read `ring[t & kTaskRingMask]`. `submit()`'s sole
// backpressure test is `seq - tail >= kTaskRingSize`, so the instant tail moves
// past t the producer may overwrite that very slot with sequence
// t + kTaskRingSize. When the ring is SATURATED — producer parked in
// submit_wait() on top of the slot the next claim frees, which is the steady
// state of any long fan-out of non-trivial tasks — the producer routinely wins
// that window: the worker executes the producer's NEW task and the task it
// claimed is never executed at all.
//
// The lost and duplicated executions balance EXACTLY (measured: 314-471 of
// 200,000 lost with an equal number of duplicates, every iteration). That is
// why it survived: the total execution count still equals the submitted count,
// so the ring's head/tail bookkeeping and the scheduler's
// tasks_submitted/tasks_completed barrier are both perfectly satisfied.
// Nothing inside the scheduler can detect it. Only a per-index census can,
// which is what this file is.
//
// End-to-end symptom: LSQB SF1 Cypher Q9 (an anti-join whose probe replay fans
// 26,364 morsels out) returned a DIFFERENT wrong count on every run at the
// default worker count — thirteen runs, thirteen values, none equal to LDBC's
// published 1,596,153,418 — each followed by SIGABRT from the per-morsel chunk
// lists two workers had raced on. CHUKONU_POOL_WORKERS=1 was exact.
//
// DISCRIMINATING POWER — and why this test drives the ring DIRECTLY
// -----------------------------------------------------------------
// A first version of this test went through Scheduler::submit_range() and was
// GREEN ON THE BROKEN RING: whether the race window opens at all depends on
// the producer/consumer speed ratio, and on a quiet box the workers drained as
// fast as the producer filled, so the ring never saturated and the defect was
// unreachable. A stress test that can be vacuously green is worse than no
// test. Two things fix that here:
//
//   1. The producer is OURS, so `submit()` returning false (ring full) is
//      COUNTABLE. The test asserts a large number of ring-full spins, i.e. it
//      proves it actually put the ring in the state the defect needs. If a
//      future change makes the ring impossible to saturate at this shape, this
//      test FAILS rather than silently passing.
//   2. The consumers call the real shipping `try_claim_and_execute()`, so the
//      code under test is the code that shipped — not a re-implementation.
//
// Verified against a header with the pre-fix ordering re-injected:
// 10/10 iterations violated exactly-once. With the fix: 0.

#include <bolt/bolt_scheduler.h>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

// Per-index execution census, plus a completion counter so the census is taken
// after every claimed task has actually finished (the ring's own wait_all only
// waits for tasks to be CLAIMED — see its doc comment).
std::atomic<std::uint32_t>* g_runs      = nullptr;
std::atomic<std::uint64_t>  g_completed{0};
std::int64_t                g_spin      = 0;

// Task body. The spin is what keeps the ring saturated; without it the
// consumers drain faster than the producer fills, the ring never fills, and
// the race window never opens. Do not "simplify" it away — the assertion on
// ring-full spins below exists to catch exactly that.
void census_body(void* arg) noexcept {
    const std::uint64_t idx = reinterpret_cast<std::uint64_t>(arg);
    g_runs[idx].fetch_add(1u, std::memory_order_relaxed);
    volatile std::int64_t sink = 0;
    for (std::int64_t k = 0; k < g_spin; ++k) sink += k ^ static_cast<std::int64_t>(idx);
    (void)sink;
    g_completed.fetch_add(1u, std::memory_order_release);
}

struct Census {
    std::uint32_t never_run   = 0;   // indices with 0 executions
    std::uint32_t duplicated  = 0;   // indices with >1 executions
    std::uint32_t extra_execs = 0;   // executions beyond one each
    std::uint64_t full_spins  = 0;   // times submit() reported the ring full
    std::uint64_t completed   = 0;   // bodies that ran to completion
};

// One saturating pass over a private TaskRing: `n_consumers` threads running
// the real try_claim_and_execute(), one producer (this thread) submitting
// `count` tasks and counting how often the ring was full.
Census saturating_pass(bolt::TaskRing* ring, std::uint32_t count,
                       int n_consumers) {
    assert(ring != nullptr);
    assert(count > 0 && n_consumers > 0);
    ring->init();
    g_completed.store(0u, std::memory_order_relaxed);
    for (std::uint32_t i = 0; i < count; ++i) {
        g_runs[i].store(0u, std::memory_order_relaxed);
    }

    std::atomic<bool> stop{false};
    std::vector<std::thread> consumers;
    consumers.reserve(static_cast<std::size_t>(n_consumers));
    for (int t = 0; t < n_consumers; ++t) {
        consumers.emplace_back([ring, &stop] {
            while (!stop.load(std::memory_order_acquire)) {
                if (!ring->try_claim_and_execute()) std::this_thread::yield();
            }
            while (ring->try_claim_and_execute()) {}   // final drain
        });
    }

    Census c{};
    for (std::uint64_t i = 0; i < count; ++i) {
        while (!ring->submit(&census_body, reinterpret_cast<void*>(i))) {
            ++c.full_spins;                       // <- the saturation witness
            bolt::cpu_pause();
        }
    }
    ring->wait_all(bolt::SpinPolicy::SpinYield);   // all CLAIMED

    // ...and then all RUN. Bounded so a genuine loss can never hang the suite;
    // on expiry we census anyway and let the assertions report what happened.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (g_completed.load(std::memory_order_acquire) < count &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    stop.store(true, std::memory_order_release);
    for (auto& t : consumers) t.join();

    c.completed = g_completed.load(std::memory_order_acquire);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t k = g_runs[i].load(std::memory_order_relaxed);
        if (k == 0u) {
            ++c.never_run;
        } else if (k > 1u) {
            ++c.duplicated;
            c.extra_execs += (k - 1u);
        }
    }
    return c;
}

// Sized from the measured reproduction: 200k tasks / 6 consumers / 2000-spin
// bodies saturates the 16384-slot ring for ~170M full-spins and takes ~0.3 s.
constexpr std::uint32_t kTasks       = 200000u;
constexpr int           kConsumers   = 6;
constexpr std::int64_t  kBodySpin    = 2000;
constexpr std::uint32_t kIterations  = 4u;
// Saturation floor. The reproduction runs ~1.6e8; anything within orders of
// magnitude of that means the ring really was full. A run that does not clear
// this bar has not tested the defect and must fail rather than pass.
constexpr std::uint64_t kMinFullSpins = 1000000u;

// The ring is large; keep one instance for the file rather than a stack local.
bolt::TaskRing g_ring;

// ---------------------------------------------------------------------------
// THE PIN.
// ---------------------------------------------------------------------------
TEST(BoltSchedulerRing, SaturatedRingRunsEveryTaskExactlyOnce) {
    // The producer must lap the ring, or the slot it would overwrite is never
    // one that has been claimed. Asserted, not assumed, so a future ring-size
    // change cannot quietly make this case vacuous.
    ASSERT_GT(kTasks, bolt::config::kTaskRingSize)
        << "task count must exceed the ring or the race cannot occur";

    g_spin = kBodySpin;
    std::vector<std::atomic<std::uint32_t>> runs(kTasks);
    g_runs = runs.data();

    for (std::uint32_t it = 0; it < kIterations; ++it) {
        const Census c = saturating_pass(&g_ring, kTasks, kConsumers);

        // Did we actually reach the state the defect needs? A green run that
        // never saturated proves nothing.
        ASSERT_GE(c.full_spins, kMinFullSpins)
            << "iteration " << it << ": the ring never saturated ("
            << c.full_spins << " full-spins) — this run did not exercise the "
               "claim-vs-overwrite window, so its result is meaningless";

        EXPECT_EQ(c.never_run, 0u)
            << "iteration " << it << ": " << c.never_run << " of " << kTasks
            << " tasks were NEVER RUN — a task was claimed but the producer "
               "overwrote its ring slot before the consumer read it";
        EXPECT_EQ(c.duplicated, 0u)
            << "iteration " << it << ": " << c.duplicated
            << " tasks ran more than once (" << c.extra_execs
            << " extra executions) — a consumer ran a slot another claim owned";
        // Stated separately: the exact balance is the SIGNATURE of this defect
        // and the reason no task-count bookkeeping can detect it. An imbalance
        // would be a different bug and should not be mistaken for this one.
        EXPECT_EQ(c.never_run, c.extra_execs)
            << "iteration " << it
            << ": lost and duplicated executions should balance exactly for "
               "the slot-overwrite race; an imbalance is a DIFFERENT defect";
        EXPECT_EQ(c.completed, static_cast<std::uint64_t>(kTasks))
            << "iteration " << it << ": only " << c.completed << " of "
            << kTasks << " task bodies ran to completion";
    }
}

// ---------------------------------------------------------------------------
// The same contract through the public API every caller actually uses. This
// one is STOCHASTIC — whether submit_range saturates the ring depends on the
// producer/consumer speed ratio, and it was green on the broken ring on a
// quiet box. It is kept as a smoke test of the API path, NOT as the pin; the
// direct-ring case above is what discriminates.
// ---------------------------------------------------------------------------
TEST(BoltSchedulerRing, SubmitRangeRunsEveryIndexExactlyOnce) {
    static bolt::Scheduler sched;
    static const bool ok = [] {
        bolt::SchedulerConfig cfg{};
        cfg.num_workers = 6;
        cfg.pin_workers = false;
        return sched.init(cfg);
    }();
    ASSERT_TRUE(ok) << "scheduler init failed";

    const std::uint32_t count = bolt::config::kTaskRingSize + 8192u;
    std::vector<std::atomic<std::uint32_t>> runs(count);
    for (std::uint32_t i = 0; i < count; ++i) runs[i].store(0u, std::memory_order_relaxed);

    struct Ctx { std::atomic<std::uint32_t>* runs; };
    Ctx ctx{runs.data()};
    sched.submit_range(
        [](void* user, std::uint32_t start, std::uint32_t end,
           std::uint32_t /*tid*/) noexcept {
            auto* c = static_cast<Ctx*>(user);
            for (std::uint32_t i = start; i < end; ++i) {
                c->runs[i].fetch_add(1u, std::memory_order_relaxed);
            }
        },
        &ctx, count, /*grain_size=*/1);
    sched.wait_all();

    std::uint32_t never_run = 0, duplicated = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t k = runs[i].load(std::memory_order_relaxed);
        if (k == 0u) ++never_run;
        else if (k > 1u) ++duplicated;
    }
    EXPECT_EQ(never_run, 0u);
    EXPECT_EQ(duplicated, 0u);

    // bolt::Scheduler has no destructor that joins its workers (the warm pool
    // is never torn down per-op), so a static one must be shut down explicitly
    // or the process aborts at static destruction with live std::threads.
    sched.shutdown();
}

}  // namespace
