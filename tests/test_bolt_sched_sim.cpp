// test_bolt_sched_sim.cpp — deterministic simulation of bolt's SPMC task ring.
//
// A SEED IS A TEST.
//
// This is the TigerBeetle VOPR model scoped to the one component whose
// nondeterminism has actually cost this project waves: `bolt::TaskRing`. The
// ring, its backpressure predicate and its completion accounting run AS
// WRITTEN — `submit_sim<>` / `try_claim_and_execute_sim<>` are the shipping
// bodies, instantiated with a different compile-time interleaving policy (see
// bolt_scheduler.h). Only the SCHEDULING is ours, and it is a pure function of
// the seed, so a run is reproducible from (seed, scenario, commit) and is the
// same on an idle laptop as on a box at load 30.
//
// WHAT IT REPLACES. test_bolt_scheduler_ring_exactly_once.cpp is a real pin but
// a STOCHASTIC one: it needs the ring genuinely saturated by real threads, so it
// asserts a floor on ring-full spins precisely because it can otherwise be
// vacuously green, and its predecessor WAS green on the broken ring on a quiet
// box. That file stays (it exercises real concurrency, real memory ordering and
// real hardware). This file is the deterministic complement: same contract, no
// dependence on the box.
//
// INVARIANTS CHECKED (the "state checkers beyond the answer"):
//   1. EXACTLY-ONCE — a per-index execution census. This is the only instrument
//      that can see the W4-L1 defect at all: its lost and duplicated executions
//      balance exactly, so every counter inside the scheduler stays consistent.
//   2. THE BALANCE SIGNATURE — never_run == extra_execs is the fingerprint of
//      the slot-overwrite race specifically; an imbalance is a DIFFERENT bug and
//      is reported as such rather than mistaken for this one.
//   3. RING BOUNDS at every interleaving point: tail <= head and
//      head - tail <= kTaskRingSize. Under real threads these two loads are not
//      a consistent snapshot and the bound is NOT an invariant (measured: such
//      an assert fires within a second). Under the simulator exactly one
//      participant runs, so the snapshot IS consistent and the bound IS
//      checkable — a checker determinism buys us that concurrency cannot have.
//   4. NON-VACUITY — every run asserts it reached the states the defect needs:
//      the producer lapped the ring, and a consumer sat between its winning CAS
//      and its body while the producer stored a slot. Across the sweep each
//      saturated scenario must additionally show backpressure firing and at
//      least one exact slot COLLISION. A run that did not reach them FAILS.
//      Which witness is asserted per seed and which only in aggregate is a
//      MEASURED choice — see the comment on check().
//
// DISCRIMINATING POWER. Proven by injection, not by argument:
// `tests/sim/inject_prefix_ring.py` re-injects the pre-fix claim-then-read
// ordering into the header, rebuilds, and requires this suite to FAIL. See
// docs/research/deterministic-scheduler-simulation.md for the recorded result
// and the named failing seeds.

#include "sim/bolt_sim_sched.h"

#include <bolt/bolt_scheduler.h>

#include <gtest/gtest.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

using bolt_sim::Hook;
using bolt_sim::Sim;

// ---------------------------------------------------------------------------
// Scenario + run state
// ---------------------------------------------------------------------------

struct Scenario {
    const char*   name;
    std::uint32_t prefill;         // submitted before the scheduled phase
    std::uint32_t extra;           // submitted under the scheduler
    std::uint32_t consumers;
    std::uint32_t parks;           // injected "worker parked at instant N"
    std::uint32_t alloc_refusals;  // injected "allocation refused at call N"
};

struct Run {
    bolt::TaskRing* ring;
    std::uint32_t*  runs;          // per-index execution census
    const std::uint8_t* skipped;   // indices the injected allocator refused
    std::uint32_t   total;
    std::uint32_t   prefill;
    std::uint32_t   executed;
    std::uint32_t   submitted;
    std::uint64_t   full_spins;    // submit() reported the ring full
    bool            producer_done;
};

Run* g_run = nullptr;
Sim  g_sim_storage;

// Bounded loop caps (Tiger Style: every loop has a fixed upper bound).
constexpr std::uint32_t kMaxSubmitAttempts = 1u << 22;
constexpr std::uint32_t kMaxConsumerSpins  = 1u << 22;
constexpr std::uint64_t kStepCap           = 40ull * 1000ull * 1000ull;

void census_body(void* arg) noexcept {
    // Plain (non-atomic) increments are correct here BY CONSTRUCTION: exactly
    // one participant runs at a time and the token handoff is release/acquire.
    // Making them atomic would mask an ordering mistake in the harness.
    assert(g_run != nullptr);
    const std::uint64_t idx = reinterpret_cast<std::uintptr_t>(arg);
    assert(idx < g_run->total);
    ++g_run->runs[idx];
    ++g_run->executed;
}

const char* ring_invariant(void* ctx) noexcept {
    assert(ctx != nullptr);
    const auto* r = static_cast<const bolt::TaskRing*>(ctx);
    const std::uint64_t h = r->head.load(std::memory_order_relaxed);
    const std::uint64_t t = r->tail.load(std::memory_order_relaxed);
    if (t > h) return "tail overtook head";
    if (h - t > bolt::kTaskRingSize) return "ring holds more than kTaskRingSize entries";
    return nullptr;
}

// ---------------------------------------------------------------------------
// Participants
// ---------------------------------------------------------------------------

void producer_main(Sim* s, std::uint32_t id) noexcept {
    assert(s != nullptr);
    assert(g_run != nullptr);
    bolt_sim::sim_enter(s, id);
    for (std::uint32_t i = g_run->prefill; i < g_run->total; ++i) {
        if (bolt_sim::sim_aborted(s)) break;
        if (g_run->skipped[i]) continue;      // injected allocation refusal
        bool ok = false;
        for (std::uint32_t a = 0; a < kMaxSubmitAttempts; ++a) {
            if (bolt_sim::sim_aborted(s)) break;
            if (g_run->ring->submit_sim<Hook>(
                    &census_body, reinterpret_cast<void*>(
                        static_cast<std::uintptr_t>(i)))) {
                ok = true;
                break;
            }
            ++g_run->full_spins;              // the saturation witness
        }
        if (!ok) break;
        ++g_run->submitted;
    }
    g_run->producer_done = true;
    bolt_sim::sim_finish(s);
}

void consumer_main(Sim* s, std::uint32_t id) noexcept {
    assert(s != nullptr);
    assert(g_run != nullptr);
    bolt_sim::sim_enter(s, id);
    // A failed claim already yields inside try_claim_and_execute_sim (at
    // kClaimTailLoaded), so an idle consumer still hands the token on.
    for (std::uint32_t spins = 0; spins < kMaxConsumerSpins; ++spins) {
        if (bolt_sim::sim_aborted(s)) break;
        if (g_run->ring->try_claim_and_execute_sim<Hook>()) { spins = 0; continue; }
        if (g_run->producer_done &&
            g_run->ring->tail.load(std::memory_order_acquire) >=
            g_run->ring->head.load(std::memory_order_acquire)) {
            break;
        }
    }
    bolt_sim::sim_finish(s);
}

// ---------------------------------------------------------------------------
// One simulated run
// ---------------------------------------------------------------------------

struct Verdict {
    std::uint32_t never_run   = 0;
    std::uint32_t duplicated  = 0;
    std::uint32_t extra_execs = 0;
    std::uint32_t skipped     = 0;
    std::uint64_t full_spins  = 0;
    std::uint64_t hazards     = 0;
    std::uint64_t collisions  = 0;
    std::uint64_t steps       = 0;
    std::uint64_t park_events = 0;
    const char*   invariant   = nullptr;
    bool          aborted     = false;
    bool          prefill_ok  = true;
};

Verdict simulate(const Scenario& sc, std::uint64_t seed) {
    assert(sc.consumers >= 1u && sc.consumers + 1u <= bolt_sim::kMaxParticipants);
    assert(sc.extra > 0u);

    static bolt::TaskRing ring;            // 256 KB — never a stack local
    const std::uint32_t total = sc.prefill + sc.extra;
    std::vector<std::uint32_t> runs(total, 0u);
    std::vector<std::uint8_t> skipped(total, 0u);

    Run run{};
    run.ring = &ring;
    run.runs = runs.data();
    run.skipped = skipped.data();
    run.total = total;
    run.prefill = sc.prefill;
    g_run = &run;

    Sim* s = &g_sim_storage;
    bolt_sim::sim_init(s, sc.consumers + 1u, seed, kStepCap, &ring);
    bolt_sim::sim_set_invariant(s, &ring_invariant, &ring);

    // Injected allocation refusals: drawn from the SAME seed stream, so they are
    // part of the run's identity. A refused index is never submitted, and the
    // census must show it never ran — accounted for exactly, never lost quietly.
    for (std::uint32_t i = 0; i < sc.alloc_refusals; ++i) {
        const std::uint32_t victim =
            sc.prefill + static_cast<std::uint32_t>(bolt_sim::sim_rand(s) % sc.extra);
        skipped[victim] = 1u;
    }
    if (sc.parks > 0u) {
        // Horizon must land INSIDE the run or the faults are scheduled past the
        // end and never fire — measured: a first version used 64*total and
        // injected exactly zero parks while reporting success on every other
        // axis. A claim goes through ~4 interleaving points, so ~4*total is the
        // step count; parks are spread over the first half of that so even the
        // last one has room to expire.
        bolt_sim::sim_seed_parks(s, sc.parks, /*horizon=*/2u * total,
                                 /*max_park_steps=*/512u);
    }

    // Prefill with the SHIPPING submit() (no interleaving policy, single
    // threaded). This constructs exactly the state that `prefill` successful
    // submits with no claims leaves behind — a reachable state, reached the
    // normal way, just without paying for a scheduled interleaving to get there.
    ring.init();
    bool prefill_ok = true;
    for (std::uint32_t i = 0; i < sc.prefill; ++i) {
        // Checked, not asserted: an assert would vanish under -DNDEBUG and a
        // short prefill would silently turn every saturated scenario into an
        // unsaturated one that still reports green.
        if (!ring.submit(&census_body,
                reinterpret_cast<void*>(static_cast<std::uintptr_t>(i)))) {
            prefill_ok = false;
            break;
        }
        ++run.submitted;
    }

    bolt_sim::g_sim = s;
    std::vector<std::thread> parts;
    parts.reserve(sc.consumers + 1u);
    parts.emplace_back(producer_main, s, 0u);
    for (std::uint32_t c = 0; c < sc.consumers; ++c) {
        parts.emplace_back(consumer_main, s, c + 1u);
    }
    for (auto& t : parts) t.join();
    bolt_sim::g_sim = nullptr;

    Verdict v{};
    for (std::uint32_t i = 0; i < total; ++i) {
        if (skipped[i]) { ++v.skipped; }
        const std::uint32_t k = runs[i];
        const std::uint32_t want = skipped[i] ? 0u : 1u;
        if (k < want) ++v.never_run;
        else if (k > want) { ++v.duplicated; v.extra_execs += (k - want); }
    }
    v.full_spins  = run.full_spins;
    v.hazards     = s->hazard_windows;
    v.collisions  = s->hazard_collisions;
    v.steps       = s->steps;
    v.park_events = s->park_events;
    v.invariant   = s->invariant_msg;
    v.aborted     = bolt_sim::sim_aborted(s);
    v.prefill_ok  = prefill_ok;
    g_run = nullptr;
    return v;
}

// ---------------------------------------------------------------------------
// Scenarios. `prefill == kTaskRingSize` puts the ring in the SATURATED state
// that is the steady state of any real fan-out and the only state in which the
// claim-vs-overwrite window is open on essentially every claim.
// ---------------------------------------------------------------------------

constexpr std::uint32_t kRing = bolt::kTaskRingSize;

const Scenario kScenarios[] = {
    // name                     prefill    extra consumers parks refusals
    {"saturated-1c",            kRing,     2048u,      1u,    0u,   0u},
    {"saturated-2c",            kRing,     2048u,      2u,    0u,   0u},
    {"saturated-3c",            kRing,     2048u,      3u,    0u,   0u},
    {"saturated-parked-2c",     kRing,     2048u,      2u,    8u,   0u},
    {"saturated-refusals-2c",   kRing,     2048u,      2u,    0u,   7u},
    // Control: starts half empty, so the first thousands of claims run with the
    // window CLOSED, and the ring only fills as the producer outruns the
    // consumers. Still laps the ring (total > kTaskRingSize) — a scenario that
    // cannot lap can never place a claimed slot in the producer's path, and the
    // non-vacuity assertion below rejects one that does not.
    {"fills-from-half-2c",      kRing / 2u, kRing,     2u,    0u,   0u},
};
constexpr std::uint32_t kScenarioCount =
    static_cast<std::uint32_t>(sizeof(kScenarios) / sizeof(kScenarios[0]));

// Non-vacuity, keyed on what the MEASUREMENT supports rather than on what would
// be nice to assert (measured over seeds 1..8, per seed, per scenario):
//
//   * `hazards` — a consumer suspended between its winning CAS and its body
//     while the producer stored a slot. 496..685 on every saturated scenario on
//     every seed. Robust per seed, so it is a per-seed ASSERTION.
//   * `collisions` — the above, with the stored slot index actually equal to the
//     claimed one. That is the defect's precondition exactly, but it is 0..74
//     and genuinely reaches 0 for some seeds at 2 and 3 consumers, where claims
//     outrun submits. Per seed it is asserted only for the single-consumer
//     scenario (4..74, never 0); across the sweep it is asserted for all of
//     them, which is where the gate's discriminating power lives.
//   * `full_spins` — backpressure firing. 0..156, zero for some seeds. Reported
//     and asserted in aggregate only. An early version asserted it per seed and
//     failed two seeds for a reason that had nothing to do with the ring.
//
// Asserting a witness that is only usually present would make the gate flaky by
// seed — the exact failure mode this file exists to remove.
void check(const Scenario& sc, std::uint64_t seed, const Verdict& v,
           bool saturated, bool require_collision) {
    const std::string where =
        std::string(sc.name) + " seed=" + std::to_string(seed);

    ASSERT_TRUE(v.prefill_ok)
        << where << ": the ring refused a prefill submit, so this scenario never "
                    "reached the fill level it names";
    ASSERT_FALSE(v.aborted)
        << where << ": the simulation exceeded its step budget (" << v.steps
        << ") — a participant made no progress, i.e. a task was lost WITHOUT a "
           "matching duplicate, or the harness deadlocked";
    EXPECT_EQ(v.invariant, nullptr)
        << where << ": ring invariant broken: "
        << (v.invariant ? v.invariant : "");

    // --- non-vacuity: did this run reach the state the defect needs? ---
    ASSERT_GT(sc.prefill + sc.extra, bolt::kTaskRingSize)
        << where << ": the producer must lap the ring or the slot it would "
                    "overwrite is never one that has been claimed";
    if (saturated) {
        ASSERT_GT(v.hazards, 0u)
            << where << ": no consumer was ever suspended between its winning "
                        "CAS and its body while the producer stored a slot — the "
                        "claim-vs-overwrite window was never open, so a green "
                        "result here proves nothing";
    }
    if (require_collision) {
        ASSERT_GT(v.collisions, 0u)
            << where << ": the producer never overwrote the exact slot a "
                        "suspended consumer had claimed — this seed did not "
                        "reach the defect's precondition";
    }

    // --- the contract ---
    EXPECT_EQ(v.never_run, 0u)
        << where << ": " << v.never_run << " tasks were NEVER RUN";
    EXPECT_EQ(v.duplicated, 0u)
        << where << ": " << v.duplicated << " tasks ran more than once ("
        << v.extra_execs << " extra executions)";
    EXPECT_EQ(v.never_run, v.extra_execs)
        << where << ": lost and duplicated executions balance exactly for the "
                    "slot-overwrite race; an imbalance is a DIFFERENT defect";
}

std::uint64_t env_u64(const char* name, std::uint64_t dflt) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return dflt;
    return std::strtoull(v, nullptr, 10);
}

// ---------------------------------------------------------------------------
// THE GATE — a seed sweep. Every failing seed is printed by number so it can be
// replayed exactly (BOLT_SIM_SEED=<n>) and kept as a permanent regression case.
// ---------------------------------------------------------------------------
TEST(BoltSchedSim, SeedSweepRunsEveryTaskExactlyOnce) {
    const std::uint64_t seeds = env_u64("BOLT_SIM_SEEDS", 12u);
    const std::uint64_t base  = env_u64("BOLT_SIM_SEED_BASE", 1u);
    ASSERT_GT(seeds, 0u);

    std::uint32_t failing = 0;
    std::uint64_t agg_steps[kScenarioCount]  = {};
    std::uint64_t agg_spins[kScenarioCount]  = {};
    std::uint64_t agg_haz[kScenarioCount]    = {};
    std::uint64_t agg_coll[kScenarioCount]   = {};
    std::uint64_t agg_parks[kScenarioCount]  = {};

    for (std::uint64_t k = 0; k < seeds; ++k) {
        const std::uint64_t seed = base + k;
        for (std::uint32_t si = 0; si < kScenarioCount; ++si) {
            const Scenario& sc = kScenarios[si];
            const Verdict v = simulate(sc, seed);
            agg_steps[si] += v.steps;
            agg_spins[si] += v.full_spins;
            agg_haz[si]   += v.hazards;
            agg_coll[si]  += v.collisions;
            agg_parks[si] += v.park_events;
            // Count failure PARTS, not HasFailure(): the latter is sticky, so
            // after the first bad pair every later pair looks like it failed
            // too and the sweep reports "1 failing pair" for a run in which
            // twenty failed. (Measured on the injected pre-fix ring.)
            const int before = ::testing::UnitTest::GetInstance()
                                   ->current_test_info()->result()->total_part_count();
            // Hazard coverage is required only where the ring starts saturated;
            // fills-from-half is a control that exercises the same code with the
            // window closed (0 collisions, by construction) and must also be
            // exactly-once. si == 0 is the single-consumer scenario, the one
            // whose collision count is reliable per seed.
            check(sc, seed, v, /*saturated=*/sc.prefill == kRing,
                  /*require_collision=*/si == 0u);
            const int after = ::testing::UnitTest::GetInstance()
                                  ->current_test_info()->result()->total_part_count();
            if (after > before) {
                ++failing;
                std::fprintf(stderr,
                    "[bolt-sim] FAILING SEED: scenario=%s seed=%llu "
                    "(replay: BOLT_SIM_SEEDS=1 BOLT_SIM_SEED_BASE=%llu)\n",
                    sc.name, static_cast<unsigned long long>(seed),
                    static_cast<unsigned long long>(seed));
            }
        }
    }
    // Show the work. A gate that reports only pass/fail cannot be audited for
    // vacuity after the fact; these are the coverage witnesses the assertions
    // above are keyed on, summed over the sweep.
    std::fprintf(stderr,
        "[bolt-sim] %llu seeds x %u scenarios\n"
        "[bolt-sim] %-24s %10s %12s %9s %10s %6s\n",
        static_cast<unsigned long long>(seeds), kScenarioCount,
        "scenario", "steps", "backpressure", "windows", "collisions", "parks");
    for (std::uint32_t si = 0; si < kScenarioCount; ++si) {
        std::fprintf(stderr, "[bolt-sim] %-24s %10llu %12llu %9llu %10llu %6llu\n",
            kScenarios[si].name,
            static_cast<unsigned long long>(agg_steps[si]),
            static_cast<unsigned long long>(agg_spins[si]),
            static_cast<unsigned long long>(agg_haz[si]),
            static_cast<unsigned long long>(agg_coll[si]),
            static_cast<unsigned long long>(agg_parks[si]));
    }
    // Aggregate non-vacuity. Per seed a collision is not guaranteed above one
    // consumer; over the sweep it must happen for EVERY saturated scenario, or
    // that scenario contributed nothing and the sweep is wider than it is deep.
    for (std::uint32_t si = 0; si < kScenarioCount; ++si) {
        if (kScenarios[si].prefill != kRing) continue;
        EXPECT_GT(agg_coll[si], 0u)
            << kScenarios[si].name << ": across all " << seeds
            << " seeds the producer never overwrote a slot a suspended consumer "
               "had claimed — this scenario never reached the defect state";
        EXPECT_GT(agg_spins[si], 0u)
            << kScenarios[si].name << ": across all " << seeds
            << " seeds backpressure never fired — the ring was never actually "
               "full, so this scenario is not the saturated case it claims to be";
    }
    EXPECT_EQ(failing, 0u) << failing << " (scenario, seed) pairs failed";
}

// ---------------------------------------------------------------------------
// The schedule must be a pure function of the seed. If this fails, every other
// result in this file is a coincidence: a failing seed would not replay and the
// suite would be a slower stress test rather than a simulator.
// ---------------------------------------------------------------------------
TEST(BoltSchedSim, SameSeedReplaysExactly) {
    const Scenario& sc = kScenarios[1];      // saturated-2c
    const Verdict a = simulate(sc, 20260905u);
    const Verdict b = simulate(sc, 20260905u);
    EXPECT_EQ(a.steps, b.steps) << "step count differed between two runs of one seed";
    EXPECT_EQ(a.full_spins, b.full_spins) << "backpressure count differed";
    EXPECT_EQ(a.hazards, b.hazards) << "hazard-window count differed";

    const Verdict c = simulate(sc, 20260906u);
    // Not a correctness requirement, but if a different seed produced an
    // identical schedule the sweep would be one test repeated N times.
    EXPECT_NE(a.steps, c.steps)
        << "two different seeds produced identical step counts — the seed is "
           "not actually driving the schedule";
}

// ---------------------------------------------------------------------------
// Injected allocation refusals must be accounted for EXACTLY: a refused index
// never runs, every other index runs once. The failure this guards against is
// a refusal being absorbed silently (the "rc=0 with 0 rows" class, one layer
// down).
// ---------------------------------------------------------------------------
TEST(BoltSchedSim, AllocationRefusalIsAccountedForExactly) {
    const Scenario& sc = kScenarios[4];      // saturated-refusals-2c
    const Verdict v = simulate(sc, 424242u);
    ASSERT_FALSE(v.aborted);
    EXPECT_GT(v.skipped, 0u) << "the refusal fault did not fire";
    EXPECT_EQ(v.never_run, 0u);
    EXPECT_EQ(v.duplicated, 0u);
}

// ---------------------------------------------------------------------------
// A parked worker must stall the system without breaking it, and must not
// deadlock it. Parks are drawn from the same seed stream, so this is one more
// axis of the same replayable schedule.
// ---------------------------------------------------------------------------
TEST(BoltSchedSim, ParkedWorkerStallsButNeverDeadlocks) {
    const Scenario& sc = kScenarios[3];      // saturated-parked-2c
    const Verdict v = simulate(sc, 909090u);
    ASSERT_FALSE(v.aborted) << "a parked worker deadlocked the run";
    EXPECT_GT(v.park_events, 0u) << "the park fault did not fire";
    EXPECT_EQ(v.never_run, 0u);
    EXPECT_EQ(v.duplicated, 0u);
}

}  // namespace
