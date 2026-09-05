// bolt_sim_sched.h — a deterministic cooperative scheduler for bolt's task ring.
//
// TEST-ONLY. Nothing in bolt links this; it exists so that a concurrency bug in
// `bolt::TaskRing` is a function of a SEED rather than a function of the box.
//
// WHY
// ---
// The W4-L1 SPMC ring defect (`docs/research/scheduler-ring-exactly-once.md`)
// had five properties that together make an ordinary stress test worthless:
//
//   * nondeterministic — ~29% of runs,
//   * LOAD-DEPENDENT, and inverted: 3 of 8 runs wrong on an idle box, 1 of 6
//     under load. A quiet box HID it,
//   * invisible to every counter the scheduler keeps, because the lost and the
//     duplicated executions balance exactly,
//   * only reachable through a gap BETWEEN two adjacent instructions, so a
//     harness that preempts between whole calls cannot see it at all,
//   * and — decisively — the first regression pin written for it was GREEN ON
//     THE BROKEN CODE, because whether the window opens depends on the
//     producer/consumer speed ratio.
//
// A test whose outcome depends on machine timing is not a gate. Here exactly
// one participant runs at any instant; who runs next is drawn from a seeded
// PRNG at every interleaving point. The schedule is therefore a pure function
// of (seed, scenario, git commit), identical on an idle laptop and on a box at
// load 30, and a failing seed replays exactly.
//
// WHAT IS AND IS NOT SIMULATED
// ----------------------------
// The RING is real: `submit_sim<>` and `try_claim_and_execute_sim<>` are the
// shipping bodies, instantiated with a different compile-time policy. The
// backpressure predicate, the CAS, the memory orders and the slot store are all
// executed as written. Only the *scheduling* is ours. We do not stub the clock
// or the filesystem, and we do not simulate the world — only the one component
// whose nondeterminism has actually cost us waves.
//
// MEMORY MODEL NOTE
// -----------------
// Participant-visible state (the census, counters, this struct's non-atomic
// fields) is plain memory with NO atomics, deliberately. `turn` is handed over
// with release/acquire, which is the happens-before edge between one holder and
// the next, and no two participants ever run concurrently. Making the census
// atomic would hide an ordering mistake in the harness rather than expose it.

#pragma once

#include <bolt/bolt_port.h>
#include <bolt/bolt_scheduler.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <thread>

namespace bolt_sim {

constexpr std::uint32_t kMaxParticipants = 8;
constexpr std::uint32_t kMaxParkFaults   = 32;
constexpr std::uint32_t kTurnNobody      = 0xFFFFFFFFu;
constexpr std::uint32_t kNotSuspended    = bolt::sched_point::kPointCount;

/// Invariant checker, run by the token holder at every interleaving point while
/// the world is quiescent. Returns nullptr when the state is legal, or a
/// message naming what broke.
typedef const char* (*InvariantFn)(void* ctx) noexcept;

/// One injected fault: participant `who` is not schedulable from step `at_step`
/// until `at_step + steps`. Our idiom for "a worker parked at a chosen instant".
struct ParkFault {
    std::uint64_t at_step;
    std::uint32_t who;
    std::uint32_t steps;
};

struct Sim {
    // The token. The only cross-thread synchronisation in the harness.
    alignas(64) std::atomic<std::uint32_t> turn;
    alignas(64) std::atomic<bool>          aborted;

    // ---- token-holder-private below this line (no atomics by design) ----
    std::uint64_t rng;
    std::uint32_t n;
    bool          alive[kMaxParticipants];
    std::uint64_t unpark_at[kMaxParticipants];   // schedulable once steps >= this
    std::uint32_t at_point[kMaxParticipants];    // where a suspended one waits

    std::uint64_t steps;
    std::uint64_t step_cap;

    ParkFault     parks[kMaxParkFaults];
    std::uint32_t n_parks;
    std::uint32_t next_park;
    std::uint64_t park_events;

    InvariantFn   check;
    void*         check_ctx;
    const char*   invariant_msg;
    std::uint64_t invariant_step;

    // Coverage witnesses — a run that did not reach the interesting states must
    // FAIL rather than pass (the vacuous-green lesson, W4-L1).
    std::uint64_t yields;
    std::uint64_t yields_at[bolt::sched_point::kPointCount];
    std::uint64_t hazard_windows;      // see sim_note_slot_write()
    std::uint64_t hazard_collisions;   // ...with a colliding slot index
    const bolt::TaskRing* ring;        // borrowed, for the hazard witness only
};

/// The participant id of the calling thread (kTurnNobody off-harness).
inline thread_local std::uint32_t tl_sim_id = kTurnNobody;
inline Sim* g_sim = nullptr;

// ---------------------------------------------------------------------------
// Seeded PRNG — splitmix64. Advanced ONLY by the token holder, so the stream is
// consumed in a single total order and the schedule replays exactly.
// ---------------------------------------------------------------------------
inline std::uint64_t sim_rand(Sim* s) noexcept {
    assert(s != nullptr);
    assert(s->n > 0 && s->n <= kMaxParticipants);
    s->rng += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = s->rng;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

inline void sim_init(Sim* s, std::uint32_t participants, std::uint64_t seed,
                     std::uint64_t step_cap, const bolt::TaskRing* ring) noexcept {
    assert(s != nullptr);
    assert(participants > 0 && participants <= kMaxParticipants);
    s->turn.store(0u, std::memory_order_relaxed);
    s->aborted.store(false, std::memory_order_relaxed);
    s->rng = seed * 0x2545F4914F6CDD1Dull + 0x9E3779B97F4A7C15ull;
    s->n = participants;
    for (std::uint32_t i = 0; i < kMaxParticipants; ++i) {
        s->alive[i]     = (i < participants);
        s->unpark_at[i] = 0u;
        s->at_point[i]  = kNotSuspended;
    }
    s->steps = 0u;
    s->step_cap = step_cap;
    s->n_parks = 0u;
    s->next_park = 0u;
    s->park_events = 0u;
    s->check = nullptr;
    s->check_ctx = nullptr;
    s->invariant_msg = nullptr;
    s->invariant_step = 0u;
    s->yields = 0u;
    for (std::uint32_t i = 0; i < bolt::sched_point::kPointCount; ++i) {
        s->yields_at[i] = 0u;
    }
    s->hazard_windows = 0u;
    s->hazard_collisions = 0u;
    s->ring = ring;
    assert(s->n == participants);
}

inline void sim_set_invariant(Sim* s, InvariantFn fn, void* ctx) noexcept {
    assert(s != nullptr);
    assert(fn != nullptr);
    s->check = fn;
    s->check_ctx = ctx;
}

/// Draw `count` park faults from the seed stream, spread over `horizon` steps.
/// Bounded by kMaxParkFaults; a request beyond that is clamped, never grown.
inline void sim_seed_parks(Sim* s, std::uint32_t count, std::uint64_t horizon,
                           std::uint32_t max_park_steps) noexcept {
    assert(s != nullptr);
    assert(horizon > 0u);
    if (count > kMaxParkFaults) count = kMaxParkFaults;
    if (max_park_steps == 0u) max_park_steps = 1u;
    std::uint64_t at = 0u;
    for (std::uint32_t i = 0; i < count; ++i) {
        at += 1u + (sim_rand(s) % (horizon / (count ? count : 1u) + 1u));
        s->parks[i].at_step = at;
        s->parks[i].who     = static_cast<std::uint32_t>(sim_rand(s) % s->n);
        s->parks[i].steps   = 1u + static_cast<std::uint32_t>(sim_rand(s) % max_park_steps);
    }
    s->n_parks = count;
    assert(s->n_parks <= kMaxParkFaults);
}

inline bool sim_aborted(const Sim* s) noexcept {
    assert(s != nullptr);
    return s->aborted.load(std::memory_order_acquire);
}

// ---------------------------------------------------------------------------
// Turn handoff
// ---------------------------------------------------------------------------

/// Block until the token is ours. Spin-then-yield: the WAIT may take any amount
/// of wall time under load, but the ORDER is fixed by `turn`, so the schedule is
/// unaffected. That is the whole point — determinism without a global lock.
inline void sim_wait_turn(Sim* s, std::uint32_t id) noexcept {
    assert(s != nullptr);
    assert(id < s->n);
    std::uint32_t spins = 0u;
    while (s->turn.load(std::memory_order_acquire) != id) {
        if (s->aborted.load(std::memory_order_relaxed)) return;
        if (++spins < 4096u) bolt::cpu_pause();
        else                 std::this_thread::yield();
    }
}

/// Pick the next participant. Parked ones are skipped; if that would leave
/// nobody, every park is released first, so a fault can stall the system but
/// never deadlock it.
inline std::uint32_t sim_pick(Sim* s) noexcept {
    assert(s != nullptr);
    assert(s->n > 0u);
    std::uint32_t cand[kMaxParticipants];
    std::uint32_t k = 0u;
    for (std::uint32_t pass = 0; pass < 2u && k == 0u; ++pass) {
        if (pass == 1u) {
            for (std::uint32_t i = 0; i < s->n; ++i) s->unpark_at[i] = 0u;
        }
        for (std::uint32_t i = 0; i < s->n; ++i) {
            if (!s->alive[i]) continue;
            if (s->unpark_at[i] > s->steps) continue;
            cand[k++] = i;
        }
    }
    if (k == 0u) return kTurnNobody;          // everyone has finished
    return cand[sim_rand(s) % k];
}

inline void sim_apply_due_parks(Sim* s) noexcept {
    assert(s != nullptr);
    assert(s->next_park <= s->n_parks);
    while (s->next_park < s->n_parks &&
           s->parks[s->next_park].at_step <= s->steps) {
        const ParkFault& f = s->parks[s->next_park];
        if (f.who < s->n && s->alive[f.who]) {
            s->unpark_at[f.who] = s->steps + f.steps;
            ++s->park_events;
        }
        ++s->next_park;
    }
}

/// The hazard witness. Called when a producer has just stored a ring slot.
///
/// The pre-fix ring read its slot AFTER the claiming CAS, so the state the
/// defect needs is: some consumer suspended at kClaimWon (post-CAS, pre-body)
/// while the producer overwrites the very slot that consumer claimed.
///
/// `hazard_windows` counts the NECESSARY half — a consumer sitting in that gap
/// during a slot store. `hazard_collisions` adds the slot-index test: the store
/// targets `head & mask` (head is not published yet, so head still reads as the
/// sequence being written), and the outstanding claims are the last `n` values
/// below `tail`. At most one of them can be congruent mod kTaskRingSize.
///
/// Reported, never argued from: the proof of discriminating power is the
/// injection battery, not this counter. The counter exists so a run that never
/// reached the state FAILS instead of passing.
inline void sim_note_slot_write(Sim* s, std::uint32_t self) noexcept {
    assert(s != nullptr);
    assert(self < s->n);
    if (s->ring == nullptr) return;
    bool suspended_in_gap = false;
    for (std::uint32_t i = 0; i < s->n; ++i) {
        if (i == self) continue;
        if (s->at_point[i] == bolt::sched_point::kClaimWon) { suspended_in_gap = true; break; }
    }
    if (!suspended_in_gap) return;
    ++s->hazard_windows;

    const std::uint64_t written = s->ring->head.load(std::memory_order_relaxed);
    const std::uint64_t tail    = s->ring->tail.load(std::memory_order_relaxed);
    for (std::uint32_t j = 1; j <= s->n && j <= tail; ++j) {
        if (((written ^ (tail - j)) & bolt::kTaskRingMask) == 0u) {
            ++s->hazard_collisions;
            return;
        }
    }
}

/// One interleaving point: check invariants, age the fault schedule, hand the
/// token to a seed-chosen participant, and block until it comes back.
inline void sim_yield(Sim* s, std::uint32_t point) noexcept {
    assert(s != nullptr);
    assert(point < bolt::sched_point::kPointCount);
    const std::uint32_t self = tl_sim_id;
    if (self == kTurnNobody) return;        // not a participant: no-op
    if (s->aborted.load(std::memory_order_relaxed)) return;

    if (s->check != nullptr && s->invariant_msg == nullptr) {
        const char* msg = s->check(s->check_ctx);
        if (msg != nullptr) { s->invariant_msg = msg; s->invariant_step = s->steps; }
    }
    if (point == bolt::sched_point::kSubmitSlotWritten) sim_note_slot_write(s, self);

    ++s->steps;
    ++s->yields;
    ++s->yields_at[point];
    if (s->steps > s->step_cap) {
        // Loud, bounded failure. From here every participant runs free so the
        // threads can unwind; the run is already a failure and its schedule no
        // longer matters.
        s->aborted.store(true, std::memory_order_release);
        s->turn.store(kTurnNobody, std::memory_order_release);
        return;
    }
    sim_apply_due_parks(s);

    const std::uint32_t next = sim_pick(s);
    s->at_point[self] = point;
    s->turn.store(next, std::memory_order_release);
    if (next != self) {
        sim_wait_turn(s, self);
        s->at_point[self] = kNotSuspended;
    } else {
        s->at_point[self] = kNotSuspended;
    }
}

/// Enter the harness as participant `id`; blocks until first scheduled.
inline void sim_enter(Sim* s, std::uint32_t id) noexcept {
    assert(s != nullptr);
    assert(id < s->n);
    tl_sim_id = id;
    sim_wait_turn(s, id);
}

/// Leave the harness: mark dead and hand the token on. Must be the last thing a
/// participant does.
inline void sim_finish(Sim* s) noexcept {
    assert(s != nullptr);
    assert(tl_sim_id < s->n || tl_sim_id == kTurnNobody);
    const std::uint32_t self = tl_sim_id;
    tl_sim_id = kTurnNobody;
    if (self == kTurnNobody) return;
    if (s->aborted.load(std::memory_order_relaxed)) return;
    s->alive[self]    = false;
    s->at_point[self] = kNotSuspended;
    const std::uint32_t next = sim_pick(s);
    s->turn.store(next, std::memory_order_release);
}

/// The interleaving policy handed to `TaskRing::submit_sim` /
/// `try_claim_and_execute_sim`. This is the ONLY difference between the ring
/// under simulation and the ring in production.
struct Hook {
    static void point(unsigned p) noexcept {
        if (g_sim != nullptr) sim_yield(g_sim, p);
    }
};

}  // namespace bolt_sim
