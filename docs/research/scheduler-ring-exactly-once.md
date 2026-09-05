# The SPMC TaskRing lost and duplicated tasks in equal numbers

*W4-L1, 2026-09-05. `bolt_scheduler.h`, `TaskRing::try_claim_and_execute`.*

## The contract that was broken

`Scheduler::submit_range(fn, user, count, grain)` promises that every index in
`[0, count)` is passed to `fn` exactly once before `wait_all()` returns. The
ring underneath it promises the same per task. It did not keep that promise.

## The defect

`try_claim_and_execute` bumped `tail` (the claim) and *then* read the slot:

```cpp
if (!tail.compare_exchange_weak(t, t + 1, release, relaxed)) return false;
Task task = ring[t & kTaskRingMask];      // <-- too late
if (task.fn) task.fn(task.arg);
```

`submit()`'s only backpressure test is `seq - tail >= kTaskRingSize`. `tail`
advances at *claim* time, so the instant a consumer moves tail from `t` to
`t+1` the producer is free to write `ring[t & kTaskRingMask]` with sequence
`t + kTaskRingSize`. The consumer then reads the slot it no longer owns and
runs the producer's **new** task; the task it actually claimed is never run.

The window is a handful of instructions wide, but it is not rare, because the
ring's *saturated* state aims the producer straight at it: when the ring is
full the producer is spinning inside `submit_wait()` reloading `tail`, so the
very next thing it does after a claim is write the slot that claim just freed.
Saturation is the steady state of any fan-out whose tasks cost more than a
submit — which is every real fan-out.

## Why it survived

**Lost and duplicated executions balance exactly.** One task is skipped for
each task run twice, so the total execution count still equals the submitted
count. The ring's `head`/`tail` bookkeeping is consistent, and
`Scheduler::wait_all()`'s `tasks_submitted`/`tasks_completed` barrier is
satisfied. No counter the scheduler keeps can see it. Only a per-index census
can.

Measured on the pre-fix ring (200,000 tasks, 6 consumers, ~2 µs bodies,
16,384-slot ring), ten consecutive iterations:

| iter | ring-full spins | never run | duplicated | extra execs |
|---|---|---|---|---|
| 0 | 174,744,902 | 370 | 370 | 370 |
| 1 | 168,763,490 | 471 | 471 | 471 |
| 2 | 173,313,705 | 314 | 314 | 314 |
| … | … | … | … | … |
| 9 | 174,812,009 | 315 | 315 | 315 |

10/10 iterations violated exactly-once. Post-fix: 0.

## What it did downstream

chukonu's morsel-parallel hash-join probe replay fans one task per buffered
morsel. LSQB SF1 Cypher Q9 — a `WHERE NOT (…)` anti-join whose probe replay
fans out 26,364 morsels — returned a **different wrong count on every run** at
the default worker count: thirteen runs, thirteen values, spread ~90,842 around
LDBC's published 1,596,153,418, each followed by SIGABRT from the per-morsel
chunk lists two workers had raced on. `CHUKONU_POOL_WORKERS=1` was exact.

A morsel never probed contributes no rows; a morsel probed twice contributes
its rows twice. The observed deltas (−58,943, +56,886, …) are one morsel's
contribution, in both directions — the signature above, seen from the top.

## The fix

Copy the slot **before** claiming it:

```cpp
const Task task = ring[t & kTaskRingMask];        // copy first
if (!tail.compare_exchange_weak(t, t + 1, release, relaxed))
    return false;                                  // not ours; discard copy
if (task.fn) task.fn(task.arg);
```

This needs no extra synchronization. The producer may only overwrite slot `t`
once `tail > t`, and a *successful* CAS with expected `== t` proves `tail` was
still `t` at claim time — so the copy, sequenced before the release CAS, read
the slot while it was still ours. A *failed* CAS means another consumer owns
`t`; the copy is discarded unused. The release CAS also forbids sinking the
copy below it. Cost: nothing — the same load, moved.

A second, latent hole in the same barrier was closed alongside it: the
trampolines incremented `stats.tasks_completed` with `memory_order_relaxed`
while `wait_all()` acquire-loads it. A relaxed increment publishes the count
without publishing the task's stores, so the waiter could observe a completed
task's memory as it was before the task ran. That increment is now
`memory_order_release` — the only synchronizes-with edge between a worker
finishing and the submitter proceeding, since the ring's `tail` CAS happens at
claim time and publishes nothing about the result.

## Two things measured that are worth keeping

**A false assert.** An early version of the fix added
`assert(h - t <= kTaskRingSize)` after the two loads. It fires within a second
under a saturated multi-consumer ring: `t` and `h` are loaded at different
instants, and another consumer advancing `tail` between the loads leaves a
stale `t` trailing `h` by more than one ring. It is not an invariant. Removed.

**A test that could not see the bug.** The first version of the regression
pin went through `Scheduler::submit_range()` and was **green on the broken
ring**: whether the window opens at all depends on the producer/consumer speed
ratio, and on a quiet box the workers drained as fast as the producer filled,
so the ring never saturated. The same standalone probe that failed 6/6 during a
busy period passed 6/6 an hour later on the same binary. A stress test that can
be vacuously green is worse than no test.

`tests/test_bolt_scheduler_ring_exactly_once.cpp` therefore drives the ring
**directly**: the producer is the test's own, so `submit()` returning false is
countable, and the test *asserts a large number of ring-full spins* — it fails
if it did not reach the state the defect needs, instead of passing. The
consumers still call the real shipping `try_claim_and_execute()`. Verified 4/4
failing on a header with the pre-fix ordering re-injected.

The `submit_range` case is kept as a smoke test of the public API path and is
explicitly documented as **not** the pin, because it was green on the broken
ring.
