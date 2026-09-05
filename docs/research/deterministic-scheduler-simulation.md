# Deterministic simulation of the task ring — a seed is a test

*W5-L2, 2026-09-05. `bolt_scheduler.h`, `tests/sim/`, `tests/test_bolt_sched_sim.cpp`.
Companion to `scheduler-ring-exactly-once.md`, which is the defect this instrument
is built to find in seconds instead of a wave.*

## Why

The W4-L1 SPMC ring defect had five properties that, together, make an ordinary
stress test worthless as a gate:

1. **Nondeterministic** — roughly 29% of runs.
2. **Load-dependent, and inverted** — 3 of 8 runs wrong on an idle box, 1 of 6
   under load. A *quiet* box hid it, which is the opposite of the usual advice.
3. **Invisible to every counter in the scheduler** — lost and duplicated
   executions balance exactly, so `tasks_submitted`/`tasks_completed` and the
   ring's own `head`/`tail` bookkeeping all stay perfectly consistent.
4. **Reachable only through a gap between two adjacent instructions**, so a
   harness that can preempt only between whole calls cannot see it at all.
5. **The first regression pin written for it was GREEN ON THE BROKEN CODE**,
   because whether the window opens depends on the producer/consumer speed ratio.

A test whose outcome depends on machine timing is not a gate. This is the
TigerBeetle VOPR model — a run is a pure function of a seed plus the commit —
scoped deliberately to the scheduler rather than the engine, because that is
where our nondeterminism actually lives.

## The design, and the one decision that mattered

Exactly one participant runs at any instant. At every interleaving point the
token holder draws the next runnable participant from a seeded splitmix64 stream
and hands the token over. Wall-clock time still varies with load; the *order*
does not.

**The ring is not re-implemented.** `TaskRing::submit_sim<Sim>` and
`try_claim_and_execute_sim<Sim>` are the shipping bodies with a compile-time
policy parameter; production instantiates `SchedSimNoop`, whose `point()` is an
empty static function. `submit()` and `try_claim_and_execute()` are one-line
forwarders. The backpressure predicate, the CAS, the memory orders and the slot
store are all executed as written.

**A macro was considered and rejected.** Two translation units including this
header with different macro definitions is an ODR violation; the linker would
keep one inline body, and — since bolt is header-only and the scheduler is
included by a dozen TUs — the one it kept could easily be the un-instrumented
one. That failure mode is a *silently vacuous simulator*, which is worse than no
simulator. A template argument makes the two versions genuinely different
functions.

### Production codegen is unchanged — measured, not asserted

Three real consumer TUs (`bolt_csv.cpp`, `bolt_parquet_write.cpp`,
`io_dispatcher.cpp`) compiled at the same path with the same flags, HEAD header
vs instrumented header:

| config | `__text` | symbol table |
|---|---|---|
| `-O3 -DNDEBUG` (timing) | **IDENTICAL** on all 3 | IDENTICAL on all 3 |
| `-O3 -g` (asserts live) | identical except 4 immediates per TU | IDENTICAL on all 3 |

The four immediates are `__LINE__` values passed to `__assert_rtn`, each shifted
by exactly **68** — the number of lines inserted above them. Instruction counts
are equal (4,538 / 4,538 and 20,676 / 20,676) and every instruction address
matches.

*A methodology note worth keeping.* The first attempt compared two include
*trees* and reported `IDENTICAL` for everything. The hash was
`e3b0c44298fc1c14…` — the SHA-256 of the empty string: the compiles had failed
and stderr was being discarded. The comparison script now refuses to compare an
object smaller than 1 KB and checks the compiler's exit status. Two include trees
also differ in path, which `-g` embeds, so the real comparison swaps the header
at the *same* path.

## What is checked

| checker | what it catches |
|---|---|
| per-index execution census | the W4-L1 class — the only instrument that can see it |
| `never_run == extra_execs` | the *signature* of the slot-overwrite race; an imbalance is reported as a DIFFERENT defect, not mistaken for this one |
| `tail <= head` and `head - tail <= kTaskRingSize` at every point | see below |
| step budget exceeded | a task lost *without* a matching duplicate, or a deadlock |

The ring-bounds check is something determinism buys that concurrency cannot.
`scheduler-ring-exactly-once.md` records that an `assert(h - t <= kTaskRingSize)`
inside `try_claim_and_execute` **fires within a second** under real threads,
because `t` and `h` are loaded at different instants — it is not an invariant
there. Under the simulator exactly one participant runs, the snapshot is
consistent, and the bound *is* an invariant. It is checked at all ~1.6 M
interleaving points of a 12-seed sweep.

## Faults, in our idiom

Injected from the same seed stream, so they are part of the run's identity:

- **Ring saturation** — scenarios prefill the ring to exactly `kTaskRingSize`,
  which is the steady state of any fan-out whose tasks cost more than a submit.
  The prefill uses the shipping `submit()` and constructs precisely the state
  that N successful submits with no claims leave behind.
- **A worker parked at a chosen instant**, for a seeded duration. If parking
  would leave nobody runnable, every park is released first — a fault may stall
  the system, never deadlock it.
- **An allocation refused at a chosen call index** — the refused task is never
  submitted, and the census must show it never ran while every other index ran
  once. Guards against a refusal being absorbed silently.

## Discriminating power — the result

`tests/sim/inject_prefix_ring.py` re-injects the pre-fix claim-then-read ordering
into the header, rebuilds, requires the suite to FAIL, then restores and requires
it to PASS. On the pre-fix ring, seeds 1–12:

| scenario | seeds failing (of 12) |
|---|---|
| `saturated-1c` | **11** |
| `saturated-parked-2c` | 8 |
| `saturated-refusals-2c` | 7 |
| `saturated-2c` | 6 |
| `saturated-3c` | 3 |
| `fills-from-half-2c` (control) | **0** |

35 of 72 (scenario, seed) pairs fail; the control — which laps the ring but never
saturates it, so the window never opens — fails on none, which is what a control
is for. Only seed 10 of 1–12 leaves `saturated-1c` green.

Every failure carries the signature: e.g. seed 1, `saturated-1c`, **22 tasks
never run and 22 extra executions**, `saturated-2c` 4 and 4,
`saturated-refusals-2c` 1 and 1.

### The property that replaces the old probe

Seed 1 on the pre-fix ring, three runs on a quiet box and three at **load average
77** (24 spinning processes on an 18-core M4):

```
quiet   22 / 4 / 1      load 77   22 / 4 / 1
quiet   22 / 4 / 1      load 77   22 / 4 / 1
quiet   22 / 4 / 1      load 77   22 / 4 / 1
```

Identical values. The old probe was 3-of-8 on an idle box and 1-of-6 under load;
this is 6-of-6 either way with the same numbers. A single-seed replay is
**0.63–0.69 s** for all six scenarios; the 12-seed sweep is ~8 s.

On the shipping ring a **100-seed × 6-scenario sweep is green** — 600 simulated
runs, 88.8 M interleaving points, 87 s — with 6,061 exact slot collisions
reached and none of them producing a lost or duplicated execution.

## Non-vacuity, and a choice made by measurement

Asserting a witness that is only *usually* present would make the gate flaky by
seed — the exact failure this file exists to remove. Measured per seed, per
scenario, over seeds 1–8:

| witness | range | asserted |
|---|---|---|
| consumer suspended post-CAS while a slot was stored | 496–685 | **per seed**, every saturated scenario |
| ...with the stored slot index equal to the claimed one | 0–74 | per seed for `saturated-1c` only (4–74, never 0); in **aggregate** for all |
| backpressure fired | 0–156 | **aggregate only** |

An earlier version asserted backpressure per seed and failed two seeds for a
reason that had nothing to do with the ring.

## Two harness bugs found by the harness's own checks

Both were in the test, not the engine, and both were caught by an assertion
written specifically to reject a vacuous run:

1. **Park faults never fired.** The horizon was `64 × total` steps while runs are
   ~`4 × total`, so every injected park was scheduled past the end of the run.
   The suite reported success on every other axis. Caught by
   `EXPECT_GT(park_events, 0)`.
2. **The failing-seed counter under-reported by an order of magnitude.**
   `::testing::Test::HasFailure()` is sticky, so after the first bad pair every
   later pair looked already-failed and the sweep printed "1 failing pair" for a
   run in which 35 failed. Now it compares `total_part_count()` before and after.

## Limits, stated

- Only the **ring** is simulated. `Scheduler::submit_range`, the trampolines and
  the `wait_all` counter barrier are exercised by the existing tests, not by this
  one. Extending the participants to drive `submit_range` is the obvious next
  step and needs no new machinery.
- The simulator explores schedules **randomly**, not exhaustively. It is a fuzzer
  with a reproducible tape, not a model checker; absence of a failing seed is not
  a proof of correctness.
- `hazard_collisions` is computed from `head`/`tail` at the moment of a store and
  attributes a claim to one of the last `n` sequence numbers below `tail`. That
  is exact while at most `n` claims are outstanding, which holds here, but it is
  a *witness for non-vacuity*, never evidence of correctness. The proof that the
  gate discriminates is the injection battery above.
- `test_bolt_scheduler_ring_exactly_once.cpp` is **not** superseded. It exercises
  real threads, real memory ordering and real hardware; this file cannot. They
  are complements.

## Reproduce

```sh
cmake -S extern/bolt -B build/bolt-sim -DBOLT_BUILD_TESTS=ON \
      -DBOLT_BUILD_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS_RELEASE="-O3 -g"          # asserts LIVE
cmake --build build/bolt-sim --target test_bolt_sched_sim -j8
./build/bolt-sim/tests/test_bolt_sched_sim                    # 12-seed sweep

BOLT_SIM_SEEDS=1 BOLT_SIM_SEED_BASE=7 ./build/bolt-sim/tests/test_bolt_sched_sim
BOLT_SIM_SEEDS=200 ./build/bolt-sim/tests/test_bolt_sched_sim  # deeper sweep

python3 extern/bolt/tests/sim/inject_prefix_ring.py --build-dir build/bolt-sim
```

`BOLT_BUILD_VULKAN=OFF` is required for a standalone bolt build on this box:
`glslc` is absent, so `bolt_vulkan.cpp` fails on a missing generated shader
header. Pre-existing and unrelated.

## Gates run for this change

| gate | result |
|---|---|
| bolt standalone suite, asserts live (`-O3 -g`) | **147/150**; pristine-HEAD baseline **146/149**, the same 3 failing (`test_bolt_kmerge`, `test_bolt_csr`, `test_bolt_parquet_write_varbinary` — tests that drive assert-guarded rejection paths in an asserts-live build). +1 test, 0 regressions |
| superproject `ctest` on `build/coverage` | **371/371, 100%** (1 skipped: `boltapi_neo4j_bolt_driver_conformance`) |
| TPC-H SF1 vs DuckDB | **22/22 value-exact** |
| production codegen | byte-identical, see table above |

The superproject numbers were taken while another agent was concurrently
rebuilding and running `ctest` in the same tree (40 objects rewritten mid-run,
three `ctest` processes observed). Contention can only turn a pass into a
failure, not the reverse, so 371/371 stands — but it is not a clean-room
measurement and a timing number must not be taken from that window.

## Still to do — outside this lane's file ownership

`test_bolt_sched_sim` is registered in `extern/bolt/tests/CMakeLists.txt`, so it
runs in a standalone bolt build. It is **not** in the superproject's
`tests/CMakeLists.txt`, which is where the 371-test gate lives — the same
"a test that never runs is not a gate" hazard this regime exists to kill
(testing-plan law #8). Four lines are needed there, mirroring
`bolt_scheduler_ring_exactly_once_test`:

```cmake
gestalt_add_test(bolt_sched_sim_test
    ${CMAKE_SOURCE_DIR}/extern/bolt/tests/test_bolt_sched_sim.cpp)
target_link_libraries(bolt_sched_sim_test PRIVATE bolt::bolt)
# deliberately NOT RUN_SERIAL: load-independence is the point
```
