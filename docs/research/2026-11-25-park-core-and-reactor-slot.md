# Park core and the reactor slot (G2CHK-1293, G2CHK-1270)

Design: Gestalt2 `docs/research/2026-11-25-thread-parking-and-reactor-sota.md` §6.1–6.3 and §6.6, with
Ben's decisions D1, D6 and D7 (2026-11-25). Epic G2CHK-1268.

## What landed

- **`bolt_park.h`: `park::Slot`.**
  - One 32-bit word per waiter (kRunning / kParked / kWoken) on its own cache line.
  - Waited on with the native wait-on-address: futex, `__ulock_wait2`, or `WaitOnAddress`, with a
    timeout.
  - Waiter: store kParked, then a seq_cst fence, then re-check, then block.
  - Waker: publish, then a seq_cst fence, then if kParked, CAS kParked→kWoken. Only the CAS winner makes
    the kernel wake.
  - `SlotKind::kReactor` swaps the kernel wait for the I/O backend's poll and the kernel wake for the
    backend's raw wake (`EVFILT_USER`, eventfd), through a function pointer. No virtual call.
  - Owner-side counters, with no shared writes: announces, parks, spurious, spin_hits. Waker-side:
    kernel_wakes.
- **`WaitPolicy` and wait classes.**
  - Classes: `reactor`, `ack`, `pool`, `flusher`, `background`.
  - Modes: busy, spin-then-park(spin_ns), park.
  - The rows live in a `bolt.park` Limits table (`bolt_park_limits.h`). The process profile is
    `park_profile`: 0 efficient, 1 balanced (the default), 2 latency.
  - Each class has `_mode` (3 = take it from the profile), `_spin_ns` and `_deadline_ms`, plus
    `park_reactor_poll_every_ns` and `park_max_spinners`.
  - All rows are kind C and resolved on every wait, so a running process can change them.
  - Balanced defaults: reactor spin 20 µs, ACK 10 µs, pool 5 µs, background park (D1).
- **`park::wait()`** for address slots takes a deadline and returns `kDeadline` (D7).
- **`park_max_spinners`** is a process-wide cap on threads in a spin phase. 0 means cores/4 (D6).
- **The reactor (G2CHK-1270).**
  - `IODispatcher` no longer polls every 1 ms.
  - Each reactor turn polls at once if work is waiting (inbox, deferred completions, shutdown).
  - Otherwise it spins per the `reactor` class. It runs a zero-timeout I/O poll every
    `poll_every_ns`, and the slot stays kRunning, so posts make no syscall.
  - Then it parks on the engine's slot. It blocks in the backend until woken, until an event, or until
    the class deadline (1 s).
  - Each backend stores kRunning as soon as its kernel wait returns, so posts made while the turn
    dispatches pay nothing.
  - `async_io::wake()` is `park::wake(slot)`. `wake_pending` is gone from kqueue, epoll and io_uring.
  - IOCP keeps its bounded 1 ms poll because several threads poll one engine (`park_protocol()` false).
- **C1 fixed.** epoll cleared `wake_pending` before draining the eventfd, so a wake landing between the
  two was eaten, and every later wake was suppressed until the 1 ms poll found it. The flag is gone. The
  eventfd is edge-triggered and never drained: every write is a new edge.
- **io_uring.** An SQE queued by a thread other than the owner now wakes a parked owner, because without
  the 1 ms poll nothing else would submit it.
- **io_uring `has_ready()`.** It reports queued SQEs and unreaped CQEs, so a spinning reactor enters the
  ring as soon as there is work (see Measurements).
- **Shutdown.** `IODispatcher::stop()` wakes each engine after setting the shutdown flag.

## Tests

- `test_bolt_park`:
  - lost-wake stress, spin-then-park and park, 20,000 hand-offs each;
  - many producers pay at most one kernel wake per park cycle;
  - the deadline is honoured;
  - profile → mode mapping;
  - spinner cap;
  - `ReactorPark.PostsToAParkedReactorAreNotLost`: 3,000 posts at random gaps, none waits out the 1 s
    park deadline;
  - `ReactorPark.IdleReactorDoesNotTick`: an idle reactor polls fewer than 20 times in 400 ms, where the
    1 ms tick polled ~400 times.
- **Falsifier twins** `test_bolt_park_falsify_2` and `_3`. They build the same stress with
  `BOLT_PARK_FALSIFY` set to a lever that models a forbidden reordering: the waiter checks before it
  announces, or announces without a re-check. Each must report `LOST_WAKE`, and both do (lost wake
  measured at 0.2 s, the stress deadline). That proves the stress can see the bug it guards against.
- **TSan.** `test_bolt_park`, `test_bolt_io_inbox` and `test_bolt_async_io` report no race in the park
  or reactor code. The only reports are pre-existing races on `std::cout` from `IODispatcher`'s
  start/stop prints.

## Measurements

Same bench set as G2CHK-1269 (Gestalt2 `docs/research/data/2026-11-24-reactor/bench/`), bolt main
`b53d005` against this branch, with libuv. Medians: 3 repetitions on macOS, 2 on Linux.

**Cross-thread wake** (post from another thread → the reactor runs it), p50 / p99 µs:

| gap between posts | bolt main | G2CHK-1269 | **this change** | libuv |
|---|---|---|---|---|
| macOS, 0 (reactor in its spin window) | 4.83 / 58.5 | 3.88 / 38.0 | **0.33 / 2.4** | 2.25 / 32.6 |
| macOS, 100 µs (parked) | 7.38 / 90.1 | 9.88 / 116 | 9.62 / 93.5 | 9.46 / 111 |
| macOS, 1000 µs (parked) | 12.1 / 149 | 12.3 / 155 | 11.9 / 192 | 11.9 / 146 |
| Linux epoll, 0 | 10.3 / 69.4 | — | **1.56 / 11.9** | 7.33 / 61.6 (libuv) |
| Linux io_uring, 0 | 6.77 / 53.9 | — | **1.58 / 12.3** | |
| Linux epoll, 100 µs | 15.4 / 79.6 | — | 14.9 / 72.5 | 16.8 / 109 |

A parked wake costs the same kernel round trip everywhere: ~10 µs on macOS, ~15 µs in the VM. The spin
window takes the wake off the kernel path. An idle reactor makes no 1 ms poll: `IdleReactorDoesNotTick`
counts fewer than 20 polls in 400 ms, where the old loop made ~400.

**Linux VM I/O**, after the io_uring `has_ready` fix described below:

| | main epoll | **park epoll** | main io_uring | **park io_uring** | libuv | raw |
|---|---|---|---|---|---|---|
| Syscalls per TCP echo round trip | 12.06 | 6.00 | 2.06 | **2.00** | 3.00 | 3.00 |
| TCP ping-pong p50 / p99 µs | 32.8 / 112 | **19.5 / 86.3** | 29.4 / 84.0 | **21.2 / 94.9** | 33.4 / 147 | 32.3 / 202 |
| TCP echo rt/s, 1 / 2 reactors | 192k / 228k | **282k / 322k** | 264k / 243k | 276k / 226k | 243k / 245k | 245k / 311k |
| UDP sink rx, datagrams/s | 652k | **1.82M** | 1.97M | 1.81M | 1.65M | 1.80M |

**io_uring bug found by this run.** Without `has_ready()`, a spinning io_uring reactor only entered the
ring on its `poll_every_ns` (5 µs) poll. A re-submitted receive SQE therefore sat unsubmitted between
polls, and UDP receive fell to 392k/s. `io_uring_io::has_ready()` now reports queued SQEs and unreaped
CQEs, both of which are reads of the shared rings, so the reactor enters the ring the moment there is
work.

**The spin window costs something, and these are the numbers.**
- On epoll, the zero-timeout polls in a 20 µs spin take TCP echo from 3.06 (G2CHK-1269) to 6.00 syscalls
  per round trip.
- On loaded macOS, against park-only mode (`BOLT_PARK_REACTOR_MODE=2`), the 16-connection echo runs 16%
  lower: 147k vs 185k rt/s.
- Ping-pong p50 is 22.0 µs vs 17.5 µs, but p99 is 139 µs vs 110 µs.
- Park-only mode costs the 0.33 µs hot wake.

The balanced default (spin 20 µs) is Ben's D1. On a host with spare cores the spin should cost less;
these runs had host load 16–73. `latency` (busy) and `efficient` (park) are one knob away:
`BOLT_PARK_PROFILE`, or `BOLT_PARK_REACTOR_MODE` per class.

**boltapi HTTP** (`bench_server` + fixed main-built loadgen, macOS, quieter host). No regression:

| Path | io | main req/s | **park req/s** | main p99 µs | **park p99 µs** |
|---|---|---|---|---|---|
| /plaintext | 1 | 129.3k | 152.6k | 598 | 590 |
| /plaintext | 2 | 151.5k | 151.6k | 438 | 524 |
| /json | 1 | 143.6k | 149.2k | 655 | 500 |
| /json | 2 | 140.3k | 168.0k | 532 | 422 |


## Gate and leases (G2CHK-1294)

**`bolt_park_gate.h`.** `park::GateT<Cap>` holds a monotone progress value and a bounded min-heap, ordered
by target, of the waiting threads' slots.

- **`advance(v)`** stores v, fences, and returns at once when nobody is registered. Otherwise it pops every
  lease with target ≤ v under a small latch, releases the latch, and wakes exactly those slots once each.
  It can deposit a handoff word in each lease first.
- **`wait(target, deadline, policy, site)`:**
  - spins on the value per the class;
  - then registers the lease in the thread's slot and parks;
  - returns `kDeadline` and removes its lease when the deadline passes;
  - returns `kExhausted` when the gate is full (Limits B: `kGateWaitersDefault`).

  Leases never expire on their own.
- **Registry.** Slots come from one startup table (`park_slots`), and each lease sits under a seqlock.
  `for_each_lease` walks every live waiter, which is the source for `admin.debug.waiters`.
  `report_stuck` names a lease older than `park_stuck_report_ms` once and never wakes it.
- **Tests (`test_bolt_park_gate`):**
  - an advance wakes only the satisfied waiters (2 of 4 at v = 20, the rest at 40);
  - the deadline removes the lease;
  - a full gate refuses;
  - the handoff reaches the woken waiter;
  - the registry lists a waiter's target and site, and the stuck report fires once;
  - a 20,000-advance lost-wake stress passes.

  TSan reports no races.

## IdleSet for worker pools (G2CHK-1295)

**`bolt_park_idle.h`.** `park::IdleSetT<Cap>` is a set of interchangeable idle workers, with three rules:

- **Spinner rule:** while a worker spins it will find the work, so `notify()` wakes nobody.
- **LIFO:** the newest idle worker is woken first.
- **Chained wake:** a worker that takes an item and sees more calls `notify()`.

Each worker parks on its own slot. `notify_all()` is for shutdown only.

**`WorkerThreadPool`.** The task and blocking queues' idle paths now use an IdleSet with the `pool` wait
class: balanced spin 5 µs, then park, with a 30 s deadline after which it re-waits. The space-side waits
(a producer facing a full queue) keep `EventCount`.

**Bug found by the existing pool suite.** In the first cut, a `notify()` that popped a worker before the
worker announced kParked could not wake it, and the worker then parked off the stack. With every worker
in that state, the queue stalled until the 30 s deadline: `test_bolt_worker_pool_park` hung in
`Doors/Park.BurstsWithIdleGapsAllRun` in park-only and spin modes. The fix: being off the stack is a
reason not to block, so `park_once`'s re-check covers it. The suite now passes in all three pool modes,
in about 6 s each, and under TSan.

**EventCount's wake-sent dedupe was not added.** Ticket G2CHK-1295 asked for a wake-sent bit on
`bolt::EventCount`. It was tried, and the existing `EventCountStress.PingPongNeverLosesAWake` hung. An
EventCount can carry waiters for different conditions (the ping-pong parks both sides on one count), and
a single in-flight bit suppresses the wake the other side needs. EventCount keeps one kernel notify per
`notify_one` while someone sleeps, and the dedupe lives where waiters are interchangeable: the IdleSet.
EventCount gained a `kernel_notifies()` diagnostic counter.
