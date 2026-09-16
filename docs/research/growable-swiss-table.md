# Growable SwissTable — design rationale (G2CHK-85)

**Date:** 2026-09-16 · **Artifact:** `include/bolt/join/bolt_swiss_growable.h`
(`bolt::SwissTableGrowable`) + `src/api/net/fd_registry.h`
(`bolt::api::net::FdRegistry<Data>`) · **Consumers fixed:**
`src/api/net/event_loop_kqueue.cpp`, `src/api/net/event_loop_epoll.cpp`.

## The gap

`bolt::SwissTable` is fixed-capacity and build-once — capacity is rounded to a
power of two at `create()` and *never changes*, and there is no erase of any
kind (not a missing rehash: the ctrl-byte vocabulary has no Deleted state, so
deletion is structurally impossible without breaking the "stop at first empty
group" probe invariant). That is the right contract for a join build side, and
the wrong one for the three real consumers the 2026-09 audit found broken by
it: the fd→HandlerData registries in the epoll/kqueue/IOCP event loops (looked
up on **every dispatched I/O event**, mutated on every socket open/close) sat
on `std::unordered_map` — the container Tiger Style bans unconditionally on
hot paths (pointer-chasing, cache-hostile, a heap allocation per insert).

## State of the art weighed

| Design | Growth | Deletion | Verdict for bolt |
|---|---|---|---|
| **Abseil `raw_hash_set`** (the SwissTable reference) | stop-the-world ×2 (pow2) at 7/8 load; below ~25/32 live it instead rehashes **in place** to drop tombstones (`DropDeletesWithoutResize` — "worth it if it reclaims ≥1/8 capacity for ≤2× the work") | tombstone (`kDeleted` ctrl byte), plus a **was-never-full** early-out: if the erased slot's ±1-group window shows empties close enough on both sides, no probe chain can ever have passed it while full → revert straight to Empty, no tombstone | **Chosen.** bolt's fixed SwissTable already borrows Abseil's probe scheme (ctrl bytes, 7-bit tags, 16-slot group SIMD scan); the growable sibling should extend that philosophy, not fork it. Every mechanism grafts onto the existing layout with zero change to the proven `find` scan — Deleted (0xFE) matches neither the tag vector nor the Empty vector for free. |
| **folly F14** | ×2, chunk-granular | tombstone-free: each 14-slot chunk keeps an *outbound overflow count* decremented along the probe path on erase | Rejected: a different layout family (14-slot chunks + per-chunk metadata + counter maintenance on every erase path). No reuse of bolt's 16-slot ctrl array, SIMD scan, or the mirror-tail trick. |
| **Robin Hood (backward-shift)** | ×2 | no tombstones: erase memmoves the following displaced cluster back one slot | Rejected: O(cluster) writes per erase; requires per-slot displacement tracking; backward shift breaks the group-scan early-exit ("empty in group ⇒ absent") unless the whole probe scheme moves to per-slot PSL ordering — again a fork, not a sibling. Its selling point (probe-length variance) matters at ~90%+ load; we cap at 7/8. |
| **Hopscotch** | ×2 | neighborhood bitmap clear | Rejected: insert-time displacement cascades (an insert can trigger a chain of relocations), per-bucket H-bit bitmaps, and unbounded relocation worst cases without extra machinery — the opposite of Tiger Style's bounded-everything. |

## What was built

`bolt::SwissTableGrowable` — same `{u64 key → u32 value}` `SwissSlot`, same
ctrl bytes + mirrored tail group, same SIMD group scan as `SwissTable`, plus:

- **erase:** Abseil's two-tier deletion. Never-full window check first (the
  common low-load case leaves **zero** tombstones — verified by test); else a
  0xFE tombstone. Tombstones on the insert probe path are **reused** (first
  Deleted slot seen wins over the terminating Empty), so steady-state churn at
  constant live count neither grows the table nor lengthens probes.
- **growth:** at 7/8 occupancy (live + tombstones) the next insert-of-a-new-key
  reallocates from the arena captured at `create()`: ×2 when live > cap/2,
  same-size tombstone purge otherwise (reclaims > 3/8 cap — over Abseil's 1/8
  bar; churn never ratchets capacity). Growth is **bounded** by a caller
  `max_capacity` ceiling; at the ceiling insert fails cleanly (`false`, table
  untouched) — never silent unbounded growth. Superseded arrays become arena
  garbage (arenas don't free; waste is a geometric series < 1× the final
  footprint — give a long-lived registry its own arena).
- **not built:** thread safety (the consumers are one-loop-one-thread; the old
  `unordered_map` was equally unsynchronized), templated payloads (bolt's
  convention is value-as-index into a caller-owned pool — a non-trivially-
  copyable payload like `std::function` cannot live in an arena-memset table
  anyway), and incremental/amortized rehash (an fd registry's rehash at
  realistic sizes is microseconds; revisit only if a latency-critical consumer
  measures a spike).

`FdRegistry<Data>` (internal to `src/api/net/`) pairs the table with a
segmented slot pool (512-entry segments, placement-new'd once, freelist
recycled). Segment addresses are **stable across growth**, which also fixes a
latent bug in the old code: a handler calling `add_fd()` mid-dispatch could
rehash the `unordered_map` under the iterator the dispatcher was still
reading (`it->second.handler(...)`).

## Measured (M4, `build/verify_release`, min-of-7 reps; VM background load present)

`bench_swiss_growable` — 1M-key sections, 64K-live churn, 1K-live registry mix
(15/16 lookup + 1/16 close/open pair), checksums forced equal across impls:

| ns/op | insert (from empty) | find-hit | find-miss | churn | registry mix |
|---|---|---|---|---|---|
| SwissTableGrowable | 18.0 | 8.0–12.1 | 3.4–3.9 | **27–28** | 4.4 |
| SwissTable (fixed, pre-sized) | 6.3–6.9 | 9.6–10.7 | 3.9–5.2 | n/a | n/a |
| std::unordered_map (reserved) | 18.9–25.6 | 10.7–13.3 | 14.7–18.7 | 71–96 | 4.0–4.2 |

Honest read: **find parity with the fixed table** (same scan; the growable's
insert-from-empty pays ~2.6× for growth rehashes — pre-size the hint when the
cardinality is known). vs `unordered_map`: **~3× on churn** (the actual
socket open/close workload; node malloc/free per insert is the map's tax),
**~4× on find-miss**, and **parity on the small fully-cached registry mix** —
at 1K entries both structures are L1-resident and the map's two dependent
loads cost about what our mix+group-scan does. The structural wins there are
the ones the audit was actually about: no per-insert heap allocation, bounded
memory with an explicit ceiling, no allocator variance in the tail, and
Tiger-Style compliance on the dispatch path.

## Verification

- `tests/test_bolt_swiss_growable.cpp` (8 tests): grow-past-initial,
  ceiling-fails-cleanly, erase both paths, 200K-op churn boundedness
  (occupancy invariant asserted every op), zero-tombstone low-load erase,
  iteration-vs-oracle, 300K-op randomized stress vs `std::unordered_map`
  oracle with periodic full-sweep comparison.
- `tests/test_bolt_event_loop.cpp` (4 tests, first-ever direct coverage of
  the reactor loops): real-socketpair dispatch, remove/double-remove ENOENT,
  handler-adds-fd-mid-dispatch, 8×64-fd churn waves — green on macOS/kqueue;
  epoll TU compiled `-Wall -Wextra` clean AND a 3×64-fd live smoke run green
  on Linux (clang 18, container).

## Follow-ups

- `event_loop_iocp.cpp` + `async_io_iocp.cpp` (Windows): same swap, same
  `FdRegistry` (key is already u64-friendly for SOCKET); not buildable or
  testable on this box — deferred exactly like the audit deferred them.
- `async_io_epoll.cpp`'s `<unordered_map>` include was DEAD (zero uses) —
  removed in this pass. `async_io_iocp.cpp`'s real map (`associated`, the
  per-op IOCP association check) goes with the Windows follow-up.
