# Per-edge tick-tock buffering protocol

## Context

Bolt's `BoltBatch` already has a tick-tock COW mechanism inside a
single batch (two physical column buffers, `read_epoch`/`write_epoch`,
swap = bit flip + dirty mask reset, ~1.3 ns measured at
`bolt_column.h:443-449`). The dataflow layer pushes this one level up:
each **edge** between two operators is a per-edge ring of `BoltBatch`
with the same tick-tock semantics. Producer writes to `write_epoch`,
publishes by flipping, consumer reads from the new `read_epoch`.

This file fixes the protocol: ring sizing, visibility / memory order,
backpressure, and garbage collection.

## Why per-edge buffers (not a global queue)

A single shared MPMC queue between all operators would force every
batch through one cache line of contention. Per-edge ring keeps the
producer / consumer pair in their own cache lines (already proven by
`SPSCChannel` in `bolt_channel.h`, 17 ns transit). The dataflow layer
is just an SPSC ring per edge, specialized to carry `BoltBatch*` and
to use Bolt's existing tick-tock primitive instead of head/tail
indexes.

## Ring size — N

Two regimes worth considering:

### N = 2 (pure tick-tock, default)

Two slots per edge. Producer writes slot W (`write_epoch`); consumer
reads slot R (`read_epoch`). Publish = `epoch ^= 1`. Producer must
wait if the consumer hasn't released the slot it's about to write to.

- **Pros:** smallest memory footprint, simplest invariants, matches
  the in-batch BoltBatch pattern exactly
- **Cons:** producer stalls on any consumer hiccup; bursty input
  produces visible jitter at the consumer

### N > 2 (bursty buffer)

`N` slots per edge with a producer cursor `wpos` and consumer cursor
`rpos`. Producer writes slot `wpos % N`, advances; consumer reads
slot `rpos % N`, advances. Backpressure: producer waits if
`wpos - rpos == N`.

- **Pros:** absorbs short consumer lag; smoother p99 under bursty
  ticks
- **Cons:** N × `BoltBatch` arena footprint per edge; more invariants
  to maintain; visibility now needs a sequence number, not just a
  bit flip

### Decision

**Default N = 2** for streaming↔streaming edges. Configurable per-edge
at graph compile time via a template parameter. N is a `constexpr`
on the edge type — never a runtime value, so the consumer-pop and
producer-push code is fully inlined and branch-free.

For pipeline-breaker → streaming edges, N = 1 (the breaker writes the
materialized state once, the downstream operators read until phase
end; no flipping).

## Memory order / visibility

Producer side, per publish:

```cpp
// producer wrote into edge.slot[write_epoch]
batch.dirty_mask_lo.store(0, std::memory_order_relaxed);  // reset
batch.dirty_mask_hi.store(0, std::memory_order_relaxed);
batch.read_epoch.store(write_epoch_was, std::memory_order_release);
                                       // ^^^^^^^ release flush guarantees
                                       // consumer's acquire-load sees all
                                       // writes from this producer.
batch.write_epoch ^= 1;
```

Consumer side, per pop:

```cpp
uint8_t e = batch.read_epoch.load(std::memory_order_acquire);
const BoltColumn& col = batch.columns[e][col_idx];
// ... read col, completely safe — release-acquire pair guarantees
// producer's writes to this slot are visible
```

This is exactly the LMAX Disruptor sequence-number protocol, scoped
down: the "sequence number" is just the epoch bit, because N=2 means
we only need to distinguish two states.

For N > 2, the epoch becomes a `uint64_t` sequence number with the
same release/acquire pair.

**Adopted:** release-store on producer publish, acquire-load on
consumer pop. No CAS on either side. No memory_order_seq_cst — too
expensive on x86, completely unnecessary for SPSC.

## Cache-line discipline

Each edge's producer cursor and consumer cursor sit on separate cache
lines (`alignas(64)`). The shared `read_epoch` byte is in the
producer's cache line; the consumer pulls it in (one cache miss per
publish, unavoidable). The slot array sits on yet another cache line.

This is the same layout as `SPSCChannel`. The 17 ns transit measurement
comes from this discipline — 4 cache-line touches per push+pop pair
plus the memory barrier.

## Backpressure

Three policies, selected per-edge at compile time:

### Block (default for streaming pipelines)

Producer spin-pauses on `BOLT_PAUSE` until the consumer releases the
slot. After N pauses, escalates to `std::this_thread::yield`.
Bounded latency (consumer can't fall arbitrarily behind because the
producer makes no progress).

### Drop-newest (for telemetry / fire-and-forget edges)

If the slot is still occupied, drop the new batch on the floor and
increment `edge.dropped_count`. Producer never stalls. Used for
edges where the consumer is intentionally allowed to skip.

### Coalesce (for stateful aggregators)

If the slot is still occupied AND the queued batch has the same
shape, merge the new batch into the queued one (operator-defined
merge fn). Useful for tick-driven aggregations where consumer lag
should produce one larger batch, not a dropped one.

The default is **block**; the others are explicit overrides.

## Garbage collection

`BoltBatch` lives in the same arena as the operator that produces
it. Slot reuse (consumer releases an N=2 slot back to the producer)
is *not* deallocation — the slot's columns just get overwritten on
the next publish. The arena resets only at the morsel boundary, so
the producer and consumer are always pointing into valid memory
within the morsel.

The consumer never frees a batch. The producer never `delete`s.
There is no per-batch allocation in the steady state.

## Failure modes

- **Producer faster than consumer (sustained):** under `block`,
  producer stalls; under `drop-newest`, batches lost; under
  `coalesce`, downstream sees larger merged batches. Measured via
  `edge.dropped_count` / `edge.coalesced_count`.
- **Consumer faster than producer:** consumer spin-pauses on the
  producer's `read_epoch` byte. No correctness issue, just wasted
  CPU. The scheduler should park the consumer thread (work-stealing
  or sleep) — this is a scheduler concern, not an edge concern.
- **Producer crashes mid-publish:** undefined. We don't recover.
  This is in-process only; no fault tolerance promised.

## What we adopt

- **N=2 default** for streaming↔streaming edges; N=1 for breaker→
  streaming; N>2 only when justified by tick-burstiness measurement
- **Release-store / acquire-load pair** for visibility; no CAS, no
  seq_cst
- **Bolt's existing `read_epoch`/`write_epoch`/`dirty_mask`** mechanism
  reused as the per-edge swap primitive — no new code for the swap
  itself
- **Three backpressure policies** (block, drop-newest, coalesce) as
  compile-time edge attributes
- **Arena-tied slot lifetime** — slot reuse is overwrite, not free

## What we skip

- **MPMC general-purpose queue** — every edge is SPSC by construction;
  fan-out / fan-in handled by operator multiplicity, not edge
  multiplicity
- **Hazard pointers / RCU** — overkill for tick-tock; release/acquire
  on a single byte gives all the visibility we need
- **Runtime-configurable N** — N is `constexpr` per edge type, never
  a runtime value
- **Sequence-number protocol for N=2** — the bit flip IS the sequence

## Followups

- Decide whether the producer's stall escalation (pause → yield →
  sleep) belongs at the edge or at the scheduler
- Profile per-edge buffer cache footprint at high operator counts
  (50+ operator graphs from chukonu's TPC-H plans)
- Backpressure policy auto-selection from edge type metadata
  (e.g. telemetry sinks default to drop-newest)

## References

- LMAX Disruptor sequence-number protocol (Thompson et al.)
- Venus ECS double-buffered entity_db
- Bolt's `bolt_column.h:443-449` (`BoltBatch::swap()`)
- Bolt's `bolt_channel.h` SPSC ring (17 ns measured transit)
