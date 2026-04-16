# Push vs pull dispatch

## Context

Two ways to schedule operator work in a dataflow graph: pull (consumer
asks producer for next batch) or push (producer fires batches at
consumer when ready). The choice shapes the executor, the edge buffer,
the backpressure model, and how cleanly tick-driven workloads fit.

This file picks one and justifies it.

## Pull / Volcano

Classic relational engine model. Every operator implements
`next() -> Batch*`. The root operator's `next()` recursively calls
its children's `next()`, which call theirs, all the way to the scan.

- **Pros:** simple to reason about; trivially handles fan-out (root
  pulls until done)
- **Cons:** every operator boundary is a virtual function call (vtable
  dispatch) or a templated callback (no inlining across operators);
  control flow is inverted at every operator (operator must be a state
  machine that yields between calls)

Used by Postgres, classic Oracle. **Tiger Style rules out the virtual
call.** Even with templates, the per-batch call overhead at 100k
ticks/s × N operators dominates.

## Push (callback / fire-and-forget)

Source operator generates batches when work is available; pushes via
a downstream `consume(batch)` call. Pipeline operators are functions
of `(in_batch) -> out_batch` plus side effects on state.

Used by Apache Arrow Acero, tremor.rs, Kafka Streams, Spark
Continuous Processing.

- **Pros:** natural fit for event-driven sources (a tick arrived → push
  it through); operators are stateless functions, easy to test;
  scheduler has full visibility into operator readiness; no
  control-flow inversion
- **Cons:** backpressure has to be explicit (producer doesn't know
  consumer's capacity by default); fan-in needs explicit
  synchronization

## Hybrid: push-with-credit

Producer pushes only if it has credit; consumer grants credit by
releasing buffer slots. Effectively push, with implicit credit = ring
buffer slot count.

This is what we land on. The N-slot edge ring (see
`tick-tock-buffering-protocol.md`) IS the credit window. Producer
pushes if there's a free slot; if not, it spins / yields under the
edge's backpressure policy. Consumer releases a slot when it advances
past it. No explicit credit messages.

## Why push for Bolt

The user's target shape (per project memory + plan):

> event-driven columns where we basically can almost do our operation
> on tick. So we have number of ticks per millisecond or second, and
> then we operate on the the the batch we have inside that tick.

Pull is wrong-shape for this. The natural model is:

```
tick arrives -> source operator produces a batch
              -> pushes to filter operator
              -> pushes to gather operator
              -> pushes to aggregate operator
              -> pushes to sink
```

A pull model would force every operator to poll its source
("any new data?") at every tick. With 100k+ ticks/sec and a 5-stage
pipeline, that's 500k poll calls/sec doing nothing. Push lets sources
be the *only* thing that initiates work; downstream operators run
exactly once per upstream batch.

## Executor shape (push)

The executor is a loop:

```cpp
void Executor::tick() noexcept {
    // 1. Pull source operators' new batches into their output edges.
    for (Operator* src : sources_) {
        src->execute(nullptr, src_outputs(src), {0, batch_size}, arena, src->state);
        publish(src_outputs(src));   // epoch flip
    }

    // 2. For each operator in topo order, if all its input edges have
    //    fresh data and it has output-edge credit, dispatch it.
    for (Operator* op : topo_order_) {
        if (!has_fresh_inputs(op)) continue;
        if (!has_output_credit(op)) continue;
        scheduler_.submit(op);   // morsel-parallel within operator
        publish(op_outputs(op)); // after morsels complete
    }
}
```

Per-operator parallelism (morsel-driven within one operator) reuses
the existing `bolt::Scheduler::submit_range`. The dataflow layer
adds the cross-operator coordination — *which* operator to dispatch
next, in what order, with what input edges ready.

## Why not pure async (no executor loop)

A purely event-driven model would have each operator wired directly
to its consumers via callbacks; no central executor. Tremor uses this
shape. It's elegant but loses the executor's ability to:

- Pin operators to NUMA nodes
- Reorder operator dispatch to prefetch hot caches
- Reschedule operators that lose their cache locality
- Apply phase barriers cleanly for pipeline breakers

The executor loop costs a few ns per tick (one walk through
`topo_order_`) and pays for itself by enabling these.

## Phase barriers and pipeline breakers

A pipeline breaker (sort, hash-join build, blocking aggregate)
disrupts pure push: downstream operators can't start until the
breaker is fully fed. The executor handles this by splitting the
graph into **phases** at compile time — each phase is a maximal
DAG of streaming operators terminating in breakers; the next phase
begins after all breakers in the previous phase signal completion
via the existing `PhaseBarrier`.

Within a phase, push dispatch flows freely. Across a phase
boundary, the executor synchronously waits for the barrier.

This is exactly DuckDB's pipeline-breaker model, ported to push.

## What we adopt

- **Push dispatch** with N-slot edge ring as implicit credit
- **Central executor loop** that walks topo order each tick, picks
  ready operators, hands morsels to the scheduler
- **Phase barriers** at pipeline breakers (existing
  `PhaseBarrier` from `bolt_scheduler.h`)
- **Source-driven cadence** — source operators are the only thing
  that kicks off work; downstream operators are reactive

## What we skip

- **Pull / Volcano `next()`** — vtable / templated callback overhead
  at the operator boundary
- **Pure async (no executor)** — loses NUMA pinning, phase barriers,
  ordered dispatch
- **Explicit credit messages** — implicit credit via the edge ring
  is enough
- **Coroutine-based dispatch** — appealing but compiler support is
  uneven on MSVC; we don't need it for the shape we want

## Followups

- Decide whether the executor walks topo order in the same thread
  every tick, or rotates across worker threads
- Measure executor-loop overhead at 100k ticks/s with 50-operator
  graphs; if > 1 µs per tick, look at incremental ready-set tracking
  instead of full topo walk
- Decide whether pull-style "request batch from source" makes sense
  for sources with finite backpressure (e.g. file scans where the
  source can produce indefinitely)

## References

- Apache Arrow Acero `ExecNode` push model
- tremor.rs pipeline DSL — pure async push
- LMAX Disruptor — push with sequence-number-based credit
- DuckDB's pipeline-breaker model and phase barriers
- Velox push execution
