# Dataflow operators — algebra, fusion, pipeline breakers

## Context

Bolt's existing primitives (arena, channels, BoltBatch COW, scheduler)
let a single kernel hit kdb+/QuestDB-class numbers. But every benchmark
hand-wires its multi-kernel pipeline — Q6's 1.9 ns/row only happens
because the bench code manually fused four kernels into one inline loop.
The dataflow layer needs a reusable composition model that doesn't
re-introduce the costs the kernels just paid to remove.

This file fixes the operator algebra. Three questions: **what is an
operator (struct shape)**, **how do operators fuse**, **how are
pipeline breakers categorized**.

## Operator shapes — the four candidates

### Volcano / iterator

```cpp
class Operator {
    virtual void open() = 0;
    virtual Batch* next() = 0;        // pull from below; vtable dispatch
    virtual void close() = 0;
};
```

Used by every classic relational engine (System R, Postgres). Per-batch
virtual call kills inlining. Ruled out by Tiger Style "no virtual
functions."

### Cascades / Memo

Operator = node in a Memo group; physical operator chosen by cost rule.
Powerful for optimizer pluggability. Heavy machinery for the runtime
side — Memo lives in the optimizer (chukonu), not in dataflow. Dataflow
consumes the *output* of plan compilation.

### Push-stream (Acero / tremor / Kafka Streams)

Source operator generates batches; pushes via a downstream callback.
Pipeline operators are functions of `(input_batch) -> output_batch` plus
side effects on state. No control-flow inversion at the consumer side.

### POD + fn pointer (Bolt's choice)

```cpp
struct Operator {
    uint16_t input_edges[kMaxOpInputs];
    uint16_t output_edges[kMaxOpOutputs];
    uint8_t  num_inputs;
    uint8_t  num_outputs;
    uint8_t  kind;                 // tag for log/debug only
    uint8_t  flags;                // pipeline_breaker, source, sink
    void*    state;                // arena-pinned, per-graph-instance
    void   (*execute)(const BoltBatch* const* in, BoltBatch* const* out,
                      Range range, Arena* arena, void* state) noexcept;
};
```

POD struct, fixed-capacity edge arrays, function pointer for execute.
Zero vtable, zero RTTI, fits Tiger Style. The runtime is push-style:
the executor walks operators in topo order and calls `execute` on each
ready operator's morsel.

**Adopted:** push-stream with POD operator structs.

## Fusion classes

Fusion = inline two adjacent operators into one to skip an intermediate
materialization. Three categories worth distinguishing.

### Trivial (predicate / projection collapse)

`filter_a | filter_b` becomes `filter_combined(a && b)`. Pure compile-
time rewrite at plan-compilation. No runtime impact.

### Compute fusion (the Q6 win)

`gather(price, sel) | gather(discount, sel) | mul | sum` collapses to:

```cpp
for (i = 0; i < n; ++i) acc += price[sel[i]] * discount[sel[i]];
```

The four-stage version goes through three arena round-trips
(`gather_p`, `gather_d`, `product`). The fused version stays in
registers + L1. Measured: 6.6× on Q6 (12.5 → 1.9 ns/row).

**Implementation:** compile-time template specialization. We don't
auto-fuse at runtime — runtime decision = runtime overhead. Instead,
named fused kernels (`sum_masked`, `dot_masked`, `filter_sum_gt`) live
in `bolt/kernels/`. Plan compilation matches a known prefix of the
operator graph against a fused-kernel template and emits a single
`Operator` whose `execute` is the fused kernel.

### Pipeline-breaking — cannot fuse

Sort, hash-join build, blocking aggregate, top-k. These materialize
all input before producing any output. Their downstream operator runs
on the materialized state, not on a stream. Fusion is impossible by
construction.

**Adopted:** named fused kernels for the small set of hot patterns
(filter+sum, gather+arithmetic+sum, hash-probe+sum). Auto-fusion is
explicitly skipped — too easy to regress; cost of missed fusion is
bounded by per-batch overhead, not per-row.

## Pipeline breaker classification

Operators split cleanly into three categories that determine how the
runtime schedules them:

| Category | Examples | Schedule shape |
|---|---|---|
| **Streaming** | filter, project, gather, scan | Per-morsel; output ready as soon as one morsel processes; fits tick-driven dispatch |
| **Pipeline-breaking, accumulating** | hash-join build, hash-aggregate, sort | Consume all input then emit; downstream pipeline can't start until they finish |
| **Pipeline-breaking, materializing** | top-k, distinct, window-function | Like accumulating but with a bounded output state |

Streaming operators run inside a tick. Breaking operators force a
phase boundary — the executor synchronizes via the existing
`PhaseBarrier` from `bolt_scheduler.h`, then the next phase begins.

**Schedule rule:** an edge between two streaming operators uses an
N=2 tick-tock buffer (the Venus pattern). An edge crossing a phase
barrier uses an N=1 "materialize" slot — the breaker writes once, the
downstream operators read until phase end.

## Operator state

Most operators are stateless across morsels (filter, project, gather).
Some carry state across ticks within an operator instance (EMA, rolling
window, hash-aggregate's table, hash-join's build-side hash table).

State is **arena-pinned at graph compile time** and lives for the
graph's lifetime — no per-tick allocation. The state pointer goes
into `Operator::state`; the operator's `execute` casts it to the
concrete state struct. State structs are POD with fixed-size buffers
(rolling-window deques become fixed ring buffers with a `constexpr`
cap, per Tiger Style).

For stateful operators that also parallelize across morsels (parallel
hash-aggregate), each worker thread gets its own per-morsel partial
state, merged at the phase barrier. This pattern already exists in
`bolt::parallel_groupby_agg_int64` — the dataflow executor will
generalize it.

## What we adopt

- **POD operator structs** with function pointers; no vtables, no
  RTTI, no inheritance
- **Push-stream dispatch** — source produces, downstream operators
  invoked by executor as their inputs become ready
- **Named fused kernels** for known-hot patterns (filter+sum,
  gather+arithmetic+sum, hash-probe+sum); plan compilation pattern-
  matches and emits a single fused `Operator`
- **Three-category pipeline-breaker classification** drives edge
  buffer choice (tick-tock for streaming↔streaming, materialize-slot
  for breaker↔streaming)
- **Arena-pinned per-graph state** for stateful operators, pre-sized
  at compile time

## What we skip

- **Volcano `next()`** — virtual call per batch, no inlining
- **Cascades Memo at runtime** — Memo machinery belongs in chukonu's
  optimizer, not in dataflow
- **Auto-fusion at runtime** — runtime choice = runtime overhead;
  fusion is a compile-time template only
- **Operator registration / plugin loading** — runtime loadable
  operators need vtables; static registration only

## Followups

- Settle the exact set of named fused kernels worth shipping (the Q6
  evidence covers gather+arith+sum; Q1 / Q3 may want others)
- Decide whether morsel-parallel breaker merge (Phase 2) becomes a
  generic dataflow primitive or stays kernel-specific
- Operator state lifetime when a graph is reset / re-entered (e.g.
  re-running a query plan with different parameters) — does state
  zero out, or does the runtime track instance generations?

## References

- Apache Arrow Acero design notes (push-stream, ExecNode model)
- Graefe, "Volcano — An Extensible and Parallel Query Evaluation
  System" (1994) — for what we're choosing not to do
- DuckDB's pipeline-breaker classification (`PhysicalOperator::Type`)
- TigerBeetle's no-vtable rule
- Bolt benchmark history: Q6 fused 1.9 ns/row, evidence for compute
  fusion's 6.6× win
