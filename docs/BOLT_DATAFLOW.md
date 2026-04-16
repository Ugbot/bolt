# Bolt Dataflow Runtime — Design

## Status

Wave 1 design draft. Synthesizes the Wave 1 research docs:
`docs/research/dataflow-operators.md`,
`docs/research/tick-tock-buffering-protocol.md`,
`docs/research/push-vs-pull-dispatch.md`,
`docs/research/gestalt-kernel-adapter.md`. Implementation lands in
Wave 2.

## Why this layer exists

Bolt's existing primitives — arena, channels, BoltBatch COW,
scheduler, branchless kernels — let a single kernel hit kdb+/QuestDB-
class numbers. But every benchmark hand-wires its multi-kernel
pipeline. Q6's 1.9 ns/row only happens because the bench code
manually fused four kernels into one inline loop. There is no
reusable composition layer.

The dataflow runtime is that layer. Goals:

1. **Compose kernels into a graph without paying the composition
   tax.** Operator-to-operator transit = epoch flip on a per-edge
   tick-tock ring buffer (~1.3 ns measured), not a `shared_ptr`
   refcount. Inner kernel call site is unchanged
   (`BOLT_RESTRICT T*, int64_t n, ...`).
2. **Tick-driven push dispatch.** Source operators are the only
   thing that initiates work. A standing graph survives many
   ticks; setup/teardown is amortized across the graph's lifetime.
3. **Stateful operators are first-class.** State is arena-pinned at
   compile time; persists across ticks; zero per-tick allocation.
4. **No perf regression** on existing Bolt benches. Framework
   overhead at the operator boundary ≤ 2 ns; inner-kernel numbers
   unchanged.

Non-goals (out of scope for this layer):

- SQL parsing, plan optimization, query rewriting (lives in chukonu)
- On-disk persistence, durability, recovery (lives in marbledb)
- Distributed coordination, network transport (separate concern)
- Runtime-pluggable operators (compile-time only; no vtables)

## Substrate primitives (recap, all already exist)

| Primitive | Location | Used for |
|---|---|---|
| `Arena`, `ArenaGuard`, `tl_arena` | `bolt_arena.h` | Per-morsel scratch; per-graph state pinning |
| `ArenaRing` | `bolt_stream.h:49-143` | Per-worker arena pool, reset-on-reuse |
| `SPSCChannel<T>` / `MPSCChannel<T>` | `bolt_channel.h` | Per-edge transport (template-generic) |
| `BoltBatch` (tick-tock COW) | `bolt_column.h:369-496` | Per-edge slot; epoch swap = publish |
| `BoltColumn` (multi-format) | `bolt_column.h` | Column data carried in batches |
| `Scheduler::submit_range` | `bolt_scheduler.h` | Morsel-parallel dispatch within an operator |
| `PhaseBarrier` | `bolt_scheduler.h` | Synchronize at pipeline-breaker boundaries |
| Kernels (`bolt::kernels::*`) | `bolt_numeric.h`, etc. | Inner loops; signatures unchanged |

The dataflow layer adds ~1500-2000 lines of new code, all in
`bolt/include/bolt/dataflow/`, that wires these together.

## Architecture

### Operator (POD)

```cpp
namespace bolt::dataflow {

constexpr uint8_t kMaxOpInputs  = 4;
constexpr uint8_t kMaxOpOutputs = 4;

struct Range {
    int64_t start;
    int64_t end;
};

using ExecuteFn = void (*)(
    const BoltBatch* const* in,
    BoltBatch* const* out,
    Range range,
    Arena* arena,
    void* state) noexcept;

enum class OpKind : uint8_t {
    Source, Sink,
    Filter, Project, Gather,
    HashJoinBuild, HashJoinProbe,
    GroupBy, GroupByMerge,
    Sort,
    Custom
};

struct OpFlags {
    static constexpr uint8_t kStreaming     = 0;
    static constexpr uint8_t kPipelineBreak = 1 << 0;
    static constexpr uint8_t kSource        = 1 << 1;
    static constexpr uint8_t kSink          = 1 << 2;
    static constexpr uint8_t kStateful      = 1 << 3;
};

struct alignas(64) Operator {
    uint16_t  input_edges[kMaxOpInputs];
    uint16_t  output_edges[kMaxOpOutputs];
    uint8_t   num_inputs;
    uint8_t   num_outputs;
    OpKind    kind;
    uint8_t   flags;
    void*     state;          // arena-pinned at graph compile
    ExecuteFn execute;
};

}  // namespace bolt::dataflow
```

POD struct, fixed-capacity edge arrays, function pointer for execute.
No vtable, no RTTI, no inheritance. Cache-line aligned because the
executor walks an array of these per tick.

### Edge — per-edge tick-tock chain buffer

```cpp
namespace bolt::dataflow {

constexpr uint8_t kEdgeStreaming     = 2;   // N=2 default
constexpr uint8_t kEdgeMaterialize   = 1;   // breaker -> stream
constexpr uint8_t kMaxEdgeBuffers    = 4;   // hard cap on N

enum class BackpressurePolicy : uint8_t {
    Block,
    DropNewest,
    Coalesce
};

struct alignas(64) Edge {
    BoltBatch slots[kMaxEdgeBuffers];   // up to N slots; only first N used
    uint8_t   num_slots;                // N (2 default, 1 for materialize)
    uint8_t   write_slot;               // producer cursor (mod N)
    uint8_t   read_slot;                // consumer cursor (mod N)
    BackpressurePolicy backpressure;
    uint16_t  producer_op;              // operator id producing into this edge
    uint16_t  consumer_op;              // operator id consuming
    alignas(64) std::atomic<uint64_t> dropped_count;
    alignas(64) std::atomic<uint64_t> coalesced_count;
};

}  // namespace bolt::dataflow
```

Each `Edge` owns its `BoltBatch` slots inline. Slot reuse = overwrite
in place. Producer flips `slots[write_slot].read_epoch` (release
store); consumer reads via acquire load. Same protocol as the in-batch
tick-tock from `bolt_column.h`, lifted one level up.

### Graph — topo-sorted, compile-once

```cpp
namespace bolt::dataflow {

constexpr uint16_t kMaxGraphOps   = 256;
constexpr uint16_t kMaxGraphEdges = 512;

struct Graph {
    Operator  ops[kMaxGraphOps];
    Edge      edges[kMaxGraphEdges];
    uint16_t  num_ops;
    uint16_t  num_edges;

    uint16_t  topo_order[kMaxGraphOps];
    uint16_t  num_phases;
    uint16_t  phase_starts[kMaxGraphOps];   // phase boundary indexes

    Arena*    state_arena;                  // owns operator state
};

}  // namespace bolt::dataflow
```

Graph is built once, never mutated during run. Topo sort happens at
compile (Kahn's algorithm). Phase boundaries marked at every
operator with `kPipelineBreak` flag. `state_arena` is the lifetime
arena for all operator state structs — never reset during run.

### Executor — tick-driven loop

```cpp
namespace bolt::dataflow {

struct ExecutorConfig {
    Scheduler*  scheduler;
    ArenaRing*  morsel_arenas;       // per-tick scratch
    uint32_t    morsel_grain_rows;   // typical: 4096
};

void execute_tick(Graph* g, ExecutorConfig* cfg) noexcept;

}  // namespace bolt::dataflow
```

Per-tick loop:

```
for phase p in g->num_phases:
    for op_idx in g->topo_order[g->phase_starts[p] .. g->phase_starts[p+1]]:
        op = g->ops[op_idx]
        if op.flags & kSource:
            morsel = source_emit_morsel(op)
        else:
            input_batches = collect_input_batches(op, g->edges)
            morsel = full_input_range(input_batches)
        cfg->scheduler->submit_range(
            op.execute, op.state, input_batches, output_batches(op),
            morsel, cfg->morsel_grain_rows, cfg->morsel_arenas);
        wait_until_op_complete(op)        # within-phase parallelism
        publish(output_edges(op))         # epoch flip
    phase_barrier_wait(p)                 # cross-phase sync
```

Single executor thread walks the topo order; the scheduler fans morsels
across worker threads inside one operator. Cross-operator parallelism
within a phase (operators that don't depend on each other) is a
followup — first version is sequential within phase, fully parallel
within operator.

### State pinning

State for stateful operators is allocated in `g->state_arena` at
graph compile, before the executor starts ticking. The state pointer
goes into `Operator::state`. The operator's `execute` casts it to
the concrete state struct.

```cpp
EMAState* s = arena_alloc<EMAState>(g->state_arena);
s->desc.value_col = 3;
s->desc.out_col   = 7;
s->alpha = 2.0 / (period + 1);
s->initialized = false;
op.state = s;
op.execute = &execute_ema;
op.flags |= OpFlags::kStateful;
```

State persists for the graph's lifetime — never reset between ticks.
Graph re-entry (e.g. re-running a query) calls `Graph::reset()`,
which zeroes state structs but keeps the arena allocation intact.

### Fusion

Compile-time only. Named fused kernels live in `bolt/kernels/`
(several already exist: `sum_masked`, `dot_masked`, the Q6 fused
loop). Plan compilation pattern-matches a known prefix of the
operator graph (e.g. `Filter | Gather | Mul | Sum`) and emits a
single `Operator` whose `execute` is the fused kernel.

We do NOT auto-fuse at runtime. Runtime decision = runtime overhead.

## Memory model and visibility

- Per-operator state: arena-pinned at compile, lives for graph's
  lifetime, never freed
- Per-tick scratch: morsel arena from `ArenaRing`, reset between
  morsels
- Per-edge slot: arena-tied, slot reuse = overwrite in place
- Cross-operator transit: release-store on producer's epoch flip,
  acquire-load on consumer's epoch read; no CAS, no seq_cst

No allocation in the steady state. No refcount traffic. No mutex.

## Backpressure

Per-edge policy, chosen at compile time:
- **Block** (default for streaming pipelines): producer spin-pauses
  on `BOLT_PAUSE`, escalates to `yield`
- **DropNewest** (telemetry sinks): drop overflow, increment counter
- **Coalesce** (stateful aggregators): merge new batch into queued
  batch via operator-defined merge fn

## Concurrency model

| Level | Parallelism |
|---|---|
| Per morsel | Yes — `Scheduler::submit_range` dispatches one operator's morsels across worker threads |
| Per operator within a phase | Single-threaded in v1 (sequential walk through topo order); full intra-phase parallelism is a Wave 13 followup |
| Per phase | Fully serial — `PhaseBarrier` between phases |
| Per graph | Single executor thread |
| Multiple graphs | Each gets its own executor and worker pool |

## Failure model

In-process only. No fault tolerance. No graceful recovery from a
worker thread crash. The `noexcept` discipline means kernels can't
throw; failures are signaled via return values where applicable
(currently most kernels are infallible).

## What this layer does NOT do

- **Does not parse SQL** — chukonu's job
- **Does not optimize plans** — chukonu's job
- **Does not persist data** — marbledb's job
- **Does not network** — separate transport module
- **Does not load operators at runtime** — compile-time only
- **Does not provide a REPL** — fixture / test code only

## Benchmarks planned (Wave 2)

- `bench_dataflow_overhead.cpp` — empty 5-operator graph, measure
  per-tick executor overhead. Target: ≤ 2 ns per operator boundary.
- `bench_tick.cpp` — 100k ticks/s × 64-row batches through a 4-op
  pipeline; report p50/p99/p999 per-tick latency. Target: p99 ≤ 5 µs.
- `bench_dataflow_q3.cpp` — TPC-H Q3 expressed as a dataflow graph;
  must match or beat current hand-wired Q3 (10.5 ns/row at 1M rows).
- `bench_ema.cpp` — EMA operator (the Tier 2 pilot from
  `gestalt-kernel-adapter.md`); expect 5-20× over the Arrow version.

Plus the existing Bolt benches (sum, filter_gt, swiss_find, Q6) must
not regress. This is the perf gate.

## Open questions to resolve in Wave 2

- Per-edge slot count: confirmed N=2 default; what's the rule for
  bumping to N=3+? Probably profile-driven on bursty workloads.
- Cross-operator parallelism within a phase: defer to Wave 13 or
  ship in v1? Ship sequential, optimize when bench data demands.
- Source operator API for tick-driven workloads — pull-from-queue
  vs callback-on-tick. Probably callback-on-tick (the source IS
  the tick).

## References

- `docs/research/dataflow-operators.md` — operator algebra
- `docs/research/tick-tock-buffering-protocol.md` — edge protocol
- `docs/research/push-vs-pull-dispatch.md` — executor model
- `docs/research/gestalt-kernel-adapter.md` — kernel port shape
- `docs/research/scheduler-design.md` — DuckDB / Polars / Seastar
  reference (predates this work; complements it)
- `bolt_column.h:443-449` — the in-batch tick-tock primitive that
  the per-edge protocol generalizes
