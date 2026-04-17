# Polars streaming engine

## Context

Polars is a Rust DataFrame library that beats Pandas by 5–30× and is
competitive with DuckDB on many analytical workloads. Its execution
engine has two modes: **eager** (run immediately, like Pandas) and
**lazy** (build an expression DAG, optimize, then execute). The lazy
engine has a **streaming** execution mode introduced in 0.18 (2023)
that processes data in morsels with bounded memory. Worth studying
because Polars solves the same problem Bolt's dataflow runtime aims
at: composable kernels with no per-op tax, with measured wins.

## Architecture

### Lazy expression DAG

Every operation on a `LazyFrame` returns a new `LazyFrame` representing
the transformation; nothing executes until `.collect()` is called.
The engine builds a logical plan tree, applies rewrites, then emits
a physical plan.

Key logical-plan optimizations:
- **Predicate pushdown** — filters move below joins, scans, aggregates
- **Projection pushdown** — only the columns that the final result
  needs are read from disk / produced by upstream
- **Slice pushdown** — `LIMIT N` propagates upstream so scans don't
  read more than necessary
- **Common subexpression elimination** — repeated expressions are
  computed once
- **Type coercion** — implicit casts inserted at the boundary, not
  the inner loop

These all happen on a logical IR before any execution; they're free
at runtime.

### Streaming execution mode

```python
df.lazy()
  .filter(col("price") > 100)
  .group_by("symbol")
  .agg(col("volume").sum())
  .collect(streaming=True)
```

In streaming mode, operators are organized into a pipeline of
**source → transform → sink** stages. The source emits morsels
(typically 50K rows). Each transform is a function `(morsel) ->
morsel`. Sinks accumulate into the final result.

Pipeline breakers (group_by, sort, join build) split the pipeline
into phases. Each phase is its own streaming pipeline; the breaker
acts as the sink for one phase and the source for the next.

### Work distribution: Rayon work-stealing

Polars uses Rayon's parallel iterator model. A morsel is split across
worker threads via `par_iter()`; each thread takes a chunk, processes
it, and the join point gathers results. No central executor — the
work-stealing pool handles dispatch implicitly.

For pipeline breakers (e.g. parallel hash join build), Polars
partitions the build side by hash, runs build in parallel, and
synchronizes at the start of probe.

## Key trade-offs Polars made

1. **Lazy is opt-in.** `eager` mode exists for Pandas-compat / REPL.
   Lazy is where the perf lives. Same pattern Bolt should follow:
   the dataflow runtime is opt-in for users who want a graph.
2. **Predicate / projection pushdown is at the logical-plan level**,
   not buried in execution. This means the IR has to be expressive
   enough to represent any operator's "what columns do I need" and
   "what predicates can I absorb." A flat physical-plan-only IR
   can't do this.
3. **Streaming uses bounded morsels** — typically 50K rows per
   morsel. The trade is morsel-overhead-per-batch vs. memory
   footprint. Polars defaults to ~50K because their measurements
   showed it sits below L2 for typical schemas.
4. **No virtual operators.** Rust enums + `match` for operator
   dispatch. Equivalent to our POD + tag-dispatch model — same
   performance properties, same compile-time guarantee.

## What we adopt

- **Pushdown lives in the optimizer (chukonu), not the runtime.**
  Bolt's dataflow runtime gets a finalized physical plan; pushdown
  rewrites happen before plan compilation.
- **Morsel size as a first-class tunable** — Bolt already has
  this via `Scheduler::grain_bytes`. Polars' 50K row default is a
  useful anchor for our typical schema shapes (reproduce that for
  TPC-H).
- **Streaming = pipeline of (source → transforms → sink) phases**,
  with breakers splitting phases. This matches our phase-barrier
  model exactly.
- **CSE / type coercion happen at logical plan**, not in operators.

## What we skip

- **Rayon work-stealing** — Bolt has its own scheduler with
  morsel-driven parallelism; same idea, different implementation.
  No reason to depend on Rayon (and we don't have Rust anyway).
- **Eager mode equivalence** — there's no "eager Bolt." The dataflow
  runtime is the only execution path; if someone wants one-shot
  computation, they build a single-operator graph.
- **Spill-to-disk for OOM** — Polars spills sort and group_by build
  state to disk under memory pressure. Bolt's hard upper bounds
  rule means we pre-allocate; if we exceed, we abort. (Marbledb
  handles persistence; the runtime doesn't spill.)

## Followups

- Steal Polars' specific predicate-pushdown rule set when chukonu's
  optimizer is designed (Wave 8)
- Measure morsel-size sensitivity for our actual workload — confirm
  or refute Polars' 50K-row default for our typical schemas

## References

- Polars docs, "Streaming" chapter:
  https://docs.pola.rs/user-guide/concepts/streaming/
- Polars source: `crates/polars-pipe/` (streaming engine)
- "Polars: A new DataFrame library" — Ritchie Vink talks
- arrow2 fork rationale (Polars' decision to fork arrow-rs)
