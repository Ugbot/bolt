# DuckDB vectorized push execution

## Context

DuckDB is the canonical reference for "embedded analytical query
engine done right." Push-based vectorized execution, 2048-tuple
vectors, multi-format columns, pipeline-breaker model, partition-on-
overflow. Many of Bolt's existing primitives (multi-format BoltColumn,
morsel-driven parallelism, branchless kernels) trace directly to
DuckDB. The dataflow runtime should explicitly track which DuckDB
patterns we adopt and which we deviate from.

## Architecture

### Vector class — multi-format columns

DuckDB's `Vector` carries a logical type plus one of five physical
formats:

| Format | Storage | Bolt equivalent |
|---|---|---|
| `FLAT` | Contiguous array | `BoltColumn::Flat` |
| `CONSTANT` | Single value + length | `BoltColumn::Constant` |
| `DICTIONARY` | (selection_vector, child_vector) | `BoltColumn::Dictionary` |
| `SEQUENCE` | (start, increment, count) | `BoltColumn::Sequence` |
| `FSST` | Compressed string format | (not in Bolt; future) |

Operators detect the physical format and dispatch to a specialized
inner loop. A `CONSTANT` column scan is a single multiply, not a
2048-element loop. We have parity here.

### Vector size: 2048 tuples (~16 KB for `INT64`)

DuckDB picked 2048 because it sits inside L1 cache for typical
schemas. Bolt uses 16384 by default (~128 KB for `INT64`) which
sits in L2, not L1. This is a deliberate divergence — Bolt targets
SIMD-bandwidth-bound kernels where larger morsels amortize loop
overhead. DuckDB targets pipeline-overhead-bound queries where
smaller batches keep latency low.

For tick-driven workloads (the dataflow runtime's target), small
batches matter more than large. **Followup:** measure what happens
to Bolt's per-tick latency at 2048 vs 16384.

### Push-based execution

Each operator implements:

```cpp
class PhysicalOperator {
    virtual SinkResult Sink(...);          // accept a chunk
    virtual SinkFinalizeResult Finalize(); // pipeline breaker complete
    virtual SourceResult GetData(...);     // produce next chunk
};
```

Source operators produce chunks; the executor pushes them down the
pipeline to sink. Pipeline breakers (`Sink` / `Finalize`) materialize
state for downstream. Streaming operators (`Filter`, `Project`) just
transform the chunk and pass through.

**Same shape as our planned executor** (push, source emits, downstream
operators react). DuckDB uses virtual functions; we use POD + fn
pointer to skip the vtable cost.

### Pipeline breakers and partition-on-overflow

DuckDB's pipeline-breaker categorization splits a query plan into
**pipelines**, each terminating in a sink. Pipelines run independently;
their state is consumed by downstream pipelines.

For breakers that may run out of RAM (hash join build, hash
aggregate), DuckDB **partitions on overflow**: when the build state
exceeds a threshold, it's hash-partitioned to disk; the probe side
gets the same partitioning, then probe runs partition-by-partition.

Bolt's "hard upper bounds, no spill" rule means we don't replicate
this. We pre-allocate the breaker's state; if it overflows, we abort
or escalate to a coarser-grained breaker. DuckDB's partition-on-
overflow is the right answer for unbounded workloads — Bolt's target
is bounded ones.

### Parallelism: morsel-driven

Each pipeline runs in parallel via morsel-driven scheduling. The
source operator's data is split into morsels (~120K tuples in
DuckDB); a thread pool claims morsels and walks them through the
pipeline. Pipeline breakers synchronize at their `Finalize`.

Bolt's `Scheduler::submit_range` is the same model. We have parity.

## Key trade-offs DuckDB made

1. **Vector size 2048 over 16K** — optimizes for L1 residency at
   the cost of more loop overhead. Right answer for OLAP, possibly
   wrong for SIMD-bound numeric kernels.
2. **Virtual operators** — every `Sink` / `GetData` call is a vtable
   dispatch. Mitigated by amortizing the call across 2048 tuples;
   per-tuple cost is irrelevant. Bolt's POD + fn pointer
   eliminates the cost entirely with no runtime trade-off.
3. **Partition-on-overflow** — costs implementation complexity but
   means DuckDB never OOMs on a sane query. Bolt skips this; if it
   matters later, marbledb-side spill is the answer (the storage
   layer has the right primitives, the runtime doesn't).
4. **Push-based execution** — picked over Volcano pull for the same
   reasons we are; this doc just confirms the choice.

## What we adopt

- **Multi-format Vector → multi-format BoltColumn** (already in
  place: Flat, Constant, Dictionary, Sequence; missing: FSST)
- **Push-based execution with pipeline breakers** (we already
  planned this in `push-vs-pull-dispatch.md`)
- **Per-pipeline state isolation** — the breaker's state is the
  pipeline boundary; downstream pipelines read it as immutable
- **Operator categorization driving scheduling** — same as our
  three-category scheme (streaming / accumulating / materializing)

## What we skip

- **Virtual operator dispatch** — POD + fn pointer wins
- **Partition-on-overflow** — Bolt does not spill; storage layer
  handles persistence concerns
- **2048-tuple vector size** — Bolt defaults to 16384; revisit per
  workload, especially for tick-driven where smaller may win
- **FSST string compression** — useful but adds complexity; deferred
  until measured-need shows up

## Followups

- Measure per-tick latency at vector sizes 1024, 2048, 4096, 8192,
  16384 once the dataflow runtime exists; pick the default by data
- Decide whether marbledb borrows DuckDB's partition-on-overflow
  pattern for its hash-aggregate spill (separate decision from
  the runtime)
- Port DuckDB's specific pipeline-breaker classification to chukonu
  when its optimizer is designed (Wave 8)

## References

- DuckDB design papers: Mühleisen & Raasveldt, "DuckDB: an Embeddable
  Analytical Database" (CIDR 2020)
- DuckDB source: `src/include/duckdb/execution/operator/`
- "DuckDB internals" talks by Mark Raasveldt
- DuckDB Vector class: `src/include/duckdb/common/types/vector.hpp`
