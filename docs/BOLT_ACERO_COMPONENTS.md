# Bolt Acero-Level Components

## What Acero Actually Contains (and What We Replace With)

Acero is six things bundled together under one namespace. We need all six.
We just don't need Arrow's implementation of them.

---

## 1. Task Scheduler (AsyncTaskScheduler)

**What Acero does:** FIFO task queue with throttling, sub-schedulers, priority,
and external task registration. Tasks are `std::function` closures submitted
to Arrow's CPU thread pool. Throttled schedulers limit concurrency (e.g. write
node uses throttle=1 to avoid re-entrant calls).

**What Venus has:** Lock-free SPMC ring buffer job system (`jobs.c`). 16384-slot
ring, cache-line padded head/tail, CAS-based claiming, spin-then-yield backoff
(SPIN_BEFORE_YIELD=1000). Job pools (Treiber stack) for zero-alloc job data.
Range jobs with automatic grain-size subdivision. Entity system jobs that accept
(read_buf, write_buf, start, end, delta_time, thread_id) — *exactly* the
morsel-driven columnar pattern.

**What FasterAPI has:** Coroutine-based IODispatcher (Seastar-inspired). 1-2 event
loop threads (kqueue/epoll/io_uring) dispatch I/O events. N worker threads execute
coroutines. C++20 coroutines with awaitable read/write.

**Bolt replacement:**

```
bolt::TaskScheduler
├── WorkerPool          — N worker threads, pinnable to cores
│   ├── Per-worker Arena (reset per task)
│   └── Per-worker SPSCChannel<Task> (work-stealing deque later)
├── TaskRing            — Venus-style SPMC ring buffer for job dispatch
│   ├── 16384 slots, cache-line padded
│   ├── CAS claim (same as Venus jobs.c)
│   └── Spin-then-yield backoff
├── TaskPool            — Treiber stack object pool for Task structs
│   └── Zero-alloc in steady state
├── RangeTask           — Auto-subdivide [0,N) across workers by grain size
│   └── Same pattern as Venus job_submit_range()
├── ColumnTask          — RangeTask specialized for (BoltBatch, start, end)
│   └── Same pattern as Venus EntitySystemJob
├── PhaseBarrier        — Multi-phase execution (Pre/Sim/Post/Render)
│   └── Venus job_begin_phase() / job_end_phase()
└── IOBridge            — FasterAPI event loop integration
    └── Coroutine awaitables for async I/O (network, disk)
```

**Key design rule:** The TaskScheduler itself allocates nothing after init.
All task data comes from the TaskPool (Treiber stack). All intermediate
compute memory comes from per-worker Arenas. The ring buffer is pre-allocated.

### Spin Policy

Venus uses a simple spin-then-yield:
```c
while (no_work) {
    if (++spin > SPIN_BEFORE_YIELD) { uv_sleep(0); spin = 0; }
}
```

We want three levels:

```cpp
enum class SpinPolicy : uint8_t {
    BusySpin,     // Pure spin. Pin to core. Use for latency-critical (< 1μs response).
    SpinYield,    // Spin N times then yield(). Default for compute workers.
    ParkWait,     // futex/condition_variable wait. Use for I/O-bound or infrequent.
};

// Per-worker configurable:
struct WorkerConfig {
    uint32_t worker_id;
    int      cpu_affinity;    // -1 = no affinity
    SpinPolicy spin_policy;
    uint32_t spin_count;      // Spins before escalation (default 1000)
    Arena*   arena;           // Pre-assigned arena
};
```

BusySpin for the morsel pipeline hot path (latency-critical operators like
filter, project). SpinYield for join build phases. ParkWait for I/O tasks
(Parquet read, network send).

---

## 2. ExecNode / Push-Based Pipeline

**What Acero does:** ExecNode base class with `InputReceived()` (push),
`StartProducing()`, `StopProducing()`. Push-based: upstream nodes call
`output->InputReceived(batch)`. ExecBatch flows between nodes.

**What Chukonu already has:** `PipelineProcessor` with cooperative state
machine (`prepare()` → `work()`). Push-based via `pushInput()`/`pullOutput()`.
This is already better than Acero — cooperative scheduling gives backpressure
control and avoids deep recursion.

**Bolt replacement:**

```
bolt::PipelineNode (replaces ExecNode)
├── Source node  — produces BoltBatches (from Parquet, CDC, MarbleDB)
├── Transform node — pure function: BoltBatch → BoltBatch
├── Sink node    — consumes BoltBatches (to MarbleDB, Parquet, Flight)
├── Pipeline breaker — accumulates (join build, sort, aggregate)
│   └── Emits only after all input consumed
└── Connected via SPSCChannel<BoltBatch>

Cooperative executor drives nodes:
  for each node in pipeline:
    if node.output_channel has space AND node has input:
      node.work()

No recursion. No std::function. No virtual dispatch on hot path.
Node type known at pipeline compile time via variant/tag.
```

---

## 3. Swiss Table Hash Map

**What Acero does:** Custom hash table with 8-byte stamps per block,
SIMD-accelerated probe (AVX2: 4 probes at once), two-pass lookup
(fast path: branch-free, optimistic; slow path: handles collisions).
1024-element mini-batches. Maps composite keys to unique integer IDs.

**What Chukonu already has:** `SIMDHashTable` with linear probing + SIMD
comparison (NEON/AVX2). 12-byte slots (hash + ChunkedRowRef). 50% load
factor. Bloom filter support.

**Bolt replacement:**

```
bolt::SwissTable
├── Block-based layout (8 slots per block, like Acero)
├── 7-bit stamps per slot (from hash) for SIMD comparison
├── Two-pass probe: fast path (branch-free) + slow path
├── Arena-allocated (no malloc during build or probe)
├── Multi-key support via hash combine
├── batch_lookup(keys[], N, output_ids[]) — vectorized
└── No ExecBatch dependency — operates on BoltColumn directly

bolt::HashJoin
├── Build phase:
│   ├── Partition input by hash (4-8 partitions per core)
│   ├── Build SwissTable per partition (parallel, arena-allocated)
│   └── Optional bloom filter per partition
├── Probe phase:
│   ├── Hash probe keys
│   ├── Partition probe batch
│   ├── Probe each partition's table (parallel, thread-safe)
│   └── Emit matches via selection vector (deferred materialization)
├── Join types: Inner, Left, Right, Full, Semi, Anti, AsOf
└── All memory from Arena — freed at pipeline end
```

---

## 4. Row Table (packed row storage for joins)

**What Acero does:** `RowTableImpl` — packs columns into variable-width rows
for hash table storage. Uses 64-bit offsets (fixed in Arrow 20.0). Row format
is: [null bitmap | fixed fields | var-length offsets | var-length data].

**Bolt replacement:**

```
bolt::RowTable
├── Arena-allocated contiguous buffer
├── Fixed-width rows: columns packed in order, aligned to 8 bytes
├── Variable-width: separate data buffer + 32-bit offsets per row
│   └── Use 64-bit offsets only when data exceeds 4GB (rare in morsels)
├── pack_columns(BoltBatch, key_columns[]) → RowTable
├── unpack_row(row_id) → values[] (used for key comparison)
└── hash_row(row_id) → uint32_t (cached in SwissTable stamp)
```

---

## 5. Expression Evaluation

**What Acero does:** `Expression` tree with `literal`, `field_ref`, `call`
nodes. Evaluation via `ExecuteScalarExpression()` which calls Arrow compute
kernels. Used in Filter, Project, and Join conditions.

**What Chukonu already has:** Full expression evaluator in
`sql/expressions/expression_evaluator.cpp` and `evaluate_expression.cpp`.
Full SQL parser. Full optimizer.

**Bolt replacement:** Keep Chukonu's expression system but have it emit
operations against BoltColumn instead of `arrow::compute::CallFunction()`.
The X-macro kernel dispatch replaces the Arrow function registry lookup.

---

## 6. Aggregate Node

**What Acero does:** `HashAggregateNode` — groups by key, accumulates
aggregates (sum, count, min, max, mean, stddev, any, all). Two phases:
consume (accumulate into hash table) and finalize (emit results).

**What Chukonu already has:** `HashAggregateOperator`, `StreamingGroupBy`,
`BucketAggregation`, `AggregateOperator`, `AggregateSinkOperator`.

**Bolt replacement:**

```
bolt::HashAggregate
├── Group keys → SwissTable → group_id
├── Per-group accumulators (arena-allocated array):
│   ├── Sum: int64_t / double running total
│   ├── Count: uint64_t
│   ├── Min/Max: type-punned int64_t (same as ColumnStats.min_value)
│   ├── Mean: running sum + count
│   └── StdDev: Welford online algorithm (sum, sum_sq, count)
├── Constant column optimization:
│   └── group_count * constant_value (no per-row accumulation)
├── Finalize: emit one BoltBatch with group keys + aggregate results
└── All arena-allocated — zero malloc
```

---

## Component Dependency Graph

```
bolt_types.h          ← zero deps (enum, constexpr, static_assert)
    ↑
bolt_arena.h          ← zero deps (malloc/free in init/shutdown only)
    ↑
bolt_channel.h        ← zero deps (atomic, array)
    ↑
bolt_column.h         ← bolt_types.h, bolt_arena.h
    ↑
bolt_kernels.h        ← bolt_column.h (filter, hash, agg, gather, sort)
    ↑
bolt_swiss_table.h    ← bolt_arena.h, bolt_kernels.h (SIMD hash probe)
    ↑
bolt_hash_join.h      ← bolt_swiss_table.h, bolt_column.h (partitioned join)
    ↑
bolt_hash_agg.h       ← bolt_swiss_table.h, bolt_kernels.h (group-by agg)
    ↑
bolt_scheduler.h      ← bolt_arena.h, bolt_channel.h (task ring, worker pool)
    ↑
bolt_pipeline.h       ← everything above (Source → Transform → Sink)
    ↑
bolt_ipc.h            ← bolt_column.h (wire format read/write)
    ↑
[optional]
bolt_arrow_interop.h  ← bolt_column.h + <arrow/api.h> (to_arrow/from_arrow)
bolt_transport.h      ← bolt_ipc.h + FasterAPI (network RPC)
bolt_parquet.h        ← bolt_column.h + parquet reader (file I/O)
```

Everything above the `[optional]` line compiles with zero external dependencies.

---

## Implementation Priority

| Component | Source | Estimated LOC | Priority |
|-----------|--------|---------------|----------|
| bolt_types.h | New | 300 | **Done** |
| bolt_arena.h | New | 200 | **Done** |
| bolt_channel.h | New | 150 | **Done** |
| bolt_column.h | New | 400 | **Done** |
| bolt_kernels.h | Chukonu kernels.h + X-macro | 600 | P0 |
| bolt_swiss_table.h | Chukonu SIMDHashTable + Acero ideas | 500 | P0 |
| bolt_hash_join.h | Chukonu SwissJoin rewrite | 400 | P0 |
| bolt_hash_agg.h | Chukonu HashAggregateOperator | 300 | P1 |
| bolt_scheduler.h | Venus jobs.c pattern | 400 | P0 |
| bolt_pipeline.h | Chukonu PipelineProcessor | 300 | P1 |
| bolt_ipc.h | New (flat binary format) | 300 | P1 |
| bolt_arrow_interop.cpp | New (C Data Interface + optional Arrow link) | 200 | P1 |
| bolt_transport.h | FasterAPI integration | 300 | P2 |
| bolt_parquet.h | Wrapper or own reader | 500-2000 | P2 |

Total core library: ~3,500 lines of header-only C++20.
Compare: Arrow C++ is ~500,000 lines. Acero alone is ~50,000 lines.
