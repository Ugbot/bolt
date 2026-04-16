# Scheduler design — lessons from DuckDB, Polars, Seastar

Bolt's scheduler (`bolt_scheduler.h`) is the piece most exposed to
platform variance: core counts, NUMA topology, P/E-core hybrids. Three
production systems have already paid the tuition on this — picking the
pieces that survive the header-only + Windows-first constraint.

## DuckDB — morsel-driven parallelism

DuckDB inherits the morsel-driven model from Leis et al. (SIGMOD 2014,
https://db.in.tum.de/~leis/papers/morsels.pdf): a fixed worker pool,
one thread per physical core, pulls "morsels" (~100K tuples in the
paper) off a global dispatcher. The dispatcher assigns morsels to
workers with affinity for the morsel's NUMA node when possible.
DuckDB's concrete row-group size is 122,880 rows; within a row group
the engine operates on 2,048-row vectors. Worker count is set via
`SET threads = N` (defaults to `num_cores`).

Parallel hash-aggregate
(https://duckdb.org/2022/03/07/aggregate-hashtable):

```text
phase 1: each thread builds its own hash table, unpartitioned
phase 2: once any thread's table exceeds ~10 000 rows, switch to
         radix-partitioned local tables (group-hash → partition id)
phase 3: each thread merges a disjoint subset of partitions in parallel
         — no inter-thread sync during merge
```

- Pool: one `std::thread` per core, lifetime = process.
- Work distribution: central dispatcher, pull-based, cache-line-padded
  task ring.
- NUMA: morsels are NUMA-compatible (dispatcher can prefer local
  morsels) but DuckDB does not bind threads to nodes by default.
- Memory pressure: partition-on-overflow rather than spill-on-overflow
  for aggregates.

## Polars — streaming + Rayon work-stealing

Polars' new streaming engine
(https://docs.pola.rs/user-guide/concepts/streaming/, Orson Peters talk
https://pola.rs/posts/talk-polars-meetup-1-streaming-engine/) runs
morsels through a pipeline of operators executed on a Rayon thread
pool. Rayon provides work-stealing: idle workers steal pending tasks
from busy workers' deques — no central dispatcher. Morsel size is
cache-aware and tunable via the `POLARS_IDEAL_MORSEL_SIZE` environment
variable; the default targets L2. Queries are LazyFrame expressions
optimised (predicate/projection pushdown, CSE) before execution
(https://docs.pola.rs/).

- Pool: global Rayon pool sized to `num_cores`.
- Work distribution: per-worker deques + work-stealing — no dispatcher
  bottleneck, but no NUMA awareness either.
- Backpressure: bounded channels between operators; producers block
  (parking Rayon's worker, which then steals) when the channel is full.
- Per-operator parallelism flags let the planner disable parallelism
  for operators where serial is provably faster (tiny inputs, strictly
  ordered sinks).

## Seastar — shared-nothing reactor

Seastar (https://seastar.io/, ScyllaDB shard-per-core
https://www.scylladb.com/product/technology/shard-per-core-architecture/)
pins one reactor thread per physical core. Each reactor owns a slice of
memory ("shard") — no data is shared across shards; cross-shard work
uses explicit message passing. The runtime binds shard memory to the
local NUMA node at startup. I/O goes through io_uring (or AIO/epoll);
CPU work runs as futures/continuations on the reactor. Because shards
are pinned and statically sized, the model is hybrid-CPU neutral — you
assign P-cores and E-cores to different shard classes at boot and the
runtime never re-pins.

- Pool: one OS thread per physical core, pinned for life.
- Work distribution: per-shard run queue; cross-shard via
  `smp::submit_to`, lock-free MPSC ring.
- NUMA: first-class — memory, file cache, network queues all shard-local.
- Backpressure: semaphores on every async operation; continuation
  scheduling is cooperative, so a slow continuation starves its shard
  only (not the whole machine).

## Comparative table

| Concern                     | DuckDB                    | Polars (streaming)          | Seastar                     |
|-----------------------------|---------------------------|-----------------------------|-----------------------------|
| Worker-to-core binding      | Unpinned `std::thread`    | Unpinned Rayon pool         | Hard-pinned, one per core   |
| Work distribution           | Central dispatcher, pull  | Per-worker deque + steal    | Per-shard queue + message   |
| NUMA awareness              | Morsel-compatible, opt-in | None                        | First-class (memory + I/O)  |
| Hybrid CPU (P/E) handling   | None (OS scheduler picks) | None (OS scheduler picks)   | Static assignment at boot   |
| Backpressure / throttling   | Bounded pipeline buffers  | Bounded channels + parking  | Per-op semaphores + coroutines |

## Portable patterns Bolt adopts

- **DuckDB morsel sizing.**
  - What we borrow: the two-tier size — large morsels (~100K rows) for
    dispatch granularity, 2,048-row vectors for kernel granularity.
  - How it fits Bolt: `TaskRing` already carries `[row_begin, row_end)`
    ranges; the inner loop in kernels is already written against 2,048
    vectors. The ratio is the same.

- **DuckDB thread-local aggregates + radix merge.**
  - What we borrow: phase-1 unpartitioned local table → phase-2 radix
    partitions on overflow → phase-3 per-partition parallel merge.
  - How it fits Bolt: slots into `bolt_groupby.h` (Phase 3 of roadmap).
    Each worker's Arena owns its local table; merge partitions become
    independent `TaskRing` entries with no cross-thread sync.

- **Polars cache-aware grain.**
  - What we borrow: env-tunable morsel size with an L2-sized default.
  - How it fits Bolt: expose `BOLT_MORSEL_ROWS` (default 65,536 rows;
    covers 256KB for 4-byte columns — L2 on most x86 parts). Compile-
    time default, runtime override at `scheduler_init`.

- **Seastar NUMA binding.**
  - What we borrow: per-worker Arena whose backing pages are allocated
    on the worker's local NUMA node; tasks preferring local-node
    morsels.
  - How it fits Bolt: `WorkerConfig` gains an optional `numa_node`
    field; `Arena::init` uses `VirtualAllocExNuma` on Windows and
    `mbind`/`numa_alloc_onnode` (via `libnuma`) on Linux. Off by default
    — enabled when `GetLogicalProcessorInformationEx(RelationNumaNodeEx)`
    reports more than one node.

- **Opt-in thread pinning.**
  - What we borrow: Seastar's "pin for life" discipline, gated behind a
    config flag rather than forced.
  - How it fits Bolt: `WorkerConfig::pin_core` is a `uint16_t` index;
    when set, `scheduler_init` issues `SetThreadAffinityMask` /
    `pthread_setaffinity_np` / `thread_policy_set` and asserts success.
    Default is unpinned so tests and short-lived embeds don't need
    capabilities.

## Patterns Bolt deliberately skips

- **Full reactor model (Seastar).** Seastar's power comes from
  owning I/O too. Bolt is in-memory execution; the caller owns I/O and
  hands us `BoltBatch`es. Adopting a reactor would force every consumer
  to adopt futures, which breaks the header-only drop-in goal.

- **Rayon-style work-stealing dispatcher.** Work-stealing wins when
  morsels have highly skewed cost. Our kernels are nearly uniform over
  a morsel (branchless, fixed-width types), so a central pull-dispatcher
  with cache-line-padded ring is simpler and gives the same throughput
  without per-worker deques. Revisit only if measured imbalance > 10%.

- **Runtime thread-class switching.** Some engines re-pin workers
  between P and E cores based on observed load. This requires quiescing
  the pool (join all workers, re-pin, resume) which defeats the
  "allocate at startup, never during execution" Tiger Style rule. We
  assign at `scheduler_init` and stay put.
