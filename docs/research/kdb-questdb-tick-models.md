# kdb+ and QuestDB — tick-driven models (the perf anchors)

## Context

The user's stated performance bar is "kdb+/QuestDB-class tick latency,
not Polars / DuckDB-class batch throughput." Both kdb+ (Kx Systems,
1993) and QuestDB (open source, 2014) are time-series databases
purpose-built for tick data — high-frequency append-only writes
plus low-latency queries over time ranges. Worth a dedicated
research doc because they are the explicit competitive target for
the dataflow runtime + marbledb.

## kdb+

### Architecture

kdb+ is a column-store with an embedded array language (`q`) acting
as both query language and stored procedure language. Tables are
just lists of columns; columns are just typed arrays. Memory layout
is identical to disk layout (mmap'd files), so reads are zero-copy.

Splay tables (`splayed` on disk) store one column per file in a
directory; partitioned tables (`partitioned`) split by date or hash
into per-partition directories. The combination — partitioned
splayed — is the canonical layout for tick data: one directory per
trading day, one file per column inside.

### Continuous queries

kdb+'s real-time engine is a separate process (`tickerplant`) that:
1. Receives updates from feed handlers
2. Logs to a write-ahead log
3. Distributes updates to one or more real-time databases (RDB)
4. End-of-day rolls the in-memory data to disk (becomes part of HDB)

Subscribers register continuous queries via `.u.sub`. The
tickerplant publishes updates to subscribers as they arrive. There
is no separate streaming engine — it's all just q functions
reacting to events.

### Why it's fast

- **Columns ARE files.** No serialization layer. mmap → read directly.
- **Append-only.** Writes are pure appends to per-column files.
  No B-tree maintenance, no compaction.
- **Time-as-implicit-key.** Queries like `select last price by sym
  from trade where date=2026.04.16, time within (09:30; 16:00)`
  use the partition layout to skip irrelevant data with no index
  consultation — the date is the directory name.
- **Vectorized q.** Every q operation is implicitly vectorized; the
  language doesn't have explicit loops.
- **Per-process simplicity.** No locks, no shared state across
  processes (RDB and HDB are separate processes); cache-friendly
  by construction.

### What we'd need to match it

- mmap-style zero-copy reads on hot data — marbledb's L0/L1 cache
  needs to deliver this
- Partition-by-date directory layout — marbledb's storage layout
  question (see `marbledb-v2/docs/research/lsm-vs-time-
  partitioned.md`)
- Continuous query model where update events trigger downstream
  operators — bolt::dataflow's tick-driven push model already
  fits this shape
- Per-tick latency in the sub-microsecond range when reading from
  L1-resident state — bolt::dataflow's overhead budget is ≤ 2 ns
  per operator boundary; this is the same number scaled to a
  modern CPU

## QuestDB

### Architecture

QuestDB is a SQL time-series database, written in Java (with
significant native code via JNI). Storage is column-oriented,
partitioned by time (day / month / year as configured), with a
**designated timestamp** column as the primary axis.

Each table is a directory of column files. Each column file is a
flat append-only buffer. Partitioning splits files by time range:
`mytable/2026-04-16/price.d` is the price column for April 16.
This is structurally identical to a kdb+ partitioned splayed
table — same idea, different language.

### Symbol type

A unique QuestDB feature: the **symbol** type is a string column
with automatic dictionary encoding. Internally a symbol is a
fixed-size integer index into a per-column dictionary. Queries
that filter or join on symbol columns operate on the integer
indices, not the strings. Bolt's `BoltColumn::Dictionary` is
the same idea.

### Out-of-order writes

QuestDB introduced out-of-order (O3) ingest in 4.x — writes
arriving with timestamps earlier than the current high-water mark
are merged into the existing partition rather than rejected. This
is critical for real tick feeds where messages can arrive out of
exchange order.

The merge happens in a background commit cycle: O3 writes go to a
per-partition staging buffer; commit merges them into the column
files. This is a streaming compaction shape — exactly the bug
marbledb's predecessor has (it loads everything into RAM instead
of streaming-merging).

### JIT-compiled filters

QuestDB compiles `WHERE` clauses to native machine code at query
time using its own JIT. For columnar scans, this skips the cost
of interpreter dispatch on each predicate evaluation. Bolt's
compile-time X-macro dispatch is the static-compile-time
equivalent — same trade, no JIT machinery needed.

### Tick-shaped benchmark

QuestDB's reference benchmark is **TSBS** (Time-Series Benchmark
Suite). Their headline: ingest at >1M rows/s/core sustained,
query a 1-day window in <100 ms on year-of-data tables. Bolt's
marbledb perf gates target the same ballpark.

## What we adopt (across both systems)

- **Time-partitioned directory layout** for marbledb's storage
  (one directory per day or hour, one file per column inside).
  This is the canonical tick-data layout; both systems use it for
  the same reasons.
- **Symbol / dictionary encoding** for low-cardinality string
  columns — Bolt has this (`Dictionary` format); marbledb's
  storage needs to preserve it on disk.
- **Designated-timestamp axis** as a first-class concept in the
  table definition — the storage layer should know which column
  is the time axis and partition / sort by it implicitly.
- **Streaming merge for out-of-order writes** — marbledb's
  compaction must do this; the v1 bug (load all into RAM) breaks
  it.
- **Continuous query as the natural shape** — kdb+'s tickerplant
  → RDB → subscriber chain is exactly the bolt::dataflow source
  → operator → sink chain. No new framework needed; just need to
  prove it works at the per-tick latency target.

## What we skip

- **q language** — Bolt's surface is C++ + chukonu's SQL. No
  embedded array language.
- **Process-per-component (kdb+ style)** — Bolt is in-process.
  Multi-graph isolation is via separate executor instances, not
  separate OS processes.
- **JIT-compiled filters** — compile-time X-macro dispatch
  achieves the same result without a JIT runtime.
- **Embedded SQL parser inside the storage engine (QuestDB style)**
  — chukonu owns SQL; marbledb only knows about typed scans.

## Per-tick latency target

Both systems publish sub-millisecond per-tick latency at high
load. The user's stated target ("100k ticks/s with p99 ≤ 50 µs
end-to-end through filter → project → aggregate → sink") is
inside this envelope and serves as the bolt::dataflow + marbledb
joint perf gate.

## Followups

- Measure marbledb's "directory of column files per day" layout
  vs. the v1 LSM SSTable model (research doc:
  `marbledb-v2/docs/research/lsm-vs-time-partitioned.md`)
- Decide whether to expose a "designated timestamp" concept in
  the marbledb schema or keep the storage layer fully agnostic
  to which column is time
- Pull TSBS down and benchmark marbledb-v1 against QuestDB to
  set a concrete delta target before the rebuild starts

## References

- kdb+ docs: https://code.kx.com/q/
- "Q for Mortals" (Jeffry Borror) — practical kdb+ tutorial
- QuestDB docs: https://questdb.io/docs/
- QuestDB out-of-order ingest blog post (2021)
- TSBS: https://github.com/timescale/tsbs (cross-system
  time-series benchmark)
