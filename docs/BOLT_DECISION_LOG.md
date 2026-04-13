# Bolt Decision Log

## Why each design decision was made, with references.

---

### Decision 1: Arena allocation instead of malloc/shared_ptr

**Problem:** Arrow uses `std::shared_ptr` for every buffer, requiring atomic
increment/decrement on every copy/destroy. In a pipeline with 8 operators
processing 100K batches/sec, that's 1.6M atomic ops/sec pure overhead.
Arrow's `MemoryPool` wraps `malloc` which contends across threads.

**Decision:** Per-thread bump allocators (arenas) that reset per morsel epoch.
Allocation is pointer arithmetic (~3ns). Deallocation is a pointer reset (~5ns).
No per-object refcounting. No cross-thread allocator contention.

**Measured:** 9,600x faster than malloc+free for 16KB allocations.

**Precedent:** Game engine frame allocators (Venus), Chronicle Queue off-heap
allocation, LMAX Disruptor pre-allocated ring buffers.

---

### Decision 2: Double-buffered clone-on-write (Venus tick-tock)

**Problem:** Arrow buffers are immutable. Any mutation requires allocating a
new buffer, copying all data, modifying, then freezing. For CDC/streaming
workloads modifying 3 of 20 columns per batch, this copies 17 columns
unnecessarily.

**Decision:** Two physical buffers per batch. Read from epoch 0, write to
epoch 1. First write to a column triggers a single memcpy (clone-on-write).
Subsequent writes within the epoch are direct pointer writes. Epoch swap
is an index flip (~1.3ns).

**Measured:** Dirty mask COW for 3/20 columns: 40.5ns (vs 76.5ns Arrow
equivalent of 3x new array + RecordBatch reconstruction).

**Precedent:** Venus ECS `entity_db` with `MARK_DIRTY` macro, game engine
double-buffered simulation state.

---

### Decision 3: Multi-format columns (Flat/Constant/Dict/Sequence/View)

**Problem:** Arrow has one physical representation per logical type. A column
where all 16K values are 42 still stores 16K × 8 bytes. A dictionary-encoded
column must be materialized to flat before most compute kernels can operate.

**Decision:** Five physical formats that the execution engine operates on
directly. A constant column is a single value + length. An aggregation over
a constant column is a single multiply, not a 16K-element loop.

**Measured:** Constant column scan: 0.7ns (vs 1,783ns flat scan). 2,500x.

**Precedent:** DuckDB Vector (Flat, Constant, Dictionary, Sequence, FSST),
QuestDB Symbol type (dictionary + optional bitmap index).

---

### Decision 4: Inline column statistics (64-byte ColumnStats)

**Problem:** Arrow IPC has column stats in serialized metadata, but
in-memory Arrays carry nothing. Every filter/join scans blind. There's
no way to skip a morsel because all values are below the filter threshold.

**Decision:** Every BoltColumn carries a 64-byte ColumnStats block inline:
min/max (zone map), null_count, distinct_count, cardinality class, sort
order, string-specific stats. Computed on ingestion, propagated through
filter/project. One cache line, always present.

**What it enables:**
- Zone map skip: `WHERE price > 100` on max=50 → skip entire morsel
- Join strategy: low cardinality → hash join, sorted → merge join
- Null fast-path: `all_valid` → skip validity bitmap checks
- Encoding promotion: `distinct_count == 1` → auto-promote to Constant

**Precedent:** DuckDB zone maps, Parquet row group statistics, QuestDB
symbol capacity/cache hints.

---

### Decision 5: Branchless inner loops with micro-adaptive dispatch

**Problem:** Branch mispredictions cost 10-20 cycles. At 50% selectivity
(worst case for predictors), a branching filter on 16K rows wastes
~120K cycles. But at extreme selectivities (<20% or >80%), the predictor
is accurate and branching is actually faster because the CPU can speculate
past the branch.

**Decision:** Two kernel variants for each operation: branchless (predicated
execution via `count += (data[i] > scalar)`) and branching (traditional `if`).
The dispatcher estimates selectivity from ColumnStats zone map and chooses
the appropriate kernel. This decision happens once per morsel per column,
not per row.

**Precedent:** Pirk et al. "Database Cracking: Fancy Scan" (DaMoN 2014) for
predicated execution. Pearce, Mohr-Daurat, Pirk "White-Box Micro-Adaptive
Query Processing" (ICDE 2025) for runtime kernel selection.

---

### Decision 6: Lock-free SPSC channels for inter-operator transit

**Problem:** Chukonu's MorselQueue uses a Vyukov bounded MPMC queue which
requires CAS operations on both push and pop. For linear pipeline stages
(one operator feeding the next), this is unnecessary — there's only one
producer and one consumer.

**Decision:** SPSC (single-producer single-consumer) ring buffer for linear
stages. No CAS on either side — producer owns `wpos_`, consumer owns `rpos_`.
MPSC for fan-in. Cache-line padded to prevent false sharing.

**Measured:** SPSC 17.2ns/op vs mutex queue 439ns/op. 25.5x faster.

**Precedent:** LMAX Disruptor, Venus job system ring buffer, Chronicle Queue
sequence-based publishing.

---

### Decision 7: Zero external dependencies

**Problem:** Arrow C++ on Windows requires ~200GB of toolchain (vcpkg,
protobuf, gRPC, thrift, boost, zlib, brotli, snappy, lz4, zstd, utf8proc,
re2, etc). Build times are hours. The dependency graph is fragile.

**Decision:** Bolt uses only C++20 standard library headers: `<cstdlib>`,
`<cstring>`, `<atomic>`, `<array>`, `<cassert>`, `<cstdint>`, `<cstddef>`.
Arrow interop is optional (`BOLT_ENABLE_ARROW_INTEROP`). The Arrow C Data
Interface structs are defined inline (they're just C structs, no library
needed). FasterAPI (vendored) provides networking when needed.

**What we give up:** Direct use of Arrow compute kernels (replaced by Bolt
kernels), Arrow Flight (replaced by Bolt Wire Protocol over FasterAPI),
Arrow Parquet reader (thin wrapper or own implementation).

---

### Decision 8: QuestDB-style sidecar indexes

**Problem:** Arrow columns have no attached acceleration structures. DuckDB
builds zone maps at the storage layer but doesn't carry them through the
pipeline. If a dictionary column is used in a WHERE clause, there's no
bitmap index to accelerate the lookup.

**Decision:** Optional sidecar index attachment points on BoltColumn: bitmap
index (for dictionary filter), bloom filter (for semi-join reduction), hash
index (for join probe), sort permutation (for order-by). All arena-allocated,
all freed automatically at epoch boundary.

**Precedent:** QuestDB Symbol INDEX (bitmap index on dictionary columns,
stored separately from data, optional per-column). Venus BH-tree (spatial
sidecar index, pre-allocated pool, reset per frame).

---

### Decision 9: German-style string views (Umbra/Polars)

**Problem:** Arrow's original string type stores all data contiguously with
an offset buffer. Appending, filtering, or gathering strings requires
copying variable-length data. High cache miss rate for long strings.

**Decision:** 16-byte inline string views. Strings ≤12 bytes are fully
inline (4 prefix + 8 data, no pointer chase). Longer strings store a 4-byte
prefix + buffer reference. Comparison: check length, then 4-byte prefix
(resolves >99% of distinct-string comparisons), then full compare only
if prefix matches.

**What it enables:**
- Filter/gather on strings is always a 16-byte copy, regardless of string length
- Prefix comparison resolves most equality checks without indirection
- Dictionary encoding uses views internally, no string data duplication

**Precedent:** Polars polars-arrow string type, DuckDB string_t, Arrow
StringViewArray (added to spec after Polars demonstrated the benefit),
Umbra/Hyper DBMS.

---

### Decision 10: Predicated dual-output partition (for hash join build)

**Problem:** Hash join build requires partitioning probe keys by hash into
buckets. The classic branching partition mispredicts at ~50% per bucket
boundary. For 8-way radix partition, every element hits a branch.

**Decision:** Predicated dual-output partition from Pirk (2014). Write to
BOTH output buffers speculatively, advance only the correct cursor. The
speculative write is harmless (overwritten next iteration) and the branch
predictor is never involved.

Extended to indexed variant (tracks original row positions for join output)
and multi-way radix scatter (pre-computed histogram + scatter loop).

**Precedent:** Pirk et al. "Database Cracking: Fancy Scan" (DaMoN 2014),
verified by Haffner et al. "Analysis and Comparison of Database Cracking
Kernels" (DaMoN 2018) which confirmed predicated is selectivity-independent
while branching is up to 7x slower at 50%.
