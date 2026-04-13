# Bolt Research Notes: Pirk et al. Techniques

## Papers Relevant to Bolt

Holger Pirk (Imperial College, ex-CWI Amsterdam / MIT CSAIL) has published
extensively on exactly the problems we're solving. His work with Boncz, Manegold,
Kersten at CWI (the MonetDB/VectorWise group) and Madden, Zaharia at MIT covers
the full stack from cache-conscious layouts to composable kernel architectures.

Here are the key papers and what we should take from each.

---

### 1. Database Cracking: Fancy Scan, Not Poor Man's Sort (DaMoN 2014)

**Core finding:** Pivoted partitioning (cracking) is CPU-bound, not memory-bound,
because of branch mispredictions in the partition loop. At 50% selectivity,
branches mispredict ~50% of the time = maximum pain.

**Predicated crack-in-two:** Replace the branching partition with predicated
execution. Speculatively write to BOTH output partitions, only advance the
cursor of the correct one. The write to the wrong partition is harmless (it
gets overwritten next iteration).

```c
// Branching (bad at 50% selectivity):
if (data[i] < pivot)
    left[l++] = data[i];
else
    right[r++] = data[i];

// Predicated (branchless, selectivity-independent):
left[l]  = data[i];
right[r] = data[i];
l += (data[i] < pivot);     // 0 or 1
r += (data[i] >= pivot);    // 0 or 1
```

**Refined partition & merge:** Predict selectivity, pre-size the output slices
accordingly. If the prediction is correct, the merge phase moves almost no data.
If wrong, the merge is still correct, just slightly more expensive.

**What Bolt should take:**
- Our `filter_gt_branchless` already uses the single-output variant of this
  (speculative write + conditional advance). But for partitioning (hash join
  build, radix partition), we should use the dual-output predicated version.
- Selectivity estimation from `ColumnStats` can feed the refined partition
  slice sizing.

**Status in bolt_branchless.h:** Partially implemented (filter). TODO: add
predicated partition for hash join build.

---

### 2. CPU and Cache Efficient Management of Memory-Resident Databases (ICDE 2013)

**Core finding:** Cache-conscious data layout matters more than algorithmic
complexity for in-memory workloads. Accessing data that's already in L1/L2
is 10-100x faster than data from L3 or main memory.

**Key techniques:**
- **Partition data to fit working set in cache:** Process in chunks that fit
  L2 (256KB-1MB typically). This is exactly our morsel/vector-size decision.
- **Prefetch next chunk:** Software prefetch (`__builtin_prefetch`) 16-32
  elements ahead in random-access patterns (hash probe, gather).
- **Minimize pointer chasing:** Inline small data. This is the German string
  view motivation — strings ≤12 bytes inline, no pointer dereference.
- **Pack hot fields together:** Cache-line-conscious struct layout. Put
  frequently accessed fields (hash, key) at the start of the struct, cold
  fields (metadata, stats) at the end.

**What Bolt should take:**
- Our 2048-element vector size (from DuckDB) fits comfortably in L1 for most
  types (2048 × 8 bytes = 16KB, L1 is 32-64KB).
- The `gather_branchless` already prefetches. Extend to hash probe.
- `BoltColumn` stats are at the end of the struct (cold), data pointer is
  at the start (hot). Good.
- `StringView` is exactly the cache-conscious string representation.

---

### 3. Voodoo - A Vector Algebra for Portable Database Performance (VLDB 2016)

**Core idea:** Define an intermediate algebra of vector operations that maps
to different hardware backends (CPU scalar, CPU SIMD, GPU OpenCL). Instead of
writing N kernels × M backends = N×M implementations, write N kernels in
Voodoo algebra, the code generator produces M backend implementations.

**The algebra:**
- `map(f, v)` — apply f to each element
- `zip(f, v1, v2)` — element-wise combine
- `fold(f, init, v)` — reduction
- `select(mask, v)` — filter (our selection vector)
- `scatter/gather` — random access
- `partition(f, v)` — split by predicate (cracking)

**What Bolt should take:**
- We're not building a code generator, but the algebra maps directly to our
  X-macro kernel inventory. Each Voodoo operation = one Bolt kernel.
- The `select` operation in Voodoo IS our branchless selection vector.
- The insight that these operations are sufficient to express all analytical
  queries validates our minimal kernel set.

---

### 4. BOSS - An Architecture for Database Kernel Composition (VLDB 2023)

**Core idea:** Don't build one monolithic DBMS. Compose it from interchangeable
kernels (Arrow for storage, Velox for processing, ArrayFire for GPU). Use
partial query evaluation — a query goes through a sequence of stages, each
stage handled by whatever kernel is best suited.

**Key design principle:** The exchange format between kernels must be
virtually overhead-free. Arrow IPC is used but the paper shows that the
overhead of moving data between kernels (Arrow ↔ Velox ↔ ArrayFire) can
be kept to near-zero with careful format alignment.

**What Bolt should take:**
- This validates our Arrow C Data Interface export approach. BOSS demonstrates
  that kernel composition with Arrow-compatible formats works.
- But BOSS still uses Arrow as the storage/exchange format. We go further:
  BoltBatch IS the format, with Arrow views as an export mechanism.
- The partial evaluation idea maps to our pipeline: each operator partially
  evaluates the query, passing BoltBatch downstream.

---

### 5. LightSaber - Efficient Window Aggregation on Multi-core Processors (SIGMOD 2020)

**Core idea:** Window aggregation has a tension between parallelism (process
windows independently) and incremental computation (share work between
overlapping windows). LightSaber builds a Parallel Aggregation Tree (PAT) that
divides aggregation into intermediate steps enabling both SIMD and multi-core
parallelism, plus a Generalized Aggregation Graph (GAG) for work-sharing.

**Performance:** 470M records/sec with 132μs average latency on a 16-core server.

**What Bolt should take:**
- The PAT structure maps to our PhaseBarrier + TaskRing: divide the morsel
  into sub-chunks for SIMD, combine results per-thread, final merge across
  threads.
- The GAG for work-sharing between overlapping windows is relevant for
  Chukonu's streaming window operators (`time_window.h`).
- Key insight: SIMD is best applied at the innermost aggregation step (sum
  within a sub-chunk), not at the window management level.

---

### 6. SonicJoin: Fast, Robust and Worst-case Optimal (EDBT 2023)

**Core idea:** Traditional hash joins degrade catastrophically with skewed
keys (one value appearing millions of times). SonicJoin provides worst-case
optimal performance regardless of data distribution.

**What Bolt should take:**
- For our `bolt_hash_join.h`, we need a skew-handling strategy. Options:
  a) Detect skew via `ColumnStats::cardinality` and switch to SonicJoin-like
     partitioning for heavy hitters.
  b) Use bloom filter pre-filtering (already planned in sidecar slots) to
     reduce probe-side volume before the join.
  c) For the common case (low skew), partitioned Swiss table is fine.

---

### 7. White-Box Micro-Adaptive Query Processing (ICDE 2025)

**Core idea:** Instead of choosing one algorithm at query compile time,
dynamically switch between implementations within a single operator based
on runtime characteristics. E.g., switch from branching to branchless filter
based on observed selectivity.

**The micro-adaptation rule:**
- Below ~20% or above ~80% selectivity: branching scan is faster (branch
  predictor is accurate, and the branching version can skip elements)
- Between 20-80% selectivity: branchless/predicated is faster (branch
  predictor fails)
- This cross-over point depends on hardware and data type

**What Bolt should take:**
- Use `ColumnStats` zone map to estimate selectivity BEFORE choosing the
  kernel. If `min_value` and `max_value` suggest selectivity < 20% or > 80%,
  use the branching kernel. Otherwise, use branchless.
- This is a one-time decision per morsel per column, not per row. The stats
  are already computed. The overhead is a single comparison.

```cpp
// In dispatch_filter:
float estimated_selectivity = estimate_selectivity(col.stats, scalar);
if (estimated_selectivity < 0.2f || estimated_selectivity > 0.8f) {
    return filter_gt_branching(data, n, scalar, out);   // Predictor wins
} else {
    return filter_gt_branchless(data, n, scalar, out);  // Predication wins
}
```

---

### 8. DEPA - Delta Shifting and Distribution Shaping for Efficient Adaptive Indexing (ICDE 2025)

**Core idea:** Adaptive indexes (like database cracking) benefit from
reshaping the data distribution to make partitioning more predictable.
By "shifting" data by its delta from the mean, you can make the partition
boundaries more even, reducing worst-case behavior.

**What Bolt should take:**
- For MarbleDB's LSM compaction, data distribution awareness during merge
  could improve zone map effectiveness. If we know the distribution shape
  during compaction, we can choose partition boundaries that minimize
  zone map overlap between SSTable blocks.
- For Bolt's branchless partitioning: knowing the distribution shape lets
  us pre-size partition buffers accurately (Pirk's refined partition & merge).

---

### 9. High-Performance Tree Indices: Locality Matters More Than One Would Think (ADMS@VLDB 2020)

**Core finding:** B-tree variants optimized for cache locality (cache-conscious
B+-trees, CSS-trees) outperform theoretically superior structures. The key
factor is fitting each tree level in a cache line.

**What Bolt should take:**
- For our sidecar sort index, a cache-line-sized B-tree node (8 keys per
  64-byte node on int64) gives better sorted-search performance than
  binary search on a flat array for large morsels.
- But for morsel sizes ≤ 2048, linear scan with branchless filter is
  faster than any tree structure (cache-line fills dominate).

---

### 10. Thriving in the No Man's Land between Compilers and Databases (CIDR 2019)

**Core idea:** There's a design space between fully interpreted (VectorWise)
and fully compiled (HyPer/Neumann) query execution. The optimal point depends
on the workload. Short queries benefit from interpretation (no compile overhead).
Long queries benefit from compilation (amortize compile cost over many rows).

**The Tectorwise/Typer comparison (VLDB 2018 Kersten et al.):**
- Vectorized (Tectorwise) and compiled (Typer/HyPer) differ by at most ~2x
- Both are 10-100x faster than row-at-a-time (PostgreSQL)
- Vectorized is simpler to implement and easier to extend
- Compiled is better for complex expressions and deeply pipelined operators

**What Bolt should take:**
- We're building a vectorized engine (like Tectorwise/DuckDB), not a compiled
  engine (like HyPer). This is the right choice for Chukonu because:
  a) We need to support UDFs (Python, JS) which can't be compiled
  b) We need streaming workloads where queries don't change often
  c) Vectorized is simpler and we're a small team
  d) The performance difference is < 2x and can be closed with SIMD
- However, for hot-path expressions (TPC-H Q6 style: simple filters +
  aggregation), JIT compilation of the expression tree could give a boost.
  This is a Phase 3+ consideration.

---

## Summary: Techniques to Implement

| Technique | From Paper | Priority | Status |
|-----------|-----------|----------|--------|
| Predicated selection (branchless filter) | Cracking (2014) | P0 | **Done** |
| Predicated dual-output partition | Cracking (2014) | P0 | TODO |
| Micro-adaptive kernel selection | White-Box (2025) | P0 | TODO |
| Cache-sized vector chunks (2048) | CPU+Cache (2013) | P0 | **Done** |
| Software prefetch in gather/probe | CPU+Cache (2013) | P0 | **Done** |
| German string views (inline ≤12) | CPU+Cache (2013) | P0 | **Done** |
| Composable kernel algebra | Voodoo (2016) | P1 | Partial |
| Arrow C Data Interface exchange | BOSS (2023) | P1 | **Done** |
| Parallel Aggregation Tree (PAT) | LightSaber (2020) | P1 | TODO |
| Skew-aware join strategy | SonicJoin (2023) | P2 | TODO |
| Cache-line B-tree for sidecar index | Tree Indices (2020) | P2 | TODO |
| Distribution-aware partitioning | DEPA (2025) | P2 | TODO |
| Expression JIT (optional) | No Man's Land (2019) | P3 | Future |

---

## References

1. Pirk et al. "Database Cracking: Fancy Scan, Not Poor Man's Sort!" DaMoN 2014
2. Pirk et al. "CPU and Cache Efficient Management of Memory-Resident Databases" ICDE 2013
3. Pirk et al. "Voodoo - A Vector Algebra for Portable Database Performance" VLDB 2016
4. Mohr-Daurat, Sun, Pirk. "BOSS - An Architecture for Database Kernel Composition" VLDB 2023
5. Theodorakis et al. "LightSaber: Efficient Window Aggregation on Multi-core Processors" SIGMOD 2020
6. Khazaie, Pirk. "SonicJoin: Fast, Robust and Worst-case Optimal" EDBT 2023
7. Pearce, Mohr-Daurat, Pirk. "White-Box Micro-Adaptive Query Processing" ICDE 2025
8. Khazaie, Pirk. "DEPA - Delta Shifting and Distribution Shaping" ICDE 2025
9. Kowalski, Kounelis, Pirk. "High-Performance Tree Indices" ADMS@VLDB 2020
10. Pirk, Giceva, Pietzuch. "Thriving in the No Man's Land between Compilers and Databases" CIDR 2019
11. Kersten et al. "Everything You Always Wanted to Know About Compiled and Vectorized Queries" VLDB 2018
12. Polychroniou et al. "Rethinking SIMD Vectorization for In-Memory Databases" SIGMOD 2015
13. Theodorakis et al. "Scabbard: Single-Node Fault-Tolerant Stream Processing" VLDB 2021
14. Theodorakis et al. "SlideSide: Fast Incremental Stream Processing for Multiple Queries" EDBT 2020
