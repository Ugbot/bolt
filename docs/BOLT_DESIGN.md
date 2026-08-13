# Project Bolt: Columnar Execution

## Design Document — Arrow Performance Gap Analysis & Optimization Strategy

**Status:** Draft — the original design rationale, kept as a historical
record. Current behaviour is described in
[`BOLT_PROJECT_MAP.md`](BOLT_PROJECT_MAP.md) and
[`BOLT_PERFORMANCE.md`](BOLT_PERFORMANCE.md).
**Date:** April 2026

---

## 1. Executive Summary

Apache Arrow provides an excellent columnar memory specification and broad ecosystem interoperability. However, its C++ implementation carries performance characteristics that are incompatible with low-latency streaming and quant workloads. This document identifies the specific gaps, cross-references them against approaches taken by Polars (arrow2 fork), DuckDB (custom vector format), and HFT systems (Chronicle Queue, LMAX Disruptor, Aeron), and proposes a concrete optimization path for Chukonu's execution engine.

The core thesis: **Arrow-the-spec is sound; Arrow-the-implementation is not designed for mutation-heavy, latency-sensitive workloads.** We keep the spec, replace the hot-path internals.

---

## 2. Arrow C++ Performance Gaps

### 2.1 Atomic Reference Counting via shared_ptr

Arrow objects are passed and stored using `std::shared_ptr` pervasively. Every copy performs an atomic increment, every destruction an atomic decrement + conditional delete. In a morsel-driven pipeline where a RecordBatch flows through N operators, that's 2N atomic operations per batch transit.

Measured cost: 6.8ns for 8x shared_ptr copies (Arrow worst case), 11.8ns with moves. Venus tick-tock epoch swap: 1.3ns.

### 2.2 Immutability as the Only Concurrency Model

Arrow buffers are frozen after construction. Mutation requires: allocate new buffer, copy, modify, freeze, wrap in new Array, wrap in new RecordBatch. Each step involves heap allocation and atomic refcount updates.

Venus uses clone-on-write via generation counters. MARK_DIRTY checks if the column has been cloned for this epoch. If not, single memcpy. All subsequent writes within the epoch are direct pointer writes.

### 2.3 Per-Operation Status Checking

Every Arrow API returns arrow::Result<T> or arrow::Status. This introduces a branch on every hot-path operation. HFT pattern: validate once at boundary, operate on raw pointers within.

### 2.4 No Arena/Epoch-Based Memory Management

Arrow's MemoryPool is a thin abstraction over malloc/jemalloc. No per-thread arenas, no epoch-based reclamation, no slab allocation. Every buffer allocation hits the global allocator.

### 2.5 No Selection Vector / Lazy Materialization

Arrow's compute kernels don't natively support selection vectors. Every Filter materializes a new array. Every Take allocates and copies. No composable filter chain.

### 2.6 Uniform Array Representation

Arrow has one array type per logical type. DuckDB has five physical representations (Flat, Constant, Dictionary, Sequence, FSST) and can operate on compressed representations directly.

### 2.7 String Type Inefficiency

Arrow's original StringArray uses contiguous data + offset array. Polars/Umbra German-style 16-byte views fix this. Arrow now has StringViewArray but Chukonu isn't using it pervasively.

---

## 3. Competitive Analysis

### 3.1 Polars — Pragmatic Arrow Fork

Forked arrow2 into polars-arrow. German-style string views ahead of spec. Hand-written Rust kernels with explicit SIMD. Rust ownership eliminates refcounting at compile time. jemalloc recommended (~25% improvement).

### 3.2 DuckDB — Custom Vector Format

Fixed 2048-tuple vectors (L1/L2 cache). Five physical vector types. PAX-oriented 120K-tuple row groups. Push-based vectorized execution. Buffer manager with strict limits and disk spill.

### 3.3 Chronicle Queue / LMAX Disruptor / Aeron

Zero allocation on hot path. Lock-free sequence-based data flow. Cache-line isolation (alignas(64)). Memory-mapped persistence. Per-thread buffers. 40us end-to-end at 99.99th percentile.

---

## 4. Venus ECS Patterns Applicable to Columnar Execution

The Venus entity_db is a double-buffered SoA columnar store. Its patterns map directly:

- **Tick-Tock Double Buffering**: Two column buffers, read/write flip, dirty bitmask, generation counters
- **Clone-on-Write per Column**: MARK_DIRTY copies only on first write per frame. 3 of 20 columns modified = 3 memcpy, not 20.
- **Deferred Structural Changes**: Lock-free SPSC ring buffer for create/destroy, processed at frame end
- **X-Macro Type Dispatch**: ENTITY_COLUMN_LIST generates per-column code at compile time

---

## 5. Architecture: Bolt Execution Layer

Bolt sits between Arrow (I/O, interop) and Chukonu operators (compute), taking over the runtime work Arrow does on the hot path.

Components:
- **BoltBatch**: Double-buffered RecordBatch replacement with COW
- **BoltColumn**: Multi-representation vector (Flat/Constant/Dictionary/Sequence/View)
- **Arena**: Per-thread bump allocator with epoch reset
- **SPSCChannel/MPSCChannel**: Lock-free ring buffers with cache-line padding
- **X-macro kernels**: Compile-time type dispatch for filter/hash/gather

---

## 6. Benchmark Results (Measured)

All benchmarks run with g++ -O3 -std=c++20 -march=native. These are
microbenchmarks — each times one operation in isolation, and the filter and
constant-scan rows compare different amounts of work rather than the same task
done faster (selection vector vs materialized batch; constant fold vs
per-row scan). Read them as evidence for the design choices, not as
end-to-end speedups.

| Metric | Arrow Baseline | Bolt Measured | Speedup |
|--------|---------------|---------------|---------|
| Buffer allocation (16KB) | 24,982 ns (malloc+free) | 2.6 ns (arena bump) | ~9,600x |
| Epoch swap | 6.8 ns (shared_ptr 8x copy) | 1.3 ns (index flip) | ~5x |
| Epoch swap (move path) | 11.8 ns (shared_ptr 8-op) | 1.3 ns | ~9x |
| COW dirty mask (3/20 cols) | 76.5 ns (3x new array+batch) | 40.5 ns | ~1.9x |
| Inter-operator channel | 439.1 ns (mutex queue) | 17.2 ns (SPSC ring) | ~25.5x |
| Filter materialization | 3,612 ns (16K rows copy) | 0.3 ns (selection vector) | ~12,000x |
| Constant column scan | 1,783 ns (iterate 16K values) | 0.7 ns (single multiply) | ~2,500x |
| COW memcpy 8KB column | N/A | 50 ns (163 GB/s) | -- |
| COW memcpy 64KB column | N/A | 1,941 ns (34 GB/s) | -- |
| COW memcpy 128KB column | N/A | 3,721 ns (35 GB/s) | -- |
| COW memcpy 2MB column | N/A | 148,871 ns (14 GB/s) | -- |

---

## 7. Migration Strategy

### Phase 1: Foundation (Weeks 1-2)
- Arena allocator with thread-local storage [DONE]
- SPSC/MPSC channels [DONE]
- GTest suite and benchmark [DONE]

### Phase 2: Batch/Column (Weeks 3-4)
- BoltColumn with Flat + Constant + View formats
- BoltBatch with double-buffering and COW
- from_arrow() / to_arrow() boundary converters
- X-macro type dispatch kernels

### Phase 3: Operator Integration (Weeks 5-8)
- Migrate FilterOperator to BoltBatch
- Migrate SwissJoinOperator build/probe
- Dictionary and Sequence column formats
- End-to-end TPC-H comparison

### Phase 4: Storage Integration (Weeks 9-12)
- MarbleDB MemTable stores BoltBatch natively
- German-style string views as default
- CDC ingestion throughput, P99 latency benchmarks

---

## 8. Open Questions

1. Vector size: DuckDB 2048 vs Chukonu 16384. Sub-vector processing?
2. FSST string compression during execution?
3. NUMA-aware channels (per-socket)?
4. GPU path for embedding inference (llama.cpp)?
5. Formal proof of SPMC sequence protocol correctness.
