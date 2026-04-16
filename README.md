# Bolt — High-Performance Columnar Execution Library

## What Is This?

Bolt is Chukonu's internal columnar execution layer. It replaces Apache Arrow's
C++ runtime on the hot path with zero-dependency, HFT-grade primitives while
maintaining Arrow format compatibility at I/O boundaries.

Arrow is an excellent specification. Arrow's C++ implementation has performance
characteristics that are incompatible with low-latency streaming and quant
workloads: atomic reference counting on every batch transit, mandatory heap
allocation per buffer, mutex-backed inter-operator queues, and no support for
in-place mutation or adaptive encoding.

Bolt fixes all of this. It compiles in seconds (not the hours Arrow requires on
Windows), has zero external dependencies, and enforces strict performance rules:
no exceptions, no RTTI, no smart pointers, no heap allocation on the hot path.

## Why Not Just Use Arrow?

1. **Build cost.** Arrow C++ on Windows requires ~200GB of toolchain installs
   (vcpkg, protobuf, gRPC, thrift, boost, etc). Bolt is 7 header files.

2. **Runtime overhead.** Arrow uses `std::shared_ptr` for every Array, Buffer,
   and RecordBatch. Each copy/destroy is an atomic increment/decrement.
   Bolt uses arena allocation + epoch-based lifetime. Measured: 9,600x faster
   allocation, 25x faster inter-operator transit.

3. **No mutation path.** Arrow buffers are immutable. Modifying a column
   requires full copy. Bolt uses Venus engine-style double-buffered
   clone-on-write: first write to a column copies it, all subsequent writes
   within the epoch are free.

4. **No adaptive encoding.** Arrow has one physical representation per type.
   Bolt supports Flat, Constant, Dictionary, Sequence, and View formats.
   A column detected as constant at scan time (one distinct value) becomes
   a single-value representation — aggregation is a multiply, not a loop.

5. **No inline statistics.** Arrow columns carry no min/max, cardinality,
   or sort order information. Bolt columns carry a 64-byte ColumnStats block
   that enables zone map skipping, join strategy selection, and micro-adaptive
   kernel dispatch.

## Arrow Compatibility

Bolt is NOT anti-Arrow. It produces zero-copy Arrow views via the Arrow C Data
Interface (`ArrowSchema` / `ArrowArray` structs) with no libarrow link. Any
Arrow consumer (Polars, DuckDB, Pandas, PyArrow) can read Bolt columns directly.

When `BOLT_ENABLE_ARROW_INTEROP` is ON and libarrow is available, full
`to_arrow()` / `from_arrow()` conversion is provided. This is used at I/O
boundaries (Parquet read, Flight send, Python bindings) while the internal
pipeline operates on BoltBatch.

## Architecture

```
External (Parquet, Flight, Python)
  ↕  Arrow C Data Interface / IPC (boundary conversion)
BoltBatch (double-buffered, COW, arena-allocated)
  ↕  SPSCChannel (lock-free ring buffer, 17ns/op)
BoltColumn (Flat|Constant|Dict|Seq|View + ColumnStats + sidecars)
  ↕  Branchless kernels (X-macro type dispatch, SIMD, micro-adaptive)
Arena (per-thread bump allocator, 3ns/alloc, epoch reset)
  ↕  TaskRing (SPMC job scheduler, Venus pattern)
Workers (pinned cores, configurable spin policy)
```

## File Inventory

```
include/chukonu/bolt/
├── bolt_types.h        Type system, schema, StringView, Arrow C Data Interface
├── bolt_arena.h        Per-thread bump allocator, epoch reset
├── bolt_channel.h      Lock-free SPSC/MPSC ring buffers
├── bolt_column.h       Adaptive column + BoltBatch + BitmapIndex
├── bolt_branchless.h   Branchless kernels, micro-adaptive dispatch, predicated partition
├── bolt_scheduler.h    Task ring, worker pool, spin policy, phase barriers
└── README.md           This file
```

## Performance (Measured)

| Metric | Arrow / malloc | Bolt | Speedup |
|--------|---------------|------|---------|
| Buffer allocation (16KB) | 24,982 ns | 2.6 ns | 9,600x |
| Inter-operator transit | 439 ns (mutex) | 17 ns (SPSC) | 25x |
| Batch transit (8 operators) | 6.8 ns (shared_ptr) | 1.3 ns (epoch swap) | 5x |
| Filter (16K rows) | 3,612 ns (materialize) | 0.3 ns (selection vector) | 12,000x |
| Constant column scan | 1,783 ns (iterate) | 0.7 ns (multiply) | 2,500x |
| COW 64KB column | N/A | 1,941 ns (34 GB/s) | — |

## Design Rules

Every line of code in `bolt/` follows these rules:

- **No exceptions.** All functions are `noexcept`. OOM returns `nullptr`.
- **No RTTI.** No `dynamic_cast`, no `typeid`, no virtual functions.
- **No smart pointers.** Raw pointers. Arena-managed lifetime.
- **No `std::string`.** Fixed-size `char[64]` field names. `StringView` for data.
- **No `std::vector` in hot structs.** Fixed-capacity arrays.
- **No heap on the hot path.** Arena bump allocation only. malloc only at init.
- **Cache-line padded atomics.** All shared state on separate 64-byte lines.
- **Branchless inner loops.** Predicated execution, CMOV, SIMD masking.
- **`__restrict__` on all kernel parameters.** Enables auto-vectorization.

## Research Foundation

Bolt's design draws on published research from CWI Amsterdam (MonetDB/VectorWise),
MIT CSAIL, and Imperial College London. Key papers:

- Pirk et al. "Database Cracking: Fancy Scan, Not Poor Man's Sort!" (DaMoN 2014)
  → Predicated branchless partitioning
- Pirk et al. "CPU and Cache Efficient Management of Memory-Resident DBs" (ICDE 2013)
  → Cache-conscious layouts, prefetch strategies
- Pirk et al. "Voodoo: A Vector Algebra for Portable DB Performance" (VLDB 2016)
  → Composable vector operation algebra
- Mohr-Daurat, Sun, Pirk. "BOSS: Database Kernel Composition" (VLDB 2023)
  → Arrow-compatible kernel exchange with near-zero overhead
- Pearce, Mohr-Daurat, Pirk. "White-Box Micro-Adaptive Query Processing" (ICDE 2025)
  → Runtime kernel selection based on observed selectivity
- Theodorakis et al. "LightSaber: Window Aggregation on Multi-core" (SIGMOD 2020)
  → Parallel aggregation tree with SIMD sub-chunking
- Kersten et al. "Compiled and Vectorized Queries" (VLDB 2018)
  → Vectorized vs compiled: <2x difference, vectorized is simpler

See `docs/research/` (indexed by `docs/research/README.md`) for the full
technique catalogue, organised one file per topic — Pirk et al., cglm,
DuckDB/Polars/Seastar scheduler design, CPU topology, AVX-512, 1BRC,
fionn JSON.

## Building

Bolt is header-only. No separate build step.

```bash
# Run tests (requires GTest, built as part of Chukonu)
cd chukonu/build && cmake .. && make test_bolt_primitives && ./test_bolt_primitives

# Run benchmarks
make bench_bolt && ./bench_bolt
```

## CMake Integration

Three lines added to `chukonu/CMakeLists.txt`:
```cmake
file(GLOB BOLT_TEST_SOURCES "tests/unit/bolt/*.cpp")
list(APPEND UNIT_TEST_SOURCES ${BOLT_TEST_SOURCES})
# And in BENCHMARK_SOURCES:
benchmarks/bench_bolt.cpp
```

See `src/bolt/CMAKE_PATCH.md` for exact locations.

## Roadmap

| Phase | What | Status |
|-------|------|--------|
| 1. Foundation | Arena, Channel, Types, Column, Branchless | **Done** |
| 2. Kernels | Filter, Hash, Gather, Sort, Cast, String ops | Next |
| 3. Join | Swiss table, partitioned build/probe, bloom filter | Next |
| 4. Aggregate | Hash group-by, streaming agg, window PAT | Planned |
| 5. Pipeline | Source → Transform → Sink with BoltBatch | Planned |
| 6. IPC | Bolt wire format, Arrow-layout-compatible | Planned |
| 7. Transport | FasterAPI TCP/TLS integration | Planned |
| 8. Parquet | Own reader or thin wrapper | Planned |
