# Bolt — Columnar Execution Core

Zero-dependency, header-heavy C++20 execution substrate underneath MarbleDB,
Chukonu, and BoltAPI. An alternative to Apache Arrow on the hot path, built
around arena lifetime and branchless kernels; the Arrow C Data Interface is
used at egress boundaries.

**It builds on Windows out of the box.** Point MSVC at the headers and go —
no vcpkg, no conan, no protobuf/gRPC/thrift/Boost toolchain, nothing to
prebuild. That is the headline: getting Arrow C++ compiling on Windows is a
multi-hour, multi-gigabyte ordeal, and Bolt skips it entirely because it is
just C++20 standard-library headers.

## Why Not Arrow

1. **Build cost — the big one.** Arrow C++ on Windows pulls in a heavy
   toolchain (vcpkg, protobuf, gRPC, thrift, Boost) and a long prebuild. Bolt
   is a collection of C++20 headers that MSVC compiles directly, so it works
   on Windows out of the box with nothing to install or prebuild.

2. **Runtime overhead.** Arrow uses `std::shared_ptr` for every Array, Buffer,
   and RecordBatch — an atomic refcount on every transit. Bolt uses arena
   allocation + epoch-based lifetime, avoiding that per-transit atomic
   traffic.

3. **No mutation path.** Arrow buffers are immutable; modifying a column
   requires a full copy. Bolt uses Venus tick-tock COW: first write copies,
   all subsequent writes within the epoch are free.

4. **No adaptive encoding.** Arrow has one physical representation per type.
   Bolt supports Flat, Constant, Dictionary, Sequence, and View formats. A
   constant-valued column becomes a single scalar — aggregation is a multiply,
   not a loop.

5. **No inline statistics.** Arrow columns carry no min/max or cardinality.
   Bolt `ColumnStats` enables zone-map skipping, join-strategy selection, and
   micro-adaptive kernel dispatch.

## Arrow Compatibility

Bolt is not anti-Arrow. It produces zero-copy Arrow views via the Arrow C Data
Interface (`ArrowSchema`/`ArrowArray`) with no libarrow link. Any Arrow consumer
(Polars, DuckDB, Pandas, PyArrow) reads Bolt columns directly. Conversion to the
full Arrow C++ representation is available at I/O boundaries when libarrow is
present.

## Performance

Bolt is faster than Arrow for the workloads it was designed for: allocation,
inter-operator transit, filtering, and constant-column scans. The gains come
from the architecture, not from tuning:

- **Arena allocation instead of per-buffer malloc.** Buffers are bump-allocated
  from a preallocated arena and freed in one pointer move, so there is no
  per-allocation malloc/free traffic on the hot path.
- **Epoch swaps instead of atomic refcounts.** Batches move between operators
  by flipping an epoch index rather than incrementing a `shared_ptr` refcount,
  removing an atomic on every transit.
- **Selection vectors and constant folding instead of materializing batches.**
  A filter emits a selection vector rather than a new batch, and a
  constant-valued column collapses a scan into a single scalar operation — both
  avoid touching every row.

It's early days, and the point above is architectural: these are the reasons
Bolt is fast for its target workloads, not a claim about any particular
benchmark result.

## What's in the Box

Headers live under `include/bolt/`. All are compiled as part of the consumer
build — there is no separate Bolt compile step.

### Core execution

| Header | What it provides |
|--------|-----------------|
| `bolt_types.h` | Type enum, StringView, Schema, Arrow C Data ABI structs |
| `bolt_arena.h` | Bump allocator (~2.6 ns/alloc), ArenaGuard, `tl_arena` thread-local |
| `bolt_arena_ring.h` | Ring-of-arenas for multi-generation scratch allocation |
| `bolt_channel.h` | Lock-free SPSC/MPSC ring buffers, cache-line padded |
| `bolt_column.h` | BoltColumn (Flat/Constant/Dict/Seq/View), ColumnStats, BoltBatch COW, BitmapIndex |
| `bolt_branchless.h` | Filter/aggregate/hash/gather kernels, micro-adaptive dispatch, predicated partition |
| `bolt_scheduler.h` | TaskRing, TaskPool, WorkerConfig, PhaseBarrier |
| `bolt_swissmap.h` / `bolt_hash.h` | SwissTable, swiss_mix, FNV-1a, wyhash |
| `bolt_ebr.h` | Epoch-Based Reclamation (lock-free snapshot lifetime) |
| `bolt_disruptor.h` | LMAX-Disruptor-style bounded ring for cross-thread handoff |
| `bolt_seqlock.h` | Seqlock for read-heavy shared state |
| `bolt_ribbon.h` | Ribbon filter (compact Bloom alternative) |
| `bolt_zonemap.h` | Zone-map (min/max block skip) |
| `bolt_hot_key_cache.h` | Hot-key row cache (bounded, lock-free) |
| `bolt_lock_free_clock_lru.h` | Clock-hand approximate LRU |
| `bolt_port.h` | `BOLT_RESTRICT`, `BOLT_FORCE_INLINE`, `BOLT_PAUSE`, etc. |
| `bolt_topology.h` | CPU topology, core affinity helpers |
| `bolt_variant_column.h` | Heterogeneous column variant |
| `bolt_row_view.h` | Zero-copy row view over a BoltBatch |
| `bolt_work_stealing_deque.h` | Chase-Lev work-stealing deque |

### Extended libraries

| Directory | What it provides |
|-----------|-----------------|
| `ingest/` | Parquet write (`bolt_parquet_write.h`), Parquet read (`bolt_parquet_read.h`), Avro (`bolt_avro.h`), Roaring bitmap (`bolt_roaring.h`), CSV, codecs (gzip/lz4/snappy/zstd) |
| `lakehouse/` | Delta + Iceberg readers/writers; object stores (local FS, S3, Azure Blob, GCS); parallel scan + scan optimizer; REST catalog fleet: Iceberg REST (OAuth2/SigV4/pagination/multi-table-commit/views/snapshots/branches-tags/stats/vended-creds), Unity Catalog, AWS Glue, Hive Metastore (Thrift), Polaris, Nessie, Gravitino |
| `crypto/` | Noise XX (X25519 + ChaCha20-Poly1305), Ed25519 sign/verify/keygen, SigV4 |
| `net/` | `bolt_tls.h` (OpenSSL TLS socket), `bolt_http_client.h` (outbound HTTP/HTTPS) |
| `kernels/` | Numeric, string, temporal, hash, sort, join (Swiss/HashJoin/GroupBy), SIMD, fintech (microstructure, volatility, risk, liquidity, cross-asset) |
| `parse/` | `fionn` JSON parser (`bolt::parse`) — zero-copy, arena-allocated |
| `wire/` | Bolt wire format for cross-process column transport |
| `stream/` | Streaming column utilities |
| `join/` | Build/probe helpers for hash and sort-merge join |
| `io/` | Async I/O primitives |
| `doc/` | Document column format |
| `ybolt/` | ycpp/Yjs CRDT runtime binding using BoltArenaAllocator (`bolt::ybolt`) |

## CMake Integration

Bolt exposes two targets: `bolt::bolt` (the main static library) and
`bolt::ybolt` (the ycpp Yjs binding — enabled when `extern/ycpp` is present).
Consumers guard with `if(NOT TARGET bolt::bolt)` so the shared bolt wiring in
Gestalt2's top-level CMake prevents duplicate definitions.

```cmake
add_subdirectory(extern/bolt)
target_link_libraries(your_target PRIVATE bolt::bolt)
```

## Build

Bolt is built as part of any consumer. To build and test standalone:

```bash
cmake -S . -B build -DBOLT_BUILD_TESTS=ON -DBOLT_BUILD_BENCHMARKS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Presets: `release`, `debug`, `msvc`, `ninja-msvc`, `clang-cl`.
GTest is fetched via `FetchContent` if not found — the build is self-contained
on Windows without vcpkg.

## Design Rules

Every line of code in `bolt/` follows these rules:

- **No exceptions.** All functions are `noexcept`. OOM returns `nullptr`.
- **No RTTI.** No `dynamic_cast`, no `typeid`, no virtual functions.
- **No smart pointers.** Raw pointers; arena-managed lifetime.
- **No `std::string`.** Fixed-size `char[64]` field names; `StringView` for data.
- **No `std::vector` in hot structs.** Fixed-capacity arrays.
- **No heap on the hot path.** Arena bump allocation only; malloc only at init.
- **Cache-line padded atomics.** All shared state on separate 64-byte lines.
- **Branchless inner loops.** Predicated execution, CMOV, SIMD masking.
- **`BOLT_RESTRICT` on all kernel parameters.** Enables auto-vectorization.
- **Compile-time type dispatch via X-macros.** One runtime switch at the boundary;
  fully specialized template in the inner loop.

## Research Foundation

- Pirk et al. "Database Cracking: Fancy Scan, Not Poor Man's Sort!" (DaMoN 2014) — predicated branchless partitioning
- Pirk et al. "CPU and Cache Efficient Management of Memory-Resident DBs" (ICDE 2013) — cache-conscious layouts, prefetch
- Pirk et al. "Voodoo: A Vector Algebra for Portable DB Performance" (VLDB 2016) — composable vector operation algebra
- Mohr-Daurat, Sun, Pirk. "BOSS: Database Kernel Composition" (VLDB 2023) — Arrow-compatible kernel exchange, near-zero overhead
- Pearce, Mohr-Daurat, Pirk. "White-Box Micro-Adaptive Query Processing" (ICDE 2025) — runtime kernel selection on observed selectivity
- Theodorakis et al. "LightSaber: Window Aggregation on Multi-core" (SIGMOD 2020) — parallel aggregation tree with SIMD sub-chunking
- Kersten et al. "Compiled and Vectorized Queries" (VLDB 2018) — vectorized vs compiled: <2× difference, vectorized is simpler

## Portability

Builds on Windows (MSVC / clang-cl — never MinGW), macOS, and Linux.
No vcpkg, no conan, no external dependencies in the core.
OpenSSL (for `net/` and `crypto/`) is the one accepted external runtime dep.

## Related

- `../marbledb` — HTAP storage engine built on Bolt
- `../chukonu` — distributed query engine executing Bolt operator graphs
- `../boltapi` — HTTP/WS/SSE framework using Bolt primitives
- `docs/` — design notes, research catalogue, decision log

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
