# Bolt Project Map

## Overview

Bolt is a zero-dependency columnar execution library embedded inside Chukonu.
It replaces Apache Arrow's C++ runtime on the hot path while maintaining
Arrow format compatibility at I/O boundaries.

Total: ~2,600 lines of header-only C++20. Compiles in seconds.

## Directory Layout

```
chukonu/
├── include/bolt/           ← Headers (the library)
│   ├── bolt_types.h                   Type enum, StringView, Schema, Arrow C Data ABI
│   ├── bolt_arena.h                   Per-thread bump allocator (9,600x faster than malloc)
│   ├── bolt_channel.h                 Lock-free SPSC/MPSC ring buffers (25x faster than mutex)
│   ├── bolt_column.h                  Adaptive column + BoltBatch + BitmapIndex
│   ├── bolt_branchless.h              Branchless kernels + micro-adaptive dispatch
│   ├── bolt_scheduler.h               Task ring, worker pool, phase barriers
│   ├── kernels/
│   │   └── bolt_numeric.h             Numeric kernel matrix: filter/agg/arith/cast (Wave A4)
│   └── README.md                      Full project documentation
│
├── src/bolt/                        ← Implementation notes
│   └── CMAKE_PATCH.md                 3-line CMake integration guide
│
├── tests/unit/bolt/                 ← Tests
│   └── test_bolt_primitives.cpp       GTest: types, arena, channels, columns, Arrow export
│
├── benchmarks/                      ← Performance validation
│   └── bench_bolt.cpp                 Arena vs malloc, SPSC vs mutex, epoch swap, COW
│
└── docs/                            ← Design documents
    ├── BOLT_DESIGN.md                 Phase 1: gap analysis, measured benchmarks
    ├── BOLT_COLUMN_FORMAT.md          Phase 2: stats, sidecars, adaptive encoding
    ├── BOLT_INDEPENDENCE.md           Zero-dependency architecture + interop matrix
    ├── BOLT_ACERO_COMPONENTS.md       What Acero provides, what Bolt replaces
    ├── BOLT_RESEARCH_NOTES.md         Thin pointer to research/
    └── research/                      Per-topic research notes:
        ├── README.md                    Index
        ├── pirk-techniques.md           Pirk et al. (14 papers)
        ├── cglm.md                      SIMD/portability patterns
        ├── scheduler-design.md          DuckDB / Polars / Seastar
        ├── cpu-topology.md              OS topology APIs
        ├── avx512-status.md             AVX-512 dispatch stub + plan
        ├── 1brc.md                      1 Billion Row Challenge
        ├── json-fionn.md                fionn vs simdjson
        ├── questdb-symbol-vs-fsst.md    SYMBOL layout + cardinality wall;
        │                                 FSST disjoint, Bolt Dict+Bitmap covers SYMBOL
        └── questdb-symbol-code-audit.md source-code audit of the SYMBOL claims
                                          (questdb/questdb@master): 7/8 confirmed,
                                          .c/.o dict vs .k/.v index correction
```

## Design Principles

These are non-negotiable:

1. **Zero external dependencies.** Bolt compiles with only a C++20 compiler.
   No Arrow, no protobuf, no gRPC, no boost, no vcpkg.

2. **No runtime overhead.** No exceptions, no RTTI, no smart pointers, no
   virtual dispatch, no `std::string`, no `std::vector` in hot structs.
   Every function is `noexcept`. OOM returns `nullptr`.

3. **Arena-managed memory.** Per-thread bump allocators. Reset per morsel
   epoch. No individual frees. Zero allocator contention between threads.

4. **Branchless inner loops.** Predicated execution (bool-to-int conditional
   advance), CMOV, SIMD masking. Micro-adaptive kernel selection based on
   column statistics: branching at extreme selectivities, branchless in the
   middle range (Pirk ICDE 2025).

5. **Arrow-compatible at boundaries.** Zero-copy export via Arrow C Data
   Interface (`ArrowSchema`/`ArrowArray`). No libarrow link needed. Any Arrow
   consumer reads Bolt columns directly.

6. **Better than Arrow internally.** Multi-format columns (Flat/Constant/
   Dictionary/Sequence/View). Inline 64-byte statistics block. Sidecar
   indexes (bitmap, bloom, sort, hash) arena-allocated per epoch.

## Provenance

Ideas come from five sources, all documented with references:

| Source | What We Took |
|--------|-------------|
| **Venus ECS** (our game engine) | Double-buffer COW, arena-per-frame, BH-tree sidecar, X-macro dispatch, deferred ops ring, spin-then-yield job system |
| **DuckDB** | Multi-format vectors, 2048 vector size, zone maps, push-based execution |
| **QuestDB** | Symbol type (dictionary + bitmap as separate concerns), mmap columns |
| **Polars/arrow2** | German-style string views, Rust-inspired ownership model |
| **Pirk et al. (CWI/MIT/Imperial)** | Predicated partitioning, micro-adaptive kernels, cache-conscious layout, composable kernel algebra, LightSaber parallel aggregation |
| **Chronicle Queue / LMAX** | Lock-free SPSC/MPSC, cache-line padding, pre-allocated ring buffers |

## Dependencies on Bolt (within Chukonu)

Bolt is currently standalone — nothing depends on it yet. Integration is
incremental. The migration path:

```
Phase 1 (current):  Bolt headers exist alongside Arrow-based operators
Phase 2:            New operators written against BoltBatch
Phase 3:            All internal operators use BoltBatch
Phase 4:            Arrow at boundaries only (Parquet, Flight, Python)
Phase 5:            Own Parquet reader, Bolt Wire Protocol, full independence
```

## Quick Start

```bash
# Everything compiles as part of the Chukonu build
cd chukonu/build
cmake .. -DCHUKONO_BUILD_TESTS=ON
make test_bolt_primitives
./test_bolt_primitives     # 25+ tests, zero Arrow dependency

make bench_bolt
./bench_bolt               # Arena, channel, epoch, COW benchmarks
```
