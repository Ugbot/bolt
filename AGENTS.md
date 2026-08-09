# Bolt Development Guidelines

## What Is Bolt?

Bolt is a zero-dependency, header-only columnar execution core. It aims to
be an ultra-performant Arrow replacement that is tiny and easy to build on
Windows/MSVC, Linux, and macOS with only the C++20 standard library.
Headers are in `include/bolt/`.

Scope: **execution + wire format**. Parquet, catalog, SQL, and query
planning live in a separate lakehouse library (user-owned) that Bolt plugs
into via Arrow C Data Interface + the Bolt wire format.

## Rules (Non-Negotiable)

### Memory & lifetime — "Tiger Style"
Follow TigerBeetle's Tiger Style. Reference:
https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md

- **Allocate at startup, never on the hot path.** Pools, ring buffers,
  arenas, column buffers, hash tables — all sized at `init()` and reused
  for the process lifetime.
- **RAII is a perf trap.** Constructor/destructor pairs in tight loops
  cause malloc churn and unpredictable tail latency. The *only* RAII
  pattern we allow is `ArenaGuard` (it swaps a thread-local pointer —
  zero allocation). Otherwise: pools for ownership, arenas for scratch,
  ring buffers for queues.
- **No smart pointers.** `std::unique_ptr` / `std::shared_ptr` are banned.
  Raw pointers; ownership is lexical, not reference-counted.
- **Hard upper bounds on everything.** Every loop, every queue, every
  buffer gets a `constexpr` cap. Over-allocate at init; never grow
  dynamically on the hot path.
- **In-place init over return-by-value** for large or atomic-holding
  structs. `init_empty(Batch*)` not `make_empty() → Batch`.
- **≥2 assertions per hot-path function.** Assert preconditions,
  postconditions, bounds. Split compound checks.
- **Functions ≤70 lines.** Parent handles control flow; pure helpers do
  the work.

### Language subset
- **No exceptions.** Everything is `noexcept`. Return `nullptr`/`false`
  on failure. (MSVC stdlib requires `/EHsc` at the ABI level, but *our
  code* never throws.)
- **No RTTI.** No `dynamic_cast`, no `typeid`, no virtual functions,
  no inheritance.
- **No `std::string`.** `char[64]` for names, `StringView` for data.
- **No `std::vector`/`std::map`/`std::deque`** in hot structs or hot
  paths. Fixed-capacity arrays, arena-backed buffers.
- **Templates are fine** for numeric specialization and compile-time
  dispatch — they win on perf. The ban is on *perf-eating* features.
- **Prefer C-style APIs** (POD + free functions) for new code so the
  surface stays C-callable for FFI / lakehouse interop. Existing
  method-style APIs stay — don't churn.

### Portability (Windows-first)
- **Must build on MSVC with only the C++20 stdlib.** No vcpkg, no conan,
  no POSIX shim required.
- **No GCC-isms in source.** `BOLT_RESTRICT`, `BOLT_FORCE_INLINE`,
  `BOLT_PAUSE`, `BOLT_PREFETCH_READ`, `bolt_ctz64`, `bolt_popcount64`,
  `bolt_aligned_alloc` — all defined in `bolt_port.h`.
- **No `pthread.h`, `sys/mman.h`, `unistd.h`** in public headers. Use
  `<thread>`, `<atomic>`, `<chrono>`, `<bit>`.
- **Windows CI is a merge-blocking gate.**

### Performance idioms
- **Cache-line pad shared atomics.** `alignas(64)` on every atomic that
  crosses threads.
- **Branchless inner loops.** Predicated execution (bool-to-int advance).
  Branch only when selectivity < 20% or > 80% (micro-adaptive dispatch).
- **`BOLT_RESTRICT` on kernel parameters.** Enables auto-vectorization.
- **Compile-time type dispatch via X-macros.** One runtime switch at the
  boundary, fully specialized template in the inner loop.

### Multiple implementations, one default
- **The general-case implementation is the default.** Multiple
  implementations of the same primitive may live in the library
  (e.g. compact-vs-padded MergeTriple, scalar vs SIMD hash probe,
  Murmur3 vs wyhash mix), but exactly one is wired in by default —
  the one that wins on the broadest workload class.
- **Alternatives ship behind compile-time switches.** Use
  `#ifndef BOLT_*` opt-in flags in `bolt_config.h`, or a
  compile-time tag-dispatch parameter (`SwissTable_PreHashed`-style
  variant types). Never runtime branches — keep dispatch zero-cost.
- **Document the trade-off inline** at the alternative's definition,
  and append an entry to `docs/research/design-log.md` recording why
  the default won and what workload would flip the choice. Future
  tuning starts from the log so we don't re-litigate decisions.

## File Structure

- `bolt_types.h` — Type enum, StringView, Schema, Arrow C Data ABI structs
- `bolt_arena.h` — Bump allocator, ArenaGuard RAII, tl_arena thread-local
- `bolt_channel.h` — SPSC/MPSC ring buffers, cache-line padded
- `bolt_column.h` — BoltColumn (multi-format), ColumnStats, BoltBatch (COW), BitmapIndex
- `bolt_branchless.h` — Filter/aggregate/hash/gather kernels, micro-adaptive dispatch, predicated partition
- `bolt_scheduler.h` — TaskRing, TaskPool, WorkerConfig, PhaseBarrier

## Key Patterns

### Venus Tick-Tock COW (from our game engine)
Two physical column buffers. Read from epoch 0, write to epoch 1.
Clone-on-write: first mutation copies, subsequent writes are free.
Swap: flip index + clear dirty mask (~1.3ns).

### Arena per Slot
Each worker thread gets its own Arena. Set `tl_arena` at start of morsel,
reset at end. All intermediates freed in one pointer move.

### Branchless Filter (Pirk DaMoN 2014)
```cpp
output[count] = i;              // Always write (speculative)
count += (data[i] > scalar);    // 0 or 1, no branch predictor
```

### Micro-Adaptive Dispatch (Pirk ICDE 2025)
Estimate selectivity from ColumnStats zone map. If extreme (<20% or >80%),
use branching kernel (predictor wins). Otherwise branchless.

### Arrow C Data Interface Export
`fill_arrow_schema()` and `fill_arrow_array()` produce zero-copy Arrow views.
No libarrow link. Any Arrow consumer reads our columns directly.

## Build & Test

```bash
# From repo root (Linux/macOS or Windows via MSVC):
cmake -S . -B build -DBOLT_BUILD_TESTS=ON -DBOLT_BUILD_BENCHMARKS=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure -C Release
```

Presets (`CMakePresets.json`): `debug`, `release`, `msvc`, `ninja-msvc`,
`clang-cl`. GTest is fetched via `FetchContent` if not found locally, so
the build is self-contained on Windows without vcpkg.

## Docs

- `docs/BOLT_PROJECT_MAP.md` — File map and design principles
- `docs/BOLT_DESIGN.md` — Gap analysis and measured benchmarks
- `docs/BOLT_COLUMN_FORMAT.md` — Stats, sidecars, adaptive encoding
- `docs/BOLT_INDEPENDENCE.md` — Zero-dependency architecture
- `docs/BOLT_ACERO_COMPONENTS.md` — What we replace from Acero
- `docs/research/` — per-topic research notes (Pirk catalogue, cglm,
  scheduler design, CPU topology, AVX-512, 1BRC, fionn JSON). Index in
  `docs/research/README.md`. **All new research lands here as its own
  file** — never append to a monolithic notes file.
