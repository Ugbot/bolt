# Bolt Roadmap

Status of the work to turn Bolt into an ultra-performant, tiny,
build-anywhere Arrow replacement. Revised after audit + cglm review + Tiger
Style adoption.

## Context

Bolt is a zero-dependency, header-only C++20 columnar execution core.
Scope: **execution + wire format**. Parquet, catalog, SQL, and query
planning live in a separate lakehouse library that plugs into Bolt via the
Arrow C Data Interface and the Bolt wire format.

Three non-negotiable invariants:

1. **Windows/MSVC is first-class.** Must build with only the C++20 stdlib.
   No vcpkg, no POSIX shim. Windows CI is a merge-blocking gate.
2. **Tiny and modular.** Each module is a header-only INTERFACE library
   with near-zero dependencies beyond `bolt::core`.
3. **No allocations on the hot path.** Tiger Style: pools, rings, arenas
   sized at `init()`. Only `ArenaGuard`-style RAII is permitted.

## Architectural principles

### Tiger Style (TigerBeetle)
Reference:
https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md

- Allocate at startup, never during execution.
- RAII is a perf trap — ctor/dtor in tight loops causes malloc churn and
  tail-latency spikes. Only `ArenaGuard` (pointer swap, zero alloc) passes.
- Pools for ownership, arenas for scratch, ring buffers for queues.
- Hard `constexpr` caps on every loop/queue/buffer.
- ≥2 assertions per hot-path function (bounds, invariants, pre/post).
- Functions ≤70 lines. Parent handles control flow; pure helpers do work.
- In-place init over return-by-value for atomic-holding or large structs.

### C-style surface over C++
- POD structs + free functions for new APIs. Existing method-style APIs on
  `BoltColumn` / `BoltBatch` stay — no churn.
- No virtual functions, no inheritance, no RTTI.
- No exceptions, no smart pointers, no `std::string` / `std::vector` /
  `std::map` in hot code.
- Templates are fully permitted — the rule is performance, not template
  avoidance. Use them for numeric-type specialization and compile-time
  dispatch.
- Public headers should be C-callable (`extern "C"` where cheap) so FFI
  and the lakehouse library integrate without pulling C++ ABI.

### Portability (Windows-first)
All GCC-isms go through `include/bolt/bolt_port.h`:
- `BOLT_RESTRICT`, `BOLT_FORCE_INLINE`, `BOLT_LIKELY/UNLIKELY`
- `BOLT_PAUSE`, `BOLT_PREFETCH_READ/WRITE`
- `bolt_ctz32/64`, `bolt_clz32/64`, `bolt_popcount32/64` (via `<bit>`)
- `bolt_aligned_alloc/free` (wraps `_aligned_malloc` on MSVC)
- `BOLT_NYI(msg)`, `BOLT_C_API`
Banned in public headers: `pthread.h`, `sys/mman.h`, `unistd.h`, raw
`__builtin_*`, `__restrict__`, `__attribute__`, `posix_memalign`,
`__thread`, `#pragma GCC`.

### cglm-inspired SIMD layer (new)
Reference: https://github.com/recp/cglm

- **One macro family selects the SIMD backend at compile time.** Write
  `bmm_cmpgt_i32(a,b)` once; preprocessor picks SSE4.2 / AVX2 / NEON. No
  runtime dispatch, no function pointers, no `#ifdef` ladders inside
  kernels. Kernel count × intrinsic count collapses into kernel count.
- **`BOLT_ALL_UNALIGNED` escape hatch.** Compile-time toggle that swaps
  aligned loads for unaligned. Lets the same code run on platforms where
  alignment can't be guaranteed.
- **Granular per-type alignment rule:**
  - `alignas(64)` — shared-atomic carriers, batch headers.
  - `alignas(32)` — SIMD column buffers.
  - Natural alignment — everything else (ties to cglm's `vec3`/`mat3`
    decision; avoids padding small structs).
- **Single-include umbrella** `include/bolt/bolt.h` pulls core + kernels +
  join + wire + arrow. Consumers who don't want module-by-module includes
  get a one-liner.
- **Skipped:** cglm's three-tier API (`glm_` / `glms_` / `glmc_`) and C11
  anonymous unions. We're header-only with one calling convention, and
  C++20 has `std::bit_cast`.

## Module layout

Each module is an INTERFACE library under `bolt::`. Core is the only
mandatory dependency; siblings are opt-in.

```
bolt::core     include/bolt/*.h              types, arena, channel, column,
                                             branchless, scheduler, port
bolt::kernels  include/bolt/kernels/*.h      numeric/string/temporal kernels
bolt::join     include/bolt/join/*.h         swiss, hashjoin, groupby
bolt::wire     include/bolt/wire/*.h         IPC/wire format
bolt::arrow    include/bolt/arrow/*.h        Arrow C Data Interface helpers
bolt::bolt     umbrella — core + all siblings
```

## Phases

### Phase 0 — Hygiene + Windows portability pass  ✅ DONE

Delivered:
- Modern CMake (3.20 baseline), namespaced targets, sibling INTERFACE libs,
  `FetchContent` GTest fallback, install/export.
- `CMakePresets.json` (debug, release, msvc, ninja-msvc, clang-cl).
- GitHub Actions CI: Linux gcc, Linux clang, macOS, **windows-msvc**
  (merge-blocking).
- `include/bolt/bolt_port.h` portability macros.
- Swept all headers: GCC-isms → portability macros, `pthread_t` →
  `std::thread`, namespace `chukonu::bolt` → `bolt`.
- `BoltBatch::make_empty()` → `init_empty(BoltBatch*)` (atomics aren't
  copyable, and Tiger Style prefers in-place init).
- `docs/BOLT_PROJECT_MAP.md` paths fixed.
- Verified: MSVC 19.37 configures + builds + 1/1 tests pass.

### Phase 1 — Finish core stubs  (2–3 days)

The `include/bolt/` core has ~10 declared-but-unimplemented methods.
File-by-file:

- **bolt_column.h**
  - `compute_stats_numeric()` — one-pass min/max/null_count/distinct
    estimate (2KB HLL-lite sketch). X-macro over `BOLT_NUMERIC_TYPES`.
  - `clone_into(Arena*)` — deep copy into arena; fix up pointers.
  - `materialize(Arena*)` — expand Constant/Sequence/Dictionary → Flat.
  - `try_promote(Arena*)` — Flat→Constant when distinct=1;
    Flat→Dictionary when `distinct * log2(distinct) < nrows * sizeof(T) * 0.7`.
  - `fill_arrow_schema/array()` — populate format string per type; wire
    validity + data buffers; `release = noop`.
- **bolt_column.h — BitmapIndex**: build/count/filter/filter_in via
  packed `uint64_t` bitmaps + `bolt_ctz64` enumerate.
- **bolt_scheduler.h**: `init()` spawns N `std::thread` workers each with
  their own Arena; `submit_range()` slices `[0,n)` into morsels;
  `submit_column_task()` fans a kernel across columns.

**Tiger Style checkpoints:** every new function ≥2 asserts, ≤70 lines; all
buffers arena-allocated; pools sized at `init()`; no `new`/`delete` in
implementation.

### Phase 1.5 — cglm-inspired SIMD layer  (1 day)

Gates Phase 2 so kernel code can target one macro family from day one.

- Add `BOLT_SIMD_*` / `bmm_*` wrappers to `bolt_port.h` covering the ops
  our kernels need: `cmp{gt,lt,eq}`, `blend`, `and`, `or`, `movemask`,
  `compressstore`, `gather`, `hadd`.
- Compile-time dispatch: one header branch per ISA (SSE4.2 / AVX2 /
  AVX-512 / NEON / scalar fallback).
- Add `BOLT_ALL_UNALIGNED` macro (aligned loads become unaligned when set).
- Add `include/bolt/bolt.h` umbrella header.
- Add `docs/research/<topic>.md` documenting what we borrowed
  from cglm and what we skipped.

### Phase 2 — Numeric kernel matrix  (3–4 days)

New header `include/bolt/kernels/bolt_numeric.h`, written once against
`bmm_*` wrappers:
- Filter: `{gt,lt,eq,ne,ge,le}` × `BOLT_NUMERIC_TYPES`.
- Aggregate: `{sum,min,max,count,avg}` × types.
- Arithmetic: `{add,sub,mul,div}` × types.
- Cast matrix.
- Micro-adaptive dispatch (Pirk ICDE 2025): branching kernel when
  zone-map selectivity < 0.2 or > 0.8, branchless in the middle range.

### Phase 3 — Hash + join + aggregate  (5–7 days)

- `bolt_swiss.h` — SwissTable: 16-byte groups, 7-bit hash tag, SIMD
  metadata scan, arena-allocated, power-of-two capacity.
- `bolt_hashjoin.h` — partitioned hash join, vectorized probe,
  selection-vector output.
- `bolt_groupby.h` — two-phase group-by (thread-local partial → merge);
  Constant-column fast path.

### Phase 4 — String + temporal kernels  (2–3 days)

- `bolt_string.h` — `utf8_{starts_with,contains,equals}` using
  `StringView` 4-byte prefix compare.
- `bolt_temporal.h` — timestamp add/diff, date_trunc, extract
  year/month/day. Pure integer math on Unix-epoch representation.

### Phase 5 — Wire format  (2–3 days)

`include/bolt/wire/bolt_wire.h`:
- Self-describing flat layout: 32-byte header, schema, column descriptors,
  buffers contiguous. Arrow-buffer-layout-compatible so `bolt::arrow` can
  alias without copy.
- `serialize(batch, buf, cap) → bytes_written`
- `deserialize(buf, len, arena) → BoltBatch`
- Versioned magic; zero-copy when buffer is mmap'd and aligned.
- `extern "C"` surface so the lakehouse library and FFI consumers bind
  without C++ ABI entanglement.

### Phase 6 — End-to-end validation  (2–3 days)

- `benchmarks/bench_tpch_lite.cpp` — Q1 (group-by agg), Q3 (filter +
  hash-join + agg), Q6 (scan + filter + sum).
- Compare vs Arrow Acero on the same host. Publish in
  `docs/BOLT_PERFORMANCE.md`.
- Integration test: BoltBatch → Arrow C Data → pyarrow → back.
- Run on both Linux/clang and Windows/MSVC.

## Verification (every phase)

On both Linux/clang and Windows/MSVC:
```
cmake --preset release
cmake --build --preset release
ctest --preset release
```
A PR does not merge until the Windows job is green.

## Out of scope

- **Parquet, catalog, SQL, query planning** — provided by the separate
  user-owned lakehouse library. Bolt plugs in via Arrow C Data Interface +
  wire format.
- Flight / transport — separate sibling module if needed.
- Python bindings — Arrow C Data Interface already bridges; any wrapper
  lives outside the core.
- Arrow `Result<T>` / `Status` adapter for drop-in call-site parity —
  deferred until real Arrow consumers are blocked.

## File inventory

Already present:
- `include/bolt/{bolt_port,bolt_types,bolt_arena,bolt_channel,bolt_column,bolt_branchless,bolt_scheduler}.h`
- `tests/test_bolt_primitives.cpp`
- `benchmarks/bench_bolt.cpp`
- `CMakeLists.txt`, `CMakePresets.json`, `cmake/BoltCompileOptions.cmake`,
  `cmake/BoltGTest.cmake`, `cmake/bolt-config.cmake.in`
- `.github/workflows/ci.yml`

Landing per phase:
- Phase 1.5: `include/bolt/bolt.h` umbrella; `BOLT_SIMD_*` block in
  `bolt_port.h`.
- Phase 2: `include/bolt/kernels/bolt_numeric.h`,
  `tests/test_bolt_kernels.cpp`.
- Phase 3: `include/bolt/join/{bolt_swiss,bolt_hashjoin,bolt_groupby}.h`,
  `tests/test_bolt_join.cpp`.
- Phase 4: `include/bolt/kernels/{bolt_string,bolt_temporal}.h`.
- Phase 5: `include/bolt/wire/bolt_wire.h`, `tests/test_bolt_wire.cpp`.
- Phase 6: `benchmarks/bench_tpch_lite.cpp`.
