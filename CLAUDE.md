# Bolt Development Guidelines

## What Is Bolt?

Bolt is Chukonu's zero-dependency columnar execution layer. It replaces
Arrow C++ on the hot path. Headers are in `include/chukonu/bolt/`.

## Rules (Non-Negotiable)

- **No exceptions.** Everything is `noexcept`. Return `nullptr` or `false` on failure.
- **No RTTI.** No `dynamic_cast`, no `typeid`, no virtual functions.
- **No smart pointers.** Raw pointers. Arena-managed lifetime.
- **No `std::string`.** `char[64]` for names, `StringView` for data.
- **No `std::vector` in hot structs.** Fixed-capacity arrays.
- **No heap on the hot path.** Arena bump allocation. `malloc` only at init.
- **All functions `noexcept`.** Compiler knows no exception tables needed.
- **Cache-line pad all shared atomics.** `alignas(64)` on every atomic that
  might be accessed by multiple threads.
- **Branchless inner loops.** Predicated execution (bool-to-int advance).
  Use branching only when selectivity < 20% or > 80% (micro-adaptive).
- **`__restrict__` on kernel parameters.** Enables auto-vectorization.
- **Compile-time type dispatch via X-macros.** One runtime switch at the
  boundary, fully specialized template in the inner loop.

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

## Testing

```bash
cd chukonu/build && cmake .. && make test_bolt_primitives && ./test_bolt_primitives
```

## Docs

- `docs/BOLT_PROJECT_MAP.md` — File map and design principles
- `docs/BOLT_DESIGN.md` — Gap analysis and measured benchmarks
- `docs/BOLT_COLUMN_FORMAT.md` — Stats, sidecars, adaptive encoding
- `docs/BOLT_INDEPENDENCE.md` — Zero-dependency architecture
- `docs/BOLT_ACERO_COMPONENTS.md` — What we replace from Acero
- `docs/BOLT_RESEARCH_NOTES.md` — Pirk et al. technique catalogue
