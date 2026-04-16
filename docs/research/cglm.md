# cglm — SIMD/portability patterns worth borrowing

[cglm](https://github.com/recp/cglm) is a fast, header-only C math library
for graphics (vec/mat/quat). It is not a database project, but its SIMD
abstraction layer is exactly the problem we have in `bolt::core`: one set
of kernel source files, many ISA backends, and no runtime dispatch budget.
Here is what we borrowed and what we skipped.

## Borrowed

**Compile-time macro dispatch.** cglm's `glmm_*` layer (see
`include/cglm/simd/intrin.h`) picks SSE / AVX / NEON at preprocessor time.
Kernel authors write `glmm_load(p)` once; the right intrinsic is selected
by `#if defined(__AVX__) ... #elif defined(__SSE2__) ...`. We adopt the
same pattern as `bmm_*` in `bolt_port.h`: one ISA-selection block sets
`BOLT_SIMD_AVX2 / _SSE42 / _NEON / _SCALAR`, and each wrapper
(`bmm_cmpgt_i32`, `bmm_movemask_i32`, etc.) has one implementation per
backend. Kernels then contain zero `#ifdef` ladders.

**`BOLT_ALL_UNALIGNED` toggle.** cglm exposes `CGLM_ALL_UNALIGNED` so
callers whose buffers cannot guarantee 16/32-byte alignment can opt into
unaligned loads globally. We mirror this as `BOLT_ALL_UNALIGNED`: when
defined, `bmm_load_*` lowers to `_mm*_loadu_*` instead of `_mm*_load_*`.
Useful for arena-backed slices that cross boundaries, mmap'd wire buffers,
and the Arrow C Data import path.

**Single-include umbrella header.** cglm ships `cglm/cglm.h` which pulls
the entire surface. We mirror with `include/bolt/bolt.h`, which pulls
`bolt::core` plus any sibling modules (`kernels/*`, `join/*`, `wire/*`,
`arrow/*`) that happen to exist via `__has_include`. Consumers get a
one-liner; we keep module-by-module includes for the rest.

**Granular per-type alignment.** cglm aligns `vec4`/`mat4` to 16 bytes
(SSE), `vec3`/`mat3` naturally (no padding on small structs). We adopt
the same philosophy with three tiers:
- `alignas(64)` — shared atomics, cache-line carriers, batch headers.
- `alignas(32)` — SIMD column buffers (AVX2-width aligned).
- Natural alignment — everything else. Avoids padding small PODs.

## Skipped

**Three-tier API (`glm_*` / `glms_*` / `glmc_*`).** cglm exposes three
calling conventions (by-pointer, by-value, explicit copy) to fit different
ergonomic tastes and ABI constraints. We are header-only C++20 with one
convention (free functions taking raw pointers, return by value for small
PODs) — a second tier buys us nothing.

**C11 anonymous unions for vector access.** cglm uses C11 anonymous
unions to expose `v.x`, `v.y`, `v.z` alongside `v.raw[0..3]`. In C++20
we have `std::bit_cast` and structured bindings, so we don't need the
aliasing gymnastics. Our `bmm_vec_*` types are either the raw ISA
intrinsic type (`__m256i`) or a one-field `std::array` wrapper in the
scalar fallback.
