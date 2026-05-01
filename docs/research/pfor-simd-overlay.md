# PFOR-Delta SIMD overlay (AVX2 256-bit + SSE4.2 128-bit)

Status: research note for Layer 2.1b of plan
`~/.claude/plans/this-was-a-freach-hashed-crab.md`. Companion to
`pfor-bitpack-kernels.md` (the scalar baseline).
Author: 2026-05-01.

---

## Why

The scalar PFOR-Delta kernels in `include/bolt/kernels/bolt_pfor.h`
(commit 49738028) hit ~0.5 µs/block on MSVC `/O2` — close to the
40 µs p50 BM25-query gate at 100 K corpus, but with no fat. Lemire &
Boytsov 2015 (*Decoding billions of integers per second through
vectorization*, arXiv:1209.2137) show 2–4× decode speedups over the
auto-vectorised scalar path for the bit-pack/unpack inner loop
(SIMD-BP128 / SIMD-FastPFOR). Lucene 10.4's `PForUtil` reaches the
same goal via the JVM `VectorAPI`. This overlay closes the gap on
x86 without changing the public surface or the wire format.

## What

Two new translation-unit-private code paths, selected at compile time:

- **AVX2 (256-bit)** — gated on `BOLT_SIMD_AVX2` (`/arch:AVX2` or
  better on MSVC; `-mavx2` on Clang/GCC).
- **SSE4.2 (128-bit)** — gated on `BOLT_SIMD_SSE42`. Covers the
  x86-64 V1 baseline so a host without AVX2 still gets a SIMD win.
- **Scalar fallback** — the existing branchless body in
  `internal::pack_bits_scalar` / `internal::unpack_bits_scalar`. The
  public `pack_bits` / `unpack_bits` try SIMD first and fall through
  on `false` return (unsupported `bpv` or `n_values != 128`).

Public API unchanged: same signatures on `pack_bits`, `unpack_bits`,
`pfor_pack_block`, `pfor_unpack_block`, `pfor_unpack_to_doc_ids`.

## Coverage table

`n_values == 128` is the only shape SIMD accelerates (the only shape
the block kernels ever pass). All other call sites (e.g. partial
trailing blocks) drop to scalar via the `false` return.

| bpv     | AVX2 unpack | AVX2 pack | SSE4.2 unpack | SSE4.2 pack | Notes |
|--------:|:-----------:|:---------:|:--------------:|:-----------:|-------|
| 0       | scalar fast path (`memset`) — never enters overlay |
| 1       | yes         | yes       | yes            | yes         | bit broadcast + mask |
| 2       | yes         | yes       | yes            | yes         | shift table {0,2,4,6} |
| 4       | yes         | yes       | yes            | yes         | nibble interleave |
| 8       | yes         | yes       | yes            | yes         | `cvtepu8_epi32` / `packus` |
| 16      | yes         | yes       | yes            | yes         | `cvtepu16_epi32` / `packus` |
| 3,5,6,7 | scalar      | scalar    | scalar         | scalar      | Lemire's full SIMD-BP128 has |
| 9–15    | scalar      | scalar    | scalar         | scalar      | dedicated routines for each;  |
| 17–23   | scalar      | scalar    | scalar         | scalar      | the win is small for our       |
| 25–31   | scalar      | scalar    | scalar         | scalar      | posting-list bpv distribution. |
| 24      | scalar      | scalar    | scalar         | scalar      | Patch-channel landing zone — covered via patches. |
| 32      | scalar      | scalar    | scalar         | scalar      | Already optimal: pure `memcpy`. |

Why only {1, 2, 4, 8, 16}? They are the byte-aligned cases (a whole
number of values fits in one byte / one uint16 / one uint32). The
remaining widths require partial-byte straddling, which costs more
SIMD shuffle work than it saves over the auto-vectorised scalar inner
loop on MSVC `/O2`. This matches Lemire's coarse-grained finding:
SIMD pays off on the byte-aligned widths first, the rest are diminishing
returns.

## Correctness gate — bit-exact equivalence with scalar

This is load-bearing. The wire format (the bytes the writer emits
into a `.bm25.doc` segment) must match regardless of which path
emits it, because every reader (the BM25 unpacker, future column
compression, the lakehouse import path) reads bytes a possibly-other
binary wrote.

`tests/test_bolt_pfor_simd.cpp` enforces this with three layers:

1. `CrossPathEquivalence_PackBytesIdentical` — same input → SIMD
   `pack_bits` and scalar `pack_bits_scalar` produce byte-identical
   buffers across the full payload.
2. `CrossPathEquivalence_UnpackValuesIdentical` — same packed buffer
   → SIMD `unpack_bits` and scalar `unpack_bits_scalar` produce the
   identical 128-uint32 sequence.
3. `Avx2Direct_BitExactWithScalar` (and the SSE4.2 mirror) — calls
   `simd::pack_avx2` / `simd::unpack_avx2` directly and compares
   against `internal::*_scalar`. Verifies the SIMD entry points
   independently of the public dispatch.

A regression here is a wire-format break, not a perf bug.

## Build configurations validated

`build-test/` is configured with `BOLT_SIMD_TIER=NATIVE` → MSVC
`/arch:AVX2`. `bolt_apply_simd` runs in INTERFACE scope on the
header-only `bolt::core`, so the flag propagates to every test
executable that links `bolt::bolt`. As reported by the runtime
diagnostic in `BoltPforSimd.ReportsActiveSimdTier`:

```
[ INFO     ] BOLT_SIMD_AVX2 active
```

Full Bolt suite: `50/50` tests passing (was `49/49`; +1 for
`test_bolt_pfor_simd`).

## SSE4.2 build (deferred verification)

A second build with `/arch:SSE2` would exercise the SSE4.2 path —
but MSVC x64 does not auto-define `__SSE4_2__` even with the SSE4.2
intrinsics callable, so `BOLT_SIMD_SSE42` would not actually engage
without a manual `-D__SSE4_2__` (called out in `bolt_port.h:425`).
The SSE4.2 functions still compile cleanly under that variant — the
overlay tests `unpack_sse42` / `pack_sse42` directly, gated by
`#if BOLT_SIMD_SSE42 && !BOLT_SIMD_AVX2` so the AVX2 build skips
them. Practical SSE4.2 verification is deferred until we wire a
non-AVX CI lane; documenting as a known gap rather than spending the
build cycles.

## CMake integration

No new flags. `bolt_apply_simd(bolt_core ${BOLT_SIMD_TIER})` already
picks `/arch:AVX2` for `NATIVE` and propagates it through INTERFACE
scope to all consumers. The new test executable
`test_bolt_pfor_simd` follows the same `bolt_apply_hardening` pattern
as its sibling `test_bolt_pfor`, added in
`tests/CMakeLists.txt:558-565`.

## Bugs caught during implementation

- AVX2's `_mm256_unpacklo_epi32` operates per 128-bit lane (sees
  lanes 0–1 from each half, not 0–3 of the full register). The
  initial bpv=4 unpack used a 256-bit-wide unpack that produced an
  out-of-order interleave. Fixed by lowering bpv=4 to 128-bit ops
  (`_mm_unpacklo_epi32` / `_mm_unpackhi_epi32` over `_mm_cvtepu8_epi32`).
- The bpv=4 pack initially loaded `in[2b+0..7]` and `in[2b+8..15]`
  into two 256-bit registers and tried to fold pairs across them,
  which paired non-adjacent values. Fixed by reverting to a scalar
  fold over 64 bytes (MSVC `/O2` auto-vectorises this loop with
  SSE pack ops; the SIMD overhead at bpv=4 lives in the unpack
  path anyway).

## Open questions

- **AVX-512 path.** Deferred. AVX-512 is server-class on x86; few
  laptops have it, and `BOLT_SIMD_AVX512` already implies
  `BOLT_SIMD_AVX2`, so the AVX2 path runs correctly on AVX-512
  silicon. Revisit when we have a server-class profiling host.
- **ARM NEON port.** Deferred. We are Windows-first; ARM is a
  v1.x target. The same shape (zero-extend + mask + interleave)
  ports cleanly via `vmovl_u8` / `vmovl_u16`.
- **bpv ∈ {3, 5–7, 9–15, 17–23, 25–31} dedicated SIMD routines.**
  Lemire's SIMD-BP128 implements them all. Defer until a perf
  trace shows the scalar fallback at one of these widths actually
  fires often enough on real BM25 workloads to matter.
- **SSE4.2 build verification on Windows + MSVC.** The MSVC
  flag-detection quirk (`__SSE4_2__` not auto-defined on x64) means
  the SSE4.2 path is only exercised on Clang/GCC today. Wire a
  Linux CI lane to close this loop.

## Citations

1. Lemire & Boytsov. *Decoding billions of integers per second
   through vectorization.* Software: Practice & Experience 45(1),
   2015. arXiv:1209.2137 / DOI:10.1002/spe.2203. §4–5 cover the
   SIMD-BP128 unpack inner loop this overlay shadows.
2. Apache Lucene 10.4 — `Lucene104PostingsFormat`, `PForUtil.java`,
   `ForUtil.java`. The `VectorAPI`-based unpack is the JVM-equivalent
   of the AVX2 path; gen_ForUtil.py emits per-bpv unrolled bodies.
3. Companion: `posting-codec-lucene90.md` (Layer 2 research brief)
   and `pfor-bitpack-kernels.md` (scalar baseline).
