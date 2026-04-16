# AVX-512 status (native ops + AVX2 fall-through)

As of wave I5 Bolt provides native 512-bit specializations in
`bolt_port.h` alongside the pre-existing 256-bit AVX2 wrappers.
`BOLT_SIMD_AVX512=1` is set whenever `__AVX512F__` is defined;
`BOLT_SIMD_AVX2=1` is *also* set because AVX-512 is a strict superset
of AVX2, so the 256-bit `bmm_vec_i32` (8-lane) / `bmm_vec_i64` (4-lane)
/ `bmm_vec_f64` (4-lane) typedefs and their ops remain the default and
execute correctly on AVX-512 silicon. Kernels opt in to wider vectors
by naming the new `_x16` / `_x8` types. The CMake tier
`BOLT_SIMD_TIER=AVX512` passes `/arch:AVX512` (MSVC) or
`-mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512cd -mavx2 -msse4.2`
(GCC/Clang).

## Native 512-bit ops (guarded by `#if BOLT_SIMD_AVX512`)

- Typedefs: `bmm_vec_i32_x16` (`__m512i`, 16 lanes), `bmm_vec_i64_x8`
  (`__m512i`, 8 lanes), `bmm_vec_f64_x8` (`__m512d`, 8 lanes), plus
  `bmm_lanes_*_x*` constants.
- Unmasked load/store: `bmm_loadu_i32_x16` (`_mm512_loadu_si512`),
  `bmm_storeu_i32_x16` (`_mm512_storeu_si512`).
- Masked load/store (kills scalar tail loops on ragged morsels):
  `bmm_maskz_loadu_i32_x16` (`_mm512_maskz_loadu_epi32`),
  `bmm_mask_store_i32_x16` (`_mm512_mask_storeu_epi32`).
- Compare-to-mask (native `__mmask16` / `__mmask8`, no movemask
  round-trip): `bmm_cmpgt_i32_x16_mask`, `bmm_cmpeq_i32_x16_mask`,
  `bmm_cmpgt_i64_x8_mask`.
- Native single-instruction compressstore:
  `bmm_compressstore_i32_x16` (`_mm512_mask_compressstoreu_epi32`),
  `bmm_compressstore_i64_x8` (`_mm512_mask_compressstoreu_epi64`).
- Conflict detection (AVX-512CD): `bmm_conflict_i32_x16`
  (`_mm512_conflict_epi32`) — per-lane mask of earlier lanes holding
  the same value, for SwissTable / groupby WAW-hazard detection
  within a probe vector.

## AVX2 fall-through

Ops that do **not** yet have `_x16` / `_x8` 512-bit forms — shuffle,
blend, gather on i64/f64, `hadd_i64` / `hadd_f64`, all i8 ops, all f32
ops, and every existing 256-bit symbol — continue to work unchanged
on AVX-512 hosts via the AVX2 branch (already compiled in because
`BOLT_SIMD_AVX2=1` is also set). This is correct by the superset
guarantee; the ops simply execute as 256-bit instructions. Future
waves widen individual ops to 512-bit as kernel demand warrants,
without disturbing the fall-through.

## Hardware availability

Intel Sapphire Rapids / Emerald Rapids servers, Ice Lake-X / -SP,
Tiger Lake mobile, Xeon Phi (legacy); AMD Zen4+ (full 512-bit data
path since Zen5). Killed in firmware on consumer Alder Lake and
Raptor Lake parts. The developer host (i9-9980HK) has no AVX-512
silicon — the `BOLT_SIMD_AVX512` block is compile-checked via the
AVX-512 CMake tier but runtime-checked only on targeted boxes.
