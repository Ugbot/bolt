# PFOR-Delta scalar bitpack kernels

Status: research note for plan `~/.claude/plans/this-was-a-freach-hashed-crab.md`
Layer 2 substrate — consumed by the next-wave MarbleDB BM25 codec rewrite.
Author: implementation agent, 2026-05-01.

> Sibling note. The lockdown of constants lives in
> `third-party/marbledb/docs/research/posting-codec-lucene90.md`. This file
> documents the *Bolt-side* scalar kernel: chosen inner-loop shape, the 8-bit
> patch ceiling rationale, the all-equal fast path, and the SIMD-overlay hook.

---

## What shipped

`include/bolt/kernels/bolt_pfor.h` (header-only, `BOLT_FORCE_INLINE`) exposes:

```cpp
namespace bolt::kernels::pfor {
  inline constexpr int32_t kBlockSize       = 128;
  inline constexpr uint8_t kMaxExceptions   = 7;
  inline constexpr int32_t kMaxBitWidth     = 32;
  inline constexpr int32_t kPforMaxPackedBytes = 1 + 4*128 + 14;  // 527

  int32_t pfor_pack_block(const uint32_t* in, uint8_t* out, int32_t cap);
  int32_t pfor_unpack_block(const uint8_t* in, int32_t cap, uint32_t* out);
  int32_t pfor_unpack_to_doc_ids(const uint8_t* in, int32_t cap,
                                  uint64_t prev_doc, uint64_t* out);
  void    pack_bits  (const uint32_t* in, int32_t n, uint8_t bpv, uint8_t* out);
  void    unpack_bits(const uint8_t*  in, int32_t n, uint8_t bpv, uint32_t* out);
}
```

Tests in `tests/test_bolt_pfor.cpp` (8 cases). `ctest` is now 49/49.

---

## Block format (locked, matches Lucene104 PForUtil)

```
+--------+----------------------------+--------------------------+
| token  | bit-packed payload         | exceptions (numExc * 2B) |
| 1B     | ceil(bpv * 128 / 8) bytes  | (idx_byte, high_byte)... |
+--------+----------------------------+--------------------------+
```

`token = (numExceptions << 5) | bpv`, with `bpv ∈ [0, 31]` and
`numExceptions ∈ [0, 7]`. Bpv=0 is the "all values are zero" fast path —
the payload is empty; only patches (if any) carry data.

### 8-bit patch ceiling rationale

Patches store `(idx_byte, high_byte)`. The high byte is 8 bits wide, so a
patched value can occupy at most `bpv + 8` bits. Encoder selection rule:

```
choose smallest b such that
    count(values > 2^b - 1) <= 7        AND
    count(values > 2^(b + 8) - 1) == 0
```

Implemented as a single suffix-summed histogram walk over bit-widths in
`pfor_pack_block` (no allocation, all stack scratch). For the rare case
where some value's bit-width is exactly 32, the chosen bpv becomes 24 and
the high 8 bits ride in the patch byte — confirmed by the
`MaxBitWidthFallback` test.

### All-equal fast path

If every value in the block is identical:
- value = 0 → `bpv = 0`, `numExc = 0`, total = **1 byte** (just the token).
- value = c ≠ 0 → `bpv = bit_width(c)`, `numExc = 0`, payload encodes `c`
  128 times. Still small (e.g. for `c = 7`, `bpv = 3`, payload = 48 B).

Validated by `AllEqualFastPath`. We do not special-case the non-zero case
because the histogram walk already produces the optimal choice and the
ratio is acceptable (worst case bpv = 31 = 4 B/value, but only when every
value sits at 31-bit which is itself rare).

---

## Chosen scalar inner loop (Lemire-shape, branchless)

`pack_bits` and `unpack_bits` both use a single bpv-parameterised loop with
the bpv hoisted ONCE outside the loop body — no `switch`, no per-bpv
specialisation. The mask (`(1ull << bpv) - 1`) is also computed once.

Pack:

```cpp
const uint64_t mask = (1ull << bpv) - 1ull;
for (int32_t i = 0; i < n; ++i) {
    const int32_t bit_off  = i * bpv;
    const int32_t byte_off = bit_off >> 3;
    const int32_t shift    = bit_off & 7;
    const uint64_t v       = uint64_t(in[i]) & mask;
    uint64_t w; std::memcpy(&w, out + byte_off, 8);
    w |= v << shift;
    std::memcpy(out + byte_off, &w, 8);
}
```

Unpack:

```cpp
const uint64_t mask = (1ull << bpv) - 1ull;
for (int32_t i = 0; i < n; ++i) {
    const int32_t bit_off  = i * bpv;
    const int32_t byte_off = bit_off >> 3;
    const int32_t shift    = bit_off & 7;
    uint64_t w; std::memcpy(&w, in + byte_off, 8);
    out[i] = uint32_t((w >> shift) & mask);
}
```

Key properties:

- **Branchless**: only the loop counter branch; no data-dependent control flow.
- **Auto-vectorisable**: MSVC `/O2` generates a tight `mov / shr / and / mov`
  body. The 8-byte spanning load via `memcpy` is folded to a single `mov`.
- **Endian-portable** in practice: we run on x86_64 / ARM64 little-endian.
  A big-endian port would need a byte-swap inside the load/store; not worth
  guarding now.
- **No tail special case**: the read window is up to 8 bytes past
  `byte_off`. The block-pack/unpack functions copy through a stack scratch
  (`kBlockSize * 4 + kBitpackTailSlack = 520 B`) so the kernel never reads
  past the caller's buffer.

bpv = 0 short-circuits to memset; bpv = 32 short-circuits to memcpy.

---

## Prefix sum for delta -> doc-ids

`pfor_unpack_to_doc_ids` decodes a delta block and prefix-sums in one pass.
The accumulator is `uint64_t` (deltas are u32, prev_doc is u64) so a
128-deep sum cannot overflow at our doc-count scale. The loop is a tight
`acc += d; out[i] = acc;` — no conditional, no early-exit.

A SIMD prefix sum (Lemire's 4-wide tree-of-shifts) is a future optimisation
and lands in `bolt_pfor_avx2.h` (see hook below).

---

## Measured numbers

`CompressionRatio` test (128 log-uniform random deltas, range [1, 1000]):

```
161 bytes / 128 deltas = 10.06 bits/delta
```

The Lemire-Boytsov 2015 number for real-corpus d-gaps is 5–7 bits/delta;
our synthetic is wider-tailed (uniform-log) so 10 is expected. The plan's
gate of `<= 12 bits/delta` (`<= 192 bytes`) holds with margin.

Decode latency was not measured here — the perf gate
(`<= 0.5 µs/block scalar`) lives downstream in MarbleDB's `bm25_query`
benchmark and is the next wave's responsibility.

---

## SIMD-overlay hook (deferred wave)

A future `bolt_pfor_avx2.h` plugs in by re-implementing only `pack_bits` /
`unpack_bits` with the same signatures. The block-pack/unpack functions
call them by name — no dispatch table, no virtual, no runtime branch.
Compile-time switch will live in `bolt_config.h` as
`BOLT_PFOR_USE_AVX2_BITPACK`. Default stays scalar until the SIMD overlay
is measured to win on Tiger-Style criteria (no allocation regression, no
misalignment trap on tail blocks).

Lemire's SIMD-BP128 reaches ~0.2-0.4 ns/int on Sandy Bridge; our scalar
target is ~2-4 ns/int. Both fit the 40 µs budget for `bm25_query` at
`q=5`, so the SIMD wave is an optimisation, not a gate.

---

## What we deliberately did NOT do

| Skipped | Reason |
|---------|--------|
| Per-bpv unrolled `unpack5`/`unpack6`/... functions | Lucene's Java codegen wins because the JIT can't hoist a `bpv` parameter; in C++ MSVC `/O2` already auto-vectorises the parameterised loop. Keep one fn, not 32. |
| Variable block size (256 in Lucene 9.9+) | `posting-codec-lucene90.md` locks 128 for v1; revisit only if perf-test shows level-0 skip overhead dominates. |
| `pos` / `pay` channel encoding | BM25-only; no proximity scoring v1. |
| Arena-allocated scratch | The 520 B stack scratch is well below the 1 MB Windows default stack; arena threading would cost a TLS lookup per call. |
| zigzag sign encoding | Doc-id deltas are unsigned. Term-frequency encoders that need signed deltas will wrap us with their own zigzag. |

---

## Citations

1. Lemire & Boytsov. *Decoding billions of integers per second through
   vectorization.* SP&E 45(1), 2015. arXiv:1209.2137.
2. Zukowski, Heman, Nes, Boncz. *Super-scalar RAM-CPU cache compression.*
   ICDE 2006. (Original PFOR.)
3. Apache Lucene 10.4 — `PForUtil.java`, `ForUtil.java` in
   `org.apache.lucene.codecs.lucene104`.
4. Pibiri & Venturini. *Techniques for inverted index compression.* ACM
   CSUR 53(6), 2020. arXiv:1908.10598.
