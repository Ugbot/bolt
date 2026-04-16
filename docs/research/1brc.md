# 1 Billion Row Challenge — techniques worth borrowing

Gunnar Morling's 1BRC (https://www.morling.dev/blog/1brc-results-are-in/)
asked contestants to aggregate min/mean/max temperature per weather
station over a 13 GB, one-billion-row text file. Final leaderboard on an
AMD EPYC 7502P (Zen2, 8 cores): 1.535 s (thomaswue / Mai / Peterssen),
1.587 s (Korzun), 1.608 s (jerrinot), 2.157 s (royvanrijn), 2.332 s
(mtopolnik). The entries are Java, but every winning trick is a
cache/branch/SIMD story that maps cleanly onto a columnar engine —
Ragnar Groot Koerkamp's C++/Rust deep dive
(https://curiouscoding.nl/posts/1brc/) confirms the same hot-path list
drops sub-second on a laptop. Bolt is not a CSV ingester, but the
kernels that win 1BRC (SWAR byte scan, branchless ASCII int parse,
cardinality-tight hash table, per-thread accumulator + merge) are the
same kernels that sit inside our filter/parse/groupby paths.

## thomaswue (winner, 1.535 s)

Source:
https://raw.githubusercontent.com/gunnarmorling/1brc/main/src/main/java/dev/morling/onebrc/CalculateAverage_thomaswue.java

Splits each mmap segment into three cursors processed in the same
thread — three independent load/parse chains exposing ILP to the
out-of-order engine without the cost of extra worker threads. The hash
table is a flat `Result[1 << 17]` (131,072 slots) where each entry
caches the station name as two `long`s (`firstNameWord`,
`secondNameWord`) so probe equality is two integer compares for the
common ≤16-byte names. A subprocess-worker trick (`spawnWorker()` with
`--worker`) lets the parent process return immediately and the OS
reclaim the mmap'd region — clean munmap without paying for it in the
timed phase. SWAR delimiter find, inlined verbatim:

```java
// XOR collapses target byte to 0; subtract-borrow lights high bit of
// each zero lane; mask isolates those high bits. bit >> 3 = byte index.
private static long findDelimiter(long word) {
    long input = word ^ 0x3B3B3B3B3B3B3B3BL;
    return (input - 0x0101010101010101L) & ~input & 0x8080808080808080L;
}
```

## mtopolnik (QuestDB, top-10)

Source:
https://raw.githubusercontent.com/gunnarmorling/1brc/main/src/main/java/dev/morling/onebrc/CalculateAverage_mtopolnik.java
and the company write-up
https://questdb.com/blog/billion-row-challenge-step-by-step/.

`STATS_TABLE_SIZE = 1 << 16` (65,536 slots), entry size ~120 bytes
(104-byte inline name slot + counters, rounded). The branchless
temperature parser is the highlight — one `long` load straddles the
sign, digits and decimal, and a single 32-bit magic multiply packs
three decimal digits into the low 10 bits:

```java
// word = 8 bytes starting at the number; dotPos = bit position of '.'
// signed = all-ones if leading '-', else zero (bit 59 trick on '-' = 0x2D).
// Magic 0x640a0001 = (100 << 24) | (10 << 8) | 1 — one multiply does
// d0*100 + d1*10 + d2 in the upper 32 bits.
private static int parseTemperature(long word, int dotPos) {
    final long signed          = (~word << 59) >> 63;
    final long removeSignMask  = ~(signed & 0xFF);
    final long digits          = ((word & removeSignMask) << (28 - dotPos)) & 0x0F000F0F00L;
    final long absValue        = ((digits * 0x640a0001) >>> 32) & 0x3FF;
    return (int) ((absValue ^ signed) - signed);         // two's-complement negate if signed
}
```

Final print phase uses k-way merge across per-thread sorted chunks
rather than a global concurrent map — concentrates all ordering work
at the end.

## jerrinot (3rd, 1.608 s)

Source:
https://raw.githubusercontent.com/gunnarmorling/1brc/main/src/main/java/dev/morling/onebrc/CalculateAverage_jerrinot.java

Same dual-cursor trick as thomaswue but two-way, and a split table:
a fast map for inline ≤15-byte names (`UNSAFE.putLong(basePtr +
FAST_MAP_NAME_PART1, ...)`) and a slow map for longer names that
stores a pointer out to heap. Work-stealing is a single
`globalCursor.addAndGet(SEGMENT_SIZE)` — no deques, no mutex, CAS-loop
gets the next slab.

## royvanrijn (7th)

Source:
https://raw.githubusercontent.com/gunnarmorling/1brc/main/src/main/java/dev/morling/onebrc/CalculateAverage_royvanrijn.java

The canonical SWAR `;` detector, same formula as thomaswue but
factored into the reader. Load-bearing identity:

```java
// For any 8-byte word, this long is nonzero iff at least one lane == 0x3B,
// and bolt_ctz64(result) >> 3 gives the index of the first match.
long comparisonResult = (lastRead ^ 0x3B3B3B3B3B3B3B3BL);
long highBitMask      = (comparisonResult - 0x0101010101010101L)
                      & (~comparisonResult & 0x8080808080808080L);
```

This is the "zero-byte find" identity from Mycroft/Alan Mycroft's
strlen trick, rebroadcast for 0x3B. It generalises to any byte by
swapping the XOR mask — the single primitive Bolt needs in
`bolt_port.h` is `bolt_swar_find_byte_u64(word, byte)`.

## merrykittyunsafe / curiouscoding

Source: the `CalculateAverage_merrykittyunsafe.java` raw URL 404'd at
fetch time (file has been renamed/removed on `main`), so the parser
technique is cross-referenced from the curiouscoding.nl deep dive
(https://curiouscoding.nl/posts/1brc/) and mtopolnik's snippet above.
The curiouscoding author independently arrived at the same magic
multiplier via a different derivation: "read four bytes as a
little-endian value, then multiply by `1 + (10 << 16) + (100 << 24)`"
collapses the three digits into one lane of the 64-bit product. The
`cmov`-not-branch framing drops branch-mispredicts from 440M to 140M
on his dataset. Bolt's `bolt::parse::parse_int10th` should take
mtopolnik's body as the x86-64 reference and swap Java `long` for
`uint64_t`.

## 1BRC technique × Bolt applicability

| Technique                         | Bolt mapping                                         | Applicability   |
|-----------------------------------|------------------------------------------------------|-----------------|
| SWAR `;` / zero-byte finder       | new `bolt_swar_find_byte_u64` in `bolt_port.h`       | HIGH            |
| mmap zero-copy file input         | future `bolt::io` sibling                            | MEDIUM (deferred) |
| Branchless ASCII int parse        | new `bolt::parse::parse_int10th`                     | HIGH            |
| Cardinality-tight hash table      | extend `SwissTable::create(tight_sizing)`            | HIGH            |
| Per-thread accum + final merge    | already `parallel_groupby_sum_int64`                 | DONE            |
| Skip floats, use int10ths         | already supported via Int32/Int64 columns            | DONE            |
| Dual-cursor ILP in one thread     | not present; SIMD lanes already expose ILP           | LOW             |
| Perfect hashing (PtrHash)         | future research — minimal-perfect-hash over dictionaries | DEFERRED     |
| GraalVM AOT native-image          | n/a — we are already AOT C++                         | SKIP            |

## What we deliberately skip

- **GraalVM native-image / JVM startup tricks.** Top-10 entries all
  compile via GraalVM to avoid JIT warm-up. We are header-only C++
  with MSVC/clang; we're already AOT, so none of this is relevant.
- **Subprocess-spawn for clean munmap.** thomaswue's
  `spawnWorker()` exists purely so the parent doesn't pay the mmap
  teardown inside the timed region. Bolt operates on in-memory
  `BoltBatch` already mapped by the caller; teardown is the caller's
  problem and there is no timer to game.
- **Java Vector API / `jdk.incubator.vector`.** Bolt already has
  `bmm_*` wrappers over SSE4.2/AVX2/AVX-512/NEON. We write the
  intrinsic directly; we don't need a portability layer that's a
  portability layer for a portability layer.
- **`sun.misc.Unsafe` addressing.** The equivalent is plain pointer
  arithmetic in C++; nothing to adopt, it's our baseline.

## Followups

- Add `bolt_swar_find_byte_u64(word, byte)` primitive to
  `bolt_port.h` alongside `bolt_ctz64`; first consumer is any future
  delimited-text import path and the `StringView` equality fast path.
- Add `bolt::parse::parse_int10th(const char*, int len) -> int32_t`
  modelled on mtopolnik's 0x640a0001 body; unit-test against a
  1M-row fuzz corpus.
- Extend `SwissTable` with a `create_tight(capacity_hint)` that picks
  the smallest power-of-two ≥ `1.25 * capacity_hint` instead of the
  current default load factor — mirrors thomaswue's 131,072-for-413-
  stations ratio where every probe hits an empty or matching slot
  in one step.
- Add an in-memory 1BRC microbench to `benchmarks/` once the two
  primitives land: pre-tokenized columnar input, measure pure
  groupby-sum throughput vs the Java winners' time minus their I/O
  share (curiouscoding estimates ~40% I/O).
- Investigate PtrHash-style minimal perfect hashing for known-
  cardinality dictionaries (station names, enum columns). Deferred
  — belongs next to the dictionary-encoding work, not in the default
  hash path.
- Revisit dual-cursor ILP only if a measured SIMD groupby shows
  front-end-bound stalls; current branchless kernels already keep
  the back end saturated.
