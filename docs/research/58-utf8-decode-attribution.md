# Utf8 parquet decode: where the time actually goes (2026-09-04)

**Lane:** decode attribution, measurement only. No engine file was edited.
**Box:** Apple M5 Max, 6 P-cores + 12 E-cores, 128 GiB. macOS arm64.
**Data:** the real ClickBench `hits.parquet` — 99,997,497 rows x 105 cols, 226
row groups, 13.76 GiB, SHA-verified. Nothing down-scaled, nothing transcoded.
**Build:** `build/coverage` (`CMAKE_BUILD_TYPE=Release`, `-O3 -g`); the
measurement binaries are compiled with the identical flag set the engine uses
(`-O3 -march=native -fno-exceptions -fno-rtti`), and `bolt_snappy.h` is
header-only so it is compiled into the harness with those same flags.

Everything below is a **decode-level micro-measurement on a fixed file**, not a
board number. Those are far less load-sensitive than end-to-end query timings —
and empirically so here: the headline A/B reproduced to within 0.7% across
three interleaved rounds while the box carried a load average of 8.7. Where a
board-level number is quoted it is labelled as such and treated as directional.

---

## TL;DR — the four things that decide the next three lanes

1. **The 12.7x "Utf8 costs more than Int64" is a BYTES ratio, not a per-value
   inefficiency.** bolt decodes `URL` at **2.69 GB/s** of decoded bytes and
   `UserID` at **2.30 GB/s** — within 17% of each other. `URL` is 8.52 GB
   uncompressed across the file and `UserID` is 0.31 GB. 27.5x the bytes at
   comparable throughput *is* the gap. There is no Utf8-specific pathology to
   find.

2. **Snappy decompression is 78–84% of a Utf8 column's decode time**, and
   68% of a compressed Int64 column's. Everything else is second order.

3. **bolt's snappy decoder is 2.19x slower than Arrow's (Google snappy) on
   byte-identical input producing byte-identical output** — 3.00 GB/s vs
   6.56 GB/s on `URL`'s real pages. The deficit is entirely in the
   back-reference/copy path: on an incompressible column (`WatchID`, ratio
   1.00, all-literal) the two are at **parity (1.02x)**. This is the largest
   measured, in-hand headroom in the decode path.

4. **Decompression is compute-bound, not memory-bound, and scales linearly.**
   Single-thread snappy runs at **4.4% of single-thread memcpy bandwidth**
   (3.00 vs 68.3 GB/s); a hot L2-resident destination and a cold 1 GiB rotating
   destination give **3.00 vs 2.99 GB/s** — no difference at all. Page-parallel
   decompression scales 5.97x at 6 threads and 15.0x at 18 threads, topping out
   at 45 GB/s against a measured 18-thread memcpy ceiling of 181 GB/s.

---

## Method

Attribution by **construction**, not by sampling. This campaign has already
learned that sampling cannot attribute this code: inlining collapses every copy
site into one frame, and it took a per-column decode measurement to settle the
`memmove` question (recorded in `PROJECT_MAP.md`). So each stage is isolated by
calling it directly.

`scratchpad/decode_attrib.cpp`
- `full` = `parquet_read_row_group_cols` on **exactly one column**, one row
  group at a time, arena reset between groups. Inlining cannot hide anything
  because only one column's code runs.
- `snappy` = `bolt::ingest::snappy_decompress` over **exactly the page bodies
  that column's chunks contain**, located by walking `PageHeader`s directly from
  the compressed bytes with a from-spec thrift-compact reader written inside the
  harness (deliberately not reusing bolt's, so the measurement does not depend
  on the code being measured).
- `rest` = `full - snappy` — dictionary build and gather, StringView
  materialisation, level handling, arena.
- The page walk also **censuses pages by encoding**, so a claim about "the
  dictionary path" is checked against whether the dictionary actually survived
  the row group rather than assumed from the footer's encoding list.

`scratchpad/decode_floor.cpp` — memcpy ceiling (1..18 threads), snappy with hot
vs cold destination, and snappy scaling over 1..18 threads. It also emits a page
manifest (`pages_col<N>_rg<M>.csv`) so an independent implementation can be timed
on the same bytes.

`scratchpad/arrow_ref.py` — times `pyarrow.decompress(codec='snappy')` (Arrow's
bundled Google snappy) on the exact page byte ranges from that manifest.

`scratchpad/sn_hash.cpp` + cross-check — hashes bolt's decompressed output per
page so it can be compared against Arrow's.

### Correctness gates actually run

- **Every decode is checksummed over VALUES** (the `hash_col` shape used by
  `bench_parquet_decode`: for Utf8 it walks each `StringView`'s bytes through
  the inline/spilled split, not the buffer address), and the harness **aborts if
  the checksum differs between passes**. No timing here is reported for a decode
  that did not produce identical bytes.
- The per-column decode checksums are **identical across separate process
  invocations** (`URL` `27dcdc35338b3289`, `UserID` `ab33e42b90e62e8d`), with
  timings reproducing to 2.3%. So the checksum is a usable old-vs-new gate for
  any future decoder change, not just an intra-run consistency check.
- **bolt's snappy output was verified byte-identical to Arrow's** on the first
  12 real pages of `URL` (11.9 MB) via an FNV-1a hash computed independently on
  both sides. The 2.19x is therefore a like-for-like comparison of two decoders
  producing the same bytes, not a comparison of different work.
- `snappy_decompress` enforces exact-fill (`hdr_len == dst_len`), so every one
  of the 348–431 pages per column also self-validated its declared length.

### One harness bug found and fixed, recorded because it inverted a conclusion

The first run of `decode_attrib` reported **zero dictionary-encoded data pages**
for every column — which would have "confirmed" the dictionary ruling far more
strongly than the truth warrants. It was wrong. The parquet `Encoding` enum is
`PLAIN=0, PLAIN_DICTIONARY=2, RLE=3, ..., RLE_DICTIONARY=8`; value 1 is the
deprecated `GROUP_VAR_INT`. The harness had assumed 1/7. Caught by hexdumping a
page header and hand-decoding the thrift compact bytes, not by the numbers
looking wrong — they looked *plausible*. The real values are now spelled out in
the source with that note attached.

---

## 1. Per-column attribution — 20 row groups, 10,749,675 rows, min-of-3

| col | type | uncomp MB | full ms | snappy ms | rest ms | **snappy %** | decoded GB/s |
|---|---|---|---|---|---|---|---|
| `Title` (2) | Utf8 | 1156.3 | 478.9 | 400.5 | 78.4 | **84%** | 2.41 |
| `Referer` (14) | Utf8 | 900.1 | 369.9 | 292.8 | 77.0 | **79%** | 2.43 |
| `URL` (13) | Utf8 | 918.5 | 341.3 | 267.6 | 73.7 | **78%** | 2.69 |
| `OriginalURL` (56) | Utf8 | 323.4 | 101.6 | 78.7 | 22.9 | 77% | 3.18 |
| `SearchPhrase` (39) | Utf8 | 104.3 | 72.7 | 49.2 | 23.5 | 68% | 1.43 |
| `EventTime` (4) | Int64 | 48.6 | 19.9 | 13.7 | 6.3 | 68% | 2.44 |
| `UserID` (9) | Int64 | 24.6 | 10.7 | 3.2 | 7.5 | 30% | 2.30 |
| `WatchID` (0) | Int64 | 91.6 | 3.9 | 1.3 | 2.6 | 33% | **23.5** |

`WatchID` is the control that proves the whole story: it is the one column
whose data does not compress (ratio 1.00, near-unique 64-bit ids), so its snappy
pass is essentially a memcpy — and it decodes **8.7x faster per byte than any
other column**. The other seven all sit in a 1.4–3.2 GB/s band set by their
snappy rate.

### Whole-file projection (single thread)

The first 20 row groups are **not** proportionally representative — per-column
byte scale factors to the full file range from 7.05x (`Title`) to 16.56x
(`OriginalURL`) against the naive 226/20 = 11.3x. So this projects by measured
**bytes** at the measured per-column rate, not by row-group count:

| col | whole-file uncomp | 1-thread decode | of which snappy |
|---|---|---|---|
| `Title` | 8.15 GB | 3.38 s | 2.82 s |
| `URL` | 8.52 GB | 3.17 s | 2.48 s |
| `Referer` | 6.58 GB | 2.70 s | 2.14 s |
| `UserID` | 0.31 GB | 0.135 s | 0.040 s |

This reconciles with the board-level decomposition that motivated the lane
(directional, different harness): `WHERE url<>''` measured 3.48 s and
`WHERE userid<>0` measured 0.27 s. Decode alone accounts for **3.17 s of the
3.48 s (91%)** on the Utf8 side and 0.135 s of 0.27 s (50%) on the Int64 side.
The decode-only ratio is 23.5x where the end-to-end ratio was 12.7x — the
difference is per-query fixed cost, which is proportionally larger on the cheap
query. Either way the conclusion is the same and now has a mechanism:
**8.52 GB / 0.31 GB = 27.5x the bytes, at 1.17x the throughput.**

---

## 2. The dictionary really is gone — now quantified per page

The lane brief records dictionary-aware aggregation as already ruled out for
`URL`. The page census says exactly how gone it is, across 20 row groups:

| col | dict pages | dict-encoded values | PLAIN values | **% of values dict-encoded** |
|---|---|---|---|---|
| `URL` | 20 (21.4 MB) | 664,576 | 10,085,099 | **6.2%** |
| `Title` | 20 (21.4 MB) | 857,088 | 9,892,587 | **8.0%** |
| `Referer` | 20 (21.5 MB) | 581,632 | 10,168,043 | **5.4%** |
| `SearchPhrase` | 20 (19.1 MB) | 4,850,510 | 5,899,165 | 45.1% |
| `UserID` | 20 (12.8 MB) | 10,389,780 | 359,895 | 96.7% |

The writer emits **exactly one** dictionary-encoded data page per row group and
then abandons the dictionary for the rest of it. For `URL` that first page holds
54,272 values out of 450,560 in the row group; the remaining ~41 pages are
PLAIN. A ~1.05 MB dictionary page is written per row group and used for 6% of
the values — bolt pays to decode it and gets almost nothing back.

Also settled: **all 105 columns are REQUIRED** (`nullable=False` on the Arrow
schema of the real file, confirming what `PROJECT_MAP.md` records). `max_def`
is 0, so **definition-level handling is 0% of decode on this dataset**. It was
on the candidate list for this lane; it is not a lever here at all.

---

## 3. The floor

| measurement | GB/s |
|---|---|
| memcpy, 1 thread | 68.3 |
| memcpy, 6 threads | 124.7 |
| memcpy, 18 threads | 180.8 |
| bolt snappy on `URL` pages, 1 thread, **hot** destination | **3.00** |
| bolt snappy on `URL` pages, 1 thread, **cold** 1 GiB rotating destination | **2.99** |
| bolt snappy, 6 threads (page-parallel) | 17.90 (5.97x) |
| bolt snappy, 18 threads (page-parallel) | 45.06 (15.0x) |

Two things fall out and they are both decisive.

**Decompression is compute-bound.** 3.00 GB/s is 4.4% of single-thread memcpy.
Hot vs cold destination is a 0.3% difference — i.e. none. Anything that
addresses memory behaviour (prefetch distance, non-temporal stores, arena
warming, destination locality, huge pages for the output) **cannot help**,
because the destination cache state provably does not matter. That is a
measured negative, not an argument.

**Page-parallel decompression has no wall anywhere near the current operating
point.** Pages are independent by format, so this scaling curve is the honest
upper bound on a page-parallel decoder — measured with no engine change. At 18
threads snappy reaches 45 GB/s against a 181 GB/s memcpy ceiling, i.e. 25% of
bandwidth. The campaign runs at 6 workers on an 18-core box.

---

## 4. bolt's snappy vs Arrow's, on identical bytes

Interleaved A/B/A/B, 3 rounds, min-of-3 within each, box load average 8.7
(busy — and it did not matter, because both sides are single-threaded and
compute-bound):

| round | bolt GB/s | Arrow GB/s |
|---|---|---|
| 1 | 3.00 | 6.55 |
| 2 | 2.98 | 6.57 |
| 3 | 3.00 | 6.49 |

**2.17–2.19x, reproducible to under 1%.** Output verified byte-identical on
12 real pages (11.9 MB). If anything the comparison flatters bolt: bolt
decompresses into a preallocated buffer while `pyarrow.decompress` allocates
its output every call, and Arrow's snappy is a prebuilt wheel binary compiled
for a baseline arm64 target while bolt's is compiled `-march=native` here.

The gap tracks the **back-reference share**, not the byte volume:

| col | compression ratio | bolt GB/s | Arrow GB/s | Arrow/bolt |
|---|---|---|---|---|
| `WatchID` | 1.00 (all literal) | 62.49 | 63.94 | **1.02x** |
| `UserID` | 1.10 | 12.15 | 14.74 | 1.21x |
| `SearchPhrase` | 2.13 | 2.00 | 3.07 | 1.53x |
| `Title` | 3.11 | 2.89 | 5.00 | 1.73x |
| `URL` | 3.06 | 3.00 | 6.57 | **2.19x** |

**bolt's literal path is already at parity; the entire deficit is in the copy /
back-reference path.** That is precisely the code `PROJECT_MAP.md` records as
having been rewritten once already (per-tag `memcpy` → fixed-width inline
stores, 1.55x) — so the previous rewrite was real and there is still 2.19x
sitting on top of it. The tree's own note that "per-tag cost is the whole cost"
holds: at 3.00 GB/s and ~6.3 output bytes per tag, bolt is spending roughly 8
cycles per tag where Arrow spends about 4.

`SearchPhrase` is the useful outlier — it is *slower* than `URL` (2.00 vs
3.00 GB/s) at a *lower* ratio, because it is short strings and empties, i.e.
more tags per output byte. Confirms the cost model is per-tag, not per-byte,
and means a fix aimed at long matches will under-deliver on that column.

---

## 5. Lever ranking

Projected column-decode gain = `snappy_share / speedup + rest_share`.

### L1 — Close the snappy back-reference gap to Arrow parity. **DO THIS FIRST.**

- **Share addressed:** 78% (`URL`), 84% (`Title`), 79% (`Referer`), 68%
  (`SearchPhrase`, `EventTime`), 30% (`UserID`).
- **Measured headroom:** 2.19x / 1.73x / 1.53x / 1.21x by column (above).
- **Projected:** `URL` column decode **1.74x**, `Title` **1.55x**. On a
  url-scanning query where decode is ~91% of the work, roughly **1.6x**.
- **Confidence: HIGH.** Measured against an independent implementation on
  identical input with verified-identical output, reproduced 3x.
- **Where to look:** the copy path only — `snappy_copy_match` and the short-
  literal lane in `bolt_snappy.h`. The literal path is at parity; do not touch
  it. Note the deliberate deviation recorded in `PROJECT_MAP.md`: the fast
  lanes are gated on `op + len + 8 <= dst_len` with an exact tail because three
  callers size their destination exactly, and removing that guard reproduces an
  ASAN heap-buffer-overflow. Any rewrite must keep that property.
- **Gate:** the differential fuzz already in `test_bolt_snappy.cpp` (synthesised
  streams, because bolt's own compressor emits a single literal chunk and cannot
  reach the changed code at all), plus a full-column decode checksum old-vs-new
  on real `hits.parquet` and `lineitem.parquet` — identical decoded bytes cannot
  change any downstream result.

### L2 — Page-parallel decompression inside one scan task. **BIGGEST CEILING, UNPROVEN DELIVERY.**

- **Primitive scaling: measured, linear, HIGH confidence** — 5.97x at 6 threads,
  15.0x at 18, no bandwidth wall (25% of memcpy at 18t).
- **But the deliverable gain is not 15x.** The engine already gets
  decompression parallelism implicitly from row-group-level worker
  parallelism. The gain is only whatever raises *effective* decode parallelism
  above the current worker count — on this 18-core box at 6 workers that is a
  ceiling of ~2.5x on the snappy share, and it is capped by whatever the
  scheduling lane already measured (op-level Amdahl ≤1.04x, P-core saturation
  at w=4 on Q6). **This lane measured the primitive, not the delivery.** Whether
  the driver can hand a scan task a pool to fan its pages across is a
  scheduling question and should be answered before anyone writes the decoder.
- Sequence it **after** L1: L1 is a pure win that also multiplies whatever L2
  eventually delivers.

### L3 — Zero-copy StringView over the decompressed page. **REAL BUT BOUNDED, AND RISKY.**

- **Share addressed:** `rest` = 22% (`URL`), 16% (`Title`).
- Confirmed from source, not inferred: `sv_from_bytes` copies **every** string
  body — `sv_copy_small` for ≤12 B into the inline `prefix`/`inline_data`
  region, `sv_copy_bulk` into the arena overflow buffer above that. For `URL`
  that is 895.8 MB of bodies copied a second time, plus 172 MB of StringViews
  written, in 73.7 ms — about **14.5 GB/s**, against a 68.3 GB/s memcpy
  ceiling, i.e. ~7.3 ns per value. The decompressed page already holds the
  bytes contiguously; a view could point into it.
- **Ceiling if the body copy vanished entirely: 1.24x** on `URL` column decode.
  It cannot be more, because 78% of the column's time is not here.
- **Risk: HIGH.** `PROJECT_MAP.md` documents three separate production bugs in
  exactly this family (G2FEAT-283 / -295 / -307): a `StringView` whose spilled
  bytes do not outlive every consumer across a morsel or partition boundary.
  A view into a page buffer makes the page buffer's lifetime load-bearing for
  every downstream operator. Only worth doing with that contract written down
  and asserted.

### NOT worth building — with the measurement that kills each

- **Anything dictionary-aware for `URL`/`Title`/`Referer`.** 6.2% / 8.0% / 5.4%
  of values are dictionary-encoded. Already ruled out; now quantified at page
  granularity. (`SearchPhrase` at 45% and `UserID` at 97% are different, but
  they are 0.83 GB and 0.31 GB whole-file — there is no time there to win.)
- **A faster snappy literal path.** `WatchID`, ratio 1.00, pure literal: bolt
  1.02x of Arrow. Parity. Nothing to get.
- **Memory-side tuning of decompression** — prefetch distance, non-temporal
  stores, arena pre-touch, output locality, huge pages for the destination.
  Hot vs cold destination is 3.00 vs 2.99 GB/s. The destination provably does
  not matter; snappy runs at 4.4% of memcpy bandwidth. This is a dead end and
  should not be re-attempted. (`PROJECT_MAP.md` already records a related
  measured negative: reusing the snappy page buffer to avoid arena first-touch
  faults saved ~3 ms in 3250.)
- **Definition-level decoding.** All 105 columns are REQUIRED; `max_def` is 0.
  Exactly 0% of decode on this dataset.
- **Page-index-driven page skipping.** Already recorded dead — `hits.parquet`
  carries no page index. No new evidence; the page walk here confirms it.

---

## Reproducing

```
clang++ -std=c++20 -O3 -g -march=native -fno-exceptions -fno-rtti -w \
  -I extern/bolt/include scratchpad/decode_attrib.cpp \
  build/coverage/extern/bolt/libbolt_ingest.a build/coverage/extern/bolt/libbolt_io.a \
  -o decode_attrib
./decode_attrib ~/benchdata/clickbench/hits.parquet 20 13,2,14,39,56,9,0,4 3
./decode_floor  ~/benchdata/clickbench/hits.parquet 10 13 3     # + writes the page manifest
python3 arrow_ref.py pages_col13_rg10.csv                       # independent snappy reference
```

Harness sources are in this session's scratchpad
(`decode_attrib.cpp`, `decode_floor.cpp`, `arrow_ref.py`, `sn_hash.cpp`).
They belong next to `benchmarks/bench_parquet_decode.cpp` if this becomes a
standing gate; nothing here supersedes anything under `benchmarks/`.
