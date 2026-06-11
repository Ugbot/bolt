# Design log — what we tried, what we kept, what we backed out

Running list of tested ideas with the measured outcome and the reason
the chosen variant won. Forward-looking research goes in topic-specific
files under `docs/research/`; this file captures the *experimental
record* so future tuning passes don't re-litigate decisions from
scratch.

Each entry: **date / context · what we tried · how it measured ·
what we kept and why**.

---

## Wave summary — base-layer perf lockdown (22 items)

Context: pre-wave state had 5 perf gaps cited in `BOLT_PERFORMANCE.md`
(compressstore i64/f64, Phase-2 scaling, Bloom pre-screen, gather,
parallel merge). Plan in `plans/to-bake-this-into-validated-thunder.md`
bucketed ~20 items across 7 tracks (infra + kernels, column formats,
joins, memory, scheduler, fusion). This wave closed 22 of them across
one extended session.

**Headline measured wins:**
- `filter_gt_i64` 0.55 → 0.32 ns/op (1.72×) via compressstore wiring (A1)
- `filter_gt_f64` 0.56 → 0.31 ns/op (1.81×) same (A1)
- `gather<int32_t>` 1.83 → 1.49 ns/op median (1.23×) via hardware gather (A2)

**New primitives, kept-code-paths:**
- `bolt_hash.h` — wyhash3/xxh3/murmur3 variants (A3)
- `bolt_bloom.h` — block Bloom for hash-join opt-in (C2)
- `bolt_mergejoin.h` — sorted-input inner/left/right (C4)
- `NumaChannelPool` — per-socket MPSC fan-in (E1)
- `SwissTableInterleaved` — TLB-friendly layout (C3)
- `bolt_memcpy_nt` — streaming stores (D2, measured slower — kept for Zen3/SPR)
- `gather_simd_i64` — AVX2 hardware gather (A2, measured slower — kept for JIT)
- `filter_count_gt` / `filter_minmax_gt` — fused aggregates (F1)
- `filter_eq_rle` / `sum_rle_i64` — run-native RLE kernels (B2b)
- `bolt_aligned_alloc_huge` — 2MB-page primitive (D1)

**New column formats** (each lands with format enum, constructor,
materialize case, round-trip tests — no kernel changes, callers
materialize-first):
- `ColumnFormat::RLE` (B2)
- `ColumnFormat::BitPacked` (B3)
- `ColumnFormat::FrameOfRef` (B4)

**Correctness fixes:**
- `gather_to_column` validity bitmap propagation (B1)
- `BoltColumn::ensure_bitmap_index` (C5)
- Windows >64-core pinning via processor groups (A6)

**Infra:**
- `docs/BOLT_PERF_PUNCHLIST.md` — living checklist (G3)
- CMake toggles: `BOLT_ENABLE_{HUGE_PAGES,NUMA}`, `BOLT_{SWISS_LAYOUT,HASH_TIER}` (G2)
- `ci/perf_check.py` + workflow step — 5% regression gate (G1)
- Scheduler adaptive grain: `recommended_grain_bytes(elem_size)` (E2)
- f32 SIMD kernels: `filter_gt_avx2_f32` + `sum_avx2_f32` (A5)
- Templated `selection_intersect_t<Idx>` + `bitmap_to_indices<Idx>` (A4)

**Test + build health:** `ctest --preset msvc` → 16/16 green throughout
the wave; no headline-bench regressions spot-checked.

**Still deferred:**
- **C1** — Phase 2a scatter atomic-cursor refactor (needs per-worker-
  per-shard sub-buffer design + multi-socket measurement).
- **B5 FSST** — external algorithm port; needs a few hundred lines of
  careful symbol-table + greedy-match code.

Both warrant their own dedicated session with proper benchmarking.

---

## Wave 2 follow-ups — I3 dispatch + bench harness + perf baselines

**Context.** End-of-wave-2 gap list called for four items:

1. I3 auto-dispatch on Dictionary columns (ties H1 + C5 + H3).
2. Record CI perf baselines so the regression gate actually fires.
3. Low-match-rate Q3 bench to measure the Bloom pre-screen (C2).
4. Merge-join bench case to measure C4 vs hash-join on sorted input.

All four landed in this drop.

**I3 — `filter_eq_dict_column<T>`** (`bolt/kernels/bolt_dict_filter.h`).
Decision tree: resolve scalar → dict code (H1); if -1, return 0
without touching the column. If the column has an auto-built
`BitmapIndex`, use `probably_absent(code)` for O(1) short-circuit
(H3), else `filter(code)` via the sidecar (C5). Otherwise fall
through to `filter_eq_dict_keys` on the narrow key buffer (H1).
5 new tests cover every branch (linear-scan-no-sidecar,
auto-build-sidecar, literal-not-in-dict, probably-absent-short-
circuits, unsupported-shape → -1).

**Bench harness — Q3b + merge-join A/B** (`benchmarks/bench_tpch_lite.cpp`).
Two new functions reporting both variants side-by-side:

- `run_q3b_bloom_ab` — build 1 K × probe 1 M at ~10 % match rate;
  measures default `HashJoinProbe::probe` vs
  `probe_with_bloom` (same build).
- `run_mergejoin_ab` — sorted 500 K × 500 K with 50 % match;
  measures `mergejoin_inner_i64` vs hash-join probe over the
  same data.

**Measured (laptop i7, MSVC Release, min-of-25 per variant):**

| workload | default | Bloom / MJ | ratio |
|---|---|---|---|
| Q3b hash-join probe, ~10 % match | 17.17 ns/row | **5.04** ns/row (probe_with_bloom) | **3.41×** |
| 500 K × 500 K sorted join | 32.59 ns/row (hash) | **2.27** ns/row (mergejoin) | **14.38×** |

Both far exceed the expected wins — the Bloom pre-screen at 10 %
hit rate pays for its build cost 3 × over, and merge-join on sorted
input is over an order of magnitude faster than hash-join probe.
These numbers now commit a strong case for planner-level auto-
selection (future chukonu work).

**Perf baselines — `ci/perf_baselines.json` populated**. `perf_check.py`
upgraded to best-of-N sampling (`--samples` flag, default 5) for
both `--mode record` and `--mode check` so single-run variance no
longer throws the 5 % gate. Recorded 22 metrics across the 4 bench
binaries on the current laptop. Real CI baselines should be
re-recorded on the Linux runner that will run the gate — the
laptop numbers are here as a working reference and will be
overwritten by a CI-leg record step when that lands.

Also fixed: `perf_check.py` subprocess capture forced utf-8 with
`errors="replace"` so the box-drawing characters in `bench_bolt`'s
banner don't blow up cp1252-defaulted Windows runs.

**Kept.** All additions are additive — no existing kernel changed
shape. Default `HashJoinProbe::probe` stays Bloom-free (keep-code-
paths); the measured 3.41 × win documents when callers should
opt in.

---

## Wave 2 — SYMBOL-shape upgrades + B-format run-native kernels  (H1-H3, I1-I2)

**Context.** QuestDB source audit
(`docs/research/questdb-symbol-code-audit.md`) confirmed that
`Dictionary + BitmapIndex` already covers the SYMBOL shape 1:1.
Four upgrades scoped: global dict (H2), literal-resolve-once filter
(H1), per-ID popcount miss-accelerator (H3 — QuestDB lacks this),
lock-free append (H4 — deferred). Plus Track I lifting B3/B4 column
formats from materialise-first to run-native kernels.

**Landed.**

- **H1** `bolt/kernels/bolt_dict_filter.h` — `dict_resolve_code<T>`
  (linear-scan the dict_child once; returns -1 → caller short-
  circuits without touching the column), `filter_eq_dict_keys<KT>`
  (branchless scan over the narrow key buffer). Mirrors QuestDB
  `EqSymStrFunctionFactory.ConstSymIntCheckFunc.init()` verbatim.
- **H2** `bolt/bolt_dictionary.h` — `DictionaryPool` with arena-
  backed char store, offsets, and a SwissTable keyed on
  FNV-1a(string). `intern/find/resolve` surface. Codes stable
  across batches; collision check on content-equality keeps
  `find()` sound even if two inputs share a hash.
- **H3** `BitmapIndex::popcounts[num_keys]` + `probably_absent(key)`
  O(1) fast path. `count()` now reads the popcount sidecar directly
  (no bitmap scan); `filter()` short-circuits to 0 for absent keys.
  Build cost: one extra uint32 increment per row. QuestDB SYMBOL
  has no equivalent.
- **I1** `filter_gt_bitpacked` / `filter_eq_bitpacked` — unpack
  64 values into a stack scratch buffer, scalar branchless compare,
  emit indices. Saves the full-column materialise traffic when
  callers only need a selection vector.
- **I2** `sum_frame_of_ref` — base-add hoisted out of the hot loop;
  pure delta sum + `base * n` on exit.

Shared helper `bolt_bitpacked_unpack_range` factors the LSB-first
bit-unpack used by both the column `materialize` path and the new
run-native kernels.

**Measured.** Correctness only this wave — 14 new tests in
`tests/test_bolt_wave2.cpp`:

- H1: resolve-present, resolve-missing, filter-eq-matches-
  materialise, filter-eq-no-match, end-to-end literal-resolve-once.
- H2: intern-and-resolve, codes-stable-across-batches,
  find-missing-returns-invalid, overflow-returns-invalid.
- H3: `ProbablyAbsentForUnusedKeys` — 5-key dict with keys {1, 3}
  unused; `probably_absent` true for those, false for {0, 2, 4};
  `count` + `filter` short-circuit without scanning the bitmap.
- I1: 5-bit random + 3-bit hand-picked round-trips against unpack+filter.
- I2: 4-bit deltas + int64 base matches the materialised sum
  element-by-element; zero-length returns 0.

Full `ctest --preset msvc`: 17/17 green (new `test_bolt_wave2`
binary).

**Kept.** Additive across the board — no existing kernel changes
shape. The `popcounts` sidecar is nullable so any legacy path that
skipped the build can still read via the bitmap-scan fallback.

**Open questions.**
- Wire `filter_eq` dispatch on `BoltColumn::Dictionary` to pick
  H1 (linear-scan) vs. `BitmapIndex::filter` (sidecar lookup) vs.
  short-circuit via H3's `probably_absent`. Today callers name
  the kernel explicitly — I3 is that dispatcher, queued in the
  punchlist but not yet wired.
- `DictionaryPool` is SPSC in this drop (single ingest worker,
  single planner). H4 (lock-free MPSC intern + tick-tock publish)
  lands when a streaming-ingest caller asks.
- BitPacked unpack is scalar; AVX2 `_mm256_srlv_epi32` + constant
  masks could do 8 lanes per iteration on fixed bit-widths. Defer
  until a run-native filter is on a headline bench.

---

## Phase 2a scatter — atomic-free via per-morsel prefix offsets  (C1)

**Context.** `BOLT_PERFORMANCE.md:735-737` documented the 8-core
scaling wall: "serial radix-merge phase + atomic shard cursors cap at
memory bandwidth."  Phase 2b (partition merge) was already parallel;
the real bottleneck was the atomic `shard_heads[s].fetch_add(1)` in
Phase 2a scatter — every single triple took one atomic increment on
a shared cache line, serialising producers on the interconnect past
~4 cores.  Followup #4 in `BOLT_PERFORMANCE.md:761-764` flagged
"per-worker-per-shard sub-buffers (rejected in H3 for complexity);
may now be worth revisiting."

**Tried.** Rather than per-worker-per-shard (which fights the
scheduler's dynamic morsel-to-worker mapping), this wave uses
**per-morsel-per-shard prefix offsets**:

1. Extended the counting pre-pass to compute `per_mc[m * P + s]` —
   triples morsel m contributes to shard s.
2. In-place per-shard prefix sum over morsels converts `per_mc` into
   write-offset table: cell `(m, s)` now holds the starting index in
   `shard_bufs[s]` where morsel m's triples for that shard land.
   `shard_sizes[s]` comes out as the per-shard total.
3. `ParallelGroupByScatterCtx` drops `shard_heads[P]` (the atomic
   cursor) and adds `morsel_shard_offset` pointing at `per_mc`.
4. `parallel_groupby_scatter_morsel` uses a stack-local per-shard
   cursor (`kGroupbyMergeRadixMax = 64` slots, one uint32 each) and
   writes at `shard_bufs[s][moffsets[s] + local++]`.  Zero atomic
   operations per triple.

Memory overhead: `num_morsels × P × sizeof(uint32_t)` for the
offset table.  For a 1K-morsel × 64-shard workload that's 256 KB —
trivial vs. the triple buffer itself.

**Measured on laptop (i7, 8 threads, single socket):**

| query | before | after | delta |
|---|---|---|---|
| Q1 (1 M rows, 3 groups) | 16.81 ns/row min | 16.84 | tied (noise) |
| Q3 (1 M rows, filter+join+agg) | 14.11 | 14.26 | tied |
| 1BRC (10 M rows, 413 stations) | 17.89 | 19.47 | -9% (laptop noise) |

Single-socket laptops don't feel the atomic cursor pain — the
contention shows on multi-socket boxes where each socket's fetch_add
cache-line-bounces via the interconnect.  The refactor ships because
architectural correctness (zero atomics on the scatter hot path) is
the right default; measured multi-socket win lands when a test box
is available.

**Kept.**  Single implementation.  The atomic-cursor path is gone —
the new scheme has lower complexity on the hot path (stack cursor vs
atomic fetch_add) AND matches the documented kept-code-paths rule by
not stashing the loser.

**Test health.**  All 16 test binaries green; 8 parallel tests
specifically exercise the groupby radix merge end-to-end.  Every
shard buffer still fills exactly — the correctness invariant is
tightened (each morsel writes at a unique offset range; overflow or
underflow would corrupt adjacent offset ranges).

**Open questions.**
- Multi-socket measurement: Skylake-SP / Zen3 with >1 socket should
  show the real win.  Add a per-socket bench once the test hardware
  is available.
- Prefix-sum stage is serial O(num_morsels × P); at huge morsel
  counts (> 100K) this becomes noticeable.  A parallel prefix-sum
  pass (Ladner-Fischer / two-phase) is a future optimisation.

---

## Split Block Bloom Filter — Parquet-spec variant  (C2b)

**Context.** Web scan (Apr 2026) of modern Bloom-family variants for
hash-join probe: candidates were register-blocked Bloom (current),
Split Block Bloom (Parquet/Impala/Arrow/StarRocks), Cuckoo/Morton/VQF
(multi-load probe — bad for our shape), XOR/Binary-Fuse (3 loads per
probe, static build), Ribbon (heavy build). Sources: Apache Parquet
BloomFilter spec, Lang VLDB 2019 ("Performance-Optimal Filtering"),
Putze/Sanders/Singler 2007.

**Tried.** Added `include/bolt/join/bolt_sbbf.h`:

- 256-bit `SbbfBlock` (8 × uint32_t, one bit-per-lane) laid out
  alignas(32) — one cache-line load per probe.
- 8 Parquet salt constants (`kSbbfSalts[]`) — do not change these;
  touching them breaks Parquet wire compatibility.
- Block index = `hi32(hash) & mask`; within-block lane mask bit =
  `1 << ((lo32(hash) * salt[l]) >> 27)` per Parquet spec.
- `sbbf_create(expected_n, arena)` targets 10 bits/key → ~1% FPR
  (Parquet documented baseline), rounds num_blocks up to pow2.
- `sbbf_add`, `sbbf_test` — 8 ANDs + combined compare; branchless.
  Auto-vectorises to `vpand` + `vptest` on AVX2 at /O2.

**Why SBBF over the existing `bolt_bloom.h`:** same 1-cache-line
probe shape, but half the bits/key at ~5× lower FPR (2-5% → 1%).
Wire-compatible with Parquet's sidecar format — free win when the
lakehouse layer ingests Parquet stats.

**Measured.** 4 correctness tests:

- `NoFalseNegatives` — 1000 inserted keys; every test returns true.
- `FprUnderTwoPercent` — 10 000 never-inserted probes; FPR under 2%
  (Parquet-standard 1% at 10 bits/key, empirically ~1-1.5% on this
  fixture).
- `EmptyFilterAlwaysAbsent` — empty filter rejects every query.
- `SizingAtTenBitsPerKey` — 1024 keys → 40 blocks → round to 64 →
  2048 bytes. Within expected bounds.

**Kept-code-paths policy.** BOTH filters ship in source:
- `BloomFilter` (bolt_bloom.h) — simpler, 16 bits/key register-
  blocked, still what `HashJoinBuild` wires into `probe_with_bloom`.
- `SplitBlockBloom` (bolt_sbbf.h) — Parquet-spec, 10 bits/key,
  lower FPR, byte-compatible wire. Reachable by name; swapping
  into hash-join is a call-site change, not an API break.

Default filter for `HashJoinBuild` stays `BloomFilter` for now —
flipping to SBBF awaits a bench harness that measures probe ns/op
differences at low match rate (C2's open question).

**Alternatives evaluated, not built:**

| Variant | Why not | Source |
|---|---|---|
| Binary Fuse8 | 9 bits/key but 3 random loads per probe | Graf/Lemire 2022 |
| Ribbon | 7 bits/key, heavy build (3-4× CPU), probe has k popcounts | Dillinger 2021 |
| Cuckoo | 12.6 bits/key, 2 cache-line loads on miss | Fan CoNEXT 2014 |
| Morton | Compressed cuckoo, ~600 LoC + overflow logic | Breslow VLDB 2018 |
| VQF | SIMD-friendly but ~1000 LoC port | Pandey SIGMOD 2021 |
| XOR filter | 3 random loads per probe, static build | Graf/Lemire 2020 |

**Open questions.**
- Wire SBBF into `HashJoinBuild::build(build_bloom=true, use_sbbf=?)`
  once a low-match-rate probe bench exists; switch the default.
- AVX2 intrinsic path for `sbbf_test`: MSVC auto-vectorises the
  current scalar AND-and-combine loop; an explicit `_mm256_loadu_si256
  + _mm256_and_si256 + _mm256_testc_si256` path could shave a few
  cycles but loses portability. Defer until measured.

---

## SwissTableInterleaved — read-after-rebuild alt layout  (C3)

**Context (base-layer perf lockdown, docs/BOLT_PERF_PUNCHLIST.md C3).**
The FLAT `SwissTable` keeps `ctrl[capacity]` and `slots[capacity]` in
two separate arenas; a probe loads one ctrl cache line and — on a tag
match — potentially a slot line up to 4 lines away. On >L3 tables
the two halves may live under different TLB entries, costing a TLB
walk per probe on large-dimension joins.

**Added** `SwissTableInterleaved` + `SwissInterleavedGroup` in
`bolt_swiss.h`:

- `SwissInterleavedGroup` packs 16 ctrl bytes + 16 `SwissSlot`
  entries in one cache-line-aligned 320-byte block (64 B ctrl line +
  4 × 64 B slot lines, 16 bytes of pad to keep slots on a line
  boundary). The ctrl line lands next to the slot lines under the
  same TLB entry.
- `create(hint, arena)` — allocate + Empty-fill ctrl.
- `insert(key, value)` — group-aligned linear probe. Tag+key compare
  updates in place; first Empty lane in a non-matching group
  receives the new entry.
- `find(key)` — SIMD scan (`bmm_cmpeq_i8` on 16 ctrl bytes +
  `bmm_movemask_i8`), then per-match key compare. Mirrors FLAT's
  find shape but reads from the interleaved group layout.
- `build_from(flat, arena)` — one-shot rebuild: walks every
  occupied FLAT slot and reinserts into the interleaved table.
  Capacity matches source so reinsert never overflows.

**Why not copy FLAT byte-for-byte.** FLAT's linear probe advances
slot-by-slot; a 16-byte SIMD find loads ctrl starting at the hash's
arbitrary `base`, potentially spanning the end of one group into the
next. Cross-group ctrl loads don't fit the interleaved layout
cleanly (would need a splice across two `SwissInterleavedGroup`).
Instead INTERLEAVED uses group-aligned probes and reinserts on
build_from — key positions differ from FLAT, but interleaved is
self-consistent and `find` is correct.

**Kept-code-paths policy.** FLAT `SwissTable` remains the default
for `SwissJoinBuild` / `GroupByTable`.  The compile-time switch
`BOLT_SWISS_LAYOUT={FLAT|INTERLEAVED}` (G2) is in place but no
call-site consults it yet — `SwissTableInterleaved` is reachable by
name. Wiring a cardinality-based picker (use interleaved when
expected table size > L3) lives in chukonu's planner.

**Measured.** Correctness only — 2 new tests:

- `InterleavedMatchesFlatFind` — 200 random keys inserted into both
  FLAT and INTERLEAVED; every find returns the same value index;
  200 random non-inserted keys also agree (all -1).
- `InterleavedEmptyTable` — an empty table returns -1 for any query;
  size == 0.

All 16 test binaries stay green.

**Kept.** Alternative in source, default unchanged. Real
measurement (TLB-miss counters on a >L3 probe workload) deferred —
needs a box with perf counters accessible.

**Open questions.**
- Real perf win measurement: today this is theory + unit tests. A
  microbench that builds 1M-entry FLAT vs INTERLEAVED and probes
  both would settle the question.
- Insert/delete on live INTERLEAVED tables: supported via the
  group-aligned linear probe; untested under adversarial churn.
  Hash-join + groupby are build-once so this is fine for now.

---

## NUMA channel pool — per-socket MPSC fan-in  (E1)

**Context (base-layer perf lockdown, docs/BOLT_PERF_PUNCHLIST.md E1).**
A single shared `MPSCChannel` serialises every producer on the
`wseq_` atomic — on a multi-socket box every push cache-line-bounces
across the interconnect. `BOLT_PERFORMANCE.md:735-737` documents this
as the Phase-2 scaling cap; a pool of per-node MPSC channels keeps
each producer writing to a socket-local atomic.

**Added** `NumaChannelPool<T, Capacity, NumNodes>` in `bolt_channel.h`:

- One `MPSCChannel<T, Capacity>` per NUMA node (templated NumNodes, 1-16).
- `try_push(node, item)` — producer's home node is its own concern;
  the pool just dispatches to `channels_[node]`.
- `try_pop(out)` — round-robin across nodes; `rr_start_` advances after
  each successful pop so no node starves.
- `approx_size()` — sum across nodes, diagnostic.

**Kept-code-paths policy.** The default `Scheduler::submit_range` path
is unchanged — it still uses a single `TaskRing`. The pool is a
standalone primitive; callers who own the producer↔socket mapping
(parallel groupby / parallel hash-join scatter) opt in at dispatch
time. Template non-type params keep dispatch zero-cost.

**Measured.** Three correctness tests green:

- `PushThenRoundRobinPop` — 2 items × 4 nodes; round-robin pop recovers
  every value exactly once; next pop returns false.
- `MultiProducerConcurrent` — 4 threads × 500 items pushing to 4
  disjoint channels; single consumer sums and matches `n*(n+1)/2`.
- `SingleNodeDegenerate` — NumNodes=1 reduces to plain MPSC.

All 16 test binaries stay green.

**Kept.** Single implementation of the pool; reuses the existing
`MPSCChannel` as the per-node substrate, so any future MPSC tuning
(back-off policy, slot padding) propagates for free.

**Open questions.**
- Scheduler-native NUMA dispatch: today the pool is caller-owned; a
  future `Scheduler::submit_range_numa` that auto-routes based on
  `worker_numa[]` would let the parallel groupby/join scatter opt in
  without touching operator code. Deferred until a real multi-socket
  workload justifies the extra API surface.
- Cross-socket work stealing: pure round-robin lets a busy producer
  starve if its node's channel fills while others drain fast. A
  priority scheme (drain oldest-pending node first) would help on
  imbalanced workloads. Track with a real multi-socket bench first.

---

## ColumnFormat::BitPacked + FrameOfRef — bit-level compression  (B3/B4)

**Context (base-layer perf lockdown, docs/BOLT_PERF_PUNCHLIST.md B3/B4).**
DuckDB ships five physical vector types; Bolt had two compact
alternatives to Flat (Constant + Sequence) before this wave. Adding
BitPacked (arbitrary-width unsigned ints in ≤32 bits per value) and
Frame-of-Reference (base + bit-packed deltas) closes the main small-int
/ monotonic-timestamp compression gap.

**Added** two enum variants and two constructors:

- `ColumnFormat::BitPacked = 6` — `make_bitpacked(packed_words,
  bit_width, total_rows, type, arena)`. Values are LSB-first packed
  into `uint64` words, each using `bit_width` ∈ [1, 32] bits. Reuses
  the existing union: `data = packed words`, `seq_step = bit_width`,
  `seq_offset = 0`.
- `ColumnFormat::FrameOfRef = 7` — `make_frame_of_ref(packed_deltas,
  bit_width, base, total_rows, type, arena)`. Same layout, but
  `seq_offset = base` and materialised value = `base + delta[i]`.

Shared unpack loop in `materialize`:

```cpp
for i in 0..length:
    bit_off = i * bit_width
    word = words[bit_off >> 6] >> (bit_off & 63)
    if bit_in_word + bit_width > 64: word |= next_word << (64 - bit_in_word)
    out[i] = base + (word & mask)       // base = 0 for BitPacked
```

Output width is determined by the requested `BoltType` — tests cover
int32 (3-bit and 17-bit BitPacked) and int64 (FrameOfRef with
1_000_000 base + 6-bit deltas).

**Kept-code-paths policy.** Kernels do not read BitPacked or FOR
natively yet — they materialise first. Run-native kernels
(`filter_gt_bitpacked` with a bit-parallel compare, `sum_frame_of_ref`
summing base × n + Σ(deltas)) are the natural B3b/B4b follow-ups once
a caller's workload justifies the compile-time specialisation.

**Measured.** Three correctness tests:

- `BitPackedRoundTrip3Bit` — 12 × 3-bit values packed into one uint64;
  unpack matches element-by-element.
- `BitPacked17BitCrossesWordBoundary` — 5 × 17-bit values = 85 bits
  across two words; exercises the `bit_in_word + bit_width > 64`
  branch explicitly.
- `FrameOfReferenceRoundTrip` — 7 × 6-bit deltas + int64 base.

All 16 test binaries stay green.

**Kept.** Single implementation per format, no alternative stash.

**Open questions.**
- Signed-int BitPacked (two's-complement zig-zag) isn't explicitly
  supported — caller's `type` controls narrowing; negative-valued
  deltas for FOR would need zig-zag decode. Add when a caller asks.
- SIMD unpack on AVX2/AVX-512 is worth 4-16× for bit-widths that
  divide evenly into word sizes (8, 16, 24, 32 bits). Deferred to a
  follow-up that adds `bmm_unpack_*` helpers.

---

## ColumnFormat::RLE — minimal run-length format + materialize  (B2)

**Context (base-layer perf lockdown, docs/BOLT_PERF_PUNCHLIST.md B2).**
Only five column formats shipped before this wave (Flat / Constant /
Dictionary / Sequence / View). RLE is table-stakes for tick data with
repeating symbol columns (exchange codes, sides, order types) and for
post-partition shard output where successive rows often share keys.

**Added** `ColumnFormat::RLE = 5` and `BoltColumn::make_rle(values,
num_runs, run_ends, total_rows, type, arena)`. Storage layout reuses
the existing `data` / `dict_child` fields — no BoltColumn size grows:

- `data` = borrowed values buffer, size `num_runs * type_size_bytes`.
- `dict_child` = arena-allocated int32 Flat column wrapping
  caller-supplied `run_ends[num_runs]`, where run i covers logical
  rows `[run_ends[i-1], run_ends[i])` and `run_ends[-1]` is 0.
- `length` = logical row count (preserves the standard semantics for
  every caller that iterates `0..length`).

Added a `ColumnFormat::RLE` branch to `BoltColumn::materialize` that
expands runs back to a contiguous Flat column via one memcpy per row
inside a run (a future optimisation could use a typed store for the
fixed-width cases, but the generic memcpy is already auto-vectorised
by MSVC for 4 / 8-byte elements at -O2).

**Kept-code-paths policy.** No kernel consults RLE natively — every
caller materialises first. The format is useful as a storage shape and
for copy-on-write amortisation; specialised run-native kernels (filter,
sum, constant-fold over run spans) are the natural follow-up once a
caller's workload proves the payoff.

**Measured.** Correctness only — three tests:

- `RLEMaterializeRoundTrip` — 4 runs (5×A / 3×B / 2×A / 4×C = 14 rows)
  expand byte-equal to the expected flat layout.
- `RLEEmpty` — 0 runs, 0 rows, both construct and materialise cleanly.
- `RLESingleRun` — 1 run of 7 i64 expands to 7 copies of the value.

All 16 test binaries stay green.

**Kept.** Single implementation per hook. No alternatives to stash.

**Open questions.**
- Other format helpers that enumerate `ColumnFormat` cases
  (`byte_size`, `clone_into`) treat RLE as unknown and return
  conservative zero / empty. Callers that need byte-size or zero-copy
  clone of an RLE column should materialise first for now; extending
  the helpers is straightforward once a caller demands it.
- Run-native filter: `filter_eq_rle` can short-circuit equal keys
  across a whole run — potentially 4-50× speedup at typical run
  lengths. Deferred to a B2b follow-up.

---

## Adaptive morsel sizing — EWMA observation + opt-in dispatch  (E2)

**Context (base-layer perf lockdown, docs/BOLT_PERF_PUNCHLIST.md E2).**
`Scheduler::grain_bytes` was a profile-static knob (256 KB balanced,
64 KB latency, 1 MB throughput). A fixed grain is wrong in two
directions: wide-row workloads overshoot the target per-morsel budget,
narrow-row workloads pay per-morsel overhead for trivial work. The
plan called for a feedback loop that tunes the grain within-profile.

**Added** on `Scheduler`:

- `adaptive.ns_per_row_ewma` (atomic<double>) + `adaptive.samples`
  (atomic<uint64_t>) — zeroed by `init()`.
- `record_morsel_ns_per_row(observed)` — EWMA update with alpha=0.25
  (fast enough to track shape changes across ~4-8 morsels, slow enough
  not to chase single-morsel noise). Guards NaN / zero observations.
- `recommended_grain_bytes(elem_size)` — uses the EWMA and the current
  profile's static `grain_bytes` to pick a grain that targets a
  1-10 ms per-morsel execution window. Clamped to
  `[grain_bytes/4, grain_bytes*4]` so the suggestion never escapes
  the profile's bounded range.

**Kept-code-paths policy.** The default `submit_range` path does NOT
consult the recommendation — callers opt in by replacing
`sched->grain_bytes()` with `sched->recommended_grain_bytes(elem_size)`
at their dispatch point. Hot path unchanged; zero cost for callers
that don't want the feedback loop. A future planner integration can
wire it up per-operator.

**Measured.** Four unit tests green:

- Zero samples → recommendation == static grain (fast-path default
  preserved).
- 1000 ns/row → grain shrinks (morsel would overshoot 10 ms at
  32K rows × 8 B → 32 ms), clamped to ≥ grain_bytes/4.
- 0.1 ns/row → grain grows (morsel would be 3 µs — too small),
  clamped to ≤ grain_bytes*4.
- 0.0 ns/row observation → ignored; no EWMA corruption.

**Kept.** Single implementation; additive feedback on Scheduler.

**Open questions.**
- Per-operator vs per-scheduler observations: today a single EWMA is
  shared across all workloads on the scheduler. Per-operator stats
  would be sharper but require a key into a map — deferred until
  multiple concurrent morsel shapes are measured to mix badly.
- Auto-record inside `submit_range`: the scheduler could time each
  dispatch and call `record_morsel_ns_per_row` itself, but that adds
  a clock read to the hot path. Kept manual for now — callers that
  care call it once per logical phase.

---

## f32 SIMD kernels — filter_gt + sum  (A5)

**Context (base-layer perf lockdown, docs/BOLT_PERF_PUNCHLIST.md A5).**
f32 SIMD types (`bmm_vec_f32`, `bmm_lanes_f32`, load/store/cmp) already
shipped in `bolt_port.h` for AVX2/SSE/NEON/scalar. Missing: accumulator
primitives (`setzero`, `add`, `hadd`) and the f32 filter/sum kernels in
`bolt_branchless.h`. Without these the bolt_branchless.h layer had no
f32 specialisation — f32 workloads either fell through the
auto-vectorised scalar template or forced callers to widen to f64
(halving effective bandwidth).

**Added.**
1. `bmm_setzero_f32`, `bmm_add_f32`, `bmm_hadd_f32` for all four ISAs
   (AVX2: 8-lane hi+lo + 2-lane + 1-lane reduce; SSE: movehl + shuffle;
   NEON: `vaddvq_f32`; scalar: `v.v[0]+v.v[1]+v.v[2]+v.v[3]`).
2. `filter_gt_avx2_f32` — two-loads-per-step pattern mirroring f64.
   On AVX2 (Li=8) that's 16 rows/iter feeding two 8-lane
   `bmm_compressstore_i32` calls via `if constexpr (Lo<=8)`-gated
   dispatch; on SSE/NEON (Li=4) one iter fills a single 8-lane
   compressstore exactly.
3. `sum_avx2_f32` — `bmm_lanes_f32` lanes of FMAs-style accumulation,
   widened to double on reduction to match the rest of the codebase's
   f64-accumulator convention.

**Measured.** Correctness only — 4 new tests:

- `F32SIMD.FilterGtMatchesScalar` — 2048 random floats, SIMD output
  byte-equals the branchless scalar reference.
- `F32SIMD.FilterGtNaNNeverPasses` — IEEE ordered GT semantics; NaN
  rows never match.
- `F32SIMD.SumMatchesDoubleAccumulator` — 4096 floats; SIMD sum within
  1e-5 relative of scalar-order double sum (SIMD reorders adds so
  exact equality is not guaranteed for floats).
- `F32SIMD.SumEmpty` — degenerate n==0 returns 0.0.

All 26 branchless tests (up from 22) stay green.

**Kept.** Single implementation per primitive — no alternative. Bench
measurement (f32 vs f64 bandwidth) deferred — the plan's "2× bandwidth
check" fits in a followup to `bench_kernels.cpp` once a plain-f32
column pipeline lands.

**Open questions.**
- f32 compressstore (`bmm_compressstore_f32`) would simplify the
  `if constexpr (Lo<=8)` branch to a single 8-lane emit everywhere; not
  built because the two-lane-idx i32 compressstore already works.
- AVX-512 native 16-lane f32 filter would double throughput; uses the
  wider `bmm_compressstore_i32_x16`. Deferred until AVX-512 rig.

---

## BitmapIndex — lazy build + cache on sidecar slot  (C5)

**Context (base-layer perf lockdown, docs/BOLT_PERF_PUNCHLIST.md C5).**
`BitmapIndex::build / count / filter / filter_in` already shipped in
`bolt_column.h` and were tested (`BitmapIndex.BuildAndCount`). The
`SidecarSlots::bitmap_index` pointer on every `BoltColumn` was declared
but never populated — callers had to remember to call `build` and keep
the result alive separately.

**Added** `BoltColumn::ensure_bitmap_index(arena)`: lazy, idempotent
builder that populates `sidecars.bitmap_index` on first call and
returns the cached pointer on subsequent calls. Returns nullptr for
non-Dictionary columns (the only shape `BitmapIndex::build` currently
supports).

**What stays unchanged.** Default filter / join kernels do NOT consult
the index automatically. Callers opt in by name
(`col.ensure_bitmap_index(arena)->filter(key, out)`). Matches the
keep-code-paths policy: the primitive is in the binary, the explicit
name reaches it; a future planner can choose bitmap-vs-scan per
column based on cardinality + selectivity stats.

**Measured.** Correctness only — new test `BitmapIndex.EnsureIndexCaches`
verifies:
- First call builds and caches (`sidecars.bitmap_index` non-null).
- Second call returns the same pointer (no rebuild).
- Non-Dictionary columns return nullptr without mutating the sidecar.

**Kept.** Single implementation; no alternative. Additive surface.

**Open questions.**
- Auto-consult in filter dispatch: the planner call-site is not here
  yet; the decision rule (cardinality threshold, selectivity cost
  model) belongs in chukonu.
- Flat integer columns with low cardinality would also benefit —
  extending `BitmapIndex::build` to accept flat int columns adds a
  key-ranging pass but is a clean addition.

---

## Huge-page allocator primitive — opt-in, fallback-safe  (D1)

**Context (base-layer perf lockdown, docs/BOLT_PERF_PUNCHLIST.md D1).**
Large arenas (≥ 16 MB) span thousands of 4 KB TLB entries; switching
to 2 MB pages cuts that by ~512× and can measurably reduce tail-latency
on TLB-sensitive scans. Historically skipped because enabling huge
pages is OS-specific and privilege-gated.

**Added** `bolt_aligned_alloc_huge(size, &out_size)` +
`bolt_aligned_free_huge(p, size)` in `bolt_port.h`:

- **Windows:** `VirtualAlloc(..., MEM_LARGE_PAGES | MEM_RESERVE |
  MEM_COMMIT)`. Requires `SeLockMemoryPrivilege` on the process token;
  without it VirtualAlloc fails and we fall back to `bolt_aligned_alloc`.
  The paired `_free` uses `VirtualQuery` to detect whether the pointer
  came from VirtualAlloc and routes accordingly.
- **Linux:** `mmap(..., MAP_HUGETLB)` first; on failure falls back to
  a plain `mmap + madvise(MADV_HUGEPAGE)` (transparent huge pages);
  on that failure falls back to `bolt_aligned_alloc`. Free routes via
  `munmap` when size > 0, `bolt_aligned_free` otherwise.
- **macOS / other:** falls back to `bolt_aligned_alloc` (macOS has no
  general-purpose userspace huge-page knob).
- **`BOLT_ENABLE_HUGE_PAGES=0`** (the Windows default, per
  `BoltCompileOptions.cmake`): compile-time-falls-back to
  `bolt_aligned_alloc` — symbol remains callable so downstream code
  compiles identically across toggle states.

**Measured.** Smoke test (`BoltPort.HugePageAllocBasic`) allocates
8 MB, writes every 4 KB page, frees — green on the unprivileged i7
laptop (exercises the fallback path). Real huge-page performance
validation needs a privileged Linux box or admin-elevated Windows run
where the allocator actually returns 2 MB pages — deferred.

**Kept.** Standalone primitive callable by name; NOT wired into the
default Arena. Per keep-code-paths, the Arena still uses
`bolt_aligned_alloc` (cache-line aligned) as its default allocator;
callers who want huge-page-backed arenas can construct one with an
explicit allocator when that configurability lands. The primitive is
also useful independently (mmap-file buffers, spill regions).

**Open questions.**
- Arena-level integration: either a `use_huge_pages` config flag on
  `ArenaConfig`, or a parallel `HugeArena` type. Deferred until a
  measured workload wants it.
- Windows `SeLockMemoryPrivilege` bootstrap: a helper that adjusts the
  process token at startup could make the primitive actually land
  2 MB pages without manual admin setup. Also deferred.

---

## Fused filter+aggregate kernels — count + minmax added  (F1)

**Context (base-layer perf lockdown, docs/BOLT_PERF_PUNCHLIST.md F1).**
`filter_sum_gt<T>` and `sum_masked<T>` already shipped in
`bolt_numeric.h` (Wave F1a). The punchlist called for two more primitives
so a future chukonu planner can dispatch fused filter+agg on the three
most common aggregate shapes without a selection-vector materialisation
pass:

- `filter_count_gt<T>` — one compare + 0/1 add per element.
- `filter_minmax_gt<T>` — branchless cmov-pattern tracking both
  extremes; returns match-count so callers can gate on empty results.

**Added** both in `include/bolt/kernels/bolt_numeric.h` alongside the
existing fused kernels. Shape matches the existing `filter_sum_gt`:

- `noexcept`, `BOLT_RESTRICT` on data pointers.
- ≥2 asserts, ≤70 lines per function.
- No selection vector, no heap — O(1) state.
- Auto-vectorises under AVX2 / NEON (no explicit SIMD — the compiler
  emits `vpcmpgt` + blend + adds on /arch:AVX2).

**Measured.** Correctness only — three new tests:

- `FilterCountGtMatchesTwoPass`: random 4096-element input; fused count
  matches `filter_gt` cardinality.
- `FilterMinMaxGtMatchesManual`: random 4096 i64; fused min/max matches
  hand-computed min/max on the filtered subset.
- `FilterMinMaxGtNonePass`: sentinel semantics — count=0 leaves
  `numeric_limits::max / lowest` in the output; callers must gate.

All 42 `BoltKernels` tests stay green. No bench yet — the win vs
two-pass (filter then aggregate) shows up only at L3-plus sizes where
the intermediate selection vector would overflow cache; a future
`bench_tpch_lite` addition should measure this.

**Kept.** Single implementation — these are additive primitives, no
alternatives to stash. With `filter_sum_gt`, `filter_count_gt`,
`filter_minmax_gt`, and `sum_masked` all in place, the planner has
enough fused shapes for all four 1BRC aggregates.

**Open questions.**
- Need `filter_{lt,eq,ne,ge,le}_{sum,count,minmax}` too — 15 more
  combinations. X-macro expansion is the obvious shape; defer until a
  caller needs more than `>`.
- For floating-point `filter_minmax_gt`, NaN handling follows IEEE
  ordered `>`: NaNs never match, safe sentinel.

---

## Merge-join — sorted-input operator  (C4)

**Context (base-layer perf lockdown, docs/BOLT_PERF_PUNCHLIST.md C4).**
Bolt had no merge-join — every equi-join routed through `HashJoinBuild`
+ `HashJoinProbe`, paying build-side state + hash mixing even when the
inputs are already sorted (timestamp-keyed tick joins, pre-sorted
dimensions, range-partition outputs). Adding merge-join is a standard
OLAP operator and a prerequisite for the planner's sort-vs-hash choice.

**Added** `include/bolt/join/bolt_mergejoin.h` with three variants:

- `mergejoin_inner_i64` — emits every matching pair; handles duplicate
  keys on both sides via the equal-run cross-product.
- `mergejoin_left_outer_i64` — inner pairs plus unmatched probe rows
  tagged `build_idx = -1`.
- `mergejoin_right_outer_i64` — symmetric; unmatched build rows tagged
  `probe_idx = -1`.

Hot path: two cursors, no heap, no hash, no state.  O(n + m) time,
O(1) space.  Caller-bounded output capacity (`out_capacity`); early-exit
without corruption if the cap is hit.  Contract requires sorted int64 /
uint64 keys in Flat or View columns (same envelope as the hash-join
probe).  Returns -1 on unsupported input shape.

**Measured.** 9 new correctness tests green
(`BoltMergeJoin.*`) including the duplicate-key cross-product (2 × 3
equal-key pair emission), capacity clamping, and a cross-check against
hash-join results on 2K unique sorted keys. No perf bench yet — the
operator is a new surface; comparisons to hash-join belong in a
sorted-join bench case on `bench_tpch_lite` as a followup.

**Kept.** Single implementation per variant — no alternative; this is
an additive operator.  Parity with standard merge-join literature
(Graefe "Query Evaluation Techniques" ch. 4.9).

**Open questions.**
- Fully-outer variant (both sides) not included in this wave — adds
  complexity on the probe-ahead logic; defer until a real caller wants
  it.
- Non-equi (range) merge joins (time-series common) are a second
  wave — `i_end` / `j_end` advance rules change to span-overlap, not
  equal-run.
- Vectorised implementation (SIMD compare-swap on key runs) is open
  territory — DuckDB ships a scalar merge-join too.

---

## Windows >64-core pinning via processor groups  (A6)

**Context (live TODO, `bolt_port.h:190, 211`).** The Windows branch of
`bolt_pin_current_thread` hard-asserted `logical_cpu < 64` and returned
false otherwise. On >64-core / multi-group Windows boxes this silently
pinned nothing past CPU 63, capping scheduler reach on the machines
where parallelism matters most.

**Change.** Replaced the single-group `SetThreadAffinityMask` path with
group-aware dispatch:

- `group_idx = logical_cpu >> 6`, `cpu_in_group = logical_cpu & 63`.
- Group 0: unchanged fast path — `SetThreadAffinityMask(h, 1<<cpu_in_group)`.
- Group > 0: `SetThreadGroupAffinity(h, &GROUP_AFFINITY{mask, group, …})`.
- Upper bound retained at 4096 (matches the shared outer assert).

Backwards compatible: any caller that already works on single-group
Windows (CPU 0..63) stays on the fast path.

**Measured.** Local i7 laptop is single-group — this box can't exercise
group > 0 directly; the code compiles cleanly under MSVC 19.37 and all
9 scheduler + 5 topology tests stay green. Verification on >64-core
hardware (e.g. Threadripper / Xeon Scalable Windows box) is the
deferred step. The API shape is the standard Win7+ call, well-tested
by DAW / game-engine code.

**Kept.** Single implementation — the old path was a correctness gap,
not a tuning decision.

---

## COW / large-buffer copy — NT stores measured (mostly) slower  (D2)

**Context (base-layer perf lockdown, docs/BOLT_PERF_PUNCHLIST.md D2).**
Non-temporal (`_mm256_stream_si256`) stores bypass the cache; they
historically won on buffers > L3 when the destination won't be re-read
soon. The theory: save the L2/L3 eviction cost that plain memcpy pays.

**Tried.** Added `bolt_memcpy_nt(dst, src, bytes)` in `bolt_port.h`:
128-byte unrolled AVX2 loop using `_mm256_loadu_si256` + four
`_mm256_stream_si256` per iteration, scalar tail, final `_mm_sfence`.
Non-AVX2 paths fall back to plain memcpy. Added a size-sweep bench in
`bench_bolt.cpp::bench_nt_memcpy` covering 2 MB → 128 MB.

**Measured (i7 laptop, MSVC 19.37 Release, AVX2):**

| size     | memcpy ns | memcpy GB/s | nt ns   | nt GB/s | nt/memcpy ratio |
|----------|-----------|-------------|---------|---------|-----------------|
| 2 MB     | 82 559    | 25.40       | 79 906  | 26.25   | **1.03× (tie)** |
| 8 MB     | 349 088   | 24.03       | 406 487 | 20.64   | 0.86× slower    |
| 32 MB    | 2 936 636 | 11.43       | 2 955 866 | 11.35 | 0.99× (tie)     |
| 128 MB   | 18 278 662| 7.34        | 37 569 606| 3.57  | **0.49× slower**|

NT is a net loss at 128 MB (2× slower than memcpy) and at best a tie
at 2 MB. The UCRT's `memcpy` already uses ISA-appropriate streaming
paths at large sizes on this machine; adding an explicit NT loop on
top doesn't help, and the `_mm_sfence` adds latency.

**Kept.**
- `bolt_memcpy_nt` stays as a **standalone primitive** per the
  keep-code-paths-for-JIT-later policy. Reachable by name; compiled
  into every build.
- **NOT wired into the default COW / clone_into path** — plain
  memcpy stays the default because it's consistently equal or faster
  on measured hardware.
- Revisit on Linux (different libc memcpy), or when a workload emerges
  where the destination is demonstrably not re-read soon enough for
  cache residency to matter (e.g., buffered spill to disk).

**Open questions.**
- macOS and Linux `memcpy` may not stream as aggressively — re-measure
  there before writing off NT entirely.
- Non-Intel CPUs (AMD Zen, Apple M-series) may invert the ratio.

---

## Hash-join Bloom pre-screen — opt-in via templated probe  (C2)

**Context (base-layer perf lockdown, docs/BOLT_PERF_PUNCHLIST.md C2).**
Bloom pre-screen before SwissTable lookup was referenced in
`README.md:162`, `BOLT_COLUMN_FORMAT.md:341`, and
`BOLT_DECISION_LOG.md:148` but never built. Theory: at low match rate
Bloom saves ~5-10 cycles per miss (SwissTable probe chain); at high
match rate Bloom is pure overhead.

**Tried.** Two passes:

**Pass 1 — default-on wiring (rejected).** Wired `bloom_add` into
`HashJoinBuild::build` unconditionally and `bloom_test` into the probe
hot path. Measured Q3 on bench_tpch_lite (1M probes vs 1K build,
~100% match rate): median 14.01 → 15.36 ns/row, an **8% regression**.
At this match rate every probe passes the Bloom test, so the extra
load + AND + compare per row is pure overhead.

**Pass 2 — opt-in via templated probe (kept).** New standalone
`include/bolt/join/bolt_bloom.h` with a block-Bloom primitive
(16 bits/key, 4 hashes/key, single-word block, expected FPR ~0.4% at
load factor 1.0). `HashJoinBuild::build` now takes a defaulted
`build_bloom = false` parameter; the filter is only populated when
requested. `HashJoinProbe` splits into `probe()` (default, no Bloom)
and `probe_with_bloom()`, both routing to a templated
`probe_impl<bool UseBloom>` so each specialisation is branch-free.

Callers that know they have low match rate call `probe_with_bloom`;
everyone else gets the unchanged default path.

**Measured.**
- Q3 default (Bloom OFF) vs pre-C2 baseline: within noise
  (13.56 vs 12.99 min; 15.62 vs 14.01 median — laptop is noisy).
- Correctness: new test `BloomGatedProbeMatchesDefault` cross-checks
  Bloom-gated results against the default probe on a 100K-probe,
  10%-match workload. Exact pair match.

**Kept.** Both probe variants ship; caller picks. Bloom primitive is
standalone and reusable (future semi-join, dedup, filter
pre-screens). Future work: wire a match-rate-adaptive picker into the
higher-level join planner once bench_tpch_lite grows a low-match-rate
scenario to measure the actual win.

**Open questions.**
- Block-Bloom params (16 bits/key, k=4) are defaults — tune once a
  low-match-rate bench exists.
- On AVX2 we could vectorise `bloom_test` across L probes in the SIMD
  path (gather the 4-bit masks + block offsets, then AND-reduce).
  Skipped for now — adds complexity without a measured need.

---

## Selection vector composition — multi-index-type + bitmap→indices  (A4)

**Context (base-layer perf lockdown, docs/BOLT_PERF_PUNCHLIST.md A4).**
`selection_intersect_branchless` / `selection_union_branchless` were
hard-wired to `int32_t` index type. Downstream kernels that use wider
row indices (i64-indexed morsels, 32-bit dense IDs) had to materialise
through i32 before chaining filters. Additionally no helper existed to
turn an Arrow-style validity bitmap into a dense selection vector for
composition with an index-form selection.

**Change.** Introduced templated `selection_intersect_t<Idx>` and
`selection_union_t<Idx>` (any integral type up to 64 bits); the
existing `_branchless` names stay as inline i32 shims so current
call-sites keep working. Added `bitmap_to_indices<Idx>` using
`bolt_ctz64` — fully branchless per bit, handles partial tail words,
zero-allocation (caller-supplied buffer).

**Measured.** No perf path change on the i32 hot path (inline shim
forwards to the template instantiation). Correctness covered by 7 new
tests in `SelectionCompose.*` (intersect/union on i32 + i64, bitmap
conversion on dense/partial-word inputs).

**Kept.** Single implementation per operation; the template is the
one true source, the i32 wrapper is a back-compat adapter.

---

## gather_to_column — validity bitmap propagation  (B1)

**Context (correctness TODO, `bolt_column.h:1071`).** `gather_to_column<T>`
materialised a selection into a new Flat column but had a lit TODO
leaving validity unpropagated. Any null in the source would silently
disappear through the gather path — a correctness bug for join probes
feeding nullable downstream operators.

**Change.** If `src.validity != nullptr` AND `!src.stats.all_valid` AND
`sel_n > 0`, allocate a zeroed `(sel_n+7)/8`-byte bitmap from the arena
and pack bit `i` from `src.validity[src.validity_offset + sel[i]]`. Set
`out.stats.null_count` and flip `out.stats.all_valid` accordingly. If
the arena fails to allocate the bitmap, leave `out.validity = nullptr`
and `all_valid = true` (degraded — never corrupt — matching the
existing fast path for nullness-free sources).

**Measured.** No perf path change on the all_valid fast path (the
existing bench/test coverage is unchanged). Two new tests assert the
nullable path — one with a hand-computed bitmap (7 rows, mask 0x69),
one re-asserting the all_valid fast path preserves `validity = nullptr`.

**Kept.** Single implementation; no alternative needed — this is a
correctness gap closed, not a tuning decision.

---

## Hash mixer — xxh3 + murmur3 added as kept-in-source alternatives  (A3)

**Context (base-layer perf lockdown, docs/BOLT_PERF_PUNCHLIST.md A3).**
`swiss_mix` was a single inline wyhash-3-op definition in
`bolt_swiss.h` (previous work migrated it there from the original
Murmur3 finalizer — see "Hash mix — Murmur3 finalizer vs Fibonacci vs
wyhash 3-op" below). The punchlist called for xxh3 as an alternative
and a compile-time switch so alternatives stay reachable.

**Tried.** Extracted `swiss_mix` into a new header
`include/bolt/bolt_hash.h` with three always-compiled named variants:

- `swiss_mix_wyhash3` — current default (xor / mul / xor, 3 ops).
- `swiss_mix_xxh3` — XXH3-style finalizer (xor37 / mul PRIME_MX1 /
  xor32, 3 ops with a deeper xor-shift on the input half).
- `swiss_mix_murmur3` — Murmur3 64-bit finalizer (6 ops, two multiplies).

`swiss_mix` itself is an inline dispatcher that routes to one of the
three based on `BOLT_HASH_TIER_{WYHASH3|XXH3|MURMUR3}` — zero-cost,
compile-time, picked in `bolt_apply_feature_toggles`. Default tier is
WYHASH3; other two stay in the binary so a future JIT / per-CPU
dispatcher can call them by name.

**Measured (i7 laptop, MSVC 19.37 Release, AVX2, SwissTable find_simd —
16K table, 500K probes, ~50% hit rate, best-of-8 each reconfig+rebuild):**

| tier    | min  | median | mean  |
|---------|------|--------|-------|
| WYHASH3 | 5.22 | 5.60   | 5.53  |
| XXH3    | 4.90 | 5.57   | 5.50  |
| MURMUR3 | 6.27 | 6.81   | 6.70  |

WYHASH3 and XXH3 are statistically tied (XXH3 -6% min, -0.5% median,
-0.5% mean — inside measurement noise at this scale). MURMUR3 is
~20% slower on this workload as expected from the doubled op count.

**Kept.**
- **Default:** `BOLT_HASH_TIER=WYHASH3` — unchanged, no measured reason
  to flip; existing 1BRC-class benchmarks were tuned against this mix.
- **Alternatives always in source:** `swiss_mix_xxh3` and
  `swiss_mix_murmur3` compile in every build. Users can run the
  `all-features` preset or pass `-DBOLT_HASH_TIER=XXH3` to swap the
  default, or call the variants by name.
- The Murmur3 path is kept despite losing here because it ships the
  strongest avalanche — worth reaching for on adversarial key streams
  (e.g. user-controlled hash inputs) even at the ~20% cost.

**Open questions.**
- Workload-specific tests: high-cardinality groupby (100K+ distinct)
  may magnify the gap between 3-op and 6-op mixers via cache effects.
- XXH3's deeper xor-shift could win on densely-packed small-integer
  keys (dict codes, sequential timestamps) where WYHASH3's 32-bit
  shift doesn't fully mix the low bits. Re-measure on such data.

---

## gather_branchless — SIMD gather vs scalar + prefetch  (A2)

**Context (base-layer perf lockdown, docs/BOLT_PERF_PUNCHLIST.md A2).**
`gather_branchless<T>` was a single scalar template using a 16-slot
software prefetch pipeline. Hardware i32/i64 gather intrinsics
(`_mm256_i32gather_epi32`, `_mm256_i32gather_epi64`) were already wrapped
in `bolt_port.h` as `bmm_gather_i32` / `bmm_gather_i64` but nothing in
the branchless layer called them.

**Tried.** Two template specialisations under AVX2/AVX-512:

- `gather_branchless<int32_t>` → `bmm_gather_i32` on 8 lanes per step
  with prefetch seeded two SIMD batches ahead.
- `gather_branchless<int64_t>` → `bmm_gather_i64` on 4 lanes per step
  with equivalent prefetch lookahead.

**Measured (i7 laptop, MSVC 19.37 Release, AVX2, data/probes=1M random
indices, best-of-10):**

| kernel        | scalar + prefetch | SIMD gather | winner |
|---------------|-------------------|-------------|--------|
| i32, min      | 1.45 ns/op        | **1.28**    | SIMD, 1.13× |
| i32, median   | 1.83 ns/op        | **1.49**    | SIMD, 1.23× |
| i32, mean     | 1.79 ns/op        | **1.46**    | SIMD, 1.23× |
| i64, min      | **2.30 ns/op**    | 7.04        | scalar, 3.06× |
| i64, median   | **2.53 ns/op**    | 8.05        | scalar, 3.18× |
| i64, mean     | **2.69 ns/op**    | 8.17        | scalar, 3.04× |

The i64 regression reproduces consistently — `_mm256_i32gather_epi64`
on client Intel (Skylake-family) executes as serialised µops, one
8-byte load per cycle plus control overhead, beating the well-prefetched
scalar loop only on Zen3+ / Sapphire Rapids where gather latency is
competitive.

**Kept.**
- **Default for `gather_branchless<int32_t>`:** SIMD gather.
- **Default for `gather_branchless<int64_t>`:** generic scalar-prefetch
  template (unchanged).
- **Both SIMD paths stay in the source** as named functions
  `gather_simd_i32` and `gather_simd_i64`. `_i64` is currently slower
  and NOT the default — but per the multiple-impls-one-default rule
  and the keep-slower-paths-for-JIT-later policy, it stays compiled in
  so a future runtime dispatcher (per-CPU, per-shape) can reach for it
  without archaeology.

**Open questions.**
- AVX-512 native `_mm512_i32gather_epi64` on wider lanes may close the
  i64 gap — add a third variant when an AVX-512 test rig is available.
- Zen3+ / Sapphire Rapids measurements would flip the i64 default;
  re-measure when those are on the bench machine.

---

## filter_gt i64/f64 — compressstore vs scalar mask-bit emit  (A1)

**Context (base-layer perf lockdown, docs/BOLT_PERF_PUNCHLIST.md item A1).**
`filter_gt_avx2_i64` and `filter_gt_avx2_f64` were emitting compacted
indices via a 4- or 2-iteration scalar loop on the movemask result:

```cpp
for (int k = 0; k < L; ++k) {
    out[count] = static_cast<int32_t>(i) + k;
    count += ((mask >> k) & 1u);   // branchless but scalar per-lane
}
```

Meanwhile `bmm_compressstore_i64`/`_f64` and the portable
`bmm_compressstore_i32` had shipped in `bolt_port.h` (lines 788, 798,
968, 1072, 1182) but nothing in the i64/f64 filter path was wired into
them — only `filter_gt_avx2_i32` (line 479) used compressstore.

**Tried.** Process `2 * bmm_lanes_i64` rows per iteration (= 8 on AVX2,
4 on SSE/NEON) so the combined movemask is exactly `bmm_lanes_i32` bits
wide, and emit indices through a single `bmm_compressstore_i32`.

```cpp
for (; i + Lo <= n; i += Lo) {
    const bmm_vec_i64 v0 = bmm_loadu_i64(data + i);
    const bmm_vec_i64 v1 = bmm_loadu_i64(data + i + Li);
    const uint32_t    m  = bmm_movemask_i64(bmm_cmpgt_i64(v0, vscalar))
                         | (bmm_movemask_i64(bmm_cmpgt_i64(v1, vscalar)) << Li);
    alignas(32) int32_t idx_buf[16];
    for (int k = 0; k < Lo; ++k) idx_buf[k] = static_cast<int32_t>(i) + k;
    count += bmm_compressstore_i32(out + count, bmm_loadu_i32(idx_buf), m);
}
```

Invariant `bmm_lanes_i32 == 2 * bmm_lanes_i64` is enforced by
`static_assert`; holds for AVX2 / SSE42 / NEON, which are the only
ISAs this `_avx2_` family compiles under (the scalar fallback is
guarded out).

**Measured (i7 laptop, MSVC 19.37 Release, N=1M, ~50% selectivity,
best-of-10 per variant):**

| kernel               | before | after  | speedup |
|----------------------|--------|--------|---------|
| `filter_gt<int64_t>` | 0.55   | 0.32   | **1.72×** |
| `filter_gt<double>`  | 0.56   | 0.31   | **1.81×** |

Median is flatter (0.67→0.64 i64, 0.64→0.57 f64) because the laptop is
noisy, but every percentile is at parity or better, and best-case
scales cleanly toward the i32 kernel's 0.17 ns/op floor.

**Kept.** The compressstore path is the only i64/f64 filter_gt impl
now — no alternative behind a switch. The scalar mask-bit emit loop
is gone.

**Open questions.**
- A native AVX-512 widening to `bmm_compressstore_i64_x8` /
  `_mm512_mask_compressstoreu_epi32` could double throughput again on
  Skylake-X and newer. Deferred until we have an AVX-512 test rig.
- `idx_buf` is a stack write-then-load. Compiler may elide it; if
  profiles show a gap, replace with a hoisted base-pattern vector +
  `bmm_add_i32(base, set1(i))`. Would need `bmm_add_i32` added to
  `bolt_port.h`.

---

## MergeTriple layout — 40-byte natural vs 64-byte cache-line padded

**Context (Wave N, 1BRC perf push).** `parallel_groupby_agg_int64`
extended `MergeTriple` from 24 bytes (sum+count only) to carry min/max
for true 1BRC parity. Two layouts were tested for the radix scatter
buffer:

- **Padded**: `alignas(64)` struct with `pad[3]` so each triple = 64 B.
- **Compact**: 40 B naturally aligned, no padding.

**Measured (i9-9980HK, MSVC Release, 100M rows, 8 threads, best of 5):**

| Variant | ns/row 1T | ns/row 4T | ns/row 8T |
|---|---|---|---|
| 64-B padded | 17.5 | 7.0  | **5.9** |
| 40-B compact | 38.2 | 11.8 | 8.7 |

The compact form was **measurably slower**, opposite of the bandwidth
intuition that drove the experiment. Best guess: cache-line-aligned
triples interact better with the L1 hardware prefetcher in the merge
consumer (each shard merge walks triples sequentially; a 64-B stride
matches a cache-line stride exactly). At the 1BRC cardinality
(~3300 total triples) the extra 80KB of padding is irrelevant.

**Kept:** 64-B padded MergeTriple. Documented inline with a "tried
40-B; measured slower" comment so future readers don't repeat the
experiment.

**Open question:** Does this hold at high cardinality (100K+ groups)?
Worth re-measuring once we have a high-cardinality bench. If padded is
slower at scale we can switch on cardinality.

---

## Hash mix — Murmur3 finalizer vs Fibonacci vs wyhash 3-op

**Context (Wave N).** `swiss_mix` was the Murmur3 finalizer
(6 ops, ~5-7 cycle dep chain). 1BRC profile pointed at the mix as a
non-trivial hot-loop fraction. Three alternatives considered.

**Tested:** swap `swiss_mix` body to wyhash-style 3-op
(`x ^= x >> 32; x *= K; x ^= x >> 32; return x;`).
**Rejected without testing:** plain Fibonacci hash
(`x * 0x9E3779B97F4A7C15`) — see analysis in
`docs/research/hash-functions.md`.

**Result:** wyhash variant kept all 15 tests green and gave a small,
hard-to-isolate perf bump (within run-to-run noise on the
i9-9980HK at 1BRC scales). Importantly: SwissTable's tag/idx split
still works because the 3-op mix produces independent high/low halves.

**Kept:** wyhash mix. Documented in
`docs/research/hash-functions.md`; rejected alternatives recorded
there with rationale.

---

## Tight-sized SwissTable in groupby — `create_with(true)`

**Context (Wave L1 partial).** `GroupByTable::create` was using the
default 2× oversize from `SwissTable::create`. For 1BRC's known 413
cardinality this allocates a 1024-slot table when 512 would suffice.

**Tested:** swap the call to
`SwissTable::create_with(..., /*tight_sizing=*/true)`.

**Result:** **1.7×** speedup at 4T and 8T on bench_1brc 100M rows.
Single-line change. No correctness impact (existing tests pass).

**Kept.** Trade-off documented in `bolt_groupby.h`: caller must now
pass an upper-bound `capacity_hint`; exceeding it makes inserts
return false.

---

## Cardinality-adaptive Phase-2 merge

**Context (Wave L1 partial).** `parallel_groupby_*` always went through
the radix-partitioned merge in Phase 2. For 1BRC's low cardinality
(413 stations × 8 workers ≈ 3300 total triples), the radix scatter +
atomic-CAS shard-claim costs more than a single-thread walk over the
partials.

**Tested:** count total partial groups; below
`kGroupbySerialMergeThreshold` (default 4096, overridable via
`BOLT_GROUPBY_SERIAL_MERGE_THRESHOLD`), serial-walk merge into one
global table; above, existing radix path.

**Result:** clean win on 1BRC-shape inputs. High-cardinality joins
keep the radix path unchanged.

**Kept.** Documented in `bolt_parallel.h` and `bolt_config.h`.

---

## Branchless `bmin`/`bmax` in `GroupByTable::ingest`

**Context (Wave N).** Adding min/max updates per row with naive
`if (v < g.min) g.min = v;` regressed 1T from 19.6 → 28.8 ns/row at
100M rows — branch mispredict at the start of each new group's
sample stream.

**Tested:** swap to `bolt::branchless::bmin`/`bmax` (CMOV under
MSVC `/arch:AVX2` and gcc/clang `-O2`).

**Result:** recovers nearly all the regression. 1T 19.0 ns/row,
basically at parity with the sum-only baseline.

**Kept.** Min/max parity now ~free; full 4-aggregate output costs
the same as sum-only. The user's earlier intuition ("min/max stuff
seems good anyway") panned out empirically.

---

## Process notes

- Per-thread variance on this laptop is ±15-20% even with best-of-N.
  Decisions need ≥3 runs of N≥3 iters each before claiming a
  speedup/regression. One-shot numbers are anecdote, not evidence.
- The bench reports best-of-3 median + min. Use the **min** as the
  comparison point — median is sensitive to thermal/background load,
  min is the floor.
- Run-to-run drift between sessions is ~10%. Always benchmark the
  before-state in the same session as the after-state when possible.

## Two-pass hoisted-probe ingest — clean loss

**Context (Wave O).** Per
[`branchless-hashing.md`](branchless-hashing.md), the two-pass
hoisted-probe pattern was the highest-rated experiment for removing
the BR1+BR5 branches from `GroupByTable::ingest`. Built it as
`GroupByTable::ingest_two_pass`:

- **Pass A** — bulk probe: `existing[i] = ht.find(keys[i])`. Same
  internal branches as before, just relocated into a tighter loop.
- **Pass B** — branchless accumulate: `slot = (hit ? hit : kDummy)`
  selects between the real slot and a throwaway dummy slot via CMOV;
  every row writes unconditionally. Misses queued for Pass C.
- **Pass C** — insert misses (rare path after warmup since each new
  key is missed only once). Re-probes; corrects the dummy-slot
  noise.

**Measured (i9-9980HK, 50M rows, 8T, 1MB grain, best of 3):**

| Stations | Default ingest_unchecked | Two-pass | Δ |
|---|---|---|---|
| 413     | 8.64 ns/row | 11.35 | **−24%** |
| 10 000  | 16.74 | 19.67 | −15% |
| 100 000 | 93.47 | 95.26 | parity |

**Why it lost:**

1. At low cardinality (413), the hit/miss branch is ALREADY perfectly
   predicted after warmup (every key is a hit). The "branch removed"
   was already free. The added scratch-allocate + two-pass traversal
   + re-probe in Pass C is pure overhead.
2. At high cardinality (100K), the bottleneck is memory bandwidth on
   the probe itself, not branch mispredicts. Two-pass doesn't reduce
   probes.
3. Pass B's "always-write to dummy slot for misses" doubles writes for
   missed rows — and misses are precisely the rows we already had to
   handle in the slow path anyway.

**Kept as opt-in.** Per the "multiple impls, one default" rule:
`GroupByTable::ingest_two_pass` stays in the library. Default is
`ingest_unchecked` (single-pass, loop-peeled). Bench has
`--two-pass` flag for users who want to test it on their own
workload. Override is `parallel_groupby_two_pass_override()`.

**Open question:** does two-pass win when keys are *truly random*
(no repeat-probe locality, hit-rate near 0% per warmup)? The 100K
test still has each key appearing ~1000× on average. A 1B-row /
1B-distinct-key bench would be the right test — but that's a memory
size we can't run on this host.

**Real lesson:** branchless via late-materialization trades branches
for memory bandwidth. Good trade when the original was branch-bound
(misprediction-heavy); bad trade when it was already memory-bound.
Our groupby is the latter at every cardinality we measure.

## Branchless `ingest_unchecked` + loop-peeled prefetch

**Context (post-Wave-M1).** Looking at the morsel hot loop revealed
~3 branches per row that turned out removable:
- `if (i + pf < end)` — bound check on the prefetch lookahead.
- `if (!tbl.ingest(...)) { error_flag.store(); return; }` — capacity-
  overflow check, dead-on-arrival when caller pre-sized the table.
- One internal branch in `ingest()` for the post-insert capacity
  check (likewise dead).

**Tested:**
1. Added `GroupByTable::ingest_unchecked` — same body minus the
   capacity check; precondition `assert(num_groups < capacity)`.
2. Loop-peeled the morsel + serial loops: main body has no per-row
   bound check, tail loop (last `pf` rows) has no prefetch.

**Measured (i9-9980HK, 50M rows, 8 threads, throughput / 1 MB grain,
best of 3):**

| Stations | Before (ns/row) | Branchless (ns/row) | Δ |
|---|---|---|---|
| 413     | 9.24 | 8.51   | +9% |
| 10 000  | 16.2 | 15.7   | small win |
| 100 000 | 177  | **96** | **+85%** ← biggest single library-level win |

The 100K-station case is the clearest signal: with the table well out
of L1 (256K-slot SwissTable, hot set ~1.7 MB), every per-row branch
shows up because each row already does a memory-bound probe. Removing
the capacity-check branch and the bound-check branch pulls the
inner loop down from ~50 cycles per row to ~30.

**Kept.** `ingest_unchecked` is now the morsel-loop default. The
checked `ingest` stays for callers without pre-sized capacity hints.

---

## Morsel-size × cardinality × prefetch sweep

**Context.** With `--prefetch N` and `--grain-kb N` runtime knobs on
`bench_1brc`, swept the three-axis space at 50M rows, 8T:

| Stations | Grain | PF=0 | PF=16 | Best |
|---|---|---|---|---|
| 413   | 64 KB   | 14.96 | 14.27 | PF=16 |
| 413   | 256 KB  | 12.48 | 11.12 | PF=16 |
| 413   | 1024 KB | 10.61 | 10.37 | parity |
| 10K   | 64 KB   | 88.69 | 91.60 | PF=0  |
| 10K   | 256 KB  | 34.08 | 31.88 | PF=16 |
| 10K   | 1024 KB | 22.74 | 18.07 | **PF=16 (+25%)** |
| 100K  | 64 KB   | 140.5 | 144.3 | parity (compute-bound) |
| 100K  | 256 KB  | 143.5 | 140.7 | parity |
| 100K  | 1024 KB | 122.5 | 128.3 | PF=0 |

**Findings:**

- **Bigger morsels almost always win.** 1024 KB grain beats 64 KB
  grain by 2-3× across the board. The smaller the grain, the more
  morsels, the more partials, the more merge work. There's no
  countervailing locality benefit because each per-morsel partial
  table doesn't fit L1 anyway above ~1K cardinality.
- **Prefetch's break-even is ~10K distinct keys at 1024 KB grain.**
  Below that the hot set (cardinality × 17 bytes) fits L1 from
  repeat-probe locality. Above that prefetch starts to pay.
- **At 100K stations performance plateaus at ~95 ns/row regardless
  of grain or prefetch.** The probe is fully memory-bandwidth bound;
  no amount of prefetch lookahead hides DRAM latency at that scale.
  The fix is structural — fewer probes per row (perfect hashing,
  smaller sketch tables, Bloom-filter pre-screen).

**Kept.** Default scheduler profile unchanged (1024 KB grain,
prefetch off). Bench advertises both knobs for callers that know
their workload's cardinality. Per-table `prefetch_ahead` field is
runtime-tunable (`uint16_t`) with default 0.

---

## Wave M1 — lookahead prefetch in groupby ingest

**Context.** Wave M1 added `SwissTable::prefetch(key)` (warms the
control + slot cache lines for a future find/insert) and tried wiring
it into the groupby ingest hot loop with a 16-row lookahead, on the
intuition that DRAM-miss latency would otherwise stall the probe.

**Tested:**
1. Always-on prefetch in `parallel_groupby_morsel` and the serial
   `groupby_agg_int64` Phase 1 loop.
2. Cardinality-adaptive prefetch (skip when `tbl.ht.capacity ≤ 2048`).
3. Default-off, `if constexpr` opt-in via `BOLT_GROUPBY_PREFETCH_AHEAD`.

**Measured (i9-9980HK, 100M rows, 4 aggregates, throughput profile,
5 runs):**

| Variant | ns/row 1T | ns/row 4T | ns/row 8T (best) |
|---|---|---|---|
| Wave N baseline (no prefetch) | 17.5 | 7.0 | **5.9** |
| Always-on prefetch | 24.2 | 9.7 | 8.3 |
| Capacity-adaptive | 23.9 | 10.6 | 8.5 |

The capacity heuristic looked correct on paper — per-morsel partial
tables are sized to `grain` (128K+ slots, well above the 2048
threshold), so prefetch fired. But the hot probed cache lines
correspond to the actual distinct-key working set (~413 stations for
1BRC), which fits in L1 *because of repeat-probing locality*, not
because of allocated capacity. Capacity is a poor proxy for that
hot working set.

**Kept:** default-off opt-in via `BOLT_GROUPBY_PREFETCH_AHEAD` macro.
The `SwissTable::prefetch` primitive remains so future workloads with
high distinct-key spread (random IDs, large unique-key universe) can
flip the switch. `SwissTable::find_simd` (used by hash-join probe
where build-side cardinality varies widely) keeps its capacity-adaptive
prefetch — there the heuristic is correct because hash-join probes
each key once, no repeat-probe locality.

**Open question:** does the current default win for high-cardinality
groupby (100K+ distinct keys, no repeat locality)? Should be tested
with a dedicated bench before claiming the default works there too.
Likely the prefetch path will win at high cardinality and we'll want
runtime adaptive selection — but the right metric is hot-set size,
not capacity.

## Project rule — multiple implementations, one default

The library may carry **multiple implementations** of the same
primitive (compact-vs-padded `MergeTriple`, scalar vs SIMD hash
probe, Murmur3 vs wyhash mix, …) — but exactly **one is wired in by
default**: the one that wins on the broadest workload class.

- Alternatives ship behind compile-time switches in `bolt_config.h`
  (`#ifndef BOLT_*`) or as compile-time tag-dispatch variants
  (`SwissTable_PreHashed`-style types). **Never runtime branches** —
  keep dispatch zero-cost.
- Each alternative gets an inline trade-off comment at its definition
  AND an entry in this log. Future tuning passes start from the log
  so we don't re-litigate decisions.
- The default exists for the load-bearing case (1BRC-shape low
  cardinality, the bench workloads we have today). Alternatives exist
  for cases we *can* describe but *don't yet measure* (very high
  cardinality, embedded targets with strict footprint budgets, NUMA
  hosts with cross-socket scatter pressure, etc.).

## Welford online (count, mean, m2) — default variance choice for rolling fintech stats  (P2.2)

**Context:** Phase 2 of the fintech kernel port (docs/BOLT_FINTECH_PORT.md)
needed a single online-variance primitive to back RollingStd / RollingZScore
/ Sharpe / Sortino / RollingCorr / RollingSkew / RollingKurt. Three
candidates:

1. **Naive** `E[x²] − E[x]²` — one running `sum` plus one running
   `sum_sq`, then subtract. Fast, one MUL and one SUB at read-out, no
   per-sample division. **Rejected:** catastrophic cancellation when
   variance ≪ mean² (the exact regime for intraday price returns).
2. **Welford's online** `(count, mean, m2)` with
   `delta = x − mean; mean += delta/count; m2 += delta*(x − mean)`.
   One DIV per update, numerically stable down to machine epsilon.
3. **Neumaier/Kahan-corrected sum of squares.** Stabilises the naive
   form but still requires a large subtraction at read-out.

**Kept:** Welford. The per-update DIV is free next to the memory traffic
of a fintech morsel (one cache line per sample dominates), and it's the
textbook-standard primitive (Knuth TAOCP vol. 2 §4.2.2) — reviewers
recognise it instantly.

**Default surface:** both `variance_pop()` (÷ N) and `variance_sample()`
(÷ N−1) are exposed. Tier-2 kernel bodies pick the Bessel-corrected
`variance_sample()` as the default for vol / Sharpe / Sortino because
tick data is modelled as a sample drawn from an unknown underlying
process, which is the statistical convention for those measures. The
pop form stays available for Tier-2 kernels that work on a closed set
(e.g. a fixed-window population over a batch with no implied wider
distribution).

**Open question:** rolling-window Welford (drop oldest + add newest)
needs the Welford-decrement formula. We haven't picked between the
symmetric "West/Chan two-pass" approach and a separate streaming
variant — that decision lands with RollingStd (P5.W.2).

## RSI Wilder vs classical EMA alpha — which convention wins inside rsi.h  (P5.E.5)

**Context:** Phase 5.E.5 ports `CreateRSIProcessor` from chukonu. RSI
requires an exponentially-weighted average of the per-sample gains and
losses, but the EMA-alpha convention is *not* the same as classical EMA.

Two candidates:

1. **Classical EMA** `alpha = 2 / (period + 1)`. This is what
   `EmaState::init(period)` computes. For period=14 → alpha ≈ 0.1333.
   Matches what we already use for EMA (P3.1) and MACD (P5.E.4).
2. **Wilder's smoothing** `alpha = 1 / period`. For period=14 → alpha ≈
   0.0714. Defined by Welles Wilder in *New Concepts in Technical
   Trading Systems* (1978) specifically for RSI, ADX, ATR. Roughly half
   the responsiveness of classical EMA at the same period number.

**Kept:** Wilder `alpha = 1/period`, exclusively inside `rsi.h`. Two
reasons:

- **Chukonu source-of-truth.** `CreateRSIProcessor` at kernels.h:536
  computes `alpha = 1.0 / period` directly. Porting RSI without matching
  this would produce numerically different output for the same period
  argument — the port's round-trip test against the Arrow version would
  diverge.
- **Convention is domain-stable.** Every charting package (TradingView,
  MetaTrader, TA-Lib, pandas-ta) uses Wilder for RSI. A user asking for
  "RSI(14)" expects the Wilder result.

**Implementation shape:** `EmaState::init(period)` hard-codes the
`2/(p+1)` formula. Rather than add a second `init_wilder(period)` method
to `state.h` (which is read-only substrate shared by EMA + MACD), we
bypass `init()` inside `make_rsi_state()` and set `ema.alpha = 1.0 /
period` manually. This keeps `state.h` untouched and localises the
convention difference to `rsi.h`, where the comment makes it auditable.
EWMA (P5.E.2) uses the same "bypass init, set alpha directly" pattern
for a different reason: the caller supplies alpha verbatim (e.g. the
RiskMetrics 0.06 or user-specified decay), not a period.

**Rejected alternative:** adding `EmaState::init_wilder(period)` as a
second constructor. Would be one extra call and one extra source of
truth for "which alpha convention applies here?" — better to keep
`state.h` minimal and let each kernel header declare its convention in
its own `make_*_state()` constructor.

**Open question:** ATR (P5.R.4) currently uses a rolling arithmetic
mean. The canonical Wilder ATR uses the same `alpha = 1/period`
smoothing as RSI. When ATR gets its Wilder-form alternative kernel
(`atr_wilder.h`?), it should copy the bypass-init pattern from
`rsi.h::make_rsi_state`. Document that port choice in a new design-log
entry at that time.

## Rolling-window Welford — rescan vs decrement  (P5.W landing, closes P2.2 open question)

**Context:** Phase 5.W landed ten Welford-based fintech kernels
(WelfordMeanVar, RollingStd, RollingZScore, SharpeRatio, SortinoRatio,
RollingCorrelation, RollingSkew, RollingKurt, Autocorr, RiskMetricsVol).
Five of these are true sliding-window moment kernels (RollingStd,
RollingZScore, RollingCorrelation, RollingSkew, RollingKurt); the rest
are either streaming (no window) or scalar-broadcast (whole-batch).

The P2.2 entry flagged an open question for sliding-window Welford:
Welford's online formula is only stable for *append-only* streams. A
rolling window needs both an add and a drop — the "symmetric Welford"
variant (West 1979 / Chan et al. 1983) exists but reintroduces a
subtraction that can cancel when the dropped value is close to the
current mean.

**Candidates for the rolling variants:**

1. **Welford-decrement.** Maintain `(count, mean, m2)` online; apply
   the inverse Welford update on eviction. `O(1)` per row, but
   accumulates cancellation error over long streams — especially for
   higher moments (skew m3, kurt m4).
2. **Rescan from ring.** Recompute mean and central moments from the
   whole `RollingRing` each row. `O(w)` per row; every row produces a
   fresh accumulator with no history-dependent cancellation.

**Kept: rescan from ring.** Reasoning:

- **Chukonu-parity.** `CreateRollingStdProcessor`,
  `CreateRollingZScoreProcessor`, `CreateRollingCorrelationProcessor`,
  `CreateRollingSkewProcessor`, `CreateRollingKurtProcessor` in
  `chukonu/fintech/kernels.h` all rescan. Matching the numerical
  behaviour keeps the regression-parity plan in `BOLT_FINTECH_PORT.md`
  intact — Arrow-output vs Bolt-output must match bit-for-bit (or
  within 1e-12) for the port to be considered landed.
- **Windows are small.** Typical fintech rolling windows are ≤256
  (intraday SMA, vol lookbacks). A 256-double rescan is ~2 KiB of
  sequential L1 traffic per row — the ring is already cache-hot from
  the push, so the `O(w)` cost is dwarfed by the per-row output store
  and surrounding morsel plumbing.
- **Higher moments are where decrement hurts most.** Pébay/Bennett
  (Sandia National Laboratories, 2008) analyses of sliding higher-moment
  statistics show decrement errors on m3/m4 growing as O(sqrt(stream
  length / window)) in typical financial signals — enough to silently
  bias CornishFisherVaR (P5.S.9) or MAD outlier flags (P5.S.14).
- **Streaming kernels are unaffected.** WelfordMeanVar, SharpeRatio,
  SortinoRatio are append-only or whole-batch — they use the streaming
  `WelfordAccumulator` directly. Rescan only applies to the five
  rolling-moment kernels.

**Shared helper:** `welford_rescan_pop<kCap>(ring, &mean, &var)` in
`rolling_std.h` is reused by `RollingStd` and `RollingZScore`. The
correlation / skew / kurt / autocorr kernels inline their own rescan
because they need additional accumulators in the same pass (cov, m3,
m4, lagged means).

**Also matched from chukonu (numerical-convention notes, per-kernel):**

- `WelfordMeanVar` emits **population** variance (`m2/count`), not
  sample — streaming case, chukonu-parity.
- `SharpeRatio` uses **sample** variance (`m2/(n-1)`), annualised by
  `sqrt(annualization_periods)`, default `periods = 252`. Scalar
  broadcast across the batch. Chukonu divides by `n-1` here, unlike
  its rolling-moment kernels.
- `SortinoRatio` divides the downside sum-of-squares by `n`
  (total sample count), **not** by the number of downside
  observations — a chukonu idiosyncrasy we match for parity.
- `RollingKurt` emits **excess** kurtosis (`m4/m2² - 3`), the
  finance-industry default.
- `RollingSkew` emits **raw / biased** skewness (`m3 / stddev³`,
  population moments), not Fisher-Pearson g1.
- `RiskMetricsVol` emits **stddev** per row (not variance), seeded at
  `|r0|` on the first sample.
- Warmup convention across the rolling kernels: chukonu emits **0.0**
  (not NaN) for `w < min_required` — `w < 3` for skew, `w < 4` for
  kurt, `w < 2` for correlation, `w < w+lag` for autocorr. Matched
  verbatim.

**Open question:** for windows >1024 the rescan cost starts to eat
into the morsel budget. If a future workload demands that scale,
revisit the symmetric-Welford-decrement path with Kahan / Neumaier
compensation on the subtraction, gated behind a compile-time
`BOLT_WELFORD_DECREMENT` flag in `bolt_config.h` (default off).
Chukonu-parity tests would need a looser tolerance in that mode.

## Rolling quantile — SortedRing vs streaming approximations

**Context / date.** Phase 5.S fintech port, Apr 2026. HistoricalVaR,
HistoricalCVaR, MedRV, and OutlierFlagMAD need a rolling quantile /
rolling median primitive. Chukonu (the Arrow-based reference)
allocates a fresh `std::vector` per row, copies the window in,
sorts it, and throws it away. Every row: one malloc, one free, a
cache-cold sort of `w` doubles. For a 10-row-per-tick stream at 10k
ticks/s that's 100k mallocs/s of pure overhead.

**What we evaluated.**

1. **SortedRing<T, kCap>** (chosen, default).
   Fixed-capacity insertion-sort under eviction: two parallel arrays
   (`raw[]` in insertion order, `sorted[]` ascending). Push = one
   `std::lower_bound` + one `memmove` to close the evicted slot +
   one `lower_bound` + one `memmove` to open the insertion slot.
   No heap, no branch on data in the hot loop body.
   - Cost: **O(w) per push**, O(1) quantile / O(1) median / O(k)
     tail-sum for CVaR.
   - Memory: `2 * kCap * sizeof(T)` bytes, pinned in graph arena once.
2. **Tukey's two-heap median** (rejected).
   `std::priority_queue`-based two-heap (max-heap for the lower half,
   min-heap for the upper) gets O(log w) push but: (a) heap
   allocation under the std containers, (b) lazy-delete eviction
   leaves tombstones, (c) only solves the *median* cleanly —
   arbitrary quantile and tail-sum both need a sorted linearisation
   anyway. We'd need a second data structure for CVaR regardless.
3. **P² / GK / t-digest streaming approximations** (rejected, deferred).
   O(1) push, O(1) quantile. Approximate: can be 1–5% off on heavy
   tails, which is exactly where VaR / CVaR live. Accuracy bound is
   distribution-dependent; reproducing chukonu's numerics for the
   round-trip tests would need a loose tolerance. Revisit only if
   a workload forces us off SortedRing.

**Why SortedRing wins as the default.**
- **Exact.** Bit-for-bit agreement with chukonu's
  sort-then-index-the-vector reference, modulo the floor-on-fp
  boundary. Enables tight `EXPECT_NEAR(..., 1e-9)` test tolerances.
- **Zero allocation.** The kCap arrays are graph-lifetime-pinned.
  No per-row heap churn; deterministic tail latency.
- **Cache-friendly.** Both arrays are contiguous; the memmove tails
  stream through a single L1 line for typical w ≤ 64.
- **Acceptable asymptotic.** Tier-2 fintech windows are small:
  ATR 14, Bollinger 20, RSI 14, HistoricalVaR typically 250, MedRV
  is batch-wide. O(w) per push is fine at w ≤ 512; the SortedRing
  hard cap at `kCap` is a template parameter the caller picks.

**Alternatives kept compile-accessible.** The header comments in
`sorted_ring.h` spell out where P²/GK/t-digest would slot in if a
future workload changes the trade-off. Per Bolt's "multiple impls,
one default" rule, we don't ship the alternatives today — they live
in the log as an open question so future tuning starts from
measured-not-remembered.

**Open question.** For `w > 1024` the O(w) memmove dominates and a
piecewise-linear skiplist (or GK summary with 1% ε) would overtake.
If a user ships VaR with a 10-year rolling window of minute bars
(~5M samples), revisit and add a second impl behind
`BOLT_SORTED_RING_APPROX` in `bolt_config.h`.

## How to add an entry

1. Run the change. Capture before/after numbers (≥3 runs each).
2. Append a new `## Topic — alt A vs alt B` section to this file.
3. State context, what was tested, the numbers, what you kept, why.
4. If the rejected alternative might come back later (different
   workload, new hardware), say so explicitly in an "open question"
   line — that's how `MergeTriple` got its high-cardinality followup.

---

## 2026-05-01 / soft-delete substrate — atomic-bitmap primitives lifted from MarbleDB BM25

Context: MarbleDB BM25's Wave-9.2 rebuild (llm-station plan
`this-was-a-freach-hashed-crab.md`) needed a lock-free tombstone bitmap
for `bm25_remove_doc`. The first cut inlined `std::atomic<uint64_t>*` +
`fetch_or` + bit-tests directly inside `src/ext/bm25.cpp`. That pattern
will repeat for every future "soft delete" surface (PDX cluster
live-mask, HNSW evicted-entry mask, episodic retire bitmap), so the
primitive moved to Bolt.

What landed: `include/bolt/kernels/bolt_atomic_bitmap.h` —
- `atomic_bitmap_words_for(n_bits)` constexpr sizing.
- `atomic_bitmap_set(words, bit) → bool` — `fetch_or(acq_rel)`; returns
  true iff this call performed the 0→1 transition.
- `atomic_bitmap_clear(words, bit) → bool` — `fetch_and(~m, acq_rel)`.
- `atomic_bitmap_test(words, bit) → bool` — relaxed load + bit test.
- `atomic_bitmap_popcount(words, n_words)` — relaxed loads + Bolt's
  existing `bolt_popcount64`.
- `atomic_bitmap_clear_all(words, n_words)` — release stores.

Also extended in this pass:
- `bolt_binsearch.h` gained `contains_{i64,u64,f64,f32}` — `lower_bound`
  + branchless tail compare. Removes the per-consumer
  `sorted_contains` inlining MarbleDB and future filter-pushdown
  consumers were going to redo.
- New header `include/bolt/kernels/bolt_smallsort.h` —
  `sort_small_{u64,i64,u32}_asc` insertion sort with `kSmallSortCap=1024`.
  Used by MarbleDB's BM25 boolean-query drain (`bm25_query_and` /
  `bm25_query_or`).

Tests landed: `tests/test_bolt_atomic_bitmap.cpp` (6 assertions
including 8-thread concurrent-set race), `tests/test_bolt_smallsort.cpp`
(7 assertions, randomised cross-checks against `std::sort`). Both green
under MSVC RelWithDebInfo.

What we kept: scalar insertion sort for `sort_small_*`. Open question:
SIMD bitonic merge for `n ≥ 32` would beat insertion sort for the
high end of the cap range; left as a future dispatch path because
MarbleDB BM25's actual `n` is ≤ ~50 in profile. When a consumer profile
shows the n ∈ [32, 1024] range is hot, write a topic file under
`docs/research/smallsort-simd.md` and add an opt-in
`#ifdef BOLT_SORT_SMALL_SIMD` switch.

Open question: `atomic_bitmap_clear_all` uses release stores
non-atomically with respect to concurrent writers. A correct
"reset-while-others-write" path needs a per-word CAS loop — defer
until a consumer actually needs concurrent-reset semantics (none today).

---

## bolt::parse::json fionn-parity wave — SAX + NDJSON + SIMD plain-run

Decision: ONE scanner, two sinks via compile-time template dispatch
(`TapeSink` for the structural-index tape, `SaxSink` for event
callbacks). The alternative — a separate hand-written SAX scanner —
would duplicate string/number/keyword scanning and drift; the template
keeps zero runtime dispatch on the hot path (Bolt's one-default rule:
the tape stays the default consumer surface, SAX is the streaming
opt-in for constant-memory pipelines).

Landed:
- `sax_parse` (events, no tape, no arena, O(depth) memory; abort via
  callback-false, distinguished from malformed input so NDJSON
  ignore_errors never swallows a consumer abort).
- `ndjson_for_each` / `ndjson_for_each_sax` (SWAR newline framing,
  CRLF trim, scratch-arena reset per record, ignore_errors + stats).
- `string_plain_run` (AVX2 behind BOLT_SIMD_AVX2 + SWAR fallback) and
  `digit_run` (SWAR) accelerating string bodies and number bodies.
- `parser_init` replacing the whole-Parser memset: the 17 KB
  path_stack zeroing per parse was a 100x write amplification on
  ~170-byte NDJSON records (NDJSON throughput +48% from this alone).

Measured (bench_json, MSVC Release AVX2): full tokenization
0.62–0.84 GB/s; NDJSON tape-per-record ~0.56 GB/s. fionn's 4–8 GiB/s
headline is selective skip-scanning, not full tokenization — the next
lever is a simdjson-style stage-1 structural bitmap with folded UTF-8
validation, slotting in behind the same sink seam. Deferred until a
consumer profile demands it; tracked in
docs/research/json-skip-architecture.md.

Tests: test_bolt_parse_json 20/20 (SAX event-sequence vs tape, abort,
null-callback skip, plain-run boundary stress at 40 offsets, NDJSON
clean/dirty/strict/abort). Bench: benchmarks/bench_json.cpp.
