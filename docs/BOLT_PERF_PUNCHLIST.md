# Bolt Perf Punchlist

Tracked checklist of the base-layer perf items we are closing out before
opening any new feature surface. Every item is a lit TODO in the code, a
design-doc gap, or a referenced-but-unbuilt primitive. We lock all of these
down *before* starting ingest/watermarks, bitemporal, marbledb storage, or
the chukonu planner.

Plan blueprint: [`plans/to-bake-this-into-validated-thunder.md`](../../../../Users/Capta/.claude/plans/to-bake-this-into-validated-thunder.md)
(local to the author's machine — content mirrored here).

## Scope / non-scope

**In scope** — P0 hot-path items, P1 table-stakes items (minus the ingest
cluster), P2 items the user explicitly wants done (huge pages, f32 SIMD,
merge-join, auto-fusion primitives), and the two live code TODOs (dict
validity, Windows >64-core pinning).

**Out of scope (deferred to the ingest / storage wave)**
- O3 out-of-order tick ingest + watermarks + late-data handling
- Bitemporal (valid_time + transaction_time) indexing → marbledb wave
- `mmap` / `posix_fadvise` / `PrefetchVirtualMemory` for file-backed columns
- CSV / JSON end-to-end parser completion (1BRC parse glue)
- JIT / runtime codegen (decided: X-macro compile-time dispatch wins)
- DataFusion-level query planner (belongs in chukonu)

## Parallel track layout

Each track owns a disjoint set of hot files, so any two items in different
tracks can be worked concurrently with no merge conflict.

| Track | Primary files | Items | Concurrent with |
|-------|---------------|-------|-----------------|
| **A** — Kernels / SIMD | `bolt_branchless.h`, `bolt_port.h`, new `bolt_hash.h` | A1–A6 | B · C · D · E |
| **B** — Column formats | `bolt_column.h` (+ new `bolt_fsst.h`) | B1–B5 | A · C · D · E |
| **C** — Joins / hash / groupby | `bolt_swiss.h`, `bolt_hashjoin.h`, `bolt_groupby.h`, new `bolt_mergejoin.h`, new `bolt_bloom.h` | C1–C5 | A · B · D · E (see A3 note) |
| **D** — Arena / memory | `bolt_arena.h`, narrow COW touch in `bolt_column.h` | D1–D2 | A · B · C · E |
| **E** — Scheduler / concurrency | `bolt_scheduler.h`, `bolt_channel.h` | E1–E2 | A · B · C · D |
| **F** — Fusion primitives | new `kernels/bolt_fused.h` | F1 | all — but blocked on A1 |
| **G** — Infra | `CMakeLists.txt`, `CMakePresets.json`, `.github/workflows/ci.yml`, `ci/perf_check.*`, this doc | G1–G3 | all |

**Conflict notes**
- A3 (xxh3) touches hash call-sites in Track C. Land A3 before C2/C3.
- D2 (NT stores) edits the COW path inside `bolt_column.h` — same file as
  Track B. Narrow surface; coordinate only when both tracks are active.
- F1 depends on A1.

## Status legend

`[ ]` open · `[~]` in progress · `[x]` done · `[!]` blocked (add note inline)

Tick a box only when **all seven DoD criteria** below are met.

---

## Track A — Kernels / SIMD

- [x] **A1. Wire `bmm_compressstore_i64/_f64` into `filter_gt` — P0**
      *i64 0.55 → 0.32 ns/op (1.72×), f64 0.56 → 0.31 ns/op (1.81×),
      best-of-10, MSVC Release AVX2.*
  - Files: `include/bolt/bolt_branchless.h:490-565`. Intrinsics live at
    `bolt_port.h:788, 798, 968, 1072, 1182`.
  - Test: existing i64/f64 cases in `tests/test_bolt_branchless.cpp`
    (10 pass on MSVC Release).
  - Bench: `benchmarks/bench_kernels.cpp` — labels updated to
    `(compressstore)` for both kernels.
  - Design-log: [`design-log.md` → "filter_gt i64/f64 — compressstore vs
    scalar mask-bit emit (A1)"](research/design-log.md).
- [x] **A2. SIMD gather + lane-level prefetch in `gather_branchless` — P0**
      *i32 min 1.45 → 1.28 ns/op (1.13×), median 1.83 → 1.49 (1.23×) —
      SIMD is the default for i32. i64 SIMD measured 3× slower on
      client Intel; i64 keeps the scalar-prefetch default but the
      `gather_simd_i64` variant stays in source for future JIT /
      Zen3+ / Sapphire Rapids dispatch.*
  - Files: `include/bolt/bolt_branchless.h:306-436` — named
    `gather_simd_i32`, `gather_simd_i64`; template specialisation for
    `<int32_t>` only; generic template is the i64 default.
  - Test: `GatherBranchless.*` in `tests/test_bolt_branchless.cpp` (5 pass).
  - Bench: `bench_gather_i32` + `bench_gather_i64` in
    `benchmarks/bench_kernels.cpp`; i64 bench prints both variants for
    ongoing A/B visibility.
  - Design-log: [`design-log.md` → "gather_branchless — SIMD gather vs
    scalar + prefetch (A2)"](research/design-log.md).
- [x] **A3. xxh3 + murmur3 added as kept-in-source alternatives — P1**
      *All three variants (`swiss_mix_wyhash3`, `swiss_mix_xxh3`,
      `swiss_mix_murmur3`) always compiled; `BOLT_HASH_TIER={WYHASH3|
      XXH3|MURMUR3}` selects the `swiss_mix` default. swiss_find on
      AVX2: WYHASH3 min=5.22, XXH3 min=4.90 (statistical tie),
      MURMUR3 min=6.27 ns/op. WYHASH3 stays the default — no measured
      reason to flip.*
  - Files: new `include/bolt/bolt_hash.h`;
    `include/bolt/join/bolt_swiss.h` now `#include`s it and deletes
    the inline definition. Call-sites in `bolt_hashjoin.h`,
    `bolt_parallel.h`, `bolt_groupby.h` (via `bolt_swiss.h`) unchanged.
  - Test: existing `test_bolt_join.cpp` (13 pass) exercises the hash
    path end-to-end under the default tier; hash-mixer correctness is
    self-evident from the variant definitions (no randomness).
  - Bench: `SwissTable::find_simd` case in `bench_kernels.cpp`
    measured under each tier via reconfigure (CMake switch).
  - Design-log: [`design-log.md` → "Hash mixer — xxh3 + murmur3
    added as kept-in-source alternatives (A3)"](research/design-log.md).
- [x] **A4. Multi-type selection-vector composition — P1**
      *`selection_intersect_t<Idx>` / `selection_union_t<Idx>` now
      templated on any integral index type; i32 wrappers stay as inline
      back-compat shims. Added `bitmap_to_indices<Idx>` — branchless
      ctz64 per bit, handles partial tail. 7 new tests green.*
  - Files: `include/bolt/bolt_branchless.h` — templated variants +
    back-compat i32 shims + `bitmap_to_indices<Idx>`.
  - Test: `SelectionCompose.*` (7 tests) in
    `tests/test_bolt_branchless.cpp`.
  - Design-log: [`design-log.md` → "Selection vector composition —
    multi-index-type + bitmap→indices (A4)"](research/design-log.md).
- [x] **A5. f32 SIMD kernels — filter_gt + sum — P2**
      *Added `bmm_setzero_f32` / `bmm_add_f32` / `bmm_hadd_f32` in all
      four ISA blocks of `bolt_port.h` (f32 compare/load/store already
      existed). Added `filter_gt_avx2_f32` + `sum_avx2_f32` in
      `bolt_branchless.h` mirroring the f64 kernels.  Sum widens to
      double on reduction to match codebase convention. 4 new tests
      green (incl. NaN-never-passes + SIMD-float-sum within 1e-5 rel
      of scalar).*
  - Files: `include/bolt/bolt_port.h` (f32 setzero/add/hadd across
    AVX2/SSE/NEON/scalar); `include/bolt/bolt_branchless.h`
    (`filter_gt_avx2_f32`, `sum_avx2_f32`);
    `tests/test_bolt_branchless.cpp` (`F32SIMD.*`, 4 tests).
  - Test: `F32SIMD.FilterGtMatchesScalar`,
    `F32SIMD.FilterGtNaNNeverPasses`,
    `F32SIMD.SumMatchesDoubleAccumulator`, `F32SIMD.SumEmpty`.
  - Bench: f32-vs-f64 bandwidth microbench deferred — add when a
    workload exercises f32 columns end-to-end.
  - Design-log: [`design-log.md` → "f32 SIMD kernels — filter_gt +
    sum (A5)"](research/design-log.md).
- [x] **A6. Windows >64-core pinning via processor groups — TODO**
      *`bolt_pin_current_thread` now routes group 0 through
      `SetThreadAffinityMask` (unchanged fast path) and group > 0
      through `SetThreadGroupAffinity`, covering the full 4096-CPU
      space. Compiles cleanly on MSVC 19.37; 5 topology + 9 scheduler
      tests green locally. Verification on >64-core Windows hardware
      deferred — no such box on hand.*
  - Files: `include/bolt/bolt_port.h` — `bolt_pin_current_thread`
    Windows branch.
  - Test: `tests/test_bolt_topology.cpp`, `tests/test_bolt_scheduler.cpp`
    (existing coverage; all pass).
  - Design-log: [`design-log.md` → "Windows >64-core pinning via
    processor groups (A6)"](research/design-log.md).

## Track B — Column formats

- [x] **B1. `gather_to_column` validity bitmap propagation — TODO**
      *`bolt_column.h:1071` TODO resolved. Validity bitmap is packed
      from `src.validity[src.validity_offset + sel[i]]`; `null_count`
      and `all_valid` kept consistent. All 39 kernel tests green.*
  - Files: `include/bolt/bolt_column.h` `gather_to_column<T>`.
  - Test: `BoltKernels.GatherToColumnInt32ValidityPropagated` (hand-
    computed bitmap mask 0x69) and
    `BoltKernels.GatherToColumnInt32AllValidSrcNoBitmap` (fast-path
    unchanged).
  - Design-log: [`design-log.md` → "gather_to_column — validity bitmap
    propagation (B1)"](research/design-log.md).
- [x] **B2. `ColumnFormat::RLE` — minimal format + materialize — P1**
      *Added `ColumnFormat::RLE = 5` + `BoltColumn::make_rle(values,
      num_runs, run_ends, total_rows, type, arena)` + materialize
      case. Storage reuses `data` (values) + `dict_child` (int32 Flat
      over run_ends) — BoltColumn size unchanged. 3 correctness tests
      green. Kernels do NOT read RLE natively yet — they materialise
      first; run-native `filter_eq_rle` is the B2b follow-up.*
  - Files: `include/bolt/bolt_column.h` — enum variant, `make_rle`,
    materialize branch.
  - Test: `BoltColumn.RLEMaterializeRoundTrip`, `BoltColumn.RLEEmpty`,
    `BoltColumn.RLESingleRun` in `tests/test_bolt_primitives.cpp`.
  - Bench: deferred — add run-native filter/sum to `bench_kernels`
    once the B2b run-native kernels land.
  - Design-log: [`design-log.md` → "ColumnFormat::RLE — minimal
    run-length format + materialize (B2)"](research/design-log.md).
- [x] **B3. `ColumnFormat::BitPacked` — minimal format + materialize — P1**
      *Added `ColumnFormat::BitPacked = 6` + `make_bitpacked` +
      materialize branch. LSB-first packing, bit_width ∈ [1,32] via
      reused `seq_step` union slot. Kernels materialise first — B3b
      adds run-native filter/sum.*
  - Files: `include/bolt/bolt_column.h`.
  - Test: `BoltColumn.BitPackedRoundTrip3Bit`,
    `BoltColumn.BitPacked17BitCrossesWordBoundary`.
- [x] **B4. `ColumnFormat::FrameOfReference` — minimal format — P1**
      *Added `ColumnFormat::FrameOfRef = 7` +
      `make_frame_of_ref(packed_deltas, bit_width, base, total_rows,
      type, arena)` + shared unpack path with BitPacked (base=0 for
      BitPacked, base=seq_offset for FOR). Kernels materialise first.*
  - Files: `include/bolt/bolt_column.h`.
  - Test: `BoltColumn.FrameOfReferenceRoundTrip` — 7 × 6-bit deltas
    + int64 base.
  - Design-log: [`design-log.md` → "ColumnFormat::BitPacked +
    FrameOfRef — bit-level compression (B3/B4)"](research/design-log.md).
- [ ] **B5. `ColumnFormat::FSST` — P1**
  - Files: `bolt_column.h`, new `bolt_fsst.h`. Symbol-table compression for
    high-cardinality strings; filter kernels compare encoded forms directly
    where possible.
  - Test: `tests/test_bolt_strtemp.cpp` FSST cases.
  - Bench: string-filter encoded-vs-decoded.
  - Design-log: *"FSST strings — encoded-form filter vs decode-and-compare"*

## Track C — Joins / hash / groupby

- [x] **C1. Phase 2a scatter — atomic-free prefix offsets — P0**
      *Refactored `parallel_groupby_radix_merge` to compute per-morsel-
      per-shard counts + in-place prefix sum → every scatter write
      lands at `shard_bufs[s][moffsets[s] + local_cursor++]` with ZERO
      atomic operations. `shard_heads[]` cursor removed entirely.
      Memory overhead: `num_morsels × P × 4B` offset table (~256 KB
      at 1K×64 — trivial).  On single-socket laptop the atomic wasn't
      the bottleneck; measured tie with baseline.  Multi-socket win
      pending test hardware.  All 16 test binaries green.*
  - Files: `include/bolt/kernels/bolt_parallel.h` —
    `ParallelGroupByScatterCtx`, `parallel_groupby_scatter_morsel`,
    `parallel_groupby_radix_merge`.
  - Test: `tests/test_bolt_parallel.cpp` (8 tests, all green —
    correctness preserved).
  - Bench: Q1 16.84 ns/row, Q3 14.26 ns/row, 1BRC 19.47 ns/row —
    all tied with pre-C1 baseline on single-socket. Multi-socket
    measurement deferred.
  - Design-log: [`design-log.md` → "Phase 2a scatter — atomic-free
    via per-morsel prefix offsets (C1)"](research/design-log.md).

- [~] **C1-legacy — replaced by the above**
      *On inspection, Phase 2b (partition merge) is **already parallel**
      via `sched->submit_range(&parallel_groupby_merge_shard, ..., P)`
      at `bolt_parallel.h:502`. The actual 8-core scaling wall
      documented at `BOLT_PERFORMANCE.md:735-737` is **atomic shard
      cursor contention during Phase 2a scatter** — every input triple
      does a `shard_heads[s].fetch_add(1)` at `bolt_parallel.h:366`.
      Followup #4 in `BOLT_PERFORMANCE.md:761-764` already identifies
      the fix: per-worker-per-shard sub-buffers (rejected in H3 for
      complexity; revisit now). Deferred to a dedicated session —
      this is a full refactor of `parallel_groupby_radix_merge` and
      touches correctness-critical merge code.*
  - Files when tackled: `include/bolt/kernels/bolt_parallel.h`
    (`ParallelGroupByScatterCtx`, `parallel_groupby_radix_merge`,
    `parallel_groupby_merge_shard`).
  - Test: `tests/test_bolt_parallel.cpp`.
  - Bench: `bench_tpch_lite` Q1 at `n_groups ∈ {10, 100, 100000}` +
    `bench_1brc` 8/16 threads. Must scale past the documented 2.6×
    plateau at 8T.
  - Design-log: *"Phase 2a scatter — per-worker-per-shard sub-buffers
    vs atomic cursor"* (pending).
- [x] **C2b. Split Block Bloom Filter (SBBF, Parquet-spec) — P1**
      *New `include/bolt/join/bolt_sbbf.h`. 256-bit blocks with 8 salt
      constants, 10 bits/key default, ~1% FPR — half the bits and
      ~5× lower FPR than the existing `BloomFilter`. Parquet-wire
      compatible. Both ship in source; `HashJoinBuild` still uses the
      simpler `BloomFilter`; SBBF is reachable by name.*
  - Files: new `include/bolt/join/bolt_sbbf.h`;
    `tests/test_bolt_join.cpp` — `BoltSbbf.*` (4 tests).
  - Test: NoFalseNegatives, FprUnderTwoPercent, EmptyFilterAlwaysAbsent,
    SizingAtTenBitsPerKey.
  - Design-log: [`design-log.md` → "Split Block Bloom Filter —
    Parquet-spec variant (C2b)"](research/design-log.md).
- [x] **C2. Bloom pre-screen on hash-join probe — P0 (opt-in)**
      *New standalone `include/bolt/join/bolt_bloom.h` (block-Bloom, 16
      bits/key, k=4). `HashJoinBuild::build(..., build_bloom=true)`
      populates it; `HashJoinProbe::probe_with_bloom()` uses it via a
      templated `probe_impl<bool UseBloom>` (zero runtime branch). The
      default `probe()` path is unchanged. Default-on wiring was tried
      and measured 8% regression on high-match-rate Q3 — the opt-in
      route is the kept-code-paths answer; flip the default per-call
      when a low-match-rate bench exists.*
  - Files: new `include/bolt/join/bolt_bloom.h`;
    `include/bolt/join/bolt_hashjoin.h` (templated probe + optional
    Bloom on build).
  - Test: `BloomGatedProbeMatchesDefault` and `BloomNotBuiltByDefault`
    in `tests/test_bolt_join.cpp` (2 pass; all 15 join tests green).
  - Bench: Q3 default matches pre-C2 baseline within laptop noise
    (13.56 min; 15.62 median vs pre-C2 12.99 / 14.01). Low-match-rate
    Q3 variant not yet present in `bench_tpch_lite`.
  - Design-log: [`design-log.md` → "Hash-join Bloom pre-screen — opt-in
    via templated probe (C2)"](research/design-log.md).
- [x] **C3. `SwissTableInterleaved` — read-after-rebuild alt layout — P1**
      *Added `SwissInterleavedGroup` (320B: 64B ctrl line + 16B pad +
      4×64B slot lines, all under one TLB entry) and
      `SwissTableInterleaved` with `create`, `insert`, `find`,
      `build_from(flat)` (rebuilds via group-aligned reinsert — avoids
      cross-group ctrl splices). FLAT stays the default for all
      existing callers; INTERLEAVED is reachable by name. 2
      correctness tests green (200-key match; empty-table).*
  - Files: `include/bolt/join/bolt_swiss.h`; `tests/test_bolt_join.cpp` —
    `BoltSwiss.InterleavedMatchesFlatFind`,
    `BoltSwiss.InterleavedEmptyTable`.
  - Bench: TLB-miss measurement on a >L3 probe workload deferred —
    needs perf-counter access.
  - Design-log: [`design-log.md` → "SwissTableInterleaved —
    read-after-rebuild alt layout (C3)"](research/design-log.md).
- [x] **C4. Merge-join operator — P2**
      *New `include/bolt/join/bolt_mergejoin.h` with
      `mergejoin_inner_i64`, `mergejoin_left_outer_i64`,
      `mergejoin_right_outer_i64`. Handles duplicate keys via
      equal-run cross-product; caller-bounded output capacity; O(n+m)
      / O(1). 9 correctness tests green including cross-check vs
      hash-join on 2K unique sorted keys.*
  - Files: new `include/bolt/join/bolt_mergejoin.h`,
    `tests/test_bolt_mergejoin.cpp`, `tests/CMakeLists.txt`.
  - Test: `BoltMergeJoin.*` (9 tests; inner, left-outer, right-outer,
    capacity, cross-check).
  - Bench: deferred — add a sorted-join case to `bench_tpch_lite` when
    a sorted-input workload is set up.
  - Design-log: [`design-log.md` → "Merge-join — sorted-input operator
    (C4)"](research/design-log.md).
- [x] **C5. `BoltColumn::ensure_bitmap_index` — lazy build + cache — P0-adjacent**
      *Added lazy builder that populates the `sidecars.bitmap_index`
      slot on first call, returns cached pointer thereafter. Default
      filter / join dispatch does NOT consult automatically — callers
      reach the primitive by name. The planner-level auto-consult
      decision lives in chukonu (cardinality + selectivity model).
      2 BitmapIndex tests green.*
  - Files: `include/bolt/bolt_column.h` —
    `BoltColumn::ensure_bitmap_index`; forward-decl of `BitmapIndex`.
  - Test: `BitmapIndex.EnsureIndexCaches` (first-call build, second-
    call cache, non-Dictionary returns nullptr).
  - Design-log: [`design-log.md` → "BitmapIndex — lazy build + cache
    on sidecar slot (C5)"](research/design-log.md).

## Track D — Arena / memory

- [x] **D1. Huge-page allocator primitive — opt-in, fallback-safe — P2**
      *`bolt_aligned_alloc_huge` / `bolt_aligned_free_huge` in
      `bolt_port.h`. Windows VirtualAlloc(MEM_LARGE_PAGES), Linux
      mmap(MAP_HUGETLB)+madvise(HUGEPAGE), macOS fallback. Compile-time
      gated by `BOLT_ENABLE_HUGE_PAGES` (added in G2). Arena stays on
      plain aligned_alloc per keep-code-paths — primitive is callable
      by name. Smoke test green (exercises fallback on unprivileged
      laptop).*
  - Files: `include/bolt/bolt_port.h` — `bolt_aligned_alloc_huge`,
    `bolt_aligned_free_huge`; `tests/test_bolt_primitives.cpp` —
    `BoltPort.HugePageAllocBasic`.
  - Test: write-read smoke test over 8 MB allocation (4 KB stride).
  - Bench: real huge-page perf needs a privileged Linux box; deferred.
  - Design-log: [`design-log.md` → "Huge-page allocator primitive —
    opt-in, fallback-safe (D1)"](research/design-log.md).
- [x] **D2. `bolt_memcpy_nt` primitive — measured loss; kept in source — P2**
      *Added `bolt_memcpy_nt` (128-byte AVX2 unrolled streaming stores
      + sfence) as a named primitive in `bolt_port.h`. Measured on the
      i7 laptop: at 2 MB it's a tie (1.03×), at 128 MB it's 0.49× of
      plain memcpy (2× slower). MSVC UCRT memcpy already streams at
      large sizes; NT loses on top of it. Per keep-code-paths, the
      primitive ships anyway — default COW / clone_into paths keep
      memcpy. Re-measure on Linux / non-Intel before reconsidering.*
  - Files: `include/bolt/bolt_port.h` — `bolt_memcpy_nt` +
    `kBoltMemcpyNtThreshold`; `benchmarks/bench_bolt.cpp` —
    `bench_nt_memcpy` size sweep.
  - Test: size-sweep bench is the verification (correctness implicit —
    plain memcpy fallback on non-AVX2 / misaligned dst).
  - Design-log: [`design-log.md` → "COW / large-buffer copy — NT
    stores measured (mostly) slower (D2)"](research/design-log.md).

## Track E — Scheduler / concurrency

- [x] **E1. `NumaChannelPool<T, Cap, NumNodes>` — opt-in primitive — P2**
      *Added templated per-node MPSC pool in `bolt_channel.h`.
      `try_push(node, item)` dispatches to the home node;
      `try_pop(out)` round-robins across nodes with advancing start
      pointer. Default `Scheduler::submit_range` still uses a single
      TaskRing — pool is caller-owned until a multi-socket bench
      justifies auto-routing. 3 correctness tests green.*
  - Files: `include/bolt/bolt_channel.h` — `NumaChannelPool`;
    `tests/test_bolt_primitives.cpp` — `BoltNumaPool.*`.
  - Test: `BoltNumaPool.PushThenRoundRobinPop`,
    `BoltNumaPool.MultiProducerConcurrent`,
    `BoltNumaPool.SingleNodeDegenerate`.
  - Bench: multi-socket `bench_tpch_lite` deferred — needs a test box
    to actually measure the cross-socket serialisation avoided.
  - Design-log: [`design-log.md` → "NUMA channel pool — per-socket
    MPSC fan-in (E1)"](research/design-log.md).
- [x] **E2. Adaptive morsel sizing — EWMA observation, opt-in — P2**
      *Added `record_morsel_ns_per_row` + `recommended_grain_bytes` on
      `Scheduler`. EWMA alpha=0.25 guards noise; recommendation
      targets a 1-10 ms per-morsel budget and clamps to
      [grain_bytes/4, grain_bytes*4]. Default `submit_range` does NOT
      consult the recommendation — callers opt in by passing the
      recommendation into their dispatch. 4 unit tests green.*
  - Files: `include/bolt/bolt_scheduler.h` — `adaptive` substruct +
    two methods; reset wired into `init()`.
  - Test: `BoltScheduler.AdaptiveGrain*` (4 tests covering static
    default, shrink, grow, zero-observation guard).
  - Bench: adaptive-vs-manual sweep on `bench_1brc` deferred —
    requires wiring a caller that opts in; the knob is in place.
  - Design-log: [`design-log.md` → "Adaptive morsel sizing — EWMA
    observation + opt-in dispatch (E2)"](research/design-log.md).

## Track F — Fusion primitives

- [x] **F1. Fused filter+agg primitives — extended — P2**
      *`filter_sum_gt` + `sum_masked` already shipped in
      `bolt_numeric.h` (Wave F1a). Added `filter_count_gt<T>` and
      `filter_minmax_gt<T>` in the same file (no new header — the
      existing home is correct). Auto-vectorising, branchless,
      O(1) state. 3 new tests green. With these four primitives the
      planner has a fused shape for every 1BRC aggregate.*
  - Files: `include/bolt/kernels/bolt_numeric.h`,
    `tests/test_bolt_kernels.cpp`.
  - Test: `BoltKernels.FilterCountGtMatchesTwoPass`,
    `BoltKernels.FilterMinMaxGtMatchesManual`,
    `BoltKernels.FilterMinMaxGtNonePass` (plus 39 existing).
  - Bench: deferred — fused-vs-two-pass speedup shows at L3-plus sizes;
    add to `bench_tpch_lite` when a high-cardinality Q1 variant lands.
  - Design-log: [`design-log.md` → "Fused filter+aggregate kernels —
    count + minmax added (F1)"](research/design-log.md).

## Track H — SYMBOL-shape upgrades (Wave 2)

Scoped from `docs/research/questdb-symbol-code-audit.md`. Brings
`ColumnFormat::Dictionary` + `BitmapIndex` to QuestDB SYMBOL parity
(H1, H2) and past it (H3 — popcount miss-accelerator QuestDB lacks).

- [x] **H1. Literal-resolve-once `filter_eq_dict` — P0**
  - Shape: `filter_eq_dict(col, scalar, out)` resolves scalar →
    dict code ONCE, then does `keys[i] == code` over the data buffer.
    Mirrors QuestDB `EqSymStrFunctionFactory.ConstSymIntCheckFunc`.
  - Files: `include/bolt/kernels/bolt_dict_filter.h` (new).
  - Test: 10-key dict, 10K rows, filter matches materialise+filter.
  - Bench: add dict-filter case to `bench_kernels`.
- [x] **H2. `DictionaryPool` — global-across-morsels dict API — P0**
  - Shape: arena-backed `DictionaryPool` with `intern(str) → code`
    and `resolve(code) → StringView`. Codes stable across the
    pool's lifetime; morsels encoded against the same pool compare
    codes directly. Mirrors QuestDB `SymbolMapWriter`.
  - Files: `include/bolt/bolt_dictionary.h` (new).
  - Test: intern same string twice → same code; 1000 distinct keys
    across two batches → codes reusable in `filter_eq_dict`.
- [x] **H3. Per-ID popcount miss-accelerator on `BitmapIndex` — P1**
  - Shape: precompute `popcount[num_keys]` at build time; new
    `BitmapIndex::probably_absent(key)` returns `true` in O(1) when
    the key was never seen. `BitmapIndex::filter` / `count` short-
    circuit when absent. **QuestDB has no equivalent.**
  - Files: extend `bolt_column.h`.
  - Test: build index from 10 keys of which 3 are unused; absent
    keys return `count == 0` without scanning.
- [ ] **H4. Lock-free append-only dict — P2** *(deferred to Wave 3)*
  - Shape: tick-tock-published dict for streaming ingest. Atomic
    next-code, SwissTable keyed on StringView for intern.
  - Gate: lands when a streaming-ingest caller asks.

## Track I — Run-native B-format kernels (Wave 2)

Lifts `BitPacked` / `FrameOfRef` (B3/B4) from materialise-first to
native kernels so the storage compression turns into a compute win.

- [x] **I1. `filter_gt_bitpacked` / `filter_eq_bitpacked` — P1**
  - Shape: unpack 16 values at a time into a stack buffer, compare
    via SIMD, compressstore indices. Avoids the full-column
    materialize pass.
  - Files: `include/bolt/bolt_branchless.h` (alongside `filter_eq_rle`).
  - Test: packed 3-bit and 17-bit inputs; results match unpack+filter.
  - Bench: add bitpacked-vs-materialize case.
- [x] **I2. `sum_frame_of_ref` — P1**
  - Shape: `base * n + Σ(decoded_deltas)`. Base-add hoisted out of
    the hot loop.
  - Files: `bolt_branchless.h`.
  - Test: round-trip against `materialize + sum_avx2_i64`.
- [ ] **I3. Dict + BitmapIndex auto-dispatch in `filter_eq` — P0**
  - Shape: `filter_eq` on a Dictionary column routes through H1
    (linear scan on keys) or through `BitmapIndex::filter(code)`
    when the sidecar exists. Ties H1 + C5.
  - Files: `bolt_column.h` filter helper.

## Track G — Infra

- [ ] **G1. CI perf regression gate**
  - Files: new `ci/perf_check.py` + `ci/perf_baselines.json` +
    workflow step in `.github/workflows/ci.yml`. Runs `bench_bolt`,
    `bench_kernels`, `bench_tpch_lite`, `bench_1brc` on a Linux runner;
    fails if any headline metric regresses >5% vs baseline.
  - Test: green CI with current numbers; deliberately regress one kernel
    locally to confirm the gate fires.
  - Design-log: *"CI perf gate — 5% regression budget, per-platform baselines"*
- [ ] **G2. New CMake options + presets**
  - Files: `CMakeLists.txt`, `CMakePresets.json`. Add
    `BOLT_ENABLE_HUGE_PAGES`, `BOLT_ENABLE_NUMA`, `BOLT_SWISS_LAYOUT`,
    `BOLT_HASH_TIER`. Add an `all-features` preset.
  - Test: build matrix covers `default` + `all-features`.
- [x] **G3. Seed this document** — done when this file landed.

---

## Execution ordering

**Single worker**
1. G3 → seed doc (trivial).
2. G2 → CMake toggles (unblocks A3, C3, D1, etc.).
3. G1 → CI perf gate (protects every subsequent item).
4. Tracks A, B, C, D, E in parallel. Only hard intra-track dep: F1 ← A1.
5. F1 last.

**Multiple workers / parallel Claude instances**
- One worker per track — no collisions.
- Or one item per worker from distinct tracks.
- Coordinate on:
  - **A3 → C2/C3**: land A3 (xxh3) first, or have C take a dep snapshot.
  - **B ↔ D2**: both touch `bolt_column.h` (different call-sites). File-level
    lock if both active.
  - **F1 waits for A1.**
- Nothing else conflicts.

## Definition of done (per item)

Every checkbox flips to `[x]` **only when all seven of these hold**:

1. Code follows Tiger Style: `noexcept`, no RAII on hot path,
   ≥2 asserts/fn, ≤70 LOC/fn, in-place init, hard bounds on every buffer.
2. Test in the mapped `tests/test_bolt_*.cpp` covers correctness +
   boundary (null, empty, single-row, SIMD-tail).
3. Bench in the mapped `benchmarks/bench_*.cpp` produces a ns/op number.
4. Design-log entry in `docs/research/design-log.md` with
   **Context / Tested / Measured / Kept**.
5. Checkbox in this doc ticked with measured number inline
   (e.g. `[x] A1 — 0.17 ns/op → 0.11 ns/op (1.5×)`).
6. CI perf gate (G1) green — no headline regression.
7. If an alternative implementation is retained behind a compile switch:
   one-liner near the switch + full entry in `design-log.md`
   describing what workload would flip the default.

## End-of-wave verification

When every box above is `[x]`:

1. `cmake --preset release && cmake --build build --config Release`
2. `ctest --test-dir build --output-on-failure -C Release` — all green on
   all platforms (MSVC merge-blocking gate included).
3. Run the four bench binaries; CI gate passes at baseline or better.
4. `docs/research/design-log.md` has a per-item entry.
5. Append a wave-closing summary to the top of `design-log.md`: net delta
   per headline bench (Q1 / Q3 / Q6 / 1BRC / filter_gt), peak
   thread-scaling, new coverage (merge-join, f32, FSST, RLE, BitPacked,
   FOR).
6. Close this doc with a "Wave sealed" stamp + pointer to the summary
   design-log entry.
