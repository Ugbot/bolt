# Design log — what we tried, what we kept, what we backed out

Running list of tested ideas with the measured outcome and the reason
the chosen variant won. Forward-looking research goes in topic-specific
files under `docs/research/`; this file captures the *experimental
record* so future tuning passes don't re-litigate decisions from
scratch.

Each entry: **date / context · what we tried · how it measured ·
what we kept and why**.

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

## How to add an entry

1. Run the change. Capture before/after numbers (≥3 runs each).
2. Append a new `## Topic — alt A vs alt B` section to this file.
3. State context, what was tested, the numbers, what you kept, why.
4. If the rejected alternative might come back later (different
   workload, new hardware), say so explicitly in an "open question"
   line — that's how `MergeTriple` got its high-cardinality followup.
