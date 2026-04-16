# Branchless hash-table probing — what's left to remove

Bolt's groupby + SwissTable hot loop has reached a perf shape where
~3-4 branches per row remain. We've already removed the easy ones
(per-row capacity check, per-row prefetch bound check, branchful
min/max via `bolt::branchless::bmin`/`bmax`). This note catalogues the
literature on removing the *remaining* branches, with a verdict per
candidate.

## The branches we want to remove

```cpp
// In SwissTable::find (one probe iteration):
const bmm_vec_i8 cgrp = bmm_loadu_i8(&ctrl[base]);
uint32_t tag_mask   = bmm_movemask_i8(bmm_cmpeq_i8(cgrp, vtag))   & 0xFFFFu;
uint32_t empty_mask = bmm_movemask_i8(bmm_cmpeq_i8(cgrp, vempty)) & 0xFFFFu;
while (tag_mask) {                          // BR1: bitmask iter (rare > 1)
    int lane = bolt_ctz32(tag_mask);
    int p    = (base + lane) & mask;
    if (slots[p].key == key) return ...;    // BR2: tag false-positive
    tag_mask &= tag_mask - 1;
}
if (empty_mask != 0) return -1;             // BR3: probe terminator
base = (base + 16) & mask;                  // BR4: probe step (cmov-able)

// In GroupByTable::ingest_unchecked:
int32_t existing = ht.find(key);
if (existing >= 0) { /* hit path */ }       // BR5: hit vs insert
// else /* insert path */
```

## Production systems we benchmarked against

### Abseil SwissTable
([abseil.io/about/design/swisstables](https://abseil.io/about/design/swisstables))

The original chunked-with-tag design Bolt borrowed. Their inner loop
has the same BR1–BR4 shape. They mitigate with: tag broadcast outside
the loop, `_pext_u32` (BMI2) for tag-mask extraction on systems with
it, and an inline `H1` precompute. Net: **same branches as us**, but
with a few cycles shaved on x86 BMI2.

### Folly F14
([F14.md design doc](https://github.com/facebook/folly/blob/main/folly/container/F14.md),
[F14Table.h](https://github.com/facebook/folly/blob/main/folly/container/detail/F14Table.h))

14-byte chunks instead of 16. Each chunk has a 14-byte data area + 2
bytes of *chunk metadata* (overflow counter + capacity scale). The
overflow counter exists specifically to **eliminate BR3**: instead of
"probe terminates when an empty slot appears", probe terminates when
the chunk's overflow counter says no more chunks were ever spilled
from this position. **One branchless terminator instead of an empty-
mask test**. They also use SIMD (SSE2/NEON) for the tag scan, same as
our `bmm_cmpeq_i8`.

What we'd borrow: **chunk-overflow counter**. Replaces BR3 with a
predicted-once load that's almost always zero. Cost: extra 2 bytes per
chunk (~12% overhead at chunk_width=16). Insert needs to maintain the
counter (one branch per insert that doesn't fit the home chunk; rare).

### Martin Ankerl's Robin Hood
([robin-hood-hashing](https://github.com/martinus/robin-hood-hashing),
[modern overview](https://andre.arko.net/2017/08/24/robin-hood-hashing/))

Each entry tracks its **probe distance** (how far it is from its ideal
home position). On lookup, you compare the *current* probe distance to
the entry's stored distance — if yours exceeds theirs, the key is
absent (would have been swapped earlier in the chain). This **replaces
BR3 with an arithmetic compare** that's CMOV-friendly.

Insert is more expensive: on collision, the entry with the lower probe
distance gets evicted. Each evicted entry's probe distance must be
updated.

What we'd borrow: **probe-distance early-exit**. Slot grows from 16→18
bytes (2-byte distance counter); could reuse a tag byte if we narrow
the tag from 7→6 bits. Mixed feelings: we lose tombstone-free property
on delete (have to shift entries back).

### Cuckoo hashing
([Pagh-Rodler paper](https://www.itu.dk/people/pagh/papers/cuckoo-jour.pdf),
[libcuckoo](https://github.com/efficient/libcuckoo))

Two hash functions; each key lives in one of exactly **two** slots.
Lookup is **fully branchless**: read both slots, parallel-compare both
keys, return the matching value (or -1 if neither matches). **BR1, BR3
both gone.** Insert: cuckoo eviction can chain; in the worst case,
table needs rebuild.

What we'd borrow: **the lookup pattern** (always read N candidate
slots, branchlessly select). Doesn't have to be Cuckoo's 2-table
scheme — could be SwissTable with a "always read up to 4 chunks" form.

Limitation: load factor < ~0.5 for stable insert; ours runs at 0.5
already so ok, but we'd lose the cardinality-tight sizing win.

### DuckDB Aggregate Hashtable
([2022/03 blog](https://duckdb.org/2022/03/07/aggregate-hashtable))

Linear probing with **hash-prefix caching**: each entry stores a
2-byte hash prefix. Lookup compares prefix first (one int compare),
then full key only on prefix-match. Avoids BR2's full-key compare for
the common false-positive-tag case — but only in tag-collision edge
cases since SwissTable's 7-bit tag already has FP rate ~1/128.

Their parallel hash-aggregate's biggest win is the **two-phase
build**: thread-local unpartitioned tables → switch to
radix-partitioned on overflow. We already do this (Wave L1).

### PtrHash (Groot Koerkamp)
([curiouscoding.nl/posts/ptrhash](https://curiouscoding.nl/posts/ptrhash/))

Minimal perfect hashing — function built once from a known key set;
lookup is **one multiply + shift, zero collisions, zero probes**. All
five branches gone, replaced by a deterministic O(1) load. Build cost
is significant but amortizes if the key set is reused.

Constraint: requires the key set to be known at build time. Doesn't
help streaming groupby with arbitrary keys; **does** help when we know
a dictionary's contents (enum columns, fixed station name sets).

## Bolt-specific candidates, ranked

| Technique | Removes | Bolt cost | Estimated win | Risk |
|---|---|---|---|---|
| **CMOV probe step** | BR4 | <10 LOC | 0.5-2% | ✅ low |
| **F14 chunk overflow counter** | BR3 | +2 B/chunk, ~30 LOC | 2-5% | medium |
| **Two-pass hoisted probe** | BR1+BR5 separation | ~80 LOC, +scratch buffer | 5-15% at high cardinality | medium |
| **AVX-512 conflict-detected vector probe** | BR1+BR5 | needs AVX-512 hw | unknown (no local hw) | high (untestable here) |
| **Robin Hood probe-distance** | BR3 | slot+2 B, +shift-on-delete | 3-8% | medium-high |
| **Cuckoo "always read N slots"** | BR1+BR3 | 2× slot reads/lookup | trades BR1 for memory bandwidth | medium |
| **PtrHash for known dictionaries** | BR1-BR5 all | offline build, fixed key set | 2-10× on dict aggs | low (orthogonal) |

### Top picks for next experiment

#### 1. CMOV probe step (BR4)

Replace:
```cpp
base = (base + kSwissGroupWidth) & mask;
```
with a predicated form so the compiler emits a single CMOV against the
"hit found" condition rather than a fall-through that's branchful when
combined with the loop iterator:

```cpp
// Probe step happens unconditionally; the find() loop continues
// only because the previous iteration didn't return — a CMOV-friendly
// shape if we restructure the find() into a fixed-bound loop.
for (uint32_t step = 0; step < max_probes; ++step) {
    base = (base_initial + step * kSwissGroupWidth) & mask;
    // ... probe body
}
```

Trades: known max_probes upper bound (we have one already at
`kSwissMaxProbeCaps`). Low risk; clean to test; small expected delta
on a workload that's already cache-bound.

#### 2. Two-pass hoisted probe (BR5 separation)

Split `ingest` into:

- **Pass A — find** (branchless): for each row, call `ht.find(key)` and
  write `existing[i]` to a scratch array. The find itself still has
  internal branches (BR1-BR3), but they're now in a tight one-pass
  loop the branch predictor can specialise on.
- **Pass B — accumulate hits** (branchless inner loop): for each row,
  if `existing[i] >= 0` use scatter-update on the payload. Otherwise
  push to a "miss queue".
- **Pass C — insert misses** (small loop, branchful but rare): walk
  the miss queue.

Pass B becomes a SIMD-friendly inner loop. The hit/miss split is
amortized over a whole morsel rather than per-row. This is the same
"hoisted key-stream" we sketched in Wave M1 but never built — now
informed by the explicit branch-removal goal.

Expected: 5-15% on high-cardinality runs (where the BR5 mispredict
rate is non-trivial). Low-cardinality runs (1BRC's 413 stations) are
nearly all-hits after warmup; predicted ~100%; little win.

## What we'd skip

- **Cuckoo** — the always-read-2 pattern doubles memory bandwidth; on
  100K-station 1BRC we already hit the bandwidth wall (~95 ns/row).
  Cuckoo would push us further into bandwidth-bound territory.
- **Robin Hood backshift on delete** — Bolt's GroupByTable doesn't
  delete; the shift complexity is overhead with no payoff.
- **Folly F14's chunk_capacity_scale byte** — useful for resizable
  tables but Bolt's tables are fixed-capacity (Tiger Style). The
  overflow counter alone is the borrowable bit.
- **Hash-prefix caching (DuckDB)** — our 7-bit Swiss tag already hits
  the same false-positive ceiling (~1/128); a 2-byte prefix would only
  save the rare BR2 mispredict at the cost of 2 B/slot.

## Empirical update (Wave O)

Built and measured the two-pass hoisted probe:

| Stations | Default | Two-pass | Δ |
|---|---|---|---|
| 413     | 8.64 ns/row | 11.35 | **−24%** |
| 10 000  | 16.74 | 19.67 | −15% |
| 100 000 | 93.47 | 95.26 | parity |

**Two-pass is a loss at every cardinality we can measure.** The
expected branch-reduction win didn't show up because:

1. At low cardinality, the hit/miss branch is perfectly predicted
   after warmup — the "branchless" alternative removes a free branch
   while adding scratch-buffer + two-pass overhead.
2. At high cardinality, the workload is **memory-bandwidth bound on
   the probe itself**, not branch-misprediction bound. Removing
   branches doesn't help; the probe still touches the same cache
   lines at the same rate.

**Re-evaluation of the other candidates in light of this:**

| Technique | Original prediction | Updated prediction |
|---|---|---|
| F14 chunk-overflow (BR3) | 2-5% | likely **<1%** (same memory pattern) |
| Robin Hood probe-distance | 3-8% | likely **<1%** + insert cost |
| Cuckoo always-read-N | trades branch for bw | now obviously bad |
| PtrHash for known dicts | 2-10× | **still the right answer** for fixed-cardinality |
| AVX-512 vector probe | unknown | may help — fewer probes per row, not fewer branches |

The branch-removal lens was wrong for our workload. The right lens is
**memory-touches-per-row reduction**. PtrHash wins on this axis (one
load instead of N probes); SIMD vectorized probe wins by amortizing
cache-line cost across N keys.

**Decision:** stop chasing branch removal in the existing SwissTable.
The remaining branches (BR1, BR2, BR3) are below the perf noise floor
on memory-bound workloads. Future hashing work targets memory traffic:
PtrHash for known dictionaries, AVX-512 vectorized probe with conflict
detection for the SIMD path.

## Followups

- **Two-pass hoisted probe**: BUILT (Wave O); measured loss at every
  cardinality; kept as opt-in via `--two-pass` flag and
  `parallel_groupby_two_pass_override()`. See empirical update above.
- **CMOV probe step**: NOT BUILT. Closer inspection of the existing
  `SwissTable::find` shows BR4 isn't actually a branch in our code
  (loop-induction-variable update). The compiler already CMOVs it.
  Closing as "no work needed".
- **F14 chunk-overflow counter**: NOT BUILT. The two-pass result
  shows our remaining branches sit below the memory-bandwidth ceiling
  at every measured cardinality; this experiment would likely return
  the same answer at higher implementation cost (modifies SwissTable
  insert + find).
- **PtrHash for known dictionaries** — separate larger research note
  in flight; pairs with future `bolt::dict::PerfectHash` for low-
  cardinality enum columns. See `1brc.md` followups. **This is the
  right next-direction work** based on the Wave O learning.
- **AVX-512 vectorized parallel probe** (Wave M2 queued) — already
  noted in `avx512-status.md`; the `bmm_conflict_i32_x16` primitive
  is in place. Needs target hardware. **Also likely-wins** because it
  reduces memory touches per row (not branches).

## Sources

- Folly F14 design doc: https://github.com/facebook/folly/blob/main/folly/container/F14.md
- Folly F14 source: https://github.com/facebook/folly/blob/main/folly/container/detail/F14Table.h
- Abseil SwissTable: https://abseil.io/about/design/swisstables
- Robin Hood (Ankerl): https://github.com/martinus/robin-hood-hashing
- Robin Hood overview: https://andre.arko.net/2017/08/24/robin-hood-hashing/
- DuckDB aggregate hashtable: https://duckdb.org/2022/03/07/aggregate-hashtable
- libcuckoo: https://github.com/efficient/libcuckoo
- Pagh-Rodler cuckoo paper: https://www.itu.dk/people/pagh/papers/cuckoo-jour.pdf
- PtrHash (Groot Koerkamp): https://curiouscoding.nl/posts/ptrhash/
- Kersten et al., "Compiled and Vectorized Queries", VLDB 2018
