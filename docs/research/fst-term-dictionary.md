# FST term dictionary (`bolt::kernels::bolt_fst`)

Layer 1.5 of plan `this-was-a-freach-hashed-crab.md`. Lucene-shape minimal
acyclic FST mapping sorted byte keys to `int64_t` outputs, used as the
term dictionary for BM25 (Layer 2.3) and as the substrate primitive for
JSONB path dictionaries (Layer 3) and a future AST identifier dictionary.

## What we read

- Lucene's `FST.java` / `FSTCompiler.java`. The 2010-era "FST in Lucene"
  Adrien Grand / Mike McCandless writeups.
- *Direct Construction of Minimal Acyclic Subsequential Transducers*,
  Mihov & Maurel (2001). The incremental algorithm that operates on
  sorted-input streams in a single pass.
- TigerBeetle Tiger Style discipline applied to all of Bolt.

## What we lifted

- **Incremental suffix minimisation while keys arrive sorted.** Caller
  emits keys in strict ascending byte order; on each `add(K)` we compute
  the common prefix length `C` with the last key, freeze every frontier
  node from `last_key_len` down to `C+1` into the packed buffer, and
  extend the frontier with new arcs for the suffix of `K`. Frozen nodes
  go through a fixed-cap dedup hash table keyed by
  `(arc_count, arc_bytes)` so identical subtrees collapse to one
  byte-range.
- **Byte-keyed arcs with attached output values.** Each arc is 14 bytes
  packed: `{ uint8 label; uint8 flags; int32 target_offset; int64 output }`.
  No struct packing pragmas; we memcpy field-by-field to dodge alignment
  traps on the int64 load.
- **Flag bits.** `IsFinal=1`, `IsLastArc=2` packed into one byte.
- **Sentinel root behaviour.** Byte offset 0 is reserved as a 0-arc
  empty node so dedup-empty-slot detection (`offset==0`) doesn't alias a
  real node, and so empty subtrees past a leaf's IsFinal arc freeze to
  the same canonical address.

## What we omitted (deferred until measured benefit)

- **Output composition for shared-suffix outputs.** Lucene pushes the
  monoid-prefix of outputs up to the parent arc so two keys that share a
  byte suffix can share the suffix node despite differing outputs. We
  attach the full output to the IsFinal arc instead. This means
  *suffix dedup only fires when the leaf outputs coincide*. With unique
  outputs (typical term dictionary) the FST still benefits from prefix
  sharing but suffix sharing is suppressed.
  - Measured impact on the `term0000..term0999` corpus: **31.5×**
    compression vs. raw `(key + output)` with identical outputs (suffix
    dedup live), **1.01×** with unique outputs (suffix dedup suppressed).
    Lucene-with-composition would keep the ratio above 5× for the
    unique-output regime — that is the next optimisation when a consumer
    profile shows the term dict dominating segment size.
- **Packed bit-width arc-target encoding.** Lucene packs target offsets
  to `ceil(log2(buf_size))` bits and uses variable-byte encoding for
  outputs. We pay 14 bytes per arc to keep memcpy access patterns
  alignment-clean. Worth revisiting once a consumer measures FST bytes
  in the segment hot path.
- **Direct-addressing arcs at high-fan-out nodes.** Lucene switches from
  linear-scan arcs to a 256-entry direct-index node when arc_count
  approaches the byte alphabet. Our linear scan is fine at typical
  arc_count < 16; a follow-up can specialise the dispatch.

## API shape (header `bolt/kernels/bolt_fst.h`)

```cpp
struct FstView { const uint8_t* nodes; int32_t bytes; int32_t root_offset; };
static_assert(sizeof(FstView) == 16);

bool fst_builder_init  (FstBuilder*, Arena*, int32_t initial_cap) noexcept;
bool fst_builder_add   (FstBuilder*, const uint8_t* key, int32_t len,
                        int64_t output) noexcept;   // strictly ascending
bool fst_builder_finish(FstBuilder*, FstView* out)  noexcept;

bool fst_lookup        (FstView, const uint8_t* key, int32_t len,
                        int64_t* out) noexcept;
bool fst_scan_prefix   (FstView, const uint8_t* prefix, int32_t plen,
                        FstScanCb, void* user, Arena* scratch) noexcept;
bool fst_scan_range    (FstView, lo, lo_len, hi, hi_len,
                        FstScanCb, void* user, Arena* scratch) noexcept;
```

Header-only with `inline` to avoid touching Bolt's CMake (Bolt's `src/`
glob is gated behind `BOLT_BUILD_PARSE` / `BOLT_BUILD_INGEST` and our
kernel doesn't need a TU).

## Tiger-Style discipline

- All allocation through `bolt::Arena*`; no raw malloc, no STL containers
  in the kernel.
- Public functions `noexcept`, `≥2 assertions`, `≤70 lines`.
- POD types with `static_assert(sizeof == N)`.
- `BOLT_FORCE_INLINE` on the hot `fst_find_arc`; `BOLT_RESTRICT` on the
  buffer pointer.
- Dedup hash overflow degrades to "no dedup, fresh node" — never crash,
  never leak. Documented inline.

## Tests (`tests/test_bolt_fst.cpp`, 11 cases, all passing)

| Case                                | What it covers                                  |
|-------------------------------------|--------------------------------------------------|
| BuildAndLookupSingleKey             | minimal smoke                                   |
| BuildAndLookupMultipleKeysSharedPrefix | apple/apply/apricot prefix dedup            |
| AscendingOrderEnforced              | rejects equal + descending adds                 |
| MissingKeyReturnsFalse              | prefix-only / extra-byte / unrelated / empty   |
| PrefixIsAlsoAKey                    | "app" + "apple" + "applecart" coexisting        |
| PrefixScanCollectsAllMatches        | DFS from prefix terminal                        |
| PrefixScanEmptyPrefixWalksWholeTree | empty prefix iterates everything                |
| RangeScanByteWise                   | `[lo, hi)` bytewise filter                      |
| CompressionRatioCheck               | 31.5× on 1k shared-output keys                  |
| CompressionRatioCheckUniqueOutputs  | sentinel for the deferred-composition regime    |
| LargeKeySetIntegrity                | 100k inserted + 10k negatives, all correct      |

## Open questions

1. When a real BM25 consumer wires this in (Layer 2.3), does the
   unique-output suffix-suppression cost dominate `.tim` file size? If
   yes, output composition jumps the queue.
2. Direct-addressed arcs vs linear scan: at what `arc_count` threshold
   does the direct-index node win on lookup latency for our workloads?
3. The 4096-entry dedup table is sized for term dictionaries up to a few
   tens of thousands of nodes. JSONB path dictionaries and AST
   identifier dictionaries may want a larger cap or a different fallback
   (e.g. open-addressed table sized at `init`). Re-measure when a second
   consumer lands.
