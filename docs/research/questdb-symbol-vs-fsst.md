# QuestDB SYMBOL vs FSST — what Bolt actually needs

Context: Bolt already has `ColumnFormat::Dictionary` (key array + values
column) and a `BitmapIndex` sidecar. Before implementing
`ColumnFormat::FSST` (Boncz/Neumann, VLDB 2020), we compared against
QuestDB's SYMBOL type to check whether FSST is redundant or fills a
different niche. Sources cited inline; `[M]` = measured by a primary
source, `[I]` = inferred from docs without a published number.

## 1. QuestDB SYMBOL physical layout

Files for a SYMBOL column [M, verified against QuestDB source — see
[`questdb-symbol-code-audit.md`](questdb-symbol-code-audit.md) for
file-path + line-range citations]:

- `<col>.d` — row-order int keys, **per-partition** (the column
  payload, one int per row). Suffix constant `FILE_SUFFIX_D` in
  `TableUtils.java`.
- `<col>.c` — symbol values, **table-global** (the string dictionary —
  actual UTF-8 bytes of distinct values). Produced by `charFileName()`
  in `TableUtils.java` (`.put(".c")`).
- `<col>.o` — offsets sidecar into `.c`, **table-global** (one 8-byte
  offset per dict entry). Produced by `offsetFileName()` in
  `TableUtils.java` (`.put(".o")`).
- When `INDEX` is set, add `<col>.k` + `<col>.v` **index** files
  (these are the bitmap-index key header + per-symbol row-ID-list
  blocks, distinct from the dictionary `.c`/`.o`). Produced by
  `BitmapIndexUtils.keyFileName` / `valueFileName`. Block size
  `cairo.index.value.block.size` default **256** (verified:
  `DefaultCairoConfiguration.getIndexValueBlockSize() { return 256; }`).

**Correction from the first draft of this note:** the dictionary pair
is `.c` + `.o`, *not* `.v` + offsets sidecar. `.k`/`.v` are exclusively
the indexed-variant's row-ID-list files. Audit against source is in
[`questdb-symbol-code-audit.md`](questdb-symbol-code-audit.md).

The dictionary is **table-global, not per-partition** — one symbol ID
means the same string across every partition ([M] QuestDB symbol
concept page). From 9.0.0 onwards, capacity is auto-managed; the old
`CAPACITY` clause is obsolete ([M] QuestDB release notes /
community).

## 2. Encode / decode cost

- **Ingest**: hash-probe the in-memory dict → existing ID or append
  new. `NOCACHE` skips the in-memory hash table and hits the `.v`
  file, trading RAM for ingest latency ([M] QuestDB symbol doc).
- **Read**: operators run on **integer IDs directly**. The string
  materialisation only happens at projection time when a row leaves
  the engine. Equality against a literal resolves the literal to an
  ID **once** at plan time and then does int-compare on `.d` ([M]
  QuestDB symbol concept + WHERE docs).

## 3. Filter / equality semantics

- `WHERE sym = 'AAPL'` → resolve `'AAPL'` once via dict, int-eq on
  `.d`. If indexed, the index gives the row-ID list directly — no
  scan ([M] QuestDB indexes doc).
- No Bloom/hash miss-accelerator on SYMBOL itself; Parquet partitions
  added Bloom filters in 9.3.4 for pushdown on Parquet-backed pages
  ([M] QuestDB 9.3.4 blog).
- **Non-exact** filters (`LIKE`, `ILIKE`, regex) do **not** use the
  index and fall back to full scan with per-row string compare ([M]
  QuestDB issue #4775).

## 4. Cardinality bounds

- Sweet spot: "repetitive" values — tickers, sides, exchange codes,
  tenants. Docs explicitly say use `VARCHAR` when values are unique
  or very high cardinality ([M] QuestDB schema docs).
- Measured degradation [M, issue #6246]: 300M rows, query returning
  ~28k rows:
  - 10 distinct symbols → milliseconds.
  - 10k distinct symbols → **15–25 s** warm, **>25 s** cold.
  - Degradation correlates with cardinality, **not** row count or
    result size. Started becoming visible around 1k distinct.
- 10M+ distinct → the in-memory dict cache "consumes significant
  memory"; docs recommend `NOCACHE` past that point ([M] QuestDB
  symbol doc).

## 5. Scan rates

No published ns/row or GB/s for SYMBOL scans in QuestDB's docs [I —
absent from all fetched pages]. The only concrete perf anchor is the
9.0.1 release note: "10×–100× speedup for index/concurrency-intensive
workloads" over 9.0.0 on indexed symbols ([M] QuestDB 9.0.1 release).

## 6. FSST vs SYMBOL head-to-head

FSST [M, Boncz/Neumann VLDB 2020 + cwida/fsst repo]:

- 255-entry symbol table, 1-byte codes, each code maps to a 1–8-byte
  symbol; 0xFF is escape for a literal byte.
- Random-access: individual strings decompress without touching
  neighbours. Not block-based.
- **Equality preserved in compressed form** — two equal strings have
  equal code bytes, so `=` works without decode.
- ~5–6 GB/s decompress, ~200–500 MB/s compress; compression ratio
  typically beats LZ4 on text-shaped data and matches it on speed.
- Wins on URLs, log lines, prefixed tenant:key, order IDs — anything
  with **substring correlation but high cardinality**.
- Loses on low-cardinality tickers — dictionary crushes it there
  because each row is already 1 int.

SYMBOL is the **low-cardinality, whole-string dedup** shape. FSST is
the **high-cardinality, substring-redundant** shape. They solve
disjoint problems; neither subsumes the other.

## 7. Parquet analogue

QuestDB's Parquet writer uses `RleDictionary` for SYMBOL and
`DeltaBinaryPacked` for timestamps ([M] QuestDB Parquet docs +
#6694). **No FSST.** FSST in Parquet is still a dev@parquet mailing
list proposal, not spec ([M] mail-archive thread). DuckDB ships FSST
natively in its storage format.

## 8. What Bolt gains by adding FSST

Bolt's `ColumnFormat::Dictionary` already matches SYMBOL's shape
one-for-one: int key column + value column + (optionally) a
`BitmapIndex` that is structurally what QuestDB's indexed-symbol
row-ID-list files are. FSST is **not redundant** — it covers the
cell QuestDB itself punts to `VARCHAR` with Parquet RLE/dictionary
fallback.

Workloads where FSST pays and Dictionary+BitmapIndex does not:

- Order IDs / execution IDs (UUID-shaped, near-unique, but share
  prefixes).
- URL columns, HTTP path / user-agent / referrer.
- Log message / error-string columns after PII-stripping.
- FIX tag-value blobs, multi-leg instrument descriptors, tenant-keyed
  compound strings.

## Recommendation

1. **Dictionary + BitmapIndex already covers the SYMBOL shape**
   (tickers, sides, exchanges, tenant enums). Do **not** reach for
   FSST for those — it's strictly worse than a 4-byte int column.

2. **Implement FSST as a sibling `ColumnFormat`, not a replacement.**
   Trigger via build-time opt-in per `feedback_multiple_impls_one_default`;
   default stays Dictionary. FSST kicks in for declared high-card
   string columns (order IDs, URLs, log lines). Kernel dispatch
   already branches on `ColumnFormat`, so the surface cost is a new
   variant + decode helper, not a hot-path branch.

3. **Quick wins on existing Dictionary that mimic SYMBOL ergonomics**
   — do these *before* FSST, they're higher leverage:

   - **Resolve-literal-to-ID once** in the filter-build step (Bolt's
     equivalent of QuestDB's planner move). Integer eq on the key
     array is what the branchless filter kernel is already good at.
   - **Global (table-wide) dictionary invariant**, not per-batch.
     QuestDB's big lesson: if the dict is global, every batch's
     int IDs are directly comparable; a merge/ASOF-join on symbol
     becomes int-int with no rehash. Today Bolt's Dictionary is
     per-column-instance — upgrading to a tick-tock global dict with
     COW promotion on new value arrival gets us the kdb/QuestDB
     semantics for free.
   - **Lock-free append-only dict for streaming ingest** — writer
     publishes new (string, id) pair under release-store; readers
     see old dict until next tick-swap. Same pattern we already use
     for COW columns.
   - **Sorted-ID invariant for monotone append** — when symbols are
     appended in first-seen order (the common tick-ingest case), ID
     order matches insertion time, so range-filters on "symbols seen
     since T" degenerate to int-range.
   - **Per-ID popcount sidecar** — cheap `uint32[dict_size]`
     histogram maintained at ingest. Gives selectivity estimates to
     the micro-adaptive dispatcher for free, and is a trivial
     pre-check for "does this value exist in this batch at all"
     (the miss-accelerator QuestDB does *not* have on SYMBOL).

4. **Defer FSST until a concrete workload demands it.** The bolt::dataflow
   tick-trading target doesn't hit FSST-shaped data on the hot path.
   Log/URL analytics or tenant-keyed OLAP would; revisit when
   marbledb or chukonu surfaces such a column.

## What we adopt / skip / followups

- **Adopt:** global dict, lock-free append-on-new, per-ID popcount
  sidecar, literal-resolve-once in filter plan.
- **Skip (for now):** FSST implementation, NOCACHE-style
  disk-backed dict (Bolt keeps everything in arena).
- **Followups:**
  - Design-log entry recording "Dictionary+BitmapIndex == SYMBOL;
    FSST deferred" with this doc as the rationale link.
  - When FSST eventually lands, keep Dictionary as default and ship
    FSST behind `BOLT_ENABLE_FSST` per the multiple-impls-one-default
    rule.

## Sources

- [QuestDB SYMBOL concept doc](https://questdb.com/docs/concepts/symbol/)
- [QuestDB indexes deep-dive](https://questdb.com/docs/concepts/deep-dive/indexes/)
- [QuestDB storage engine overview](https://questdb.com/docs/architecture/storage-engine/)
- [QuestDB schema design essentials](https://questdb.com/docs/schema-design-essentials/)
- [QuestDB issue #6246 — symbol cardinality perf degradation](https://github.com/questdb/questdb/issues/6246)
- [QuestDB issue #4775 — non-exact symbol filters skip index](https://github.com/questdb/questdb/issues/4775)
- [QuestDB issue #2168 — index block capacity and large symbol tables](https://github.com/questdb/questdb/issues/2168)
- [QuestDB Parquet tasks #4738](https://github.com/questdb/questdb/issues/4738)
- [QuestDB Parquet TODO #6694](https://github.com/questdb/questdb/issues/6694)
- [QuestDB 9.0.1 release notes — indexed symbol speedup](https://github.com/questdb/questdb/releases/tag/9.0.1)
- [QuestDB 9.3.4 — Parquet Bloom filters](https://questdb.com/blog/questdb-9-3-4-and-enterprise-3-2-4-release/)
- [Boncz, Neumann, Leis — FSST: Fast Random Access String Compression, VLDB 2020 (PDF)](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf)
- [cwida/fsst reference implementation](https://github.com/cwida/fsst)
- [dev@parquet — FSST encoding proposal thread](http://www.mail-archive.com/dev@parquet.apache.org/msg26813.html)
