# Parquet reader completeness: everything hardwood reads that bolt does not

> **STATUS 2026-08-25: ALL NINE closed, plus one that was not on the list.**
> List/map values now read via Dremel assembly (see
> [`parquet-writer-completeness.md`](parquet-writer-completeness.md) for the
> column-shape decision that unblocked it), and LZ4_RAW — which this plan did
> not flag — turned out to be unreadable too, since bolt's only LZ4 was behind
> a find_package. Both are fixed. The status below is the 2026-08-24 snapshot.
>
> **STATUS 2026-08-24: eight of nine gaps closed.** Everything below now reads
> except list/map VALUES. Verified byte-exact against pyarrow on 82 fixtures:
> BYTE_STREAM_SPLIT, all three delta encodings, DATA_PAGE_V2, GZIP + LZ4_RAW,
> nested structs (including all three definition levels with nulls), and any
> file that merely CONTAINS a list — its scalar columns read via projection.
>
> The same logical data written four ways decodes to one checksum,
> 373c6b93d4394863, so four independent decode paths agree.
>
> **The one remaining readability gap, and why it stops here.** Materialising
> list/map values needs Dremel record assembly AND a list column shape, and the
> shape is the blocker, not the decoding. bolt has a single general `dict_child`
> slot reused three ways — run-ends, VarBinary BYTE offsets, dictionary
> keys→values. A typed list needs offsets in ELEMENTS plus a typed child column,
> which collides with VarBinary's byte-offset meaning. So it needs either a new
> `ColumnFormat` or a two-level `dict_child` chain in `bolt_column.h` — a public
> type chukonu and marbledb both consume. That is an API decision to take
> deliberately, not a decode detail to improvise.
>
> Sequence when picking it up: settle the representation in `bolt_column.h`
> FIRST; then repetition-level decoding, which is nearly free because
> `unpack_le_bounded` and `level_bit_width()` already generalise and `max_rep`
> is already computed per leaf; then assembly. The fixtures and
> `scripts/parquet_ref_hash.py` already cover the verification.
>
> Also still open, both smaller and neither a readability gap: page-index and
> bloom-filter wiring need a predicate plumbed through the read API, which has
> no predicate parameter today — so they are more than "wiring". BROTLI and LZO
> have no decoder in bolt at all and fail closed.
>
> **UPDATE 2026-08-24, after the WRITER work
> ([`parquet-writer-completeness.md`](parquet-writer-completeness.md)):**
> two corrections to the table below.
>
> Rows 8 and 9 said "BUILT, NOT WIRED" and called them "the cheapest wins in
> the table". Half of that is wrong: chukonu's `parquet_scan_op.cpp` already
> calls `pq_bloom_may_contain` and `pq_stat_range_i64` to skip whole chunks,
> so chunk-level pruning IS wired, just on the consumer side rather than
> inside bolt. What genuinely has no consumer is PAGE-level skipping via
> `pq_read_column_index` — and until now bolt could not WRITE a page index or
> a bloom filter at all, so neither path could be exercised end to end from
> within this tree. Both can now, and both are covered by fixtures.
>
> The remaining bolt-side blocker for general page skipping is narrower than
> "a predicate parameter": `parquet_read_col_chunk_pages`, the resumable
> page-range decoder a consumer would jump with, is documented PLAIN-only and
> returns false on a dictionary page. Dictionary encoding is now the writer's
> recommended default, so that is the next thing to close.

Written 2026-08-24, after a day spent making the reader 1.53x faster and then
discovering it cannot open several kinds of file the ecosystem routinely
produces. Speed on files we can already read is worth less than being able to
read the files at all — that lesson is already in this tree once, as the zstd
finding in PROJECT_MAP ("a reader that needs a find_package to open a real table
is not a reader").

Comparison target: [hardwood](https://hardwood.dev), a Java parquet parser whose
stated goal is "support all Parquet files which are supported by the canonical
parquet-java library".

## What bolt cannot read today

Every row below is a file bolt REJECTS. It fails closed — `decode_data_page`
returns false and the row group fails loudly — so none of this is a correctness
bug. It is absence.

| # | Gap | Status in bolt | Evidence |
|---|---|---|---|
| 1 | `DELTA_BINARY_PACKED` | **DONE** | reads; verified vs pyarrow |
| 2 | `DELTA_BYTE_ARRAY` | **DONE** | reads; needed a growable Utf8 spill buffer |
| 3 | `DELTA_LENGTH_BYTE_ARRAY` | **DONE** | reads |
| 4 | `BYTE_STREAM_SPLIT` | **DONE** | reads; width must come from the PHYSICAL type |
| 5 | `DATA_PAGE_V2` | **DONE** | reads; levels assembled uncompressed ahead of values |
| 6a | Nested STRUCT columns | **DONE** | schema walked depth-first; decoder generalised to max_def |
| 6b | Files CONTAINING a list | **DONE** | the list blocks only itself; scalars read via projection |
| 6c | List/map VALUES | **DONE** | Dremel assembly; ColumnFormat::Nested settled the shape |
| 7 | GZIP / LZ4_RAW | **DONE** | GZIP via bolt's own inflate_raw. LZ4_RAW needed find_package(lz4) and so did NOT read on a default build — now bolt's own block codec. BROTLI/LZO still fail closed |
| 8 | Page index (skip pages by stats) | BUILT, NOT WIRED | `bolt_parquet_pageindex.h` included only by its own .cpp and test |
| 9 | Bloom filter (prune by equality) | BUILT, NOT WIRED | `bolt_parquet_bloom.h` same |

Items 8 and 9 are the cheapest wins in the table: the code exists and is tested,
it is simply not connected to the read path. `PqChunk::column_index_offset` is
parsed and never read.

## Why the benchmarks never caught any of this

TPC-H and ClickBench are both written PLAIN/dictionary, DATA_PAGE v1, SNAPPY,
flat schema, and neither carries a page index or bloom filter. The entire
standing gate is blind to items 1-9 by construction. That is not a criticism of
the benchmarks; it is the reason a capability inventory has to be done against
a reference implementation rather than against our own test data.

## Sequence

Ordered by (files unlocked) / (effort), not by interest.

### Phase 1 — wire up what already exists

**1.1 Page index.** `bolt_parquet_pageindex.h` is complete and tested. Read
`PqChunk::column_index_offset`, and where a scan carries a predicate, skip pages
whose min/max excludes it. Note the honest caveat already recorded in
PROJECT_MAP: neither `lineitem.parquet` nor `hits.parquet` carries an index, so
this fires on ZERO files in this tree — it is capability for real lakehouse
data, and must not be sold as a benchmark win.

**1.2 Bloom filter.** Same shape: parse `bloom_filter_offset`, consult on
equality predicates. Same caveat — no file here has one.

Both are wiring, not new algorithms. Verify with files written by pyarrow with
`write_page_index=True` / `write_bloom_filter=True`, asserting that a predicate
that excludes everything reads zero pages, and that results are unchanged.

### Phase 2 — the encodings (the actual blocker)

**2.1 `BYTE_STREAM_SPLIT`.** Start here: it is a pure transpose and needs no new
concepts. For a width-W type the page holds all byte 0s, then all byte 1s, and
so on; the decoder is a strided gather that vectorises well. Increasingly the
default for FLOAT/DOUBLE.

**2.2 `DELTA_BINARY_PACKED`.** The integer encoding of the parquet V2 writer —
Spark with `writer.version=v2` emits it for every INT32/INT64. Blocks of
miniblocks, each with its own bit width, a per-block minimum delta, and a
zig-zag varint first value. Reuses the group-of-8 unpack lane already in
`unpack_le_bounded`.

**2.3 `DELTA_LENGTH_BYTE_ARRAY`.** A DELTA_BINARY_PACKED length block followed by
concatenated bytes. Nearly free once 2.2 exists.

**2.4 `DELTA_BYTE_ARRAY`.** Prefix lengths and suffix lengths, both
delta-packed, plus suffix bytes; each value is rebuilt from its predecessor's
prefix. The usual choice for sorted string columns.

### Phase 3 — DATA_PAGE_V2

Independent of the encodings and equally blocking: a V2 file fails on BOTH the
page type and the encoding. V2 moves def/rep levels OUT of the compressed body
(they are stored uncompressed with explicit byte lengths) and adds an
`is_compressed` flag and a null count. Mechanically simpler than v1 in places —
the level sections no longer have to be found by decoding.

### Phase 4 — codecs

GZIP and LZ4_RAW are the two that appear in the wild often enough to matter
(LZ4_RAW is Hadoop-era, GZIP is everywhere). bolt already owns a zlib inflate
(`bolt_inflate.h`) and an LZ4 wrapper, so this is dispatch plumbing, with the
same "no find_package required" rule the zstd decoder was written to satisfy.
BROTLI and LZO are rarer; decline explicitly rather than silently.

### Phase 5 — nested columns

The largest by far, and deliberately last. bolt's reader is flat-only by design;
supporting lists and maps means repetition levels and inverse-Dremel record
assembly, which changes the output model (a leaf value no longer corresponds to
a row). Worth doing only when a real workload needs it, and worth designing
against `BoltColumn`'s list representation before any decoding is written.

## Verification standard for every item

The same standard the group-of-8 unpack was held to today, because these are all
silent-wrong-data risks:

1. **Fixtures written by a reference writer**, not by bolt. pyarrow can emit
   every encoding and page version above; bolt's own writer cannot, so a
   round-trip through it proves nothing about interoperability.
2. **Assert VALUES, not row counts.** A transpose off by one stride, or a delta
   block with a wrong miniblock width, still produces exactly the right number
   of rows. Compare against `table.to_pydict()`.
3. **Sweep the parameter space**, not one happy case: bit widths, block and
   miniblock boundaries, first/last value, empty pages, single-row pages, nulls
   interleaved.
4. **Prove the gate discriminates** by injecting a defect and confirming the
   test fails. Twice today a passing suite turned out not to reach the code
   under test.

## What NOT to take from hardwood

Recorded so the comparison is not re-run:

- **SIMD `countNonNulls` / `markNulls`** — bolt beats these by DELETING the work
  (definition levels for a nullable-but-null-free column are no longer
  materialised at all), which is strictly better than vectorising them.
- **Per-type `applyDictionary*`** — bolt's compile-time `W` gather is the same
  idea, already in place.
- **Virtual-thread parallelism** — JVM-specific; chukonu already parallelises
  across row groups.
- **Dictionary string interning** — relies on Java object identity; does not map
  onto `StringView`.

## One idea worth taking that is not a gap

`BatchSizing` computes a batch size so that all projected column arrays fit in
L2 (6 MB target), derived from per-column byte width and list fan-out, rather
than a fixed row count. chukonu uses a fixed 65536-row morsel regardless of
schema width — about right for TPC-H's 16 columns, far past L2 for ClickBench's
105. Worth a probe. Note PROJECT_MAP already records morsel size as "ruled out",
but that was about INCREASING it; shrinking it for wide schemas is a different
question and has not been measured.
