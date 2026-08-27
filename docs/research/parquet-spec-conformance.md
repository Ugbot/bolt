# Parquet spec conformance: the specification of the work

Target, in order: **conform to `parquet-format` first, then be as fast as
possible under the usual bolt rules.** Correctness is a merge gate; speed is
measured after, on a quiet box, against a recorded baseline.

Companion document: `parquet-gap-analysis.md` surveys bolt against Arrow and
DuckDB *as implementations*. This one is against the **specification**, which
is a different and larger set — the spec mandates reader behaviours no
comparison to another implementation will surface, and three of the highest
-value items below fall in exactly that category.

Normative sources (`apache/parquet-format`): `parquet.thrift`,
`LogicalTypes.md`, `Encodings.md`, `Compression.md`, `PageIndex.md`,
`BloomFilter.md`, `Encryption.md`.

**Tracker:** `G2PQ-7` (epic) at http://127.0.0.1:8765/tracker, broken into
A/B/C/D/E stories matching the sections below. The three silent-misread items
(A1, B1, B2) and the highest value-per-effort item (A3) carry priority 1;
E depends on A and B so performance cannot be measured against a moving
correctness target.

## The rule this is measured by

A conformant reader **opens every legal file**. A conformant writer **emits
only legal files and declares exactly what it emitted**. Everything else is
a performance question.

Two failure modes are called out separately throughout, because they need
different evidence:

* **Refusal** — the file does not open. Loud, findable, safe.
* **Silent misread** — the file opens and the values are wrong. This is what
  the spec's compatibility rules and ordering semantics protect against, and
  it is why several items below cannot be verified by round-tripping through
  bolt.

## A. Reader conformance — open every legal file

### A1. Legacy 2-level LIST encodings *(spec-mandated, currently missing)*

`LogicalTypes.md` defines **backward-compatibility rules** for repeated
fields: a `repeated` field that is not inside a 3-level `LIST` group is
itself the list, and older writers (Hive, Impala, pre-1.0 parquet-mr, Thrift)
emit exactly that. bolt's schema walk assumes the modern 3-level shape.

This is the single most under-appreciated gap: such files are common in older
lakes, and the failure mode is **not** a clean refusal — a 2-level list read
as if it were 3-level misassigns nesting.

The spec's rules are precise and must be implemented as written, not
approximated:
- a `repeated` field with no `LIST` annotation is a list of its own type;
- inside a `LIST` group, if the repeated child is a group with exactly one
  field, that field is the element; otherwise the repeated group *is* the
  element;
- element naming (`array`, `<name>_tuple`) disambiguates the ambiguous cases.

### A2. Nested repetition (`max_rep >= 2`)

`list<list<T>>`, `list<struct<...>>`, `map<K, list<V>>`. Refused today.
Level *decoding* is already depth-agnostic; the missing piece is assembly —
one offset array per repetition level.

### A3. `parquet_read_file` rejects any file containing a repeated column

Not a decoder gap: `parquet_read_list_column` handles single-level lists
correctly, but the primary entry point refuses `max_rep != 0`, so a file with
one `list<int64>` cannot be opened by the obvious call. Largest
value-per-effort item in the whole document.

### A4. FIXED_LEN_BYTE_ARRAY beyond DECIMAL

Blocks `UUID` (FLBA 16), `FLOAT16` (FLBA 2), `INTERVAL` (FLBA 12), and raw
fixed-width binary. `plain_flba` already exists; this is type mapping.

### A5. Remaining logical types

`ENUM` (BYTE_ARRAY), `UNKNOWN`/null, `INTERVAL`, and full `VARIANT`
(currently shape-only). Each is an annotation over storage bolt already
decodes.

### A6. Codecs: legacy LZ4 framing, BROTLI

bolt reads `LZ4_RAW` but not the Hadoop-framed `LZ4` that older parquet-mr
emits, and not `BROTLI` at all. Per the standing rule that produced
`bolt_zstd_dec.cpp` and `bolt_inflate.h` — *a reader that needs a
`find_package` to open a real table is not a reader* — BROTLI needs an owned
decoder. LZO is explicitly out (see Non-goals).

### A7. Page CRC32 verification

The reader **skips** `PageHeader.crc`. The spec defines it as CRC32 of the
*compressed* page data. Verifying it converts silent corruption into a clean
error. `crc32_ieee` already exists in `bolt_deflate.cpp`.

### A8. Column chunks in a separate file (`ColumnChunk.file_path`)

Legal per the thrift and used by some producers. Currently unhandled; must
either be supported or refused explicitly rather than misread as an offset
into the current file.

## B. Writer conformance — emit only legal files, and declare them

### B1. Statistics ordering semantics *(silent-misread risk)*

The spec defines, per type, which comparison governs `min`/`max`:
signed for the integer types, **unsigned** for `BYTE_ARRAY`/FLBA,
`TypeDefinedOrder` recorded in `FileMetaData.column_orders`. The deprecated
`min`/`max` fields and the modern `min_value`/`max_value` differ precisely in
this ordering, which is why both exist.

Get this wrong and every reader that trusts the statistic skips real matches.
bolt writes statistics and a `ColumnOrder`; the ordering must be **proven per
type** against the spec, and the deprecated fields either written with the
correct legacy semantics or omitted.

### B2. Float/double statistics: NaN and signed zero *(silent-misread risk)*

The spec is explicit: NaN must never be written as a min or max; if all
values are NaN the statistic is omitted. `-0.0` must compare equal to `+0.0`,
and a writer should store `-0.0` as a min and `+0.0` as a max to stay
conservative. A reader must treat a missing statistic as "unknown", never as
"empty range".

### B3. `key_value_metadata`

Not written at all. Carries `ARROW:schema` (how Arrow restores type detail
bolt would otherwise drop) and Iceberg/Delta field IDs.

### B4. `EncodingStats` / `PageEncodingStats`

`ColumnMetaData.encoding_stats` lets a reader know how many pages use each
encoding without walking them. bolt declares `encodings` but not the stats.

### B5. `SizeStatistics` and level histograms

`unencoded_byte_array_data_bytes` plus repetition/definition level
histograms, in both `ColumnMetaData` and `ColumnIndex`. Lets a reader size
buffers before decoding.

### B6. `SortingColumn`

`RowGroup.sorting_columns` — declares sortedness so a consumer can skip a
sort. Only legal to write when actually sorted, so it needs a verified path.

### B7. Nested writing: MAP, STRUCT, nested LIST

Follows A2: level *generation* is the inverse of assembly.

### B8. RowGroup completeness

`ordinal`, `file_offset`, `total_compressed_size` — declared fields bolt
should populate for readers that use them for planning.

## C. Encryption (spec section of its own)

`AES_GCM_V1` and `AES_GCM_CTR_V1`, footer and per-column keys, footer
signing, `crypto_metadata` / `encrypted_column_metadata`. `bolt::crypto`
already has AES primitives. Gated on demand — no encrypted table exists in
this tree — but it *is* part of the spec, so it is listed rather than
quietly dropped.

## D. Geospatial

`GEOMETRY` / `GEOGRAPHY` logical types and `GeospatialStatistics`
(parquet-format 2.11). Lowest priority: DuckDB does not implement them
either, so there is no consumer here.

## E. Performance — after conformance, under the usual bolt rules

Sequenced deliberately after A–B: every item below changes how bytes are
produced or consumed, so a correctness regression introduced here would be
indistinguishable from one introduced by a feature.

* **E1. Parallel column decode.** The writer has `encode_pool` (measured
  1.34×/2.08×/3.33× at 2/4/8 threads); the reader is serial. Arrow and DuckDB
  both decode columns in parallel. `parquet_read_row_group_cols` already
  isolates per-column decode, so this is a fan-out over a borrowed
  `bolt::Scheduler`.
* **E2. `chunk_write_dict` index encoding.** Measured: dictionary work is
  ~24 ms of an ~87 ms encode, and it belongs to the columns that genuinely
  build a dictionary — 400,000 RLE-encoded index values per column — not to
  the ones that abandon one. This is the identified hot spot, not a guess.
* **E3. Page-level skipping using the ColumnIndex on read.** bolt writes and
  parses the index but does not use it to skip pages during a scan.
* **E4. Predicate pushdown into the bloom filter on read.**

### Rules for every performance item

1. **No claim without a quiet box.** A measured artifact of this exact class:
   writing benchmark output into a *fresh* directory costs ~17 ms more than
   overwriting existing files — enough to fabricate a large "improvement"
   when before/after differ in that respect.
2. **Interleaved A/B, min-of-N, both directions.**
3. **Value-level regression gate before any perf claim**, since every item
   here changes what decides a row.
4. **A measured negative is a deliverable.** Record it in the header where
   the code would have gone, with the numbers, so it is not re-attempted.

## Non-goals, with reasons

* **BIT_PACKED** — deprecated by the spec in favour of RLE. Add only if a
  real file demands it.
* **LZO** — DuckDB refuses it, Arrow's support is partial, licensing is why.
  Not worth an owned decoder.
* **INT96 writing** — deprecated. bolt already *reads* it (→ µs), which is
  the direction that matters.
* **`INDEX_PAGE`** — defined in the thrift, never produced by any writer.
  Refuse explicitly.

## Verification standard

Inherited from what this tree already enforces, and non-negotiable for the
items above:

* **Assert VALUES, not row counts.** A level bug, a mis-ordered statistic and
  a dropped column all preserve row counts.
* **The oracle must be independent.** bolt reading what bolt wrote proves the
  pair agree, not that the file is right. Established oracles here: pyarrow
  against a model re-derived in python, DuckDB's own split-block bloom
  implementation, a from-spec thrift decoder, libzstd / liblz4 / python-zlib.
* **Every gate injection-tests itself.** A gate that cannot fail is
  indistinguishable from one that passed.
* **Sweep the parameter space**, and fuzz anything that turns file bytes into
  indices, under ASAN+UBSAN.
