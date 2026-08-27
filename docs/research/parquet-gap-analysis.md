# Parquet: what bolt still lacks, against Arrow, DuckDB and the spec

Status of `bolt::ingest::parquet` as of 2026-08-27 (bolt `784c963`), and what
is left to be "complete".

## How this was determined

Every row below is either read out of bolt's source, or **measured** against a
reference on this machine (pyarrow 21.0.0, DuckDB 1.4.5). Where a claim is
about a reference implementation rather than something observed, it says so.
This matters because two claims that "looked obvious" turned out false when
tested, and both are recorded here:

* **DuckDB does not consult the ColumnIndex.** Flipping every byte of a bolt
  file's 2288-byte page-index region leaves every range-query answer
  unchanged. Any gap analysis that assumes DuckDB exercises the page index is
  wrong, and so is any gate built on it.
* **DuckDB writes LZ4 and BROTLI; it refuses LZO.** Not assumed from docs --
  each codec was actually written to disk.
* **bolt reads STRUCT columns fine.** A grep says otherwise, and a grep is
  wrong: a struct's leaves have `max_rep == 0`, so they take the ordinary flat
  path and arrive as `c.a`, `c.b`. It is flattening, not refusal.

Every "bolt cannot open this" claim below was checked by writing the file with
pyarrow and handing it to `parquet_read_file`, not by reading bolt's source.

## The matrix

`R` = read, `W` = write, `--` = absent, `(x)` = present with the caveat noted.

| Feature | bolt R | bolt W | Arrow C++ | DuckDB |
|---|---|---|---|---|
| **Physical types** ||||
| BOOLEAN / INT32 / INT64 / FLOAT / DOUBLE / BYTE_ARRAY | R | W | R/W | R/W |
| INT96 (legacy timestamp) | R → µs | -- | R/W | R |
| FIXED_LEN_BYTE_ARRAY | (R) DECIMAL only | -- | R/W | R/W |
| **Logical types** ||||
| STRING, DATE, TIME, TIMESTAMP, INTEGER | R | W | R/W | R/W |
| DECIMAL (INT32/INT64/FLBA ≤16B → Decimal128) | R | W | R/W | R/W |
| JSON / BSON | R | W | R/W | R/W |
| VARIANT | (R) shape only | -- | R/W | R |
| UUID | -- | -- | R/W | R/W |
| FLOAT16 | -- | -- | R/W | R/W |
| ENUM | -- | -- | R/W | R/W |
| INTERVAL | -- | -- | R/W | R |
| GEOMETRY / GEOGRAPHY (+ geo statistics) | -- | -- | R/W | -- |
| **Encodings** ||||
| PLAIN, PLAIN_DICTIONARY, RLE, RLE_DICTIONARY | R | W | R/W | R/W |
| DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY | R | W | R/W | R/W |
| BYTE_STREAM_SPLIT | R | W | R/W | R/W |
| BIT_PACKED (deprecated) | -- | -- | R | R |
| **Compression** ||||
| UNCOMPRESSED, SNAPPY, GZIP, ZSTD, LZ4_RAW | R | W | R/W | R/W |
| LZ4 (legacy Hadoop framing) | -- | -- | R/W | R/W |
| BROTLI | -- | -- | R/W | R/W |
| LZO | -- | -- | (R) | -- |
| **Structure** ||||
| Flat columns, multiple row groups, DATA_PAGE v1 + v2 | R | W | R/W | R/W |
| `list<T>`, map leaves (`max_rep == 1`) | (R) per-column API only | W (list) | R/W | R/W |
| Any list/map column via `parquet_read_file` | -- | n/a | R | R |
| STRUCT (read as flattened leaf columns) | R | -- | R/W (nested) | R/W (nested) |
| Nested lists (`max_rep ≥ 2`) | -- | -- | R/W | R/W |
| MAP as a first-class column | (R) leaves | -- | R/W | R/W |
| **Metadata / indexes** ||||
| Statistics, ColumnIndex + OffsetIndex, bloom filters | R | W | R/W | W-only(*) |
| Page CRC32 checksums | skipped | -- | R/W | R/W |
| `sorting_columns` | -- | -- | R/W | R |
| `key_value_metadata` (incl. `ARROW:schema`) | -- | -- | R/W | R/W |
| `size_statistics` (unencoded sizes, level histograms) | -- | -- | R/W | -- |
| Modular encryption (AES-GCM / AES-GCM-CTR) | -- | -- | R/W | R/W |
| **Engine** ||||
| Projection pushdown, resumable page-range decode | R | n/a | R | R |
| Parallel column encode | n/a | W | W | W |
| Parallel column decode | -- | n/a | R | R |

(*) DuckDB writes a page index and bloom filters, but was **measured** not to
consult the ColumnIndex when reading.

## What is left, ranked

Ranked by what actually breaks if it stays missing, not by effort.

### 1. `parquet_read_file` refuses any file containing a list or map

**Measured**, and the biggest practical gap on the list. `parquet_read_file`
rejects a column chunk whose `max_rep != 0`, so a file with even a *single*
`list<int64>` -- a shape bolt decodes correctly through
`parquet_read_list_column` -- cannot be read through the primary entry point
at all. A mixed flat+list file fails the same way.

So this is not a decoder gap. The decoder is right and tested; the main API
just cannot assemble a batch containing a Nested column. Everything needed is
present: `ColumnFormat::Nested` holds the column, and
`parquet_read_list_column` produces it.

**Effort: small**, and it converts an already-working decoder from
"reachable only if the caller knows to use a different function per column"
into "open the file". Highest value-per-unit-effort item here.

### 2. Nested lists (`max_rep ≥ 2`) and nested-group writing

`list<list<T>>`, `list<struct<...>>`, and a map whose value is a list are
refused -- loudly, since `parquet_read_list_column` returns false rather than
guessing, and a wrong nesting silently reshapes data. Verified against
pyarrow-written files.

A *flat* struct is fine (it flattens to leaf columns), so the practical
blast radius is narrower than it first looks: it is repeated nesting that
fails, not grouping.

The level *decoding* is already depth-agnostic -- `list_page_levels` reads
rep/def at the correct bit widths for any `max_rep`. What is missing is
assembly: one offset array per repetition level, tracking which level each
rep value re-opens. `ColumnFormat::Nested` can already hold a child column, so
the representation exists.

**Effort: large.** Multi-level Dremel is intricate and fails silently. It
needs pyarrow fixtures across the nesting matrix (list of lists, list of
structs, struct of lists, map of structs, nulls at every level, empty vs null
at every level) and an extension of `CorruptLevelsNeverCrash`.

### 3. FIXED_LEN_BYTE_ARRAY beyond DECIMAL

`parquet_map_type` rejects any non-DECIMAL FLBA. That blocks UUID, FLOAT16,
and any raw fixed-width binary column -- and FLBA is not exotic; it is how
UUIDs and small hashes are usually stored.

Verified: pyarrow writes FLOAT16 and `binary(16)`, DuckDB reads both, and
`parquet_read_file` returns false on each.

**Effort: small.** The plain FLBA decoder (`plain_flba`) already exists and is
used for decimals; this is mostly a type-mapping decision (`BoltType::Binary`
with a fixed width, plus `BoltLogical::Uuid`) and the writer side.

### 4. LZ4 (legacy) and BROTLI codecs

bolt reads LZ4_RAW but not the legacy Hadoop-framed LZ4, and not BROTLI at
all. Both are written by real producers: BROTLI by pyarrow and DuckDB on
request, legacy LZ4 by older parquet-mr. A file using either does not open.

**Effort: small for LZ4 legacy** (framing over the existing raw decoder).
**Medium for BROTLI** -- it needs an owned decoder, which is the same argument
that produced `bolt_zstd_dec.cpp` and `bolt_inflate.h`: a reader that needs a
`find_package` to open a real table is not a reader.

### 5. Page CRC32 checksums

The reader **skips** the `crc` field; the writer never emits one. Arrow and
DuckDB both write and verify them. Silent corruption that a checksum would
catch currently reaches the decoder, where it becomes either a decode failure
with a confusing message or, worse, plausible wrong values.

**Effort: small.** CRC32 already exists in-tree (`bolt_deflate.cpp`'s
`crc32_ieee`). Verify on read behind a flag, emit on write.

### 6. `key_value_metadata`

bolt writes none. This is how Arrow round-trips its own schema (`ARROW:schema`)
and how Iceberg/Delta stash field IDs. Without it, a bolt-written file loses
type detail that Arrow would otherwise restore, and cannot carry field IDs.

**Effort: small** (a thrift list of string pairs in the footer), and it
unblocks Iceberg field-ID mapping.

### 7. Parallel column decode

The goal explicitly asks for multithreaded. The **writer** has it
(`ParquetWriteOpts::encode_pool`, measured 1.34×/2.08×/3.33× at 2/4/8
threads). The **reader** does not: `parquet_read_file` decodes column chunks
serially. Arrow and DuckDB both decode columns in parallel.

**Effort: medium.** `parquet_read_row_group_cols` already decodes a chosen
column set independently, which is the hard part; this is a fan-out over a
borrowed `bolt::Scheduler`, mirroring the writer's pattern. Needs a quiet box
to measure -- see the caveat at the end.

### 8. `sorting_columns` and `size_statistics`

Both are pure metadata a reader can exploit: `sorting_columns` lets a consumer
skip a sort; `size_statistics` (unencoded byte counts and rep/def level
histograms) lets it size buffers before decoding. Neither is required for
correctness.

**Effort: small each.** `size_statistics` is newer -- Arrow 18+ writes it,
DuckDB does not.

### 9. MAP and STRUCT *writing*

bolt writes `list<T>` but refuses a List whose element is List/Struct/Map, and
cannot write a MAP or STRUCT column at all. Follows naturally from item 2;
the level *generation* is the inverse of the assembly.

### 10. Modular encryption

AES-GCM / AES-GCM-CTR, footer and per-column keys. Arrow and DuckDB both
support it (DuckDB's was verified writing here). `bolt::crypto` already has
AES-capable primitives.

**Effort: large**, and gated on demand -- nothing in this tree reads or writes
an encrypted table today.

### 11. GEOMETRY / GEOGRAPHY

Parquet 2.11 logical types plus geospatial statistics. Arrow 21 has them;
**DuckDB does not**. Lowest priority: no consumer here wants them, and the
reference set does not agree they matter yet.

## Deliberately not planned

* **BIT_PACKED** -- deprecated by the spec, replaced by RLE. Arrow and DuckDB
  read it for old files only. Add only if a real file demands it.
* **LZO** -- DuckDB refuses it outright, Arrow's support is partial, and the
  license history is why. Not worth an owned decoder.
* **INT96 writing** -- deprecated; bolt already *reads* it (→ µs), which is
  the direction that matters for opening old files.

## Verified NOT to be gaps

Recorded so they are not re-investigated:

* **STRUCT columns read correctly**, flattened to their leaves. Only
  *repeated* nesting is unsupported.
* **All 8 live encodings** are read and written. Only deprecated BIT_PACKED is
  absent.
* **Every codec bolt reads, it also writes**, with no `find_package` for any
  of them.
* **DECIMAL** reaches Decimal128 via FLBA up to 16 bytes (precision ≤ 38).
  The `scale > 18` rejection on INT32/INT64-backed decimals is a *scale*
  bound, not a precision ceiling.
* **The page index, bloom filters and LIST writer are all correct**, now
  established by independent oracles rather than by bolt's own reader: a
  from-spec thrift decoder (2208 pages, every declared bound brackets its
  page), DuckDB's own split-block bloom implementation (225 present values,
  none wrongly excluded), and pyarrow against a python-re-derived model.

## Caveat on anything performance-shaped

Items 6 and any future optimisation need a quiet machine. At the time of
writing this box carries another agent's VM at ~536% CPU plus two test
binaries; wall-clock here is not quotable. A separately measured artifact of
the same class: writing benchmark output into a *fresh* directory costs ~17 ms
more than overwriting existing files, which is enough to fabricate a large
"improvement" if before/after differ in that respect.
