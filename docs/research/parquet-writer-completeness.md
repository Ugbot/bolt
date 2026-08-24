# Parquet writer completeness: what bolt could read but not write

Written 2026-08-24, immediately after
[`parquet-reader-completeness-plan.md`](parquet-reader-completeness-plan.md)
took the READER to eight of nine gaps closed. Doing that inventory made the
opposite asymmetry obvious: bolt could read files it was structurally unable
to produce.

The reader note's framing applies here unchanged — "a reader that cannot open
a file the ecosystem routinely produces is not slow, it is absent" — and the
writer's version is that a file bolt writes should not be one the ecosystem
treats as unusual. Before this work bolt emitted PLAIN values only, in
exactly ONE data page per column chunk, with no page index and no bloom
filter. Arrow and parquet-mr both default to dictionary-encoded ~1 MiB pages.

## What was missing, and what it cost

| Gap | Before | Now |
|---|---|---|
| Dictionary (RLE_DICTIONARY) | absent — Arrow's DEFAULT | writes DICTIONARY_PAGE + RLE_DICTIONARY, per-chunk fallback to PLAIN |
| Data page splitting | ONE page per chunk | budgeted, default 1 MiB, clamped [4 KiB, 512 MiB] |
| Page index (ColumnIndex/OffsetIndex) | option accepted and ignored | real, with a PROVEN boundary_order |
| Bloom filter | option accepted and ignored | split-block SBBF, parquet-mr sizing |
| BYTE_STREAM_SPLIT | readable, not writable | writes; Decimal128 reversed to BE first |
| DELTA_BINARY_PACKED | readable, not writable | writes; wraps at the int64 range |
| DELTA_LENGTH_BYTE_ARRAY | readable, not writable | writes |
| DELTA_BYTE_ARRAY | readable, not writable | writes; front coding |
| Parallel encoding | none | per-column, on a borrowed pool |
| DATA_PAGE_V2 | not written | still not written (v1 only) |
| GZIP / ZSTD / LZ4 | not written | still not written — see below |

## One of these was a corruption bug, not a size choice

`PageHeader.uncompressed_page_size` and `compressed_page_size` are int32. The
old writer cast the whole-chunk page size straight into them, so a column
chunk past 2 GiB wrote a silently wrong header — a corrupt file, produced
without any error. That is the reason page splitting has no "unlimited"
setting: 0 selects the default rather than disabling the budget, and
`emit_page` refuses to truncate rather than casting.

## Decisions worth not re-litigating

**Dictionary fallback is per CHUNK, never mid-chunk.** A chunk whose data
pages disagree about their encoding is legal parquet and a well-known source
of reader bugs. We decline to produce one; if the dictionary exceeds its
ceiling the whole chunk is rewritten PLAIN, at the cost of the abandoned
build. Nothing has been written to the file at that point, so the cost is
bounded and local.

**BOOLEAN is never dictionary encoded**, and never gets a bloom filter. A
two-entry dictionary plus an index stream is strictly larger than one bit per
value, and a two-valued domain makes a filter pointless. parquet-mr and Arrow
both skip it.

**boundary_order is proven, not guessed.** Statistics are stored
little-endian, so ordering them lexicographically would mislabel every
numeric column — and a wrong ASCENDING makes a reader's binary search skip
pages that DO match, which is silently dropped rows rather than a slow query.
The comparison dispatches on the real type and falls back to UNORDERED the
moment anything is ambiguous.

**The RLE hybrid only opens a run on an 8-value boundary.** A bit-packed run
carries whole groups of 8, so cutting elsewhere needs a short group, and
padding one injects values that were never in the input. Padding at the very
END is safe and standard because the reader is bounded by num_values. The
cost is at most 7 values of a repeat falling into the packed buffer instead
of a run.

**DELTA arithmetic is unsigned.** `v[i] - v[i-1]` is not representable as an
int64 for a sequence crossing the range; the encoder subtracts in uint64 so
it wraps two's-complement and the decoder's `cur += min_delta + packed` wraps
identically. Signed arithmetic here would be undefined behaviour, not merely
wrong. parquet-mr relies on the same property.

**A partial DELTA block still writes the full bit-width array.** The reader
consumes `miniblocks_per_block` width bytes unconditionally but only reads
data for the miniblocks it needs — and DELTA_LENGTH_BYTE_ARRAY and
DELTA_BYTE_ARRAY locate their byte payload by how many bytes the length block
consumed. One miniblock too many or too few silently relocates every string.

**Bloom filters are flushed after each row group**, parquet-mr's
AFTER_ROWGROUP default, so live filter memory is one row group's worth rather
than the file's. Their bytes deliberately do NOT count toward any chunk's
`total_compressed_size`: that field bounds the chunk's PAGE region, and a
reader walking pages past its end fails.

**The encode pool is borrowed, never owned.** `bolt::Scheduler` has no
destructor that joins its workers, so a library that quietly creates one owns
a teardown hazard it cannot discharge. `ParquetWriteOpts::encode_pool` takes
a pointer; nullptr is serial. Columns are encoded in waves sized to the pool,
so peak scratch is bounded by WORKER count, not column count — a 256-column
schema does not hold 256 dictionaries.

## Verification standard

The reader plan's four rules, applied unchanged, because every item above is
a silent-wrong-data risk rather than a crash:

1. **Fixtures read by a REFERENCE implementation.** pyarrow reads eleven
   fixtures and agrees on every value, on which chunks are really dictionary
   encoded, on the declared encoding of each delta/BSS column, and on where
   the page-index locators are. A bolt-to-bolt round trip proves only that
   the writer and reader share a reading of the spec — which is exactly the
   failure being guarded against.
2. **Assert VALUES.** A page cut in the wrong place, a dictionary index off
   by one, or a transpose off by one stride all produce exactly the right
   number of rows.
3. **Sweep the parameter space.** Dictionary index bit widths 0/1/2/4/5/8/9;
   delta bit widths 0..62 by stride, crossing every byte boundary and the
   56-bit point where the packer changes strategy; value counts at 1, 2, 8,
   31, 32, 33, 127, 128, 129, 1000, 4097 around the 128-value block and
   32-value miniblock; page cuts landing between groups of 8; nulls
   straddling page boundaries; all-null pages; empty strings; a value that is
   a strict prefix of its predecessor; the 12-byte inline/spill boundary;
   int64 delta wraparound.
4. **Prove the gate discriminates.** Every suite has a DiscriminatingPower
   test that perturbs one value, one string or one null and confirms the same
   comparison the other tests rely on reports that exact row. The pyarrow
   script is injection-tested the same way — a wrong value, a wrong encoding
   name and a wrong index expectation must each fail it.

Two properties get special treatment because passing them is not enough:

- **A bloom filter needs BOTH directions.** No false negatives is the
  correctness property; a bounded false-positive rate is the usefulness one.
  A filter with every bit set has no false negatives either and prunes
  nothing.
- **Parallel encoding is held to BYTE-IDENTITY with a serial write**, at
  1/2/3/4/8 threads and across 12 repeats — not "it still parses" and not
  "the values still round-trip", both of which pass while a race reorders
  chunks. Identity is achievable by construction because placement order does
  not depend on completion order. The parallel path was separately confirmed
  to actually RUN (8 distinct encode thread ids across 87 waves), because
  otherwise the whole suite would pass just as well if dispatch had silently
  done nothing.

## Found while building this

**An "interop fixture for external readers" contained non-deterministic
garbage.** `build_stats_batch` called `alloc_columns(2)`, and
`InteropFixtureForExternalReaders` then filled in columns 2..4 — writing
three `BoltColumn`s past the end of the array, over the arena allocations
that immediately followed, which were that same function's StringView and
spill buffers. The fixture's id column read 6169264384 where 1002 was
written, and three consecutive runs produced three different files. It
survived because the test asserts round-trip consistency and bolt's reader
faithfully returned the corrupted values it had been handed. Only a
byte-identity baseline surfaced it.

**A dropped pool task could have placed a stale chunk.** Output slots are
reused across waves and `submit_range` drops a task silently when its payload
pool is exhausted; a slot still carrying the previous wave's `ok = true`
would then be written as that column's chunk. Slots are invalidated before
dispatch, so a dropped task is a clean write failure instead of wrong bytes.

## Still open

**DATA_PAGE_V2 is not written.** bolt READS it. Writing it is mechanical —
levels move out of the compressed body with explicit byte lengths, plus an
`is_compressed` flag and a null count — and buys interoperability with
nothing that cannot already read v1, which is why it sits below the items
above rather than being skipped for a hard reason.

**GZIP / ZSTD / LZ4_RAW are not written, and this one is not merely
unfinished.** bolt has no COMPRESSOR for any of them that does not need a
`find_package`: `bolt_gzip.h`, `bolt_lz4.h` and `bolt_zstd.h` are all
opt-in shims, and none is available on a default build. bolt owns a
self-contained zstd DECODER (`bolt_zstd_dec.cpp`) precisely because the same
problem blocked READING pyiceberg's default output; the writing side would
need the same treatment — a self-contained DEFLATE for GZIP, and a zstd
compressor — which is a much larger piece of work than a codec dispatch. The
honest statement is that bolt writes UNCOMPRESSED and SNAPPY because those
are the two it can compress without a dependency, not because the others were
overlooked.

**Page-level pruning on the READ side is a consumer step, not a bolt gap.**
`bolt_parquet_pageindex.h` and `bolt_parquet_bloom.h` were described in the
reader plan as "BUILT, NOT WIRED", and that framing is worth correcting:
chukonu's `parquet_scan_op.cpp` already calls `pq_bloom_may_contain` and
`pq_stat_range_i64` to skip whole chunks. What no consumer does yet is
PAGE-level skipping via `pq_read_column_index` — and until this work there
was no way to write a page index at all, so the path could not be exercised
end to end from within this tree. It can now.

The one thing that WOULD need a bolt change to make page skipping general:
`parquet_read_col_chunk_pages` (the resumable page-range decoder a consumer
would jump with) is documented PLAIN-only and returns false on a dictionary
page. Since dictionary encoding is now the writer's recommended default, that
is the next thing to close.

**List/map values remain the one readability gap**, for the reason the reader
plan records: the blocker is the COLUMN SHAPE, not the decoding, and it is an
API decision about `bolt_column.h` — a public type chukonu and marbledb both
consume — that should be taken deliberately rather than improvised.
