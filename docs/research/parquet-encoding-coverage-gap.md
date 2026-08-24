# The parquet reader rejects three encodings modern writers emit by default

Found 2026-08-24 by comparing bolt's reader against [hardwood](https://hardwood.dev),
a Java parquet parser, while borrowing its bit-unpacking technique.

## The gap

`decode_data_page` handles four encodings and ends with
`return false;  // encoding outside v1`:

```
kEncPlain = 0, kEncPlainDict = 2, kEncRle = 3, kEncRleDict = 8
```

hardwood ships decoders for those **plus** `DeltaBinaryPacked`,
`DeltaByteArray`, `DeltaLengthByteArray` and `ByteStreamSplit`.

MEASURED, not inferred from the constant list. Three files written with pyarrow,
200,000 rows each, read back through `bench_parquet_decode`:

| written with | bolt |
|---|---|
| `PLAIN` / `RLE_DICTIONARY` | reads, 5.2 ms |
| `DELTA_BINARY_PACKED` (int64) | **row group 0 failed** |
| `DELTA_BYTE_ARRAY` (string) | **row group 0 failed** |
| `BYTE_STREAM_SPLIT` (double) | **row group 0 failed** |

## Why it matters more than another few percent of decode

This is the same shape as the zstd finding already recorded in PROJECT_MAP —
"zstd is pyiceberg's DEFAULT parquet codec and bolt's reader rejected it
outright, so the most common real Iceberg table on earth did not read". A reader
that cannot open a file the ecosystem routinely produces is not slow, it is
absent.

Who emits these:

- **`DELTA_BINARY_PACKED`** is the integer encoding of the parquet **V2** writer.
  Spark with `parquet.writer.version=v2`, and parquet-java's v2 path, produce it
  for INT32/INT64 as a matter of course.
- **`BYTE_STREAM_SPLIT`** is increasingly the default for FLOAT/DOUBLE — it is
  the recommended encoding for floating-point columns and is what
  scientific/numeric datasets are commonly written with. pyarrow and DuckDB both
  emit it on request and read it always.
- **`DELTA_BYTE_ARRAY`** is the usual choice for sorted or
  common-prefix string columns.

TPC-H and ClickBench happen to be written PLAIN/dictionary, which is why every
benchmark in this tree passes and the gap has stayed invisible.

## Note on the failure mode

It fails closed — `decode_data_page` returns false and the row group fails
loudly, so no wrong answers. That is the right behaviour for an unsupported
encoding, and it is why this is a capability gap rather than a correctness bug.

## Shape of the fix

Each is a self-contained page decoder behind the existing dispatch, in the same
place `kEncPlain` and `kEncRleDict` are handled:

- `DELTA_BINARY_PACKED` — miniblock/blocks of deltas with a per-miniblock bit
  width. Reuses the group-of-8 unpack lane already in `unpack_le_bounded`.
- `DELTA_LENGTH_BYTE_ARRAY` — a DELTA_BINARY_PACKED length block followed by
  concatenated bytes. Trivial once the above exists.
- `DELTA_BYTE_ARRAY` — prefix lengths + suffix lengths (both delta-packed) plus
  suffix bytes; each value is built from the previous one's prefix.
- `BYTE_STREAM_SPLIT` — a pure transpose: for a width-W type, all byte 0s, then
  all byte 1s, and so on. The decoder is a strided gather and vectorises well.

Verification should follow the pattern used for the group-of-8 unpack: write the
fixtures with pyarrow at each encoding, and assert VALUES against
`table.to_pydict()` rather than row counts — a transpose that is off by one
stride still produces the right number of rows.
