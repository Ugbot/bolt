# Bolt VarBinary Column Format (Layer 1.1)

*Plan: `this-was-a-freach-hashed-crab.md` — substrate Layer 1.1.*
*Status: shipped — `Format::VarBinary` constructor + accessor + byte_size; tests green (6/6).*

## Problem

`BoltColumn` had 8 formats — Flat, Constant, Dictionary, Sequence, View, RLE,
BitPacked, FrameOfRef. **All eight assumed fixed-width `type_size_bytes`.**
The two declared-but-unsupported types `BoltType::Utf8` and `BoltType::Binary`
were rejected in every dispatch (wire serialise, wire stream, SST writer,
zone-map). `StringView` (16-byte POD with 4-byte prefix) was defined in
`bolt_types.h` but not plumbed end-to-end.

This was the foundation hole — without a variable-width substrate, no JSONB,
no Utf8 column, no JSON path-shredding, no FST term dictionary could land.

## Decision

Add `ColumnFormat::VarBinary = 8` with the Arrow variable-binary shape:

| field        | role                                                              |
| ------------ | ----------------------------------------------------------------- |
| `data`       | flat byte buffer — concatenated payloads in row order             |
| `dict_child` | a `Format::Flat` Int32 column of length+1 offsets                 |
| `validity`   | optional Arrow null bitmap                                        |
| `length`     | row count (number of payloads = `dict_child.length - 1`)          |
| `type`       | `BoltType::Utf8` or `BoltType::Binary` (or future JSONB sub-type) |

Row `i` spans bytes `[offsets[i], offsets[i+1])`. Total payload bytes =
`offsets[length]`. Validity follows Arrow convention (bit 1 = valid).

We reuse the `dict_child` slot for the offsets column rather than carrying a
fourth pointer on `BoltColumn` — keeps the struct size unchanged and makes
the offsets column a first-class `BoltColumn` that travels through the same
clone / wire / SST plumbing as everything else.

## Hot-path accessor

```cpp
BOLT_FORCE_INLINE void var_binary_at(int64_t row,
                                      const uint8_t** out_data,
                                      int32_t* out_len) const noexcept {
    const int32_t* offs = static_cast<const int32_t*>(dict_child->data);
    const int32_t  start = offs[row];
    const int32_t  end   = offs[row + 1];
    *out_data = static_cast<const uint8_t*>(data) + start;
    *out_len  = end - start;
}
```

Two loads from `offs`, one pointer add, one subtract — no branches, no
allocations, fits in 4 instructions. Designed for the hot path of the
JSON parser tape walk and JSONB path-lookup.

## What we kept

- **Arrow's three-buffer layout** (validity + offsets + data). Lets us export
  a Bolt VarBinary column to Arrow consumers via the existing C Data
  Interface plumbing without reshaping anything. Confirmed against the
  Arrow Format spec (offsets length = N+1; nulls allowed; offsets monotonic).
- **Int32 offsets** (not Int64). 2 GB per granule cap is generous — anything
  beyond that should be a new granule, not a single column. Matches Arrow
  `Utf8` / `Binary`; the `LargeUtf8` / `LargeBinary` (Int64) variant is
  deferred until a workload wants it.

## What we rejected

- **In-place mutation.** Like all Bolt columns, `VarBinary` is build-once,
  read-many. Mutations route through the existing tick-tock COW pattern at
  the BoltBatch level.
- **A second pointer field on `BoltColumn` for offsets.** Would have been
  cleaner conceptually but bloats every column header (Flat, Constant,
  Dictionary, ...) for the benefit of one variant. Reusing `dict_child`
  is the minimal-surface choice and keeps `sizeof(BoltColumn)` unchanged.
- **Inline-vs-spilled `StringView` at the column level.** `StringView` is
  the per-row in-memory view; the on-column storage is the offsets+data pair.
  No need to mix the two. (A future Arrow-Utf8View bridge could materialise
  `StringView`s into a side buffer if a consumer asks.)
- **Variable-width entries inside the existing fixed-width formats** (RLE
  with strings, Dictionary with var-width values). Both are straightforward
  composites once `VarBinary` exists; deferred until a consumer needs them.

## Tests

`tests/test_bolt_var_column.cpp` — 6 tests, all passing:

- `EmptyColumn` — n=0 returns a well-formed `Format::VarBinary` with
  `byte_size() == 0`.
- `InlineRowsAreReadable` — 5 rows of mixed widths (`alpha`, `bravo`, `c`,
  `deltaecho`, ``); accessor returns the right `(ptr, len)` for each.
- `ZeroLengthRowBetweenNonEmpty` — empty payload between two non-empty
  rows; offset equality is the empty-row signal (no separate flag needed).
- `ByteSizeAccountsForBuffers` — `byte_size()` includes payload + offsets
  + validity bitmap (when present).
- `RejectsNullOffsetsForNonEmptyColumn` — passing `offsets=nullptr` with
  `total_rows > 0` falls back to `make_empty()` (assertion-friendly,
  doesn't crash on bad input).
- `NonContiguousAccessIsCorrect` — random-order accesses match sequential.

## What remains for full v2 wire-format integration

Layer 1.1 landed the in-memory shape. The wire format upgrade and SST
plumbing were in scope on the plan but are deferred to dedicated sub-waves
to keep this PR small and reviewable:

- `bolt_wire.h` / `bolt_wire_stream.h` — bump format version 1 → 2; emit
  three-buffer-per-column layout when `format == VarBinary`.
- `marbledb/internal/sstable_format.h` — `is_variable_width` flag on per-
  column metadata; SST writer emits the offsets buffer alongside the data
  buffer.

These don't block Layers 1.2 / 1.2b / 1.3 (parse/variant/string-kernels
all run on in-memory columns). They block Layer 3 (`Field::Json`
on-disk format) — schedule alongside that wave.

## References

- Arrow columnar format spec — variable-size binary layout
- The original `external/bolt/include/bolt/bolt_column.h` 8-format taxonomy
- Postgres JSONB Jentry header (informs future on-disk variant; see
  `jsonb-binary-format.md` when Layer 1.4 ships)
