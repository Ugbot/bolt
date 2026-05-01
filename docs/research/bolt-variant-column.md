# Bolt Variant Column

Layer 1.2b of `this-was-a-freach-hashed-crab.md`. Adds a sum-type
column primitive on top of the existing `BoltColumn` so a logical
field can carry one of N typed payloads per row. Used by JSON
ingestion, schema-on-read, and ClickHouse-style `Dynamic` projection.

## Reference

ClickHouse's `Dynamic` type (`src/Columns/ColumnDynamic.h`,
`src/Columns/ColumnVariant.h`). The shape is well-trodden and
fits Bolt's POD-plus-arena discipline.

## What we lifted

- **UInt8 discriminator per row.** One byte selects the variant index;
  reserved values 254 (overflow → `shared_variant`) and 255 (null).
- **Per-variant typed columns.** The `variants` array holds N fully
  typed `BoltColumn`s — Int32, Float32, Utf8, etc. — at full row
  width. Reads are constant-time after the byte load.
- **Shared variant tail.** A single `VarBinary` column carries
  payloads whose type spilled past the 254-variant cap, encoded as
  `(type_id, value)` blobs. The variant column itself only needs to
  surface the discriminator byte; decoding lives in the consumer.
- **Compact-discriminator pass.** Reuses Bolt's existing `Constant`
  format. When every row shares a discriminator, the Flat UInt8
  buffer is replaced by `BoltColumn::make_constant<uint8_t>(...)` —
  zero per-row storage, accessor stays branchless thanks to the
  stride-zero load through `inline_value`.

## What we rejected

- **Per-row encoded type tags inside each variant.** ClickHouse pays
  for that flexibility with serialization branching. We avoid it: a
  variant is a homogeneous typed column, so the type is implicit and
  kernels (filter, group-by, hash) inherit Bolt's existing fast paths.
- **More than 254 typed variants.** ClickHouse has the same cap
  (`MAX_NUMBER_OF_VARIANTS = 255` minus null). Pushing higher would
  cost a UInt16 discriminator and double the per-row tag traffic for
  workloads that never need it. The shared-overflow tail handles the
  long tail.
- **Std-vector-of-variant fallback.** The old MarbleDB sketch held a
  `std::vector<BoltColumn>`. We arena-allocate a flat array instead —
  Tiger Style, no heap.
- **Runtime branch in the hot accessor.** `variant_column_at` does a
  load, a single bounds compare, and a CMOV. Tested on MSVC `/O2`
  via the disassembly of the test binary — no `jcc` on the
  in-range path. The Constant-discriminator case multiplies the row
  index by `0` so the same load services both Flat and Constant
  shapes; the compiler folds the constant stride.

## Layout

```
discriminator   : BoltColumn   // UInt8, Flat or Constant
shared_variant  : BoltColumn   // VarBinary tail (may be empty)
variants        : BoltColumn*  // arena-allocated [variant_count]
length          : int64_t
variant_count   : int32_t
_pad            : uint8_t[4]
```

`sizeof(VariantColumn) == 2 * sizeof(BoltColumn) + 64` (the tail
fields round up to BoltColumn's 64-byte alignment). Static-asserted.

## Open questions

- **Promotion / demotion.** When a JSON column drifts from one variant
  to two, who rewrites the discriminator? The current API is
  build-once; an `append` path that grows the discriminator and
  back-fills overflow rows is future work.
- **shared_variant decode.** We expose the discriminator byte and
  leave `(type_id, value)` decoding to the caller. A small typed
  reader helper (probably `(BoltType, StringView)`-shaped) likely
  belongs next to the parse layer, not in `bolt_column`.
- **Vectorised gather.** `variant_column_at` is row-at-a-time. A
  block kernel that walks 64 rows and emits `(variant_idx[64],
  out_index[64])` with SIMD would let downstream operators dispatch
  per-variant in cache-friendly stripes — Layer 1.3 candidate.
- **Stats roll-up.** The variant column has no aggregate `ColumnStats`
  yet. The discriminator's distinct-count is a useful selectivity
  proxy and should land alongside the compact pass.
