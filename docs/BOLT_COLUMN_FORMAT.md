# Bolt Column Format: Beyond Arrow

## Design Document — Adaptive Column Format with Sidecar Indexes

**Extends:** `docs/BOLT_DESIGN.md` (Phase 1: Arena + Channel primitives)
**Status:** Design — ready for implementation discussion
**Date:** April 2026

---

## 1. The Problem with Arrow Compatibility

Arrow is a *wire format* and an *interop specification*. It's excellent at being a
lingua franca between systems. But treating it as an execution format forces several
compromises:

1. **No column statistics at the execution level.** Arrow has column-level metadata
   in IPC (min/max in RecordBatch stats), but the in-memory Array carries no stats.
   Every filter/join must scan blind — there's no zone map, no bloom filter, no
   cardinality estimate attached to the data as it flows through the pipeline.

2. **No sidecar indexes.** Arrow columns are flat buffers. There's no way to attach
   a B-tree, bitmap index, or hash index to a column in-flight. DuckDB builds its
   own zone maps per 2048-element vector. Parquet has row group statistics. But
   Arrow-the-format has nothing between "scan everything" and "build a separate
   index data structure."

3. **Fixed string representation.** Even with StringView (German-style), Arrow's
   string format doesn't support dictionary-encoded strings that survive through
   the pipeline. DuckDB does this with FSST. QuestDB does it with Symbols.
   Arrow has DictionaryArray, but it's a separate type that most kernels don't
   optimize for.

4. **No adaptive encoding.** A column that arrives as Flat might be better served
   as Dictionary (low cardinality), Constant (single value after filter), or
   Run-Length-Encoded (sorted data). Arrow commits to one representation at
   construction time. DuckDB can switch vector formats mid-pipeline.

**Our thesis:** Keep Arrow IPC/Flight at the boundary for interop. Internally,
use a richer column format that carries statistics, supports sidecar indexes,
and adapts encoding on the fly.

---

## 2. What Others Do

### 2.1 DuckDB: Inline Statistics per Vector

Every DuckDB `Vector` (2048 elements) carries:

- **Validity mask** (same as Arrow)
- **Vector type** (Flat, Constant, Dictionary, Sequence, FSST)
- **Auxiliary data** per type (dict child, sequence offset/step, etc.)

But critically, DuckDB's storage layer maintains **zone maps** per column segment
(120K row groups). During scan, zone maps allow predicate pushdown without
touching data. During execution, the Vector type itself encodes information
(Constant = all values equal, Dictionary = low cardinality).

### 2.2 QuestDB: Symbol Tables with Bitmap Indexes

QuestDB's SYMBOL type is the most interesting design for categorical data:

- **Dictionary file (.v):** Unique string values stored once
- **Key file (.k):** Integer keys per row mapping to dictionary
- **Offset file (.o):** Offsets into the dictionary for variable-length values
- **Bitmap index:** For indexed symbols, a separate bitmap tracks which rows
  contain each distinct value. This is a *sidecar index* — updated incrementally
  as data arrives.

The key insight: the dictionary and the index are *separate concerns*. The
dictionary is the encoding (storage efficiency). The index is the acceleration
structure (query speed). QuestDB allows you to have dictionary encoding without
an index (cheap) or with an index (fast filters but 2x write cost).

Symbol capacity auto-scales, the dictionary is cached in memory by default,
and the bitmap index uses a block-based design (256 row IDs per block by default)
that balances memory and lookup speed.

### 2.3 Venus: BH-Tree as Sidecar Spatial Index

The Venus BH-tree (`BHtree.h`) demonstrates the sidecar pattern perfectly:

```c
bh_tree t;
bh_tree_init(&t, max_nodes, agg_stride);  // Pre-allocate
bh_build(&t, items, count, ...);           // Build from column data
bh_query_radius(&t, pos, radius, ...);     // Query
bh_tree_reset(&t);                         // Reset per frame, reuse memory
```

Critical design points:
- **Pre-allocated pool** (`pool_mem`, `pool_cap`) — no per-node malloc
- **Reset per frame** (`bh_tree_reset`) — same epoch pattern as Arena
- **Generic aggregate** (`agg_stride` bytes per node) — the tree carries
  domain-specific summary data alongside spatial partitioning
- **Callback-based** (`bh_pos_fn`, `bh_accum_fn`) — the tree doesn't know
  what it's indexing, just how to partition and summarize

This maps directly to a column sidecar index: pre-allocate the index structure,
build it from column data when the column is materialized, query it during
filter/join, reset it when the morsel is done.

### 2.4 Arrow IPC Statistics

Arrow IPC *does* have statistics in the schema metadata:

```
RecordBatch.metadata:
  ARROW:column_stats:{col_idx}: {
    "min": ..., "max": ...,
    "null_count": ..., "distinct_count": ...
  }
```

But these are:
- Only available after serialization/deserialization (IPC boundary)
- Not maintained during compute (filters don't update stats on output)
- Not available on in-memory arrays constructed programmatically
- String-encoded JSON, requiring parsing

We need stats that are *live* — maintained as data flows through the pipeline.

---

## 3. Proposed: BoltColumn with Stats + Sidecar Indexes

### 3.1 Column Statistics Block

Every `BoltColumn` carries an inline statistics block. This is cheap (64 bytes)
and maintained incrementally:

```cpp
struct ColumnStats {
    // Zone map (maintained on construction + mutation)
    int64_t min_value;        // Type-punned, valid for fixed-width types
    int64_t max_value;
    uint32_t null_count;
    uint32_t distinct_count;  // Approximate (HyperLogLog) or exact if < threshold

    // Encoding hints (set by analysis, used by adaptive encoder)
    uint16_t cardinality_class;  // 0=unknown, 1=constant, 2=low(<256),
                                  // 3=medium(<64K), 4=high
    uint16_t sort_order;         // 0=unknown, 1=ascending, 2=descending,
                                  // 3=nearly_sorted, 4=unsorted
    bool all_valid;              // True = no nulls (skip validity checks)
    bool is_monotonic;           // True = strictly increasing/decreasing

    // String-specific (for German-style views)
    uint32_t max_string_len;     // Longest string in column
    uint32_t total_string_bytes; // Total string data (for arena sizing)
    uint16_t avg_string_len;     // Average (for prefetch distance)
    bool all_inline;             // True = all strings <= 12 bytes (no indirection)
};
```

**When stats are updated:**
- On `BoltColumn::from_arrow()` — compute during ingestion (one pass)
- On `BoltColumn::clone()` — copy stats (free)
- On filter output — adjust min/max, null_count (from selection vector)
- On constant detection — if cardinality == 1, promote to Constant format
- On sort detection — if ascending, set `sort_order` for merge join path

**What this enables:**
- **Zone map skipping:** `WHERE price > 100` on a column with `max_value = 50` → skip entire morsel
- **Join strategy selection:** Low cardinality → hash join. Already sorted → merge join.
- **Null fast-path:** `all_valid = true` → skip validity bitmap checks entirely
- **String optimization:** `all_inline = true` → never chase pointers into data buffers

### 3.2 Adaptive Column Encoding

The column format auto-selects encoding based on stats:

```
Input column (Flat, 16K rows int64) arrives from Parquet scan
  → Stats computed: cardinality=1, min=max=42
  → Auto-promote to Constant format
  → All downstream operators see constant_value<int64_t>() = 42
  → Aggregation: sum = 42 * 16384 in one multiply

Input column (Flat, 16K rows string) arrives from CDC
  → Stats computed: cardinality=47, max_string_len=8
  → Auto-promote to Dictionary format
  → 47-entry dict + 16K uint8 indices (vs 16K × 16-byte views)
  → Hash join: hash 47 values once, not 16K

Input column (Flat, 16K rows int64) after ORDER BY
  → Stats computed: sort_order=ascending, is_monotonic=true
  → Mark as Sequence if perfectly arithmetic
  → Or mark sort_order for merge join optimizer hint
```

### 3.3 Sidecar Index Slots

Each `BoltColumn` has optional sidecar index attachment points:

```cpp
class BoltColumn {
    // ... existing data, validity, format fields ...

    // Inline statistics (always present, 64 bytes)
    ColumnStats stats_;

    // Sidecar indexes (optional, arena-allocated)
    // These are ephemeral — built per-morsel, reset with arena
    struct SidecarSlots {
        void* bitmap_index;    // BitmapIndex* — for low-cardinality filter acceleration
        void* hash_index;      // HashIndex* — for probe-side join acceleration
        void* sort_index;      // SortIndex* — permutation array for order-by
        void* bloom_filter;    // BloomFilter* — for semi-join reduction
    } sidecars_ = {};
};
```

**Sidecar lifecycle:**
1. Column arrives (from Parquet, CDC, or upstream operator)
2. Stats computed during ingestion pass (or copied from source)
3. Optimizer examines stats, decides which sidecars to build
4. Sidecars built from arena (no malloc, no refcounting)
5. Downstream operators use sidecars for acceleration
6. Arena resets at morsel boundary — sidecars freed automatically

**QuestDB-style bitmap index for symbols/labels:**

```cpp
struct BitmapIndex {
    // For a Dictionary column with K distinct values:
    // bitmap[k] has one bit per row — 1 if row has value k
    // Total size: K * (num_rows / 8) bytes
    //
    // For K=47, 16K rows: 47 * 2KB = 94KB
    // One-time build cost: ~50μs
    // Filter cost: popcount on bitmap slice = nanoseconds

    uint64_t** bitmaps;   // Array of K bitmap pointers
    uint32_t num_keys;
    uint32_t num_rows;
    Arena* arena;         // Lifetime tied to arena

    // Build from Dictionary column
    static BitmapIndex* build(const BoltColumn& dict_col, Arena* arena);

    // Filter: return selection vector of rows matching value k
    int64_t filter(uint32_t key, int32_t* output_indices) const;

    // Multi-value filter: OR of multiple bitmaps
    int64_t filter_in(const uint32_t* keys, uint32_t num_keys,
                      int32_t* output_indices) const;
};
```

### 3.4 German-Style Strings + Symbol Encoding

Combine Polars/Umbra string views with QuestDB symbol semantics:

```
String column arrives with stats:
  cardinality_class == LOW (< 256 distinct values)
  → Encode as Symbol:
     - Dictionary: arena-allocated array of string views (16 bytes each)
     - Keys: arena-allocated uint8_t array (one byte per row for < 256 values)
     - Optional bitmap index built from keys
     - String comparison → integer comparison on keys

  cardinality_class == MEDIUM (256 - 64K distinct values)
  → Encode as Dictionary with uint16 keys

  cardinality_class == HIGH (> 64K distinct values)
  → Keep as German-style views (inline ≤ 12 bytes, pointer for longer)
     - all_inline flag enables tight scan loop without indirection
     - max_string_len enables fixed-width treatment if all strings same size
```

**The QuestDB insight applied:** The dictionary and the index are separate concerns.
A column can be dictionary-encoded without being indexed (saves memory). Or it can
be indexed without being dictionary-encoded (bitmap on the raw values). The
`cardinality_class` in stats drives which combination to use.

### 3.5 Arrow IPC Compatibility Layer

At the boundary (Flight, Parquet, IPC), convert:

```
BoltColumn (internal)          →  Arrow Array (external)
──────────────────────          ─────────────────────────
Flat + stats                  →  Array + RecordBatch.metadata stats
Constant                      →  Single-value array (or extension type)
Dictionary (Symbol)           →  DictionaryArray
Sequence                      →  Materialize to flat Int64Array
View                          →  SlicedArray (offset + length)
Stats                         →  ARROW:column_stats metadata
Bitmap sidecar                →  Dropped (ephemeral, not serializable)
```

The reverse direction:
```
Arrow Array (from Parquet)     →  BoltColumn (internal)
──────────────────────          ─────────────────────────
DictionaryArray               →  Dictionary column + stats
Array + IPC stats              →  Flat column + stats (reuse IPC stats)
Array (no stats)               →  Flat column + compute stats in one pass
```

---

## 4. Integration with MarbleDB

MarbleDB's LSM storage already has column-level metadata in its SSTable format.
The BoltColumn stats block should align with MarbleDB's metadata so that:

1. **Scan produces BoltColumns with pre-populated stats** — MarbleDB knows min/max
   per column per SSTable block from its zone maps. These propagate directly into
   BoltColumn stats without recomputation.

2. **Compaction can use stats for encoding decisions** — When MarbleDB merges
   SSTables, it can observe cardinality and sort order, then choose optimal
   encoding for the merged output.

3. **MemTable ↔ BoltBatch is zero-copy** — The active MemTable stores data in
   BoltBatch format. Reads from MemTable return BoltColumns directly. Flush to
   SSTable serializes from BoltColumn → Parquet/native format with stats.

---

## 5. Implementation Priority

### Phase 2a: Stats (this sprint)
- Add `ColumnStats` to `BoltColumn`
- Compute stats in `from_arrow()` one-pass
- Propagate stats through filter/project
- Zone map skip in morsel source (trivial but high-value)

### Phase 2b: Adaptive Encoding
- Constant detection and auto-promotion
- Dictionary encoding for low-cardinality strings
- Symbol type with QuestDB-style separate dict + keys
- Stats-driven encoding selection

### Phase 2c: Sidecar Indexes
- Bitmap index for dictionary columns
- Bloom filter for semi-join reduction (SwissJoin build side)
- Sort index for merge join path selection
- All arena-allocated, all ephemeral

### Phase 2d: Arrow IPC Bridge
- `to_arrow()` serializes stats to RecordBatch metadata
- `from_arrow()` deserializes stats from IPC metadata
- Flight integration: stats propagate across agents
- Parquet reader: Parquet row group stats → BoltColumn stats

---

## 6. Open Questions

1. **HyperLogLog for distinct_count:** Use a 64-register HLL for approximate
   distinct count? 64 bytes per column, ~2% error. Or exact count for
   cardinality < 1024 (fits in a small hash set)?

2. **Stats update on mutation:** When BoltBatch COW triggers and a column is
   modified, stats become stale. Options: (a) invalidate stats, recompute
   lazily; (b) maintain incrementally (complex for min/max deletion);
   (c) mark as approximate (good enough for optimizer hints).

3. **Bitmap index threshold:** At what cardinality does bitmap index stop
   being useful? QuestDB defaults to 256 row IDs per index block. For 16K-row
   morsels with cardinality > 1000, the bitmap might be larger than the data.

4. **FSST for high-cardinality strings:** DuckDB uses FSST (Fast Static Symbol
   Table) compression for strings during execution. Worth implementing? It
   requires building a symbol table from a sample, then encoding/decoding
   through the pipeline. Good for repeated substrings (URLs, paths, labels).

5. **Venus spatial index for vector columns:** BoltColumn for HNSW embedding
   vectors — should the BH-tree/KD-tree sidecar support nearest-neighbor
   queries during pipeline execution? Relevant for AI memory retrieval
   pipelines where vector search is fused with SQL filters.
