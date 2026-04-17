# QuestDB SYMBOL — source-code audit of the research note

Follow-up to [`questdb-symbol-vs-fsst.md`](questdb-symbol-vs-fsst.md). The
original note summarised QuestDB's SYMBOL type from published docs and
community posts. This audit walks the actual Java source in
`questdb/questdb@master` (Apache 2.0) and pins each claim to a file path
plus line range. Anything a doc says but the code contradicts is marked
REFUTED and corrected.

All paths are relative to the repo root. Line numbers are from `master`
at audit time (2026-04-17).

---

## Claim 1a: SYMBOL dictionary produces `.d`/`.k`/`.v`/`.o` files

Research note says: *".d — row-order int keys … .k — symbol keys file …
.v — symbol values file … .o — offsets sidecar into .v"*

Source evidence:

- `core/src/main/java/io/questdb/cairo/TableUtils.java` **line 88**:
  `public static final String FILE_SUFFIX_D = ".d";`
- `core/src/main/java/io/questdb/cairo/TableUtils.java` **line 89**:
  `public static final String FILE_SUFFIX_I = ".i";`
- `core/src/main/java/io/questdb/cairo/TableUtils.java` **~line 1707**:
  `offsetFileName(...){ path.concat(columnName).put(".o"); }`
- `core/src/main/java/io/questdb/cairo/TableUtils.java` **~line 1641**:
  `charFileName(...){ path.concat(columnName).put(".c"); }`
- `core/src/main/java/io/questdb/cairo/BitmapIndexUtils.java`
  **~lines 45 / 103**:
  `keyFileName(...){ ... put(".k"); }` and
  `valueFileName(...){ ... put(".v"); }`

Status: **PARTIALLY CONFIRMED — name mismatch.**

Correction: the **dictionary** files are `.c` (characters, i.e. the
string payload) and `.o` (offsets into `.c`), **not** `.v` and a
separate offsets sidecar. The `.k`/`.v` pair belongs to the **bitmap
index** (row-ID lists per key), which is only present when the column
was declared `INDEX`. The research note conflated the two pairs. So
the correct summary is:

- Always: `<col>.d` (per-row int key column, per-partition),
  `<col>.c` (string bytes, table-global),
  `<col>.o` (offsets into `.c`, table-global).
- If `INDEX`: plus `<col>.k` (index header / hash buckets) and
  `<col>.v` (row-ID list blocks).

---

## Claim 1b: `.d` is per-partition; dictionary files are table-global

Research note says: *"The dictionary is table-global, not
per-partition — one symbol ID means the same string across every
partition."*

Source evidence:

- `core/src/test/java/io/questdb/test/cairo/SymbolMapTest.java`
  **~lines 107–112, 263–268**: `SymbolMapWriter` is instantiated with
  `path = new Path().of(configuration.getDbRoot())` (root, not
  partition) and only the `columnName` — no partition timestamp
  parameter.
- `core/src/main/java/io/questdb/cairo/SymbolMapWriter.java`
  constructor signature takes `(CairoConfiguration, Path,
  CharSequence columnName, long columnNameTxn, int symbolCapacity,
  …)` — **no partition timestamp.** Compare this with `.d` column
  writers, which are opened per partition.
- QuestDB storage docs (root-directory-structure page) show
  partition directories containing `mycolumn.d`, but `.c`/`.o`
  appear at the table-root level in the file layout tree.

Status: **CONFIRMED.**

Correction: none. This is the load-bearing invariant for Bolt's
global-dictionary design — symbol IDs in `.d` from any partition
compare directly, no rehash at merge/join.

---

## Claim 2: `.d` holds int IDs; string materialisation is late

Research note says: *"operators run on integer IDs directly. The
string materialisation only happens at projection time."*

Source evidence:

- `core/src/main/java/io/questdb/cairo/SymbolMapReaderImpl.java`
  **~lines 105–115** (`keyOf(CharSequence value)`):
  `return SymbolMapWriter.offsetToKey(offsetOffset);` — returns `int`.
- `core/src/main/java/io/questdb/cairo/SymbolMapReaderImpl.java`
  **~line 184** (`valueOf(int key)`):
  `return charMem.getStrA(offsetMem.getLong(SymbolMapWriter.keyToOffset(key)));`
  — the only path that reconstitutes a string, gated on a projection
  actually asking for one.
- `core/src/main/java/io/questdb/cairo/SymbolMapWriter.java`
  **~line 354**:
  `return (int) ((offset - HEADER_SIZE) / 8L);` — the int key is
  literally an array index into the `.o` file.

Status: **CONFIRMED.**

---

## Claim 3: equality literal is resolved to an int **once** at plan time

Research note says: *"Equality against a literal resolves the literal
to an ID once at plan time and then does int-compare on `.d`."*

Source evidence:

- `core/src/main/java/io/questdb/griffin/engine/functions/eq/EqSymStrFunctionFactory.java`
  **inner class `ConstSymIntCheckFunc`, ~lines 109–113** (`init()`):
  `valueIndex = staticSymbolTable.keyOf(constant);
   exists = (valueIndex != SymbolTable.VALUE_NOT_FOUND);`
- Same file, **~lines 103–105** (`getBool(Record)` hot path):
  `return negated != (exists && arg.getInt(rec) == valueIndex);`

So: `keyOf(constant)` runs in `init()` (cursor bind, not per-row), the
result is cached in `valueIndex`, and the per-row check is a single
`int == int`. Exactly the "resolve-once" pattern the note claims.

Status: **CONFIRMED.**

---

## Claim 4: indexed-SYMBOL default block size 256

Research note says: *"block size 256 (`cairo.index.value.block.size`)."*

Source evidence:

- `core/src/main/java/io/questdb/cairo/DefaultCairoConfiguration.java`
  **~lines 1024–1026**:
  `public int getIndexValueBlockSize() { return 256; }`

Also found nearby and worth recording for future tuning:

- `getDefaultSymbolCapacity() { return 128; }` (line ~921).
- `autoScaleSymbolCapacity() { return true; }` (line ~106).
- `autoScaleSymbolCapacityThreshold() { return 0.8; }` (line ~110).

The index file structure: `.k` is the per-symbol header (which block
to read), `.v` is blocks of 256 row-IDs per key by default — i.e. the
thing Bolt's `BitmapIndex` is structurally equivalent to.

Status: **CONFIRMED.**

---

## Claim 5: cardinality degradation wall (issue #6246)

Research note says: *"300M rows … 10 distinct symbols → ms, 10k →
15–25 s warm, >25 s cold."*

Source evidence: GitHub issue
`https://github.com/questdb/questdb/issues/6246` — title "Query
performance degradation with high SYMBOL cardinality", reporter
measurements match verbatim: 300M rows, ~28k-row result set, 10
distinct = ms, 10k distinct = 15–25 s warm, >25 s cold first-fetch.

No in-source comment documents a hard cardinality ceiling; the
`autoScaleSymbolCapacityThreshold = 0.8` in
`DefaultCairoConfiguration.java` implies the dict grows rather than
capping, which is consistent with the observed degradation being a
*performance* rather than *correctness* wall.

Status: **CONFIRMED** (issue text); **NOT FOUND** (any in-source
comment explicitly calling out the cardinality ceiling). The note's
framing ("measured wall, not declared limit") is accurate.

---

## Claim 6: No FSST / substring compression for VARCHAR

Research note says: *"QuestDB's Parquet writer uses RleDictionary for
SYMBOL … No FSST."*

Source evidence:

- `core/src/main/java/io/questdb/cairo/VarcharTypeDriver.java` — the
  on-disk layout is **not** FSST or any substring compression. It's
  a two-file auxiliary+data split with three encoding paths:
  - `HEADER_FLAG_INLINED` (1): string ≤ 9 bytes, fully inlined in
    16-byte aux entry (constant `VARCHAR_MAX_BYTES_FULLY_INLINED = 9`).
  - Long form: aux entry = 6-byte prefix + 48-bit offset into data
    file, full UTF-8 bytes in data file
    (`VARCHAR_INLINED_PREFIX_BYTES = 6`).
  - `HEADER_FLAG_ASCII` (2) / `VARCHAR_HEADER_FLAG_NULL` (4) flags.
  Aux entry is 16 bytes (`VARCHAR_AUX_WIDTH_BYTES = 2 * Long.BYTES`).
- The string "FSST" does not appear in `VarcharTypeDriver.java`, in
  any file under `cairo/`, or in any search of the repo.

Status: **CONFIRMED.**

Correction: the research note was broadly right about absence of
FSST, but under-described the VARCHAR format. The 6-byte prefix +
short-string inlining is actually similar to Umbra/DuckDB's "German
strings" / string-view layout — this is a useful data point on its
own and worth its own follow-up note. It gives prefix-equality
shortcut and a cheap no-dereference hash, but no substring
compression across rows.

---

## Claim 7: Parquet SYMBOL handling

Research note says: *"QuestDB's Parquet writer uses RleDictionary for
SYMBOL."*

Source evidence:

- `core/src/main/java/io/questdb/griffin/engine/table/parquet/PartitionEncoder.java`
  **~lines 216–232**:
  `if (ColumnType.isSymbol(columnType)) { SymbolMapReader … =
   tableReader.getSymbolMapReader(i); MemoryR symbolValuesMem =
   symbolMapReader.getSymbolValuesColumn(); MemoryR symbolOffsetsMem
   = symbolMapReader.getSymbolOffsetsColumn(); … }`
  The writer ships the raw dict (`.c` + `.o`) plus the int `.d`
  column to the native encoder; actual Parquet dict/RLE encoding is
  in C++/Rust native code (`encodePartition()`), not in this Java
  file.
- `core/src/main/java/io/questdb/griffin/engine/table/parquet/PartitionDecoder.java`
  **~lines 339–376** (`copyToSansUnsupported` on the `Metadata`
  inner class): reads Parquet columns back as SYMBOL **by default**;
  a `treatSymbolsAsVarchar` flag forces conversion to VARCHAR on
  import. So Parquet dict-encoded strings round-trip as SYMBOL
  unless the caller opts out.

Status: **CONFIRMED** on the *behaviour* (dict-encoded → SYMBOL). The
note's specific claim that the *on-wire encoding is RleDictionary*
is not verifiable from Java alone — that call lives in the native
encoder and would need a Rust/C++ audit. Marking that sub-claim
**PARTIALLY CONFIRMED** pending native-side audit.

---

## Claim 8: Global-across-partitions dictionary invariant

See Claim 1b. Source evidence identical. **CONFIRMED.**

This is the most important claim for Bolt's design — the int IDs in
every partition's `.d` share one namespace, so cross-morsel
comparisons are just int compares and a global dict is the natural
default.

---

## Bolt impact

None of the corrections change the recommendation. Specifically:

1. The `.v` vs `.c` / `.o` filename mix-up in the research note is
   cosmetic — the **structure** (int column + string dict + offsets
   + optional row-ID-list index) is exactly what Bolt's
   `ColumnFormat::Dictionary` + `BitmapIndex` already model. Bolt's
   `values` column ≡ QuestDB's `.c`, Bolt's implicit offset array
   ≡ QuestDB's `.o`, Bolt's key array ≡ QuestDB's `.d`, Bolt's
   `BitmapIndex` ≡ QuestDB's `.k` + `.v` (index variant).

2. Literal-resolve-once (Claim 3) is *confirmed in source* —
   `ConstSymIntCheckFunc.init()` doing `keyOf(constant)` once and
   the hot path being `arg.getInt(rec) == valueIndex` is the pattern
   Bolt should adopt verbatim in the filter-build step.

3. Global dictionary (Claim 8) is *confirmed in source* via
   `SymbolMapWriter`'s construction-site (no partition timestamp,
   path is table root). This validates the "upgrade Bolt Dictionary
   to table-global tick-tock dict" follow-up in the original note.

4. VARCHAR Claim 6: the note was right that QuestDB has no FSST, but
   *under-described* the VARCHAR format. QuestDB's "6-byte prefix +
   inline-or-offset" layout is Umbra/DuckDB-shaped. That's an
   independent data point worth its own note — it doesn't overlap
   with FSST (compression) but does overlap with Bolt's choice of
   short-string representation. **Followup:** add
   `questdb-varchar-layout.md` when someone touches Bolt string
   storage; not urgent.

5. Cardinality wall (Claim 5) remains an observational wall from
   issue #6246, not a coded-in limit — reinforces the original
   recommendation to build a per-ID popcount sidecar for fast
   `does this key even exist` pre-checks, which QuestDB demonstrably
   lacks.

**Recommendation unchanged:** Dictionary + BitmapIndex covers the
SYMBOL shape; FSST deferred to a separate `ColumnFormat` behind
`BOLT_ENABLE_FSST`; the four original follow-ups (global dict,
literal-resolve-once, per-ID popcount, lock-free append-on-new)
stand. Update the original note only to fix the `.v` → `.c`/`.o`
filename error.

## Sources (primary)

- `core/src/main/java/io/questdb/cairo/TableUtils.java` (file suffix
  constants, `offsetFileName`, `charFileName`).
- `core/src/main/java/io/questdb/cairo/BitmapIndexUtils.java`
  (`keyFileName`, `valueFileName`).
- `core/src/main/java/io/questdb/cairo/SymbolMapWriter.java`
  (constructor, `put0`, `offsetToKey`).
- `core/src/main/java/io/questdb/cairo/SymbolMapReaderImpl.java`
  (`keyOf`, `valueOf`).
- `core/src/main/java/io/questdb/cairo/VarcharTypeDriver.java`
  (`VARCHAR_MAX_BYTES_FULLY_INLINED`, `VARCHAR_INLINED_PREFIX_BYTES`,
  `HEADER_FLAG_*`, `appendValue`).
- `core/src/main/java/io/questdb/cairo/DefaultCairoConfiguration.java`
  (`getIndexValueBlockSize`, `getDefaultSymbolCapacity`,
  `autoScaleSymbolCapacity*`).
- `core/src/main/java/io/questdb/griffin/engine/functions/eq/EqSymStrFunctionFactory.java`
  (`ConstSymIntCheckFunc`).
- `core/src/main/java/io/questdb/griffin/engine/table/parquet/PartitionEncoder.java`
  (`isSymbol` branch, symbol values/offsets hand-off to native
  encoder).
- `core/src/main/java/io/questdb/griffin/engine/table/parquet/PartitionDecoder.java`
  (`treatSymbolsAsVarchar` in `Metadata.copyToSansUnsupported`).
- `core/src/test/java/io/questdb/test/cairo/SymbolMapTest.java`
  (confirms `new SymbolMapWriter(configuration, path, columnName,
  ...)` with `path = dbRoot`, no partition timestamp).
- GitHub issue `questdb/questdb#6246` (cardinality degradation
  measurements).
