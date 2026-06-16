# bolt::lakehouse — feature-complete Delta + Iceberg, no Rust

Replaces the old Gestalt `lakehouse/` library (Rust-binding `delta-
kernel-rs` for Delta; hand-written C++ for Iceberg + manifest writer/
reader + REST catalog + parallel Parquet reader; ~14.5 K LoC). Same
surface, same feature set, **C++ all the way down** on bolt
primitives. No Rust, no Arrow runtime on hot paths, no Java/JNI.

Lives at `extern/bolt/lakehouse/` — sibling to `extern/bolt/ingest/`
and `extern/bolt/net/`. Chukonu's existing lakehouse connectors
(`lakehouse_source.cpp`, `lakehouse_iceberg_source.cpp`,
`lakehouse_sink.cpp`) become thin wrappers around `bolt::lakehouse`.

## Non-negotiables

- **Tiger Style:** PODs + free functions, ≥2 asserts/fn, ≤70-line
  fns, no exceptions, no smart pointers, no RTTI, all allocs via
  `bolt::Arena`, every loop has a `constexpr` cap.
- **Bolt on the hot path; Arrow only at egress.** Internal data is
  `BoltBatch`. Arrow C Data Interface only at external boundaries.
- **No external runtime deps** beyond OpenSSL (via TlsSocket) and the
  bolt-vendored zstd/snappy/lz4 codecs.
- **Cross-platform** — Windows MSVC / macOS clang / Linux clang+gcc.
  No POSIX-only headers in public surface.

## Architecture — one trait, two implementations

```cpp
namespace bolt::lakehouse {

enum class TableFormat : uint8_t { kDelta = 0, kIceberg = 1 };

struct TableHandle;     // opaque, format-specific state

// Format-agnostic API; dispatches on TableFormat at session init.
struct ReadOptions {
    int64_t   snapshot_id;          // -1 = latest; > 0 = time travel
    int64_t   timestamp_ms;          // -1 = ignore; takes precedence over snapshot_id
    Predicate predicates[16];
    uint32_t  n_predicates;
    char      projection[64][64];
    uint32_t  n_projection;
    bool      include_deletes;       // CDF / Delta CDC mode
    bool      include_partition_columns;
};

struct WriteOptions {
    char         partition_columns[8][64];
    uint32_t     n_partition_columns;
    char         sort_columns[8][64];     // Iceberg sort order; ignored on Delta
    uint32_t     n_sort_columns;
    Compression  compression;             // SNAPPY/GZIP/ZSTD/LZ4
    int32_t      target_file_size_mb;
    int32_t      target_row_group_rows;
    bool         emit_stats;
    bool         emit_bloom_filters;
    bool         emit_page_index;
};

// Table lifecycle.
bool table_open  (TableHandle**, Arena*, Catalog*, const char* namespace_,
                  const char* name, TableFormat hint) noexcept;
bool table_create(TableHandle**, Arena*, Catalog*, const char* namespace_,
                  const char* name, const Schema*, const WriteOptions*,
                  TableFormat format) noexcept;
void table_close (TableHandle*) noexcept;

// Read path.
struct ScanHandle;
bool  scan_open      (ScanHandle**, TableHandle*, const ReadOptions*) noexcept;
bool  scan_next_batch(ScanHandle*, BoltBatch* out, bool* out_eof) noexcept;
void  scan_close     (ScanHandle*) noexcept;

// Write path — append.
struct AppendHandle;
bool append_open  (AppendHandle**, TableHandle*) noexcept;
bool append_write (AppendHandle*, const BoltBatch*) noexcept;
bool append_commit(AppendHandle*) noexcept;
void append_close (AppendHandle*) noexcept;

// Update / delete / merge.
bool table_update(TableHandle*, const Predicate*, const Assignment*, uint32_t n) noexcept;
bool table_delete(TableHandle*, const Predicate*) noexcept;
bool table_merge (TableHandle*, const MergeSpec*) noexcept;

// Admin.
bool table_history(TableHandle*, HistoryEntry out[], uint32_t cap, uint32_t* n) noexcept;
bool table_snapshots(TableHandle*, SnapshotEntry out[], uint32_t cap, uint32_t* n) noexcept;
bool table_optimize(TableHandle*, const OptimizeOptions*) noexcept;
bool table_vacuum  (TableHandle*, uint64_t retention_hours, bool dry_run,
                    char (*out_deleted)[512], uint32_t cap, uint32_t* n) noexcept;
bool table_rollback(TableHandle*, int64_t target_version) noexcept;
bool table_branch_create(TableHandle*, const char* branch, int64_t base_snapshot) noexcept;
bool table_tag_create   (TableHandle*, const char* tag, int64_t snapshot) noexcept;

}  // namespace bolt::lakehouse
```

The trait dispatches on `TableFormat`; Delta + Iceberg each implement
the entire surface. `TableHandle` is opaque; the format-specific
state lives behind it.

## Module layout

```
extern/bolt/lakehouse/
├── include/bolt/lakehouse/
│   ├── format.h               — TableFormat enum + Schema + Predicate + Assignment
│   ├── handle.h               — TableHandle / ScanHandle / AppendHandle opaque types
│   ├── catalog.h              — Catalog trait POD
│   ├── object_store.h         — ObjectStore trait POD
│   ├── delta/
│   │   ├── log.h              — _delta_log parser (JSON actions + checkpoint .parquet)
│   │   ├── snapshot.h         — replay actions → live file set
│   │   ├── deletion_vector.h  — Roaring bitmap (.dv) deserialise
│   │   ├── column_mapping.h   — id ↔ physical-name ↔ logical-name
│   │   ├── generated_column.h — recompute on read
│   │   ├── cdf.h              — CDF (_change_data/*.parquet) reader
│   │   ├── writer.h           — Add action emission + commit JSON
│   │   ├── optimize.h         — bin-pack + Z-ORDER
│   │   └── checkpoint.h       — checkpoint write + read
│   ├── iceberg/
│   │   ├── metadata.h         — metadata.json reader/writer
│   │   ├── manifest.h         — Avro manifest reader/writer
│   │   ├── snapshot.h         — snapshot mgmt (incl. branches/tags)
│   │   ├── partition.h        — PartitionSpec + transform fns
│   │   ├── sort_order.h
│   │   ├── statistics.h       — Puffin statistics files
│   │   ├── writer.h           — manifest list + manifest + data file writer
│   │   ├── delete_file.h      — equality + position deletes
│   │   ├── view.h             — Iceberg views
│   │   └── transform.h        — identity/year/month/day/hour/bucket/truncate
│   ├── parquet/
│   │   └── scan.h             — wraps bolt::ingest::parquet for vectorized scan
│   ├── avro/                   — NEW bolt primitive used by Iceberg manifests
│   │   ├── reader.h
│   │   ├── writer.h
│   │   ├── schema.h
│   │   └── codec.h            — null + deflate + snappy + zstd object container
│   ├── catalog/
│   │   ├── filesystem.h
│   │   ├── hive_metastore.h   — Thrift binary client
│   │   ├── iceberg_rest.h     — Apache REST Catalog full spec
│   │   ├── unity.h            — Databricks UC
│   │   ├── glue.h             — AWS Glue Data Catalog
│   │   ├── polaris.h          — Snowflake Polaris (IRC + extras)
│   │   ├── nessie.h           — Project Nessie
│   │   └── gravitino.h        — Apache Gravitino
│   └── object_store/
│       ├── filesystem.h
│       ├── s3.h               — SigV4 (already in chukonu — port up)
│       ├── azure.h            — Azure Blob (SharedKey + SAS)
│       ├── gcs.h              — Google Cloud Storage (HMAC + OAuth)
│       └── compat.h           — MinIO / R2 / Wasabi endpoint override
├── src/...   (mirrors include/ tree)
├── tests/    (one cpp per public header surface)
├── docs/
│   ├── DESIGN.md             — this doc, lifted
│   ├── DELTA_SPEC_NOTES.md   — what protocol features land per phase
│   ├── ICEBERG_SPEC_NOTES.md — same for Iceberg v1/v2/v3
│   └── PERF.md               — measured benchmarks per gate
└── CMakeLists.txt
```

## Bolt primitives this depends on

Already shipped:
- `bolt::Arena`, `bolt::BoltBatch`, `bolt::BoltColumn`, `bolt::StringView`
- `bolt::ingest::parquet_read_meta` + `parquet_read_row_group` + the new `parquet_write_*`
- `bolt::ingest::snappy_decompress` + `_compress`
- `bolt::io::crc32c`
- `bolt::net::TcpSocket`, `bolt::net::TlsSocket`
- `bolt::SwissTable` (path → live-file lookup)
- `bolt::channel::SPSCChannel` (async object-store fetch queue)

To add upstream:
- `bolt::ingest::bolt_zstd_decompress` / `_compress` (zstd library vendored)
- `bolt::ingest::bolt_gzip_decompress` (zlib vendored, decompress-only)
- `bolt::ingest::bolt_lz4_decompress` / `_compress`
- `bolt::ingest::roaring_deserialize` (Roaring bitmap for Delta deletion vectors)
- `bolt::parse::bolt_json` already exists (fionn) — used for Delta `_delta_log/*.json` + Iceberg `metadata.json` + REST catalog responses
- `bolt::ingest::bolt_avro` — **NEW** (Iceberg manifests + Schema Registry payloads)
- `bolt::crypto::sha256` from chukonu — port to `bolt::crypto`
- `bolt::crypto::sigv4` — extract from chukonu's S3 client into bolt

## Phase plan

### W1 — Foundations (1.5 weeks)
- W1.1 `bolt::ingest::bolt_avro` — Avro 1.11 object container format reader + writer. Codecs: null, deflate, snappy, zstd. Schema parse (JSON). Per-type decoders: null/bool/int/long/float/double/bytes/string/record/enum/array/map/union/fixed. Maps directly to `BoltBatch` rows.
- W1.2 `bolt::ingest::roaring_deserialize` — Roaring bitmap deserialise for Delta DVs. 64-bit container variant.
- W1.3 `bolt::ingest::zstd` + `gzip` + `lz4` — codec wrappers (vendoring zstd + zlib + lz4).
- W1.4 `bolt::crypto::sigv4` — extracted from chukonu's S3 client.
- W1.5 `bolt::lakehouse::ObjectStore` trait + filesystem + S3 implementations (port from chukonu).
- W1.6 `bolt::lakehouse::Catalog` trait + filesystem implementation.

### W2 — Delta read (2 weeks)
- W2.1 `_delta_log/*.json` parser via `bolt::parse::bolt_json` — Protocol / Metadata / Add / Remove / CommitInfo / DomainMetadata / CDC actions.
- W2.2 Checkpoint `.parquet` reader (action subset embedded as Parquet rows).
- W2.3 Snapshot replay: walk actions in version order, merge into live-file set keyed by path. Dedup via Remove actions.
- W2.4 Column mapping (NAME / ID modes per Protocol).
- W2.5 Generated columns: recompute on read using the metadata's `delta.generationExpression`.
- W2.6 Deletion vectors: load `.dv` file, apply Roaring bitmap to filter row indices in each Parquet row group.
- W2.7 Predicate pushdown via Add action's `stats` JSON (numRecords + minValues + maxValues + nullCount).
- W2.8 Partition pruning via Add action's `partitionValues`.
- W2.9 Time travel: `versionAsOf` / `timestampAsOf` (binary-search the log).
- W2.10 CDF mode: read `_change_data/*.parquet` for insert/update_preimage/update_postimage/delete.

### W3 — Delta write (2 weeks)
- W3.1 Schema inference + serialisation to Metadata action.
- W3.2 Parquet writer wrapper that emits Add actions per file via `bolt::ingest::parquet_write_*`.
- W3.3 Commit JSON append (path = `_delta_log/<version>.json`). O_EXCL race-safe (already in v1 sink).
- W3.4 Concurrent-writer conflict detection: re-read log between snapshot read + commit write; abort + retry on conflict.
- W3.5 UPDATE: per-file rewrite (or DV emit for Delta 3+ tables).
- W3.6 DELETE: emit DV when supported; else per-file rewrite.
- W3.7 MERGE INTO: compile to source-target join + per-file rewrite plan.
- W3.8 OPTIMIZE: bin-pack small files; Z-ORDER BY (Hilbert-curve ordering on N columns).
- W3.9 VACUUM: dry-run + commit-aware delete of files older than retention.
- W3.10 Checkpoint writer: every 10 commits (configurable), aggregate live set into a checkpoint Parquet.
- W3.11 RESTORE TABLE TO VERSION N: revert via new commit.

### W4 — Iceberg read (2 weeks)
- W4.1 `metadata.json` reader. v1 + v2 + v3 spec parse. Snapshot list, schema list, partition spec list, sort order list, properties.
- W4.2 Snapshot resolution: snapshot_id, branch, tag, timestamp. version-hint.text fallback.
- W4.3 Manifest list (Avro) reader.
- W4.4 Manifest (Avro) reader: data file entries + delete file entries (v2 only).
- W4.5 Partition transforms: identity / year / month / day / hour / bucket(N) / truncate(W).
- W4.6 Partition pruning via partition spec evolution (apply correct spec per partition_spec_id per file).
- W4.7 Stats-based file skip via `lower_bounds` / `upper_bounds` / `null_value_counts`.
- W4.8 Equality deletes: load delete files; filter scan rows.
- W4.9 Position deletes: load file-path + position; mask rows during scan.
- W4.10 Schema evolution by field-id: resolve column lineage across renames.
- W4.11 Column projection via field-id (not name).

### W5 — Iceberg write (2 weeks)
- W5.1 Manifest writer (Avro).
- W5.2 Manifest list writer (Avro).
- W5.3 metadata.json writer with monotonic-int version + atomic rename via ObjectStore `put_if_absent`.
- W5.4 Data file writer (Parquet) with field-id metadata in column path.
- W5.5 New snapshot creation: append / overwrite / rewrite / delete operation summaries.
- W5.6 Schema evolution: add/drop/rename column → new schema entry + new metadata.json.
- W5.7 Partition spec evolution: new spec id; existing manifests retain their spec id.
- W5.8 Sort order evolution.
- W5.9 Compaction: rewrite manifests; rewrite data files into bigger files.
- W5.10 Expire snapshots + remove orphans.
- W5.11 Equality + position delete file writers.
- W5.12 Branches + tags: `set-snapshot-ref` updates in metadata.json.
- W5.13 Iceberg View support (v1 view spec).

### W6 — Catalogs (1.5 weeks)
- W6.1 **Iceberg REST Catalog** full Apache spec: OAuth2 + SigV4 auth, server config discovery, namespace CRUD, table CRUD, multi-table txn commits, view CRUD, snapshot mgmt, register-table, vended credentials, branches/tags, statistics, pagination.
- W6.2 **Hive Metastore** Thrift client (minimal codec, get/create/alter/drop {database, table, partition}).
- W6.3 **Unity Catalog** Databricks REST: catalogs/schemas/tables/volumes/external-locations/permissions/shares/recipients/lineage/tags.
- W6.4 **AWS Glue Data Catalog** SigV4: Database/Table/Partition CRUD; schema registry; Lake Formation grants.
- W6.5 **Polaris** (IRC + Polaris-specific OAuth wrapper).
- W6.6 **Nessie** (IRC + commit/merge/transplant/compare-references native endpoints).
- W6.7 **Gravitino** (multi-catalog gateway).

### W7 — Object stores (1 week)
- W7.1 **Azure Blob Storage** — Shared Key + SAS auth; PUT/GET/range/list/delete + multi-part block upload.
- W7.2 **Google Cloud Storage** — HMAC auth (preferred for parity with S3 client) + OAuth2 ServiceAccount path.
- W7.3 **S3-compatible** — already done; verify MinIO / R2 / Wasabi / Backblaze endpoints work with `endpoint_url` override.
- W7.4 **Async I/O queue** — `bolt::channel::SPSCChannel` between scan-orchestrator and object-store-fetcher threads. Prefetch the next N row-groups while the current one is decoding.

### W8 — Optimisations (1 week)
- W8.1 Vectorized scan via BoltBatch + selection vectors.
- W8.2 Column-pruning + projection pushdown.
- W8.3 Predicate pushdown to Parquet via stats + bloom + page index.
- W8.4 Deletion-vector filtering inline with row-group decode.
- W8.5 Partition pruning.
- W8.6 Multi-threaded row-group scan via `bolt::scheduler`.
- W8.7 Read-ahead via `SPSCChannel`-fed prefetch queue.
- W8.8 Per-column-type SIMD scans where bolt has the kernels (numeric filters).

### W9 — Migration of existing connectors (3 days)
- W9.1 Rewrite `chukonu/src/connectors/lakehouse_source.cpp` to call `bolt::lakehouse::scan_*`.
- W9.2 Rewrite `chukonu/src/connectors/lakehouse_iceberg_source.cpp` to call `bolt::lakehouse::scan_*` with `format=kIceberg`.
- W9.3 Rewrite `chukonu/src/connectors/lakehouse_sink.cpp` to call `bolt::lakehouse::append_*`.
- W9.4 Rewrite `chukonu/src/connectors/lakehouse_admin.cpp` to delegate.
- W9.5 Delete the chukonu-side manifest/log parser code (moved up to bolt).

## Verification gates

Per phase, run:

- `ctest -R bolt_lakehouse_<phase>` — unit + integration tests under `extern/bolt/lakehouse/tests/`.
- Reference dataset roundtrip: write 1M rows, read back, byte-compare. Both Delta and Iceberg.
- Perf gate: scan 100M rows / 50 columns / SNAPPY / 256 MiB row groups → ≥ 5 GB/s on 8 cores, p99 ≤ 5 ms per row group.
- Compatibility gate: Delta-Spark, delta-rs, pyiceberg, AWS Athena/EMR Iceberg all able to round-trip data we wrote. (Manual; documented harness.)

## Migration sequencing

W1 → W2 || W4 (Delta read + Iceberg read can land in parallel since
the format dispatch is at the top of the trait).
After W2 + W4: W3 || W5 (Delta write + Iceberg write parallel).
W6 || W7 in parallel after W1.5 (catalogs need ObjectStore;
ObjectStore extensions don't depend on Delta/Iceberg).
W8 follows W2 + W4 (need a working read path to optimize against).
W9 follows everything else.

Estimated total: 13.5 weeks single-track, 7-8 weeks with parallelism.
The existing chukonu Lakehouse v2 work (Phases A-G) provides the
scaffold; this rewrite replaces and exceeds it.

## What lands where

Code: `extern/bolt/lakehouse/`. Bolt becomes the canonical lakehouse
library; chukonu connectors are thin shims (~50-100 lines each).

Tracker: epic G2CLUS-LAKE (will be opened by this commit), phases as
sub-tasks.
