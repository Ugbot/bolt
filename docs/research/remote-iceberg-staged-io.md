# Bounded staged Iceberg writes and versioned ranged scans

Qualified on macOS arm64 with the existing Apple Clang release build,
2026-10-21. Parent Gestalt2 acceptance runs 29 CTests / 271 executed GoogleTests,
zero runtime skips, in 32.82s. This is a scoped gate, not full repository,
Windows/Linux, RSS or performance qualification.

## Writer transport

`iceberg/write_io.h` adds a startup opt-in `FileWriteIO` to writer-created
table handles. Set it immediately after create/open, before the first mutation
or append handle. The table copies the existing absolute private staging root,
explicit nonzero per-file byte cap and callback/context. Context remains borrowed.
Generic Parquet and existing local Iceberg callers retain their defaults.

Append, overwrite and positional-delete sources share one staging path:

1. Reserve a complete candidate key, canonical location, table UUID, metadata
   versions and snapshot. Exclusively create its own staging directory; never
   truncate an existing retained candidate.
2. Serialize through `parquet_write_open_bounded`. Every sink write charges the
   cap before writing, including magic, pages, indexes and footer. Sticky failure
   refuses close and removes the incomplete unpublished source.
3. Check flush and close. Reopen the exact closed file, verify its extent, and
   compute SHA-256 with one 64 KiB buffer before external publication.
4. Invoke one `publish_if_absent` callback with `StagedDataFile`. The callback
   must provide real atomic conditional publication and classify all ambiguous
   initiation, completion or cleanup outcomes as `kOsWriteUncertain`.
5. Publish existing immutable manifests/list/metadata only after definite data
   ACK. Preserve source and all potentially referenced objects on uncertainty.

`CommitObjectKind::kDataFile = 4` preserves previous enum values. The copied data
receipt includes exact key/location/UUID, full hash/length, base and candidate
metadata versions, snapshot, local path and operation digest. The digest uses
domain `bolt.iceberg.data.v1`, uint64 little-endian length-prefixed text fields
(key, location, UUID, source hash), then uint64 length, base version, candidate
version and snapshot. It excludes the local path. It identifies a candidate;
it is not a durable exactly-once command ID.

`table_staged_data_file` exposes the most recent receipt and cleanup status.
`table_uncertain_data_file` preserves the earlier data receipt even when the
later uncertain candidate is metadata. A retained source fences another
file-producing mutation that would overwrite its only receipt. Definite ACK
and failed local cleanup remain separate outcomes. Definite data conflict never
deletes another writer's key; uncertainty never triggers cleanup or retry.

Flush/close is not fsync. The caller owns crash-durable prepared records,
retained-stage inventory and aggregate disk/concurrency/encoder memory budgets.
The generic callback does not implement S3 multipart, catalog-pointer commits,
restart reconciliation or database namespace identity. Those remain parent seams.

## Reader transport

Version-capable stores automatically use bounded sources through the existing
Iceberg scan API. Each data or positional-delete file chooses one store key,
captures one strong versioned HEAD, and pins all exact conditional ranges to
that descriptor. Failure never falls through to another key or repins the file.
Data-source failures latch the scan; delete-source failure refuses scan opening.
Equality deletes remain explicitly unsupported.

The convenience scan owns an 8 MiB footer/compressed-column workspace and a
1 MiB transport arena allocated once at startup. `iceberg_scan_open_with_io`
allows caller-owned finite workspace and exclusive transport arena. Startup
freezes/probes transport reservation. Each source request stays <=1 MiB even
though raw versioned S3 transport permits 8 MiB; the cache also keeps 1 MiB pages.
File size may exceed workspace; a footer or compressed chunk exceeding workspace
refuses explicitly. Metadata and decoded output have their own arena budgets.

Source workspace, transport, known vtable/allocator state and writable result
descriptors are admitted before mutation. Borrowed implementation extent is
opaque; its immutable storage must remain outside reused buffers. The metadata
arena resets per file, while copied delete paths/positions and decoded strings
remain in owned storage. Nonversioned legacy stores retain the contiguous reader;
a partially implemented version interface refuses.

## Interoperability and verification corrections

Ordinary Iceberg data files now stamp exact source field IDs in Parquet, including
rename stability and drop/add identity. Generic Parquet behavior is unchanged.
Raw footer regression checks source ID 7, rename preserving 7, and replacement
field 8 while older files still carry 7. Positional deletes retain reserved
`file_path=2147483546` / `pos=2147483545` IDs. These requirements follow the
[Iceberg specification](https://iceberg.apache.org/spec/#parquet).

The source-only large staged fixture initially supplied a frozen 4 MiB metadata
arena, smaller than the existing parser's initial 128-column/4096-chunk POD
reservation. Execution exposed `Capacity`. The corrected startup budget derives
from actual structure sizes plus alignment; the >16 MiB file, 40 distinct row
groups, all 2,621,440 exact values, 1 MiB source workspace and 4 KiB requests
remain unchanged. Reader integration separately checks an actual >16 MiB table
with a 256 KiB workspace, tiny requests and every row.

The parent real-R2 acceptance publishes a 20,978,484-byte Parquet source in three
reported multipart parts, checks a complete downloaded SHA and all rows, compares
two cached/uncached scans, applies two positional deletes and reopens. PyIceberg
and DuckDB independently verify all values and physical source field ID in both
snapshots: 2,621,440 before and 2,621,438 after. Cleanup occurs only after both
oracles succeed, deleting its ten owned keys and checking absence. This proves
standalone physical table I/O; it does not qualify a mounted HTTP proxy/catalog,
WAL-backed export projection or durable parent receipt protocol.

Incremental `crypto/sha256.h` reuses the maintained implementation used by SigV4.
It adds copied 112-byte state, bounded/alias-safe update and consuming final,
with NIST known answers and split/overflow/phase tests. The previous narrowing
before taking the remaining block minimum is corrected. No >4 GiB data run is
claimed. Signing and existing one-shot/HMAC interfaces retain their contracts.
