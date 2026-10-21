# Exact immutable Iceberg metadata loading

`lakehouse/iceberg/metadata_io.h` exposes `metadata_load_exact` for a
catalog-selected metadata object. Selection and version acquisition belong to
the caller. The loader never discovers a newer path, repins, retries, or falls
back to an unconditional GET.

## Identity and lifetime

The input separates ObjectStore root, raw relative key, absolute metadata URI,
table location, table UUID and the strong version/extent. Metadata may be
outside the table location. URI mapping uses the existing S3 percent encoder;
the first implementation admits a documented canonical S3 subset only.

Each conditional range is exact and at most the configured request size and
1 MiB. Distinct frozen caller arenas retain metadata/parser arrays and recycle
provider scratch before each request. The caller keeps those arenas and the
immutable provider alive. Outputs publish together only after full transport,
identity, parser and SHA-256 success. Scratch may be consumed on failure.

The maintained parser exposes raw JSON slices. This API conservatively refuses
escaped root keys and escaped selected identity strings, including escaped
aliases after literal identities. It does not implement general JSON identity
normalization. Existing parser workspace is about twelve token bytes per input
byte in addition to retained JSON; callers must admit that finite budget.

## Qualification

Seventeen native regressions cover exact counters/schema/hash, separate table
and store namespaces, pinned replacement refusal, short and failed ranges,
per-request scratch reuse, aliases, unchanged outputs, raw identity ambiguity,
and a real metadata document larger than 16 MiB. In the controlled regression
check, removing escape refusal fails two of the three new cases; escaped-only
identity keys were already refused by the old missing-identity check.

The Gestalt2 consumer qualified the same implementation against actual R2 with
127-byte metadata ranges, exact UUID/location/snapshot/sequence, copied strong
identity and full digest. Its physical table also passed all-value PyIceberg
and DuckDB checks before owned-fixture cleanup. These are macOS component
checks. Catalog authority, durable bindings, live synthetic remote/local
stitching, Windows/Linux and performance acceptance are separate work.

## Parsed current-snapshot presence

`Metadata::current_snapshot_present` preserves whether the original JSON has a
numeric `current-snapshot-id`; numeric signed `-1`, zero and other negative IDs
are present. Missing and JSON null are absent. The copied Metadata returned by
`metadata_load_exact` retains this distinction. Consumers selecting exact
snapshots must compare signed IDs by equality, require membership for every
present current ID and selected ID, and use presence rather than the legacy
`current_snapshot_id == -1` sentinel to establish absence.

Writer serialization and the ordinary parser's legacy membership behavior stay
compatible: old owned empty metadata emits numeric `-1`. That old sentinel is
still readable by the ordinary parser; an exact binding consumer can refuse it
when no matching snapshot exists and require null/omitted for explicit absence.
The provenance flag describes parsed input and does not change writer mutation
or serialization semantics.

The [Iceberg snapshot ID specification](https://iceberg.apache.org/spec/#assignment-of-snapshot-ids-and-current-snapshot-id)
notes the Java V1/V2 `-1` convention for no current snapshot. Gestalt2's durable
remote binding component deliberately refuses numeric `-1` with no matching
listed snapshot; it admits explicit null/omitted absence or explicit numeric
`-1` with matching listed snapshot. This strict component policy does not
change Bolt's legacy writer/parser compatibility.
