# Iceberg publication with an uncertain acknowledgement

Small conditional S3 PUTs can commit before their HTTP reply is lost. Transport
failure, HTTP 409, and server errors therefore return `kOsWriteUncertain`;
an acknowledged HTTP 412 remains `kOsExists`. No automatic retry is performed.

Iceberg maps uncertain conditional manifest, manifest-list or metadata writes to
`CommitError::kUncertain`. The handle becomes sticky: subsequent mutations,
expiry and retention refuse, and cleanup preserves every candidate object that
may be referenced by the publication. Definite conflict cleanup is unchanged.

`table_uncertain_commit` copies a bounded receipt containing object kind, full
store key and canonical location, table UUID, exact SHA-256 and byte length,
base/candidate metadata versions and snapshot ID. `table_create_ex` exposes the
same evidence when initial metadata creation is uncertain. The legacy create
interface still returns false and a null handle on failure. Overlong receipt
identity is refused before conditional remote I/O.

Reconciliation is external: establish exact object/table/version identity and
candidate bytes before reopening a writer. HEAD size, one absence observation,
or blindly retrying the mutation is insufficient. Receipts are copied in memory;
this interface does not add a durable reconciliation journal.

The path-based reader treats `version-hint.text` as advisory and ranks a complete
bounded metadata listing even when the hinted version exists. This discovers
accepted commits whose hint update was never acknowledged, including retained
version gaps. Hint-object and listing I/O errors fail closed. An explicit
catalog-provided metadata open remains authoritative and does not list.

## Verification on 2026-10-21

The existing Gestalt2 Clang/Ninja release build passed 13 deterministic
`test_bolt_iceberg_uncertain_commit` cases, 16 S3 upload cases including actual
HTTP acknowledgement loss, 12 real-reader cases, nine ordinary writer cases,
and four concurrent commit cases. Exact reopened row values and positional
deletes discriminate stale metadata reads. The initial run exposed two stale
hint failures; both pass after the read discovery fix. Independent source
review found no required blocker in this scope.

This verifies local fault injection and the maintained S3 adapter. Large remote
Parquet transport, a complete remote catalog/proxy/cache deployment, durable
reconciliation, Windows/Linux and performance gates remain separate work.
