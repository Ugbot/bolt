# Versioned object ranges, cache and bounded Parquet source

Qualified in the existing Gestalt2 Clang/Ninja release build on 2026-10-21.
The combined gate passed 19 CTests / 193 executed GoogleTests in31.82s.
Two declared disabled network cases are excluded; actual R2 acceptance ran
separately. No Windows/Linux, performance or complete remote Iceberg deployment
qualification is implied.

## Strong object identity and conditional ranges

`ObjectStoreVT` appends optional `head_versioned` and `get_range_if_match`.
`ObjectVersion` owns an exact strong quoted ETag and object extent. Existing
operations remain available; unsupported backends return NotImplemented without
emulating a conditional request. The VT layout grows: rebuild all consumers.

S3 signs `If-Match` in canonical header order through the additive SigV4 input.
The conditional range accepts only exact HTTP206, Content-Range, ETag and body
length. HTTP412 becomes `kOsPreconditionFailed`; missing/weak/duplicate/malformed
validators, ignored ranges, truncation and wrong extents cannot publish bytes.
The current raw conditional range limit is1MiB, supplied through caller-owned
bounded scratch. Providers own admitted-error output initialization. Invalid
aliases refuse without modifying borrowed input/output spans.

R2 documents conditional and ranged GetObject support in its
[S3 compatibility table](https://developers.cloudflare.com/r2/api/s3/api/).
The [S3 GetObject contract](https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetObject.html)
defines If-Match and412. These source contracts informed the implementation;
the actual R2 test below qualifies this adapter against the configured service.

## Range cache

`object_store/range_cache.h` exposes one thread-confined cache, explicit
0..64 one-MiB pages and a separate exactly one-MiB I/O arena. It allocates its
bounded backing at init and uses no lookup/fill locks or hidden growth. The
maintained HTTP client still has its own bounded wire staging; this is not an
allocation-free HTTP or total-RSS claim.

Every logical read performs HEAD. Pages match key, strong ETag, extent and
offset; misses use If-Match and never repin or retry. A pinned expected version
refuses a changed object. Cache-off uses the same validated transport without
stored pages. Counters expose HEADs, range calls, successful origin body bytes,
requested bytes, page lifecycle and attempted/acknowledged/uncertain writes.
Failed transfers may consume wire bytes not included in successful-body counts.

PUT, conditional creation and DELETE invalidate before forwarding and preserve
actual origin outcomes. They never populate from an unverified write. Legacy
whole-object GET is forwarded and is explicitly outside the range cache.
Input/output aliases with identity, private state and resettable I/O storage
refuse before mutation. Startup outputs cannot alias the borrowed origin VT.

## Prefix adapter

`object_store/prefix.h` provides a table-relative view over a copied origin
handle. It validates bounded raw relative keys, strips and validates full LIST
prefixes, preserves optional operations/conditional outcomes, and adds no HTTP,
allocation or retries. Nested views flatten at init. The empty relative key
addresses the exact prefix object, including an S3 root directory marker.
The adapter/backend/vtable lifetimes are explicit; state is initialized at its
final address and remains immutable.

## Parquet source

`ingest/bolt_parquet_source.h` reads a pinned `ParquetByteSource` through an
exact callback, requesting at most1MiB per call. The caller owns finite footer/
compressed-column workspace and separate bounded metadata/output arenas. Total
file size is not capped at16MiB. Metadata and decoded variable data do not borrow
the reusable workspace. Each selected compressed chunk must fit the chosen
workspace; unsupported encodings or exceeded budgets fail explicitly.

The existing contiguous decoder and signatures remain, with a shared explicit
buffer-base decoder bridge. Sixteen source tests cover an actual encoded file
larger than16MiB with256KiB workspace/4KiB requests and exact values, externally
authored fixtures, projection/dictionary/nullable strings, metadata ownership,
transport/capacity failures and no partially published column descriptors.
Invalid descriptor aliases refuse before I/O; admitted failures publish zero
rows and leave output column descriptors unchanged.

This source abstraction is not yet wired through every Iceberg data/delete-file
reader, and the staged remote Parquet writer remains separate work.

## Actual R2 cache acceptance

The Gestalt2 fixture uses a unique prefix in an existing shared bucket and a
2,097,325-byte known payload. One selected case passes in9.087s, zero skips.
Across two reads, cache-on makes three range GETs/2,097,325 body bytes;
cache-off makes six/4,194,650. The warm read adds one freshness HEAD and no body
GET. Exact payloads also pass after external same-size replacement and a
write-through update; an old pin refuses; conditional replacement refuses
without changing data; DELETE and subsequent absence are verified. Only the
run-owned object is deleted. This tests the maintained object-store adapters,
not an HTTP proxy mount or complete remote Iceberg catalog.
