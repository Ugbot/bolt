# Bounded S3 multipart upload

`bolt::lakehouse::s3_upload_stream` uploads a known-size object with sequential,
bounded S3 multipart requests. The API is in
[`s3_upload.h`](../../include/bolt/lakehouse/object_store/s3_upload.h); the
existing `ObjectStoreVT` layout is unchanged. The S3 `put_object` adapters keep
single PUT for small objects and use multipart for objects larger than 16 MiB.

## Source and lifetime

The caller supplies `S3UploadSource::read_part(context, offset, length, data)`.
Calls are synchronous and forward-only at increasing offsets. A successful
callback returns exactly the requested readable bytes; a short read must be an
error. The returned span is borrowed and must stay immutable until the next
callback or until `s3_upload_stream` returns. A file source can reuse one
preallocated part buffer. The contiguous `put_object` adapter points directly
at the caller's existing body and does not copy the whole object or allocate a
second part buffer.

Keep the store configuration immutable and the callback context alive for the
whole call. Each concurrent upload needs its own source context, result and
arena; there is no shared upload workspace. The callback must have its own
finite bound. Cancellation is checked between phases and after callbacks, not
inside a running callback or HTTP request.

## Geometry and memory budget

The default part size is 8 MiB. Admitted part sizes are 5–16 MiB, with a
maximum of 10,000 parts; the final part may be smaller. The resulting object
limits are 78.125 GiB at the default and 156.25 GiB at 16 MiB. Unknown-length
sources and geometry exceeding the configured `max_parts` are refused before
remote effects. Empty objects use an ordinary empty PutObject.

The caller provides an `Arena` for bounded upload metadata and completion XML.
Those allocations are reserved before multipart initiation. At the 10,000-part
limit, part receipts use 1,280,000 bytes and completion XML uses up to 8,520,064
bytes, plus allocator alignment/backing. HTTP control-response storage is
capped at 64 KiB and reused. Budget the arena, the at-most-one-part source
staging buffer, and transport buffers independently for every simultaneous
upload. The configured S3 store must remain immutable until all its calls
return.

## Request lifecycle and outcomes

The implementation sends one CreateMultipartUpload, sequential UploadPart
requests, and one CompleteMultipartUpload. It signs ordinary SigV4 requests,
preserves canonical query ordering and endpoint authority (including a
nondefault port), and does not retry implicitly. Upload IDs are decoded from
the initiation XML and encoded once for later requests. ETags are opaque, have
a bounded size, and are XML-escaped in the ordered completion manifest; they
are not treated as MD5 hashes. Only the documented flat result XML shape,
optional XML declaration and default S3 namespace are accepted.

`S3UploadOptions::if_absent` requests conditional creation on multipart
completion with `If-None-Match: *`. HTTP 412 is the definitive
`kOsExists` result. Other conflict responses are not retried or recreated.
The existing ObjectStore adapters keep zero/small objects on PutObject;
the typed streaming API uses multipart for every nonempty source.

Cancellation or source/part failure before completion attempts one abort.
`S3UploadResult` keeps the primary status, phase, HTTP status, part/byte
progress, upload ID and separate abort status. A failed abort marks remote state
uncertain. Once completion may have been sent, any result without a fully
verified completion response is `kOsWriteUncertain` (except definitive 412).
Do not blindly retry, recreate, or abort that object: completion may have
committed even when its response was lost or malformed. Reconcile the key using
application-specific identity/content metadata before deciding whether a new
write is safe. A lost initiation response can leave an unidentified orphan
multipart upload. The one-shot `s3_abort_multipart` helper does not prove an
in-flight remote part has finished; orphan reclamation may require a separate
bucket lifecycle or ListParts policy.

## HTTP and ABI integration

The HTTP client uses fixed-capacity POD request and response structures. Requests
must be zero-initialized before filling. Its `HttpHeader` and `HttpRequest`
layouts changed as part of the framing/timeout work, so rebuild every consumer
of the Bolt HTTP headers together; do not mix old and new binaries. HTTP
response heads are capped at 8 KiB, wire staging is bounded by the body budget
plus 64 KiB of framing overhead, and request/response body capacity remains
bounded. Per-request timeout is 1–60,000 ms (zero selects 10,000 ms); synchronous
DNS resolution is outside that socket deadline.

## Scope and evidence

The final scoped Bolt CTest run in `build/release` passed 14/14 CTest targets:
89 GoogleTest executions, 87 passing and two Azure/GCS cases skipped; the S3
MinIO cases ran.
The native multipart suite covers request signing and framing, geometry,
conditional completion, cancellation, malformed/truncated responses, abort
outcomes, uncertainty and concurrent-call isolation.

The actual native SST MinIO fixture has separately round-tripped a 34,508,928
byte MBLT SST with 32,768 rows and 256-wide vectors, verifying the downloaded
data bits and SHA-256 prefix `a8228982`. The strengthened service-reported
multipart part-count oracle is still pending; this is not final
external-service qualification. G2ICE-249 remains
open until that evidence is recorded. Windows/Linux, broader S3-compatible
services, TieringManager activation, automatic crash recovery, and throughput or
tail-latency qualification are outside these results.
