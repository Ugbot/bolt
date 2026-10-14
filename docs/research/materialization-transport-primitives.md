# Bounded response heads and explicit TLS I/O deadlines

The authoritative-owner sink needs a complete response body and caller-owned
buffers. Bolt now supplies two narrow reusable primitives; neither establishes
storage durability, validates application receipts or authorizes a write.

[`http_response_head_parse`](../../include/bolt/net/bolt_http_response_head.h)
parses an HTTP/1.0 or HTTP/1.1 response head into borrowed slices without
allocation or socket IO. It returns incomplete, invalid or complete, caps the
head at 8192 bytes and headers at 32, and checks status, CRLF, field names and
values. Duplicate Content-Length, duplicate Transfer-Encoding and their combined
presence refuse. An overflowing Content-Length refuses. Body framing is a
separate caller decision: the parser exposes transfer encoding, while the first
materialization consumer explicitly refuses it and requires Content-Length.
The input buffer must outlive every returned slice. Existing `http_send` and
generic REST behavior are unchanged.

[`tls_send_all_timeout` and `tls_recv_some_timeout`](../../include/bolt/net/bolt_tls.h)
accept an explicit 1..60000 ms per-call steady-clock deadline; zero, overflow,
unconnected sockets and unrepresentable signed IO lengths fail with -1. They
reuse the existing TLS loops and verification behavior. Positive progress and
WANT_READ/WANT_WRITE waits consult the same deadline. Existing send/receive
functions retain their 5000 ms default. `TlsConfig::recv_timeout_ms` is not
retained by the live socket; consumers needing an explicit IO budget must call
these APIs rather than assume that field affects legacy calls. TLS-disabled
builds return failure, matching the existing primitive's behavior.

These are cooperative IO bounds, not preemption. An individual OpenSSL call
can consume CPU before the next deadline check. The IP-based TLS connect API
uses its existing shared nonblocking-connect/handshake deadline; these additions
do not change it or add query-cancellation polling inside handshake. Upstream
hostname resolution is synchronous and outside that deadline. Consumers using
a separate blocking plain-connect path retain its OS-controlled wait. Caller
retry/total deadlines, cancellation between calls, response body capacities and
credential ownership remain consumer responsibilities. OpenSSL still allocates
per connection. No end-to-end cancellation latency or HFT qualification follows
from these primitives.

## Scoped local evidence

Darwin 25.6.0 arm64; Apple Clang 21.0.0 (clang-2100.3.34.2), target
arm64-apple-darwin25.6.0. Existing superproject `build/release`, Release `-O3 -g`.

```text
cmake --build build/release --target test_bolt_http_response_head test_bolt_tls test_materialization_owner_sink test_materialization_send_timeout -j2
Final exit 0.
ctest --test-dir build/release -R '^(test_bolt_http_response_head|chukonu_materialization_owner_sink|chukonu_materialization_send_timeout)$' --output-on-failure
3/3 passed; 1.13 seconds. Response-head parser: 2 internal cases, 0.33 seconds.
GTEST_FILTER='BoltTls.CtxInitAndFree:BoltTls.PeerResetDoesNotRaiseSigpipe:BoltTls.Explicit*' ctest --test-dir build/release -R '^test_bolt_tls$' --output-on-failure
1/1 passed; 4 internal cases; 0 skips; 0.44 seconds.
explicit TLS receive timeout: budget_ms=40 elapsed_ms=42
explicit TLS send timeout: budget_ms=40 elapsed_ms=41
```

The [response-head tests](../../tests/test_bolt_http_response_head.cpp) cover
every truncation of a valid head, borrowed-field lifetime, malformed status and
CRLF, ambiguous lengths/encodings, length overflow, head/header limits and the
separation of header parsing from body framing. The
[TLS tests](../../tests/test_bolt_tls.cpp) reuse the existing owned in-process
certificate/server setup, including the legacy peer-reset regression. A silent
handshaken peer and explicitly small socket buffers prove actual configured
receive/send timeout behavior. Self-signed trust bypass exists only in the local
test setup; the new production APIs do not bypass verification. The public
internet TLS case was excluded. Socket access was approved and no socket check
was skipped. These timing values are single correctness observations, not
latency targets or benchmark percentiles. Windows/Linux execution gates remain
open; this receipt proves macOS behavior only.

Raw receipts: `/tmp/g2-materialization-sink-build-green.log`,
`/tmp/g2-materialization-sink-ctest.log`,
`/tmp/g2-materialization-sink-lasttest.log`,
`/tmp/g2-materialization-tls-ctest.log`,
`/tmp/g2-materialization-tls-lasttest.log`. Remaining build warnings are inherited
Bolt column/types warnings; no new parser/TLS warning remains.
