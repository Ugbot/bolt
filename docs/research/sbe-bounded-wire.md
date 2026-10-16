# Bounded SBE wire primitives

Date: 2026-10-13. This is the first generic Bolt SBE slice. It has no CME
dependency, allocator, IO, exception or RTTI path. It is usable by a generated
codec or a bounded descriptor interpreter. It does not yet make SBE equivalent
to Bolt's existing Avro/Protobuf/Parquet ingestion interfaces.

Tracking: G2FEAT-36 (wire slice), G2FEAT-44 (format performance). Initial native
strict/ASan/UBSan tests pass, including existing ByteReader and CME framing.
Profiling identified missed inlining; using Bolt's portability annotation on
SBE and byte-reader helpers reduced the new decoder's warmed median from about
21.2 to 3.73 ns/message. The independent checked direct decoder is about
2.3 ns/message: a 1.61–1.64x ratio, still outside the proposed 1.20 gate. These
are cache-resident template-50 microbenchmark windows, not individual tails or
general-format parity. Gestalt's `docs/research/cme-sbe-platform-20261013/validation`
stores raw pairs, source/binary/compiler hashes and DTrace run
`0bd58d1d-bcd3-4c31-a19c-b8f4aeeefadb`.

## Contract

`bolt/wire/sbe_writer.h` adds the schema-neutral write side. Caller-owned
`Writer`/`WriteBlock` support checked scalar conversion, common headers,
versioned fields, explicit optional presence/nulls, fixed bytes, group dimensions
and variable data. Offsets, lengths, versions and counts are checked before
narrowing. Each failed operation preserves bytes and cursor/result descriptors;
whole-message atomicity requires the generated encoder's complete preflight.
No XML, allocation or exchange-layout assumptions enter these writers.
`tests/test_sbe_writer.cpp` contains 18 literal-byte and refusal/alias cases.

Include `bolt/wire/sbe.h`, `sbe_fields.h`, and `sbe_groups.h`.
Everything is in `bolt::wire::sbe`; the existing `ByteReader`/`ByteView` own
the bounds/lifetime seam. Input storage is caller-owned, immutable, alive for
every borrowed view, and is exactly one transport-framed message. Failed
operations preserve outputs and cursors. Successful variable/group operations
advance only the bytes they explicitly consumed.

| Capability | Implemented contract |
|---|---|
| Primitive fields | All SBE integer widths, float/double, compile-time little/big endian, unaligned-safe |
| Header | Common four-uint16 header; explicit schema/template/version-range checks |
| Fixed block | Acting block length sets tail position; typed offset and byte-slice access |
| Evolution | `Absent` for `sinceVersion` beyond acting version; truncation is an error |
| Optional | Caller-supplied sentinel, distinct `Null`; floating NaN sentinel handles all NaNs |
| Groups | Packed schema-selected block/count widths, caps and fixed-byte lower-bound validation |
| Nested groups | Explicit schema-order traversal; consume nested/data tails before next outer entry |
| Variable data | Schema-selected unsigned length type, explicit compile-time cap, borrowed bytes |
| Completion | Caller invokes `finish_message` after all schema fields; unknown tail fails |

SBE's field encodings and group dimensions come from its XML schema. Block
length excludes nested groups/variable fields; it is not a message length.
Byte order applies to scalar encodings. Constants occupy no wire bytes;
composites, enum/set membership and semantic constraints require schema
information. [FIX SBE standard](https://www.fixtrading.org/standards/sbe/),
[schema specification](https://github.com/FIXTradingCommunity/fix-simple-binary-encoding/blob/master/v1-0-STANDARD/doc/04MessageSchema.md).

`open_group` takes a schema traversal depth starting at one. It performs no
recursion or automatic unknown-group skipping. A generated adapter must supply
the correct depth and bound its own traversal. All loops inside these headers
are compile-time scalar widths (at most eight bytes). Message cap defaults to
16 MiB and is a template argument; per-group count/depth and data caps are
explicit template arguments. These are generic library bounds, not negotiated
exchange limits. An exchange adapter must choose its actual limits.

## Independent evidence and limitations

The [fixture provenance](../../tests/data/sbe-fixtures.md) records CME's
historical version-9/template-50 packet and the independently specified
synthetic nested-group case. Tests decode the entire limits/banding body,
including the nullable price, not merely its framing. Every truncated prefix
of both messages must fail without publishing a partial result. Additional
tests cover identity/version refusal, padding, empty groups, malformed counts,
depth/length caps, big endian, floating-point values/null, and borrowed slices.

`test_bolt_sbe` links only `bolt::core`. `bench_sbe_decode` alternates 400 pairs
after warmup, using 1024 varying 54-byte messages and 128 rounds per window.
It emits every paired window and checksum, then p50/p99 of window-average
ns/message. These are **not individual-message tail percentiles**. Both paths
check bounds, IDs/version and all the same fields. The reference uses
independent `memcpy` loads and explicit schema offsets. It is hand-written,
not Aeron's generated codec. A generated-code comparison and wider message
matrix are still required before a format-parity performance claim.

## Remaining format parity work

1. Bounded cold-path XML/XInclude loader, validated schema/IR and hash pin;
   constants, composites, enum/set domains, field offsets/version rules and
   custom header/dimension layouts. Reject unsupported schema constructs.
2. Generated typed readers and writers with exact round-trip/differential
   fixtures, negative semantic tests, model-independent representation.
3. `bolt/ingest` SBE projection into reusable column buffers; Chukonu source
   and pychukonu dataframe integration, nullable integer/decimal fidelity.
4. Fuzz corpus/differential codecs, SIMD/batch projection profiling and
   allocation instrumentation. No bytes/rows lost under explicit backpressure.

Runtime XML parsing or per-message descriptor allocation is not part of the
proposed fast path. Descriptor dispatch and generated code should be separate
compile-time strategies, selected on whole-workload evidence.
