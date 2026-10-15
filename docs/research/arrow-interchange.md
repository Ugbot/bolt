# Owned Arrow C Data interchange

G2FEAT-156, 2026-10-15. `bolt::interop` supplies canonical Arrow C Data/C
Stream ABI declarations, a bounded immutable snapshot owner, and bounded flat
copy ingress. It depends on Bolt core only: no libarrow, PyArrow, Python,
FlatBuffers runtime, or query/session dependency. IPC byte/file parsing lives
separately in [the ingest reader](arrow-ipc-reader.md).

## Immutable export ownership

`interop/arrow_owner.h` admits an explicit total owner/arena/descriptor budget,
field cap and fixed lease count at creation. The trusted cold builder allocates
buffers in the supplied no-growth arena, publishes once, then relinquishes
mutable access. Publication copies format/name/topology and checks supported
flat layouts, required buffers, full backing spans and variable offsets.
Builder responsibility includes valid allocation extents, alignment, payload
encoding and declared decimal precision. Backing containment is not proof of
an individual allocation's extent.

Fresh schema/array exports and materialized single-batch streams borrow shared
immutable buffers using independent descriptors. Moving a child retains the
backing until that child releases; schema, array, stream and original owner
lifetimes are independent. Distinct leases can release across threads without
Python/GIL access. Stream callbacks and access to one descriptor are caller
serialized. Release callbacks allocate nothing; fixed-pool exhaustion refuses
without consuming a stream batch or mutating output shells.

Nested/dictionary/metadata-bearing exports and unsupported formats refuse.
The C Stream represents one already materialized bounded batch, including an
empty batch, then EOF; it does not claim incremental unbounded query execution.

## Exact bounded copy ingress

`bolt_arrow_import.h` borrows a schema and each foreign array for the duration
of the call; the caller owns and releases foreign roots. Copy chunks into a
scratch arena, then concatenate once into a distinct destination arena only
after the consumer has observed clean stream EOF. Append errors poison the
state. Finish publishes output only after all columns succeed; failures can
consume destination scratch space and do not promise rollback of its arena.

Limits admit batches, rows, columns and copied bytes, including transient chunks,
final copies, metadata and alignment slack. This is not a cap on already-reserved
arena backing pages. Both arenas require default allocation alignment at most
64 bytes; larger alignment refuses before allocation because it exceeds the
charged padding contract.

Flat signed/unsigned integer widths, Float32/64, bool, UTF8/Binary (32/64-bit
offsets compacted to owned Int32), dates, timezone-free timestamp/duration and
Decimal128 are supported within documented bounds. Dictionaries, nested types,
extensions, timezone and unsupported decimal shapes refuse. Exact decimal
precision and temporal kind/unit live in `ImportState.fields`; consumers must
retain those sidecars when adapting the physical Bolt batch. Non-microsecond
temporals use raw integer lanes without rescaling.

Foreign pointers follow the C ABI readable-memory contract, which supplies no
buffer lengths. Checked offsets/arithmetic cannot prove external allocations
exist or are sufficiently large. UTF8 payload validity and decimal precision
compliance are trusted producer properties. IPC byte parsing has independent
length/UTF8 validation and must not inherit that trust assumption.

## Verification scope

`test_bolt_arrow_owner`, `test_bolt_arrow_import`, `test_bolt_arena_ranges`,
`test_bolt_arrow_ipc` and `test_bolt_arrow_ipc_reader` pass in the local Release
Clang build. Coverage includes moved children, independent/concurrent lease
release, all temporal units with exact extreme tick bits and empty output,
full backing admission, resource refusal, exact integer widths, multibatch
ownership, null slices and no-allocation IPC next calls.

Chukonu adapters have separate native/Python protocol tests, including tests
that block PyArrow imports and independent test-only reference consumers.
Spark migration, ordinary IPC Source registration, performance comparisons,
Linux sanitizer/container and Windows/MSVC qualification remain separate gates.
