# Owned Arrow IPC reader (G2FEAT-155)

The reader accepts Arrow IPC streams and `ARROW1` files into ordinary typed
`BoltBatch` columns. All production parsing, I/O and ownership are Bolt code.
There is no Arrow or FlatBuffers runtime dependency, generated parser, Python
row decoder, or replacement internal batch format. This is a selected flat
format subset, not complete Arrow IPC qualification.

## Format pin and independent evidence

Selected specification: Apache Arrow tag `apache-arrow-21.0.0`:

| Primary source | SHA-256 |
|---|---|
| [Schema.fbs](https://raw.githubusercontent.com/apache/arrow/apache-arrow-21.0.0/format/Schema.fbs) | `e46bc54638a2de387113ae658861428d705fa90cadcc1d4def6113d9e406d6f8` |
| [Message.fbs](https://raw.githubusercontent.com/apache/arrow/apache-arrow-21.0.0/format/Message.fbs) | `32a1464cede1b90d1351faf65726d465d95fbf3a85bd96e2d7222ac979fe20e4` |
| [File.fbs](https://raw.githubusercontent.com/apache/arrow/apache-arrow-21.0.0/format/File.fbs) | `646bf4e173192900f64c08823b67f5e0a8c2cbcd2d48c83f8b42faa864ab8087` |

The [official columnar specification](https://arrow.apache.org/docs/format/Columnar.html#serialization-and-interprocess-communication-ipc)
defines optional EOS, prefix padding, null bitmaps and physical buffer layout.
The pinned [official reader](https://github.com/apache/arrow/blob/apache-arrow-21.0.0/cpp/src/arrow/ipc/reader.cc)
confirms file block offsets identify encapsulated messages, metadata lengths
include the prefix, and bodies begin at `offset + metadata_length`. These are
protocol references only; their implementation is not copied or linked.

`tests/data/arrow_ipc/generate.py` prepares small independent reference fixtures
using **test-only PyArrow 21.0.0**. Values are authored mathematical examples,
not authentic feed captures. `manifest.json` pins every generated file and the
generator. Modern/V4/legacy stream and file fixtures contain 23 exact flat columns, a
nonempty batch, an empty batch, and a repeated nonempty batch. Additional
reference fixtures cover empty schema, dictionary, nested, negative-scale
decimal and Zstandard-compressed bodies. Native expected values are separately
asserted in `test_bolt_arrow_ipc_reader.cpp`, including UInt64 maximum, integer
extrema, half-float NaN bits, negative zero, binary zero/255, decimal128 limbs,
non-microsecond temporal units/timezone, nulls and variable offsets.

## Memory and lifetime contract

1. Call `arrow_ipc_reader_workspace_size(limits, bytes)`; allocate an aligned
   (`kIpcReaderAlignment == 64`) caller-owned workspace of that exact bound.
2. Open either an immutable borrowed byte span or a path. The path opener uses
   `bolt::io::mapped_file_open`, with lexical close through the reader. Mapping
   does not copy the full file into process-owned decompression buffers.
3. `arrow_ipc_reader_schema` returns an exact structural schema view. Names,
   timezone and key/value strings borrow immutable input; descriptor arrays
   borrow workspace. The caller must retain both until close.
4. `arrow_ipc_reader_next` returns a reader-owned, read-only `BoltBatch`. The
   copied payload and columns remain valid until the next **successful** next
   or close. An error or EOF leaves the previous batch, output pointer and
   cursor unchanged. Copy output before another successful next if retention
   is required.

All descriptor arrays, layout scratch, column descriptors and output payload
are reserved once from workspace. There is no allocator, Arena allocation or
Arena growth in next. Output copies use checked byte counts and 64-byte aligned
slots. The workspace budget includes each array and alignment, not just body
bytes. Limits independently bound schema fields, total schema key/value pairs,
metadata bytes per message/footer, batch rows/body, file batch count and copied
output bytes. Zero counts permit an empty schema or empty batch. Mapping is
startup I/O; page faults remain OS-managed and are not a latency guarantee.

Caller exclusively owns the reader. Limits, workspace and argument shells must be disjoint. Input must stay immutable, mapped files
must not be changed/truncated concurrently, and workspace/input/reader/output
argument shells must not overlap. No concurrent reader calls. An open reader
must not be copied; close is explicit and idempotent. Open refusal preserves
reader ownership but admitted workspace may have been scratch-written. Closed
reader schema/next return `Closed`.

## Admitted values and schema boundary

- Signed/unsigned 8/16/32/64-bit integers and half/single/double floats retain
  exact widths and bits. No conversion through double or signed Int64.
- Boolean Arrow bitmaps become Bolt byte values; validity remains LSB-first.
  Null counts are checked against actual bitmap bits before publication.
- Binary uses Bolt `VarBinary` and a reader-owned offsets child; Utf8 uses
  ordinary Bolt `StringView` plus copied overflow backing. Signed 32-bit offsets
  must be nonnegative, monotonic and bounded, including null slots. Valid Utf8
  values are checked using the owned strict UTF-8 code-point primitive; null
  payload is uninterpreted. Empty strings differ from nulls.
- Date32/day and Date64/millisecond retain their native Bolt types.
- Time, Timestamp and Duration retain exact Int32/Int64 physical lanes. The
  accompanying descriptor carries the Arrow logical tag, unit and timestamp
  timezone. Consumers must use that descriptor; these are not implicitly
  relabelled as Bolt microsecond Timestamp/Duration or rescaled.
- Decimal128 uses actual Bolt Decimal128, precision 1..38, scale 0..38; exact
  signed precision/scale also remain in the descriptor. Negative/out-of-Bolt
  scales and other decimal widths are explicitly unsupported. No binary
  downgrade or floating conversion.
- Fixed-size binary widths 1..16 use Bolt's existing 16-byte zero-padded slots,
  `fixed_width`/schema `fixed_size` and exact descriptor byte width. Larger
  widths are explicitly unsupported.

Schema fields carry Arrow tags and first-child/child-count slots so later
nested decoding can extend the same schema surface. This first slice admits
only leaves. Names longer than Bolt's field-name capacity, or names containing
embedded NUL, are unsupported rather than truncated. Schema and field custom
metadata are retained verbatim as length-bearing strings, including embedded
NUL values. File footer custom metadata is exposed separately.

## Framing and refusal

Modern continuation framing and legacy 4-byte framing are accepted with
aligned encapsulated metadata/body. Metadata versions V4 and V5 and
little-endian bodies are admitted. EOS is optional for a complete immutable
span; exhaustion precisely at a message boundary is EOF. Partial prefix,
metadata or body is `Truncated`. Bytes after an explicit EOS are `BadInput`.

Files validate both magics, signed footer length, footer schema equality and
version, dictionary absence, bounded record-batch index, alignment, metadata
and body lengths, and the final EOS/clean boundary. The admitted file profile
has contiguous ordered embedded batches and matching footer index. Arbitrary
gaps/reordered blocks are `Unsupported`; entries outside the message region
or inconsistent message sizes are refused. A footer cannot authorize bytes
outside the embedded stream. Batch payload/node/offset/null validation occurs
before output is copied. FlatBuffers references, signed vtable offsets,
object sizes, vectors and strings are explicitly bounded without a runtime
FlatBuffers dependency. The verifier also enforces format-relative scalar/table/
vector alignment and validates UTF-8 in field names, timezones and key/value
metadata using the owned UTF-8 decoder. Borrowed input itself need not be aligned;
all wire loads remain byte-based.

Nested, dictionary, compressed, large-offset and view arrays, union/tensor
messages, unknown schema features, unsupported metadata versions and big-endian
bodies return `Unsupported`. They are not skipped. Corruption/limits/unsupported
next calls retain the exact next-message position and can be inspected/retried;
there is no silent partial success. Message-level application custom metadata
is not currently exposed as a batch API; schema/field/footer metadata is.

## Verification handoff

Root owns native CMake registration and serialized build/test execution. Add
three production TUs to the existing owned ingest target:

- `src/ingest/bolt_arrow_ipc_reader.cpp`
- `src/ingest/bolt_arrow_ipc_reader_schema.cpp`
- `src/ingest/bolt_arrow_ipc_reader_batch.cpp`

Register isolated `test_bolt_arrow_ipc_reader`, linked as the existing IPC writer
test plus GTest main, with `BOLT_TEST_DATA_DIR` pointing to `tests/data`. Its
heap-counter replacement is isolated to this executable. Tests cover official
reference-generated file/stream values and ordinary mapped opener, empty
schema/batches, unsupported cases, all truncation offsets through the first
batch, metadata/rows/output/workspace budgets, root/footer/buffer/null/offset/
UTF8 corruption, alias refusal, exact retry after refusal, previous payload
preservation and allocation-free repeated next. Native/platform/performance
results must be recorded by the serialized root gate; author has not run native
builds or tests. IPC writing and C Data/C Stream lifetime/export integration are
separate work; this reader does not change their capabilities.
