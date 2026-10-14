# Schema-driven bounded SBE decoding

Implemented 2026-10-14 for G2FEAT-37. The reusable cold compiler is
`ai-grind/src/devtools_mcp/sbe/`: `compile_schema(bytes, budgets=..., expect_sha256=...)
-> SchemaIR`, `generate_cpp(ir, namespace=...) -> str`, and
`python -m devtools_mcp.sbe compile`. Python and ai-grind's existing lxml are
authoring dependencies only. Generated C++ includes Bolt headers and the C++20
standard library; it has no XML, heap ownership or schema lookup on the wire path.

The [FIX version-1 schema specification](https://github.com/FIXTradingCommunity/fix-simple-binary-encoding/blob/master/v1-0-STANDARD/doc/04MessageSchema.md)
defines byte order, type references, field offsets and ordered group/data tails.
This implementation compiles the explicit subset below. It is neither a complete
XSD validator nor a current CME production-schema compiler. Reconstructed
historical XML is explicitly labelled and never represented as exchange XML.

## Supported and rejected layouts

| Feature | Implemented contract |
|---|---|
| Primitive types | char; signed/unsigned 8/16/32/64-bit integers; IEEE float/double; LE and BE |
| Fixed arrays | Bounded typed indexed getters, including char arrays and embedded NUL; invalid index preserves output |
| Composites | Inline/referenced/nested fixed composites; resolved cycle detection; explicit nonoverlapping offsets; typed leaf getters such as `get_price_amount_mantissa` |
| Enum | Unsigned/char backing, named typed enum constants, known-domain checking and value sinceVersion checking; optional field null sentinels |
| Set | Unsigned backing, checked bit indices, scalar mask and named bit getters; unknown mask bits are retained |
| Constant | Scalar encoding constants and constant enum valueRef; zero wire bytes; absent before sinceVersion |
| Optional | Scalar integer/float/char and enum null; default/explicit sentinels; distinct `Absent`, `Null`, `Truncated`; failed getters preserve destination |
| Version | Schema/type/field/value version metadata; getters check acting version; minimum fixed block depends on active fields; current version additionally requires declared reserved blockLength; future versions rejected |
| Deprecated | Retained metadata, not removal from the wire layout; field metadata emitted alongside offsets/sinceVersion |
| Header | Common packed four-uint16 header only; schema/template/version checked before callbacks |
| Groups | Packed unsigned <=32-bit blockLength/count dimensions; nested groups followed by entry data in schema order; bounded count/depth and borrowed entry blocks |
| Variable data | Packed unsigned length then variable byte/char payload; compile-time payload cap; embedded NUL retained |
| Padding | Explicit field/composite offsets and reserved blockLength; tail begins at the received acting blockLength |
| Unsupported | Custom/padded headers or dimensions/data prefixes; optional composites/arrays/sets; constant arrays; changing wire members inside a composite across versions; any unknown node/attribute/layout; XML includes |

Versioned fixed-field additions must append, so an older decoder view never
uses a newly shifted existing offset. A whole composite field may be introduced
in a later version, but versioned changes to its internal wire members are
rejected. Each generated function is capped at 70 lines; oversized blocks/tails
fail generation explicitly and must be split. Character encoding, epoch,
time-unit and semantic annotations do not perform transcoding or unit conversion.
Scalar min/max metadata is validated for representability at compile time;
typed scalar getters decode wire values without business-range checks.

## Cold input and IR contract

The compiler receives one immutable bytes object. The CLI reads at most the
file cap plus one byte and requires an expected SHA-256; mismatch leaves an
existing output untouched. The resulting frozen dataclasses contain only
immutable tuples/scalars and resolved encodings, with schema identity, version,
byte order and the exact XML hash. Generated headers preserve identity/version
and hash as compile constants. `generate_cpp` consumes compiler-produced IR;
it rejects unsupported emission cases rather than omitting them.

Default budgets: 2 MiB XML; 32,768 XML nodes; 4,096 named types; 1,024 messages;
4,096 schema groups; 16,384 members; nesting 8; 65,535 array elements and fixed
block bytes; 16 MiB wire message; 65,535 group entries; 1 MiB variable payload.
The fixture command tightens group entries to 255 and payload bytes to 1,024.
Integer representation limits and positive budgets are validated before parsing.

The hardened lxml parser sets `resolve_entities=False`, `load_dtd=False`,
`no_network=True`, `huge_tree=False`, and parses in-memory bytes. Every DTD is
rejected; namespace/node/attribute allowlists reject XInclude and unknown schema
extensions. No resolver, network fetch, include expansion, schema download or
automatic decompression is invoked. These options follow the
[lxml parser documentation](https://lxml.de/parsing.html#parser-options).

## Generated runtime contract

`decode_<Message>(ByteView, visitor)` validates a bounded message and calls
`visitor.on_message_<Message>(view)`, then each nested entry/data callback in
wire order. Every callback returns `bolt::wire::sbe::Status`; the first failure
propagates immediately. The decoder rejects an unconsumed tail. Enum domains
are checked when the typed getter is called; traversal alone validates structure,
not every field value. All views borrow caller storage.

Callbacks may already have effects before a later truncation/error. Stage effects
externally when the application requires an all-or-nothing commit. The bounded
Bolt primitive cursor transactionality does not make visitor application state
transactional. An input-null/length mismatch returns an explicit error, including
with assertions enabled.

## Fixtures, pins and reproduction

All paths below are relative to the Bolt root. `tests/data/sbe-codegen/` contains
four XML/header pairs. No encoder creates these oracle bytes.
The reusable compiler source pins are recorded in
[compiler-sha256.txt](../../tests/data/sbe-codegen/compiler-sha256.txt).

| XML | Exact SHA-256 |
|---|---|
| features.xml | `c9b9b13ce93741c6cd62a6d2ee07ddb576f2a4c1e0cccbef9d43eb8cd9583469` |
| features-big.xml | `eaaa291516fa0b0c3c59b88110c395f5bf87735f3758aee23cf0127836adb843` |
| nested.xml | `c0123e165f34f519015c3fd544e6b085942d41abd48f600dd12746aab89dd13c` |
| historical-limits.xml | `fbe6acdc36f04dd5d72be7c5ced0bcf80a28bac04d88241b2caeb3db61ee4bd7` |

From Bolt root, with ai-grind installed or on PYTHONPATH:

```sh
python -m devtools_mcp.sbe compile tests/data/sbe-codegen/features.xml --output tests/data/sbe-codegen/features.h --namespace generated_features --expect-sha256 c9b9b13ce93741c6cd62a6d2ee07ddb576f2a4c1e0cccbef9d43eb8cd9583469 --max-group-entries 255 --max-data-bytes 1024
python -m devtools_mcp.sbe compile tests/data/sbe-codegen/features-big.xml --output tests/data/sbe-codegen/features-big.h --namespace generated_features_big --expect-sha256 eaaa291516fa0b0c3c59b88110c395f5bf87735f3758aee23cf0127836adb843 --max-group-entries 255 --max-data-bytes 1024
python -m devtools_mcp.sbe compile tests/data/sbe-codegen/nested.xml --output tests/data/sbe-codegen/nested.h --namespace generated_nested --expect-sha256 c0123e165f34f519015c3fd544e6b085942d41abd48f600dd12746aab89dd13c --max-group-entries 255 --max-data-bytes 1024
python -m devtools_mcp.sbe compile tests/data/sbe-codegen/historical-limits.xml --output tests/data/sbe-codegen/historical-limits.h --namespace generated_historical_limits --expect-sha256 fbe6acdc36f04dd5d72be7c5ced0bcf80a28bac04d88241b2caeb3db61ee4bd7 --max-group-entries 255 --max-data-bytes 1024
```

`features` uses independently hand-written LE/BE bytes covering arrays, composites,
float null, optional enum null, constant enum, padding and older acting blocks.
`nested` uses the preexisting owned `sbe_fixtures::nested` bytes and checks callback
order, empty inner group/data, older root blocks, count caps and failure propagation.
`historical-limits` decodes the existing independently published CME template-50
bytes and checks every expected field. The original capture attribution/hash is
in [sbe-fixtures.md](../../tests/data/sbe-fixtures.md). The reconstructed subset
only supports that historical example and is not a production Book46 manifest.

## Validation receipt and remaining gates

On macOS Clang, `tests/test_bolt_sbe_codegen.cpp` passes with `-std=c++20 -O2
-Wall -Wextra -Werror`, again with `-DNDEBUG`, and with `-O1 -g
-fsanitize=address,undefined -fno-omit-frame-pointer`. Every proper truncation of
the feature, historical and nested messages fails explicitly. The ai-grind test
suite `tests/test_sbe_compiler.py` has 48 passing cases; Ruff and mypy pass for
the compiler module. Its tests include independent offsets/bytes, malformed
schema cases, resource limits, cycles, unsafe XML, deterministic emission and
hash-pinned reusable CLI output preservation.

This is correctness evidence, not differential performance qualification.
G2FEAT-44 remains open: compare the generated reader to an independent generated
reader on the same pinned public/owned schema and bytes, including latency/tail
measurements. Windows MSVC/Linux container gates and actual versioned production
CME XML/captures remain separate validation work. SBE-to-BoltBatch ingestion,
schema registry/materialization and CME economics are not implemented by this
compiler slice.

Independent review rejected array-valued header/group dimension members even
when total header size remains eight bytes, and resolved versioned references
inside otherwise unversioned composites. Their previous acceptance produced a
wire layout the generated reader could not honor. Decimal metadata conversion
now bounds significant digits before Python integer conversion, so hostile
5,000-digit metadata returns `SchemaError`, not an uncaught digit-limit exception.

`devtools_sbe_compile` is registered in ai-grind's existing MCP tool inventory.
It receives UTF-8 XML as an in-memory string, requires the exact lowercase
SHA-256, and returns structured status/provenance/header or status/code/detail
without a header on failure. It accepts no paths, URLs or executable input and
writes no files. The cold callable limits XML to 256 KiB, emitted header to
1 MiB, messages to 64, members to 2,048, and concurrent worker-thread compilations
to two. Default wire count/payload caps remain 65,535 entries / 1 MiB; callers
may tighten them. Its registration and real registered `mcp.call_tool` dispatch,
hash mismatch and limit refusal are covered by the focused Python suite.
The CLI remains a separate explicitly path-based authoring boundary.

Review verification (2026-10-14): 48 focused pytest cases pass, Ruff is clean,
and mypy reports no issues in the six compiler/tool source files. Raw Python
receipt: `/private/tmp/g2-sbe-review-pytest.log`. The original pinned parser was
independently checked to accept both invalid scalar layouts and the versioned
reference, and to leak `ValueError` for oversized decimal metadata; new reject
regressions cover each. No local running-service rollout or full G2FEAT-37
acceptance is asserted by these checks.

Integer min/max metadata ordering now compares exact integer literals, so contradictory bounds above 2^53 are rejected rather than rounded equal by floating-point conversion.

Native review receipt: isolated `test_bolt_sbe_codegen` CMake target builds with
`-j2`; its CTest passes (1/1, 0.30 s). Standalone Clang strict `-O2 -Wall
-Wextra -Werror`, `-O2 -DNDEBUG` with the same warning gate, and `-O1 -g
-fsanitize=address,undefined -fno-omit-frame-pointer` each compile and run with
exit zero. Receipts are `/private/tmp/g2-sbe-review-ctest.log`,
`g2-sbe-review-strict.log`, `g2-sbe-review-ndebug.log`, and
`g2-sbe-review-asan.log`, with adjacent build logs. All four checked-in headers
were compared byte-for-byte to current compiler output and match. Focused Black
checks pass; the ai-grind prose scan reports zero findings for the changed
parser/tool/test files. These checks do not establish the deferred Windows,
Linux container, independent differential-reader or production-schema gates.

Resolved encoding DAGs have an explicit expanded-work budget (default 65,536
nodes, absolute cap 1,048,576), independently of encoded block size. Depth,
variable-layout detection and expanded size are memoized by encoding identity;
this includes inline and referenced composites without recursively hashing a
frozen DAG. Counts refuse before addition would exceed the remaining budget.
Selected message fields share the compile budget. Generation preflights expanded
work before leaf-list creation and additionally counts traversal steps.

Output is bounded during `emit`, before storing each line, rather than after
building the complete header. The default is 4 MiB (absolute cap 16 MiB); the
MCP callable tightens expanded work to 4,096 nodes and output to 1 MiB. The
existing generated-function limit remains in force. Oversized reusable
zero-byte constant composites return `SchemaError` from the direct API and a
structured MCP error without a header. A sub-8-KiB schema whose repeated graph
would expand to 8**8 leaves is rejected without executing that expansion.
Tests also admit a small reused DAG and check the exact output byte boundary.
After this review correction, 48 focused cases pass in 1.78 s; Ruff, Black and
mypy remain clean, and all four generated fixtures still match byte-for-byte.
Generated native output is unchanged, so the prior strict/NDEBUG/sanitizer
receipts remain applicable to the checked-in readers. No native rebuild was
required for the cold compiler-only correction.
