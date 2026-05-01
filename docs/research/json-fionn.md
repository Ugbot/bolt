# fionn — JSON parsing techniques

**Fionn** (https://github.com/darach/fionn) is an experimental Rust library
for selective JSON parsing designed for high-throughput, constant-memory
workloads. Unlike traditional DOM parsers, fionn uses a **skip-based
architecture** that traverses JSON without materializing all intermediate
values, achieving 8+ GiB/s throughput (29.5x faster than serde_json) while
consuming O(1) memory instead of O(document size). The claim is particularly
strong for selective extraction: a 1 GB document uses ~1 KB of RAM with
fionn's skip model vs. 3 GB with serde_json DOM parsing.

## Architecture: skip-based traversal vs. structural bitmaps

Fionn diverges from the canonical two-stage simdjson model. Where simdjson
uses Stage 1 (find all structural characters via SIMD scan) followed by
Stage 2 (walk the structural index to build a tape), fionn implements four
pluggable skip-scanning strategies that traverse JSON in a single pass:

- **Scalar** (1.5 GiB/s): Byte-by-byte fallback; portable, no SIMD.
- **Langdale** (987 MiB/s): XOR-based prefix scan over escape bitmaps.
- **JsonSki** (1.0 GiB/s): Bracket-counting; nesting-depth aware exit.
- **AVX2** (4-8 GiB/s): Full SIMD on x86_64; falls back to JsonSki elsewhere.

The skip model is SAX-style (event-driven) rather than DOM: fionn emits
structural positions or schema-matched field names, not a complete tree.
For JSONL inputs, a SimdLineSeparator pairs with per-line skip scanning.

## Key techniques

**String detection & escape handling:**
Fionn uses memchr::memmem::find() for literal field-name searches, not the
prefix-XOR quote-bitmap technique simdjson uses. For narrow schemas this is
faster. The Langdale variant applies prefix-XOR to escape bitmaps (backslash
positions) rather than quotes, inferring string boundaries afterward.

**Number parsing:**
Deferred to the caller. Numeric fields are returned as raw byte slices,
keeping the skip phase allocation-free and letting the consumer choose
int/float/Decimal parsing based on type context.

**Whitespace skipping:**
Langdale uses SIMD memchr comparisons; JsonSki and Scalar step character-by-
character while counting brackets. No dedicated SIMD whitespace scan.

**UTF-8 validation:**
Not explicitly validated during skip scanning; simdjson does this in Stage 1.
Fionn assumes valid input or defers validation to the value consumer.

**Memory layout:**
Purely streaming (SAX events). No DOM tree or lazy on-demand access like
simdjson's On-Demand API. Tradeoff: zero intermediate allocation, but the
caller must consume events immediately (no random access to re-visit).

## Comparison table — fionn vs. simdjson vs. Bolt

| Dimension                    | fionn                           | simdjson                  | Bolt potential                 |
|------------------------------|---------------------------------|---------------------------|--------------------------------|
| Stage-1 model                | Skip-scanning (4 variants)      | Structural bitmap (SIMD)  | stage-1 if full JSON; skip if NDJSON |
| Stage-2 model                | SAX events; no tree             | Tape walking; DOM/OnDemand| Column projection (BoltBatch)  |
| Quote handling               | Literal + Langdale              | Prefix-XOR bitmap         | Literal for NDJSON             |
| Number parsing               | Caller-supplied (deferred)      | Branchless in stage 2     | parse_int10th (int10ths)       |
| UTF-8 validation             | Deferred / skipped              | Stage 1 (interleaved)     | Out-of-scope v0                |
| Streaming / SAX              | Yes, event-driven               | No (DOM or on-demand)     | YES: emits BoltBatch columns   |
| Memory model                 | O(1) skip                       | O(document) or O(1)       | O(schema) per morsel           |
| Dependencies                 | memchr, alloc                   | std only                  | bolt::core, bolt_port.h        |
| Language                     | Rust                            | C++                       | C++20                          |

## What Bolt could borrow

**HIGH applicability:**

- **Selective field extraction**: Langdale prefix-XOR for escape bitmaps is
  a direct win for line-delimited JSON. Apply after bolt_swar_find_byte_u64
  locates field names.

- **SAX/streaming architecture**: Emit column-wise events (one BoltColumn per
  field) as you scan. Aligns perfectly with BoltBatch output model.

- **Bracket-counting for nested structures**: JsonSki's bracket-depth tracking
  is simpler than full structural indexing for skipping unneeded nesting.

**MEDIUM applicability:**

- **Escape-bitmap prefix-XOR** (Langdale): Similar to simdjson's quote-boundary
  technique but applied to backslashes. Useful for full string parsing.

- **Multi-strategy dispatch**: Fionn's runtime selection of 4 algorithms based
  on CPU features mirrors our bmm_* dispatch. Follow the same pattern for
  stage-1 strategy selection based on BOLT_SIMD_*.

**LOW applicability:**

- **Number parsing**: Fionn defers this. Bolt already has parse_int10th.
  Floats and arbitrary-precision decimals out-of-scope for v0.

- **Full UTF-8 validation**: Deferred to v1. For ASCII NDJSON keys, skipping
  is fast enough that a separate UTF-8 pass is acceptable.

- **Format conversion (YAML/TOML/CSV)**: Fionn ships tape-to-tape transforms.
  Not relevant for bolt::ingest::json (columnar, not nested tape).

## What we deliberately skip

- **DOM construction**: Streaming columns only.
- **On-demand lazy access**: No re-visiting; consume during scan.
- **Nested extraction (JSONPath/XPath)**: v0 is flat schema only.
- **Schema-on-read fuzziness**: v0 enforces strictly; no coercion.

## Mapping to Bolt's existing parts

- **Structural character detection**: bolt_swar_find_byte_u64 already scans
  for delimiters. Extend for ", :, {, }, [, ] to build stage-1 bitmap.

- **Escape handling**: Langdale prefix-XOR maps to bmm_cmpgt_i8 / bmm_and /
  bmm_movemask — scan 32 bytes for backslashes, XOR-collapse, correct
  string boundaries.

- **Streaming into BoltBatch**: Our CSV loader already uses SWAR loops.
  JSON equivalent: scan field names, skip unwanted fields, parse leaf
  values into columns. Structure mirrors perfectly.

- **Number parsing**: parse_int10th covers int10ths. For JSON numbers
  with optional . and exponents, a parse_json_number wrapper is natural.

## Sources

- Fionn repository: https://github.com/darach/fionn
- Fionn README: https://raw.githubusercontent.com/darach/fionn/main/README.md
- Fionn-simd module: https://github.com/darach/fionn/tree/main/crates/fionn-simd/src
- SimdJSON HACKING.md: https://github.com/simdjson/simdjson/blob/master/HACKING.md
- SimdJSON documentation: https://github.com/simdjson/simdjson/tree/master/doc

## Bolt port — implementation notes

Layer 1.3 of the `this-was-a-freach-hashed-crab.md` plan landed
`bolt::parse::json`. It rebuilds Fionn's architecture in C++20 (no Rust
imported), Tiger-Style throughout: `noexcept` everywhere, no STL in
production code, all allocation through `bolt::Arena`, depth cap at 64,
PathFilter cap at 1024 paths.

The four-stage walk-through, the path-filter design, the
`iter_skip_to_close` cost model, the deferred-SIMD TODO, and the measured
token-count delta on the synthetic filter test (full=796 vs filtered=11
on `interest = {/wanted, /also/0}`, **~98.6% reduction**) live in the
sibling note `json-skip-architecture.md`. This note stays as the
upstream-research log; that note is the implementation log.

Files: `include/bolt/parse/bolt_json.h`,
`src/parse/bolt_json.cpp`, `tests/test_bolt_parse_json.cpp`. Wired into
the existing `bolt::parse` static library (one-line CMake addition).
