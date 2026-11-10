# YAML onto the fionn tape — `bolt::parse::yaml`

G2GRAPH-453. Header `include/bolt/parse/bolt_yaml.h`; implementation
`src/parse/bolt_yaml.cpp` (stage 1, tape, aux) with
`bolt_yaml_scalars.inc` and `bolt_yaml_blocks.inc`; tests
`tests/test_bolt_yaml.cpp`; CLI and benchmark `tests/bench_yaml/`. Follows
the JSON reader's fionn-derived design (`json-skip-architecture.md`) and
fionn's own YAML scanner (`crates/fionn-simd/src/formats/yaml.rs`: line /
indentation / indicator masks).

## One tape for JSON and YAML

The reader emits `json::Token`s — Begin/EndObject, Begin/EndArray, Key,
String, Int64, Float64, BoolTrue, BoolFalse, Null, End — into a
`json::StructuralIndex`. Everything that walks a JSON tape walks a YAML one:
`iter_skip_to_close`, `iter_int64` / `iter_float64` (lazy), `value_from_index`,
`serialize`, and the JSON-LD expander (`chukonu::rdf::jsonld_value_to_rdf`).
`to_json` is a thin adapter: build the tape, then `value_from_index` +
`serialize`.

## Zero-copy and the aux buffer

A token's bytes are JSON string content (escapes encoded) or a JSON number,
wherever they live. YAML scalars that already are — plain scalars without `"`,
`\` or control bytes, single-line quoted scalars whose escapes are all JSON
escapes, numbers spelled as JSON spells them, `null` / `true` / `false` — are
`(offset, length)` slices of the source. Everything else is materialised once
into `StructuralIndex::aux`, JSON-escaped, and the token carries
`flags = kTokenAux`: YAML-only escapes (`\x`, `\e`, `\N`, ...), folded
multi-line plain and quoted scalars, block scalars, number spellings JSON lacks
(`0x1F`, `+5`, `.5`, `1.`), and `~` / `Null` (pointing at fixed aux offsets of
`null` / `true` / `false`). `json::token_bytes(idx, tok)` resolves either;
`iter_string` reports aux bytes as buffer index 1. The JSON scanner never sets
the flag, so existing JSON consumers are unchanged.

## Stage 1: the line table

One word-at-a-time pass counts newlines (exact SWAR zero-byte mask over
`w ^ 0x0A..`), a second records line starts from the same masks, and each
line's indentation is a SWAR leading-space count (first non-`0x20` byte via
`countr_zero`). Each line is classified content / blank / comment /
tab-indented. TODO(SIMD): NEON / AVX2 16- or 32-byte compares + movemask, as
in bolt_json's deferred SIMD stage; the SWAR path is the reference.

## Stage 2: a state machine over lines

Block structure never rescans bytes: the next content line, its column and
its class come from the table (`next_content`, `ln_of` with a locality hint so
the forward-moving cursor is O(1)). Within a line, plain scalars are scanned
word at a time: a word with no `:` / `#` (or flow indicator) is skipped
whole, and the same pass records whether the bytes are JSON content (no `"`,
`\`, control byte) — which decides zero-copy without a second pass. A
mapping key's scan is reused for the key itself.

## Filtered builds

`build_index(..., filter, ...)` takes a `json::PathFilter` from
`compile_paths` and applies `path_filter_admits` exactly as
`build_index_filtered` does: keys and array indices extend the path; a
subtree outside every interest path is still parsed (so it is still
validated — an anchor in a skipped subtree is still refused) but emits no
tokens. Knowledge ingest builds `index.md` with `/base_iri` only.

## Semantics, refusals, budgets

YAML 1.2 core schema (dates are strings); refused by name: anchors and
aliases, tags, directives, complex keys, several documents, tab indentation,
non-finite floats, integers beyond int64, multi-line plain scalars inside flow
collections, single-pair mappings in flow sequences. Budgets: input bytes,
depth (<= 256; <= 64 when filtered), nodes; refused by name.

## Measured (2026-11-10, M5 Max, release, one thread)

`scripts/verify_bolt_yaml_tape.py` in Gestalt2 over the frontmatter of LOKF's
examples and the full Innermost Loop KB (6140 documents, 5.85 MB):

| | MB/s |
|---|---|
| tape build (`build_index`) | 661 |
| `to_json` over the tape | 484 |
| previous text emitter (`to_json`, pre-tape) | 406 |
| rapidyaml 0.16.0 `parse_in_place` into a reused tree | 309 |

Correctness on the same run: 6140/6140 documents equal to ruamel.yaml in YAML
1.2 mode (dates compared as instants) and to PyYAML; yaml-test-suite (data
branch): of 256 single-document tests with `in.json`, 81 are refused by name
and the claimed 175 all equal `in.json`.
