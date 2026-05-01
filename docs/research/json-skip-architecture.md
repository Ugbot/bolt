# Skip-aware JSON parser — `bolt::parse::json`

**Layer 1.3** of `this-was-a-freach-hashed-crab.md`. Header
`include/bolt/parse/bolt_json.h`, impl `src/parse/bolt_json.cpp`, tests at
`tests/test_bolt_parse_json.cpp`. Lifts Fionn's design (skip-aware tape +
precompiled path filter) — no Rust code imported.

## What we built

Four cooperating stages, each crisp on the boundary so we can swap one out
without touching the others.

### Stage 1 — structural-index scan

Single linear pass over the source byte buffer. Scalar fallback only in
this wave; AVX2 / SSE4.2 are deferred (see "SIMD deferred" below). The pass
emits exactly one `Token` per JSON structural element:

- `BeginObject` / `EndObject` / `BeginArray` / `EndArray` — depth markers,
  no payload, `length == 0`.
- `Key` — the bytes between the quotes of an object key. Zero-copy slice
  into the source.
- `String` — same shape as `Key` but emitted as a value.
- `Int64` / `Float64` — raw number bytes, parsed lazily on demand.
- `BoolTrue` / `BoolFalse` / `Null` — keyword markers.
- `End` — sentinel.

UTF-8 is validated mid-scan with the Hoehrmann DFA (single 364-byte table)
and rejects overlong (`0xC0 0x80`), surrogates, and truncated tail bytes.
Bare control bytes < 0x20 inside strings reject. Escape sequences consume
two source bytes without further validation — `\uXXXX` is the consumer's
problem at materialise time.

### Stage 2 — tape

The same pass writes Tokens into a buffer allocated up front from the
caller's `Arena*`. Capacity sized to `src_len + 2` — a worst-case bound
because every token covers at least one source byte (the only zero-byte
token is the `End` sentinel; we add 2 of slack). No realloc loop.

`sizeof(Token) == 12` with explicit `static_assert`. POD; no destructors.

### Stage 3 — iterator with `skip_to_close`

`Iterator` is `{ const StructuralIndex*, int32_t cursor }` — 16 bytes,
trivially copyable. `iter_skip_to_close` walks the tape (not the byte
stream) counting structural depth; lands on the first index past the
matching `End{Object,Array}`. Cost is O(tokens-skipped), not
O(bytes-skipped) — that's the headline win versus a full re-parse.

For scalars at the cursor, `skip_to_close` advances by one (the scalar
*is* its own close). Test
`SkipToCloseEqualsSequentialAdvance` asserts the skip lands at exactly the
same cursor as a manual depth-tracking advance loop.

### Stage 4 — path filter

`compile_paths` ingests slash-prefixed interest paths (`/foo`, `/foo/0/bar`)
and stores them in an FNV-1a closed-addressing hash table. Capacity =
`next_pow2(count * 4)`. Caps: 1024 paths, 256 bytes per path, 64 nesting
depth — all `static constexpr` in the header.

`build_index_filtered` carries a `path_stack` while parsing. Whenever a
key or array index is entered, the joined path is checked: if it isn't a
prefix of any interest path, the recursive call into the subtree runs with
`emit_tokens=false`. Bytes are still consumed (we have to find the matching
close), but **zero tokens** are written for the skipped subtree.

The prefix test is currently a linear scan over occupied slots. That's
fine for the v1 cap of 1024 paths; the obvious upgrade is a precomputed
prefix-trie at compile time — left as a TODO when a benchmark says it
matters.

## Path filter — what passes and what doesn't

A path passes when the current path-stack is a prefix of any interest path
**and** the prefix ends on a `/` boundary (or end-of-path). So
`/wanted` matches `/wanted/x/y`, but `/want` does not match `/wanted`. The
boundary check stops false hits where one declared path is a textual
prefix of an unrelated key.

## Numbers parsed lazily

Stage 1 only records `(start, length)` for `Int64` / `Float64` tokens.
`iter_int64` / `iter_float64` copy into a stack buffer (max 64 bytes,
asserted) and call `std::strtoll` / `std::strtod`. These two are
non-throwing and acceptable per the plan. The length bound prevents the
copy from running off the source buffer; the C-strtod's stop pointer
must equal the buffer end to accept (rejects `"12abc"`-style trailing
garbage that the lexer would never produce, but defends in depth).

## Cost model

| Operation                       | Cost                            |
|---------------------------------|---------------------------------|
| `build_index` over N source     | O(N) bytes scanned, O(T) tokens |
| `build_index_filtered`, F paths | O(N + T·F) — F=interest paths   |
| `iter_skip_to_close` over K     | O(K) tape steps (no byte scan)  |
| `iter_int64` / `iter_float64`   | O(L) where L ≤ 64               |
| `compile_paths` of P paths      | O(P · max_path_len)             |

## Measured

`tests/test_bolt_parse_json.cpp::FilterTokenDelta` builds a synthetic
document with one large `"big"` array (64 entries, each a 3-field object
with a 3-element tag array) and a small `"wanted"` plus `"also/0"` slice.
Token counts:

- **Unfiltered**: 796 tokens.
- **Filtered (interest = `/wanted`, `/also/0`)**: 11 tokens.
- **Delta**: 785 tokens skipped, ~98.6% reduction.

The reduction matches the structural ratio of the document — the parser
walks the bytes but the tape only carries what the consumer asked for.

## SIMD deferred

The structural-index scan is scalar in this wave. A `find any of `"\:,{}[]
in 32 bytes` AVX2 kernel and an SSE4.2 fallback are the natural next step,
following the Bolt convention of `BOLT_SIMD_*` compile-time selection. The
TODO marker is in `src/parse/bolt_json.cpp` — search for `// SIMD`.

UTF-8 validation already runs on the scalar path; the AVX2 lookup-table
approach (Lemire 2018) folds it into the structural scan. That's a single
unified upgrade.

## What we didn't take from Fionn

- **Tape format** — Fionn's tape entries are 8 bytes; ours are 12 because
  we keep `start` and `length` separated for direct slicing. Worth
  revisiting if memory bandwidth dominates a real workload.
- **Multi-strategy dispatch** — Fionn ships four scan strategies
  (Scalar / Langdale / JsonSki / AVX2). We're committing to one default
  per Bolt's "one default, alternatives behind compile flags" rule. A
  compile-flag opt-in for AVX2 is the right shape when we add it.
- **Number-as-byte-slice everywhere** — Fionn defers all numeric parsing
  to the consumer. We materialise lazily via `iter_int64` /
  `iter_float64` so the consumer doesn't re-derive the parser; the
  byte-slice is still available via `iter_string` if a caller wants raw.

## Open questions

- Whether to precompute a per-depth prefix-set in `PathFilter` to make
  `path_is_prefix` O(1) instead of O(P). Cap=1024 makes the linear scan
  fine for v1; the trie matters only at higher P.
- AVX2 / AVX-512 structural scan upgrade path. Fold UTF-8 validation in.
- Streaming variant — `build_index` requires the full source up front.
  A chunked variant would let consumers pipeline NDJSON. Defer until a
  consumer asks.
