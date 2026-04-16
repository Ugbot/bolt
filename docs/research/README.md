# Bolt research notes

Per-topic write-ups of external systems, papers, and techniques the design
draws from. One file per topic. Cited primary sources inline; every claim
should be link-traceable.

**Process rule:** any web research, paper summary, or external-system
analysis lands here as its own `.md` file — not in conversation summaries
or plan prose. New topic → new file + add an entry below.

## Index

### Foundational technique catalogue

- [pirk-techniques.md](pirk-techniques.md) — Holger Pirk et al. (cracking,
  cache-conscious layout, Voodoo algebra, BOSS composition, LightSaber
  PAT, micro-adaptive kernels, DEPA, tree indices, vectorised vs
  compiled). The original 14-paper catalogue Bolt was bootstrapped from.

### Decisions log

- [design-log.md](design-log.md) — running record of perf experiments
  with measured outcomes: what we tried, what we kept, what we backed
  out and why. Append-only; future tuning starts here so we don't
  re-litigate `MergeTriple` 40-vs-64-byte etc.

### Hashing

- [branchless-hashing.md](branchless-hashing.md) — branch-removal
  techniques for open-addressing hash tables (F14 chunk-overflow
  counter, Robin Hood probe-distance early-exit, Cuckoo always-read-N,
  CMOV probe step, two-pass hoisted probe). Catalogues which Bolt
  branches each technique removes; ranks the candidates worth
  building locally vs deferring (PtrHash, AVX-512).

### SIMD + portability

- [cglm.md](cglm.md) — `glmm_*` macro dispatch, `CGLM_ALL_UNALIGNED`,
  single-include umbrella, granular per-type alignment. Source for
  Bolt's `bmm_*` / `BOLT_ALL_UNALIGNED` / `bolt/bolt.h` patterns.
- [avx512-status.md](avx512-status.md) — native AVX-512 specializations
  (`bmm_compressstore_i32_x16`, `bmm_conflict_i32_x16`, masked
  load/store) + AVX2 fall-through for unspecialised ops; hardware
  availability matrix.
- [hash-functions.md](hash-functions.md) — trade-off space for
  Bolt's open-addressing hash mix (Murmur3 vs Fibonacci vs wyhash vs
  perfect hashing); why `swiss_mix` keeps Murmur3 today and the two
  safe paths to a cheaper mix later.

### Scheduler / parallelism

- [scheduler-design.md](scheduler-design.md) — DuckDB morsel-driven
  parallelism, Polars Rayon work-stealing, Seastar shared-nothing
  reactor. Comparative table; what we adopt vs skip.
- [cpu-topology.md](cpu-topology.md) — OS APIs Bolt's
  `bolt_topology.h` reads (Windows
  `GetLogicalProcessorInformationEx`, Linux sysfs, macOS
  `hw.perflevel*`, Intel hybrid spec).

### Ingest / parsing

- [1brc.md](1brc.md) — 1 Billion Row Challenge winners (thomaswue,
  mtopolnik, jerrinot, royvanrijn, merrykittyunsafe). SWAR `;` finder,
  branchless `0x640a0001` ASCII int parser, cardinality-tight hash
  table, dual-cursor ILP. Sources for `bolt_swar_*` and
  `bolt::parse::parse_int10th`.
- [json-fionn.md](json-fionn.md) — fionn vs simdjson; skip-based
  vs structural-bitmap parsing, Langdale escape-bitmap prefix-XOR,
  SAX-style streaming. Reference material for the future
  `bolt::ingest::json` module.

## How to add a new entry

1. Create `docs/research/<topic-slug>.md`.
2. Lead with a one-paragraph context: what the source is, why it
   matters to Bolt.
3. Quote primary-source code snippets (verbatim, with brief
   annotation) where they're load-bearing for a Bolt design choice.
4. End with a "what we adopt / what we skip / followups" section.
5. Append the entry to the index above with a one-line hook.

`docs/BOLT_RESEARCH_NOTES.md` is now a thin pointer to this index —
historical link target only; new content goes here.
