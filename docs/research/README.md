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
- [`../BOLT_PERF_PUNCHLIST.md`](../BOLT_PERF_PUNCHLIST.md) — the live
  checklist of base-layer perf items being closed out before any new
  feature surface opens. Each ticked box links to a design-log entry.

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
- [scheduler-ring-exactly-once.md](scheduler-ring-exactly-once.md) — the
  SPMC `TaskRing` lost and duplicated tasks in exactly equal numbers on a
  saturated ring (claim-then-read vs copy-then-claim), so no scheduler
  counter could detect it; the downstream symptom was a nondeterministic
  wrong answer from a parallel anti-join. Includes why the first
  regression test was green on the broken ring.
- [deterministic-scheduler-simulation.md](deterministic-scheduler-simulation.md)
  — the TigerBeetle VOPR model scoped to the task ring: one participant runs
  at a time and the interleaving is drawn from a seed, so a schedule is a pure
  function of (seed, commit) and a failing seed replays exactly. The ring runs
  AS WRITTEN via a compile-time policy parameter (production codegen proven
  byte-identical at `-O3 -DNDEBUG`). Fails on the injected pre-fix ring for 11
  of 12 seeds, in 0.63 s, with identical values at load average 77 — replacing
  a probe that needed a busy box. Also records two harness bugs its own
  non-vacuity assertions caught.

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

### Dataflow runtime (Wave 1)

Design-blocking research:

- [dataflow-operators.md](dataflow-operators.md) — operator algebra
  (POD struct + fn ptr, push-stream, no vtable), fusion classes
  (trivial / compute / pipeline-breaking), three-category pipeline
  breaker classification driving edge buffer choice.
- [tick-tock-buffering-protocol.md](tick-tock-buffering-protocol.md)
  — per-edge ring of BoltBatch with epoch swap; N=2 default,
  release/acquire visibility, three backpressure policies
  (block / drop-newest / coalesce), arena-tied slot lifetime.
- [push-vs-pull-dispatch.md](push-vs-pull-dispatch.md) — push with
  N-slot edge ring as implicit credit; central executor walks topo
  order each tick, scheduler fans morsels per operator; phase
  barriers at pipeline breakers.
- [gestalt-kernel-adapter.md](gestalt-kernel-adapter.md) — porting
  shape for the 74 fintech kernels: Tier 1 mechanical, Tier 2
  stateful (arena-pinned POD state), Tier 3 deferred until
  event-time + keyed state primitives land. Worked EMA diff.

Comparative analysis (validates the choices above):

- [polars-streaming-engine.md](polars-streaming-engine.md) — lazy
  expression DAG, predicate / projection / slice pushdown, streaming
  morsels (~50K rows), pipeline-breaker phases, Rayon work-stealing.
- [duckdb-vectorized-push.md](duckdb-vectorized-push.md) — Vector
  multi-format (Flat / Constant / Dictionary / Sequence / FSST),
  push-based execution, pipeline-breaker categorization,
  partition-on-overflow, morsel-driven parallelism. Many of Bolt's
  existing primitives trace here.
- [tremor-rs-lessons.md](tremor-rs-lessons.md) — TQL-defined
  pipelines, contraflow back-pressure channel, connector / operator
  separation, circuit breakers. We adopt the connector model
  conceptually; skip the DSL.
- [kdb-questdb-tick-models.md](kdb-questdb-tick-models.md) — the
  reference points Bolt's tick workloads are measured against. kdb+
  tickerplant + RDB + HDB chain;
  QuestDB time-partitioned column files, symbol type, O3 ingest,
  JIT filters. The continuous-query shape these model is the same
  shape bolt::dataflow's push dispatch produces.
- [questdb-symbol-vs-fsst.md](questdb-symbol-vs-fsst.md) — SYMBOL
  physical layout (`.d` per-partition + `.c`/`.o` table-global dict
  + optional `.k`/`.v` bitmap-index sidecar), ID-first operators,
  measured cardinality wall (1k fine, 10k → 15–25 s on 300M rows per
  issue #6246). FSST (Boncz/Neumann VLDB 2020) solves a disjoint
  problem (substring-redundant high-card strings). Bolt's
  `Dictionary + BitmapIndex` already covers the SYMBOL shape; FSST
  deferred. Quick wins enumerated: global dict, literal-resolve-once,
  per-ID popcount sidecar, lock-free append, sorted-ID invariant.
- [questdb-symbol-code-audit.md](questdb-symbol-code-audit.md) —
  source-code audit of the claims in `questdb-symbol-vs-fsst.md`
  against `questdb/questdb@master`. Eight claims verified; seven
  CONFIRMED, one PARTIALLY CONFIRMED (`.v` was conflated with `.c`
  in the dictionary layout — `.v` is actually the index row-list
  file; fixed inline in the original note). File-path + line-range
  citations for every claim. Bolt recommendation unchanged:
  Dictionary + BitmapIndex covers SYMBOL, FSST deferred. Side
  finding: QuestDB VARCHAR uses a Umbra-style 6-byte-prefix +
  inline-or-offset layout (not FSST, not plain length-prefixed).

Synthesized into the design doc [`../BOLT_DATAFLOW.md`](../BOLT_DATAFLOW.md).

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
