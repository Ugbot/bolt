# Bolt documentation

Start with [`BOLT_PROJECT_MAP.md`](BOLT_PROJECT_MAP.md) for the file map and
design principles, then follow whichever track below matches what you need.

> **A note on the project names you'll see.** Bolt was extracted from a set of
> sibling projects: **MarbleDB** (storage engine), **Chukonu** (distributed
> query engine), **BoltAPI** (HTTP/WS/SSE framework), **BoltLLM** (inference),
> and **Gestalt** (the engine several of these grew out of). **None of them are
> publicly available.** They appear throughout these docs — and in some source
> comments — because they are the workloads that drove a given decision. Read
> those mentions as the reason a design choice was made, not as code or repos
> you can go and look at. Bolt itself depends on none of them.

## Architecture and design

| Document | What it covers |
|----------|----------------|
| [BOLT_PROJECT_MAP.md](BOLT_PROJECT_MAP.md) | File map, directory layout, design principles. The entry point. |
| [BOLT_COLUMN_FORMAT.md](BOLT_COLUMN_FORMAT.md) | Column representations, statistics, sidecars, adaptive encoding. |
| [BOLT_DATAFLOW.md](BOLT_DATAFLOW.md) | Operator algebra, push dispatch, edge buffering, pipeline breakers. |
| [BOLT_INDEPENDENCE.md](BOLT_INDEPENDENCE.md) | What Bolt provides in place of each Arrow subsystem, and why the core has no dependencies. |
| [BOLT_ACERO_COMPONENTS.md](BOLT_ACERO_COMPONENTS.md) | Component-by-component comparison against Arrow Acero. |
| [BOLT_FINTECH_PORT.md](BOLT_FINTECH_PORT.md) | The fintech kernel family: microstructure, volatility, risk, liquidity, cross-asset. |

## Performance

| Document | What it covers |
|----------|----------------|
| [BOLT_PERFORMANCE.md](BOLT_PERFORMANCE.md) | The architectural reasons Bolt is fast, plus the microbenchmarks and their methodology. |
| [BOLT_DECISION_LOG.md](BOLT_DECISION_LOG.md) | Each design decision with the problem, the choice, and the measurement behind it. |
| [research/design-log.md](research/design-log.md) | Append-only experimental record: what was tried, what was kept, what was backed out. |

**How to read the numbers.** Every measurement in these docs is a
microbenchmark — one operation timed in isolation, on one machine, at one
compiler setting. They exist to justify specific design choices and to stop
old decisions being re-litigated. They are not end-to-end speedups, and some
comparisons involve different amounts of work (a selection vector instead of a
materialized batch, a constant fold instead of a scan). Bolt is early; treat
the figures as directional.

## Research notes

[`research/`](research/README.md) holds per-topic write-ups of the external
systems, papers, and techniques the design draws from — Pirk et al., DuckDB,
Polars, QuestDB, kdb+, the 1BRC entries, simdjson/fionn, and others. Each note
cites its primary sources and ends with what Bolt adopts and what it skips.
The index in [`research/README.md`](research/README.md) lists them all.

Quoted third-party source is attributed inline with its project and license
at the point of quotation.

## Planning and status

| Document | What it covers |
|----------|----------------|
| [BOLT_ROADMAP.md](BOLT_ROADMAP.md) | Planned work by wave. |
| [BOLT_PERF_PUNCHLIST.md](BOLT_PERF_PUNCHLIST.md) | Working checklist of base-layer perf items. Tracks in-progress engineering, so it goes stale faster than the design docs. |
| [BOLT_DESIGN.md](BOLT_DESIGN.md) | The original Arrow gap analysis, kept as a historical record of the founding rationale. |

## Conventions

- New research goes in `research/<topic-slug>.md` with an index entry in
  `research/README.md` — never appended to a monolithic notes file.
- Perf experiments get an entry in `research/design-log.md` recording what won
  and what workload would flip the choice.
- `BOLT_RESEARCH_NOTES.md` is a historical link target only; it points at the
  `research/` index.
