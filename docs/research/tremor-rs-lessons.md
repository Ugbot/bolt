# tremor.rs — lessons for the dataflow runtime

## Context

tremor.rs (https://www.tremor.rs/) is a Rust-based event-processing
system designed for high-throughput, low-latency data flow with
explicit pipeline definition. Its model is pure push: events come
in, flow through a graph of operators defined in a DSL ("tremor
query"), exit at sinks. Used in production at Wayfair for high-
volume ingestion. Worth studying for back-pressure design and
connector model — both directly relevant to Bolt's dataflow runtime.

## Architecture

### Pipeline DSL

tremor pipelines are written in **Tremor Query Language (TQL)**, a
declarative DSL:

```
define pipeline ema_pipeline
pipeline
    define operator ema from generic::ema
    with
        period = 20
    end;

    select event from in into ema;
    select event from ema into out;
end;
```

The DSL compiles to a static operator graph. Operators are defined
as Rust structs implementing an `Operator` trait. There is no
runtime plan rewriting — the graph is built once at deploy time.

### Operator interface

```rust
pub trait Operator {
    fn on_event(&mut self, port: &str, event: Event)
        -> Result<EventAndInsights>;

    fn handles_signal(&self) -> bool { false }
    fn on_signal(&mut self, signal: &mut Signal)
        -> Result<EventAndInsights> { ... }

    fn handles_contraflow(&self) -> bool { false }
    fn on_contraflow(&mut self, insight: &mut Event) { ... }
}
```

Three message types flow through:
- **Event** — normal data, downstream
- **Signal** — control data, downstream (e.g. tick, drain, pause)
- **Contraflow** — feedback upstream (e.g. "I'm full, slow down")

The contraflow channel is what makes tremor's back-pressure work
end-to-end: a downstream sink can signal "I'm overloaded" and the
upstream source receives it without needing a side channel.

### Connectors

tremor separates **connectors** (sources / sinks for external
systems: Kafka, files, HTTP, DNS) from **operators** (pure
transforms). Connectors carry cross-cutting concerns: TLS,
authentication, retry, codec parsing.

The runtime isolates connector failures from the pipeline — a
crashed Kafka client doesn't kill the pipeline; the framework
retries with backoff.

### Circuit breakers

Each connector has a circuit breaker that opens on sustained
failure. The breaker propagates a contraflow message saying "stop
sending me events" up the pipeline; the source pauses or drops
according to its policy.

This is conceptually identical to Bolt's per-edge backpressure,
except tremor's is system-wide (any sink can signal any source via
the contraflow channel).

## Key trade-offs tremor made

1. **DSL-defined pipelines** — operators are wired in TQL, not
   in Rust code. Trade: deploy-time flexibility vs. compile-time
   type safety. tremor's DSL has its own type system to recover
   some safety, but it's a runtime check, not a compile-time one.
2. **Contraflow for back-pressure** — explicit upstream signal
   channel rather than implicit credit / blocking. More flexible
   (you can signal anything, not just "I'm full") but requires
   every operator to handle contraflow correctly.
3. **Per-connector retry / circuit-break logic** — pulled out of
   the pipeline into the connector layer. Means pipeline operators
   stay pure transforms; connectors handle the messy world.
4. **Single-threaded per pipeline** by default — parallelism is via
   running multiple pipeline instances, not by parallelizing within
   a pipeline. Same trade DuckDB made for OLTP-shaped workloads;
   different from Bolt's morsel-parallel-within-operator model.

## What we adopt

- **Connector / operator separation** — keep the dataflow runtime
  pure; sources / sinks that touch the outside world are wrapped
  in adapter operators. Marbledb's `BoltBatch` reader is a
  connector; a tick stream from a TCP socket is a connector;
  neither is a "real" operator.
- **Contraflow concept** for backpressure-as-explicit-signal —
  Bolt's per-edge `dropped_count` and `coalesced_count` already
  capture some of this; if we need "sink signals source," we add
  a contraflow channel as a side band, not in the data path.
- **Circuit-breaker pattern** for sources that can fail (network,
  disk) — keep the runtime alive, isolate the failure to the
  source.

## What we skip

- **DSL-defined pipelines** — Bolt is C++; pipelines are built in
  C++ via the `Graph` API. chukonu's SQL surface produces graphs;
  no separate runtime DSL.
- **Single-threaded per pipeline** — Bolt is morsel-parallel within
  every operator; we want both the per-operator parallelism and
  multiple-graphs-running.
- **Generic event message** — tremor events are loosely typed
  (basically JSON-like maps). Bolt batches are typed columns; no
  schema-on-read.
- **Runtime operator loading** — tremor lets you load operator
  modules at runtime; Bolt's POD + fn pointer model means
  everything is compile-time.

## Followups

- Decide whether to add a contraflow side-channel between sinks
  and sources, or rely on per-edge backpressure being sufficient
- Connector model design — what does a "Bolt connector" look like
  vs. a "Bolt operator"? Probably an operator with `kSource` or
  `kSink` flag and an adapter to the external system.

## References

- tremor.rs official site: https://www.tremor.rs/
- tremor-runtime source: https://github.com/tremor-rs/tremor-runtime
- TQL language docs: https://www.tremor.rs/docs/tremor-script/
- Wayfair usage: tremor was originally built for Wayfair's logging
  ingestion at TB/day scale
