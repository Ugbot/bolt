# Gestalt kernel adapter — porting shape

## Context

74 fintech kernels written for Apache Arrow in a predecessor engine (not
publicly available) are the forcing function for Bolt's dataflow runtime
— the API shape that satisfies them is the API shape that ships. This
file fixes the mechanical port pattern (Tier 1 stateless, Tier 2
stateful) and documents the worked example for EMA.

Categories from the survey of those kernels:

- **Tier 1 — mechanical (~26 kernels, ~35%):** stateless within batch.
  Midprice, microprice, spread, imbalance, log returns, Sharpe,
  Sortino, SMA. Direct port: type swap + arena swap.
- **Tier 2 — stateful (~30 kernels, ~40%):** EMA, EWMA, EWCOV, rolling
  correlation/skew/kurtosis, Welford, 1D Kalman, autocorr. Need
  persistent state struct.
- **Tier 3 — framework features missing (~18 kernels, ~25%):**
  Hayashi-Yoshida cov/corr, lead-lag, pairs spread, triangular arb,
  VPIN. Need event-time windows + keyed state. Deferred until those
  primitives land.

## Today's shape (Arrow / chukonu predecessor)

```cpp
// gestalt/chukonu/include/chukonu/fintech/kernels.h:482-516
inline MorselProcessorFunc CreateEMAProcessor(
    int32_t value_col,
    int32_t period,
    const std::string& output_col = "ema") {

    return [value_col, period, output_col](
        std::shared_ptr<arrow::RecordBatch> batch
    ) -> std::shared_ptr<arrow::RecordBatch> {
        auto values = std::static_pointer_cast<arrow::DoubleArray>(
            batch->column(value_col));

        const double alpha = 2.0 / (period + 1);
        std::deque<double> window;     // <- per-call allocation
        arrow::DoubleBuilder builder;  // <- per-call allocation

        for (int64_t i = 0; i < batch->num_rows(); ++i) {
            const double v = values->Value(i);
            // ... ema math, builder.Append(...)
        }

        std::shared_ptr<arrow::Array> result;
        builder.Finish(&result);            // <- another allocation
        return AppendColumn(batch, output_col, result);   // <- another
    };
}
```

Five allocations per batch. State (the deque) is local — broken across
batches in any streaming context. `dynamic_pointer_cast` per column.
`std::string` for column names. `std::shared_ptr<RecordBatch>` for the
batch itself.

This is precisely why the predecessor is too slow.

## Target shape (Bolt dataflow)

Two parts: (1) the operator definition the kernel becomes, and
(2) the kernel body itself.

### Operator definition (Tier 1 — stateless)

```cpp
// bolt/include/bolt/kernels/fintech/spread.h
struct SpreadOpDesc {
    int32_t bid_col;
    int32_t ask_col;
    int32_t out_col;
};

void execute_spread(
    const BoltBatch* const* in,
    BoltBatch* const* out,
    Range range,
    Arena* arena,
    void* state) noexcept
{
    const auto* desc = static_cast<const SpreadOpDesc*>(state);
    BOLT_ASSERT(in[0] != nullptr);
    BOLT_ASSERT(out[0] != nullptr);
    BOLT_ASSERT(range.end > range.start);

    const double* BOLT_RESTRICT bid =
        in[0]->col(desc->bid_col).data<double>();
    const double* BOLT_RESTRICT ask =
        in[0]->col(desc->ask_col).data<double>();
    double* BOLT_RESTRICT result =
        out[0]->mut_col(desc->out_col).data<double>();

    for (int64_t i = range.start; i < range.end; ++i) {
        result[i] = ask[i] - bid[i];   // branchless, vectorizes
    }
}
```

State for stateless operators is just the operator's column-binding
descriptor (which column is bid, ask, output). Pinned at graph
compile, no per-tick allocation.

### Operator definition (Tier 2 — stateful)

```cpp
// bolt/include/bolt/kernels/fintech/ema.h
struct EMAState {
    SpreadOpDesc desc;       // (or a nested column-binding struct)
    double alpha;
    double ema_value;        // running state across ticks
    bool   initialized;
};

void execute_ema(
    const BoltBatch* const* in,
    BoltBatch* const* out,
    Range range,
    Arena* arena,
    void* state) noexcept
{
    auto* s = static_cast<EMAState*>(state);
    const double* BOLT_RESTRICT v =
        in[0]->col(s->desc.value_col).data<double>();
    double* BOLT_RESTRICT result =
        out[0]->mut_col(s->desc.out_col).data<double>();

    int64_t i = range.start;
    if (!s->initialized && i < range.end) {
        s->ema_value = v[i];
        s->initialized = true;
        result[i] = v[i];
        ++i;
    }
    for (; i < range.end; ++i) {
        s->ema_value = s->alpha * v[i] +
                       (1.0 - s->alpha) * s->ema_value;
        result[i] = s->ema_value;
    }
}
```

State struct is POD, pinned in the graph's arena at compile time.
Persists across ticks for the graph's lifetime. **No `std::deque`** —
the stateful kernels that today use a deque (rolling-window) become
fixed-size ring buffers in the state struct.

### Operator definition (rolling window — Tier 2 with bounded ring)

```cpp
struct RollingMeanState {
    int32_t value_col;
    int32_t out_col;
    uint32_t window_size;        // <= kMaxRollingWindow (constexpr)
    double ring[kMaxRollingWindow];
    uint32_t ring_pos;
    uint32_t ring_count;
    double sum;
};
```

`kMaxRollingWindow` is a `constexpr` upper bound (e.g. 4096). Operators
that need larger windows must specialize. Hard upper bounds — Tiger
Style.

## Worked diff: EMA before / after

```diff
- inline MorselProcessorFunc CreateEMAProcessor(
-     int32_t value_col,
-     int32_t period,
-     const std::string& output_col = "ema") {
-     return [value_col, period, output_col](
-         std::shared_ptr<arrow::RecordBatch> batch
-     ) -> std::shared_ptr<arrow::RecordBatch> {
-         auto values = std::static_pointer_cast<arrow::DoubleArray>(
-             batch->column(value_col));
-         const double alpha = 2.0 / (period + 1);
-         std::deque<double> window;
-         arrow::DoubleBuilder builder;
-         /* ... ema loop ... */
-         std::shared_ptr<arrow::Array> result;
-         builder.Finish(&result);
-         return AppendColumn(batch, output_col, result);
-     };
- }
+ struct EMAState { ... };  // (above)
+
+ void execute_ema(
+     const BoltBatch* const* in,
+     BoltBatch* const* out,
+     Range range,
+     Arena* arena,
+     void* state) noexcept;
+
+ EMAState* make_ema_state(Arena* graph_arena, int32_t value_col,
+                          int32_t out_col, int32_t period) noexcept;
```

Allocations per batch: **0** (vs 5 in the Arrow version).
State carry across batches: **correct** (vs broken).
Type dispatch: **compile-time** (vs `dynamic_pointer_cast`).
Column names in hot path: **none** (vs `std::string`).
Refcount per call: **none** (vs every batch).

## Tier 1 port template (mechanical translation)

For each Tier 1 kernel:

1. Define a `<KernelName>OpDesc` POD with column indices
2. Write `execute_<kernelname>(in, out, range, arena, state)` body
   that loops `[range.start, range.end)`
3. Use `BOLT_RESTRICT` on every input/output pointer
4. Use `bolt::kernels::sum<T>` etc. where the math is already in
   Bolt's kernel library
5. Add a constructor `make_<kernelname>_desc(Arena*, ...)`
6. Add to the dataflow operator registry as a known kind tag
7. Bench against the Arrow version; expect 5-20× faster

Estimated ~1-2 hours per Tier 1 kernel once the template is set;
26 kernels = ~50-100 hours total.

## Tier 2 port template (stateful)

Same as Tier 1 plus:

8. Define `<KernelName>State` POD (extends the `OpDesc` with running
   state)
9. Replace `std::deque` / `std::vector` window with fixed-size ring
   buffer in the state struct (constexpr cap)
10. State struct allocated in graph arena at compile time
11. Add reset() to zero state for graph re-entry
12. Bench against the Arrow version

Estimated ~3-4 hours per Tier 2 kernel; 30 kernels = ~90-120 hours.

## Tier 3 deferral

Tier 3 kernels (Hayashi-Yoshida, lead-lag, pairs spread, triangular
arb, VPIN) need:

- **Event-time windowing** — group by event timestamp range, not
  morsel boundary
- **Keyed state** — separate state per symbol / pair / instrument

Neither exists in `bolt::dataflow` yet. Plan: Wave 13 lands these
primitives, then Tier 3 kernels port. Don't try to special-case Tier
3 in the runtime — wait for the framework to grow up to them.

## Per-kernel progress reports

Per the user's "report per piece" rule, each ported kernel gets its
own log file:

```
bolt/docs/research/fintech-port-log-<kernel>.md
```

Contents per file:
- Pre/post benchmark numbers (ns/row, throughput)
- Line-count delta
- Any state-contract change required
- Numerical-stability notes if relevant (Welford vs Kahan vs naive)

## What we adopt

- **POD operator state** pinned in graph arena at compile time
- **Fixed-size ring buffers** in state structs for rolling windows;
  `constexpr` upper bound
- **Compile-time column-index binding** in the descriptor; no
  string lookups in the hot path
- **`BOLT_RESTRICT` on every kernel pointer** — non-negotiable for
  vectorization
- **One progress report per kernel ported** (followups + bench delta)

## What we skip

- **`std::deque` / `std::vector` for windows** — fixed ring only
- **Lambda capture for state** — broken across batches; doesn't
  survive graph re-entry
- **`arrow::Builder` chains** — every allocation must come from the
  arena
- **`dynamic_pointer_cast` per column** — type known at compile time
  via the operator descriptor
- **`std::string` column names in hot dispatch** — descriptor has
  `int32_t` column indices

## Followups

- Settle the `kMaxRollingWindow` constant (4096? 65536?). Measure
  cache footprint.
- Decide on numerical-stability default for rolling stats (Welford
  vs Neumaier vs naive). Probably Welford for rolling variance,
  Neumaier for sum.
- Tier 3 design — event-time windowing, watermarks, keyed state.
  Separate research docs in Wave 13.
