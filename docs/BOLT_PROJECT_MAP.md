# Bolt Project Map

## Overview

Bolt is a zero-dependency columnar execution library embedded inside Chukonu.
It replaces Apache Arrow's C++ runtime on the hot path while maintaining
Arrow format compatibility at I/O boundaries.

Total: ~2,600 lines of header-only C++20. Compiles in seconds.

## Directory Layout

```
chukonu/
├── include/bolt/           ← Headers (the library)
│   ├── bolt_types.h                   Type enum, StringView, Schema, Arrow C Data ABI
│   ├── bolt_arena.h                   Per-thread bump allocator (9,600x faster than malloc)
│   ├── bolt_channel.h                 Lock-free SPSC/MPSC ring buffers (25x faster than mutex)
│   ├── bolt_column.h                  Adaptive column + BoltBatch + BitmapIndex
│   ├── bolt_branchless.h              Branchless kernels + micro-adaptive dispatch
│   ├── bolt_scheduler.h               Task ring, worker pool, phase barriers
│   ├── kernels/
│   │   ├── bolt_numeric.h             Numeric kernel matrix: filter/agg/arith/cast (Wave A4)
│   │   └── fintech/
│   │       ├── column_primitives.h   Phase 1: diff/lag/log/sum_of_squares/running_max_drawdown
│   │       ├── state.h               Phase 2: RollingRing / WelfordAccumulator / EmaState
│   │       ├── midprice.h            Phase 4a: (bid+ask)/2
│   │       ├── microprice.h          Phase 4a: size-weighted midprice
│   │       ├── l1_imbalance.h        Phase 4a: order-book imbalance
│   │       ├── quoted_spread.h       Phase 4a: ask - bid
│   │       ├── effective_spread.h    Phase 4a: 2·|exec - mid|
│   │       ├── signed_volume.h       Phase 4a: int64 side · double qty
│   │       ├── arrival_price_impact.h Phase 4a: TCA bps vs arrival mid
│   │       ├── forward_points.h      Phase 4a: FX IRP forward - spot
│   │       ├── funding_cost.h        Phase 4a: position · price · rate
│   │       ├── cip_basis.h           Phase 4a: CIP basis (log-form)
│   │       ├── funding_apr.h         Phase 4a: rate · periods · 100
│   │       ├── perp_fair_price.h     Phase 4a: spot · (1 + fr·h/8)
│   │       ├── log_returns.h        Phase 4b: log(p[i]/p[i-1]), 0 at i=0
│   │       ├── ofi.h                Phase 4b: buy_vol - sell_vol
│   │       ├── implementation_shortfall.h Phase 4b: 10000·(exec-bench)/bench bps
│   │       ├── open_interest_delta.h Phase 4b: oi[i]-oi[i-1], 0 at i=0
│   │       ├── corwin_schultz.h     Phase 4b: high-low pair spread estimator
│   │       ├── garman_klass.h       Phase 4b: OHLC volatility
│   │       ├── parkinson.h          Phase 4b: high-low volatility
│   │       ├── rogers_satchell.h    Phase 4b: drift-independent OHLC vol
│   │       ├── realized_var.h       Phase 4b: Σr² broadcast
│   │       ├── bipower_var.h        Phase 4b: jump-robust (π/2)·Σ|r_i||r_{i-1}|
│   │       ├── stale_quote_detector.h Phase 4b: ts-lastUpdate>max_age flag
│   │       ├── price_band_guard.h   Phase 4b: |order/ref-1|≥band flag
│   │       ├── fat_finger_guard.h   Phase 4b: band OR notional flag
│   │       ├── circuit_breaker.h    Phase 4b: |p/ref-1|>thr flag (batch-local)
│   │       ├── monotonic_deque.h    Phase 5.R: IndexRing<kCap> sliding-window extrema
│   │       ├── sma.h                Phase 5.R: rolling mean (SMA)
│   │       ├── bollinger_bands.h    Phase 5.R: SMA ± k·stddev (3 cols)
│   │       ├── donchian_channel.h   Phase 5.R: rolling high/low channel (2 cols)
│   │       ├── atr.h                Phase 5.R: rolling mean of True Range
│   │       ├── vwap_twap.h          Phase 5.R: rolling VWAP + rolling TWAP
│   │       ├── rolling_min_max.h    Phase 5.R: O(1) amortised rolling min/max
│   │       ├── stochastic_osc.h     Phase 5.R: Stochastic %K oscillator
│   │       ├── ema.h                Phase 3 / P5.E.1: worked Tier 2 template; arena-pinned EMAState
│   │       ├── ewma.h               Phase 5.E.2: caller-supplied alpha EWMA
│   │       ├── ewcov.h              Phase 5.E.3: two-variable EW covariance (RiskMetrics lambda)
│   │       ├── macd.h               Phase 5.E.4: MACD line/signal/histogram (3 chained EmaState)
│   │       ├── rsi.h                Phase 5.E.5: Wilder RSI (alpha = 1/period)
│   │       ├── welford_mean_var.h   Phase 5.W.1: streaming (mean, pop_var) per row
│   │       ├── rolling_std.h        Phase 5.W.2: rolling pop_stddev (rescan-Welford)
│   │       ├── rolling_zscore.h     Phase 5.W.3: (x - rolling_mean)/rolling_stddev
│   │       ├── sharpe_ratio.h       Phase 5.W.4: scalar Sharpe, sample-var, annualized
│   │       ├── sortino_ratio.h     Phase 5.W.5: scalar Sortino, downside-dev
│   │       ├── rolling_correlation.h Phase 5.W.6: rolling Pearson corr (two rings, rescan)
│   │       ├── rolling_skew.h       Phase 5.W.7: rolling raw skewness (pop moments)
│   │       ├── rolling_kurt.h       Phase 5.W.8: rolling EXCESS kurtosis
│   │       ├── autocorr.h           Phase 5.W.9: rolling Pearson autocorr at lag k
│   │       └── risk_metrics_vol.h   Phase 5.W.10: EWMA of r² (RiskMetrics'94), emits stddev
│   └── README.md                      Full project documentation
│
├── src/bolt/                        ← Implementation notes
│   └── CMAKE_PATCH.md                 3-line CMake integration guide
│
├── tests/unit/bolt/                 ← Tests
│   └── test_bolt_primitives.cpp       GTest: types, arena, channels, columns, Arrow export
│
├── benchmarks/                      ← Performance validation
│   └── bench_bolt.cpp                 Arena vs malloc, SPSC vs mutex, epoch swap, COW
│
└── docs/                            ← Design documents
    ├── BOLT_DESIGN.md                 Phase 1: gap analysis, measured benchmarks
    ├── BOLT_COLUMN_FORMAT.md          Phase 2: stats, sidecars, adaptive encoding
    ├── BOLT_INDEPENDENCE.md           Zero-dependency architecture + interop matrix
    ├── BOLT_ACERO_COMPONENTS.md       What Acero provides, what Bolt replaces
    ├── BOLT_RESEARCH_NOTES.md         Thin pointer to research/
    └── research/                      Per-topic research notes:
        ├── README.md                    Index
        ├── pirk-techniques.md           Pirk et al. (14 papers)
        ├── cglm.md                      SIMD/portability patterns
        ├── scheduler-design.md          DuckDB / Polars / Seastar
        ├── cpu-topology.md              OS topology APIs
        ├── avx512-status.md             AVX-512 dispatch stub + plan
        ├── 1brc.md                      1 Billion Row Challenge
        ├── json-fionn.md                fionn vs simdjson
        ├── questdb-symbol-vs-fsst.md    SYMBOL layout + cardinality wall;
        │                                 FSST disjoint, Bolt Dict+Bitmap covers SYMBOL
        └── questdb-symbol-code-audit.md source-code audit of the SYMBOL claims
                                          (questdb/questdb@master): 7/8 confirmed,
                                          .c/.o dict vs .k/.v index correction
```

## Design Principles

These are non-negotiable:

1. **Zero external dependencies.** Bolt compiles with only a C++20 compiler.
   No Arrow, no protobuf, no gRPC, no boost, no vcpkg.

2. **No runtime overhead.** No exceptions, no RTTI, no smart pointers, no
   virtual dispatch, no `std::string`, no `std::vector` in hot structs.
   Every function is `noexcept`. OOM returns `nullptr`.

3. **Arena-managed memory.** Per-thread bump allocators. Reset per morsel
   epoch. No individual frees. Zero allocator contention between threads.

4. **Branchless inner loops.** Predicated execution (bool-to-int conditional
   advance), CMOV, SIMD masking. Micro-adaptive kernel selection based on
   column statistics: branching at extreme selectivities, branchless in the
   middle range (Pirk ICDE 2025).

5. **Arrow-compatible at boundaries.** Zero-copy export via Arrow C Data
   Interface (`ArrowSchema`/`ArrowArray`). No libarrow link needed. Any Arrow
   consumer reads Bolt columns directly.

6. **Better than Arrow internally.** Multi-format columns (Flat/Constant/
   Dictionary/Sequence/View). Inline 64-byte statistics block. Sidecar
   indexes (bitmap, bloom, sort, hash) arena-allocated per epoch.

## Provenance

Ideas come from five sources, all documented with references:

| Source | What We Took |
|--------|-------------|
| **Venus ECS** (our game engine) | Double-buffer COW, arena-per-frame, BH-tree sidecar, X-macro dispatch, deferred ops ring, spin-then-yield job system |
| **DuckDB** | Multi-format vectors, 2048 vector size, zone maps, push-based execution |
| **QuestDB** | Symbol type (dictionary + bitmap as separate concerns), mmap columns |
| **Polars/arrow2** | German-style string views, Rust-inspired ownership model |
| **Pirk et al. (CWI/MIT/Imperial)** | Predicated partitioning, micro-adaptive kernels, cache-conscious layout, composable kernel algebra, LightSaber parallel aggregation |
| **Chronicle Queue / LMAX** | Lock-free SPSC/MPSC, cache-line padding, pre-allocated ring buffers |

## Dependencies on Bolt (within Chukonu)

Bolt is currently standalone — nothing depends on it yet. Integration is
incremental. The migration path:

```
Phase 1 (current):  Bolt headers exist alongside Arrow-based operators
Phase 2:            New operators written against BoltBatch
Phase 3:            All internal operators use BoltBatch
Phase 4:            Arrow at boundaries only (Parquet, Flight, Python)
Phase 5:            Own Parquet reader, Bolt Wire Protocol, full independence
```

## Quick Start

```bash
# Everything compiles as part of the Chukonu build
cd chukonu/build
cmake .. -DCHUKONO_BUILD_TESTS=ON
make test_bolt_primitives
./test_bolt_primitives     # 25+ tests, zero Arrow dependency

make bench_bolt
./bench_bolt               # Arena, channel, epoch, COW benchmarks
```
