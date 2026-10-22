# Bolt Project Map

Listener accept retirement is under implementation in `api/core/accept_retirement.h`
and the reactor backends. The checked listener API retains ownership on every
non-complete result; legacy stop remains a blocking owner-progress operation.
The new cross-platform lifecycle target is registered, but qualification and
main integration are pending. [Contract](research/accept-retirement.md).

IDOT's compiled scalar/dispatch sources use baseline compiler options; core ISA
requirements remain on the consumer interface. Shared layout/feature definitions
still reach every compiled tier. The existing IDOT CTest benchmark runs serially;
its speedup floor is unchanged. [Contract](research/idot-isa-isolation.md).

`bolt_seqlock.h` stores trivially-copyable snapshots in inline atomic cells;
`try_read` has a fixed attempt bound and preserves caller lifetime pins.
The legacy `read()` wrapper still spins until success.
[Representation, ordering proof and qualification limits](research/seqlock-atomic-payload.md).

`lakehouse/iceberg/pinned_read.h` opens exact retained metadata over a bounded
caller-admitted `object_store/pinned.h` directory. Manifest/list/data/delete
reads use only copied strong versions; `object_store/s3_uri.h` maps original
absolute URIs once against the store namespace. Signed snapshot selection is
separate from legacy latest sentinels. Flat data and position deletes bind by
retained Parquet field IDs; `parquet_read_meta_source_flat_strict` validates the
exact fetched footer without extra transport calls. Full graph admission and
durable publication remain caller responsibilities. Unsupported types/options
are enumerated in the pinned-read public header; this is not a complete mounted
Iceberg service or read/write cache.

`lakehouse/iceberg/metadata_io.h` loads one caller-selected immutable metadata
object through strong conditional ranges, distinct finite retained/scratch
arenas, exact table identity and complete SHA-256. It performs no discovery or
repinning; a declared raw JSON identity subset refuses escaped aliases.
[Contract and qualification](research/iceberg-metadata-exact.md).

`thread/bolt_thread_qos.h` / `bolt::thread` provide cold current-thread policy
configuration with explicit native readback. Inherit makes no OS request;
unsupported/failure is explicit. Configuration is not effective scheduling or
latency proof. [Contract](THREAD_QOS.md).

`io/bolt_mapped_ring.h` adds a fixed, physically preallocated page-aware event
ring in `bolt::io_file`: whole-event publication, explicit reader leases,
lossless backpressure, optional gap-reporting overwrite, live logical caps and
separate explicit flush progress. `bolt_mapped_ring_batch.h` encodes native
frames directly into reservations and borrows decoded batches under leases.
Writable mapping mechanics extend `io/bolt_file_io.h`. Owner-local only;
independent-process/restart contracts remain separate.
[API, ownership and measurement scope](research/mapped-event-ring.md).

`wire/xchannel.h` / `xchannel_format.h` admit bounded FORMAT4 prefixes and
peek/consume opaque records over caller-pinned mappings, with exact committed
Roll successor checks. FORMAT3 refuses explicitly. Local framing tests pass;
actual upstream-writer interoperability remains separate.
[Contract and qualification](research/xchannel-reader.md).

`io/bolt_file_io.h::mapped_file_open_bounded` validates inclusive file-size
bounds on the same opened handle before mapping its observed extent. Exact
bounds admit fixed-size inputs without reopening a pathname; later growth
does not expand the view. Caller-owned lifetime still excludes truncation.

`ingest/bolt_parquet_write.h` admits optional integer width/signedness and
millisecond/microsecond/nanosecond timestamp annotations over exact physical
lanes. Unsigned lanes suppress unproved signed statistics; legacy zero metadata
retains its output policy. `kernels/bolt_decimal.h` also validates complete
Decimal128 mantissas against declared precision before cold IO.

`schema/exact_field.h` adds bounded exact descriptor metadata for canonical
schemas: width, decimal/presence/sentinel, temporal and group interpretation.
This metadata-only primitive adds no decoder or planner execution capability.
The metadata layout is defined in [the public header](../include/bolt/schema/exact_field.h).
`bolt_type_validity.h` validates admitted enum values without using diagnostic
names; internal Decimal64 is valid while reserved holes and NUM_TYPES refuse.

`ingest/bolt_arrow_ipc_reader.h` reads bounded Arrow IPC streams and ARROW1
files through immutable spans or the owned file mapper into reusable typed
Bolt batches. Unsupported nested/dictionary/compressed shapes refuse explicitly.
[Contract and independent fixtures](research/arrow-ipc-reader.md).

`interop/arrow_c.h` owns the canonical C Data/C Stream ABI; `arrow_owner.h`
provides immutable snapshots with independent descriptor leases, and
`bolt_arrow_import.h` provides bounded exact flat C Data copy ingress.
`bolt::interop` has no Arrow/Python runtime dependency. Consumer schema sidecars
must retain precision and temporal kind/unit; generic batches alone do not
encode every Arrow logical declaration. Native tests cover ownership, limits,
exact fields and the no-allocation IPC next path. Wider engine integration and
performance qualification remain separate gates.
[Interchange ownership, admission and verification](research/arrow-interchange.md).

`kernels/bolt_integer_filter.h` compares exact signed/unsigned widths with
signed64 literals without narrowing/wrap, including null-mask offsets and
selected rows. [Scope and evidence](research/integer-literal-filters.md).

`ingest/bolt_zstd_stream.h` adds bounded caller-memory block/history-ring
decoding with explicit consumption, backpressure, frame integrity and EOF.
[Contract and scoped native evidence](research/zstd-streaming.md).

`lakehouse/object_store/s3_upload.h` provides bounded synchronous S3 multipart
and streamed PUT for known-size objects; reconciliation and caller-memory
budgets are explicit. [Usage, limits and evidence](research/s3-multipart-upload.md).

`wire/dbn.h` adds borrowed bounded DBN1/2/3 metadata/record framing;
`wire/fast.h` adds the exact stop-bit/presence-map/dictionary FAST1.1 subset.
Typed Databento extraction and T7 venue templates remain in Chukonu.
[Contracts, pins and verification scope](research/dbn-fast-wire.md).

Raw LZ4 supports exact decoded length for Parquet and a bounded-output API for
transports without an uncompressed-size prefix. Both use the same allocation-free
decoder with compile-time policy. [Contract and tests](research/lz4-bounded-transport.md).

## Overview

Bolt is a zero-dependency columnar execution library. It is an alternative to
Apache Arrow's C++ runtime on the hot path, and keeps Arrow format
compatibility at I/O boundaries.

Header-only C++20, built directly by the consumer — including MSVC on
Windows, with no package manager or prebuilt dependency.

Stable persisted byte hashing: `bolt_hash.h` supplies explicit WYHASH3-v1
scalar/accumulator helpers; `kernels/bolt_hash_sv.h` supplies scalar/batch
StringView equivalents. Generic hashing remains tier-selected.
[Contract and three-tier tests](research/persisted-byte-hash-v1-2026-10-08.md).

## Directory Layout

Shared-state layout is selected by `cmake/BoltAlignment.cmake` and exported
through `bolt::core`: cache isolation defaults to 64 bytes, with explicit
128-byte or compiler-target AUTO options; channel slots are independently
64/128 bytes. Header-only callers also default to 64. Changing the default
requires fresh consumer memory and tail-latency evidence; the historical
Apple AUTO policy was not adopted during main convergence. Payload and wire
layouts keep their existing alignment.

The optional compiled `bolt::reactor` also owns native event loops. Its internal
`src/api/net/event_handler_registry.h` layers bounded callback pins and
registration identities over the shared `FdRegistry`. The epoll/kqueue backends
snapshot identities before dispatch and use fixed 256-event batches; callbacks
can remove/re-register handlers without destroying an active callable. Nested
polling returns `EBUSY`. Portable and real-socket tests cover these lifetimes.
IOCP handler migration still needs a cancel-and-drain protocol; see the
[platform I/O audit](research/g2chk-85-platform-io-audit.md).

`include/bolt/wire/sbe{,_fields,_groups}.h` adds generic bounded SBE wire
primitives over `ByteReader`: endian-specialized scalars, header identity,
version/optional fields, explicit nested-group traversal and bounded variable
data. This does not yet include an XML compiler, generated codec or BoltBatch
source. [Contract, evidence and remaining work](research/sbe-bounded-wire.md).

`wire/mold_udp64.h` and `wire/soup_bin_tcp.h` provide allocation-free borrowed
framing for MoldUDP64 downstream datagrams and one SoupBinTCP frame in a TCP
prefix. Whole UDP frames validate before escape; Soup reports NeedMore versus
consumed bytes and checks direction/type lengths. No sockets/session lifecycle
or exchange recovery is implied. Independent framing/overflow tests passed
Release Clang on 2026-10-14. [Contract](research/market-feed-framing.md).

`net/bolt_udp.h` adds `udp_recv_from_checked` alongside the unchanged legacy
receive API. It admits only whole IPv4 datagrams as Complete, distinguishes
Idle/Interrupted/Truncated/Error, and publishes peer/count only on Complete.
The implementation uses at most one receive syscall with no retry or explicit
allocation: POSIX `recvmsg(..., 0)` checks output MSG_TRUNC; Windows
`recvfrom(..., 0)` maps WSAEMSGSIZE. Local macOS qualification passed nine
GTests in one CTest target on 2026-10-16; Windows/Linux native qualification,
latency and CME sequence/recovery remain separate gates.

`Arena::overlaps_backing` admits cold-path external buffers against every
retained arena block, including reset/unused/oversize backing. Callers must
quiesce arena mutation. The bounded scan allocates nothing and does not alter
allocation/reset behavior; range/overflow tests pass. Materialization capture
uses it before any scratch reset to refuse aliased input/output (G2FEAT-143).

```
bolt/
├── include/bolt/           ← Headers (the library)
│   ├── bolt_types.h                   Type enum, StringView, Schema, Arrow C Data ABI
│   ├── bolt_decimal_scale.h           Lightweight shared AVG scale min(s+4,38) (G2CHK-736)
│   ├── bolt_arena.h                   Per-thread bump allocator
│   ├── api/core/stacked_thread.h      Explicit-stack native thread; checked noexcept callback start/join (G2FEAT-248)
│   ├── bolt_channel.h                 Lock-free SPSC/MPSC ring buffers
│   ├── bolt_column.h                  Adaptive column + BoltBatch + BitmapIndex
│   ├── bolt_branchless.h              Branchless kernels + micro-adaptive dispatch
│   ├── bolt_scheduler.h               Task ring, worker pool, phase barriers
│   ├── io/
│   │   └── bolt_crc32c.h              I/O-adjacent pure-compute: CRC32C (Castagnoli),
│   │                                   SSE4.2 / ARM CRC / slicing-by-8 software fallback
│   ├── kernels/
│   │   ├── bolt_binsearch.h           Branchless binary search (i64/u64/f64) —
│   │   │                               exact + lower_bound + upper_bound, cmov loop
│   │   ├── bolt_utf8_clone.h          Checked min(span,sum) single-base spill clone (G2CHK-714)
│   │   ├── bolt_numeric.h             Numeric kernel matrix: filter/agg/arith/cast (Wave A4)
│   │   ├── promql.h                   PromQL kernels aggregator (rate/deriv/histogram)
│   │   ├── promql_rate.h              rate/increase/delta/irate/idelta/resets/changes —
│   │   │                               counter-reset-aware, boundary extrapolation
│   │   ├── promql_deriv.h             deriv/predict_linear + OLS linear_regression
│   │   ├── promql_histogram.h         histogram_quantile — classic le-bucket interpolation
│   │   │                               (Prometheus-exact; oracle-tested test_promql_kernels)
│   │   └── fintech/
│   │       ├── book_types.h          Runtime-admitted exact values, scales, cuts and statuses
│   │       ├── book_l2.h             Price-keyed AVL book, atomic chunked events/replacements
│   │       ├── book_mbp.h            Distinct ranked MBP, explicit bounded rank shifts
│   │       ├── book_detail.h         Caller-storage banks, sparse before-images and lifecycle
│   │       ├── book_l2_tree.h        Array-indexed AVL/ordered-neighbor implementation
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
│   │       ├── risk_metrics_vol.h   Phase 5.W.10: EWMA of r² (RiskMetrics'94), emits stddev
│   │       ├── sorted_ring.h        Phase 5.S primitive: fixed-cap rolling sorted window
│   │       │                         (O(w) push, O(1) quantile/median; no heap)
│   │       ├── historical_var.h     Phase 5.S.4: rolling (1-conf) quantile VaR
│   │       ├── historical_cvar.h    Phase 5.S.5: mean of sorted-tail (Expected Shortfall)
│   │       ├── med_rv.h             Phase 5.S.8: batch-scalar median realised volatility
│   │       ├── outlier_flag_mad.h   Phase 5.S.14: rolling median + 1.4826·MAD flag (int64)
│   │       ├── max_drawdown.h       Phase 5.S.1: streaming peak, per-row absolute drawdown
│   │       ├── drawdown.h           Phase 5.S.2: streaming peak, per-row fractional drawdown
│   │       ├── cusum.h              Phase 5.S.3: two-sided Page CUSUM with threshold reset
│   │       ├── almgren_chriss.h     Phase 5.S.13: optimal-execution schedule (cyclic emit)
│   │       ├── throttle_check.h    Phase 5.S.15: sliding-window event-count tripwire
│   │       ├── kalman1d.h           Phase 5.S.6: scalar 1-D Kalman filter (seed x=obs[0])
│   │       ├── kalman_hedge_ratio.h Phase 5.S.7: Kalman hedge ratio (emits ratio + spread)
│   │       ├── kyle_lambda.h        Phase 5.S.11: rolling OLS slope of Δp on signed volume
│   │       ├── roll_spread.h        Phase 5.S.12: 2·sqrt(-cov(Δp, Δp_{-1})), clamped ≥0
│   │       ├── cornish_fisher_var.h Phase 5.S.9: batch-scalar VaR with skew/kurt correction
│   │       └── amihud.h             Phase 5.S.10: per-row |r|/v with v>0 cmov guard
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

## Adoption path

Bolt is designed to be adopted incrementally by an Arrow-based engine rather
than in one cutover. The path the original consumers followed:

```
Phase 1 (current):  Bolt headers exist alongside Arrow-based operators
Phase 2:            New operators written against BoltBatch
Phase 3:            All internal operators use BoltBatch
Phase 4:            Arrow at boundaries only (Parquet, Flight, Python)
Phase 5:            Own Parquet reader, Bolt Wire Protocol, full independence
```

## Quick Start

```bash
cmake -S . -B build -DBOLT_BUILD_TESTS=ON -DBOLT_BUILD_BENCHMARKS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Presets: `release`, `debug`, `msvc`, `ninja-msvc`, `clang-cl`. GTest is fetched
via `FetchContent` when not found locally, so the build is self-contained on
Windows without vcpkg.

G2CHK-714 clone bounds and format/refusal contracts: [Utf8 clone copy bound](research/utf8-clone-copy-bound.md).

Event-loop compile-time allocation bounds live in `include/bolt/api/net/event_loop_limits.h` (kind B): 16 callback pins and 256 readiness events. The registry and epoll/kqueue share these declarations.
