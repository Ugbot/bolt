# Bolt fintech kernel port — from chukonu/gestalt

## Source

`C:\code\gestalt\chukonu\include\chukonu\fintech\kernels.h` — 74 kernels
written against Apache Arrow, `MorselProcessorFunc` signature, 5
allocations per batch, `std::deque` rolling windows, `std::string`
column names in the hot path. This plan ports the survivable subset
into Bolt's zero-allocation Tiger-Style shape.

Template doc: `docs/research/gestalt-kernel-adapter.md`.

## Why port them

They are the forcing function for Bolt's operator surface. The API
shape that satisfies all 74 kernels is the shape that ships —
ArrowAPI+`std::shared_ptr`+heap-per-batch is what made the
predecessor too slow; porting proves Bolt's primitives (arena-pinned
POD state, compile-time column binding, `BOLT_RESTRICT` everywhere)
actually cover a realistic workload, not just the TPC-H microbenches.

## Categorisation

- **Tier 0 — primitives** (new column ops + state types that every
  later tier composes on). ~7 items.
- **Tier 1 — mechanical arithmetic** (~26 kernels). Stateless within
  a batch; pure column math. Direct port.
- **Tier 2 — single-batch stateful** (~30 kernels). POD state struct
  + a `RollingRing` / `Welford` / `Ema` substrate.
- **Tier 3 — framework-gated** (~11 kernels). Need event-time
  windows + keyed state. Wait for Wave 3 (`bolt::dataflow`).

## Ordering contract

Each item's dependencies are called out inline. Phases execute in
order; items **within** a phase are independent and can be parallel-
agented. No item lands without: POD `State` struct, `execute_<name>`
free function, round-trip test against a hand-computed reference,
design-log entry if numerical-stability choice is made.

---

## Phase 1 — column primitives (stateless, additive)

No framework work, no new state types. These are plain free
functions that compose on existing `bolt_branchless.h` + `bolt_numeric.h`.

| # | Kernel | Why | Deps |
|---|--------|-----|------|
| P1.1 | **`diff_column<T>`** — `out[i] = data[i] - data[i-1]`; `out[0] = 0` | Unlocks LogReturns, Diff, OpenInterestDelta, OFI, and every change-of-price kernel | none |
| P1.2 | **`lag_column<T>(k)`** — `out[i] = data[i-k]`; zero-fill first k | Unlocks Lag, LeadLag, Autocorr, pairs math | none |
| P1.3 | **`log_column<T>`** — element-wise natural log, double out | Unlocks LogReturns, Parkinson, RogersSatchell, GarmanKlass | none |
| P1.4 | **`sum_of_squares<T>`** — fused Σx², shape mirrors `filter_sum_gt` | Unlocks RealizedVar, RiskMetricsVol, Parkinson-squared | none |
| P1.5 | **`running_max_drawdown<T>`** — single-pass `max_so_far` tracking, emit `(max_so_far - x[i])` | Unlocks MaxDrawdown, Drawdown | none |

## Phase 2 — state substrate (POD + method body only)

| # | State | Why | Deps |
|---|-------|-----|------|
| P2.1 | **`RollingRing<T, kCap>`** — fixed ring + running sum + push/pop | Substrate for SMA, BollingerBands, Donchian, ATR, MedRV, RiskMetricsVol, RollingMin/Max | none |
| P2.2 | **`WelfordAccumulator`** — `(count, mean, m2)` + scalar `update(x)` | Substrate for RollingStd, RollingZScore, Sharpe, Sortino, RollingCorr, Skew, Kurt, WelfordMeanVar | none |
| P2.3 | **`EmaState`** — `(alpha, value, initialized)` POD + `update(x)` | Substrate for EMA, EWMA, EWCOV, MACD (3 EMAs), RiskMetricsVol | none |

## Phase 3 — worked template (first Tier 2 port)

The plan's worked example — proves the state-substrate shape + bench
methodology is correct before we fan out.

| # | Kernel | Why | Deps |
|---|--------|-----|------|
| P3.1 | **`execute_ema` + `EMAState`** (worked example from gestalt-kernel-adapter.md) | Validates P2.3 state-pinning, benchmark vs Arrow EMAProcessor | P2.3 |

## Phase 4 — Tier 1 mechanical (parallel-agent safe)

All stateless within a batch. Each lands with `execute_<kernel>` +
`<Kernel>OpDesc` + round-trip test. Ordered by tick-feed usefulness.

| # | Kernel | Formula | Deps |
|---|--------|---------|------|
| P4.1  | **Midprice** | `(bid + ask) / 2` | add, mul |
| P4.2  | **Microprice** | `(ask·bid_sz + bid·ask_sz) / (bid_sz + ask_sz)` | add, mul, div |
| P4.3  | **L1Imbalance** | `(bid_sz - ask_sz) / (bid_sz + ask_sz)` | add, sub, div |
| P4.4  | **QuotedSpread** | `ask - bid` | sub |
| P4.5  | **EffectiveSpread** | `2 · \|mid - trade_price\|` | sub, babs, mul |
| P4.6  | **LogReturns** | `log(p[i] / p[i-1])` | P1.3 log, P1.1 diff or div |
| P4.7  | **SignedVolume** | `sign(side) · qty` | mul |
| P4.8  | **OFI** | order-flow imbalance (bid/ask-size deltas with signed aggregation) | P1.1 diff |
| P4.9  | **ArrivalPriceImpact** | arithmetic over price + arrival | sub, div |
| P4.10 | **ImplementationShortfall** | `Σ (exec_px - arrival) · qty` | mul, sum |
| P4.11 | **ForwardPoints** | `forward - spot` | sub |
| P4.12 | **FundingCost** | time-weighted funding arithmetic | mul, add |
| P4.13 | **CIPBasis** | covered-interest-parity basis arithmetic | sub, div |
| P4.14 | **FundingAPR** | funding-rate APR conversion | mul |
| P4.15 | **PerpFairPrice** | perp fair-price arithmetic | add, mul |
| P4.16 | **OpenInterestDelta** | `oi[i] - oi[i-1]` | P1.1 diff |
| P4.17 | **CorwinSchultz** | OHLC-based spread estimator | log, arithmetic |
| P4.18 | **GarmanKlass** | OHLC volatility estimator | P1.3 log |
| P4.19 | **Parkinson** | `ln(high/low)²` | P1.3 log, P1.4 sumSq |
| P4.20 | **RogersSatchell** | OHLC volatility | P1.3 log |
| P4.21 | **RealizedVar** | `Σ r²` over returns | P1.3 log, P1.4 sumSq |
| P4.22 | **BipowerVar** | `Σ \|r_i·r_{i-1}\|` | P1.2 lag, babs, mul |
| P4.23 | **StaleQuoteDetector** | timestamp diff > threshold → flag | P1.1 diff, filter |
| P4.24 | **PriceBandGuard** | `\|px - ref\| > band` → flag | sub, babs, filter |
| P4.25 | **FatFingerGuard** | qty > ceiling or px > band → flag | filter |
| P4.26 | **CircuitBreaker** | running `max\|Δ\|` threshold → flag | filter |

## Phase 5 — Tier 2 stateful (parallel-agent safe after P2/P3)

Grouped by substrate so agents can batch by skill. Each item lands
with `<Kernel>State` + `execute_<kernel>` + numerical-stability note
if relevant.

### P5.R — RollingRing-based

| # | Kernel | Why | Deps |
|---|--------|-----|------|
| P5.R.1 | **SMA** (rolling mean) | most-used indicator | P2.1 |
| P5.R.2 | **BollingerBands** (SMA ± k·stddev) | adds Welford | P2.1, P2.2 |
| P5.R.3 | **DonchianChannel** (rolling min/max) | needs monotonic deque | P2.1 |
| P5.R.4 | **ATR** (avg true range) | rolling mean of TR | P2.1 |
| P5.R.5 | **VWAP / TWAP** (rolling, per-window) | rolling price-weighted sum | P2.1 |
| P5.R.6 | **RollingMin / RollingMax** | monotonic deque | P2.1 |
| P5.R.7 | **StochasticOsc** | `(x - low) / (high - low)` rolling | P5.R.3 |

Phase 5.R status (landed):
- [x] P5.R.1 SMA
- [x] P5.R.2 BollingerBands
- [x] P5.R.3 DonchianChannel
- [x] P5.R.4 ATR  (rolling-mean form — Wilder's EMA shape deferred to 5.E)
- [x] P5.R.5 VWAP / TWAP  (rolling; chukonu's batch-aggregate shape dropped)
- [x] P5.R.6 RollingMin / RollingMax
- [x] P5.R.7 StochasticOsc (%K only; %D is SMA(K,3) downstream)

### P5.W — Welford-based (moments)

| # | Kernel | Why | Deps |
|---|--------|-----|------|
| P5.W.1 | **WelfordMeanVar** | direct exposure | P2.2 |
| P5.W.2 | **RollingStd** | `sqrt(WelfordVar)` | P2.2 |
| P5.W.3 | **RollingZScore** | `(x - mean) / stddev` | P5.W.2 |
| P5.W.4 | **SharpeRatio** | `(mean - rfr) / stddev` (scalar) | P2.2 |
| P5.W.5 | **SortinoRatio** | mean / downside-dev | P2.2 |
| P5.W.6 | **RollingCorrelation** | two-variable Welford | P2.2 extension |
| P5.W.7 | **RollingSkew** | 3rd central moment online | P2.2 extension |
| P5.W.8 | **RollingKurt** | 4th central moment online | P2.2 extension |
| P5.W.9 | **Autocorr** | `cov(x, lag(x, k)) / var(x)` | P1.2, P2.2 |
| P5.W.10 | **RiskMetricsVol** | EWMA of r² (variance) | P2.3 + P1.4 |

Phase 5.W status (landed):
- [x] P5.W.1 WelfordMeanVar (streaming; chukonu uses POP variance, not sample)
- [x] P5.W.2 RollingStd (pop-var rescan; no warmup NaN — chukonu parity)
- [x] P5.W.3 RollingZScore (stddev==0 → 0.0, chukonu parity, not NaN)
- [x] P5.W.4 SharpeRatio (scalar broadcast; SAMPLE variance; annualized sqrt(periods))
- [x] P5.W.5 SortinoRatio (scalar broadcast; downside_dev divides by n not n_downside)
- [x] P5.W.6 RollingCorrelation (rescan; w<2 → 0.0)
- [x] P5.W.7 RollingSkew (raw/biased; w<3 → 0.0)
- [x] P5.W.8 RollingKurt (EXCESS kurtosis, kurt - 3; w<4 → 0.0)
- [x] P5.W.9 Autocorr (single ring size w+lag; first w+lag-1 rows → 0.0)
- [x] P5.W.10 RiskMetricsVol (seed var₀=r₀²; emits stddev not variance)

Rolling-window Welford stability: for the rolling variants (P5.W.2/3/6/7/8)
we rescan the `RollingRing` on each row rather than doing a Welford
decrement. O(w) per row, numerically safe, cache-hot for small windows.
See `docs/research/design-log.md` entry "Rolling-window Welford — rescan
vs decrement".

### P5.E — EMA-based

| # | Kernel | Why | Deps |
|---|--------|-----|------|
| P5.E.1 | **EMA** | done in Phase 3 | — |
| P5.E.2 | **EWMA** | equivalent to EMA | P2.3 |
| P5.E.3 | **EWCOV** | two-variable EMA | P2.3 |
| P5.E.4 | **MACD** | 3 EMAs chained (12, 26, 9) | P5.E.1 |
| P5.E.5 | **RSI** | Wilder's smoothing (exponential) | P2.3 |

### P5.S — single-state / specialised

| # | Kernel | Why | Deps |
|---|--------|-----|------|
| P5.S.1 | **MaxDrawdown** | running max tracking | P1.5 |
| P5.S.2 | **Drawdown** | per-row drawdown from running max | P1.5 |
| P5.S.3 | **CUSUM** | cumulative sum with threshold reset | none |
| P5.S.4 | **HistoricalVaR** | sorted ring for rolling quantile | new primitive: sorted ring |
| P5.S.5 | **HistoricalCVaR** | mean of tail quantile | P5.S.4 |
| P5.S.6 | **Kalman1D** | single-state Kalman update | POD state |
| P5.S.7 | **KalmanHedgeRatio** | 1D Kalman on regression | P5.S.6 |
| P5.S.8 | **MedRV** | median-based realized vol | rolling median |
| P5.S.9 | **CornishFisherVaR** | parametric VaR with skew/kurt correction | P5.W.7, P5.W.8 |
| P5.S.10 | **Amihud** | `\|r\| / volume`, rolling mean | P2.1 |
| P5.S.11 | **KyleLambda** | rolling regression slope | P2.2 extension |
| P5.S.12 | **RollSpread** | `2·sqrt(-cov(Δp, Δp_{-1}))` | P1.1 diff, P2.2 |
| P5.S.13 | **AlmgrenChriss** | optimal-execution schedule, scalar math | POD state |
| P5.S.14 | **OutlierFlagMAD** | median + MAD | rolling median |
| P5.S.15 | **ThrottleCheck** | count per interval | time-window ring |

## Phase 6 — Tier 3 (framework-gated, DEFERRED)

Require event-time windows + keyed state. Wait for `bolt::dataflow`
Wave 3. NOT scheduled in this plan.

- Hayashi-Yoshida Cov
- Hayashi-Yoshida Corr
- XCorr LeadLag
- PairsSpread
- TriangularArb
- VPIN
- LeadLag (cross-series)

---

## Execution mechanics

Module path: `include/bolt/kernels/fintech/`. One header per kernel
(or per tightly-coupled group). Operators follow the gestalt-kernel-
adapter.md template verbatim — `<KernelName>OpDesc` POD +
`execute_<kernelname>(in, out, range, arena, state)` signature.

Tests: `tests/test_bolt_fintech_<phase>.cpp`, one file per phase so
failures cluster. Each test is a round-trip against a scalar reference
computed inline (no Arrow dependency to compare against in-tree).

Design-log: each phase appends a single entry when it lands, plus
per-kernel entries only for ones making a numerical-stability
decision (Welford variant, Kahan summation, etc.).

## Checkbox tracker

Legend: `[ ]` open · `[~]` in progress · `[x]` done

### Phase 1 (column primitives)
- [x] P1.1 diff_column
- [x] P1.2 lag_column
- [x] P1.3 log_column
- [x] P1.4 sum_of_squares
- [x] P1.5 running_max_drawdown

### Phase 2 (state substrate)
- [x] P2.1 RollingRing<T, kCap>
- [x] P2.2 WelfordAccumulator
- [x] P2.3 EmaState

### Phase 3 (worked EMA)
- [x] P3.1 execute_ema  (include/bolt/kernels/fintech/ema.h; Tier 2 template
  validated — POD state, arena-pinned via `make_ema_state`, persists across
  `execute_ema` calls, zero hot-path allocation. Also satisfies P5.E.1.)

### Phase 4 (Tier 1 mechanical — 26 kernels)
- [x] P4.1 Midprice  · [x] P4.2 Microprice · [x] P4.3 L1Imbalance
- [x] P4.4 QuotedSpread · [x] P4.5 EffectiveSpread · [x] P4.6 LogReturns
- [x] P4.7 SignedVolume · [x] P4.8 OFI · [x] P4.9 ArrivalPriceImpact
- [x] P4.10 ImplementationShortfall · [x] P4.11 ForwardPoints · [x] P4.12 FundingCost
- [x] P4.13 CIPBasis · [x] P4.14 FundingAPR · [x] P4.15 PerpFairPrice
- [x] P4.16 OpenInterestDelta · [x] P4.17 CorwinSchultz · [x] P4.18 GarmanKlass
- [x] P4.19 Parkinson · [x] P4.20 RogersSatchell · [x] P4.21 RealizedVar
- [x] P4.22 BipowerVar · [x] P4.23 StaleQuoteDetector · [x] P4.24 PriceBandGuard
- [x] P4.25 FatFingerGuard · [x] P4.26 CircuitBreaker

### Phase 5 (Tier 2 stateful — 32 kernels)
RollingRing, Welford, EMA, and specialised groups — see tables above.

#### P5.R (RollingRing-based)
- [x] P5.R.1 SMA  (`include/bolt/kernels/fintech/sma.h`)
- [x] P5.R.2 BollingerBands  (`include/bolt/kernels/fintech/bollinger_bands.h`;
  two RollingRings over x and x², population stddev — deviates from Welford
  to stay O(1) amortised under eviction; cancellation guard clamps var≥0)
- [x] P5.R.3 DonchianChannel  (`include/bolt/kernels/fintech/donchian_channel.h`;
  two IndexRing<kCap> monotonic deques; emits upper/lower, mid dropped)
- [x] P5.R.4 ATR  (`include/bolt/kernels/fintech/atr.h`;
  rolling mean of True Range — *not* Wilder's EMA like chukonu; Wilder
  shape deferred to Phase 5.E if needed)
- [x] P5.R.5 VWAP / TWAP  (`include/bolt/kernels/fintech/vwap_twap.h`;
  per-row rolling, not chukonu's batch-aggregate fill shape)
- [x] P5.R.6 RollingMin / RollingMax  (`include/bolt/kernels/fintech/rolling_min_max.h`;
  O(1) amortised via shared `monotonic_deque.h`)
- [x] P5.R.7 StochasticOsc  (`include/bolt/kernels/fintech/stochastic_osc.h`;
  %K only — %D is SMA(K,3) downstream; composes DonchianChannelState)

#### P5.E (EMA-based)
- [x] P5.E.1 EMA  (covered by P3.1 — lives in `include/bolt/kernels/fintech/ema.h`)
- [x] P5.E.2 EWMA  (`include/bolt/kernels/fintech/ewma.h`; caller-supplied alpha)
- [x] P5.E.3 EWCOV  (`include/bolt/kernels/fintech/ewcov.h`; RiskMetrics lambda-form)
- [x] P5.E.4 MACD  (`include/bolt/kernels/fintech/macd.h`; three chained EmaState;
  emits line/signal/histogram)
- [x] P5.E.5 RSI  (`include/bolt/kernels/fintech/rsi.h`; Wilder smoothing
  α = 1/period — see design-log entry "RSI Wilder vs classical EMA alpha")

#### P5.S (specialised — subset: sorted-window kernels)
- [x] P5.S.4 HistoricalVaR  (`include/bolt/kernels/fintech/historical_var.h`;
  SortedRing-backed rolling quantile; `var_index = floor((1-conf)*nw)`;
  partial-window VaR from row 0 — chukonu parity, no NaN warmup)
- [x] P5.S.5 HistoricalCVaR  (`include/bolt/kernels/fintech/historical_cvar.h`;
  mean of `sorted[0..var_index)` tail; 0.0 when `var_index == 0`)
- [x] P5.S.8 MedRV  (`include/bolt/kernels/fintech/med_rv.h`;
  batch-scalar broadcast, not rolling — chukonu's shape;
  scale = π/(6 - 4√3 + π), bias = n/(n-2), median-of-three via branchless
  sorting network)
- [x] P5.S.14 OutlierFlagMAD  (`include/bolt/kernels/fintech/outlier_flag_mad.h`;
  two SortedRings; MAD scaled by 1.4826; emits int64_t flag — matches
  chukonu's output type; scaled_mad ≤ 1e-15 → flag = 0)

#### P5.S (specialised — subset: simple POD-state kernels)
- [x] P5.S.1 MaxDrawdown  (`include/bolt/kernels/fintech/max_drawdown.h`;
  streaming peak + absolute per-row drawdown `peak - x[i]`; deviates from
  chukonu's broadcast-scalar fraction — Bolt uses per-row absolute form
  (matches P1.5 primitive); cross-batch peak persists)
- [x] P5.S.2 Drawdown  (`include/bolt/kernels/fintech/drawdown.h`;
  streaming peak + fractional per-row drawdown `(peak - x) / peak`;
  peak ≤ 1e-15 → 0.0 guard, chukonu parity)
- [x] P5.S.3 CUSUM  (`include/bolt/kernels/fintech/cusum.h`;
  two-sided Page CUSUM with symmetric threshold reset; deviates from
  chukonu's one-sided + warmup-mean variant — caller supplies target,
  no embedded statistics; emits two columns (cp, cn) pre-reset)
- [x] P5.S.13 AlmgrenChriss  (`include/bolt/kernels/fintech/almgren_chriss.h`;
  schedule precomputed at make-time into fixed array[4096 max];
  per-row emit is `schedule[row_count % n_slices]`; row_count persists
  across batches for seamless cyclic wrap)
- [x] P5.S.15 ThrottleCheck  (`include/bolt/kernels/fintech/throttle_check.h`;
  single-global sliding-window tripwire; fixed-capacity int64 timestamp
  ring; amortised O(1) eviction; unit-agnostic timestamps;
  keyed/per-symbol throttle deferred to Phase 6 when `bolt::dataflow`
  keyed state lands)

#### P5.S (specialised — subset: Kalman + rolling regression)
- [x] P5.S.6 Kalman1D  (`include/bolt/kernels/fintech/kalman1d.h`;
  seed `x_hat = obs[0]`, `P = p_init` — matches chukonu's hard-coded
  `P=1.0`; straight-line recurrence, no per-row branch; state persists
  across calls)
- [x] P5.S.7 KalmanHedgeRatio  (`include/bolt/kernels/fintech/kalman_hedge_ratio.h`;
  `beta_init = 1`, `P_init = 1` — chukonu parity; `|pb| <= 1e-15`
  fallback compiles to cmov; emits two columns (hedge ratio, Kalman spread))
- [x] P5.S.9 CornishFisherVaR  (`include/bolt/kernels/fintech/cornish_fisher_var.h`;
  scalar broadcast; POPULATION moments (`/n` — chukonu parity);
  A&S 26.2.3 normal-quantile approximation; CF expansion
  `z + (z²-1)s/6 + (z³-3z)k/24 - (2z³-5z)s²/36`; `m2 <= 1e-15` guards
  skew and kurt to 0.0; one-shot quantile branches hoisted out of loop)
- [x] P5.S.10 Amihud  (`include/bolt/kernels/fintech/amihud.h`;
  per-row `|r|/v`, not rolling mean — chukonu shape; `v > 0` cmov guard;
  `denom = v>0 ? v : 1.0` dodge avoids MSVC C4723; plan doc's
  "rolling mean" wording deviates from chukonu — we follow chukonu)
- [x] P5.S.11 KyleLambda  (`include/bolt/kernels/fintech/kyle_lambda.h`;
  rolling OLS slope of Δp on signed-volume; two RollingRings + rescan
  (design-log "Rolling-window Welford — rescan vs decrement" applies);
  warmup-split at `need=2`; `var_sv <= 1e-15` cmov guard)
- [x] P5.S.12 RollSpread  (`include/bolt/kernels/fintech/roll_spread.h`;
  POPULATION `cov(Δp_i, Δp_{i-1})` over window; two RollingRings +
  rescan; warmup-split at `need_seen=4` (combines stream-index `i<2`
  no-push AND `w<2` no-output into one up-front deficit); `cov >= 0`
  cmov clamps spread to 0; state carries `prev_price`, `prev_diff`,
  `seen` across calls)

### Phase 6 (Tier 3 — DEFERRED)
Waits for event-time + keyed state; not in this plan.

---

## Unfinished kernels — explicit status

**Landed: 67 of 74 chukonu kernels** (plus 2 chukonu kernels covered
directly by Phase-1 primitives: `CreateDiffProcessor` → `diff_column`,
`CreateLagProcessor` → `lag_column`).

The 7 remaining kernels are all **Phase 6 Tier-3**. Each is blocked
on the same two `bolt::dataflow` primitives (event-time windowing +
keyed state) that Wave 3 will ship. NOT blocked on chukonu math or
on any Bolt kernel primitive — the math is understood; the framework
hook is not there yet.

| Kernel | chukonu line | Why it's Tier-3 | What unblocks it |
|---|---:|---|---|
| **PairsSpread** | 776 | Needs per-pair keyed state (one spread state per instrument pair); batch-scoped implementation would conflate pairs. | `bolt::dataflow` keyed-state primitive keyed on `(symbol_a, symbol_b)`. |
| **TriangularArb** | 1535 | Needs **synchronised** event-time join across 3 FX legs (e.g. EURUSD, USDJPY, EURJPY); batch-local math only catches ticks that happen to co-arrive within the morsel. | Event-time synchronisation window with watermarks, plus 3-way keyed join on currency triangle. |
| **HayashiYoshidaCov** | 1711 | Non-synchronous covariance estimator for tick data; requires pairing trades by overlapping tick timestamps from two irregular series. Native event-time semantics. | Event-time windowing with per-series cursors + overlap detection. |
| **HayashiYoshidaCorr** | 1798 | Correlation form of HayashiYoshidaCov. Same framework blocker. | Same as HayashiYoshidaCov. |
| **XCorrLeadLag** | 1896 | Cross-correlation at arbitrary lags between two series; needs alignable event-time indices, not row indices. | Event-time primitive + per-series ring that indexes by timestamp, not position. |
| **LeadLag** | 3240 | Cross-series lead/lag estimation; needs event-time aligned pairs (distinct from P1.2 `lag_column` which is in-series row-lag). | Same as XCorrLeadLag. |
| **VPIN** | 3743 | Volume-synchronised Probability of Informed Trading; partitions the tick stream into equal-volume buckets, not equal-time or equal-row. Requires volume-bucket windowing, a fundamentally different morsel shape than row-count or wall-clock. | Volume-bucket window primitive in `bolt::dataflow` (new window type alongside row-count / event-time). |

When Wave 3 lands event-time + keyed state, the port template is
clear:

1. Each Tier-3 kernel becomes `<Name>State` (per-key) + `execute_<name>`.
2. Keyed state lookup replaces the single-state pointer.
3. Event-time primitives (watermark, `TimestampRing`, volume-bucket
   window) are consumed like today's `RollingRing` but indexed by a
   non-row domain.
4. Math copies chukonu verbatim exactly as Phase 4/5 did — the port
   risk is framework plumbing, not numerics.

**Other possible follow-ups** (not Tier-3, but spotted during the port
and worth tracking for a future minor wave):

- **ATR — Wilder smoothing variant.** Shipped as rolling-mean TR per
  plan; chukonu uses Wilder EMA. A second header `atr_wilder.h`
  composes `EmaState` (α=1/period) over TR.
- **VWAP/TWAP — batch-aggregate variant.** Shipped as per-row
  rolling; chukonu also has a batch-aggregate broadcast form. Trivial
  to add as `vwap_batch.h` alongside the rolling version.
- **DonchianChannel mid column.** Chukonu emits 3 columns (upper,
  lower, mid); Bolt ships 2 (upper, lower). Mid is `(upper+lower)/2`
  downstream — add iff a caller wants it without the compose step.
- **StochasticOsc %D.** Chukonu emits both %K and %D (= SMA of %K);
  Bolt ships %K only. Caller composes %D via `execute_sma` on the %K
  output. Add iff a caller wants the fused version.
- **CUSUM — chukonu parity variant.** Current `cusum.h` is two-sided
  Page with caller-supplied target; chukonu's one-sided warmup-mean
  variant could ship as `cusum_chukonu.h` if a caller needs bit-for-
  bit parity on historical runs.
- **MaxDrawdown — scalar-broadcast variant.** Current `max_drawdown.h`
  is per-row absolute (streaming-friendly, matches P1.5). Chukonu's
  scalar-max-fraction form could ship as `max_drawdown_batch.h` for
  callers that work offline.
- **ArrayKalman / MultiDimKalman.** Chukonu has Kalman1D; a 2-D or
  n-D variant with matrix state is natural but not in the original 74.

None of these are blocking; each is a single-header add when a
downstream caller surfaces demand.
