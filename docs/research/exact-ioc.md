# Exact IOC planning and consumption

`bolt/kernels/fintech/exact_ioc.h` supplies bounded numerical execution against
a caller-owned displayed depth image. It plans a partial aggressive IOC and
separately applies its per-level takes to a consumed-liquidity overlay. It
reuses the checked Int64 arithmetic in `exact_depth.h`; it introduces no runtime
dependency, allocation, IO, clock or private book.

## Public contract

The functions are `noexcept` templates in `bolt::fintech`:

```cpp
template <ExactIocSide Side, std::uint32_t Capacity, typename Level>
ExactIocStatus exact_ioc_plan(
    const Level* levels, std::uint32_t count,
    const std::uint64_t* consumed, std::uint32_t consumed_count,
    std::uint32_t requested, std::int64_t limit_ticks,
    std::int64_t adverse_ticks, std::uint32_t* takes,
    std::uint32_t takes_count, ExactIocResult* out) noexcept;

template <std::uint32_t Capacity, typename Level>
ExactIocStatus exact_ioc_apply(
    const Level* levels, std::uint32_t count,
    std::uint64_t* consumed, std::uint32_t consumed_count,
    const std::uint32_t* takes, std::uint32_t takes_count) noexcept;
```

`Side` is compile-time `ExactIocSide::Buy` or `Sell`. `Capacity` is positive.
`Level` must be standard layout and trivially copyable, with exactly Int64
`price_ticks` and UInt32, UInt64 or Int64 `quantity`. Bolt `BookLevel` meets this
contract. Int64 quantities must be positive, as must unsigned quantities.
Prices, limits and adverse adjustments share the caller's native integer price
unit; no scaling, floating point or money conversion occurs here.

`count`, `consumed_count` and `takes_count` must be at most `Capacity`, with
both storage counts at least `count`. Each nonempty declared span must point
to valid aligned storage, have representable byte extent, and be disjoint from
every other declared span. Planning also checks the result object's extent
against all spans. These numerical checks cannot prove a pointer's lifetime or
allocation size: the caller owns that obligation. Zero-length spans may be
null. `out` is always required. The active level count may be zero, while
`requested` must be positive UInt32 and `adverse_ticks` nonnegative Int64.
Only `takes[0..count)` and, on apply, `consumed[0..count)` may be written;
declared storage beyond the active count is unchanged.

`ExactIocResult` is a 40-byte, eight-byte-aligned standard-layout, trivially
copyable object. It has UInt32 `requested`, `filled`, `remaining`,
`levels_touched`; Int64 `weighted_ticks`, `marginal_ticks`; Bool `price_limited`;
and seven reserved bytes, zero on success. `weighted_ticks` is the checked sum
of stressed price times take, without an implicit sell-side sign reversal.
`marginal_ticks` is the last filled stressed price. With no fill both are zero.
`levels_touched` counts nonzero takes.

## Planning and refusal

Planning first validates every active level: positive displayed quantity,
consumed quantity no greater than displayed, and strictly increasing ask
prices for Buy or strictly decreasing bid prices for Sell. Duplicate prices
are invalid. This full structural pass includes suffixes beyond the requested
fill or limit boundary. The image must remain immutable for the call.

Traversal skips fully consumed levels. Buy stress is `price + adverse`; Sell
stress is `price - adverse`. Buy accepts stressed prices at or below the limit;
Sell accepts prices at or above it. The first available level outside the limit
ends traversal and sets `price_limited`. Filling the complete request ends
traversal without examining more prices. Structural validation still covers
those unused levels, but their stress arithmetic is irrelevant: a far
`INT64_MAX` ask cannot invalidate an already filled prefix merely because an
unused positive adjustment would overflow. Fully consumed prices are likewise
not stressed.

Available quantity is UInt64. The kernel takes the minimum with the remaining
UInt32 request before narrowing, so a displayed quantity above `UINT32_MAX`
cannot wrap into a smaller apparent size. Stressed prices, each weighted term,
and the accumulated weighted sum use checked Int64 arithmetic. Overflow on an
actually examined price or weighted term refuses the whole plan. Negative
prices and exact `INT64_MIN`/`INT64_MAX` results are valid when representable.

| Status | Meaning and publication |
| --- | --- |
| `Ok` | Publish the entire result and active takes, including partial, empty, exhausted or price-limited outcomes. `filled + remaining == requested`. |
| `Invalid` | Bad spans, count, request, adjustment or structural depth/overlay. Leave the published result and overlay unchanged. |
| `Overflow` | Examined price or weighted arithmetic is not representable in Int64. Leave the published result and overlay unchanged. |

The UInt32 takes array is unpublished scratch. Its active entries are cleared
after span admission, and a refusal may leave cleared or partially calculated
entries there. The caller must inspect status before using either the scratch
or result. Depth and consumed inputs are never changed by planning. There is
no capacity-sized automatic scratch array: only the caller's takes storage,
one result candidate and four fixed byte ranges are used.

## Applying a successful plan

`exact_ioc_apply` validates all active quantities, current overlays and takes
before mutating any overlay entry. Each take must fit the current displayed
quantity minus consumption. A late invalid take leaves the entire overlay
unchanged; an accepted second pass adds all takes without UInt64 wrap.
Apply does not reprice, sort, validate an IOC limit or require a particular
aggregate take count. Its job is atomic availability consumption, so it needs
no side template parameter. A caller passes takes from an accepted plan against
the same depth image and coordinates exclusive overlay access across plan and
apply. Atomicity here is all-or-nothing mutation within the call, not an atomic
operation across threads.

The caller applies each accepted plan once. Bolt has no order identity or
deduplication state; repeating an apply may consume more liquidity when enough
remains. Repeated *planning* against the updated overlay sees only remaining
liquidity. The owning compute operator decides when a genuinely new committed
image allows its overlay to be reset or remapped, and owns order deduplication.
Do not reset consumption merely to retry an order.

## Evidence and remaining qualification

`tests/test_bolt_exact_ioc.cpp` contains independent literal Buy/Sell and
negative-price results; exact, partial, empty and stressed-limit outcomes;
successive plans and applies; UInt32/UInt64/Int64 sizes; native `BookLevel`;
malformed suffixes; overlapping/misaligned/undersized spans; byte-preserved
refusal outputs; traversed arithmetic overflows after a nonzero first take;
unused or unavailable overflowing far prices; and all-or-nothing apply refusal.
It also checks the unchanged arithmetic primitive's refusal behavior. These
are synthetic kernel tests; compile and execution qualification is pending
the root-owned build at this implementation handoff.

This kernel does not implement a paper engine, strategy, definition handling,
causal time, fees, account cash, risk, routing or live venue execution. Native
paper-path integration, maintained benchmark/profile results and Windows/Linux
qualification remain separate work. No throughput or latency result is claimed.
