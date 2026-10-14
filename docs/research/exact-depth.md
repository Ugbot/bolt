# Exact displayed-depth arithmetic

`include/bolt/kernels/fintech/exact_depth.h` provides portable checked signed
64-bit addition/subtraction, multiplication by an unsigned 32-bit quantity,
bounded depth consumption, and decimal formatting of a price quoted at scale
100. These functions allocate no memory and return explicit failure on overflow.

`exact_depth<Capacity>` accepts caller-owned price-ordered POD levels exposing
`price_ticks` and `quantity`. The caller validates ordering and side semantics.
The result includes known capacity, weighted price sum, marginal price and
consumed quantity. Insufficient depth never returns a partial weighted price.
Negative prices are supported. This primitive does not infer deeper liquidity,
combine resources across alternative routes, or predict execution.

`exact_price_text` preserves at least two and at most eight decimal places.
If a rational requires more places, the output truncates toward zero and the
exactness output is false. Consumers must preserve the original rational.

`test_bolt_exact_depth` is independently buildable without Chukonu or a test
framework. It covers the worked 40-unit depth sum, negative prices, insufficient
depth, integer extrema and decimal exactness.

`exact_depth_imbalance<Capacity>` computes a strict rational threshold direction
from bid and ask displayed quantities. It reuses bounded depth capacity, then
checks signed conversion, total, difference, and both rational products. Output
is `-1`, `0`, or `1`; exact threshold equality is neutral. Every failure clears
output to zero: invalid ratio/count returns `Invalid`, a zero-capacity side
returns `Insufficient`, and an unrepresentable product returns `Overflow`.
Pointers must be non-null caller-owned arrays; level quantities are unsigned
32-bit. Ordering, side identity and book eligibility remain caller concerns.
Signed extreme prices are accepted because capacity consumption requests only
one unit. No price, instrument, inventory, fee or execution policy is inferred.

Tests cover positive/negative equality and just-over thresholds, full unsigned
quantities, signed extreme prices, invalid ratios/counts, empty/zero-quantity
sides, and both threshold/difference product overflows with output reset.
The Chukonu backtest caller propagates refusal before publishing staged account,
orders or point outputs; existing admission bounds make overflow unreachable.

The optional `test_bolt_exact_depth --bench` compares the prior validated scalar
sum/ratio calculation with the checked primitive on identical ten-level books.
Five alternating-order runs each measure 10,000 batches of 64 evaluations;
setup, scalar oracle checks and sorting are outside timing. Timed code has no
heap calls by inspection; an allocation interceptor is not installed. Both
paths produced checksum 337440 and batch p50 167 ns / p99 250 ns in every run
on the current macOS Clang Release build (2026-10-14). Timer overhead is included.
Raw evidence: [helper.txt](data/exact-imbalance-20261014/helper.txt). Its original
`allocations=0` label was based on source inspection, not an interceptor; the
maintained benchmark now says `heap_calls=0_by_inspection` explicitly.

Five resident synthetic delayed-settlement backtest runs before/after the
caller change each verified their existing oracle over 100,000 samples. Before
p50 was 42 ns in all runs and p99 was 83, 83, 83, 125, 83 ns; after p50 was
42 ns in all runs and p99 was 84 ns in all runs. Allocation count remains
unmeasured (`null`) in that benchmark. Raw evidence is
retained in Chukonu's `docs/research/data/cme-backtest-20261014/` directory as
`before.jsonl` and `after.jsonl`. Coarse clock resolution and
process scheduling limit these observations; they do not qualify live feeds,
durability, queue latency, or an HFT system, or establish general performance
non-regression.
