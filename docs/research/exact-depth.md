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
