# Reuse preallocated arena backing without growing

`Arena::set_growth_enabled(false)` is an opt-in startup policy. The default
remains enabled. It permits allocations from existing normal and oversized
blocks, including reuse after `reset()`, but returns `nullptr` before any new
backing allocation. Both normal growth and the separate oversized allocation
path check the policy. The ordinary bump fast path has no new branch.

Callers preallocate or warm their required backing, freeze it before entering
the hot path, and handle exhaustion explicitly. `max_block_size` is not a total
arena budget: it limits the growth size of individual ordinary blocks. Freezing
is the mechanism for preventing new backing once a startup budget is installed.
Allocation, reset and trim callers must be quiescent when changing policy.
Trimming can reduce the retained budget; it does not re-enable growth.

Three focused tests cover normal and oversized exhaustion without changing
allocated/reserved counts or existing contents; reuse of multiple warm blocks
without overlap; reset/compact and cold re-enablement. The existing concurrent
arena and OS-release tests also pass: three CTests, 1.04 seconds. Evidence:
`/private/tmp/g2-arena-frozen-build.log` and
`/private/tmp/g2-arena-frozen-ctest.log`.

This does not make a consumer allocation-free by itself. Every arena and other
allocator in that consumer must have its own checked budget. In particular,
Marble point-read allocation failures must propagate as errors, never misses,
and its current optimistic transaction object/arena lifecycle still allocates
per transaction. The materialization consumer tracks those prerequisites
separately before any whole-path HFT claim.
