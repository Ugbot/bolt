# Runtime-admitted L2 and ranked MBP kernels

G2FEAT-74: reusable header-only state/update/read primitives in
`include/bolt/kernels/fintech/book_l2.h` and `book_mbp.h`. MBO, connectors,
Chukonu SQL/DataFrames, persistence/checkpoints and trading remain separate work.

## Contract and implementation steps

The accepted layout uses an array-indexed AVL tree with ordered neighbor links
for price-keyed L2 and contiguous ranked side arrays for MBP. Caller-owned
prewarmed banks and a sparse before-image journal provide atomic complete
events across arbitrary input chunks without cloning the entire book each tick.

- [x] Shared POD definitions, runtime storage sizing/admission and event lifecycle.
- [x] L2 absolute Upsert, strict Delete and side Clear, ordered lookup/read.
- [x] MBP 1-based Insert/Change/Delete and side Clear with explicit rank shifts.
- [x] Independent map/vector oracle tests, refusal/rollback and replacement tests.
- [x] Local Release Clang compile/test and independent implementation review.
- [ ] Matched old/new depth/update/read latency and allocation performance gate.

Every API returns `BookStatus` or a copied `BookReadInfo`. Unknown enum values,
invalid scales/identities, invalid payloads and budget exhaustion refuse.
Insertion at capacity never silently drops a worst level. L2 deletion of a
missing price refuses; MBP insertion accepts ranks 1..count+1, change/delete
accept 1..count, and clear requires rank zero. MBP rank is source-defined: the
kernel does not invent price-order or queue-priority rules.

`BookDefinition` requires nonzero book/instrument/definition/tick-rule identities
and positive integer scales. Values mean `price_ticks / price_scale` and
`quantity / quantity_scale`; there is no floating-point conversion or rounding.
Prices cover the entire signed int64 domain including negative endpoints;
comparisons never subtract prices. Quantity and optional order count are u64.
Order-count validity is explicitly 0/1, with absent count represented by zero.
Stored quantities must be positive; deletion uses its explicit action. Per-side
total quantity is checked against u64 overflow before mutation.

## Lifecycle, visibility and source identity

1. Call the model's `storage_size` with admitted `BookLimits`. Supply at least
   those bytes to `init`; arbitrary byte-buffer alignment is supported. Init
   warms all banks/journal/directory storage once. State and storage are disjoint,
   lexically owned and remain live until the caller retires the book.
2. `begin(cut, Delta)` or explicit caller-authorized `Replacement` starts an
   event. `stage` calls may span input chunks; nothing publishes until `commit`.
   `event_operations` charges every submitted operation, not every input chunk.
3. Deltas mutate only journaled slots. Directory entries retain the first full
   before-image of every touched slot/tree/list/free-list field. Committed reads
   use the prior roots/count/cut and those images even after erase/reuse or AVL
   rotations. Repeated updates of the same slot consume one journal record.
4. A replacement starts an empty inactive bank; commit swaps banks. Levels absent
   from the replacement disappear. An empty replacement is a valid empty baseline.
   There is no old-book copy. Snapshot construction consumes event-operation and
   level capacity; the unused delta journal is not charged.
5. Failed staging rolls back every recorded slot immediately and leaves the old
   committed metadata intact. The failure is sticky and `commit` refuses until
   `abort`. Explicit abort also restores successful but unfinished deltas.

`BookCut` contains opaque source epoch, monotonically increasing execution
generation, and ordered event ordinal (`position`). Epoch and generation must
be nonzero. A fresh book accepts an admitted replacement at ordinal zero.
Delta requires an existing valid baseline, identical epoch/generation and a
strictly increasing ordinal. Same-identity replacement also requires increasing
ordinal. Changed identities require explicit replacement with a strictly newer
execution generation; a numerically larger epoch never authorizes stale work.
Adapters increment execution generation on reset/rebaseline/epoch change. Restore
into fresh state requires caller admission. The kernel does not certify gaps,
durability, freshness, continuity, risk limits or trading authority.

`info.valid` certifies only a kernel baseline. `invalidate` refuses during a
staged event; otherwise it revokes readable validity without changing cut or
data. Delta cannot revalidate it. An admitted replacement can establish a new
kernel baseline. Revision overflow refuses before opening an event.

## Bounds, readers and layout trade-offs

All calls have one writer and **no concurrent readers**. Reads copy committed
levels into disjoint caller output storage; copies may be handed to another
thread. Mutable state/banks and their lifetimes cannot be shared concurrently.
Output overlap with book/state or the output-count location refuses. Small depth
output buffers return a bounded prefix, not a capacity error; offset is zero-based.
Price lookup is logarithmic and top is constant-time. L2 depth traversal costs
`O(offset + returned levels)`; MBP depth costs `O(returned levels)`.

Runtime depth, operation and journal budgets determine allocation and admission.
The only global ceilings are u32 slot/index/operation representability. Storage
sizing uses checked admission and u64 products, includes worst-case alignment
padding, and rejects insufficient or address-overflowing spans. There is no
growing scratch, hot allocation, network I/O or hidden quantity narrowing into
the existing uint32 depth helpers; returned levels retain u64 quantity.

AVL mutations touch `O(log depth)` nodes plus neighbors, with before-image
directory lookup O(1). Side clear can visit all admitted levels and consumes
the corresponding journal budget. MBP rank shifts naturally touch bounded
`O(depth)` slots. Full snapshot L2 construction is O(depth log depth); its empty
bank/free-list reset is O(admitted capacity). These are complexity statements,
not measured latency claims.

A sorted-array L2 would impose O(depth) insertion/deletion shifts at broad depth;
an unconditional double-buffer clone would copy O(depth) for every tick. Neither
is selected as an unmeasured general default. Root-owned benchmarks must compare
shallow/deep mutation and read workloads before production performance claims.
Tests in `tests/test_bolt_book.cpp` use independent std::map/vector oracles only
in the test harness; production headers contain no heap-backed containers.

The 2026-10-14 synthetic Chukonu Source-to-book runner exercised these L2 kernels
at depth8 over 48/1,000/10,000 books, observing approximately10.79M/7.97M/7.62M
normalized messages/s respectively. These short runs include generation/column
copying and do not qualify isolated kernel tails, durability or a comparator
gate. Devtools DTrace run `f2483cba-b517-4f17-b1b7-f42c4a31e211` attributes about
55% of519 samples to the book-consumer path. Dedicated book-oracle CTest passed;
MBP performance and broad depth/read workloads remain unmeasured.
