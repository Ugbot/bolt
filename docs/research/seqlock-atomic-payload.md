# Atomic payload representation for Seqlock

`bolt_seqlock.h` protects a trivially copyable `T` with one single-writer
sequence and inline `std::atomic<uint64_t>` payload cells. The last cell copies
only its remaining bytes; unused bytes of its private staging word are zero.
All shared payload access uses those same atomic cells. `memcpy` only accesses
private caller objects and a private staging word, never shared plain payload.
No `atomic_ref` overlays arbitrary subobjects, and no large `atomic<T>`, heap,
reader lock, platform-specific path or sanitizer exception is involved.

## Ordering proof

An exclusive init stores the initial cells before the object is published.
Writers retain external single-writer ownership. `write` publishes odd sequence
by acquire/release RMW, release-stores all cells, then publishes the even epoch
by acquire/release RMW. Readers first acquire an even epoch E, acquire-load cells
into their output, then acquire-load the sequence again.

The first sequence acquire makes E's cell stores happen-before the cell loads.
Atomic coherence prevents loading values older than those E stores. If a cell
load reads a later writer W's release store, that acquire synchronizes with W;
W's preceding odd sequence modification happens-before the final sequence load.
Sequence coherence then prevents the final sequence load from returning old E.
Equal even samples therefore accept only a completed coherent epoch. A rejected
mixed candidate has no plain shared access and hence no payload data race.
This reasoning uses ordinary atomic synchronization and coherence under the C++
memory model ([atomics.order](https://eel.is/c++draft/atomics.order),
[intro.races](https://eel.is/c++draft/intro.races)). No fence-only argument is
used to legalize a concurrent plain copy.

The existing practical contract excludes a full 64-bit sequence wrap during an
attempt. Snapshot coherence is distinct from pointee/mapping lifetime: existing
WAL tailer/flusher pins and reclamation owners remain necessary and unchanged.

## Bounds, layout and limits

There are `ceil(sizeof(T)/8)` inline lock-free cells. Supported targets must
provide always-lock-free, eight-byte `atomic<uint64_t>`. The member `value`
remains isolated from `Sequence` by the configured cache isolation unit, and
retains stronger `alignof(T)` where necessary. Existing alignment assertions
remain in force. This process-local representation is not a durable format.

`try_read` makes at most 128 attempts, each bounded by the cell count. It needs
only one private eight-byte staging cell in addition to the caller's output;
there is no second full `T` temporary. Failed output is unspecified, as before.
`read()` retains its legacy spin-until-success behavior. It requires writer
progress and a stable interval; it is not bounded or universally wait-free.
The source retains release/acquire cell operations rather than making an
unmeasured paired-fence optimization.

Direct controls cover sequential init/write, tiny and odd tails, initialized
padding, stronger alignment and pointer bits, correlated values spanning many
cells with one writer/four readers, and real odd sequence publication paused
before payload mutation. That paused-owner test requires bounded refusal and
exact recovery after normal publication. It never inspects failed output.
Every accepted concurrent value is checked; continuous writing may legitimately
refuse all bounded concurrent attempts. After the actual writer completes, each
reader receives a bounded stable opportunity and must read the exact final
epoch and all correlated fields. Thus progress is required when the writer is
stable, without assuming fairness or success during its continuous updates.

Actual Linux TSAN, an old-source discriminating negative with exact restoration,
MSVC/clang-cl Windows CI, and stack/performance checks belong to execution
qualification. The representation/proof alone is not a passing gate or a
zero-regression measurement. Prior native WAL trim/subscriber reports and root
snapshot-fallback reports reached the same old plain payload race.
