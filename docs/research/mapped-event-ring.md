# Page-aware mapped event ring

`io/bolt_mapped_ring.h`, linked through `bolt::io_file`, implements a native
single-producer/multi-reader ring over one fixed writable mapping. No database,
tables or external transport are required. This is owner-local storage: the
current API does not provide independent-process attachment or restart recovery.

## Layout and ownership

Cold open creates a new file exclusively, physically allocates its extent,
maps it once and optionally touches each writable OS page. Existing files are
never truncated. Capacity respects discovered OS page and mapping-offset
granularity. Prefaulting does not lock residency or prevent later page faults.
The owned writable mapper also offers explicit synchronous page-range flush.

Each whole event has a 64-byte CRC-bearing header followed by payload and
64-byte span padding. Events pack across pages; an explicit padding header
handles physical-end wrap. Absolute byte position and event sequence validate
generation reuse. Two maximum spans plus one page must fit the admitted
capacity, ensuring empty-ring progress without splitting an event across the
physical end. Oversized events refuse; physical storage never grows implicitly.

One producer reserves exact bytes, fills the borrowed region, then commits or
cancels. Only commit release-publishes the complete event. Readers acquire a
committed prefix and pin their cursor's page until release. All configured
readers participate in retention from startup. Reader registration/attachment
while running is not supplied by this first owner API.

Lossless reserve returns `kFull`. Explicit overwrite first excludes every
reader from acquiring a new view; an active view refuses eviction. Idle readers
may lose their unread prefix and receive its exact sequence interval as
`kGap` before subsequent data. Repeated evictions merge unreported gaps.
Outstanding reservations and stale reservation/view tokens cannot authorize
reuse. `unpin` releases a view without consuming its event; a retry gets a new
lease. The caller quiesces roles before close, which refuses active borrows.

## Admission and persistence

Optional borrowed Bolt budgets reserve file count, full mapped disk capacity
and mapped-plus-owner memory at open. Failures roll back; close releases the
charges. Byte units and count units use separate hierarchies. A live logical
cap can change within physical capacity, on page boundaries, provided accepted
unread/pinned/reserved commitments still fit. Physical resize is cold work.

The default is ephemeral reuse. Opt-in `retain_until_flush` additionally pins
unflushed pages until explicit flush. Flush is serialized with producer work,
checks both mapping flush and file sync, and advances its durable byte prefix
only after success. Ephemeral reuse of unsynced bytes permanently prevents
certification of a contiguous durable prefix for that owner; subsequent flush
returns `kUnsupported`. There is no per-event sync or implied durable ACK.
The durable progress field alone is not a recovery protocol.

## Native batches

`io/bolt_mapped_ring_batch.h` reuses Bolt's existing native frame codec directly
inside the reservation. It does not allocate a temporary encoding buffer or
invent source EOF/control semantics. Read payloads remain borrowed under the
explicit ring lease. Column descriptors use caller-owned Arena storage; callers
must prewarm sufficient capacity and disable growth before claiming a bounded
allocation-free descriptor path. Codec/descriptor refusal preserves outputs
and unpins without consuming the event. Frame integrity failures are separate.

## Qualification boundaries

Independent tests cover literal binary/nullable/unsigned values, actual file
geometry, multi-page events, packed records, full refusal, reader fences,
cancel/stale tokens, physical wrap, accumulated overwrite gaps, corruption,
startup budget rollback/release, live logical caps, explicit flush and a
20,000-event concurrent producer/consumer payload check. The native batch
tests cover direct framing and descriptor-refusal retry.

`bench_mapped_ring` is a bounded, maintained C++ benchmark for a 64-byte
owner-local reserve/fill/commit/peek/verify/release cycle. Startup mapping,
sample allocation and prefaulting precede exported allocation-probe markers;
sorting, reporting and close follow them. An intentional allocation control
uses the same interval. This benchmark is not inter-thread, IPC, Chukonu
packet-to-Book or fsync latency. OS-managed page residency, slow-reader wrap,
cross-process lifecycle and sustained full-pipeline tails require separate
evidence. Windows/Linux code is present; local macOS execution does not prove
those platforms.
