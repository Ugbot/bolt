# Listener accept retirement

## Scope and evidence

The observed Linux io_uring failure was a listener port remaining bound after
plain close while an ACCEPT still held a kernel file reference. The independent
POSIX client failures were EINTR from connect/read. Root owns the actual old
source witness and all qualification. This document describes source contracts;
it does not claim that the new source has passed execution.

## Checked operation and ownership

`CoroTcpListener::try_retire_accepts(steady_clock::time_point)` accepts one
absolute monotonic deadline and returns `Complete`, `Pending`, `Unavailable`,
or `Error`. It publishes intent and observes fixed startup records. It does not
poll an engine, join a thread, take a blocking mutex, allocate, or reclaim the
record array. OS wake calls cannot provide a universal hard real-time guarantee.

`Complete` certifies that this listener's accept loops have exited, their actual
callbacks have returned, tracked cancel requests have completed, and its listen
fds have closed. It does not certify client-handler completion or dispatcher
join. Every other result retains all owner/storage obligations. The shared
dispatcher must keep progressing; independently stopping it can produce
Unavailable and does not authorize destruction.

Lifecycle callers externally serialize start, checked retirement, and stop.
Retain listener, dispatcher, worker pool and borrowed handler captures through
their documented lifetimes. Checked Complete permits another listener to bind
the port. Call legacy stop before restarting this object. The startup record
array remains allocated until that lifecycle is finished.

Legacy void `stop()` and the destructor repeat checked units while their owner
makes progress, then join owned resources. They preserve the existing cold
blocking contract and have no universal deadline. Call outside owned reactor and
worker threads, before shared progress is stopped. A permanent backend error
cannot be converted to success by a timeout; checked callers can observe it and
must retain the owner. Legacy stop cannot safely finish such a failure by
discarding live storage.

## Startup and reactor selection

Allocate at most 4096 records at startup. Record each fd and its actual reactor
before publishing a coroutine. Initially mark unpublished loops exited; set a
loop live before its queued start is admitted. Thus a queued initial coroutine
cannot be mistaken for an empty, already retired listener.

Tracked accept callbacks and registration stay on the recorded per-thread
reactor even with `inline_resume=false`. Ordinary client I/O retains its worker
resumption policy. Off-reactor tracked dispatcher registration refuses instead
of falling back to engine zero. Shared IOCP retains its normal completion/worker
model; operation counts and loop exit protect that cross-thread lifetime.

Readiness backends have an existing fixed 65536-entry fd table. Tracked accepts
outside it refuse with EOVERFLOW before registration. Listener startup performs
the same cold validation, returns -1/EOVERFLOW, and retires the unregistered fd.
An asynchronous submission refusal exits the loop, stops admission, and exposes
a nonzero `accept_error()` while keeping retirement records. It never enters an
immediate failed-await retry loop. Proof errors and submission errors are
separate facts: a refused, unregistered fd can still be closed safely.

Partial socket/post startup failure uses the same retirement path for already
published loops. Native startup errors survive cleanup. A failed bounded inbox
admission reports ENOBUFS; it does not silently drop a coroutine frame.

## Backend completion witnesses

The engine has 4096 fixed retirement admission slots. Checked admission scans
within the supplied deadline. A nonblocking poll-owner guard serializes progress
on shared IOCP and prevents two owners from retaining a record pointer past its
final Complete store. Completion removes the queue slot before publishing the
record's final acknowledgment, and never touches that record afterward.

Epoll/kqueue remove the actual accept registration, invoke its cancellation
callback on the existing reactor, and wait for callback return plus loop exit.
Removal/close errors remain explicit and retain ownership. IOCP uses CancelIoEx
and the actual original AcceptEx completion; ERROR_NOT_FOUND alone is not drain.
Failed AcceptEx completion closes its unused accepted socket. Registry close
uses a try-lock; contention leaves the record pending.

io_uring uses user_data-targeted ASYNC_CANCEL at the existing Linux 5.11 floor.
Each of 4096 ordinary slots remains live until the original CQE, actual callback
return, and every published cancel CQE are observed. Cancel user_data uses a
distinct high tag, decoded before ordinary slot indices. Only one cancel is
outstanding per target. ENOENT/EALREADY allow a later retry; neither supplies the
original completion witness. Retaining the slot prevents cancellation-token ABA.
Control SQEs do not consume a spare ordinary op slot. Full SQ space retains
intent and continues bounded polling batches.

Published tail versus kernel-consumed head determines pending submissions;
partial enter and EINTR never clear unconsumed work. A producer-serialized
64-bit consumed counter extends the kernel's wrapping head. Every SQ admission
observes head, and no more than SQ capacity can be consumed without another
producer observation. Thus the observation gap is bounded, while an old pending
accept's age is unrestricted by a 32-bit half-range test. An original's absolute
submission ticket must be consumed before cancellation. Counter exhaustion or
an impossible observation gap becomes EOVERFLOW with ownership retained.

The private enter test seam only limits one real enter or omits one enter with
EINTR. It cannot fabricate a CQE, callback, consumed head, or slot release. Normal
operation calls the real syscall. Original/cancel order predicates are shared
with direct state controls; actual kernel delivery order is not forced by those
controls.

## POSIX client fixture

The natural fanout still makes exactly 64 connections per reactor-count case,
with the same two owner replies and distribution assertions. Each connection
uses one connect and one shared exchange deadline. Nonblocking readiness plus
SO_ERROR and actual peer identity are required; EINTR never causes a replacement
client or a second connect. Exact one-byte I/O distinguishes bytes, EOF,
EAGAIN/EINTR and real errors, with at most 1024 attempts and remaining deadline.
Cleanup preserves errno. The separate queued-stop EOF observation has its own
post-stop read deadline.

The macOS queued-client stop fixture first observes its real accept-callback
return barrier and active queued handler, then retires acceptor 0 with one
absolute deadline while target reactor 1 remains paused. Only after Complete
does it stop the dispatcher, release/join target 1, and require active=0 plus
actual peer EOF/reset. Stopping dispatcher progress before retiring accepts
violates the listener ownership contract. If checked retirement fails, the
fixture closes the peer, always releases target 1, keeps dispatcher progress
alive through actual handler exit, and performs owner cleanup before its fatal
assertion. That failure cleanup retains the cold blocking lifetime obligation;
it does not relabel Pending as success or claim generic engine drain.

Fixture syscall controls inject EINTR only; success still requires real socket
readiness, peer state, and exact transferred bytes. Controls also cover a real
empty-socket deadline, EOF and EBADF. No injected success or sleeps supply proof.

## Source qualification controls

`test_bolt_accept_retirement` links bolt::reactor and GTest on all platforms.
It covers real backend pending/cancel/rebind, positive original callback
requesting retirement before return, queued startup behind a paused reactor (or
actual IOCP startup worker), a client racing retirement, shared survivor, and
preserved client worker resumption. Per-thread controls use two engines with
inline resumption disabled and inspect actual continuation owner and untouched
engine-zero accept count. Linux controls cover partial startup unwind, full
4096-op io_uring pool with a 64-entry SQ, limited/interrupted actual enter, and
real listener admission exhaustion with observable failure and cleanup.

High-fd controls use actual duplicated/open descriptors. They explicitly skip
when OS resource limits cannot admit the control. IOCP has no readiness-table
control; io_uring-specific cases explicitly report backend unavailability.
Windows still runs the backend-neutral state and native IOCP lifecycle controls.
The full high-fd startup fixture owns at most 65538 descriptor references and
does not change process resource limits.

State controls cover original-first/cancel-first/callback-last predicates,
ENOENT/EALREADY retention, partial/interrupt accounting, wrap and ancient pending
accept tickets. They do not prove that both physical CQE orders occurred in a
particular kernel run. Actual kernel tests must be qualified on unconfined
io_uring as well as epoll and the applicable macOS/Windows backends; fallback or
source inspection cannot qualify io_uring. Final compiler/runtime/sanitizer
results belong to root's frozen-source run.

## Remaining scope

Generic io_uring `close_async` remains plain close. It does not establish pending
read/write/UDP cancellation or callback/buffer drain. This change does not add a
generic engine drain, bounded dispatcher join, or active-handler cancellation.
Do not destroy an engine with borrowed generic operations based on this accept
certificate. Full Iceberg offline installation/reopen, source receipt/transport
and row completeness remain separate work.
