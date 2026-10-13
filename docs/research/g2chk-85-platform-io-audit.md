# G2CHK-85 platform I/O audit

Date: 2026-09-21

## Scope

This pass reviewed the three deferred platform backends from the owned-primitives
audit:

- `src/api/net/event_loop_iocp.cpp`
- `src/api/core/async_io_iocp.cpp`
- `src/api/core/async_io_uring.cpp`

The review distinguishes registries that can use the existing single-threaded
`FdRegistry` from synchronization that protects a larger cross-thread protocol.

## IOCP event-loop handler registry

`IOCPEventLoop::handlers_` remains an `std::unordered_map`. A direct
`FdRegistry<IOCPHandlerData>` substitution is unsafe under the current dispatch
contract: a handler can call `remove_fd()` for its own descriptor while its
`std::function` callback is executing. `FdRegistry::erase()` immediately resets
the payload and would therefore destroy the active callable from inside itself.
The existing map has the same lifetime hazard, but changing the container
without defining a dispatch pin would preserve or sharpen it rather than solve
it.

This pass deliberately leaves `event_loop_iocp.cpp` unchanged. A portable
`EventHandlerRegistry` now provides bounded callable pins for the epoll and
kqueue readiness loops: dispatch swaps the active `std::function` into one of
16 preconstructed pin slots, and restores it only when the fd and monotonic
registration token still match. Self-removal, replacement, and remove/re-add
with immediate slot reuse therefore cannot destroy or resurrect the executing
callable. Recursive `poll()` is refused before either backend overwrites its
shared event batch. Each backend now drains readiness in fixed batches of 256
events and snapshots every event's registration token before invoking the
first callback. A callback that removes and re-adds a different ready fd can
therefore make the old dequeued entry stale, but cannot redirect that entry to
the replacement handler. Missing and stale entries are skipped; recursive or
over-depth dispatch remains an explicit error. Readiness beyond the fixed
batch remains queued in the kernel for a later poll.

Applying that lifetime fix to IOCP is still insufficient. Each IOCP handler has
an outstanding overlapped `WSARecv`. If a callback removes and re-adds the same
open socket, a new-token read can be posted while the old read remains owned by
the kernel. The old read can consume bytes and then be discarded as stale,
silently losing input. Disabling READ has the same ownership issue: it cannot
withdraw a read already submitted. A correct IOCP migration therefore needs
cancel-and-drain semantics (for example, `CancelIoEx` plus a tombstone retained
until the cancellation completion) or an interface where removal owns socket
close. Token checks alone solve stale callback lifetime, not kernel-operation
ownership, so the attempted IOCP wiring was rejected and the file restored.

## IOCP async association registry

`iocp_io::impl::associated` is different: async submissions may arrive from
multiple threads, so the surrounding mutex is part of its contract.
`FdRegistry` is intentionally not thread-safe. The narrow internal
`AssociatedHandleRegistry` transaction helper therefore requires external
serialization; `async_io_iocp.cpp` retains `assoc_mutex` across the complete
check, bounded reservation, one-time `CreateIoCompletionPort` call, and
rollback. Keeping the mutex across the cold first-association syscall also
closes the old check/unlock/associate race in which two first submissions could
both attempt association.

The six submission entry points now propagate association failure before
allocating an operation. Previously they ignored the return value and could
submit overlapped I/O whose completion was not routed to this IOCP.

Close is part of the same association transaction. `close_async()` now holds
`assoc_mutex` across `closesocket()` and removes the cached association only
after a successful close. A failed close retains the entry because the OS still
owns that socket; the WinSock error is captured immediately and restored after
unlocking. This also prevents a successful close from racing a new submission
that observes the old association state while Windows reuses the numeric socket
value.

The helper keys are `uint64_t`, and the portable tests use keys that differ only
above bit 32. This adds no narrowing between `SOCKET` and the registry on Win64.
There is, however, a pre-existing public-interface limitation: Bolt's
`async_io` and `EventLoop` APIs expose descriptors as `int`, while Win64 defines
`SOCKET` as pointer-sized `UINT_PTR`. The cast to `int` can truncate before
either IOCP implementation reaches this registry. Correcting that ABI across
all backends and callers is outside this registry migration.

The operation free list remains a separate issue. It uses a mutex-protected
`std::vector<iocp_op*>` and falls back to `new` when empty. Replacing it safely
requires Windows execution coverage for concurrent submit/completion and pool
exhaustion; it is not a registry substitution and was not changed here.

## FdRegistry initialization failure

`FdRegistry` previously asserted that its initial 64-slot Swiss table allocation
could not fail and then unconditionally dereferenced the table in every method.
That made ordinary arena allocation refusal a debug abort or invalid access.
It now accepts an optional `ArenaConfig` and treats initial allocation failure
as an empty, permanently refusing registry: `find()` and `insert()` return
`nullptr`, `erase()` returns `false`, and `size()` returns zero. The default
configuration and initialized hot path are unchanged.

The portable test drives this path without a global failure hook. It supplies a
valid 64-byte alignment with a `SIZE_MAX` initial block request, which Bolt's
aligned allocator deterministically rejects through its overflow guard before
calling the platform allocator.

## io_uring submit mutex

`submit_mu` is on every submit, but it does not merely protect a container. One
critical section atomically coordinates three pieces of state:

1. allocation or release in the fixed operation-slot free stack;
2. construction and release-publication of one SQE plus the shared SQ tail;
3. transfer of `pending_submits` to the owner thread before `io_uring_enter`.

The normal server path submits on the ring owner thread, where the mutex is
uncontended. Cross-thread submission is permitted for pre-pinning client work,
so deleting the lock or swapping only the free stack for `NodePool` would leave
the SQ tail and batching counter racy.

Bolt's current `MPSCChannel` is not a drop-in replacement: once a producer
claims a position it spins until capacity becomes available and cannot report a
full queue. A correct redesign needs a bounded, fail-fast multi-producer command
ring whose single consumer owns the op pool and SQ, plus wake-up and shutdown
semantics and tests for queue saturation, cross-thread connect, callback
resubmission, SQ-full rollback, and stop races. No such partial rewrite is made
in this pass.

## Verification and platform limit

Reproducible checks for this increment:

```sh
# Linux/Clang container, at most three compiler jobs.
cmake -S /src -B /build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
  -DBOLT_BUILD_TESTS=ON \
  -DBOLT_BUILD_BENCHMARKS=OFF
cmake --build /build --target test_bolt_swiss_growable test_bolt_event_loop \
  test_associated_handle_registry test_event_handler_registry -j3
ctest --test-dir /build --output-on-failure \
  -R 'test_bolt_(swiss_growable|event_loop)|test_associated_handle_registry|test_event_handler_registry'

# Source audit: async IOCP no longer owns an unordered association map;
# event-loop IOCP intentionally remains deferred as described above.
rg -n 'std::unordered_map|#include <unordered_map>' \
  src/api/core/async_io_iocp.cpp
git diff --check
```

The Linux tests exercise `SwissTableGrowable`, `FdRegistry` churn and capacity,
full-width association keys, one-time association, platform-failure rollback,
close-failure retention, close-success removal, deterministic initial-registry
allocation refusal, bounded dispatch depth, busy dispatch, self-removal,
same-fd remove/re-add ABA, stale readiness after replacement, fixed-batch
draining beyond 256 ready fds, slot reuse, nested-poll refusal, stable handler
addresses, dispatch, modify, and remove behavior. They do not compile the
`_WIN32` translation units or prove all six Windows call sites execute the
propagation branch. An MSVC or clang-cl Windows build and the Bolt test suite
remain required before claiming the async IOCP changes are platform validated.

Observed on 2026-09-21 in `chukonu-clang:19` (Linux ARM64, Clang 19.1.7/lld),
using the distinct `/private/tmp/g2chk85-bolt-linux-20260921-1` build directory
and `-j3`: the association/registry executable passed all 5 cases, and all
three targeted CTest entries passed with 0 failures in 0.11 seconds total. The
target rebuilt without compiler warnings. This is correctness evidence only;
no timing benchmark was run.

The readiness-loop increment was rebuilt separately in
`/private/tmp/g2chk85-event-pin-linux-20260921-1` with the same Linux ARM64
Clang 19/lld image and `-j3`. `test_event_handler_registry` passed 5/5 cases,
`test_bolt_event_loop` passed 9/9 cases, and their targeted CTest run passed
2/2 entries with 0 failures in 0.07 seconds. The build emitted existing
diagnostics in unrelated Bolt and GoogleTest sources; the touched epoll source
compiled without a diagnostic. No timing benchmark was run.

Parent verification after the final review corrections also passed the complete
five-case association/registry executable under macOS ASan + UBSan (CTest 1/1,
0.83 seconds). This verifies the portable transaction and allocation-failure
logic; it does not substitute for the Windows gate above.

Final parent macOS ASan + UBSan verification passed all 14 readiness-lifetime
cases (5 registry and 9 real-backend cases), with no skips, and both targeted
CTest entries passed in 0.65 seconds. The four new real-backend regressions had
also been run against the archived pre-change kqueue implementation:
self-removal and self-replacement destroyed active callables, recursive poll
was accepted, and stale readiness reached a replacement handler. The separate
300-fd case passes the current bounded batching contract. Logs are retained at
`/tmp/g2-callback-verified-asan-{build,test}.log`; the before-change evidence
is at `/tmp/g2-callback-lifetime-baseline/`.
