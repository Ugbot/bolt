# Reactor: register sockets once, try I/O before waiting (G2CHK-1269)

Plan item 1 of Gestalt2 `docs/research/2026-11-24-reactor-vs-venus-libuv-seastar.md` (epic G2CHK-1268).

## What changed

The kqueue and epoll backends, for ops submitted on the thread that polls the engine:

- **Register once.**
  - kqueue: one `EV_CLEAR` knote per filter, with the change folded into the next wait's changelist
    (no extra syscall).
  - epoll: one `EPOLLIN|EPOLLOUT|EPOLLRDHUP|EPOLLET` registration per fd.
  - An op only stores its pool index in the fd's read or write slot.
- **Try first.**
  - `read_async` and `recvfrom_async` try `recv`/`recvfrom(MSG_DONTWAIT)` unless the fd is known to be
    drained. A short read counts as drained (Venus `read_drained`); a datagram socket counts as drained
    only after EAGAIN.
  - `write_async` always tries `send(MSG_DONTWAIT)` first.
  - A finished try returns 1 with the result through the new `inline_result` out-param, and the callback
    is not called. With no out-param, the result is delivered from a deferred ring at the top of the next
    poll.
- **No fcntl per op.** `O_NONBLOCK` is set once per fd epoch, because callers rely on it
  (boltapi `UdpTransport` drains with raw `recvfrom`). `sendto_async` uses `MSG_DONTWAIT`.
- **fd epochs.** A process-wide per-fd epoch is bumped by every `close_async`, by accept completion,
  and by every one-shot registration. A registration is trusted only for the epoch it was made in, so a
  reused fd number registers again. `async_io::forget_fd(fd)` covers fds closed with a raw `close()`.
- **epoll has a read slot and a write slot per fd**, so a read and a write can be in flight on one fd.
  Before, the write's `EPOLL_CTL_MOD` replaced the read's interest.
- **Other cases keep the previous one-shot path:**
  - other threads' submits;
  - engines polled by several threads;
  - non-socket fds;
  - fds ≥ 65536;
  - accept and connect.
- **`IODispatcher` awaitables.** Read, write and recvfrom `await_suspend` return `bool`, and resume
  without suspending when the op finished inline. They resume inline only with `inline_resume` on.
- **Inline budget.** Each reactor turn allows `limits::kMaxInlineIoStreak` (64) inline completions. The
  next op goes through the deferred ring: a yield, so one busy socket cannot starve the reactor.
- **Not changed:** io_uring takes the new parameter and ignores it; its completion path is G2CHK-1272.
  IOCP is unchanged (its analogue, `FILE_SKIP_COMPLETION_PORT_ON_SUCCESS`, belongs with RIO in
  G2CHK-1272; untested here, no Windows host).

## Tests (test_bolt_async_io, macOS kqueue and Linux epoll)

New and red before this change:

| Test | What it checks |
|---|---|
| `ReadWithQueuedDataFinishesInline` | returns 1, callback never runs |
| `FinishedReadWithoutInlineSlotIsDeliveredByPoll` | a finished read with no inline slot is delivered by the next poll |
| `SteadyStateReadsAndWritesRegisterOnce` | ≤ 2 kernel registrations over 500 round trips; the old path made one per op |
| `ReadinessConsumedWhileIdleIsNotLost` | an edge consumed while nothing was armed does not lose the data |
| `ReusedFdNumberIsRegisteredAgain` | a reused fd number registers again |
| `FdIsNonBlockingAfterAnOp` | the fd is `O_NONBLOCK` after one op |
| `QueuedDatagramsAllArriveAfterOneEdge` | 8 queued datagrams all arrive after one edge |
| `WriteThatFitsFinishesInline` | a write that fits finishes on the submit path |
| `ReadAndWriteOnOneFdCompleteIndependently` | read and write in flight on one fd; the epoll leg was compiled out before |

`QueuedDatagramsAllArriveAfterOneEdge` caught a bug in the first cut of this change. That cut left a
datagram fd marked drained after a readiness-driven receive, which cut macOS UDP receive to 100k/s.

Suites passing:
- macOS (native), all eight:
  - test_bolt_async_io 12/12
  - test_bolt_event_loop 9/9
  - test_bolt_io_inbox 6/6
  - test_bolt_accept_retirement 12/12
  - test_bolt_coro_listener_port_clash 7/7
  - test_bolt_coro_listener_bind_host 4/4
  - test_timer_queue 7/7
  - test_bolt_worker_pool_park 25/25
- Linux (Ubuntu 24.04 container, clang 18): the first six. The Linux run of accept retirement has one
  backend-parameterised case skipped by design.

## Measurements

The benchmark set is Gestalt2 `docs/research/data/2026-11-24-reactor/bench/`. bolt `main` (`b53d005`)
and this branch ran alternately, with libuv and a hand-written loop, as medians of 3. Host load was
16–73, from other work. Read the ratios.

**macOS, M5 Max, kqueue:**

| | bolt main | this change | libuv | raw loop |
|---|---|---|---|---|
| TCP ping-pong p50 / p99 µs | 42.2 / 229 | **21.3 / 82.5** | 19.5 / 72.3 | 31.3 / 119 |
| TCP echo, 1 conn, 1 reactor, rt/s | 15.5k | **38.7k** | 44.3k | 23.2k |
| TCP echo, 16 conns, 1 reactor | 82.9k | **139.5k** | 160.9k | 148.5k |
| TCP echo, 16 conns, 2 reactors | 78.5k | **174.6k** | 120.4k | 187.3k |
| UDP sink rx, datagrams/s (rerun after the fix) | 182k | **276k** | 286k | 354k |
| Cross-thread wake, hot, p50 µs | 10.1 | 2.3 | 2.8 | — |

The second UDP ping-pong run is noise-dominated: bolt main 43.5 µs, this change 52.7 µs, libuv 28.7 µs,
with p99 1264 / 283 / 181 µs. The first run gave main 53.3 µs and this change 24.7 µs.

**Linux VM** (Docker, 4 CPUs, kernel 7.0.14-linuxkit; syscall counts exact, timings indicative):

| | main epoll | **new epoll** | main io_uring | new io_uring | libuv | raw |
|---|---|---|---|---|---|---|
| Syscalls per TCP echo round trip (strace) | 12.06 | **3.06** | 2.06 | 2.06 | 3.00 | 3.00 |
| TCP echo rt/s, 1 reactor | 124.9k | **163.6k** | 152.1k | 134.7k | 196.2k | 185.8k |
| TCP echo rt/s, 2 reactors | 100.4k | **150.5k** | 138.6k | 158.6k | 180.6k | 185.9k |
| UDP sink rx, datagrams/s | 536k | **1.50M** | 1.35M | 1.34M | 1.29M | 1.43M |

The epoll path reaches the libuv/raw syscall floor. UDP receive is 2.8× faster.

**boltapi HTTP** (`boltapi_bench_server` + one fixed `boltapi_loadgen` built on main; 32 connections,
5 s; macOS). No regression:

| Path | io threads | main req/s | **new req/s** | main p99 µs | **new p99 µs** |
|---|---|---|---|---|---|
| /plaintext | 1 | 86.5k | 87.2k | 1008 | 1098 |
| /plaintext | 2 | 81.8k | 110.6k | 1212 | 811 |
| /json | 1 | 78.6k | 106.7k | 1425 | 786 |
| /json | 2 | 91.5k | 116.5k | 1196 | 598 |
