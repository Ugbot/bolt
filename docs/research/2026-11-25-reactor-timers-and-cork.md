# Reactor timers and per-turn write corking (G2CHK-1273)

Plan item 5 of Gestalt2 `docs/research/2026-11-24-reactor-vs-venus-libuv-seastar.md`.

## Timers

- **`IODispatcher::sleep_for(ns)` / `sleep_until(deadline_ns)`** park a coroutine on its reactor's timer
  heap. The heap is 4-ary, owner-thread only, and holds `limits::kReactorTimersMax` (16,384) timers per
  reactor; a sleep past that is refused (`false`). A turn fires at most `kReactorTimersPerTurn` (256).
- **Wait timeout.** A reactor waits until the earliest deadline (rounded up to the next µs) or its class
  deadline. It never ticks: `ReactorWaitingOnATimerParks` sees fewer than 40 polls over a 300 ms sleep.
- **Slack.** Timers fire early by at most `park_reactor_timer_slack_ns` (Limits C, default 0) and never
  late. `SleepIsNeverEarlyAndBoundedLate` asserts lateness ≥ 0 over 500 sleeps.
- **Precision.**
  - kqueue: a wait of up to 100 ms also arms a one-shot `EVFILT_TIMER` with `NOTE_CRITICAL`. XNU
    coalesces a plain kevent timeout by up to ~1 ms, and the timer opts out of that.
  - epoll: waits use `epoll_pwait2` (µs) where the kernel has it. Elsewhere a sub-millisecond timeout
    rounds UP to 1 ms; it used to truncate to a busy 0.
  - io_uring: the existing ns `EXT_ARG`.
- **Before this change** the reactor had no timer at all (`core::timer_queue` was unused). Callers slept
  on the 1 ms poll tick.

Timer lateness (fire − deadline), p50 / p99 µs, 300 sleeps × 3 repetitions:

| | delay | bolt main (`timer_queue` + poll) | **IODispatcher sleep** | libuv (ms timers) |
|---|---|---|---|---|
| macOS | 250 µs | 66 / 93 | **10.4 / 39** | 959 / 1036 |
| macOS | 1 ms | 210 / 277 | **17.8 / 102** | 212 / 278 |
| macOS | 5 ms | 1050 / 1269 | **19.0 / 441** | 1043 / 1287 |
| Linux VM | 250 µs | 124 / 171 | 121 / 148 | 918 / 1842 |
| Linux VM | 1 ms | 124 / 447 | 161 / 485 | 224 / 1107 |
| Linux VM | 5 ms | 1652 / 2489 | 1730 / 2391 | 1767 / 2524 |

On macOS, `NOTE_CRITICAL` cuts lateness 10–50×. In the Docker VM every timer, libuv's included, is
bounded by the VM's own timer resolution. A bare-metal Linux run is pending with the quiet-host rerun
(G2CHK-1277).

## Write corking: one flush per turn

`IODispatcher::write_corked(fd, buf, len)` queues a write into the reactor's cork table. Each turn ends,
before the reactor waits, with one `sendmsg` (scatter/gather) per fd for everything that turn corked.

- A writer resumes only when all of its bytes are on the wire (result = len). After a short write or
  EAGAIN, the remainder waits on a new `async_io::writable_async` (kqueue `EVFILT_WRITE`, epoll
  register-once write slot, io_uring `POLL_ADD`) and goes out on the next turn, in order. The stream is
  never reordered.
- `async_close` fails the fd's queued writes with −1 before it closes the fd, so a reused fd number
  never receives another socket's bytes.
- Bounds (Limits B): 64 fds and 32 writes per fd per reactor. A full queue flushes at once; a queue
  still full after that refuses the write with −1.
- Tests (`test_bolt_write_cork`):
  - 20 writes corked in one turn arrive in order;
  - 8 × 64 KiB through a 4 KiB send buffer arrive in order, each writer resuming only when its bytes
    are out;
  - a corked write off an IO thread is refused.
- Adoption is a separate change: boltapi writes an HTTP response's header and body as two corked writes,
  pipelined responses share one flush, and the transport layer (B0) flushes per turn. That change
  carries the HTTP numbers.
