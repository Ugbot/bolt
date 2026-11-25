#pragma once

#include <cstddef>
#include <cstdint>

namespace bolt::api::net::limits {

// Kind B: compile-time allocation bounds for loop-owned storage. Adjusting
// these requires rebuilding the loop and its consumers together. No queue
// grows during poll/dispatch: excess readiness stays in the kernel, and a
// nested dispatch beyond the pin stack is explicitly refused.
inline constexpr uint32_t kMaxDispatchDepth = 16;
inline constexpr size_t kEventBatchCapacity = 256;
// Persistent kqueue registrations queued for the next wait's changelist before
// a forced flush (G2CHK-1269).
inline constexpr size_t kReadinessChangeBatch = 64;
// I/O ops one reactor turn may complete inline (try-first succeeded) before the
// next one is deferred to the following poll, so a busy socket cannot starve
// the rest of the reactor (G2CHK-1269).
inline constexpr uint32_t kMaxInlineIoStreak = 64;
// Timers one reactor holds at once (IODispatcher::sleep_for / sleep_until);
// past it a sleep is refused (resumes at once, returns false) (G2CHK-1273).
inline constexpr uint32_t kReactorTimersMax = 16384;
// Write corking (IODispatcher::write_corked): fds with queued writes per
// reactor, and writes queued per fd. A full queue flushes at once; still full,
// the write is refused with -1 (G2CHK-1273).
inline constexpr uint32_t kCorkedFds = 64;
inline constexpr uint32_t kCorkedWritesPerFd = 32;
// kqueue waits up to this long also arm a one-shot non-coalesced EVFILT_TIMER
// (XNU coalesces plain kevent timeouts by up to ~1 ms) (G2CHK-1273).
inline constexpr uint32_t kPreciseWaitMaxUs = 100000;
// Due timers one reactor turn fires before it polls again.
inline constexpr uint32_t kReactorTimersPerTurn = 256;

static_assert(kMaxDispatchDepth > 0);
static_assert(kEventBatchCapacity > 0 && kEventBatchCapacity <= INT32_MAX);
static_assert(kReadinessChangeBatch > 0 && kReadinessChangeBatch <= 1024);
static_assert(kMaxInlineIoStreak > 0);
static_assert(kReactorTimersMax > 0 && kReactorTimersPerTurn > 0);

}  // namespace bolt::api::net::limits
