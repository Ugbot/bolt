// bolt_park_limits.h — the park layer's Limits rows (G2CHK-1293).
//
// One row set per wait class: mode, spin budget, deadline. A class's mode
// defaults to the process profile (park_profile); setting the class row
// overrides it. Every row is kind C (a running process can change it) and
// reported with the rest of bolt's Limits.
//
//   park_profile        0 efficient (park everywhere), 1 balanced (spin then
//                       park), 2 latency (busy reactors, flushers, ACK waits)
//   park_<class>_mode   0 busy, 1 spin then park, 2 park, 3 from park_profile
//   park_<class>_spin_ns          spin budget of spin-then-park
//   park_<class>_deadline_ms      longest single park before the waiter
//                                 re-checks (0 = class has no deadline)
//   park_reactor_poll_every_ns    zero-timeout poll interval while a reactor spins
//   park_max_spinners  process-wide cap on threads in a spin phase (0 = cores/4)
#pragma once

#include "bolt/bolt_limits.h"

#include <cstdint>

#define BOLT_PARK_LIMITS(X)                                                     \
    X(park_profile, kBudget, "profile", 1, 0, 2, "BOLT_PARK_PROFILE", nullptr, \
      "process wait profile: 0 efficient, 1 balanced, 2 latency")              \
    X(park_reactor_mode, kBudget, "mode", 3, 0, 3, "BOLT_PARK_REACTOR_MODE",    \
      nullptr, "reactor wait: 0 busy, 1 spin then park, 2 park, 3 profile")    \
    X(park_reactor_spin_ns, kBudget, "ns", 20000, 0, 1000000000,               \
      "BOLT_PARK_REACTOR_SPIN_NS", nullptr, "reactor spin before it parks")    \
    X(park_reactor_deadline_ms, kBudget, "ms", 1000, 1, 3600000,               \
      "BOLT_PARK_REACTOR_DEADLINE_MS", nullptr,                                \
      "longest reactor park with no timer due")                                \
    X(park_reactor_poll_every_ns, kBudget, "ns", 5000, 0, 1000000000,          \
      "BOLT_PARK_REACTOR_POLL_EVERY_NS", nullptr,                              \
      "zero-timeout I/O poll interval while a reactor spins")                  \
    X(park_ack_mode, kBudget, "mode", 3, 0, 3, "BOLT_PARK_ACK_MODE", nullptr,  \
      "durability / quorum ACK wait: 0 busy, 1 spin, 2 park, 3 profile")       \
    X(park_ack_spin_ns, kBudget, "ns", 10000, 0, 1000000000,                   \
      "BOLT_PARK_ACK_SPIN_NS", nullptr, "ACK waiter spin before it parks")     \
    X(park_ack_deadline_ms, kBudget, "ms", 30000, 0, 3600000,                  \
      "BOLT_PARK_ACK_DEADLINE_MS", nullptr, "ACK wait deadline")               \
    X(park_pool_mode, kBudget, "mode", 3, 0, 3, "BOLT_PARK_POOL_MODE", nullptr, \
      "idle pool worker: 0 busy, 1 spin, 2 park, 3 profile")                   \
    X(park_pool_spin_ns, kBudget, "ns", 5000, 0, 1000000000,                   \
      "BOLT_PARK_POOL_SPIN_NS", nullptr, "idle worker spin before it parks")   \
    X(park_pool_deadline_ms, kBudget, "ms", 30000, 0, 3600000,                 \
      "BOLT_PARK_POOL_DEADLINE_MS", nullptr, "pool wait deadline")             \
    X(park_flusher_mode, kBudget, "mode", 3, 0, 3, "BOLT_PARK_FLUSHER_MODE",    \
      nullptr, "WAL flusher / committer: 0 busy, 1 spin, 2 park, 3 profile")   \
    X(park_flusher_spin_ns, kBudget, "ns", 10000, 0, 1000000000,               \
      "BOLT_PARK_FLUSHER_SPIN_NS", nullptr, "flusher spin before it parks")    \
    X(park_flusher_deadline_ms, kBudget, "ms", 30000, 0, 3600000,              \
      "BOLT_PARK_FLUSHER_DEADLINE_MS", nullptr, "flusher wait deadline")       \
    X(park_background_mode, kBudget, "mode", 2, 0, 3,                          \
      "BOLT_PARK_BACKGROUND_MODE", nullptr,                                    \
      "background work (flush, compaction, tiering): 2 park by default")       \
    X(park_background_spin_ns, kBudget, "ns", 0, 0, 1000000000,                \
      "BOLT_PARK_BACKGROUND_SPIN_NS", nullptr, "background spin")              \
    X(park_background_deadline_ms, kBudget, "ms", 0, 0, 3600000,               \
      "BOLT_PARK_BACKGROUND_DEADLINE_MS", nullptr,                             \
      "background wait deadline (0 = stuck report only)")                     \
    X(park_max_spinners, kBudget, "threads", 0, 0, 1u << 16,                   \
      "BOLT_PARK_MAX_SPINNERS", nullptr,                                       \
      "threads in a spin phase at once, process-wide (0 = cores / 4)")        \
    X(park_slots, kFixedAtAlloc, "threads", 4096, 64, 1u << 16,                \
      "BOLT_PARK_SLOTS", nullptr,                                              \
      "threads that can hold a park slot (lease registry size)")              \
    X(park_stuck_report_ms, kBudget, "ms", 10000, 1, 86400000,                 \
      "BOLT_PARK_STUCK_REPORT_MS", nullptr,                                    \
      "a lease older than this is reported once by the stuck-waiter report")

namespace bolt {
namespace park {
// Kind B: waiters one Gate holds at once (a full gate refuses the next waiter
// with kExhausted naming gate_waiters). Gates may pick a smaller capacity.
inline constexpr uint32_t kGateWaitersDefault = 256;
static_assert(kGateWaitersDefault > 0 && kGateWaitersDefault <= 65536);
}  // namespace park

BOLT_LIMITS_TABLE(bolt_park_limits, "bolt.park", BOLT_PARK_LIMITS)
}  // namespace bolt
