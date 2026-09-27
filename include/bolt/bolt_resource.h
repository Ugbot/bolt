// bolt_resource.h — ResourceExhausted and BOLT_BOUND_CHECK.
//
// ResourceExhausted{knob, requested, limit} is the one shape every repo's
// status type carries when a budget or a bound refuses work. `knob` is the
// Limits-registry name that raises it, so the message can say how to fix it.
//
// The failing site records the detail in a per-thread slot (set_resource_
// exhausted) and returns its own status code; the layer that talks to the
// client reads the slot to build the message. The slot is overwritten by the
// next failure on the same thread, so read it immediately.
//
// BOLT_BOUND_CHECK is a release-enforced check for caller-supplied counts.
// Unlike assert() it never compiles out: a bound that only an assert guards is
// out-of-bounds memory in a release build.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "bolt/bolt_limits.h"
#include "bolt/bolt_port.h"

namespace bolt {

struct ResourceExhausted {
    const char* knob      = nullptr;   // registry name, never nullptr when set
    uint64_t    requested = 0;         // what was asked for (bytes for memory budgets)
    uint64_t    limit     = 0;         // the bound in force
    uint64_t    in_use    = 0;         // already consumed when refused
};

namespace detail {
inline ResourceExhausted& resource_exhausted_slot() noexcept {
    thread_local ResourceExhausted r{};
    return r;
}
}  // namespace detail

inline void set_resource_exhausted(const char* knob, uint64_t requested,
                                   uint64_t limit, uint64_t in_use = 0) noexcept {
    assert(knob != nullptr);
    ResourceExhausted& r = detail::resource_exhausted_slot();
    r.knob = knob;
    r.requested = requested;
    r.limit = limit;
    r.in_use = in_use;
}

inline void clear_resource_exhausted() noexcept {
    detail::resource_exhausted_slot() = ResourceExhausted{};
}

// The last refusal on this thread; knob == nullptr when none.
inline ResourceExhausted last_resource_exhausted() noexcept {
    return detail::resource_exhausted_slot();
}

// "resource_exhausted: <knob> requested N (in use M) > limit L; raise <ENV>".
// Always NUL-terminates; returns the length written (<= cap - 1).
inline size_t format_resource_exhausted(const ResourceExhausted& r, char* buf,
                                        size_t cap) noexcept {
    assert(buf != nullptr && cap > 0);
    const char* knob = r.knob != nullptr ? r.knob : "unknown";
    const char* env = limits_env_for(r.knob);
    int n = 0;
    if (env != nullptr) {
        n = std::snprintf(buf, cap,
                          "resource_exhausted: %s requested %llu (in use %llu) "
                          "exceeds limit %llu; raise it with %s",
                          knob, static_cast<unsigned long long>(r.requested),
                          static_cast<unsigned long long>(r.in_use),
                          static_cast<unsigned long long>(r.limit), env);
    } else {
        n = std::snprintf(buf, cap,
                          "resource_exhausted: %s requested %llu (in use %llu) "
                          "exceeds limit %llu",
                          knob, static_cast<unsigned long long>(r.requested),
                          static_cast<unsigned long long>(r.in_use),
                          static_cast<unsigned long long>(r.limit));
    }
    if (n < 0) { buf[0] = '\0'; return 0; }
    return static_cast<size_t>(n) < cap ? static_cast<size_t>(n) : cap - 1;
}

}  // namespace bolt

// Release-enforced bound: `return <ret>` when cond is false.
#define BOLT_BOUND_CHECK(cond, ret)            \
    do {                                       \
        if (BOLT_UNLIKELY(!(cond))) return ret; \
    } while (0)

// Release-enforced capacity check that also records ResourceExhausted:
// refuses (returns ret) when `requested > limit`.
#define BOLT_BOUND_CHECK_CAP(requested, limit, knob, ret)                     \
    do {                                                                      \
        const uint64_t bolt_bc_req_ = static_cast<uint64_t>(requested);       \
        const uint64_t bolt_bc_lim_ = static_cast<uint64_t>(limit);           \
        if (BOLT_UNLIKELY(bolt_bc_req_ > bolt_bc_lim_)) {                     \
            ::bolt::set_resource_exhausted(knob, bolt_bc_req_, bolt_bc_lim_); \
            return ret;                                                       \
        }                                                                     \
    } while (0)
