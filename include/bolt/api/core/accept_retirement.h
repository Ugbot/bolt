#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cerrno>

namespace bolt::api::core {
enum class AcceptRetirementStatus : uint8_t { Complete, Pending, Unavailable, Error };
using AcceptRetirementDeadline = std::chrono::steady_clock::time_point;
// Startup-owned, immovable record. Its listener and engine outlive every
// non-Complete result. Lifecycle calls are externally serialized.
struct AcceptRetirement {
    static constexpr uint32_t kCapacity = 4096u;
    static constexpr int kReadinessFdLimit = 1 << 16;
    int fd = -1;
    uint32_t reactor = 0u; // immutable actual registration/progress owner
    std::atomic<bool> requested{false};
    std::atomic<bool> loop_exited{false};
    std::atomic<bool> admitted{false};
    std::atomic<int> admission_error{0};
    std::atomic<bool> complete{false};
    std::atomic<uint32_t> operations{0u};
    std::atomic<int> error{0};
};

namespace detail {
// Actual io_uring slot lifetime, separate from kernel ABI/storage. The poll
// owner alone changes these facts; a retained slot is the cancellation token's
// ABA protection. Tests exercise both CQE orders using these same predicates.
struct AcceptCompletionWitness {
    bool original_seen = false;
    bool callback_done = false;
    bool cancel_inflight = false;
    bool cancel_retry = true;
    bool recyclable() const noexcept {
        return original_seen && callback_done && !cancel_inflight;
    }
    int cancel_completed(int result) noexcept {
        cancel_inflight = false;
        cancel_retry = result == -ENOENT || result == -EALREADY;
        return result < 0 && !cancel_retry ? -result : 0;
    }
};
// Unsigned counters wrap naturally. At most one bounded ring's entries can be
// outstanding; enter's return value is not a consumption/accounting witness.
constexpr unsigned pending_submissions(unsigned published, unsigned consumed) noexcept {
    return published - consumed;
}
struct SubmissionProgress {
    uint32_t last_head = 0u;
    uint64_t consumed = 0u;
    // Every producer observes before publishing, so unseen kernel progress is
    // bounded by ring capacity even if an ACCEPT lives for billions of submits.
    bool observe(uint32_t head, uint32_t capacity) noexcept {
        const uint32_t delta = head - last_head;
        if (delta > capacity || UINT64_MAX - consumed < delta) return false;
        consumed += delta; last_head = head; return true;
    }
};
constexpr bool original_consumed(uint64_t consumed, uint64_t original_ticket) noexcept {
    return consumed >= original_ticket;
}
} // namespace detail
} // namespace bolt::api::core
