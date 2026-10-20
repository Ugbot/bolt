// IODispatcher::post_to_io_thread under a full inbox (G2ETL-54).
//
// A full inbox must never hang the poster: a foreign thread waits for the IO
// thread to drain, the IO thread posting to itself is refused (it is the only
// consumer), and a post to a dispatcher that is not running is refused.

#include <bolt/api/net/io_dispatcher.h>

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <coroutine>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

using bolt::api::net::IODispatcher;
using bolt::api::net::IODispatcherConfig;

constexpr size_t kInboxCapacity = 256;

// Fire-and-forget coroutine: suspended at start, destroys itself at the end.
struct Detached {
    struct promise_type {
        Detached get_return_object() {
            return Detached{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() noexcept { std::abort(); }
    };
    std::coroutine_handle<promise_type> h;
};

Detached bump(std::atomic<size_t>* n) {
    n->fetch_add(1, std::memory_order_relaxed);
    co_return;
}

// Runs fn on a thread and exits the process if it does not return in time:
// the pre-fix failure mode is a poster spinning forever.
template <typename Fn>
void run_bounded(Fn fn, const char* what) {
    std::atomic<bool> done{false};
    std::thread t([&] { fn(); done.store(true, std::memory_order_release); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!done.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() > deadline) {
            std::fprintf(stderr, "HANG: %s did not return within 10s\n", what);
            std::fflush(stderr);
            std::_Exit(1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    t.join();
}

IODispatcherConfig one_thread() {
    IODispatcherConfig cfg{};
    cfg.num_io_threads = 1;
    cfg.pin_io_threads = false;
    return cfg;
}

TEST(IoInbox, FullInboxOnStoppedDispatcherIsRefused) {
    IODispatcher io(one_thread());
    for (size_t i = 0; i < kInboxCapacity; ++i) {
        ASSERT_TRUE(io.post_to_io_thread(0, std::noop_coroutine()));
    }
    bool posted = true;
    run_bounded([&] { posted = io.post_to_io_thread(0, std::noop_coroutine()); },
                "post to a full, stopped inbox");
    EXPECT_FALSE(posted);
    EXPECT_EQ(io.inbox_posts_rejected(), 1u);
}

TEST(IoInbox, ForeignPosterWaitsForDrain) {
    IODispatcher io(one_thread());
    io.start();
    constexpr size_t kPosts = kInboxCapacity * 16;
    std::atomic<size_t> ran{0};
    run_bounded([&] {
        for (size_t i = 0; i < kPosts; ++i) {
            ASSERT_TRUE(io.post_to_io_thread(0, bump(&ran).h));
        }
    }, "foreign poster");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (ran.load() < kPosts && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(ran.load(), kPosts);
    EXPECT_EQ(io.inbox_posts_rejected(), 0u);
    io.stop();
}

struct SelfPost {
    IODispatcher* io;
    std::atomic<size_t>* ran;
    std::atomic<size_t> accepted{0};
    std::atomic<size_t> refused{0};
    std::atomic<bool> done{false};
};

Detached flood_self(SelfPost* s) {
    for (size_t i = 0; i < kInboxCapacity + 8; ++i) {
        Detached d = bump(s->ran);
        if (s->io->post_to_io_thread(0, d.h)) {
            s->accepted.fetch_add(1);
        } else {
            d.h.destroy();
            s->refused.fetch_add(1);
        }
    }
    s->done.store(true, std::memory_order_release);
    co_return;
}

TEST(IoInbox, IoThreadPostingToItselfIsRefusedNotDeadlocked) {
    IODispatcher io(one_thread());
    io.start();
    std::atomic<size_t> ran{0};
    SelfPost s{&io, &ran};
    ASSERT_TRUE(io.post_to_io_thread(0, flood_self(&s).h));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!s.done.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() > deadline) {
            std::fprintf(stderr, "HANG: IO thread deadlocked posting to itself\n");
            std::fflush(stderr);
            std::_Exit(1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(s.accepted.load() + s.refused.load(), kInboxCapacity + 8);
    EXPECT_GE(s.refused.load(), 1u);
    EXPECT_EQ(io.inbox_posts_rejected(), s.refused.load());
    while (ran.load() < s.accepted.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(ran.load(), s.accepted.load());
    io.stop();
}


TEST(IoInbox, TryPostRejectsInvalidAndStoppedOwnershipStaysWithCaller) {
    std::atomic<size_t> ran{0};  // outlives dispatcher and every callback
    IODispatcher io(one_thread());
    auto early = bump(&ran);
    const bool before = io.try_post_to_io_thread(0, early.h);
    EXPECT_FALSE(before);
    if (!before) early.h.destroy();
    EXPECT_FALSE(io.try_post_to_io_thread(0, {}));
    auto invalid = bump(&ran);
    const bool bad_index = io.try_post_to_io_thread(1, invalid.h);
    EXPECT_FALSE(bad_index);
    if (!bad_index) invalid.h.destroy();
    io.start();
    auto valid = bump(&ran);
    const bool accepted = io.try_post_to_io_thread(0, valid.h);
    EXPECT_TRUE(accepted);
    if (!accepted) valid.h.destroy();
    io.stop();
    EXPECT_EQ(ran.load(), 1u);
    auto late = bump(&ran);
    const bool after = io.try_post_to_io_thread(0, late.h);
    EXPECT_FALSE(after);
    if (!after) late.h.destroy();
    EXPECT_FALSE(io.post_to_io_thread(0, std::noop_coroutine()));
}

struct HeldReactor {
    std::atomic<bool> entered{false}, release{false};
};

Detached hold_reactor(HeldReactor* gate) {
    gate->entered.store(true);
    while (!gate->release.load()) gate->release.wait(false);
    co_return;
}

TEST(IoInbox, TryPostFullInboxReturnsBeforeTargetCanDrain) {
    HeldReactor gate;
    std::atomic<size_t> ran{0};
    std::atomic<bool> finished{false};
    bool posted = true;
    IODispatcher io(one_thread());
    io.start();
    auto hold = hold_reactor(&gate);
    const bool held = io.post_to_io_thread(0, hold.h);
    EXPECT_TRUE(held);
    if (!held) hold.h.destroy();
    for (unsigned i = 0; i < 1000 && !gate.entered.load(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_TRUE(gate.entered.load());
    for (size_t i = 0; i < kInboxCapacity; ++i)
        EXPECT_TRUE(io.try_post_to_io_thread(0, std::noop_coroutine()));
    std::thread poster([&] {
        auto task = bump(&ran);
        posted = io.try_post_to_io_thread(0, task.h);
        if (!posted) task.h.destroy();
        finished.store(true);
    });
    for (unsigned i = 0; i < 1000 && !finished.load(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_TRUE(finished.load()) << "try-post waited for target progress";
    gate.release.store(true);  // always release before joining, including failures
    gate.release.notify_all();
    poster.join();
    io.stop();
    EXPECT_FALSE(posted);
    EXPECT_EQ(ran.load(), 0u);
}

constexpr size_t kTryPublishers = 4, kTryRace = 64, kTryLate = 16, kTryBacklog = 64;
constexpr size_t kTryStride = kTryRace + kTryLate;
constexpr size_t kTryTracked = kTryBacklog + kTryPublishers * kTryStride;
struct TryPostLedger {
    HeldReactor gate;
    std::array<std::atomic<size_t>, kTryTracked> ran{};
    std::array<bool, kTryTracked> accepted{};
    std::atomic<bool> begin{false}, late{false};
    std::atomic<size_t> ready{0};
    void submit(IODispatcher& io, size_t id) {
        auto task = bump(&ran[id]);
        accepted[id] = io.try_post_to_io_thread(0, task.h);
        if (!accepted[id]) task.h.destroy();
    }
};

void race_try_posts_with_stop(bool hold_until_published) {
    TryPostLedger ledger;  // all callbacks/captures survive I/O join
    IODispatcher io(one_thread());
    io.start();
    auto hold = hold_reactor(&ledger.gate);
    const bool held = io.post_to_io_thread(0, hold.h);
    EXPECT_TRUE(held);
    if (!held) hold.h.destroy();
    for (unsigned i = 0; i < 1000 && !ledger.gate.entered.load(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_TRUE(ledger.gate.entered.load());
    for (size_t id = 0; id < kTryBacklog; ++id) {
        ledger.submit(io, id);
        EXPECT_TRUE(ledger.accepted[id]);
        EXPECT_EQ(ledger.ran[id].load(), 0u);
    }
    std::array<std::thread, kTryPublishers> publishers;
    for (size_t p = 0; p < publishers.size(); ++p) publishers[p] = std::thread([&, p] {
        const size_t base = kTryBacklog + p * kTryStride;
        ledger.ready.fetch_add(1);
        while (!ledger.begin.load()) ledger.begin.wait(false);
        for (size_t i = 0; i < kTryRace; ++i) ledger.submit(io, base + i);
        while (!ledger.late.load()) ledger.late.wait(false);
        for (size_t i = 0; i < kTryLate; ++i) ledger.submit(io, base + kTryRace + i);
    });
    std::thread stopper([&] {
        ledger.ready.fetch_add(1);
        while (!ledger.begin.load()) ledger.begin.wait(false);
        io.stop();
    });
    for (unsigned i = 0; i < 1000 && ledger.ready.load() != kTryPublishers + 1; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_EQ(ledger.ready.load(), kTryPublishers + 1);
    ledger.begin.store(true);
    ledger.begin.notify_all();
    if (!hold_until_published) {
        ledger.gate.release.store(true);
        ledger.gate.release.notify_all();
    }
    for (unsigned i = 0; i < 1000 && io.is_running(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_FALSE(io.is_running());
    ledger.late.store(true);
    ledger.late.notify_all();
    for (auto& publisher : publishers) publisher.join();
    ledger.gate.release.store(true);
    ledger.gate.release.notify_all();
    stopper.join();
    for (size_t id = 0; id < kTryTracked; ++id)
        EXPECT_EQ(ledger.ran[id].load(), ledger.accepted[id] ? 1u : 0u) << "ID " << id;
    for (size_t p = 0; p < kTryPublishers; ++p)
        for (size_t i = 0; i < kTryLate; ++i)
            EXPECT_FALSE(ledger.accepted[kTryBacklog + p * kTryStride + kTryRace + i]);
}

TEST(IoInbox, TryPostConcurrentStopDrainsAcceptedHandlesExactlyOnce) {
    // Scheduling stress cannot force every internal publication interleave.
    for (unsigned round = 0; round < 8; ++round) {
        SCOPED_TRACE(round);
        race_try_posts_with_stop((round & 1u) == 0);
    }
}

}  // namespace
