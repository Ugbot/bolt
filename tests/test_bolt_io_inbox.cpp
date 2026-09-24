// IODispatcher::post_to_io_thread under a full inbox (G2ETL-54).
//
// A full inbox must never hang the poster: a foreign thread waits for the IO
// thread to drain, the IO thread posting to itself is refused (it is the only
// consumer), and a post to a dispatcher that is not running is refused.

#include <bolt/api/net/io_dispatcher.h>

#include <gtest/gtest.h>

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

}  // namespace
