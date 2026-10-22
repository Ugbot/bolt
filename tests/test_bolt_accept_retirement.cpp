#include <gtest/gtest.h>
#include "bolt/api/net/coro_tcp_listener.h"

#include <atomic>
#include <chrono>
#include <climits>
#include <cerrno>
#include <memory>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace core = bolt::api::core;
namespace net = bolt::api::net;
using Status = core::AcceptRetirementStatus;
using Clock = std::chrono::steady_clock;

#ifdef __linux__
namespace bolt::api::core {
struct AcceptRetirementTestAccess {
    static void limit(io_uring_io& io, unsigned n) { io.test_limit_next_enter(n); }
    static void interrupt(io_uring_io& io) { io.test_interrupt_next_enter(); }
    static unsigned pending(io_uring_io& io) { return io.test_pending_submissions(); }
};
}
#endif

namespace bolt::api::net {
// Observation of the actual startup-unwind transition, never a success flag or
// injected completion. Used to release a deliberately paused reactor safely.
struct AcceptRetirementTestAccess {
    static bool stopping(const CoroTcpListener& listener) {
        return listener.stop_requested_.load(std::memory_order_acquire);
    }
};
}

namespace {
int listen_socket(uint16_t* port) {
    net::sys::startup();
    const int fd = static_cast<int>(socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (fd < 0) return -1;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(*port);
    socklen_t bytes = sizeof(address);
    if (bind(fd, reinterpret_cast<sockaddr*>(&address), bytes) != 0 ||
        listen(fd, 128) != 0 || getsockname(fd, reinterpret_cast<sockaddr*>(&address), &bytes) != 0 ||
        net::sys::set_nonblocking(fd) != 0) {
        net::sys::close_socket(fd); return -1;
    }
    *port = ntohs(address.sin_port);
    return fd;
}
uint16_t free_port() {
    uint16_t port = 0;
    const int fd = listen_socket(&port);
    if (fd >= 0) net::sys::close_socket(fd);
    return fd >= 0 ? port : 0;
}
bool still_bound(int fd) {
    sockaddr_in address{}; socklen_t bytes = sizeof(address);
    return getsockname(fd, reinterpret_cast<sockaddr*>(&address), &bytes) == 0;
}
template<class Predicate> bool until(Predicate done) {
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    for (unsigned n = 0; n < 1000000u && Clock::now() < deadline; ++n) {
        if (done()) return true;
        std::this_thread::yield();
    }
    return done();
}
int connect_local(uint16_t port) {
    const int fd = static_cast<int>(socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (fd < 0) return -1;
    if (net::sys::set_nonblocking(fd) != 0) { net::sys::close_socket(fd); return -1; }
    sockaddr_in peer{}; peer.sin_family = AF_INET;
    peer.sin_addr.s_addr = htonl(INADDR_LOOPBACK); peer.sin_port = htons(port);
    const int result = connect(fd, reinterpret_cast<sockaddr*>(&peer), sizeof(peer)); // one connect
#ifdef _WIN32
    const int error = result == 0 ? 0 : WSAGetLastError();
    const bool pending = error == WSAEWOULDBLOCK || error == WSAEINPROGRESS || error == WSAEINTR;
#else
    const int error = result == 0 ? 0 : errno;
    const bool pending = error == EINPROGRESS || error == EINTR;
#endif
    if (result != 0 && !pending) { net::sys::close_socket(fd); return -1; }
    const bool connected = until([&] {
        int socket_error = 0; socklen_t bytes = sizeof(socket_error);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&socket_error), &bytes) != 0 || socket_error != 0) return false;
        bytes = sizeof(peer);
        return getpeername(fd, reinterpret_cast<sockaddr*>(&peer), &bytes) == 0 &&
            peer.sin_family == AF_INET && peer.sin_addr.s_addr == htonl(INADDR_LOOPBACK) && peer.sin_port == htons(port);
    });
    if (!connected) { net::sys::close_socket(fd); return -1; }
    return fd;
}

// The storage owner never frees a pending kernel operation on a failed ASSERT.
// Cleanup uses the same cold progress contract as the listener's legacy stop.
struct DirectOwner {
    std::unique_ptr<core::async_io> io;
    core::AcceptRetirement record;
    uint16_t port = 0;
    explicit DirectOwner(core::io_backend backend) {
        core::async_io_config config;
        config.backend = backend; config.queue_depth = 64u;
        io = core::async_io::create(config);
        record.fd = listen_socket(&port);
        record.loop_exited.store(true);
    }
    ~DirectOwner() {
        if (!io || record.fd < 0) return;
        record.loop_exited.store(true, std::memory_order_release);
        while (io->try_retire_accepts(record, Clock::now() + std::chrono::milliseconds(10)) != Status::Complete)
            io->poll(1000u);
    }
    bool drain() {
        const auto deadline = Clock::now() + std::chrono::seconds(5);
        for (unsigned n = 0; n < 10000u && Clock::now() < deadline; ++n) {
            const auto result = io->try_retire_accepts(record, deadline);
            if (result == Status::Complete) return true;
            if (result == Status::Error || result == Status::Unavailable) return false;
            io->poll(1000u);
        }
        return record.complete.load(std::memory_order_acquire);
    }
};

class AcceptBackend : public ::testing::TestWithParam<core::io_backend> {};

TEST_P(AcceptBackend, PendingRetainsFdThenRealCancellationReleasesIt) {
    DirectOwner owner(GetParam());
    ASSERT_NE(owner.io, nullptr);
    if (owner.io->backend() != GetParam()) GTEST_SKIP() << "requested backend unavailable; actual=" << owner.io->backend_name();
    SCOPED_TRACE(owner.io->backend_name());
    ASSERT_GE(owner.record.fd, 0);
    owner.record.loop_exited.store(false);
    unsigned callbacks = 0;
    ASSERT_EQ(owner.io->accept_async(owner.record.fd, [&](const core::io_event& event) {
        EXPECT_EQ(event.result, -1); ++callbacks;
        owner.record.loop_exited.store(true, std::memory_order_release);
    }, nullptr, &owner.record), 0);
    owner.io->poll(0u); // actual original SQE reaches the kernel on io_uring
    const auto before = Clock::now();
    EXPECT_EQ(owner.io->try_retire_accepts(owner.record, before), Status::Pending);
    EXPECT_LT(Clock::now() - before, std::chrono::milliseconds(100));
    EXPECT_TRUE(still_bound(owner.record.fd));
    EXPECT_FALSE(owner.record.complete.load());
    EXPECT_EQ(owner.record.operations.load(), 1u);
    EXPECT_TRUE(owner.drain());
    EXPECT_EQ(callbacks, 1u);
    EXPECT_EQ(owner.record.operations.load(), 0u);
    EXPECT_FALSE(still_bound(owner.record.fd));
    uint16_t same_port = owner.port;
    const int replacement = listen_socket(&same_port);
    EXPECT_GE(replacement, 0); // descriptor/port reuse only after acknowledgment
    if (replacement >= 0) net::sys::close_socket(replacement);
}

TEST_P(AcceptBackend, SuccessfulOriginalCallbackRequestsRetirementBeforeReturning) {
    DirectOwner owner(GetParam());
    ASSERT_NE(owner.io, nullptr);
    if (owner.io->backend() != GetParam()) GTEST_SKIP() << "actual=" << owner.io->backend_name();
    ASSERT_GE(owner.record.fd, 0);
    unsigned callbacks = 0;
    ASSERT_EQ(owner.io->accept_async(owner.record.fd, [&](const core::io_event& event) {
        EXPECT_GE(event.result, 0); ++callbacks;
        if (event.result >= 0) net::sys::close_socket(static_cast<int>(event.result));
        EXPECT_EQ(owner.record.operations.load(), 1u); // callback has not returned
        EXPECT_EQ(owner.io->try_retire_accepts(owner.record, Clock::now() + std::chrono::seconds(1)), Status::Pending);
        EXPECT_FALSE(owner.record.complete.load());
        EXPECT_TRUE(still_bound(owner.record.fd));
    }, nullptr, &owner.record), 0);
    const int client = connect_local(owner.port);
    ASSERT_GE(client, 0);
    net::sys::close_socket(client);
    for (unsigned n = 0; n < 10000u && callbacks == 0u; ++n) owner.io->poll(1000u);
    EXPECT_EQ(callbacks, 1u); // positive original must precede retirement intent
    ASSERT_TRUE(owner.drain());
    EXPECT_EQ(callbacks, 1u);
}

#ifndef _WIN32
TEST_P(AcceptBackend, HighFdTrackedAdmissionRefusesBeforeRegistrationAndRetainsCleanup) {
    DirectOwner owner(GetParam());
    ASSERT_NE(owner.io, nullptr);
    if (GetParam() == core::io_backend::io_uring) GTEST_SKIP() << "numeric table bound applies to readiness backends";
    ASSERT_GE(owner.record.fd, 0);
    const int high = fcntl(owner.record.fd, F_DUPFD_CLOEXEC, 65536);
    const int error = errno;
    if (high < 0) GTEST_SKIP() << "OS descriptor limit does not permit high-fd control: " << error;
    net::sys::close_socket(owner.record.fd);
    owner.record.fd = high; // still exclusive startup, before any publication
    unsigned callbacks = 0u;
    errno = 0;
    const int result = owner.io->accept_async(high, [&](const core::io_event&) { ++callbacks; }, nullptr, &owner.record);
    const int refusal = errno;
    EXPECT_EQ(result, -1); EXPECT_EQ(refusal, EOVERFLOW);
    EXPECT_EQ(owner.record.operations.load(), 0u); EXPECT_EQ(callbacks, 0u);
    EXPECT_TRUE(still_bound(high));
    EXPECT_TRUE(owner.drain()); EXPECT_FALSE(still_bound(high));
}
#endif

#ifdef __linux__
INSTANTIATE_TEST_SUITE_P(Native, AcceptBackend, ::testing::Values(core::io_backend::epoll, core::io_backend::io_uring));
#elif defined(_WIN32)
INSTANTIATE_TEST_SUITE_P(Native, AcceptBackend, ::testing::Values(core::io_backend::iocp));
#else
INSTANTIATE_TEST_SUITE_P(Native, AcceptBackend, ::testing::Values(core::io_backend::kqueue));
#endif

TEST(AcceptRetirement, OriginalAndCancelOrdersRetainTokenThroughCallbackReturn) {
    for (unsigned order = 0; order < 3u; ++order) {
        core::detail::AcceptCompletionWitness witness;
        witness.cancel_inflight = true;
        EXPECT_FALSE(witness.recyclable());
        if (order != 1u) witness.original_seen = true;
        else EXPECT_EQ(witness.cancel_completed(0), 0);
        if (order == 2u) witness.callback_done = true;
        EXPECT_FALSE(witness.recyclable());
        if (order != 1u) EXPECT_EQ(witness.cancel_completed(0), 0);
        else witness.original_seen = true;
        EXPECT_EQ(witness.recyclable(), order == 2u);
        witness.callback_done = true;
        EXPECT_TRUE(witness.recyclable());
    }
    core::detail::AcceptCompletionWitness absent;
    absent.cancel_inflight = true;
    for (const int result : {ENOENT, EALREADY}) {
        EXPECT_EQ(absent.cancel_completed(-result), 0);
        EXPECT_TRUE(absent.cancel_retry);
        EXPECT_FALSE(absent.recyclable());
        absent.cancel_inflight = true;
    }
}

TEST(AcceptRetirement, PartialEnterAndInterruptKeepUnconsumedSubmissionAcrossWrap) {
    using core::detail::pending_submissions;
    using core::detail::original_consumed;
    const unsigned published = 2u;
    const uint64_t original_ticket = uint64_t{UINT_MAX} + 2u;
    core::detail::SubmissionProgress progress{UINT_MAX - 1u, UINT_MAX - 1u};
    EXPECT_EQ(pending_submissions(published, UINT_MAX - 1u), 4u);
    EXPECT_FALSE(original_consumed(progress.consumed, original_ticket));
    EXPECT_TRUE(progress.observe(0u, 64u));
    EXPECT_EQ(pending_submissions(published, 0u), 2u); // partial enter
    EXPECT_FALSE(original_consumed(progress.consumed, original_ticket));
    EXPECT_TRUE(progress.observe(0u, 64u));
    EXPECT_EQ(pending_submissions(published, 0u), 2u); // EINTR, unchanged kernel head
    EXPECT_TRUE(progress.observe(1u, 64u));
    EXPECT_TRUE(original_consumed(progress.consumed, original_ticket));
    EXPECT_EQ(pending_submissions(published, published), 0u);
}

TEST(AcceptRetirement, AncientPendingOriginalRemainsAdmittedBeyondHalfRangeAndHeadWrap) {
    core::detail::SubmissionProgress progress{0x80000010u, 0x80000010ull};
    EXPECT_TRUE(core::detail::original_consumed(progress.consumed, 1u));
    EXPECT_TRUE(progress.observe(0x80000020u, 64u));
    EXPECT_TRUE(core::detail::original_consumed(progress.consumed, 1u));
    progress = core::detail::SubmissionProgress{UINT_MAX - 1u, uint64_t{UINT_MAX} - 1u};
    EXPECT_TRUE(progress.observe(1u, 64u));
    EXPECT_EQ(progress.consumed, uint64_t{UINT_MAX} + 2u);
    EXPECT_TRUE(core::detail::original_consumed(progress.consumed, 1u));
    EXPECT_FALSE(progress.observe(1000u, 64u)); // observation gap violates producer bound
}

#ifdef __linux__
TEST(AcceptRetirement, UringRealOriginalsSurviveLimitedEnterThenInterruptedEnter) {
    DirectOwner owner(core::io_backend::io_uring);
    ASSERT_NE(owner.io, nullptr);
    if (owner.io->backend() != core::io_backend::io_uring) GTEST_SKIP() << "io_uring unavailable";
    ASSERT_GE(owner.record.fd, 0);
    auto& ring = static_cast<core::io_uring_io&>(*owner.io);
    unsigned callbacks = 0u;
    for (unsigned n = 0; n < 2u; ++n) ASSERT_EQ(ring.accept_async(owner.record.fd, [&](const core::io_event& event) {
        EXPECT_EQ(event.result, -1); ++callbacks;
    }, nullptr, &owner.record), 0);
    core::AcceptRetirementTestAccess::limit(ring, 1u); // real enter admits only the initial wake SQE
    ring.poll(0u);
    EXPECT_EQ(core::AcceptRetirementTestAccess::pending(ring), 2u);
    core::AcceptRetirementTestAccess::interrupt(ring); // no syscall or CQE fabricated
    ring.poll(0u);
    EXPECT_EQ(core::AcceptRetirementTestAccess::pending(ring), 2u);
    EXPECT_EQ(callbacks, 0u);
    EXPECT_EQ(owner.record.operations.load(), 2u);
    EXPECT_TRUE(owner.drain()); // actual kernel original + cancel CQEs release both slots
    EXPECT_EQ(callbacks, 2u);
}

TEST(AcceptRetirement, UringFullOperationPoolAndSmallSqStillAdmitControlCancellation) {
    DirectOwner owner(core::io_backend::io_uring);
    ASSERT_NE(owner.io, nullptr);
    if (owner.io->backend() != core::io_backend::io_uring) GTEST_SKIP() << "io_uring unavailable";
    ASSERT_GE(owner.record.fd, 0);
    unsigned admitted = 0u, callbacks = 0u;
    for (unsigned attempt = 0; attempt < 16384u && admitted < 4096u; ++attempt) {
        const int result = owner.io->accept_async(owner.record.fd, [&](const core::io_event& event) {
            EXPECT_EQ(event.result, -1); ++callbacks;
        }, nullptr, &owner.record);
        if (result == 0) ++admitted;
        else owner.io->poll(0u); // actual full SQ, no clients and no free op slot
    }
    EXPECT_EQ(admitted, 4096u);
    EXPECT_EQ(owner.record.operations.load(), 4096u);
    EXPECT_EQ(owner.io->accept_async(owner.record.fd, [](const core::io_event&) {}, nullptr, &owner.record), -1);
    EXPECT_TRUE(owner.drain());
    EXPECT_EQ(callbacks, 4096u);
    EXPECT_EQ(owner.record.operations.load(), 0u);
}
#endif

core::detached_task owner_pause(std::atomic<bool>* entered, std::atomic<bool>* release, std::atomic<bool>* exited) {
    entered->store(true, std::memory_order_release);
    while (!release->load(std::memory_order_acquire)) std::this_thread::yield();
    exited->store(true, std::memory_order_release);
    co_return;
}
core::detached_task filler() { co_return; }
struct Pause {
    std::atomic<bool> entered{false}, release{false}, exited{false};
    bool published = false;
    ~Pause() {
        release.store(true, std::memory_order_release);
        while (published && !exited.load(std::memory_order_acquire)) std::this_thread::yield();
    }
    bool post(net::IODispatcher& io, size_t reactor) {
        auto task = owner_pause(&entered, &release, &exited);
        auto handle = task.release();
        if (!io.try_post_to_io_thread(reactor, handle)) { handle.destroy(); return false; }
        published = true;
        return until([&] { return entered.load(std::memory_order_acquire); });
    }
    bool post_worker(core::WorkerThreadPool& workers) {
        auto task = owner_pause(&entered, &release, &exited);
        auto handle = task.release();
        if (!workers.submit(handle)) { handle.destroy(); return false; }
        published = true;
        return until([&] { return entered.load(std::memory_order_acquire); });
    }
};
struct Shared {
    core::WorkerThreadPool workers{core::WorkerPoolConfig{.num_workers = 1, .blocking_workers = 0}};
    net::IODispatcher io;
    explicit Shared(size_t reactors = 1u, bool inline_resume = true,
        core::io_backend backend = core::io_backend::auto_detect) : io(net::IODispatcherConfig{
        .num_io_threads = reactors, .io_config = core::async_io_config{.backend = backend},
        .worker_pool = &workers, .inline_resume = inline_resume}) {
        workers.start(); io.start();
    }
    ~Shared() { io.stop(); workers.stop(); }
};
net::CoroTcpListenerConfig listener_config(uint16_t port) {
    net::CoroTcpListenerConfig config; config.port = port; config.num_io_threads = 1u;
    return config;
}
core::coro_task<void> counted(net::IODispatcher&, int fd, std::atomic<unsigned>* count) {
    net::sys::close_socket(fd); count->fetch_add(1u, std::memory_order_release); co_return;
}
bool listener_drained(net::CoroTcpListener& listener) {
    return until([&] { return listener.try_retire_accepts(Clock::now() + std::chrono::milliseconds(5)) == Status::Complete; });
}

struct TrackedLoop {
    Shared& shared;
    core::AcceptRetirement record;
    std::atomic<unsigned> completions{0u};
    std::atomic<bool> wrong_owner{false};
    uint16_t port = 0;
    explicit TrackedLoop(Shared& owner) : shared(owner) {
        record.fd = listen_socket(&port); record.reactor = 1u;
        record.loop_exited.store(true);
    }
    ~TrackedLoop() {
        if (record.fd < 0) return;
        while (shared.io.try_retire_accepts(1u, record, Clock::now() + std::chrono::milliseconds(5)) != Status::Complete)
            std::this_thread::yield();
    }
    static core::detached_task run(TrackedLoop* owner) {
        auto& record = owner->record;
        while (!record.requested.load(std::memory_order_acquire)) {
            const int fd = co_await owner->shared.io.async_accept(record.fd, &record);
            if (fd < 0) break;
            net::sys::close_socket(fd);
            if (owner->shared.io.current_io_thread_index() != record.reactor) owner->wrong_owner.store(true);
            owner->completions.fetch_add(1u, std::memory_order_release);
        }
        record.loop_exited.store(true, std::memory_order_release);
    }
    bool start() {
        record.loop_exited.store(false);
        auto task = run(this); auto handle = task.release();
        if (shared.io.try_post_to_io_thread(1u, handle)) return true;
        handle.destroy(); record.loop_exited.store(true); return false;
    }
};

TEST(AcceptRetirement, WorkerResumePolicyDoesNotMigrateTrackedAcceptRegistrationToEngineZero) {
    Shared shared(2u, false);
    if (!shared.io.per_thread_engines()) GTEST_SKIP() << "IOCP has a shared completion engine";
    TrackedLoop owner(shared);
    ASSERT_GE(owner.record.fd, 0);
    ASSERT_TRUE(owner.start());
    for (unsigned n = 1u; n <= 2u; ++n) {
        const int client = connect_local(owner.port);
        ASSERT_GE(client, 0); net::sys::close_socket(client);
        ASSERT_TRUE(until([&] { return owner.completions.load(std::memory_order_acquire) == n; }));
    }
    ASSERT_TRUE(until([&] { return shared.io.get_stats().accepts_started >= 3u && owner.record.operations.load() == 1u; }));
    EXPECT_FALSE(owner.wrong_owner.load());
    ASSERT_NE(shared.io.io_engine(), nullptr);
    EXPECT_EQ(shared.io.io_engine()->get_stats().accepts, 0u);
    EXPECT_TRUE(until([&] { return shared.io.try_retire_accepts(1u, owner.record,
        Clock::now() + std::chrono::milliseconds(5)) == Status::Complete; }));
}

TEST(AcceptRetirement, QueuedInitialStartStaysOwnedWhileActualReactorIsPaused) {
    Shared shared;
    const uint16_t port = free_port(); ASSERT_NE(port, 0);
    std::atomic<unsigned> calls{0};
    net::CoroTcpListener listener(listener_config(port), [&](net::IODispatcher& io, int fd) {
        return counted(io, fd, &calls);
    }, &shared.io, &shared.workers);
    Pause pause;
    ASSERT_TRUE(shared.io.per_thread_engines() ? pause.post(shared.io, 0u) : pause.post_worker(shared.workers));
    ASSERT_EQ(listener.start_background(), 0);
    const auto before = Clock::now();
    EXPECT_EQ(listener.try_retire_accepts(before + std::chrono::milliseconds(5)), Status::Pending);
    EXPECT_LT(Clock::now() - before, std::chrono::milliseconds(100));
    EXPECT_EQ(listener.start_background(), -1);
    EXPECT_EQ(errno, EBUSY);
    uint16_t same = port; const int premature = listen_socket(&same);
    EXPECT_LT(premature, 0);
    if (premature >= 0) net::sys::close_socket(premature);
    pause.release.store(true, std::memory_order_release);
    ASSERT_TRUE(listener_drained(listener));
    EXPECT_EQ(calls.load(), 0u);
    listener.stop();
    same = port; const int replacement = listen_socket(&same);
    EXPECT_GE(replacement, 0);
    if (replacement >= 0) net::sys::close_socket(replacement);
    EXPECT_TRUE(shared.io.is_running());
}

TEST(AcceptRetirement, ClientArrivingWhileOwnerPausedCannotStartHandlerAfterRetirementRequest) {
    Shared shared;
    const uint16_t port = free_port(); ASSERT_NE(port, 0);
    std::atomic<unsigned> calls{0};
    net::CoroTcpListener listener(listener_config(port), [&](net::IODispatcher& io, int fd) {
        return counted(io, fd, &calls);
    }, &shared.io, &shared.workers);
    ASSERT_EQ(listener.start_background(), 0);
    ASSERT_TRUE(until([&] { return shared.io.io_engine()->get_stats().accepts == 1u; }));
    Pause pause; ASSERT_TRUE(pause.post(shared.io, 0u));
    const int client = connect_local(port);
    ASSERT_GE(client, 0);
    EXPECT_EQ(listener.try_retire_accepts(Clock::now() + std::chrono::milliseconds(5)), Status::Pending);
    pause.release.store(true, std::memory_order_release);
    const bool drained = listener_drained(listener);
    net::sys::close_socket(client);
    EXPECT_TRUE(drained);
    EXPECT_EQ(calls.load(), 0u);
    EXPECT_EQ(listener.get_stats().connections_active, 0u);
    listener.stop();
}

core::coro_task<void> worker_handler(net::IODispatcher& io, int fd,
    std::atomic<bool>* worker, std::atomic<unsigned>* calls) {
    char byte = 0;
    const auto result = co_await io.async_read(fd, &byte, 1u);
    worker->store(result == 1 && byte == 'w' && io.current_io_thread_index() == SIZE_MAX, std::memory_order_release);
    net::sys::close_socket(fd);
    calls->fetch_add(1u, std::memory_order_release);
}
TEST(AcceptRetirement, PinningAcceptControlPreservesClientWorkerResumePolicy) {
    Shared shared(2u, false);
    const uint16_t port = free_port(); ASSERT_NE(port, 0);
    std::atomic<bool> worker{false}; std::atomic<unsigned> calls{0};
    auto config = listener_config(port); config.num_io_threads = 2u;
    net::CoroTcpListener listener(config, [&](net::IODispatcher& io, int fd) {
        return worker_handler(io, fd, &worker, &calls);
    }, &shared.io, &shared.workers);
    ASSERT_EQ(listener.start_background(), 0);
    const int client = connect_local(port);
    ASSERT_GE(client, 0);
    const char byte = 'w';
    const auto sent = net::sys::send_bytes(client, &byte, 1u);
    net::sys::close_socket(client);
    ASSERT_EQ(sent, 1);
    EXPECT_TRUE(until([&] { return calls.load(std::memory_order_acquire) == 1u; }));
    EXPECT_TRUE(worker.load(std::memory_order_acquire));
    EXPECT_TRUE(until([&] { return listener.get_stats().connections_active == 0u; }));
    listener.stop();
}

TEST(AcceptRetirement, RetiringOneSharedListenerLeavesTheOtherAccepting) {
    Shared shared;
    const uint16_t first_port = free_port(), second_port = free_port();
    ASSERT_NE(first_port, 0); ASSERT_NE(second_port, 0); ASSERT_NE(first_port, second_port);
    std::atomic<unsigned> calls{0};
    auto handler = [&](net::IODispatcher& io, int fd) { return counted(io, fd, &calls); };
    net::CoroTcpListener first(listener_config(first_port), handler, &shared.io, &shared.workers);
    net::CoroTcpListener survivor(listener_config(second_port), handler, &shared.io, &shared.workers);
    ASSERT_EQ(first.start_background(), 0);
    ASSERT_EQ(survivor.start_background(), 0);
    ASSERT_TRUE(listener_drained(first)); first.stop();
    EXPECT_TRUE(shared.io.is_running());
    const int client = connect_local(second_port);
    ASSERT_GE(client, 0);
    net::sys::close_socket(client);
    EXPECT_TRUE(until([&] { return calls.load(std::memory_order_acquire) == 1u; }));
    EXPECT_TRUE(until([&] { return survivor.get_stats().connections_active == 0u; }));
    survivor.stop();
}

#ifndef _WIN32
struct LowDescriptors {
    std::vector<int> owned;
    ~LowDescriptors() { for (int fd : owned) close(fd); }
    bool fill() {
        constexpr unsigned cap = static_cast<unsigned>(core::AcceptRetirement::kReadinessFdLimit) + 2u;
        owned.reserve(cap);
        const int source = open("/dev/null", O_RDONLY | O_CLOEXEC);
        if (source < 0) return false;
        owned.push_back(source);
        for (unsigned n = 1u; n < cap; ++n) {
            const int fd = fcntl(source, F_DUPFD_CLOEXEC, 3);
            if (fd < 0) return false;
            owned.push_back(fd);
            if (fd >= core::AcceptRetirement::kReadinessFdLimit) return true;
        }
        return false;
    }
};
TEST(AcceptRetirement, HighFdListenerStartupReturnsExplicitRefusalAndReleasesItsPort) {
    rlimit limit{};
    ASSERT_EQ(getrlimit(RLIMIT_NOFILE, &limit), 0);
    if (limit.rlim_cur <= static_cast<rlim_t>(core::AcceptRetirement::kReadinessFdLimit + 64))
        GTEST_SKIP() << "OS soft descriptor limit cannot exercise high-fd listener startup";
#ifdef __linux__
    Shared shared(1u, true, core::io_backend::epoll);
#else
    Shared shared(1u, true, core::io_backend::kqueue);
#endif
    const auto port = free_port(); ASSERT_NE(port, 0);
    LowDescriptors descriptors;
    if (!descriptors.fill()) GTEST_SKIP() << "OS resource admission refused bounded high-fd fixture: " << errno;
    std::atomic<unsigned> calls{0};
    net::CoroTcpListener listener(listener_config(port), [&](net::IODispatcher& io, int fd) {
        return counted(io, fd, &calls);
    }, &shared.io, &shared.workers);
    errno = 0;
    const int result = listener.start_background();
    const int error = errno;
    EXPECT_EQ(result, -1); EXPECT_EQ(error, EOVERFLOW);
    EXPECT_EQ(listener.accept_error(), EOVERFLOW);
    EXPECT_FALSE(listener.is_running()); EXPECT_EQ(calls.load(), 0u);
    uint16_t same_port = port; const int replacement = listen_socket(&same_port);
    EXPECT_GE(replacement, 0);
    if (replacement >= 0) net::sys::close_socket(replacement);
}
#endif

#if defined(__linux__)
struct BorrowedAccepts {
    Shared& shared;
    core::AcceptRetirement record;
    std::atomic<unsigned> callbacks{0u};
    uint16_t port = 0;
    explicit BorrowedAccepts(Shared& owner) : shared(owner) {
        record.fd = listen_socket(&port); record.loop_exited.store(true);
    }
    ~BorrowedAccepts() {
        if (record.fd >= 0) while (!drain()) std::this_thread::yield();
    }
    bool drain() {
        return until([&] { return shared.io.try_retire_accepts(0u, record,
            Clock::now() + std::chrono::milliseconds(5)) == Status::Complete; });
    }
};
TEST(AcceptRetirement, RealUringAdmissionExhaustionStopsListenerAndExposesErrorWithOwnersRetained) {
    Shared shared;
    ASSERT_NE(shared.io.io_engine(), nullptr);
    if (shared.io.io_engine()->backend() != core::io_backend::io_uring) GTEST_SKIP() << "io_uring unavailable";
    BorrowedAccepts pending(shared);
    ASSERT_GE(pending.record.fd, 0);
    std::atomic<unsigned> handlers{0};
    const auto port = free_port(); ASSERT_NE(port, 0);
    net::CoroTcpListener listener(listener_config(port), [&](net::IODispatcher& io, int fd) {
        return counted(io, fd, &handlers);
    }, &shared.io, &shared.workers);
    Pause pause; ASSERT_TRUE(pause.post(shared.io, 0u));
    unsigned admitted = 0u;
    for (; admitted < 4097u; ++admitted) {
        const int result = shared.io.io_engine()->accept_async(pending.record.fd, [&](const core::io_event& event) {
            EXPECT_EQ(event.result, -1); pending.callbacks.fetch_add(1u, std::memory_order_release);
        }, nullptr, &pending.record);
        if (result != 0) break;
    }
    ASSERT_GT(admitted, 0u); ASSERT_LE(admitted, 4096u);
    ASSERT_EQ(listener.start_background(), 0); // queued startup ownership accepted
    pause.release.store(true, std::memory_order_release);
    ASSERT_TRUE(until([&] { return !listener.is_running(); }));
    EXPECT_NE(listener.accept_error(), 0);
    EXPECT_EQ(listener.start_background(), -1); // failed but not yet retired
    EXPECT_EQ(errno, EBUSY);
    listener.stop();
    EXPECT_TRUE(pending.drain());
    EXPECT_EQ(pending.callbacks.load(std::memory_order_acquire), admitted);
    EXPECT_EQ(handlers.load(), 0u);
    uint16_t same = port; const int replacement = listen_socket(&same);
    EXPECT_GE(replacement, 0);
    if (replacement >= 0) net::sys::close_socket(replacement);
}

TEST(AcceptRetirement, PartialStartUnwindRetiresAlreadyPublishedAcceptOnItsOwner) {
    Shared shared(2u);
    std::atomic<unsigned> calls{0};
    auto config = listener_config(free_port()); config.num_io_threads = 2u;
    ASSERT_NE(config.port, 0);
    net::CoroTcpListener listener(config, [&](net::IODispatcher& io, int fd) { return counted(io, fd, &calls); }, &shared.io, &shared.workers);
    Pause pause; ASSERT_TRUE(pause.post(shared.io, 1u));
    unsigned published = 0u;
    for (unsigned n = 0; n < 1024u; ++n) {
        auto task = filler(); auto handle = task.release();
        if (!shared.io.try_post_to_io_thread(1u, handle)) { handle.destroy(); break; }
        ++published;
    }
    ASSERT_GT(published, 0u); ASSERT_LT(published, 1024u);
    // Existing stop_requested is the real unwind entry, after reactor-0's
    // accept was published and reactor-1 startup admission was refused.
    std::atomic<bool> saw_unwind{false};
    std::thread release([&] {
        saw_unwind.store(until([&] { return net::AcceptRetirementTestAccess::stopping(listener); }));
        pause.release.store(true, std::memory_order_release);
    });
    const int result = listener.start_background();
    release.join();
    EXPECT_EQ(result, -1); EXPECT_TRUE(saw_unwind.load());
    EXPECT_EQ(listener.accept_error(), ENOBUFS);
    EXPECT_FALSE(listener.is_running()); EXPECT_EQ(calls.load(), 0u);
    uint16_t port = config.port; const int replacement = listen_socket(&port);
    EXPECT_GE(replacement, 0);
    if (replacement >= 0) net::sys::close_socket(replacement);
}
#endif
} // namespace
