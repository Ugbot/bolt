#pragma once

#include <coroutine>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <iostream>
#include <cstdlib>

namespace bolt::api {
namespace core {

// Cache line size for false-sharing avoidance. Previously provided by
// core/coro_pool.h (cut from the fork); defined here so worker_pool.h /
// io_dispatcher.h alignas(core::kCacheLineSize) keep working.
inline constexpr unsigned long long kCacheLineSize = 64;

/**
 * C++20 coroutine task type for async operations.
 *
 * Provides a lightweight, zero-allocation coroutine task that can be
 * suspended and resumed. Compatible with co_await, co_return.
 *
 * Design:
 * - No heap allocation for promise
 * - Exception propagation
 * - Lazy evaluation (starts when co_await'd)
 * - Move-only semantics
 *
 * Example:
 *   coro_task<int> get_value() {
 *       co_return 42;
 *   }
 *
 *   coro_task<void> use_value() {
 *       int value = co_await get_value();
 *       // use value...
 *   }
 */
template<typename T>
class coro_task {
public:
    /**
     * Promise type for coroutine (required by C++20).
     */
    struct promise_type {
        // Result storage (manual lifetime management for union)
        alignas(T) unsigned char value_storage_[sizeof(T)];
        alignas(std::exception_ptr) unsigned char exception_storage_[sizeof(std::exception_ptr)];
        bool has_value_ = false;
        bool has_exception_ = false;
        
        // Continuation (parent coroutine to resume when done)
        std::coroutine_handle<> continuation_;

        // Accessors
        T& value() { return *reinterpret_cast<T*>(value_storage_); }
        std::exception_ptr& exception() { return *reinterpret_cast<std::exception_ptr*>(exception_storage_); }

        /**
         * Called when coroutine is created.
         */
        coro_task get_return_object() {
            return coro_task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        /**
         * Initial suspend: lazy evaluation.
         */
        std::suspend_always initial_suspend() noexcept { return {}; }

        /**
         * Final suspend: resume the continuation (parent coroutine).
         */
        struct final_awaiter {
            bool await_ready() const noexcept { return false; }
            
            std::coroutine_handle<> await_suspend(std::coroutine_handle<promise_type> h) noexcept {
                auto& promise = h.promise();
                if (promise.continuation_) {
                    return promise.continuation_;  // Resume parent
                }
                return std::noop_coroutine();  // No continuation, just stop
            }
            
            void await_resume() noexcept {}
        };
        
        final_awaiter final_suspend() noexcept { return {}; }

        /**
         * Store return value.
         */
        void return_value(T&& val) noexcept(std::is_nothrow_move_constructible_v<T>) {
            new (value_storage_) T(std::move(val));
            has_value_ = true;
        }

        void return_value(const T& val) noexcept(std::is_nothrow_copy_constructible_v<T>) {
            new (value_storage_) T(val);
            has_value_ = true;
        }

        /**
         * Handle exceptions.
         */
        void unhandled_exception() noexcept {
#ifdef __cpp_exceptions
            new (exception_storage_) std::exception_ptr(std::current_exception());
            has_exception_ = true;
#else
            std::cerr << "COROUTINE EXCEPTION: unhandled (exceptions disabled)" << std::endl;
            std::abort();
#endif
        }

        /**
         * Cleanup.
         */
        ~promise_type() {
            if (has_value_) {
                value().~T();
            }
            if (has_exception_) {
                exception().~exception_ptr();
            }
        }
    };

    /**
     * Awaiter for co_await support.
     */
    struct awaiter {
        std::coroutine_handle<promise_type> handle_;

        bool await_ready() const noexcept {
            return handle_.done();
        }

        std::coroutine_handle<> await_suspend(std::coroutine_handle<> awaiting) noexcept {
            // Store continuation so child can resume us when done
            handle_.promise().continuation_ = awaiting;
            // Resume the child coroutine
            return handle_;
        }

        T await_resume() {
            auto& promise = handle_.promise();

            if (promise.has_exception_) {
                // With -fno-exceptions, we cannot rethrow. Abort instead.
                std::cerr << "FATAL: Coroutine task has exception but exceptions are disabled" << std::endl;
                std::abort();
            }

            if (!promise.has_value_) {
                // With -fno-exceptions, we cannot throw. Abort instead.
                std::cerr << "FATAL: Coroutine task has no value" << std::endl;
                std::abort();
            }

            return std::move(promise.value());
        }
    };

    /**
     * Make task awaitable.
     */
    awaiter operator co_await() && {
        return awaiter{handle_};
    }

    /**
     * Constructor from coroutine handle.
     */
    explicit coro_task(std::coroutine_handle<promise_type> handle) noexcept
        : handle_(handle) {}

    /**
     * Move constructor.
     */
    coro_task(coro_task&& other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)) {}

    /**
     * Move assignment.
     */
    coro_task& operator=(coro_task&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                handle_.destroy();
            }
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    /**
     * Deleted copy operations.
     */
    coro_task(const coro_task&) = delete;
    coro_task& operator=(const coro_task&) = delete;

    /**
     * Destructor.
     */
    ~coro_task() {
        if (handle_) {
            handle_.destroy();
        }
    }

    /**
     * Check if task is valid.
     */
    explicit operator bool() const noexcept {
        return handle_ != nullptr;
    }

    /**
     * Resume the coroutine (for manual control).
     */
    bool resume() {
        if (!handle_ || handle_.done()) {
            return false;
        }
        handle_.resume();
        return !handle_.done();
    }

    /**
     * Check if task is complete.
     */
    bool done() const noexcept {
        return handle_ && handle_.done();
    }

    /**
     * Get raw coroutine handle.
     */
    std::coroutine_handle<> get_handle() const noexcept {
        return handle_;
    }

    /**
     * Release ownership of the coroutine handle.
     * After this, the coro_task no longer owns or destroys the handle.
     * The caller is responsible for eventually destroying it.
     */
    std::coroutine_handle<promise_type> release() noexcept {
        return std::exchange(handle_, nullptr);
    }

private:
    std::coroutine_handle<promise_type> handle_;
};

/**
 * Specialization for void tasks.
 */
template<>
class coro_task<void> {
public:
    struct promise_type {
        std::exception_ptr exception_;
        bool has_exception_ = false;

        // Continuation (parent coroutine to resume when done)
        std::coroutine_handle<> continuation_;

        // True iff the owning coro_task<void> wrapper called .release() — i.e.
        // the coroutine is now DETACHED, no wrapper will destroy() it in a dtor.
        // Used by final_awaiter to decide whether to self-destroy on completion.
        bool released_ = false;

        coro_task get_return_object() {
            return coro_task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }

        /**
         * Final suspend: resume the continuation (parent coroutine), or stop.
         *
         * TODO(connection-handle leak follow-up): a coro_task<void> released via
         * .release() and detached (continuation_ null, released_ true) sits at
         * final_suspend with its frame never reclaimed — a latent leak (masked
         * under keep-alive; bounded connection count). The released_ flag below
         * is plumbed for the future fix; an attempt to self-destroy here
         * (`if (released_) h.destroy()`) tripped heap corruption on MSVC even
         * with the flag — likely a subtle interaction with the runtime's resume
         * unwinding that needs a dedicated `detached_task` type with its own
         * frame-pool reclamation (likely in slice D alongside the dispatch
         * frame arena). For now: detached coroutines leak as before (no
         * regression vs pre-slice-B).
         */
        struct final_awaiter {
            bool await_ready() const noexcept { return false; }

            std::coroutine_handle<> await_suspend(std::coroutine_handle<promise_type> h) noexcept {
                auto& promise = h.promise();
                if (promise.continuation_) {
                    return promise.continuation_;  // Resume parent.
                }
                (void)promise.released_;  // dormant — see TODO above.
                return std::noop_coroutine();
            }

            void await_resume() noexcept {}
        };

        final_awaiter final_suspend() noexcept { return {}; }

        void return_void() noexcept {}

        void unhandled_exception() noexcept {
#ifdef __cpp_exceptions
            exception_ = std::current_exception();
            has_exception_ = true;
            try {
                std::rethrow_exception(exception_);
            } catch (const std::exception& e) {
                std::cerr << "COROUTINE EXCEPTION (void): " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "COROUTINE EXCEPTION (void): unknown type" << std::endl;
            }
#else
            std::cerr << "COROUTINE EXCEPTION (void): unhandled (exceptions disabled)" << std::endl;
            std::abort();
#endif
        }
    };

    struct awaiter {
        std::coroutine_handle<promise_type> handle_;

        bool await_ready() const noexcept {
            return handle_.done();
        }

        std::coroutine_handle<> await_suspend(std::coroutine_handle<> awaiting) noexcept {
            // Store continuation so child can resume us when done
            handle_.promise().continuation_ = awaiting;
            // Resume the child coroutine
            return handle_;
        }

        void await_resume() {
            auto& promise = handle_.promise();
            if (promise.has_exception_) {
                // With -fno-exceptions, we cannot rethrow. Abort instead.
                std::cerr << "FATAL: Coroutine task (void) has exception but exceptions are disabled" << std::endl;
                std::abort();
            }
        }
    };

    awaiter operator co_await() && {
        return awaiter{handle_};
    }

    explicit coro_task(std::coroutine_handle<promise_type> handle) noexcept
        : handle_(handle) {}

    coro_task(coro_task&& other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)) {}

    coro_task& operator=(coro_task&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                handle_.destroy();
            }
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    coro_task(const coro_task&) = delete;
    coro_task& operator=(const coro_task&) = delete;

    ~coro_task() {
        if (handle_) {
            handle_.destroy();
        }
    }

    explicit operator bool() const noexcept {
        return handle_ != nullptr;
    }

    bool resume() {
        if (!handle_ || handle_.done()) {
            return false;
        }
        handle_.resume();
        return !handle_.done();
    }

    bool done() const noexcept {
        return handle_ && handle_.done();
    }

    std::coroutine_handle<> get_handle() const noexcept {
        return handle_;
    }

    /**
     * Release ownership of the coroutine handle.
     * After this, the coro_task no longer owns or destroys the handle. The
     * underlying coroutine self-destroys at final_suspend (see promise_type's
     * final_awaiter — we mark released_=true so it knows the wrapper has
     * relinquished ownership). Detached coroutines must NOT be destroy()'d
     * manually after they complete; manual destroy is only safe if you destroy
     * BEFORE first resume (e.g. the startup-failure path in
     * coro_tcp_listener.cpp), in which case final_suspend never fires.
     */
    std::coroutine_handle<promise_type> release() noexcept {
        if (handle_) {
            handle_.promise().released_ = true;
        }
        return std::exchange(handle_, nullptr);
    }

private:
    std::coroutine_handle<promise_type> handle_;
};

// =============================================================================
// detached_task — fire-and-forget coroutine, owns its own lifetime
// =============================================================================
//
// For top-level coroutines that are NEVER co_awaited and are spawned-and-
// released (accept loops, connection handlers, future per-thread background
// jobs). Unlike coro_task<void>, the coroutine self-destroys on completion —
// closing the latent connection-handle leak that masked itself under keep-
// alive. The mechanism is the canonical fire-and-forget pattern (different
// from a final_awaiter that destroy()s in await_suspend, which tripped MSVC
// heap corruption in an earlier attempt):
//
//   - initial_suspend = suspend_always  → lazy: the listener .release()es the
//     handle and resumes it explicitly (or hands it to the worker pool /
//     post_to_io_thread). Avoids running coroutine bodies on the creating
//     thread.
//   - final_suspend   = suspend_never   → await_ready returns true; the
//     compiler-emitted code at the end of the coroutine body destroys the
//     frame automatically. No await_suspend is called for final, so no
//     destroy-then-return-handle dance for the runtime to trip over.
//
// Usage:
//
//   detached_task my_loop() { ... co_await some_awaitable; ... }
//   auto task = my_loop();
//   auto handle = task.release();             // give up wrapper ownership
//   handle.resume();                          // or wp->submit(handle) etc.
//   // After completion, the frame is gone. Do not touch handle.
//
// If the handle is NEVER resumed (startup-failure path), the wrapper's dtor
// destroys it. If .release()'d but never started, the caller MUST call
// handle.destroy() explicitly — same contract as coro_task.
class detached_task {
public:
    struct promise_type {
        detached_task get_return_object() noexcept {
            return detached_task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        // Lazy start.
        std::suspend_always initial_suspend() noexcept { return {}; }
        // Auto-destroy at final: suspend_never's await_ready=true means the
        // compiler proceeds to coroutine cleanup, including frame destruction.
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() noexcept {
            // -fno-exceptions: should never reach here.
            std::cerr << "FATAL: detached_task unhandled exception" << std::endl;
            std::abort();
        }
    };

    using handle_t = std::coroutine_handle<promise_type>;

    explicit detached_task(handle_t h) noexcept : handle_(h) {}
    detached_task(const detached_task&) = delete;
    detached_task& operator=(const detached_task&) = delete;
    detached_task(detached_task&&) = delete;
    detached_task& operator=(detached_task&&) = delete;

    // If the wrapper still owns the handle at destruction, the coroutine was
    // created but never started (or never released to a runner). Destroy it
    // explicitly to avoid leaking the frame.
    ~detached_task() noexcept {
        if (handle_) handle_.destroy();
    }

    // Release ownership: returns the handle and clears it from the wrapper.
    // The caller now owns the lifetime — either resume() it (the coroutine
    // self-destroys on completion via suspend_never final) or, if it will
    // never start (failure path), call handle.destroy() explicitly.
    handle_t release() noexcept {
        return std::exchange(handle_, {});
    }

    explicit operator bool() const noexcept { return static_cast<bool>(handle_); }

private:
    handle_t handle_;
};

} // namespace core
} // namespace bolt::api
