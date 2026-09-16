// StackedThread — a joinable OS thread created with an EXPLICIT stack size.
//
// WHY THIS EXISTS (G2ICE-111 / G2ICE-112): every thread that resumes a
// boltapi request-handling coroutine — the WorkerThreadPool's main and
// blocking workers, and (with IODispatcherConfig::inline_resume, the
// default) the IODispatcher's own I/O threads — was created via a bare
// `std::thread`, which offers NO portable way to request a stack size and
// therefore silently inherits the platform default pthread stack: 512 KiB
// on macOS, commonly 1-2 MiB on Linux distros, never explicitly raised
// anywhere in this codebase. Deeply recursive canonical-plan-building code
// in chukonu holds NodeBindings/ColBinding-shaped locals (grown ~4x by a
// column-count cap raise) as ordinary stack frames; on the default stack
// that recursion exhausts the guard page and takes the whole process down
// — not a per-request failure, a full crash of every connection gestaltd is
// serving. See docs/PUNCHCARD.md and tracker G2ICE-111/G2ICE-112 in the
// Gestalt2 superproject for the full incident writeup.
//
// Raising the stack size on every thread that can resume handler code closes
// the whole CLASS of "a sufficiently deep/wide recursive call chain
// overflows the request thread's stack" bugs at once — including ones no one
// has audited yet — rather than requiring a function-by-function fix for
// every recursive function that happens to hold a large local. It is not a
// substitute for bounding genuinely unbounded recursion (a stack of any
// finite size can still be exhausted by an adversarial-enough input), but it
// converts "512 KiB, exhausted by an ordinary two-way JOIN" into "8 MiB,
// which comfortably fits the current worst-known call depth with real
// headroom" — the same order of magnitude as a Linux pthread default and
// larger than the 512 KiB-1 MiB range that made this bug reachable at all.
//
// std::thread has no constructor parameter for stack size on ANY platform —
// the platform API (pthread_attr_setstacksize / the dwStackSize argument to
// _beginthreadex) has to be called directly, so this class exists instead of
// a std::thread. It intentionally mirrors just the slice of std::thread's
// interface the two call sites in this library need (move-only,
// joinable()/join(), constructible from a callable + its bound arguments) so
// it drops into `std::vector<std::thread>` call sites with a one-line type
// change.
#pragma once

#include <cassert>
#include <cstddef>
#include <functional>
#include <tuple>
#include <utility>

#if defined(_WIN32)
#include <process.h>  // _beginthreadex
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace bolt::api::core {

/// Default stack size for boltapi request-handling threads: 32 MiB.
///
/// G2CHK-81 (2026-09-16): was 8 MiB. A binary that links chukonu with
/// CHUKONU_WITH_GRAPH_FRONTENDS=ON (the vendored ANTLR4 C++ runtime for
/// the openCypher/SPARQL frontends — default-on since G2GRAPH-1, the
/// same day this was found) carries a STATIC thread-local-storage block
/// of ~14.5 MiB (measured: gestaltd's `.tbss` section is 0xe82aac =
/// 15,225,004 bytes). glibc's NPTL reserves that static-TLS block out of
/// every new thread's stack allocation before a single byte is available
/// for real call frames — for a thread with NO explicit stacksize this
/// silently grows the default allocation to fit (observed: ~13.8 MiB
/// threads under an 8 MiB RLIMIT_STACK), but for a thread created with an
/// EXPLICIT `pthread_attr_setstacksize` smaller than the required TLS
/// floor, `pthread_create` flatly refuses with EINVAL — proven live via
/// gdb: `pthread_attr_setstacksize(&attr, 8*1024*1024)` succeeds (glibc
/// doesn't validate against the running process's actual TLS need at
/// that call), but the following `pthread_create` returns errno 22
/// (EINVAL) every time, and manually forcing 32 MiB via gdb made the
/// identical call succeed. So the *previous* 8 MiB comment ("matches the
/// typical Linux pthread default... trivial for any real deployment")
/// stopped being true the moment ANTLR4 got linked into this binary by
/// default — it wasn't a documentation error when written, it was
/// invalidated by a build-graph change elsewhere. 32 MiB clears the
/// current ~14.5 MiB floor with ~17 MiB of real headroom to spare, and
/// stays trivial in absolute terms: at the default thread counts (1 IO
/// thread + 8 workers + up to 8 blocking workers), 17 * 32 MiB = 544 MiB
/// total — still negligible for any real deployment, and every other
/// StackedThread call site (marbledb's compaction/commit/ttl-reaper
/// background threads, see marbledb G2CHK-81) rides this same constant so
/// a future TLS-floor increase only needs fixing in one place.
inline constexpr std::size_t kDefaultStackBytes = 32u * 1024u * 1024u;

class StackedThread {
public:
    StackedThread() noexcept = default;

    /// Spawns a new OS thread running `fn(args...)` with an explicitly
    /// requested stack of `stack_bytes`. `fn`/`args` are decay-copied into a
    /// heap-allocated closure (mirrors std::thread's by-value capture
    /// semantics) that the spawned thread frees itself just before
    /// returning.
    template <typename Fn, typename... Args>
    StackedThread(std::size_t stack_bytes, Fn&& fn, Args&&... args) {
        assert(stack_bytes > 0);
        auto* closure = new std::function<void()>(
            [f = std::forward<Fn>(fn),
             tup = std::make_tuple(std::forward<Args>(args)...)]() mutable {
                std::apply(f, std::move(tup));
            });
        spawn(stack_bytes, closure);
        assert(joinable());
    }

    ~StackedThread() {
        // A still-joinable StackedThread at destruction is a caller bug (the
        // owning pool must join() before the vector holding this element is
        // cleared) — matches std::thread's std::terminate-on-destroy-while-
        // joinable contract, but as a debug assertion rather than a
        // terminate, consistent with this repo's "asserts are for
        // programmer errors" convention.
        assert(!joinable());
    }

    StackedThread(const StackedThread&) = delete;
    StackedThread& operator=(const StackedThread&) = delete;

    StackedThread(StackedThread&& other) noexcept
        : handle_(other.handle_), has_handle_(other.has_handle_) {
        other.has_handle_ = false;
#if defined(_WIN32)
        other.handle_ = nullptr;
#endif
    }

    StackedThread& operator=(StackedThread&& other) noexcept {
        assert(!joinable());  // never silently leak a still-joinable thread
        if (this != &other) {
            handle_ = other.handle_;
            has_handle_ = other.has_handle_;
            other.has_handle_ = false;
#if defined(_WIN32)
            other.handle_ = nullptr;
#endif
        }
        return *this;
    }

    bool joinable() const noexcept { return has_handle_; }

    void join() noexcept {
        assert(joinable());
#if defined(_WIN32)
        WaitForSingleObject(handle_, INFINITE);
        CloseHandle(handle_);
        handle_ = nullptr;
#else
        pthread_join(handle_, nullptr);
#endif
        has_handle_ = false;
        assert(!joinable());
    }

private:
    void spawn(std::size_t stack_bytes, std::function<void()>* closure);

#if defined(_WIN32)
    static unsigned __stdcall trampoline(void* arg) {
        auto* closure = static_cast<std::function<void()>*>(arg);
        (*closure)();
        delete closure;
        return 0;
    }
    HANDLE handle_ = nullptr;
#else
    static void* trampoline(void* arg) {
        auto* closure = static_cast<std::function<void()>*>(arg);
        (*closure)();
        delete closure;
        return nullptr;
    }
    pthread_t handle_{};
#endif
    bool has_handle_ = false;
};

inline void StackedThread::spawn(std::size_t stack_bytes,
                                  std::function<void()>* closure) {
    assert(!has_handle_);
#if defined(_WIN32)
    // _beginthreadex's stack_size sets the COMMIT size of the new thread's
    // stack (matching pthread_attr_setstacksize's semantics below); the OS
    // reserves address space beyond it on demand up to the executable's
    // configured reserve, so this is safe to raise without a matching
    // reserve-side change.
    uintptr_t h = _beginthreadex(nullptr, static_cast<unsigned>(stack_bytes),
                                  &trampoline, closure, 0, nullptr);
    assert(h != 0);
    handle_ = reinterpret_cast<HANDLE>(h);
    has_handle_ = (h != 0);
#else
    pthread_attr_t attr;
    int rc = pthread_attr_init(&attr);
    assert(rc == 0);
    rc = pthread_attr_setstacksize(&attr, stack_bytes);
    assert(rc == 0);
    rc = pthread_create(&handle_, &attr, &trampoline, closure);
    pthread_attr_destroy(&attr);
    assert(rc == 0);
    has_handle_ = (rc == 0);
#endif
}

}  // namespace bolt::api::core
