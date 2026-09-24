// test_bolt_scheduler_worker_stack.cpp — G2ICE-166.
//
// Scheduler workers must get a real stack regardless of how much static TLS
// the linking binary carries. On glibc a default-attr thread's stack is
// RLIMIT_STACK (8 MiB) with the static-TLS block carved out of it; once the
// TLS exceeds the rlimit, glibc clamps to its minimum and the thread keeps
// only a few KiB for call frames. chukonu's thread_local scratch tables put
// ~11.3 MiB in static TLS, so marbledb's maint workers died on the first
// flush task (a 4 KiB frame) — a SIGSEGV in the guard page that gdb reported
// as flush_zone_write_open(db=0x0, t=0x0, z=0x0).
//
// This binary reproduces that shape: a 12 MiB thread_local pushes static TLS
// past the default 8 MiB rlimit, and each task touches 256 KiB of stack. With
// std::thread workers it SIGSEGVs on Linux; with explicitly-sized workers it
// passes. (macOS allocates TLS off-stack, so there the test cannot fail.)

#include "bolt/bolt_scheduler.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace {

constexpr std::size_t kBigTlsBytes = 12u * 1024u * 1024u;
thread_local unsigned char g_big_tls[kBigTlsBytes];

constexpr std::size_t kTaskStackBytes = 256u * 1024u;
constexpr uint32_t    kTasks = 64u;

std::atomic<uint32_t> g_done{0u};
std::atomic<uint64_t> g_sum{0u};

BOLT_NOINLINE uint64_t touch_stack(uint32_t seed) noexcept {
    volatile unsigned char buf[kTaskStackBytes];
    for (std::size_t i = 0; i < kTaskStackBytes; i += 512u) {
        buf[i] = static_cast<unsigned char>(seed + i);
    }
    uint64_t s = 0u;
    for (std::size_t i = 0; i < kTaskStackBytes; i += 512u) s += buf[i];
    return s;
}

void task(void* arg) noexcept {
    const auto seed = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(arg));
    g_big_tls[seed % kBigTlsBytes] = static_cast<unsigned char>(seed);
    g_sum.fetch_add(touch_stack(seed), std::memory_order_relaxed);
    g_done.fetch_add(1u, std::memory_order_release);
}

// glibc sizes default thread stacks from RLIMIT_STACK read at process start,
// so pin it to the common 8 MiB and re-exec; otherwise an unlimited or large
// rlimit in the test environment would hide the defect.
void pin_stack_rlimit(char** argv) noexcept {
#if defined(__linux__)
    constexpr rlim_t kPinned = 8u * 1024u * 1024u;
    struct rlimit rl{};
    if (getrlimit(RLIMIT_STACK, &rl) != 0 || rl.rlim_cur == kPinned) return;
    if (rl.rlim_max != RLIM_INFINITY && rl.rlim_max < kPinned) return;
    rl.rlim_cur = kPinned;
    if (setrlimit(RLIMIT_STACK, &rl) != 0) return;
    execv("/proc/self/exe", argv);
#else
    (void)argv;
#endif
}

}  // namespace

int main(int /*argc*/, char** argv) {
    pin_stack_rlimit(argv);
    g_big_tls[0] = 1u;  // keep the TLS block referenced
    static bolt::Scheduler sched;
    bolt::SchedulerConfig cfg{};
    cfg.num_workers = 4u;
    if (!sched.init(cfg)) {
        std::fprintf(stderr, "[FAIL] Scheduler::init\n");
        return 1;
    }
    for (uint32_t i = 0; i < kTasks; ++i) {
        sched.submit(&task, reinterpret_cast<void*>(uintptr_t{i + 1u}));
    }
    for (uint64_t spin = 0; g_done.load(std::memory_order_acquire) < kTasks;
         ++spin) {
        if (spin > (1ull << 32)) {
            std::fprintf(stderr, "[FAIL] tasks did not finish\n");
            return 1;
        }
        std::this_thread::yield();
    }
    sched.shutdown();
    if (g_done.load() != kTasks || g_sum.load() == 0u) {
        std::fprintf(stderr, "[FAIL] done=%u\n", g_done.load());
        return 1;
    }
    std::printf("test_bolt_scheduler_worker_stack: OK (%u tasks, %zu KiB stack each, "
                "%zu MiB static TLS)\n", kTasks, kTaskStackBytes / 1024u,
                kBigTlsBytes >> 20);
    return 0;
}
