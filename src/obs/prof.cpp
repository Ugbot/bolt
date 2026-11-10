// prof.cpp — ITIMER_PROF sampling into a fixed array, symbolized after the run.
#include "bolt/obs/crash.h"

#if BOLT_OBS_PROFILE && !defined(_WIN32)

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <map>
#include <sys/time.h>
#include <thread>
#include <vector>

#include "bolt/obs/obs_limits.h"

namespace bolt::prof {

namespace {

constexpr int kFrames = 32;
constexpr int kSkipFrames = 2;   // handler + signal trampoline

struct Sample {
    void* pcs[kFrames];
    int n;
};

std::atomic<bool> g_busy{false};
Sample* g_samples = nullptr;
std::uint32_t g_cap = 0;
std::atomic<std::uint32_t> g_next{0};
std::atomic<std::uint64_t> g_dropped{0};

extern "C" void bolt_prof_handler(int) {
    const int saved = errno;
    const std::uint32_t i = g_next.fetch_add(1, std::memory_order_relaxed);
    if (i >= g_cap) { g_dropped.fetch_add(1, std::memory_order_relaxed); errno = saved; return; }
    g_samples[i].n = backtrace(g_samples[i].pcs, kFrames);
    errno = saved;
}

std::string symbol(void* pc) {
    Dl_info info;
    if (dladdr(pc, &info) == 0 || info.dli_sname == nullptr) {
        char b[24];
        std::snprintf(b, sizeof(b), "0x%llx", static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(pc)));
        return b;
    }
    int status = 0;
    char* d = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
    std::string out = status == 0 && d != nullptr ? d : info.dli_sname;
    std::free(d);
    std::replace(out.begin(), out.end(), ';', ':');
    std::replace(out.begin(), out.end(), ' ', '_');
    return out;
}

void fold(std::uint32_t n, Result* out) {
    std::map<std::string, std::uint64_t> stacks;
    std::map<void*, std::string> names;
    for (std::uint32_t i = 0; i < n; ++i) {
        const Sample& s = g_samples[i];
        std::string key;
        for (int f = s.n - 1; f >= kSkipFrames; --f) {
            auto it = names.find(s.pcs[f]);
            if (it == names.end()) it = names.emplace(s.pcs[f], symbol(s.pcs[f])).first;
            if (!key.empty()) key.push_back(';');
            key += it->second;
        }
        ++stacks[key];
    }
    for (const auto& [k, v] : stacks) out->folded.append(k).append(" ").append(std::to_string(v)).push_back('\n');
}

}  // namespace

bool run(std::uint32_t seconds, std::uint32_t hz, Result* out) noexcept {
    assert(out != nullptr);
    assert(seconds > 0 && hz > 0);
    bool expected = false;
    if (!g_busy.compare_exchange_strong(expected, true)) return false;
    void* warm[4];
    (void)backtrace(warm, 4);
    g_cap = static_cast<std::uint32_t>(obs::obs_limits_value(obs::obs_limits_id::obs_profile_samples));
    g_samples = static_cast<Sample*>(std::malloc(sizeof(Sample) * g_cap));
    if (g_samples == nullptr) { g_busy.store(false); return false; }
    g_next.store(0);
    g_dropped.store(0);
    struct sigaction sa{}, old{};
    sa.sa_handler = bolt_prof_handler;
    sa.sa_flags = SA_RESTART;
    sigemptyset(&sa.sa_mask);
    ::sigaction(SIGPROF, &sa, &old);
    itimerval timer{};
    timer.it_interval.tv_usec = static_cast<int>(1000000u / hz);
    timer.it_value = timer.it_interval;
    ::setitimer(ITIMER_PROF, &timer, nullptr);
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    itimerval off{};
    ::setitimer(ITIMER_PROF, &off, nullptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));   // a handler may still be in flight
    struct sigaction ignore{};
    ignore.sa_handler = SIG_IGN;
    ::sigaction(SIGPROF, &ignore, nullptr);
    const std::uint32_t n = std::min(g_next.load(), g_cap);
    fold(n, out);
    out->samples = n;
    out->dropped = g_dropped.load();
    ::sigaction(SIGPROF, &old, nullptr);
    std::free(g_samples);
    g_samples = nullptr;
    g_busy.store(false);
    return true;
}

}  // namespace bolt::prof

#elif BOLT_OBS_PROFILE

namespace bolt::prof {
bool run(std::uint32_t, std::uint32_t, Result*) noexcept { return false; }
}  // namespace bolt::prof

#endif
