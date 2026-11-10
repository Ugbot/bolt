// bolt/obs/crash.h — fatal-signal reports and an opt-in sampling profiler.
// Both are cold: nothing runs until a signal arrives or a run is requested.
#pragma once

#include <cstdint>
#include <string>

#include "bolt/obs/obs_config.h"

namespace bolt::crash {

struct Config {
    const char* dir = nullptr;       // reports land in <dir>/crash-<ts>-<pid>.txt; nullptr: stderr only
    const char* identity = "bolt";   // one line: service, version, build
};

#if BOLT_OBS_CRASH
bool install(const Config& config) noexcept;
void thread_init() noexcept;   // alternate signal stack, so a stack overflow can still report
void thread_fini() noexcept;
#else
inline bool install(const Config&) noexcept { return false; }
inline void thread_init() noexcept {}
inline void thread_fini() noexcept {}
#endif

}  // namespace bolt::crash

namespace bolt::prof {

struct Result {
    std::uint64_t samples = 0;
    std::uint64_t dropped = 0;
    std::string folded;   // "a;b;c 12\n", ready for flamegraph.pl
};

#if BOLT_OBS_PROFILE
// Blocks the caller for `seconds`. One run at a time: false when busy or unsupported.
bool run(std::uint32_t seconds, std::uint32_t hz, Result* out) noexcept;
#else
inline bool run(std::uint32_t, std::uint32_t, Result*) noexcept { return false; }
#endif

}  // namespace bolt::prof
