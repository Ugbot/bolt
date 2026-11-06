// bolt/obs/context.h — the per-thread request context logs and spans carry.
#pragma once

#include <cstdint>

#include "bolt/obs/obs_config.h"

namespace bolt::obs {

// W3C trace context. lo alone is the 64-bit id the gateway and WAL frames carry.
struct TraceCtx {
    std::uint64_t hi = 0;
    std::uint64_t lo = 0;
    std::uint64_t span = 0;
    std::uint8_t  sampled = 0;
};

#if BOLT_OBS_LOG || BOLT_OBS_TRACE
inline constinit thread_local TraceCtx t_trace_ctx{};
inline TraceCtx& trace_ctx() noexcept { return t_trace_ctx; }

// Installs a context on the thread that runs a request, restores on exit.
class ScopedTraceCtx {
public:
    explicit ScopedTraceCtx(const TraceCtx& ctx) noexcept : saved_(t_trace_ctx) { t_trace_ctx = ctx; }
    ~ScopedTraceCtx() { t_trace_ctx = saved_; }
    ScopedTraceCtx(const ScopedTraceCtx&) = delete;
    ScopedTraceCtx& operator=(const ScopedTraceCtx&) = delete;
private:
    TraceCtx saved_;
};
#else
class ScopedTraceCtx {
public:
    explicit ScopedTraceCtx(const TraceCtx&) noexcept {}
};
#endif

}  // namespace bolt::obs
