#pragma once
#include <cstdint>

namespace bolt {
enum class ThreadQos : std::uint32_t { Inherit, Latency };
enum class ThreadQosStatus : std::uint32_t { Inherited, Configured, Unsupported, Failed, InvalidInput };
enum class ThreadQosPlatform : std::uint32_t { None, MacQos, WindowsPriority };
struct ThreadQosResult {
    ThreadQosStatus status=ThreadQosStatus::Inherited;
    ThreadQosPlatform platform=ThreadQosPlatform::None;
    std::int32_t native_requested=0,native_observed=0,native_relative_priority=0;
    std::uint32_t os_error=0;
};
static_assert(sizeof(ThreadQosResult)==24);
// Cold current-thread configuration, never a CPU pin or latency promise.
// Inherit performs no OS call and does not reset a prior explicit setting.
// Latency requests macOS USER_INTERACTIVE QoS/relative0, or Windows HIGHEST
// thread priority (not TIME_CRITICAL); other platforms return Unsupported.
// Configured means the setter succeeded and immediate configuration readback
// matched. The OS may still preempt, boost, clamp or migrate the thread.
// No process policy, realtime scheduler, CPU isolation or system setting changes.
// Caller owns this thread's policy through return; a successful request persists
// until changed or thread exit. Failure never silently falls back to inheritance;
// a setter success followed by readback failure may leave the setting changed.
ThreadQosResult thread_qos_apply_current(ThreadQos) noexcept;
} // namespace bolt
