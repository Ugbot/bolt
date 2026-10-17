#include "bolt/thread/bolt_thread_qos.h"
#include <cassert>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

namespace bolt {
namespace {
ThreadQosResult latency() noexcept {
    static_assert(sizeof(ThreadQosResult)==24);
    ThreadQosResult result{};
    result.status=ThreadQosStatus::Failed;
#if defined(_WIN32)
    result.platform=ThreadQosPlatform::WindowsPriority;
    result.native_requested=THREAD_PRIORITY_HIGHEST;
    const auto thread=GetCurrentThread();
    if(!SetThreadPriority(thread,THREAD_PRIORITY_HIGHEST)) {
        result.os_error=GetLastError();
        return result;
    }
    const auto observed=GetThreadPriority(thread);
    if(observed==THREAD_PRIORITY_ERROR_RETURN) {
        result.os_error=GetLastError();
        return result;
    }
    result.native_observed=observed;
    if(observed==THREAD_PRIORITY_HIGHEST)result.status=ThreadQosStatus::Configured;
#elif defined(__APPLE__)
    result.platform=ThreadQosPlatform::MacQos;
    result.native_requested=static_cast<std::int32_t>(QOS_CLASS_USER_INTERACTIVE);
    const int set_error=pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
    if(set_error!=0) {
        result.os_error=static_cast<std::uint32_t>(set_error);
        return result;
    }
    qos_class_t observed=QOS_CLASS_UNSPECIFIED;
    int relative=0;
    const int get_error=pthread_get_qos_class_np(pthread_self(),&observed,&relative);
    if(get_error!=0) {
        result.os_error=static_cast<std::uint32_t>(get_error);
        return result;
    }
    result.native_observed=static_cast<std::int32_t>(observed);
    result.native_relative_priority=relative;
    if(observed==QOS_CLASS_USER_INTERACTIVE && relative==0)result.status=ThreadQosStatus::Configured;
#else
    result.status=ThreadQosStatus::Unsupported;
#endif
    assert(result.status!=ThreadQosStatus::Inherited);
    assert(result.status!=ThreadQosStatus::InvalidInput);
    return result;
}
} // namespace
ThreadQosResult thread_qos_apply_current(ThreadQos policy) noexcept {
    static_assert(sizeof(ThreadQos)==sizeof(std::uint32_t));
    static_assert(sizeof(ThreadQosStatus)==sizeof(std::uint32_t));
    if(policy==ThreadQos::Inherit)return {};
    if(policy==ThreadQos::Latency)return latency();
    ThreadQosResult result{};
    result.status=ThreadQosStatus::InvalidInput;
    return result;
}
} // namespace bolt
