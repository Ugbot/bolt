#include "bolt/thread/bolt_thread_qos.h"
#include <gtest/gtest.h>
#include <limits>
#include <thread>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

namespace {
struct NativePolicy {
    std::int32_t value=0,relative=0;
    std::uint32_t error=0;
};
NativePolicy read_policy() noexcept {
    NativePolicy out{};
#if defined(_WIN32)
    const auto value=GetThreadPriority(GetCurrentThread());
    if(value==THREAD_PRIORITY_ERROR_RETURN)out.error=GetLastError();
    else out.value=value;
#elif defined(__APPLE__)
    qos_class_t value=QOS_CLASS_UNSPECIFIED;int relative=0;
    const int error=pthread_get_qos_class_np(pthread_self(),&value,&relative);
    out.error=static_cast<std::uint32_t>(error);
    if(error==0){out.value=static_cast<std::int32_t>(value);out.relative=relative;}
#endif
    return out;
}
void expect_empty(const bolt::ThreadQosResult& result,bolt::ThreadQosStatus status) {
    EXPECT_EQ(result.status,status);
    EXPECT_EQ(result.platform,bolt::ThreadQosPlatform::None);
    EXPECT_EQ(result.native_requested,0);EXPECT_EQ(result.native_observed,0);
    EXPECT_EQ(result.native_relative_priority,0);EXPECT_EQ(result.os_error,0u);
}
} // namespace

TEST(ThreadQos, InheritAndInvalidLeaveIsolatedThreadConfigurationUnchanged) {
    std::thread thread([] {
        const auto before=read_policy();
        expect_empty(bolt::thread_qos_apply_current(bolt::ThreadQos::Inherit),bolt::ThreadQosStatus::Inherited);
        constexpr std::uint32_t invalid[]={2u,std::numeric_limits<std::uint32_t>::max()};
        for(const auto value:invalid)
            expect_empty(bolt::thread_qos_apply_current(static_cast<bolt::ThreadQos>(value)),bolt::ThreadQosStatus::InvalidInput);
        const auto after=read_policy();
        EXPECT_EQ(after.error,before.error);EXPECT_EQ(after.value,before.value);EXPECT_EQ(after.relative,before.relative);
    });
    thread.join();
}

TEST(ThreadQos, LatencyReportsExactRequestedConfigurationOrTruthfulRefusal) {
    std::thread thread([] {
        const auto result=bolt::thread_qos_apply_current(bolt::ThreadQos::Latency);
#if defined(_WIN32) || defined(__APPLE__)
#if defined(_WIN32)
        EXPECT_EQ(result.platform,bolt::ThreadQosPlatform::WindowsPriority);
        EXPECT_EQ(result.native_requested,THREAD_PRIORITY_HIGHEST);
#else
        EXPECT_EQ(result.platform,bolt::ThreadQosPlatform::MacQos);
        EXPECT_EQ(result.native_requested,static_cast<std::int32_t>(QOS_CLASS_USER_INTERACTIVE));
#endif
        ASSERT_TRUE(result.status==bolt::ThreadQosStatus::Configured || result.status==bolt::ThreadQosStatus::Failed);
        if(result.status==bolt::ThreadQosStatus::Configured) {
            const auto native=read_policy();
            ASSERT_EQ(native.error,0u);
            EXPECT_EQ(result.os_error,0u);EXPECT_EQ(native.value,result.native_requested);
            EXPECT_EQ(result.native_observed,native.value);
            EXPECT_EQ(result.native_relative_priority,0);EXPECT_EQ(native.relative,0);
        } else if(result.os_error==0u) {
            // A successful readback can reject a configuration mismatch without
            // inventing an OS error. A native failure retains its actual code.
            EXPECT_TRUE(result.native_observed!=result.native_requested || result.native_relative_priority!=0);
        }
#else
        expect_empty(result,bolt::ThreadQosStatus::Unsupported);
#endif
        const auto before=read_policy();
        expect_empty(bolt::thread_qos_apply_current(bolt::ThreadQos::Inherit),bolt::ThreadQosStatus::Inherited);
        const auto after=read_policy();
        EXPECT_EQ(after.error,before.error);EXPECT_EQ(after.value,before.value);EXPECT_EQ(after.relative,before.relative);
    });
    thread.join();
}
