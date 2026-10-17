// Encoding admission and actual public refusal; never pin to a valid CPU.
#include "bolt/bolt_port.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <limits>

TEST(ThreadAffinity, RepresentabilityChecksEncodingRatherThanCpuCount) {
    EXPECT_EQ(kBoltThreadAffinityCpuCapacity,4096u);
#if defined(_WIN32) || defined(__APPLE__)
    // Pure checks may include offline CPUs; no OS affinity call is made.
    EXPECT_TRUE(bolt_thread_affinity_cpu_representable(0u));
    EXPECT_TRUE(bolt_thread_affinity_cpu_representable(63u));
    EXPECT_TRUE(bolt_thread_affinity_cpu_representable(64u));
    EXPECT_TRUE(bolt_thread_affinity_cpu_representable(4095u));
#elif defined(__linux__)
    constexpr auto set_capacity=static_cast<std::uint32_t>(CPU_SETSIZE);
    constexpr auto limit=set_capacity<4096u?set_capacity:4096u;
    static_assert(limit>0u);
    EXPECT_TRUE(bolt_thread_affinity_cpu_representable(0u));
    EXPECT_TRUE(bolt_thread_affinity_cpu_representable(limit-1u));
    EXPECT_FALSE(bolt_thread_affinity_cpu_representable(limit));
    EXPECT_FALSE(bolt_thread_affinity_cpu_representable(set_capacity));
#else
    EXPECT_FALSE(bolt_thread_affinity_cpu_representable(0u));
    EXPECT_FALSE(bolt_thread_affinity_cpu_representable(1u));
    EXPECT_FALSE(bolt_thread_affinity_cpu_representable(4095u));
#endif
    EXPECT_FALSE(bolt_thread_affinity_cpu_representable(4096u));
    EXPECT_FALSE(bolt_thread_affinity_cpu_representable(4097u));
    EXPECT_FALSE(bolt_thread_affinity_cpu_representable(std::numeric_limits<std::uint32_t>::max()));
}

TEST(ThreadAffinity, InvalidCpuReturnsFalseWithoutAssertingOrChangingAffinity) {
    // All IDs here are unrepresentable: this exercises the public wrapper but
    // reaches neither a valid OS affinity request nor a macOS affinity hint.
    constexpr std::uint32_t invalid[]={4096u,4097u,std::numeric_limits<std::uint32_t>::max()};
    for(const auto cpu:invalid) {
        SCOPED_TRACE(cpu);
        EXPECT_FALSE(bolt_pin_current_thread(cpu));
    }
#if defined(__linux__)
    // Catch the platform's narrower encoding boundary as well as Bolt's cap.
    EXPECT_FALSE(bolt_pin_current_thread(static_cast<std::uint32_t>(CPU_SETSIZE)));
#elif !defined(_WIN32) && !defined(__APPLE__)
    EXPECT_FALSE(bolt_pin_current_thread(0u));
#endif
}
