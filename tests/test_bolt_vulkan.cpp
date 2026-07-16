// test_bolt_vulkan.cpp -- coverage for bolt::vulkan's device/queue/memory
// scaffolding (BLLM-46). Runs against whatever REAL Vulkan device this
// machine has (this box: AMD Radeon 8060S integrated GPU) -- self-skips
// (not a hard failure) if no Vulkan-capable device is present, matching
// this module's own "runtime-detected, CPU fallback always works"
// contract. STL is allowed in tests; production code stays Tiger-Style.

#include <gtest/gtest.h>

#include <cstring>

#include "bolt/bolt_vulkan.h"

using bolt::vulkan::BufferHandle;
using bolt::vulkan::Context;

TEST(BoltVulkan, AvailableDoesNotCrashRegardlessOfHardware) {
    // Must be callable (and return SOME bool) whether or not a device
    // exists -- this is the one assertion that should hold on ANY machine,
    // Vulkan-capable or not.
    const bool result = bolt::vulkan::available();
    EXPECT_TRUE(result == true || result == false);  // trivially true; documents intent
}

TEST(BoltVulkan, ContextCreateSucceedsOnThisRealDevice) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    EXPECT_TRUE(ctx.is_valid());
    EXPECT_GT(std::strlen(ctx.device_name()), 0u);
    EXPECT_GT(ctx.device_local_heap_bytes(), 0u);
    ctx.destroy();
    EXPECT_FALSE(ctx.is_valid());
}

TEST(BoltVulkan, DestroyIsSafeOnNeverCreatedContext) {
    Context ctx;
    ctx.destroy();  // must not crash
    EXPECT_FALSE(ctx.is_valid());
}

TEST(BoltVulkan, DestructorRunsCleanlyWithoutExplicitDestroy) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    {
        Context ctx;
        ASSERT_TRUE(ctx.create());
    }  // ~Context() must clean up without leaking/crashing
    SUCCEED();
}

TEST(BoltVulkan, AllocateAndMapHostVisibleBuffer) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());

    BufferHandle handle;
    ASSERT_TRUE(ctx.allocate_buffer(4096, /*prefer_host_visible=*/true, &handle));
    EXPECT_TRUE(handle.host_visible);
    EXPECT_EQ(handle.size_bytes, 4096u);

    void* ptr = ctx.map_buffer(handle);
    ASSERT_NE(ptr, nullptr);
    std::memset(ptr, 0xAB, 4096);
    // Re-read through the same mapping -- proves the map is a real,
    // writable view over the allocation, not a stub returning garbage.
    EXPECT_EQ(static_cast<unsigned char*>(ptr)[0], 0xAB);
    EXPECT_EQ(static_cast<unsigned char*>(ptr)[4095], 0xAB);
    ctx.unmap_buffer(handle);

    ctx.free_buffer(&handle);
    EXPECT_EQ(handle.buffer, nullptr);
    EXPECT_EQ(handle.memory, nullptr);
}

TEST(BoltVulkan, AllocateDeviceLocalBufferSucceeds) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());

    BufferHandle handle;
    // On this box's integrated GPU, DEVICE_LOCAL-preferring allocation may
    // still land on a host-visible-capable heap (UMA) -- the point of this
    // test is just that allocation succeeds and produces a usable handle,
    // not that host_visible is false.
    ASSERT_TRUE(ctx.allocate_buffer(1024 * 1024, /*prefer_host_visible=*/false, &handle));
    EXPECT_NE(handle.buffer, nullptr);
    EXPECT_NE(handle.memory, nullptr);
    ctx.free_buffer(&handle);
}

TEST(BoltVulkan, FreeBufferIsSafeOnDefaultConstructedHandle) {
    if (!bolt::vulkan::available()) {
        GTEST_SKIP() << "No Vulkan-capable device on this machine";
    }
    Context ctx;
    ASSERT_TRUE(ctx.create());
    BufferHandle handle;  // never allocated
    ctx.free_buffer(&handle);  // must not crash
    SUCCEED();
}

TEST(BoltVulkan, AllocateBufferFailsCleanlyOnInvalidContext) {
    Context ctx;  // create() never called
    BufferHandle handle;
    EXPECT_FALSE(ctx.allocate_buffer(1024, true, &handle));
    EXPECT_EQ(handle.buffer, nullptr);
}
