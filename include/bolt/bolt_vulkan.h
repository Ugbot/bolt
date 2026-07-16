#pragma once
// bolt_vulkan.h -- BLLM-46: Vulkan device/queue/memory scaffolding behind
// bolt::compute's reserved Vulkan device slot (see bolt_compute.h's
// `Device` enum and its `vulkan_table` hard-failing stubs -- BLLM-13).
//
// Scope: instance + physical/logical device + one dedicated compute queue
// + a basic buffer allocator. NOT compute kernels/shaders (BLLM-47's job,
// "adapt dequant+GEMM shaders") -- this is the plumbing those kernels will
// dispatch through once they exist. bolt_compute.h's vulkan_table entries
// still point at device_not_implemented() until BLLM-47 lands; this module
// is additive and does not change that dispatch yet.
//
// Runtime-detected, CPU fallback always works: `available()` probes for a
// real Vulkan-capable device and returns false (never asserts/aborts) if
// none exists -- no SDK installed, no driver, or a driver that exposes no
// compute-capable queue family. `Context::create()` follows the same
// contract: false on ANY failure, leaving the Context inert. Callers MUST
// check these return values and fall back to the CPU path in
// bolt::compute's dispatch table if either fails -- this module never
// silently degrades to a wrong answer, it only ever cleanly fails to be
// available (mirroring bolt_compute.h's own "never silently falling back
// to CPU FROM a dispatched Vulkan call" contract, one layer up: THIS layer
// is what lets a caller decide whether to dispatch to Vulkan at all).
//
// Compiled (not header-only): unlike bolt_compute.h's CPU-scalar kernels,
// Vulkan's C API + loader linkage doesn't suit a header-only inline
// function -- same rationale as bolt::parse (bolt_jinja.h's own file
// header), which is why this is bolt::vulkan, a separate optional compiled
// library gated on CMake finding the Vulkan SDK (BOLT_BUILD_VULKAN),
// rather than folded into bolt_compute.h itself.
//
// Handles are opaque (void*) in this header so callers that never touch
// Vulkan directly don't need <vulkan/vulkan.h> in their own translation
// units -- only bolt_vulkan.cpp includes it.

#include <cstdint>

namespace bolt::vulkan {

// True iff a Vulkan instance can be created AND at least one physical
// device exposes a queue family supporting compute. Creates and
// immediately destroys a throwaway instance -- intended for one-time
// startup capability detection, not a hot-path check.
bool available() noexcept;

// One allocated GPU-visible buffer. `buffer`/`memory` are opaque
// (VkBuffer/VkDeviceMemory) handles -- only bolt_vulkan.cpp's
// implementation interprets them.
struct BufferHandle {
    void*    buffer = nullptr;
    void*    memory = nullptr;
    uint64_t size_bytes = 0;
    bool     host_visible = false;  // true iff map_buffer() will work on this handle
};

// Owns one Vulkan instance + physical/logical device + one compute queue.
// Tiger-Style single-owner: not copyable/movable (owns live Vulkan
// handles with real destruction-order requirements).
class Context {
public:
    Context() noexcept = default;
    ~Context() noexcept;

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    // Creates an instance, picks the first physical device exposing a
    // compute-capable queue family (preferring a discrete GPU over an
    // integrated one if more than one device qualifies), then creates a
    // logical device + that one queue. Returns false (leaving `this`
    // inert -- `is_valid()` stays false) on ANY failure: no Vulkan SDK/
    // driver, no compute-capable device, or the driver rejecting device
    // creation. May be called at most once per instance.
    bool create() noexcept;

    // Destroys the device/instance if valid; safe to call on an inert
    // (create() never called, or create() returned false) Context too.
    void destroy() noexcept;

    bool is_valid() const noexcept { return device_ != nullptr; }

    // Valid only if is_valid() -- a short driver-reported device name and
    // the size of the largest DEVICE_LOCAL memory heap, respectively.
    const char* device_name() const noexcept { return device_name_; }
    uint64_t device_local_heap_bytes() const noexcept { return device_local_heap_bytes_; }

    // Allocates a buffer of `size_bytes`. `prefer_host_visible` selects a
    // HOST_VISIBLE|HOST_COHERENT memory type when available (needed for
    // map_buffer() -- useful for staging/upload buffers a CPU writes
    // directly) over a DEVICE_LOCAL-only type; on this box's INTEGRATED
    // AMD GPU these are typically the same heap, so the distinction
    // mostly matters for future discrete-GPU support, where DEVICE_LOCAL-
    // only memory is faster for the GPU to read but not CPU-mappable.
    // Returns false (leaving `*out` default-constructed) on any failure.
    // Precondition: is_valid().
    bool allocate_buffer(uint64_t size_bytes, bool prefer_host_visible,
                        BufferHandle* out) noexcept;

    // Frees a handle allocate_buffer() returned; safe on a
    // default-constructed (never-allocated) handle. Zeroes `*handle`.
    void free_buffer(BufferHandle* handle) noexcept;

    // Maps a host-visible buffer for direct CPU read/write. Returns
    // nullptr if `handle.host_visible` is false or the map call itself
    // fails. Precondition: is_valid().
    void* map_buffer(const BufferHandle& handle) noexcept;

    // Unmaps a buffer map_buffer() returned non-null for. Precondition:
    // is_valid().
    void unmap_buffer(const BufferHandle& handle) noexcept;

private:
    void*    instance_ = nullptr;         // VkInstance
    void*    physical_device_ = nullptr;  // VkPhysicalDevice
    void*    device_ = nullptr;           // VkDevice
    void*    queue_ = nullptr;            // VkQueue
    uint32_t queue_family_index_ = 0;
    char     device_name_[256] = {};
    uint64_t device_local_heap_bytes_ = 0;
};

}  // namespace bolt::vulkan
