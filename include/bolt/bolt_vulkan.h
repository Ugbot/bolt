#pragma once
// bolt_vulkan.h -- BLLM-46: Vulkan device/queue/memory scaffolding behind
// bolt::compute's reserved Vulkan device slot (see bolt_compute.h's
// `Device` enum and its `vulkan_table` hard-failing stubs -- BLLM-13).
//
// Scope: instance + physical/logical device + one dedicated compute queue
// + a basic buffer allocator, PLUS (BLLM-47) the compiled matmul+dequant
// compute pipeline itself -- `Context::create_matmul_pipeline` /
// `matmul_dequant` below. bolt_compute.h's vulkan_table entries still
// point at device_not_implemented() (wiring this into that dispatch table
// is a separate follow-up, same seam bolt::rocm's matmul_dequant left
// open -- see bolt_compute.h's SEAM note on matmul_rocm_stub); this module
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

// Weight dtype tag for matmul_dequant (BLLM-47) -- mirrors
// bolt::rocm::WeightDType and the subset of bolt::DType (bolt_tensor.h)
// that bolt_compute.h's matmul_row_dot_f32 dequantizes today. Restated by
// value (not #include bolt_tensor.h) for the same reason bolt_rocm.h
// restates it: callers that only want the Vulkan kernel shouldn't have to
// link bolt::core's Tensor type. F16/BF16 are intentionally absent.
enum class WeightDType : uint8_t {
    F32  = 0,
    Int8 = 1,
    Int4 = 2,
    Int2 = 3,
};

// One compiled compute pipeline for a specific WeightDType's matmul+dequant
// shader (BLLM-47). Owns a shader module, descriptor set layout, pipeline
// layout, pipeline, a one-set descriptor pool + the set itself, and a
// dedicated command pool + command buffer -- everything matmul_dequant()
// needs to record and submit a dispatch without allocating per-call.
// Opaque handles; only bolt_vulkan.cpp interprets them. Tiger-Style
// single-owner: default-constructible (inert), destroyed via
// Context::destroy_matmul_pipeline (this struct has no destructor of its
// own -- ownership is explicit, matching this module's Context contract).
struct MatmulPipeline {
    void*       shader_module = nullptr;         // VkShaderModule
    void*       descriptor_set_layout = nullptr;  // VkDescriptorSetLayout
    void*       pipeline_layout = nullptr;        // VkPipelineLayout
    void*       pipeline = nullptr;               // VkPipeline
    void*       descriptor_pool = nullptr;        // VkDescriptorPool
    void*       descriptor_set = nullptr;         // VkDescriptorSet
    void*       command_pool = nullptr;           // VkCommandPool
    void*       command_buffer = nullptr;         // VkCommandBuffer
    WeightDType dtype = WeightDType::F32;

    bool is_valid() const noexcept { return pipeline != nullptr; }
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

    // ---- BLLM-47: matmul+dequant compute pipeline -----------------------

    // Creates the compiled compute pipeline for one WeightDType's
    // matmul+dequant shader (embedded SPIR-V -- see shaders/matmul_dequant_
    // {f32,int8,int4,int2}.comp and bolt_vulkan.cpp's kShaderFor()).
    // Returns false (leaving `*out` default-constructed) on any failure.
    // Precondition: is_valid(). May be called more than once (one pipeline
    // per dtype needed); each call owns independent resources, freed by
    // destroy_matmul_pipeline().
    bool create_matmul_pipeline(WeightDType dtype, MatmulPipeline* out) noexcept;

    // Destroys a pipeline create_matmul_pipeline() returned; safe on a
    // default-constructed (never-created) pipeline. Zeroes `*pipeline`.
    void destroy_matmul_pipeline(MatmulPipeline* pipeline) noexcept;

    // Dispatches the GEMV: out[r] = dot(weight.row(r), activation), r in
    // [0, out_rows), with on-the-fly dequant for dtype != F32 -- same
    // convention (kInt4/kInt2 offsets, per-row `scale`) as
    // bolt_compute.h's matmul_row_dot_f32 CPU reference and
    // bolt::rocm::matmul_dequant. All buffer args are Vulkan-resident
    // BufferHandles from allocate_buffer(); `scale` is ignored (may be a
    // default-constructed handle) when `pipeline.dtype() ==
    // WeightDType::F32`. Blocks until the dispatch completes
    // (vkQueueWaitIdle) -- not a fire-and-forget async dispatch.
    // Precondition: is_valid(), and `pipeline` was created with the same
    // dtype these buffers were packed for.
    bool matmul_dequant(const MatmulPipeline& pipeline, const BufferHandle& weight,
                        const BufferHandle& scale, const BufferHandle& activation,
                        const BufferHandle& out, int32_t out_rows, int32_t cols) noexcept;

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
