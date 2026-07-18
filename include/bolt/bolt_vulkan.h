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
    // BLLM-83: true iff this allocation ALSO landed on a DEVICE_LOCAL heap
    // (real on UMA/integrated GPUs -- confirmed on this box's AMD Radeon
    // 8060S, which exposes memory types that are simultaneously
    // HOST_VISIBLE|HOST_COHERENT|DEVICE_LOCAL). When true and host_visible
    // is also true, writes via map_buffer() land directly in
    // GPU-fast-path memory -- no separate staging buffer + device-local
    // copy step is needed, matching ds4_metal.m's MTLResourceStorageModeShared
    // zero-copy pattern one layer down (Vulkan's equivalent is memory-type
    // selection, not a separate storage-mode API).
    bool     device_local = false;
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

    // ---- BLLM-176: on-device RMSNorm (first non-GEMV op of the resident-graph
    // forward, BLLM-159). out[i] = x[i]/sqrt(mean(x^2)+eps)*weight[i]. All args
    // are Vulkan-resident BufferHandles; one workgroup normalizes the whole
    // n-wide vector. Blocks until complete (vkQueueWaitIdle). ----
    bool create_rmsnorm_pipeline(MatmulPipeline* out) noexcept;
    bool rmsnorm(const MatmulPipeline& pipeline, const BufferHandle& x, const BufferHandle& weight,
                 const BufferHandle& out, int32_t n, float eps) noexcept;

    // ---- BLLM-179: on-device elementwise (add/SwiGLU/GeGLU). out[i] = op(a[i],
    // b[i]); op 0=a+b (residual/bias), 1=silu(a)*b (SwiGLU), 2=gelu(a)*b (GeGLU).
    // One invocation per element. Blocks until complete. ----
    bool create_elementwise_pipeline(MatmulPipeline* out) noexcept;
    bool elementwise(const MatmulPipeline& pipeline, const BufferHandle& a, const BufferHandle& b,
                     const BufferHandle& out, int32_t n, int32_t op) noexcept;

    // ---- BLLM-177: on-device RoPE (rotate_half, matches rope_half_split).
    // Rotates a [num_heads, head_dim] buffer in place. Blocks until complete. ----
    bool create_rope_pipeline(MatmulPipeline* out) noexcept;
    bool rope(const MatmulPipeline& pipeline, const BufferHandle& x, int32_t head_dim,
              int32_t num_heads, int32_t position, float theta) noexcept;

    // ---- BLLM-178: on-device decode-step causal attention with flash-style
    // online softmax (the Vulkan twin of bolt::rocm::attention). `q` is
    // [num_heads, head_dim]; `k_cache`/`v_cache` are [seq_len, num_kv_heads,
    // head_dim]; `out` is [num_heads, head_dim]. GQA: query head h reads kv
    // head h/(num_heads/num_kv_heads). One workgroup per query head; requires
    // head_dim <= 256. Blocks until complete. ----
    bool create_attention_pipeline(MatmulPipeline* out) noexcept;
    bool attention(const MatmulPipeline& pipeline, const BufferHandle& q,
                   const BufferHandle& k_cache, const BufferHandle& v_cache,
                   const BufferHandle& out, int32_t num_heads, int32_t num_kv_heads,
                   int32_t head_dim, int32_t seq_len, float scale) noexcept;

    // ---- BLLM-81: int8-ACTIVATION matmul (the real expert-FFN numerics
    // tier, distinct from matmul_dequant's float-activation tier above) ---

    // Creates the compiled pipeline for the int8-activation variant of one
    // weight dtype's matmul shader (matmul_dequant_int8act_{int8,int4}.comp).
    // `dtype` must be Int8 or Int4 (Int2/F32 have no int8-activation hot
    // path in boltllm today -- see bolt::compute::detail's own CPU
    // idot_gemv_int8/int4, which only cover those two). Returns false on
    // any failure or an unsupported dtype. Precondition: is_valid().
    bool create_int8_activation_pipeline(WeightDType dtype, MatmulPipeline* out) noexcept;

    // Same lifetime contract as destroy_matmul_pipeline() -- shares
    // implementation, just documented separately for discoverability next
    // to create_int8_activation_pipeline().
    void destroy_int8_activation_pipeline(MatmulPipeline* pipeline) noexcept;

    // Dispatches the GEMV: out[r] = dot(activation_q, weight.row(r)), r in
    // [0, out_rows), where BOTH activation and weight are already-quantized
    // int8/int4 integers -- matching boltllm::quant::idot_gemv_int8_scalar/
    // idot_gemv_int4_scalar's exact contract (IdotDispatch.cpp): int32
    // accumulation of act_q[i]*weight_q[i] (or dequant-then-multiply for
    // Int4 weight), a SINGLE final act_scale*weight_scale float multiply.
    // `activation` must be a device buffer of already-quantized int8_t
    // values (the caller quantizes host-side via
    // boltllm::quant::quantize_act_row_int8 or equivalent BEFORE upload --
    // this function never quantizes on-device). `act_scale` is the single
    // scalar boltllm::quant::quantize_act_row_int8 produced for that whole
    // activation row. Blocks until the dispatch completes.
    // Precondition: is_valid(), pipeline created via
    // create_int8_activation_pipeline() with a matching dtype.
    bool matmul_dequant_int8_activation(const MatmulPipeline& pipeline, const BufferHandle& weight,
                                        const BufferHandle& weight_scale,
                                        const BufferHandle& activation, float act_scale,
                                        const BufferHandle& out, int32_t out_rows,
                                        int32_t cols) noexcept;

    // ---- BLLM-87: BATCHED int8-ACTIVATION matmul -- amortizes one
    // dispatch's overhead (descriptor set update, command buffer submit,
    // queue wait) across MANY tokens against the SAME resident weight,
    // instead of one matmul_dequant_int8_activation() call per token. See
    // matmul_dequant_int8act_int8_batched.comp's header for the full
    // design rationale (this is the viable alternative to a per-token
    // device-resident expert cache, which BLLM-85's investigation found
    // upload-cost-dominated and not worth building).

    // Creates the compiled pipeline for the batched int8-activation variant
    // of one weight dtype's matmul shader
    // (matmul_dequant_int8act_{int8,int4}_batched.comp). Same dtype scope
    // as create_int8_activation_pipeline() (Int8/Int4 only). A pipeline
    // created here is NOT interchangeable with one from
    // create_int8_activation_pipeline() (different descriptor layout/push
    // constants) -- only pass it to matmul_dequant_int8_activation_batched().
    bool create_int8_activation_batched_pipeline(WeightDType dtype, MatmulPipeline* out) noexcept;

    // Same lifetime contract as destroy_matmul_pipeline().
    void destroy_int8_activation_batched_pipeline(MatmulPipeline* pipeline) noexcept;

    // Dispatches out[t][r] = dot(activation[t], weight.row(r)) *
    // act_scale[t] * weight_scale[r] for EVERY (row, token) pair in ONE
    // Vulkan submit. `activation` is `n_tokens` already-quantized int8_t
    // rows, TIGHTLY PACKED token-major (row t at byte offset t*cols, no
    // per-token padding -- matches boltllm::quant::quantize_act_row_int8's
    // own per-row packing, just concatenated across tokens). `act_scale` is
    // `n_tokens` floats, one per token (each token's own
    // quantize_act_row_int8-produced scale). `out` is `n_tokens * out_rows`
    // floats, token-major (`out[t*out_rows + r]`). Blocks until the
    // dispatch completes. Precondition: is_valid(), pipeline created via
    // create_int8_activation_batched_pipeline() with a matching dtype,
    // n_tokens >= 1.
    bool matmul_dequant_int8_activation_batched(const MatmulPipeline& pipeline,
                                                 const BufferHandle& weight,
                                                 const BufferHandle& weight_scale,
                                                 const BufferHandle& activation,
                                                 const BufferHandle& act_scale,
                                                 const BufferHandle& out, int32_t out_rows,
                                                 int32_t cols, int32_t n_tokens) noexcept;

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
