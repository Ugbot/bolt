#pragma once
// bolt_cuda.h -- BLLM-182: CUDA/NVIDIA matmul+dequant + on-device-forward
// kernel core behind bolt::compute's reserved CUDA device slot. A near-1:1
// mirror of bolt::rocm (bolt_rocm.h) -- CUDA and HIP are source-compatible,
// so this is the ROCm backend transcribed s/hip/cuda/, kept auditable as a
// straight diff against its ROCm twin.
//
// This module ships the actual compute kernels: a GEMV (matmul) with
// on-the-fly F32/Int8/Int4/Int2 dequant, plus the on-device forward ops
// (rmsnorm, rope, elementwise/SwiGLU, decode attention), all numerically
// identical to bolt_compute.h's CPU references (same kInt4/kInt2 offsets,
// same per-row `scale` convention) but executed as real CUDA `__global__`
// kernels, one thread-block per output row / head.
//
// Runtime-detected, CPU fallback always works: `available()` probes for a
// real CUDA-capable device and returns false (never asserts/aborts) if
// none exists. `Context::create()` follows the same contract.
//
// STATUS (2026-07-18): structurally complete, a mechanical transcription of
// the ROCm backend which is itself VERIFIED on real gfx1151 hardware
// (13/13 test_bolt_rocm), but this CUDA twin is UNVALIDATED -- this dev box
// (AMD Ryzen AI Max+ 395 / Radeon 8060S) has NO NVIDIA GPU and NO CUDA
// toolkit (no nvcc), so it cannot be compiled or run here. CMakeLists.txt
// only builds bolt::cuda when it finds a real nvcc / CMAKE_CUDA_COMPILER,
// so this never blocks anyone else's build; hardware validation is deferred
// to whoever next has a working CUDA toolchain + NVIDIA device. Per this
// project's measure-before-claiming discipline, BLLM-182 stays OPEN until
// then -- exactly as bolt::rocm (BLLM-78) stayed unvalidated until ROCm 7.1
// landed on this box.
//
// Compiled (not header-only): CUDA kernels require nvcc, so this is a
// separate optional compiled library gated on BOLT_BUILD_CUDA, same shape
// as bolt::rocm/bolt::vulkan. Handles are opaque; only bolt_cuda.cpp
// includes <cuda_runtime.h>.

#include <cstdint>

namespace bolt::cuda {

// True iff a CUDA-capable device is present and cudaGetDeviceCount succeeds
// with a non-zero count. Intended for one-time startup capability
// detection, not a hot-path check.
bool available() noexcept;

// One device-resident allocation (a CUDA device pointer, opaque here).
struct DeviceBuffer {
    void*    ptr = nullptr;
    uint64_t size_bytes = 0;
};

// Weight dtype tag mirroring the subset of bolt::DType (bolt_tensor.h)
// that bolt_compute.h's matmul_row_dot_f32 dequantizes today (see that
// file's architectural choice #1). Restated by value (not #include
// bolt_tensor.h) so callers that only want the CUDA kernel don't have to
// link bolt::core's Tensor type -- kept in lock-step by value with
// bolt::DType; F16/BF16 are intentionally absent (bolt_compute.h's own
// CPU tier has no implementation for them yet either).
enum class WeightDType : uint8_t {
    F32  = 0,
    Int8 = 1,
    Int4 = 2,
    Int2 = 3,
};

// Owns one CUDA device context (device selection + capability query).
// Tiger-Style single-owner: not copyable/movable.
class Context {
public:
    Context() noexcept = default;
    ~Context() noexcept;

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    // Selects device 0 (no multi-GPU policy yet) and queries its
    // properties. Returns false (leaving `this` inert) on any failure:
    // no CUDA runtime, no device, or the runtime rejecting device
    // selection. May be called at most once per instance.
    bool create() noexcept;

    // Resets to the inert state; safe to call on a Context that was never
    // created or whose create() returned false.
    void destroy() noexcept;

    bool is_valid() const noexcept { return valid_; }

    // Valid only if is_valid().
    const char* device_name() const noexcept { return device_name_; }
    uint64_t device_local_heap_bytes() const noexcept { return device_local_heap_bytes_; }
    // Reported by cudaDeviceProp::warpSize -- 32 or 64 depending on
    // gfx target/compile mode. Informational only: matmul_dequant()'s
    // kernel uses a runtime block-wide shared-memory reduction (see
    // bolt_cuda.cpp) rather than a wavefront-width-assuming shuffle, so
    // correctness never depends on this value being read accurately.
    int32_t wavefront_size() const noexcept { return wavefront_size_; }

    // Allocates a device-resident buffer. Returns false (leaving `*out`
    // default-constructed) on failure. Precondition: is_valid().
    bool allocate(uint64_t size_bytes, DeviceBuffer* out) noexcept;

    // Frees a handle allocate() returned; safe on a default-constructed
    // (never-allocated) handle. Zeroes `*buf`.
    void free(DeviceBuffer* buf) noexcept;

    // Host-to-device / device-to-host blocking copies. `size_bytes` must
    // not exceed the destination/source buffer's allocated size.
    // Precondition: is_valid().
    bool copy_to_device(const void* host_src, DeviceBuffer* dst, uint64_t size_bytes) noexcept;
    bool copy_to_host(const DeviceBuffer& src, void* host_dst, uint64_t size_bytes) noexcept;

private:
    bool     valid_ = false;
    int32_t  device_index_ = -1;
    int32_t  wavefront_size_ = 0;
    char     device_name_[256] = {};
    uint64_t device_local_heap_bytes_ = 0;
};

// GEMV: out[r] = dot(weight.row(r), activation), r in [0, out_rows), with
// on-the-fly dequant matching bolt_compute.h's matmul_row_dot_f32 exactly.
// All pointers except `ctx` are DEVICE pointers (obtained via
// Context::allocate + copy_to_device) -- this function launches a kernel
// and blocks until it completes (cudaDeviceSynchronize), it does not copy
// host memory itself. `scale_device` may be null only when
// dtype == WeightDType::F32 (no scale used); required otherwise.
// Returns false on invalid arguments or any launch/sync failure.
// Precondition: ctx.is_valid().
bool matmul_dequant(Context& ctx, const void* weight_device, WeightDType dtype,
                     const float* scale_device, const float* activation_device,
                     float* out_device, int32_t out_rows, int32_t cols) noexcept;

// BLLM-91: int8-ACTIVATION matmul -- the CUDA twin of
// bolt::vulkan::Context::matmul_dequant_int8_activation (BLLM-81), the
// real expert-FFN numerics tier (distinct from matmul_dequant above, which
// matches lm_head's float-activation tier). Matches
// boltllm::quant::idot_gemv_int8/int4_scalar's exact contract
// (IdotDispatch.cpp): int32 accumulation of act_q[i]*weight_q[i] (or
// dequant-then-multiply for Int4 weight), a SINGLE final
// act_scale*weight_scale float multiply. `dtype` must be Int8 or Int4 (no
// int8-activation hot path exists for F32/Int2 weights in boltllm today).
// `activation_device` is an already-quantized int8_t device buffer (the
// caller quantizes host-side via boltllm::quant::quantize_act_row_int8 or
// equivalent BEFORE upload -- this function never quantizes on-device).
// Returns false on invalid arguments, an unsupported dtype, or any
// launch/sync failure. Precondition: ctx.is_valid().
bool matmul_dequant_int8_activation(Context& ctx, const void* weight_device, WeightDType dtype,
                                     const float* weight_scale_device,
                                     const int8_t* activation_device, float act_scale,
                                     float* out_device, int32_t out_rows, int32_t cols) noexcept;

// ---- On-device forward ops (CUDA twins of the bolt::vulkan shaders), for the
// resident-graph GPU forward (BLLM-181). All buffers are device pointers from
// Context::allocate(); each blocks until complete (cudaDeviceSynchronize). ----

// BLLM-184: y[i] = x[i]/sqrt(mean(x^2)+eps) * w[i].
bool rmsnorm(Context& ctx, const float* x_device, const float* w_device, float* y_device, int32_t n,
             float eps) noexcept;

// BLLM-185: rotate_half RoPE on a [num_heads, head_dim] buffer, in place.
bool rope(Context& ctx, float* x_device, int32_t head_dim, int32_t num_heads, int32_t position,
          float theta) noexcept;

// BLLM-186: elementwise op 0=a+b (residual/bias), 1=silu(a)*b (SwiGLU),
// 2=gelu(a)*b (GeGLU).
bool elementwise(Context& ctx, const float* a_device, const float* b_device, float* out_device,
                 int32_t n, int32_t op) noexcept;

// BLLM-187: decode-step causal attention with flash-style online softmax. `q`
// is [num_heads, head_dim]; `k_cache`/`v_cache` are [seq_len, num_kv_heads,
// head_dim] (the KV cache including the current position); `out` is
// [num_heads, head_dim]. GQA: query head h reads kv head h/(num_heads/
// num_kv_heads). `scale` is the softmax scale (typically 1/sqrt(head_dim)).
// Precondition: num_heads % num_kv_heads == 0 and head_dim <= 256.
bool attention(Context& ctx, const float* q_device, const float* k_cache_device,
               const float* v_cache_device, float* out_device, int32_t num_heads,
               int32_t num_kv_heads, int32_t head_dim, int32_t seq_len, float scale) noexcept;

}  // namespace bolt::cuda
