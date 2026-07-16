// bolt_rocm.cpp -- BLLM-78: real HIP kernels behind bolt::rocm (see
// bolt_rocm.h for the full contract). Written against AMD's documented
// HIP runtime API (hipGetDeviceCount/hipSetDevice/hipGetDeviceProperties/
// hipMalloc/hipFree/hipMemcpy, `__global__`/`<<<>>>` kernel launch syntax
// -- all hipcc-standard). UNVALIDATED against a real hipcc/device: this
// dev box has no ROCm/HIP SDK for native Windows (see bolt_rocm.h's
// status note and BLLM-78 tracker notes). This file is only compiled
// when CMakeLists.txt finds a real hipcc (BOLT_BUILD_ROCM); until then it
// is not part of any build and cannot break one.

#include "bolt/bolt_rocm.h"

#include <hip/hip_runtime.h>

#include <cstdio>

namespace bolt::rocm {

namespace {

// MUST stay byte-for-byte identical to
// bolt::compute::detail::kInt4DequantOffset/kInt2DequantOffset
// (bolt_compute.h) and boltllm::quant::QuantTypes.h's originals.
constexpr int32_t kInt4DequantOffset = 8;
constexpr int32_t kInt2DequantOffset = 2;

// One thread-block computes one output row's dot product. Threads stride
// over `cols`, then a block-wide shared-memory tree reduction combines
// the partial sums. Deliberately NOT a raw wavefront-shuffle reduction
// (ds4's own approach, per BLLM-78's prior-art notes): that assumes a
// fixed 64-wide wavefront, which is CDNA's/wave64-compiled-RDNA's width
// but NOT necessarily this box's target (gfx1151/Strix Halo is an RDNA
// part; RDNA defaults to wave32 unless compiled -mwavefrontsize64). A
// block-wide reduction sized to the actual runtime launch dimension is
// correct regardless of the compiled wavefront width -- important here
// since there is no hardware access to verify that width empirically.
constexpr int32_t kBlockSize = 256;

__global__ void matmul_dequant_f32_kernel(const float* weight, const float* activation,
                                           float* out, int32_t cols) {
    __shared__ float shared[kBlockSize];
    const int32_t row = blockIdx.x;
    const float* row_ptr = weight + static_cast<int64_t>(row) * cols;

    float acc = 0.0f;
    for (int32_t i = threadIdx.x; i < cols; i += blockDim.x) {
        acc += activation[i] * row_ptr[i];
    }
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = shared[0];
}

__global__ void matmul_dequant_int8_kernel(const int8_t* weight, const float* scale,
                                            const float* activation, float* out, int32_t cols) {
    __shared__ float shared[kBlockSize];
    const int32_t row = blockIdx.x;
    const int8_t* row_ptr = weight + static_cast<int64_t>(row) * cols;

    float acc = 0.0f;
    for (int32_t i = threadIdx.x; i < cols; i += blockDim.x) {
        acc += activation[i] * static_cast<float>(row_ptr[i]);
    }
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = shared[0] * scale[row];
}

// Int4: two 4-bit codes packed per byte, low nibble first -- same layout
// bolt_compute.h's matmul_row_dot_f32 (Int4 case) and boltllm::quant's
// packing convention use.
__global__ void matmul_dequant_int4_kernel(const uint8_t* weight, const float* scale,
                                            const float* activation, float* out, int32_t cols) {
    __shared__ float shared[kBlockSize];
    const int32_t row = blockIdx.x;
    const int32_t packed_cols = (cols + 1) / 2;
    const uint8_t* row_ptr = weight + static_cast<int64_t>(row) * packed_cols;

    float acc = 0.0f;
    for (int32_t i = threadIdx.x; i < cols; i += blockDim.x) {
        const uint8_t byte = row_ptr[i >> 1];
        const int32_t nibble = (i & 1) ? (byte >> 4) : (byte & 0x0F);
        const int32_t v = nibble - kInt4DequantOffset;
        acc += activation[i] * static_cast<float>(v);
    }
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = shared[0] * scale[row];
}

// Int2: four 2-bit codes packed per byte, lowest bits first -- same
// layout bolt_compute.h's matmul_row_dot_f32 (Int2 case) uses.
__global__ void matmul_dequant_int2_kernel(const uint8_t* weight, const float* scale,
                                            const float* activation, float* out, int32_t cols) {
    __shared__ float shared[kBlockSize];
    const int32_t row = blockIdx.x;
    const int32_t packed_cols = (cols + 3) / 4;
    const uint8_t* row_ptr = weight + static_cast<int64_t>(row) * packed_cols;

    float acc = 0.0f;
    for (int32_t i = threadIdx.x; i < cols; i += blockDim.x) {
        const uint8_t byte = row_ptr[i >> 2];
        const int32_t sh = (i & 3) * 2;
        const int32_t v = static_cast<int32_t>((byte >> sh) & 0x03) - kInt2DequantOffset;
        acc += activation[i] * static_cast<float>(v);
    }
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = shared[0] * scale[row];
}

}  // namespace

bool available() noexcept {
    int32_t device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess) return false;
    return device_count > 0;
}

Context::~Context() noexcept { destroy(); }

bool Context::create() noexcept {
    int32_t device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0) return false;

    const int32_t chosen = 0;  // first device -- no multi-GPU selection policy yet.
    if (hipSetDevice(chosen) != hipSuccess) return false;

    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, chosen) != hipSuccess) return false;

    std::snprintf(device_name_, sizeof(device_name_), "%s", props.name);
    device_local_heap_bytes_ = static_cast<uint64_t>(props.totalGlobalMem);
    wavefront_size_ = props.warpSize;
    device_index_ = chosen;
    valid_ = true;
    return true;
}

void Context::destroy() noexcept {
    valid_ = false;
    device_index_ = -1;
    wavefront_size_ = 0;
    device_name_[0] = '\0';
    device_local_heap_bytes_ = 0;
}

bool Context::allocate(uint64_t size_bytes, DeviceBuffer* out) noexcept {
    if (out == nullptr) return false;
    *out = DeviceBuffer{};
    if (!valid_ || size_bytes == 0) return false;

    void* ptr = nullptr;
    if (hipMalloc(&ptr, size_bytes) != hipSuccess) return false;
    out->ptr = ptr;
    out->size_bytes = size_bytes;
    return true;
}

void Context::free(DeviceBuffer* buf) noexcept {
    if (buf == nullptr || buf->ptr == nullptr) return;
    hipFree(buf->ptr);
    buf->ptr = nullptr;
    buf->size_bytes = 0;
}

bool Context::copy_to_device(const void* host_src, DeviceBuffer* dst, uint64_t size_bytes) noexcept {
    if (host_src == nullptr || dst == nullptr || dst->ptr == nullptr) return false;
    if (size_bytes > dst->size_bytes) return false;
    return hipMemcpy(dst->ptr, host_src, size_bytes, hipMemcpyHostToDevice) == hipSuccess;
}

bool Context::copy_to_host(const DeviceBuffer& src, void* host_dst, uint64_t size_bytes) noexcept {
    if (host_dst == nullptr || src.ptr == nullptr) return false;
    if (size_bytes > src.size_bytes) return false;
    return hipMemcpy(host_dst, src.ptr, size_bytes, hipMemcpyDeviceToHost) == hipSuccess;
}

bool matmul_dequant(Context& ctx, const void* weight_device, WeightDType dtype,
                     const float* scale_device, const float* activation_device,
                     float* out_device, int32_t out_rows, int32_t cols) noexcept {
    if (!ctx.is_valid() || weight_device == nullptr || activation_device == nullptr ||
        out_device == nullptr || out_rows <= 0 || cols <= 0) {
        return false;
    }
    if (dtype != WeightDType::F32 && scale_device == nullptr) return false;

    const dim3 grid(static_cast<unsigned int>(out_rows));
    const dim3 block(static_cast<unsigned int>(kBlockSize));

    switch (dtype) {
        case WeightDType::F32:
            matmul_dequant_f32_kernel<<<grid, block>>>(
                static_cast<const float*>(weight_device), activation_device, out_device, cols);
            break;
        case WeightDType::Int8:
            matmul_dequant_int8_kernel<<<grid, block>>>(
                static_cast<const int8_t*>(weight_device), scale_device, activation_device,
                out_device, cols);
            break;
        case WeightDType::Int4:
            matmul_dequant_int4_kernel<<<grid, block>>>(
                static_cast<const uint8_t*>(weight_device), scale_device, activation_device,
                out_device, cols);
            break;
        case WeightDType::Int2:
            matmul_dequant_int2_kernel<<<grid, block>>>(
                static_cast<const uint8_t*>(weight_device), scale_device, activation_device,
                out_device, cols);
            break;
        default:
            return false;
    }

    return hipDeviceSynchronize() == hipSuccess;
}

}  // namespace bolt::rocm
