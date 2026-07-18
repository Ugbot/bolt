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

// BLLM-189: templated on Accumulate -- <false> writes out[row]=dot (plain
// matmul), <true> writes out[row]+=dot (fuses the residual-add into the
// projection, removing a separate elementwise kernel). Accumulate is a
// compile-time constant (bolt's compile-time-dispatch idiom), so the branch
// vanishes -- the hot loop stays branch-free.
template <bool Accumulate>
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
    if (threadIdx.x == 0) out[row] = Accumulate ? (out[row] + shared[0]) : shared[0];
}

template <bool Accumulate>
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
    if (threadIdx.x == 0) {
        const float v = shared[0] * scale[row];
        out[row] = Accumulate ? (out[row] + v) : v;
    }
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

// BLLM-91: int8-ACTIVATION kernels -- the real expert-FFN numerics tier
// (distinct from the float-activation kernels above, which match lm_head's
// tier -- see bolt_rocm.h's matmul_dequant_int8_activation doc). Both
// activation AND weight are already-quantized integers; accumulate in
// int32 (safe without overflow for any realistic hidden size -- max |term|
// = 127*127 = 16129, staying well inside int32 range even at cols in the
// hundreds of thousands), single final act_scale*weight_scale float
// multiply, matching boltllm::quant::idot_gemv_int8/int4_scalar's exact
// contract (IdotDispatch.cpp) byte-for-byte.
__global__ void matmul_dequant_int8act_int8_kernel(const int8_t* weight, const float* weight_scale,
                                                    const int8_t* activation, float act_scale,
                                                    float* out, int32_t cols) {
    __shared__ int32_t shared[kBlockSize];
    const int32_t row = blockIdx.x;
    const int8_t* row_ptr = weight + static_cast<int64_t>(row) * cols;

    int32_t acc = 0;
    for (int32_t i = threadIdx.x; i < cols; i += blockDim.x) {
        acc += static_cast<int32_t>(activation[i]) * static_cast<int32_t>(row_ptr[i]);
    }
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = static_cast<float>(shared[0]) * act_scale * weight_scale[row];
}

// Int4 weight, low-nibble-first packing, offset kInt4DequantOffset -- same
// layout matmul_dequant_int4_kernel uses, just int8-activation numerics.
__global__ void matmul_dequant_int8act_int4_kernel(const uint8_t* weight, const float* weight_scale,
                                                    const int8_t* activation, float act_scale,
                                                    float* out, int32_t cols) {
    __shared__ int32_t shared[kBlockSize];
    const int32_t row = blockIdx.x;
    const int32_t packed_cols = (cols + 1) / 2;
    const uint8_t* row_ptr = weight + static_cast<int64_t>(row) * packed_cols;

    int32_t acc = 0;
    for (int32_t i = threadIdx.x; i < cols; i += blockDim.x) {
        const uint8_t byte = row_ptr[i >> 1];
        const int32_t nibble = (i & 1) ? (byte >> 4) : (byte & 0x0F);
        const int32_t w_v = nibble - kInt4DequantOffset;
        acc += static_cast<int32_t>(activation[i]) * w_v;
    }
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = static_cast<float>(shared[0]) * act_scale * weight_scale[row];
}

// BLLM-184: RMSNorm -- y[i]=x[i]/sqrt(mean(x^2)+eps)*w[i]. One block, tree reduce.
__global__ void rmsnorm_kernel(const float* x, const float* w, float* y, int32_t n, float eps) {
    __shared__ float shared[kBlockSize];
    __shared__ float inv_rms;
    float ss = 0.0f;
    for (int32_t i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    shared[threadIdx.x] = ss;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) inv_rms = rsqrtf(shared[0] / static_cast<float>(n) + eps);
    __syncthreads();
    const float r = inv_rms;
    for (int32_t i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * r * w[i];
}

// BLLM-185: RoPE (rotate_half) on [num_heads, head_dim], in place, one thread
// per (head, j) pair -- matches rope_half_split exactly.
__global__ void rope_kernel(float* x, int32_t head_dim, int32_t num_heads, int32_t position,
                            float theta) {
    const int32_t hd2 = head_dim / 2;
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_heads * hd2) return;
    const int32_t head = idx / hd2;
    const int32_t j = idx % hd2;
    const int32_t base = head * head_dim;
    const float inv = powf(theta, -2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
    const float ang = static_cast<float>(position) * inv;
    const float cs = cosf(ang), sn = sinf(ang);
    const float a = x[base + j], b = x[base + j + hd2];
    x[base + j] = a * cs - b * sn;
    x[base + j + hd2] = b * cs + a * sn;
}

// BLLM-186: elementwise -- op 0=add(a+b), 1=SwiGLU(silu(a)*b), 2=GeGLU(gelu(a)*b).
__global__ void elementwise_kernel(const float* a, const float* b, float* o, int32_t n, int32_t op) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float av = a[i], bv = b[i];
    if (op == 0) {
        o[i] = av + bv;
    } else if (op == 1) {
        o[i] = (av / (1.0f + expf(-av))) * bv;
    } else {
        const float g = 0.5f * av * (1.0f + tanhf(0.7978845608028654f * (av + 0.044715f * av * av * av)));
        o[i] = g * bv;
    }
}

// BLLM-187: decode-step attention with flash-style online softmax. ONE block
// per query head; blockDim is a power-of-two >= head_dim, one thread per dim
// (inactive threads d>=head_dim contribute 0 to the QK dot and skip the write,
// so the power-of-two tree reduction stays valid). Mirrors qwen2_attention_'s
// numerics exactly: stream over cached KV in one pass keeping a running max
// `m`, denominator `l`, and rescaled per-dim context `ch`. GQA: query head h
// reads kv head h/group_size. m/l/corr/p are computed identically by every
// thread from the broadcast score, so they stay in lockstep with no extra sync.
__global__ void attention_kernel(const float* q, const float* k_cache, const float* v_cache,
                                 float* out, int32_t head_dim, int32_t num_kv_heads,
                                 int32_t group_size, int32_t seq_len, float scale) {
    __shared__ float red[kBlockSize];
    __shared__ float s_bcast;
    const int32_t h = blockIdx.x;
    const int32_t kv_head = h / group_size;
    const int32_t d = threadIdx.x;
    const bool active = d < head_dim;
    const float qd = active ? q[static_cast<int64_t>(h) * head_dim + d] : 0.0f;

    float ch = 0.0f, m = -INFINITY, l = 0.0f;
    for (int32_t t = 0; t < seq_len; ++t) {
        const int64_t off = (static_cast<int64_t>(t) * num_kv_heads + kv_head) * head_dim + d;
        const float kd = active ? k_cache[off] : 0.0f;
        red[threadIdx.x] = qd * kd;
        __syncthreads();
        for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) s_bcast = red[0] * scale;
        __syncthreads();
        const float s = s_bcast;
        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new);
        const float p = expf(s - m_new);
        l = l * corr + p;
        ch = ch * corr + p * (active ? v_cache[off] : 0.0f);
        m = m_new;
    }
    if (active) out[static_cast<int64_t>(h) * head_dim + d] = ch / l;
}

// BLLM-189: fused gate+up+SwiGLU FFN kernel. ONE block per FFN row computes BOTH
// gate=dot(Wg[row],x) and up=dot(Wu[row],x) over the shared activation in a
// single pass, then act[row]=silu(gate)*up. Replaces 3 kernels (gate matmul, up
// matmul, SwiGLU elementwise) + 2 intermediate buffers with 1 launch -- the
// op-fusion lever (fewer serial data-dependent kernels) the resident-graph decode
// measurement identified (docs/perf/gpu-backends-parity.md). Branch-free inner
// loop, two shared-mem tree reductions. F32-weight tier.
__global__ void fused_swiglu_f32_kernel(const float* wg, const float* wu, const float* x,
                                        float* act, int32_t cols) {
    __shared__ float sg[kBlockSize];
    __shared__ float su[kBlockSize];
    const int32_t row = blockIdx.x;
    const float* gr = wg + static_cast<int64_t>(row) * cols;
    const float* ur = wu + static_cast<int64_t>(row) * cols;
    float ga = 0.0f, ua = 0.0f;
    for (int32_t i = threadIdx.x; i < cols; i += blockDim.x) {
        const float xi = x[i];
        ga += xi * gr[i];
        ua += xi * ur[i];
    }
    sg[threadIdx.x] = ga;
    su[threadIdx.x] = ua;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sg[threadIdx.x] += sg[threadIdx.x + stride];
            su[threadIdx.x] += su[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float g = sg[0];
        act[row] = (g / (1.0f + expf(-g))) * su[0];  // silu(gate) * up
    }
}

// Int8-weight twin: per-row scales, int8 weight x float activation, one scale
// multiply per dot (matches matmul_dequant_int8_kernel's contract byte-for-byte).
__global__ void fused_swiglu_int8_kernel(const int8_t* wg, const int8_t* wu, const float* wg_scale,
                                         const float* wu_scale, const float* x, float* act,
                                         int32_t cols) {
    __shared__ float sg[kBlockSize];
    __shared__ float su[kBlockSize];
    const int32_t row = blockIdx.x;
    const int8_t* gr = wg + static_cast<int64_t>(row) * cols;
    const int8_t* ur = wu + static_cast<int64_t>(row) * cols;
    float ga = 0.0f, ua = 0.0f;
    for (int32_t i = threadIdx.x; i < cols; i += blockDim.x) {
        const float xi = x[i];
        ga += xi * static_cast<float>(gr[i]);
        ua += xi * static_cast<float>(ur[i]);
    }
    sg[threadIdx.x] = ga;
    su[threadIdx.x] = ua;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sg[threadIdx.x] += sg[threadIdx.x + stride];
            su[threadIdx.x] += su[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float g = sg[0] * wg_scale[row];
        act[row] = (g / (1.0f + expf(-g))) * (su[0] * wu_scale[row]);
    }
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

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return false;
    stream_ = stream;

    std::snprintf(device_name_, sizeof(device_name_), "%s", props.name);
    device_local_heap_bytes_ = static_cast<uint64_t>(props.totalGlobalMem);
    wavefront_size_ = props.warpSize;
    device_index_ = chosen;
    valid_ = true;
    return true;
}

void Context::destroy() noexcept {
    free_graph();
    if (stream_ != nullptr) {
        hipStreamDestroy(static_cast<hipStream_t>(stream_));
        stream_ = nullptr;
    }
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

bool Context::end_batch() noexcept {
    batch_mode_ = false;
    return hipStreamSynchronize(static_cast<hipStream_t>(stream_)) == hipSuccess;
}

bool Context::begin_graph_capture() noexcept {
    if (!valid_ || stream_ == nullptr) return false;
    free_graph();
    batch_mode_ = true;  // ops skip their per-op sync while being captured
    return hipStreamBeginCapture(static_cast<hipStream_t>(stream_),
                                 hipStreamCaptureModeThreadLocal) == hipSuccess;
}

bool Context::end_graph_capture() noexcept {
    batch_mode_ = false;
    hipGraph_t graph = nullptr;
    if (hipStreamEndCapture(static_cast<hipStream_t>(stream_), &graph) != hipSuccess) return false;
    hipGraphExec_t exec = nullptr;
    const hipError_t rc = hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    hipGraphDestroy(graph);
    if (rc != hipSuccess) return false;
    graph_exec_ = exec;
    return true;
}

bool Context::graph_launch() noexcept {
    if (graph_exec_ == nullptr || stream_ == nullptr) return false;
    if (hipGraphLaunch(static_cast<hipGraphExec_t>(graph_exec_),
                       static_cast<hipStream_t>(stream_)) != hipSuccess) {
        return false;
    }
    return hipStreamSynchronize(static_cast<hipStream_t>(stream_)) == hipSuccess;
}

void Context::free_graph() noexcept {
    if (graph_exec_ != nullptr) {
        hipGraphExecDestroy(static_cast<hipGraphExec_t>(graph_exec_));
        graph_exec_ = nullptr;
    }
}

bool matmul_dequant(Context& ctx, const void* weight_device, WeightDType dtype,
                     const float* scale_device, const float* activation_device,
                     float* out_device, int32_t out_rows, int32_t cols) noexcept {
    if (!ctx.is_valid() || weight_device == nullptr || activation_device == nullptr ||
        out_device == nullptr || out_rows <= 0 || cols <= 0) {
        return false;
    }
    if (dtype != WeightDType::F32 && scale_device == nullptr) return false;

    const hipStream_t st = static_cast<hipStream_t>(ctx.stream());
    const dim3 grid(static_cast<unsigned int>(out_rows));
    const dim3 block(static_cast<unsigned int>(kBlockSize));

    switch (dtype) {
        case WeightDType::F32:
            matmul_dequant_f32_kernel<false><<<grid, block, 0, st>>>(
                static_cast<const float*>(weight_device), activation_device, out_device, cols);
            break;
        case WeightDType::Int8:
            matmul_dequant_int8_kernel<false><<<grid, block, 0, st>>>(
                static_cast<const int8_t*>(weight_device), scale_device, activation_device,
                out_device, cols);
            break;
        case WeightDType::Int4:
            matmul_dequant_int4_kernel<<<grid, block, 0, st>>>(
                static_cast<const uint8_t*>(weight_device), scale_device, activation_device,
                out_device, cols);
            break;
        case WeightDType::Int2:
            matmul_dequant_int2_kernel<<<grid, block, 0, st>>>(
                static_cast<const uint8_t*>(weight_device), scale_device, activation_device,
                out_device, cols);
            break;
        default:
            return false;
    }

    return ctx.batching() ? true : (hipStreamSynchronize(st) == hipSuccess);
}

bool matmul_dequant_residual(Context& ctx, const void* weight_device, WeightDType dtype,
                             const float* scale_device, const float* activation_device,
                             float* out_device, int32_t out_rows, int32_t cols) noexcept {
    if (!ctx.is_valid() || weight_device == nullptr || activation_device == nullptr ||
        out_device == nullptr || out_rows <= 0 || cols <= 0) {
        return false;
    }
    if (dtype == WeightDType::Int8 && scale_device == nullptr) return false;

    const hipStream_t st = static_cast<hipStream_t>(ctx.stream());
    const dim3 grid(static_cast<unsigned int>(out_rows));
    const dim3 block(static_cast<unsigned int>(kBlockSize));
    // out[row] += dot(...) -- fuses the residual-add into the projection. F32/Int8
    // only (the resident-graph decode tiers); Int4/Int2 keep the unfused path.
    if (dtype == WeightDType::F32) {
        matmul_dequant_f32_kernel<true><<<grid, block, 0, st>>>(
            static_cast<const float*>(weight_device), activation_device, out_device, cols);
    } else if (dtype == WeightDType::Int8) {
        matmul_dequant_int8_kernel<true><<<grid, block, 0, st>>>(
            static_cast<const int8_t*>(weight_device), scale_device, activation_device, out_device,
            cols);
    } else {
        return false;
    }
    return ctx.batching() ? true : (hipStreamSynchronize(st) == hipSuccess);
}

bool matmul_dequant_int8_activation(Context& ctx, const void* weight_device, WeightDType dtype,
                                     const float* weight_scale_device,
                                     const int8_t* activation_device, float act_scale,
                                     float* out_device, int32_t out_rows, int32_t cols) noexcept {
    if (!ctx.is_valid() || weight_device == nullptr || weight_scale_device == nullptr ||
        activation_device == nullptr || out_device == nullptr || out_rows <= 0 || cols <= 0) {
        return false;
    }
    if (dtype != WeightDType::Int8 && dtype != WeightDType::Int4) return false;

    const hipStream_t st = static_cast<hipStream_t>(ctx.stream());
    const dim3 grid(static_cast<unsigned int>(out_rows));
    const dim3 block(static_cast<unsigned int>(kBlockSize));

    switch (dtype) {
        case WeightDType::Int8:
            matmul_dequant_int8act_int8_kernel<<<grid, block, 0, st>>>(
                static_cast<const int8_t*>(weight_device), weight_scale_device, activation_device,
                act_scale, out_device, cols);
            break;
        case WeightDType::Int4:
            matmul_dequant_int8act_int4_kernel<<<grid, block, 0, st>>>(
                static_cast<const uint8_t*>(weight_device), weight_scale_device, activation_device,
                act_scale, out_device, cols);
            break;
        default:
            return false;
    }

    return ctx.batching() ? true : (hipStreamSynchronize(st) == hipSuccess);
}

bool rmsnorm(Context& ctx, const float* x_device, const float* w_device, float* y_device, int32_t n,
             float eps) noexcept {
    if (!ctx.is_valid() || x_device == nullptr || w_device == nullptr || y_device == nullptr ||
        n <= 0) {
        return false;
    }
    const hipStream_t st = static_cast<hipStream_t>(ctx.stream());
    rmsnorm_kernel<<<dim3(1), dim3(kBlockSize), 0, st>>>(x_device, w_device, y_device, n, eps);
    return ctx.batching() ? true : (hipStreamSynchronize(st) == hipSuccess);
}

bool rope(Context& ctx, float* x_device, int32_t head_dim, int32_t num_heads, int32_t position,
          float theta) noexcept {
    if (!ctx.is_valid() || x_device == nullptr || head_dim <= 0 || num_heads <= 0) return false;
    const hipStream_t st = static_cast<hipStream_t>(ctx.stream());
    const int32_t total = num_heads * (head_dim / 2);
    const dim3 grid(static_cast<unsigned int>((total + kBlockSize - 1) / kBlockSize));
    rope_kernel<<<grid, dim3(kBlockSize), 0, st>>>(x_device, head_dim, num_heads, position, theta);
    return ctx.batching() ? true : (hipStreamSynchronize(st) == hipSuccess);
}

bool elementwise(Context& ctx, const float* a_device, const float* b_device, float* out_device,
                 int32_t n, int32_t op) noexcept {
    if (!ctx.is_valid() || a_device == nullptr || b_device == nullptr || out_device == nullptr ||
        n <= 0) {
        return false;
    }
    const hipStream_t st = static_cast<hipStream_t>(ctx.stream());
    const dim3 grid(static_cast<unsigned int>((n + kBlockSize - 1) / kBlockSize));
    elementwise_kernel<<<grid, dim3(kBlockSize), 0, st>>>(a_device, b_device, out_device, n, op);
    return ctx.batching() ? true : (hipStreamSynchronize(st) == hipSuccess);
}

bool attention(Context& ctx, const float* q_device, const float* k_cache_device,
               const float* v_cache_device, float* out_device, int32_t num_heads,
               int32_t num_kv_heads, int32_t head_dim, int32_t seq_len, float scale) noexcept {
    if (!ctx.is_valid() || q_device == nullptr || k_cache_device == nullptr ||
        v_cache_device == nullptr || out_device == nullptr) {
        return false;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || seq_len <= 0) return false;
    if (num_heads % num_kv_heads != 0 || head_dim > kBlockSize) return false;
    const int32_t group_size = num_heads / num_kv_heads;
    const hipStream_t st = static_cast<hipStream_t>(ctx.stream());
    int32_t block = 1;
    while (block < head_dim) block <<= 1;  // power-of-two >= head_dim for the tree reduce
    attention_kernel<<<dim3(static_cast<unsigned int>(num_heads)),
                       dim3(static_cast<unsigned int>(block)), 0, st>>>(
        q_device, k_cache_device, v_cache_device, out_device, head_dim, num_kv_heads, group_size,
        seq_len, scale);
    return ctx.batching() ? true : (hipStreamSynchronize(st) == hipSuccess);
}

bool fused_swiglu(Context& ctx, const void* wg, const void* wu, WeightDType dtype,
                  const float* wg_scale, const float* wu_scale, const float* x, float* act,
                  int32_t rows, int32_t cols) noexcept {
    if (!ctx.is_valid() || wg == nullptr || wu == nullptr || x == nullptr || act == nullptr ||
        rows <= 0 || cols <= 0) {
        return false;
    }
    if (dtype != WeightDType::F32 && dtype != WeightDType::Int8) return false;
    if (dtype == WeightDType::Int8 && (wg_scale == nullptr || wu_scale == nullptr)) return false;

    const hipStream_t st = static_cast<hipStream_t>(ctx.stream());
    const dim3 grid(static_cast<unsigned int>(rows));
    const dim3 block(static_cast<unsigned int>(kBlockSize));
    if (dtype == WeightDType::F32) {
        fused_swiglu_f32_kernel<<<grid, block, 0, st>>>(static_cast<const float*>(wg),
                                                        static_cast<const float*>(wu), x, act, cols);
    } else {
        fused_swiglu_int8_kernel<<<grid, block, 0, st>>>(static_cast<const int8_t*>(wg),
                                                         static_cast<const int8_t*>(wu), wg_scale,
                                                         wu_scale, x, act, cols);
    }
    return ctx.batching() ? true : (hipStreamSynchronize(st) == hipSuccess);
}

}  // namespace bolt::rocm
