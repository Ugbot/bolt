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

// BLLM-189: two-pass attention shared-memory caps. head_dim <= 256 (already
// enforced), scores buffer holds the KV context length. 4096 covers decode
// contexts up to 4K tokens at 16 KB shared; longer falls back to the online
// kernel (which never materializes scores).
constexpr int32_t kMaxHeadDim = 256;
constexpr int32_t kMaxAttnSeq = 4096;

// BLLM-190: sum-reduce a value across the wavefront via shuffle -- no shared
// memory, no __syncthreads. Width-agnostic (folds `warpSize` lanes, 32 on RDNA
// / 64 on CDNA). This is the reduction the fast GEMVs below use instead of a
// block-wide shared-memory tree (which cost ~log2(256) syncs for a few bytes of
// per-thread work on the memory-bound decode GEMVs -- the lm_head bottleneck).
__device__ inline float warp_reduce_sum(float v) {
    for (int32_t offset = warpSize / 2; offset > 0; offset >>= 1) v += __shfl_down(v, offset);
    return v;
}

// BLLM-190: ONE WAVEFRONT per output row (llama.cpp mul_mat_vec shape) instead
// of one whole block -- the lanes stride over `cols`, then a shuffle reduction
// (no sync). A block packs blockDim/warpSize rows, so occupancy is high even at
// small row counts and there are zero __syncthreads on the GEMV hot path.
// Templated on Accumulate (bolt compile-time dispatch): <false> writes
// out[row]=dot, <true> writes out[row]+=dot (fuses the residual-add).
template <bool Accumulate>
__global__ void matmul_dequant_f32_kernel(const float* weight, const float* activation, float* out,
                                          int32_t rows, int32_t cols) {
    const int32_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
    if (row >= rows) return;
    const int32_t lane = threadIdx.x % warpSize;
    const float* row_ptr = weight + static_cast<int64_t>(row) * cols;
    float acc = 0.0f;
    for (int32_t i = lane; i < cols; i += warpSize) acc += activation[i] * row_ptr[i];
    acc = warp_reduce_sum(acc);
    if (lane == 0) out[row] = Accumulate ? (out[row] + acc) : acc;
}

template <bool Accumulate>
__global__ void matmul_dequant_int8_kernel(const int8_t* weight, const float* scale,
                                            const float* activation, float* out, int32_t rows,
                                            int32_t cols) {
    const int32_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
    if (row >= rows) return;
    const int32_t lane = threadIdx.x % warpSize;
    const int8_t* row_ptr = weight + static_cast<int64_t>(row) * cols;
    float acc = 0.0f;
    // BLLM-190: vectorized fast path -- each lane loads 4 int8 weights as one
    // uint32 and 4 activations as one float4, 4 MACs/iter (4x fewer loads, more
    // ILP -> closer to the bandwidth wall). Requires cols % 4 == 0 so every row
    // starts 4-byte aligned; the scalar warp loop handles any remainder / odd cols.
    const int32_t cols4 = cols & ~3;
    if ((cols & 3) == 0) {
        const uint32_t* w4 = reinterpret_cast<const uint32_t*>(row_ptr);
        const float4* x4 = reinterpret_cast<const float4*>(activation);
        for (int32_t j = lane; j < cols4 / 4; j += warpSize) {
            const uint32_t w = w4[j];
            const float4 x = x4[j];
            acc += x.x * static_cast<float>(static_cast<int8_t>(w & 0xffu)) +
                   x.y * static_cast<float>(static_cast<int8_t>((w >> 8) & 0xffu)) +
                   x.z * static_cast<float>(static_cast<int8_t>((w >> 16) & 0xffu)) +
                   x.w * static_cast<float>(static_cast<int8_t>((w >> 24) & 0xffu));
        }
    } else {
        for (int32_t i = lane; i < cols4; i += warpSize) acc += activation[i] * static_cast<float>(row_ptr[i]);
        for (int32_t i = cols4 + lane; i < cols; i += warpSize) acc += activation[i] * static_cast<float>(row_ptr[i]);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        const float v = acc * scale[row];
        out[row] = Accumulate ? (out[row] + v) : v;
    }
}

// Int4: two 4-bit codes packed per byte, low nibble first -- same layout
// bolt_compute.h's matmul_row_dot_f32 (Int4 case) and boltllm::quant's
// packing convention use.
template <bool Accumulate>
__global__ void matmul_dequant_int4_kernel(const uint8_t* weight, const float* scale,
                                            const float* activation, float* out, int32_t rows,
                                            int32_t cols) {
    const int32_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
    if (row >= rows) return;
    const int32_t lane = threadIdx.x % warpSize;
    const int32_t packed_cols = (cols + 1) / 2;
    const uint8_t* row_ptr = weight + static_cast<int64_t>(row) * packed_cols;
    float acc = 0.0f;
    // BLLM-190: vectorized -- one uint32 = 4 packed bytes = 8 nibbles (cols),
    // matched with two float4 activations. nibble k = (w >> 4k) & 0xF.
    if ((cols & 7) == 0) {
        const uint32_t* w8 = reinterpret_cast<const uint32_t*>(row_ptr);
        const float4* x4 = reinterpret_cast<const float4*>(activation);
        for (int32_t j = lane; j < cols / 8; j += warpSize) {
            const uint32_t w = w8[j];
            const float4 xa = x4[2 * j];
            const float4 xb = x4[2 * j + 1];
            acc += xa.x * static_cast<float>(static_cast<int32_t>(w & 0xFu) - kInt4DequantOffset) +
                   xa.y * static_cast<float>(static_cast<int32_t>((w >> 4) & 0xFu) - kInt4DequantOffset) +
                   xa.z * static_cast<float>(static_cast<int32_t>((w >> 8) & 0xFu) - kInt4DequantOffset) +
                   xa.w * static_cast<float>(static_cast<int32_t>((w >> 12) & 0xFu) - kInt4DequantOffset) +
                   xb.x * static_cast<float>(static_cast<int32_t>((w >> 16) & 0xFu) - kInt4DequantOffset) +
                   xb.y * static_cast<float>(static_cast<int32_t>((w >> 20) & 0xFu) - kInt4DequantOffset) +
                   xb.z * static_cast<float>(static_cast<int32_t>((w >> 24) & 0xFu) - kInt4DequantOffset) +
                   xb.w * static_cast<float>(static_cast<int32_t>((w >> 28) & 0xFu) - kInt4DequantOffset);
        }
    } else {
        for (int32_t i = lane; i < cols; i += warpSize) {
            const uint8_t byte = row_ptr[i >> 1];
            const int32_t nibble = (i & 1) ? (byte >> 4) : (byte & 0x0F);
            acc += activation[i] * static_cast<float>(nibble - kInt4DequantOffset);
        }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        const float v = acc * scale[row];
        out[row] = Accumulate ? (out[row] + v) : v;
    }
}

// BLLM-200: GPU MXFP4 matvec -- the expert-weight tier for gpt-oss (MoE). The
// experts stay resident on the iGPU in their native MXFP4 (0.53 B/weight, ~60GB
// fits the UMA); this kernel dequants IN THE DOT (E8M0 block scale 2^(e-127) x
// E2M1 codebook), one wavefront per output row, lanes striding the cols/32
// 17-byte blocks, shuffle reduction. Same MXFP4 layout as GgufDequant.cpp's
// dequant_block_mxfp4 and the CPU core's mxfp4_row_dot.
__device__ const float kMxfp4Dev[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__global__ void matmul_mxfp4_kernel(const uint8_t* weight, const float* activation, float* out,
                                    int32_t rows, int32_t cols) {
    const int32_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
    if (row >= rows) return;
    const int32_t lane = threadIdx.x % warpSize;
    const int32_t nb = cols >> 5;  // cols / 32 blocks
    const uint8_t* wr = weight + static_cast<int64_t>(row) * nb * 17;
    float acc = 0.0f;
    for (int32_t b = lane; b < nb; b += warpSize) {
        const uint8_t* blk = wr + static_cast<int64_t>(b) * 17;
        const float scale = exp2f(static_cast<float>(static_cast<int32_t>(blk[0]) - 127));
        const uint8_t* qs = blk + 1;
        const float* xb = activation + static_cast<int64_t>(b) * 32;
        float ba = 0.0f;
        for (int32_t j = 0; j < 16; ++j)
            ba += kMxfp4Dev[qs[j] & 0x0F] * xb[j] + kMxfp4Dev[qs[j] >> 4] * xb[j + 16];
        acc += scale * ba;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) out[row] = acc;
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
// per query head; blockDim is a power-of-two >= head_dim, one thread per dim.
// Kept as the LARGE-CONTEXT fallback (scores never materialized) for
// seq_len > kMaxAttnSeq; the two-pass kernel below is the fast path.
__global__ void attention_online_kernel(const float* q, const float* k_cache, const float* v_cache,
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

// BLLM-189: two-pass decode attention -- the fast path. The online kernel does
// a block reduction PER timestep (~seq_len * log2(blockDim) __syncthreads, the
// #1 decode sink in the profile). This kernel materializes the scores in shared
// memory and does just TWO block reductions total (max, then sum), then a
// sync-free weighted-V pass:
//   pass 1: 256 threads stride over timesteps, each computes a full QK dot ->
//           scores[t] (q cached in shared, no per-timestep sync)
//   reduce: block max over scores; block sum of exp(scores-max)
//   pass 2: 256 threads stride over head dims, each sums weight[t]*V[t][d]
// Mathematically identical to the online softmax; ~20 syncs vs ~768. Requires
// seq_len <= kMaxAttnSeq and head_dim <= kMaxHeadDim (host dispatches to the
// online kernel otherwise). GQA: query head h reads kv head h/group_size.
__global__ void attention_twopass_kernel(const float* q, const float* k_cache,
                                         const float* v_cache, float* out, int32_t head_dim,
                                         int32_t num_kv_heads, int32_t group_size, int32_t seq_len,
                                         float scale) {
    __shared__ float q_sh[kMaxHeadDim];
    __shared__ float scores[kMaxAttnSeq];
    __shared__ float red[kBlockSize];
    const int32_t h = blockIdx.x;
    const int32_t kv_head = h / group_size;
    const int32_t tid = threadIdx.x;

    for (int32_t i = tid; i < head_dim; i += blockDim.x) q_sh[i] = q[static_cast<int64_t>(h) * head_dim + i];
    __syncthreads();

    // Pass 1: scores[t] = scale * dot(q, K[t]).  No per-timestep sync.
    for (int32_t t = tid; t < seq_len; t += blockDim.x) {
        const float* kt = k_cache + (static_cast<int64_t>(t) * num_kv_heads + kv_head) * head_dim;
        float s = 0.0f;
        for (int32_t d = 0; d < head_dim; ++d) s += q_sh[d] * kt[d];
        scores[t] = s * scale;
    }
    __syncthreads();

    // Block max over scores.
    float pm = -INFINITY;
    for (int32_t t = tid; t < seq_len; t += blockDim.x) pm = fmaxf(pm, scores[t]);
    red[tid] = pm;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] = fmaxf(red[tid], red[tid + stride]);
        __syncthreads();
    }
    const float mmax = red[0];
    __syncthreads();

    // exp(scores-max) in place + block sum -> denominator.
    float pl = 0.0f;
    for (int32_t t = tid; t < seq_len; t += blockDim.x) {
        const float p = expf(scores[t] - mmax);
        scores[t] = p;
        pl += p;
    }
    red[tid] = pl;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    const float inv_l = 1.0f / red[0];
    __syncthreads();

    // Pass 2: out[d] = inv_l * sum_t scores[t] * V[t][d].  No sync.
    for (int32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int32_t t = 0; t < seq_len; ++t)
            acc += scores[t] * v_cache[(static_cast<int64_t>(t) * num_kv_heads + kv_head) * head_dim + d];
        out[static_cast<int64_t>(h) * head_dim + d] = acc * inv_l;
    }
}

// BLLM-189: fused gate+up+SwiGLU FFN kernel. ONE block per FFN row computes BOTH
// gate=dot(Wg[row],x) and up=dot(Wu[row],x) over the shared activation in a
// single pass, then act[row]=silu(gate)*up. Replaces 3 kernels (gate matmul, up
// matmul, SwiGLU elementwise) + 2 intermediate buffers with 1 launch -- the
// op-fusion lever (fewer serial data-dependent kernels) the resident-graph decode
// measurement identified (docs/perf/gpu-backends-parity.md). Branch-free inner
// loop, two shared-mem tree reductions. F32-weight tier.
// BLLM-190: one wavefront per FFN row, both gate & up dots reduced by shuffle
// (no sync). silu(gate)*up.
__global__ void fused_swiglu_f32_kernel(const float* wg, const float* wu, const float* x,
                                        float* act, int32_t rows, int32_t cols) {
    const int32_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
    if (row >= rows) return;
    const int32_t lane = threadIdx.x % warpSize;
    const float* gr = wg + static_cast<int64_t>(row) * cols;
    const float* ur = wu + static_cast<int64_t>(row) * cols;
    float ga = 0.0f, ua = 0.0f;
    for (int32_t i = lane; i < cols; i += warpSize) {
        const float xi = x[i];
        ga += xi * gr[i];
        ua += xi * ur[i];
    }
    ga = warp_reduce_sum(ga);
    ua = warp_reduce_sum(ua);
    if (lane == 0) act[row] = (ga / (1.0f + expf(-ga))) * ua;
}

// Int8-weight twin: per-row scales, one scale multiply per dot (matches
// matmul_dequant_int8_kernel's contract byte-for-byte).
__global__ void fused_swiglu_int8_kernel(const int8_t* wg, const int8_t* wu, const float* wg_scale,
                                         const float* wu_scale, const float* x, float* act,
                                         int32_t rows, int32_t cols) {
    const int32_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
    if (row >= rows) return;
    const int32_t lane = threadIdx.x % warpSize;
    const int8_t* gr = wg + static_cast<int64_t>(row) * cols;
    const int8_t* ur = wu + static_cast<int64_t>(row) * cols;
    float ga = 0.0f, ua = 0.0f;
    if ((cols & 3) == 0) {  // BLLM-190: 4 int8 as uint32 + float4 activation
        const uint32_t* g4 = reinterpret_cast<const uint32_t*>(gr);
        const uint32_t* u4 = reinterpret_cast<const uint32_t*>(ur);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int32_t j = lane; j < cols / 4; j += warpSize) {
            const uint32_t g = g4[j], u = u4[j];
            const float4 xv = x4[j];
            ga += xv.x * static_cast<float>(static_cast<int8_t>(g & 0xffu)) +
                  xv.y * static_cast<float>(static_cast<int8_t>((g >> 8) & 0xffu)) +
                  xv.z * static_cast<float>(static_cast<int8_t>((g >> 16) & 0xffu)) +
                  xv.w * static_cast<float>(static_cast<int8_t>((g >> 24) & 0xffu));
            ua += xv.x * static_cast<float>(static_cast<int8_t>(u & 0xffu)) +
                  xv.y * static_cast<float>(static_cast<int8_t>((u >> 8) & 0xffu)) +
                  xv.z * static_cast<float>(static_cast<int8_t>((u >> 16) & 0xffu)) +
                  xv.w * static_cast<float>(static_cast<int8_t>((u >> 24) & 0xffu));
        }
    } else {
        for (int32_t i = lane; i < cols; i += warpSize) {
            const float xi = x[i];
            ga += xi * static_cast<float>(gr[i]);
            ua += xi * static_cast<float>(ur[i]);
        }
    }
    ga = warp_reduce_sum(ga);
    ua = warp_reduce_sum(ua);
    if (lane == 0) {
        const float g = ga * wg_scale[row];
        act[row] = (g / (1.0f + expf(-g))) * (ua * wu_scale[row]);
    }
}

// Int4-weight twin: nibble-unpack (low-first, offset kInt4DequantOffset) both Wg
// and Wu, per-row scales. Reads 0.5 B/weight -- the bandwidth tier.
__global__ void fused_swiglu_int4_kernel(const uint8_t* wg, const uint8_t* wu,
                                         const float* wg_scale, const float* wu_scale,
                                         const float* x, float* act, int32_t rows, int32_t cols) {
    const int32_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
    if (row >= rows) return;
    const int32_t lane = threadIdx.x % warpSize;
    const int32_t packed = (cols + 1) / 2;
    const uint8_t* gr = wg + static_cast<int64_t>(row) * packed;
    const uint8_t* ur = wu + static_cast<int64_t>(row) * packed;
    float ga = 0.0f, ua = 0.0f;
    if ((cols & 7) == 0) {  // BLLM-190: uint32 = 8 nibbles, two float4 activations
        const uint32_t* g8 = reinterpret_cast<const uint32_t*>(gr);
        const uint32_t* u8 = reinterpret_cast<const uint32_t*>(ur);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int32_t j = lane; j < cols / 8; j += warpSize) {
            const uint32_t g = g8[j], u = u8[j];
            const float4 xa = x4[2 * j];
            const float4 xb = x4[2 * j + 1];
            const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
            for (int32_t n = 0; n < 8; ++n) {
                const int32_t sh = 4 * n;
                ga += xs[n] * static_cast<float>(static_cast<int32_t>((g >> sh) & 0xFu) - kInt4DequantOffset);
                ua += xs[n] * static_cast<float>(static_cast<int32_t>((u >> sh) & 0xFu) - kInt4DequantOffset);
            }
        }
    } else {
        for (int32_t i = lane; i < cols; i += warpSize) {
            const float xi = x[i];
            const int32_t sh = (i & 1) * 4;  // 0 for low nibble, 4 for high
            ga += xi * static_cast<float>(static_cast<int32_t>((gr[i >> 1] >> sh) & 0x0F) - kInt4DequantOffset);
            ua += xi * static_cast<float>(static_cast<int32_t>((ur[i >> 1] >> sh) & 0x0F) - kInt4DequantOffset);
        }
    }
    ga = warp_reduce_sum(ga);
    ua = warp_reduce_sum(ua);
    if (lane == 0) {
        const float g = ga * wg_scale[row];
        act[row] = (g / (1.0f + expf(-g))) * (ua * wu_scale[row]);
    }
}

// BLLM-189: fused QKV projection. ONE block per output row across the
// concatenated [q | k | v] rows -- one launch replaces three matmuls that all
// read the same normed activation. The which-tensor selection is uniform per
// block (constant across the block's threads), so there is no intra-block
// divergence and the hot dot loop stays branch-free. F32-weight tier.
__global__ void fused_qkv_f32_kernel(const float* wq, const float* wk, const float* wv,
                                     const float* x, float* q, float* k, float* v, int32_t q_rows,
                                     int32_t kv_rows, int32_t cols) {
    const int32_t total = q_rows + 2 * kv_rows;
    const int32_t b = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
    if (b >= total) return;
    const int32_t lane = threadIdx.x % warpSize;
    const float* wrow;
    float* out;
    int32_t orow;
    if (b < q_rows) {
        wrow = wq + static_cast<int64_t>(b) * cols; out = q; orow = b;
    } else if (b < q_rows + kv_rows) {
        orow = b - q_rows; wrow = wk + static_cast<int64_t>(orow) * cols; out = k;
    } else {
        orow = b - q_rows - kv_rows; wrow = wv + static_cast<int64_t>(orow) * cols; out = v;
    }
    float acc = 0.0f;
    for (int32_t i = lane; i < cols; i += warpSize) acc += x[i] * wrow[i];
    acc = warp_reduce_sum(acc);
    if (lane == 0) out[orow] = acc;
}

// Int8-weight twin: each of q/k/v carries its own per-row scale array. Warp-per-
// row + vectorized (4 int8 as uint32, float4 activation) when cols % 4 == 0.
__global__ void fused_qkv_int8_kernel(const int8_t* wq, const int8_t* wk, const int8_t* wv,
                                      const float* sq, const float* sk, const float* sv,
                                      const float* x, float* q, float* k, float* v, int32_t q_rows,
                                      int32_t kv_rows, int32_t cols) {
    const int32_t total = q_rows + 2 * kv_rows;
    const int32_t b = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
    if (b >= total) return;
    const int32_t lane = threadIdx.x % warpSize;
    const int8_t* wrow;
    const float* scl;
    float* out;
    int32_t orow;
    if (b < q_rows) {
        wrow = wq + static_cast<int64_t>(b) * cols; scl = sq; out = q; orow = b;
    } else if (b < q_rows + kv_rows) {
        orow = b - q_rows; wrow = wk + static_cast<int64_t>(orow) * cols; scl = sk; out = k;
    } else {
        orow = b - q_rows - kv_rows; wrow = wv + static_cast<int64_t>(orow) * cols; scl = sv; out = v;
    }
    float acc = 0.0f;
    if ((cols & 3) == 0) {
        const uint32_t* w4 = reinterpret_cast<const uint32_t*>(wrow);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int32_t j = lane; j < cols / 4; j += warpSize) {
            const uint32_t w = w4[j];
            const float4 xv = x4[j];
            acc += xv.x * static_cast<float>(static_cast<int8_t>(w & 0xffu)) +
                   xv.y * static_cast<float>(static_cast<int8_t>((w >> 8) & 0xffu)) +
                   xv.z * static_cast<float>(static_cast<int8_t>((w >> 16) & 0xffu)) +
                   xv.w * static_cast<float>(static_cast<int8_t>((w >> 24) & 0xffu));
        }
    } else {
        for (int32_t i = lane; i < cols; i += warpSize) acc += x[i] * static_cast<float>(wrow[i]);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) out[orow] = acc * scl[orow];
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
    const dim3 block(static_cast<unsigned int>(kBlockSize));
    // BLLM-190: warp-per-row grid -- blockDim/warpSize rows per block.
    int32_t wps = ctx.wavefront_size();
    if (wps <= 0) wps = 32;
    const int32_t wpb = kBlockSize / wps;
    const dim3 wgrid(static_cast<unsigned int>((out_rows + wpb - 1) / wpb));
    const dim3 rgrid(static_cast<unsigned int>(out_rows));  // Int2: one block per row (legacy)

    switch (dtype) {
        case WeightDType::F32:
            matmul_dequant_f32_kernel<false><<<wgrid, block, 0, st>>>(
                static_cast<const float*>(weight_device), activation_device, out_device, out_rows,
                cols);
            break;
        case WeightDType::Int8:
            matmul_dequant_int8_kernel<false><<<wgrid, block, 0, st>>>(
                static_cast<const int8_t*>(weight_device), scale_device, activation_device,
                out_device, out_rows, cols);
            break;
        case WeightDType::Int4:
            matmul_dequant_int4_kernel<false><<<wgrid, block, 0, st>>>(
                static_cast<const uint8_t*>(weight_device), scale_device, activation_device,
                out_device, out_rows, cols);
            break;
        case WeightDType::Int2:
            matmul_dequant_int2_kernel<<<rgrid, block, 0, st>>>(
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
    if (dtype != WeightDType::F32 && scale_device == nullptr) return false;

    const hipStream_t st = static_cast<hipStream_t>(ctx.stream());
    const dim3 block(static_cast<unsigned int>(kBlockSize));
    int32_t wps = ctx.wavefront_size();
    if (wps <= 0) wps = 32;
    const int32_t wpb = kBlockSize / wps;
    const dim3 wgrid(static_cast<unsigned int>((out_rows + wpb - 1) / wpb));
    // out[row] += dot(...) -- fuses the residual-add into the projection. F32/Int8/
    // Int4 (the resident-graph decode tiers); Int2 keeps the unfused path.
    if (dtype == WeightDType::F32) {
        matmul_dequant_f32_kernel<true><<<wgrid, block, 0, st>>>(
            static_cast<const float*>(weight_device), activation_device, out_device, out_rows, cols);
    } else if (dtype == WeightDType::Int8) {
        matmul_dequant_int8_kernel<true><<<wgrid, block, 0, st>>>(
            static_cast<const int8_t*>(weight_device), scale_device, activation_device, out_device,
            out_rows, cols);
    } else if (dtype == WeightDType::Int4) {
        matmul_dequant_int4_kernel<true><<<wgrid, block, 0, st>>>(
            static_cast<const uint8_t*>(weight_device), scale_device, activation_device, out_device,
            out_rows, cols);
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
    const dim3 grid(static_cast<unsigned int>(num_heads));
    if (seq_len <= kMaxAttnSeq && head_dim <= kMaxHeadDim) {
        // BLLM-189 fast path: two block reductions total instead of one per timestep.
        attention_twopass_kernel<<<grid, dim3(static_cast<unsigned int>(kBlockSize)), 0, st>>>(
            q_device, k_cache_device, v_cache_device, out_device, head_dim, num_kv_heads, group_size,
            seq_len, scale);
    } else {  // large-context fallback: never materializes the scores
        int32_t block = 1;
        while (block < head_dim) block <<= 1;  // power-of-two >= head_dim for the tree reduce
        attention_online_kernel<<<grid, dim3(static_cast<unsigned int>(block)), 0, st>>>(
            q_device, k_cache_device, v_cache_device, out_device, head_dim, num_kv_heads, group_size,
            seq_len, scale);
    }
    return ctx.batching() ? true : (hipStreamSynchronize(st) == hipSuccess);
}

bool fused_swiglu(Context& ctx, const void* wg, const void* wu, WeightDType dtype,
                  const float* wg_scale, const float* wu_scale, const float* x, float* act,
                  int32_t rows, int32_t cols) noexcept {
    if (!ctx.is_valid() || wg == nullptr || wu == nullptr || x == nullptr || act == nullptr ||
        rows <= 0 || cols <= 0) {
        return false;
    }
    if (dtype != WeightDType::F32 && dtype != WeightDType::Int8 && dtype != WeightDType::Int4)
        return false;
    if (dtype != WeightDType::F32 && (wg_scale == nullptr || wu_scale == nullptr)) return false;

    const hipStream_t st = static_cast<hipStream_t>(ctx.stream());
    const dim3 block(static_cast<unsigned int>(kBlockSize));
    int32_t wps = ctx.wavefront_size();
    if (wps <= 0) wps = 32;
    const int32_t wpb = kBlockSize / wps;
    const dim3 grid(static_cast<unsigned int>((rows + wpb - 1) / wpb));  // warp-per-row
    if (dtype == WeightDType::F32) {
        fused_swiglu_f32_kernel<<<grid, block, 0, st>>>(static_cast<const float*>(wg),
                                                        static_cast<const float*>(wu), x, act, rows,
                                                        cols);
    } else if (dtype == WeightDType::Int8) {
        fused_swiglu_int8_kernel<<<grid, block, 0, st>>>(static_cast<const int8_t*>(wg),
                                                         static_cast<const int8_t*>(wu), wg_scale,
                                                         wu_scale, x, act, rows, cols);
    } else {
        fused_swiglu_int4_kernel<<<grid, block, 0, st>>>(static_cast<const uint8_t*>(wg),
                                                         static_cast<const uint8_t*>(wu), wg_scale,
                                                         wu_scale, x, act, rows, cols);
    }
    return ctx.batching() ? true : (hipStreamSynchronize(st) == hipSuccess);
}

bool fused_qkv(Context& ctx, const void* wq, const void* wk, const void* wv, WeightDType dtype,
               const float* sq, const float* sk, const float* sv, const float* x, float* q,
               float* k, float* v, int32_t q_rows, int32_t kv_rows, int32_t cols) noexcept {
    if (!ctx.is_valid() || wq == nullptr || wk == nullptr || wv == nullptr || x == nullptr ||
        q == nullptr || k == nullptr || v == nullptr || q_rows <= 0 || kv_rows <= 0 || cols <= 0) {
        return false;
    }
    if (dtype != WeightDType::F32 && dtype != WeightDType::Int8) return false;
    if (dtype == WeightDType::Int8 && (sq == nullptr || sk == nullptr || sv == nullptr)) return false;

    const hipStream_t st = static_cast<hipStream_t>(ctx.stream());
    const dim3 block(static_cast<unsigned int>(kBlockSize));
    int32_t wps = ctx.wavefront_size();
    if (wps <= 0) wps = 32;
    const int32_t wpb = kBlockSize / wps;
    const dim3 grid(static_cast<unsigned int>((q_rows + 2 * kv_rows + wpb - 1) / wpb));  // warp-per-row
    if (dtype == WeightDType::F32) {
        fused_qkv_f32_kernel<<<grid, block, 0, st>>>(
            static_cast<const float*>(wq), static_cast<const float*>(wk),
            static_cast<const float*>(wv), x, q, k, v, q_rows, kv_rows, cols);
    } else {
        fused_qkv_int8_kernel<<<grid, block, 0, st>>>(
            static_cast<const int8_t*>(wq), static_cast<const int8_t*>(wk),
            static_cast<const int8_t*>(wv), sq, sk, sv, x, q, k, v, q_rows, kv_rows, cols);
    }
    return ctx.batching() ? true : (hipStreamSynchronize(st) == hipSuccess);
}

bool matmul_mxfp4(Context& ctx, const void* weight_device, const float* activation_device,
                  float* out_device, int32_t out_rows, int32_t cols) noexcept {
    if (!ctx.is_valid() || weight_device == nullptr || activation_device == nullptr ||
        out_device == nullptr || out_rows <= 0 || cols <= 0 || (cols & 31) != 0) {
        return false;
    }
    const hipStream_t st = static_cast<hipStream_t>(ctx.stream());
    int32_t wps = ctx.wavefront_size();
    if (wps <= 0) wps = 32;
    const int32_t wpb = kBlockSize / wps;
    const dim3 grid(static_cast<unsigned int>((out_rows + wpb - 1) / wpb));
    const dim3 block(static_cast<unsigned int>(kBlockSize));
    matmul_mxfp4_kernel<<<grid, block, 0, st>>>(static_cast<const uint8_t*>(weight_device),
                                                activation_device, out_device, out_rows, cols);
    return ctx.batching() ? true : (hipStreamSynchronize(st) == hipSuccess);
}

}  // namespace bolt::rocm
