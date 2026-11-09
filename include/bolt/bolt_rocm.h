#pragma once
// bolt_rocm.h -- BLLM-78: ROCm/HIP matmul+dequant kernel core behind
// bolt::compute's reserved ROCm device slot (see bolt_compute.h's `Device`
// enum and its per-device dispatch tables -- BLLM-13).
//
// Unlike bolt::vulkan (BLLM-46, device/queue/buffer plumbing ONLY, kernels
// deferred to BLLM-47), this module ships the actual compute kernel: a
// GEMV (matmul) with on-the-fly F32/Int8/Int4/Int2 dequant, numerically
// identical to bolt_compute.h's `detail::matmul_row_dot_f32` CPU reference
// (same kInt4/kInt2 offsets, same per-row `scale` convention) but executed
// as a real HIP `__global__` kernel, one thread-block per output row.
//
// Runtime-detected, CPU fallback always works: `available()` probes for a
// real ROCm-capable device and returns false (never asserts/aborts) if
// none exists. `Context::create()` follows the same contract.
//
// STATUS (2026-07-16): structurally complete, written against AMD's
// documented HIP runtime API, but UNVALIDATED against a real hipcc/device
// -- this dev box (AMD Ryzen AI Max+ 395 / Radeon 8060S, gfx1151) has no
// ROCm/HIP SDK for native Windows (no HIP_PATH, no hipcc, no ROCm install
// dir; see BLLM-78 tracker notes for the ds4-confirmed Linux-only-driver
// constraint on this exact chip). CMakeLists.txt only builds bolt::rocm
// when it finds a real hipcc, so this never blocks anyone else's build;
// hardware validation is deferred to whoever next has a working ROCm
// toolchain against this or an equivalent gfx target.
//
// Compiled (not header-only): HIP kernels require hipcc, so this is a
// separate optional compiled library gated on BOLT_BUILD_ROCM, same shape
// as bolt::parse/bolt::vulkan. Handles are opaque; only bolt_rocm.cpp
// includes <hip/hip_runtime.h>.

#include <cstdint>

namespace bolt::rocm {

// True iff a ROCm-capable device is present and hipGetDeviceCount succeeds
// with a non-zero count. Intended for one-time startup capability
// detection, not a hot-path check.
bool available() noexcept;

// One device-resident allocation (a HIP device pointer, opaque here).
struct DeviceBuffer {
    void*    ptr = nullptr;
    uint64_t size_bytes = 0;
};

// Weight dtype tag mirroring the subset of bolt::DType (bolt_tensor.h)
// that bolt_compute.h's matmul_row_dot_f32 dequantizes today (see that
// file's architectural choice #1). Restated by value (not #include
// bolt_tensor.h) so callers that only want the ROCm kernel don't have to
// link bolt::core's Tensor type -- kept in lock-step by value with
// bolt::DType; F16/BF16 are intentionally absent (bolt_compute.h's own
// CPU tier has no implementation for them yet either).
enum class WeightDType : uint8_t {
    F32  = 0,
    Int8 = 1,
    Int4 = 2,
    Int2 = 3,
};

// Owns one HIP device context (device selection + capability query).
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
    // no ROCm runtime, no device, or the runtime rejecting device
    // selection. May be called at most once per instance.
    bool create() noexcept;

    // Resets to the inert state; safe to call on a Context that was never
    // created or whose create() returned false.
    void destroy() noexcept;

    bool is_valid() const noexcept { return valid_; }

    // Valid only if is_valid().
    const char* device_name() const noexcept { return device_name_; }
    uint64_t device_local_heap_bytes() const noexcept { return device_local_heap_bytes_; }
    // Reported by hipDeviceProp_t::warpSize -- 32 or 64 depending on
    // gfx target/compile mode. Informational only: matmul_dequant()'s
    // kernel uses a runtime block-wide shared-memory reduction (see
    // bolt_rocm.cpp) rather than a wavefront-width-assuming shuffle, so
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

    // BLLM-188: batch mode for the resident-graph forward. Between begin_batch()
    // and end_batch(), the compute ops (matmul_dequant/rmsnorm/rope/elementwise/
    // attention) skip their per-op hipDeviceSynchronize -- the kernels serialize
    // in issue order on the default stream, so correctness is preserved and the
    // whole layer/graph costs ONE sync instead of one-per-op (this is the lever
    // the per-op GEMV bench, docs/perf/gpu-backends-parity.md, identified as the
    // decode-throughput cap). end_batch() performs the single sync and returns
    // false iff any batched launch failed. Nesting is not supported; begin then
    // end. Outside a batch (the default) every op syncs as before.
    void begin_batch() noexcept { batch_mode_ = true; }
    bool end_batch() noexcept;
    // True between begin_batch() and end_batch(); the free-function ops read
    // this to decide whether to skip their per-op sync.
    bool batching() const noexcept { return batch_mode_; }

    // The HIP stream every op launches on (opaque hipStream_t). Non-default so
    // the op sequence can be captured into a graph. Valid only if is_valid().
    void* stream() const noexcept { return stream_; }

    // BLLM-189: HIP graph capture -- the lever past the per-launch overhead the
    // resident-graph decode measured. begin_graph_capture() starts recording the
    // ops issued on this Context's stream (they skip their per-op sync, like a
    // batch); end_graph_capture() ends recording and instantiates an executable
    // graph; graph_launch() replays that whole captured sequence in ONE
    // launch + sync, amortizing the ~362-launches/token cost to ~0. The captured
    // graph bakes in the exact kernel arguments (device pointers, dims), so it is
    // valid only while those resident buffers + shapes are unchanged -- re-capture
    // if the position/shape changes. Returns false on any HIP failure.
    bool begin_graph_capture() noexcept;
    bool end_graph_capture() noexcept;
    bool graph_launch() noexcept;
    bool has_graph() const noexcept { return graph_exec_ != nullptr; }
    void free_graph() noexcept;

private:
    bool     valid_ = false;
    bool     batch_mode_ = false;
    int32_t  device_index_ = -1;
    int32_t  wavefront_size_ = 0;
    char     device_name_[256] = {};
    uint64_t device_local_heap_bytes_ = 0;
    void*    stream_ = nullptr;      // hipStream_t (opaque)
    void*    graph_exec_ = nullptr;  // hipGraphExec_t (opaque)
};

// GEMV: out[r] = dot(weight.row(r), activation), r in [0, out_rows), with
// on-the-fly dequant matching bolt_compute.h's matmul_row_dot_f32 exactly.
// All pointers except `ctx` are DEVICE pointers (obtained via
// Context::allocate + copy_to_device) -- this function launches a kernel
// and blocks until it completes (hipDeviceSynchronize), it does not copy
// host memory itself. `scale_device` may be null only when
// dtype == WeightDType::F32 (no scale used); required otherwise.
// Returns false on invalid arguments or any launch/sync failure.
// Precondition: ctx.is_valid().
// BLLM-200: optional per-row `bias` (nullptr = none) is folded into the output
// (out = dot + bias), removing a separate bias-add launch (router/lm_head/proj).
bool matmul_dequant(Context& ctx, const void* weight_device, WeightDType dtype,
                     const float* scale_device, const float* activation_device,
                     float* out_device, int32_t out_rows, int32_t cols,
                     const float* bias = nullptr) noexcept;

// BLLM-189: like matmul_dequant but ACCUMULATES -- out[r] += dot(weight.row(r),
// activation) -- fusing a residual/skip add into the projection GEMV (removes a
// separate elementwise-add kernel from the decode chain). `out_device` must hold
// the residual on entry. F32 and Int8 weights only (the resident-graph tiers);
// returns false for Int4/Int2. Precondition: ctx.is_valid().
bool matmul_dequant_residual(Context& ctx, const void* weight_device, WeightDType dtype,
                             const float* scale_device, const float* activation_device,
                             float* out_device, int32_t out_rows, int32_t cols,
                             const float* bias = nullptr) noexcept;

// BLLM-91: int8-ACTIVATION matmul -- the HIP twin of
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

// ---- On-device forward ops (HIP twins of the bolt::vulkan shaders), for the
// resident-graph GPU forward (BLLM-181). All buffers are device pointers from
// Context::allocate(); each blocks until complete (hipDeviceSynchronize). ----

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
// BLLM-200: `sinks_device` (per-head learned softmax-denominator logit; nullptr =
// none) and `window_start` (sliding-window: attend keys [window_start, seq_len);
// 0 = full causal) are optional gpt-oss extensions. Handled on the two-pass path
// (seq_len <= kMaxAttnSeq); the long-context online fallback ignores them.
bool attention(Context& ctx, const float* q_device, const float* k_cache_device,
               const float* v_cache_device, float* out_device, int32_t num_heads,
               int32_t num_kv_heads, int32_t head_dim, int32_t seq_len, float scale,
               const float* sinks_device = nullptr, int32_t window_start = 0) noexcept;

// BLLM-200: gpt-oss expert activation. gate/up are the [n] outputs of two separate
// matmul_mxfp4 calls; gate_bias/up_bias are optional per-column biases (nullptr =
// none). out[i] = (g*sigmoid(alpha*g))*(u+1), g=min(gate+gb,limit),
// u=clamp(up+ub,-limit,limit). gpt-oss uses alpha=1.702, limit=7.
bool swiglu_oai(Context& ctx, const float* gate_device, const float* up_device,
                const float* gate_bias_device, const float* up_bias_device, float* out_device,
                int32_t n, float alpha, float limit) noexcept;

// BLLM-200: NeoX RoPE + optional YaRN, in place on [num_heads, head_dim]. `base`
// is the rope theta; when `yarn` the extrapolate/interpolate blend uses freq_scale
// + beta_fast/beta_slow + orig_ctx (mscale = attn_factor*(1+0.1*ln(1/freq_scale))).
// Matches the CPU rope_neox_yarn exactly. Precondition: head_dim even.
bool rope_neox_yarn(Context& ctx, float* x_device, int32_t head_dim, int32_t num_heads,
                    int32_t position, float base, bool yarn, float freq_scale, float ext_factor,
                    float attn_factor, float beta_fast, float beta_slow, int32_t orig_ctx) noexcept;

// BLLM-200: scaled accumulate with optional bias -- out[i] += scale*(x[i]+bias[i]),
// i in [0,n). The MoE expert-mixing step (weight the down-projection by its routing
// softmax weight, fold in down_bias). bias_device may be null.
bool axpy_bias(Context& ctx, const float* x_device, const float* bias_device, float scale,
               float* out_device, int32_t n) noexcept;

// BLLM-189: fused gate+up+SwiGLU FFN. Computes act[r] = silu(dot(Wg[r],x)) *
// dot(Wu[r],x) for r in [0,rows) in ONE kernel -- the op-fusion lever that
// replaces gate-matmul + up-matmul + SwiGLU-elementwise (3 launches + 2 buffers)
// with 1. `wg`/`wu` are row-major [rows, cols] of dtype (F32 or Int8); for Int8,
// `wg_scale`/`wu_scale` are the per-row scales (required; ignored for F32).
// `x` is the [cols] float activation, `act` the [rows] float output. Same per-row
// scale + dequant contract as matmul_dequant. Precondition: ctx.is_valid().
bool fused_swiglu(Context& ctx, const void* wg, const void* wu, WeightDType dtype,
                  const float* wg_scale, const float* wu_scale, const float* x, float* act,
                  int32_t rows, int32_t cols) noexcept;

// BLLM-200: GPU MXFP4 matvec -- out[r] = dot(dequant(weight.row(r)), activation)
// for r in [0,out_rows). `weight_device` is one expert matrix's resident MXFP4
// bytes ([out_rows, cols], row = (cols/32)*17 bytes); dequant happens in the dot
// (E8M0 scale + E2M1 codebook). `cols` must be a multiple of 32. f32 activation.
// The gpt-oss expert tier (experts stay MXFP4-resident on the iGPU). Precondition:
// ctx.is_valid().
bool matmul_mxfp4(Context& ctx, const void* weight_device, const float* activation_device,
                  float* out_device, int32_t out_rows, int32_t cols) noexcept;

// BLLM-200: bandwidth-optimized MXFP4 matvec. The weight is repacked (at upload)
// into struct-of-arrays: `codes_device` = each 32-element block's 16 E2M1 code
// bytes contiguous (16-byte aligned; row r at codes + r*(cols/32)*16), and
// `scales_device` = the E8M0 scale bytes (row r at scales + r*(cols/32)). Reads
// codes as aligned uint4 + activation as float4 -- same result as matmul_mxfp4 but
// coalesced/vectorized (the iGPU-bandwidth path for the gpt-oss experts). `cols`
// must be a multiple of 32. Precondition: ctx.is_valid().
bool matmul_mxfp4_aligned(Context& ctx, const void* codes_device, const void* scales_device,
                          const float* activation_device, float* out_device, int32_t out_rows,
                          int32_t cols) noexcept;

// BLLM-200 diagnostic: raw device read bandwidth in GB/s (streams `bytes` through a
// grid-stride sum, `iters` times, hipEvent-timed). The memory-bound decode ceiling.
bool bandwidth_probe(Context& ctx, uint64_t bytes, int32_t iters, double* gbps) noexcept;

// BLLM-200: GPU top-K + softmax router selection -- writes sel_device[K] (expert
// indices, top-K of E by raw logit) + selw_device[K] (softmax weights over the K)
// so the MoE runs with NO per-layer host round-trip (was the sole decode sync).
// K <= 64. Precondition: ctx.is_valid().
bool moe_topk_softmax(Context& ctx, const float* logits_device, int32_t* sel_device,
                      float* selw_device, int32_t E, int32_t K) noexcept;

// BLLM-200: FUSED gpt-oss expert gate+up+swiglu_oai in ONE launch, DEVICE-INDEXED:
// expert e = sel[kk] read on-device. gate/up are split codes+scales (aligned) with
// shared [expert_inter, hidden] shape -> shared cstride (bytes/expert of codes) and
// sstride (bytes/expert of scales). gbias/ubias base pointers may be null (per-
// expert bias stride == rows). act[r]=swiglu_oai(dot(gate[e].r,x)+gb, dot(up[e].r,x)+ub).
bool moe_gate_up_swiglu_mxfp4(Context& ctx, const void* gcodes, const void* gscales,
                              const void* ucodes, const void* uscales, int64_t cstride,
                              int64_t sstride, const float* gbias, const float* ubias,
                              const int32_t* sel, int32_t kk, const float* x, float* act,
                              int32_t rows, int32_t cols, float alpha, float limit) noexcept;

// BLLM-200: FUSED expert down-projection + weighted residual accumulate, ONE launch,
// DEVICE-INDEXED. hidden[r] += selw[kk]*(dot(down[e].r,act)+dbias[e][r]), e=sel[kk].
// down is split codes+scales; dbias base may be null (per-expert stride == rows).
bool moe_down_accum_mxfp4(Context& ctx, const void* codes, const void* scales, int64_t cstride,
                          int64_t sstride, const float* dbias, const float* selw, const int32_t* sel,
                          int32_t kk, const float* act, float* hidden, int32_t rows,
                          int32_t cols) noexcept;

// BLLM-200: BATCHED expert kernels -- all K selected experts in ONE launch each
// (vs K separate gate_up + K down). gate_up writes act[K*ei] (row vr -> expert
// sel[vr/ei], local vr%ei); down reads act[kk*ei..] and atomicAdd's each expert's
// weighted contribution into hidden[row]. Fewer, bigger memory-bound kernels ->
// better DRAM saturation. `cols` is the shared-dim (H for gate_up, ei for down).
bool moe_gate_up_batched(Context& ctx, const void* gcodes, const void* gscales, const void* ucodes,
                         const void* uscales, int64_t cstride, int64_t sstride, const float* gbias,
                         const float* ubias, const int32_t* sel, int32_t K, const float* x,
                         float* act, int32_t ei, int32_t cols, float alpha, float limit) noexcept;
bool moe_down_batched(Context& ctx, const void* codes, const void* scales, int64_t cstride,
                      int64_t sstride, const float* dbias, const float* selw, const int32_t* sel,
                      int32_t K, const float* act, float* hidden, int32_t H, int32_t cols) noexcept;

// BLLM-189: fused QKV projection -- one kernel produces q[q_rows], k[kv_rows],
// v[kv_rows] from the shared normed activation `x[cols]`, replacing three
// matmul_dequant calls with one launch. `wq`/`wk`/`wv` are row-major weights of
// dtype (F32 or Int8); for Int8 `sq`/`sk`/`sv` are the per-row scales (required;
// ignored for F32). Same dequant contract as matmul_dequant. Precondition:
// ctx.is_valid().
// BLLM-200: optional per-row biases qb/kb/vb (nullptr = none) are folded into the
// projection -- out = dot + bias -- removing three separate elementwise bias-add
// launches per layer (the gpt-oss dense path is kernel-count bound).
bool fused_qkv(Context& ctx, const void* wq, const void* wk, const void* wv, WeightDType dtype,
               const float* sq, const float* sk, const float* sv, const float* x, float* q,
               float* k, float* v, int32_t q_rows, int32_t kv_rows, int32_t cols,
               const float* qb = nullptr, const float* kb = nullptr,
               const float* vb = nullptr) noexcept;

}  // namespace bolt::rocm
