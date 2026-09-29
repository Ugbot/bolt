# perf-pass-gpu-simd: unmerged BLLM/STATION findings

Preserved 2026-09-29 from bolt branch `perf-pass-gpu-simd` @ 4f497af (35 commits, 2026-07-18..07-25), which is not on main and awaits a decision. The work is for the boltllm / llm-station consumer (BLLM-*/STATION-* tickets): SIMD f32/f16 GEMV, ROCm/CUDA/Vulkan on-device kernels, a byte-faithful JSON writer, reactor timers and async file I/O. Below: its design-log entries verbatim, then the commit log (subjects carry the measured tokens/s).


## 2026-07-18 / perf-pass — f32/f16 float-activation SIMD GEMV (compute_gemv.h)

LLM decode is memory-bandwidth-bound: every weight is read once per token.
The dense-family hot path (attention Q/K/V/O, dense FFN, lm_head) was a plain
scalar f32 dot (`bolt::compute::matmul`'s F32 tier + the consumer's
`linear_forward`), measured at a flat ~5 GF/s (~10 GB/s) — it cannot even
saturate DRAM. An AVX-512 FMA GEMV hits the CPU's real memory wall (~19-25
GF/s, ~40-50 GB/s), a measured ~4x on real model shapes (67 MB attn / 235 MB
FFN) and ~12x L2-resident.

Decision: add an f32/f16 float-activation SIMD GEMV as a NEW default tier,
living in bolt (the shared substrate) as `bolt::compute::matmul_f32` /
`matmul_f16` (compute_gemv.h), NOT in the consumer. It rides in the existing
per-source-ISA `bolt::compute_idot` library and reuses that module's exact
shape: intrinsic kernels in dedicated TUs compiled at `/arch:AVX2` (+F16C) and
`/arch:AVX512`, a baseline-ISA CPUID probe, and a C++11 magic-static
function-pointer table resolved ONCE (AVX-512F > AVX2+FMA+F16C > scalar), hot
path = one indirect call. This is the same justified exception to bolt's
"no runtime dispatch" rule that idot_dispatch.cpp documents (probe amortized
over a heavy GEMV, not paid per element).

Algorithm (ggml's ggml_vec_dot_f32 as the reference, written bolt-native — no
vendored source, so bolt stays zero-dependency): 4 independent f32 accumulators
over a STEP-unrolled body (32-wide AVX2 / 64-wide AVX-512) to hide FMA latency,
tree-reduce, then a final f64 horizontal sum + f64 scalar tail. f16 weights
upcast on the fly (_mm256_cvtph_ps / _mm512_cvtph_ps) so the GEMV reads 2 bytes
per weight instead of 4.

Why the default wins / what flips it: SIMD reorders the f32 reduction, so
matmul_f32 is NOT bit-exact vs the scalar f64 reference (unlike the integer
IDOT kernels) — it agrees to ~1e-6 relative, inside every consumer oracle's
1e-3 tolerance and preserving greedy token-exactness. The scalar tier stays
reachable (and is what non-AVX2 hardware runs + what test_bolt_gemv compares
against). Gate: bench_compute_gemv asserts a 2.5x GFLOPS floor over scalar on
model shapes when a SIMD tier is present, catching a silent /arch-flag or
CPUID-probe regression to scalar (which the equality test cannot). The choice
flips only if a consumer needs bit-exact reproducibility across ISA tiers — it
would then pin the scalar tier.

## 2026-07-18 / perf-pass 2 — threaded + quantized dense decode: near llama.cpp parity

Follow-up to the f32/f16 SIMD GEMV entry. Established the llama.cpp baseline on
the target box (Ryzen AI Max+ 395, 16 cores, ~256 GB/s UMA) with a real model
(Qwen2.5-0.5B-Q8_0): llama.cpp tg128 = 116 t/s; boltllm f32-resident = 27 t/s
(23%). Decode is memory-bandwidth-bound and the gap mapped almost exactly to
bytes/weight (f32 4B vs Q8_0 ~1B).

Closed the gap on the CPU with three stacked, measured levers, all preserving
coherence:
1. Threaded per-layer GEMV: fan attention/FFN GEMVs across the shared pool by
   output-row blocks (one core hits ~46 GB/s of the ~256 aggregate). Measured
   2.0-2.6x over single-core SIMD. Zero numeric change (row-block independent).
2. Quantized-resident dense weights (opt-in BOLTLLM_DENSE_QUANT=int8|int4):
   requantize at load, route attention/lm_head through a threaded int8-activation
   IDOT and the dense FFN through the existing ExpertFfn IDOT dispatch. Int8: 60
   t/s coherent (52% of llama.cpp); Int4: 90 t/s (77%) but per-row-scale int4 is
   too coarse for a 0.5B model (degraded output -> Q4_K block-quant is the fix).
   f32 forward-oracles stay green (opt-in). Int8 is the fast-and-coherent default.
3. Branch-free online-softmax attention (flash-attention-style): one streaming
   pass over KV per head (running max + f64 denom + rescaled context), no scores
   buffer, no separate softmax pass. Token-exact (oracle green).

GPU verdict (measured, not assumed): lm_head on the iGPU (26 t/s) LOSES to the
threaded CPU (27) on UMA -- shared memory + per-dispatch overhead. With the CPU
now at 60-90 t/s, GPU offload is not viable on this box (same reasoning that
cancelled the per-token expert-cache). HFT tail: p99/p50 = 1.12x (f32) / 1.18x
(int8) -- the tight tail corroborates zero-hot-path allocation.

Net: boltllm 23% -> 52% of llama.cpp at full coherence (Int8), decode tail
HFT-tight, all wins behind perf tripwires. Remaining for full parity: Q4_K
block-quant (quality-preserving 4-bit) + rolling the quantized path to the other
dense families (Qwen2 is the proven reference).

## 2026-07-18 / perf-pass 2b — block-scale Q4 (quality-preserving 4-bit) + alignment discipline

Per-row Int4 (perf-pass 2) hit near-parity SPEED (90 t/s) but degraded coherence
on small models -- one scale per row is too coarse. Added block-scale Q4
(BOLTLLM_DENSE_QUANT=q4b): 32-element blocks each with its own f32 scale (the
Q4_0 essence), boltllm/quant/BlockQ4.h -- format + quantizer + threaded
dequant-in-dot GEMV, wired into Qwen2 attention/lm_head + a manual block-Q4
SwiGLU FFN. Result: coherent 4-bit restored ("...capital of France is Paris...")
at ~0.625 B/weight, f32 oracles green (opt-in). The GEMV is scalar for now
(compute-bound ~23 t/s) -- a SIMD block-dequant kernel (BLLM-173) turns the
bandwidth into speed.

Standing perf discipline for all kernels (user directive): maintain page +
stride alignment and every perf trick. 64-byte-aligned buffers (bolt_aligned_alloc,
not std::vector's 16), row byte-strides padded to a 64-byte multiple (no
split-line SIMD loads, no page bisecting a hot vector), aligned loads once
guaranteed, 4-accumulator STEP-unroll, BOLT_RESTRICT, prefetch, branch-free,
zero hot-path alloc, elevated-/arch TU. Required from the start on BLLM-173.

## 2026-07-18 / ROCm backend — native AMD on-device forward (BLLM-181, 184-188)

Vulkan was the portable fallback; ROCm/HIP is AMD's native compute and the fast
path on the target Radeon 8060S (RDNA 3.5, gfx1151). ROCm 7.1 is now installed
on this box, so bolt::rocm is testable here for the first time. Built the full
on-device op set as HIP kernels mirroring the Vulkan shaders 1:1, each verified
GPU-vs-CPU on the real device:

- matmul_dequant (F32/Int8/Int4/Int2) + int8-activation — pre-existing (BLLM-78/91).
- rmsnorm (BLLM-184): shared-mem tree reduction + rsqrtf.
- rope (BLLM-185): rotate_half in place, matches rope_half_split.
- elementwise (BLLM-186): op0=add, op1=SwiGLU (silu*b), op2=GeGLU (gelu*b).
- attention (BLLM-187): flash-style online softmax, one block per query head,
  GQA-aware, blockDim power-of-two >= head_dim so the tree-reduced QK dot stays
  valid with inactive threads contributing 0. Mirrors qwen2_attention_ exactly.

Coherence gate (BLLM-188): a FULL pre-norm transformer layer (Qwen2/Llama-style,
GQA, no biases, SwiGLU) computed ENTIRELY on-device — rmsnorm -> Q/K/V matmul ->
RoPE -> attention -> O matmul -> residual -> rmsnorm -> gate/up -> SwiGLU ->
down -> residual — with activations staying RESIDENT across the whole layer (only
the input hidden uploaded once, final hidden read back once; the current token's
K/V matmul writes directly into its KV-cache slot on-device, no round-trip).
Matches a CPU reference within 1e-3 on the real gfx1151. This is the technical
crux of the resident-graph win on UMA: keep the graph on-device, pay upload/
readback once per token instead of per-op. Build: clang++ HIP direct .obj->.lib
(BLLM-99), host stays MSVC, -DBOLT_BUILD_ROCM=ON; 13/13 test_bolt_rocm PASSED.

Still open (BLLM-188 tail): wiring this chain into boltllm's Engine (GpuContext
is Vulkan-only today) + an end-to-end tok/s measurement vs Vulkan and the 60-90
t/s CPU on a real GGUF. The kernels + coherence are proven; the engine-level
backend selection and measurement are the remaining integration.

## 2026-07-18 / measured: ROCm vs Vulkan vs CPU (BLLM-188/148)

bench_gpu_backends.cpp: matmul_dequant F32 GEMV (the dominant decode op) at real
model shapes, effective GB/s (decode is bandwidth-bound; ~256 GB/s UMA wall).
GPU = resident weight + per-op {act-in, dispatch, wait, out}; CPU = bolt SIMD
GEMV, single-thread + persistent warm pool (32 threads spawned once).

Typical GB/s (median of 3, ~±15% run variance):
  shape                cpu-1t  cpu-Nt  vulkan  rocm
  0.9K^2 (3 MiB,L2)    ~150    ~64     ~45     ~28
  4K^2   (64 MiB)      ~49     ~82     ~140    ~149
  11K4K  (172 MiB)     ~40     ~70     ~145    ~188

Verdict (the per-op measure-gated call BLLM-148 wanted): on real-sized weights
(64-172 MiB) the GPU wins ~1.8-2.5x over the threaded CPU (ROCm ~188 GB/s = 74%
of the UMA wall on the biggest GEMV; Vulkan close). On tiny L2-resident weights
(3 MiB) the CPU wins big (~150 GB/s, never touches DRAM) -- GPU dispatch/transfer
overhead dominates. ROCm ~= Vulkan at the kernel level, ROCm modestly ahead on
the largest shape. => move big attention/FFN GEMVs to GPU, keep small/norm ops
on CPU. The per-op transfer+WaitIdle overhead measured here is exactly what the
engine-wired resident graph (BLLM-174 resident activations + BLLM-175 batched
single-submit) removes -- that's the next lever, now with a measured ceiling.

## 2026-07-18 / ROCm resident-graph decode measured + launch-overhead finding (BLLM-188/189)

Added bolt::rocm batch mode (begin_batch/end_batch): between them the compute ops
skip their per-op hipDeviceSynchronize (kernels serialize on the default stream),
so a whole token costs ONE sync. bench_gpu_backends.cpp runs a full resident-graph
decode step at Qwen2.5-0.5B shape (24 layers + lm_head), weights resident, only
logits read back.

Measured (Radeon 8060S, stable):
  F32  weights: ~13.3 ms/token -> ~75 t/s  (65% of llama.cpp's 116)
  Int8 weights: ~10.3 ms/token -> ~97 t/s  (84%) -- MEASURED, beats boltllm CPU
                                                    (Int8 60 / Int4 90 t/s)

KEY FINDING: Int8 is only 1.3x faster than F32, NOT the 4x its byte savings imply.
F32 GEMVs are bandwidth-bound (~146 GB/s, launches hidden behind long kernels);
Int8 GEMVs are 4x faster so the ~362 kernel LAUNCHES/token (~20us each ~= 7ms)
become the exposed floor. Int8's bandwidth ceiling is ~3.4ms (~290 t/s) -- launch
overhead caps it at ~10ms. => the bottleneck flipped from bandwidth to launch
overhead. Next lever to pass llama.cpp: hipGraph capture (replay the token's
kernel sequence, no per-launch cost) + op fusion (norm+matmul, gate+up+SwiGLU),
targeting Int8's ~290 t/s ceiling = ~2.5x llama.cpp. New ticket BLLM-189.

## 2026-07-18 / hipGraph measured -> NOT the lever; op fusion is (BLLM-189)

Hypothesis from the prior entry: Int8's shortfall is CPU kernel-LAUNCH overhead;
fix with hipGraph capture (record token once, replay). Built it (bolt::rocm
Context: non-default stream + begin/end_graph_capture + graph_launch). RESULT:
zero improvement -- Int8 96 t/s batch vs 96 hipGraph (0.5B), 13.3 vs 13.3 (8B).
Hypothesis WRONG. hipGraph only removes CPU launch cost; since it didn't move,
the bottleneck is GPU-side serial execution of ~361 data-dependent kernels/token
(each reads the prior op's output -> no overlap; fixed ~28us dispatch/drain each
~= 10ms, matching Int8's floor).

Corroboration: F32 8B hits 208 GB/s = ~UMA wall (GEMV kernels efficient, not the
problem). Int8-vs-F32 speedup grows with model size (0.5B 1.34x -> 8B 1.93x) as
bigger GEMVs stay bandwidth-bound long enough to amortize per-kernel latency, but
never 4x while kernel COUNT stays ~361.

=> Real lever to pass llama.cpp is OP FUSION (fewer/bigger kernels): norm+matmul,
fused QKV (one GEMV off shared normed input), fused gate+up+SwiGLU, fused
attention -- ~15 kernels/layer -> ~5. hipGraph kept as a shipped primitive (right
tool when CPU launch IS the bound: many tiny independent kernels) but not the
decode lever. Measure-don't-assume win: the disproven hypothesis localized the
real cause. BLLM-189 redirected to fusion.

## 2026-07-18 / op fusion landing -- 97 -> 110 t/s Int8 0.5B (BLLM-189)

The lever from the prior entry, executed. Each fusion TigerStyle + branch-free +
coherence-gated (resident-layer test now runs the FUSED path, still matches CPU
1e-3) + unit-tested.

  step                                     kern/layer  0.5B Int8
  unfused                                     15         97 t/s   (84% of llama.cpp)
  + fused gate+up+SwiGLU (3->1)               13        108 t/s   (93%)
  + residual-add folded into O/down matmul    11        110 t/s   (95%)

- fused_swiglu (f32/int8): one block per FFN row does both gate & up dots in one
  pass (two shared-mem reductions) + silu(gate)*up. Removes 2 big GEMV launches +
  gate/up/act intermediate buffers. Biggest single jump (+11 t/s).
- matmul_dequant_f32/int8_kernel templated on <bool Accumulate> (compile-time
  dispatch, branch vanishes); matmul_dequant_residual launches <true> so
  out[r]+=dot folds the skip-add into the projection. Small gain (+2, the elt-add
  was cheap) but removes 2 launches/layer.

Still latency-bound (9.08ms Int8 = 54 GB/s effective vs 3.3ms bandwidth ceiling;
266 kernels x ~22us). Next: fused QKV (3->1), merge 2 rope launches -> cross 116.
Open caveat: 116 is llama.cpp CPU-only; fair GPU-vs-GPU needs llama.cpp's Vulkan
backend on this iGPU.

## 2026-07-18 / fused QKV perf-neutral -> the bound is TRAFFIC, not kernel count (BLLM-189)

Added fused_qkv (f32/int8): one launch produces q/k/v from the shared normed
input (canonical llama.cpp fusion), coherence-verified via the resident-layer
test. Perf: NEUTRAL (110 -> 110 t/s Int8 0.5B). Contrast fused_swiglu (+11).
Difference: swiglu removed the gate/up/act INTERMEDIATE BUFFERS (real traffic);
QKV only removed launches (q/k/v still written). => at this size decode is
memory-traffic-bound (weights + intermediates), NOT launch-count-bound -- same
reason hipGraph did nothing. Corroboration: F32 0.5B = 155 GB/s (bandwidth-bound,
near GEMV ceiling); Int8 = 54 GB/s (too few weight bytes to saturate -> floored by
fixed per-op work). fused_qkv kept (correct, standard, cuts launches for
batched/other-hw regimes) but perf-neutral for single-token 0.5B.

Status: 110 t/s Int8 = 95% of llama.cpp CPU baseline; 8B GPU (15.6 t/s) laps
llama.cpp CPU decode. Next: (1) rocprof to find the real ~9ms sink before more
fusion; (2) fair GPU-vs-GPU number (llama.cpp Vulkan on this iGPU -- 116 is
CPU-only, and llama.cpp's GPU path may itself sit below 116 on a tiny 0.5B model).

## 2026-07-19 / Int4 fused kernels + per-op profile -> attention is the sink (BLLM-189)

Int4 weight tier added to the fused kernels (fused_swiglu_int4 nibble-unpack;
matmul_dequant_int4 templated <Accumulate> for residual O/down; qkv unfused in
Int4 since fused_qkv is perf-neutral). Coherence-gated (FusedSwigluInt4 test).

Resident-graph decode: 0.5B Int4 = 115 t/s (Int8 111, F32 77) = 99% of llama.cpp
CPU baseline (116). 8B Int4 = 17.2. Int4's gain modest (+4 on 0.5B) -- weight
GEMVs latency-bound at this size.

Per-op profile (profile_rocm_ops, rocprof-absent substitute -- each op K x in one
batch, one sync amortized, x per-token count):
  attention (S=128)      79 us x24 = 1.91 ms   <-- #1 sink
  lm_head (151936x896) 1869 us x1  = 1.87 ms
  fused_swiglu           48 us x24 = 1.16
  down-resid             25 us x24 = 0.59
  qkv/O/rmsnorm/rope                = 0.79
  SUM                               ~6.3 ms

Two sinks: attention + lm_head. lm_head = one unavoidable huge GEMV (Int4 halves).
ATTENTION is a weak kernel: 1 block/head (14 blocks -> poor occupancy), 64/256
threads used, serial 128-step loop w/ per-step block reduction. Next lever: proper
flash-decode attention (parallelize timesteps, one end-reduction) -> cut several-
fold, push past 116.

## 2026-07-19 / two-pass flash-decode attention -- the #1 sink, 10x'd (BLLM-189)

The online attention kernel did a block reduction PER timestep (~seq_len *
log2(blockDim) __syncthreads) -- the profile's #1 decode sink (79us/call, 1.91
ms/token). Replaced with a two-pass kernel (attention_twopass_kernel): pass 1
materializes scores[t] in shared (256 threads stride timesteps, full QK dot each,
no per-step sync); TWO block reductions total (max, then sum of exp); pass 2
strides head dims summing weight[t]*V[t][d]. ~20 syncs vs ~768. Same numerics
(coherence tests green). Kept the online kernel as the >4096-context fallback.

attention: 79 -> 8 us/call (10x). SUM 6.34 -> 4.60 ms.
Synthetic 0.5B: Int8 111 -> 156 t/s, Int4 115 -> 165. 8B: Int8 15.4 -> 16.5.
REAL qwen2.5-0.5b-q8_0: Int8 116.9 -> 123.6 t/s, coherent -> PASSES llama.cpp's 116.

Now the sinks are lm_head (1.87 ms, one huge GEMV) + fused_swiglu (1.14). FTZ +
attention together took the real model from 115 to 123.6 t/s.

## 2026-07-19 / warp-per-row GEMV -- llama.cpp mul_mat_vec shape (BLLM-190)

Fair GPU-vs-GPU: llama.cpp Vulkan on the 8060S hits 304 t/s on qwen2.5-0.5b-q8_0
(our 123.6 only beat their CPU 116). The gap was our naive GEMV: one block/row
with an 8-round shared-mem block reduction for ~3.5 bytes/thread of work
(lm_head ran at 73 GB/s). Rewrote matmul_dequant f32/int8/int4 (+residual) and
fused_swiglu to ONE WAVEFRONT PER ROW with a shuffle reduction (warp_reduce_sum,
__shfl_down, width-agnostic via warpSize) -- no shared mem, no __syncthreads,
blockDim/warpSize rows per block for high occupancy. Coherence tests green (16/16).

lm_head: 1867 -> 1024 us (73 -> 133 GB/s). fused_swiglu 47->21. SUM 4.60->3.08 ms.
Synthetic 0.5B: Int8 156 -> 206, Int4 165 -> 247.
REAL qwen2.5-0.5b-q8_0 Int8: 123.6 -> 163 t/s, coherent (52 CPU this morning -> 163).
=> 54% of llama.cpp's GPU 304, closing.

Note: warp-per-row favors TALL matrices (lm_head 151936 rows). WIDE ones (8B FFN
down/gate, 14336 cols) regress slightly (fewer lanes per long row) -- 8B Int8
16.5 -> 13.7. An adaptive multi-warp-per-row for wide cols is the follow-up.
Next levers to chase 304: vectorized loads (int8x4/float4), Int4 lm_head, faster
attention occupancy (split-K).

## 2026-07-19 / vectorized int8 GEMV loads -- lm_head at the bandwidth wall (BLLM-190)

Each warp lane now loads 4 int8 weights as one uint32 + 4 activations as one
float4 (4 MACs/iter, 4x fewer loads, more ILP). Requires cols%4==0 (all Qwen2.5
dims qualify); scalar warp loop handles odd cols. Coherence green (16/16).

lm_head: 1024 -> 617 us = 220 GB/s (AT the wall). down-resid 27->9 (the wide-
matrix warp regression is GONE -- vectorization fixed it), O 5.3->3.5. SUM
3.08 -> 2.19 ms. Synthetic 0.5B Int8 206 -> 264.6 (now BEATS Int4 248 -- int8 is
vectorized, int4 not yet). 8B Int8 recovered 13.7 -> 16.2.
REAL qwen2.5-0.5b-q8_0 Int8: 163 -> 217.8 t/s, coherent+fluent.

Session arc: 52 (CPU this morning) -> 217.8 t/s = 4.2x, now 72% of llama.cpp's
GPU 304. Remaining sinks: lm_head 0.62 (bandwidth-bound now), fused_swiglu 0.50
(not yet vectorized), qkv 0.27. Next: vectorize fused_swiglu + fused_qkv + int4.

## 2026-07-19 / vectorized fused_swiglu + fused_qkv (BLLM-190)

Rolled the 4-int8-as-uint32 + float4 vectorization into fused_swiglu_int8 and
fused_qkv_int8 (also converted fused_qkv f32/int8 from block-per-row to warp-
per-row). 16/16 coherence green.

fused_qkv 11.2->3.8 us, fused_swiglu 20.8->10.9 us. SUM 2.19->1.78 ms.
Synthetic 0.5B Int8 264.6 -> 280.3 t/s. REAL qwen2.5-0.5b-q8_0 Int8: 217.8 ->
244 t/s, coherent.

Session arc: 52 (CPU) -> 244 t/s = 4.7x, now 80% of llama.cpp's GPU 304.
Sinks now: lm_head 0.64 (bandwidth-bound), fused_swiglu 0.26, down 0.22,
attention 0.19, rmsnorm 0.18. Next: Int4 lm_head (mixed, halves the top sink),
vectorize the Int4 kernels (Int4 240 < Int8 280 because not vectorized).

## Commit log

- 2c6e57d 2026-07-18 feat: SIMD f32/f16 GEMV + GPU on-device ops (BLLM perf pass)
- 22bad01 2026-07-18 feat(gpu): on-device elementwise shader (add/SwiGLU/GeGLU) [BLLM-179]
- 1f6b6cb 2026-07-18 feat(gpu): on-device RoPE shader (rotate_half) [BLLM-177]
- 61e46d5 2026-07-18 feat(rocm): on-device HIP rmsnorm/rope/elementwise kernels (BLLM-184/185/186)
- ffd0da9 2026-07-18 feat(rocm): on-device HIP decode attention kernel (BLLM-187)
- 2bdb4ef 2026-07-18 test(rocm): full resident-graph transformer layer, on-device, verified (BLLM-188)
- aab8491 2026-07-18 feat(cuda): bolt::cuda NVIDIA backend — mirror of ROCm, build-gated (BLLM-182)
- 31d5316 2026-07-18 feat(vulkan): on-device attention shader — flash-style online softmax (BLLM-178)
- e2500ea 2026-07-18 feat(reactor): timer_queue — backend-agnostic timers for the reactor (STATION-82)
- 2acf784 2026-07-18 feat(parse): byte-faithful JSON writer for bolt::parse::json (STATION-80)
- 8121237 2026-07-18 bench(gpu): measured ROCm vs Vulkan vs CPU for the decode GEMV (BLLM-188/148)
- df64cff 2026-07-18 feat(rocm): batch mode + measured resident-graph decode (BLLM-188)
- c703388 2026-07-18 feat(rocm): hipGraph capture + 2-size decode measure — bottleneck localized (BLLM-189)
- 1f8c32a 2026-07-18 feat(io): bolt async File I/O API on the hoisted reactor (STATION-84)
- de4312e 2026-07-18 feat(rocm): op fusion -- fused SwiGLU FFN + residual-into-matmul (BLLM-189)
- cdb0036 2026-07-18 feat(rocm): fused QKV projection -- perf-neutral, reframes the bound (BLLM-189)
- 8985423 2026-07-19 feat(rocm): Int4 fused kernels + per-op profile -- 115 t/s, at llama.cpp parity (BLLM-189)
- 064250a 2026-07-19 feat(json): nlohmann-faithful emit mode for bolt's JSON writer (STATION-87)
- c0464c3 2026-07-19 fix(rocm): flush denormals to zero -- fixes 6x Int4 decode slowdown (BLLM-189)
- c65a8c6 2026-07-19 perf(rocm): two-pass flash-decode attention -- 10x the #1 sink (BLLM-189)
- 9c7c3c9 2026-07-19 perf(rocm): warp-per-row GEMV (llama.cpp mul_mat_vec shape) -- 163 t/s real (BLLM-190)
- e7d7726 2026-07-19 perf(rocm): vectorized int8 GEMV loads -- lm_head at bandwidth wall, 217 t/s real (BLLM-190)
- 090e926 2026-07-19 perf(rocm): vectorize fused_swiglu + fused_qkv -- 244 t/s real (BLLM-190)
- 1edfb8e 2026-07-19 perf(rocm): vectorize Int4 GEMV loads -- 409 t/s synthetic (BLLM-190)
- b9c5b81 2026-07-19 feat(rocm): GPU MXFP4 matvec kernel -- gpt-oss expert tier on the iGPU (BLLM-200)
- 286b0d4 2026-07-19 feat(rocm): gpt-oss GPU kernels — attention sinks+SWA, swiglu_oai (BLLM-200)
- 900f4b5 2026-07-19 feat(rocm): NeoX+YaRN rope + axpy_bias kernels (BLLM-200)
- 559b950 2026-07-19 feat(rocm): aligned + fused MXFP4 expert kernels (BLLM-200)
- aed85ba 2026-07-19 feat(rocm): GPU top-K routing -> whole gpt-oss token in one batch (BLLM-200)
- f33270c 2026-07-19 diag(rocm): raw device read-bandwidth probe (BLLM-200)
- 8295d69 2026-07-19 perf(rocm): fold biases into matmuls + 16-byte int8 weight loads (BLLM-200)
- 4575f61 2026-07-20 perf(rocm): int4 bias-fold support in matmul_dequant kernels (BLLM-200)
- e679010 2026-07-20 perf(rocm): dynamic-shared scores in two-pass attention (BLLM-200)
- a9d409c 2026-07-20 wip(rocm): batched expert kernels + multi-size bandwidth probe (BLLM-200)
- 4f497af 2026-07-25 chore(async_io): drop stale "check kernel version" TODO on io_uring select
