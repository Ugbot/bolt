// bench_gpu_backends.cpp -- BLLM-188/148: measured ROCm vs Vulkan vs CPU for
// the dominant decode op (matmul_dequant F32 GEMV) at real model shapes on this
// UMA box. Decode is memory-bandwidth-bound, so the headline metric is
// effective GB/s (weight_bytes / time). Each GPU number is the REALISTIC
// per-op cost the resident-weight path pays: activation in, dispatch + wait,
// output out (weights uploaded ONCE, resident). CPU is bolt's SIMD GEMV, single
// thread and fanned across all cores (the fallback boltllm actually runs).
//
// Only meaningful when a real device exists; each backend self-skips if its
// available() is false. STL allowed in benches; production stays Tiger-Style.

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "bolt/compute_gemv.h"

#if defined(BOLT_HAVE_ROCM)
#include "bolt/bolt_rocm.h"
#endif
#if defined(BOLT_HAVE_VULKAN)
#include "bolt/bolt_vulkan.h"
#endif

namespace {

using Clock = std::chrono::steady_clock;

double median_us(std::vector<double>& xs) {
    std::sort(xs.begin(), xs.end());
    return xs[xs.size() / 2];
}

double gbps(int64_t weight_bytes, double us) {
    return static_cast<double>(weight_bytes) / (us * 1e-6) / 1e9;
}

struct Shape {
    int32_t rows, cols;
    const char* label;
};

// Representative decode GEMV shapes (f32-resident weight).
const Shape kShapes[] = {
    {896, 896, "qwen0.5b hidden  (0.9K x 0.9K, 3 MiB)"},
    {4096, 4096, "llama attn/o     (4K x 4K, 64 MiB)"},
    {11008, 4096, "llama ffn up     (11K x 4K, 172 MiB)"},
};

void fill(std::vector<float>& v, uint32_t salt) {
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = static_cast<float>((int32_t)((i * 131u + salt * 17u) % 19u) - 9) * 0.05f;
}

// ---- CPU: bolt SIMD GEMV across a PERSISTENT warm worker pool (mirrors
// boltllm's SharedSchedulerPool -- workers are spawned ONCE, not per call, so
// the measurement is compute + sync, not thread-creation overhead). ----
class WarmPool {
public:
    explicit WarmPool(unsigned n) : n_(n) {
        for (unsigned i = 0; i < n_; ++i) workers_.emplace_back([this, i] { worker(i); });
    }
    ~WarmPool() {
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; ++gen_; }
        cv_start_.notify_all();
        for (auto& t : workers_) t.join();
    }
    void run(const float* w, const float* x, float* out, int32_t rows, int32_t cols) {
        {
            std::lock_guard<std::mutex> lk(m_);
            w_ = w; x_ = x; out_ = out; rows_ = rows; cols_ = cols;
            block_ = (rows + static_cast<int32_t>(n_) - 1) / static_cast<int32_t>(n_);
            done_ = 0;
            ++gen_;
        }
        cv_start_.notify_all();
        std::unique_lock<std::mutex> lk(m_);
        cv_done_.wait(lk, [this] { return done_ == n_; });
    }

private:
    void worker(unsigned id) {
        uint64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(m_);
            cv_start_.wait(lk, [this, &seen] { return stop_ || gen_ != seen; });
            if (stop_) return;
            seen = gen_;
            const float* w = w_; const float* x = x_; float* out = out_;
            const int32_t rows = rows_, cols = cols_, block = block_;
            lk.unlock();
            const int32_t r0 = static_cast<int32_t>(id) * block;
            if (r0 < rows) {
                const int32_t rn = std::min(block, rows - r0);
                bolt::compute::matmul_f32(w + static_cast<int64_t>(r0) * cols, x, out + r0, rn, cols);
            }
            lk.lock();
            if (++done_ == n_) cv_done_.notify_one();
        }
    }
    unsigned n_;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_start_, cv_done_;
    const float* w_ = nullptr; const float* x_ = nullptr; float* out_ = nullptr;
    int32_t rows_ = 0, cols_ = 0, block_ = 0;
    uint64_t gen_ = 0;
    unsigned done_ = 0;
    bool stop_ = false;
};

#if defined(BOLT_HAVE_ROCM)
// BLLM-188: full resident-graph decode step at Qwen2.5-0.5B shape, chained
// entirely on-device in BATCH mode (one hipDeviceSynchronize per TOKEN, not per
// op). Weights are resident (uploaded once); per token we run all layers' ops
// back-to-back on the stream and read back only the logits. This is the decode
// throughput the resident graph unlocks -- the per-op transfer+sync overhead the
// GEMV table pays is gone. Weights are uninitialized (perf is bandwidth, not
// values). Returns median ms/token, or 0 on setup failure.
struct ModelCfg {
    int H, layers, heads, kv_heads, head_dim, ffn, vocab;
};

double bench_rocm_resident_forward(bolt::rocm::Context& ctx, const ModelCfg& cfg, bool int8,
                                   bool use_graph) {
    using namespace bolt::rocm;
    const int H = cfg.H, layers = cfg.layers, heads = cfg.heads, kv_heads = cfg.kv_heads;
    const int head_dim = cfg.head_dim, ffn = cfg.ffn, vocab = cfg.vocab;
    const int q_dim = heads * head_dim;        // 896
    const int kv_dim = kv_heads * head_dim;    // 128
    const int S = 128;                          // KV cache length (mid-sequence)
    const int pos = S - 1;
    const float eps = 1e-5f, theta = 1e6f;
    const float scale = 1.0f / 8.0f;            // 1/sqrt(64)
    const WeightDType wdt = int8 ? WeightDType::Int8 : WeightDType::F32;
    const uint64_t wbpe = int8 ? 1u : 4u;       // weight bytes per element

    // GEMV-weight buffers (dtype-sized) + their per-row scale buffers (Int8 only).
    std::vector<DeviceBuffer> wq, wk, wv, wo, wg, wu, wd;
    std::vector<DeviceBuffer> sq, sk, sv, so, sg, su, sd;
    // f32 norm weights + f32 KV cache (norms/activations stay f32 in both modes).
    std::vector<DeviceBuffer> an, fn, kc, vc;
    bool ok = true;
    auto aw = [&](int64_t rows, int64_t cols, std::vector<DeviceBuffer>& wv_, std::vector<DeviceBuffer>& sv_) {
        DeviceBuffer w, s;
        ok = ok && ctx.allocate(static_cast<uint64_t>(rows * cols) * wbpe, &w);
        wv_.push_back(w);
        if (int8) { ok = ok && ctx.allocate(static_cast<uint64_t>(rows) * 4, &s); sv_.push_back(s); }
    };
    auto af = [&](int64_t elems, std::vector<DeviceBuffer>& into) {
        DeviceBuffer b; ok = ok && ctx.allocate(static_cast<uint64_t>(elems) * 4, &b); into.push_back(b);
    };
    for (int l = 0; l < layers; ++l) {
        af(H, an); af(H, fn); af((int64_t)S * kv_dim, kc); af((int64_t)S * kv_dim, vc);
        aw(q_dim, H, wq, sq); aw(kv_dim, H, wk, sk); aw(kv_dim, H, wv, sv); aw(H, q_dim, wo, so);
        aw(ffn, H, wg, sg); aw(ffn, H, wu, su); aw(H, ffn, wd, sd);
    }
    DeviceBuffer d_fn, d_lm, s_lm, hid, nrm, q, ctxb, ao, gate, up, act, ffo, logits;
    ok = ok && ctx.allocate((uint64_t)H * 4, &d_fn) &&
         ctx.allocate((uint64_t)vocab * H * wbpe, &d_lm) &&
         (!int8 || ctx.allocate((uint64_t)vocab * 4, &s_lm)) &&
         ctx.allocate((uint64_t)H * 4, &hid) && ctx.allocate((uint64_t)H * 4, &nrm) &&
         ctx.allocate((uint64_t)q_dim * 4, &q) && ctx.allocate((uint64_t)q_dim * 4, &ctxb) &&
         ctx.allocate((uint64_t)H * 4, &ao) && ctx.allocate((uint64_t)ffn * 4, &gate) &&
         ctx.allocate((uint64_t)ffn * 4, &up) && ctx.allocate((uint64_t)ffn * 4, &act) &&
         ctx.allocate((uint64_t)H * 4, &ffo) && ctx.allocate((uint64_t)vocab * 4, &logits);
    if (!ok) { std::printf("  (resident-forward: device alloc failed -- out of memory?)\n"); return 0.0; }

    std::vector<float> logits_host(static_cast<size_t>(vocab));
    auto fp = [](const DeviceBuffer& b) { return static_cast<const float*>(b.ptr); };
    auto mp = [](const DeviceBuffer& b) { return static_cast<float*>(b.ptr); };
    auto sc = [&](std::vector<DeviceBuffer>& sv_, int l) { return int8 ? fp(sv_[l]) : nullptr; };

    // The op sequence only -- no batch/sync/readback (so it can be either batched
    // or captured into a graph).
    auto record_ops = [&]() {
        for (int l = 0; l < layers; ++l) {
            float* kslot = mp(kc[l]) + (int64_t)pos * kv_dim;
            float* vslot = mp(vc[l]) + (int64_t)pos * kv_dim;
            rmsnorm(ctx, fp(hid), fp(an[l]), mp(nrm), H, eps);
            matmul_dequant(ctx, wq[l].ptr, wdt, sc(sq, l), fp(nrm), mp(q), q_dim, H);
            matmul_dequant(ctx, wk[l].ptr, wdt, sc(sk, l), fp(nrm), kslot, kv_dim, H);
            matmul_dequant(ctx, wv[l].ptr, wdt, sc(sv, l), fp(nrm), vslot, kv_dim, H);
            rope(ctx, mp(q), head_dim, heads, pos, theta);
            rope(ctx, kslot, head_dim, kv_heads, pos, theta);
            attention(ctx, fp(q), fp(kc[l]), fp(vc[l]), mp(ctxb), heads, kv_heads, head_dim, S, scale);
            // O-proj with FUSED residual add (BLLM-189): hid += Wo·ctx.
            matmul_dequant_residual(ctx, wo[l].ptr, wdt, sc(so, l), fp(ctxb), mp(hid), H, q_dim);
            rmsnorm(ctx, fp(hid), fp(fn[l]), mp(nrm), H, eps);
            // FUSED gate+up+SwiGLU (BLLM-189): 3 kernels -> 1.
            fused_swiglu(ctx, wg[l].ptr, wu[l].ptr, wdt, sc(sg, l), sc(su, l), fp(nrm), mp(act), ffn, H);
            // down-proj with FUSED residual add: hid += Wd·act.
            matmul_dequant_residual(ctx, wd[l].ptr, wdt, sc(sd, l), fp(act), mp(hid), H, ffn);
        }
        rmsnorm(ctx, fp(hid), fp(d_fn), mp(nrm), H, eps);
        matmul_dequant(ctx, d_lm.ptr, wdt, int8 ? fp(s_lm) : nullptr, fp(nrm), mp(logits), vocab, H);
    };

    std::function<void()> run;
    if (use_graph) {
        ctx.begin_graph_capture();
        record_ops();
        if (!ctx.end_graph_capture()) {
            std::printf("  (graph capture failed)\n");
        }
        run = [&]() {
            ctx.graph_launch();  // whole token replayed in one launch + one sync
            ctx.copy_to_host(logits, logits_host.data(), logits_host.size() * sizeof(float));
        };
    } else {
        run = [&]() {
            ctx.begin_batch();
            record_ops();
            ctx.end_batch();  // ONE sync for the whole token
            ctx.copy_to_host(logits, logits_host.data(), logits_host.size() * sizeof(float));
        };
    }

    for (int i = 0; i < 3; ++i) run();  // warmup
    std::vector<double> ts;
    for (int i = 0; i < 20; ++i) {
        auto a = Clock::now();
        run();
        ts.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a).count());
    }
    ctx.free_graph();  // release the captured graph before its buffers go away
    auto freev = [&](std::vector<DeviceBuffer>& v) { for (auto& b : v) ctx.free(&b); };
    freev(wq); freev(wk); freev(wv); freev(wo); freev(wg); freev(wu); freev(wd);
    freev(sq); freev(sk); freev(sv); freev(so); freev(sg); freev(su); freev(sd);
    freev(an); freev(fn); freev(kc); freev(vc);
    ctx.free(&d_fn); ctx.free(&d_lm); ctx.free(&s_lm); ctx.free(&hid); ctx.free(&nrm); ctx.free(&q);
    ctx.free(&ctxb); ctx.free(&ao); ctx.free(&gate); ctx.free(&up); ctx.free(&act);
    ctx.free(&ffo); ctx.free(&logits);
    return median_us(ts) / 1000.0;  // ms/token
}
#endif

}  // namespace

int main() {
    const int kWarmup = 5, kIters = 30;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    WarmPool pool(hw);  // persistent workers, spawned once (like SharedSchedulerPool)

    std::printf("bench_gpu_backends -- matmul_dequant F32 GEMV, effective GB/s (higher better)\n");
    std::printf("CPU SIMD tier: avx512=%d avx2=%d ; hw_threads=%u\n\n",
                bolt::compute::gemv_avx512_available() ? 1 : 0,
                bolt::compute::gemv_avx2_available() ? 1 : 0, hw);

#if defined(BOLT_HAVE_ROCM)
    bolt::rocm::Context rocm;
    const bool rocm_ok = bolt::rocm::available() && rocm.create();
    std::printf("ROCm:   %s\n", rocm_ok ? rocm.device_name() : "(unavailable)");
#else
    std::printf("ROCm:   (not built)\n");
#endif
#if defined(BOLT_HAVE_VULKAN)
    bolt::vulkan::Context vk;
    const bool vk_ok = bolt::vulkan::available() && vk.create();
    std::printf("Vulkan: %s\n", vk_ok ? "available" : "(unavailable)");
#else
    std::printf("Vulkan: (not built)\n");
#endif
    std::printf("\n%-38s | %10s | %10s | %10s | %10s\n", "shape", "cpu-1t", "cpu-Nt", "vulkan",
                "rocm");
    std::printf("%.38s-+-%.10s-+-%.10s-+-%.10s-+-%.10s\n",
                "--------------------------------------------------",
                "----------", "----------", "----------", "----------");

    for (const Shape& s : kShapes) {
        const int64_t wbytes = static_cast<int64_t>(s.rows) * s.cols * 4;
        std::vector<float> w(static_cast<size_t>(s.rows) * s.cols), x(static_cast<size_t>(s.cols)),
            out(static_cast<size_t>(s.rows));
        fill(w, 1);
        fill(x, 2);
        const int iters = wbytes > (32LL << 20) ? kIters : kIters * 3;

        // CPU single-thread.
        for (int i = 0; i < kWarmup; ++i) bolt::compute::matmul_f32(w.data(), x.data(), out.data(), s.rows, s.cols);
        std::vector<double> t1;
        for (int i = 0; i < iters; ++i) {
            auto a = Clock::now();
            bolt::compute::matmul_f32(w.data(), x.data(), out.data(), s.rows, s.cols);
            t1.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a).count());
        }
        // CPU all-cores (warm pool).
        for (int i = 0; i < kWarmup; ++i) pool.run(w.data(), x.data(), out.data(), s.rows, s.cols);
        std::vector<double> tn;
        for (int i = 0; i < iters; ++i) {
            auto a = Clock::now();
            pool.run(w.data(), x.data(), out.data(), s.rows, s.cols);
            tn.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a).count());
        }
        double vk_gbps = 0.0, rocm_gbps = 0.0;

#if defined(BOLT_HAVE_VULKAN)
        if (vk_ok) {
            using namespace bolt::vulkan;
            MatmulPipeline pipe;
            BufferHandle wb, sb, ab, ob;
            if (vk.create_matmul_pipeline(WeightDType::F32, &pipe) &&
                vk.allocate_buffer(w.size() * sizeof(float), false, &wb) &&
                vk.allocate_buffer(x.size() * sizeof(float), true, &ab) &&
                vk.allocate_buffer(out.size() * sizeof(float), true, &ob)) {
                if (void* wp = vk.map_buffer(wb)) { std::memcpy(wp, w.data(), w.size() * sizeof(float)); vk.unmap_buffer(wb); }
                void* ap = vk.map_buffer(ab);
                void* op = vk.map_buffer(ob);
                for (int i = 0; i < kWarmup; ++i) vk.matmul_dequant(pipe, wb, sb, ab, ob, s.rows, s.cols);
                std::vector<double> tv;
                for (int i = 0; i < iters; ++i) {
                    auto a = Clock::now();
                    if (ap) std::memcpy(ap, x.data(), x.size() * sizeof(float));  // fresh activation
                    vk.matmul_dequant(pipe, wb, sb, ab, ob, s.rows, s.cols);
                    if (op) std::memcpy(out.data(), op, out.size() * sizeof(float));  // read result
                    tv.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a).count());
                }
                vk_gbps = gbps(wbytes, median_us(tv));
                if (ap) vk.unmap_buffer(ab);
                if (op) vk.unmap_buffer(ob);
                vk.free_buffer(&wb); vk.free_buffer(&ab); vk.free_buffer(&ob);
                vk.destroy_matmul_pipeline(&pipe);
            }
        }
#endif
#if defined(BOLT_HAVE_ROCM)
        if (rocm_ok) {
            using namespace bolt::rocm;
            DeviceBuffer wb, ab, ob;
            if (rocm.allocate(w.size() * sizeof(float), &wb) &&
                rocm.allocate(x.size() * sizeof(float), &ab) &&
                rocm.allocate(out.size() * sizeof(float), &ob) &&
                rocm.copy_to_device(w.data(), &wb, w.size() * sizeof(float))) {  // weight resident
                for (int i = 0; i < kWarmup; ++i) {
                    rocm.copy_to_device(x.data(), &ab, x.size() * sizeof(float));
                    matmul_dequant(rocm, wb.ptr, WeightDType::F32, nullptr,
                                   static_cast<const float*>(ab.ptr), static_cast<float*>(ob.ptr),
                                   s.rows, s.cols);
                }
                std::vector<double> tr;
                for (int i = 0; i < iters; ++i) {
                    auto a = Clock::now();
                    rocm.copy_to_device(x.data(), &ab, x.size() * sizeof(float));  // fresh activation
                    matmul_dequant(rocm, wb.ptr, WeightDType::F32, nullptr,
                                   static_cast<const float*>(ab.ptr), static_cast<float*>(ob.ptr),
                                   s.rows, s.cols);
                    rocm.copy_to_host(ob, out.data(), out.size() * sizeof(float));  // read result
                    tr.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a).count());
                }
                rocm_gbps = gbps(wbytes, median_us(tr));
                rocm.free(&wb); rocm.free(&ab); rocm.free(&ob);
            }
        }
#endif
        std::printf("%-38s | %8.1f G | %8.1f G | %8.1f G | %8.1f G\n", s.label, gbps(wbytes, median_us(t1)),
                    gbps(wbytes, median_us(tn)), vk_gbps, rocm_gbps);
    }
    std::printf("\nNote: GPU numbers include per-op activation upload + dispatch + wait + readback\n");
    std::printf("(weights resident). On UMA all paths hit the same LPDDR5X; ~256 GB/s is the wall.\n");

#if defined(BOLT_HAVE_ROCM)
    if (rocm_ok) {
        std::printf("\n--- ROCm resident-graph decode (whole token on-device, one sync/token) ---\n");
        const ModelCfg kSmall{896, 24, 14, 2, 64, 4864, 151936};    // Qwen2.5-0.5B
        const ModelCfg kLarge{4096, 32, 32, 8, 128, 14336, 128256};  // Llama-3-8B shape
        auto suite = [&](const char* name, const ModelCfg& cfg) {
            std::printf("  [%s]  H=%d layers=%d ffn=%d GQA=%d/%d vocab=%d\n", name, cfg.H, cfg.layers,
                        cfg.ffn, cfg.heads, cfg.kv_heads, cfg.vocab);
            auto row = [&](const char* tag, bool i8, bool g) {
                const double ms = bench_rocm_resident_forward(rocm, cfg, i8, g);
                if (ms > 0.0)
                    std::printf("    %-24s %7.2f ms/token  ->  %7.1f tok/s\n", tag, ms, 1000.0 / ms);
            };
            row("F32  batch:", false, false);
            row("Int8 batch:", true, false);
            row("Int8 hipGraph:", true, true);
        };
        suite("Qwen2.5-0.5B", kSmall);
        suite("Llama-3-8B", kLarge);
        std::printf("  ---\n");
        std::printf("  llama.cpp (Qwen2.5-0.5B Q8_0, CPU-only llama-bench): 116 tok/s\n");
        std::printf("  boltllm CPU: Int8 ~60, Int4 ~90 tok/s. llama.cpp 8B-class CPU decode ~ a few t/s.\n");
    }
#endif
    return 0;
}
