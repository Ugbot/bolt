// bench_bolt.cpp — Bolt primitives performance benchmark
//
// Validates the core performance claims:
//   Arena vs malloc:     ~9,600x faster for 16KB allocations
//   SPSC channel:        ~25x faster than mutex queue
//   Epoch swap:          ~5-9x faster than shared_ptr transit
//
// Build: part of Chukonu benchmark suite (CMake bench_bolt target)
// Standalone: g++ -O3 -std=c++20 -march=native -pthread bench_bolt.cpp

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <mutex>
#include <memory>
#include <atomic>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_channel.h"
#include "bolt/bolt_port.h"

using Clock = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;
using namespace bolt;

template <typename T>
inline void DoNotOptimize(T&& val) {
#if defined(_MSC_VER) && !defined(__clang__)
    (void)val;
    _ReadWriteBarrier();
#else
    asm volatile("" : : "g"(val) : "memory");
#endif
}
inline void ClobberMemory() {
#if defined(_MSC_VER) && !defined(__clang__)
    _ReadWriteBarrier();
#else
    asm volatile("" ::: "memory");
#endif
}

// ============================================================================

void bench_arena_vs_malloc() {
    constexpr size_t N = 100000;
    constexpr size_t SZ = 16384;

    printf("=== Arena vs malloc (%zuKB × %zu) ===\n", SZ/1024, N);

    // malloc + free
    {
        std::vector<void*> ptrs(N);
        auto start = Clock::now();
        for (int iter = 0; iter < 10; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                ptrs[i] = std::malloc(SZ);
                DoNotOptimize(ptrs[i]);
            }
            for (size_t i = 0; i < N; ++i) std::free(ptrs[i]);
        }
        auto end = Clock::now();
        printf("  malloc+free:     %8.1f ns/alloc\n",
               (double)std::chrono::duration_cast<ns>(end - start).count() / (N * 10));
    }

    // Arena
    {
        Arena arena;
        auto start = Clock::now();
        for (int iter = 0; iter < 10; ++iter) {
            for (size_t i = 0; i < N; ++i) {
                void* p = arena.allocate(SZ);
                DoNotOptimize(p);
            }
            arena.reset();
        }
        auto end = Clock::now();
        printf("  arena+reset:     %8.1f ns/alloc\n",
               (double)std::chrono::duration_cast<ns>(end - start).count() / (N * 10));
    }
    printf("\n");
}

void bench_epoch_swap() {
    constexpr size_t N = 50000000;
    printf("=== Epoch swap vs shared_ptr (N=%zuM) ===\n", N/1000000);

    // Epoch swap
    {
        struct alignas(64) { uint8_t idx = 0; std::atomic<uint64_t> dirty{0}; } db;
        auto start = Clock::now();
        for (size_t i = 0; i < N; ++i) {
            db.idx ^= 1;
            db.dirty.store(0, std::memory_order_release);
            DoNotOptimize(db.idx);
        }
        auto end = Clock::now();
        printf("  epoch swap:          %5.1f ns\n",
               (double)std::chrono::duration_cast<ns>(end - start).count() / N);
    }

    // shared_ptr 8× copy (Arrow pipeline transit without moves)
    {
        auto sp = std::make_shared<int>(42);
        auto start = Clock::now();
        for (size_t i = 0; i < N; ++i) {
            auto p1 = sp; auto p2 = p1; auto p3 = p2; auto p4 = p3;
            auto p5 = p4; auto p6 = p5; auto p7 = p6; auto p8 = p7;
            DoNotOptimize(p8.get());
        }
        auto end = Clock::now();
        printf("  shared_ptr 8× copy:  %5.1f ns\n",
               (double)std::chrono::duration_cast<ns>(end - start).count() / N);
    }
    printf("\n");
}

void bench_spsc_channel() {
    constexpr size_t N = 10000000;
    printf("=== SPSC Channel vs mutex (N=%zuM) ===\n", N/1000000);

    // SPSC
    {
        SPSCChannel<uint64_t, 8192> ch;
        std::thread producer([&]() {
            for (uint64_t i = 0; i < N; ++i) {
                uint64_t v = i + 1;
                while (!ch.try_push(std::move(v))) {}
            }
        });

        auto start = Clock::now();
        uint64_t count = 0, sum = 0;
        while (count < N) {
            uint64_t val;
            if (ch.try_pop(&val)) { sum += val; count++; }
        }
        auto end = Clock::now();
        producer.join();

        printf("  SPSC:    %6.1f ns/op  (%5.1f M ops/sec)\n",
               (double)std::chrono::duration_cast<ns>(end - start).count() / N,
               N / ((double)std::chrono::duration_cast<ns>(end - start).count() / 1e9) / 1e6);
    }

    // mutex queue
    {
        std::mutex mtx;
        std::vector<uint64_t> queue;
        queue.reserve(8192);

        std::thread producer([&]() {
            for (uint64_t i = 0; i < N; ++i) {
                std::lock_guard<std::mutex> lk(mtx);
                queue.push_back(i + 1);
            }
        });

        auto start = Clock::now();
        uint64_t count = 0;
        while (count < N) {
            std::lock_guard<std::mutex> lk(mtx);
            if (!queue.empty()) {
                DoNotOptimize(queue.back());
                queue.pop_back();
                count++;
            }
        }
        auto end = Clock::now();
        producer.join();

        printf("  mutex:   %6.1f ns/op  (%5.1f M ops/sec)\n",
               (double)std::chrono::duration_cast<ns>(end - start).count() / N,
               N / ((double)std::chrono::duration_cast<ns>(end - start).count() / 1e9) / 1e6);
    }
    printf("\n");
}

void bench_cow_memcpy() {
    printf("=== COW memcpy cost (per-column clone-on-write) ===\n");
    struct { const char* label; size_t size; } specs[] = {
        {"2K×i32 (8KB)",    2048*4},
        {"16K×i32 (64KB)",  16384*4},
        {"16K×i64 (128KB)", 16384*8},
        {"256K×i64 (2MB)",  262144*8},
    };

    for (auto& s : specs) {
        void* src = bolt_aligned_alloc(64, s.size);
        void* dst = bolt_aligned_alloc(64, s.size);
        std::memset(src, 0xAB, s.size);

        constexpr size_t ITERS = 50000;
        auto start = Clock::now();
        for (size_t i = 0; i < ITERS; ++i) {
            std::memcpy(dst, src, s.size);
            ClobberMemory();
        }
        auto end = Clock::now();
        double per_ns = (double)std::chrono::duration_cast<ns>(end - start).count() / ITERS;
        printf("  %s: %8.1f ns  (%.1f GB/s)\n", s.label, per_ns, s.size / per_ns);

        bolt_aligned_free(src);
        bolt_aligned_free(dst);
    }
    printf("\n");
}

// D2 — non-temporal (streaming) store copy; wins when dst size > L3 and
// the destination will not be re-read soon. Below L3 memcpy stays ahead
// because its stores land in cache for the follow-up reader.
void bench_nt_memcpy() {
    printf("=== bolt_memcpy_nt vs memcpy (streaming NT stores) ===\n");
    struct { const char* label; size_t size; } specs[] = {
        {"256K×i64   (2MB)",   262144ull*8},
        {"1M×i64     (8MB)",  1048576ull*8},
        {"4M×i64    (32MB)",  4194304ull*8},
        {"16M×i64  (128MB)", 16777216ull*8},
    };

    for (auto& s : specs) {
        void* src = bolt_aligned_alloc(64, s.size);
        void* dst = bolt_aligned_alloc(64, s.size);
        std::memset(src, 0xAB, s.size);

        // Fewer iters for big sizes to keep wall-time bounded.
        size_t iters = s.size >= (16ull << 20) ? 200
                    : s.size >= ( 8ull << 20) ? 1000
                                              : 5000;

        auto t0 = Clock::now();
        for (size_t i = 0; i < iters; ++i) { std::memcpy(dst, src, s.size); ClobberMemory(); }
        auto t1 = Clock::now();
        double mc_ns = (double)std::chrono::duration_cast<ns>(t1 - t0).count() / iters;

        auto t2 = Clock::now();
        for (size_t i = 0; i < iters; ++i) { bolt::bolt_memcpy_nt(dst, src, s.size); ClobberMemory(); }
        auto t3 = Clock::now();
        double nt_ns = (double)std::chrono::duration_cast<ns>(t3 - t2).count() / iters;

        printf("  %s: memcpy %7.0f ns (%.2f GB/s)  nt %7.0f ns (%.2f GB/s)  ratio %.2f×\n",
               s.label, mc_ns, s.size / mc_ns, nt_ns, s.size / nt_ns, mc_ns / nt_ns);

        bolt_aligned_free(src);
        bolt_aligned_free(dst);
    }
    printf("\n");
}

int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Bolt Primitives Benchmark                          ║\n");
    printf("║  Arena / Channel / COW — Chukonu Hot Path            ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    bench_arena_vs_malloc();
    bench_epoch_swap();
    bench_spsc_channel();
    bench_cow_memcpy();
    bench_nt_memcpy();

    return 0;
}
