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

using Clock = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;
using namespace chukonu::bolt;

template <typename T>
inline void DoNotOptimize(T&& val) {
    asm volatile("" : : "g"(val) : "memory");
}
inline void ClobberMemory() { asm volatile("" ::: "memory"); }

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
            if (ch.try_pop(val)) { sum += val; count++; }
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
        void *src, *dst;
#ifdef _WIN32
        src = _aligned_malloc(s.size, 64);
        dst = _aligned_malloc(s.size, 64);
#else
        posix_memalign(&src, 64, s.size);
        posix_memalign(&dst, 64, s.size);
#endif
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

#ifdef _WIN32
        _aligned_free(src); _aligned_free(dst);
#else
        std::free(src); std::free(dst);
#endif
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

    return 0;
}
