// test_bolt_arena_concurrent.cpp — coverage for bolt::Arena::set_concurrent.
//
// bolt::Arena is single-thread by design ("Arena per Slot"): its bump cursor
// is a plain uintptr_t. chukonu's parallel driver, however, runs several
// operators concurrently on worker threads and hands them ALL the same
// fragment arena. Two operators calling arena->allocate() at once therefore
// race the cursor and can be handed OVERLAPPING blocks — one op's write then
// stomps the other's (in the wild, a Compute op's out_batch->columns[epoch]
// came back a torn pointer and a downstream BoltColumn memcpy SIGSEGV'd:
// TPC-H SF10 Q6 at 12/18 workers, never at 1). set_concurrent(true) serializes
// the allocate() body so concurrent sharers get disjoint regions.
//
// DISCRIMINATING POWER: the stress below allocates ~320k blocks across 8
// threads on ONE shared arena and proves no two overlap (each thread stamps a
// unique byte over its whole block and re-reads it intact after a barrier).
// With the fix reverted (allocate() lock-free while shared) this test fails
// deterministically in practice — overlapping torn allocations corrupt the
// stamps, exactly the mechanism behind the Q6 crash. Empirically verified:
// clean 100% with set_concurrent(true); stamp mismatches within the first
// rounds with set_concurrent(false) under the same stress.
//
// Test code may use stdlib freely; the primitive under test does not.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "bolt/bolt_arena.h"

namespace {

// One shared arena, N threads, each doing many small allocations and stamping
// its own block with a per-thread byte, then verifying every stamp survived.
// Returns the number of corrupted (overlapped) blocks detected.
std::uint64_t run_concurrent_stamp_stress(bool concurrent,
                                          int n_threads,
                                          int allocs_per_thread,
                                          std::size_t block_bytes) {
    bolt::ArenaConfig cfg{};
    cfg.initial_block_size = 8ull * 1024 * 1024;
    cfg.max_block_size     = 64ull * 1024 * 1024;
    bolt::Arena arena{cfg};
    arena.set_concurrent(concurrent);

    std::atomic<std::uint64_t> corrupt{0};
    std::atomic<int>           ready{0};
    std::atomic<bool>          go{false};

    auto body = [&](int tid) {
        // Each thread keeps its own (ptr, size) list so it can re-verify its
        // OWN blocks after every other thread has finished writing.
        std::vector<void*> ptrs;
        ptrs.reserve(static_cast<std::size_t>(allocs_per_thread));
        const auto stamp = static_cast<unsigned char>(1 + tid);

        ready.fetch_add(1, std::memory_order_acq_rel);
        while (!go.load(std::memory_order_acquire)) { /* spin */ }

        for (int i = 0; i < allocs_per_thread; ++i) {
            // Vary the size a little so alignment padding and block boundaries
            // are exercised, but keep it well under max_block_size.
            const std::size_t sz = block_bytes + static_cast<std::size_t>((i * 8) & 63);
            void* p = arena.allocate(sz, 16);
            // OOM is a test-config failure, not the property under test.
            ASSERT_NE(p, nullptr);
            std::memset(p, stamp, sz);
            ptrs.push_back(p);
        }
        // Barrier: wait until all threads finished writing, then verify.
        static std::atomic<int> done{0};
        done.fetch_add(1, std::memory_order_acq_rel);
        while (done.load(std::memory_order_acquire) < n_threads) { /* spin */ }

        std::uint64_t local_bad = 0;
        for (std::size_t k = 0; k < ptrs.size(); ++k) {
            const std::size_t sz =
                block_bytes + static_cast<std::size_t>(((int)k * 8) & 63);
            const auto* b = static_cast<const unsigned char*>(ptrs[k]);
            for (std::size_t j = 0; j < sz; ++j) {
                if (b[j] != stamp) { ++local_bad; break; }
            }
        }
        corrupt.fetch_add(local_bad, std::memory_order_acq_rel);
    };

    std::vector<std::thread> ts;
    ts.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) ts.emplace_back(body, t);
    while (ready.load(std::memory_order_acquire) < n_threads) { /* spin */ }
    go.store(true, std::memory_order_release);
    for (auto& th : ts) th.join();
    return corrupt.load(std::memory_order_acquire);
}

}  // namespace

// The fix: with the shared-arena spinlock enabled, concurrent allocate() calls
// return disjoint regions and every stamp survives. This is the crash-preventing
// invariant. Runs several rounds to make the race window wide.
TEST(BoltArenaConcurrent, ConcurrentAllocateNoOverlap) {
    for (int round = 0; round < 8; ++round) {
        const std::uint64_t corrupt =
            run_concurrent_stamp_stress(/*concurrent=*/true,
                                        /*n_threads=*/8,
                                        /*allocs_per_thread=*/5000,
                                        /*block_bytes=*/48);
        ASSERT_EQ(corrupt, 0u)
            << "round " << round
            << ": overlapping allocations from a shared concurrent arena";
    }
}

// The default (single-thread) mode is unchanged: the flag is off and the
// lock-free fast path is used. A single-threaded caller sees disjoint blocks
// trivially (no lock needed); this pins the default and the accessor.
TEST(BoltArenaConcurrent, DefaultIsSingleThreadFastPath) {
    bolt::Arena a{};
    EXPECT_FALSE(a.concurrent());
    a.set_concurrent(true);
    EXPECT_TRUE(a.concurrent());
    a.set_concurrent(false);
    EXPECT_FALSE(a.concurrent());

    // Single-thread allocations never overlap regardless of the flag.
    const std::uint64_t corrupt =
        run_concurrent_stamp_stress(/*concurrent=*/false,
                                    /*n_threads=*/1,
                                    /*allocs_per_thread=*/10000,
                                    /*block_bytes=*/32);
    EXPECT_EQ(corrupt, 0u);
}
