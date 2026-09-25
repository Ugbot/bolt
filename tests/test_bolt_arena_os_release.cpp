// test_bolt_arena_os_release.cpp — G2LAUNCH-35: trimmed arena blocks must go
// back to the OS, not sit dirty in an allocator cache.
//
// Through malloc, macOS's large cache kept freed arena blocks dirty, so a
// long-lived process (ClickBench board, gestaltd) ratcheted to roughly the sum
// of its queries' peaks and hit the memory ceiling at 16 threads. Large blocks
// now come from mmap/munmap. The test touches ~1 GiB of arena blocks, trims
// them with reset_keep(), and requires the process's private footprint to
// drop by most of that. Reverting the mmap path fails it on macOS.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "bolt/bolt_arena.h"

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <cstdio>
#include <unistd.h>
#endif

namespace {

// Private resident bytes of this process, or 0 when not measurable here.
std::uint64_t private_footprint_bytes() {
#if defined(__APPLE__)
    task_vm_info_data_t vi{};
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&vi), &cnt) != KERN_SUCCESS) {
        return 0;
    }
    return static_cast<std::uint64_t>(vi.phys_footprint);
#elif defined(__linux__)
    std::FILE* f = std::fopen("/proc/self/statm", "r");
    if (f == nullptr) return 0;
    unsigned long long size = 0, resident = 0, shared = 0;
    const int n = std::fscanf(f, "%llu %llu %llu", &size, &resident, &shared);
    std::fclose(f);
    if (n != 3 || resident < shared) return 0;
    return (resident - shared) * static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
#else
    return 0;
#endif
}

constexpr std::uint64_t kMiB = 1024ull * 1024ull;

TEST(ArenaOsRelease, TrimmedBlocksLeaveTheProcess) {
    if (private_footprint_bytes() == 0) GTEST_SKIP() << "no footprint probe";
    bolt::ArenaConfig cfg{};
    cfg.initial_block_size = 64 * kMiB;
    cfg.max_block_size     = 256 * kMiB;
    bolt::Arena arena{cfg};

    // Warm-up epoch first, so allocator bookkeeping is not measured.
    for (int round = 0; round < 2; ++round) {
        const std::uint64_t before = private_footprint_bytes();
        std::uint64_t touched = 0;
        for (int i = 0; i < 16; ++i) {
            auto* p = static_cast<unsigned char*>(arena.allocate(64 * kMiB, 64));
            ASSERT_NE(p, nullptr);
            std::memset(p, 0xA5, 64 * kMiB);
            touched += 64 * kMiB;
        }
        ASSERT_GE(arena.total_reserved(), touched);
        const std::uint64_t grown = private_footprint_bytes();
        ASSERT_GE(grown, before + touched / 2) << "probe did not see the pages";

        arena.reset_keep(64 * kMiB);
        ASSERT_LE(arena.total_reserved(), 64 * kMiB);
        const std::uint64_t after = private_footprint_bytes();
        // Everything but the kept first block (and slack) must be gone.
        EXPECT_LE(after, before + 192 * kMiB)
            << "round " << round << ": before=" << (before >> 20)
            << "MiB grown=" << (grown >> 20) << "MiB after=" << (after >> 20)
            << "MiB";
    }
}

}  // namespace
