// test_bolt_pool_lifecycle.cpp — TypedBatchPool / ArenaRing slot lifetime.
//
// G2CHK-148: slots held `Arena arena;` as a plain member, so constructing a
// pool ran Arena() (a 4 MiB initial block) for EVERY slot, and the first
// acquire of a slot placement-new'd a second Arena over the first, dropping
// its block. LSan saw one 4 MiB direct leak per acquired slot. These tests
// measure heap bytes in use around construction, acquire and destruction.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <new>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_arena_ring.h"
#include "bolt/bolt_batch_pool.h"

#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define POOL_TEST_ASAN 1
#  endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#  define POOL_TEST_ASAN 1
#endif

#if defined(POOL_TEST_ASAN)
#  include <sanitizer/allocator_interface.h>
#elif defined(__APPLE__)
#  include <malloc/malloc.h>
#elif defined(__linux__) && defined(__GLIBC__)
#  include <malloc.h>
#endif

namespace {

constexpr std::size_t kBlock = std::size_t{1} << 20;   // 1 MiB initial block
constexpr uint32_t    kSlots = 8u;

bool heap_in_use(std::size_t* out) {
#if defined(POOL_TEST_ASAN)
    *out = __sanitizer_get_current_allocated_bytes();
    return true;
#elif defined(__APPLE__)
    malloc_statistics_t st{};
    malloc_zone_statistics(nullptr, &st);
    *out = st.size_in_use;
    return true;
#elif defined(__linux__) && defined(__GLIBC__)
    const struct mallinfo2 mi = mallinfo2();
    *out = mi.uordblks + mi.hblkhd;
    return true;
#else
    (void)out;
    return false;
#endif
}

bolt::ArenaConfig small_cfg() {
    bolt::ArenaConfig cfg{};
    cfg.initial_block_size = kBlock;
    return cfg;
}

// Growth in heap bytes, tolerant of unrelated sub-block noise.
std::ptrdiff_t delta(std::size_t before, std::size_t after) {
    return static_cast<std::ptrdiff_t>(after) - static_cast<std::ptrdiff_t>(before);
}

template <typename Pool, typename AcquireRelease>
void check_pool_lifecycle(AcquireRelease&& cycle) {
    std::size_t base = 0;
    if (!heap_in_use(&base)) GTEST_SKIP() << "no heap statistics on this platform";

    // Default Arena config is 4 MiB; an eager per-slot Arena() shows up here.
    alignas(Pool) static unsigned char storage[sizeof(Pool)];
    Pool* pool = new (storage) Pool();
    std::size_t after_ctor = 0;
    ASSERT_TRUE(heap_in_use(&after_ctor));
    EXPECT_LT(delta(base, after_ctor), static_cast<std::ptrdiff_t>(kBlock))
        << "pool construction allocated arena blocks";

    pool->init(small_cfg());
    cycle(pool);
    std::size_t after_use = 0;
    ASSERT_TRUE(heap_in_use(&after_use));
    EXPECT_GE(delta(base, after_use), static_cast<std::ptrdiff_t>(kBlock))
        << "acquire did not construct the slot arena";

    pool->~Pool();
    std::size_t after_dtor = 0;
    ASSERT_TRUE(heap_in_use(&after_dtor));
    EXPECT_LT(delta(base, after_dtor), static_cast<std::ptrdiff_t>(kBlock))
        << "arena blocks survived pool destruction (leaked)";
}

}  // namespace

TEST(TypedBatchPoolLifecycle, NoEagerArenaAndNoLeakOnAcquire) {
    using Pool = bolt::TypedBatchPool<kSlots>;
    check_pool_lifecycle<Pool>([](Pool* p) {
        bolt::TypedBatchPoolSlot* a = p->acquire();
        bolt::TypedBatchPoolSlot* b = p->acquire();
        ASSERT_NE(a, nullptr);
        ASSERT_NE(b, nullptr);
        ASSERT_EQ(a->batch.arena, &a->arena);
        ASSERT_NE(a->arena.allocate(64), nullptr);
        p->release(a);
        p->release(b);
        ASSERT_EQ(p->acquire(), b);   // LIFO reuse, no fresh construction
    });
}

TEST(ArenaRingLifecycle, NoEagerArenaAndNoLeakOnAcquire) {
    using Pool = bolt::ArenaRing<kSlots>;
    check_pool_lifecycle<Pool>([](Pool* p) {
        bolt::Arena* a = p->acquire();
        bolt::Arena* b = p->acquire();
        ASSERT_NE(a, nullptr);
        ASSERT_NE(b, nullptr);
        ASSERT_NE(a->allocate(64), nullptr);
        p->release(a);
        p->release(b);
        ASSERT_EQ(p->acquire(), b);
    });
}

// Explicit destroy() followed by the destructor must not double-free.
TEST(ArenaRingLifecycle, DestroyThenDestructIsSafe) {
    bolt::ArenaRing<kSlots>* ring = new bolt::ArenaRing<kSlots>();
    ring->init(small_cfg());
    ASSERT_NE(ring->acquire(), nullptr);
    ring->destroy();
    delete ring;
    SUCCEED();
}
