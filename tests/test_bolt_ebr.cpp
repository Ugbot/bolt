// test_bolt_ebr.cpp — GTest suite for bolt::Ebr.
//
// Validates the 3-epoch reclamation scheme:
//   1) Single-threaded enter/retire/collect — freed after 2 advances.
//   2) Pinned reader blocks free until exit.
//   3) Multi-threaded soak: 4 workers + 1 collector, shadow counter
//      asserts no use-after-free and all retires eventually free.
//   4) Retire queue full returns false.
//   5) ebr_destroy drains everything.
//
// Uses std::thread + std::atomic in tests (allowed — not hot path).

#include <gtest/gtest.h>

#include "bolt/bolt_ebr.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

using namespace bolt;

// ---------------------------------------------------------------------------
// Test payload: a heap-allocated struct with a live/dead flag that we
// check before deref to catch use-after-free in the multi-threaded test.
// ---------------------------------------------------------------------------
struct TestObj {
    std::atomic<uint32_t> magic;     // kAlive while reachable, kDead after free
    uint64_t              payload;   // initialised to v, scrubbed to kScrub on free
};

static constexpr uint32_t kAlive = 0xA11AEu;  // "Alive-ish" — hex digits only
static constexpr uint32_t kDead  = 0xDEADu;
static constexpr uint64_t kScrub = 0xDEADBEEFBADDCAFEull;

// Global counter of total free calls — used to verify destroy drains.
static std::atomic<uint64_t> g_free_count{0};

static TestObj* make_obj(uint64_t v) noexcept {
    auto* o = static_cast<TestObj*>(std::malloc(sizeof(TestObj)));
    new (&o->magic) std::atomic<uint32_t>(kAlive);
    o->payload = v;
    return o;
}

// Shadow-UAF pattern: mark magic=kDead AND scrub payload BEFORE std::free
// so a racing reader that still dereferences a freed pointer observes a
// corrupt pattern (kDead / kScrub) instead of lingering-but-valid bytes.
static void free_obj(void* p) noexcept {
    auto* o = static_cast<TestObj*>(p);
    o->magic.store(kDead, std::memory_order_release);
    o->payload = kScrub;
    std::free(o);
    g_free_count.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Test 1 — single-threaded retire/collect freed after 2 advances.
// ---------------------------------------------------------------------------
TEST(BoltEbr, SingleThreadedRetireAndCollect) {
    g_free_count.store(0, std::memory_order_relaxed);

    Ebr e;
    ebr_init(&e, 1);

    TestObj* obj = make_obj(42);
    ASSERT_TRUE(ebr_retire(&e, 0, obj, &free_obj));

    // No readers pinned. Two advances are needed before the object
    // enqueued in epoch 0 sits in the (ge+1 mod 3) drain slot.
    uint32_t freed = 0;
    for (int i = 0; i < 4; ++i) freed += ebr_try_advance_and_collect(&e);
    EXPECT_EQ(freed, 1u);
    EXPECT_EQ(g_free_count.load(), 1u);

    ebr_destroy(&e);
}

// ---------------------------------------------------------------------------
// Test 2 — a pinned reader must block drain of its epoch.
// ---------------------------------------------------------------------------
TEST(BoltEbr, PinnedReaderBlocksReclaim) {
    g_free_count.store(0, std::memory_order_relaxed);

    Ebr e;
    ebr_init(&e, 2);

    // Shard 0 retires; shard 1 is a reader.
    TestObj* obj = make_obj(7);
    ebr_enter(&e, 1);
    ASSERT_TRUE(ebr_retire(&e, 0, obj, &free_obj));

    // Advance many times — reader pinned at epoch 0 blocks drain of slot 0.
    for (int i = 0; i < 8; ++i) (void)ebr_try_advance_and_collect(&e);
    EXPECT_EQ(g_free_count.load(), 0u);
    EXPECT_EQ(obj->magic.load(), kAlive);

    // Reader exits; eventually the object is freed.
    ebr_exit(&e, 1);
    uint32_t freed = 0;
    for (int i = 0; i < 8 && freed == 0; ++i) {
        freed += ebr_try_advance_and_collect(&e);
    }
    EXPECT_EQ(freed, 1u);
    EXPECT_EQ(g_free_count.load(), 1u);

    ebr_destroy(&e);
}

// ---------------------------------------------------------------------------
// Test 3 — multi-threaded soak. 4 worker threads continually publish new
// objects, read them under pin, and retire the old. 1 collector thread
// advances the epoch. A reader that observes kDead is a use-after-free.
// ---------------------------------------------------------------------------
TEST(BoltEbr, MultiThreadedNoUseAfterFree) {
    g_free_count.store(0, std::memory_order_relaxed);

    Ebr e;
    ebr_init(&e, 5);  // 4 workers + 1 collector shard (unused by collector)

    constexpr int kWorkers         = 4;
    constexpr int kOpsPerWorker    = 4000;
    std::atomic<TestObj*> published[kWorkers];
    for (int i = 0; i < kWorkers; ++i) published[i].store(make_obj(0));

    std::atomic<uint32_t> uaf_seen{0};
    std::atomic<uint64_t> total_retired{0};
    std::atomic<bool>     stop{false};

    auto worker = [&](int id) noexcept {
        for (int i = 0; i < kOpsPerWorker; ++i) {
            // Read phase — pin, load, verify BOTH magic and payload-not-
            // scrubbed; unpin. Either check failing means a freed object
            // was reachable under pin (use-after-free).
            ebr_enter(&e, static_cast<uint32_t>(id));
            TestObj* snap = published[id].load(std::memory_order_acquire);
            const uint32_t m = snap->magic.load(std::memory_order_acquire);
            const uint64_t p = snap->payload;
            if (m != kAlive || p == kScrub) {
                uaf_seen.fetch_add(1, std::memory_order_relaxed);
            }
            ebr_exit(&e, static_cast<uint32_t>(id));

            // Publish phase — install a fresh object, retire the old.
            TestObj* fresh = make_obj(static_cast<uint64_t>(i));
            TestObj* old   = published[id].exchange(fresh);

            // Push back until the retire lands. On queue-full we drive the
            // collector ourselves and yield to let it advance. Unbounded
            // retry is safe in this test because the collector thread is
            // always running; production code would set a cap + escalate.
            while (!ebr_retire(&e, static_cast<uint32_t>(id), old, &free_obj)) {
                (void)ebr_try_advance_and_collect(&e);
                std::this_thread::yield();
            }
            total_retired.fetch_add(1, std::memory_order_relaxed);
        }
    };

    auto collector = [&]() noexcept {
        while (!stop.load(std::memory_order_relaxed)) {
            (void)ebr_try_advance_and_collect(&e);
            std::this_thread::yield();
        }
    };

    std::thread ths[kWorkers];
    std::thread coll(collector);
    for (int i = 0; i < kWorkers; ++i) ths[i] = std::thread(worker, i);
    for (int i = 0; i < kWorkers; ++i) ths[i].join();
    stop.store(true, std::memory_order_relaxed);
    coll.join();

    EXPECT_EQ(uaf_seen.load(), 0u) << "reader observed a freed object";

    // Flush any lingering retires and the still-published head pointers.
    for (int i = 0; i < kWorkers; ++i) {
        TestObj* head = published[i].exchange(nullptr);
        ASSERT_TRUE(ebr_retire(&e, static_cast<uint32_t>(i), head, &free_obj));
        total_retired.fetch_add(1, std::memory_order_relaxed);
    }
    ebr_destroy(&e);

    EXPECT_EQ(g_free_count.load(), total_retired.load())
        << "every retired object should eventually be freed";
}

// ---------------------------------------------------------------------------
// Test 4 — retire queue full returns false.
// ---------------------------------------------------------------------------
TEST(BoltEbr, RetireFullReturnsFalse) {
    g_free_count.store(0, std::memory_order_relaxed);

    Ebr e;
    ebr_init(&e, 1);

    // Fill the epoch-0 slot of shard 0 completely.
    for (uint32_t i = 0; i < kEbrRetirePerEpoch; ++i) {
        ASSERT_TRUE(ebr_retire(&e, 0, make_obj(i), &free_obj));
    }
    // Next retire into the same epoch slot must fail.
    TestObj* overflow = make_obj(999);
    EXPECT_FALSE(ebr_retire(&e, 0, overflow, &free_obj));
    // Caller is responsible for overflow; free it directly to avoid leaks.
    free_obj(overflow);

    ebr_destroy(&e);
    EXPECT_EQ(g_free_count.load(), kEbrRetirePerEpoch + 1u);
}

// ---------------------------------------------------------------------------
// Test 5 — destroy drains everything across multiple epochs.
// ---------------------------------------------------------------------------
TEST(BoltEbr, DestroyDrainsAllEpochs) {
    g_free_count.store(0, std::memory_order_relaxed);

    Ebr e;
    ebr_init(&e, 2);

    // Retire an object, advance, retire another, advance, retire a third.
    // Each sits in a different epoch's retire slot.
    ASSERT_TRUE(ebr_retire(&e, 0, make_obj(1), &free_obj));
    (void)ebr_try_advance_and_collect(&e);
    ASSERT_TRUE(ebr_retire(&e, 0, make_obj(2), &free_obj));
    (void)ebr_try_advance_and_collect(&e);
    ASSERT_TRUE(ebr_retire(&e, 1, make_obj(3), &free_obj));

    ebr_destroy(&e);
    EXPECT_EQ(g_free_count.load(), 3u);
}
