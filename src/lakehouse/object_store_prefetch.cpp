// bolt/lakehouse/object_store_prefetch.cpp — async object prefetcher.
//
// One worker thread drains a request SPSC ring, calls os_get for each URI,
// and pushes the result on a result SPSC ring. The arena passed at open
// time owns both the Prefetcher itself and the fetched buffers (allocated
// inside the get_object call).

#include "bolt/lakehouse/object_store/prefetch.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#include "bolt/bolt_channel.h"   // SPSCChannel

namespace bolt {
namespace lakehouse {
namespace prefetch {

// Capacity is a compile-time template arg for SPSCChannel; we cap at 64
// (kPrefetchMaxDepth) and use the full ring regardless of requested depth.
// Depth still controls the producer-side back-pressure point — when more
// than `depth` requests are in flight the producer's try_push returns false.
static constexpr size_t kRingCapacity = 64u;

struct Prefetcher {
    Arena*                                       arena;
    ObjectStore*                                 store;
    SPSCChannel<PrefetchRequest, kRingCapacity>  req_ring;
    SPSCChannel<PrefetchResult,  kRingCapacity>  res_ring;
    std::thread                                  worker;
    std::atomic<bool>                            running;
    std::atomic<bool>                            stop_flag;
    uint32_t                                     depth;
    std::atomic<uint32_t>                        in_flight;
    uint8_t                                      _pad[4];
};

namespace {

void worker_loop(Prefetcher* p) noexcept {
    assert(p != nullptr);
    while (!p->stop_flag.load(std::memory_order_acquire)) {
        PrefetchRequest req;
        if (!p->req_ring.try_pop(&req)) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }
        PrefetchResult res;
        std::memset(&res, 0, sizeof(res));
        res.slot_id = req.slot_id;
        const uint8_t* data = nullptr;
        uint64_t len = 0;
        const int rc = os_get(p->store, req.uri, p->arena, &data, &len);
        res.status = rc;
        if (rc == kOsOk) {
            res.buf = const_cast<uint8_t*>(data);
            res.len = static_cast<uint32_t>(len);
            // Optional range slice when caller asked for one.
            if (req.range_lo >= 0 && req.range_hi >= req.range_lo) {
                const uint32_t lo = static_cast<uint32_t>(req.range_lo);
                const uint32_t hi = static_cast<uint32_t>(req.range_hi);
                if (lo < res.len) {
                    const uint32_t end = hi + 1u > res.len ? res.len : hi + 1u;
                    res.buf = res.buf + lo;
                    res.len = end - lo;
                }
            }
            res.ok = true;
        } else {
            res.ok = false;
        }
        // Spin until the result ring has room — bounded by consumer pace.
        // The producer is expected to poll fairly; this is the natural
        // back-pressure point.
        while (!p->res_ring.try_push(static_cast<PrefetchResult&&>(res))) {
            if (p->stop_flag.load(std::memory_order_acquire)) return;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        p->in_flight.fetch_sub(1u, std::memory_order_acq_rel);
    }
}

}  // namespace

bool prefetcher_open(Prefetcher** out, Arena* arena, ObjectStore* store,
                     uint32_t depth) noexcept {
    assert(out != nullptr);
    assert(arena != nullptr);
    assert(store != nullptr);
    if (out == nullptr || arena == nullptr || store == nullptr) return false;
    if (depth == 0u) depth = kPrefetchDefaultDepth;
    if (depth > kPrefetchMaxDepth) return false;
    // Placement-new the Prefetcher in the arena so its SPSC channels (which
    // are not trivially default-constructible in C++ language terms) get
    // their constructors called.
    void* mem = arena->allocate(sizeof(Prefetcher), alignof(Prefetcher));
    if (mem == nullptr) return false;
    Prefetcher* p = new (mem) Prefetcher();
    p->arena = arena;
    p->store = store;
    p->running.store(true, std::memory_order_release);
    p->stop_flag.store(false, std::memory_order_release);
    p->depth = depth;
    p->in_flight.store(0u, std::memory_order_release);
    p->worker = std::thread(worker_loop, p);
    *out = p;
    assert(*out != nullptr);
    return true;
}

bool prefetcher_submit(Prefetcher* p, const PrefetchRequest* req) noexcept {
    assert(p != nullptr);
    assert(req != nullptr);
    if (p == nullptr || req == nullptr) return false;
    const uint32_t live = p->in_flight.load(std::memory_order_acquire);
    if (live >= p->depth) return false;
    PrefetchRequest copy = *req;
    if (!p->req_ring.try_push(static_cast<PrefetchRequest&&>(copy))) {
        return false;
    }
    p->in_flight.fetch_add(1u, std::memory_order_acq_rel);
    return true;
}

bool prefetcher_poll(Prefetcher* p, PrefetchResult* out,
                     uint32_t timeout_ms) noexcept {
    assert(p != nullptr);
    assert(out != nullptr);
    if (p == nullptr || out == nullptr) return false;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    while (true) {
        if (p->res_ring.try_pop(out)) return true;
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void prefetcher_close(Prefetcher* p) noexcept {
    if (p == nullptr) return;
    p->stop_flag.store(true, std::memory_order_release);
    if (p->worker.joinable()) p->worker.join();
    p->running.store(false, std::memory_order_release);
    // Arena owns the memory; no free here. Run dtor to release std::thread
    // resources cleanly.
    p->~Prefetcher();
}

}  // namespace prefetch
}  // namespace lakehouse
}  // namespace bolt
