// fd_registry.h — internal fd → handler-data registry for the event loops.
//
// G2CHK-85 audit: the epoll/kqueue/IOCP event loops kept their fd→HandlerData
// mapping in a std::unordered_map, looked up on EVERY dispatched I/O event —
// exactly the container Tiger Style bans on hot paths (pointer-chasing,
// cache-hostile, allocate-per-insert). This registry replaces it with:
//
//   - bolt::SwissTableGrowable (fd → slot index): SIMD group-scan probe,
//     real erase (tombstone + never-full revert), arena-backed bounded
//     growth. One table per event loop; one loop = one thread, so no locks
//     (matches the old unordered_map's implicit single-thread contract).
//   - a segmented slot pool for the payload: handler data (std::function —
//     non-trivially-copyable, so it cannot live inside an arena-memset
//     table) sits in fixed 512-entry segments allocated from the same
//     arena and placement-new'd once. Segment addresses are STABLE: a
//     Data* stays valid across insert-driven table growth — which also
//     fixes a latent bug in the unordered_map version, where a handler
//     that called add_fd() mid-dispatch could rehash the map and dangle
//     the iterator the dispatcher was still reading through.
//   - a freelist chaining released slots (parallel u32 array per segment,
//     so Data needs no intrusive field).
//
// Bounded everything: kMaxSegs * kSegSize slots hard cap (524,288 fds per
// loop), table ceiling sized to keep <= 50% steady load at that cap; when
// full, insert() returns nullptr and the caller reports ENOMEM — no silent
// growth past the bound.
//
// Not thread-safe, by design — see the thread-safety note in
// bolt_swiss_growable.h.

#pragma once

#include <cassert>
#include <cstdint>
#include <new>

#include "bolt/bolt_arena.h"
#include "bolt/join/bolt_swiss_growable.h"

namespace bolt::api {
namespace net {

template <typename Data>
class FdRegistry {
public:
    static constexpr uint32_t kNilSlot = 0xFFFFFFFFu;
    static constexpr uint32_t kSegBits = 9;              // 512 entries/segment
    static constexpr uint32_t kSegSize = 1u << kSegBits;
    static constexpr uint32_t kMaxSegs = 1024;           // 512K fds hard cap
    static constexpr uint32_t kTableCeiling = 1u << 21;  // <=50% load at cap

    FdRegistry() noexcept
        : arena_(registry_arena_config()), seg_count_(0), slot_count_(0),
          free_head_(kNilSlot) {
        for (uint32_t i = 0; i < kMaxSegs; ++i) {
            segs_[i] = nullptr;
            seg_links_[i] = nullptr;
        }
        links_ok_ = SwissTableGrowable::create(&table_, 64, &arena_, kTableCeiling);
        assert(links_ok_ && "fd registry table creation cannot fail at 64 slots");
    }

    ~FdRegistry() noexcept {
        // Destroy every constructed Data (std::function captures must free).
        for (uint32_t i = 0; i < slot_count_; ++i) {
            slot_at(i)->~Data();
        }
    }

    FdRegistry(const FdRegistry&) = delete;
    FdRegistry& operator=(const FdRegistry&) = delete;

    // Hot path: one SIMD-probed lookup + one indexed load. nullptr = absent.
    Data* find(uint64_t fd) noexcept {
        assert(links_ok_);
        const int32_t v = table_.find(fd);
        if (v < 0) return nullptr;
        assert(static_cast<uint32_t>(v) < slot_count_);
        return slot_at(static_cast<uint32_t>(v));
    }

    // Insert-or-overwrite (unordered_map operator[] semantics): an existing
    // fd returns its current slot for the caller to overwrite. Returns
    // nullptr only when the registry is at its hard cap (caller → ENOMEM).
    Data* insert(uint64_t fd) noexcept {
        assert(links_ok_);
        const int32_t existing = table_.find(fd);
        if (existing >= 0) return slot_at(static_cast<uint32_t>(existing));

        uint32_t idx = kNilSlot;
        if (!alloc_slot(&idx)) return nullptr;
        if (!table_.insert(fd, idx)) {  // ceiling or arena OOM — undo cleanly
            release_slot(idx);
            return nullptr;
        }
        assert(idx < slot_count_);
        return slot_at(idx);
    }

    // Erase. Returns false if `fd` is absent. The slot's Data is reset to a
    // default-constructed state (releasing captures) and recycled.
    bool erase(uint64_t fd) noexcept {
        assert(links_ok_);
        const int32_t v = table_.find(fd);
        if (v < 0) return false;
        const bool erased = table_.erase(fd);
        assert(erased);
        (void)erased;
        release_slot(static_cast<uint32_t>(v));
        return true;
    }

    uint32_t size() const noexcept { return table_.size; }

private:
    static ArenaConfig registry_arena_config() noexcept {
        ArenaConfig cfg;
        cfg.initial_block_size = 64u * 1024;       // registries start small
        cfg.max_block_size = 64u * 1024 * 1024;
        return cfg;
    }

    Data* slot_at(uint32_t idx) const noexcept {
        assert((idx >> kSegBits) < seg_count_);
        assert(segs_[idx >> kSegBits] != nullptr);
        return &segs_[idx >> kSegBits][idx & (kSegSize - 1u)];
    }

    bool alloc_slot(uint32_t* out) noexcept {
        assert(out != nullptr);
        if (free_head_ != kNilSlot) {
            const uint32_t idx = free_head_;
            free_head_ = link_at(idx);
            *out = idx;
            return true;
        }
        if (slot_count_ == seg_count_ * kSegSize) {
            if (seg_count_ >= kMaxSegs) return false;  // hard cap, honest fail
            Data* seg = static_cast<Data*>(
                arena_.allocate(sizeof(Data) * kSegSize, alignof(Data)));
            uint32_t* links = arena_.allocate_array<uint32_t>(kSegSize);
            if (!seg || !links) return false;
            for (uint32_t i = 0; i < kSegSize; ++i) {
                new (&seg[i]) Data();
                links[i] = kNilSlot;
            }
            segs_[seg_count_] = seg;
            seg_links_[seg_count_] = links;
            ++seg_count_;
        }
        assert(slot_count_ < seg_count_ * kSegSize);
        *out = slot_count_++;
        return true;
    }

    void release_slot(uint32_t idx) noexcept {
        assert(idx < slot_count_);
        *slot_at(idx) = Data();  // drop captures now, not at pool teardown
        link_at(idx) = free_head_;
        free_head_ = idx;
    }

    uint32_t& link_at(uint32_t idx) const noexcept {
        assert((idx >> kSegBits) < seg_count_);
        return seg_links_[idx >> kSegBits][idx & (kSegSize - 1u)];
    }

    Arena arena_;
    SwissTableGrowable table_;
    Data* segs_[kMaxSegs];
    uint32_t* seg_links_[kMaxSegs];
    uint32_t seg_count_;
    uint32_t slot_count_;
    uint32_t free_head_;
    bool links_ok_ = false;
};

}  // namespace net
}  // namespace bolt::api
