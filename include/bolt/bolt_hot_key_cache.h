// bolt_hot_key_cache.h — access-count-promoted hot-key LRU with
// true row-data caching. Lock-free via seqlock-per-slot semantics.
//
// RULES: No exceptions. No RTTI. Fixed capacity. Zero allocation.
//
// Motivation. v1 MarbleDB had an Aerospike-inspired HotKeyCache
// that pushed point-lookup throughput to 500 K – 1 M ops/s on
// skewed workloads. The key design points were:
//   - Copy the row DATA into the cache (not just location).
//   - Promote a key into the cache only after N accesses (default 3)
//     so random-uniform traffic doesn't pollute it.
//   - LRU/frequency-weighted eviction.
//   - Lock-free reads via RCU (v1 used `atomic_load(shared_ptr)`;
//     we use a per-slot seqlock pattern since Bolt bans smart ptrs).
//
// This is a template on:
//   K            — key type (int64 / uint64 / a fixed-size struct)
//   RowBytes     — size of the cached row payload in bytes
//   Capacity     — number of slots (power of 2)
//   PromotionThreshold — access count required before insertion
//
// Layout per slot: 64-byte `key_tag` atomic (key << 16 | gen << 8 |
// flags) + 64-byte `access_count` atomic + `layout_gen` + row bytes.
// Slot is cache-line-padded to avoid false sharing between hot slots.
//
// Read path:
//   1. Hash key → slot index (linear probe up to 8 slots).
//   2. Load `key_tag` with acquire. If flags show "writing", skip
//      (treat as miss) — caller falls through to the next layer.
//   3. Compare key + layout_gen. Mismatch → probe next slot.
//   4. On match: bump `access_count` (relaxed), return `row_data`.
//
// Write path (producer calling `try_promote`):
//   1. Look up the slot; if key already cached, update layout_gen +
//      row data in-place (seqlock).
//   2. If not cached, consult a separate `AccessTracker` (also in
//      this header) — only if tracker count >= threshold does the
//      key go into the cache, evicting the LRU victim.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/bolt_sequence.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {

namespace hotkey_detail {

// Splitmix-style mix for power-of-2 buckets.
BOLT_FORCE_INLINE uint64_t mix64(uint64_t x) noexcept {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

}  // namespace hotkey_detail

static constexpr uint8_t kHotKeyFlagOccupied = 1u;
static constexpr uint8_t kHotKeyFlagWriting  = 2u;
static constexpr uint32_t kHotKeyProbeDist   = 8u;

template <typename K, uint32_t RowBytes, uint32_t Capacity,
          uint32_t PromotionThreshold = 3>
struct alignas(64) HotKeyCache {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity power of 2");
    static_assert(RowBytes >= 8 && RowBytes <= 256, "RowBytes bound");

    struct alignas(64) Slot {
        // Packed: [63..16] = key bits (K fits in 48 bits minimum),
        // [15..8] = flags, [7..0] = layout_gen tag. Carried as a
        // single atomic so readers can load the whole metadata in
        // one op.
        std::atomic<uint64_t> key_tag;
        // Access counter — relaxed atomic, bumped on every hit.
        std::atomic<uint64_t> access_count;
        // Layout generation at insert time. Stale layout → miss.
        uint64_t              layout_gen;
        // Commit LSN of the cached row (the writer's monotonic version).
        // `lookup_copy` rejects a slot whose commit_lsn is older than the
        // caller's freshness horizon, so a cache that lags the authoritative
        // store by a write is treated as a miss (caller reads the store)
        // rather than serving a stale row. Written under the seqlock.
        uint64_t              commit_lsn;
        // Row data copy.
        uint8_t               row_data[RowBytes];
    };
    static_assert(sizeof(Slot) <= 384, "Slot exceeds 384B — bump padding math");

    Slot    slots[Capacity];
    // Monotonic access-tick counter for LRU scoring.
    Sequence access_tick;
    // Cache-global monotonic write counter. Its low 8 bits are stamped into
    // every written tag's [7:0] "version" field so `lookup_copy`'s seqlock
    // re-check detects a same-key re-insert that completed during the copy
    // (a bare key+flags tag would be identical across such a write → torn
    // read). Globally monotonic so concurrent inserts never share a version;
    // aliasing would need 256 cache-wide inserts inside one ≤256 B copy,
    // which is not physically reachable.
    alignas(64) std::atomic<uint64_t> write_seq;

    static constexpr uint32_t kMask = Capacity - 1;
    static constexpr uint64_t kKeyMask48 = 0xFFFFFFFFFFFFull;

    BOLT_FORCE_INLINE void init() noexcept {
        for (uint32_t i = 0; i < Capacity; ++i) {
            slots[i].key_tag.store(0, std::memory_order_relaxed);
            slots[i].access_count.store(0, std::memory_order_relaxed);
            slots[i].layout_gen = 0;
            slots[i].commit_lsn = 0;
            std::memset(slots[i].row_data, 0, RowBytes);
        }
        access_tick.store_relaxed(0);
        write_seq.store(0, std::memory_order_relaxed);
    }

    BOLT_FORCE_INLINE uint32_t bucket(uint64_t key_u64) const noexcept {
        return static_cast<uint32_t>(hotkey_detail::mix64(key_u64)) & kMask;
    }

    // Lookup. Returns pointer to row_data on hit, nullptr on miss.
    // Bumps access_count on hit (relaxed atomic).
    BOLT_FORCE_INLINE const uint8_t* lookup(K key, uint64_t layout_gen) noexcept {
        const uint64_t ku = static_cast<uint64_t>(key);
        uint32_t idx = bucket(ku);
        for (uint32_t probe = 0; probe < kHotKeyProbeDist; ++probe) {
            Slot& s = slots[idx];
            const uint64_t tag = s.key_tag.load(std::memory_order_acquire);
            const uint8_t  flags = static_cast<uint8_t>((tag >> 8) & 0xFFu);
            if ((flags & kHotKeyFlagOccupied) == 0u) return nullptr;
            if ((flags & kHotKeyFlagWriting)  != 0u) {
                // Slot is mid-write — retry on next probe (rare).
                idx = (idx + 1) & kMask;
                continue;
            }
            const uint64_t slot_key = tag >> 16;
            if (slot_key == (ku & 0xFFFFFFFFFFFFu) && s.layout_gen == layout_gen) {
                s.access_count.fetch_add(1, std::memory_order_relaxed);
                return s.row_data;
            }
            idx = (idx + 1) & kMask;
        }
        return nullptr;
    }

    // Seqlock-validated copy. Like `lookup`, but copies the matched row's
    // bytes into the caller's buffer under a seqlock re-check so a concurrent
    // `insert` / `invalidate_key` overwriting the slot mid-copy is detected
    // and reported as a miss (return false) rather than yielding a TORN row.
    //
    // Returning the raw `row_data` pointer (as `lookup` does) is unsafe for
    // readers that then dereference it — the slot has no pin count, so a
    // writer can overwrite it during the read. Callers that need the row
    // bytes (the MarbleDB get path) MUST use this instead of `lookup`.
    //
    // Protocol: load key_tag (tag1, acquire); require occupied + not-writing
    // + key match + layout_gen match + commit_lsn ≥ `min_commit_lsn`; memcpy
    // `len` bytes; acquire-fence; re-load key_tag (tag2). Success iff
    // tag2 == tag1 (the slot did not change identity/version during the copy).
    // `len` must be ≤ RowBytes.
    //
    // `min_commit_lsn` is the caller's freshness horizon (e.g. the store's
    // committed LSN at read start): a slot whose cached row predates it has
    // been superseded by a newer committed write not yet reflected here, so
    // it is reported as a miss and the caller reads the authoritative store.
    BOLT_FORCE_INLINE bool lookup_copy(K key, uint64_t layout_gen,
                                       uint64_t min_commit_lsn,
                                       uint8_t* dst, uint32_t len) noexcept {
        assert(dst != nullptr);
        assert(len <= RowBytes);
        const uint64_t ku = static_cast<uint64_t>(key);
        uint32_t idx = bucket(ku);
        for (uint32_t probe = 0; probe < kHotKeyProbeDist; ++probe) {
            Slot& s = slots[idx];
            const uint64_t tag1 = s.key_tag.load(std::memory_order_acquire);
            const uint8_t  flags1 = static_cast<uint8_t>((tag1 >> 8) & 0xFFu);
            if ((flags1 & kHotKeyFlagOccupied) == 0u) return false;
            if ((flags1 & kHotKeyFlagWriting) != 0u) {
                idx = (idx + 1) & kMask;
                continue;
            }
            if ((tag1 >> 16) != (ku & kKeyMask48) ||
                s.layout_gen != layout_gen) {
                idx = (idx + 1) & kMask;
                continue;
            }
            // Freshness gate: the row read here is consistent (tag1 stable, so
            // commit_lsn is from a completed insert, validated by tag2 below).
            if (s.commit_lsn < min_commit_lsn) return false;   // lagging → miss
            std::memcpy(dst, s.row_data, len);
            // Full barrier: an ACQUIRE fence only stops later ops from moving
            // BEFORE it — it would NOT stop the row_data copy (prior loads)
            // from sinking BELOW the tag2 reload on weak memory (ARM), which
            // let a torn copy slip past the seqlock re-check. A seq_cst fence
            // orders the copy strictly before tag2, so any concurrent
            // overwrite is detected.
            std::atomic_thread_fence(std::memory_order_seq_cst);
            // Full-tag compare: key + flags + version. A same-key re-insert
            // bumps the [7:0] version, so an overwrite that completed during
            // the copy (writing bit missed) still changes the tag → miss.
            const uint64_t tag2 = s.key_tag.load(std::memory_order_acquire);
            if (tag2 == tag1) {
                s.access_count.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            return false;   // slot changed under us → treat as a miss
        }
        return false;
    }

    // Insert a row into the cache, evicting the LRU victim if needed.
    // Caller is expected to gate this with their own access-count
    // tracker (promote only if threshold met). Single-writer contract
    // is NOT required — concurrent inserts on different slots are
    // fine; concurrent inserts on the same slot race via seqlock bits.
    BOLT_FORCE_INLINE void insert(K key, uint64_t layout_gen,
                                   uint64_t commit_lsn,
                                   const uint8_t* row, uint32_t row_len) noexcept {
        assert(row != nullptr);
        assert(row_len <= RowBytes);
        const uint64_t ku = static_cast<uint64_t>(key);
        uint32_t idx = bucket(ku);
        // Find either a matching slot (upsert) or a free slot.
        uint32_t victim_idx = idx;
        uint64_t victim_tick = UINT64_MAX;
        for (uint32_t probe = 0; probe < kHotKeyProbeDist; ++probe) {
            Slot& s = slots[idx];
            const uint64_t tag = s.key_tag.load(std::memory_order_acquire);
            const uint8_t  flags = static_cast<uint8_t>((tag >> 8) & 0xFFu);
            if ((flags & kHotKeyFlagOccupied) == 0u) {
                victim_idx = idx;
                victim_tick = 0;
                break;
            }
            const uint64_t slot_key = tag >> 16;
            if (slot_key == (ku & 0xFFFFFFFFFFFFu)) {
                victim_idx = idx;
                victim_tick = 0;
                break;
            }
            const uint64_t ac = s.access_count.load(std::memory_order_relaxed);
            if (ac < victim_tick) {
                victim_tick = ac;
                victim_idx  = idx;
            }
            idx = (idx + 1) & kMask;
        }
        // Seqlock write: set writing bit, memcpy, clear writing bit +
        // install final tag. The tag carries a per-write version in [7:0]
        // (cache-global monotonic) so `lookup_copy` detects a same-key
        // overwrite that completed inside its copy window.
        Slot& v = slots[victim_idx];
        const uint64_t ver =
            (write_seq.fetch_add(1, std::memory_order_relaxed) + 1u) & 0xFFu;
        const uint64_t key_bits = (ku & kKeyMask48) << 16;
        const uint64_t write_tag = key_bits |
            (static_cast<uint64_t>(kHotKeyFlagWriting) << 8) | ver;
        v.key_tag.store(write_tag, std::memory_order_release);
        v.layout_gen = layout_gen;
        v.commit_lsn = commit_lsn;
        std::memcpy(v.row_data, row, row_len);
        if (row_len < RowBytes) {
            std::memset(v.row_data + row_len, 0, RowBytes - row_len);
        }
        v.access_count.store(1, std::memory_order_relaxed);
        const uint64_t stable_tag = key_bits |
            (static_cast<uint64_t>(kHotKeyFlagOccupied) << 8) | ver;
        v.key_tag.store(stable_tag, std::memory_order_release);
    }

    // Invalidate all entries for a layout_gen mismatch — used when an
    // SST swap happens. O(Capacity); not on hot path.
    BOLT_FORCE_INLINE void invalidate_all() noexcept {
        for (uint32_t i = 0; i < Capacity; ++i) {
            slots[i].key_tag.store(0, std::memory_order_release);
        }
    }

    // Invalidate a single key — used on put / delete to kill stale
    // cached rows. O(probe-distance). Silently no-op if the key isn't
    // cached. Safe from any thread; uses release-store so a subsequent
    // acquire-load lookup misses on the invalidated slot.
    BOLT_FORCE_INLINE void invalidate_key(K key) noexcept {
        const uint64_t ku = static_cast<uint64_t>(key);
        uint32_t idx = bucket(ku);
        for (uint32_t probe = 0; probe < kHotKeyProbeDist; ++probe) {
            Slot& s = slots[idx];
            const uint64_t tag = s.key_tag.load(std::memory_order_acquire);
            const uint8_t  flags = static_cast<uint8_t>((tag >> 8) & 0xFFu);
            if ((flags & kHotKeyFlagOccupied) == 0u) return;
            const uint64_t slot_key = tag >> 16;
            if (slot_key == (ku & 0xFFFFFFFFFFFFu)) {
                s.key_tag.store(0, std::memory_order_release);
                return;
            }
            idx = (idx + 1) & kMask;
        }
    }

    // Occupancy — approximate count of populated slots. Used by the
    // write path to gate `invalidate_key` calls: when the cache is
    // empty, skip the probe entirely.
    BOLT_FORCE_INLINE uint32_t approx_occupancy() const noexcept {
        uint32_t count = 0;
        for (uint32_t i = 0; i < Capacity; ++i) {
            const uint64_t tag = slots[i].key_tag.load(std::memory_order_relaxed);
            const uint8_t flags = static_cast<uint8_t>((tag >> 8) & 0xFFu);
            if ((flags & kHotKeyFlagOccupied) != 0u) ++count;
        }
        return count;
    }
};

// Parallel access tracker — counts accesses per key in a small hash
// table so the caller can gate promotion on the threshold. Lock-free
// via atomic counters.
template <typename K, uint32_t Capacity>
struct alignas(64) AccessTracker {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity power of 2");
    struct Slot {
        std::atomic<uint64_t> key;
        std::atomic<uint32_t> count;
    };
    Slot slots[Capacity];
    static constexpr uint32_t kMask = Capacity - 1;

    BOLT_FORCE_INLINE void init() noexcept {
        for (uint32_t i = 0; i < Capacity; ++i) {
            slots[i].key.store(0, std::memory_order_relaxed);
            slots[i].count.store(0, std::memory_order_relaxed);
        }
    }

    // Bump counter for `key` and return the new count.
    BOLT_FORCE_INLINE uint32_t bump(K key) noexcept {
        const uint64_t ku = static_cast<uint64_t>(key);
        uint32_t idx = static_cast<uint32_t>(hotkey_detail::mix64(ku)) & kMask;
        for (uint32_t probe = 0; probe < 4; ++probe) {
            Slot& s = slots[idx];
            uint64_t existing = s.key.load(std::memory_order_acquire);
            if (existing == ku) {
                return s.count.fetch_add(1, std::memory_order_relaxed) + 1;
            }
            if (existing == 0u) {
                uint64_t zero = 0;
                if (s.key.compare_exchange_strong(
                        zero, ku, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    return s.count.fetch_add(1, std::memory_order_relaxed) + 1;
                }
                // Someone else took this slot — retry.
                existing = s.key.load(std::memory_order_acquire);
                if (existing == ku) {
                    return s.count.fetch_add(1, std::memory_order_relaxed) + 1;
                }
            }
            idx = (idx + 1) & kMask;
        }
        return 0;  // tracker full — caller treats as below threshold
    }

    // Read the current count for `key` WITHOUT incrementing it. Returns 0 if
    // the key is absent. Used by a write-side promoter to decide whether a
    // key is hot enough to cache, without polluting the access counts that
    // the read side maintains.
    BOLT_FORCE_INLINE uint32_t peek(K key) const noexcept {
        const uint64_t ku = static_cast<uint64_t>(key);
        uint32_t idx = static_cast<uint32_t>(hotkey_detail::mix64(ku)) & kMask;
        for (uint32_t probe = 0; probe < 4; ++probe) {
            const Slot& s = slots[idx];
            const uint64_t existing = s.key.load(std::memory_order_acquire);
            if (existing == ku) return s.count.load(std::memory_order_relaxed);
            if (existing == 0u) return 0u;
            idx = (idx + 1) & kMask;
        }
        return 0u;
    }

    BOLT_FORCE_INLINE void reset() noexcept {
        for (uint32_t i = 0; i < Capacity; ++i) {
            slots[i].key.store(0, std::memory_order_release);
            slots[i].count.store(0, std::memory_order_release);
        }
    }
};

}  // namespace bolt
