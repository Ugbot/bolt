// bolt_swiss_growable.h — growable SwissTable sibling with real deletion
// (Tiger Style, arena-allocated).
//
// `SwissTable` (bolt_swiss.h) is build-once / fixed-capacity by design — right
// for a join build side, wrong for a registry that lives as long as the process
// and churns (an event loop's fd→handler map, a connection table, a live LRU
// index). `SwissTableGrowable` is the sibling for THAT shape:
//
//   - same 16-slot group layout, same ctrl-byte scheme (0x80 = Empty, low-7-bit
//     tag from `swiss_tag`), same SIMD group scan, same {u64 key → u32 value}
//     slot — value-as-index, exactly like its siblings ("the 32-bit value field
//     is meant as an index, not a payload" — SwissTableSalted's words). A
//     caller with a fat or non-POD payload keeps it in its own stable pool and
//     stores the pool index here.
//   - REAL erase via tombstones (ctrl 0xFE = Deleted) with Abseil's
//     "was-never-full" early-out: when the erased slot's ±1-group window still
//     contains empties on both sides close enough that no probe chain can have
//     passed through it while full, the slot goes straight back to Empty and no
//     tombstone is ever created. Low-load churn (the fd-registry case) therefore
//     accumulates ZERO tombstones in the common case.
//   - a real grow/rehash path: at 7/8 occupancy (live + tombstones) the table
//     either doubles (live-dominated) or rehashes at the same size to purge
//     tombstones (tombstone-dominated), allocating fresh ctrl/slot arrays from
//     the SAME arena it was created with. Growth is still bounded: a caller-
//     supplied `max_capacity` (default kSwissMaxProbeCaps) is a hard ceiling
//     past which insert fails cleanly (returns false, table untouched).
//
// Design rationale (weighed vs Abseil raw_hash_set, folly F14, Robin Hood
// backward-shift, hopscotch): docs/research/growable-swiss-table.md.
// Short version: this IS the Abseil design (7/8 max load, tombstone + purge,
// never-full erase early-out, 2x pow2 growth) grafted onto bolt's existing
// SwissTable layout, because bolt's fixed table already borrows Abseil's probe
// scheme — the growable sibling should extend it, not fork the philosophy.
// Backward-shift/Robin-Hood erase was rejected (O(cluster) memmove per erase +
// breaks the group-scan "stop at first empty group" invariant as implemented);
// F14's chunk overflow counters were rejected (14-slot chunks + per-chunk
// metadata = a different layout family, no reuse of the proven scan);
// hopscotch was rejected (insert-time displacement cascades, unbounded worst
// case without extra machinery).
//
// Arena ownership: the table CAPTURES the Arena* passed to create() and grows
// from it. The arena must outlive the table. Superseded arrays are NOT freed
// (arenas don't free) — they become arena garbage until the arena is reset or
// destroyed. Total waste is bounded by a geometric series: < 1x the final
// array footprint for pure growth. Give a long-lived registry its own arena.
//
// Thread-safety: NONE (by design). The real consumers (epoll/kqueue/IOCP event
// loops) are one-loop-one-thread — each loop thread owns its registry
// exclusively, so a lock-free or locked design would be pure overhead.
//
// Rules: noexcept, no exceptions, no RTTI, bounded probes, ≥2 asserts per
// non-trivial function. Fallible ops return bool/-1, never throw.

#pragma once

#include "bolt/join/bolt_swiss.h"

#include <bit>

namespace bolt {

// Tombstone ctrl byte. High bit set so it can never equal a tag (tags are
// 7-bit), distinct from Empty (0x80) so the group scan's empty test skips it.
inline constexpr uint8_t kSwissCtrlDeleted = 0xFE;

struct SwissTableGrowable {
    uint8_t*   ctrl;          // capacity + kSwissGroupWidth bytes (mirror tail)
    SwissSlot* slots;         // capacity entries
    Arena*     arena;         // growth allocator — captured at create()
    uint32_t   capacity;      // power of two, >= kSwissMinCapacity
    uint32_t   mask;          // capacity - 1
    uint32_t   size;          // live entries
    uint32_t   tombstones;    // Deleted ctrl bytes currently in the array
    uint32_t   max_capacity;  // hard growth ceiling (power of two)
    uint32_t   _pad;

    // Max occupancy (live + tombstones) before a rehash/grow: 7/8 of capacity
    // (Abseil's load factor). Guarantees >= capacity/8 Empty bytes, so every
    // probe loop terminates at a group containing an Empty.
    uint32_t load_limit() const noexcept { return capacity - capacity / 8u; }

    // ------------------------------------------------------------------
    // Factory. `initial_hint` = expected live entries (sized 2x, like
    // SwissTable::create); `max_cap` = hard ceiling on grown capacity.
    // Returns false on arena OOM or a nonsensical ceiling.
    // ------------------------------------------------------------------
    static bool create(SwissTableGrowable* out, uint64_t initial_hint,
                       Arena* a, uint32_t max_cap = kSwissMaxProbeCaps) noexcept {
        assert(out != nullptr);
        assert(a != nullptr);
        uint32_t ceiling = swiss_round_up_pow2(max_cap);
        uint32_t cap = swiss_round_up_pow2(initial_hint * 2);
        if (cap > ceiling) cap = ceiling;

        uint8_t* c = nullptr;
        SwissSlot* s = nullptr;
        if (!alloc_arrays(&c, &s, cap, a)) return false;

        out->ctrl = c;
        out->slots = s;
        out->arena = a;
        out->capacity = cap;
        out->mask = cap - 1;
        out->size = 0;
        out->tombstones = 0;
        out->max_capacity = ceiling;
        out->_pad = 0;
        assert((out->capacity & out->mask) == 0);
        assert(out->capacity <= out->max_capacity);
        return true;
    }

    // ------------------------------------------------------------------
    // Find. Returns value, or -1 if not present. Identical scan to
    // SwissTable::find_mixed — Deleted bytes match neither the tag vector
    // nor the Empty vector, so tombstones are skipped for free.
    // ------------------------------------------------------------------
    int32_t find(uint64_t key) const noexcept {
        const int64_t p = find_slot(key);
        return p < 0 ? -1 : static_cast<int32_t>(slots[p].value);
    }

    // ------------------------------------------------------------------
    // Insert-or-update. Existing key: value overwritten in place, size
    // unchanged, never grows. New key: placed on the first Deleted slot of
    // its probe path if one exists (tombstone reuse — churn at steady size
    // does not grow), else the first Empty; grows/purges first if occupancy
    // would exceed 7/8. Returns false ONLY when growth is needed but the
    // ceiling is reached with no tombstones to purge, or the arena is OOM —
    // the table is untouched in that case.
    //
    // Iterator/index invalidation: a grow/purge reallocates the arrays, so
    // slot indices from next_live() are invalid after any insert of a NEW key.
    // ------------------------------------------------------------------
    bool insert(uint64_t key, uint32_t value) noexcept {
        assert(ctrl != nullptr);
        assert(slots != nullptr);
        uint32_t pos = 0;
        bool on_tombstone = false;
        int64_t found = probe_for_insert(key, &pos, &on_tombstone);
        if (found >= 0) {
            slots[found].value = value;  // update in place
            return true;
        }
        if (!on_tombstone && size + tombstones + 1 > load_limit()) {
            if (!grow_for_insert()) return false;
            found = probe_for_insert(key, &pos, &on_tombstone);
            assert(found < 0);           // key was absent; rehash keeps it so
            assert(!on_tombstone);       // fresh arrays have no tombstones
        }
        place(pos, key, value, on_tombstone);
        return true;
    }

    // ------------------------------------------------------------------
    // Erase. Returns false if `key` is not present. Applies Abseil's
    // was-never-full test: load the group AT the slot and the group ONE
    // GROUP BEFORE it; if both contain an Empty and the gap between the
    // nearest empties spanning the slot is under one group width, then no
    // probe sequence can ever have walked past this slot while it looked
    // full — so it reverts to Empty (no tombstone). Otherwise it becomes a
    // tombstone, reclaimed by the next purge/growth rehash.
    // ------------------------------------------------------------------
    bool erase(uint64_t key) noexcept {
        assert(ctrl != nullptr);
        assert(slots != nullptr);
        const int64_t p = find_slot(key);
        if (p < 0) return false;

        using namespace bolt::simd;
        const bmm_vec_i8 vempty = bmm_set1_i8(static_cast<int8_t>(kSwissCtrlEmpty));
        const uint32_t idx = static_cast<uint32_t>(p);
        const uint32_t before = (idx - kSwissGroupWidth) & mask;

        const bmm_vec_i8 gafter = bmm_loadu_i8(
            reinterpret_cast<const int8_t*>(ctrl + idx));
        const bmm_vec_i8 gbefore = bmm_loadu_i8(
            reinterpret_cast<const int8_t*>(ctrl + before));
        const uint32_t empty_after = static_cast<uint32_t>(bmm_movemask_i8(
            bmm_cmpeq_i8(gafter, vempty))) & 0xFFFFu;
        const uint32_t empty_before = static_cast<uint32_t>(bmm_movemask_i8(
            bmm_cmpeq_i8(gbefore, vempty))) & 0xFFFFu;

        // Trailing zeros of `after` = distance from the slot forward to the
        // nearest Empty; leading zeros of `before` (as a 16-bit mask) =
        // distance from the slot backward to the nearest Empty. If the two
        // empties sit within one group width of each other, every 16-byte
        // probe window covering this slot also covers an Empty → never full.
        bool never_full = false;
        if (empty_after != 0 && empty_before != 0) {
            const uint32_t fwd = static_cast<uint32_t>(bolt_ctz32(empty_after));
            const uint32_t back = static_cast<uint32_t>(
                std::countl_zero(static_cast<uint16_t>(empty_before)));
            never_full = (fwd + back) < kSwissGroupWidth;
        }

        set_ctrl(idx, never_full ? kSwissCtrlEmpty : kSwissCtrlDeleted);
        slots[idx].key = 0;
        slots[idx].value = 0;
        assert(size > 0);
        --size;
        if (!never_full) ++tombstones;
        return true;
    }

    // ------------------------------------------------------------------
    // Iteration: index of the first live slot >= `start`, or `capacity`
    // when none. Pattern:
    //   for (uint32_t i = t.next_live(0); i < t.capacity; i = t.next_live(i+1))
    //       use(t.slots[i].key, t.slots[i].value);
    // Indices are invalidated by any insert of a new key (may rehash).
    // ------------------------------------------------------------------
    uint32_t next_live(uint32_t start) const noexcept {
        assert(ctrl != nullptr);
        assert(start <= capacity);
        for (uint32_t i = start; i < capacity; ++i) {
            if ((ctrl[i] & 0x80u) == 0) return i;  // tag byte = live
        }
        return capacity;
    }

private:
    static bool alloc_arrays(uint8_t** c_out, SwissSlot** s_out,
                             uint32_t cap, Arena* a) noexcept {
        assert(cap >= kSwissMinCapacity);
        assert((cap & (cap - 1)) == 0);
        uint8_t* c = static_cast<uint8_t*>(a->allocate(cap + kSwissGroupWidth, 16));
        SwissSlot* s = a->allocate_array<SwissSlot>(cap);
        if (!c || !s) return false;
        memset(c, kSwissCtrlEmpty, cap + kSwissGroupWidth);
        memset(s, 0, cap * sizeof(SwissSlot));
        *c_out = c;
        *s_out = s;
        return true;
    }

    // Write a ctrl byte, maintaining the mirrored first group at the tail
    // (SIMD loads at base near capacity read ctrl[capacity..capacity+15]).
    void set_ctrl(uint32_t i, uint8_t b) noexcept {
        assert(i < capacity);
        ctrl[i] = b;
        if (i < kSwissGroupWidth) ctrl[capacity + i] = b;
    }

    // Slot index of `key`, or -1. Same group scan as SwissTable::find_mixed.
    int64_t find_slot(uint64_t key) const noexcept {
        assert(ctrl != nullptr);
        assert(capacity > 0);
        using namespace bolt::simd;
        const uint64_t h = swiss_mix(key);
        const uint8_t tag = swiss_tag(h);
        const bmm_vec_i8 vtag = bmm_set1_i8(static_cast<int8_t>(tag));
        const bmm_vec_i8 vempty = bmm_set1_i8(static_cast<int8_t>(kSwissCtrlEmpty));
        uint32_t base = static_cast<uint32_t>(h) & mask;
        for (uint32_t probes = 0; probes < capacity; probes += kSwissGroupWidth) {
            const bmm_vec_i8 g = bmm_loadu_i8(
                reinterpret_cast<const int8_t*>(ctrl + base));
            uint32_t tag_mask = static_cast<uint32_t>(bmm_movemask_i8(
                bmm_cmpeq_i8(g, vtag))) & 0xFFFFu;
            const uint32_t empty_mask = static_cast<uint32_t>(bmm_movemask_i8(
                bmm_cmpeq_i8(g, vempty))) & 0xFFFFu;
            while (tag_mask) {
                const uint32_t lane = static_cast<uint32_t>(bolt_ctz32(tag_mask));
                const uint32_t p = (base + lane) & mask;
                if (slots[p].key == key) return static_cast<int64_t>(p);
                tag_mask &= tag_mask - 1;
            }
            if (empty_mask != 0) return -1;
            base = (base + kSwissGroupWidth) & mask;
        }
        return -1;
    }

    // One probe walk that serves insert: returns the slot index of an
    // existing live `key` (>= 0), or -1 with *pos = the placement slot —
    // the FIRST Deleted slot seen on the probe path if any (tombstone
    // reuse), else the first Empty — and *on_tombstone saying which.
    int64_t probe_for_insert(uint64_t key, uint32_t* pos,
                             bool* on_tombstone) const noexcept {
        assert(pos != nullptr);
        assert(on_tombstone != nullptr);
        using namespace bolt::simd;
        const uint64_t h = swiss_mix(key);
        const uint8_t tag = swiss_tag(h);
        const bmm_vec_i8 vtag = bmm_set1_i8(static_cast<int8_t>(tag));
        const bmm_vec_i8 vempty = bmm_set1_i8(static_cast<int8_t>(kSwissCtrlEmpty));
        const bmm_vec_i8 vdel = bmm_set1_i8(static_cast<int8_t>(kSwissCtrlDeleted));
        uint32_t base = static_cast<uint32_t>(h) & mask;
        int64_t first_del = -1;
        for (uint32_t probes = 0; probes < capacity; probes += kSwissGroupWidth) {
            const bmm_vec_i8 g = bmm_loadu_i8(
                reinterpret_cast<const int8_t*>(ctrl + base));
            uint32_t tag_mask = static_cast<uint32_t>(bmm_movemask_i8(
                bmm_cmpeq_i8(g, vtag))) & 0xFFFFu;
            const uint32_t empty_mask = static_cast<uint32_t>(bmm_movemask_i8(
                bmm_cmpeq_i8(g, vempty))) & 0xFFFFu;
            const uint32_t del_mask = static_cast<uint32_t>(bmm_movemask_i8(
                bmm_cmpeq_i8(g, vdel))) & 0xFFFFu;
            while (tag_mask) {
                const uint32_t lane = static_cast<uint32_t>(bolt_ctz32(tag_mask));
                const uint32_t p = (base + lane) & mask;
                if (slots[p].key == key) return static_cast<int64_t>(p);
                tag_mask &= tag_mask - 1;
            }
            if (first_del < 0 && del_mask != 0) {
                const uint32_t lane = static_cast<uint32_t>(bolt_ctz32(del_mask));
                first_del = static_cast<int64_t>((base + lane) & mask);
            }
            if (empty_mask != 0) {
                if (first_del >= 0) {
                    *pos = static_cast<uint32_t>(first_del);
                    *on_tombstone = true;
                } else {
                    const uint32_t lane =
                        static_cast<uint32_t>(bolt_ctz32(empty_mask));
                    *pos = (base + lane) & mask;
                    *on_tombstone = false;
                }
                return -1;
            }
            base = (base + kSwissGroupWidth) & mask;
        }
        // Unreachable: load_limit() guarantees Empty bytes exist.
        assert(false && "SwissTableGrowable probe found no terminator");
        *pos = 0;
        *on_tombstone = false;
        return -1;
    }

    void place(uint32_t pos, uint64_t key, uint32_t value,
               bool on_tombstone) noexcept {
        assert(pos < capacity);
        assert(ctrl[pos] == (on_tombstone ? kSwissCtrlDeleted : kSwissCtrlEmpty));
        const uint8_t tag = swiss_tag(swiss_mix(key));
        set_ctrl(pos, tag);
        slots[pos].key = key;
        slots[pos].value = value;
        ++size;
        if (on_tombstone) {
            assert(tombstones > 0);
            --tombstones;
        }
        assert(size + tombstones <= load_limit());
    }

    // Pick the next capacity and rehash. Live-dominated (size > cap/2):
    // double, up to the ceiling. Tombstone-dominated (size <= cap/2, so
    // tombstones > 3/8 cap): rehash at the SAME size — purging reclaims
    // > cap * 3/8 occupancy, well past Abseil's "reclaim at least 1/8 or
    // resize" bar, and steady-state churn at a stable live count never
    // ratchets capacity. At the ceiling with tombstones, purge; at the
    // ceiling without, fail.
    bool grow_for_insert() noexcept {
        assert(size + tombstones + 1 > load_limit());
        uint64_t new_cap;
        if (size > capacity / 2u) {
            new_cap = static_cast<uint64_t>(capacity) * 2u;
            if (new_cap > max_capacity) {
                if (tombstones == 0) return false;  // hard ceiling, honest fail
                new_cap = capacity;                 // purge-only
            }
        } else {
            new_cap = capacity;  // tombstone purge at same size
        }
        if (size + 1 > new_cap - new_cap / 8u) return false;
        return rehash(static_cast<uint32_t>(new_cap));
    }

    bool rehash(uint32_t new_cap) noexcept {
        assert(new_cap >= kSwissMinCapacity);
        assert(new_cap <= max_capacity);
        assert(size <= new_cap - new_cap / 8u);
        uint8_t* nc = nullptr;
        SwissSlot* ns = nullptr;
        if (!alloc_arrays(&nc, &ns, new_cap, arena)) return false;

        const uint32_t nmask = new_cap - 1u;
        uint32_t moved = 0;
        for (uint32_t i = 0; i < capacity; ++i) {
            if ((ctrl[i] & 0x80u) != 0) continue;  // Empty or Deleted
            const uint64_t key = slots[i].key;
            const uint64_t h = swiss_mix(key);
            const uint8_t tag = swiss_tag(h);
            uint32_t idx = static_cast<uint32_t>(h) & nmask;
            while (nc[idx] != kSwissCtrlEmpty) idx = (idx + 1u) & nmask;
            nc[idx] = tag;
            if (idx < kSwissGroupWidth) nc[new_cap + idx] = tag;
            ns[idx].key = key;
            ns[idx].value = slots[i].value;
            ++moved;
        }
        assert(moved == size);
        (void)moved;

        ctrl = nc;          // old arrays become arena garbage (documented)
        slots = ns;
        capacity = new_cap;
        mask = nmask;
        tombstones = 0;
        return true;
    }
};

}  // namespace bolt
