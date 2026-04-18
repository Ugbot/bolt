// bolt_kmerge.h — K-way sorted-run merge over int64/uint64 keys.
//
// Built for LSM compaction (MarbleDB): merge up to 64 sorted key runs into
// one ascending output stream, emitting (input_id, key) pairs so the caller
// can gather matching row bodies from the correct input.
//
// Tiger Style:
//   - POD + free functions, no exceptions, noexcept everywhere.
//   - No heap; the min-heap lives on the stack (fixed cap 64 entries).
//   - Hard upper bounds; ≥2 asserts per function; outer function ≤70 lines.
//
// Shape recap:
//   KMergeInput carries a sorted run (keys[], pos, len) plus a caller-
//   assigned input_id. The driver advances pos on the input that supplied
//   the most recent emitted top, then either pushes its next key onto the
//   heap or sift_down's to fill the vacated slot.
//
// Fan-in rationale: the compaction research picks 64 as the practical max
// — wider fan-ins need double-buffered IO and a loser-tree. We stop at 64
// and document the cap; callers above 64 run a two-pass merge.
//
// Stability: output is ordered by key ascending. For equal keys across
// inputs the tie-break is the current heap sift order — NOT a stable
// input_id order. Duplicates *within* one input preserve arrival order
// (guaranteed: a single input advances its pos monotonically).
//
// 2-input fast path: when num_inputs == 2 the heap overhead dominates, so
// we fall through to a 2-way two-pointer loop mirroring the shape of
// bolt::mergejoin_inner_i64.

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_branchless.h"
#include "bolt/bolt_port.h"

namespace bolt {

// ---------------------------------------------------------------------------
// Public constants + POD types
// ---------------------------------------------------------------------------

/// Hard fan-in cap. Wider merges run in two passes.
inline constexpr uint32_t kKMergeMaxInputs = 64;

/// One sorted run. `keys` must point to `len` ascending int64/uint64 values;
/// `pos` is the read cursor (updated by the merge driver). `input_id` is
/// echoed to the output so the caller can gather row bodies from the
/// originating input.
struct KMergeInput {
    const int64_t* keys;     // sorted ascending (reinterpret for u64 variant)
    int64_t        pos;      // current read position
    int64_t        len;      // total keys in this input
    uint32_t       input_id; // caller-assigned, passed through to output
    uint32_t       _pad;     // keep size = 32 bytes
};

static_assert(sizeof(KMergeInput) == 32,
              "KMergeInput must stay 32 bytes for cache friendliness");

// ---------------------------------------------------------------------------
// Internal heap representation
// ---------------------------------------------------------------------------
namespace detail {

/// Heap entry: min-ordered by `key`. `slot` is the 0..num_inputs-1 index
/// into the caller's `inputs[]` array — lets the driver reach the source
/// run in O(1) without scanning for `input_id`. Caller's `input_id` is
/// recovered via `inputs[slot].input_id` on emit (one indirection,
/// cache-hot for n ≤ 64).
///
/// Keeping the struct a clean 16 bytes (key + slot + pad) is a cache-line
/// power-of-two. Explicit pad makes the layout assumption load-bearing.
struct KMergeHeapEntry {
    int64_t  key;
    uint32_t slot;
    uint32_t _pad;
};

static_assert(sizeof(KMergeHeapEntry) == 16,
              "KMergeHeapEntry layout assumption");
static_assert(alignof(KMergeHeapEntry) == 8,
              "KMergeHeapEntry alignment assumption");

BOLT_FORCE_INLINE bool kmerge_i64_less(const KMergeHeapEntry& a,
                                       const KMergeHeapEntry& b) noexcept {
    // Signed compare for the i64 variant.
    return a.key < b.key;
}

BOLT_FORCE_INLINE bool kmerge_u64_less(const KMergeHeapEntry& a,
                                       const KMergeHeapEntry& b) noexcept {
    // u64 variant reinterprets the 64-bit payload as unsigned.
    const uint64_t ak = static_cast<uint64_t>(a.key);
    const uint64_t bk = static_cast<uint64_t>(b.key);
    return ak < bk;
}

/// Percolate slot `i` downward until the heap property is restored.
/// Branchless child pick via `bolt::branchless::bselect` (→ cmov on x86,
/// csel on ARM64). The loop-termination checks (`l >= n`, heap property
/// satisfied) stay as explicit branches — they're loop-exit conditions,
/// not per-iteration predication, so CMOV buys nothing there. The swap
/// itself is unconditional once we've decided to descend.
template <bool (*Less)(const KMergeHeapEntry&, const KMergeHeapEntry&)>
BOLT_FORCE_INLINE void kmerge_sift_down(KMergeHeapEntry* BOLT_RESTRICT heap,
                                        uint32_t n,
                                        uint32_t i) noexcept {
    assert(heap != nullptr);
    assert(n <= kKMergeMaxInputs);
    assert(i < n || n == 0);
    while (true) {
        const uint32_t l = 2u * i + 1u;
        const uint32_t r = l + 1u;
        if (l >= n) return;
        // Branchless pick of the smaller child. When `r == n` (no right
        // child) we force `child = l` by feeding `l` into both sides of
        // the right-half select.
        const bool r_in = (r < n);
        const uint32_t r_or_l =
            bolt::branchless::bselect<uint32_t>(r_in, r, l);
        const bool r_smaller = Less(heap[r_or_l], heap[l]);
        const uint32_t child =
            bolt::branchless::bselect<uint32_t>(r_smaller, r_or_l, l);
        if (!Less(heap[child], heap[i])) return;
        const KMergeHeapEntry tmp = heap[i];
        heap[i] = heap[child];
        heap[child] = tmp;
        i = child;
    }
}

/// Percolate slot `i` upward until the heap property is restored.
/// No branchless rewrite here — sift_up's per-iteration test IS the
/// loop-exit condition, not a data-dependent branch we could CMOV away.
template <bool (*Less)(const KMergeHeapEntry&, const KMergeHeapEntry&)>
BOLT_FORCE_INLINE void kmerge_sift_up(KMergeHeapEntry* BOLT_RESTRICT heap,
                                      uint32_t i) noexcept {
    assert(heap != nullptr);
    assert(i < kKMergeMaxInputs);
    while (i > 0) {
        const uint32_t p = (i - 1u) / 2u;
        if (!Less(heap[i], heap[p])) return;
        const KMergeHeapEntry tmp = heap[i];
        heap[i] = heap[p];
        heap[p] = tmp;
        i = p;
    }
}

// ---------------------------------------------------------------------------
// 2-input fast path — pure two-pointer merge (no heap).
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE int64_t kmerge_two_way_i64(KMergeInput* BOLT_RESTRICT inputs,
                                             int32_t* BOLT_RESTRICT out_id,
                                             int64_t* BOLT_RESTRICT out_key,
                                             int64_t capacity) noexcept {
    assert(inputs != nullptr);
    assert(capacity >= 0);

    KMergeInput& a = inputs[0];
    KMergeInput& b = inputs[1];
    int64_t count = 0;
    while (a.pos < a.len && b.pos < b.len) {
        if (count >= capacity) return count;
        const int64_t ak = a.keys[a.pos];
        const int64_t bk = b.keys[b.pos];
        if (ak <= bk) {
            out_id[count]  = static_cast<int32_t>(a.input_id);
            out_key[count] = ak;
            ++a.pos;
        } else {
            out_id[count]  = static_cast<int32_t>(b.input_id);
            out_key[count] = bk;
            ++b.pos;
        }
        ++count;
    }
    while (a.pos < a.len) {
        if (count >= capacity) return count;
        out_id[count]  = static_cast<int32_t>(a.input_id);
        out_key[count] = a.keys[a.pos++];
        ++count;
    }
    while (b.pos < b.len) {
        if (count >= capacity) return count;
        out_id[count]  = static_cast<int32_t>(b.input_id);
        out_key[count] = b.keys[b.pos++];
        ++count;
    }
    return count;
}

BOLT_FORCE_INLINE int64_t kmerge_two_way_u64(KMergeInput* BOLT_RESTRICT inputs,
                                             int32_t* BOLT_RESTRICT out_id,
                                             int64_t* BOLT_RESTRICT out_key,
                                             int64_t capacity) noexcept {
    assert(inputs != nullptr);
    assert(capacity >= 0);

    KMergeInput& a = inputs[0];
    KMergeInput& b = inputs[1];
    int64_t count = 0;
    while (a.pos < a.len && b.pos < b.len) {
        if (count >= capacity) return count;
        const uint64_t ak = static_cast<uint64_t>(a.keys[a.pos]);
        const uint64_t bk = static_cast<uint64_t>(b.keys[b.pos]);
        if (ak <= bk) {
            out_id[count]  = static_cast<int32_t>(a.input_id);
            out_key[count] = static_cast<int64_t>(ak);
            ++a.pos;
        } else {
            out_id[count]  = static_cast<int32_t>(b.input_id);
            out_key[count] = static_cast<int64_t>(bk);
            ++b.pos;
        }
        ++count;
    }
    while (a.pos < a.len) {
        if (count >= capacity) return count;
        out_id[count]  = static_cast<int32_t>(a.input_id);
        out_key[count] = a.keys[a.pos++];
        ++count;
    }
    while (b.pos < b.len) {
        if (count >= capacity) return count;
        out_id[count]  = static_cast<int32_t>(b.input_id);
        out_key[count] = b.keys[b.pos++];
        ++count;
    }
    return count;
}

// ---------------------------------------------------------------------------
// General heap-driven merge. `Less` selects signed vs unsigned ordering.
//
// Heap entries carry the 0..num_inputs-1 `slot` index (NOT the caller's
// `input_id`) so the driver reaches the source run in O(1). The caller's
// `input_id` is recovered via `inputs[slot].input_id` exactly once per
// emitted row.
//
// Exhausted-run handling: shrink. When a run hits EOF we move the last
// heap element into slot 0 and sift down. Sentinel-INT64_MAX was
// considered and rejected — it would require a signed-vs-unsigned
// sentinel per variant and bloat the comparator.
// ---------------------------------------------------------------------------
template <bool (*Less)(const KMergeHeapEntry&, const KMergeHeapEntry&)>
BOLT_FORCE_INLINE int64_t kmerge_heap_run(KMergeInput* BOLT_RESTRICT inputs,
                                          uint32_t num_inputs,
                                          int32_t* BOLT_RESTRICT out_id,
                                          int64_t* BOLT_RESTRICT out_key,
                                          int64_t capacity) noexcept {
    assert(inputs != nullptr);
    assert(num_inputs > 0 && num_inputs <= kKMergeMaxInputs);

    KMergeHeapEntry heap[kKMergeMaxInputs];
    uint32_t n = 0;

    // Seed the heap with the first key from every non-empty input.
    for (uint32_t i = 0; i < num_inputs; ++i) {
        if (inputs[i].pos < inputs[i].len) {
            heap[n].key  = inputs[i].keys[inputs[i].pos];
            heap[n].slot = i;
            heap[n]._pad = 0;
            kmerge_sift_up<Less>(heap, n);
            ++n;
        }
    }

    int64_t count = 0;
    while (n > 0) {
        if (count >= capacity) return count;
        const KMergeHeapEntry top = heap[0];
        assert(top.slot < num_inputs);
        KMergeInput& in = inputs[top.slot];
        out_id[count]  = static_cast<int32_t>(in.input_id);
        out_key[count] = top.key;
        ++count;
        ++in.pos;

        if (in.pos < in.len) {
            heap[0].key  = in.keys[in.pos];
            // slot + pad unchanged
            kmerge_sift_down<Less>(heap, n, 0);
        } else {
            --n;
            if (n > 0) {
                heap[0] = heap[n];
                kmerge_sift_down<Less>(heap, n, 0);
            }
        }
    }
    return count;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Public API — i64
// ---------------------------------------------------------------------------

/// Merge up to `num_inputs` sorted int64 runs into one ascending output.
/// Writes (input_id, key) pairs in order. Output arrays must have capacity
/// ≥ sum of input lengths. Returns the number of pairs written, or -1 on
/// invalid arguments (null pointer, num_inputs == 0 or > kKMergeMaxInputs,
/// negative capacity).
inline int64_t kmerge_i64(KMergeInput* BOLT_RESTRICT inputs,
                          uint32_t num_inputs,
                          int32_t* BOLT_RESTRICT out_input_id,
                          int64_t* BOLT_RESTRICT out_key,
                          int64_t capacity) noexcept {
    assert(capacity >= 0);
    assert(num_inputs <= kKMergeMaxInputs);
    if (inputs == nullptr && num_inputs > 0)           return -1;
    if (num_inputs == 0)                               return 0;
    if (num_inputs > kKMergeMaxInputs)                 return -1;
    if (capacity < 0)                                  return -1;
    if (out_input_id == nullptr || out_key == nullptr) return -1;

    // Per-input bounds (pos in [0, len], len ≥ 0, keys != nullptr if len > 0).
    for (uint32_t i = 0; i < num_inputs; ++i) {
        const KMergeInput& in = inputs[i];
        if (in.len < 0)                       return -1;
        if (in.pos < 0 || in.pos > in.len)    return -1;
        if (in.len > 0 && in.keys == nullptr) return -1;
    }

    if (num_inputs == 1) {
        KMergeInput& s = inputs[0];
        int64_t n = 0;
        while (s.pos < s.len && n < capacity) {
            out_input_id[n] = static_cast<int32_t>(s.input_id);
            out_key[n]      = s.keys[s.pos++];
            ++n;
        }
        return n;
    }
    if (num_inputs == 2) {
        return detail::kmerge_two_way_i64(inputs, out_input_id, out_key, capacity);
    }
    return detail::kmerge_heap_run<detail::kmerge_i64_less>(
        inputs, num_inputs, out_input_id, out_key, capacity);
}

// ---------------------------------------------------------------------------
// Public API — u64 (same shape, unsigned ordering)
// ---------------------------------------------------------------------------

/// Merge up to `num_inputs` sorted uint64 runs. `keys` is declared int64 on
/// the input struct for ABI uniformity; this entrypoint reinterprets it as
/// unsigned for the comparison. Same return contract as kmerge_i64.
inline int64_t kmerge_u64(KMergeInput* BOLT_RESTRICT inputs,
                          uint32_t num_inputs,
                          int32_t* BOLT_RESTRICT out_input_id,
                          int64_t* BOLT_RESTRICT out_key,
                          int64_t capacity) noexcept {
    assert(capacity >= 0);
    assert(num_inputs <= kKMergeMaxInputs);
    if (inputs == nullptr && num_inputs > 0)           return -1;
    if (num_inputs == 0)                               return 0;
    if (num_inputs > kKMergeMaxInputs)                 return -1;
    if (capacity < 0)                                  return -1;
    if (out_input_id == nullptr || out_key == nullptr) return -1;

    for (uint32_t i = 0; i < num_inputs; ++i) {
        const KMergeInput& in = inputs[i];
        if (in.len < 0)                       return -1;
        if (in.pos < 0 || in.pos > in.len)    return -1;
        if (in.len > 0 && in.keys == nullptr) return -1;
    }

    if (num_inputs == 1) {
        KMergeInput& s = inputs[0];
        int64_t n = 0;
        while (s.pos < s.len && n < capacity) {
            out_input_id[n] = static_cast<int32_t>(s.input_id);
            out_key[n]      = s.keys[s.pos++];
            ++n;
        }
        return n;
    }
    if (num_inputs == 2) {
        return detail::kmerge_two_way_u64(inputs, out_input_id, out_key, capacity);
    }
    return detail::kmerge_heap_run<detail::kmerge_u64_less>(
        inputs, num_inputs, out_input_id, out_key, capacity);
}

}  // namespace bolt
