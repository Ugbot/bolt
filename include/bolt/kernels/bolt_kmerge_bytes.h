// bolt_kmerge_bytes.h — k-way merge of sorted runs of variable-length,
// order-preserving byte keys (MSEG B10).
//
// MarbleDB stores every primary key as canonical key_encode() bytes: fixed
// cells big-endian with the sign bit flipped, then an optional var-length
// final cell, so a composite key of any kinds orders correctly under
// memcmp. A multi-column merge is therefore a merge over these bytes; the
// i64/u64 kmerge (bolt_kmerge.h) covers single integer keys only.
//
// Order: bytes ascending under memcmp, a proper prefix before any longer key
// that extends it (memcmp on the common length, then length). Equal keys
// come out in input-slot order (slot 0 first), and within one input in row
// order, so the merge is deterministic and stable: a caller that lists runs
// newest-first gets the newest version of each key first, which is what a
// newest-wins dedup driver needs. `out_new_key` marks the first row of each
// distinct key across call boundaries.
//
// Resumable: the merge state holds the heap and cursors, so a bounded output
// buffer never loses rows; call kmerge_bytes_next until it returns 0.
//
// Tiger Style: POD + free functions, noexcept, no allocation (the state is
// caller-owned), every loop bounded by the output capacity or a run length.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_kmerge.h"

namespace bolt {

// ---------------------------------------------------------------------------
// One run of keys. Variable-length keys use Arrow/VarBinary layout
// (`offsets[len + 1]` into `bytes`); fixed-length keys set `offsets` null and
// `stride` to the key width (e.g. 8 for a bare int64 key, 16 for (i64, i64)).
// ---------------------------------------------------------------------------
struct KeyBytesColumn {
    const uint8_t* bytes;
    const int32_t* offsets;   // nullptr => fixed stride
    int64_t        len;       // number of keys
    uint32_t       stride;    // fixed key width when offsets == nullptr
    uint32_t       _pad;
};
static_assert(sizeof(KeyBytesColumn) == 32, "KeyBytesColumn layout");

BOLT_FORCE_INLINE void key_bytes_at(const KeyBytesColumn& c, int64_t i,
                                    const uint8_t** p, uint32_t* n) noexcept {
    assert(i >= 0 && i < c.len);
    assert(p != nullptr && n != nullptr);
    if (c.offsets != nullptr) {
        const int32_t lo = c.offsets[i];
        *p = c.bytes + lo;
        *n = static_cast<uint32_t>(c.offsets[i + 1] - lo);
    } else {
        *p = c.bytes + static_cast<size_t>(i) * c.stride;
        *n = c.stride;
    }
}

/// View a column as keys: VarBinary (Binary / Utf8 / Symbol: payload in
/// `data`, int32 offsets in `dict_child`) or a Flat fixed-width column whose
/// rows are the key bytes (FixedSizeBinary, UUID, Decimal128 ...). False for
/// anything else (a Flat Utf8 StringView column has no contiguous bytes).
inline bool key_bytes_from_column(const BoltColumn& c, KeyBytesColumn* out) noexcept {
    assert(out != nullptr);
    assert(c.length >= 0);
    memset(out, 0, sizeof(*out));
    out->len = c.length;
    if (c.format == ColumnFormat::VarBinary) {
        if (c.length > 0 && (c.dict_child == nullptr || c.dict_child->data == nullptr))
            return false;
        out->bytes = static_cast<const uint8_t*>(c.data);
        out->offsets = c.length > 0 ? static_cast<const int32_t*>(c.dict_child->data)
                                    : nullptr;
        return true;
    }
    if (c.format == ColumnFormat::Flat && c.type != BoltType::Utf8 &&
        c.type_size_bytes > 0) {
        out->bytes = static_cast<const uint8_t*>(c.data);
        out->stride = c.type_size_bytes;
        return true;
    }
    return false;
}

/// memcmp order with "shorter prefix first": <0, 0, >0.
BOLT_FORCE_INLINE int key_bytes_cmp(const uint8_t* a, uint32_t an,
                                    const uint8_t* b, uint32_t bn) noexcept {
    assert(a != nullptr || an == 0);
    assert(b != nullptr || bn == 0);
    const uint32_t m = an < bn ? an : bn;
    const int c = (m == 0) ? 0 : memcmp(a, b, m);
    if (c != 0) return c;
    return (an < bn) ? -1 : (an > bn ? 1 : 0);
}

/// First 8 key bytes as a big-endian integer, zero padded: comparing two
/// prefixes orders keys exactly when they differ, and ties only when the
/// first 8 bytes (padded) agree, where the full compare decides.
BOLT_FORCE_INLINE uint64_t key_bytes_prefix(const uint8_t* p, uint32_t n) noexcept {
    assert(p != nullptr || n == 0);
    uint64_t v = 0;
    const uint32_t m = n < 8u ? n : 8u;
    for (uint32_t i = 0; i < m; ++i) v |= static_cast<uint64_t>(p[i]) << (56u - 8u * i);
    return v;
}

// ---------------------------------------------------------------------------
// Merge state.
// ---------------------------------------------------------------------------
struct KMergeBytesInput {
    KeyBytesColumn keys;      // sorted ascending (key_bytes_cmp)
    int64_t        pos;       // read cursor
    uint32_t       input_id;  // echoed to the output
    uint32_t       _pad;
};

namespace detail {
struct KMergeBytesEntry {
    uint64_t       prefix;
    const uint8_t* ptr;
    uint32_t       len;
    uint32_t       slot;
};
static_assert(sizeof(KMergeBytesEntry) == 24, "KMergeBytesEntry layout");
}  // namespace detail

struct KMergeBytes {
    KMergeBytesInput*         inputs;   // caller-owned, num_inputs entries
    uint32_t                  num_inputs;
    uint32_t                  heap_n;
    const uint8_t*            last_ptr; // last emitted key (into an input run)
    uint32_t                  last_len;
    uint32_t                  has_last;
    detail::KMergeBytesEntry  heap[kKMergeMaxInputs];
};

namespace detail {

BOLT_FORCE_INLINE bool kmb_less(const KMergeBytesEntry& a,
                                const KMergeBytesEntry& b) noexcept {
    if (a.prefix != b.prefix) return a.prefix < b.prefix;
    const int c = key_bytes_cmp(a.ptr, a.len, b.ptr, b.len);
    if (c != 0) return c < 0;
    return a.slot < b.slot;
}

inline void kmb_sift_down(KMergeBytesEntry* h, uint32_t n, uint32_t i) noexcept {
    assert(h != nullptr);
    assert(n <= kKMergeMaxInputs);
    for (uint32_t guard = 0; guard < kKMergeMaxInputs; ++guard) {   // depth <= log2(64)
        const uint32_t l = 2u * i + 1u;
        if (l >= n) return;
        const uint32_t r = l + 1u;
        const uint32_t c = (r < n && kmb_less(h[r], h[l])) ? r : l;
        if (!kmb_less(h[c], h[i])) return;
        const KMergeBytesEntry t = h[i]; h[i] = h[c]; h[c] = t;
        i = c;
    }
}

inline void kmb_sift_up(KMergeBytesEntry* h, uint32_t i) noexcept {
    assert(h != nullptr);
    assert(i < kKMergeMaxInputs);
    while (i > 0) {                      // bounded: i halves each step
        const uint32_t p = (i - 1u) / 2u;
        if (!kmb_less(h[i], h[p])) return;
        const KMergeBytesEntry t = h[i]; h[i] = h[p]; h[p] = t;
        i = p;
    }
}

inline void kmb_load(KMergeBytesEntry* e, const KMergeBytesInput& in,
                     uint32_t slot) noexcept {
    assert(e != nullptr);
    assert(in.pos < in.keys.len);
    key_bytes_at(in.keys, in.pos, &e->ptr, &e->len);
    e->prefix = key_bytes_prefix(e->ptr, e->len);
    e->slot = slot;
}

}  // namespace detail

/// Validate the runs and seed the heap. Returns false on a bad argument
/// (null, 0 or > kKMergeMaxInputs inputs, a cursor out of range). Runs must
/// already be sorted; kmerge_bytes_check_sorted verifies one in O(n).
inline bool kmerge_bytes_init(KMergeBytes* m, KMergeBytesInput* inputs,
                              uint32_t num_inputs) noexcept {
    assert(m != nullptr);
    assert(inputs != nullptr || num_inputs == 0);
    if (inputs == nullptr || num_inputs == 0 || num_inputs > kKMergeMaxInputs)
        return false;
    m->inputs = inputs;
    m->num_inputs = num_inputs;
    m->heap_n = 0;
    m->last_ptr = nullptr;
    m->last_len = 0;
    m->has_last = 0;
    for (uint32_t i = 0; i < num_inputs; ++i) {
        const KMergeBytesInput& in = inputs[i];
        if (in.keys.len < 0 || in.pos < 0 || in.pos > in.keys.len) return false;
        if (in.keys.len > 0 && in.keys.bytes == nullptr &&
            !(in.keys.offsets == nullptr && in.keys.stride == 0)) return false;
        if (in.pos == in.keys.len) continue;
        detail::kmb_load(&m->heap[m->heap_n], in, i);
        detail::kmb_sift_up(m->heap, m->heap_n);
        ++m->heap_n;
    }
    return true;
}

/// Emit up to `capacity` (input_id, row) pairs in merged order. `out_new_key`
/// (nullable) gets 1 for the first row of each distinct key, 0 for a repeat
/// of the previous key. Returns rows written; 0 when every run is drained.
inline int64_t kmerge_bytes_next(KMergeBytes* m, int32_t* BOLT_RESTRICT out_id,
                                 int64_t* BOLT_RESTRICT out_row,
                                 uint8_t* BOLT_RESTRICT out_new_key,
                                 int64_t capacity) noexcept {
    assert(m != nullptr);
    assert(capacity >= 0);
    assert(out_id != nullptr && out_row != nullptr);
    int64_t count = 0;
    while (m->heap_n > 0 && count < capacity) {     // bounded by capacity
        const detail::KMergeBytesEntry top = m->heap[0];
        KMergeBytesInput& in = m->inputs[top.slot];
        out_id[count] = static_cast<int32_t>(in.input_id);
        out_row[count] = in.pos;
        if (out_new_key != nullptr) {
            out_new_key[count] = static_cast<uint8_t>(
                m->has_last == 0 ||
                key_bytes_cmp(m->last_ptr, m->last_len, top.ptr, top.len) != 0);
        }
        m->last_ptr = top.ptr;
        m->last_len = top.len;
        m->has_last = 1;
        ++count;
        ++in.pos;
        if (in.pos < in.keys.len) {
            detail::kmb_load(&m->heap[0], in, top.slot);
        } else {
            --m->heap_n;
            m->heap[0] = m->heap[m->heap_n];
        }
        detail::kmb_sift_down(m->heap, m->heap_n, 0);
    }
    assert(count <= capacity);
    return count;
}

/// True when `c` is non-decreasing under key_bytes_cmp.
inline bool kmerge_bytes_check_sorted(const KeyBytesColumn& c) noexcept {
    assert(c.len >= 0);
    assert(c.bytes != nullptr || c.len == 0 || c.stride == 0);
    for (int64_t i = 1; i < c.len; ++i) {
        const uint8_t *a = nullptr, *b = nullptr;
        uint32_t an = 0, bn = 0;
        key_bytes_at(c, i - 1, &a, &an);
        key_bytes_at(c, i, &b, &bn);
        if (key_bytes_cmp(a, an, b, bn) > 0) return false;
    }
    return true;
}

}  // namespace bolt
