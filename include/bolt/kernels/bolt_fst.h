// bolt/kernels/bolt_fst.h - Finite-state-transducer term dictionary.
//
// Layer 1.5 of plan `this-was-a-freach-hashed-crab.md`. Lucene-style minimal
// acyclic byte-keyed FST: maps sorted byte keys to int64 outputs in a packed
// node-arc graph with shared prefixes (and best-effort shared suffixes via a
// fixed-cap dedup hash table).
//
// On-disk node layout (packed, written/read via memcpy - no alignment trap):
//   node      := { uint16_t arc_count; Arc[arc_count] }
//   Arc (14B) := { uint8_t label;
//                  uint8_t flags;          // bit0=IsFinal, bit1=IsLastArc
//                  int32_t target_offset;  // byte offset of child node, or -1
//                  int64_t output;         // attached payload when IsFinal
//                }
//
// Build algorithm (incremental suffix-minimisation, Lucene-shape):
//   - Caller adds keys in strict ascending byte order.
//   - We hold a "frontier" of in-progress nodes, one per depth of the last
//     key. On add(K), we find the common prefix length C with last_key.
//     For depth d in [last_key_len ... C+1], freeze that frontier node:
//       - serialise its arcs into the buffer
//       - dedup against a hash table keyed by (arc_count, arc_bytes); on a
//         hit we re-use the existing node offset, on a miss we keep the new
//         offset. Hash overflow degrades to "no dedup, fresh node" - this
//         is correct, just less compressed.
//   - Then extend the frontier from depth C..K_len-1 with arcs labelled by
//     K's bytes. Mark the terminal arc IsFinal and stash `output` on it.
//
// Lookup is a tight per-byte loop: linear scan of arcs at the current node
// (most nodes have <8 arcs in byte alphabets) until we find the matching
// label, then jump to the child node. On the last byte, the IsFinal flag
// gates "key exists" and the arc's `output` is the payload.
//
// Tiger Style:
//   - noexcept everywhere, no exceptions, no RTTI, no virtual.
//   - All allocation through `bolt::Arena*`. No raw malloc, no STL containers.
//   - POD with `static_assert(sizeof(...) == N, ...)`.
//   - >=2 assertions on every public function, <=70 lines per function.
//   - BOLT_FORCE_INLINE on hot lookup, BOLT_RESTRICT on kernel pointer params.
//
// What we lifted from Lucene:
//   - Incremental suffix minimisation while keys arrive sorted.
//   - Byte-keyed arcs with attached output values.
//   - IsFinal / IsLastArc flags packed into a single byte.
//
// What we omitted (deferred until measured benefit):
//   - Output composition for shared-suffix outputs (Lucene's "outputs"
//     monoid). When two keys share a suffix but have different outputs we
//     keep separate suffix nodes - prefix sharing still kicks in.
//   - Packed bit-width arc-target encoding. We pay 14 bytes per arc; Lucene
//     packs target offsets to ceil(log2(buf_size)) bits.
//
// Research note: docs/research/fst-term-dictionary.md.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace kernels {

// --- Public POD types ------------------------------------------------------

// Read-only FST view. 16 bytes; the buffer it points at lives in the arena
// the builder was initialised with (or is otherwise caller-owned).
struct FstView {
    const uint8_t* nodes;
    int32_t        bytes;
    int32_t        root_offset;
};
static_assert(sizeof(FstView) == 16, "FstView is 16 bytes");

// Arc flag bits. Packed into Arc::flags (1 byte).
namespace fst_flag {
inline constexpr uint8_t kIsFinal   = 0x01u;
inline constexpr uint8_t kIsLastArc = 0x02u;
}  // namespace fst_flag

// Per-arc on-disk size; not a struct because we memcpy field-by-field to
// avoid alignment penalties on platforms that police 8-byte int64 loads.
inline constexpr int32_t kFstArcBytes = 14;

// --- Builder ---------------------------------------------------------------

// kFstMaxKeyLen also bounds the frontier depth. 256-byte keys cover term
// dictionaries (longest realistic identifier) and JSONB path components.
inline constexpr int32_t kFstMaxKeyLen     = 256;

// Cap on arcs at any single frontier node. 256 covers the full byte alphabet
// (one arc per possible label); we never need more.
inline constexpr int32_t kFstMaxArcsPerNode = 256;

// Dedup hash table size (power of two). 4096 entries give a good
// prefix/suffix sharing rate on the workloads we care about (term dicts,
// path dicts) while keeping the table cheap. Overflow -> no dedup.
inline constexpr int32_t kFstDedupSlots = 4096;

// In-progress arc held in the frontier. POD; not serialised in this shape.
struct FstPendingArc {
    int64_t output;
    int32_t target;   // child node offset in `buf`, -1 if no child
    uint8_t label;
    uint8_t is_final; // 0/1
    uint8_t _pad[6];
};
static_assert(sizeof(FstPendingArc) == 24, "FstPendingArc is 24 bytes");

// One frontier slot per depth. arcs[] is fixed-cap; we never overflow
// because byte alphabet has 256 distinct labels.
struct FstFrontierNode {
    FstPendingArc arcs[kFstMaxArcsPerNode];
    int32_t       arc_count;
    int32_t       _pad;
};

// Dedup hash entry. key_hash is the hash of (arc_count + serialised arcs);
// node_offset is where in `buf` that frozen node lives.
struct FstDedupEntry {
    uint64_t key_hash;
    int32_t  node_offset;
    int32_t  arc_count;
};
static_assert(sizeof(FstDedupEntry) == 16, "FstDedupEntry is 16 bytes");

struct FstBuilder {
    Arena*    arena;
    uint8_t*  buf;
    int32_t   capacity;
    int32_t   size;

    // Frontier state - one node per active depth.
    FstFrontierNode* frontier;     // [kFstMaxKeyLen + 1]
    int32_t          frontier_len; // active depth (== last_key_len after first add)

    // Last key we added (for ascending check + common-prefix find).
    uint8_t* last_key;            // [kFstMaxKeyLen]
    int32_t  last_key_len;
    int64_t  last_output;

    // Dedup hash table.
    FstDedupEntry* dedup;          // [kFstDedupSlots]
    int32_t        dedup_count;

    int32_t  num_keys;
    int32_t  finished;             // 0/1
};

// --- Internal helpers (inline) --------------------------------------------

namespace fst_detail {

BOLT_FORCE_INLINE uint64_t mix64(uint64_t x) noexcept {
    // Splitmix64 finaliser - good avalanche, two muls + two xorshifts.
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

inline bool grow_buf(FstBuilder* b, int32_t need) noexcept {
    assert(b != nullptr);
    assert(need >= 0);
    if (b->size + need <= b->capacity) return true;
    int32_t new_cap = b->capacity > 0 ? b->capacity : 64;
    while (new_cap < b->size + need) {
        if (new_cap > (1 << 28)) return false;  // 256 MiB ceiling - sanity
        new_cap *= 2;
    }
    uint8_t* nb = b->arena->allocate_array<uint8_t>(static_cast<size_t>(new_cap));
    if (!nb) return false;
    if (b->size > 0) std::memcpy(nb, b->buf, static_cast<size_t>(b->size));
    b->buf = nb;
    b->capacity = new_cap;
    return true;
}

inline void write_arc(uint8_t* dst, uint8_t label, uint8_t flags,
                      int32_t target, int64_t output) noexcept {
    assert(dst != nullptr);
    dst[0] = label;
    dst[1] = flags;
    std::memcpy(dst + 2,  &target, 4);
    std::memcpy(dst + 6,  &output, 8);
}

inline void read_arc(const uint8_t* src, uint8_t* label, uint8_t* flags,
                     int32_t* target, int64_t* output) noexcept {
    assert(src != nullptr);
    *label = src[0];
    *flags = src[1];
    std::memcpy(target, src + 2, 4);
    std::memcpy(output, src + 6, 8);
}

// Hash a serialised node (arc_count + arc_bytes) for the dedup table.
inline uint64_t hash_node_bytes(int32_t arc_count, const uint8_t* bytes) noexcept {
    assert(arc_count >= 0);
    assert(bytes != nullptr || arc_count == 0);
    uint64_t h = 0xcbf29ce484222325ULL;  // FNV-ish seed; we mix64 below
    h ^= static_cast<uint64_t>(arc_count);
    h = mix64(h);
    const int32_t total = arc_count * kFstArcBytes;
    for (int32_t i = 0; i < total; i += 8) {
        uint64_t chunk = 0;
        const int32_t take = (total - i) >= 8 ? 8 : (total - i);
        std::memcpy(&chunk, bytes + i, static_cast<size_t>(take));
        h ^= chunk; h = mix64(h);
    }
    return h;
}

// Look up (or insert) a node we just serialised at offset `tentative_offset`.
// Returns the canonical offset to use (existing if dedup hit, tentative
// otherwise). On dedup hit, the caller rewinds `b->size` to drop the
// tentative bytes. Linear-probe with a power-of-two table; on full table we
// degrade to "no dedup" which is just `return tentative_offset`.
inline int32_t dedup_lookup_or_insert(FstBuilder* b, int32_t tentative_offset,
                                       int32_t arc_count) noexcept {
    assert(b != nullptr);
    assert(tentative_offset >= 0);
    assert(arc_count >= 0);
    const uint8_t* node_bytes = b->buf + tentative_offset + 2;  // skip arc_count
    uint64_t h = hash_node_bytes(arc_count, node_bytes);
    if (b->dedup_count >= (kFstDedupSlots * 3) / 4) {
        return tentative_offset;  // table near-full: skip dedup
    }
    uint32_t mask = static_cast<uint32_t>(kFstDedupSlots) - 1u;
    uint32_t slot = static_cast<uint32_t>(h) & mask;
    for (uint32_t probe = 0; probe < kFstDedupSlots; ++probe) {
        FstDedupEntry* e = &b->dedup[(slot + probe) & mask];
        if (e->node_offset == 0 && e->arc_count == 0 && e->key_hash == 0) {
            // Empty slot - insert. (offset==0 is the root and is reserved
            // for the first frozen node; dedup never tries to alias to it.)
            e->key_hash = h;
            e->node_offset = tentative_offset;
            e->arc_count = arc_count;
            ++b->dedup_count;
            return tentative_offset;
        }
        if (e->key_hash == h && e->arc_count == arc_count) {
            const int32_t total = arc_count * kFstArcBytes;
            if (std::memcmp(b->buf + e->node_offset + 2, node_bytes,
                            static_cast<size_t>(total)) == 0) {
                return e->node_offset;  // dedup hit
            }
        }
    }
    return tentative_offset;  // table truly full
}

// Serialise the frontier node at `depth` into the buffer; mark its last arc
// IsLastArc; dedup; return the canonical offset (or -1 on OOM).
inline int32_t freeze_frontier_node(FstBuilder* b, int32_t depth) noexcept {
    assert(b != nullptr);
    assert(depth >= 0 && depth <= kFstMaxKeyLen);
    FstFrontierNode* fn = &b->frontier[depth];
    const int32_t arc_count = fn->arc_count;
    const int32_t need = 2 + arc_count * kFstArcBytes;
    if (!grow_buf(b, need)) return -1;
    const int32_t tentative = b->size;
    uint16_t ac16 = static_cast<uint16_t>(arc_count);
    std::memcpy(b->buf + tentative, &ac16, 2);
    for (int32_t i = 0; i < arc_count; ++i) {
        FstPendingArc* a = &fn->arcs[i];
        uint8_t flags = 0;
        if (a->is_final) flags |= fst_flag::kIsFinal;
        if (i + 1 == arc_count) flags |= fst_flag::kIsLastArc;
        write_arc(b->buf + tentative + 2 + i * kFstArcBytes,
                  a->label, flags, a->target, a->output);
    }
    b->size += need;
    int32_t canonical = dedup_lookup_or_insert(b, tentative, arc_count);
    if (canonical != tentative) {
        b->size = tentative;  // rewind: re-use existing node
    }
    fn->arc_count = 0;
    return canonical;
}

}  // namespace fst_detail

// --- Public API ------------------------------------------------------------

inline bool fst_builder_init(FstBuilder* b, Arena* arena,
                              int32_t initial_cap) noexcept {
    assert(b != nullptr);
    assert(arena != nullptr);
    if (!b || !arena) return false;
    if (initial_cap < 64) initial_cap = 64;
    std::memset(b, 0, sizeof(*b));
    b->arena = arena;
    b->buf = arena->allocate_array<uint8_t>(static_cast<size_t>(initial_cap));
    if (!b->buf) return false;
    b->capacity = initial_cap;
    // Reserve byte 0 as a sentinel "empty node" so dedup-empty-slot
    // detection (offset==0 means empty) doesn't collide with a real node.
    // We write a 0-arc node at offset 0; the real root will live elsewhere.
    if (b->capacity < 2) return false;
    std::memset(b->buf, 0, 2);
    b->size = 2;
    b->frontier = arena->allocate_array<FstFrontierNode>(
        static_cast<size_t>(kFstMaxKeyLen + 1));
    if (!b->frontier) return false;
    std::memset(b->frontier, 0,
                sizeof(FstFrontierNode) * static_cast<size_t>(kFstMaxKeyLen + 1));
    b->last_key = arena->allocate_array<uint8_t>(
        static_cast<size_t>(kFstMaxKeyLen));
    if (!b->last_key) return false;
    b->dedup = arena->allocate_array<FstDedupEntry>(
        static_cast<size_t>(kFstDedupSlots));
    if (!b->dedup) return false;
    std::memset(b->dedup, 0,
                sizeof(FstDedupEntry) * static_cast<size_t>(kFstDedupSlots));
    return true;
}

// Append an arc to the frontier node at depth `d`. Returns false on overflow.
inline bool fst_frontier_push_arc(FstBuilder* b, int32_t depth, uint8_t label,
                                   int32_t target, int64_t output,
                                   bool is_final) noexcept {
    assert(b != nullptr);
    assert(depth >= 0 && depth <= kFstMaxKeyLen);
    FstFrontierNode* fn = &b->frontier[depth];
    if (fn->arc_count >= kFstMaxArcsPerNode) return false;
    FstPendingArc* a = &fn->arcs[fn->arc_count++];
    a->label = label;
    a->target = target;
    a->output = output;
    a->is_final = is_final ? 1u : 0u;
    std::memset(a->_pad, 0, sizeof(a->_pad));
    return true;
}

inline bool fst_builder_add(FstBuilder* b,
                             const uint8_t* key, int32_t key_len,
                             int64_t output) noexcept {
    assert(b != nullptr);
    assert(key != nullptr || key_len == 0);
    if (!b || b->finished) return false;
    if (key_len <= 0 || key_len > kFstMaxKeyLen) return false;
    // Strict-ascending precondition.
    if (b->num_keys > 0) {
        const int32_t cmp_len = b->last_key_len < key_len ? b->last_key_len
                                                          : key_len;
        int c = std::memcmp(b->last_key, key, static_cast<size_t>(cmp_len));
        if (c > 0) return false;
        if (c == 0 && b->last_key_len >= key_len) return false;
    }
    // Find common prefix length with last key.
    int32_t common = 0;
    while (common < b->last_key_len && common < key_len &&
           b->last_key[common] == key[common]) {
        ++common;
    }
    // Freeze frontier nodes from deepest down to depth common+1; record the
    // canonical offset of each frozen node into its parent's last arc.
    // Empty frontier slots (the "below a leaf" depths past a key's terminal
    // arc) freeze into the sentinel 0-arc node at offset 0 - the parent's
    // IsFinal flag still carries the answer for that key, and onward
    // lookups simply find no arc at the empty target.
    for (int32_t depth = b->last_key_len; depth > common; --depth) {
        FstFrontierNode* fn = &b->frontier[depth];
        if (fn->arc_count == 0) continue;  // dedup-to-sentinel handled below
        int32_t off = fst_detail::freeze_frontier_node(b, depth);
        if (off < 0) return false;
        FstFrontierNode* parent = &b->frontier[depth - 1];
        assert(parent->arc_count > 0);
        parent->arcs[parent->arc_count - 1].target = off;
    }
    // Extend the frontier with new arcs for the suffix of `key`.
    for (int32_t depth = common; depth < key_len; ++depth) {
        const bool last_byte = (depth + 1 == key_len);
        const int64_t out_val = last_byte ? output : 0;
        if (!fst_frontier_push_arc(b, depth, key[depth], -1, out_val,
                                   last_byte)) {
            return false;
        }
    }
    std::memcpy(b->last_key, key, static_cast<size_t>(key_len));
    b->last_key_len = key_len;
    b->last_output = output;
    ++b->num_keys;
    return true;
}

inline bool fst_builder_finish(FstBuilder* b, FstView* out) noexcept {
    assert(b != nullptr);
    assert(out != nullptr);
    if (!b || !out || b->finished) return false;
    // Empty FST - root is the sentinel 0-arc node at offset 0.
    if (b->num_keys == 0) {
        out->nodes = b->buf;
        out->bytes = b->size;
        out->root_offset = 0;
        b->finished = 1;
        return true;
    }
    // Freeze every frontier node from last_key_len down to 1; the depth-0
    // node becomes the root.
    for (int32_t depth = b->last_key_len; depth > 0; --depth) {
        FstFrontierNode* fn = &b->frontier[depth];
        if (fn->arc_count == 0) continue;
        int32_t off = fst_detail::freeze_frontier_node(b, depth);
        if (off < 0) return false;
        FstFrontierNode* parent = &b->frontier[depth - 1];
        assert(parent->arc_count > 0);
        parent->arcs[parent->arc_count - 1].target = off;
    }
    int32_t root = fst_detail::freeze_frontier_node(b, 0);
    if (root < 0) return false;
    out->nodes = b->buf;
    out->bytes = b->size;
    out->root_offset = root;
    b->finished = 1;
    return true;
}

// --- Lookup ----------------------------------------------------------------

// Internal: find the arc at `node_off` whose label == byte. Returns the arc
// byte-offset within the buffer, or -1 if no such arc.
BOLT_FORCE_INLINE int32_t fst_find_arc(const uint8_t* BOLT_RESTRICT buf,
                                        int32_t node_off,
                                        uint8_t byte) noexcept {
    assert(buf != nullptr);
    assert(node_off >= 0);
    uint16_t ac16;
    std::memcpy(&ac16, buf + node_off, 2);
    const int32_t arc_count = ac16;
    const int32_t base = node_off + 2;
    for (int32_t i = 0; i < arc_count; ++i) {
        const int32_t off = base + i * kFstArcBytes;
        if (buf[off] == byte) return off;
    }
    return -1;
}

inline bool fst_lookup(FstView v,
                        const uint8_t* key, int32_t key_len,
                        int64_t* out) noexcept {
    assert(out != nullptr);
    assert(key != nullptr || key_len == 0);
    if (!v.nodes || key_len <= 0) return false;
    int32_t node = v.root_offset;
    for (int32_t d = 0; d < key_len; ++d) {
        int32_t arc = fst_find_arc(v.nodes, node, key[d]);
        if (arc < 0) return false;
        const bool last = (d + 1 == key_len);
        if (last) {
            const uint8_t flags = v.nodes[arc + 1];
            if ((flags & fst_flag::kIsFinal) == 0) return false;
            int64_t output;
            std::memcpy(&output, v.nodes + arc + 6, 8);
            *out = output;
            return true;
        }
        int32_t target;
        std::memcpy(&target, v.nodes + arc + 2, 4);
        if (target < 0) return false;  // no child - key not present
        node = target;
    }
    return false;
}

// --- Scan ------------------------------------------------------------------

using FstScanCb = bool (*)(void* user, const uint8_t* key, int32_t key_len,
                            int64_t output);

// DFS state frame for iterative scanning. arc_idx is the next arc to visit
// at this node; node_off is the node's byte offset.
struct FstScanFrame {
    int32_t node_off;
    int32_t arc_idx;
    int32_t arc_count;
    int32_t _pad;
};
static_assert(sizeof(FstScanFrame) == 16, "FstScanFrame is 16 bytes");

namespace fst_detail {

// Walk the FST from root following `prefix`. On success returns the node
// offset reached after consuming all prefix bytes (this is the node from
// which all keys with that prefix branch). Sets `*matched_terminal_output`
// to the output if the prefix itself is a key; *prefix_is_key flags it.
inline bool walk_prefix(FstView v, const uint8_t* prefix, int32_t prefix_len,
                        int32_t* out_node, bool* prefix_is_key,
                        int64_t* prefix_output) noexcept {
    assert(out_node != nullptr);
    assert(prefix_is_key != nullptr);
    *prefix_is_key = false;
    *prefix_output = 0;
    int32_t node = v.root_offset;
    for (int32_t d = 0; d < prefix_len; ++d) {
        int32_t arc = fst_find_arc(v.nodes, node, prefix[d]);
        if (arc < 0) return false;
        const bool last = (d + 1 == prefix_len);
        if (last) {
            const uint8_t flags = v.nodes[arc + 1];
            if (flags & fst_flag::kIsFinal) {
                *prefix_is_key = true;
                std::memcpy(prefix_output, v.nodes + arc + 6, 8);
            }
        }
        int32_t target;
        std::memcpy(&target, v.nodes + arc + 2, 4);
        if (last) { *out_node = target; return true; }
        if (target < 0) return false;
        node = target;
    }
    *out_node = node;
    return true;
}

// Iterative DFS over (node, arc_idx) frames. Emits keys via `cb`; returns
// false if cb stops early or if any internal cap is hit. `key_buf` is the
// scratch key buffer (capacity `key_cap`); on entry `key_buf[0..prefix_len)`
// already holds the prefix bytes.
inline bool dfs_emit(FstView v, int32_t start_node,
                     uint8_t* key_buf, int32_t prefix_len, int32_t key_cap,
                     FstScanFrame* stack, int32_t stack_cap,
                     FstScanCb cb, void* user) noexcept {
    assert(key_buf != nullptr);
    assert(stack != nullptr);
    assert(prefix_len >= 0);
    if (start_node < 0) return true;  // empty subtree - nothing to emit
    int32_t sp = 0;
    if (sp >= stack_cap) return false;
    uint16_t ac16;
    std::memcpy(&ac16, v.nodes + start_node, 2);
    stack[sp++] = FstScanFrame{ start_node, 0, static_cast<int32_t>(ac16), 0 };
    while (sp > 0) {
        FstScanFrame* f = &stack[sp - 1];
        if (f->arc_idx >= f->arc_count) { --sp; continue; }
        // depth = number of label bytes accumulated above the current frame.
        // Bottom frame sits at depth=prefix_len (its first arc lands at that
        // byte index), next frame down adds one byte per descent.
        const int32_t depth = prefix_len + (sp - 1);
        if (depth >= key_cap) return false;
        const int32_t arc_off = f->node_off + 2 + f->arc_idx * kFstArcBytes;
        ++f->arc_idx;
        const uint8_t label = v.nodes[arc_off];
        const uint8_t flags = v.nodes[arc_off + 1];
        int32_t target;
        std::memcpy(&target, v.nodes + arc_off + 2, 4);
        key_buf[depth] = label;
        const int32_t emit_len = depth + 1;
        if (flags & fst_flag::kIsFinal) {
            int64_t output;
            std::memcpy(&output, v.nodes + arc_off + 6, 8);
            if (!cb(user, key_buf, emit_len, output)) return true;
        }
        if (target >= 0) {
            if (sp >= stack_cap) return false;
            uint16_t cac;
            std::memcpy(&cac, v.nodes + target, 2);
            stack[sp++] = FstScanFrame{ target, 0,
                                         static_cast<int32_t>(cac), 0 };
        }
    }
    return true;
}

}  // namespace fst_detail

inline bool fst_scan_prefix(FstView v,
                             const uint8_t* prefix, int32_t prefix_len,
                             FstScanCb cb, void* user,
                             Arena* scratch) noexcept {
    assert(cb != nullptr);
    assert(scratch != nullptr);
    if (!cb || !scratch || prefix_len < 0) return false;
    int32_t after_node = -1;
    bool prefix_is_key = false;
    int64_t prefix_output = 0;
    if (prefix_len > 0) {
        if (!fst_detail::walk_prefix(v, prefix, prefix_len, &after_node,
                                      &prefix_is_key, &prefix_output)) {
            return true;  // no match - empty result, not an error
        }
    } else {
        after_node = v.root_offset;
    }
    constexpr int32_t kKeyCap = kFstMaxKeyLen;
    constexpr int32_t kStackCap = kFstMaxKeyLen + 1;
    uint8_t* key_buf = scratch->allocate_array<uint8_t>(kKeyCap);
    FstScanFrame* stack = scratch->allocate_array<FstScanFrame>(kStackCap);
    if (!key_buf || !stack) return false;
    if (prefix_len > 0) std::memcpy(key_buf, prefix, static_cast<size_t>(prefix_len));
    if (prefix_is_key) {
        if (!cb(user, key_buf, prefix_len, prefix_output)) return true;
    }
    return fst_detail::dfs_emit(v, after_node, key_buf, prefix_len, kKeyCap,
                                 stack, kStackCap, cb, user);
}

// Range filter trampoline - we still DFS the root subtree but the callback
// drops keys outside [lo, hi). Simple and correct; a tighter walk that
// prunes entire subtrees byte-wise can replace this once measured.
struct FstRangeCtx {
    const uint8_t* lo;
    int32_t        lo_len;
    const uint8_t* hi;
    int32_t        hi_len;
    FstScanCb      inner;
    void*          user;
};

inline int fst_bytewise_cmp(const uint8_t* a, int32_t alen,
                             const uint8_t* b, int32_t blen) noexcept {
    assert(a != nullptr || alen == 0);
    assert(b != nullptr || blen == 0);
    const int32_t n = alen < blen ? alen : blen;
    int c = std::memcmp(a, b, static_cast<size_t>(n));
    if (c != 0) return c;
    if (alen == blen) return 0;
    return alen < blen ? -1 : 1;
}

inline bool fst_range_trampoline(void* user, const uint8_t* k, int32_t klen,
                                  int64_t output) noexcept {
    FstRangeCtx* ctx = static_cast<FstRangeCtx*>(user);
    if (fst_bytewise_cmp(k, klen, ctx->lo, ctx->lo_len) < 0) return true;
    if (fst_bytewise_cmp(k, klen, ctx->hi, ctx->hi_len) >= 0) return true;
    return ctx->inner(ctx->user, k, klen, output);
}

inline bool fst_scan_range(FstView v,
                            const uint8_t* lo, int32_t lo_len,
                            const uint8_t* hi, int32_t hi_len,
                            FstScanCb cb, void* user,
                            Arena* scratch) noexcept {
    assert(cb != nullptr);
    assert(scratch != nullptr);
    if (!cb || !scratch) return false;
    FstRangeCtx ctx{ lo, lo_len, hi, hi_len, cb, user };
    return fst_scan_prefix(v, nullptr, 0, &fst_range_trampoline, &ctx, scratch);
}

}  // namespace kernels
}  // namespace bolt
