// bolt_jsonb.h — sorted-keys binary JSON container (Layer 1.4)
//
// Mirrors Postgres JSONB: 4-byte Jentry header per child (28-bit length-or-
// offset + 1-bit is-offset flag + 3-bit type tag), object keys sorted at
// finalise (length asc, then lex asc), stride-based offset/length encoding so
// that every 32nd Jentry carries an absolute offset (cheap random access)
// while the rest carry lengths (cheap to bit-pack later, no quadratic prefix
// sum on lookup).
//
// Tiger-Style discipline: no exceptions, no RTTI, no virtuals; no STL in this
// header; all allocation through a caller-supplied `bolt::Arena`. Path lookup
// never asserts on user input — malformed buffers return false.
//
// On-disk layout (little-endian throughout):
//
//   Container := ContainerHeader (4B)
//                KeysBytes      (4B)             // total bytes of key payload
//                ValuesBytes    (4B)             // total bytes of value payload
//                Jentry[N]                       // arrays / scalar wrapper
//                Jentry[N_keys] Jentry[N_keys]   // objects: keys then vals
//                payload[]                       // raw bytes referenced
//
//   For arrays / scalar wrappers, KeysBytes is 0; ValuesBytes is the total
//   payload bytes of the (single) array's children. Two 4-byte counters keep
//   the format uniform across containers and let the reader derive the
//   length of any slot — including those at stride boundaries — without a
//   forward scan that needs an external bound.
//
//   ContainerHeader :=
//     bits  0..27  : count (number of children for array; key/value pairs for
//                    object). Top-level scalar is encoded as a single-element
//                    array (Postgres convention) so the format is uniform.
//     bits 28..30  : flags  (bit 28 = is_object, bit 29 = is_array,
//                            bit 30 = is_scalar_wrapper)
//     bit  31      : reserved (zero)
//
//   Jentry :=
//     bits  0..27  : length (default) OR absolute offset from end of Jentry
//                    table (when is_offset=1).
//     bit  28      : is_offset (1 → bits 0..27 are an offset, 0 → length)
//     bits 29..31  : JsonbTag
//
// Stride rule: Jentry index i carries an offset when (i % 32) == 0 and i > 0.
// All other indices carry the length of that child's payload bytes. Random
// access at index i is `payload_base + offset[floor(i/32)] + sum(lengths[...
// floor(i/32) + 1 .. i])`. Index 0 is implicitly offset 0.
//
// Numbers: Int64 / Float64 little-endian fixed 8 bytes. Booleans and null
// carry zero payload bytes (the tag conveys the value).

#pragma once

#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"

namespace bolt::doc {

enum class JsonbTag : uint8_t {
    String    = 0,
    Int64     = 1,
    Float64   = 2,
    BoolTrue  = 3,
    BoolFalse = 4,
    Null      = 5,
    Object    = 6,
    Array     = 7,
};

struct Jentry { uint32_t bits; };
static_assert(sizeof(Jentry) == 4, "Jentry is 4 bytes");

struct ContainerHeader { uint32_t bits; };
static_assert(sizeof(ContainerHeader) == 4, "ContainerHeader is 4 bytes");

struct JsonbView {
    const uint8_t* data;
    int32_t        bytes;
};

struct JsonbValue {
    JsonbTag       tag;
    const uint8_t* bytes;     // String: ptr into payload
    int32_t        len;       // String: byte length
    int64_t        i64;       // Int64
    double         f64;       // Float64
    JsonbView      container; // Object/Array: nested view
};

// ---------------------------------------------------------------------------
// Bit layout helpers (BOLT_FORCE_INLINE; pure bit-twiddling; trivially safe).
// ---------------------------------------------------------------------------

static constexpr uint32_t kJEntryLenMask    = 0x0FFFFFFFu; // 28 bits
static constexpr uint32_t kJEntryOffsetFlag = 0x10000000u; // bit 28
static constexpr uint32_t kJEntryTagShift   = 29;          // bits 29..31

static constexpr uint32_t kHdrCountMask     = 0x0FFFFFFFu;
static constexpr uint32_t kHdrFlagObject    = 0x10000000u;
static constexpr uint32_t kHdrFlagArray     = 0x20000000u;
static constexpr uint32_t kHdrFlagScalar    = 0x40000000u;

static constexpr int32_t  kStride           = 32;
static constexpr int32_t  kMaxDepth         = 64;
// Maximum encoded payload — the 28-bit Jentry length bound. Buffers larger
// than this cannot be addressed by the format.
static constexpr int32_t  kMaxBytes         = 0x0FFFFFFF;

BOLT_FORCE_INLINE uint32_t je_make_len(JsonbTag tag, uint32_t len) noexcept {
    return (static_cast<uint32_t>(tag) << kJEntryTagShift) | (len & kJEntryLenMask);
}
BOLT_FORCE_INLINE uint32_t je_make_off(JsonbTag tag, uint32_t off) noexcept {
    return (static_cast<uint32_t>(tag) << kJEntryTagShift)
         | kJEntryOffsetFlag
         | (off & kJEntryLenMask);
}
BOLT_FORCE_INLINE JsonbTag je_tag(uint32_t je) noexcept {
    return static_cast<JsonbTag>((je >> kJEntryTagShift) & 0x7u);
}
BOLT_FORCE_INLINE bool je_is_offset(uint32_t je) noexcept {
    return (je & kJEntryOffsetFlag) != 0;
}
BOLT_FORCE_INLINE uint32_t je_value(uint32_t je) noexcept {
    return je & kJEntryLenMask;
}

BOLT_FORCE_INLINE uint32_t hdr_count(uint32_t h) noexcept { return h & kHdrCountMask; }
BOLT_FORCE_INLINE bool     hdr_is_object(uint32_t h) noexcept { return (h & kHdrFlagObject) != 0; }
BOLT_FORCE_INLINE bool     hdr_is_array(uint32_t h) noexcept { return (h & kHdrFlagArray) != 0; }
BOLT_FORCE_INLINE bool     hdr_is_scalar(uint32_t h) noexcept { return (h & kHdrFlagScalar) != 0; }

// ---------------------------------------------------------------------------
// Reader — bounded, never-throws, never-asserts on user data.
// ---------------------------------------------------------------------------

namespace detail {

BOLT_FORCE_INLINE bool read_u32_le(const uint8_t* BOLT_RESTRICT p,
                                   int32_t bytes, int32_t off,
                                   uint32_t* BOLT_RESTRICT out) noexcept {
    if (off < 0 || bytes < 4 || off > bytes - 4) return false;
    uint32_t v;
    std::memcpy(&v, p + off, 4);
    *out = v;
    return true;
}

BOLT_FORCE_INLINE bool read_i64_le(const uint8_t* BOLT_RESTRICT p,
                                   int32_t bytes, int32_t off,
                                   int64_t* BOLT_RESTRICT out) noexcept {
    if (off < 0 || bytes < 8 || off > bytes - 8) return false;
    int64_t v;
    std::memcpy(&v, p + off, 8);
    *out = v;
    return true;
}

BOLT_FORCE_INLINE bool read_f64_le(const uint8_t* BOLT_RESTRICT p,
                                   int32_t bytes, int32_t off,
                                   double* BOLT_RESTRICT out) noexcept {
    if (off < 0 || bytes < 8 || off > bytes - 8) return false;
    double v;
    std::memcpy(&v, p + off, 8);
    *out = v;
    return true;
}

// Resolve the (offset_in_payload, length_bytes) for child `idx` in a Jentry
// table whose first entry is at `je_base`. `count` is the table length in
// entries. `payload_total` is the total bytes the table's payload region
// occupies (== sum of all child lengths) — needed to compute the length of
// the last child when its Jentry is an offset slot. Pass 0 to mean "infer
// from buffer end" (only valid when the payload region runs to buffer end).
// Returns false if the table is malformed.
inline bool resolve_extent(const uint8_t* BOLT_RESTRICT data, int32_t bytes,
                           int32_t je_base, int32_t count, int32_t idx,
                           int32_t payload_total,
                           int32_t* BOLT_RESTRICT out_off,
                           int32_t* BOLT_RESTRICT out_len,
                           JsonbTag* BOLT_RESTRICT out_tag) noexcept {
    if (idx < 0 || idx >= count) return false;
    if (je_base < 0 || count < 0) return false;
    if (static_cast<int64_t>(je_base) + static_cast<int64_t>(count) * 4 > bytes) {
        return false;
    }

    // Find the most recent stride boundary <= idx that carries an offset, then
    // walk forward summing lengths.
    int32_t base_idx = (idx / kStride) * kStride;
    int32_t base_off = 0;
    if (base_idx != 0) {
        uint32_t je;
        if (!read_u32_le(data, bytes, je_base + base_idx * 4, &je)) return false;
        if (!je_is_offset(je)) return false;
        base_off = static_cast<int32_t>(je_value(je));
    }

    // Strategy: derive the length of any slot i by reading forward to the
    // next stride boundary (or end of count) and subtracting the offset
    // delta. With kStride=32, the read-forward is bounded at 31 slots.
    //
    // Helper: cumulative offset of slot i within the run that starts at
    // `run_base_idx` (a stride boundary, or 0). For i==run_base_idx → 0;
    // otherwise sum lengths of slots [run_base_idx .. i-1]. Length slots
    // contribute their value; offset slots contribute their (next_off -
    // this_off) which we resolve recursively when the next slot is also an
    // offset (we cap recursion depth at the stride distance).
    auto run_base_off = [&](int32_t run_base_idx, int32_t* out_off) noexcept -> bool {
        if (run_base_idx == 0) { *out_off = 0; return true; }
        uint32_t je;
        if (!read_u32_le(data, bytes, je_base + run_base_idx * 4, &je)) return false;
        if (!je_is_offset(je)) return false;
        *out_off = static_cast<int32_t>(je_value(je));
        return true;
    };

    auto slot_length = [&](int32_t i, uint32_t je,
                           int32_t* out) noexcept -> bool {
        if (!je_is_offset(je)) {
            *out = static_cast<int32_t>(je_value(je));
            return *out >= 0;
        }
        // Offset slot. Walk forward to the next stride boundary or end-of-
        // table, accumulate length-slot values, then `len = next_off -
        // (my_off + accumulated_lengths)`. Bounded by kStride.
        const int32_t my_off = static_cast<int32_t>(je_value(je));
        int32_t acc = 0;
        int32_t j = i + 1;
        while (j < count && (j % kStride) != 0) {
            uint32_t je_j;
            if (!read_u32_le(data, bytes, je_base + j * 4, &je_j)) return false;
            if (je_is_offset(je_j)) return false;
            acc += static_cast<int32_t>(je_value(je_j));
            ++j;
        }
        int32_t next_off;
        if (j == count) {
            next_off = payload_total > 0
                ? payload_total
                : bytes - (je_base + count * 4);
        } else {
            uint32_t je_j;
            if (!read_u32_le(data, bytes, je_base + j * 4, &je_j)) return false;
            if (!je_is_offset(je_j)) return false;
            next_off = static_cast<int32_t>(je_value(je_j));
        }
        *out = next_off - my_off - acc;
        return *out >= 0;
    };

    int32_t off = base_off;
    int32_t len = 0;
    JsonbTag tag = JsonbTag::Null;
    for (int32_t i = base_idx; i <= idx; ++i) {
        uint32_t je;
        if (!read_u32_le(data, bytes, je_base + i * 4, &je)) return false;
        const bool is_off = je_is_offset(je);
        if (i == idx) {
            tag = je_tag(je);
            if (!slot_length(i, je, &len)) return false;
            if (is_off) off = static_cast<int32_t>(je_value(je));
            break;
        }
        int32_t step = 0;
        if (!slot_length(i, je, &step)) return false;
        off += step;
    }
    (void)run_base_off;  // currently unused; kept for future extensions
    if (off < 0 || len < 0) return false;
    const int32_t payload_base = je_base + count * 4;
    if (static_cast<int64_t>(payload_base) + off + len > bytes) return false;
    *out_off = payload_base + off;
    *out_len = len;
    *out_tag = tag;
    return true;
}

// Decode a child whose extent has been resolved. Fills `out`. Returns false
// if the payload is malformed for the tag.
inline bool decode_child(const uint8_t* data, int32_t bytes,
                         int32_t off, int32_t len, JsonbTag tag,
                         JsonbValue* out) noexcept {
    out->tag = tag;
    out->bytes = nullptr;
    out->len = 0;
    out->i64 = 0;
    out->f64 = 0.0;
    out->container.data = nullptr;
    out->container.bytes = 0;
    switch (tag) {
        case JsonbTag::String: {
            if (off < 0 || len < 0) return false;
            if (static_cast<int64_t>(off) + len > bytes) return false;
            out->bytes = data + off;
            out->len = len;
            return true;
        }
        case JsonbTag::Int64: {
            if (len != 8) return false;
            return read_i64_le(data, bytes, off, &out->i64);
        }
        case JsonbTag::Float64: {
            if (len != 8) return false;
            return read_f64_le(data, bytes, off, &out->f64);
        }
        case JsonbTag::BoolTrue:
        case JsonbTag::BoolFalse:
        case JsonbTag::Null: {
            return len == 0;
        }
        case JsonbTag::Object:
        case JsonbTag::Array: {
            if (len < 4) return false;
            if (static_cast<int64_t>(off) + len > bytes) return false;
            out->container.data = data + off;
            out->container.bytes = len;
            return true;
        }
    }
    return false;
}

// Read header trio (count, keys_bytes, values_bytes). Returns false if the
// container is malformed (header doesn't fit, declared tables/payload exceed
// buffer).
inline bool read_container(const uint8_t* data, int32_t bytes,
                           uint32_t* hdr, int32_t* keys_bytes,
                           int32_t* values_bytes) noexcept {
    if (data == nullptr) return false;
    if (bytes < 12) return false;
    uint32_t h, kb, vb;
    std::memcpy(&h, data + 0, 4);
    std::memcpy(&kb, data + 4, 4);
    std::memcpy(&vb, data + 8, 4);
    if ((kb & ~kJEntryLenMask) != 0) return false;
    if ((vb & ~kJEntryLenMask) != 0) return false;
    *hdr = h;
    *keys_bytes = static_cast<int32_t>(kb);
    *values_bytes = static_cast<int32_t>(vb);
    return true;
}

// Read a key string (Jentry table starts at `je_base`, holds 2*count entries:
// keys then values). Fills `*kp,*klen` for key index `idx`.
inline bool read_key(const uint8_t* data, int32_t bytes,
                     int32_t je_base, int32_t pair_count, int32_t keys_bytes,
                     int32_t idx, const uint8_t** kp, int32_t* klen) noexcept {
    int32_t off, len;
    JsonbTag tag;
    if (!resolve_extent(data, bytes, je_base, pair_count, idx,
                        /*payload_total=*/keys_bytes,
                        &off, &len, &tag)) return false;
    if (tag != JsonbTag::String) return false;
    // The values Jentry table sits between the keys Jentry table and any
    // payload bytes. resolve_extent computed `payload_base = je_base +
    // count*4` (immediately after keys table); we shift past the values
    // Jentry table.
    off += pair_count * 4;
    if (off < 0 || static_cast<int64_t>(off) + len > bytes) return false;
    *kp = data + off;
    *klen = len;
    return true;
}

// Read value for object index `idx`.
inline bool read_value(const uint8_t* data, int32_t bytes,
                       int32_t je_base, int32_t pair_count,
                       int32_t keys_bytes, int32_t values_bytes,
                       int32_t idx, JsonbValue* out) noexcept {
    int32_t off, len;
    JsonbTag tag;
    const int32_t values_je_base = je_base + pair_count * 4;
    if (!resolve_extent(data, bytes, values_je_base, pair_count, idx,
                        /*payload_total=*/values_bytes,
                        &off, &len, &tag)) return false;
    // Resolve_extent computed payload_base = values_je_base + count*4 — the
    // start of the keys payload region. Skip past keys to reach values.
    off += keys_bytes;
    if (off < 0 || static_cast<int64_t>(off) + len > bytes) return false;
    return decode_child(data, bytes, off, len, tag, out);
}

// Lex-then-length comparator used by both sort and binary-search.
//   primary: shorter < longer
//   secondary: memcmp ascending
BOLT_FORCE_INLINE int key_cmp(const uint8_t* a, int32_t alen,
                              const uint8_t* b, int32_t blen) noexcept {
    if (alen != blen) return alen < blen ? -1 : 1;
    if (alen == 0) return 0;
    return std::memcmp(a, b, static_cast<size_t>(alen));
}

inline bool array_lookup(JsonbView v, int32_t idx, JsonbValue* out) noexcept {
    uint32_t hdr;
    int32_t kb, vb;
    if (!read_container(v.data, v.bytes, &hdr, &kb, &vb)) return false;
    if (!hdr_is_array(hdr) && !hdr_is_scalar(hdr)) return false;
    if (kb != 0) return false;  // arrays carry no keys
    const int32_t count = static_cast<int32_t>(hdr_count(hdr));
    if (idx < 0 || idx >= count) return false;
    int32_t off, len;
    JsonbTag tag;
    if (!resolve_extent(v.data, v.bytes, /*je_base=*/12, count, idx,
                        /*payload_total=*/vb, &off, &len, &tag)) return false;
    return decode_child(v.data, v.bytes, off, len, tag, out);
}

inline bool object_lookup(JsonbView v, const char* key, int32_t klen,
                          JsonbValue* out) noexcept {
    if (klen < 0) return false;
    uint32_t hdr;
    int32_t kb, vb;
    if (!read_container(v.data, v.bytes, &hdr, &kb, &vb)) return false;
    if (!hdr_is_object(hdr)) return false;
    const int32_t count = static_cast<int32_t>(hdr_count(hdr));
    if (count == 0) {
        out->tag = JsonbTag::Null;
        out->len = 0;
        return false;
    }
    const uint8_t* needle = reinterpret_cast<const uint8_t*>(key);
    int32_t lo = 0, hi = count;
    while (lo < hi) {
        const int32_t mid = lo + ((hi - lo) >> 1);
        const uint8_t* kp = nullptr;
        int32_t kl = 0;
        if (!read_key(v.data, v.bytes, /*je_base=*/12, count, kb, mid,
                      &kp, &kl)) return false;
        const int c = key_cmp(kp, kl, needle, klen);
        if (c == 0) {
            return read_value(v.data, v.bytes, /*je_base=*/12, count, kb, vb,
                              mid, out);
        }
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return false;
}

}  // namespace detail

// Public lookup. `path_kind[i] == 0` → object key (path_str[i], path_len[i]);
// `== 1` → array index (path_idx[i]). Returns true if traversal completed
// (value or "absent"); false only if the buffer is malformed. When traversal
// terminates early because a key was missing, sets out->tag = Null and
// returns true.
inline bool path_lookup(JsonbView view,
                        const uint8_t* BOLT_RESTRICT path_kind,
                        const char* const* BOLT_RESTRICT path_str,
                        const int32_t* BOLT_RESTRICT path_len,
                        const int32_t* BOLT_RESTRICT path_idx,
                        int32_t depth,
                        JsonbValue* out) noexcept {
    assert(out != nullptr);
    assert(depth >= 0);
    if (depth < 0 || depth > kMaxDepth) return false;
    if (view.data == nullptr || view.bytes < 12) return false;

    out->tag = JsonbTag::Null;
    out->len = 0;
    out->bytes = nullptr;
    out->container.data = nullptr;
    out->container.bytes = 0;

    JsonbView cur = view;
    JsonbValue tmp{};
    // For depth 0, "the value" is the root container itself unwrapped. If the
    // root is a scalar wrapper, decode child 0; else surface Object/Array.
    if (depth == 0) {
        uint32_t hdr; int32_t kb, vb;
        if (!detail::read_container(cur.data, cur.bytes, &hdr, &kb, &vb)) return false;
        if (hdr_is_scalar(hdr)) {
            return detail::array_lookup(cur, 0, out);
        }
        out->tag = hdr_is_object(hdr) ? JsonbTag::Object : JsonbTag::Array;
        out->container = cur;
        return true;
    }

    for (int32_t i = 0; i < depth; ++i) {
        const uint8_t kind = path_kind ? path_kind[i] : 0;
        uint32_t hdr; int32_t kb, vb;
        if (!detail::read_container(cur.data, cur.bytes, &hdr, &kb, &vb)) return false;
        const int32_t hcount = static_cast<int32_t>(hdr_count(hdr));
        const int32_t je_count = hdr_is_object(hdr) ? hcount * 2 : hcount;
        if (static_cast<int64_t>(12) + static_cast<int64_t>(je_count) * 4
              + static_cast<int64_t>(kb) + static_cast<int64_t>(vb)
            > cur.bytes) {
            return false;
        }
        if (kind == 0) {
            if (!hdr_is_object(hdr)) return false;
            if (path_str == nullptr || path_len == nullptr) return false;
            const char* k = path_str[i];
            const int32_t kl = path_len[i];
            if (!detail::object_lookup(cur, k, kl, &tmp)) {
                out->tag = JsonbTag::Null;
                return true;  // missing key → absent value, not malformed
            }
        } else if (kind == 1) {
            if (!hdr_is_array(hdr) && !hdr_is_scalar(hdr)) return false;
            if (path_idx == nullptr) return false;
            if (!detail::array_lookup(cur, path_idx[i], &tmp)) {
                out->tag = JsonbTag::Null;
                return true;
            }
        } else {
            return false;
        }

        if (i + 1 == depth) {
            *out = tmp;
            return true;
        }
        // Step into nested container; non-container mid-path ⇒ absent.
        if (tmp.tag != JsonbTag::Object && tmp.tag != JsonbTag::Array) {
            out->tag = JsonbTag::Null;
            return true;
        }
        cur = tmp.container;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Builder. Stack-based: every begin_object/array pushes a frame; emit_*
// records (tag, payload-bytes, key/value role) into a per-frame child list.
// `finish` walks the stack bottom-up, sorting object keys, computing offsets,
// and copying into a flat arena buffer. Cap depth at kMaxDepth.
// ---------------------------------------------------------------------------

struct JsonbChild {
    JsonbTag tag;
    int32_t  payload_off;   // offset within JsonbBuilder.scratch
    int32_t  payload_len;   // bytes
    // For object-frame children, these address the key in the same scratch.
    int32_t  key_off;
    int32_t  key_len;
    // For nested-container children, points to the child container's
    // finalised JsonbView in the builder's nested-buffer slab. -1 if not a
    // container.
    int32_t  nested_off;    // offset in scratch
    int32_t  nested_len;    // bytes
};

struct JsonbFrame {
    bool     is_object;
    int32_t  child_first;   // index into JsonbBuilder.children
    int32_t  child_count;
    bool     awaiting_value; // object: just saw a key, expecting value
    int32_t  pending_key_off;
    int32_t  pending_key_len;
};

struct JsonbBuilder {
    Arena*     arena;
    uint8_t*   scratch;
    int32_t    scratch_cap;
    int32_t    scratch_size;

    JsonbChild* children;
    int32_t     children_cap;
    int32_t     children_size;

    JsonbFrame  frames[kMaxDepth];
    int32_t     frame_top;          // -1 ⇒ no frame yet (at root)

    bool        root_emitted;       // a root scalar / container has been emitted
    bool        root_is_container;  // top-level is object/array
    bool        finished;
    bool        ok;                 // sticky error flag
    int32_t     root_child_idx;     // index in `children` of the root child
};

namespace detail {

inline bool scratch_reserve(JsonbBuilder* b, int32_t need) noexcept {
    if (need <= b->scratch_cap - b->scratch_size) return true;
    int32_t new_cap = b->scratch_cap > 0 ? b->scratch_cap : 64;
    while (new_cap - b->scratch_size < need) {
        if (new_cap > (kMaxBytes >> 1)) return false;
        new_cap *= 2;
    }
    uint8_t* nb = static_cast<uint8_t*>(
        b->arena->allocate(static_cast<size_t>(new_cap), alignof(uint8_t)));
    if (!nb) return false;
    if (b->scratch_size > 0) {
        std::memcpy(nb, b->scratch, static_cast<size_t>(b->scratch_size));
    }
    b->scratch = nb;
    b->scratch_cap = new_cap;
    return true;
}

inline bool children_reserve(JsonbBuilder* b, int32_t extra) noexcept {
    if (b->children_size + extra <= b->children_cap) return true;
    int32_t new_cap = b->children_cap > 0 ? b->children_cap : 16;
    while (new_cap < b->children_size + extra) new_cap *= 2;
    JsonbChild* nb = static_cast<JsonbChild*>(
        b->arena->allocate(static_cast<size_t>(new_cap) * sizeof(JsonbChild),
                           alignof(JsonbChild)));
    if (!nb) return false;
    if (b->children_size > 0) {
        std::memcpy(nb, b->children,
                    static_cast<size_t>(b->children_size) * sizeof(JsonbChild));
    }
    b->children = nb;
    b->children_cap = new_cap;
    return true;
}

inline int32_t scratch_append(JsonbBuilder* b, const void* data, int32_t n) noexcept {
    if (n < 0) { b->ok = false; return -1; }
    if (!scratch_reserve(b, n)) { b->ok = false; return -1; }
    const int32_t off = b->scratch_size;
    if (n > 0) std::memcpy(b->scratch + off, data, static_cast<size_t>(n));
    b->scratch_size += n;
    return off;
}

inline bool push_child(JsonbBuilder* b, JsonbChild c) noexcept {
    if (b->frame_top < 0) {
        // Root-level: only one root child allowed.
        if (b->root_emitted) { b->ok = false; return false; }
        b->root_emitted = true;
        if (!children_reserve(b, 1)) return false;
        b->root_child_idx = b->children_size;
        b->children[b->children_size++] = c;
        return true;
    }
    JsonbFrame& f = b->frames[b->frame_top];
    if (f.is_object) {
        if (!f.awaiting_value) { b->ok = false; return false; }
        c.key_off = f.pending_key_off;
        c.key_len = f.pending_key_len;
        f.awaiting_value = false;
    }
    if (!children_reserve(b, 1)) return false;
    if (f.child_first < 0) f.child_first = b->children_size;
    b->children[b->children_size++] = c;
    f.child_count++;
    return true;
}

}  // namespace detail

inline bool jsonb_builder_init(JsonbBuilder* b, Arena* arena, int32_t initial_cap) noexcept {
    assert(b != nullptr);
    assert(arena != nullptr);
    if (!b || !arena) return false;
    if (initial_cap < 64) initial_cap = 64;
    std::memset(b, 0, sizeof(*b));
    b->arena = arena;
    b->frame_top = -1;
    b->root_child_idx = -1;
    b->ok = true;
    b->scratch = static_cast<uint8_t*>(
        arena->allocate(static_cast<size_t>(initial_cap), alignof(uint8_t)));
    if (!b->scratch) { b->ok = false; return false; }
    b->scratch_cap = initial_cap;
    b->children = static_cast<JsonbChild*>(
        arena->allocate(static_cast<size_t>(16) * sizeof(JsonbChild), alignof(JsonbChild)));
    if (!b->children) { b->ok = false; return false; }
    b->children_cap = 16;
    return true;
}

inline bool jsonb_begin_object(JsonbBuilder* b) noexcept {
    assert(b != nullptr);
    if (!b || !b->ok || b->finished) return false;
    if (b->frame_top + 1 >= kMaxDepth) { b->ok = false; return false; }
    b->frame_top++;
    JsonbFrame& f = b->frames[b->frame_top];
    f.is_object = true;
    f.child_first = -1;
    f.child_count = 0;
    f.awaiting_value = false;
    f.pending_key_off = -1;
    f.pending_key_len = 0;
    return true;
}

inline bool jsonb_begin_array(JsonbBuilder* b) noexcept {
    assert(b != nullptr);
    if (!b || !b->ok || b->finished) return false;
    if (b->frame_top + 1 >= kMaxDepth) { b->ok = false; return false; }
    b->frame_top++;
    JsonbFrame& f = b->frames[b->frame_top];
    f.is_object = false;
    f.child_first = -1;
    f.child_count = 0;
    f.awaiting_value = false;
    return true;
}

// Forward decl for end_*: defined further down because they materialise a
// nested container's bytes into scratch using an emit-style helper.
inline bool jsonb_end_object(JsonbBuilder* b) noexcept;
inline bool jsonb_end_array(JsonbBuilder* b) noexcept;

inline bool jsonb_emit_key(JsonbBuilder* b, const char* k, int32_t klen) noexcept {
    assert(b != nullptr);
    assert(k != nullptr || klen == 0);
    if (!b || !b->ok || b->finished) return false;
    if (b->frame_top < 0) { b->ok = false; return false; }
    JsonbFrame& f = b->frames[b->frame_top];
    if (!f.is_object) { b->ok = false; return false; }
    if (f.awaiting_value) { b->ok = false; return false; }
    if (klen < 0 || klen > kMaxBytes) { b->ok = false; return false; }
    const int32_t off = detail::scratch_append(b, k, klen);
    if (off < 0) return false;
    f.pending_key_off = off;
    f.pending_key_len = klen;
    f.awaiting_value = true;
    return true;
}

inline bool jsonb_emit_string(JsonbBuilder* b, const char* s, int32_t slen) noexcept {
    assert(b != nullptr);
    assert(s != nullptr || slen == 0);
    if (!b || !b->ok || b->finished) return false;
    if (slen < 0 || slen > kMaxBytes) { b->ok = false; return false; }
    const int32_t off = detail::scratch_append(b, s, slen);
    if (off < 0) return false;
    JsonbChild c{};
    c.tag = JsonbTag::String;
    c.payload_off = off;
    c.payload_len = slen;
    c.key_off = -1; c.key_len = 0;
    c.nested_off = -1; c.nested_len = 0;
    return detail::push_child(b, c);
}

inline bool jsonb_emit_int64(JsonbBuilder* b, int64_t v) noexcept {
    assert(b != nullptr);
    if (!b || !b->ok || b->finished) return false;
    const int32_t off = detail::scratch_append(b, &v, 8);
    if (off < 0) return false;
    JsonbChild c{};
    c.tag = JsonbTag::Int64;
    c.payload_off = off; c.payload_len = 8;
    c.key_off = -1; c.key_len = 0;
    c.nested_off = -1; c.nested_len = 0;
    return detail::push_child(b, c);
}

inline bool jsonb_emit_float64(JsonbBuilder* b, double v) noexcept {
    assert(b != nullptr);
    if (!b || !b->ok || b->finished) return false;
    const int32_t off = detail::scratch_append(b, &v, 8);
    if (off < 0) return false;
    JsonbChild c{};
    c.tag = JsonbTag::Float64;
    c.payload_off = off; c.payload_len = 8;
    c.key_off = -1; c.key_len = 0;
    c.nested_off = -1; c.nested_len = 0;
    return detail::push_child(b, c);
}

inline bool jsonb_emit_bool(JsonbBuilder* b, bool v) noexcept {
    assert(b != nullptr);
    if (!b || !b->ok || b->finished) return false;
    JsonbChild c{};
    c.tag = v ? JsonbTag::BoolTrue : JsonbTag::BoolFalse;
    c.payload_off = b->scratch_size;
    c.payload_len = 0;
    c.key_off = -1; c.key_len = 0;
    c.nested_off = -1; c.nested_len = 0;
    return detail::push_child(b, c);
}

inline bool jsonb_emit_null(JsonbBuilder* b) noexcept {
    assert(b != nullptr);
    if (!b || !b->ok || b->finished) return false;
    JsonbChild c{};
    c.tag = JsonbTag::Null;
    c.payload_off = b->scratch_size;
    c.payload_len = 0;
    c.key_off = -1; c.key_len = 0;
    c.nested_off = -1; c.nested_len = 0;
    return detail::push_child(b, c);
}

namespace detail {

// Sort the children [first .. first+count) of an object frame by (klen, lex).
// Insertion sort is fine — keys per object are usually small; for large
// objects the cost is still O(n^2 * klen) which beats heap-sort's overhead at
// n<256, and we cap depth at 64 so the nested cost is bounded.
inline void sort_object_children(JsonbBuilder* b, int32_t first, int32_t count) noexcept {
    JsonbChild* arr = b->children + first;
    const uint8_t* base = b->scratch;
    for (int32_t i = 1; i < count; ++i) {
        JsonbChild key = arr[i];
        int32_t j = i - 1;
        while (j >= 0) {
            const int c = key_cmp(base + arr[j].key_off, arr[j].key_len,
                                  base + key.key_off,    key.key_len);
            if (c <= 0) break;
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
}

// Encode a frame (`is_object` + child run [first..first+count)) into a flat
// byte sequence appended to `b->scratch`. Returns the (offset,len) of the
// container in scratch via out_off / out_len. The container's payload pulls
// from existing scratch slices for keys / scalar values; nested containers
// are recursively self-contained slabs already in scratch.
inline bool encode_frame(JsonbBuilder* b, bool is_object,
                         int32_t first, int32_t count,
                         int32_t* out_off, int32_t* out_len) noexcept {
    if (count < 0 || count > static_cast<int32_t>(kHdrCountMask)) {
        b->ok = false; return false;
    }
    if (is_object) sort_object_children(b, first, count);

    // Compute total size: 4 (header) + jentries + payloads.
    const int32_t je_count = is_object ? count * 2 : count;
    int64_t payload_bytes = 0;
    if (is_object) {
        for (int32_t i = 0; i < count; ++i) {
            payload_bytes += b->children[first + i].key_len;
        }
    }
    for (int32_t i = 0; i < count; ++i) {
        payload_bytes += b->children[first + i].payload_len;
    }
    if (payload_bytes > kMaxBytes) { b->ok = false; return false; }

    // Layout: hdr(4) + keys_bytes(4) + values_bytes(4) + jentries + payload.
    const int32_t header_bytes = 12;
    int32_t keys_total_bytes = 0;
    if (is_object) {
        for (int32_t i = 0; i < count; ++i) {
            keys_total_bytes += b->children[first + i].key_len;
        }
    }
    int32_t values_total_bytes = 0;
    for (int32_t i = 0; i < count; ++i) {
        values_total_bytes += b->children[first + i].payload_len;
    }
    const int32_t total = header_bytes + je_count * 4 + static_cast<int32_t>(payload_bytes);
    if (total > kMaxBytes) { b->ok = false; return false; }
    if (!scratch_reserve(b, total)) return false;

    const int32_t base = b->scratch_size;
    uint8_t* w = b->scratch + base;
    std::memset(w, 0, static_cast<size_t>(header_bytes + je_count * 4));
    int32_t payload_cursor = header_bytes + je_count * 4;

    // Helper: write Jentry into absolute table slot `abs_slot`. The "is
    // offset slot?" decision is based on the table-relative `rel_slot`
    // (which resets at the start of each Jentry table — keys vs values).
    auto write_jentry = [&](int32_t abs_slot, int32_t rel_slot, JsonbTag tag,
                            int32_t cumulative_off, int32_t len) {
        uint32_t je;
        if (rel_slot > 0 && (rel_slot % kStride) == 0) {
            je = je_make_off(tag, static_cast<uint32_t>(cumulative_off));
        } else {
            je = je_make_len(tag, static_cast<uint32_t>(len));
        }
        std::memcpy(w + header_bytes + abs_slot * 4, &je, 4);
    };

    if (is_object) {
        int32_t off = 0;
        for (int32_t i = 0; i < count; ++i) {
            const JsonbChild& c = b->children[first + i];
            write_jentry(/*abs=*/i, /*rel=*/i, JsonbTag::String, off, c.key_len);
            std::memcpy(w + payload_cursor, b->scratch + c.key_off,
                        static_cast<size_t>(c.key_len));
            payload_cursor += c.key_len;
            off += c.key_len;
        }
        int32_t voff = 0;
        for (int32_t i = 0; i < count; ++i) {
            const JsonbChild& c = b->children[first + i];
            write_jentry(/*abs=*/count + i, /*rel=*/i, c.tag, voff, c.payload_len);
            if (c.payload_len > 0) {
                std::memcpy(w + payload_cursor, b->scratch + c.payload_off,
                            static_cast<size_t>(c.payload_len));
                payload_cursor += c.payload_len;
            }
            voff += c.payload_len;
        }
    } else {
        int32_t off = 0;
        for (int32_t i = 0; i < count; ++i) {
            const JsonbChild& c = b->children[first + i];
            write_jentry(/*abs=*/i, /*rel=*/i, c.tag, off, c.payload_len);
            if (c.payload_len > 0) {
                std::memcpy(w + payload_cursor, b->scratch + c.payload_off,
                            static_cast<size_t>(c.payload_len));
                payload_cursor += c.payload_len;
            }
            off += c.payload_len;
        }
    }

    uint32_t hdr = static_cast<uint32_t>(count)
                 | (is_object ? kHdrFlagObject : kHdrFlagArray);
    std::memcpy(w + 0, &hdr, 4);
    uint32_t kb = static_cast<uint32_t>(keys_total_bytes);
    uint32_t vb = static_cast<uint32_t>(values_total_bytes);
    std::memcpy(w + 4, &kb, 4);
    std::memcpy(w + 8, &vb, 4);

    if (payload_cursor != total) { b->ok = false; return false; }
    b->scratch_size += total;
    *out_off = base;
    *out_len = total;
    return true;
}

}  // namespace detail

inline bool jsonb_end_object(JsonbBuilder* b) noexcept {
    assert(b != nullptr);
    if (!b || !b->ok || b->finished) return false;
    if (b->frame_top < 0) { b->ok = false; return false; }
    JsonbFrame& f = b->frames[b->frame_top];
    if (!f.is_object || f.awaiting_value) { b->ok = false; return false; }
    int32_t off = 0, len = 0;
    if (!detail::encode_frame(b, true,
                              f.child_first < 0 ? 0 : f.child_first,
                              f.child_count, &off, &len)) return false;
    b->frame_top--;
    JsonbChild c{};
    c.tag = JsonbTag::Object;
    c.payload_off = off;
    c.payload_len = len;
    c.key_off = -1; c.key_len = 0;
    c.nested_off = off; c.nested_len = len;
    return detail::push_child(b, c);
}

inline bool jsonb_end_array(JsonbBuilder* b) noexcept {
    assert(b != nullptr);
    if (!b || !b->ok || b->finished) return false;
    if (b->frame_top < 0) { b->ok = false; return false; }
    JsonbFrame& f = b->frames[b->frame_top];
    if (f.is_object || f.awaiting_value) { b->ok = false; return false; }
    int32_t off = 0, len = 0;
    if (!detail::encode_frame(b, false,
                              f.child_first < 0 ? 0 : f.child_first,
                              f.child_count, &off, &len)) return false;
    b->frame_top--;
    JsonbChild c{};
    c.tag = JsonbTag::Array;
    c.payload_off = off;
    c.payload_len = len;
    c.key_off = -1; c.key_len = 0;
    c.nested_off = off; c.nested_len = len;
    return detail::push_child(b, c);
}

// Finalise. After this call, `out` references arena-resident bytes valid for
// the lifetime of the arena. The builder is exhausted.
inline bool jsonb_finish(JsonbBuilder* b, JsonbView* out) noexcept {
    assert(b != nullptr);
    assert(out != nullptr);
    if (!b || !out) return false;
    if (b->finished) return false;
    if (!b->ok) return false;
    if (b->frame_top != -1) return false;       // unterminated container
    if (!b->root_emitted) return false;
    b->finished = true;

    if (b->root_child_idx < 0 || b->root_child_idx >= b->children_size) return false;
    JsonbChild root = b->children[b->root_child_idx];

    if (root.tag == JsonbTag::Object || root.tag == JsonbTag::Array) {
        // The container's bytes are already a self-contained slab in scratch.
        const int32_t total = root.nested_len;
        uint8_t* dst = static_cast<uint8_t*>(
            b->arena->allocate(static_cast<size_t>(total), alignof(uint8_t)));
        if (!dst) return false;
        std::memcpy(dst, b->scratch + root.nested_off, static_cast<size_t>(total));
        out->data = dst;
        out->bytes = total;
        return true;
    }

    // Root scalar — wrap as 1-element scalar container with full header.
    const int32_t total = 12 + 4 + root.payload_len;
    uint8_t* dst = static_cast<uint8_t*>(
        b->arena->allocate(static_cast<size_t>(total), alignof(uint8_t)));
    if (!dst) return false;
    uint32_t hdr = 1u | kHdrFlagArray | kHdrFlagScalar;
    uint32_t kb = 0;
    uint32_t vb = static_cast<uint32_t>(root.payload_len);
    std::memcpy(dst + 0, &hdr, 4);
    std::memcpy(dst + 4, &kb, 4);
    std::memcpy(dst + 8, &vb, 4);
    uint32_t je = je_make_len(root.tag, static_cast<uint32_t>(root.payload_len));
    std::memcpy(dst + 12, &je, 4);
    if (root.payload_len > 0) {
        std::memcpy(dst + 16, b->scratch + root.payload_off,
                    static_cast<size_t>(root.payload_len));
    }
    out->data = dst;
    out->bytes = total;
    return true;
}

}  // namespace bolt::doc
