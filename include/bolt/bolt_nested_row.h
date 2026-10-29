// bolt/bolt_nested_row.h — one row of a Nested column (STRUCT / LIST / MAP, any depth) as bytes, and back.
//
// A storage engine keeps a nested value per row while it is buffered or merged (memtable, compaction, a query's sort key);
// the columnar form is rebuilt for a consumer. The encoding is deterministic, so equal values have equal bytes:
//   nullable node: 1 byte, 0 = NULL (nothing follows) or 1
//   fixed leaf:    its value bytes (little-endian, the type's width)
//   var leaf:      u32 length + bytes
//   struct:        each field in order
//   list / map:    u32 count + that many elements (a map element is its {key, value} struct)
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <new>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_arena_vec.h"
#include "bolt/bolt_column.h"

namespace bolt {
namespace nested_row {

inline constexpr uint32_t kMaxNodes = 256;
inline constexpr uint32_t kMaxDepth = 32;
inline constexpr uint32_t kMaxCount = 1u << 30;     // elements of one list / map value

// A type tree: node 0 is the column; a node's children are nodes[first_child .. first_child + n_children).
struct TypeNode {
    BoltType    type;
    BoltLogical logical;
    uint8_t     nullable;
    uint8_t     _pad;
    uint16_t    first_child;
    uint16_t    n_children;
    uint16_t    fixed_bytes;     // a fixed leaf's value width, 0 for a var leaf / group
    uint8_t     fixed_width;     // FixedSizeBinary's true width (BoltColumn::fixed_width)
    uint8_t     scale;
};
struct TypeTree {
    TypeNode nodes[kMaxNodes];
    uint32_t n = 0;
};

inline bool is_group(BoltType t) noexcept { return t == BoltType::Struct || t == BoltType::List || t == BoltType::Map; }
inline bool is_var_leaf(const BoltColumn& c) noexcept {
    return c.format == ColumnFormat::VarBinary || c.type == BoltType::Utf8 || c.type == BoltType::Binary;
}

namespace detail {
inline bool tree_add(const BoltColumn& c, TypeTree* t, uint32_t idx, uint32_t depth) noexcept {
    assert(t != nullptr && idx < t->n);
    if (depth > kMaxDepth) return false;
    TypeNode& nd = t->nodes[idx];
    nd = TypeNode{};
    nd.type = c.type;
    nd.logical = c.logical;
    nd.scale = c.decimal_scale;
    nd.nullable = c.validity != nullptr ? 1u : 0u;
    if (!c.is_nested()) {
        nd.fixed_bytes = is_var_leaf(c) ? 0u : c.type_size_bytes;
        nd.fixed_width = c.fixed_width;
        return is_var_leaf(c) || c.type_size_bytes > 0u;
    }
    const uint32_t nk = static_cast<uint32_t>(c.child_count());
    if (nk == 0u || t->n + nk > kMaxNodes) return false;
    nd.first_child = static_cast<uint16_t>(t->n);
    nd.n_children = static_cast<uint16_t>(nk);
    t->n += nk;
    for (uint32_t k = 0; k < nk; ++k) {                              // bounded: kMaxNodes
        if (!tree_add(*c.child_at(k), t, nd.first_child + k, depth + 1u)) return false;
    }
    return true;
}
}  // namespace detail

// The tree of `c` (every node nullable when it carries a validity bitmap; mark more nullable by hand if a column may later
// carry nulls). False past kMaxNodes / kMaxDepth or on a leaf of no known width.
inline bool tree_from_column(const BoltColumn& c, TypeTree* out) noexcept {
    assert(out != nullptr);
    out->n = 1u;
    return detail::tree_add(c, out, 0u, 0u);
}

// ---- encode -----------------------------------------------------------------------------------------------------------------

struct Writer {
    uint8_t* p;
    size_t   cap;
    size_t   n;
    bool failed = false;
    void put(const void* src, size_t len) noexcept {
        if (len == 0) return;
        if (len > SIZE_MAX - n) { failed = true; return; }
        if (n + len <= cap && p != nullptr) std::memcpy(p + n, src, len);
        n += len;                                                   // counts past `cap`: the caller sizes and retries
    }
    void u8(uint8_t v) noexcept { put(&v, 1u); }
    void u32(uint32_t v) noexcept { put(&v, 4u); }
};

namespace detail {
inline bool enc(const TypeTree& t, uint32_t ni, const BoltColumn& c, int64_t i, uint32_t depth, Writer* w) noexcept {
    assert(ni < t.n && w != nullptr);
    if (depth > kMaxDepth || i < 0 || i >= c.length) return false;
    const TypeNode& nd = t.nodes[ni];
    if (nd.nullable != 0u) {
        const bool present = !c.is_null(i);
        w->u8(present ? 1u : 0u);
        if (!present) return true;
    } else if (c.is_null(i)) {
        return false;                                               // a NULL where the tree says required
    }
    if (nd.type == BoltType::Struct) {
        if (!c.is_nested() || c.child_count() != nd.n_children) return false;
        for (uint32_t k = 0; k < nd.n_children; ++k) {               // bounded: kMaxNodes
            if (!enc(t, nd.first_child + k, *c.child_at(k), i, depth + 1u, w)) return false;
        }
        return true;
    }
    if (nd.type == BoltType::List || nd.type == BoltType::Map) {
        const int32_t* offs = c.list_offsets();
        if (offs == nullptr || c.list_element() == nullptr || nd.n_children != 1u) return false;
        const int32_t lo = offs[i], hi = offs[i + 1];
        if (hi < lo || static_cast<uint32_t>(hi - lo) > kMaxCount) return false;
        w->u32(static_cast<uint32_t>(hi - lo));
        for (int32_t e = lo; e < hi; ++e) {                          // bounded: kMaxCount
            if (!enc(t, nd.first_child, *c.list_element(), e, depth + 1u, w)) return false;
        }
        return true;
    }
    if (nd.fixed_bytes == 0u) {
        const uint8_t* bp = nullptr;
        int32_t len = 0;
        if (c.format == ColumnFormat::VarBinary) {
            if (c.dict_child == nullptr || c.dict_child->data == nullptr) return false;
            const auto* offsets = static_cast<const int32_t*>(c.dict_child->data);
            if (offsets[i] < 0 || offsets[i+1] < offsets[i]) return false;
            len = offsets[i+1] - offsets[i];
            if (len && c.data == nullptr) return false;
            if (len) bp = static_cast<const uint8_t*>(c.data) + offsets[i];
        } else c.utf8_at_or_empty(i, &bp, &len);
        if (len < 0) return false;
        w->u32(static_cast<uint32_t>(len));
        w->put(bp, static_cast<size_t>(len));
        return true;
    }
    if (c.data == nullptr || c.type_size_bytes != nd.fixed_bytes) return false;
    w->put(static_cast<const uint8_t*>(c.data) + static_cast<size_t>(i) * nd.fixed_bytes, nd.fixed_bytes);
    return true;
}
}  // namespace detail

// Encode row `i` of `c` (whose shape is `t`) into out[0..cap). Returns the bytes the row needs (> cap: nothing usable was
// written, retry with that size), or 0 when the value does not match the tree.
inline size_t encode_row(const TypeTree& t, const BoltColumn& c, int64_t i, uint8_t* out, size_t cap) noexcept {
    assert(t.n > 0u);
    Writer w{out, cap, 0u};
    if (!detail::enc(t, 0u, c, i, 0u, &w) || w.failed) return 0u;
    return w.n == 0u ? 1u : w.n;                                     // a value is never empty; 1 keeps 0 as the error
}

// ---- decode -----------------------------------------------------------------------------------------------------------------

// Per-node output buffers while rows are decoded.
struct NodeBuf {
    ArenaVec<uint8_t> valid;     // one byte per value (packed at finish)
    ArenaVec<uint8_t> bytes;     // fixed values / var payload
    ArenaVec<int32_t> offs;      // var leaf: value offsets; list / map: element offsets (both len + 1 at finish)
    int64_t           len = 0;
};

struct Decoder {
    const TypeTree* t = nullptr;
    Arena*          arena = nullptr;
    NodeBuf*        bufs = nullptr;    // t->n entries
};

inline bool decoder_init(Decoder* d, const TypeTree* t, Arena* arena) noexcept {
    assert(d != nullptr && t != nullptr && arena != nullptr);
    d->t = t;
    d->arena = arena;
    d->bufs = static_cast<NodeBuf*>(arena->allocate(sizeof(NodeBuf) * t->n, alignof(NodeBuf)));
    if (d->bufs == nullptr) return false;
    for (uint32_t k = 0; k < t->n; ++k) {                            // bounded: kMaxNodes
        new (&d->bufs[k]) NodeBuf();
        if (!d->bufs[k].valid.init(arena, nullptr, 0u) || !d->bufs[k].bytes.init(arena, nullptr, 0u) ||
            !d->bufs[k].offs.init(arena, nullptr, 0u) || !d->bufs[k].offs.try_push(0)) return false;
    }
    return true;
}

namespace detail {
struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    bool take(void* dst, size_t len) noexcept {
        if (static_cast<size_t>(end - p) < len) return false;
        if (dst != nullptr) std::memcpy(dst, p, len);
        p += len;
        return true;
    }
};

// A NULL (or the slot under a NULL struct): one value of `ni` and, for a struct, of each field.
inline bool push_null(Decoder* d, uint32_t ni, uint32_t depth) noexcept {
    if (depth > kMaxDepth) return false;
    const TypeNode& nd = d->t->nodes[ni];
    NodeBuf& b = d->bufs[ni];
    if (!b.valid.try_push(0u)) return false;
    ++b.len;
    if (nd.type == BoltType::Struct) {
        for (uint32_t k = 0; k < nd.n_children; ++k) if (!push_null(d, nd.first_child + k, depth + 1u)) return false;
        return true;
    }
    if (nd.type == BoltType::List || nd.type == BoltType::Map) return b.offs.try_push(b.offs.back());
    if (nd.fixed_bytes == 0u) return b.offs.try_push(b.offs.back());
    return b.bytes.resize(b.bytes.size() + nd.fixed_bytes);
}

inline bool dec(Decoder* d, uint32_t ni, Reader* r, uint32_t depth) noexcept {
    if (depth > kMaxDepth) return false;
    const TypeNode& nd = d->t->nodes[ni];
    NodeBuf& b = d->bufs[ni];
    if (nd.nullable != 0u) {
        uint8_t f = 0;
        if (!r->take(&f, 1u) || f > 1u) return false;
        if (f == 0u) return push_null(d, ni, depth);
    }
    if (!b.valid.try_push(1u)) return false;
    ++b.len;
    if (nd.type == BoltType::Struct) {
        for (uint32_t k = 0; k < nd.n_children; ++k) if (!dec(d, nd.first_child + k, r, depth + 1u)) return false;
        return true;
    }
    if (nd.type == BoltType::List || nd.type == BoltType::Map) {
        uint32_t cnt = 0;
        if (!r->take(&cnt, 4u) || cnt > kMaxCount) return false;
        for (uint32_t e = 0; e < cnt; ++e) if (!dec(d, nd.first_child, r, depth + 1u)) return false;
        return b.offs.try_push(static_cast<int32_t>(d->bufs[nd.first_child].len));
    }
    if (nd.fixed_bytes == 0u) {
        uint32_t len = 0;
        if (!r->take(&len, 4u) || static_cast<size_t>(r->end - r->p) < len) return false;
        const size_t at = b.bytes.size();
        if (!b.bytes.resize(at + len)) return false;
        if (!r->take(b.bytes.data() + at, len)) return false;
        return b.offs.try_push(static_cast<int32_t>(b.bytes.size()));
    }
    const size_t at = b.bytes.size();
    return b.bytes.resize(at + nd.fixed_bytes) && r->take(b.bytes.data() + at, nd.fixed_bytes);
}
}  // namespace detail

// Append one encoded row (or a NULL row when `bytes` is nullptr; the root must be nullable then).
inline bool decoder_push(Decoder* d, const uint8_t* bytes, size_t len) noexcept {
    assert(d != nullptr && d->bufs != nullptr);
    if (bytes == nullptr) return d->t->nodes[0].nullable != 0u && detail::push_null(d, 0u, 0u);
    detail::Reader r{bytes, bytes + len};
    return detail::dec(d, 0u, &r, 0u) && r.p == r.end;
}

namespace detail {
inline uint8_t* pack_validity(const NodeBuf& b, bool nullable, Arena* a) noexcept {
    if (!nullable) return nullptr;
    auto* v = static_cast<uint8_t*>(a->allocate_zeroed(static_cast<size_t>((b.len + 7) / 8) + 1u));
    if (v == nullptr) return nullptr;
    for (int64_t i = 0; i < b.len; ++i) if (b.valid[static_cast<size_t>(i)] != 0u) v[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
    return v;
}

inline bool build(const Decoder& d, uint32_t ni, uint32_t depth, BoltColumn* out) noexcept {
    if (depth > kMaxDepth) return false;
    const TypeNode& nd = d.t->nodes[ni];
    const NodeBuf& b = d.bufs[ni];
    uint8_t* valid = pack_validity(b, nd.nullable != 0u, d.arena);
    if (nd.nullable != 0u && valid == nullptr) return false;
    if (nd.type == BoltType::Struct) {
        auto* kids = static_cast<BoltColumn*>(d.arena->allocate(sizeof(BoltColumn) * nd.n_children, alignof(BoltColumn)));
        if (kids == nullptr) return false;
        for (uint32_t k = 0; k < nd.n_children; ++k) if (!build(d, nd.first_child + k, depth + 1u, &kids[k])) return false;
        *out = BoltColumn::make_struct(kids, nd.n_children, b.len, valid, d.arena);
    } else if (nd.type == BoltType::List || nd.type == BoltType::Map) {
        BoltColumn el{};
        if (!build(d, nd.first_child, depth + 1u, &el)) return false;
        *out = BoltColumn::make_list(&el, b.offs.data(), b.len, valid, d.arena);
        out->type = nd.type;
    } else if (nd.fixed_bytes == 0u) {
        *out = BoltColumn::make_var_binary(const_cast<uint8_t*>(b.bytes.data()), valid, b.offs.data(), b.len, nd.type, d.arena);
    } else {
        *out = BoltColumn::make_flat(const_cast<uint8_t*>(b.bytes.data()), valid, b.len, nd.type);
        out->type_size_bytes = nd.fixed_bytes;
        out->fixed_width = nd.fixed_width;
    }
    out->logical = nd.logical;
    out->decimal_scale = nd.scale;
    return out->format != ColumnFormat::Flat || out->data != nullptr || b.len == 0;
}
}  // namespace detail

// The decoded rows as one Nested (or leaf) column.
inline bool decoder_finish(const Decoder& d, BoltColumn* out) noexcept {
    assert(out != nullptr && d.t != nullptr);
    return detail::build(d, 0u, 0u, out);
}

}  // namespace nested_row
}  // namespace bolt
