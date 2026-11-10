// bolt/bolt_nested_value.h — navigate and build ONE nested value in the bolt_nested_row encoding, without decoding the
// row into columns. Element access, struct field access, list iteration, size, leaf comparison and value building for
// expression evaluators (element_at, getField, array_contains, sort_array, array(...), struct(...), split, explode).
//
// The encoded bytes of a subtree are a valid standalone value of that subtree's type: a list element or a struct field
// is handed out as a span into the parent, with its nullable flag, so it can be stored as a value of the child type.
//
// RULES: No exceptions, no heap, all fns noexcept, bounded loops (depth <= kMaxDepth, counts <= kMaxCount).
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_nested_row.h"

namespace bolt {
namespace nested_row {

// A borrowed type tree: nodes[0] is the root. Lets a caller keep only the
// nodes it uses (a plan blob) instead of a full kMaxNodes TypeTree.
struct TreeView {
    const TypeNode* nodes;
    uint32_t        n;
};
inline TreeView view(const TypeTree& t) noexcept { return TreeView{t.nodes, t.n}; }

struct Span {
    const uint8_t* p;
    uint32_t       len;
};

// A leaf value: is_null, or (bytes, len) of its payload (fixed width, or the var bytes without the u32 length).
struct LeafView {
    bool           is_null;
    const uint8_t* bytes;
    uint32_t       len;
};

namespace value_detail {
inline bool skip(TreeView t, uint32_t ni, detail::Reader* r, uint32_t depth) noexcept {
    assert(r != nullptr);
    if (depth > kMaxDepth || ni >= t.n) return false;
    const TypeNode& nd = t.nodes[ni];
    if (nd.nullable != 0u) {
        uint8_t f = 0;
        if (!r->take(&f, 1u) || f > 1u) return false;
        if (f == 0u) return true;
    }
    if (nd.type == BoltType::Struct) {
        for (uint32_t k = 0; k < nd.n_children; ++k) if (!skip(t, nd.first_child + k, r, depth + 1u)) return false;
        return true;
    }
    if (nd.type == BoltType::List || nd.type == BoltType::Map) {
        uint32_t cnt = 0;
        if (!r->take(&cnt, 4u) || cnt > kMaxCount) return false;
        for (uint32_t e = 0; e < cnt; ++e) if (!skip(t, nd.first_child, r, depth + 1u)) return false;
        return true;
    }
    if (nd.fixed_bytes == 0u) {
        uint32_t len = 0;
        return r->take(&len, 4u) && r->take(nullptr, len);
    }
    return r->take(nullptr, nd.fixed_bytes);
}

// Reads node ni's null flag: 1 present, 0 null, -1 malformed.
inline int open(const TypeNode& nd, detail::Reader* r) noexcept {
    if (nd.nullable == 0u) return 1;
    uint8_t f = 0;
    if (!r->take(&f, 1u) || f > 1u) return -1;
    return f;
}
}  // namespace value_detail

// Span of one value of node `ni` at the front of `in`.
inline bool value_span(TreeView t, uint32_t ni, const uint8_t* p, uint32_t len, Span* out) noexcept {
    assert(out != nullptr);
    detail::Reader r{p, p + len};
    if (!value_detail::skip(t, ni, &r, 0u)) return false;
    *out = Span{p, static_cast<uint32_t>(r.p - p)};
    return true;
}

// A leaf of node `ni` from its standalone encoding.
inline bool leaf_view(TreeView t, uint32_t ni, Span v, LeafView* out) noexcept {
    assert(out != nullptr && ni < t.n);
    const TypeNode& nd = t.nodes[ni];
    if (is_group(nd.type)) return false;
    detail::Reader r{v.p, v.p + v.len};
    const int o = value_detail::open(nd, &r);
    if (o < 0) return false;
    *out = LeafView{o == 0, nullptr, 0};
    if (o == 0) return true;
    uint32_t n = nd.fixed_bytes;
    if (n == 0u && !r.take(&n, 4u)) return false;
    if (static_cast<size_t>(r.end - r.p) < n) return false;
    out->bytes = r.p;
    out->len = n;
    return true;
}

// Iterates the elements of a LIST (or the entries of a MAP) value of node `ni`.
struct ListCursor {
    const TypeNode* nodes;
    uint32_t        n_nodes;
    uint32_t        elem;
    detail::Reader  r;
    uint32_t        remaining;
    uint32_t        count;
};

// 1: open, 0: the list is NULL, -1: malformed.
inline int list_open(TreeView t, uint32_t ni, Span v, ListCursor* c) noexcept {
    assert(c != nullptr && ni < t.n);
    const TypeNode& nd = t.nodes[ni];
    if (nd.type != BoltType::List && nd.type != BoltType::Map) return -1;
    *c = ListCursor{t.nodes, t.n, nd.first_child, detail::Reader{v.p, v.p + v.len}, 0u, 0u};
    const int o = value_detail::open(nd, &c->r);
    if (o <= 0) return o;
    uint32_t cnt = 0;
    if (!c->r.take(&cnt, 4u) || cnt > kMaxCount) return -1;
    c->remaining = cnt;
    c->count = cnt;
    return 1;
}

inline bool list_next(ListCursor* c, Span* elem) noexcept {
    assert(c != nullptr && elem != nullptr);
    if (c->remaining == 0u) return false;
    const uint8_t* at = c->r.p;
    if (!value_detail::skip(TreeView{c->nodes, c->n_nodes}, c->elem, &c->r, 1u)) { c->remaining = 0; return false; }
    *elem = Span{at, static_cast<uint32_t>(c->r.p - at)};
    --c->remaining;
    return true;
}

// Field k of a STRUCT value of node `ni`. 1: found, 0: the struct is NULL, -1: malformed / out of range.
inline int struct_field(TreeView t, uint32_t ni, Span v, uint32_t k, Span* out) noexcept {
    assert(out != nullptr && ni < t.n);
    const TypeNode& nd = t.nodes[ni];
    if (nd.type != BoltType::Struct || k >= nd.n_children) return -1;
    detail::Reader r{v.p, v.p + v.len};
    const int o = value_detail::open(nd, &r);
    if (o <= 0) return o;
    for (uint32_t f = 0; f < k; ++f) if (!value_detail::skip(t, nd.first_child + f, &r, 1u)) return -1;
    const uint8_t* at = r.p;
    if (!value_detail::skip(t, nd.first_child + k, &r, 1u)) return -1;
    *out = Span{at, static_cast<uint32_t>(r.p - at)};
    return 1;
}

// Orders two present leaves of one type: <0, 0, >0. Floats order NaN above every value (Spark / SQL order).
inline int leaf_compare(const TypeNode& nd, const LeafView& a, const LeafView& b) noexcept {
    assert(!a.is_null && !b.is_null);
    switch (nd.type) {
        case BoltType::Int8: case BoltType::Int16: case BoltType::Int32: case BoltType::Int64:
        case BoltType::Date32: case BoltType::Timestamp: case BoltType::Duration: {
            int64_t x = 0, y = 0;
            if (a.len > 8 || b.len > 8) return 0;
            std::memcpy(&x, a.bytes, a.len); std::memcpy(&y, b.bytes, b.len);
            if (a.len < 8) { const int s = 64 - 8 * static_cast<int>(a.len); x = (x << s) >> s; y = (y << s) >> s; }
            return x < y ? -1 : x > y ? 1 : 0;
        }
        case BoltType::Float32: case BoltType::Float64: {
            double x = 0, y = 0;
            if (nd.type == BoltType::Float32) { float f, g; std::memcpy(&f, a.bytes, 4); std::memcpy(&g, b.bytes, 4); x = f; y = g; }
            else { std::memcpy(&x, a.bytes, 8); std::memcpy(&y, b.bytes, 8); }
            const bool xn = x != x, yn = y != y;
            if (xn || yn) return xn == yn ? 0 : xn ? 1 : -1;
            return x < y ? -1 : x > y ? 1 : 0;
        }
        default: {
            const uint32_t m = a.len < b.len ? a.len : b.len;
            const int c = m ? std::memcmp(a.bytes, b.bytes, m) : 0;
            if (c != 0) return c;
            return a.len < b.len ? -1 : a.len > b.len ? 1 : 0;
        }
    }
}

// ---- build ----------------------------------------------------------------------------------------------------------------
// Appends parts of one value to a Writer (which counts past its capacity; the caller sizes and retries).

inline void put_present(Writer* w) noexcept { w->u8(1u); }
inline void put_null(Writer* w) noexcept { w->u8(0u); }
inline void put_var(Writer* w, const void* bytes, uint32_t len) noexcept { w->u32(len); w->put(bytes, len); }
inline void put_fixed(Writer* w, const void* bytes, uint32_t width) noexcept { w->put(bytes, width); }
// A list / map count placeholder; patch it with patch_count once the elements are written.
inline size_t put_count(Writer* w) noexcept { const size_t at = w->n; w->u32(0u); return at; }
inline void patch_count(Writer* w, size_t at, uint32_t count) noexcept {
    assert(at + 4u <= w->n);
    if (w->p != nullptr && at + 4u <= w->cap) std::memcpy(w->p + at, &count, 4u);
}

}  // namespace nested_row
}  // namespace bolt
