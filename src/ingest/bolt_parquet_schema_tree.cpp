// bolt_parquet_schema_tree.cpp — the parquet group tree and its Bolt-shaped
// field interpretation. See include/bolt/ingest/bolt_parquet_nested.h.

#include "bolt/ingest/bolt_parquet_nested.h"

#include <cassert>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt_parquet_meta_internal.h"

namespace bolt {
namespace ingest {
namespace parquet {

namespace {

constexpr int32_t kCtMap = 1, kCtMapKeyValue = 2, kCtList = 3;

struct OpenGroup {
    int32_t node;
    int32_t left;        // children still to come
    int32_t last_child;  // -1 until the first child is linked
};

void node_from_elem(const PqRawElem& e, PqSchemaNode* n) noexcept {
    assert(n != nullptr);
    std::memset(n, 0, sizeof(*n));
    std::memcpy(n->name, e.col.name, kPqMaxNameBytes);
    n->name[kPqMaxNameBytes - 1] = '\0';
    n->field_id = e.col.field_id;
    n->has_field_id = e.col.has_field_id;
    n->converted = e.col.converted;
    n->logical = e.col.logical;
    n->repetition = e.repeated ? 2u : (e.col.optional ? 1u : 0u);
    n->parent = -1;
    n->first_child = -1;
    n->next_sibling = -1;
    n->column = -1;
    assert(n->repetition <= 2u);
}

// Link element i under the innermost open group and set its levels.
bool attach_child(PqSchemaTree* t, OpenGroup* stk, uint32_t sp,
                  int32_t i) noexcept {
    assert(t != nullptr && stk != nullptr && sp > 0);
    OpenGroup& g = stk[sp - 1];
    PqSchemaNode& p = t->nodes[g.node];
    PqSchemaNode& n = t->nodes[i];
    if (g.left <= 0) return false;
    n.parent = g.node;
    if (g.last_child < 0) p.first_child = i;
    else t->nodes[g.last_child].next_sibling = i;
    g.last_child = i;
    p.n_children += 1;
    const uint32_t d = p.def + (n.repetition != 0u ? 1u : 0u);
    const uint32_t r = p.rep + (n.repetition == 2u ? 1u : 0u);
    if (d > 255u || r > 255u) return false;
    n.def = static_cast<uint8_t>(d);
    n.rep = static_cast<uint8_t>(r);
    assert(n.def >= p.def && n.rep >= p.rep);
    return true;
}

// Close every group whose children are all seen; records leaf ranges.
void close_groups(PqSchemaTree* t, OpenGroup* stk, uint32_t* sp) noexcept {
    assert(t != nullptr && stk != nullptr && sp != nullptr);
    for (uint32_t k = 0; k <= kPqMaxNestDepth && *sp > 0; ++k) {
        OpenGroup& g = stk[*sp - 1];
        if (--g.left > 0) return;
        t->nodes[g.node].col_end = t->n_columns;
        *sp -= 1;
    }
}

// Walk the DFS-serialized list<SchemaElement> into t->nodes.
bool parse_nodes(TcCursor* c, uint32_t n, PqSchemaTree* t) noexcept {
    assert(c != nullptr && t != nullptr && t->nodes != nullptr);
    OpenGroup stk[kPqMaxNestDepth + 1];
    uint32_t sp = 0;
    for (uint32_t i = 0; i < n; ++i) {
        PqRawElem e;
        if (!pq_parse_schema_elem(c, &e)) return false;
        PqSchemaNode* nd = &t->nodes[i];
        node_from_elem(e, nd);
        nd->col_begin = t->n_columns;
        if (i == 0) {
            if (e.num_children <= 0) return false;
            nd->repetition = 0;
            stk[0] = OpenGroup{0, e.num_children, -1};
            sp = 1;
            continue;
        }
        if (sp == 0 || !attach_child(t, stk, sp, static_cast<int32_t>(i))) {
            return false;
        }
        if (e.num_children > 0) {
            if (sp > kPqMaxNestDepth) return false;
            stk[sp++] = OpenGroup{static_cast<int32_t>(i), e.num_children, -1};
            continue;
        }
        if (e.has_type == 0u) return false;
        nd->column = static_cast<int32_t>(t->n_columns++);
        nd->col_end = t->n_columns;
        close_groups(t, stk, &sp);
    }
    t->n_nodes = n;
    return sp == 0;
}

bool is_list_group(const PqSchemaNode& n) noexcept {
    return n.converted == kCtList ||
           n.logical == static_cast<int32_t>(PqLogical::List);
}

bool is_map_group(const PqSchemaNode& n) noexcept {
    return n.converted == kCtMap || n.converted == kCtMapKeyValue ||
           n.logical == static_cast<int32_t>(PqLogical::Map);
}

struct Builder {
    PqSchemaTree* t;
    uint32_t      cap;
};

int32_t new_field(Builder* b, PqFieldKind kind, int32_t node, uint32_t idef,
                  uint32_t irep, uint32_t vdef) noexcept {
    assert(b != nullptr && node >= 0);
    if (b->t->n_fields >= b->cap) return -1;
    const PqSchemaNode& n = b->t->nodes[node];
    const int32_t k = static_cast<int32_t>(b->t->n_fields++);
    PqField* f = &b->t->fields[k];
    std::memset(f, 0, sizeof(*f));
    std::memcpy(f->name, n.name, kPqMaxNameBytes);
    f->field_id = n.field_id;
    f->has_field_id = n.has_field_id;
    f->node = node;
    f->column = (kind == PqFieldKind::Leaf) ? n.column : -1;
    f->first_child = -1;
    f->next_sibling = -1;
    f->col_begin = n.col_begin;
    f->col_end = n.col_end;
    f->kind = kind;
    f->inst_def = static_cast<uint8_t>(idef);
    f->inst_rep = static_cast<uint8_t>(irep);
    f->valid_def = static_cast<uint8_t>(vdef);
    f->nullable = vdef > idef ? 1u : 0u;
    assert(vdef >= idef);
    return k;
}

void link_child(PqSchemaTree* t, int32_t parent, int32_t* last,
                int32_t child) noexcept {
    assert(t != nullptr && last != nullptr);
    assert(parent >= 0 && child >= 0);
    if (*last < 0) t->fields[parent].first_child = child;
    else t->fields[*last].next_sibling = child;
    *last = child;
    t->fields[parent].n_children += 1;
}

bool derive(Builder* b, int32_t node, uint32_t idef, uint32_t irep,
            uint32_t depth, int32_t* out) noexcept;

// A Struct over `node`'s children at position (idef, irep).
bool derive_struct(Builder* b, int32_t node, uint32_t idef, uint32_t irep,
                   uint32_t depth, int32_t* out) noexcept {
    assert(b != nullptr && out != nullptr);
    const PqSchemaNode& n = b->t->nodes[node];
    const int32_t s = new_field(b, PqFieldKind::Struct, node, idef, irep, n.def);
    if (s < 0 || n.n_children == 0u) return false;
    int32_t last = -1;
    int32_t c = n.first_child;
    for (uint32_t k = 0; k < n.n_children; ++k) {
        if (c < 0) return false;
        int32_t f = -1;
        if (!derive(b, c, idef, irep, depth + 1u, &f)) return false;
        link_child(b->t, s, &last, f);
        c = b->t->nodes[c].next_sibling;
    }
    *out = s;
    return true;
}

// The element (or entry) of a list-shaped field; `rep_node` is the repeated
// node and `elem` the node the element is read from (itself for 2-level).
bool derive_element(Builder* b, int32_t rep_node, int32_t elem,
                    bool elem_is_group_self, uint32_t depth,
                    int32_t* out) noexcept {
    assert(b != nullptr && out != nullptr);
    const PqSchemaNode& r = b->t->nodes[rep_node];
    if (elem != rep_node) return derive(b, elem, r.def, r.rep, depth, out);
    if (elem_is_group_self) return derive_struct(b, rep_node, r.def, r.rep, depth, out);
    const int32_t f = new_field(b, PqFieldKind::Leaf, rep_node, r.def, r.rep, r.def);
    *out = f;
    return f >= 0;
}

// A List or Map field over a repeated node `rep_node`.
bool derive_listlike(Builder* b, PqFieldKind kind, int32_t named,
                     int32_t rep_node, int32_t elem, bool group_self,
                     uint32_t idef, uint32_t irep, uint32_t vdef,
                     uint32_t depth, int32_t* out) noexcept {
    assert(b != nullptr && out != nullptr);
    const PqSchemaNode& r = b->t->nodes[rep_node];
    const int32_t l = new_field(b, kind, named, idef, irep, vdef);
    if (l < 0 || r.repetition != 2u || r.rep != irep + 1u) return false;
    b->t->fields[l].elem_def = r.def;
    b->t->fields[l].elem_rep = r.rep;
    int32_t e = -1;
    if (!derive_element(b, rep_node, elem, group_self, depth + 1u, &e)) {
        return false;
    }
    int32_t last = -1;
    link_child(b->t, l, &last, e);
    *out = l;
    return true;
}

// LIST group: the backward-compatibility rules of LogicalTypes.md.
bool derive_list(Builder* b, int32_t node, uint32_t idef, uint32_t irep,
                 uint32_t depth, int32_t* out) noexcept {
    const PqSchemaNode& x = b->t->nodes[node];
    if (x.n_children != 1u || x.first_child < 0) return false;
    const int32_t ri = x.first_child;
    const PqSchemaNode& r = b->t->nodes[ri];
    if (r.repetition != 2u) return false;
    if (r.column >= 0) {                                   // 2-level leaf
        return derive_listlike(b, PqFieldKind::List, node, ri, ri, false,
                               idef, irep, x.def, depth, out);
    }
    char tuple[kPqMaxNameBytes + 8];
    const size_t xl = std::strlen(x.name);
    std::memcpy(tuple, x.name, xl);
    std::memcpy(tuple + xl, "_tuple", 7);
    const bool self = r.n_children > 1u || std::strcmp(r.name, "array") == 0 ||
                      std::strcmp(r.name, tuple) == 0;
    return derive_listlike(b, PqFieldKind::List, node, ri,
                           self ? ri : r.first_child, self, idef, irep, x.def,
                           depth, out);
}

// MAP / MAP_KEY_VALUE group: one repeated group of {key, value}.
bool derive_map(Builder* b, int32_t node, uint32_t idef, uint32_t irep,
                uint32_t depth, int32_t* out) noexcept {
    const PqSchemaNode& x = b->t->nodes[node];
    if (x.n_children != 1u || x.first_child < 0) return false;
    const int32_t kv = x.first_child;
    const PqSchemaNode& r = b->t->nodes[kv];
    if (r.repetition != 2u || r.column >= 0 || r.n_children != 2u) return false;
    return derive_listlike(b, PqFieldKind::Map, node, kv, kv, true, idef, irep,
                           x.def, depth, out);
}

bool derive(Builder* b, int32_t node, uint32_t idef, uint32_t irep,
            uint32_t depth, int32_t* out) noexcept {
    assert(b != nullptr && out != nullptr);
    if (depth > 2u * kPqMaxNestDepth + 2u || node <= 0 ||
        static_cast<uint32_t>(node) >= b->t->n_nodes) return false;
    const PqSchemaNode& x = b->t->nodes[node];
    const bool group = x.column < 0;
    if (x.repetition == 2u) {        // a repeated field outside any LIST group
        if (x.def == 0u) return false;
        return derive_listlike(b, PqFieldKind::List, node, node, node, group,
                               idef, irep, x.def - 1u, depth, out);
    }
    if (group && is_list_group(x)) return derive_list(b, node, idef, irep, depth, out);
    if (group && is_map_group(x)) return derive_map(b, node, idef, irep, depth, out);
    if (group) return derive_struct(b, node, idef, irep, depth, out);
    const int32_t f = new_field(b, PqFieldKind::Leaf, node, idef, irep, x.def);
    *out = f;
    return f >= 0;
}

bool derive_top(PqSchemaTree* t) noexcept {
    assert(t != nullptr && t->n_nodes > 0);
    Builder b{t, 2u * t->n_nodes};
    const PqSchemaNode& root = t->nodes[0];
    int32_t c = root.first_child;
    for (uint32_t k = 0; k < root.n_children; ++k) {
        if (c < 0) return false;
        int32_t f = -1;
        if (!derive(&b, c, 0u, 0u, 0u, &f)) return false;
        t->top[t->n_top++] = f;
        c = t->nodes[c].next_sibling;
    }
    assert(t->n_top == root.n_children);
    return t->n_top > 0;
}

}  // namespace

bool parquet_schema_tree_slice(const uint8_t* metadata, uint32_t length,
                               Arena* arena, PqSchemaTree* out) noexcept {
    assert(arena != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    if (metadata == nullptr || length == 0) return false;
    TcCursor c{metadata, metadata + length};
    int16_t fid = 0;
    uint8_t ft = 0;
    for (uint32_t guard = 0; guard < 1024u; ++guard) {   // bounded field walk
        if (!tc_field(&c, &fid, &ft)) return false;     // no schema field
        if (fid != 2) {
            if (!tc_skip(&c, ft, 0)) return false;
            continue;
        }
        uint8_t et = 0;
        uint32_t n = 0;
        if (ft != kTcList || !tc_list(&c, &et, &n) || et != kTcStruct ||
            n < 2u || n > kPqMaxSchemaElems) return false;
        out->nodes = arena->allocate_array<PqSchemaNode>(n);
        out->fields = arena->allocate_array<PqField>(2u * n);
        out->top = arena->allocate_array<int32_t>(n);
        if (out->nodes == nullptr || out->fields == nullptr || out->top == nullptr) {
            return false;
        }
        return parse_nodes(&c, n, out) && derive_top(out);
    }
    return false;
}

bool parquet_read_schema_tree(const uint8_t* buf, uint64_t len, Arena* arena,
                              PqSchemaTree* out) noexcept {
    assert(arena != nullptr && out != nullptr);
    uint64_t off = 0;
    uint32_t mlen = 0;
    if (buf == nullptr || !pq_locate_footer(buf, len, &off, &mlen)) return false;
    assert(off + mlen <= len);
    return parquet_schema_tree_slice(buf + off, mlen, arena, out);
}

int32_t pq_field_find_id(const PqSchemaTree* tree, int32_t parent,
                         int32_t field_id) noexcept {
    assert(tree != nullptr);
    if (parent < -1 || parent >= static_cast<int32_t>(tree->n_fields)) return -1;
    if (parent == -1) {
        for (uint32_t k = 0; k < tree->n_top; ++k) {
            const PqField& f = tree->fields[tree->top[k]];
            if (f.has_field_id != 0u && f.field_id == field_id) return tree->top[k];
        }
        return -1;
    }
    int32_t c = tree->fields[parent].first_child;
    for (uint32_t k = 0; k < tree->fields[parent].n_children && c >= 0; ++k) {
        const PqField& f = tree->fields[c];
        if (f.has_field_id != 0u && f.field_id == field_id) return c;
        c = f.next_sibling;
    }
    return -1;
}

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
