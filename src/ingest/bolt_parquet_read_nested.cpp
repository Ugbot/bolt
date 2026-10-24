// bolt_parquet_read_nested.cpp — Dremel record assembly of one top-level
// field into a Nested BoltColumn tree. See bolt_parquet_nested.h.
//
// Every field is assembled from the level stream of the FIRST leaf under it
// (all leaves of a subtree agree on its instances), so a field never needs
// more than one pass over one leaf's (rep, def) arrays. Children are built
// from their own first leaf and must come out at the parent's instance
// count -- a disagreement means the file or the tree lies, and is refused.

#include "bolt/ingest/bolt_parquet_nested.h"

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt_parquet_read_internal.h"

namespace bolt {
namespace ingest {
namespace parquet {

namespace {

struct NestCtx {
    const PqSchemaTree* t;
    const PqLeafSlots*  leaves;   // [col - col_base]
    uint32_t            col_base;
    uint32_t            n_leaves;
    Arena*              arena;
};

const PqLeafSlots* rep_leaf(const NestCtx* x, const PqField& f) noexcept {
    assert(x != nullptr);
    if (f.col_begin < x->col_base || f.col_begin >= f.col_end) return nullptr;
    const uint32_t k = f.col_begin - x->col_base;
    return k < x->n_leaves ? &x->leaves[k] : nullptr;
}

inline bool is_inst(const PqLeafSlots* s, int64_t j, uint32_t d,
                    uint32_t r) noexcept {
    return s->def[j] >= d && s->rep[j] <= r;
}

// Instances of a field and a validity bitmap (nullptr when none is null).
bool field_validity(const NestCtx* x, const PqField& f, const PqLeafSlots* s,
                    int64_t* n, uint8_t** validity) noexcept {
    assert(x != nullptr && s != nullptr);
    assert(n != nullptr && validity != nullptr);
    int64_t cnt = 0, nulls = 0;
    for (int64_t j = 0; j < s->n_slots; ++j) {
        if (!is_inst(s, j, f.inst_def, f.inst_rep)) continue;
        cnt += 1;
        nulls += s->def[j] < f.valid_def ? 1 : 0;
    }
    *n = cnt;
    *validity = nullptr;
    if (nulls == 0) return true;
    const uint64_t nb = (static_cast<uint64_t>(cnt) + 7u) / 8u;
    auto* bm = static_cast<uint8_t*>(x->arena->allocate(nb, 1));
    if (bm == nullptr) return false;
    std::memset(bm, 0xFF, nb);
    int64_t i = 0;
    for (int64_t j = 0; j < s->n_slots; ++j) {
        if (!is_inst(s, j, f.inst_def, f.inst_rep)) continue;
        if (s->def[j] < f.valid_def) {
            bm[i >> 3] = static_cast<uint8_t>(bm[i >> 3] & ~(1u << (i & 7)));
        }
        ++i;
    }
    assert(i == cnt);
    *validity = bm;
    return true;
}

bool build(const NestCtx* x, int32_t fi, uint32_t depth, BoltColumn* out) noexcept;

bool build_leaf(const NestCtx* x, const PqField& f, BoltColumn* out) noexcept {
    assert(x != nullptr && out != nullptr);
    const PqLeafSlots* s = rep_leaf(x, f);
    if (s == nullptr || f.column != static_cast<int32_t>(f.col_begin)) return false;
    int64_t n = 0;
    for (int64_t j = 0; j < s->n_slots; ++j) n += is_inst(s, j, f.inst_def, f.inst_rep);
    if (n != s->n_values || s->values.length != n) return false;
    *out = s->values;
    return true;
}

bool build_struct(const NestCtx* x, const PqField& f, uint32_t depth,
                  BoltColumn* out) noexcept {
    assert(x != nullptr && out != nullptr);
    const PqLeafSlots* s = rep_leaf(x, f);
    int64_t n = 0;
    uint8_t* validity = nullptr;
    if (s == nullptr || f.n_children == 0u ||
        !field_validity(x, f, s, &n, &validity)) return false;
    auto* kids = x->arena->allocate_array<BoltColumn>(f.n_children);
    if (kids == nullptr) return false;
    int32_t c = f.first_child;
    for (uint32_t k = 0; k < f.n_children; ++k) {
        if (c < 0 || !build(x, c, depth + 1u, &kids[k])) return false;
        if (kids[k].length != n) return false;
        c = x->t->fields[c].next_sibling;
    }
    *out = BoltColumn::make_struct(kids, f.n_children, n, validity, x->arena);
    return out->data != nullptr;
}

// Offsets of a List/Map: one container per instance, one element per slot
// that reaches the repeated node without repeating deeper.
bool list_offsets(const NestCtx* x, const PqField& f, const PqLeafSlots* s,
                  int64_t n, int32_t** offs, int64_t* n_elems) noexcept {
    assert(x != nullptr && s != nullptr);
    assert(offs != nullptr && n_elems != nullptr);
    auto* o = x->arena->allocate_array<int32_t>(static_cast<size_t>(n) + 1u);
    if (o == nullptr) return false;
    int64_t cont = -1, elems = 0;
    for (int64_t j = 0; j < s->n_slots; ++j) {
        if (is_inst(s, j, f.inst_def, f.inst_rep)) {
            if (++cont >= n) return false;
            o[cont] = static_cast<int32_t>(elems);
        }
        if (!is_inst(s, j, f.elem_def, f.elem_rep)) continue;
        if (cont < 0 || ++elems > INT32_MAX) return false;
    }
    if (cont + 1 != n) return false;
    o[n] = static_cast<int32_t>(elems);
    *offs = o;
    *n_elems = elems;
    return true;
}

bool build_list(const NestCtx* x, const PqField& f, uint32_t depth,
                BoltColumn* out) noexcept {
    assert(x != nullptr && out != nullptr);
    const PqLeafSlots* s = rep_leaf(x, f);
    int64_t n = 0, n_elems = 0;
    uint8_t* validity = nullptr;
    int32_t* offs = nullptr;
    if (s == nullptr || f.n_children != 1u || f.first_child < 0 ||
        f.elem_rep != f.inst_rep + 1u || f.elem_def <= f.inst_def ||
        !field_validity(x, f, s, &n, &validity) ||
        !list_offsets(x, f, s, n, &offs, &n_elems)) return false;
    BoltColumn elem;
    if (!build(x, f.first_child, depth + 1u, &elem) || elem.length != n_elems) {
        return false;
    }
    if (f.kind == PqFieldKind::Map &&
        (elem.type != BoltType::Struct || elem.child_count() != 2)) return false;
    *out = BoltColumn::make_list(&elem, offs, n, validity, x->arena);
    if (out->data == nullptr) return false;
    if (f.kind == PqFieldKind::Map) out->type = BoltType::Map;
    return true;
}

bool build(const NestCtx* x, int32_t fi, uint32_t depth, BoltColumn* out) noexcept {
    assert(x != nullptr && out != nullptr);
    if (fi < 0 || static_cast<uint32_t>(fi) >= x->t->n_fields ||
        depth > 2u * kPqMaxNestDepth + 2u) return false;
    const PqField& f = x->t->fields[fi];
    switch (f.kind) {
        case PqFieldKind::Leaf:   return build_leaf(x, f, out);
        case PqFieldKind::Struct: return build_struct(x, f, depth, out);
        case PqFieldKind::List:
        case PqFieldKind::Map:    return build_list(x, f, depth, out);
    }
    return false;
}

// The tree must describe this footer: same leaves at the same levels.
bool tree_matches(const PqMeta* meta, const PqSchemaTree* t,
                  const PqField& f) noexcept {
    assert(meta != nullptr && t != nullptr);
    if (t->n_columns != meta->n_columns || f.col_end > meta->n_columns ||
        f.col_begin >= f.col_end) return false;
    for (uint32_t i = 0; i < t->n_nodes; ++i) {
        const PqSchemaNode& nd = t->nodes[i];
        if (nd.column < 0 || static_cast<uint32_t>(nd.column) < f.col_begin ||
            static_cast<uint32_t>(nd.column) >= f.col_end) continue;
        const PqColumn& c = meta->columns[nd.column];
        if (c.max_def != nd.def || c.max_rep != nd.rep) return false;
    }
    return true;
}

}  // namespace

bool parquet_read_nested_column_tree(const uint8_t* buf, uint64_t len,
                                     const PqMeta* meta,
                                     const PqSchemaTree* tree,
                                     uint32_t row_group, uint32_t top_field,
                                     Arena* arena, BoltColumn* out,
                                     int64_t* out_rows) noexcept {
    if (buf == nullptr || meta == nullptr || tree == nullptr || arena == nullptr ||
        out == nullptr || out_rows == nullptr) return false;
    if (row_group >= meta->n_row_groups || top_field >= tree->n_top) return false;
    const PqField& f = tree->fields[tree->top[top_field]];
    if (!tree_matches(meta, tree, f)) return false;
    const int64_t rows = meta->row_groups[row_group].num_rows;
    assert(f.col_end <= meta->n_columns);
    if (f.kind == PqFieldKind::Leaf && meta->columns[f.column].max_def <= 1u) {
        const uint16_t c = static_cast<uint16_t>(f.column);
        return parquet_read_row_group_cols(buf, len, meta, row_group, &c, 1,
                                           arena, out, out_rows);
    }
    const uint32_t n = f.col_end - f.col_begin;
    auto* leaves = arena->allocate_array<PqLeafSlots>(n);
    if (leaves == nullptr) return false;
    for (uint32_t k = 0; k < n; ++k) {
        if (!pq_decode_leaf_slots(buf, len, meta, row_group, row_group + 1u,
                                  f.col_begin + k, arena, &leaves[k]) ||
            leaves[k].n_rows != rows) return false;
    }
    const NestCtx x{tree, leaves, f.col_begin, n, arena};
    BoltColumn col;
    if (!build(&x, tree->top[top_field], 0u, &col) || col.length != rows) {
        return false;
    }
    *out = col;
    *out_rows = rows;
    return true;
}

bool parquet_read_nested_column(const uint8_t* buf, uint64_t len,
                                const PqMeta* meta, uint32_t row_group,
                                uint32_t top_field, Arena* arena,
                                BoltColumn* out, int64_t* out_rows) noexcept {
    if (arena == nullptr || meta == nullptr) return false;
    PqSchemaTree tree;
    if (!parquet_read_schema_tree(buf, len, arena, &tree)) return false;
    assert(tree.n_top > 0);
    return parquet_read_nested_column_tree(buf, len, meta, &tree, row_group,
                                           top_field, arena, out, out_rows);
}

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
