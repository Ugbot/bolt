// bolt_parquet_write_nested.inc — STRUCT / LIST / MAP columns declared by a
// ParquetNestedNode tree (ParquetWriteColumn::nested).
//
// Included ONCE from bolt_parquet_write.cpp inside its anonymous namespace,
// after the LIST chunk writer and the schema-element writers. A size seam,
// not a public header.
//
// One logical column expands into one physical chunk per LEAF node. Each
// leaf is shredded independently from the batch column: a walk from the
// column root down the leaf's path emits Dremel rep/def levels per row, the
// present values are gathered into a DENSE flat column, and pages are cut on
// row boundaries over both. Leaves are PLAIN encoded; no dictionary, no
// bloom filter.
//
// Canonical shapes:
//   struct: <opt|req> group name { fields... }
//   list:   <opt|req> group name (LIST) { repeated group list {
//               <opt|req> element } }
//   map:    <opt|req> group name (MAP) { repeated group key_value {
//               required key; <opt|req> value } }

// ===== validation =========================================================

bool nested_is_group(BoltType t) noexcept {
    return t == BoltType::Struct || t == BoltType::List || t == BoltType::Map;
}

bool nested_is_repeated(BoltType t) noexcept {
    return t == BoltType::List || t == BoltType::Map;
}

// A leaf node as the flat column every encoder and the leaf schema-element
// writer already consume.
ParquetWriteColumn nested_leaf_column(const ParquetNestedNode& n,
                                      const char* name) noexcept {
    assert(name != nullptr);
    assert(!nested_is_group(n.type));
    ParquetWriteColumn c{};
    std::strncpy(c.name, name, sizeof(c.name) - 1u);
    c.type = n.type;
    c.precision = n.precision;
    c.scale = n.scale;
    c.nullable = n.nullable;
    c.logical = n.logical;
    c.has_field_id = n.has_field_id;
    c.field_id = n.field_id;
    c.ts_local = n.ts_local;
    c.integer_bits = n.integer_bits;
    c.integer_signed = n.integer_signed;
    c.timestamp_unit = n.timestamp_unit;
    c.time_micros = n.time_micros;
    c.fixed_len = n.fixed_len;
    return c;
}

// The logical-annotation rules parquet_write_open applies to a flat column.
bool nested_leaf_logical_ok(const ParquetWriteColumn& c) noexcept {
    assert(c.nested == nullptr);
    const BoltLogical lg = static_cast<BoltLogical>(c.logical);
    if (c.logical > static_cast<std::uint8_t>(BoltLogical::Unknown)) return false;
    if (lg == BoltLogical::Json && c.type != BoltType::Utf8) return false;
    if (lg == BoltLogical::Bson && c.type != BoltType::Binary &&
        c.type != BoltType::Utf8) {
        return false;
    }
    if (lg == BoltLogical::Enum && c.type != BoltType::Utf8) return false;
    if (lg == BoltLogical::Unknown && !c.nullable) return false;
    return true;
}

bool nested_node_valid(const ParquetNestedNode* t, std::uint32_t n,
                       std::uint32_t i) noexcept {
    assert(t != nullptr && i < n);
    assert(n <= kPwMaxNestedNodes);
    const ParquetNestedNode& nd = t[i];
    if (strnlen(nd.name, sizeof(nd.name)) >= sizeof(nd.name)) return false;
    if (!nested_is_group(nd.type)) {
        if (nd.n_children != 0u) return false;
        const ParquetWriteColumn c = nested_leaf_column(nd, "leaf");
        return type_supported(c) && nested_leaf_logical_ok(c);
    }
    if (nd.logical != 0u || nd.n_children == 0u) return false;
    if (nd.type == BoltType::List && nd.n_children != 1u) return false;
    if (nd.type == BoltType::Map && nd.n_children != 2u) return false;
    // Children strictly after their parent: no cycles, single forward pass.
    if (nd.first_child <= i) return false;
    if (static_cast<std::uint32_t>(nd.first_child) + nd.n_children > n) return false;
    if (nd.type == BoltType::Map && t[nd.first_child].nullable) return false;
    return true;
}

// Every non-root node has exactly one parent; struct fields are named; root
// to leaf depth and repetition stay inside the writer's level tables.
bool nested_tree_valid(const ParquetNestedNode* t, std::uint32_t n) noexcept {
    assert(t != nullptr);
    assert(n >= 1u && n <= kPwMaxNestedNodes);
    std::uint8_t parents[kPwMaxNestedNodes] = {};
    std::uint8_t depth[kPwMaxNestedNodes] = {};
    std::uint8_t reps[kPwMaxNestedNodes] = {};
    depth[0] = 1u;
    reps[0] = nested_is_repeated(t[0].type) ? 1u : 0u;
    for (std::uint32_t i = 0; i < n; ++i) {
        if (i > 0u && parents[i] != 1u) return false;   // parent precedes child
        for (std::uint32_t j = 0; j < t[i].n_children; ++j) {
            const std::uint32_t k = t[i].first_child + j;
            if (++parents[k] > 1u) return false;
            if (t[i].type == BoltType::Struct && t[k].name[0] == '\0') return false;
            depth[k] = static_cast<std::uint8_t>(depth[i] + 1u);
            reps[k] = static_cast<std::uint8_t>(
                reps[i] + (nested_is_repeated(t[k].type) ? 1u : 0u));
            if (depth[k] > kPwMaxNestedDepth) return false;
            if (reps[k] > kPwMaxNestedRep) return false;
        }
    }
    return true;
}

bool nested_column_valid(const ParquetWriteColumn& c) noexcept {
    const ParquetNestedNode* t = c.nested;
    const std::uint32_t n = c.n_nested;
    if (t == nullptr || n == 0u || n > kPwMaxNestedNodes) return false;
    if (!nested_is_group(t[0].type) || t[0].type != c.type) return false;
    if (t[0].name[0] == '\0') return false;
    if (c.logical != 0u || c.list_depth != 0u) return false;
    if (c.encoding > static_cast<std::uint8_t>(PqWriteEncoding::Plain)) return false;
    for (std::uint32_t i = 0; i < n; ++i) {
        if (!nested_node_valid(t, n, i)) return false;
    }
    return nested_tree_valid(t, n);
}

// ===== leaf planning (at open) ============================================

// The schema name of child slot `j` of `parent`: canonical for LIST / MAP,
// the node's own for a struct field.
const char* nested_child_name(const ParquetNestedNode& parent, std::uint32_t j,
                              const ParquetNestedNode& child) noexcept {
    assert(nested_is_group(parent.type));
    assert(j < parent.n_children);
    if (parent.type == BoltType::List) return "element";
    if (parent.type == BoltType::Map) return (j == 0u) ? "key" : "value";
    return child.name;
}

void nested_leaf_levels(const ParquetWriter* w, PwLeaf* L) noexcept {
    assert(w != nullptr && L != nullptr);
    assert(L->path_len >= 1u && L->path_len <= kPwMaxNestedDepth);
    std::uint32_t def = 0, rep = 0;
    for (std::uint32_t k = 0; k < L->path_len; ++k) {
        const ParquetNestedNode& nd = w->nodes[L->path[k]];
        def += nd.nullable ? 1u : 0u;
        if (nested_is_repeated(nd.type)) { ++def; ++rep; }
    }
    assert(def < kPqWMaxDefBuckets && rep < kPqWMaxRepBuckets);
    L->max_def = static_cast<std::uint8_t>(def);
    L->max_rep = static_cast<std::uint8_t>(rep);
}

// Pre-order DFS of one column's node table (based at `nb` in w->nodes),
// appending one PwLeaf per leaf in schema order.
bool plan_nested_leaves(ParquetWriter* w, std::uint32_t col,
                        std::uint32_t nb, std::uint32_t n) noexcept {
    assert(w != nullptr);
    assert(n >= 1u && n <= kPwMaxNestedNodes);
    std::uint32_t st_node[kPwMaxNestedNodes];
    std::uint8_t  st_depth[kPwMaxNestedNodes];
    std::uint32_t path[kPwMaxNestedDepth];
    std::uint32_t sp = 0;
    st_node[sp] = 0u; st_depth[sp] = 0u; ++sp;
    for (std::uint32_t guard = 0; sp > 0u && guard < n; ++guard) {
        --sp;
        const std::uint32_t rel = st_node[sp];
        const std::uint8_t d = st_depth[sp];
        assert(d < kPwMaxNestedDepth);
        path[d] = nb + rel;
        const ParquetNestedNode& nd = w->nodes[nb + rel];
        if (nested_is_group(nd.type)) {
            for (std::uint32_t j = nd.n_children; j-- > 0u;) {
                if (sp >= kPwMaxNestedNodes) return false;
                st_node[sp] = nd.first_child + j;
                st_depth[sp] = static_cast<std::uint8_t>(d + 1u);
                ++sp;
            }
            continue;
        }
        PwLeaf L{};
        L.col = col;
        L.nested = true;
        L.node_base = nb;
        L.path_len = static_cast<std::uint8_t>(d + 1u);
        for (std::uint32_t k = 0; k <= d; ++k) L.path[k] = path[k];
        nested_leaf_levels(w, &L);
        L.sch = nested_leaf_column(nd, nd.name);
        w->leaves.push_back(L);
    }
    return sp == 0u;
}

// ===== schema elements + leaf paths (footer) ==============================

constexpr std::int32_t kConvMap = 1;   // ConvertedType MAP

void write_nested_group(TcOut* o, const ParquetNestedNode& nd, const char* name,
                        std::uint32_t n_children) noexcept {
    assert(o != nullptr && name != nullptr);
    assert(nested_is_group(nd.type));
    tc_put_field(o, 3, kFI32);
    tc_put_zigzag(o, nd.nullable ? kRepOptional : kRepRequired);
    tc_put_field(o, 4, kFBinary);
    tc_put_string(o, name);
    tc_put_field(o, 5, kFI32);
    tc_put_zigzag(o, static_cast<std::int64_t>(n_children));
    if (nd.type != BoltType::Struct) {
        const bool list = nd.type == BoltType::List;
        tc_put_field(o, 6, kFI32);
        tc_put_zigzag(o, list ? kConvList : kConvMap);
        tc_put_field(o, 10, kFStruct);
        tc_put_field(o, list ? 3 : 2, kFStruct);   // LogicalType LIST / MAP
        tc_put_stop(o);
        tc_put_stop(o);
    }
    if (nd.has_field_id) {
        tc_put_field(o, 9, kFI32);
        tc_put_zigzag(o, nd.field_id);
    }
    tc_put_stop(o);
}

void write_repeated_group(TcOut* o, const char* name,
                          std::uint32_t n_children) noexcept {
    assert(o != nullptr && name != nullptr);
    assert(n_children >= 1u);
    tc_put_field(o, 3, kFI32);
    tc_put_zigzag(o, kRepRepeated);
    tc_put_field(o, 4, kFBinary);
    tc_put_string(o, name);
    tc_put_field(o, 5, kFI32);
    tc_put_zigzag(o, static_cast<std::int64_t>(n_children));
    tc_put_stop(o);
}

// Recursion is bounded by kPwMaxNestedDepth (validated at open).
void write_nested_elements(TcOut* o, const ParquetWriter* w, std::uint32_t nb,
                           std::uint32_t rel, const char* name,
                           std::uint32_t depth) noexcept {
    assert(o != nullptr && w != nullptr);
    assert(depth < kPwMaxNestedDepth);
    const ParquetNestedNode& nd = w->nodes[nb + rel];
    if (!nested_is_group(nd.type)) {
        write_schema_element_col(o, nested_leaf_column(nd, name));
        return;
    }
    const bool repeated = nested_is_repeated(nd.type);
    write_nested_group(o, nd, name, repeated ? 1u : nd.n_children);
    if (repeated) {
        write_repeated_group(o, nd.type == BoltType::List ? "list" : "key_value",
                             nd.n_children);
    }
    for (std::uint32_t j = 0; j < nd.n_children; ++j) {
        const ParquetNestedNode& ch = w->nodes[nb + nd.first_child + j];
        write_nested_elements(o, w, nb, nd.first_child + j,
                              nested_child_name(nd, j, ch), depth + 1u);
    }
}

std::uint32_t nested_element_count(const ParquetWriter* w, std::uint32_t nb,
                                   std::uint32_t n) noexcept {
    assert(w != nullptr);
    assert(n >= 1u && n <= kPwMaxNestedNodes);
    std::uint32_t c = n;
    for (std::uint32_t i = 0; i < n; ++i) {
        c += nested_is_repeated(w->nodes[nb + i].type) ? 1u : 0u;
    }
    return c;
}

// The schema name of path step k (k > 0 names it as its parent's child).
const char* nested_path_name(const ParquetWriter* w, const PwLeaf& L,
                             std::uint32_t k) noexcept {
    assert(w != nullptr && k < L.path_len);
    const ParquetNestedNode& nd = w->nodes[L.path[k]];
    if (k == 0u) return nd.name;
    const ParquetNestedNode& parent = w->nodes[L.path[k - 1u]];
    const std::uint32_t j = L.path[k] - (L.node_base + parent.first_child);
    return nested_child_name(parent, j, nd);
}

void write_nested_path(TcOut* o, const ParquetWriter* w,
                       const PwLeaf& L) noexcept {
    assert(o != nullptr && w != nullptr);
    assert(L.nested && L.path_len >= 1u);
    std::uint32_t segs = 0;
    for (std::uint32_t k = 0; k < L.path_len; ++k) {
        segs += 1u + (nested_is_repeated(w->nodes[L.path[k]].type) ? 1u : 0u);
    }
    tc_put_list_hdr(o, kFBinary, segs);
    for (std::uint32_t k = 0; k < L.path_len; ++k) {
        tc_put_string(o, nested_path_name(w, L, k));
        const BoltType t = w->nodes[L.path[k]].type;
        if (t == BoltType::List) tc_put_string(o, "list");
        if (t == BoltType::Map) tc_put_string(o, "key_value");
    }
}

std::int32_t nested_leaf_physical(const ParquetWriteColumn& sch) noexcept {
    if (sch.type == BoltType::Binary && sch.fixed_len != 0u) return kPtFlba;
    return bolt_to_pq_physical(sch.type);
}

// ===== shredding ==========================================================

// Per-chunk walk tables for one leaf: the batch column at every path step
// (fixed for the row group) and the def level a NULL at that step writes.
struct NestedWalk {
    const ParquetNestedNode* nodes[kPwMaxNestedDepth];
    const BoltColumn*        cols[kPwMaxNestedDepth];
    std::uint8_t             def_null[kPwMaxNestedDepth];
    std::uint8_t             rep_lvl[kPwMaxNestedDepth];   // repeated steps only
    std::uint32_t            len;
};

// The batch column for path step k + 1, or nullptr when the batch does not
// have the declared shape.
const BoltColumn* nested_step_child(const BoltColumn* c, const ParquetNestedNode& nd,
                                    std::uint32_t slot) noexcept {
    assert(c != nullptr);
    assert(slot < nd.n_children);
    if (c->format != ColumnFormat::Nested || c->type != nd.type) return nullptr;
    if (nd.type == BoltType::Struct) {
        return (c->child_count() == nd.n_children) ? c->child_at(slot) : nullptr;
    }
    if (c->list_offsets() == nullptr) return nullptr;
    const BoltColumn* e = c->list_element();
    if (e == nullptr || nd.type == BoltType::List) return e;
    if (e->format != ColumnFormat::Nested || e->type != BoltType::Struct) return nullptr;
    return (e->child_count() == 2) ? e->child_at(slot) : nullptr;
}

bool nested_walk_init(const ParquetWriter* w, const PwLeaf& L,
                      const BoltColumn& top, NestedWalk* nw) noexcept {
    assert(w != nullptr && nw != nullptr);
    assert(L.nested && L.path_len >= 1u);
    nw->len = L.path_len;
    const BoltColumn* c = &top;
    std::uint32_t def = 0, rep = 0;
    for (std::uint32_t k = 0; k < L.path_len; ++k) {
        const ParquetNestedNode& nd = w->nodes[L.path[k]];
        nw->nodes[k] = &nd;
        nw->cols[k] = c;
        nw->def_null[k] = static_cast<std::uint8_t>(def);
        def += nd.nullable ? 1u : 0u;
        nw->rep_lvl[k] = 0u;
        if (nested_is_repeated(nd.type)) {
            ++def;
            nw->rep_lvl[k] = static_cast<std::uint8_t>(++rep);
        }
        if (k + 1u == L.path_len) break;
        const std::uint32_t slot = L.path[k + 1u] - (L.node_base + nd.first_child);
        c = nested_step_child(c, nd, slot);
        if (c == nullptr) return false;
    }
    assert(def == L.max_def && rep == L.max_rep);
    // The leaf must carry the declared flat type.
    return c->type == L.sch.type && c->format != ColumnFormat::Nested;
}

// Dremel levels of one path, generalized over every node kind. `idx` indexes
// cols[k]; rep_in is the repetition level of the first slot this emits.
struct NestedShred {
    std::vector<std::uint8_t> rep;
    std::vector<std::uint8_t> def;
    std::vector<std::int64_t> row_slot;   // row -> first level slot (rows + 1)
    std::vector<std::int64_t> row_val;    // row -> first dense value (rows + 1)
    std::vector<std::int64_t> vidx;       // present leaf values, leaf-column rows
};

// A REQUIRED node holding a null, a null map entry, or offsets past their
// child refuse the chunk: the batch contradicts the declared schema.
bool shred_value(const NestedWalk& nw, std::uint32_t k, std::int64_t idx,
                 std::uint8_t rep_in, NestedShred* out) noexcept {
    assert(out != nullptr);
    assert(k < nw.len);
    const BoltColumn* c = nw.cols[k];
    const ParquetNestedNode& nd = *nw.nodes[k];
    if (idx < 0 || idx >= c->length) return false;
    if (read_valid(c->validity, c->validity_offset, idx) == 0u) {
        if (!nd.nullable) return false;
        out->rep.push_back(rep_in);
        out->def.push_back(nw.def_null[k]);
        return true;
    }
    const std::uint8_t d_self =
        static_cast<std::uint8_t>(nw.def_null[k] + (nd.nullable ? 1u : 0u));
    if (k + 1u == nw.len) {
        out->rep.push_back(rep_in);
        out->def.push_back(d_self);
        out->vidx.push_back(idx);
        return true;
    }
    if (nd.type == BoltType::Struct) return shred_value(nw, k + 1u, idx, rep_in, out);
    const std::int32_t* offs = c->list_offsets();
    const std::int32_t lo = offs[idx];
    const std::int32_t hi = offs[idx + 1];
    if (lo < 0 || hi < lo) return false;
    if (hi == lo) {
        out->rep.push_back(rep_in);
        out->def.push_back(d_self);
        return true;
    }
    const BoltColumn* entries = (nd.type == BoltType::Map) ? c->list_element() : nullptr;
    for (std::int32_t e = lo; e < hi; ++e) {
        if (entries != nullptr &&
            (e >= entries->length ||
             read_valid(entries->validity, entries->validity_offset, e) == 0u)) {
            return false;
        }
        const std::uint8_t r = (e == lo) ? rep_in : nw.rep_lvl[k];
        if (!shred_value(nw, k + 1u, e, r, out)) return false;
    }
    return true;
}

bool shred_rows(const NestedWalk& nw, std::int64_t start, std::int64_t rows,
                NestedShred* out) noexcept {
    assert(out != nullptr);
    assert(start >= 0 && rows >= 0);
    out->rep.clear();
    out->def.clear();
    out->vidx.clear();
    out->row_slot.assign(static_cast<std::size_t>(rows) + 1u, 0);
    out->row_val.assign(static_cast<std::size_t>(rows) + 1u, 0);
    for (std::int64_t r = 0; r < rows; ++r) {
        out->row_slot[static_cast<std::size_t>(r)] =
            static_cast<std::int64_t>(out->rep.size());
        out->row_val[static_cast<std::size_t>(r)] =
            static_cast<std::int64_t>(out->vidx.size());
        if (!shred_value(nw, 0u, start + r, 0u, out)) return false;
    }
    out->row_slot[static_cast<std::size_t>(rows)] =
        static_cast<std::int64_t>(out->rep.size());
    out->row_val[static_cast<std::size_t>(rows)] =
        static_cast<std::int64_t>(out->vidx.size());
    return true;
}

// ===== dense leaf =========================================================

// The present values of a leaf, contiguous, so the flat PLAIN encoder and the
// statistics see an ordinary REQUIRED column. Byte arrays become VarBinary
// (offsets + pool); fixed-width lanes are copied. Must not be moved once
// built: `col.dict_child` points at `off_col`.
struct DenseLeaf {
    std::vector<std::uint8_t> bytes;
    std::vector<std::int32_t> offs;
    BoltColumn                off_col;
    BoltColumn                col;
};

std::uint32_t nested_lane_bytes(BoltType t) noexcept {
    switch (t) {
        case BoltType::Bool:       return 1u;
        case BoltType::Int32:
        case BoltType::Date32:
        case BoltType::Float32:    return 4u;
        case BoltType::Int64:
        case BoltType::Timestamp:
        case BoltType::Float64:    return 8u;
        case BoltType::UUID:
        case BoltType::Decimal128: return 16u;
        default:                   return 0u;
    }
}

bool gather_dense_bytes(const BoltColumn& leaf, BoltType t,
                        const std::vector<std::int64_t>& vidx,
                        DenseLeaf* d) noexcept {
    assert(d != nullptr);
    assert(t == BoltType::Utf8 || t == BoltType::Binary);
    if (leaf.format != ColumnFormat::Flat && leaf.format != ColumnFormat::View &&
        leaf.format != ColumnFormat::VarBinary) {
        return false;
    }
    if (leaf.format == ColumnFormat::VarBinary && leaf.dict_child == nullptr) return false;
    d->offs.assign(vidx.size() + 1u, 0);
    for (std::size_t i = 0; i < vidx.size(); ++i) {
        const RowVal v = row_val_bytes(leaf, vidx[i]);
        if (v.n == 0xFFFFFFFFu) return false;
        if (d->bytes.size() + v.n > 0x7FFFFFFFull) return false;
        if (v.n != 0u) d->bytes.insert(d->bytes.end(), v.p, v.p + v.n);
        d->offs[i + 1u] = static_cast<std::int32_t>(d->bytes.size());
    }
    d->bytes.push_back(0u);   // keeps the pool non-null; never referenced
    d->off_col = BoltColumn::make_flat(d->offs.data(), nullptr,
                                       static_cast<std::int64_t>(d->offs.size()),
                                       BoltType::Int32);
    d->col = BoltColumn::make_empty();
    d->col.format = ColumnFormat::VarBinary;
    d->col.type = t;
    d->col.data = d->bytes.data();
    d->col.dict_child = &d->off_col;
    d->col.length = static_cast<std::int64_t>(vidx.size());
    return true;
}

bool gather_dense(const BoltColumn& leaf, BoltType t,
                  const std::vector<std::int64_t>& vidx, DenseLeaf* d) noexcept {
    assert(d != nullptr);
    d->bytes.clear();
    d->offs.clear();
    if (t == BoltType::Utf8 || t == BoltType::Binary) {
        return gather_dense_bytes(leaf, t, vidx, d);
    }
    const std::uint32_t lane = nested_lane_bytes(t);
    if (lane == 0u || leaf.format != ColumnFormat::Flat) return false;
    if (leaf.data == nullptr && !vidx.empty()) return false;
    d->bytes.resize(vidx.size() * lane + 1u);
    const std::uint8_t* src = static_cast<const std::uint8_t*>(leaf.data);
    for (std::size_t i = 0; i < vidx.size(); ++i) {
        std::memcpy(d->bytes.data() + i * lane,
                    src + static_cast<std::size_t>(vidx[i]) * lane, lane);
    }
    d->col = BoltColumn::make_flat(d->bytes.data(), nullptr,
                                   static_cast<std::int64_t>(vidx.size()), t);
    return true;
}

// ===== pages ==============================================================

// Hybrid-encoded level run with no length prefix (the v2 layout).
void encode_level_raw(const std::uint8_t* lv, std::size_t n, std::uint32_t max_lvl,
                      std::vector<std::uint8_t>* out) noexcept {
    assert(out != nullptr);
    assert(max_lvl > 0u);
    std::vector<std::uint32_t> tmp(n);
    for (std::size_t i = 0; i < n; ++i) tmp[i] = lv[i];
    rle_hybrid_encode(tmp.data(), n, bit_width_for(max_lvl), out);
}

// Levels then values for slots [s0, s1). A level stream whose max is 0 is
// absent, as for a flat REQUIRED column.
void nested_page_payload(const PwLeaf& L, const NestedShred& sh, bool v2,
                         std::int64_t s0, std::int64_t s1,
                         const std::vector<std::uint8_t>& vals,
                         std::vector<std::uint8_t>* payload,
                         std::uint32_t* level_bytes, std::uint32_t* rep_bytes) noexcept {
    assert(payload != nullptr && level_bytes != nullptr && rep_bytes != nullptr);
    assert(s0 <= s1);
    payload->clear();
    const std::size_t n = static_cast<std::size_t>(s1 - s0);
    if (L.max_rep > 0u) {
        if (v2) encode_level_raw(sh.rep.data() + s0, n, L.max_rep, payload);
        else (void)encode_level_stream(sh.rep.data() + s0, n, L.max_rep, payload);
    }
    *rep_bytes = v2 ? static_cast<std::uint32_t>(payload->size()) : 0u;
    if (L.max_def > 0u) {
        if (v2) encode_level_raw(sh.def.data() + s0, n, L.max_def, payload);
        else (void)encode_level_stream(sh.def.data() + s0, n, L.max_def, payload);
    }
    *level_bytes = v2 ? static_cast<std::uint32_t>(payload->size()) : 0u;
    payload->insert(payload->end(), vals.begin(), vals.end());
}

// PLAIN bytes of dense values [v0, v1): an upper bound used to cut pages.
std::uint64_t dense_plain_bytes(const DenseLeaf& d, BoltType t,
                                std::int64_t v0, std::int64_t v1) noexcept {
    assert(v0 <= v1);
    const std::uint64_t n = static_cast<std::uint64_t>(v1 - v0);
    if (t == BoltType::Utf8 || t == BoltType::Binary) {
        return 4u * n + static_cast<std::uint64_t>(
            d.offs[static_cast<std::size_t>(v1)] - d.offs[static_cast<std::size_t>(v0)]);
    }
    return n * nested_lane_bytes(t);
}

// One page grows a row at a time; a record never splits across pages.
std::int64_t nested_page_end(const NestedShred& sh, const DenseLeaf& d, BoltType t,
                             std::int64_t r0, std::int64_t rows,
                             std::uint64_t budget) noexcept {
    assert(r0 <= rows);
    assert(budget > 0u);
    std::int64_t r1 = r0;
    while (r1 < rows) {
        const std::size_t a = static_cast<std::size_t>(r0);
        const std::size_t b = static_cast<std::size_t>(r1 + 1);
        const std::uint64_t est =
            dense_plain_bytes(d, t, sh.row_val[a], sh.row_val[b]) +
            static_cast<std::uint64_t>(sh.row_slot[b] - sh.row_slot[a]);
        if (r1 > r0 && est > budget) break;
        ++r1;
    }
    return r1;
}

std::int64_t nested_slot_nulls(const NestedShred& sh, std::uint8_t max_def,
                               std::int64_t s0, std::int64_t s1) noexcept {
    assert(s0 <= s1);
    assert(static_cast<std::size_t>(s1) <= sh.def.size());
    std::int64_t n = 0;
    for (std::int64_t i = s0; i < s1; ++i) n += (sh.def[static_cast<std::size_t>(i)] < max_def);
    return n;
}

bool nested_emit_pages(ParquetWriter* w, ChunkOut* out, const PwLeaf& L,
                       const NestedShred& sh, const DenseLeaf& d,
                       std::int64_t rows, ChunkRec* rec) noexcept {
    assert(w != nullptr && out != nullptr && rec != nullptr);
    assert(rows >= 0);
    const std::uint64_t budget = page_budget_bytes(w->opts);
    const bool v2 = w->opts.data_page_v2;
    std::vector<std::uint8_t> vals, payload;
    std::int64_t r0 = 0;
    std::uint32_t pages = 0;
    for (;;) {
        if (++pages > kPwMaxPagesPerChunk) return false;
        const std::int64_t r1 = nested_page_end(sh, d, L.sch.type, r0, rows, budget);
        const std::int64_t s0 = sh.row_slot[static_cast<std::size_t>(r0)];
        const std::int64_t s1 = sh.row_slot[static_cast<std::size_t>(r1)];
        const std::int64_t v0 = sh.row_val[static_cast<std::size_t>(r0)];
        const std::int64_t v1 = sh.row_val[static_cast<std::size_t>(r1)];
        vals.clear();
        if (!encode_plain_range(d.col, L.sch, v0, v1, nullptr, false, &vals)) return false;
        std::uint32_t level_bytes = 0, rep_bytes = 0;
        nested_page_payload(L, sh, v2, s0, s1, vals, &payload, &level_bytes, &rep_bytes);
        const std::int64_t nulls = nested_slot_nulls(sh, L.max_def, s0, s1);
        PageSpec spec{false, v2, s1 - s0, nulls, kEncPlain, level_bytes,
                      rep_bytes, r1 - r0};
        std::int64_t off = 0;
        std::int32_t sz = 0;
        if (!emit_page(w, out, payload, spec, rec, &off, &sz)) return false;
        if (rec->data_page_offset < 0) rec->data_page_offset = off;
        StatBuf ps;
        std::memset(&ps, 0, sizeof(ps));
        if (w->opts.emit_page_index && w->opts.emit_statistics) {
            (void)compute_stats(d.col, v0, v1, nullptr, false, &ps);
        }
        note_page(w, out, rec, off, sz, r0, nulls, s1 - s0, ps);
        r0 = r1;
        if (r0 >= rows) break;
    }
    return true;
}

void nested_chunk_stats(const ParquetWriter* w, const PwLeaf& L,
                        const NestedShred& sh, const DenseLeaf& d,
                        ChunkRec* rec) noexcept {
    assert(w != nullptr && rec != nullptr);
    assert(sh.rep.size() == sh.def.size());
    const std::int64_t n_vals = static_cast<std::int64_t>(sh.vidx.size());
    rec->num_values = static_cast<std::int64_t>(sh.def.size());
    rec->null_count = nested_slot_nulls(sh, L.max_def, 0, rec->num_values);
    rec->null_count_known = true;
    rec->encoding = kEncPlain;
    if (w->opts.emit_statistics) {
        (void)compute_stats(d.col, 0, n_vals, nullptr, false, &rec->st);
    }
    if (!w->opts.emit_size_statistics) return;
    if (L.max_def > 1u || L.max_rep > 0u) {
        rec->level_hist_known = true;
        rec->max_def = L.max_def;
        rec->max_rep = L.max_rep;
        std::memset(rec->def_hist, 0, sizeof(rec->def_hist));
        std::memset(rec->rep_hist, 0, sizeof(rec->rep_hist));
        for (std::size_t i = 0; i < sh.def.size(); ++i) {
            rec->def_hist[sh.def[i]] += 1;
            rec->rep_hist[sh.rep[i]] += 1;
        }
    }
    if ((L.sch.type == BoltType::Utf8 || L.sch.type == BoltType::Binary) &&
        L.sch.fixed_len == 0u) {
        std::int64_t bytes = 0;
        if (sum_byte_array_data_bytes(d.col, 0, n_vals, nullptr, false, &bytes)) {
            rec->byte_array_bytes_known = true;
            rec->byte_array_bytes = bytes;
        }
    }
}

// One leaf chunk of a nested column for rows [start, start + rows) of the
// UNSLICED batch column -- the walk indexes it directly, so struct columns
// (which slice_column cannot slice) need no view.
bool chunk_write_nested(ParquetWriter* w, ChunkOut* out, const BoltColumn& top,
                        const PwLeaf& L, std::int64_t start,
                        std::int64_t rows) noexcept {
    assert(w != nullptr && out != nullptr);
    assert(L.nested && start >= 0 && rows >= 0);
    out->bytes.clear();
    out->pages.clear();
    out->bloom.clear();
    out->ok = false;
    ChunkRec* rec = &out->rec;
    std::memset(rec, 0, sizeof(*rec));
    rec->data_page_offset = -1;
    rec->dict_page_offset = -1;
    if (top.length < start + rows) return false;
    NestedWalk nw;
    if (!nested_walk_init(w, L, top, &nw)) return false;
    NestedShred sh;
    if (!shred_rows(nw, start, rows, &sh)) return false;
    DenseLeaf d;
    if (!gather_dense(*nw.cols[nw.len - 1u], L.sch.type, sh.vidx, &d)) return false;
    nested_chunk_stats(w, L, sh, d, rec);
    if (!nested_emit_pages(w, out, L, sh, d, rows, rec)) return false;
    out->ok = true;
    return true;
}
