// Top-level typed layout and child-ID conformance for the ordinary Iceberg scan.
bool schema_has_nested(const Schema* schema) noexcept {
    assert(sizeof(SchemaField::type) > 0); assert(kMaxColumns > 0);
    if (schema == nullptr) return false;
    for (uint32_t i = 0; i < schema->n_fields; ++i) {
        const auto* t = schema->fields[i].type;
        if (!std::strcmp(t,"struct") || !std::strcmp(t,"list") || !std::strcmp(t,"map")) return true;
    }
    return false;
}

int32_t nested_find(const ScanHandle* s, int32_t parent, const SchemaField& field,
                    bool ids) noexcept {
    assert(s != nullptr); assert(parent >= -1);
    const auto& tree = s->pinned.tree;
    const uint32_t count = parent < 0 ? tree.n_top : tree.fields[parent].n_children;
    int32_t child = parent < 0 ? -1 : tree.fields[parent].first_child;
    int32_t found = -1;
    const NameMapEntry* mapping = nullptr;
    if (!ids) for (uint32_t i = 0; i < s->nm_count; ++i)
        if (s->nm_entries[i].field_id == field.id) mapping = &s->nm_entries[i];
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t at = parent < 0 ? tree.top[i] : child;
        if (at < 0 || static_cast<uint32_t>(at) >= tree.n_fields) return -2;
        const auto& physical = tree.fields[at];
        bool hit = ids ? physical.has_field_id && physical.field_id == field.id : false;
        if (!ids && mapping != nullptr) for (uint32_t j = 0; j < mapping->n_names; ++j)
            hit |= std::strcmp(physical.name,mapping->names[j]) == 0;
        if (!ids && s->nm_count == 0) hit = std::strcmp(physical.name,field.name) == 0;
        if (hit) { if (found >= 0) return -2; found = at; }
        child = physical.next_sibling;
    }
    return found;
}

bool nested_has_ids(const pq::PqSchemaTree& tree) noexcept {
    assert(tree.nodes != nullptr); assert(tree.n_nodes > 0);
    for (uint32_t i = 0; i < tree.n_nodes; ++i) if (tree.nodes[i].has_field_id) return true;
    return false;
}

bool bind_nested_schema(ScanHandle* s, const pq::PqMeta* meta, const Schema* schema) noexcept {
    assert(s != nullptr && meta != nullptr); assert(s->meta_arena != nullptr);
    const auto& tree = s->pinned.tree;
    if (schema == nullptr || schema->n_fields == 0 || schema->n_fields > kMaxColumns ||
        tree.n_top == 0 || tree.n_top > kMaxColumns || tree.n_columns != meta->n_columns) return false;
    const bool ids = nested_has_ids(tree);
    auto* sorted = s->meta_arena->allocate_array<int32_t>(tree.n_nodes);
    if (sorted == nullptr) return false;
    uint32_t used = 0;
    for (uint32_t i = 0; i < tree.n_nodes; ++i) if (tree.nodes[i].has_field_id) {
        if (tree.nodes[i].field_id <= 0) return false;
        sorted[used++] = tree.nodes[i].field_id;
    }
    std::sort(sorted,sorted + used);
    for (uint32_t i = 1; i < used; ++i) if (sorted[i] == sorted[i-1u]) return false;
    if (!ids && !s->nm_loaded) {
        s->nm_loaded = true;
        if (!metadata_name_mapping(&s->table->meta,s->scratch,&s->nm_entries,&s->nm_count)) return false;
    }
    if (s->colmap == nullptr || s->map_schema != schema) {
        s->colmap = s->scratch->allocate_array<int32_t>(schema->n_fields);
        if (s->colmap == nullptr) return false;
    }
    for (uint32_t i = 0; i < schema->n_fields; ++i) {
        const int32_t field = nested_find(s,-1,schema->fields[i],ids);
        if (field == -2) return false;
        s->colmap[i] = -1;
        for (uint32_t j = 0; j < tree.n_top; ++j)
            if (tree.top[j] == field) s->colmap[i] = static_cast<int32_t>(j);
    }
    for (uint32_t i = 0; i < tree.n_top; ++i) s->source_indices[i] = static_cast<uint16_t>(i);
    s->map_schema = schema; s->use_map = true; s->nested_layout = true;
    s->decode_columns = tree.n_top; s->output_columns = schema->n_fields;
    return true;
}

bool nested_absent(const SchemaField&, int64_t, bool, Arena*, uint32_t, BoltColumn*) noexcept;
bool nested_absent_group(const SchemaField& f, int64_t rows, Arena* arena,
    uint32_t depth, BoltColumn* out) noexcept {
    assert(arena != nullptr && out != nullptr); assert(depth <= kIcebergSchemaMaxDepth);
    auto* children = arena->allocate_array<BoltColumn>(f.n_children + 1u);
    auto* valid = static_cast<uint8_t*>(arena->allocate_zeroed((static_cast<uint64_t>(rows)+7u)/8u+1u));
    if (children == nullptr || valid == nullptr) return false;
    const bool structure = std::strcmp(f.type,"struct") == 0;
    for (uint32_t i = 0; i < f.n_children; ++i)
        if (!nested_absent(f.children[i],structure ? rows : 0,true,arena,depth+1u,&children[i])) return false;
    if (structure) {
        *out = BoltColumn::make_struct(children,f.n_children,rows,valid,arena);
    } else {
        auto* offsets = arena->allocate_array<int32_t>(static_cast<size_t>(rows)+1u);
        if (offsets == nullptr) return false;
        std::memset(offsets,0,(static_cast<size_t>(rows)+1u)*sizeof(int32_t));
        BoltColumn element = children[0];
        if (std::strcmp(f.type,"map") == 0) {
            element = BoltColumn::make_struct(children,2,0,nullptr,arena);
            if (element.data == nullptr) return false;
        }
        *out = BoltColumn::make_list(&element,offsets,rows,valid,arena);
        if (std::strcmp(f.type,"map") == 0) out->type = BoltType::Map;
    }
    return out->data != nullptr;
}

bool nested_absent(const SchemaField& f, int64_t rows, bool masked, Arena* arena,
                   uint32_t depth, BoltColumn* out) noexcept {
    assert(arena != nullptr && out != nullptr); assert(rows >= 0);
    if (depth > kIcebergSchemaMaxDepth || rows > INT32_MAX) return false;
    const bool group = !std::strcmp(f.type,"struct") || !std::strcmp(f.type,"list") || !std::strcmp(f.type,"map");
    if (group) {
        if (f.initial_default_kind != DefaultKind::kNone || (f.required && !masked && rows != 0)) return false;
        return nested_absent_group(f,rows,arena,depth,out);
    }
    uint32_t fixed = 0, scale = 0;
    BoltType type = want_bolt_type(f.type);
    if (type == BoltType::NA) type = fill_bolt_type(f.type,&fixed,&scale);
    if (!masked && f.initial_default_kind != DefaultKind::kNone)
        return default_column(f,type,rows,arena,out);
    if (type == BoltType::NA || (f.required && !masked && rows != 0)) return false;
    const size_t width = type == BoltType::Binary ? sizeof(StringView) : bolt::type_size(type);
    auto* data = arena->allocate_zeroed(width*(static_cast<size_t>(rows)+1u));
    auto* valid = static_cast<uint8_t*>(arena->allocate_zeroed((static_cast<uint64_t>(rows)+7u)/8u+1u));
    if (data == nullptr || valid == nullptr) return false;
    *out = BoltColumn::make_flat(data,valid,rows,type);
    if (type == BoltType::Binary) out->type_size_bytes = sizeof(StringView);
    out->fixed_width = static_cast<uint8_t>(fixed); out->decimal_scale = static_cast<uint8_t>(scale);
    out->stats.all_valid = rows == 0; out->stats.null_count = static_cast<uint32_t>(rows);
    return true;
}

bool nested_conform(ScanHandle*,const SchemaField&,int32_t,const BoltColumn&,bool,uint32_t,BoltColumn*) noexcept;
bool nested_children(ScanHandle* s, const SchemaField& field, int32_t parent,
    const BoltColumn& src, bool ids, uint32_t depth, BoltColumn* out) noexcept {
    assert(s != nullptr && out != nullptr); assert(depth <= kIcebergSchemaMaxDepth);
    const auto& tree = s->pinned.tree;
    const auto& physical = tree.fields[parent];
    if (src.type != BoltType::Struct || src.child_count() != physical.n_children) return false;
    auto* children = s->scratch->allocate_array<BoltColumn>(field.n_children + 1u);
    if (children == nullptr) return false;
    for (uint32_t i = 0; i < field.n_children; ++i) {
        const auto& desired = field.children[i];
        const int32_t found = nested_find(s,parent,desired,ids);
        if (found == -2) return false;
        if (found < 0) {
            if (!nested_absent(desired,src.length,false,s->scratch,depth+1u,&children[i])) return false;
            continue;
        }
        int32_t current = physical.first_child;
        uint32_t ordinal = 0;
        for (; ordinal < physical.n_children && current != found; ++ordinal)
            current = tree.fields[current].next_sibling;
        const auto* child = src.child_at(ordinal);
        if (ordinal == physical.n_children || child == nullptr ||
            !nested_conform(s,desired,found,*child,ids,depth+1u,&children[i])) return false;
    }
    *out = src; out->data = children; out->seq_offset = field.n_children;
    return true;
}

bool nested_conform(ScanHandle* s, const SchemaField& field, int32_t index,
    const BoltColumn& src, bool ids, uint32_t depth, BoltColumn* out) noexcept {
    assert(s != nullptr && out != nullptr); assert(index >= 0);
    if (depth > kIcebergSchemaMaxDepth) return false;
    const auto& tree = s->pinned.tree;
    const auto& physical = tree.fields[index];
    if (ids && (!physical.has_field_id || physical.field_id != field.id)) return false;
    if (field.required && tree.nodes[physical.node].repetition == 1u) return false;
    if (physical.kind == pq::PqFieldKind::Leaf) {
        auto column = s->cur_meta->columns[physical.column];
        column.has_field_id = 1; column.field_id = field.id;
        if (!schema_leaf_matches_parquet(field,column)) {
            SchemaField old = field;
            if (!std::strcmp(field.type,"long") && column.physical == pq::PqType::Int32) std::strcpy(old.type,"int");
            else if (!std::strcmp(field.type,"double") && column.physical == pq::PqType::Float) std::strcpy(old.type,"float");
            else return false;
            if (!schema_leaf_matches_parquet(old,column)) return false;
        }
        return promote_column(src,want_bolt_type(field.type),src.length,s->scratch,out,field.name);
    }
    const auto kind = physical.kind;
    const char* type = kind == pq::PqFieldKind::Struct ? "struct" : kind == pq::PqFieldKind::List ? "list" : "map";
    if (std::strcmp(field.type,type) != 0 || src.format != ColumnFormat::Nested) return false;
    if (kind == pq::PqFieldKind::Struct) return nested_children(s,field,index,src,ids,depth,out);
    const auto* element = src.list_element();
    if (element == nullptr || physical.first_child < 0) return false;
    BoltColumn child{};
    if (kind == pq::PqFieldKind::List) {
        if (field.n_children != 1u || !nested_conform(s,field.children[0],physical.first_child,
            *element,ids,depth+1u,&child)) return false;
    } else if (field.n_children != 2u || !nested_children(s,field,physical.first_child,
        *element,ids,depth,&child)) return false;
    auto selected = BoltColumn::make_list(&child,src.list_offsets(),src.length,src.validity,s->scratch);
    if (selected.data == nullptr) return false;
    *out = src; out->data = selected.data;
    return true;
}

bool conform_nested_group(ScanHandle* s, const BoltColumn* decoded,
    BoltColumn* out, int64_t rows) noexcept {
    assert(s != nullptr && s->map_schema != nullptr); assert(out != nullptr && decoded != nullptr);
    const bool ids = nested_has_ids(s->pinned.tree);
    for (uint32_t i = 0; i < s->map_schema->n_fields; ++i) {
        const auto& field = s->map_schema->fields[i];
        const int32_t col = s->colmap[i];
        if (col < 0) {
            if (!nested_absent(field,rows,false,s->scratch,0,&out[i])) return false;
        } else if (!nested_conform(s,field,s->pinned.tree.top[col],decoded[col],ids,0,&out[i])) return false;
    }
    return true;
}
