// Nested source admission shares the flat adapter's span/range primitives.
bool nested_metadata_spans(const ParquetByteSource* source,
    const ParquetSourceWorkspace* w, const Arena* arena, const PqMeta* meta,
    const PqSchemaTree* tree) noexcept {
    assert(source != nullptr && w != nullptr);
    assert(arena != nullptr && meta != nullptr);
    const Span inputs[]{{source, sizeof(*source)}, {w, sizeof(*w)}, {arena, sizeof(*arena)},
        {w->bytes, static_cast<size_t>(w->capacity)}, {meta, sizeof(*meta)}};
    return disjoint({tree, sizeof(*tree)}, inputs, 5);
}

bool nested_tree_spans(const ParquetSourceWorkspace* w, const PqSchemaTree* tree,
    const BoltColumn* out, const int64_t* rows) noexcept {
    assert(w != nullptr && out != nullptr);
    assert(rows != nullptr);
    if (tree == nullptr || tree->n_nodes == 0 || tree->n_nodes > kPqMaxSchemaElems ||
        tree->n_fields == 0 || tree->n_fields > 2u * kPqMaxSchemaElems ||
        tree->n_top == 0 || tree->n_top > kMaxColumns) return false;
    const Span inputs[]{{tree, sizeof(*tree)}, {tree->nodes, sizeof(PqSchemaNode) * tree->n_nodes},
        {tree->fields, sizeof(PqField) * tree->n_fields}, {tree->top, sizeof(int32_t) * tree->n_top}};
    if (!disjoint({w->bytes, static_cast<size_t>(w->capacity)}, inputs, 4) ||
        !disjoint({out, sizeof(*out)}, inputs, 4) || !disjoint({rows, sizeof(*rows)}, inputs, 4))
        return false;
    for (uint32_t i = 0; i < tree->n_top; ++i)
        if (tree->top[i] < 0 || static_cast<uint32_t>(tree->top[i]) >= tree->n_fields) return false;
    return true;
}

bool nested_spans(const ParquetByteSource* source, const ParquetSourceWorkspace* w,
    const PqMeta* meta, const PqSchemaTree* tree, const Arena* arena,
    const BoltColumn* out, const int64_t* rows) noexcept {
    assert(source != nullptr && w != nullptr && meta != nullptr);
    assert(arena != nullptr && out != nullptr && rows != nullptr);
    // Reuse flat metadata/output admission with an inert borrowed index.
    const uint16_t index = 0;
    if (!projection_spans(source, w, meta, &index, 1, arena, out, rows) ||
        !nested_tree_spans(w, tree, out, rows)) return false;
    const Span inputs[]{{source, sizeof(*source)}, {w, sizeof(*w)}, {arena, sizeof(*arena)},
        {meta, sizeof(*meta)}, {meta->columns, sizeof(PqColumn) * meta->n_columns},
        {meta->chunks, sizeof(PqChunk) * meta->n_chunks}};
    return disjoint({tree, sizeof(*tree)}, inputs, 6);
}

Status nested_extents(const ParquetByteSource& source, const ParquetSourceWorkspace& w,
    const PqMeta& meta, const PqSchemaTree& tree, uint32_t row_group,
    const PqField& field) noexcept {
    assert(row_group < meta.n_row_groups);
    assert(meta.n_columns <= kMaxColumns);
    if (source.size < 12 || meta.n_row_groups > kPqMaxRowGroups ||
        tree.n_columns != meta.n_columns || field.col_begin >= field.col_end ||
        field.col_end > meta.n_columns) return Status::BadArg;
    const auto& group = meta.row_groups[row_group];
    if (group.num_rows < 0 || group.chunk_off > meta.n_chunks ||
        meta.n_columns > meta.n_chunks - group.chunk_off) return Status::DecodeFailed;
    for (uint32_t col = field.col_begin; col < field.col_end; ++col) {
        const auto& chunk = meta.chunks[group.chunk_off + col];
        if (chunk.num_values < group.num_rows || chunk.num_values < 0 ||
            (meta.columns[col].max_rep == 0 && chunk.num_values != group.num_rows))
            return Status::DecodeFailed;
        if (group.num_rows == 0 && chunk.total_compressed_size == 0) continue;
        uint64_t offset = 0, length = 0;
        if (!chunk_extent(chunk, source.size - 8u, &offset, &length)) return Status::DecodeFailed;
        if (length > w.capacity) return Status::Capacity;
    }
    return Status::Ok;
}
