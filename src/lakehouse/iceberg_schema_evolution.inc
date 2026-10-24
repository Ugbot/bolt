// bolt/lakehouse/iceberg_schema_evolution.inc — schema, partition-spec and sort-order evolution. Included inside
// iceberg_writer.cpp at namespace scope; not a standalone TU.

// ---------------------------------------------------------------------------
// Schema evolution
// ---------------------------------------------------------------------------
//
// G2ICE-89. The Iceberg spec makes schema evolution ADDITIVE: a schema is
// immutable once published, and add/drop/rename append a NEW Schema object
// with its own schema-id to metadata.schemas[], moving current-schema-id to
// it. This is precisely so a snapshot committed before the change keeps
// pointing at the schema-id its data files were actually written under
// (`publish_snapshot` records `s.schema_id = current_schema_id` at commit
// time -- correct already, and unchanged by this fix) -- so a time-travel
// read of that old snapshot still resolves the shape its files really have,
// not the shape the table grew into later.
//
// This used to mutate `th->meta.schemas[0]` IN PLACE: every existing
// schema-id (including ones old snapshots point at) instantly reflected
// whatever the table's CURRENT shape happened to be, and `n_schemas` never
// left 1. Confirmed against pyiceberg (StaticTable): a snapshot written
// under a 2-field schema, then evolved with `table_add_column`, resolved via
// pyiceberg's own `schemas()[snapshot.schema_id]` to the 3-field shape --
// binding a column to rows written before that column existed.

// Begins an evolution: fills `next` with a DEEP copy of the current schema
// (its own field array) under a fresh schema-id, for the caller to mutate.
// Nothing in `th` changes until commit_schema_evolution.
bool begin_schema_evolution(TableHandle* th, Schema* next) noexcept {
    assert(th != nullptr && next != nullptr);
    if (!write_ready(th)) return false;
    Metadata& m = th->meta;
    if (m.n_schemas == 0) return false;
    const Schema* cur = metadata_current_schema(&m);
    if (cur == nullptr) return false;
    int32_t max_id = cur->schema_id;
    for (uint32_t i = 0; i < m.n_schemas; ++i) {          // bounded
        if (m.schemas[i].schema_id > max_id) max_id = m.schemas[i].schema_id;
    }
    // One spare slot so a field add cannot fail after the copy.
    if (!schema_copy(next, cur, th->arena, m.budget)) return false;
    if (!schema_reserve(next, th->arena, m.budget, next->n_fields + 1u))
        return false;
    next->schema_id = max_id + 1;
    assert(next->schema_id != cur->schema_id);
    return true;
}

// Publishes `next` as the new current schema and persists metadata.json. On a
// failed persist the schema is withdrawn again.
bool commit_schema_evolution(TableHandle* th, const Schema* next) noexcept {
    assert(th != nullptr && next != nullptr);
    const int32_t prev_id = th->meta.current_schema_id;
    Schema* slot = metadata_push_schema(&th->meta);
    if (slot == nullptr) return false;
    *slot = *next;
    th->meta.current_schema_id = next->schema_id;
    th->dirty_meta |= kDirtySchema;
    if (persist_metadata(th)) return true;
    th->meta.n_schemas--;
    th->meta.current_schema_id = prev_id;
    return false;
}

bool table_add_column(TableHandle* th, const char* name, BoltType type,
                      bool nullable) noexcept {
    assert(th != nullptr && name != nullptr);
    if (std::strlen(name) >= sizeof(SchemaField::name)) return false;
    Schema next{};
    Schema* s = &next;
    if (!begin_schema_evolution(th, s)) return false;
    assert(s->n_fields < s->cap_fields);
    SchemaField& f = s->fields[s->n_fields];
    std::memset(&f, 0, sizeof(f));
    // last-column-id + 1: n_fields + 1 reused a dropped column's id.
    int32_t max_id = 0;
    for (uint32_t i = 0; i < th->meta.n_schemas; ++i) {       // bounded
        const Schema& o = th->meta.schemas[i];
        for (uint32_t j = 0; j < o.n_fields; ++j)
            if (o.fields[j].id > max_id) max_id = o.fields[j].id;
    }
    f.id = max_id + 1;
    f.required = !nullable;
    std::strncpy(f.name, name, sizeof(f.name) - 1u);
    std::strncpy(f.type, bolt_type_iceberg_name(type), sizeof(f.type) - 1u);
    s->n_fields++;
    return commit_schema_evolution(th, s);
}

bool table_drop_column(TableHandle* th, const char* name) noexcept {
    assert(th != nullptr && name != nullptr);
    Schema next{};
    Schema* s = &next;
    if (!begin_schema_evolution(th, s)) return false;
    uint32_t out = 0;
    bool dropped = false;
    for (uint32_t i = 0; i < s->n_fields; ++i) {
        if (std::strcmp(s->fields[i].name, name) == 0) { dropped = true; continue; }
        if (out != i) s->fields[out] = s->fields[i];
        ++out;
    }
    if (!dropped) return false;
    s->n_fields = out;
    return commit_schema_evolution(th, s);
}

bool table_rename_column(TableHandle* th, const char* from,
                         const char* to) noexcept {
    assert(th != nullptr && from != nullptr && to != nullptr);
    if (std::strlen(to) >= sizeof(SchemaField::name)) return false;
    Schema next{};
    Schema* s = &next;
    if (!begin_schema_evolution(th, s)) return false;
    for (uint32_t i = 0; i < s->n_fields; ++i) {
        if (std::strcmp(s->fields[i].name, from) == 0) {
            std::memset(s->fields[i].name, 0, sizeof(s->fields[i].name));
            std::strncpy(s->fields[i].name, to, sizeof(s->fields[i].name) - 1u);
            return commit_schema_evolution(th, s);
        }
    }
    return false;
}

bool table_update_partition_spec(TableHandle* th,
                                  const PartitionSpec* spec) noexcept {
    assert(th != nullptr && spec != nullptr);
    if (!write_ready(th)) return false;
    if (th->meta.n_specs >= kIcebergMaxSpecs) return false;
    th->meta.specs[th->meta.n_specs] = *spec;
    th->meta.current_spec_id = spec->spec_id;
    th->meta.n_specs++;
    th->dirty_meta |= kDirtySpec;
    return persist_metadata(th);
}

bool table_update_sort_order(TableHandle* th, const SortOrder* sort) noexcept {
    assert(th != nullptr && sort != nullptr);
    if (!write_ready(th)) return false;
    if (th->meta.n_sort_orders >= kIcebergMaxSortOrders) return false;
    th->meta.sort_orders[th->meta.n_sort_orders] = *sort;
    th->meta.default_sort_order_id = sort->order_id;
    th->meta.n_sort_orders++;
    th->dirty_meta |= kDirtySort;
    return persist_metadata(th);
}

