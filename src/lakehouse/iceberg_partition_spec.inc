// bolt/lakehouse/iceberg_partition_spec.inc — validating and numbering a partition spec a writer adds (G2ICE-292).
// Included inside an anonymous namespace of iceberg_writer.cpp before table_create_ex; not a standalone TU.
//
// Spec ("Partitioning", "Partition Evolution"):
//   * a writer must not commit data under a spec with an unknown transform, so such a spec is refused here, as is a
//     transform the spec does not define for its source type, a source column the current schema lacks (a void
//     field excepted), and a field name that is empty, repeated, or not an Avro name (the manifest's partition
//     record is named by it);
//   * partition field ids are unique across every spec of a v2 table and are REUSED for an equivalent field (same
//     source, transform and name) of an earlier spec; new ones continue from last-partition-id (1000 for the
//     first); a caller's id >= 1000 that no other field holds is kept;
//   * a writer must not add a spec equivalent to an existing one: that spec becomes the default instead.

bool spec_name_ok(const char* n) noexcept {
    assert(n != nullptr);
    if (n[0] == '\0' || (n[0] >= '0' && n[0] <= '9')) return false;
    for (uint32_t i = 0; i < kIcebergMaxFieldName; ++i) {                 // bounded: name width
        const char c = n[i];
        if (c == '\0') return true;
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) return false;
    }
    return false;
}

bool same_field(const PartitionField& a, const PartitionField& b) noexcept {
    return a.source_id == b.source_id && a.transform.kind == b.transform.kind && a.transform.param == b.transform.param &&
           std::strncmp(a.name, b.name, kIcebergMaxFieldName) == 0;
}

bool same_spec(const PartitionSpec& a, const PartitionSpec& b) noexcept {
    if (a.n_fields != b.n_fields) return false;
    for (uint32_t i = 0; i < a.n_fields; ++i)                             // bounded: kIcebergMaxFieldsPerSpec
        if (!same_field(a.fields[i], b.fields[i])) return false;
    return true;
}

// The field id `f` takes: an equivalent existing field's, else the caller's when free, else the next unused one.
int32_t assign_field_id(const Metadata* m, const PartitionSpec* out, uint32_t upto, const PartitionField& f,
                        int32_t* last) noexcept {
    assert(m != nullptr && out != nullptr && last != nullptr);
    bool taken = f.field_id < 1000;
    for (uint32_t s = 0; s < m->n_specs; ++s)                             // bounded: kIcebergMaxSpecs
        for (uint32_t i = 0; i < m->specs[s].n_fields; ++i) {             // bounded: kIcebergMaxFieldsPerSpec
            const PartitionField& e = m->specs[s].fields[i];
            if (same_field(e, f)) return e.field_id;
            if (e.field_id == f.field_id) taken = true;
        }
    for (uint32_t i = 0; i < upto; ++i) taken = taken || out->fields[i].field_id == f.field_id;   // bounded by upto
    if (!taken) { if (f.field_id > *last) *last = f.field_id; return f.field_id; }
    return ++*last;
}

bool spec_field_valid(const Schema* sch, const PartitionSpec* in, uint32_t i) noexcept {
    assert(in != nullptr && i < in->n_fields);
    const PartitionField& f = in->fields[i];
    if (!spec_name_ok(f.name)) return false;
    for (uint32_t j = 0; j < i; ++j)                                      // bounded by i
        if (std::strncmp(in->fields[j].name, f.name, kIcebergMaxFieldName) == 0) return false;
    IcebergTypeInfo src{}, res{};
    bool found = false;
    for (uint32_t k = 0; sch != nullptr && k < sch->n_fields && !found; ++k)   // bounded by n_fields
        if (sch->fields[k].id == f.source_id) found = iceberg_type_parse(sch->fields[k].type, &src) || true;
    if (!found && f.transform.kind != TransformKind::kVoid) return false;
    return partition_result_type(f.transform, src, &res);
}

// `in` checked and numbered against the table's existing specs into `out`. *existing = the id of an equivalent spec
// already in the table (then nothing is added), else -1.
bool normalize_spec(const Metadata* m, const Schema* sch, const PartitionSpec* in, PartitionSpec* out,
                    int32_t* existing) noexcept {
    assert(m != nullptr && in != nullptr && out != nullptr && existing != nullptr);
    *existing = -1;
    if (in->n_fields > kIcebergMaxFieldsPerSpec) return false;
    for (uint32_t i = 0; i < in->n_fields; ++i)                           // bounded: kIcebergMaxFieldsPerSpec
        if (!spec_field_valid(sch, in, i)) {
            std::fprintf(stderr, "[iceberg_writer] partition field %u ('%.*s') is invalid: unknown transform, a transform "
                         "not defined for its source type, a missing source column, or a repeated / non-Avro name\n",
                         i, static_cast<int>(kIcebergMaxFieldName), in->fields[i].name);
            return false;
        }
    for (uint32_t s = 0; s < m->n_specs; ++s)                             // bounded: kIcebergMaxSpecs
        if (same_spec(m->specs[s], *in)) { *existing = m->specs[s].spec_id; *out = m->specs[s]; return true; }
    *out = *in;
    int32_t last = 999, max_spec = -1;
    bool spec_taken = false;
    for (uint32_t s = 0; s < m->n_specs; ++s) {                           // bounded: kIcebergMaxSpecs
        if (m->specs[s].spec_id > max_spec) max_spec = m->specs[s].spec_id;
        spec_taken = spec_taken || m->specs[s].spec_id == in->spec_id;
        for (uint32_t i = 0; i < m->specs[s].n_fields; ++i)               // bounded
            if (m->specs[s].fields[i].field_id > last) last = m->specs[s].fields[i].field_id;
    }
    if (spec_taken || in->spec_id < 0) out->spec_id = max_spec + 1;
    for (uint32_t i = 0; i < in->n_fields; ++i)                           // bounded: kIcebergMaxFieldsPerSpec
        out->fields[i].field_id = assign_field_id(m, out, i, in->fields[i], &last);
    assert(out->n_fields == in->n_fields);
    return true;
}
