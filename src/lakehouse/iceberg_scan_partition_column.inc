// iceberg_scan_partition_column.inc — G2ICE-300: the `_partition` metadata column. Included by iceberg_scan.cpp after
// FileTask. SQL has no struct type yet, so the struct is carried as its JSON text, the shape Spark's to_json(_partition)
// gives: the table's unified partition type (every partition field of every spec, ordered by field id; a field the
// file's spec lacks, or a null value, is left out as to_json leaves nulls out), dates (day transform, identity of a date) as "YYYY-MM-DD", other values as numbers or
// strings.

namespace {

// Days since 1970-01-01 -> y, m, d (Howard Hinnant's civil_from_days).
void scan_civil(int64_t z, int64_t* y, uint32_t* m, uint32_t* d) noexcept {
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const uint32_t doe = static_cast<uint32_t>(z - era * 146097);
    const uint32_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const uint32_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const uint32_t mp = (5 * doy + 2) / 153;
    *d = doy - (153 * mp + 2) / 5 + 1;
    *m = mp < 10 ? mp + 3 : mp - 9;
    *y = static_cast<int64_t>(yoe) + era * 400 + (*m <= 2 ? 1 : 0);
}

bool json_put(char* out, uint32_t cap, uint32_t* n, const char* s, uint32_t len) noexcept {
    if (*n + len + 1u > cap) return false;
    std::memcpy(out + *n, s, len);
    *n += len;
    out[*n] = '\0';
    return true;
}

bool json_put_str(char* out, uint32_t cap, uint32_t* n, const char* s, uint32_t len) noexcept {
    if (!json_put(out, cap, n, "\"", 1)) return false;
    for (uint32_t i = 0; i < len; ++i) {                                  // bounded by len
        const unsigned char c = static_cast<unsigned char>(s[i]);
        char esc[8];
        int k = 0;
        if (c == '"' || c == '\\') k = std::snprintf(esc, sizeof(esc), "\\%c", c);
        else if (c < 0x20) k = std::snprintf(esc, sizeof(esc), "\\u%04x", c);
        else { esc[0] = static_cast<char>(c); k = 1; }
        if (!json_put(out, cap, n, esc, static_cast<uint32_t>(k))) return false;
    }
    return json_put(out, cap, n, "\"", 1);
}

bool is_date_field(const PartitionField& f, const Schema* sch) noexcept {
    if (f.transform.kind == TransformKind::kDay) return true;
    if (f.transform.kind != TransformKind::kIdentity || sch == nullptr) return false;
    for (uint32_t i = 0; i < sch->n_fields; ++i)                         // bounded: the schema's fields
        if (sch->fields[i].id == f.source_id) return std::strcmp(sch->fields[i].type, "date") == 0;
    return false;
}

bool json_put_value(char* out, uint32_t cap, uint32_t* n, const PartitionValue& v, bool date) noexcept {
    char buf[64];
    int k = 0;
    if (v.is_null) return json_put(out, cap, n, "null", 4);
    if (v.is_str) return json_put_str(out, cap, n, v.str, partition_value_bytes_len(v));
    if (v.is_float) k = std::snprintf(buf, sizeof(buf), "%.17g", v.f64);
    else if (date) {
        int64_t y = 0;
        uint32_t mo = 0, d = 0;
        scan_civil(v.i64, &y, &mo, &d);
        k = std::snprintf(buf, sizeof(buf), "\"%04lld-%02u-%02u\"", static_cast<long long>(y), mo, d);
    } else k = std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v.i64));
    return k > 0 && json_put(out, cap, n, buf, static_cast<uint32_t>(k));
}

// Bytes the JSON of any file's tuple can take: braces, and per unified field its escaped name (6x), a colon, a comma and
// the widest escaped value (6x a PartitionValue's bytes, or a number / date).
uint32_t partition_json_bound(const Metadata& m) noexcept {
    uint32_t n = 3u;
    for (uint32_t s = 0; s < m.n_specs; ++s)
        for (uint32_t f = 0; f < m.specs[s].n_fields; ++f)
            n += 6u * static_cast<uint32_t>(std::strlen(m.specs[s].fields[f].name)) + 6u * kLakeMaxValBytes + 40u;
    return n;
}

// The JSON of data file `e`'s partition tuple over the table's unified partition type, into `out` (cap bytes).
bool partition_json(const Metadata& m, const DataFileRef& e, char* out, uint32_t cap) noexcept {
    assert(out != nullptr && cap > 2u);
    const PartitionSpec* own = metadata_spec(&m, e.partition_spec_id);
    const Schema* sch = metadata_current_schema(&m);
    uint32_t n = 0;
    out[0] = '\0';
    if (!json_put(out, cap, &n, "{", 1)) return false;
    int32_t last = INT32_MIN;
    for (uint32_t emitted = 0; emitted < kIcebergMaxSpecs * kIcebergMaxFieldsPerSpec; ++emitted) {   // bounded
        const PartitionField* next = nullptr;                             // the smallest field id above `last`
        for (uint32_t s = 0; s < m.n_specs; ++s)
            for (uint32_t f = 0; f < m.specs[s].n_fields; ++f) {
                const PartitionField& pf = m.specs[s].fields[f];
                if (pf.field_id > last && (next == nullptr || pf.field_id < next->field_id)) next = &pf;
            }
        if (next == nullptr) break;
        last = next->field_id;
        const PartitionValue* val = nullptr;
        for (uint32_t f = 0; own != nullptr && f < own->n_fields && f < e.n_partition; ++f)
            if (own->fields[f].field_id == next->field_id) val = &e.partition[f];   // values are the spec's ordinals
        if (val == nullptr || val->is_null) continue;                     // to_json leaves null fields out
        if (n > 1u && !json_put(out, cap, &n, ",", 1)) return false;
        if (!json_put_str(out, cap, &n, next->name, static_cast<uint32_t>(std::strlen(next->name))) || !json_put(out, cap, &n, ":", 1))
            return false;
        if (!json_put_value(out, cap, &n, *val, is_date_field(*next, sch))) return false;
    }
    return json_put(out, cap, &n, "}", 1);
}

}  // namespace
