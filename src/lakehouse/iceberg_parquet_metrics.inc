// iceberg_parquet_metrics.inc — Iceberg data-file metrics from a parquet file (G2ICE-296). Included at the end of
// iceberg_parquet_stats.cpp: it reuses that file's footer folding for the fixed-width integer types, whose parquet
// statistics are exactly Iceberg's bounds. See bolt/lakehouse/iceberg/metrics.h for the rules.

namespace {

bool ascii_ieq(const char* a, uint32_t an, const char* b) noexcept {
    assert(a != nullptr && b != nullptr);
    const uint32_t bn = static_cast<uint32_t>(std::strlen(b));
    if (an != bn) return false;
    for (uint32_t i = 0; i < an; ++i) {                                  // bounded by an
        const char x = (a[i] >= 'A' && a[i] <= 'Z') ? static_cast<char>(a[i] + 32) : a[i];
        if (x != b[i]) return false;
    }
    return true;
}

// Bytes of the UTF-8 sequence led by `b` (1 for a stray continuation or invalid lead: one unit, never a split).
uint32_t utf8_seq_len(uint8_t b) noexcept {
    if (b < 0x80u) return 1u;
    if ((b & 0xE0u) == 0xC0u) return 2u;
    if ((b & 0xF0u) == 0xE0u) return 3u;
    if ((b & 0xF8u) == 0xF0u) return 4u;
    return 1u;
}

uint32_t utf8_decode(const uint8_t* p, uint32_t n) noexcept {
    assert(p != nullptr && n >= 1u && n <= 4u);
    if (n == 1u) return p[0];
    uint32_t cp = p[0] & (0xFFu >> (n + 1u));
    for (uint32_t i = 1; i < n; ++i) cp = (cp << 6) | (p[i] & 0x3Fu);     // bounded: 4
    return cp;
}

uint32_t utf8_encode(uint32_t cp, uint8_t* out) noexcept {
    assert(out != nullptr && cp <= 0x10FFFFu);
    if (cp < 0x80u) { out[0] = static_cast<uint8_t>(cp); return 1u; }
    if (cp < 0x800u) { out[0] = static_cast<uint8_t>(0xC0u | (cp >> 6)); out[1] = static_cast<uint8_t>(0x80u | (cp & 0x3Fu)); return 2u; }
    if (cp < 0x10000u) {
        out[0] = static_cast<uint8_t>(0xE0u | (cp >> 12));
        out[1] = static_cast<uint8_t>(0x80u | ((cp >> 6) & 0x3Fu));
        out[2] = static_cast<uint8_t>(0x80u | (cp & 0x3Fu));
        return 3u;
    }
    out[0] = static_cast<uint8_t>(0xF0u | (cp >> 18));
    out[1] = static_cast<uint8_t>(0x80u | ((cp >> 12) & 0x3Fu));
    out[2] = static_cast<uint8_t>(0x80u | ((cp >> 6) & 0x3Fu));
    out[3] = static_cast<uint8_t>(0x80u | (cp & 0x3Fu));
    return 4u;
}

// Iceberg-Java UnicodeUtil.incrementCodePoint: skips the surrogate range; 0 = no successor.
uint32_t next_code_point(uint32_t cp) noexcept {
    if (cp == 0xD7FFu) return 0xE000u;
    if (cp >= 0x10FFFFu) return 0u;
    return cp + 1u;
}

// Start of the code point that ends at `end` in p[0..end): a lead byte and up to three continuation bytes after it; a
// stray continuation byte is a unit of its own (as a forward scan would see it).
uint32_t utf8_prev_start(const uint8_t* p, uint32_t end) noexcept {
    assert(p != nullptr && end > 0u);
    uint32_t s = end - 1u;
    for (uint32_t k = 0; k < 3u && s > 0u && (p[s] & 0xC0u) == 0x80u; ++k) --s;   // bounded: 3
    if ((p[s] & 0xC0u) == 0x80u) return end - 1u;
    return s + utf8_seq_len(p[s]) == end ? s : end - 1u;
}

// Increment the truncated string prefix out[0..*n) in place (cap bytes): the last code point that has a successor
// whose encoding still fits is replaced by it and everything after it dropped.
bool utf8_increment(uint8_t* out, uint32_t* n, uint32_t cap) noexcept {
    assert(out != nullptr && n != nullptr);
    uint32_t end = *n;
    while (end > 0u) {                                                    // bounded: *n bytes
        const uint32_t at = utf8_prev_start(out, end);
        const uint32_t l = end - at;
        const uint32_t nxt = next_code_point(l <= 4u ? utf8_decode(out + at, l) : 0x10FFFFu);
        end = at;
        if (nxt == 0u) continue;
        uint8_t enc[4];
        const uint32_t el = utf8_encode(nxt, enc);
        if (at + el > cap) continue;
        std::memcpy(out + at, enc, el);
        *n = at + el;
        return true;
    }
    return false;
}

bool bytes_increment(uint8_t* out, uint32_t* n) noexcept {
    assert(out != nullptr && n != nullptr);
    while (*n > 0u) {                                                     // bounded by *n
        if (out[*n - 1u] != 0xFFu) { ++out[*n - 1u]; return true; }
        --*n;
    }
    return false;
}

}  // namespace

uint32_t truncate_lower_len(bool utf8, const uint8_t* v, uint32_t n, uint32_t units, uint32_t cap) noexcept {
    assert(v != nullptr || n == 0u);
    if (!utf8) return n < units ? (n < cap ? n : cap) : (units < cap ? units : cap);
    uint32_t at = 0, k = 0;
    while (at < n && k < units) {                                        // bounded by n
        const uint32_t l = utf8_seq_len(v[at]);
        const uint32_t step = (at + l <= n) ? l : 1u;
        if (at + step > cap) break;
        at += step;
        ++k;
    }
    return at;
}

bool truncate_upper(bool utf8, const uint8_t* v, uint32_t n, uint32_t units, uint8_t* out, uint32_t cap,
                    uint32_t* out_len) noexcept {
    assert(out != nullptr && out_len != nullptr);
    assert(v != nullptr || n == 0u);
    const uint32_t pl = truncate_lower_len(utf8, v, n, units, cap);
    if (pl > 0u) std::memcpy(out, v, pl);
    *out_len = pl;
    if (pl == n) return true;                                            // nothing was cut: the value bounds itself
    return utf8 ? utf8_increment(out, out_len, cap) : bytes_increment(out, out_len);
}

bool metrics_mode_parse(const char* s, uint32_t len, MetricsModeSpec* out) noexcept {
    assert(out != nullptr);
    if (s == nullptr) return false;
    while (len > 0u && (s[0] == ' ')) { ++s; --len; }
    while (len > 0u && s[len - 1u] == ' ') --len;
    if (ascii_ieq(s, len, "none")) { *out = MetricsModeSpec{MetricsMode::kNone, 0u}; return true; }
    if (ascii_ieq(s, len, "counts")) { *out = MetricsModeSpec{MetricsMode::kCounts, 0u}; return true; }
    if (ascii_ieq(s, len, "full")) { *out = MetricsModeSpec{MetricsMode::kFull, 0u}; return true; }
    if (len < 11u || !ascii_ieq(s, 9u, "truncate(") || s[len - 1u] != ')') return false;
    uint64_t v = 0;
    for (uint32_t i = 9u; i + 1u < len; ++i) {                            // bounded by len
        if (s[i] < '0' || s[i] > '9' || v > UINT32_MAX / 10u) return false;
        v = v * 10u + static_cast<uint32_t>(s[i] - '0');
    }
    if (v == 0u || v > UINT32_MAX) return false;
    *out = MetricsModeSpec{MetricsMode::kTruncate, static_cast<uint32_t>(v)};
    return true;
}

namespace {

// The Iceberg type a parquet leaf was written for (decoded lanes: FLOAT widens to double; UUID / FLBA / decimal raw).
IcebergTypeInfo leaf_type(const pq::PqColumn& c) noexcept {
    IcebergTypeInfo t{};
    const auto lg = static_cast<pq::PqLogical>(c.logical);
    if (lg == pq::PqLogical::Decimal || c.converted == 5) {
        t.type = IcebergType::kDecimal;
        t.precision = static_cast<uint8_t>(c.precision > 0 && c.precision <= 38 ? c.precision : 38);
        t.scale = static_cast<uint8_t>(c.scale > 0 ? c.scale : 0);
        t.fixed_len = decimal_required_bytes(t.precision);                 // the identity value's width; bounds are re-encoded minimal
        return t;
    }
    switch (c.physical) {
        case pq::PqType::Float:  t.type = IcebergType::kFloat; break;
        case pq::PqType::Double: t.type = IcebergType::kDouble; break;
        case pq::PqType::ByteArray:
            t.type = (lg == pq::PqLogical::String || lg == pq::PqLogical::Json || lg == pq::PqLogical::Enum ||
                      c.converted == 0) ? IcebergType::kString : IcebergType::kBinary;
            if (lg == pq::PqLogical::Geometry) t.type = IcebergType::kGeometry;     // G2ICE-327
            if (lg == pq::PqLogical::Geography) t.type = IcebergType::kGeography;
            break;
        case pq::PqType::FixedLenByteArray:
            t.type = lg == pq::PqLogical::Uuid ? IcebergType::kUuid : IcebergType::kFixed;
            t.fixed_len = static_cast<uint32_t>(c.type_length > 0 ? c.type_length : 0);
            break;
        default: t.type = IcebergType::kUnknown; break;   // integers, booleans, temporals: the footer's own bounds
    }
    return t;
}

// What one leaf's values contribute: NaN count and bounds. Strings / binary keep the smallest prefix (lower) and the
// largest prefix one unit longer (upper: it shows whether the largest value was cut).
struct ValueFold {
    IcebergTypeInfo t;
    MetricsModeSpec mode;
    bool            ok = true;      // every value read
    int64_t         nan = 0;
    bool            have = false;
    bool            refused = false; // a bound over iceberg_bound_bytes (or arena exhaustion): the file is refused
    PartitionValue  lo, hi;         // non-byte types: identity values
    Arena*          a = nullptr;    // byte bounds: copies, grown by doubling
    uint8_t*        lo_b = nullptr;
    uint8_t*        hi_b = nullptr;
    uint32_t        lo_n = 0, hi_n = 0, lo_cap = 0, hi_cap = 0;
    kernels::wkb::Bounds geo;       // G2ICE-327: a geometry's x / y box (Iceberg-Java GeometryBoundsBuilder)
};

// Fixed is compared and kept as bytes too (never truncated; unbounded by a PartitionValue's width).
bool is_byte_type(IcebergType t) noexcept {
    return t == IcebergType::kString || t == IcebergType::kBinary || t == IcebergType::kFixed;
}

uint32_t fold_units(const ValueFold& f) noexcept {
    return f.mode.mode == MetricsMode::kTruncate && f.t.type != IcebergType::kFixed ? f.mode.length : UINT32_MAX;
}

uint64_t bound_budget() noexcept { return bolt_limits_value(bolt_limits_id::iceberg_bound_bytes); }

bool fold_keep(ValueFold* f, uint8_t** buf, uint32_t* cap, uint32_t* len, const uint8_t* p, uint32_t n) noexcept {
    assert(f != nullptr && f->a != nullptr);
    if (n > *cap) {
        const uint64_t want = n < 64u ? 64u : (uint64_t{n} * 2u > UINT32_MAX ? n : uint64_t{n} * 2u);
        uint8_t* b = static_cast<uint8_t*>(f->a->allocate(static_cast<size_t>(want), 1));
        if (b == nullptr) { set_resource_exhausted("arena_max_blocks", want, 0); f->refused = true; return false; }
        *buf = b;
        *cap = static_cast<uint32_t>(want);
    }
    if (n > 0u) std::memcpy(*buf, p, n);
    *len = n;
    return true;
}

int bytes_cmp(const uint8_t* a, uint32_t an, const uint8_t* b, uint32_t bn) noexcept {
    const uint32_t n = an < bn ? an : bn;
    const int c = n == 0u ? 0 : std::memcmp(a, b, n);
    return c != 0 ? c : (an > bn) - (an < bn);
}

// The lower bound is the smallest truncated prefix; the upper keeps one more unit so emit_value_bounds can tell a cut
// largest value. A lower over the bound budget, or an upper more than one code point over, refuses the file.
void fold_bytes(ValueFold* f, const uint8_t* p, uint32_t n) noexcept {
    assert(f != nullptr);
    if (f->refused) return;
    const bool u8 = f->t.type == IcebergType::kString;
    const uint32_t units = fold_units(*f);
    const uint32_t ln = truncate_lower_len(u8, p, n, units, UINT32_MAX);
    const uint32_t hn = truncate_lower_len(u8, p, n, units == UINT32_MAX ? units : units + 1u, UINT32_MAX);
    const uint64_t cap = bound_budget();
    if (ln > cap || hn > cap + 4u) {
        set_resource_exhausted("iceberg_bound_bytes", ln > cap ? ln : hn, cap);
        f->refused = true;
        return;
    }
    if (!f->have || bytes_cmp(p, ln, f->lo_b, f->lo_n) < 0) {
        if (!fold_keep(f, &f->lo_b, &f->lo_cap, &f->lo_n, p, ln)) return;
    }
    if (!f->have || bytes_cmp(p, hn, f->hi_b, f->hi_n) > 0) {
        if (!fold_keep(f, &f->hi_b, &f->hi_cap, &f->hi_n, p, hn)) return;
    }
    f->have = true;
}

void fold_row(ValueFold* f, const BoltColumn& c, int64_t r) noexcept {
    assert(f != nullptr);
    if (c.is_null(r) || f->refused) return;
    if (f->t.type == IcebergType::kGeography) return;                  // Iceberg-Java keeps counts only
    if (f->t.type == IcebergType::kGeometry) {
        if (c.format != ColumnFormat::Flat && c.format != ColumnFormat::View && c.format != ColumnFormat::VarBinary) { f->ok = false; return; }
        const uint8_t* p = nullptr;
        int32_t n = 0;
        c.utf8_at(r, &p, &n);
        if (n < 0) f->geo.incomplete = true;
        else (void)kernels::wkb::bounds_add(&f->geo, p, static_cast<size_t>(n));
        return;
    }
    if (f->t.type == IcebergType::kFixed && (c.type == BoltType::FixedSizeBinary || c.type == BoltType::UUID)) {
        // fixed[L <= 16] decodes into 16-byte slots (wider decodes as variable bytes, below)
        if (c.data == nullptr || f->t.fixed_len == 0u || f->t.fixed_len > 16u) { f->ok = false; return; }
        fold_bytes(f, static_cast<const uint8_t*>(c.data) + static_cast<uint64_t>(r) * 16u, f->t.fixed_len);
        return;
    }
    if (is_byte_type(f->t.type)) {
        if (c.format != ColumnFormat::Flat && c.format != ColumnFormat::View && c.format != ColumnFormat::VarBinary) { f->ok = false; return; }
        const uint8_t* p = nullptr;
        int32_t n = 0;
        c.utf8_at(r, &p, &n);
        if (n < 0) { f->ok = false; return; }
        fold_bytes(f, p, static_cast<uint32_t>(n));
        return;
    }
    PartitionField id{};
    id.transform.kind = TransformKind::kIdentity;
    PartitionValue v;
    if (!partition_value_of(id, f->t, c, r, &v)) { f->ok = false; return; }
    if (v.is_null) return;
    if (v.is_float && std::isnan(v.f64)) { ++f->nan; return; }
    if (!f->have || partition_value_compare(f->t, v, f->lo, false) < 0) f->lo = v;
    if (!f->have || partition_value_compare(f->t, v, f->hi, false) > 0) f->hi = v;
    f->have = true;
}

}  // namespace

namespace {

// Every row group's values of leaf `c` folded into `f` (one decode per row group, its buffers released after).
void fold_leaf_values(const uint8_t* bytes, uint64_t len, const pq::PqMeta* meta, uint16_t c, ValueFold* f) noexcept {
    assert(meta != nullptr && f != nullptr);
    Arena a;
    for (uint32_t g = 0; g < meta->n_row_groups && f->ok && !f->refused; ++g) {        // bounded: kPqMaxRowGroups
        a.reset();
        BoltColumn col{};
        int64_t rows = 0;
        if (!pq::parquet_read_row_group_cols(bytes, len, meta, g, &c, 1u, &a, &col, &rows)) { f->ok = false; break; }
        for (int64_t r = 0; r < rows && r < col.length; ++r) fold_row(f, col, r);   // bounded: the row group
    }
}

// The bounds a value fold proves, in Iceberg's single-value encoding. False = refused (ResourceExhausted is set).
bool emit_value_bounds(const ValueFold& f, ColumnStatEntry* e, Arena* a) noexcept {
    assert(e != nullptr && a != nullptr);
    if (f.refused) return false;
    if (f.t.type == IcebergType::kGeometry) {                           // G2ICE-327: (xmin, ymin) / (xmax, ymax)
        uint8_t lo[16], hi[16];
        if (!f.ok || f.mode.mode == MetricsMode::kCounts || !kernels::wkb::iceberg_xy_bounds(f.geo, lo, hi)) return true;
        return stats_set_bound(e, a, true, lo, 16u) && stats_set_bound(e, a, false, hi, 16u);
    }
    if (f.t.type == IcebergType::kGeography) return true;
    if (!f.ok || !f.have || f.mode.mode == MetricsMode::kCounts) return true;
    if (is_byte_type(f.t.type)) {
        if (!stats_set_bound(e, a, true, f.lo_b, f.lo_n)) return false;
        uint8_t* up = static_cast<uint8_t*>(a->allocate(f.hi_n + 4u, 1));
        if (up == nullptr) { set_resource_exhausted("arena_max_blocks", f.hi_n + 4u, 0); return false; }
        uint32_t un = 0;
        const bool u8 = f.t.type == IcebergType::kString;
        if (truncate_upper(u8, f.hi_b, f.hi_n, fold_units(f), up, f.hi_n + 4u, &un)) return stats_set_bound(e, a, false, up, un);
        return true;
    }
    uint8_t lo[kLakeMaxValBytes], hi[kLakeMaxValBytes];
    uint32_t ln = 0, hn = 0;
    if (f.t.type == IcebergType::kFloat) {                               // decoded as double: back to its 4 bytes
        const float x = static_cast<float>(f.lo.f64), y = static_cast<float>(f.hi.f64);
        std::memcpy(lo, &x, 4); std::memcpy(hi, &y, 4);
        return stats_set_bound(e, a, true, lo, 4u) && stats_set_bound(e, a, false, hi, 4u);
    }
    if (!partition_value_encode(f.t, f.lo, lo, kLakeMaxValBytes, &ln) || !partition_value_encode(f.t, f.hi, hi, kLakeMaxValBytes, &hn))
        return true;                                                     // identity types are at most 16 bytes
    return stats_set_bound(e, a, true, lo, ln) && stats_set_bound(e, a, false, hi, hn);
}

// Sizes and counts of leaf `c` from the footer. False = a malformed footer.
bool footer_counts(const pq::PqMeta* meta, uint32_t c, bool counts, ColumnStatEntry* e) noexcept {
    assert(meta != nullptr && e != nullptr);
    int64_t size = 0, values = 0, nulls = 0;
    bool nulls_known = true;
    for (uint32_t g = 0; g < meta->n_row_groups; ++g) {                 // bounded: kPqMaxRowGroups
        const pq::PqRowGroup& rg = meta->row_groups[g];
        if (c >= rg.chunk_count || rg.chunk_off + c >= meta->n_chunks) return false;
        const pq::PqChunk& ch = meta->chunks[rg.chunk_off + c];
        size += ch.total_compressed_size;
        values += ch.num_values;
        if (ch.null_count < 0) nulls_known = false; else nulls += ch.null_count;
    }
    e->has_column_size = true;
    e->column_size = size;
    e->no_null_count = !(counts && nulls_known);
    e->null_count = e->no_null_count ? 0 : nulls;
    e->has_value_count = counts;
    e->value_count = counts ? values : 0;
    return true;
}

// The integer-like bounds (booleans, ints, longs, dates, times, timestamps) straight from the footer statistics.
bool footer_bounds(const pq::PqMeta* meta, uint32_t c, ColumnStatEntry* e, Arena* a) noexcept {
    assert(meta != nullptr && e != nullptr);
    const BoundKind k = bound_kind(meta->columns[c]);
    if (k == BoundKind::kNone || k == BoundKind::kFloat || k == BoundKind::kDouble || k == BoundKind::kBytes) return true;
    ColFold f{};
    f.nulls_known = true;
    f.lo_ok = f.hi_ok = true;
    for (uint32_t g = 0; g < meta->n_row_groups; ++g) {                 // bounded: kPqMaxRowGroups
        const pq::PqRowGroup& rg = meta->row_groups[g];
        fold_chunk(k, meta->chunks[rg.chunk_off + c], &f);
    }
    return emit_bounds(k, f, e, a);
}

// One leaf's entry. False = a malformed footer or a refused budget.
bool leaf_metrics(const uint8_t* bytes, uint64_t len, const pq::PqMeta* meta, uint32_t c, const MetricsLeaf& lf,
                  ColumnStatEntry* e, Arena* a) noexcept {
    assert(meta != nullptr && e != nullptr && a != nullptr);
    const pq::PqColumn& col = meta->columns[c];
    const bool repeated = col.max_rep != 0u;                            // inside a list or map: its size only
    if (!footer_counts(meta, c, !repeated, e)) return false;
    if (col.variant_leaf != 0u) {                                       // G2ICE-327: Iceberg-Java's variant metrics: the group's
        e->has_column_size = false;                                     // value / null counts (from its metadata leaf), no size,
        e->column_size = 0;                                             // no bounds (shredded bounds: G2ICE-310)
        return true;
    }
    if (repeated) return true;
    ValueFold vf{};
    vf.t = leaf_type(col);
    vf.mode = lf.mode;
    vf.a = a;
    const bool floats = vf.t.type == IcebergType::kFloat || vf.t.type == IcebergType::kDouble;
    const bool bounds = lf.mode.mode == MetricsMode::kTruncate || lf.mode.mode == MetricsMode::kFull;
    if (vf.t.type == IcebergType::kUnknown) {
        return !bounds || footer_bounds(meta, c, e, a);
    }
    if ((!floats && !bounds) || vf.t.type == IcebergType::kGeography) return true;   // G2ICE-327: geography: counts only
    fold_leaf_values(bytes, len, meta, static_cast<uint16_t>(c), &vf);
    if (floats && vf.ok && !vf.refused) { e->has_nan_count = true; e->nan_count = vf.nan; }
    return emit_value_bounds(vf, e, a);
}

// split_offsets: each row group's first byte (its first column's dictionary page, else first data page).
bool split_offsets(const pq::PqMeta* meta, FileStats* out, Arena* a) noexcept {
    assert(meta != nullptr && out != nullptr);
    out->n_split_offsets = 0;
    out->split_offsets = nullptr;
    if (meta->n_row_groups == 0u || meta->n_columns == 0u) return true;
    int64_t* offs = a->allocate_array<int64_t>(meta->n_row_groups);
    if (offs == nullptr) { set_resource_exhausted("arena_max_blocks", uint64_t{meta->n_row_groups} * 8u, 0); return false; }
    for (uint32_t g = 0; g < meta->n_row_groups; ++g) {                 // bounded: kPqMaxRowGroups
        const pq::PqRowGroup& rg = meta->row_groups[g];
        if (rg.chunk_count == 0u || rg.chunk_off >= meta->n_chunks) return true;   // no split offsets for this footer
        const pq::PqChunk& ch = meta->chunks[rg.chunk_off];
        offs[g] = ch.dictionary_page_offset > 0 && ch.dictionary_page_offset < ch.data_page_offset
                      ? ch.dictionary_page_offset : ch.data_page_offset;
    }
    out->split_offsets = offs;
    out->n_split_offsets = meta->n_row_groups;
    return true;
}

}  // namespace

bool file_metrics_from_parquet(const uint8_t* bytes, uint64_t len, const pq::PqMeta* meta,
                               const MetricsLeaf* leaves, uint32_t n_leaves, Arena* a, FileStats* out) noexcept {
    assert(out != nullptr && a != nullptr);
    if (bytes == nullptr || meta == nullptr || out == nullptr || a == nullptr || (leaves == nullptr && n_leaves != 0u)) return false;
    if (meta->n_columns > kMaxColumns || meta->n_row_groups > pq::kPqMaxRowGroups) return false;
    if (meta->n_row_groups > 0u && meta->chunks == nullptr) return false;
    out->n_cols = 0;
    const uint32_t nc = meta->n_columns < n_leaves ? meta->n_columns : n_leaves;
    for (uint32_t c = 0; c < nc; ++c) {                                  // bounded: kMaxColumns
        if (leaves[c].field_id <= 0 || leaves[c].mode.mode == MetricsMode::kNone) continue;
        if (meta->columns[c].variant_leaf > 1u) continue;                // G2ICE-327: a variant counts once, on its metadata leaf
        ColumnStatEntry* e = stats_add_col(out, a, leaves[c].field_id);
        if (e == nullptr) return false;                                  // over iceberg_stat_columns
        if (!leaf_metrics(bytes, len, meta, c, leaves[c], e, a)) return false;
    }
    if (!split_offsets(meta, out, a)) return false;
    out->has_sort_order_id = true;                                       // synthesized rows follow no table sort order
    out->sort_order_id = 0;
    return true;
}

namespace {

#include "iceberg_batch_metrics.inc"

}  // namespace

bool file_metrics_from_batches(const BoltBatch* const* batches, uint32_t n_batches, const Schema* sch, const MetricsModeSpec& mode,
                               Arena* a, FileStats* out) noexcept {
    assert(out != nullptr && a != nullptr);
    if (batches == nullptr || sch == nullptr || out == nullptr || a == nullptr || n_batches == 0u) return false;
    out->n_cols = 0;
    out->n_split_offsets = 0;
    out->split_offsets = nullptr;
    if (mode.mode == MetricsMode::kNone) return true;
    if(sch->n_fields>kMaxColumns || (sch->n_fields && !sch->fields))return false;
    uint32_t path[kIcebergSchemaMaxDepth+1u]{},visited=0;
    for(uint32_t i=0;i<sch->n_fields;++i) {
        if(!batch_field_metrics(batches,n_batches,sch->fields[i].name,sch->fields[i],path,0,&visited,mode,out,a))return false;
    }
    return true;
}
