// Private helpers, included after the table/scan definitions.
bool pinned_open_admit(TableHandle** out, Arena* owner, Arena* metadata,
    PinnedObjectStore* source, const PinnedTableRead* read,
    const uint8_t* bytes, uint64_t length) noexcept {
    assert(sizeof(size_t) <= sizeof(uint64_t)); assert(kOsPinnedMaxEntries <= 65536);
    using detail::ObjectSpan;
    if (out == nullptr || owner == nullptr || metadata == nullptr || source == nullptr ||
        read == nullptr || bytes == nullptr || length == 0 || length > SIZE_MAX ||
        source->count > kOsPinnedMaxEntries || source->io == nullptr || owner == metadata)
        return false;
    const ObjectSpan retained[]{{source,sizeof(*source)},
        {source->entries,static_cast<size_t>(source->count)*sizeof(PinnedObject)},
        {source->origin.vt,sizeof(ObjectStoreVT)}, {source->origin.impl,1},
        {source->io,sizeof(Arena)}, {read,sizeof(*read)},
        {bytes,static_cast<size_t>(length)}, {out,sizeof(*out)}};
    const ObjectSpan arenas[]{{owner,sizeof(Arena)},{metadata,sizeof(Arena)}};
    // Complete raw object-span admission precedes inspecting proposed arenas.
    for (const auto span : retained) if (!detail::object_span_valid(span)) return false;
    if (!detail::object_span_valid(arenas[0]) || !detail::object_span_valid(arenas[1]) ||
        detail::object_spans_overlap(arenas[0],arenas[1])) return false;
    for (const auto arena : arenas) {
        for (const auto span : retained)
            if (detail::object_spans_overlap(arena,span)) return false;
        if (source->io->overlaps_backing(arena.data,arena.size)) return false;
    }
    for (uint32_t i=0; i+1<sizeof(retained)/sizeof(retained[0]); ++i)
        if (detail::object_spans_overlap(retained[i],retained[7])) return false;
    if (source->io->overlaps_backing(out,sizeof(*out)) ||
        source->io->overlaps_backing(read,sizeof(*read)) ||
        source->io->overlaps_backing(bytes,static_cast<size_t>(length))) return false;
    if (owner->growth_enabled() || metadata->growth_enabled() || owner->concurrent() || metadata->concurrent())
        return false;
    Arena* workspaces[]{owner,metadata};
    for (Arena* arena : workspaces) {
        for (const auto span : retained)
            if (arena->overlaps_backing(span.data,span.size)) return false;
        if (arena->overlaps_backing(arena == owner ? metadata : owner,sizeof(Arena))) return false;
    }
    return true;
}

bool pinned_read_valid(PinnedObjectStore* source, const PinnedTableRead& read,
    ObjectStore* view) noexcept {
    assert(source != nullptr && view != nullptr); assert(sizeof(read.sha256)==65);
    if (pinned_object_store_view(source,view) != kOsOk ||
        read.parquet_metadata_bytes == 0 || read.parquet_metadata_bytes > (1u<<30) ||
        read.manifest_bytes == 0 || read.manifest_bytes > (1u<<30) ||
        (read.selection != SnapshotSelection::Exact && read.selection != SnapshotSelection::Empty) ||
        (read.selection == SnapshotSelection::Empty && read.snapshot_id != 0) ||
        read.sha256[64] != 0) return false;
    for (uint32_t i=0; i<64; ++i)
        if (!((read.sha256[i]>='0' && read.sha256[i]<='9') ||
              (read.sha256[i]>='a' && read.sha256[i]<='f'))) return false;
    detail::ObjectSpan root{};
    if (!detail::object_key_span(read.object.store_location,sizeof(read.object.store_location),&root) ||
        std::strcmp(source->namespace_uri,read.object.store_location) != 0) return false;
    char key[kOsMaxKey];
    if (s3_uri_object_key(read.object.store_location,read.object.table_location,key,sizeof(key)) !=
        S3UriStatus::Ok) return false;
    // Directory HEAD is a local copied pin, never origin discovery.
    ObjectVersion pin{};
    return os_head_versioned(view,read.object.object_key,source->io,&pin)==kOsOk &&
        pin.size==read.object.version.size &&
        std::strcmp(pin.etag,read.object.version.etag)==0;
}

bool open_pinned_table(TableHandle** out, Arena* owner, Arena* metadata,
    PinnedObjectStore* source, const PinnedTableRead* read,
    const uint8_t* bytes, uint64_t length) noexcept {
    assert(sizeof(TableHandle)>0); assert(sizeof(MetadataLoadResult)>0);
    if (!pinned_open_admit(out,owner,metadata,source,read,bytes,length)) return false;
    ObjectStore view{};
    // Validate exact metadata identity before comparing any untrusted strings.
    Metadata parsed{}; MetadataLoadResult validated{};
    if (metadata_validate_exact(&read->object,&read->metadata_limits,bytes,length,
            metadata,&parsed,&validated)!=MetadataLoadStatus::Ok ||
        std::memcmp(validated.sha256,read->sha256,sizeof(read->sha256))!=0 ||
        !pinned_read_valid(source,*read,&view)) return false;
    Snapshot selected{};
    if (read->selection==SnapshotSelection::Empty ? parsed.current_snapshot_present :
        !snapshot_resolve_selected(&parsed,read->snapshot_id,-1,read->selection,&selected)) return false;
    if (read->selection==SnapshotSelection::Exact) {
        uint32_t matches=0;
        for (uint32_t i=0; i<parsed.n_snapshots; ++i)
            if (parsed.snapshots[i].snapshot_id==read->snapshot_id) ++matches;
        if (matches!=1) return false;
    }
    auto* h=owner->allocate_array<TableHandle>(1);
    if (h==nullptr) return false;
    std::memset(h,0,sizeof(*h));
    h->arena=owner; h->os=view; h->meta=parsed; h->meta_loaded=true;
    h->strict_pinned=true; h->selection=read->selection; h->selected_snapshot=read->snapshot_id;
    h->parquet_metadata_bytes=read->parquet_metadata_bytes; h->manifest_bytes=read->manifest_bytes;
    h->retained_arena=metadata; h->pinned_source=source;
    h->request_bytes=source->limits.request_bytes;
    std::memcpy(h->fs_root,read->object.store_location,sizeof(read->object.store_location));
    *out=h;
    assert(h->meta_loaded && h->strict_pinned); return true;
}

bool table_read_ref(TableHandle* h, const char* path, Arena* scratch,
    const uint8_t** body, uint64_t* length) noexcept {
    assert(h!=nullptr && path!=nullptr); assert(scratch!=nullptr);
    if (!h->strict_pinned)
        return read_ref(&h->os,h->meta.location,h->fs_root,h->table_rel,path,scratch,body,length);
    char key[kOsMaxKey];
    return s3_uri_object_key(h->fs_root,path,key,sizeof(key))==S3UriStatus::Ok &&
        os_get(&h->os,key,scratch,body,length)==kOsOk;
}

bool pinned_scan_options(ScanHandle* s, const ReadOptions* supplied) noexcept {
    assert(s!=nullptr && s->table!=nullptr); assert(s->table->strict_pinned);
    const auto* h=s->table;
    if (supplied==nullptr) {
        s->opts.snapshot_selection=h->selection; s->opts.snapshot_id=h->selected_snapshot;
    }
    // These options are not yet implemented by this scanner: refuse rather
    // than silently returning an unprojected/unfiltered alternative result.
    return s->opts.snapshot_selection==h->selection && s->opts.snapshot_id==h->selected_snapshot &&
        s->opts.timestamp_ms<0 && s->opts.n_projection==0 && !s->opts.include_partition_columns &&
        !s->opts.include_deletes && s->opts.n_predicates==0;
}

bool pinned_scan_io_admit(TableHandle* h, const detail::ObjectSpan* inputs, uint32_t count,
    detail::ObjectSpan output, Arena* scratch) noexcept {
    assert(h!=nullptr && h->pinned_source!=nullptr); assert(count<=3);
    using detail::ObjectSpan;
    const auto* source=h->pinned_source;
    const ObjectSpan protected_spans[]{{h,sizeof(*h)}, {source,sizeof(*source)},
        {source->entries,static_cast<size_t>(source->count)*sizeof(PinnedObject)},
        {source->origin.vt,sizeof(ObjectStoreVT)}, {h->os.vt,sizeof(ObjectStoreVT)},
        {source->io,sizeof(Arena)}, {h->arena,sizeof(Arena)}, {h->retained_arena,sizeof(Arena)}};
    if (!detail::object_span_valid(output)) return false;
    for (uint32_t i=0; i<count; ++i) {
        if (!detail::object_span_valid(inputs[i])) return false;
        for (const auto span : protected_spans)
            if (detail::object_spans_overlap(inputs[i],span)) return false;
        if (h->arena->overlaps_backing(inputs[i].data,inputs[i].size) ||
            h->retained_arena->overlaps_backing(inputs[i].data,inputs[i].size) ||
            source->io->overlaps_backing(inputs[i].data,inputs[i].size)) return false;
    }
    for (const auto span : protected_spans)
        if (detail::object_spans_overlap(output,span) ||
            (scratch!=nullptr && detail::object_spans_overlap({scratch,sizeof(Arena)},span))) return false;
    if (h->arena->overlaps_backing(output.data,output.size) ||
        h->retained_arena->overlaps_backing(output.data,output.size) ||
        source->io->overlaps_backing(output.data,output.size)) return false;
    if (scratch==nullptr) return true;
    const ObjectSpan object{scratch,sizeof(Arena)};
    if (!detail::object_span_valid(object) || h->arena->overlaps_backing(object.data,object.size) ||
        h->retained_arena->overlaps_backing(object.data,object.size) ||
        source->io->overlaps_backing(object.data,object.size)) return false;
    for (uint32_t i=0; i<count; ++i)
        if (!detail::object_span_valid(inputs[i]) || detail::object_spans_overlap(object,inputs[i])) return false;
    if (detail::object_spans_overlap(object,output)) return false;
    for (const auto span : protected_spans)
        if (scratch->overlaps_backing(span.data,span.size)) return false;
    return true;
}

bool select_pinned_schema(ScanHandle* s) noexcept {
    assert(s!=nullptr && s->table!=nullptr); assert(s->table->strict_pinned);
    const auto& meta=s->table->meta;
    const Schema* selected=nullptr;
    const int32_t selected_id=s->snap.schema_id_present ? s->snap.schema_id : meta.current_schema_id;
    for (uint32_t i=0; i<meta.n_schemas; ++i) {
        if (meta.schemas[i].schema_id!=selected_id) continue;
        if (selected!=nullptr) return false;
        selected=&meta.schemas[i];
    }
    if (selected==nullptr || selected->n_fields==0 || selected->n_fields>kMaxColumns) return false;
    for (uint32_t i=0; i<selected->n_fields; ++i) {
        const auto& field=selected->fields[i];
        if (field.id<=0 || field.name[0]==0 || std::strchr(field.name,'\\')!=nullptr ||
            std::strcmp(field.type,"struct")==0) return false;
        for (uint32_t j=0; j<i; ++j)
            if (field.id==selected->fields[j].id || std::strcmp(field.name,selected->fields[j].name)==0)
                return false;
    }
    s->selected_schema=selected; return true;
}

bool pinned_type_matches(const SchemaField& field, const pq::PqColumn& column) noexcept {
    return schema_leaf_matches_parquet(field,column);
}

// The decoded type an Iceberg primitive type implies; Invalid for types this check does not model (decimal, binary, nested, ...).
BoltType want_bolt_type(const char* t) noexcept {
    assert(t!=nullptr);
    if (std::strcmp(t,"long")==0) return BoltType::Int64;
    if (std::strcmp(t,"int")==0) return BoltType::Int32;
    if (std::strcmp(t,"double")==0) return BoltType::Float64;
    if (std::strcmp(t,"float")==0) return BoltType::Float64;   // the parquet decoder widens FLOAT to Float64 (G2ICE-379)
    if (std::strcmp(t,"boolean")==0) return BoltType::Bool;
    if (std::strcmp(t,"string")==0) return BoltType::Utf8;
    if (std::strcmp(t,"binary")==0) return BoltType::Binary;
    unsigned fixed = 0;
    if (std::sscanf(t,"fixed[%u]",&fixed)==1 && fixed>16u) return BoltType::Binary;
    if (std::strcmp(t,"date")==0) return BoltType::Date32;
    return BoltType::NA;
}

// Map this file's columns onto the table's current schema by field id (files without any field ids fall back to the column name, the
// spec's default name mapping). A schema column the file lacks reads as null (evolved schemas); a file column the schema lacks is dropped.
bool bind_current_schema(ScanHandle* s, const pq::PqMeta* meta) noexcept {
    assert(s!=nullptr && meta!=nullptr);
    const Schema* cur=metadata_current_schema(&s->table->meta);
    s->use_map=false;
    if (cur==nullptr || cur->n_fields==0) return true;                  // no schema to conform to: the file's own layout
    if (cur->n_fields>kMaxColumns || meta->n_columns>kMaxColumns) return false;
    bool file_has_ids=false;
    for (uint32_t i=0; i<meta->n_columns; ++i) file_has_ids=file_has_ids||meta->columns[i].has_field_id;
    if (s->colmap==nullptr || s->map_schema!=cur) {
        s->colmap=s->scratch->allocate_array<int32_t>(cur->n_fields);
        s->want_type=s->scratch->allocate_array<BoltType>(cur->n_fields);
        if (s->colmap==nullptr || s->want_type==nullptr) return false;
        s->map_schema=cur;
        for (uint32_t j=0; j<cur->n_fields; ++j) s->want_type[j]=want_bolt_type(cur->fields[j].type);
    }
    // Files written without field ids bind through the table's default name mapping (spec), falling back to the current column name only
    // when the table carries no mapping.
    if (!file_has_ids && !s->nm_loaded) {
        s->nm_loaded=true;
        if (!metadata_name_mapping(&s->table->meta,s->scratch,&s->nm_entries,&s->nm_count)) return false;
    }
    for (uint32_t j=0; j<cur->n_fields; ++j) {
        int32_t found=-1;
        const NameMapEntry* nm=nullptr;
        if (!file_has_ids) for (uint32_t e=0; e<s->nm_count; ++e) if (s->nm_entries[e].field_id==cur->fields[j].id) nm=&s->nm_entries[e];
        for (uint32_t i=0; i<meta->n_columns; ++i) {
            const auto& col=meta->columns[i];
            bool by_name=false;
            if (!file_has_ids) {
                if (nm!=nullptr) { for (uint32_t k=0; k<nm->n_names; ++k) by_name=by_name||std::strcmp(col.name,nm->names[k])==0; }
                else if (s->nm_count==0) by_name=std::strcmp(col.name,cur->fields[j].name)==0;
            }
            const bool hit=file_has_ids ? (col.has_field_id && col.field_id==cur->fields[j].id) : by_name;
            if (!hit) continue;
            if (found>=0) return false;                                  // two file columns claim one field: ambiguous, never guess
            found=static_cast<int32_t>(i);
        }
        s->colmap[j]=found;
    }
    s->use_map=true;
    s->decode_columns=meta->n_columns;
    s->output_columns=cur->n_fields;
    return true;
}

// Widen a decoded column to the type the table schema now declares (spec promotions: int->long, float->double).
bool promote_column(const BoltColumn& src, BoltType want, int64_t rows, Arena* a, BoltColumn* out, const char* what) noexcept {
    assert(a!=nullptr && out!=nullptr);
    // Representation-equivalent decodes are not a mismatch: a parquet boolean decodes as a one-byte integer, a date as int32/date32, a
    // string as utf8 or binary.
    // (bolt's parquet decoder returns a boolean as an integer column; the spec allows no evolution to or from boolean)
    const auto byte_like=[](BoltType t){ return t==BoltType::Bool || t==BoltType::Int8 || t==BoltType::UInt8 || t==BoltType::Int64; };
    const auto date_like=[](BoltType t){ return t==BoltType::Date32 || t==BoltType::Int32; };
    const bool same_repr=(want==BoltType::Bool && byte_like(src.type)) || (date_like(want) && date_like(src.type) && want==BoltType::Date32) ||
        (want==BoltType::Utf8 && src.type==BoltType::Binary);
    if (want==BoltType::Binary && src.type==BoltType::Utf8) { *out=src;out->type=BoltType::Binary;return true; }
    if (want==BoltType::NA || src.type==want || same_repr) { *out=src; return true; }
    if (src.format==ColumnFormat::Flat && src.type==BoltType::Int32 && want==BoltType::Int64) {
        auto* d=a->allocate_array<int64_t>(static_cast<size_t>(rows)+1u);
        if (d==nullptr) return false;
        const int32_t* in=static_cast<const int32_t*>(src.data);
        for (int64_t r=0; r<rows; ++r) d[r]=in[r];
        *out=src; out->data=d; out->type=BoltType::Int64; out->type_size_bytes=sizeof(int64_t); return true;
    }
    if (src.format==ColumnFormat::Flat && src.type==BoltType::Float32 && want==BoltType::Float64) {
        auto* d=a->allocate_array<double>(static_cast<size_t>(rows)+1u);
        if (d==nullptr) return false;
        const float* in=static_cast<const float*>(src.data);
        for (int64_t r=0; r<rows; ++r) d[r]=static_cast<double>(in[r]);
        *out=src; out->data=d; out->type=BoltType::Float64; out->type_size_bytes=sizeof(double); return true;
    }
    std::fprintf(stderr,"[iceberg_scan] REFUSED: column '%s' is decoded as type %d but the table schema declares type %d, and that is not a spec promotion\n",
                 what,static_cast<int>(src.type),static_cast<int>(want));
    return false;
}

// "YYYY-MM-DD" as days since 1970-01-01 (proleptic Gregorian).
bool date_days(const char* t, int32_t* out) noexcept {
    int y = 0, m = 0, d = 0, used = 0;
    if (std::sscanf(t, "%d-%d-%d%n", &y, &m, &d, &used) != 3 || t[used] != '\0' || m < 1 || m > 12 || d < 1 || d > 31) return false;
    y -= m <= 2 ? 1 : 0;
    const int64_t era = (y >= 0 ? y : y - 399) / 400, yoe = y - era * 400;
    const int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    *out = static_cast<int32_t>(era * 146097 + doe - 719468);
    return true;
}

// G2ICE-326: the lane a column absent from a data file is filled in, for types `want_bolt_type` passes through (the type the
// parquet decoder gives a present column of it); `fixed` is fixed[L]'s L, `scale` a decimal's.
BoltType fill_bolt_type(const char* t, uint32_t* fixed, uint32_t* scale) noexcept {
    assert(t != nullptr && fixed != nullptr && scale != nullptr);
    *fixed = 0u;
    *scale = 0u;
    unsigned p = 0, sc = 0, len = 0;
    if (std::strcmp(t, "time") == 0) return BoltType::Duration;
    if (std::strcmp(t, "timestamp") == 0 || std::strcmp(t, "timestamptz") == 0) return BoltType::Timestamp;
    if (std::strcmp(t, "uuid") == 0) return BoltType::UUID;
    if (std::strcmp(t, "binary") == 0) return BoltType::Binary;
    if (std::sscanf(t, "decimal(%u,%u)", &p, &sc) == 2 && sc <= p && p <= 38u) {
        *scale = sc;
        return p <= 18u ? BoltType::Decimal64 : BoltType::Decimal128;
    }
    if (std::sscanf(t, "fixed[%u]", &len) == 1 && len > 0u) {
        *fixed = len;
        return len <= 16u ? BoltType::FixedSizeBinary : BoltType::Binary;
    }
    return BoltType::NA;
}

// A JSON string body unescaped (\uXXXX with surrogate pairs to UTF-8) into `out` (cap bytes); its length, or -1 when malformed.
int32_t json_unescape_into(const char* in, char* out, uint32_t cap) noexcept {
    uint32_t n = 0;
    const auto put = [&](uint32_t b) { if (n + 1u >= cap) return false; out[n++] = static_cast<char>(b); return true; };
    const auto hex4 = [](const char* h, uint32_t* v) {
        char tmp[5] = {h[0], h[1], h[2], h[3], '\0'};
        char* e = nullptr;
        if (std::strlen(tmp) != 4u) return false;
        *v = static_cast<uint32_t>(std::strtoul(tmp, &e, 16));
        return *e == '\0';
    };
    for (const char* q = in; *q != '\0'; ++q) {                               // bounded: the text
        if (*q != '\\') { if (!put(static_cast<unsigned char>(*q))) return -1; continue; }
        const char c = *++q;
        uint32_t cp = 0;
        switch (c) {
            case '"': case '\\': case '/': cp = static_cast<unsigned char>(c); break;
            case 'b': cp = '\b'; break;
            case 'f': cp = '\f'; break;
            case 'n': cp = '\n'; break;
            case 'r': cp = '\r'; break;
            case 't': cp = '\t'; break;
            case 'u': {
                if (!hex4(q + 1, &cp)) return -1;
                q += 4;
                if (cp >= 0xD800u && cp <= 0xDBFFu) {
                    uint32_t lo = 0;
                    if (q[1] != '\\' || q[2] != 'u' || !hex4(q + 3, &lo) || lo < 0xDC00u || lo > 0xDFFFu) return -1;
                    cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                    q += 6;
                } else if (cp >= 0xDC00u && cp <= 0xDFFFu) {
                    return -1;
                }
                break;
            }
            default: return -1;
        }
        bool ok = true;
        if (cp < 0x80u) ok = put(cp);
        else if (cp < 0x800u) ok = put(0xC0u | (cp >> 6)) && put(0x80u | (cp & 0x3Fu));
        else if (cp < 0x10000u) ok = put(0xE0u | (cp >> 12)) && put(0x80u | ((cp >> 6) & 0x3Fu)) && put(0x80u | (cp & 0x3Fu));
        else ok = put(0xF0u | (cp >> 18)) && put(0x80u | ((cp >> 12) & 0x3Fu)) && put(0x80u | ((cp >> 6) & 0x3Fu)) && put(0x80u | (cp & 0x3Fu));
        if (!ok) return -1;
    }
    out[n] = '\0';
    return static_cast<int32_t>(n);
}

// Hex text into bytes; the byte count, or -1.
int32_t hex_into(const char* t, uint8_t* out, uint32_t cap) noexcept {
    const auto nib = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
    uint32_t n = 0;
    for (const char* q = t; q[0] != '\0'; q += 2) {                          // bounded: the text
        if (q[1] == '\0' || n >= cap) return -1;
        const int h = nib(q[0]), l = nib(q[1]);
        if (h < 0 || l < 0) return -1;
        out[n++] = static_cast<uint8_t>((h << 4) | l);
    }
    return static_cast<int32_t>(n);
}

// A decimal's text at `scale` as a two's-complement little-endian int128 in `out` (16 bytes); false past 38 digits or the scale.
bool decimal_le128(const char* t, uint32_t scale, uint8_t* out) noexcept {
    const bool neg = t[0] == '-';
    if (neg || t[0] == '+') ++t;
    uint64_t lo = 0, hi = 0;
    uint32_t frac = 0, digits = 0;
    bool dot = false;
    for (const char* q = t; ; ++q) {                                         // bounded: the text, then the scale
        char c = *q;
        if (c == '.' && !dot) { dot = true; continue; }
        if (c == '\0') { if (frac >= scale) break; c = '0'; --q; }
        if (c < '0' || c > '9' || (dot && frac >= scale)) return false;
        if (dot) ++frac;
        if (++digits > 39u) return false;
        const uint64_t a = (lo & 0xFFFFFFFFu) * 10u + static_cast<uint64_t>(c - '0');
        const uint64_t b = (lo >> 32) * 10u + (a >> 32);
        lo = (b << 32) | (a & 0xFFFFFFFFu);
        hi = hi * 10u + (b >> 32);
    }
    if ((hi >> 63) != 0u) return false;
    if (neg) { lo = ~lo + 1u; hi = ~hi + (lo == 0u ? 1u : 0u); }
    std::memcpy(out, &lo, 8u);
    std::memcpy(out + 8, &hi, 8u);
    return true;
}

// "[YYYY-MM-DDT]HH:MM:SS[.ffffff][+00:00|Z]" as micros (since the epoch, or since midnight when `time_only`).
bool micros_of(const char* t, bool time_only, int64_t* out) noexcept {
    int32_t days = 0;
    const char* tp = t;
    if (!time_only) {
        char date[16] = {};
        const char* tsep = std::strchr(t, 'T');
        if (tsep == nullptr || tsep - t >= 15) return false;
        std::memcpy(date, t, static_cast<size_t>(tsep - t));
        if (!date_days(date, &days)) return false;
        tp = tsep + 1;
    }
    int h = 0, mi = 0, se = 0, used = 0;
    if (std::sscanf(tp, "%d:%d:%d%n", &h, &mi, &se, &used) != 3 || h < 0 || h > 23 || mi < 0 || mi > 59 || se < 0 || se > 59) return false;
    const char* q = tp + used;
    int64_t frac = 0, unit = 1000000;
    if (*q == '.') for (++q; *q >= '0' && *q <= '9'; ++q) if (unit > 1) { unit /= 10; frac += (*q - '0') * unit; }   // bounded: the text
    if (*q != '\0' && std::strcmp(q, "+00:00") != 0 && std::strcmp(q, "Z") != 0) return false;
    *out = ((static_cast<int64_t>(days) * 24 + h) * 60 + mi) * 60 * 1000000 + static_cast<int64_t>(se) * 1000000 + frac;
    return true;
}

// Fill the bytes of one value of `want` from a default's text into `cell` (8 bytes); false when the text is not that type.
bool default_cell(const SchemaField& f, BoltType want, uint8_t* cell) noexcept {
    const char* t = f.initial_default;
    char* end = nullptr;
    const bool str = f.initial_default_kind == DefaultKind::kString;
    switch (want) {
        case BoltType::Int64: case BoltType::Int32: {
            const long long v = std::strtoll(t, &end, 10);
            if (str || end == t || *end != '\0') return false;
            if (want == BoltType::Int64) { const int64_t x = v; std::memcpy(cell, &x, 8); }
            else { const int32_t x = static_cast<int32_t>(v); std::memcpy(cell, &x, 4); }
            return true;
        }
        case BoltType::Float64: case BoltType::Float32: {
            const double v = std::strtod(t, &end);
            if (str || end == t || *end != '\0') return false;
            if (want == BoltType::Float64) std::memcpy(cell, &v, 8);
            else { const float x = static_cast<float>(v); std::memcpy(cell, &x, 4); }
            return true;
        }
        case BoltType::Bool:
            if (str || (std::strcmp(t, "true") != 0 && std::strcmp(t, "false") != 0)) return false;
            cell[0] = t[0] == 't' ? 1u : 0u;
            return true;
        case BoltType::Date32: {
            int32_t d = 0;
            if (!str || !date_days(t, &d)) return false;
            std::memcpy(cell, &d, 4);
            return true;
        }
        case BoltType::Timestamp: case BoltType::Duration: {               // G2ICE-326: micros (since the epoch / midnight)
            int64_t us = 0;
            if (!str || !micros_of(t, want == BoltType::Duration, &us)) return false;
            std::memcpy(cell, &us, 8);
            return true;
        }
        case BoltType::Decimal64: case BoltType::Decimal128: {
            uint32_t fixed = 0, scale = 0;
            uint8_t v[16];
            if (!str || fill_bolt_type(f.type, &fixed, &scale) != want || !decimal_le128(t, scale, v)) return false;
            int64_t lo = 0, hi = 0;
            std::memcpy(&lo, v, 8u);
            std::memcpy(&hi, v + 8, 8u);
            if (want == BoltType::Decimal64 && hi != (lo < 0 ? -1 : 0)) return false;   // a decimal(p <= 18) mantissa fits 64 bits
            std::memcpy(cell, v, want == BoltType::Decimal64 ? 8u : 16u);
            return true;
        }
        case BoltType::UUID: {
            char hex[33] = {};
            uint32_t k = 0;
            for (const char* q = t; *q != '\0' && k < 32u; ++q) if (*q != '-') hex[k++] = *q;   // bounded: 36
            return str && std::strlen(t) == 36u && hex_into(hex, cell, 16u) == 16;
        }
        case BoltType::FixedSizeBinary: {
            uint32_t fixed = 0, scale = 0;
            (void)fill_bolt_type(f.type, &fixed, &scale);
            std::memset(cell, 0, 16u);
            return str && fixed > 0u && hex_into(t, cell, 16u) == static_cast<int32_t>(fixed);
        }
        default: return false;
    }
}

// A column of `rows` copies of the field's v3 initial-default (spec "Default values": a reader fills it for files without the field).
bool default_column(const SchemaField& f, BoltType want, int64_t rows, Arena* a, BoltColumn* out) noexcept {
    assert(a != nullptr && out != nullptr);
    const size_t n = static_cast<size_t>(rows) + 1u;
    if ((want == BoltType::Utf8 || want == BoltType::Binary) && f.initial_default_kind == DefaultKind::kString) {
        // A string's text unescaped; binary / fixed[L > 16] defaults are hex (spec Appendix D), read as their bytes.
        char text[limits::kDefaultTextMax];
        const bool bytes = std::strcmp(f.type, "string") != 0;
        const int32_t got = bytes ? hex_into(f.initial_default, reinterpret_cast<uint8_t*>(text), sizeof(text))
                                  : json_unescape_into(f.initial_default, text, sizeof(text));
        if (got < 0) return false;
        const uint32_t len = static_cast<uint32_t>(got);
        auto* sv = a->allocate_array<StringView>(n);
        auto* spill = static_cast<char*>(a->allocate(len + 1u));
        if (sv == nullptr || spill == nullptr) return false;
        std::memcpy(spill, text, len);
        spill[len] = '\0';
        StringView v;
        std::memset(&v, 0, sizeof(v));
        v.length = len;
        std::memcpy(v.prefix, text, len < 4u ? len : 4u);
        if (len > 4u && len <= 12u) std::memcpy(v.inline_data, text + 4, len - 4u);
        for (size_t r = 0; r < n; ++r) sv[r] = v;
        *out = BoltColumn::make_flat(sv, nullptr, rows, want);
        if (want == BoltType::Binary) out->type_size_bytes = sizeof(StringView);
        out->str_overflow_base = spill;
        return true;
    }
    uint8_t cell[16] = {};
    const size_t width = want == BoltType::NA ? 0u : bolt::type_size(want);
    if (f.initial_default_kind == DefaultKind::kUnsupported || width == 0u || width > 16u || !default_cell(f, want, cell)) {
        std::fprintf(stderr, "[iceberg_scan] REFUSED: column '%s' (field id %d) is absent from a data file and its initial-default ('%s', "
                     "type %s) is not one this reader fills (G2ICE-326)\n", f.name, f.id, f.initial_default, f.type);
        return false;
    }
    auto* d = static_cast<uint8_t*>(a->allocate(width * n, 8));
    if (d == nullptr) return false;
    for (size_t r = 0; r < n; ++r) std::memcpy(d + r * width, cell, width);
    *out = BoltColumn::make_flat(d, nullptr, rows, want);
    uint32_t fixed = 0, scale = 0;
    if (fill_bolt_type(f.type, &fixed, &scale) == want) {           // G2ICE-326: as the decoder labels a present column
        out->decimal_scale = static_cast<uint8_t>(scale);
        if (want == BoltType::FixedSizeBinary) out->fixed_width = static_cast<uint8_t>(fixed);
    }
    return true;
}

// Build the emitted columns (current-schema order) from the decoded file columns.
bool conform_group(ScanHandle* s, const BoltColumn* dec, BoltColumn* out, int64_t rows) noexcept {
    assert(s!=nullptr && s->use_map && s->map_schema!=nullptr);
    assert(dec!=nullptr && out!=nullptr);
    for (uint32_t j=0; j<s->map_schema->n_fields; ++j) {
        const char* name=s->map_schema->fields[j].name;
        if (s->colmap[j]>=0) {
            if (!promote_column(dec[s->colmap[j]],s->want_type[j],rows,s->scratch,&out[j],name)) return false;
            continue;
        }
        // Absent from this file: the column was added after the file was written. Reads its v3 initial-default, else null.
        uint32_t fixed = 0, scale = 0;
        const BoltType absent = s->want_type[j] != BoltType::NA ? s->want_type[j] : fill_bolt_type(s->map_schema->fields[j].type, &fixed, &scale);
        if (s->map_schema->fields[j].initial_default_kind != DefaultKind::kNone) {
            if (!default_column(s->map_schema->fields[j], absent, rows, s->scratch, &out[j])) return false;
            continue;
        }
        if (s->map_schema->fields[j].required) {
            std::fprintf(stderr,"[iceberg_scan] REFUSED: required column '%s' (field id %d) has no data in a data file and no default; the file or the table is corrupt\n",
                         name,s->map_schema->fields[j].id);
            return false;
        }
        const BoltType t=absent;
        if (t==BoltType::NA) {
            std::fprintf(stderr,"[iceberg_scan] REFUSED: column '%s' is absent from a data file and its type cannot be null-filled here\n",name);
            return false;
        }
        const size_t width=t==BoltType::Binary?sizeof(StringView):bolt::type_size(t);
        void* d=s->scratch->allocate_zeroed(width*(static_cast<size_t>(rows)+1u));
        auto* v=static_cast<uint8_t*>(s->scratch->allocate_zeroed((static_cast<size_t>(rows)+7u)/8u+1u));
        if (d==nullptr || v==nullptr) return false;
        out[j]=BoltColumn::make_flat(d,v,rows,t);
        if (t==BoltType::Binary) out[j].type_size_bytes=sizeof(StringView);
        if (t==BoltType::FixedSizeBinary) out[j].fixed_width=static_cast<uint8_t>(fixed);
        out[j].stats.all_valid=(rows==0);
        out[j].stats.null_count=rows;
    }
    return true;
}

bool bind_data_columns(ScanHandle* s, const pq::PqMeta* meta) noexcept {
    assert(s!=nullptr && meta!=nullptr); assert(s->table!=nullptr);
    s->output_columns=meta->n_columns;
    if (!s->table->strict_pinned) return bind_current_schema(s,meta);
    const auto* schema=s->selected_schema;
    if (schema==nullptr || meta->n_columns>kMaxColumns) return false;
    for (uint32_t i=0; i<meta->n_columns; ++i) {
        const auto& column=meta->columns[i];
        if (!column.has_field_id || column.field_id<=0 || column.max_rep!=0 || column.max_def>1)
            return false;
        for (uint32_t j=0; j<i; ++j)
            if (column.field_id==meta->columns[j].field_id) return false;
    }
    for (uint32_t i=0; i<schema->n_fields; ++i) {
        uint32_t physical=meta->n_columns;
        for (uint32_t j=0; j<meta->n_columns; ++j)
            if (schema->fields[i].id==meta->columns[j].field_id) physical=j;
        if (physical==meta->n_columns) return false;
        if (!pinned_type_matches(schema->fields[i],meta->columns[physical])) return false;
        s->source_indices[i]=static_cast<uint16_t>(physical);
    }
    s->output_columns=schema->n_fields;
    assert(s->output_columns<=kMaxColumns); return true;
}

bool delete_column_indices(const ScanHandle* s, const pq::PqMeta* meta,
    int32_t* path, int32_t* pos) noexcept {
    assert(s!=nullptr && meta!=nullptr); assert(path!=nullptr && pos!=nullptr);
    *path=-1; *pos=-1;
    for (uint32_t c=0; c<meta->n_columns; ++c) {
        const auto& column=meta->columns[c];
        const bool is_path=s->table->strict_pinned ? column.has_field_id && column.field_id==2147483546 :
            std::strcmp(column.name,"file_path")==0;
        const bool is_pos=s->table->strict_pinned ? column.has_field_id && column.field_id==2147483545 :
            std::strcmp(column.name,"pos")==0;
        if (is_path) { if (*path>=0) return false; *path=static_cast<int32_t>(c); }
        if (is_pos) { if (*pos>=0) return false; *pos=static_cast<int32_t>(c); }
    }
    if (*path<0 || *pos<0) return false;
    if (!s->table->strict_pinned) return true;
    SchemaField field{}; field.id=1;
    std::strcpy(field.type,"string");
    if (!pinned_type_matches(field,meta->columns[*path])) return false;
    std::strcpy(field.type,"long");
    return pinned_type_matches(field,meta->columns[*pos]) && meta->columns[*path].max_rep==0 &&
        meta->columns[*pos].max_rep==0 && meta->columns[*path].max_def<=1 && meta->columns[*pos].max_def<=1;
}
