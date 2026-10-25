// bolt/lakehouse/iceberg_metadata_merge.inc — lossless metadata on commit (G2ICE-279). Included inside
// iceberg_writer.cpp's anonymous namespace, before iceberg_commit.inc.
//
// The writer's Metadata struct does not model every table-metadata field (refs, sort-order fields,
// statistics, partition-statistics, encryption-keys, foreign properties, field docs, defaults ...).
// So a commit to an OPENED table re-emits the document it read and only replaces what the commit owns:
// the snapshot list and its bookkeeping, plus refs.main. Every other member stays byte-for-byte.

constexpr uint32_t kMergeMaxMembers = 128u;   // members of one metadata document / of its refs or properties

struct JsonMember {
    const char* key;      // points into the document, past the opening quote
    uint32_t    key_len;
    const char* val;      // raw value text
    uint32_t    val_len;
};

// End of the JSON value starting at p; string- and escape-aware, depth-counted.
const char* json_value_end(const char* p, const char* end) noexcept {
    assert(p != nullptr && end != nullptr);
    if (p >= end) return nullptr;
    if (*p == '"') {
        for (const char* q = p + 1; q < end; ++q) {        // bounded by the document
            if (*q == '\\') { ++q; continue; }
            if (*q == '"') return q + 1;
        }
        return nullptr;
    }
    if (*p == '{' || *p == '[') {
        int depth = 0;
        bool in_str = false;
        for (const char* q = p; q < end; ++q) {            // bounded by the document
            const char c = *q;
            if (in_str) { if (c == '\\') ++q; else if (c == '"') in_str = false; continue; }
            if (c == '"') in_str = true;
            else if (c == '{' || c == '[') ++depth;
            else if ((c == '}' || c == ']') && --depth == 0) return q + 1;
        }
        return nullptr;
    }
    const char* q = p;
    while (q < end && *q != ',' && *q != '}' && *q != ']' && *q != ' ' && *q != '\n' && *q != '\r' && *q != '\t') ++q;
    return q > p ? q : nullptr;
}

const char* skip_ws(const char* p, const char* end) noexcept {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    return p;
}

// Members of the object at [p, end). False on malformed input or more than kMergeMaxMembers.
bool json_object_members(const char* p, const char* end, JsonMember* out, uint32_t* n) noexcept {
    assert(out != nullptr && n != nullptr);
    *n = 0;
    p = skip_ws(p, end);
    if (p >= end || *p != '{') return false;
    p = skip_ws(p + 1, end);
    if (p < end && *p == '}') return true;
    while (p < end) {                                       // bounded by the document
        if (*p != '"' || *n >= kMergeMaxMembers) return false;
        const char* kend = json_value_end(p, end);
        if (kend == nullptr) return false;
        JsonMember& m = out[(*n)++];
        m.key = p + 1;
        m.key_len = static_cast<uint32_t>(kend - p - 2);
        p = skip_ws(kend, end);
        if (p >= end || *p != ':') return false;
        p = skip_ws(p + 1, end);
        const char* vend = json_value_end(p, end);
        if (vend == nullptr) return false;
        m.val = p;
        m.val_len = static_cast<uint32_t>(vend - p);
        p = skip_ws(vend, end);
        if (p < end && *p == ',') { p = skip_ws(p + 1, end); continue; }
        return p < end && *p == '}';
    }
    return false;
}

bool member_is(const JsonMember& m, const char* k) noexcept {
    const size_t n = std::strlen(k);
    return m.key_len == n && std::memcmp(m.key, k, n) == 0;
}

const JsonMember* find_member(const JsonMember* ms, uint32_t n, const JsonMember& like) noexcept {
    for (uint32_t i = 0; i < n; ++i)                       // bounded by n
        if (ms[i].key_len == like.key_len && std::memcmp(ms[i].key, like.key, like.key_len) == 0) return &ms[i];
    return nullptr;
}

struct MergeBuf {
    char*    d;
    uint64_t len;
    uint64_t cap;
    bool     ok;
};

void mb_put(MergeBuf* b, const char* s, uint64_t n) noexcept {
    if (!b->ok || b->len + n > b->cap) { b->ok = false; return; }
    std::memcpy(b->d + b->len, s, n);
    b->len += n;
}

void mb_member(MergeBuf* b, bool* first, const char* key, uint32_t key_len, const char* val, uint32_t val_len) noexcept {
    if (!*first) mb_put(b, ",", 1);
    *first = false;
    mb_put(b, "\"", 1);
    mb_put(b, key, key_len);
    mb_put(b, "\":", 2);
    mb_put(b, val, val_len);
}

// Object `base` with `over`'s members replacing/adding (refs: the commit's main over the table's refs).
void mb_object_overlay(MergeBuf* b, const JsonMember* base, uint32_t nb, const JsonMember* over, uint32_t no) noexcept {
    mb_put(b, "{", 1);
    bool first = true;
    for (uint32_t i = 0; i < nb; ++i) {                     // bounded
        const JsonMember* o = find_member(over, no, base[i]);
        const JsonMember& m = o != nullptr ? *o : base[i];
        mb_member(b, &first, m.key, m.key_len, m.val, m.val_len);
    }
    for (uint32_t i = 0; i < no; ++i)                       // bounded
        if (find_member(base, nb, over[i]) == nullptr) mb_member(b, &first, over[i].key, over[i].key_len, over[i].val, over[i].val_len);
    mb_put(b, "}", 1);
}

// The raw text of `key`'s value in the object at [obj, end); false when absent or malformed. `scratch` holds
// kMergeMaxMembers members.
bool object_value(const char* obj, const char* end, const char* key, JsonMember* scratch,
                  const char** v, uint32_t* n) noexcept {
    assert(obj != nullptr && key != nullptr);
    assert(scratch != nullptr && v != nullptr && n != nullptr);
    uint32_t nm = 0;
    if (!json_object_members(obj, end, scratch, &nm)) return false;
    for (uint32_t i = 0; i < nm; ++i)                       // bounded by nm
        if (member_is(scratch[i], key)) { *v = scratch[i].val; *n = scratch[i].val_len; return true; }
    return false;
}

// Next element of a JSON array: *p at the element (after '[' or ','); on success [*elem, *elem_end) is it and *p
// moves past it and its separator. False at ']' or on malformed input.
bool array_next(const char** p, const char* end, const char** elem, const char** elem_end) noexcept {
    assert(p != nullptr && elem != nullptr && elem_end != nullptr);
    const char* q = skip_ws(*p, end);
    if (q >= end || *q == ']') return false;
    const char* e = json_value_end(q, end);
    if (e == nullptr) return false;
    *elem = q;
    *elem_end = e;
    q = skip_ws(e, end);
    if (q < end && *q == ',') ++q;
    *p = q;
    return true;
}

// The snapshot object of `prior_arr` whose snapshot-id text equals `id`/`id_len`, or nullptr.
const char* prior_snapshot(const char* arr, const char* end, const char* id, uint32_t id_len, JsonMember* scratch,
                           const char** out_end) noexcept {
    assert(arr != nullptr && id != nullptr);
    if (arr >= end || *arr != '[') return nullptr;
    const char* p = arr + 1;
    const char* e = nullptr;
    const char* ee = nullptr;
    while (array_next(&p, end, &e, &ee)) {                  // bounded by the document
        const char* v = nullptr; uint32_t n = 0;
        if (object_value(e, ee, "snapshot-id", scratch, &v, &n) && n == id_len && std::memcmp(v, id, n) == 0) {
            *out_end = ee;
            return e;
        }
    }
    return nullptr;
}

// gen's snapshot list, but the prior document's own object for every snapshot both hold: snapshots are immutable,
// and the prior object carries what this writer does not model (another writer's summary, for one).
void mb_snapshots(MergeBuf* b, const JsonMember& prior, const JsonMember& gen, JsonMember* scratch) noexcept {
    assert(b != nullptr && scratch != nullptr);
    const char* pend = prior.val + prior.val_len;
    const char* gend = gen.val + gen.val_len;
    mb_put(b, "[", 1);
    if (gen.val_len == 0 || gen.val[0] != '[') { b->ok = false; return; }
    const char* p = gen.val + 1;
    const char* e = nullptr;
    const char* ee = nullptr;
    bool first = true;
    while (array_next(&p, gend, &e, &ee)) {                 // bounded by the document
        const char* id = nullptr; uint32_t id_len = 0;
        const char* pe = nullptr;
        const char* po = object_value(e, ee, "snapshot-id", scratch, &id, &id_len)
                             ? prior_snapshot(prior.val, pend, id, id_len, scratch, &pe) : nullptr;
        if (!first) mb_put(b, ",", 1);
        first = false;
        if (po != nullptr) mb_put(b, po, static_cast<uint64_t>(pe - po));
        else               mb_put(b, e, static_cast<uint64_t>(ee - e));
    }
    mb_put(b, "]", 1);
}

// What a commit through this handle changed, beyond its snapshot (set by the mutating table_* calls).
constexpr uint32_t kDirtySchema = 1u, kDirtySpec = 2u, kDirtySort = 4u, kDirtyRefs = 8u, kDirtyProps = 16u;

bool commit_owned_key(const JsonMember& m, uint32_t dirty) noexcept {
    static const char* const kOwned[] = {"snapshots", "snapshot-log", "current-snapshot-id", "last-sequence-number",
                                         "last-updated-ms", "metadata-log", "next-row-id"};
    for (const char* k : kOwned) if (member_is(m, k)) return true;
    if ((dirty & kDirtySchema) && (member_is(m, "schemas") || member_is(m, "current-schema-id") || member_is(m, "last-column-id"))) return true;
    if ((dirty & kDirtySpec) && (member_is(m, "partition-specs") || member_is(m, "default-spec-id") || member_is(m, "last-partition-id"))) return true;
    if ((dirty & kDirtySort) && (member_is(m, "sort-orders") || member_is(m, "default-sort-order-id"))) return true;
    return false;
}

bool name_dropped(const JsonMember& m, const char (*dropped)[64], uint32_t n) noexcept {
    for (uint32_t i = 0; i < n; ++i) if (member_is(m, dropped[i])) return true;   // bounded by n
    return false;
}

// The document a commit publishes: `prior` (what the table held) with the commit-owned members of `gen` (what
// metadata_json_emit produced) in their place, refs.main from `gen`, and properties `gen` adds that `prior` lacks.
bool metadata_merge(const uint8_t* prior, uint64_t prior_len, const uint8_t* gen, uint64_t gen_len, uint32_t dirty,
                    const char (*dropped)[64], uint32_t n_dropped, Arena* a, const uint8_t** out, uint64_t* out_len) noexcept {
    assert(prior != nullptr && gen != nullptr && a != nullptr);
    assert(out != nullptr && out_len != nullptr);
    auto* pmem = a->allocate_array<JsonMember>(kMergeMaxMembers * 4u);
    if (pmem == nullptr) return false;
    JsonMember* gmem = pmem + kMergeMaxMembers;
    JsonMember* sub_p = gmem + kMergeMaxMembers;
    JsonMember* sub_g = sub_p + kMergeMaxMembers;
    uint32_t np = 0, ng = 0;
    const char* pp = reinterpret_cast<const char*>(prior);
    const char* gp = reinterpret_cast<const char*>(gen);
    if (!json_object_members(pp, pp + prior_len, pmem, &np) || !json_object_members(gp, gp + gen_len, gmem, &ng)) return false;
    MergeBuf b{static_cast<char*>(a->allocate(prior_len + gen_len + 64u)), 0, prior_len + gen_len + 64u, true};
    if (b.d == nullptr) return false;
    mb_put(&b, "{", 1);
    bool first = true;
    for (uint32_t i = 0; i < np; ++i) {                     // bounded by the member cap
        const JsonMember& r = pmem[i];
        const JsonMember* g = find_member(gmem, ng, r);
        if (g != nullptr && member_is(r, "snapshots")) {
            if (!first) mb_put(&b, ",", 1);
            first = false;
            mb_put(&b, "\"snapshots\":", 12);
            mb_snapshots(&b, r, *g, sub_p);
            continue;
        }
        if (g != nullptr && commit_owned_key(r, dirty)) { mb_member(&b, &first, r.key, r.key_len, g->val, g->val_len); continue; }
        if (g != nullptr && (member_is(r, "refs") || member_is(r, "properties"))) {
            uint32_t a1 = 0, a2 = 0;
            if (!json_object_members(r.val, r.val + r.val_len, sub_p, &a1) ||
                !json_object_members(g->val, g->val + g->val_len, sub_g, &a2)) return false;
            const bool refs = member_is(r, "refs");
            uint32_t keep = 0;
            for (uint32_t k = 0; k < a2; ++k) {             // refs: main (or every ref this handle set); properties: missing (or every one this handle set)
                const bool take = refs ? ((dirty & kDirtyRefs) != 0u || member_is(sub_g[k], "main"))
                                       : ((dirty & kDirtyProps) != 0u || find_member(sub_p, a1, sub_g[k]) == nullptr);
                if (take) sub_g[keep++] = sub_g[k];
            }
            if (refs && n_dropped > 0u) {                   // a ref this handle dropped leaves the table's refs too
                uint32_t kp = 0;
                for (uint32_t k = 0; k < a1; ++k) if (!name_dropped(sub_p[k], dropped, n_dropped)) sub_p[kp++] = sub_p[k];
                a1 = kp;
            }
            if (!first) mb_put(&b, ",", 1);
            first = false;
            mb_put(&b, "\"", 1);
            mb_put(&b, r.key, r.key_len);
            mb_put(&b, "\":", 2);
            mb_object_overlay(&b, sub_p, a1, sub_g, keep);
            continue;
        }
        mb_member(&b, &first, r.key, r.key_len, r.val, r.val_len);
    }
    for (uint32_t i = 0; i < ng; ++i)                       // bounded: members the table never had (its first snapshot)
        if (find_member(pmem, np, gmem[i]) == nullptr) mb_member(&b, &first, gmem[i].key, gmem[i].key_len, gmem[i].val, gmem[i].val_len);
    mb_put(&b, "}", 1);
    if (!b.ok) return false;
    *out = reinterpret_cast<const uint8_t*>(b.d);
    *out_len = b.len;
    return true;
}
