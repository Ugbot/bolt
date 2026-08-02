// bolt/ingest/bolt_avro.cpp — Avro 1.11 OCF reader + writer. See bolt_avro.h.

#include "bolt/ingest/bolt_avro.h"

#include <cstring>

#include "bolt/ingest/bolt_gzip.h"
#include "bolt/ingest/bolt_snappy.h"
#include "bolt/ingest/bolt_zstd.h"
#include "bolt/parse/bolt_json.h"

namespace bolt {
namespace ingest {

namespace {

const uint8_t kMagic[4] = {'O', 'b', 'j', 0x01};

// ---- varint / zigzag ------------------------------------------------------

// Decode an Avro long (zig-zag varint). Advances *off. False on truncation /
// > 10 bytes.
bool read_long(const uint8_t* p, uint64_t len, uint64_t* off,
               int64_t* out) noexcept {
    assert(p != nullptr);
    assert(off != nullptr);
    uint64_t u = 0;
    uint32_t shift = 0;
    for (uint32_t i = 0; i < 10u; ++i) {           // bounded: 10 bytes max
        if (*off >= len) return false;
        const uint8_t b = p[(*off)++];
        u |= static_cast<uint64_t>(b & 0x7Fu) << shift;
        if ((b & 0x80u) == 0u) {
            *out = static_cast<int64_t>((u >> 1) ^ (~(u & 1u) + 1u));
            return true;
        }
        shift += 7u;
    }
    return false;
}

// Encode an Avro long (zig-zag varint) into dst; returns bytes written (<=10).
uint32_t write_long(int64_t v, uint8_t* dst) noexcept {
    assert(dst != nullptr);
    uint64_t u = (static_cast<uint64_t>(v) << 1) ^
                 static_cast<uint64_t>(v >> 63);
    uint32_t n = 0;
    do {                                            // bounded: <= 10 iters
        uint8_t b = static_cast<uint8_t>(u & 0x7Fu);
        u >>= 7;
        if (u != 0u) b |= 0x80u;
        dst[n++] = b;
    } while (u != 0u && n < 10u);
    return n;
}

// Read a length-prefixed byte slice (bytes/string). Slice points into src.
bool read_bytes(const uint8_t* p, uint64_t len, uint64_t* off,
                const uint8_t** out, uint32_t* out_len) noexcept {
    assert(off != nullptr);
    assert(out != nullptr);
    int64_t n = 0;
    if (!read_long(p, len, off, &n)) return false;
    if (n < 0 || static_cast<uint64_t>(n) > len - *off) return false;
    *out = p + *off;
    *out_len = static_cast<uint32_t>(n);
    *off += static_cast<uint64_t>(n);
    return true;
}

// ---- schema JSON parsing (SAX over bolt::parse::json) ---------------------
// We only need the flattened top-level record field list (name + primitive
// type, nullable if a ["null", T] union). The header schema is a record.

// Sentinel: no "null" branch seen in the union currently being parsed.
constexpr uint8_t kNoNullBranch = 0xFFu;

struct SchemaCtx {
    AvroField* fields;
    uint32_t   max_fields;     // caller cap on `fields`
    uint32_t   n;
    uint32_t   depth;          // object nesting depth
    bool       in_fields;      // inside the top record's "fields" array
    bool       expect_name;    // next string is a field name
    bool       expect_type;    // next string is a field type
    char       pending_name[kAvroMaxName];
    bool       have_name;
    bool       saw_null_union; // bare "type":"null"
    // Union state — a field's "type" may be an ARRAY of branches. Both
    // ["null",T] and [T,"null"] occur in real registry schemas, so the branch
    // INDEX that means null is recorded rather than assumed (see AvroField::
    // null_branch). Resolved on ']', not on the first branch string, so
    // ordering cannot change the outcome.
    bool       in_union;
    uint8_t    union_n;         // branches seen so far
    uint8_t    union_null_idx;  // index of the "null" branch, or kNoNullBranch
    bool       union_have_type; // a non-null branch was seen
    AvroType   union_type;      // first non-null branch's type
    bool       ok;
};

// Append one resolved field. Bounded by the caller's max_fields.
void commit_field(SchemaCtx* s, AvroType t, bool nullable,
                  uint8_t null_branch) noexcept {
    assert(s != nullptr);
    assert(s->n <= s->max_fields);
    if (!s->have_name || s->n >= s->max_fields) return;
    AvroField* f = &s->fields[s->n++];
    std::memset(f, 0, sizeof(*f));
    std::strncpy(f->name, s->pending_name, kAvroMaxName - 1u);
    f->type        = t;
    f->nullable    = nullable;
    f->null_branch = nullable ? null_branch : 0u;
    s->have_name      = false;
    s->saw_null_union = false;
    s->expect_type    = false;
}

AvroType type_from_str(const char* s, uint32_t len) noexcept {
    assert(s != nullptr);
    auto eq = [&](const char* k) noexcept {
        return std::strlen(k) == len && std::memcmp(s, k, len) == 0;
    };
    if (eq("null"))    return AvroType::kNull;
    if (eq("boolean")) return AvroType::kBoolean;
    if (eq("int"))     return AvroType::kInt;
    if (eq("long"))    return AvroType::kLong;
    if (eq("float"))   return AvroType::kFloat;
    if (eq("double"))  return AvroType::kDouble;
    if (eq("bytes"))   return AvroType::kBytes;
    if (eq("string"))  return AvroType::kString;
    return AvroType::kString;  // fallback for named/complex — treated as string
}

bool sx_key(void* c, const char* b, int32_t len) noexcept {
    SchemaCtx* s = static_cast<SchemaCtx*>(c);
    const auto k = [&](const char* kk) noexcept {
        return static_cast<int32_t>(std::strlen(kk)) == len &&
               std::memcmp(b, kk, static_cast<size_t>(len)) == 0;
    };
    if (k("fields")) { s->in_fields = true; }
    else if (k("name") && s->in_fields) { s->expect_name = true; }
    else if (k("type") && s->in_fields) { s->expect_type = true; }
    return true;
}

bool sx_string(void* c, const char* b, int32_t len) noexcept {
    SchemaCtx* s = static_cast<SchemaCtx*>(c);
    if (s->expect_name && !s->have_name) {
        uint32_t n = static_cast<uint32_t>(len);
        if (n >= kAvroMaxName) n = kAvroMaxName - 1u;
        std::memcpy(s->pending_name, b, n);
        s->pending_name[n] = '\0';
        s->have_name = true;
        s->expect_name = false;
        return true;
    }
    if (s->in_union) {                       // a branch of ["...", "..."]
        const AvroType t = type_from_str(b, static_cast<uint32_t>(len));
        if (t == AvroType::kNull) {
            if (s->union_null_idx == kNoNullBranch) s->union_null_idx = s->union_n;
        } else if (!s->union_have_type) {
            s->union_type      = t;
            s->union_have_type = true;
        }
        if (s->union_n < 0xFEu) ++s->union_n;   // bounded; index fits uint8
        return true;
    }
    if (s->expect_type) {                    // plain "type": "long"
        const AvroType t = type_from_str(b, static_cast<uint32_t>(len));
        if (t == AvroType::kNull) { s->saw_null_union = true; return true; }
        commit_field(s, t, /*nullable=*/s->saw_null_union, /*null_branch=*/0u);
        return true;
    }
    return true;
}

bool sx_begin_obj(void* c) noexcept {
    SchemaCtx* s = static_cast<SchemaCtx*>(c);
    s->depth++;
    return true;
}
bool sx_end_obj(void* c) noexcept {
    SchemaCtx* s = static_cast<SchemaCtx*>(c);
    if (s->depth > 0) s->depth--;
    return true;
}

// An array opened while a field's "type" is pending IS that field's union.
// (The enclosing "fields": [...] array opens when expect_type is false, so the
// two cannot be confused.)
bool sx_begin_arr(void* c) noexcept {
    SchemaCtx* s = static_cast<SchemaCtx*>(c);
    if (s->expect_type && !s->in_union) {
        s->in_union        = true;
        s->union_n         = 0u;
        s->union_null_idx  = kNoNullBranch;
        s->union_have_type = false;
        s->union_type      = AvroType::kNull;
    }
    return true;
}

// Resolve the union here — after every branch is known — so branch ORDER
// cannot change the result. ["null",T] yields null_branch 0, [T,"null"] 1.
bool sx_end_arr(void* c) noexcept {
    SchemaCtx* s = static_cast<SchemaCtx*>(c);
    if (!s->in_union) return true;
    const bool nullable = (s->union_null_idx != kNoNullBranch);
    const AvroType t =
        s->union_have_type ? s->union_type : AvroType::kNull;
    commit_field(s, t, nullable,
                 nullable ? s->union_null_idx : 0u);
    s->in_union = false;
    return true;
}

// Parse the schema JSON; fill fields[] (bounded by max_fields). Returns the
// field count, 0 on failure.
uint32_t parse_schema(const uint8_t* json, uint32_t json_len,
                      AvroField* fields, uint32_t max_fields) noexcept {
    assert(json != nullptr);
    assert(fields != nullptr);
    SchemaCtx s;
    std::memset(&s, 0, sizeof(s));
    s.fields         = fields;
    s.max_fields     = max_fields;
    s.union_null_idx = kNoNullBranch;
    s.ok             = true;
    parse::json::SaxHandler h;
    std::memset(&h, 0, sizeof(h));
    h.ctx = &s;
    h.on_begin_object = sx_begin_obj;
    h.on_end_object   = sx_end_obj;
    h.on_begin_array  = sx_begin_arr;
    h.on_end_array    = sx_end_arr;
    h.on_key          = sx_key;
    h.on_string       = sx_string;
    if (!parse::json::sax_parse(json, static_cast<int32_t>(json_len), &h)) {
        return 0;
    }
    return s.n;
}

// ---- header metadata map decode -------------------------------------------

AvroCodec codec_from(const uint8_t* b, uint32_t len) noexcept {
    auto eq = [&](const char* k) noexcept {
        return std::strlen(k) == len && std::memcmp(b, k, len) == 0;
    };
    if (len == 0 || eq("null"))    return AvroCodec::kNull;
    if (eq("deflate")) return AvroCodec::kDeflate;
    if (eq("snappy"))  return AvroCodec::kSnappy;
    if (eq("zstandard") || eq("zstd")) return AvroCodec::kZstd;
    return AvroCodec::kUnknown;
}

}  // namespace

bool avro_read_header(const uint8_t* src, uint64_t src_len,
                      Arena* arena, AvroHeader* out,
                      uint64_t* body_offset) noexcept {
    assert(src != nullptr || src_len == 0);
    assert(out != nullptr);
    if (src == nullptr || arena == nullptr || out == nullptr) return false;
    if (src_len < kAvroMagicLen + 1u) return false;
    if (std::memcmp(src, kMagic, kAvroMagicLen) != 0) return false;

    std::memset(out, 0, sizeof(*out));
    out->codec = AvroCodec::kNull;

    uint64_t off = kAvroMagicLen;
    // Metadata is map<string,bytes>: blocks of (long count, entries...), 0 ends.
    const uint8_t* schema_json = nullptr;
    uint32_t schema_len = 0;
    for (uint32_t blk = 0; blk < 64u; ++blk) {     // bounded blocks
        int64_t count = 0;
        if (!read_long(src, src_len, &off, &count)) return false;
        if (count == 0) break;
        if (count < 0) {                            // negative ⇒ size-prefixed
            int64_t bytesz = 0;
            if (!read_long(src, src_len, &off, &bytesz)) return false;
            count = -count;
        }
        if (count > static_cast<int64_t>(kAvroMaxFields)) return false;
        for (int64_t i = 0; i < count; ++i) {       // bounded by count
            const uint8_t* k = nullptr; uint32_t klen = 0;
            const uint8_t* v = nullptr; uint32_t vlen = 0;
            if (!read_bytes(src, src_len, &off, &k, &klen)) return false;
            if (!read_bytes(src, src_len, &off, &v, &vlen)) return false;
            if (klen == 11u && std::memcmp(k, "avro.schema", 11) == 0) {
                schema_json = v; schema_len = vlen;
            } else if (klen == 10u && std::memcmp(k, "avro.codec", 10) == 0) {
                out->codec = codec_from(v, vlen);
            }
        }
    }
    if (out->codec == AvroCodec::kUnknown) return false;
    if (schema_json == nullptr) return false;

    out->n_fields = parse_schema(schema_json, schema_len, out->field,
                                 kAvroMaxFields);
    if (out->n_fields == 0) return false;

    if (off + kAvroSyncLen > src_len) return false;
    std::memcpy(out->sync, src + off, kAvroSyncLen);
    off += kAvroSyncLen;
    if (body_offset != nullptr) *body_offset = off;
    return true;
}

namespace {

// Decode one row of primitives at *off (decompressed block buffer `p`).
bool decode_row(const uint8_t* p, uint64_t len, uint64_t* off,
                const AvroField* fields, uint32_t n_fields,
                AvroValue* vals) noexcept {
    assert(p != nullptr);
    assert(off != nullptr);
    for (uint32_t f = 0; f < n_fields; ++f) {       // bounded: n_fields
        AvroValue* val = &vals[f];
        std::memset(val, 0, sizeof(*val));
        val->type = fields[f].type;
        const AvroField* fd = &fields[f];
        if (fd->nullable) {                          // union index precedes value
            int64_t branch = 0;
            if (!read_long(p, len, off, &branch)) return false;
            // Which index means null is schema-declared (AvroField::null_branch),
            // not assumed: ["null",T] gives 0 (what Iceberg emits), [T,"null"]
            // gives 1. A zeroed field decodes null-first, so OCF callers that
            // predate this field are unaffected.
            if (branch == static_cast<int64_t>(fd->null_branch)) {
                val->is_null = true;
                continue;
            }
        }
        switch (fd->type) {
            case AvroType::kNull: val->is_null = true; break;
            case AvroType::kBoolean: {
                if (*off >= len) return false;
                val->num.i64 = p[(*off)++] != 0 ? 1 : 0;
                break;
            }
            case AvroType::kInt:
            case AvroType::kLong: {
                if (!read_long(p, len, off, &val->num.i64)) return false;
                break;
            }
            case AvroType::kFloat: {
                if (*off + 4u > len) return false;
                float fv = 0.0f;
                std::memcpy(&fv, p + *off, 4);
                val->num.f64 = static_cast<double>(fv);
                *off += 4u;
                break;
            }
            case AvroType::kDouble: {
                if (*off + 8u > len) return false;
                double dv = 0.0;
                std::memcpy(&dv, p + *off, 8);
                val->num.f64 = dv;
                *off += 8u;
                break;
            }
            case AvroType::kBytes:
            case AvroType::kString: {
                if (!read_bytes(p, len, off, &val->bytes, &val->bytes_len))
                    return false;
                break;
            }
        }
    }
    return true;
}

// Decompress one raw object block into the arena per the codec.
bool decompress_block(AvroCodec codec, const uint8_t* in, uint64_t in_len,
                      Arena* arena, const uint8_t** out, uint64_t* out_len,
                      int64_t obj_count) noexcept {
    assert(arena != nullptr);
    assert(out != nullptr);
    (void)obj_count;
    if (codec == AvroCodec::kNull) { *out = in; *out_len = in_len; return true; }
    if (codec == AvroCodec::kSnappy) {
        // Avro snappy block = snappy-raw bytes + 4-byte big-endian CRC32 tail.
        if (in_len < 4u) return false;
        const uint64_t body = in_len - 4u;
        uint64_t ulen = 0;
        if (snappy_uncompressed_len(in, body, &ulen) == 0) return false;
        uint8_t* buf = arena->allocate_array<uint8_t>(
            ulen == 0 ? 1u : static_cast<size_t>(ulen));
        if (buf == nullptr) return false;
        if (!snappy_decompress(in, body, buf, ulen)) return false;
        *out = buf; *out_len = ulen;
        return true;
    }
    // deflate / zstd: the OCF block does not carry the decompressed length and
    // our exact-fill codec wrappers require it, so the streaming deflate/zstd
    // Avro blocks are not supported in W1 (null + snappy are). Callers writing
    // OCFs here use the null codec; manifest reads we exercise are snappy/null.
    (void)arena; (void)in; (void)in_len; (void)out_len;
    return false;
}

}  // namespace

bool avro_read(const uint8_t* src, uint64_t src_len, Arena* arena,
               void* ctx, AvroRowFn on_row, int64_t* out_rows) noexcept {
    assert(src != nullptr || src_len == 0);
    assert(on_row != nullptr);
    if (src == nullptr || arena == nullptr || on_row == nullptr) return false;

    AvroHeader hdr;
    uint64_t off = 0;
    if (!avro_read_header(src, src_len, arena, &hdr, &off)) return false;

    AvroValue vals[kAvroMaxFields];
    int64_t row_index = 0;
    for (uint32_t blk = 0; blk < (1u << 24); ++blk) {   // bounded block count
        if (off >= src_len) break;
        int64_t count = 0, block_size = 0;
        if (!read_long(src, src_len, &off, &count)) return false;
        if (!read_long(src, src_len, &off, &block_size)) return false;
        if (count < 0 || block_size < 0) return false;
        if (static_cast<uint64_t>(block_size) > src_len - off) return false;

        const uint8_t* raw = src + off;
        off += static_cast<uint64_t>(block_size);

        const uint8_t* body = nullptr; uint64_t body_len = 0;
        if (!decompress_block(hdr.codec, raw,
                              static_cast<uint64_t>(block_size), arena, &body,
                              &body_len, count)) return false;

        uint64_t boff = 0;
        for (int64_t r = 0; r < count; ++r) {        // bounded by count
            if (!decode_row(body, body_len, &boff, hdr.field, hdr.n_fields,
                            vals)) return false;
            if (!on_row(ctx, vals, hdr.n_fields, row_index)) return false;
            ++row_index;
        }
        // Sync marker after the block.
        if (off + kAvroSyncLen > src_len) return false;
        if (std::memcmp(src + off, hdr.sync, kAvroSyncLen) != 0) return false;
        off += kAvroSyncLen;
    }
    if (out_rows != nullptr) *out_rows = row_index;
    return true;
}

// ---- bare-datum surface (streaming carriers; see bolt_avro.h) --------------
// Thin, allocation-free wrappers over the SAME parse_schema / decode_row the
// OCF path uses — deliberately not a second decoder, so the container and
// streaming paths can never diverge.

bool avro_parse_schema(const uint8_t* json, uint32_t json_len,
                       AvroField* out_fields, uint32_t max_fields,
                       uint32_t* out_n) noexcept {
    assert(out_fields != nullptr || max_fields == 0);
    assert(out_n != nullptr);
    if (json == nullptr || out_fields == nullptr || out_n == nullptr) return false;
    if (json_len == 0u || max_fields == 0u) return false;
    const uint32_t n = parse_schema(json, json_len, out_fields, max_fields);
    *out_n = n;
    return n != 0u;
}

bool avro_decode_datum(const uint8_t* src, uint64_t src_len,
                       const AvroField* fields, uint32_t n_fields,
                       AvroValue* out_vals, uint64_t* out_consumed) noexcept {
    assert(src != nullptr || src_len == 0);
    assert(fields != nullptr || n_fields == 0);
    if (src == nullptr || fields == nullptr || out_vals == nullptr) return false;
    if (n_fields == 0u) return false;
    uint64_t off = 0;
    if (!decode_row(src, src_len, &off, fields, n_fields, out_vals)) return false;
    assert(off <= src_len);              // decode_row never runs past the end
    if (out_consumed != nullptr) *out_consumed = off;
    return true;
}

// ---- writer ---------------------------------------------------------------

namespace {

// Append a length-prefixed byte slice (Avro bytes/string).
uint32_t enc_bytes(uint8_t* dst, const uint8_t* b, uint32_t len) noexcept {
    assert(dst != nullptr);
    uint32_t n = write_long(static_cast<int64_t>(len), dst);
    if (len > 0) std::memcpy(dst + n, b, len);
    return n + len;
}

// Build the JSON schema "{\"type\":\"record\",\"name\":\"r\",\"fields\":[...]}".
uint32_t build_schema_json(const AvroField* fields, uint32_t n_fields,
                           char* dst, uint32_t cap) noexcept {
    assert(dst != nullptr);
    auto put = [&](uint32_t* p, const char* s) noexcept -> bool {
        const size_t sl = std::strlen(s);
        if (*p + sl >= cap) return false;
        std::memcpy(dst + *p, s, sl); *p += static_cast<uint32_t>(sl);
        return true;
    };
    const char* prim[] = {"null","boolean","int","long","float","double",
                          "bytes","string"};
    uint32_t p = 0;
    if (!put(&p, "{\"type\":\"record\",\"name\":\"r\",\"fields\":[")) return 0;
    for (uint32_t i = 0; i < n_fields; ++i) {        // bounded
        if (i && !put(&p, ",")) return 0;
        if (!put(&p, "{\"name\":\"") || !put(&p, fields[i].name) ||
            !put(&p, "\",\"type\":")) return 0;
        const char* tn = prim[static_cast<uint32_t>(fields[i].type)];
        if (fields[i].nullable) {
            if (!put(&p, "[\"null\",\"") || !put(&p, tn) || !put(&p, "\"]"))
                return 0;
        } else {
            if (!put(&p, "\"") || !put(&p, tn) || !put(&p, "\"")) return 0;
        }
        if (!put(&p, "}")) return 0;
    }
    if (!put(&p, "]}")) return 0;
    return p;
}

}  // namespace

uint64_t avro_write_max_len(const AvroField* fields, uint32_t n_fields,
                            uint64_t total_value_bytes, int64_t n_rows) noexcept {
    assert(fields != nullptr);
    assert(n_rows >= 0);
    (void)fields;   // size bound depends only on field count + value bytes
    // magic(4) + metadata map (schema + codec, generous 4 KB) + sync(16) +
    // block header(20) + per-cell 10-byte long + value bytes + trailing sync.
    const uint64_t per_cell = 10u;
    const uint64_t cells = static_cast<uint64_t>(n_rows) * n_fields;
    return 4u + 4096u + 16u + 20u + cells * per_cell + total_value_bytes +
           16u + 64u;
}

bool avro_write(const AvroField* fields, uint32_t n_fields,
                const AvroValue* rows, int64_t n_rows,
                const uint8_t sync[kAvroSyncLen],
                uint8_t* dst, uint64_t* dst_len) noexcept {
    assert(fields != nullptr);
    assert(dst_len != nullptr);
    if (dst == nullptr || dst_len == nullptr) return false;
    if (n_fields == 0 || n_fields > kAvroMaxFields) return false;
    if (n_rows < 0) return false;
    const uint64_t cap = *dst_len;
    uint64_t o = 0;

    auto room = [&](uint64_t need) noexcept { return o + need <= cap; };

    if (!room(kAvroMagicLen)) return false;
    std::memcpy(dst + o, kMagic, kAvroMagicLen); o += kAvroMagicLen;

    // Metadata map: 2 entries (avro.schema, avro.codec).
    char schema[4096];
    const uint32_t slen = build_schema_json(fields, n_fields, schema,
                                            sizeof(schema));
    if (slen == 0) return false;
    if (!room(64u + slen)) return false;
    o += write_long(2, dst + o);                     // 2 map entries
    o += enc_bytes(dst + o, reinterpret_cast<const uint8_t*>("avro.schema"), 11);
    o += enc_bytes(dst + o, reinterpret_cast<const uint8_t*>(schema), slen);
    o += enc_bytes(dst + o, reinterpret_cast<const uint8_t*>("avro.codec"), 10);
    o += enc_bytes(dst + o, reinterpret_cast<const uint8_t*>("null"), 4);
    o += write_long(0, dst + o);                     // end map

    if (!room(kAvroSyncLen)) return false;
    std::memcpy(dst + o, sync, kAvroSyncLen); o += kAvroSyncLen;

    // One object block. Encode rows into the area after the block header, then
    // backfill count + size. Reserve up to 20 bytes for the two longs.
    const uint64_t hdr_pos = o;
    uint8_t tmp[20];
    const uint32_t cnt_bytes = write_long(n_rows, tmp);
    // We don't know block_size yet; encode body at a tentative offset and shift.
    // Simpler: encode body into a temp cursor first measuring size.
    const uint64_t body_start_guess = hdr_pos + cnt_bytes + 10u;
    if (body_start_guess > cap) return false;
    uint64_t b = body_start_guess;
    for (int64_t r = 0; r < n_rows; ++r) {           // bounded by n_rows
        const AvroValue* row = rows + static_cast<uint64_t>(r) * n_fields;
        for (uint32_t f = 0; f < n_fields; ++f) {    // bounded
            const AvroValue* v = &row[f];
            const AvroField* fd = &fields[f];
            if (fd->nullable) {
                if (!room(b - o + 10u)) return false;
                b += write_long(v->is_null ? 0 : 1, dst + b);
                if (v->is_null) continue;
            }
            switch (fd->type) {
                case AvroType::kNull: break;
                case AvroType::kBoolean:
                    if (b + 1u > cap) return false;
                    dst[b++] = v->num.i64 != 0 ? 1u : 0u; break;
                case AvroType::kInt:
                case AvroType::kLong:
                    if (b + 10u > cap) return false;
                    b += write_long(v->num.i64, dst + b); break;
                case AvroType::kFloat: {
                    if (b + 4u > cap) return false;
                    float fv = static_cast<float>(v->num.f64);
                    std::memcpy(dst + b, &fv, 4); b += 4u; break;
                }
                case AvroType::kDouble: {
                    if (b + 8u > cap) return false;
                    double dv = v->num.f64;
                    std::memcpy(dst + b, &dv, 8); b += 8u; break;
                }
                case AvroType::kBytes:
                case AvroType::kString: {
                    if (b + 10u + v->bytes_len > cap) return false;
                    b += enc_bytes(dst + b, v->bytes, v->bytes_len); break;
                }
            }
        }
    }
    const uint64_t body_len = b - body_start_guess;

    // Now write count + size at hdr_pos, then shift body left to be contiguous.
    uint8_t hdr[20];
    uint32_t hp = 0;
    hp += write_long(n_rows, hdr + hp);
    hp += write_long(static_cast<int64_t>(body_len), hdr + hp);
    std::memmove(dst + hdr_pos + hp, dst + body_start_guess, body_len);
    std::memcpy(dst + hdr_pos, hdr, hp);
    o = hdr_pos + hp + body_len;

    if (!room(kAvroSyncLen)) return false;
    std::memcpy(dst + o, sync, kAvroSyncLen); o += kAvroSyncLen;

    *dst_len = o;
    return true;
}

}  // namespace ingest
}  // namespace bolt
