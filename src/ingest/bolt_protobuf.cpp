// bolt/ingest/bolt_protobuf.cpp — dynamic protobuf decoder. See bolt_protobuf.h.
//
// No code generation, no libprotobuf. Layer 2 bootstraps off layer 1 using the
// constant field numbers of descriptor.proto declared below.

#include "bolt/ingest/bolt_protobuf.h"

#include <cstring>

namespace bolt {
namespace ingest {

namespace {

// ---- descriptor.proto bootstrap ------------------------------------------
// These are descriptor.proto's own field numbers. They are part of the stable,
// wire-visible contract of protobuf itself — a small constant table, not
// generated code. This is the standard way to parse a FileDescriptorSet
// without already having a protobuf compiler.

constexpr int32_t kFdsFile          = 1;   // FileDescriptorSet.file

constexpr int32_t kFdpPackage       = 2;   // FileDescriptorProto.package
constexpr int32_t kFdpMessageType   = 4;   // FileDescriptorProto.message_type

constexpr int32_t kDpName           = 1;   // DescriptorProto.name
constexpr int32_t kDpField          = 2;   // DescriptorProto.field
constexpr int32_t kDpNestedType     = 3;   // DescriptorProto.nested_type

constexpr int32_t kFdName           = 1;   // FieldDescriptorProto.name
constexpr int32_t kFdNumber         = 3;   // FieldDescriptorProto.number
constexpr int32_t kFdLabel          = 4;   // FieldDescriptorProto.label
constexpr int32_t kFdType           = 5;   // FieldDescriptorProto.type
constexpr int32_t kFdTypeName       = 6;   // FieldDescriptorProto.type_name

// ---- small helpers --------------------------------------------------------

bool copy_str(char* dst, uint32_t cap, const uint8_t* src, uint32_t len) noexcept {
    assert(dst != nullptr);
    assert(cap > 0u);
    if (len >= cap) return false;                 // fail closed, never truncate
    if (len > 0u) std::memcpy(dst, src, len);
    dst[len] = '\0';
    return true;
}

// "prefix.name" into dst. Empty prefix ⇒ just name.
bool join_name(char* dst, uint32_t cap, const char* prefix,
               const char* name) noexcept {
    assert(dst != nullptr);
    assert(name != nullptr);
    const size_t pl = (prefix != nullptr) ? std::strlen(prefix) : 0u;
    const size_t nl = std::strlen(name);
    const size_t need = (pl > 0u) ? (pl + 1u + nl) : nl;
    if (need >= cap) return false;
    if (pl > 0u) {
        std::memcpy(dst, prefix, pl);
        dst[pl] = '.';
        std::memcpy(dst + pl + 1u, name, nl);
    } else {
        std::memcpy(dst, name, nl);
    }
    dst[need] = '\0';
    return true;
}

bool is_numeric_type(PbType t) noexcept {
    switch (t) {
        case PbType::kDouble: case PbType::kFloat:
        case PbType::kInt64:  case PbType::kUint64: case PbType::kInt32:
        case PbType::kFixed64: case PbType::kFixed32: case PbType::kBool:
        case PbType::kUint32: case PbType::kEnum:
        case PbType::kSfixed32: case PbType::kSfixed64:
        case PbType::kSint32: case PbType::kSint64:
            return true;
        default:
            return false;
    }
}

// Wire type a scalar type uses when NOT packed.
PbWireType native_wire(PbType t) noexcept {
    switch (t) {
        case PbType::kDouble: case PbType::kFixed64: case PbType::kSfixed64:
            return PbWireType::kFixed64;
        case PbType::kFloat: case PbType::kFixed32: case PbType::kSfixed32:
            return PbWireType::kFixed32;
        case PbType::kString: case PbType::kBytes: case PbType::kMessage:
            return PbWireType::kLenDelim;
        default:
            return PbWireType::kVarint;
    }
}

bool read_fixed32(const uint8_t* src, uint64_t len, uint64_t* off,
                  uint32_t* out) noexcept {
    assert(off != nullptr);
    assert(out != nullptr);
    if (*off + 4u > len) return false;
    uint32_t v = 0;
    std::memcpy(&v, src + *off, 4u);              // little-endian on target
    *off += 4u;
    *out = v;
    return true;
}

bool read_fixed64(const uint8_t* src, uint64_t len, uint64_t* off,
                  uint64_t* out) noexcept {
    assert(off != nullptr);
    assert(out != nullptr);
    if (*off + 8u > len) return false;
    uint64_t v = 0;
    std::memcpy(&v, src + *off, 8u);
    *off += 8u;
    *out = v;
    return true;
}

bool read_slice(const uint8_t* src, uint64_t len, uint64_t* off,
                const uint8_t** out, uint64_t* out_len) noexcept {
    assert(off != nullptr);
    uint64_t n = 0;
    if (!pb_read_varint(src, len, off, &n)) return false;
    if (n > len - *off) return false;             // no overflow: *off <= len
    *out = src + *off;
    *out_len = n;
    *off += n;
    return true;
}

// Fill one PbValue from a varint payload, per the declared type.
void set_from_varint(PbValue* v, PbType t, uint64_t u) noexcept {
    assert(v != nullptr);
    switch (t) {
        case PbType::kSint32: case PbType::kSint64:
            v->num.i64 = pb_zigzag_decode(u); break;
        case PbType::kBool:
            v->num.i64 = (u != 0u) ? 1 : 0; break;
        case PbType::kInt32:
            v->num.i64 = static_cast<int32_t>(static_cast<uint32_t>(u)); break;
        case PbType::kUint32:
            v->num.u64 = static_cast<uint32_t>(u); break;
        case PbType::kUint64:
            v->num.u64 = u; break;
        default:  // kInt64, kEnum
            v->num.i64 = static_cast<int64_t>(u); break;
    }
}

void set_from_fixed32(PbValue* v, PbType t, uint32_t raw) noexcept {
    assert(v != nullptr);
    if (t == PbType::kFloat) {
        float f = 0.0f;
        std::memcpy(&f, &raw, 4u);
        v->num.f64 = static_cast<double>(f);
    } else if (t == PbType::kSfixed32) {
        v->num.i64 = static_cast<int32_t>(raw);
    } else {
        v->num.u64 = raw;
    }
}

void set_from_fixed64(PbValue* v, PbType t, uint64_t raw) noexcept {
    assert(v != nullptr);
    if (t == PbType::kDouble) {
        double d = 0.0;
        std::memcpy(&d, &raw, 8u);
        v->num.f64 = d;
    } else if (t == PbType::kSfixed64) {
        v->num.i64 = static_cast<int64_t>(raw);
    } else {
        v->num.u64 = raw;
    }
}

// Decode ONE value of `t` at *off (wire type already known to match).
bool decode_one(const uint8_t* src, uint64_t len, uint64_t* off,
                PbType t, PbWireType wire, PbValue* out) noexcept {
    assert(off != nullptr);
    assert(out != nullptr);
    if (wire == PbWireType::kVarint) {
        uint64_t u = 0;
        if (!pb_read_varint(src, len, off, &u)) return false;
        set_from_varint(out, t, u);
        return true;
    }
    if (wire == PbWireType::kFixed32) {
        uint32_t raw = 0;
        if (!read_fixed32(src, len, off, &raw)) return false;
        set_from_fixed32(out, t, raw);
        return true;
    }
    if (wire == PbWireType::kFixed64) {
        uint64_t raw = 0;
        if (!read_fixed64(src, len, off, &raw)) return false;
        set_from_fixed64(out, t, raw);
        return true;
    }
    if (wire == PbWireType::kLenDelim) {
        const uint8_t* p = nullptr;
        uint64_t       n = 0;
        if (!read_slice(src, len, off, &p, &n)) return false;
        out->bytes     = p;
        out->bytes_len = static_cast<uint32_t>(n);
        return true;
    }
    return false;                                  // groups: rejected
}

// Count (and optionally capture the first) values in a PACKED run.
bool scan_packed(const uint8_t* p, uint64_t n, PbType t,
                 PbValue* first, uint32_t* out_count) noexcept {
    assert(out_count != nullptr);
    const PbWireType w = native_wire(t);
    uint64_t off = 0;
    uint32_t cnt = 0;
    while (off < n) {                              // bounded by n
        PbValue tmp{};
        if (!decode_one(p, n, &off, t, w, &tmp)) return false;
        if (cnt == 0u && first != nullptr) {
            const uint32_t keep_rep = first->rep_count;
            tmp.type       = t;
            tmp.is_null    = false;
            tmp.msg_index  = -1;
            tmp.rep_count  = keep_rep;
            *first         = tmp;
        }
        ++cnt;
    }
    *out_count = cnt;
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Layer 1 — wire reader
// ---------------------------------------------------------------------------

bool pb_read_varint(const uint8_t* src, uint64_t len, uint64_t* off,
                    uint64_t* out) noexcept {
    assert(src != nullptr || len == 0u);
    assert(off != nullptr && out != nullptr);
    uint64_t u = 0;
    uint32_t shift = 0;
    for (uint32_t i = 0; i < kPbVarintMaxBytes; ++i) {   // bounded: 10 bytes
        if (*off >= len) return false;
        const uint8_t b = src[(*off)++];
        u |= static_cast<uint64_t>(b & 0x7Fu) << shift;
        if ((b & 0x80u) == 0u) { *out = u; return true; }
        shift += 7u;
    }
    return false;                                        // overlong varint
}

bool pb_read_tag(const uint8_t* src, uint64_t len, uint64_t* off,
                 int32_t* out_field, PbWireType* out_wire) noexcept {
    assert(off != nullptr);
    assert(out_field != nullptr && out_wire != nullptr);
    uint64_t tag = 0;
    if (!pb_read_varint(src, len, off, &tag)) return false;
    const uint64_t fnum = tag >> 3;
    if (fnum == 0u || fnum > 0x1FFFFFFFull) return false; // 0 and >2^29-1 invalid
    *out_field = static_cast<int32_t>(fnum);
    *out_wire  = static_cast<PbWireType>(tag & 0x7u);
    return true;
}

bool pb_skip_field(const uint8_t* src, uint64_t len, uint64_t* off,
                   PbWireType wire) noexcept {
    assert(off != nullptr);
    assert(*off <= len);
    switch (wire) {
        case PbWireType::kVarint: {
            uint64_t u = 0;
            return pb_read_varint(src, len, off, &u);
        }
        case PbWireType::kFixed64:
            if (*off + 8u > len) return false;
            *off += 8u;
            return true;
        case PbWireType::kFixed32:
            if (*off + 4u > len) return false;
            *off += 4u;
            return true;
        case PbWireType::kLenDelim: {
            const uint8_t* p = nullptr;
            uint64_t n = 0;
            return read_slice(src, len, off, &p, &n);
        }
        default:
            return false;                            // groups: unsupported
    }
}

// ---------------------------------------------------------------------------
// Layer 2 — descriptor parsing
// ---------------------------------------------------------------------------

namespace {

bool parse_field_desc(const uint8_t* p, uint64_t n, PbFieldDesc* out) noexcept {
    assert(out != nullptr);
    std::memset(out, 0, sizeof(*out));
    out->msg_index = -1;
    uint64_t off = 0;
    while (off < n) {                                // bounded by n
        int32_t    f = 0;
        PbWireType w = PbWireType::kVarint;
        if (!pb_read_tag(p, n, &off, &f, &w)) return false;
        if (f == kFdName && w == PbWireType::kLenDelim) {
            const uint8_t* s = nullptr; uint64_t sl = 0;
            if (!read_slice(p, n, &off, &s, &sl)) return false;
            if (!copy_str(out->name, kPbMaxName, s, static_cast<uint32_t>(sl)))
                return false;
        } else if (f == kFdTypeName && w == PbWireType::kLenDelim) {
            const uint8_t* s = nullptr; uint64_t sl = 0;
            if (!read_slice(p, n, &off, &s, &sl)) return false;
            if (!copy_str(out->type_name, kPbMaxFullName, s,
                          static_cast<uint32_t>(sl))) return false;
        } else if (f == kFdNumber && w == PbWireType::kVarint) {
            uint64_t u = 0;
            if (!pb_read_varint(p, n, &off, &u)) return false;
            out->number = static_cast<int32_t>(static_cast<uint32_t>(u));
        } else if (f == kFdLabel && w == PbWireType::kVarint) {
            uint64_t u = 0;
            if (!pb_read_varint(p, n, &off, &u)) return false;
            out->label = static_cast<PbLabel>(u & 0xFFu);
        } else if (f == kFdType && w == PbWireType::kVarint) {
            uint64_t u = 0;
            if (!pb_read_varint(p, n, &off, &u)) return false;
            out->type = static_cast<PbType>(u & 0xFFu);
        } else {
            if (!pb_skip_field(p, n, &off, w)) return false;
        }
    }
    out->repeated = (out->label == PbLabel::kRepeated);
    out->nullable = !out->repeated && (out->label != PbLabel::kRequired);
    return true;
}

// Parse a DescriptorProto into set->msg[*], hoisting nested types with dotted
// names. Depth-bounded; the recursion is the cold descriptor path only.
bool parse_message_desc(const uint8_t* p, uint64_t n, const char* prefix,
                        PbDescriptorSet* set, uint32_t depth) noexcept {
    assert(set != nullptr);
    assert(p != nullptr || n == 0u);
    if (depth > kPbMaxNesting) return false;
    if (set->n_msgs >= kPbMaxMessages) return false;

    const uint32_t my = set->n_msgs++;
    PbMessageDesc* m = &set->msg[my];
    std::memset(m, 0, sizeof(*m));

    // Pass 1: name (needed as the prefix for nested types).
    uint64_t off = 0;
    char local[kPbMaxName];
    local[0] = '\0';
    while (off < n) {
        int32_t f = 0; PbWireType w = PbWireType::kVarint;
        if (!pb_read_tag(p, n, &off, &f, &w)) return false;
        if (f == kDpName && w == PbWireType::kLenDelim) {
            const uint8_t* s = nullptr; uint64_t sl = 0;
            if (!read_slice(p, n, &off, &s, &sl)) return false;
            if (!copy_str(local, kPbMaxName, s, static_cast<uint32_t>(sl)))
                return false;
        } else if (!pb_skip_field(p, n, &off, w)) {
            return false;
        }
    }
    if (!join_name(m->full_name, kPbMaxFullName, prefix, local)) return false;

    // Pass 2: fields, then nested types (which append after us).
    off = 0;
    while (off < n) {
        int32_t f = 0; PbWireType w = PbWireType::kVarint;
        if (!pb_read_tag(p, n, &off, &f, &w)) return false;
        if (f == kDpField && w == PbWireType::kLenDelim) {
            const uint8_t* s = nullptr; uint64_t sl = 0;
            if (!read_slice(p, n, &off, &s, &sl)) return false;
            if (m->n_fields >= kPbMaxFields) return false;
            if (!parse_field_desc(s, sl, &m->field[m->n_fields])) return false;
            ++m->n_fields;
        } else if (f == kDpNestedType && w == PbWireType::kLenDelim) {
            const uint8_t* s = nullptr; uint64_t sl = 0;
            if (!read_slice(p, n, &off, &s, &sl)) return false;
            // `m` may be re-read after the recursive append; the array is
            // fixed-capacity so the pointer stays valid.
            if (!parse_message_desc(s, sl, set->msg[my].full_name, set,
                                    depth + 1u)) return false;
        } else if (!pb_skip_field(p, n, &off, w)) {
            return false;
        }
    }
    return true;
}

bool parse_file_desc(const uint8_t* p, uint64_t n, PbDescriptorSet* set) noexcept {
    assert(set != nullptr);
    char pkg[kPbMaxFullName];
    pkg[0] = '\0';
    uint64_t off = 0;
    while (off < n) {                                // pass 1: package
        int32_t f = 0; PbWireType w = PbWireType::kVarint;
        if (!pb_read_tag(p, n, &off, &f, &w)) return false;
        if (f == kFdpPackage && w == PbWireType::kLenDelim) {
            const uint8_t* s = nullptr; uint64_t sl = 0;
            if (!read_slice(p, n, &off, &s, &sl)) return false;
            if (!copy_str(pkg, kPbMaxFullName, s, static_cast<uint32_t>(sl)))
                return false;
        } else if (!pb_skip_field(p, n, &off, w)) {
            return false;
        }
    }
    off = 0;
    while (off < n) {                                // pass 2: messages
        int32_t f = 0; PbWireType w = PbWireType::kVarint;
        if (!pb_read_tag(p, n, &off, &f, &w)) return false;
        if (f == kFdpMessageType && w == PbWireType::kLenDelim) {
            const uint8_t* s = nullptr; uint64_t sl = 0;
            if (!read_slice(p, n, &off, &s, &sl)) return false;
            if (!parse_message_desc(s, sl, pkg, set, 0u)) return false;
        } else if (!pb_skip_field(p, n, &off, w)) {
            return false;
        }
    }
    return true;
}

}  // namespace

int32_t pb_find_message(const PbDescriptorSet* set, const char* full_name) noexcept {
    assert(set != nullptr);
    assert(full_name != nullptr);
    const char* want = (full_name[0] == '.') ? full_name + 1 : full_name;
    for (uint32_t i = 0; i < set->n_msgs; ++i) {     // bounded: kPbMaxMessages
        if (std::strcmp(set->msg[i].full_name, want) == 0) {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

bool pb_parse_descriptor_set(const uint8_t* src, uint64_t len,
                             Arena* arena, PbDescriptorSet* out) noexcept {
    assert(out != nullptr);
    assert(src != nullptr || len == 0u);
    (void)arena;                     // descriptor table is caller-owned storage
    std::memset(out, 0, sizeof(*out));
    uint64_t off = 0;
    while (off < len) {                              // bounded by len
        int32_t f = 0; PbWireType w = PbWireType::kVarint;
        if (!pb_read_tag(src, len, &off, &f, &w)) return false;
        if (f == kFdsFile && w == PbWireType::kLenDelim) {
            const uint8_t* s = nullptr; uint64_t sl = 0;
            if (!read_slice(src, len, &off, &s, &sl)) return false;
            if (!parse_file_desc(s, sl, out)) return false;
        } else if (!pb_skip_field(src, len, &off, w)) {
            return false;
        }
    }
    // Resolve message-typed fields to indexes now that every name is known.
    for (uint32_t i = 0; i < out->n_msgs; ++i) {
        PbMessageDesc* m = &out->msg[i];
        for (uint32_t j = 0; j < m->n_fields; ++j) {
            PbFieldDesc* fd = &m->field[j];
            if (fd->type == PbType::kMessage && fd->type_name[0] != '\0') {
                fd->msg_index = pb_find_message(out, fd->type_name);
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Layer 3 — message decode
// ---------------------------------------------------------------------------

bool pb_decode_message(const uint8_t* src, uint64_t len,
                       const PbMessageDesc* desc, const PbDescriptorSet* set,
                       PbValue* out_vals, uint32_t out_cap) noexcept {
    assert(desc != nullptr);
    assert(out_vals != nullptr);
    if (desc->n_fields > out_cap) return false;
    (void)set;

    for (uint32_t i = 0; i < desc->n_fields; ++i) {
        PbValue* v = &out_vals[i];
        std::memset(v, 0, sizeof(*v));
        v->type      = desc->field[i].type;
        v->is_null   = true;
        v->msg_index = desc->field[i].msg_index;
    }

    uint64_t off = 0;
    while (off < len) {                              // bounded by len
        int32_t    fnum = 0;
        PbWireType w    = PbWireType::kVarint;
        if (!pb_read_tag(src, len, &off, &fnum, &w)) return false;

        uint32_t slot = UINT32_MAX;
        for (uint32_t i = 0; i < desc->n_fields; ++i) {
            if (desc->field[i].number == fnum) { slot = i; break; }
        }
        if (slot == UINT32_MAX) {                    // unknown ⇒ skip (fwd compat)
            if (!pb_skip_field(src, len, &off, w)) return false;
            continue;
        }

        const PbFieldDesc* fd = &desc->field[slot];
        PbValue*           v  = &out_vals[slot];

        // Packed repeated scalars: length-delimited payload for a numeric type.
        if (fd->repeated && w == PbWireType::kLenDelim &&
            is_numeric_type(fd->type)) {
            const uint8_t* p = nullptr; uint64_t n = 0;
            if (!read_slice(src, len, &off, &p, &n)) return false;
            uint32_t cnt = 0;
            PbValue* first = (v->rep_count == 0u) ? v : nullptr;
            if (!scan_packed(p, n, fd->type, first, &cnt)) return false;
            v->rep_count += cnt;
            if (cnt > 0u) v->is_null = false;
            continue;
        }

        if (w != native_wire(fd->type)) {            // type/wire mismatch: skip
            if (!pb_skip_field(src, len, &off, w)) return false;
            continue;
        }

        PbValue tmp{};
        if (!decode_one(src, len, &off, fd->type, w, &tmp)) return false;
        // Repeated ⇒ keep FIRST; scalar ⇒ last-one-wins (protobuf semantics).
        if (!fd->repeated || v->rep_count == 0u) {
            const uint32_t keep = v->rep_count;
            tmp.type      = fd->type;
            tmp.is_null   = false;
            tmp.msg_index = fd->msg_index;
            tmp.rep_count = keep;
            *v = tmp;
        }
        v->rep_count += 1u;
        v->is_null = false;
    }
    return true;
}

bool pb_decode_repeated_field(const uint8_t* src, uint64_t len,
                              int32_t field_number, PbType type,
                              PbValue* out_vals, uint32_t out_cap,
                              uint32_t* out_n) noexcept {
    assert(out_vals != nullptr || out_cap == 0u);
    assert(out_n != nullptr);
    *out_n = 0;
    const PbWireType want = native_wire(type);
    uint64_t off = 0;
    while (off < len) {                              // bounded by len
        int32_t    fnum = 0;
        PbWireType w    = PbWireType::kVarint;
        if (!pb_read_tag(src, len, &off, &fnum, &w)) return false;
        if (fnum != field_number) {
            if (!pb_skip_field(src, len, &off, w)) return false;
            continue;
        }
        if (w == PbWireType::kLenDelim && is_numeric_type(type)) {
            const uint8_t* p = nullptr; uint64_t n = 0;   // packed run
            if (!read_slice(src, len, &off, &p, &n)) return false;
            uint64_t o2 = 0;
            while (o2 < n) {
                PbValue tmp{};
                if (!decode_one(p, n, &o2, type, want, &tmp)) return false;
                if (*out_n < out_cap) {
                    tmp.type = type; tmp.is_null = false;
                    tmp.msg_index = -1; tmp.rep_count = 1u;
                    out_vals[*out_n] = tmp;
                }
                ++(*out_n);
            }
            continue;
        }
        if (w != want) {
            if (!pb_skip_field(src, len, &off, w)) return false;
            continue;
        }
        PbValue tmp{};
        if (!decode_one(src, len, &off, type, w, &tmp)) return false;
        if (*out_n < out_cap) {
            tmp.type = type; tmp.is_null = false;
            tmp.msg_index = -1; tmp.rep_count = 1u;
            out_vals[*out_n] = tmp;
        }
        ++(*out_n);
    }
    if (*out_n > out_cap) *out_n = out_cap;          // report what we wrote
    return true;
}

// ---------------------------------------------------------------------------
// Confluent framing
// ---------------------------------------------------------------------------

bool pb_confluent_parse(const uint8_t* src, uint64_t len,
                        uint32_t* out_schema_id,
                        int32_t* out_msg_index, uint32_t idx_cap,
                        uint32_t* out_n_idx,
                        const uint8_t** out_payload,
                        uint64_t* out_payload_len) noexcept {
    assert(out_schema_id != nullptr && out_n_idx != nullptr);
    assert(out_payload != nullptr && out_payload_len != nullptr);
    if (src == nullptr || len < 5u) return false;
    if (src[0] != 0x00u) return false;                       // magic byte
    *out_schema_id = (static_cast<uint32_t>(src[1]) << 24) |
                     (static_cast<uint32_t>(src[2]) << 16) |
                     (static_cast<uint32_t>(src[3]) << 8)  |
                      static_cast<uint32_t>(src[4]);
    uint64_t off = 5u;
    uint64_t count = 0;
    if (!pb_read_varint(src, len, &off, &count)) return false;
    if (count == 0u) {                                       // shorthand for [0]
        *out_n_idx = 1u;
        if (idx_cap >= 1u && out_msg_index != nullptr) out_msg_index[0] = 0;
    } else {
        if (count > kPbMaxMsgIndexes) return false;
        *out_n_idx = static_cast<uint32_t>(count);
        for (uint64_t i = 0; i < count; ++i) {               // bounded
            uint64_t raw = 0;
            if (!pb_read_varint(src, len, &off, &raw)) return false;
            const int64_t v = pb_zigzag_decode(raw);
            if (i < idx_cap && out_msg_index != nullptr) {
                out_msg_index[i] = static_cast<int32_t>(v);
            }
        }
    }
    *out_payload     = src + off;
    *out_payload_len = len - off;
    return true;
}

}  // namespace ingest
}  // namespace bolt
