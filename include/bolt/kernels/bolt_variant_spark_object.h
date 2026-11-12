// bolt/kernels/bolt_variant_spark_object.h — Spark to_variant_object(x) (G2CHK-963): a struct / array / map value (a nested-row
// value with its type tree) as one variant. A struct and a string-keyed map become objects (values in field order, entries
// sorted by key, the metadata dictionary in first-seen key order: VariantBuilder), an array an array, a NULL field or element a
// variant null, and every leaf what CAST(leaf AS VARIANT) gives: the narrowest integer, a decimal sized by its value's digits.
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/kernels/bolt_json_spark_nested.h"
#include "bolt/kernels/bolt_variant_spark.h"

namespace bolt {
namespace kernels {
namespace variantspark {

namespace object_detail {

namespace nrv = ::bolt::nested_row;

inline bool leaf(detail::Enc* e, const nrv::TypeNode& nd, const nrv::LeafView& lv) noexcept {
    uint8_t b[24];
    uint32_t n = 0;
    switch (nd.type) {
        case BoltType::Bool: n = encode_bool(b, sizeof b, lv.bytes[0] != 0); break;
        case BoltType::Int8: case BoltType::Int16: case BoltType::Int32: case BoltType::Int64: {
            const int64_t v = jsonspark::tojd::read_int(lv.bytes, lv.len);
            if (nd.scale != 0) {                                     // a narrow decimal rides the integer lane with its scale
                const int64_t hi = v < 0 ? -1 : 0;
                n = encode_decimal(b, sizeof b, hi, static_cast<uint64_t>(v),
                                   decimal_value_digits(hi, static_cast<uint64_t>(v)), nd.scale);
            } else {
                n = encode_int(b, sizeof b, v, long_width(v));
            }
            break;
        }
        case BoltType::Float64: { double d; std::memcpy(&d, lv.bytes, 8); n = encode_double(b, sizeof b, d); break; }
        case BoltType::Float32: { float f; std::memcpy(&f, lv.bytes, 4); n = encode_float(b, sizeof b, f); break; }
        case BoltType::Decimal128: {
            uint64_t lo = 0;
            int64_t hi = 0;
            std::memcpy(&lo, lv.bytes, 8);
            std::memcpy(&hi, lv.bytes + 8, 8);
            n = encode_decimal(b, sizeof b, hi, lo, decimal_value_digits(hi, lo), nd.scale);
            break;
        }
        case BoltType::Date32: { int32_t d; std::memcpy(&d, lv.bytes, 4); n = encode_temporal(b, sizeof b, kDate, d); break; }
        case BoltType::Timestamp: case BoltType::TimestampNtz: {
            int64_t us; std::memcpy(&us, lv.bytes, 8);
            n = encode_temporal(b, sizeof b, nd.type == BoltType::TimestampNtz ? kTimestampNtz : kTimestamp, us);
            break;
        }
        case BoltType::Utf8: case BoltType::Binary: {                // written in place: up to the variant size limit
            const bool str = nd.type == BoltType::Utf8;
            const uint32_t hdr = str && lv.len < 64u ? 1u : 5u;
            if (!detail::room(e, hdr + lv.len)) return false;
            uint8_t* q = e->out->value + e->out->value_n;
            if (str) encode_string(q, hdr + lv.len, lv.bytes, lv.len);
            else encode_binary(q, hdr + lv.len, lv.bytes, lv.len);
            e->out->value_n += hdr + lv.len;
            return true;
        }
        default: e->st = Status::Malformed; return false;          // admitted by the caller's type check
    }
    if (n == 0) { e->st = Status::Malformed; return false; }
    return detail::put_bytes(e, b, n);
}

inline bool value(detail::Enc* e, const jsonspark::Tree& c, uint32_t ni, nrv::Span v, uint32_t id, uint32_t depth) noexcept;

inline bool null_value(detail::Enc* e, uint32_t id) noexcept {
    return detail::add_entry(e, id) && detail::put_byte(e, detail::prim(kNull));
}

inline bool fields(detail::Enc* e, const jsonspark::Tree& c, uint32_t ni, nrv::Span v, uint32_t depth) noexcept {
    const nrv::TypeNode& nd = c.t.nodes[ni];
    for (uint32_t k = 0; k < nd.n_children; ++k) {                  // bounded by the struct's fields
        nrv::Span fs{};
        if (nrv::struct_field(c.t, ni, v, k, &fs) != 1) { e->st = Status::Malformed; return false; }
        const uint32_t ci = nd.first_child + k;
        uint32_t fid = 0;
        if (!detail::key_id(e, reinterpret_cast<const uint8_t*>(c.meta[ci].name), c.meta[ci].name_len, &fid) ||
            !value(e, c, ci, fs, fid, depth + 1u)) return false;
    }
    return true;
}

inline bool entries(detail::Enc* e, const jsonspark::Tree& c, uint32_t ni, nrv::Span v, uint32_t depth, bool map) noexcept {
    nrv::ListCursor cur{};
    if (nrv::list_open(c.t, ni, v, &cur) != 1) { e->st = Status::Malformed; return false; }
    const uint32_t en = c.t.nodes[ni].first_child;
    nrv::Span x{};
    while (nrv::list_next(&cur, &x)) {                               // bounded by the list's count
        if (!map) {
            if (!value(e, c, en, x, 0, depth + 1u)) return false;
            continue;
        }
        const uint32_t kn = c.t.nodes[en].first_child;
        nrv::Span ks{}, vs{};
        nrv::LeafView kl{};
        if (nrv::struct_field(c.t, en, x, 0, &ks) != 1 || nrv::struct_field(c.t, en, x, 1, &vs) != 1 ||
            !nrv::leaf_view(c.t, kn, ks, &kl) || kl.is_null || c.t.nodes[kn].type != BoltType::Utf8) {
            e->st = Status::Malformed;
            return false;
        }
        uint32_t fid = 0;
        if (!detail::key_id(e, kl.bytes, kl.len, &fid) || !value(e, c, kn + 1u, vs, fid, depth + 1u)) return false;
    }
    return true;
}

inline bool value(detail::Enc* e, const jsonspark::Tree& c, uint32_t ni, nrv::Span v, uint32_t id, uint32_t depth) noexcept {
    assert(ni < c.t.n);
    if (depth > nrv::kMaxDepth + 1u) { e->st = Status::Malformed; return false; }
    const nrv::TypeNode& nd = c.t.nodes[ni];
    if (jsonspark::tojd::is_null(nd, v)) return null_value(e, id);
    if (nd.type == BoltType::Struct && nd.logical == BoltLogical::Variant) {   // an embedded variant (G2CHK-984)
        nrv::Span ms{}, vs{};
        nrv::LeafView ml{}, vl{};
        if (nd.n_children != 2 || nrv::struct_field(c.t, ni, v, 0, &ms) != 1 || nrv::struct_field(c.t, ni, v, 1, &vs) != 1 ||
            !nrv::leaf_view(c.t, nd.first_child, ms, &ml) || !nrv::leaf_view(c.t, nd.first_child + 1u, vs, &vl) ||
            ml.is_null || vl.is_null) {
            e->st = Status::Malformed;
            return false;
        }
        return detail::append_variant(e, ml.bytes, ml.len, vl.bytes, vl.len, id, 0);
    }
    if (nd.type == BoltType::Struct || nd.type == BoltType::List || nd.type == BoltType::Map) {
        const bool obj = nd.type != BoltType::List;
        if (!detail::open_container(e, obj, id)) return false;
        const bool ok = nd.type == BoltType::Struct ? fields(e, c, ni, v, depth) : entries(e, c, ni, v, depth, nd.type == BoltType::Map);
        return ok && detail::finish_container(e);
    }
    nrv::LeafView lv{};
    if (!nrv::leaf_view(c.t, ni, v, &lv)) { e->st = Status::Malformed; return false; }
    if (lv.is_null) return null_value(e, id);
    return detail::add_entry(e, id) && leaf(e, nd, lv);
}

}  // namespace object_detail

// to_variant_object of the value `v` of node `root` (a struct, list or map) into `out`. Status::DuplicateKey cannot occur for a
// struct (distinct names) or a map (distinct keys); Malformed is a value the type does not describe.
inline Status to_variant_object(const jsonspark::Tree& c, uint32_t root, nested_row::Span v, Workspace* w, Encoded* out) noexcept {
    assert(w != nullptr && out != nullptr && root < c.t.n);
    std::memset(w->table, 0, sizeof(uint32_t) * w->table_cap);
    w->key_off[0] = 0;
    out->value_n = 0;
    out->metadata_n = 0;
    detail::Enc e{w, out, 0, 0, 0, 0, Status::Ok};
    if (!object_detail::value(&e, c, root, v, 0, 0)) return e.st == Status::Ok ? Status::Malformed : e.st;
    return detail::write_metadata(&e) ? Status::Ok : e.st;
}

}  // namespace variantspark
}  // namespace kernels
}  // namespace bolt
