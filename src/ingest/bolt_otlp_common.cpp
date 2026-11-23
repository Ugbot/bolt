// OTLP shared decoding: AnyValue, KeyValue, resource stream bytes. See bolt_otlp.h.

#include "bolt/ingest/bolt_otlp.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace bolt::ingest::otlp {

using namespace bolt::ingest::pb;

bool out_init(Out* o, bolt::Arena* a, std::size_t in_bytes) noexcept {
    assert(o != nullptr && a != nullptr);
    o->cap = in_bytes * 6u + 256u;
    o->p = a->allocate_array<char>(o->cap);
    o->n = 0;
    o->ok = o->p != nullptr;
    return o->ok;
}

void out_put(Out* o, std::string_view s) noexcept {
    assert(o != nullptr);
    if (!o->ok || o->cap - o->n < s.size()) { o->ok = false; return; }
    if (!s.empty()) std::memcpy(o->p + o->n, s.data(), s.size());
    o->n += s.size();
}

void out_json_string(Out* o, std::string_view s) noexcept {
    assert(o != nullptr);
    static const char kHex[] = "0123456789abcdef";
    if (!o->ok || o->cap - o->n < s.size() * 6u + 2u) { o->ok = false; return; }
    char* w = o->p + o->n;
    *w++ = '"';
    for (const char ch : s) {   // bounded: s.size()
        const auto c = static_cast<unsigned char>(ch);
        if (c == '"' || c == '\\') { *w++ = '\\'; *w++ = ch; }
        else if (c >= 0x20) { *w++ = ch; }
        else if (c == '\n') { *w++ = '\\'; *w++ = 'n'; }
        else if (c == '\t') { *w++ = '\\'; *w++ = 't'; }
        else if (c == '\r') { *w++ = '\\'; *w++ = 'r'; }
        else { *w++ = '\\'; *w++ = 'u'; *w++ = '0'; *w++ = '0'; *w++ = kHex[c >> 4]; *w++ = kHex[c & 15]; }
    }
    *w++ = '"';
    o->n = static_cast<std::size_t>(w - o->p);
}

namespace {

void out_i64(Out* o, std::int64_t v) noexcept {
    char buf[24];
    const int n = std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v));
    out_put(o, std::string_view(buf, n > 0 ? static_cast<std::size_t>(n) : 0u));
}

// JSON has no NaN or infinity: those go out as strings.
void out_f64(Out* o, double v) noexcept {
    if (std::isnan(v)) { out_put(o, "\"NaN\""); return; }
    if (std::isinf(v)) { out_put(o, v > 0 ? "\"+Inf\"" : "\"-Inf\""); return; }
    char buf[32];
    const int n = std::snprintf(buf, sizeof(buf), "%.17g", v);
    out_put(o, std::string_view(buf, n > 0 ? static_cast<std::size_t>(n) : 0u));
}

void out_hex(Out* o, std::string_view bytes) noexcept {
    static const char kHex[] = "0123456789abcdef";
    if (!o->ok || o->cap - o->n < bytes.size() * 2u + 2u) { o->ok = false; return; }
    char* w = o->p + o->n;
    *w++ = '"';
    for (const char ch : bytes) {   // bounded: bytes.size()
        const auto c = static_cast<unsigned char>(ch);
        *w++ = kHex[c >> 4]; *w++ = kHex[c & 15];
    }
    *w++ = '"';
    o->n = static_cast<std::size_t>(w - o->p);
}

}  // namespace

bool any_json(Cursor v, Out* o, std::uint32_t depth) noexcept {
    assert(o != nullptr);
    if (depth > kValueDepth) return false;
    if (v.empty()) { out_put(o, "null"); return o->ok; }   // an AnyValue with nothing set
    std::uint32_t field = 0; std::uint8_t wire = 0;
    if (!read_tag(&v, &field, &wire)) return false;
    if (field == 1 && wire == kWireLen) {                  // string_value
        Cursor s; if (!read_len_delimited(&v, &s)) return false;
        out_json_string(o, view_of(s));
    } else if (field == 2 && wire == kWireVarint) {        // bool_value
        std::uint64_t b = 0; if (!read_varint(&v, &b)) return false;
        out_put(o, b ? "true" : "false");
    } else if (field == 3 && wire == kWireVarint) {        // int_value
        std::uint64_t i = 0; if (!read_varint(&v, &i)) return false;
        out_i64(o, static_cast<std::int64_t>(i));
    } else if (field == 4 && wire == kWireF64) {           // double_value
        std::uint64_t bits = 0; if (!read_fixed64(&v, &bits)) return false;
        double d = 0; std::memcpy(&d, &bits, 8);
        out_f64(o, d);
    } else if (field == 5 && wire == kWireLen) {           // array_value { repeated AnyValue values = 1 }
        Cursor arr; if (!read_len_delimited(&v, &arr)) return false;
        out_put(o, "[");
        bool first = true;
        while (!arr.empty()) {   // bounded: the array's bytes
            std::uint32_t f = 0; std::uint8_t w = 0;
            if (!read_tag(&arr, &f, &w)) return false;
            if (f != 1 || w != kWireLen) { if (!skip_field(&arr, w)) return false; continue; }
            Cursor item; if (!read_len_delimited(&arr, &item)) return false;
            if (!first) out_put(o, ",");
            first = false;
            if (!any_json(item, o, depth + 1)) return false;
        }
        out_put(o, "]");
    } else if (field == 6 && wire == kWireLen) {           // kvlist_value { repeated KeyValue values = 1 }
        Cursor kvs; if (!read_len_delimited(&v, &kvs)) return false;
        out_put(o, "{");
        bool first = true;
        while (!kvs.empty()) {   // bounded: the list's bytes
            std::uint32_t f = 0; std::uint8_t w = 0;
            if (!read_tag(&kvs, &f, &w)) return false;
            if (f != 1 || w != kWireLen) { if (!skip_field(&kvs, w)) return false; continue; }
            Cursor kv; if (!read_len_delimited(&kvs, &kv)) return false;
            std::string_view key; Cursor val{};
            if (!key_value(kv, &key, &val)) return false;
            if (!first) out_put(o, ",");
            first = false;
            out_json_string(o, key); out_put(o, ":");
            if (!any_json(val, o, depth + 1)) return false;
        }
        out_put(o, "}");
    } else if (field == 7 && wire == kWireLen) {           // bytes_value
        Cursor b; if (!read_len_delimited(&v, &b)) return false;
        out_hex(o, view_of(b));
    } else {
        if (!skip_field(&v, wire)) return false;
        out_put(o, "null");
    }
    return o->ok;
}

bool any_text(Cursor v, Out* o) noexcept {
    assert(o != nullptr);
    Cursor peek = v;
    std::uint32_t field = 0; std::uint8_t wire = 0;
    if (!peek.empty() && read_tag(&peek, &field, &wire) && field == 1 && wire == kWireLen) {
        Cursor s; if (!read_len_delimited(&peek, &s)) return false;
        out_put(o, view_of(s));
        return o->ok;
    }
    if (v.empty()) return true;   // no value: empty text
    return any_json(v, o, 0);
}

bool key_value(Cursor kv, std::string_view* key, Cursor* value) noexcept {
    assert(key != nullptr && value != nullptr);
    *key = std::string_view(); value->p = nullptr; value->end = nullptr;
    while (!kv.empty()) {   // bounded: the message's bytes
        std::uint32_t f = 0; std::uint8_t w = 0;
        if (!read_tag(&kv, &f, &w)) return false;
        if (f == 1 && w == kWireLen) { Cursor k; if (!read_len_delimited(&kv, &k)) return false; *key = view_of(k); }
        else if (f == 2 && w == kWireLen) { if (!read_len_delimited(&kv, value)) return false; }
        else if (!skip_field(&kv, w)) return false;
    }
    return true;
}

bool kv_members(Cursor msg, std::uint32_t field, Out* o, std::uint32_t* members) noexcept {
    assert(o != nullptr && members != nullptr);
    while (!msg.empty()) {   // bounded: the message's bytes
        std::uint32_t f = 0; std::uint8_t w = 0;
        if (!read_tag(&msg, &f, &w)) return false;
        if (f != field || w != kWireLen) { if (!skip_field(&msg, w)) return false; continue; }
        Cursor kv; if (!read_len_delimited(&msg, &kv)) return false;
        std::string_view key; Cursor val{};
        if (!key_value(kv, &key, &val)) return false;
        if (*members != 0) out_put(o, ",");
        ++*members;
        out_json_string(o, key); out_put(o, ":");
        if (!any_json(val, o, 0)) return false;
    }
    return o->ok;
}

bool resource_stream(Cursor resource, bolt::Arena* a, std::string_view* out) noexcept {
    assert(a != nullptr && out != nullptr);
    struct Attr { std::string_view key; Cursor val; };
    Attr attrs[kIdentityAttrs];
    std::uint32_t n = 0;
    const std::size_t in_bytes = resource.remaining();
    while (!resource.empty()) {   // bounded: the message's bytes
        std::uint32_t f = 0; std::uint8_t w = 0;
        if (!read_tag(&resource, &f, &w)) return false;
        if (f != 1 || w != kWireLen) { if (!skip_field(&resource, w)) return false; continue; }
        Cursor kv; if (!read_len_delimited(&resource, &kv)) return false;
        if (n == kIdentityAttrs) return false;
        if (!key_value(kv, &attrs[n].key, &attrs[n].val)) return false;
        ++n;
    }
    std::sort(attrs, attrs + n, [](const Attr& x, const Attr& y) { return x.key < y.key; });
    Out o;
    if (!out_init(&o, a, in_bytes)) return false;
    for (std::uint32_t i = 0; i < n; ++i) {
        if (i != 0 && attrs[i].key == attrs[i - 1].key) continue;   // a repeated name: the first wins
        out_put(&o, attrs[i].key); out_put(&o, std::string_view("\x1F", 1));
        if (!any_text(attrs[i].val, &o)) return false;
        out_put(&o, std::string_view("\x1E", 1));
    }
    *out = o.view();
    return o.ok;
}

bool read_scope(Cursor scope, Scope* s) noexcept {
    assert(s != nullptr);
    while (!scope.empty()) {   // bounded: the message's bytes
        std::uint32_t f = 0; std::uint8_t w = 0;
        if (!read_tag(&scope, &f, &w)) return false;
        if ((f == 1 || f == 2) && w == kWireLen) {
            Cursor v; if (!read_len_delimited(&scope, &v)) return false;
            (f == 1 ? s->name : s->version) = view_of(v);
        } else if (!skip_field(&scope, w)) return false;
    }
    return true;
}

void close_attrs(Out* o, const Scope& scope, std::uint32_t* members) noexcept {
    assert(o != nullptr && members != nullptr);
    if (!scope.name.empty()) {
        out_put(o, (*members)++ ? ",\"otel.scope.name\":" : "\"otel.scope.name\":");
        out_json_string(o, scope.name);
    }
    if (!scope.version.empty()) {
        out_put(o, (*members)++ ? ",\"otel.scope.version\":" : "\"otel.scope.version\":");
        out_json_string(o, scope.version);
    }
    out_put(o, "}");
}

void out_hex_string(Out* o, std::string_view bytes) noexcept { out_hex(o, bytes); }

}  // namespace bolt::ingest::otlp
