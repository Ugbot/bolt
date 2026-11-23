// OTLP traces decode. See bolt_otlp.h.
//
// opentelemetry.proto.collector.trace.v1.ExportTraceServiceRequest:
//   resource_spans = 1 { resource = 1; scope_spans = 2 { scope = 1; spans = 2 } }
//   Span: trace_id = 1, span_id = 2, trace_state = 3, parent_span_id = 4, name = 5, kind = 6,
//         start_time_unix_nano = 7 (fixed64), end_time_unix_nano = 8 (fixed64), attributes = 9,
//         events = 11 { time_unix_nano = 1 (fixed64); name = 2; attributes = 3 },
//         links = 13 { trace_id = 1; span_id = 2; trace_state = 3; attributes = 4 },
//         status = 15 { message = 2; code = 3 }, flags = 16 (fixed32)

#include "bolt/ingest/bolt_otlp.h"

#include <cassert>
#include <cstdio>


namespace bolt::ingest::otlp {

using namespace bolt::ingest::pb;

namespace {

void put_u64(Out* o, std::uint64_t v) noexcept {
    char buf[24];
    const int n = std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(v));
    out_put(o, std::string_view(buf, n > 0 ? static_cast<std::size_t>(n) : 0u));
}

// Span.events as a JSON array into `o`; `*count` is how many there were.
bool events_json(Cursor span, Out* o, std::uint32_t* count) noexcept {
    out_put(o, "[");
    const bool ok = each_len(span, 11, [&](Cursor ev) {
        std::uint64_t ts = 0; std::string_view name;
        Cursor walk = ev;
        while (!walk.empty()) {   // bounded: the event's bytes
            std::uint32_t f = 0; std::uint8_t w = 0;
            if (!read_tag(&walk, &f, &w)) return false;
            if (f == 1 && w == kWireF64) { if (!read_fixed64(&walk, &ts)) return false; }
            else if (f == 2 && w == kWireLen) { Cursor v; if (!read_len_delimited(&walk, &v)) return false; name = view_of(v); }
            else if (!skip_field(&walk, w)) return false;
        }
        out_put(o, (*count)++ ? ",{\"ts\":" : "{\"ts\":");
        put_u64(o, ts);
        out_put(o, ",\"name\":"); out_json_string(o, name);
        out_put(o, ",\"attrs\":{");
        std::uint32_t members = 0;
        if (!kv_members(ev, 3, o, &members)) return false;
        out_put(o, "}}");
        return o->ok;
    });
    out_put(o, "]");
    return ok && o->ok;
}

// Span.links as a JSON array into `o`.
bool links_json(Cursor span, Out* o, std::uint32_t* count) noexcept {
    out_put(o, "[");
    const bool ok = each_len(span, 13, [&](Cursor ln) {
        std::string_view trace, sid;
        Cursor walk = ln;
        while (!walk.empty()) {   // bounded: the link's bytes
            std::uint32_t f = 0; std::uint8_t w = 0;
            if (!read_tag(&walk, &f, &w)) return false;
            if ((f == 1 || f == 2) && w == kWireLen) {
                Cursor v; if (!read_len_delimited(&walk, &v)) return false;
                (f == 1 ? trace : sid) = view_of(v);
            } else if (!skip_field(&walk, w)) return false;
        }
        out_put(o, (*count)++ ? ",{\"trace_id\":" : "{\"trace_id\":");
        out_hex_string(o, trace);
        out_put(o, ",\"span_id\":"); out_hex_string(o, sid);
        out_put(o, ",\"attrs\":{");
        std::uint32_t members = 0;
        if (!kv_members(ln, 4, o, &members)) return false;
        out_put(o, "}}");
        return o->ok;
    });
    out_put(o, "]");
    return ok && o->ok;
}

bool read_status(Cursor st, std::string_view* msg, std::uint64_t* code) noexcept {
    while (!st.empty()) {   // bounded: the message's bytes
        std::uint32_t f = 0; std::uint8_t w = 0;
        if (!read_tag(&st, &f, &w)) return false;
        if (f == 2 && w == kWireLen) { Cursor v; if (!read_len_delimited(&st, &v)) return false; *msg = view_of(v); }
        else if (f == 3 && w == kWireVarint) { if (!read_varint(&st, code)) return false; }
        else if (!skip_field(&st, w)) return false;
    }
    return true;
}

bool decode_span(Cursor span, const Scope& scope, std::string_view stream, bolt::Arena* a, SpanRows* rows,
                 std::uint32_t i) noexcept {
    const Cursor whole = span;
    const std::size_t in_bytes = span.remaining();
    std::uint64_t start = 0, end = 0, kind = 0, code = 0;
    std::uint32_t flags = 0;
    std::string_view trace, sid, parent, state, name, msg;
    while (!span.empty()) {   // bounded: the span's bytes
        std::uint32_t f = 0; std::uint8_t w = 0;
        if (!read_tag(&span, &f, &w)) return false;
        if (f == 7 && w == kWireF64) { if (!read_fixed64(&span, &start)) return false; }
        else if (f == 8 && w == kWireF64) { if (!read_fixed64(&span, &end)) return false; }
        else if (f == 6 && w == kWireVarint) { if (!read_varint(&span, &kind)) return false; }
        else if (f == 16 && w == kWireF32) { if (!read_fixed32(&span, &flags)) return false; }
        else if (w == kWireLen && (f <= 5 || f == 15)) {
            Cursor v; if (!read_len_delimited(&span, &v)) return false;
            if (f == 1) trace = view_of(v);
            else if (f == 2) sid = view_of(v);
            else if (f == 3) state = view_of(v);
            else if (f == 4) parent = view_of(v);
            else if (f == 5) name = view_of(v);
            else if (!read_status(v, &msg, &code)) return false;
        } else if (!skip_field(&span, w)) return false;
    }
    if (trace.size() != 16 || sid.size() != 8 || all_zero(sid)) return false;
    Out attrs, events, links;
    if (!out_init(&attrs, a, in_bytes + scope.name.size() + scope.version.size()) ||
        !out_init(&events, a, in_bytes) || !out_init(&links, a, in_bytes)) return false;
    std::uint32_t members = 0, n_events = 0, n_links = 0;
    out_put(&attrs, "{");
    if (!kv_members(whole, 9, &attrs, &members)) return false;
    close_attrs(&attrs, scope, &members);
    if (!attrs.ok || !events_json(whole, &events, &n_events) || !links_json(whole, &links, &n_links)) return false;
    rows->start_ts[i] = static_cast<std::int64_t>(start);
    rows->duration_ns[i] = end > start ? static_cast<std::int64_t>(end - start) : 0;
    rows->flags[i] = flags;
    rows->kind[i] = kind <= 5u ? static_cast<std::uint8_t>(kind) : std::uint8_t{0};
    rows->status_code[i] = code <= 2u ? static_cast<std::uint8_t>(code) : std::uint8_t{0};
    rows->trace_id[i] = trace;
    rows->span_id[i] = sid;
    rows->parent_span_id[i] = (parent.size() == 8 && !all_zero(parent)) ? parent : std::string_view();
    rows->name[i] = name;
    rows->trace_state[i] = state;
    rows->status_msg[i] = msg;
    rows->attrs[i] = members != 0 ? attrs.view() : std::string_view();
    rows->events[i] = n_events != 0 ? events.view() : std::string_view();
    rows->links[i] = n_links != 0 ? links.view() : std::string_view();
    rows->stream[i] = stream;
    return true;
}

}  // namespace

bool count_spans(std::string_view body, std::uint64_t* spans) noexcept {
    assert(spans != nullptr);
    *spans = 0;
    return each_len(cursor_of(body), 1, [&](Cursor rs) {
        return each_len(rs, 2, [&](Cursor ss) {
            return each_len(ss, 2, [&](Cursor) { ++*spans; return true; });
        });
    });
}

Status decode_spans(std::string_view body, bolt::Arena* a, SpanRows* rows) noexcept {
    assert(a != nullptr && rows != nullptr);
    std::uint64_t count = 0;
    if (!count_spans(body, &count) || count > UINT32_MAX) return Status::kMalformed;
    const auto n = static_cast<std::uint32_t>(count);
    const std::uint32_t cap = n ? n : 1u;
    rows->n = n;
    rows->start_ts = a->allocate_array<std::int64_t>(cap);
    rows->duration_ns = a->allocate_array<std::int64_t>(cap);
    rows->flags = a->allocate_array<std::int64_t>(cap);
    rows->kind = a->allocate_array<std::uint8_t>(cap);
    rows->status_code = a->allocate_array<std::uint8_t>(cap);
    std::string_view** texts[] = {&rows->trace_id, &rows->span_id, &rows->parent_span_id, &rows->name, &rows->trace_state,
                                  &rows->status_msg, &rows->attrs, &rows->events, &rows->links, &rows->stream};
    for (std::string_view** t : texts) {
        *t = a->allocate_array<std::string_view>(cap);
        if (*t == nullptr) return Status::kArena;
    }
    if (!rows->start_ts || !rows->duration_ns || !rows->flags || !rows->kind || !rows->status_code)
        return Status::kArena;
    std::uint32_t i = 0;
    const bool ok = each_len(cursor_of(body), 1, [&](Cursor rs) {
        Cursor resource; std::string_view stream;
        if (!first_len(rs, 1, &resource)) return false;
        if (resource.p != nullptr && !resource_stream(resource, a, &stream)) return false;
        return each_len(rs, 2, [&](Cursor ss) {
            Cursor sc; Scope scope;
            if (!first_len(ss, 1, &sc)) return false;
            if (sc.p != nullptr && !read_scope(sc, &scope)) return false;
            return each_len(ss, 2, [&](Cursor span) { return i < n && decode_span(span, scope, stream, a, rows, i++); });
        });
    });
    if (!ok || i != n) return Status::kMalformed;
    return Status::kOk;
}

}  // namespace bolt::ingest::otlp
