// OTLP logs decode. See bolt_otlp.h.
//
// opentelemetry.proto.collector.logs.v1.ExportLogsServiceRequest:
//   resource_logs = 1 { resource = 1; scope_logs = 2 { scope = 1 { name = 1; version = 2 }; log_records = 2 } }
//   LogRecord: time_unix_nano = 1 (fixed64), severity_number = 2, severity_text = 3, body = 5 (AnyValue),
//              attributes = 6, flags = 8, trace_id = 9, span_id = 10, observed_time_unix_nano = 11, event_name = 12

#include "bolt/ingest/bolt_otlp.h"

#include <cassert>


namespace bolt::ingest::otlp {

using namespace bolt::ingest::pb;

namespace {



bool decode_record(Cursor rec, const Scope& scope, std::string_view stream, std::int64_t now_ns, bolt::Arena* a,
                   LogRows* rows, std::uint32_t i) noexcept {
    const std::size_t in_bytes = rec.remaining();
    std::uint64_t time = 0, observed = 0, sev = 0;
    Cursor body{nullptr, nullptr};
    std::string_view sev_text, trace, span, event;
    const Cursor whole = rec;
    while (!rec.empty()) {   // bounded: the record's bytes
        std::uint32_t f = 0; std::uint8_t w = 0;
        if (!read_tag(&rec, &f, &w)) return false;
        if (f == 1 && w == kWireF64) { if (!read_fixed64(&rec, &time)) return false; }
        else if (f == 11 && w == kWireF64) { if (!read_fixed64(&rec, &observed)) return false; }
        else if (f == 2 && w == kWireVarint) { if (!read_varint(&rec, &sev)) return false; }
        else if (w == kWireLen && (f == 3 || f == 5 || f == 9 || f == 10 || f == 12)) {
            Cursor v; if (!read_len_delimited(&rec, &v)) return false;
            if (f == 3) sev_text = view_of(v);
            else if (f == 5) body = v;
            else if (f == 9) trace = view_of(v);
            else if (f == 10) span = view_of(v);
            else event = view_of(v);
        } else if (!skip_field(&rec, w)) return false;
    }
    Out text, attrs;
    if (!out_init(&text, a, body.p != nullptr ? body.remaining() : 0u)) return false;
    if (body.p != nullptr && !any_text(body, &text)) return false;
    if (!out_init(&attrs, a, in_bytes + scope.name.size() + scope.version.size())) return false;
    std::uint32_t members = 0;
    out_put(&attrs, "{");
    if (!kv_members(whole, 6, &attrs, &members)) return false;
    close_attrs(&attrs, scope, &members);
    if (!text.ok || !attrs.ok) return false;
    const std::uint64_t ts = time != 0 ? time : (observed != 0 ? observed : static_cast<std::uint64_t>(now_ns));
    rows->ts[i] = static_cast<std::int64_t>(ts);
    rows->observed_ts[i] = static_cast<std::int64_t>(observed);
    rows->severity[i] = sev <= 24u ? static_cast<std::uint8_t>(sev) : std::uint8_t{0};
    rows->body[i] = text.view();
    rows->stream[i] = stream;
    rows->attrs[i] = members != 0 ? attrs.view() : std::string_view();
    rows->severity_text[i] = sev_text;
    rows->trace_id[i] = (trace.size() == 16 && !all_zero(trace)) ? trace : std::string_view();
    rows->span_id[i] = (span.size() == 8 && !all_zero(span)) ? span : std::string_view();
    rows->event_name[i] = event;
    return true;
}

}  // namespace

bool count_logs(std::string_view body, std::uint64_t* records) noexcept {
    assert(records != nullptr);
    *records = 0;
    Cursor req{reinterpret_cast<const std::uint8_t*>(body.data()),
               reinterpret_cast<const std::uint8_t*>(body.data()) + body.size()};
    return each_len(req, 1, [&](Cursor rl) {
        return each_len(rl, 2, [&](Cursor sl) {
            return each_len(sl, 2, [&](Cursor) { ++*records; return true; });
        });
    });
}

Status decode_logs(std::string_view body, std::int64_t now_ns, bolt::Arena* a, LogRows* rows) noexcept {
    assert(a != nullptr && rows != nullptr);
    std::uint64_t count = 0;
    if (!count_logs(body, &count) || count > UINT32_MAX) return Status::kMalformed;
    const auto n = static_cast<std::uint32_t>(count);
    const std::uint32_t cap = n ? n : 1u;
    rows->n = n;
    rows->ts = a->allocate_array<std::int64_t>(cap);
    rows->observed_ts = a->allocate_array<std::int64_t>(cap);
    rows->severity = a->allocate_array<std::uint8_t>(cap);
    std::string_view** texts[] = {&rows->body, &rows->stream, &rows->attrs, &rows->severity_text,
                                  &rows->trace_id, &rows->span_id, &rows->event_name};
    for (std::string_view** t : texts) {
        *t = a->allocate_array<std::string_view>(cap);
        if (*t == nullptr) return Status::kArena;
    }
    if (rows->ts == nullptr || rows->observed_ts == nullptr || rows->severity == nullptr) return Status::kArena;
    Cursor req{reinterpret_cast<const std::uint8_t*>(body.data()),
               reinterpret_cast<const std::uint8_t*>(body.data()) + body.size()};
    std::uint32_t i = 0;
    const bool ok = each_len(req, 1, [&](Cursor rl) {
        Cursor resource; std::string_view stream;
        if (!first_len(rl, 1, &resource)) return false;
        if (resource.p != nullptr && !resource_stream(resource, a, &stream)) return false;
        return each_len(rl, 2, [&](Cursor sl) {
            Cursor sc; Scope scope;
            if (!first_len(sl, 1, &sc)) return false;
            if (sc.p != nullptr && !read_scope(sc, &scope)) return false;
            return each_len(sl, 2, [&](Cursor rec) {
                return i < n && decode_record(rec, scope, stream, now_ns, a, rows, i++);
            });
        });
    });
    if (!ok || i != n) return Status::kMalformed;
    return Status::kOk;
}

}  // namespace bolt::ingest::otlp
