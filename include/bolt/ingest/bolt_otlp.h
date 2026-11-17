// bolt/ingest/bolt_otlp.h — OTLP (OpenTelemetry protocol) protobuf decode.
//
// Hand-decoded over bolt_pb_cursor.h: no protobuf runtime, no generated code.
// One decoder for every consumer (gestaltd's /v1/* doors, chukonu's otlp://
// source). Two outputs from the same walk:
//
//   rows     arrays of views into the request body and the arena (zero copy)
//   batch    a BoltBatch with the fixed schema below (text packed per column)
//
// A malformed request is refused whole; nothing is half decoded.
//
// otlp_logs (kLogCols columns, in this order):
//   ts Int64 (epoch ns; time_unix_nano, else observed, else now), observed_ts Int64,
//   severity Int64 (OTel 0-24), severity_text Utf8, body Utf8 (a string body as
//   its bytes, any other AnyValue as JSON), attrs Utf8 (JSON object of the
//   record's attributes plus otel.scope.name / .version; empty = none),
//   stream Utf8 (the resource's attributes as canonical bytes), trace_id Utf8
//   (32 lowercase hex or empty), span_id Utf8 (16 hex or empty), event_name
//   Utf8, log_key Int64 (see log_key()).
//
// otlp_spans (kSpanCols columns):
//   start_ts Int64, duration_ns Int64, trace_id Utf8 (32 hex), span_id Utf8 (16 hex),
//   parent_span_id Utf8 (16 hex or empty), name Utf8, kind Int64 (0-5),
//   status_code Int64 (0-2), status_msg Utf8, trace_state Utf8, attrs Utf8,
//   events Utf8 (JSON array [{ts,name,attrs}]), links Utf8 (JSON array
//   [{trace_id,span_id,attrs}]), stream Utf8, flags Int64.
//
// Ids are hex text in batches (OTLP/JSON's own spelling): every engine operator
// reads Utf8 as StringView, and raw id bytes are not UTF-8. Rows keep the raw bytes.
//
// Canonical stream bytes: for each resource attribute, sorted by name,
// `name 0x1F value 0x1E`; a repeated name keeps the first.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_otlp_limits.h"
#include "bolt/ingest/bolt_pb_cursor.h"

namespace bolt::ingest::otlp {

inline constexpr std::uint32_t kLogCols = 11;
inline constexpr std::uint32_t kSpanCols = 15;

enum class Status : std::uint8_t { kOk = 0, kMalformed = 1, kArena = 2 };

// ---- shared pieces ---------------------------------------------------------

// A bounded text buffer in arena memory. A write past `cap` latches ok = false.
struct Out {
    char*       p = nullptr;
    std::size_t n = 0, cap = 0;
    bool        ok = true;
    std::string_view view() const noexcept { return std::string_view(p, n); }
};

// Room for anything `in_bytes` of OTLP can render to (a byte escapes to at most six).
bool out_init(Out* o, bolt::Arena* a, std::size_t in_bytes) noexcept;
void out_put(Out* o, std::string_view s) noexcept;
void out_json_string(Out* o, std::string_view s) noexcept;   // quoted and escaped
void out_hex_string(Out* o, std::string_view bytes) noexcept;   // quoted lowercase hex

// AnyValue as a JSON value. False on malformed input or nesting past kValueDepth.
bool any_json(pb::Cursor v, Out* o, std::uint32_t depth = 0) noexcept;
// AnyValue as text: a string value as its bytes, anything else as JSON.
bool any_text(pb::Cursor v, Out* o) noexcept;
// One KeyValue: key and the value's cursor (empty when the value is absent).
bool key_value(pb::Cursor kv, std::string_view* key, pb::Cursor* value) noexcept;
// Appends `"key":value` members (comma separated, no braces) for every KeyValue
// in field `field` of `msg`. `*members` counts what was written.
bool kv_members(pb::Cursor msg, std::uint32_t field, Out* o, std::uint32_t* members) noexcept;
// A Resource message's attributes as canonical stream bytes. More than kIdentityAttrs is refused.
bool resource_stream(pb::Cursor resource, bolt::Arena* a, std::string_view* out) noexcept;

// Calls fn(sub) for every length-delimited `field` of `msg`; other fields are skipped.
template <class Fn>
inline bool each_len(pb::Cursor msg, std::uint32_t field, Fn&& fn) noexcept {
    while (!msg.empty()) {   // bounded: the message's bytes
        std::uint32_t f = 0; std::uint8_t w = 0;
        if (!pb::read_tag(&msg, &f, &w)) return false;
        if (f != field || w != pb::kWireLen) { if (!pb::skip_field(&msg, w)) return false; continue; }
        pb::Cursor sub; if (!pb::read_len_delimited(&msg, &sub)) return false;
        if (!fn(sub)) return false;
    }
    return true;
}

// The first length-delimited `field` of `msg` (out->p == nullptr when absent).
inline bool first_len(pb::Cursor msg, std::uint32_t field, pb::Cursor* out) noexcept {
    out->p = nullptr; out->end = nullptr;
    bool seen = false;
    return each_len(msg, field, [&](pb::Cursor sub) { if (!seen) { *out = sub; seen = true; } return true; });
}

inline bool all_zero(std::string_view id) noexcept {
    for (const char c : id) if (c != 0) return false;   // bounded: an id's bytes
    return true;
}

inline pb::Cursor cursor_of(std::string_view body) noexcept {
    const auto* p = reinterpret_cast<const std::uint8_t*>(body.data());
    return pb::Cursor{p, p + body.size()};
}

// InstrumentationScope { name = 1; version = 2 }.
struct Scope { std::string_view name, version; };
bool read_scope(pb::Cursor scope, Scope* s) noexcept;
// Closes a JSON object opened with "{" after `members` attribute members, adding the scope's name and version.
void close_attrs(Out* o, const Scope& scope, std::uint32_t* members) noexcept;

// ---- request identity ------------------------------------------------------

// Identity of a request's bytes (wyhash3 v1, the persisted bolt hash).
std::uint64_t request_key(std::string_view body) noexcept;
// log_key of record `ordinal` of a request: splitmix64(request_key, ordinal) kept
// to 63 bits. Persisted in otel_logs' primary key: never change the mix.
std::int64_t log_key(std::uint64_t request_key, std::uint32_t ordinal) noexcept;

// ---- logs ------------------------------------------------------------------

// A request's records. Text points into the arena and into the request body,
// which must outlive the rows.
struct LogRows {
    std::uint32_t     n = 0;
    std::int64_t*     ts = nullptr;
    std::int64_t*     observed_ts = nullptr;
    std::uint8_t*     severity = nullptr;
    std::string_view* body = nullptr;
    std::string_view* stream = nullptr;
    std::string_view* attrs = nullptr;
    std::string_view* severity_text = nullptr;
    std::string_view* trace_id = nullptr;
    std::string_view* span_id = nullptr;
    std::string_view* event_name = nullptr;
};

// Log records in an ExportLogsServiceRequest; false when it is not well formed.
bool count_logs(std::string_view body, std::uint64_t* records) noexcept;
Status decode_logs(std::string_view body, std::int64_t now_ns, bolt::Arena* a, LogRows* rows) noexcept;

// ---- spans -----------------------------------------------------------------

struct SpanRows {
    std::uint32_t     n = 0;
    std::int64_t*     start_ts = nullptr;
    std::int64_t*     duration_ns = nullptr;
    std::int64_t*     flags = nullptr;
    std::uint8_t*     kind = nullptr;
    std::uint8_t*     status_code = nullptr;
    std::string_view* trace_id = nullptr;
    std::string_view* span_id = nullptr;
    std::string_view* parent_span_id = nullptr;
    std::string_view* name = nullptr;
    std::string_view* trace_state = nullptr;
    std::string_view* status_msg = nullptr;
    std::string_view* attrs = nullptr;
    std::string_view* events = nullptr;
    std::string_view* links = nullptr;
    std::string_view* stream = nullptr;
};

// Spans in an ExportTraceServiceRequest; false when it is not well formed.
bool count_spans(std::string_view body, std::uint64_t* spans) noexcept;
// A span without a 16-byte trace id or a non-zero 8-byte span id refuses the request.
Status decode_spans(std::string_view body, bolt::Arena* a, SpanRows* rows) noexcept;

// ---- batches ---------------------------------------------------------------

// Pack `n_parts` decoded requests into one batch of the fixed schema. `out` is
// init_empty'd with out->arena set; columns and text live in that arena, so the
// request bodies may be released afterwards. request_keys[p] keys part p's
// log_key column. False on arena exhaustion or text past a column's 2 GiB.
bool logs_to_batch(const LogRows* parts, const std::uint64_t* request_keys, std::uint32_t n_parts,
                   bolt::BoltBatch* out) noexcept;
bool spans_to_batch(const SpanRows* parts, std::uint32_t n_parts, bolt::BoltBatch* out) noexcept;

// The fixed schemas, in column order (kLogCols / kSpanCols entries).
struct Field { const char* name; bolt::BoltType type; };
const Field* log_fields() noexcept;
const Field* span_fields() noexcept;

// Text cell `row` of a Utf8 column (a view into the column's bytes).
std::string_view batch_text(const bolt::BoltColumn& col, std::int64_t row) noexcept;

}  // namespace bolt::ingest::otlp
