// OTLP rows packed into fixed-schema batches, and request identity. See bolt_otlp.h.

#include "bolt/ingest/bolt_otlp.h"
#include "bolt/ingest/bolt_otlp_metrics.h"

#include <cassert>
#include <cstring>

#include "bolt/bolt_hash.h"

namespace bolt::ingest::otlp {

namespace {

constexpr std::uint64_t kKeyMask = 0x7FFFFFFFFFFFFFFFull;

// Fills one Utf8 column from text cells get(part, row); `hex` spells bytes as lowercase hex.
template <class Get>
bool pack_text(const std::uint32_t* counts, std::uint32_t n_parts, std::int64_t total, bool hex, Get&& get,
               bolt::Arena* a, bolt::BoltColumn* col) noexcept {
    assert(a != nullptr && col != nullptr && total >= 0);
    std::uint64_t bytes = 0;
    for (std::uint32_t p = 0; p < n_parts; ++p)       // bounded: the parts' rows
        for (std::uint32_t i = 0; i < counts[p]; ++i) bytes += get(p, i).size();
    if (hex) bytes *= 2u;
    if (bytes > static_cast<std::uint64_t>(INT32_MAX)) return false;
    char* data = a->allocate_array<char>(bytes ? bytes : 1u);
    auto* offs = a->allocate_array<std::int32_t>(static_cast<std::uint64_t>(total) + 1u);
    if (data == nullptr || offs == nullptr) return false;
    static const char kHex[] = "0123456789abcdef";
    std::int64_t r = 0;
    std::int32_t at = 0;
    offs[0] = 0;
    for (std::uint32_t p = 0; p < n_parts; ++p) {
        for (std::uint32_t i = 0; i < counts[p]; ++i) {
            const std::string_view s = get(p, i);
            if (!hex) {
                if (!s.empty()) std::memcpy(data + at, s.data(), s.size());
                at += static_cast<std::int32_t>(s.size());
            } else {
                for (const char ch : s) {   // bounded: an id's bytes
                    const auto c = static_cast<unsigned char>(ch);
                    data[at++] = kHex[c >> 4]; data[at++] = kHex[c & 15];
                }
            }
            offs[++r] = at;
        }
    }
    assert(r == total && static_cast<std::uint64_t>(at) == bytes);
    *col = bolt::BoltColumn::make_utf8_from_packed(data, nullptr, offs, total, a);
    return total == 0 || col->data != nullptr;
}

template <class Get>
bool pack_i64(const std::uint32_t* counts, std::uint32_t n_parts, std::int64_t total, Get&& get, bolt::Arena* a,
              bolt::BoltColumn* col) noexcept {
    assert(a != nullptr && col != nullptr && total >= 0);
    auto* v = a->allocate_array<std::int64_t>(total ? static_cast<std::uint64_t>(total) : 1u);
    if (v == nullptr) return false;
    std::int64_t r = 0;
    for (std::uint32_t p = 0; p < n_parts; ++p)       // bounded: the parts' rows
        for (std::uint32_t i = 0; i < counts[p]; ++i) v[r++] = get(p, i);
    assert(r == total);
    *col = bolt::BoltColumn::make_flat(v, nullptr, total, bolt::BoltType::Int64);
    return true;
}

template <class Get>
bool pack_f64(const std::uint32_t* counts, std::uint32_t n_parts, std::int64_t total, Get&& get, bolt::Arena* a,
              bolt::BoltColumn* col) noexcept {
    assert(a != nullptr && col != nullptr && total >= 0);
    auto* v = a->allocate_array<double>(total ? static_cast<std::uint64_t>(total) : 1u);
    if (v == nullptr) return false;
    std::int64_t r = 0;
    for (std::uint32_t p = 0; p < n_parts; ++p)       // bounded: the parts' rows
        for (std::uint32_t i = 0; i < counts[p]; ++i) v[r++] = get(p, i);
    assert(r == total);
    *col = bolt::BoltColumn::make_flat(v, nullptr, total, bolt::BoltType::Float64);
    return true;
}

bool begin_batch(bolt::BoltBatch* out, std::uint32_t cols, std::int64_t rows) noexcept {
    assert(out != nullptr && out->arena != nullptr);
    if (!bolt::BoltBatch::alloc_columns(out, out->arena, cols)) return false;
    out->num_rows = rows;
    out->schema.num_fields = 0;
    return true;
}

}  // namespace

const Field* log_fields() noexcept {
    using T = bolt::BoltType;
    static const Field kFields[kLogCols] = {
        {"ts", T::Int64}, {"observed_ts", T::Int64}, {"severity", T::Int64}, {"severity_text", T::Utf8},
        {"body", T::Utf8}, {"attrs", T::Utf8}, {"stream", T::Utf8}, {"trace_id", T::Utf8}, {"span_id", T::Utf8},
        {"event_name", T::Utf8}, {"log_key", T::Int64}};
    return kFields;
}

const Field* span_fields() noexcept {
    using T = bolt::BoltType;
    static const Field kFields[kSpanCols] = {
        {"start_ts", T::Int64}, {"duration_ns", T::Int64}, {"trace_id", T::Utf8}, {"span_id", T::Utf8},
        {"parent_span_id", T::Utf8}, {"name", T::Utf8}, {"kind", T::Int64}, {"status_code", T::Int64},
        {"status_msg", T::Utf8}, {"trace_state", T::Utf8}, {"attrs", T::Utf8}, {"events", T::Utf8},
        {"links", T::Utf8}, {"stream", T::Utf8}, {"flags", T::Int64}};
    return kFields;
}

const Field* metric_fields() noexcept {
    using T = bolt::BoltType;
    static const Field kFields[kMetricCols] = {
        {"labels", T::Utf8}, {"ts_ms", T::Int64}, {"start_ts_ms", T::Int64}, {"value", T::Float64},
        {"hist", T::Utf8}, {"request_key", T::Int64}};
    return kFields;
}

std::uint64_t request_key(std::string_view body) noexcept {
    const std::uint64_t h = bolt::hash_bytes_wyhash3_v1(body.empty() ? "" : body.data(), body.size());
    return h != 0 ? h : 1u;
}

std::int64_t log_key(std::uint64_t request_key, std::uint32_t ordinal) noexcept {
    assert(request_key != 0);
    std::uint64_t z = request_key + (static_cast<std::uint64_t>(ordinal) + 1) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return static_cast<std::int64_t>(z & kKeyMask);
}

std::string_view batch_text(const bolt::BoltColumn& col, std::int64_t row) noexcept {
    assert(col.type == bolt::BoltType::Utf8 && row >= 0 && row < col.length);
    const auto* sv = static_cast<const bolt::StringView*>(col.data) + row;
    if (sv->length <= 12u) return std::string_view(sv->prefix, sv->length);
    return std::string_view(static_cast<const char*>(col.str_overflow_base) + sv->ref.offset, sv->length);
}

bool logs_to_batch(const LogRows* parts, const std::uint64_t* request_keys, std::uint32_t n_parts,
                   bolt::BoltBatch* out) noexcept {
    assert(out != nullptr && out->arena != nullptr && (n_parts == 0 || (parts != nullptr && request_keys != nullptr)));
    bolt::Arena* a = out->arena;
    auto* counts = a->allocate_array<std::uint32_t>(n_parts ? n_parts : 1u);
    if (counts == nullptr) return false;
    std::int64_t total = 0;
    for (std::uint32_t p = 0; p < n_parts; ++p) { counts[p] = parts[p].n; total += parts[p].n; }
    if (!begin_batch(out, kLogCols, total)) return false;
    bolt::BoltColumn* c = out->columns[out->read_epoch];
    bool ok = pack_i64(counts, n_parts, total, [&](std::uint32_t p, std::uint32_t i) { return parts[p].ts[i]; }, a, &c[0]);
    ok = ok && pack_i64(counts, n_parts, total, [&](std::uint32_t p, std::uint32_t i) { return parts[p].observed_ts[i]; }, a, &c[1]);
    ok = ok && pack_i64(counts, n_parts, total, [&](std::uint32_t p, std::uint32_t i) { return std::int64_t{parts[p].severity[i]}; }, a, &c[2]);
    ok = ok && pack_text(counts, n_parts, total, false, [&](std::uint32_t p, std::uint32_t i) { return parts[p].severity_text[i]; }, a, &c[3]);
    ok = ok && pack_text(counts, n_parts, total, false, [&](std::uint32_t p, std::uint32_t i) { return parts[p].body[i]; }, a, &c[4]);
    ok = ok && pack_text(counts, n_parts, total, false, [&](std::uint32_t p, std::uint32_t i) { return parts[p].attrs[i]; }, a, &c[5]);
    ok = ok && pack_text(counts, n_parts, total, false, [&](std::uint32_t p, std::uint32_t i) { return parts[p].stream[i]; }, a, &c[6]);
    ok = ok && pack_text(counts, n_parts, total, true, [&](std::uint32_t p, std::uint32_t i) { return parts[p].trace_id[i]; }, a, &c[7]);
    ok = ok && pack_text(counts, n_parts, total, true, [&](std::uint32_t p, std::uint32_t i) { return parts[p].span_id[i]; }, a, &c[8]);
    ok = ok && pack_text(counts, n_parts, total, false, [&](std::uint32_t p, std::uint32_t i) { return parts[p].event_name[i]; }, a, &c[9]);
    ok = ok && pack_i64(counts, n_parts, total, [&](std::uint32_t p, std::uint32_t i) { return log_key(request_keys[p], i); }, a, &c[10]);
    if (!ok) return false;
    const Field* kFields = log_fields();
    for (std::uint32_t i = 0; i < kLogCols; ++i) out->schema.add_field(kFields[i].name, kFields[i].type, false);
    return out->schema.num_fields == kLogCols;
}

bool spans_to_batch(const SpanRows* parts, std::uint32_t n_parts, bolt::BoltBatch* out) noexcept {
    assert(out != nullptr && out->arena != nullptr && (n_parts == 0 || parts != nullptr));
    bolt::Arena* a = out->arena;
    auto* counts = a->allocate_array<std::uint32_t>(n_parts ? n_parts : 1u);
    if (counts == nullptr) return false;
    std::int64_t total = 0;
    for (std::uint32_t p = 0; p < n_parts; ++p) { counts[p] = parts[p].n; total += parts[p].n; }
    if (!begin_batch(out, kSpanCols, total)) return false;
    bolt::BoltColumn* c = out->columns[out->read_epoch];
#define BOLT_OTLP_I64(col, expr) \
    ok = ok && pack_i64(counts, n_parts, total, [&](std::uint32_t p, std::uint32_t i) { return std::int64_t{expr}; }, a, &c[col])
#define BOLT_OTLP_TEXT(col, hex, field) \
    ok = ok && pack_text(counts, n_parts, total, hex, [&](std::uint32_t p, std::uint32_t i) { return parts[p].field[i]; }, a, &c[col])
    bool ok = true;
    BOLT_OTLP_I64(0, parts[p].start_ts[i]);
    BOLT_OTLP_I64(1, parts[p].duration_ns[i]);
    BOLT_OTLP_TEXT(2, true, trace_id);
    BOLT_OTLP_TEXT(3, true, span_id);
    BOLT_OTLP_TEXT(4, true, parent_span_id);
    BOLT_OTLP_TEXT(5, false, name);
    BOLT_OTLP_I64(6, parts[p].kind[i]);
    BOLT_OTLP_I64(7, parts[p].status_code[i]);
    BOLT_OTLP_TEXT(8, false, status_msg);
    BOLT_OTLP_TEXT(9, false, trace_state);
    BOLT_OTLP_TEXT(10, false, attrs);
    BOLT_OTLP_TEXT(11, false, events);
    BOLT_OTLP_TEXT(12, false, links);
    BOLT_OTLP_TEXT(13, false, stream);
    BOLT_OTLP_I64(14, parts[p].flags[i]);
#undef BOLT_OTLP_I64
#undef BOLT_OTLP_TEXT
    if (!ok) return false;
    const Field* kFields = span_fields();
    for (std::uint32_t i = 0; i < kSpanCols; ++i) out->schema.add_field(kFields[i].name, kFields[i].type, false);
    return out->schema.num_fields == kSpanCols;
}

bool metrics_to_batch(const MetricRows* parts, const std::uint64_t* request_keys, std::uint32_t n_parts,
                      bolt::BoltBatch* out) noexcept {
    assert(out != nullptr && out->arena != nullptr && (n_parts == 0 || (parts != nullptr && request_keys != nullptr)));
    bolt::Arena* a = out->arena;
    auto* counts = a->allocate_array<std::uint32_t>(n_parts ? n_parts : 1u);
    if (counts == nullptr) return false;
    std::int64_t total = 0;
    for (std::uint32_t p = 0; p < n_parts; ++p) { counts[p] = parts[p].n; total += parts[p].n; }
    if (!begin_batch(out, kMetricCols, total)) return false;
    bolt::BoltColumn* c = out->columns[out->read_epoch];
    bool ok = pack_text(counts, n_parts, total, false, [&](std::uint32_t p, std::uint32_t i) { return parts[p].labels[i]; }, a, &c[0]);
    ok = ok && pack_i64(counts, n_parts, total, [&](std::uint32_t p, std::uint32_t i) { return parts[p].ts_ms[i]; }, a, &c[1]);
    ok = ok && pack_i64(counts, n_parts, total, [&](std::uint32_t p, std::uint32_t i) { return parts[p].start_ts_ms[i]; }, a, &c[2]);
    ok = ok && pack_f64(counts, n_parts, total, [&](std::uint32_t p, std::uint32_t i) { return parts[p].value[i]; }, a, &c[3]);
    ok = ok && pack_text(counts, n_parts, total, true, [&](std::uint32_t p, std::uint32_t i) { return parts[p].hist[i]; }, a, &c[4]);
    ok = ok && pack_i64(counts, n_parts, total, [&](std::uint32_t p, std::uint32_t) {
        return static_cast<std::int64_t>(request_keys[p] & kKeyMask); }, a, &c[5]);
    if (!ok) return false;
    const Field* kFields = metric_fields();
    for (std::uint32_t i = 0; i < kMetricCols; ++i) out->schema.add_field(kFields[i].name, kFields[i].type, false);
    return out->schema.num_fields == kMetricCols;
}

}  // namespace bolt::ingest::otlp
