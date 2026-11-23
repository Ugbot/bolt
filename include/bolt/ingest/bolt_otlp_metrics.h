// bolt/ingest/bolt_otlp_metrics.h — OTLP metrics decoded to Prometheus-shaped
// points (the translation Prometheus's own OTLP receiver applies: underscore
// escaping with unit and _total suffixes).
//
//   name     non [A-Za-z0-9:] runs become one "_"; the unit is appended
//            (ms -> milliseconds, By -> bytes, x/s -> x_per_second, "1" on a
//            gauge -> ratio); a monotonic sum ends in _total
//   labels   point attributes (invalid characters -> "_"), the promoted
//            resource attributes, job = service.namespace/service.name,
//            instance = service.instance.id
//   target_info  one point per resource carrying every resource attribute,
//                at the resource's newest point
//   histogram    <name>_bucket{le} (cumulative), _sum, _count: one point each
//   exponential histogram   ONE point with `hist` set (a native histogram);
//                a scale above kExpHistogramScale is merged down
//   summary      <name>{quantile}, _sum, _count
//   a point flagged "no recorded value" is a stale marker (value = the stale
//   NaN bit pattern 0x7ff0000000000002)
//
// otlp_metric_points (kMetricCols columns, in this order):
//   labels Utf8       the series: `name 0x1F value 0x1E` pairs sorted by name,
//                     `__name__` included (bolt/kernels/packed_labels.h reads it;
//                     the bytes gestalt_series stores). A repeated name keeps
//                     the first; an empty value is dropped.
//   ts_ms Int64       epoch ms
//   start_ts_ms Int64 epoch ms, 0 = unknown
//   value Float64     the sample; for a native histogram its count
//   hist Utf8         empty for a float sample; else lowercase hex of the
//                     native histogram bytes (HistHead, then n_pos + n_neg f64)
//   request_key Int64 the request the point arrived in (request_key() kept to
//                     63 bits): points of one request are adjacent, so the
//                     store can count duplicates per request
//
// Refused whole, by name: delta temporality (Prometheus refuses it too), a
// label name or value holding 0x1E / 0x1F. Exemplars are counted and dropped.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_otlp.h"

namespace bolt::ingest::otlp {

inline constexpr std::uint32_t kMetricCols = 6;
inline constexpr std::uint64_t kStaleNaNBits = 0x7ff0000000000002ull;

// The Prometheus name of an OTLP metric (kind: 0 gauge, 1 monotonic sum, 2 other)
// into out[0, cap). False when it does not fit or holds more than kNameTokens words.
bool metric_name(std::string_view name, std::string_view unit, int kind, char* out, std::size_t cap,
                 std::size_t* len) noexcept;
// A label name; needs cap >= name.size() + 4.
std::size_t label_name(std::string_view name, char* out, std::size_t cap) noexcept;

// Native histogram bytes: this head (little-endian, 56 bytes), then n_pos and n_neg doubles.
struct HistHead {
    std::int32_t  schema;       // -4..kExpHistogramScale
    std::int32_t  pos_offset;   // Prometheus bucket index of pos[0]
    std::int32_t  neg_offset;
    std::uint32_t n_pos;
    std::uint32_t n_neg;
    std::uint32_t _pad;
    double        zero_threshold;
    double        zero_count;
    double        count;
    double        sum;
};
static_assert(sizeof(HistHead) == 56, "HistHead is 56 B");
// Head and bucket counts of `raw`. False when malformed or a side is over `cap`.
bool hist_decode(std::string_view raw, HistHead* head, double* pos, double* neg, std::uint32_t cap) noexcept;

struct MetricRows {
    std::uint32_t     n = 0;
    std::string_view* labels = nullptr;
    std::int64_t*     ts_ms = nullptr;
    std::int64_t*     start_ts_ms = nullptr;
    double*           value = nullptr;
    std::string_view* hist = nullptr;        // raw bytes; empty for a float sample
    std::uint64_t     metrics = 0;
    std::uint64_t     points = 0;            // data points translated (target_info not counted)
    std::uint64_t     exemplars_dropped = 0;
    std::string_view  refused;               // the metric a kDelta refusal names (a view of the body)
};

// Text points into the arena and the request body. On any refusal rows->n is 0.
Status decode_metrics(std::string_view body, bolt::Arena* a, MetricRows* rows) noexcept;
// The sentence a refusal is answered with; out is NUL terminated. Returns its length.
std::size_t metrics_refusal(Status s, const MetricRows& rows, char* out, std::size_t cap) noexcept;

// As logs_to_batch. A stage may change values and drop rows; request_key keeps the grouping.
bool metrics_to_batch(const MetricRows* parts, const std::uint64_t* request_keys, std::uint32_t n_parts,
                      bolt::BoltBatch* out) noexcept;
const Field* metric_fields() noexcept;

// `hex` (lowercase or uppercase, even length) to out[0, hex.size() / 2). False on a bad digit.
bool hex_decode(std::string_view hex, char* out) noexcept;

}  // namespace bolt::ingest::otlp
