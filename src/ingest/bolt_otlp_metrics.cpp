// OTLP metrics decode. See bolt_otlp_metrics.h.
//
// opentelemetry.proto.collector.metrics.v1.ExportMetricsServiceRequest:
//   resource_metrics = 1 { resource = 1; scope_metrics = 2 { scope = 1; metrics = 2 } }
//   Metric: name = 1, unit = 3, gauge = 5, sum = 7, histogram = 9, exponential_histogram = 10, summary = 11
//   Sum / Histogram / ExponentialHistogram: data_points = 1, aggregation_temporality = 2; Sum.is_monotonic = 3
//   NumberDataPoint: start = 2, time = 3 (fixed64), as_double = 4, exemplars = 5, as_int = 6, attributes = 7, flags = 8
//   HistogramDataPoint: start = 2, time = 3, count = 4, sum = 5, bucket_counts = 6, explicit_bounds = 7,
//                       exemplars = 8, attributes = 9, flags = 10
//   ExponentialHistogramDataPoint: attributes = 1, start = 2, time = 3, count = 4, sum = 5, scale = 6 (sint32),
//                       zero_count = 7, positive = 8 { offset = 1 (sint32); bucket_counts = 2 }, negative = 9,
//                       flags = 10, exemplars = 11, zero_threshold = 14
//   SummaryDataPoint: start = 2, time = 3, count = 4, sum = 5, quantile_values = 6 { quantile = 1; value = 2 },
//                     attributes = 7, flags = 8

#include "bolt/ingest/bolt_otlp_metrics.h"

#include <cassert>
#include <cstdio>
#include <cstring>

#include "bolt/kernels/bolt_prom_float.h"

namespace bolt::ingest::otlp {

using namespace bolt::ingest::pb;

namespace {

constexpr std::uint32_t kFlagNoRecordedValue = 1u;

// Resource attributes copied onto every series (the OpenTelemetry demo's Prometheus promotes these).
constexpr std::string_view kPromoted[] = {
    "service.instance.id", "service.name", "service.namespace", "service.version", "service.criticality",
    "cloud.availability_zone", "cloud.region", "deployment.environment.name", "k8s.cluster.name",
    "k8s.container.name", "k8s.cronjob.name", "k8s.daemonset.name", "k8s.deployment.name", "k8s.job.name",
    "k8s.namespace.name", "k8s.node.name", "k8s.pod.name", "k8s.replicaset.name", "k8s.statefulset.name",
    "container.name", "host.name", "postgresql.database.name", "postgresql.schema.name", "postgresql.table.name",
    "postgresql.index.name", "kafka.cluster.alias", "collector.instance.id",
};

struct Pair { std::string_view name, value; };

// An append-only array in arena memory (doubles; the old block stays in the arena).
template <class T>
struct Grow {
    bolt::Arena*  a = nullptr;
    T*            p = nullptr;
    std::uint32_t n = 0, cap = 0;
    bool push(const T& v) noexcept {
        assert(a != nullptr);
        if (n == cap) {
            const std::uint32_t nc = cap != 0 ? cap * 2u : 64u;
            T* np = a->allocate_array<T>(nc);
            if (np == nullptr) return false;
            for (std::uint32_t i = 0; i < n; ++i) np[i] = p[i];   // bounded: n
            p = np; cap = nc;
        }
        p[n++] = v;
        return true;
    }
};

struct Point { std::uint64_t start = 0, time = 0, flags = 0; };

struct Ctx {
    bolt::Arena*           a;
    MetricRows*            rows;
    Status                 fail = Status::kMalformed;   // what a false return means
    Grow<std::string_view> labels, hist;
    Grow<std::int64_t>     ts, st;
    Grow<double>           value;
    Grow<Pair>             res_series;   // job, instance and the promoted attributes
    Grow<Pair>             res_info;     // job, instance and every attribute
    std::int64_t           newest_ms = 0;
    Pair                   scratch[kSeriesLabels];
};

double f64_of(std::uint64_t bits) noexcept { double d; std::memcpy(&d, &bits, 8); return d; }
std::int64_t zigzag32(std::uint64_t v) noexcept { return static_cast<std::int64_t>((v >> 1) ^ (~(v & 1) + 1)); }
std::int64_t ms_of(std::uint64_t ns) noexcept { return static_cast<std::int64_t>(ns / 1000000u); }

bool arena_fail(Ctx* c) noexcept { c->fail = Status::kArena; return false; }

bool concat(Ctx* c, std::string_view x, std::string_view y, std::string_view* out) noexcept {
    char* p = c->a->allocate_array<char>(x.size() + y.size() + 1u);
    if (p == nullptr) return arena_fail(c);
    if (!x.empty()) std::memcpy(p, x.data(), x.size());
    if (!y.empty()) std::memcpy(p + x.size(), y.data(), y.size());
    *out = std::string_view(p, x.size() + y.size());
    return true;
}

// Every KeyValue in field `field` of `msg` as (label name, text). A string value is a view of the body.
bool read_attrs(Cursor msg, std::uint32_t field, Ctx* c, Grow<Pair>* out, Grow<Pair>* raw_keys = nullptr) noexcept {
    return each_len(msg, field, [&](Cursor kv) {
        std::string_view key; Cursor val{};
        if (!key_value(kv, &key, &val)) return false;
        std::string_view text;
        Cursor peek = val;
        std::uint32_t f = 0; std::uint8_t w = 0;
        if (val.p != nullptr && !peek.empty() && read_tag(&peek, &f, &w) && f == 1 && w == kWireLen) {
            Cursor s; if (!read_len_delimited(&peek, &s)) return false;
            text = view_of(s);
        } else if (val.p != nullptr && !val.empty()) {
            Out o;
            if (!out_init(&o, c->a, val.remaining())) return arena_fail(c);
            if (!any_json(val, &o, 0)) return false;
            text = o.view();
        }
        char* name = c->a->allocate_array<char>(key.size() + 4u);
        if (name == nullptr) return arena_fail(c);
        const std::size_t n = label_name(key, name, key.size() + 4u);
        if (raw_keys != nullptr && !raw_keys->push(Pair{key, text})) return arena_fail(c);
        if (!out->push(Pair{std::string_view(name, n), text})) return arena_fail(c);
        return true;
    });
}

std::string_view find_raw(const Grow<Pair>& attrs, std::string_view key) noexcept {
    for (std::uint32_t i = 0; i < attrs.n; ++i) if (attrs.p[i].name == key) return attrs.p[i].value;
    return std::string_view();
}

bool has_raw(const Grow<Pair>& attrs, std::string_view key) noexcept {
    for (std::uint32_t i = 0; i < attrs.n; ++i) if (attrs.p[i].name == key) return true;
    return false;
}

bool resource_labels(Cursor resource, Ctx* c) noexcept {
    c->res_series.n = 0; c->res_info.n = 0;
    Grow<Pair> raw, translated;
    raw.a = c->a; translated.a = c->a;
    if (resource.p != nullptr && !read_attrs(resource, 1, c, &translated, &raw)) return false;
    if (has_raw(raw, "service.name")) {
        const std::string_view name = find_raw(raw, "service.name"), ns = find_raw(raw, "service.namespace");
        std::string_view job = name;
        if (!ns.empty()) {
            std::string_view head;
            if (!concat(c, ns, "/", &head) || !concat(c, head, name, &job)) return false;
        }
        if (!c->res_series.push(Pair{"job", job}) || !c->res_info.push(Pair{"job", job})) return arena_fail(c);
    }
    if (has_raw(raw, "service.instance.id")) {
        const Pair inst{"instance", find_raw(raw, "service.instance.id")};
        if (!c->res_series.push(inst) || !c->res_info.push(inst)) return arena_fail(c);
    }
    for (std::uint32_t i = 0; i < raw.n; ++i) {   // bounded: the resource's attributes
        if (!c->res_info.push(translated.p[i])) return arena_fail(c);
        bool promoted = false;
        for (const std::string_view k : kPromoted) promoted = promoted || k == raw.p[i].name;
        if (promoted && !c->res_series.push(translated.p[i])) return arena_fail(c);
    }
    return true;
}

bool has_separator(std::string_view s) noexcept {
    return std::memchr(s.data(), 0x1E, s.size()) != nullptr || std::memchr(s.data(), 0x1F, s.size()) != nullptr;
}

// One row. Labels in push order: __name__, attrs, the extra pair, `tail`; sorted by name, a repeated
// name keeps the first, an empty name or value is dropped.
bool emit(Ctx* c, std::string_view name, const Grow<Pair>* attrs, std::string_view extra_name, std::string_view extra_value,
          const Grow<Pair>& tail, std::int64_t ts_ms, std::int64_t st_ms, double v, std::string_view hist) noexcept {
    Pair* s = c->scratch;
    std::uint32_t n = 0;
    const auto add = [&](const Pair& p) noexcept {
        if (p.name.empty() || p.value.empty()) return true;
        if (has_separator(p.name) || has_separator(p.value)) { c->fail = Status::kLabelBytes; return false; }
        if (n == kSeriesLabels) return false;
        std::uint32_t at = n++;   // stable insertion: after every name <= p.name
        while (at != 0 && s[at - 1].name > p.name) { s[at] = s[at - 1]; --at; }   // bounded: kSeriesLabels
        s[at] = p;
        return true;
    };
    if (!add(Pair{"__name__", name})) return false;
    for (std::uint32_t i = 0; attrs != nullptr && i < attrs->n; ++i) if (!add(attrs->p[i])) return false;
    if (!extra_name.empty() && !add(Pair{extra_name, extra_value})) return false;
    for (std::uint32_t i = 0; i < tail.n; ++i) if (!add(tail.p[i])) return false;
    std::size_t bytes = 0;
    for (std::uint32_t i = 0; i < n; ++i) bytes += s[i].name.size() + s[i].value.size() + 2u;
    char* p = c->a->allocate_array<char>(bytes + 1u);
    if (p == nullptr) return arena_fail(c);
    std::size_t at = 0;
    for (std::uint32_t i = 0; i < n; ++i) {   // bounded: kSeriesLabels
        if (i != 0 && s[i].name == s[i - 1].name) continue;
        std::memcpy(p + at, s[i].name.data(), s[i].name.size()); at += s[i].name.size();
        p[at++] = '\x1F';
        std::memcpy(p + at, s[i].value.data(), s[i].value.size()); at += s[i].value.size();
        p[at++] = '\x1E';
    }
    assert(at <= bytes);
    if (!c->labels.push(std::string_view(p, at)) || !c->ts.push(ts_ms) || !c->st.push(st_ms) || !c->value.push(v) ||
        !c->hist.push(hist)) return arena_fail(c);
    return true;
}

bool sample(Ctx* c, std::string_view name, const Grow<Pair>& attrs, std::string_view extra_name,
            std::string_view extra_value, double v, const Point& p) noexcept {
    const double out = (p.flags & kFlagNoRecordedValue) ? f64_of(kStaleNaNBits) : v;
    const std::int64_t ts = ms_of(p.time);
    if (!emit(c, name, &attrs, extra_name, extra_value, c->res_series, ts, ms_of(p.start), out, std::string_view())) return false;
    if (ts > c->newest_ms) c->newest_ms = ts;
    ++c->rows->points;
    return true;
}

// The fields every data point kind shares.
bool point_head(Cursor dp, std::uint32_t flags_field, std::uint32_t exemplar_field, Point* p, MetricRows* rows) noexcept {
    while (!dp.empty()) {   // bounded: the point's bytes
        std::uint32_t f = 0; std::uint8_t w = 0;
        if (!read_tag(&dp, &f, &w)) return false;
        if (f == 2 && w == kWireF64) { if (!read_fixed64(&dp, &p->start)) return false; }
        else if (f == 3 && w == kWireF64) { if (!read_fixed64(&dp, &p->time)) return false; }
        else if (f == flags_field && w == kWireVarint) { if (!read_varint(&dp, &p->flags)) return false; }
        else { if (f == exemplar_field && w == kWireLen) ++rows->exemplars_dropped; if (!skip_field(&dp, w)) return false; }
    }
    return true;
}

// Repeated fixed64 in `field` of `msg`, packed or not.
bool read_fixed64s(Cursor msg, std::uint32_t field, Ctx* c, Grow<std::uint64_t>* out) noexcept {
    while (!msg.empty()) {   // bounded: the message's bytes
        std::uint32_t f = 0; std::uint8_t w = 0;
        if (!read_tag(&msg, &f, &w)) return false;
        if (f == field && w == kWireF64) {
            std::uint64_t v = 0; if (!read_fixed64(&msg, &v)) return false;
            if (!out->push(v)) return arena_fail(c);
        } else if (f == field && w == kWireLen) {
            Cursor sub; if (!read_len_delimited(&msg, &sub)) return false;
            if (sub.remaining() % 8 != 0) return false;
            while (!sub.empty()) {   // bounded: the field's bytes
                std::uint64_t v = 0; if (!read_fixed64(&sub, &v)) return false;
                if (!out->push(v)) return arena_fail(c);
            }
        } else if (!skip_field(&msg, w)) return false;
    }
    return true;
}

bool number_points(Cursor data, std::string_view name, Ctx* c) noexcept {
    return each_len(data, 1, [&](Cursor dp) {
        Point p;
        if (!point_head(dp, 8, 5, &p, c->rows)) return false;
        Grow<Pair> attrs; attrs.a = c->a;
        if (!read_attrs(dp, 7, c, &attrs)) return false;
        double v = 0;
        Cursor walk = dp;
        while (!walk.empty()) {   // bounded: the point's bytes
            std::uint32_t f = 0; std::uint8_t w = 0;
            if (!read_tag(&walk, &f, &w)) return false;
            if ((f == 4 || f == 6) && w == kWireF64) {
                std::uint64_t bits = 0; if (!read_fixed64(&walk, &bits)) return false;
                v = f == 4 ? f64_of(bits) : static_cast<double>(static_cast<std::int64_t>(bits));
            } else if (!skip_field(&walk, w)) return false;
        }
        return sample(c, name, attrs, {}, {}, v, p);
    });
}

bool histogram_points(Cursor data, std::string_view name, Ctx* c) noexcept {
    std::string_view bucket, sum, count_name;
    if (!concat(c, name, "_bucket", &bucket) || !concat(c, name, "_sum", &sum) || !concat(c, name, "_count", &count_name)) return false;
    return each_len(data, 1, [&](Cursor dp) {
        Point p;
        if (!point_head(dp, 10, 8, &p, c->rows)) return false;
        Grow<Pair> attrs; attrs.a = c->a;
        Grow<std::uint64_t> counts, bounds; counts.a = c->a; bounds.a = c->a;
        if (!read_attrs(dp, 9, c, &attrs) || !read_fixed64s(dp, 6, c, &counts) || !read_fixed64s(dp, 7, c, &bounds)) return false;
        if (counts.n != 0 && counts.n != bounds.n + 1u) return false;
        std::uint64_t count = 0, sum_bits = 0; bool has_sum = false;
        Cursor walk = dp;
        while (!walk.empty()) {   // bounded: the point's bytes
            std::uint32_t f = 0; std::uint8_t w = 0;
            if (!read_tag(&walk, &f, &w)) return false;
            if (f == 4 && w == kWireF64) { if (!read_fixed64(&walk, &count)) return false; }
            else if (f == 5 && w == kWireF64) { if (!read_fixed64(&walk, &sum_bits)) return false; has_sum = true; }
            else if (!skip_field(&walk, w)) return false;
        }
        std::uint64_t running = 0;
        for (std::uint32_t i = 0; i < counts.n; ++i) {   // bounded: the point's buckets
            running += counts.p[i];
            std::string_view le = "+Inf";
            if (i < bounds.n) {
                char* text = c->a->allocate_array<char>(bolt::kernels::k_prom_float_chars);
                if (text == nullptr) return arena_fail(c);
                le = std::string_view(text, bolt::kernels::prom_float_chars(f64_of(bounds.p[i]), text));
            }
            if (!sample(c, bucket, attrs, "le", le, static_cast<double>(running), p)) return false;
        }
        if (counts.n == 0 && !sample(c, bucket, attrs, "le", "+Inf", static_cast<double>(count), p)) return false;
        if (has_sum && !sample(c, sum, attrs, {}, {}, f64_of(sum_bits), p)) return false;
        return sample(c, count_name, attrs, {}, {}, static_cast<double>(count), p);
    });
}

// Buckets { offset = 1 (sint32); bucket_counts = 2 (packed varint) } as (first Prometheus index, counts).
bool exp_buckets(Cursor bk, std::int32_t downscale, std::int32_t* offset, double* out, std::uint32_t* n_out) noexcept {
    std::int64_t off = 0;
    std::uint64_t counts[kHistBucketsPerSide];
    std::uint32_t n = 0;
    const auto push = [&](std::uint64_t v) noexcept { if (n == kHistBucketsPerSide) return false; counts[n++] = v; return true; };
    while (!bk.empty()) {   // bounded: the message's bytes
        std::uint32_t f = 0; std::uint8_t w = 0;
        if (!read_tag(&bk, &f, &w)) return false;
        std::uint64_t v = 0;
        if (f == 1 && w == kWireVarint) { if (!read_varint(&bk, &v)) return false; off = zigzag32(v); }
        else if (f == 2 && w == kWireLen) {
            Cursor sub; if (!read_len_delimited(&bk, &sub)) return false;
            while (!sub.empty()) { if (!read_varint(&sub, &v) || !push(v)) return false; }   // bounded: the field's bytes
        } else if (f == 2 && w == kWireVarint) { if (!read_varint(&bk, &v) || !push(v)) return false; }
        else if (!skip_field(&bk, w)) return false;
    }
    *n_out = 0; *offset = 0;
    if (n == 0) return true;
    // OTLP bucket i is (base^i, base^(i+1)]; Prometheus bucket k is (base^(k-1), base^k]: k = i + 1, then merged down.
    const auto prom = [&](std::int64_t i) noexcept { return (i >> downscale) + 1; };
    const std::int64_t first = prom(off), last = prom(off + static_cast<std::int64_t>(n) - 1);
    const auto width = static_cast<std::uint32_t>(last - first + 1);
    assert(width <= n);
    for (std::uint32_t i = 0; i < width; ++i) out[i] = 0.0;
    for (std::uint32_t i = 0; i < n; ++i)   // bounded: kHistBucketsPerSide
        out[static_cast<std::size_t>(prom(off + static_cast<std::int64_t>(i)) - first)] += static_cast<double>(counts[i]);
    *offset = static_cast<std::int32_t>(first);
    *n_out = width;
    return true;
}

bool exp_histogram_points(Cursor data, std::string_view name, Ctx* c) noexcept {
    return each_len(data, 1, [&](Cursor dp) {
        Point p;
        if (!point_head(dp, 10, 11, &p, c->rows)) return false;
        Grow<Pair> attrs; attrs.a = c->a;
        if (!read_attrs(dp, 1, c, &attrs)) return false;
        HistHead h{};
        std::int64_t scale = 0;
        Cursor pos{nullptr, nullptr}, neg{nullptr, nullptr};
        Cursor walk = dp;
        while (!walk.empty()) {   // bounded: the point's bytes
            std::uint32_t f = 0; std::uint8_t w = 0;
            if (!read_tag(&walk, &f, &w)) return false;
            std::uint64_t v = 0;
            if ((f == 4 || f == 5 || f == 7 || f == 14) && w == kWireF64) {
                if (!read_fixed64(&walk, &v)) return false;
                if (f == 4) h.count = static_cast<double>(v);
                else if (f == 5) h.sum = f64_of(v);
                else if (f == 7) h.zero_count = static_cast<double>(v);
                else h.zero_threshold = f64_of(v);
            } else if (f == 6 && w == kWireVarint) { if (!read_varint(&walk, &v)) return false; scale = zigzag32(v); }
            else if ((f == 8 || f == 9) && w == kWireLen) { if (!read_len_delimited(&walk, f == 8 ? &pos : &neg)) return false; }
            else if (!skip_field(&walk, w)) return false;
        }
        if (scale < -4) return false;
        const std::int32_t down = scale > kExpHistogramScale ? static_cast<std::int32_t>(scale - kExpHistogramScale) : 0;
        h.schema = static_cast<std::int32_t>(scale) - down;
        double pb[kHistBucketsPerSide], nb[kHistBucketsPerSide];
        if (pos.p != nullptr && !exp_buckets(pos, down, &h.pos_offset, pb, &h.n_pos)) return false;
        if (neg.p != nullptr && !exp_buckets(neg, down, &h.neg_offset, nb, &h.n_neg)) return false;
        const std::size_t bytes = sizeof(HistHead) + (static_cast<std::size_t>(h.n_pos) + h.n_neg) * 8u;
        char* raw = c->a->allocate_array<char>(bytes);
        if (raw == nullptr) return arena_fail(c);
        std::memcpy(raw, &h, sizeof(HistHead));
        if (h.n_pos != 0) std::memcpy(raw + sizeof(HistHead), pb, h.n_pos * 8u);
        if (h.n_neg != 0) std::memcpy(raw + sizeof(HistHead) + h.n_pos * 8u, nb, h.n_neg * 8u);
        const std::int64_t ts = ms_of(p.time);
        if (!emit(c, name, &attrs, {}, {}, c->res_series, ts, ms_of(p.start), h.count, std::string_view(raw, bytes))) return false;
        if (ts > c->newest_ms) c->newest_ms = ts;
        ++c->rows->points;
        return true;
    });
}

bool summary_points(Cursor data, std::string_view name, Ctx* c) noexcept {
    std::string_view sum, count_name;
    if (!concat(c, name, "_sum", &sum) || !concat(c, name, "_count", &count_name)) return false;
    return each_len(data, 1, [&](Cursor dp) {
        Point p;
        if (!point_head(dp, 8, 0, &p, c->rows)) return false;
        Grow<Pair> attrs; attrs.a = c->a;
        if (!read_attrs(dp, 7, c, &attrs)) return false;
        std::uint64_t count = 0, sum_bits = 0;
        Cursor walk = dp;
        while (!walk.empty()) {   // bounded: the point's bytes
            std::uint32_t f = 0; std::uint8_t w = 0;
            if (!read_tag(&walk, &f, &w)) return false;
            if (f == 4 && w == kWireF64) { if (!read_fixed64(&walk, &count)) return false; }
            else if (f == 5 && w == kWireF64) { if (!read_fixed64(&walk, &sum_bits)) return false; }
            else if (!skip_field(&walk, w)) return false;
        }
        const bool ok = each_len(dp, 6, [&](Cursor qv) {
            std::uint64_t q = 0, v = 0;
            while (!qv.empty()) {   // bounded: the message's bytes
                std::uint32_t f = 0; std::uint8_t w = 0;
                if (!read_tag(&qv, &f, &w)) return false;
                if ((f == 1 || f == 2) && w == kWireF64) { if (!read_fixed64(&qv, f == 1 ? &q : &v)) return false; }
                else if (!skip_field(&qv, w)) return false;
            }
            char* text = c->a->allocate_array<char>(bolt::kernels::k_prom_float_chars);
            if (text == nullptr) return arena_fail(c);
            const std::string_view quantile(text, bolt::kernels::prom_float_chars(f64_of(q), text));
            return sample(c, name, attrs, "quantile", quantile, f64_of(v), p);
        });
        if (!ok) return false;
        if (!sample(c, sum, attrs, {}, {}, f64_of(sum_bits), p)) return false;
        return sample(c, count_name, attrs, {}, {}, static_cast<double>(count), p);
    });
}

// aggregation_temporality (field 2) and is_monotonic (field 3) of a Sum / Histogram message.
bool data_flags(Cursor data, std::uint64_t* temporality, std::uint64_t* monotonic) noexcept {
    while (!data.empty()) {   // bounded: the message's bytes
        std::uint32_t f = 0; std::uint8_t w = 0;
        if (!read_tag(&data, &f, &w)) return false;
        if (f == 2 && w == kWireVarint) { if (!read_varint(&data, temporality)) return false; }
        else if (f == 3 && w == kWireVarint) { if (!read_varint(&data, monotonic)) return false; }
        else if (!skip_field(&data, w)) return false;
    }
    return true;
}

bool one_metric(Cursor metric, Ctx* c) noexcept {
    std::string_view name, unit;
    Cursor data{nullptr, nullptr};
    std::uint32_t kind_field = 0;
    while (!metric.empty()) {   // bounded: the metric's bytes
        std::uint32_t f = 0; std::uint8_t w = 0;
        if (!read_tag(&metric, &f, &w)) return false;
        if (w == kWireLen && (f == 1 || f == 3 || f == 5 || f == 7 || f == 9 || f == 10 || f == 11)) {
            Cursor v; if (!read_len_delimited(&metric, &v)) return false;
            if (f == 1) name = view_of(v);
            else if (f == 3) unit = view_of(v);
            else { data = v; kind_field = f; }
        } else if (!skip_field(&metric, w)) return false;
    }
    if (kind_field == 0) return true;   // a metric with no data
    std::uint64_t temporality = 0, monotonic = 0;
    if (kind_field != 5 && kind_field != 11 && !data_flags(data, &temporality, &monotonic)) return false;
    if (temporality == 1) { c->fail = Status::kDelta; c->rows->refused = name; return false; }
    ++c->rows->metrics;
    const int kind = kind_field == 5 ? 0 : (kind_field == 7 && monotonic ? 1 : 2);
    const std::size_t cap = name.size() + unit.size() + 64u;
    char* text = c->a->allocate_array<char>(cap);
    if (text == nullptr) return arena_fail(c);
    std::size_t len = 0;
    if (!metric_name(name, unit, kind, text, cap, &len)) return false;
    const std::string_view prom(text, len);
    if (kind_field == 5 || kind_field == 7) return number_points(data, prom, c);
    if (kind_field == 9) return histogram_points(data, prom, c);
    if (kind_field == 10) return exp_histogram_points(data, prom, c);
    return summary_points(data, prom, c);
}

}  // namespace

Status decode_metrics(std::string_view body, bolt::Arena* a, MetricRows* rows) noexcept {
    assert(a != nullptr && rows != nullptr);
    *rows = MetricRows{};
    Ctx c{};
    c.a = a; c.rows = rows; c.fail = Status::kMalformed;
    c.labels.a = a; c.hist.a = a; c.ts.a = a; c.st.a = a; c.value.a = a; c.res_series.a = a; c.res_info.a = a;
    const bool ok = each_len(cursor_of(body), 1, [&](Cursor res_metrics) {
        Cursor resource;
        if (!first_len(res_metrics, 1, &resource)) return false;
        if (!resource_labels(resource, &c)) return false;
        c.newest_ms = 0;
        const bool walked = each_len(res_metrics, 2, [&](Cursor scope_metrics) {
            return each_len(scope_metrics, 2, [&](Cursor metric) { return one_metric(metric, &c); });
        });
        if (!walked) return false;
        if (c.newest_ms != 0 && c.res_info.n != 0)   // target_info at the resource's newest point
            return emit(&c, "target_info", nullptr, {}, {}, c.res_info, c.newest_ms, 0, 1.0, std::string_view());
        return true;
    });
    if (!ok) {
        const std::string_view refused = rows->refused;
        *rows = MetricRows{};
        rows->refused = refused;
        assert(c.fail != Status::kOk);
        return c.fail;
    }
    rows->n = c.labels.n;
    rows->labels = c.labels.p; rows->hist = c.hist.p;
    rows->ts_ms = c.ts.p; rows->start_ts_ms = c.st.p; rows->value = c.value.p;
    assert(c.ts.n == rows->n && c.value.n == rows->n);
    return Status::kOk;
}

std::size_t metrics_refusal(Status s, const MetricRows& rows, char* out, std::size_t cap) noexcept {
    assert(out != nullptr && cap >= 160u && s != Status::kOk);
    int n = 0;
    if (s == Status::kDelta) {
        n = std::snprintf(out, cap, "metric '%.*s' has delta temporality, which is not supported; export cumulative",
                          static_cast<int>(rows.refused.size() < 120 ? rows.refused.size() : 120), rows.refused.data());
    } else if (s == Status::kLabelBytes) {
        n = std::snprintf(out, cap, "a label name or value contains \\x1e or \\x1f, which the series labels column cannot store");
    } else if (s == Status::kArena) {
        n = std::snprintf(out, cap, "out of memory decoding the OTLP metrics request");
    } else {
        n = std::snprintf(out, cap, "malformed OTLP metrics request");
    }
    return n > 0 ? (static_cast<std::size_t>(n) < cap ? static_cast<std::size_t>(n) : cap - 1u) : 0u;
}

}  // namespace bolt::ingest::otlp
