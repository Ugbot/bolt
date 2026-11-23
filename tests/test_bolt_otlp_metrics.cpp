// OTLP metrics decoded to Prometheus-shaped points: names, labels, target_info,
// every data point kind, the batch, and what is refused.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_otlp_metrics.h"
#include "otlp_test_pb.h"

namespace o = bolt::ingest::otlp;
using namespace otlp_test;

namespace {

constexpr std::uint64_t kT = 1700000000000000000ull;   // ns
constexpr std::int64_t kTms = 1700000000000ll;

std::uint64_t bits(double d) { std::uint64_t b; std::memcpy(&b, &d, 8); return b; }
std::string packed64(const std::vector<std::uint64_t>& v) {
    std::string s; for (std::uint64_t x : v) s.append(reinterpret_cast<const char*>(&x), 8); return s;
}
std::uint64_t zz(std::int64_t v) { return (static_cast<std::uint64_t>(v) << 1) ^ static_cast<std::uint64_t>(v >> 63); }

// NumberDataPoint with a double (as_int when `as_int`).
std::string number_point(double v, const std::vector<std::string>& attrs, bool as_int = false, std::uint64_t flags = 0) {
    Pb p; p.fixed64(2, kT - 60000000000ull); p.fixed64(3, kT);
    if (as_int) p.fixed64(6, static_cast<std::uint64_t>(static_cast<std::int64_t>(v))); else p.fixed64(4, bits(v));
    for (const auto& a : attrs) p.len(7, a);
    if (flags) p.u64(8, flags);
    return p.b;
}

std::string metric(const std::string& name, const std::string& unit, std::uint32_t kind_field, const std::string& data) {
    Pb m; m.len(1, name); if (!unit.empty()) m.len(3, unit); m.len(kind_field, data); return m.b;
}

std::string gauge(const std::vector<std::string>& points) { Pb g; for (const auto& p : points) g.len(1, p); return g.b; }
std::string sum(const std::vector<std::string>& points, std::uint64_t temporality, bool monotonic) {
    Pb s; for (const auto& p : points) s.len(1, p); s.u64(2, temporality); if (monotonic) s.u64(3, 1); return s.b;
}

std::string request(const std::vector<std::string>& resource_attrs, const std::vector<std::string>& metrics) {
    Pb res; for (const auto& a : resource_attrs) res.len(1, a);
    Pb sm; for (const auto& m : metrics) sm.len(2, m);
    Pb rm_; rm_.len(1, res.b); rm_.len(2, sm.b);
    Pb req; req.len(1, rm_.b);
    return req.b;
}

const std::vector<std::string> kResource = {kv("service.name", any_str("checkout")), kv("service.namespace", any_str("otel-demo")),
                                            kv("service.instance.id", any_str("i-1")), kv("process.pid", any_int(42))};

using Labels = std::map<std::string, std::string>;
Labels labels_of(std::string_view packed) {
    Labels l;
    std::size_t i = 0;
    while (i < packed.size()) {
        const std::size_t sep = packed.find('\x1F', i), end = packed.find('\x1E', i);
        l[std::string(packed.substr(i, sep - i))] = std::string(packed.substr(sep + 1, end - sep - 1));
        i = end + 1;
    }
    return l;
}

// The first row named `name` whose labels include `extra`; -1 when none.
int find(const o::MetricRows& rows, const std::string& name, const Labels& extra = {}) {
    for (std::uint32_t r = 0; r < rows.n; ++r) {
        const Labels l = labels_of(rows.labels[r]);
        if (!l.count("__name__") || l.at("__name__") != name) continue;
        bool ok = true;
        for (const auto& [k, v] : extra) ok = ok && l.count(k) && l.at(k) == v;
        if (ok) return static_cast<int>(r);
    }
    return -1;
}

std::string mname(const char* name, const char* unit, int kind) {
    char buf[256]; std::size_t n = 0;
    EXPECT_TRUE(o::metric_name(name, unit, kind, buf, sizeof(buf), &n));
    return std::string(buf, n);
}
std::string lname(const char* name) { char buf[64]; return std::string(buf, o::label_name(name, buf, sizeof(buf))); }

TEST(OtlpMetrics, NamesFollowPrometheusOtlpTranslation) {
    EXPECT_EQ(mname("http.server.duration", "ms", 2), "http_server_duration_milliseconds");
    EXPECT_EQ(mname("app.requests", "{request}", 1), "app_requests_total");
    EXPECT_EQ(mname("app.requests.total", "", 1), "app_requests_total");
    EXPECT_EQ(mname("system.cpu.utilization", "1", 0), "system_cpu_utilization_ratio");
    EXPECT_EQ(mname("system.network.io", "By/s", 2), "system_network_io_bytes_per_second");
    EXPECT_EQ(mname("process.runtime.memory_bytes", "By", 0), "process_runtime_memory_bytes");
    EXPECT_EQ(mname("3d.frames", "", 0), "_3d_frames");
    EXPECT_EQ(lname("http.status_code"), "http_status_code");
    EXPECT_EQ(lname("1st"), "key_1st");
    EXPECT_EQ(lname("_x"), "key_x");
    EXPECT_EQ(lname(".x"), "key_x");
    EXPECT_EQ(lname("__x"), "__x");
    char tiny[4]; std::size_t n = 0;
    EXPECT_FALSE(o::metric_name("a.long.metric.name", "", 2, tiny, sizeof(tiny), &n));
}

TEST(OtlpMetrics, GaugesSumsLabelsAndTargetInfo) {
    const std::string req = request(kResource, {
        metric("queue.depth", "", 5, gauge({number_point(7.5, {kv("queue.name", any_str("orders"))})})),
        metric("app.requests", "{request}", 7, sum({number_point(12, {kv("http.status_code", any_int(200))}, true)}, 2, true)),
        metric("app.inflight", "", 7, sum({number_point(3, {})}, 2, false)),
        metric("gone", "", 5, gauge({number_point(0, {}, false, 1)})),
    });
    bolt::Arena a; o::MetricRows rows;
    ASSERT_TRUE(o::decode_metrics(req, &a, &rows) == o::Status::kOk);
    EXPECT_EQ(rows.metrics, 4u);
    EXPECT_EQ(rows.points, 4u);
    ASSERT_EQ(rows.n, 5u);   // four points and target_info
    const int g = find(rows, "queue_depth");
    ASSERT_GE(g, 0);
    EXPECT_EQ(labels_of(rows.labels[g]), (Labels{{"__name__", "queue_depth"}, {"queue_name", "orders"}, {"job", "otel-demo/checkout"},
                                     {"instance", "i-1"}, {"service_name", "checkout"}, {"service_namespace", "otel-demo"},
                                     {"service_instance_id", "i-1"}}));
    EXPECT_EQ(rows.labels[g].substr(0, 9), std::string_view("__name__\x1F"));   // sorted by name
    EXPECT_EQ(rows.value[g], 7.5);
    EXPECT_EQ(rows.ts_ms[g], kTms);
    EXPECT_EQ(rows.start_ts_ms[g], kTms - 60000);
    EXPECT_TRUE(rows.hist[g].empty());
    const int c = find(rows, "app_requests_total", {{"http_status_code", "200"}});
    ASSERT_GE(c, 0);
    EXPECT_EQ(rows.value[c], 12.0);
    EXPECT_GE(find(rows, "app_inflight"), 0);
    const int gone = find(rows, "gone");
    ASSERT_GE(gone, 0);
    EXPECT_EQ(bits(rows.value[gone]), o::kStaleNaNBits);
    const int info = find(rows, "target_info");
    ASSERT_GE(info, 0);
    EXPECT_EQ(labels_of(rows.labels[info]), (Labels{{"__name__", "target_info"}, {"job", "otel-demo/checkout"}, {"instance", "i-1"},
                                        {"service_name", "checkout"}, {"service_namespace", "otel-demo"},
                                        {"service_instance_id", "i-1"}, {"process_pid", "42"}}));
    EXPECT_EQ(rows.value[info], 1.0);
    EXPECT_EQ(rows.ts_ms[info], kTms);
}

TEST(OtlpMetrics, HistogramsBecomeCumulativeBucketsSumAndCount) {
    Pb dp; dp.fixed64(3, kT); dp.fixed64(4, 10); dp.fixed64(5, bits(123.5));
    dp.len(6, packed64({2, 5, 3})); dp.len(7, packed64({bits(0.1), bits(1)})); dp.len(9, kv("route", any_str("/cart")));
    Pb h; h.len(1, dp.b); h.u64(2, 2);
    const std::string req = request(kResource, {metric("http.server.duration", "ms", 9, h.b)});
    bolt::Arena a; o::MetricRows rows;
    ASSERT_TRUE(o::decode_metrics(req, &a, &rows) == o::Status::kOk);
    const std::string base = "http_server_duration_milliseconds";
    const int b0 = find(rows, base + "_bucket", {{"le", "0.1"}, {"route", "/cart"}});
    const int b1 = find(rows, base + "_bucket", {{"le", "1"}});
    const int b2 = find(rows, base + "_bucket", {{"le", "+Inf"}});
    ASSERT_TRUE(b0 >= 0 && b1 >= 0 && b2 >= 0);
    EXPECT_EQ(rows.value[b0], 2.0);
    EXPECT_EQ(rows.value[b1], 7.0);
    EXPECT_EQ(rows.value[b2], 10.0);
    EXPECT_EQ(rows.value[find(rows, base + "_sum")], 123.5);
    EXPECT_EQ(rows.value[find(rows, base + "_count")], 10.0);
}

TEST(OtlpMetrics, ExponentialHistogramsAndSummaries) {
    Pb pos; pos.u64(1, zz(-2)); { Pb c; c.varint(4); c.varint(0); c.varint(6); pos.len(2, c.b); }
    Pb dp; dp.fixed64(3, kT); dp.fixed64(4, 11); dp.fixed64(5, bits(9.0)); dp.u64(6, zz(3)); dp.fixed64(7, 1); dp.len(8, pos.b);
    Pb eh; eh.len(1, dp.b); eh.u64(2, 2);
    Pb q; q.fixed64(1, bits(0.99)); q.fixed64(2, bits(250.0));
    Pb sp; sp.fixed64(3, kT); sp.fixed64(4, 5); sp.fixed64(5, bits(600.0)); sp.len(6, q.b);
    Pb su; su.len(1, sp.b);
    const std::string req = request(kResource, {metric("rpc.latency", "s", 10, eh.b), metric("gc.pause", "ms", 11, su.b)});
    bolt::Arena a; o::MetricRows rows;
    ASSERT_TRUE(o::decode_metrics(req, &a, &rows) == o::Status::kOk);
    const int e = find(rows, "rpc_latency_seconds");
    ASSERT_GE(e, 0);
    o::HistHead h; double hp[o::kHistBucketsPerSide], hn[o::kHistBucketsPerSide];
    ASSERT_TRUE(o::hist_decode(rows.hist[e], &h, hp, hn, o::kHistBucketsPerSide));
    EXPECT_EQ(h.schema, 3);
    EXPECT_EQ(h.count, 11.0);
    EXPECT_EQ(rows.value[e], 11.0);
    EXPECT_EQ(h.sum, 9.0);
    EXPECT_EQ(h.zero_count, 1.0);
    EXPECT_EQ(h.pos_offset, -1);                       // OTLP index -2 is Prometheus index -1
    ASSERT_EQ(h.n_pos, 3u);
    EXPECT_EQ(std::vector<double>(hp, hp + 3), (std::vector<double>{4, 0, 6}));
    EXPECT_EQ(h.n_neg, 0u);
    EXPECT_EQ(rows.ts_ms[e], kTms);
    EXPECT_FALSE(o::hist_decode(rows.hist[e].substr(1), &h, hp, hn, o::kHistBucketsPerSide));
    EXPECT_EQ(rows.value[find(rows, "gc_pause_milliseconds", {{"quantile", "0.99"}})], 250.0);
    EXPECT_EQ(rows.value[find(rows, "gc_pause_milliseconds_sum")], 600.0);
    EXPECT_EQ(rows.value[find(rows, "gc_pause_milliseconds_count")], 5.0);
}

TEST(OtlpMetrics, AScaleAboveEightIsMergedDown) {
    Pb pos; pos.u64(1, zz(0)); { Pb c; for (int i = 0; i < 8; ++i) c.varint(1); pos.len(2, c.b); }
    Pb dp; dp.fixed64(3, kT); dp.fixed64(4, 8); dp.u64(6, zz(10)); dp.len(8, pos.b);
    Pb eh; eh.len(1, dp.b); eh.u64(2, 2);
    bolt::Arena a; o::MetricRows rows;
    ASSERT_TRUE(o::decode_metrics(request(kResource, {metric("x", "", 10, eh.b)}), &a, &rows) == o::Status::kOk);
    o::HistHead h; double hp[o::kHistBucketsPerSide], hn[o::kHistBucketsPerSide];
    ASSERT_TRUE(o::hist_decode(rows.hist[find(rows, "x")], &h, hp, hn, o::kHistBucketsPerSide));
    EXPECT_EQ(h.schema, 8);
    EXPECT_EQ(h.pos_offset, 1);
    ASSERT_EQ(h.n_pos, 2u);
    EXPECT_EQ(hp[0], 4.0);                             // scale 10 -> 8: four source buckets per bucket
    EXPECT_EQ(hp[1], 4.0);
}

TEST(OtlpMetrics, DeltaTemporalitySeparatorBytesAndMalformedRequestsAreRefused) {
    bolt::Arena a; o::MetricRows rows;
    const std::string delta = request(kResource, {metric("ok", "", 5, gauge({number_point(1, {})})),
                                                  metric("d.count", "", 7, sum({number_point(1, {})}, 1, true))});
    const o::Status s = o::decode_metrics(delta, &a, &rows);
    EXPECT_TRUE(s == o::Status::kDelta);
    EXPECT_EQ(rows.n, 0u);                             // refused whole
    char why[256];
    EXPECT_NE(std::string(why, o::metrics_refusal(s, rows, why, sizeof(why))).find("'d.count' has delta temporality"), std::string::npos);
    const std::string sep = request(kResource, {metric("ok", "", 5, gauge({number_point(1, {kv("k", any_str("a\x1E" "b"))})}))});
    EXPECT_TRUE(o::decode_metrics(sep, &a, &rows) == o::Status::kLabelBytes);
    EXPECT_EQ(rows.n, 0u);
    const std::string good = request(kResource, {metric("ok", "", 5, gauge({number_point(1, {})}))});
    for (std::size_t cut = 1; cut < good.size(); ++cut) {
        const bool ok = o::decode_metrics(good.substr(0, cut), &a, &rows) == o::Status::kOk;
        EXPECT_FALSE(ok && rows.points == 1) << cut;   // a prefix never yields the whole request
    }
    EXPECT_TRUE(o::decode_metrics(std::string(), &a, &rows) == o::Status::kOk);
    EXPECT_EQ(rows.n, 0u);
}

TEST(OtlpMetrics, RepeatedAndEmptyLabelsFollowPrometheus) {
    // A point attribute named like a promoted resource label wins (it comes first); an empty value is dropped.
    const std::string req = request(kResource, {metric("m", "", 5, gauge({number_point(1, {
        kv("service.name", any_str("override")), kv("empty", any_str("")), kv("n", any_int(7)), kv("b", any_bool(true))})}))});
    bolt::Arena a; o::MetricRows rows;
    ASSERT_TRUE(o::decode_metrics(req, &a, &rows) == o::Status::kOk);
    const Labels l = labels_of(rows.labels[find(rows, "m")]);
    EXPECT_EQ(l.at("service_name"), "override");
    EXPECT_EQ(l.count("empty"), 0u);
    EXPECT_EQ(l.at("n"), "7");
    EXPECT_EQ(l.at("b"), "true");
}

TEST(OtlpMetrics, BatchCarriesTheFixedSchemaAndRequestKeys) {
    Pb pos; pos.u64(1, zz(0)); { Pb c; c.varint(2); pos.len(2, c.b); }
    Pb dp; dp.fixed64(3, kT); dp.fixed64(4, 2); dp.u64(6, zz(0)); dp.len(8, pos.b);
    Pb eh; eh.len(1, dp.b); eh.u64(2, 2);
    const std::string req = request(kResource, {metric("g", "", 5, gauge({number_point(1.5, {})})), metric("h", "", 10, eh.b)});
    bolt::Arena a;
    o::MetricRows parts[2];
    ASSERT_TRUE(o::decode_metrics(req, &a, &parts[0]) == o::Status::kOk);
    ASSERT_TRUE(o::decode_metrics(req, &a, &parts[1]) == o::Status::kOk);
    const std::uint64_t keys[2] = {o::request_key(req), o::request_key("other")};
    bolt::Arena ba;
    auto* b = ba.allocate_array<bolt::BoltBatch>(1);
    bolt::BoltBatch::init_empty(b);
    b->arena = &ba;
    ASSERT_TRUE(o::metrics_to_batch(parts, keys, 2, b));
    ASSERT_EQ(b->num_rows, 6);   // g, h, target_info, twice
    ASSERT_EQ(b->num_cols, o::kMetricCols);
    const bolt::BoltColumn* c = b->columns[b->read_epoch];
    EXPECT_EQ(o::batch_text(c[0], 0), parts[0].labels[0]);
    EXPECT_EQ(static_cast<const std::int64_t*>(c[1].data)[0], kTms);
    EXPECT_EQ(static_cast<const double*>(c[3].data)[0], 1.5);
    EXPECT_EQ(o::batch_text(c[4], 0), "");
    const std::string_view hex = o::batch_text(c[4], 1);
    ASSERT_EQ(hex.size(), parts[0].hist[1].size() * 2u);
    std::string raw(hex.size() / 2, '\0');
    ASSERT_TRUE(o::hex_decode(hex, raw.data()));
    EXPECT_EQ(raw, std::string(parts[0].hist[1]));
    EXPECT_FALSE(o::hex_decode("zz", raw.data()));
    const auto* rk = static_cast<const std::int64_t*>(c[5].data);
    EXPECT_EQ(rk[0], rk[2]);
    EXPECT_NE(rk[0], rk[3]);
    EXPECT_GE(rk[3], 0);
    for (std::uint32_t i = 0; i < o::kMetricCols; ++i) EXPECT_STREQ(b->schema.fields[i].name, o::metric_fields()[i].name);
}

}  // namespace
