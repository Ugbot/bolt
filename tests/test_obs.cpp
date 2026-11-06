#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#include "bolt/obs/log.h"
#include "bolt/obs/metrics.h"
#include "bolt/obs/trace.h"

BOLT_LOG_MODULE("test");

namespace {

std::string read_all(const std::string& path) {
    std::ifstream f(path);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

std::string temp_path(const char* stem) {
    const char* d = std::getenv("TMPDIR");
    return std::string(d != nullptr ? d : "/tmp") + "/" + stem + std::to_string(::getpid());
}

TEST(ObsLog, FormatsCapturedArgumentsOffThread) {
    const std::string path = temp_path("bolt_obs_log");
    bolt::log::Config c;
    c.path = path.c_str();
    c.format = bolt::log::Format::Json;
    c.spec = "info";
    ASSERT_TRUE(bolt::log::start(c));
    std::string held = "hello";
    const char* bounded = "abcdef";
    bolt::obs::ScopedTraceCtx ctx(bolt::obs::TraceCtx{0, 0xabcd, 0, 0});
    BLOG_INFO("s=%s d=%d u=%llu x=%x f=%.2f p=%.*s z=%zu %%\n", held.c_str(), -7,
              static_cast<unsigned long long>(1ull << 40), 255u, 1.5, 3, bounded, static_cast<std::size_t>(9));
    BLOG_DEBUG("not at this level");
    BLOG_EVENT(Warn, "slow_query", "\"elapsed_ms\":12");
    bolt::log::stop();
    const std::string got = read_all(path);
    EXPECT_NE(got.find("s=hello d=-7 u=1099511627776 x=ff f=1.50 p=abc z=9 %"), std::string::npos) << got;
    EXPECT_NE(got.find("\"trace_id\":\"000000000000abcd\""), std::string::npos) << got;
    EXPECT_NE(got.find("\"event\":\"slow_query\",\"elapsed_ms\":12"), std::string::npos) << got;
    EXPECT_EQ(got.find("not at this level"), std::string::npos);
    std::remove(path.c_str());
}

TEST(ObsLog, LiveModuleLevels) {
    char err[64];
    EXPECT_TRUE(bolt::log::set_spec("warn,test=debug", err, sizeof(err)));
    EXPECT_TRUE(bolt::log::enabled(bolt_log_module_, bolt::log::Level::Debug));
    EXPECT_TRUE(bolt::log::set_spec("error", err, sizeof(err)));
    EXPECT_FALSE(bolt::log::enabled(bolt_log_module_, bolt::log::Level::Warn));
    EXPECT_FALSE(bolt::log::set_spec("bogus", err, sizeof(err)));
    EXPECT_TRUE(bolt::log::set_spec("info", err, sizeof(err)));
}

TEST(ObsLog, FullRingDropsAndCountsNeverBlocks) {
    const std::string path = temp_path("bolt_obs_drop");
    bolt::log::Config c;
    c.path = path.c_str();
    c.spec = "info";
    ASSERT_TRUE(bolt::log::start(c));
    const std::uint64_t before = bolt::log::stats().dropped;
    std::vector<std::thread> ts;
    std::atomic<std::uint64_t> attempted{0};
    for (int t = 0; t < 4; ++t) {
        ts.emplace_back([&attempted] {
            for (int i = 0; i < 200000; ++i) { BLOG_INFO("n=%d", i); attempted.fetch_add(1, std::memory_order_relaxed); }
        });
    }
    for (auto& t : ts) t.join();
    bolt::log::stop();
    const auto s = bolt::log::stats();
    std::uint64_t written = 0;
    const std::string got = read_all(path);
    for (char ch : got) written += ch == '\n';
    EXPECT_GT(s.dropped, before);
    EXPECT_GE(written + (s.dropped - before), attempted.load());
    std::remove(path.c_str());
}

TEST(ObsMetrics, CountersSumAcrossThreads) {
    auto c = bolt::metrics::counter("t_events_total", "events", "kind=\"a\"");
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t) ts.emplace_back([c] { for (int i = 0; i < 100000; ++i) c.inc(); });
    for (auto& t : ts) t.join();
    EXPECT_EQ(bolt::metrics::counter_value(c), 800000u);
    bolt::metrics::set_enabled(false);
    c.inc(5);
    EXPECT_EQ(bolt::metrics::counter_value(c), 800000u);
    bolt::metrics::set_enabled(true);
}

TEST(ObsMetrics, HistogramBoundsAndOpenMetrics) {
    auto h = bolt::metrics::histogram("t_latency_seconds", "lat", "", bolt::metrics::kLatency);
    h.observe(1024);       // exactly the top of bucket 0
    h.observe(1025);
    h.observe(1ull << 50); // past every finite bucket
    std::string prom;
    bolt::metrics::render(&prom, bolt::metrics::Format::Prometheus);
    EXPECT_NE(prom.find("t_latency_seconds_bucket{le=\"1.024e-06\"} 1"), std::string::npos) << prom;
    EXPECT_NE(prom.find("t_latency_seconds_bucket{le=\"2.048e-06\"} 2"), std::string::npos);
    EXPECT_NE(prom.find("t_latency_seconds_bucket{le=\"+Inf\"} 3"), std::string::npos);
    EXPECT_NE(prom.find("t_latency_seconds_count 3"), std::string::npos);
    std::string om;
    bolt::metrics::render(&om, bolt::metrics::Format::OpenMetrics);
    EXPECT_NE(om.find("# TYPE t_events counter"), std::string::npos) << om;
    EXPECT_NE(om.find("t_events_total{kind=\"a\"}"), std::string::npos);
}

TEST(ObsMetrics, LegacyBodyToOpenMetrics) {
    std::string out;
    bolt::metrics::to_openmetrics("# HELP a_total x y\n# TYPE a_total counter\na_total 1\n# HELP g z\n# TYPE g gauge\ng 2\n", &out);
    EXPECT_EQ(out, "# HELP a x y\n# TYPE a counter\na_total 1\n# HELP g z\n# TYPE g gauge\ng 2\n");
}

struct Slot { bolt::metrics::Counter c; };

TEST(ObsMetrics, LabelTableIsBounded) {
    bolt::metrics::LabelTable t(4, sizeof(Slot), [](std::string_view k, void* s, void*) {
        const std::string l = "k=\"" + std::string(k) + "\"";
        static_cast<Slot*>(s)->c = bolt::metrics::counter("t_labeled_total", "x", l);
    }, nullptr);
    auto* a = static_cast<Slot*>(t.find_or_add("a"));
    EXPECT_EQ(a, t.find_or_add("a"));
    for (int i = 0; i < 100; ++i) static_cast<Slot*>(t.find_or_add(std::to_string(i)))->c.inc();
    EXPECT_EQ(t.size(), 4u);
    std::string prom;
    bolt::metrics::render(&prom, bolt::metrics::Format::Prometheus);
    EXPECT_NE(prom.find("t_labeled_total{k=\"other\"} 97"), std::string::npos) << prom;
}

TEST(ObsTrace, SampledSpansNestAndUnsampledCostNothing) {
    ASSERT_TRUE(bolt::trace::start(nullptr, nullptr));
    bolt::trace::set_rate_ppm(1000000);
    EXPECT_TRUE(bolt::trace::sample(77));
    {
        bolt::obs::ScopedTraceCtx ctx(bolt::obs::TraceCtx{1, 2, 0, 1});
        bolt::trace::Span root("root");
        ASSERT_TRUE(root.active());
        bolt::trace::Span child("child");
        child.attrs("\"rows\":3");
    }
    {
        bolt::obs::ScopedTraceCtx ctx(bolt::obs::TraceCtx{1, 3, 0, 0});
        bolt::trace::Span skipped("skipped");
        EXPECT_FALSE(skipped.active());
    }
    bolt::trace::stop();
    std::string json;
    bolt::trace::recent_json(&json, 10, 2);
    EXPECT_NE(json.find("\"name\":\"child\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"rows\":3"), std::string::npos);
    EXPECT_EQ(json.find("skipped"), std::string::npos);
    bolt::trace::set_rate_ppm(0);
    EXPECT_FALSE(bolt::trace::sample(77));
}

}  // namespace
