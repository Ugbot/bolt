// OTLP decode: hand-encoded requests to rows and to fixed-schema batches;
// malformed input refused whole.

#include "bolt/ingest/bolt_otlp.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "otlp_test_pb.h"

namespace o = bolt::ingest::otlp;

namespace {

using namespace otlp_test;

TEST(OtlpLogs, DecodesResourceScopeAndEveryRecordField) {
    const std::string req = logs::sample_request();
    std::uint64_t count = 0;
    ASSERT_TRUE(o::count_logs(req, &count));
    EXPECT_EQ(count, 3u);
    bolt::Arena a;
    o::LogRows rows;
    ASSERT_TRUE((o::decode_logs(req, 77, &a, &rows) == o::Status::kOk));
    ASSERT_EQ(rows.n, 3u);
    const std::string stream = std::string("host.name\x1Fh1\x1Eprocess.pid\x1F") + "42\x1Eservice.name\x1F" + "checkout\x1E";
    for (std::uint32_t i = 0; i < 3; ++i) EXPECT_EQ(rows.stream[i], stream) << i;
    EXPECT_EQ(rows.ts[0], 1700000000000000000ll);
    EXPECT_EQ(rows.observed_ts[0], 1700000000500000000ll);
    EXPECT_EQ(rows.severity[0], 9);
    EXPECT_EQ(rows.severity_text[0], "INFO");
    EXPECT_EQ(rows.body[0], "order placed \"ok\"\n");
    EXPECT_EQ(rows.trace_id[0], logs::kTrace);
    EXPECT_EQ(rows.span_id[0], logs::kSpan);
    EXPECT_EQ(rows.event_name[0], "order.placed");
    EXPECT_EQ(rows.attrs[0], "{\"http.status_code\":200,\"ok\":true,\"ratio\":0.5,\"tags\":[\"a\",2],\"nested\":{\"k\":\"v\"},"
                             "\"otel.scope.name\":\"io.demo\",\"otel.scope.version\":\"1.2\"}");
    EXPECT_EQ(rows.ts[1], 1700000001000000000ll);          // no time: the observed time
    EXPECT_EQ(rows.severity[1], 17);
    EXPECT_EQ(rows.body[1], "{\"msg\":\"boom\"}");           // a non-string body is JSON
    EXPECT_TRUE(rows.trace_id[1].empty());                 // all-zero id
    EXPECT_TRUE(rows.span_id[1].empty());                  // wrong length
    EXPECT_EQ(rows.ts[2], 77);                             // neither time: now
    EXPECT_EQ(rows.observed_ts[2], 0);
    EXPECT_EQ(rows.body[2], "");
    EXPECT_EQ(rows.attrs[2], "{\"otel.scope.name\":\"io.demo\",\"otel.scope.version\":\"1.2\"}");
}

TEST(OtlpLogs, EveryTruncationAndABadWireTypeAreRefused) {
    const std::string req = logs::sample_request();
    bolt::Arena a;
    for (std::size_t cut = 1; cut < req.size(); ++cut) {
        o::LogRows rows;
        const std::string part = req.substr(0, cut);
        std::uint64_t count = 0;
        const bool counted = o::count_logs(part, &count);
        const bool decoded = (o::decode_logs(part, 1, &a, &rows) == o::Status::kOk);
        EXPECT_EQ(counted && count == 3, decoded && rows.n == 3) << cut;   // a prefix never yields the whole request
        EXPECT_FALSE(decoded && rows.n == 3) << cut;
    }
    std::string bad = req;
    bad[0] = static_cast<char>((1u << 3) | 3u);   // field 1 as a start-group
    o::LogRows rows;
    EXPECT_FALSE((o::decode_logs(bad, 1, &a, &rows) == o::Status::kOk));
    EXPECT_TRUE((o::decode_logs(std::string(), 1, &a, &rows) == o::Status::kOk));
    EXPECT_EQ(rows.n, 0u);
}

TEST(OtlpSpans, DecodesSpansEventsLinksAndStatus) {
    const std::string req = spans::sample_request();
    bolt::Arena a;
    o::SpanRows rows;
    ASSERT_TRUE((o::decode_spans(req, &a, &rows) == o::Status::kOk));
    ASSERT_EQ(rows.n, 2u);
    EXPECT_EQ(rows.stream[0], std::string("service.name\x1F") + "checkout\x1E");
    EXPECT_EQ(rows.name[0], "GET /checkout");
    EXPECT_EQ(rows.start_ts[0], 1700000000000000000ll);
    EXPECT_EQ(rows.duration_ns[0], 250000000ll);
    EXPECT_EQ(rows.kind[0], 2);
    EXPECT_EQ(rows.status_code[0], 2);
    EXPECT_EQ(rows.status_msg[0], "boom");
    EXPECT_TRUE(rows.parent_span_id[0].empty());
    EXPECT_EQ(rows.attrs[0], "{\"http.status_code\":500,\"otel.scope.name\":\"io.demo\"}");
    EXPECT_EQ(rows.events[0], "[{\"ts\":1700000000100000000,\"name\":\"exception\",\"attrs\":{\"exception.type\":\"IOError\"}}]");
    EXPECT_TRUE(rows.links[0].empty());
    EXPECT_EQ(rows.parent_span_id[1], spans::kRoot);
    EXPECT_EQ(rows.trace_state[1], "k=v");
    EXPECT_EQ(rows.duration_ns[1], 40000000ll);
    EXPECT_EQ(rows.links[1], "[{\"trace_id\":\"0102030405060708090a0b0c0d0e0f10\",\"span_id\":\"a1a2a3a4a5a6a7a8\",\"attrs\":{}}]");
    EXPECT_TRUE(rows.events[1].empty());
}

TEST(OtlpSpans, TruncationsAndSpansWithoutIdsAreRefused) {
    const std::string req = spans::sample_request();
    bolt::Arena a;
    for (std::size_t cut = 1; cut < req.size(); ++cut) {
        o::SpanRows rows;
        EXPECT_FALSE((o::decode_spans(req.substr(0, cut), &a, &rows) == o::Status::kOk) && rows.n == 2) << cut;
    }
    spans::Span no_trace; no_trace.trace = "short"; no_trace.id = spans::kRoot; no_trace.name = "x";
    spans::Span zero_span; zero_span.id = std::string(8, '\0'); zero_span.name = "x";
    o::SpanRows rows;
    EXPECT_FALSE((o::decode_spans(spans::request("s", {no_trace}), &a, &rows) == o::Status::kOk));
    EXPECT_FALSE((o::decode_spans(spans::request("s", {zero_span}), &a, &rows) == o::Status::kOk));
    EXPECT_TRUE((o::decode_spans(std::string(), &a, &rows) == o::Status::kOk));
    EXPECT_EQ(rows.n, 0u);
}

TEST(OtlpBatch, LogsPackEveryColumnAndKeysFollowTheRequestBytes) {
    const std::string req = logs::sample_request();
    bolt::Arena a;
    o::LogRows parts[2];
    ASSERT_TRUE(o::decode_logs(req, 77, &a, &parts[0]) == o::Status::kOk);
    ASSERT_TRUE(o::decode_logs(req, 77, &a, &parts[1]) == o::Status::kOk);
    const std::uint64_t keys[2] = {o::request_key(req), o::request_key("other")};
    ASSERT_NE(keys[0], keys[1]);
    bolt::Arena ba;
    auto* b = ba.allocate_array<bolt::BoltBatch>(1);
    bolt::BoltBatch::init_empty(b);
    b->arena = &ba;
    ASSERT_TRUE(o::logs_to_batch(parts, keys, 2, b));
    ASSERT_EQ(b->num_rows, 6);
    ASSERT_EQ(b->num_cols, o::kLogCols);
    const bolt::BoltColumn* c = b->columns[b->read_epoch];
    const auto i64 = [&](int col, int r) { return static_cast<const std::int64_t*>(c[col].data)[r]; };
    EXPECT_EQ(i64(0, 0), 1700000000000000000ll);
    EXPECT_EQ(i64(1, 0), 1700000000500000000ll);
    EXPECT_EQ(i64(2, 0), 9);
    EXPECT_EQ(o::batch_text(c[3], 0), "INFO");
    EXPECT_EQ(o::batch_text(c[4], 0), "order placed \"ok\"\n");
    EXPECT_EQ(o::batch_text(c[5], 2), "{\"otel.scope.name\":\"io.demo\",\"otel.scope.version\":\"1.2\"}");
    EXPECT_EQ(o::batch_text(c[6], 5), parts[0].stream[2]);
    EXPECT_EQ(o::batch_text(c[7], 0), "0102030405060708090a0b0c0d0e0f10");
    EXPECT_EQ(o::batch_text(c[8], 0), "a1a2a3a4a5a6a7a8");
    EXPECT_EQ(o::batch_text(c[7], 1), "");
    EXPECT_EQ(o::batch_text(c[9], 3), "order.placed");
    for (int r = 0; r < 3; ++r) {
        EXPECT_EQ(i64(10, r), o::log_key(keys[0], static_cast<std::uint32_t>(r)));   // a retry of the same bytes: the same keys
        EXPECT_NE(i64(10, r), i64(10, r + 3));                                       // other bytes: other keys
        EXPECT_GE(i64(10, r), 0);
    }
    EXPECT_STREQ(b->schema.fields[10].name, "log_key");
}

TEST(OtlpBatch, SpansPackIdsAsHex) {
    const std::string req = spans::sample_request();
    bolt::Arena a;
    o::SpanRows rows;
    ASSERT_TRUE(o::decode_spans(req, &a, &rows) == o::Status::kOk);
    bolt::Arena ba;
    auto* b = ba.allocate_array<bolt::BoltBatch>(1);
    bolt::BoltBatch::init_empty(b);
    b->arena = &ba;
    ASSERT_TRUE(o::spans_to_batch(&rows, 1, b));
    ASSERT_EQ(b->num_rows, 2);
    ASSERT_EQ(b->num_cols, o::kSpanCols);
    const bolt::BoltColumn* c = b->columns[b->read_epoch];
    EXPECT_EQ(static_cast<const std::int64_t*>(c[1].data)[1], 40000000ll);
    EXPECT_EQ(o::batch_text(c[2], 0), "0102030405060708090a0b0c0d0e0f10");
    EXPECT_EQ(o::batch_text(c[3], 1), "b1b2b3b4b5b6b7b8");
    EXPECT_EQ(o::batch_text(c[4], 0), "");
    EXPECT_EQ(o::batch_text(c[4], 1), "a1a2a3a4a5a6a7a8");
    EXPECT_EQ(o::batch_text(c[5], 0), "GET /checkout");
    EXPECT_EQ(static_cast<const std::int64_t*>(c[7].data)[0], 2);
    EXPECT_EQ(o::batch_text(c[8], 0), "boom");
    EXPECT_EQ(o::batch_text(c[13], 1), std::string("service.name\x1F") + "checkout\x1E");
    bolt::BoltBatch* e = ba.allocate_array<bolt::BoltBatch>(1);
    bolt::BoltBatch::init_empty(e);
    e->arena = &ba;
    ASSERT_TRUE(o::spans_to_batch(nullptr, 0, e));
    EXPECT_EQ(e->num_rows, 0);
}

}  // namespace
