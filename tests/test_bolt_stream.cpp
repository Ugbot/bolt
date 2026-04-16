// test_bolt_stream.cpp — GTest suite for bolt::stream (v0 pipeline).
//
// Covers: lifecycle, single-stage copy, two-stage double+sum, drop semantics,
// arena ring reuse under tiny capacity, and umbrella header reachability.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "bolt/bolt.h"  // umbrella — also exercises __has_include pull

using namespace bolt;
using namespace bolt::stream;

namespace {

SchedulerConfig make_latency_cfg() noexcept {
    SchedulerConfig cfg = SchedulerConfig::with_profile(TuneProfile::Latency);
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw >= 2) cfg.num_workers = (hw >= 4) ? 4u : hw;
    else         cfg.num_workers = 2;
    return cfg;
}

// Build a single-column Int64 BoltBatch whose data buffer lives inside
// `arena` (values copied in). Does NOT allocate the BoltBatch itself; caller
// provides it.
void make_int64_batch(BoltBatch* out, Arena* arena,
                      const int64_t* values, int64_t n) noexcept {
    BoltBatch::init_empty(out);
    out->arena     = arena;
    out->num_rows  = n;
    out->num_cols  = 1;

    int64_t* buf = static_cast<int64_t*>(arena->allocate(n * sizeof(int64_t), alignof(int64_t)));
    ASSERT_NE(buf, nullptr);
    for (int64_t i = 0; i < n; ++i) buf[i] = values[i];

    BoltColumn c = BoltColumn::make_flat(buf, nullptr, n, BoltType::Int64);
    out->columns[out->read_epoch][0]  = c;
    out->columns[out->write_epoch][0] = c;
}

// ---- Stage callbacks ----

// Shallow copy: forward the single column view from in_batch to out_batch.
bool stage_copy(const BoltBatch* in_batch, BoltBatch* out_batch,
                Scheduler* /*sched*/, Arena* /*arena*/,
                void* /*state*/) noexcept {
    out_batch->num_rows = in_batch->num_rows;
    out_batch->num_cols = in_batch->num_cols;
    for (uint32_t i = 0; i < in_batch->num_cols; ++i) {
        const BoltColumn& src = in_batch->col(i);
        out_batch->columns[out_batch->read_epoch][i]  = src;
        out_batch->columns[out_batch->write_epoch][i] = src;
    }
    return true;
}

// Double every Int64 in column 0 into a fresh arena-backed buffer.
bool stage_double(const BoltBatch* in_batch, BoltBatch* out_batch,
                  Scheduler* /*sched*/, Arena* arena,
                  void* /*state*/) noexcept {
    const BoltColumn& src = in_batch->col(0);
    const int64_t n = src.length;
    int64_t* out = static_cast<int64_t*>(arena->allocate(n * sizeof(int64_t), alignof(int64_t)));
    if (!out) return false;
    const int64_t* in = src.typed_data<int64_t>();
    for (int64_t i = 0; i < n; ++i) out[i] = in[i] * 2;

    BoltColumn c = BoltColumn::make_flat(out, nullptr, n, BoltType::Int64);
    out_batch->num_rows = n;
    out_batch->num_cols = 1;
    out_batch->columns[out_batch->read_epoch][0]  = c;
    out_batch->columns[out_batch->write_epoch][0] = c;
    return true;
}

// Reduce column 0 (Int64) to a 1-row Int64 sum, stored in arena.
bool stage_sum(const BoltBatch* in_batch, BoltBatch* out_batch,
               Scheduler* /*sched*/, Arena* arena,
               void* /*state*/) noexcept {
    const BoltColumn& src = in_batch->col(0);
    const int64_t n = src.length;
    const int64_t* in = src.typed_data<int64_t>();
    int64_t s = 0;
    for (int64_t i = 0; i < n; ++i) s += in[i];

    int64_t* out = static_cast<int64_t*>(arena->allocate(sizeof(int64_t), alignof(int64_t)));
    if (!out) return false;
    *out = s;

    BoltColumn c = BoltColumn::make_flat(out, nullptr, 1, BoltType::Int64);
    out_batch->num_rows = 1;
    out_batch->num_cols = 1;
    out_batch->columns[out_batch->read_epoch][0]  = c;
    out_batch->columns[out_batch->write_epoch][0] = c;
    return true;
}

struct DropEveryOther {
    std::atomic<uint32_t> calls{0};
};

bool stage_drop_every_other(const BoltBatch* in_batch, BoltBatch* out_batch,
                            Scheduler* /*sched*/, Arena* /*arena*/,
                            void* state) noexcept {
    auto* s = static_cast<DropEveryOther*>(state);
    const uint32_t n = s->calls.fetch_add(1, std::memory_order_relaxed);
    if ((n & 1) != 0) return false;
    return stage_copy(in_batch, out_batch, nullptr, nullptr, nullptr);
}

// ---- Sinks ----

struct CountSink {
    std::atomic<uint32_t> seen{0};
    int64_t last_sum = 0;
    std::vector<int64_t> sums;
};

void sink_count(const BoltBatch* /*result*/, void* user) noexcept {
    auto* s = static_cast<CountSink*>(user);
    s->seen.fetch_add(1, std::memory_order_relaxed);
}

void sink_record_sum(const BoltBatch* result, void* user) noexcept {
    auto* s = static_cast<CountSink*>(user);
    const int64_t v = result->col(0).typed_data<int64_t>()[0];
    s->last_sum = v;
    s->sums.push_back(v);
    s->seen.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Create / destroy
// ---------------------------------------------------------------------------
TEST(BoltStream, PipelineCreateDestroy) {
    Scheduler sched{};
    ASSERT_TRUE(sched.init(make_latency_cfg()));

    CountSink sink;
    PipelineConfig cfg;
    cfg.sched              = &sched;
    cfg.sink               = &sink_count;
    cfg.sink_user          = &sink;
    cfg.in_flight_capacity = 4;

    Pipeline p{};
    ASSERT_TRUE(pipeline_create(&p, cfg));
    EXPECT_EQ(p.num_stages, 0u);
    EXPECT_GE(p.arena_ring.capacity, 4u);
    EXPECT_EQ(p.arena_ring.capacity & (p.arena_ring.capacity - 1), 0u);

    pipeline_destroy(&p);
    sched.shutdown();
}

// ---------------------------------------------------------------------------
// 2. Single stage filter (copy-through)
// ---------------------------------------------------------------------------
TEST(BoltStream, SingleStageFilter) {
    Scheduler sched{};
    ASSERT_TRUE(sched.init(make_latency_cfg()));

    CountSink sink;
    PipelineConfig cfg;
    cfg.sched = &sched; cfg.sink = &sink_count; cfg.sink_user = &sink;
    cfg.in_flight_capacity = 8;

    Pipeline p{};
    ASSERT_TRUE(pipeline_create(&p, cfg));
    ASSERT_TRUE(pipeline_add_stage(&p, &stage_copy, nullptr, "copy"));

    Arena input_arena;
    const int64_t vals[5] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 5; ++i) {
        input_arena.reset();
        BoltBatch b;
        make_int64_batch(&b, &input_arena, vals, 5);
        EXPECT_TRUE(pipeline_push(&p, &b));
    }
    EXPECT_EQ(sink.seen.load(), 5u);
    EXPECT_EQ(p.batches_pushed.load(),  5u);
    EXPECT_EQ(p.batches_drained.load(), 5u);
    EXPECT_EQ(p.batches_dropped.load(), 0u);

    pipeline_destroy(&p);
    sched.shutdown();
}

// ---------------------------------------------------------------------------
// 3. Two-stage pipeline: double -> sum
// ---------------------------------------------------------------------------
TEST(BoltStream, TwoStagePipeline) {
    Scheduler sched{};
    ASSERT_TRUE(sched.init(make_latency_cfg()));

    CountSink sink;
    PipelineConfig cfg;
    cfg.sched = &sched; cfg.sink = &sink_record_sum; cfg.sink_user = &sink;
    cfg.in_flight_capacity = 4;

    Pipeline p{};
    ASSERT_TRUE(pipeline_create(&p, cfg));
    ASSERT_TRUE(pipeline_add_stage(&p, &stage_double, nullptr, "double"));
    ASSERT_TRUE(pipeline_add_stage(&p, &stage_sum,    nullptr, "sum"));

    // Push 3 batches with known sums * 2.
    const int64_t v0[3] = {1, 2, 3};     // sum*2 = 12
    const int64_t v1[4] = {10, 20, 30, 40};  // 200
    const int64_t v2[2] = {7, 8};        // 30

    Arena a;
    BoltBatch b;
    a.reset(); make_int64_batch(&b, &a, v0, 3); EXPECT_TRUE(pipeline_push(&p, &b));
    a.reset(); make_int64_batch(&b, &a, v1, 4); EXPECT_TRUE(pipeline_push(&p, &b));
    a.reset(); make_int64_batch(&b, &a, v2, 2); EXPECT_TRUE(pipeline_push(&p, &b));

    ASSERT_EQ(sink.sums.size(), 3u);
    EXPECT_EQ(sink.sums[0], 12);
    EXPECT_EQ(sink.sums[1], 200);
    EXPECT_EQ(sink.sums[2], 30);

    pipeline_drain(&p);  // no-op in v0, must not crash.

    pipeline_destroy(&p);
    sched.shutdown();
}

// ---------------------------------------------------------------------------
// 4. Stage returns false on every other batch.
// ---------------------------------------------------------------------------
TEST(BoltStream, StageDropsBatch) {
    Scheduler sched{};
    ASSERT_TRUE(sched.init(make_latency_cfg()));

    CountSink sink;
    PipelineConfig cfg;
    cfg.sched = &sched; cfg.sink = &sink_count; cfg.sink_user = &sink;
    cfg.in_flight_capacity = 4;

    Pipeline p{};
    ASSERT_TRUE(pipeline_create(&p, cfg));
    DropEveryOther state;
    ASSERT_TRUE(pipeline_add_stage(&p, &stage_drop_every_other, &state, "drop"));

    Arena a;
    const int64_t vals[3] = {1, 2, 3};
    for (int i = 0; i < 6; ++i) {
        a.reset();
        BoltBatch b;
        make_int64_batch(&b, &a, vals, 3);
        pipeline_push(&p, &b);
    }
    EXPECT_EQ(p.batches_pushed.load(),  6u);
    EXPECT_EQ(p.batches_dropped.load(), 3u);
    EXPECT_EQ(p.batches_drained.load(), 3u);
    EXPECT_EQ(sink.seen.load(),         3u);

    pipeline_destroy(&p);
    sched.shutdown();
}

// ---------------------------------------------------------------------------
// 5. Arena ring reuse under tight capacity.
// ---------------------------------------------------------------------------
TEST(BoltStream, ArenaRingBackpressure) {
    Scheduler sched{};
    ASSERT_TRUE(sched.init(make_latency_cfg()));

    CountSink sink;
    PipelineConfig cfg;
    cfg.sched = &sched; cfg.sink = &sink_count; cfg.sink_user = &sink;
    cfg.in_flight_capacity = 2;  // rounded to 2

    Pipeline p{};
    ASSERT_TRUE(pipeline_create(&p, cfg));
    ASSERT_TRUE(pipeline_add_stage(&p, &stage_copy, nullptr, "copy"));
    EXPECT_EQ(p.arena_ring.capacity, 2u);

    Arena a;
    const int64_t vals[2] = {9, 10};
    for (int i = 0; i < 10; ++i) {
        a.reset();
        BoltBatch b;
        make_int64_batch(&b, &a, vals, 2);
        EXPECT_TRUE(pipeline_push(&p, &b));
    }
    EXPECT_EQ(p.batches_drained.load(), 10u);
    EXPECT_EQ(sink.seen.load(),         10u);

    pipeline_destroy(&p);
    sched.shutdown();
}

// ---------------------------------------------------------------------------
// 6. Umbrella header reachability — compile-time check.
// ---------------------------------------------------------------------------
TEST(BoltStream, UmbrellaIncludesStream) {
    static_assert(sizeof(bolt::stream::Pipeline) > 0,
                  "umbrella bolt.h must pull in bolt/stream/bolt_stream.h");
    static_assert(bolt::stream::kStreamMaxStages == 16,
                  "stage bound documented in header");
    SUCCEED();
}
