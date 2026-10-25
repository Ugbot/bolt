// test_bolt_iceberg_metrics.cpp — G2ICE-296: Iceberg data-file metrics from a parquet file.
//
// The expected bytes are what Iceberg-Java 1.12 (Spark 4) wrote for the same values (probe recorded in the G2ICE-296
// tracker notes): float/double bounds exclude NaN and keep -0.0 below +0.0, string bounds truncate to 16 code points
// with the upper's last code point incremented, binary truncates to 16 bytes with a carry over 0xFF, a leaf inside a
// list gets only its size. The manifest round trip checks the bytes a client reads.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_budget.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_resource.h"
#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/manifest_avro.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/metrics.h"

namespace ice = bolt::lakehouse::iceberg;
namespace pq  = bolt::ingest::parquet;

namespace {

std::string unhex(const char* h) {
    std::string out;
    for (size_t i = 0; h[i] != '\0' && h[i + 1] != '\0'; i += 2) {
        out.push_back(static_cast<char>(std::stoi(std::string(h + i, 2), nullptr, 16)));
    }
    return out;
}

std::string upper_of(bool utf8, const std::string& v, uint32_t units, bool* ok) {
    uint8_t out[64];
    uint32_t n = 0;
    *ok = ice::truncate_upper(utf8, reinterpret_cast<const uint8_t*>(v.data()), static_cast<uint32_t>(v.size()), units,
                              out, sizeof(out), &n);
    return std::string(reinterpret_cast<const char*>(out), n);
}

uint8_t* bitmap(bolt::Arena* a, const std::vector<bool>& nulls) {
    const size_t n = nulls.size();
    auto* bm = static_cast<uint8_t*>(a->allocate((n + 7u) / 8u, 8u));
    std::memset(bm, 0, (n + 7u) / 8u);
    for (size_t i = 0; i < n; ++i) {
        if (!nulls[i]) bm[i >> 3] = static_cast<uint8_t>(bm[i >> 3] | (1u << (i & 7u)));
    }
    return bm;
}

bolt::BoltColumn bytes_col(bolt::Arena* a, const std::vector<std::string>& v, const std::vector<bool>& nulls,
                           bolt::BoltType t) {
    auto* offs = a->allocate_array<int32_t>(v.size() + 1u);
    std::string pool;
    for (size_t i = 0; i < v.size(); ++i) { offs[i] = static_cast<int32_t>(pool.size()); pool += v[i]; }
    offs[v.size()] = static_cast<int32_t>(pool.size());
    auto* data = static_cast<char*>(a->allocate(pool.size() + 1u, 8u));
    std::memcpy(data, pool.data(), pool.size());
    bolt::BoltColumn c = bolt::BoltColumn::make_var_binary(data, bitmap(a, nulls), offs, static_cast<int64_t>(v.size()), t, a);
    c.stats.all_valid = false;
    return c;
}

template <typename T>
bolt::BoltColumn flat_col(bolt::Arena* a, const std::vector<T>& v, const std::vector<bool>& nulls, bolt::BoltType t) {
    bolt::BoltColumn c = bolt::BoltColumn::make_flat_alloc(static_cast<int64_t>(v.size()), t, a);
    std::memcpy(c.data, v.data(), v.size() * sizeof(T));
    c.validity = bitmap(a, nulls);
    c.stats.all_valid = false;
    return c;
}

const double kNaN = std::numeric_limits<double>::quiet_NaN();

// One row group, the Java probe's values: id, f (float), d (double), s (string), bin (binary), zero (double). `batch_out`
// (optional) receives the batch written.
bool write_file(bolt::Arena* a, const uint8_t** out, uint64_t* out_len, const bolt::BoltBatch** batch_out = nullptr) {
    const std::vector<bool> none(4, false), n3 = {false, false, true, false};
    pq::ParquetWriteOpts o{};
    o.n_columns = 6;
    o.emit_statistics = true;
    const char* names[6] = {"id", "f", "d", "s", "bin", "zero"};
    const bolt::BoltType types[6] = {bolt::BoltType::Int64, bolt::BoltType::Float32, bolt::BoltType::Float64,
                                     bolt::BoltType::Utf8, bolt::BoltType::Binary, bolt::BoltType::Float64};
    for (uint32_t i = 0; i < 6; ++i) {
        std::snprintf(o.columns[i].name, sizeof(o.columns[i].name), "%s", names[i]);
        o.columns[i].type = types[i];
        o.columns[i].nullable = i != 0u;
        o.columns[i].has_field_id = true;
        o.columns[i].field_id = static_cast<int32_t>(i + 1u);
    }
    auto* b = a->allocate_array<bolt::BoltBatch>(1);
    bolt::BoltBatch::init_empty(b);
    b->num_cols = 6;
    b->num_rows = 4;
    bolt::BoltBatch::alloc_columns(b, a, 6);
    for (uint32_t i = 0; i < 6; ++i) b->schema.add_field(names[i], types[i], i != 0u);
    bolt::BoltColumn* c = b->columns[b->read_epoch];
    c[0] = flat_col<int64_t>(a, {1, 2, 3, 4}, none, bolt::BoltType::Int64);
    c[1] = flat_col<float>(a, {std::nanf(""), 1.5f, -2.5f, std::nanf("")}, none, bolt::BoltType::Float32);
    c[2] = flat_col<double>(a, {-0.0, 0.0, 0.0, 3.0}, n3, bolt::BoltType::Float64);
    c[3] = bytes_col(a, {"abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrstuvwxzz\xc3\xa9\xc3\xa9", "",
                         unhex("f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880"
                               "f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880f09f988078")},
                     n3, bolt::BoltType::Utf8);
    c[4] = bytes_col(a, {unhex("00ffff"), unhex("ff"), "", ""}, n3, bolt::BoltType::Binary);
    c[5] = flat_col<double>(a, {0.0, -0.0, 0.0, 0.0}, none, bolt::BoltType::Float64);
    pq::ParquetWriter* w = pq::parquet_write_open_mem(&o);
    if (w == nullptr || !pq::parquet_write_row_group(w, b)) return false;
    if (batch_out != nullptr) *batch_out = b;
    return pq::parquet_write_close_mem(w, a, out, out_len);
}

void metrics(bolt::Arena* a, ice::MetricsModeSpec mode, ice::FileStats* st, const pq::PqMeta** meta_out = nullptr) {
    const uint8_t* buf = nullptr;
    uint64_t len = 0;
    ASSERT_TRUE(write_file(a, &buf, &len));
    auto* meta = a->allocate_array<pq::PqMeta>(1);
    ASSERT_TRUE(pq::parquet_read_meta(buf, len, a, meta));
    ice::MetricsLeaf leaves[6];
    for (uint32_t i = 0; i < 6; ++i) { leaves[i].field_id = static_cast<int32_t>(i + 1u); leaves[i].mode = mode; }
    std::memset(st, 0, sizeof(*st));
    ASSERT_TRUE(ice::file_metrics_from_parquet(buf, len, meta, leaves, 6, a, st));
    if (meta_out != nullptr) *meta_out = meta;
}

const ice::ColumnStatEntry* find(const ice::FileStats& s, int32_t fid) {
    for (uint32_t i = 0; i < s.n_cols; ++i) if (s.cols[i].field_id == fid) return &s.cols[i];
    return nullptr;
}

std::string lo(const ice::ColumnStatEntry* e) { return e->has_lower ? std::string(e->lower, e->lower_len) : "<none>"; }
std::string hi(const ice::ColumnStatEntry* e) { return e->has_upper ? std::string(e->upper, e->upper_len) : "<none>"; }

}  // namespace

TEST(IcebergMetrics, TruncationMatchesIcebergJava) {
    bool ok = false;
    EXPECT_EQ(upper_of(false, unhex("0102030405060708090a0b0c0d0e0f1011121314"), 16, &ok), unhex("0102030405060708090a0b0c0d0e0f11"));
    EXPECT_TRUE(ok);
    EXPECT_EQ(upper_of(false, unhex("0000000000000000000000000000ffffffff"), 16, &ok), unhex("0000000000000000000000000001"));
    EXPECT_EQ(upper_of(false, unhex("000000000000000000000000000000ffffff"), 16, &ok), unhex("000000000000000000000000000001"));
    (void)upper_of(false, std::string(20, '\xff'), 16, &ok);
    EXPECT_FALSE(ok) << "all 0xFF: no upper bound";
    EXPECT_EQ(upper_of(true, std::string(20, 'a'), 16, &ok), std::string(15, 'a') + "b");
    const std::string max = unhex("f48fbfbf");                            // U+10FFFF
    EXPECT_EQ(upper_of(true, std::string(15, 'a') + max + max, 16, &ok), std::string(14, 'a') + "b");
    std::string all_max;
    for (int i = 0; i < 17; ++i) all_max += max;
    (void)upper_of(true, all_max + "z", 16, &ok);
    EXPECT_FALSE(ok) << "16 x U+10FFFF: no upper bound";
    EXPECT_EQ(upper_of(true, std::string(15, 'a') + unhex("ed9fbf") + "x", 16, &ok), std::string(15, 'a') + unhex("ee8080"))
        << "U+D7FF increments past the surrogates to U+E000";
    EXPECT_EQ(upper_of(true, "short", 16, &ok), "short") << "a value that fits bounds itself";
    const std::string lower(16, 'a');
    EXPECT_EQ(ice::truncate_lower_len(true, reinterpret_cast<const uint8_t*>(all_max.data()), 68, 16, 64), 64u);
    EXPECT_EQ(ice::truncate_lower_len(false, reinterpret_cast<const uint8_t*>(lower.data()), 16, 4, 64), 4u);
}

TEST(IcebergMetrics, ModeParsing) {
    ice::MetricsModeSpec m;
    ASSERT_TRUE(ice::metrics_mode_parse("truncate(4)", 11, &m));
    EXPECT_EQ(m.mode, ice::MetricsMode::kTruncate);
    EXPECT_EQ(m.length, 4u);
    ASSERT_TRUE(ice::metrics_mode_parse("Full", 4, &m));
    EXPECT_EQ(m.mode, ice::MetricsMode::kFull);
    ASSERT_TRUE(ice::metrics_mode_parse("counts", 6, &m));
    EXPECT_EQ(m.mode, ice::MetricsMode::kCounts);
    ASSERT_TRUE(ice::metrics_mode_parse("none", 4, &m));
    EXPECT_EQ(m.mode, ice::MetricsMode::kNone);
    EXPECT_FALSE(ice::metrics_mode_parse("truncate(0)", 11, &m));
    EXPECT_FALSE(ice::metrics_mode_parse("truncate(x)", 11, &m));
    EXPECT_FALSE(ice::metrics_mode_parse("trunc", 5, &m));
}

TEST(IcebergMetrics, DefaultTruncate16MatchesIcebergJava) {
    bolt::Arena a;
    ice::FileStats st;
    const pq::PqMeta* meta = nullptr;
    metrics(&a, ice::MetricsModeSpec{}, &st, &meta);
    ASSERT_EQ(st.n_cols, 6u);
    for (uint32_t i = 0; i < 6; ++i) {
        EXPECT_TRUE(st.cols[i].has_column_size);
        EXPECT_EQ(st.cols[i].column_size, meta->chunks[i].total_compressed_size);
        EXPECT_TRUE(st.cols[i].has_value_count);
        EXPECT_EQ(st.cols[i].value_count, 4);
    }
    const ice::ColumnStatEntry* f = find(st, 2);
    EXPECT_TRUE(f->has_nan_count);
    EXPECT_EQ(f->nan_count, 2);
    EXPECT_EQ(lo(f), unhex("000020c0"));
    EXPECT_EQ(hi(f), unhex("0000c03f"));
    const ice::ColumnStatEntry* d = find(st, 3);
    EXPECT_EQ(d->null_count, 1);
    EXPECT_EQ(d->nan_count, 0);
    EXPECT_EQ(lo(d), unhex("0000000000000080")) << "-0.0 sorts before +0.0";
    EXPECT_EQ(hi(d), unhex("0000000000000840"));
    const ice::ColumnStatEntry* s = find(st, 4);
    EXPECT_FALSE(s->has_nan_count);
    EXPECT_EQ(lo(s), "abcdefghijklmnop");
    EXPECT_EQ(hi(s), unhex("f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880"
                           "f09f9880f09f9880f09f9880f09f9880f09f9880f09f9881"));
    const ice::ColumnStatEntry* bin = find(st, 5);
    EXPECT_EQ(lo(bin), "");
    EXPECT_EQ(hi(bin), unhex("ff"));
    const ice::ColumnStatEntry* z = find(st, 6);
    EXPECT_EQ(lo(z), unhex("0000000000000080"));
    EXPECT_EQ(hi(z), unhex("0000000000000000"));
    const ice::ColumnStatEntry* id = find(st, 1);
    EXPECT_EQ(lo(id), unhex("0100000000000000"));
    EXPECT_EQ(hi(id), unhex("0400000000000000"));
    ASSERT_EQ(st.n_split_offsets, 1u);
    EXPECT_EQ(st.split_offsets[0], 4);
    EXPECT_TRUE(st.has_sort_order_id);
}

TEST(IcebergMetrics, ModesNoneCountsFull) {
    bolt::Arena a;
    ice::FileStats st;
    metrics(&a, ice::MetricsModeSpec{ice::MetricsMode::kCounts, 0}, &st);
    ASSERT_EQ(st.n_cols, 6u);
    EXPECT_FALSE(find(st, 4)->has_lower);
    EXPECT_EQ(find(st, 2)->nan_count, 2) << "counts mode keeps NaN counts";
    metrics(&a, ice::MetricsModeSpec{ice::MetricsMode::kFull, 0}, &st);
    EXPECT_EQ(lo(find(st, 4)), "abcdefghijklmnopqrstuvwxyz");
    EXPECT_EQ(hi(find(st, 4)), unhex("f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880"
                                     "f09f9880f09f9880f09f9880f09f9880f09f9880f09f9880f09f988078"))
        << "G2ICE-336: full keeps the whole value (69 bytes), as Iceberg-Java does";
    metrics(&a, ice::MetricsModeSpec{ice::MetricsMode::kNone, 0}, &st);
    EXPECT_EQ(st.n_cols, 0u) << "none: not even the size";
    metrics(&a, ice::MetricsModeSpec{ice::MetricsMode::kTruncate, 4}, &st);
    EXPECT_EQ(lo(find(st, 4)), "abcd");
    EXPECT_EQ(hi(find(st, 4)), unhex("f09f9880f09f9880f09f9880f09f9881"));
}

TEST(IcebergMetrics, ManifestRoundTripsEveryMetric) {
    bolt::Arena a;
    ice::DataFileRef f{};
    f.status  = ice::ManifestStatus::kAdded;
    f.content = ice::FileContent::kData;
    f.snapshot_id = 9;
    std::snprintf(f.file_path, sizeof(f.file_path), "s3://wh/d/t/data/r.parquet");
    metrics(&a, ice::MetricsModeSpec{}, &f.stats);
    f.stats.record_count = 4;
    f.stats.file_size_in_bytes = 1234;
    static const char kSchema[] = "{\"type\":\"struct\",\"schema-id\":0,\"fields\":[]}";
    const uint8_t* mbuf = nullptr;
    uint64_t mlen = 0;
    ASSERT_TRUE(ice::manifest_write_avro(&f, 1, 9, 1, kSchema, sizeof(kSchema) - 1u, 0, &a, &mbuf, &mlen));
    ice::DataFileRef out[2]{};
    uint32_t n = 0;
    ASSERT_TRUE(ice::manifest_parse_avro(mbuf, mlen, &a, 0, out, 2, &n));
    ASSERT_EQ(n, 1u);
    const ice::FileStats& r = out[0].stats;
    for (uint32_t i = 1; i <= 6; ++i) {
        const ice::ColumnStatEntry* w = find(f.stats, static_cast<int32_t>(i));
        const ice::ColumnStatEntry* g = find(r, static_cast<int32_t>(i));
        ASSERT_NE(g, nullptr);
        EXPECT_EQ(g->column_size, w->column_size);
        EXPECT_EQ(g->value_count, w->value_count);
        EXPECT_EQ(g->null_count, w->null_count);
        EXPECT_EQ(g->has_nan_count, w->has_nan_count);
        EXPECT_EQ(g->nan_count, w->nan_count);
        EXPECT_EQ(lo(g), lo(w)) << i;
        EXPECT_EQ(hi(g), hi(w)) << i;
    }
    ASSERT_EQ(r.n_split_offsets, 1u);
    EXPECT_EQ(r.split_offsets[0], 4);
    EXPECT_TRUE(r.has_sort_order_id);
    EXPECT_EQ(r.sort_order_id, 0);
}

// G2ICE-323: a v2 position-delete entry that names its one data file says so in referenced_data_file.
TEST(IcebergMetrics, V2ManifestCarriesReferencedDataFile) {
    bolt::Arena a;
    ice::DataFileRef f{};
    f.status  = ice::ManifestStatus::kAdded;
    f.content = ice::FileContent::kPositionDeletes;
    f.snapshot_id = 9;
    std::snprintf(f.file_path, sizeof(f.file_path), "s3://wh/d/t/data/d.parquet");
    std::snprintf(f.referenced_data_file, sizeof(f.referenced_data_file), "s3://wh/d/t/data/r.parquet");
    f.stats.record_count = 2;
    f.stats.file_size_in_bytes = 99;
    static const char kSchema[] = "{\"type\":\"struct\",\"schema-id\":0,\"fields\":[]}";
    const uint8_t* mbuf = nullptr;
    uint64_t mlen = 0;
    ASSERT_TRUE(ice::manifest_write_avro(&f, 1, 9, 1, kSchema, sizeof(kSchema) - 1u, 0, &a, &mbuf, &mlen));
    ice::DataFileRef out[2]{};
    uint32_t n = 0;
    ASSERT_TRUE(ice::manifest_parse_avro(mbuf, mlen, &a, 0, out, 2, &n));
    ASSERT_EQ(n, 1u);
    EXPECT_STREQ(out[0].referenced_data_file, "s3://wh/d/t/data/r.parquet");
    const std::string body(reinterpret_cast<const char*>(mbuf), mlen);
    EXPECT_NE(body.find("\"field-id\":143"), std::string::npos);
    EXPECT_NE(body.find("format-version"), std::string::npos);
}

// bolt's own Iceberg writer computes metrics from the batches it is about to write: they equal the ones read off the file.
TEST(IcebergMetrics, WriterBatchesAgreeWithTheWrittenFile) {
    bolt::Arena a;
    const uint8_t* buf = nullptr;
    uint64_t len = 0;
    const bolt::BoltBatch* batch = nullptr;
    ASSERT_TRUE(write_file(&a, &buf, &len, &batch));
    auto* meta = a.allocate_array<pq::PqMeta>(1);
    ASSERT_TRUE(pq::parquet_read_meta(buf, len, &a, meta));
    ice::MetricsLeaf leaves[6];
    for (uint32_t i = 0; i < 6; ++i) leaves[i].field_id = static_cast<int32_t>(i + 1u);
    ice::FileStats file{}, mem{};
    ASSERT_TRUE(ice::file_metrics_from_parquet(buf, len, meta, leaves, 6, &a, &file));
    ice::SchemaField fields[6]{};
    const char* names[6] = {"id", "f", "d", "s", "bin", "zero"};
    const char* types[6] = {"long", "float", "double", "string", "binary", "double"};
    for (uint32_t i = 0; i < 6; ++i) {
        fields[i].id = static_cast<int32_t>(i + 1u);
        std::snprintf(fields[i].name, sizeof(fields[i].name), "%s", names[i]);
        std::snprintf(fields[i].type, sizeof(fields[i].type), "%s", types[i]);
    }
    ice::Schema sch{};
    sch.n_fields = 6;
    sch.fields = fields;
    ASSERT_TRUE(ice::file_metrics_from_batches(&batch, 1u, &sch, ice::MetricsModeSpec{}, &a, &mem));
    ASSERT_EQ(mem.n_cols, 6u);
    for (uint32_t i = 1; i <= 6; ++i) {
        const ice::ColumnStatEntry* f = find(file, static_cast<int32_t>(i));
        const ice::ColumnStatEntry* g = find(mem, static_cast<int32_t>(i));
        ASSERT_NE(g, nullptr);
        EXPECT_EQ(g->value_count, f->value_count) << i;
        EXPECT_EQ(g->null_count, f->null_count) << i;
        EXPECT_EQ(g->has_nan_count, f->has_nan_count) << i;
        EXPECT_EQ(g->nan_count, f->nan_count) << i;
        EXPECT_EQ(lo(g), lo(f)) << i;
        EXPECT_EQ(hi(g), hi(f)) << i;
        EXPECT_FALSE(g->has_column_size) << "a size belongs to the written file";
    }
}

// G2ICE-336: what one file may carry is a budget, not a slot. Past it the file is refused by the knob's name; under it
// every column and every byte is kept.
namespace {

struct LimitGuard {
    uint32_t id;
    uint64_t saved;
    explicit LimitGuard(bolt::bolt_limits_id i, uint64_t v) : id(static_cast<uint32_t>(i)) {
        saved = bolt::limits_value(bolt::bolt_limits(), id);
        EXPECT_TRUE(bolt::limits_set(&bolt::bolt_limits(), id, v, nullptr, 0));
    }
    ~LimitGuard() { (void)bolt::limits_set(&bolt::bolt_limits(), id, saved, nullptr, 0); }
};

// `nc` long columns c<i> = i * 10 + row, two rows; field id i + 1.
const bolt::BoltBatch* wide_batch(bolt::Arena* a, uint32_t nc, ice::Schema* sch) {
    auto* b = a->allocate_array<bolt::BoltBatch>(1);
    bolt::BoltBatch::init_empty(b);
    b->num_cols = nc;
    b->num_rows = 2;
    bolt::BoltBatch::alloc_columns(b, a, nc);
    auto* fields = a->allocate_array<ice::SchemaField>(nc);
    std::memset(fields, 0, sizeof(ice::SchemaField) * nc);
    for (uint32_t i = 0; i < nc; ++i) {
        char name[16];
        std::snprintf(name, sizeof(name), "c%u", i);
        b->schema.add_field(name, bolt::BoltType::Int64, false);
        b->columns[b->read_epoch][i] = flat_col<int64_t>(a, {int64_t{i} * 10, int64_t{i} * 10 + 1}, {false, false},
                                                         bolt::BoltType::Int64);
        fields[i].id = static_cast<int32_t>(i + 1u);
        std::snprintf(fields[i].name, sizeof(fields[i].name), "%s", name);
        std::snprintf(fields[i].type, sizeof(fields[i].type), "long");
    }
    *sch = ice::Schema{};
    sch->n_fields = nc;
    sch->fields = fields;
    return b;
}

}  // namespace

TEST(IcebergMetricsBudget, EveryColumnPastTheOld32AndThroughTheManifest) {
    bolt::Arena a;
    ice::Schema sch;
    const bolt::BoltBatch* b = wide_batch(&a, 150u, &sch);
    ice::DataFileRef f{};
    f.status = ice::ManifestStatus::kAdded;
    f.content = ice::FileContent::kData;
    std::snprintf(f.file_path, sizeof(f.file_path), "s3://wh/t/data/a.parquet");
    f.stats.record_count = 2;
    f.stats.file_size_in_bytes = 100;
    ASSERT_TRUE(ice::file_metrics_from_batches(&b, 1u, &sch, ice::MetricsModeSpec{}, &a, &f.stats));
    ASSERT_EQ(f.stats.n_cols, 150u);
    static const char kSchema[] = "{\"type\":\"struct\",\"schema-id\":0,\"fields\":[]}";
    const uint8_t* mbuf = nullptr;
    uint64_t mlen = 0;
    ASSERT_TRUE(ice::manifest_write_avro(&f, 1, 9, 1, kSchema, sizeof(kSchema) - 1u, 0, &a, &mbuf, &mlen));
    ice::DataFileRef* out = nullptr;
    uint32_t n = 0;
    ASSERT_TRUE(ice::manifest_parse_avro_grow(mbuf, mlen, &a, 0, 1u, &out, &n));
    ASSERT_EQ(n, 1u);
    ASSERT_EQ(out[0].stats.n_cols, 150u);
    for (uint32_t i = 0; i < 150u; ++i) {
        const ice::ColumnStatEntry* e = find(out[0].stats, static_cast<int32_t>(i + 1u));
        ASSERT_NE(e, nullptr) << i;
        ASSERT_TRUE(e->has_lower && e->has_upper && e->lower_len == 8u) << i;
        int64_t lo_v = 0, hi_v = 0;
        std::memcpy(&lo_v, e->lower, 8);
        std::memcpy(&hi_v, e->upper, 8);
        EXPECT_EQ(lo_v, int64_t{i} * 10) << i;
        EXPECT_EQ(hi_v, int64_t{i} * 10 + 1) << i;
        EXPECT_EQ(e->value_count, 2) << i;
    }
}

TEST(IcebergMetricsBudget, ColumnsPastTheBudgetAreRefusedByName) {
    bolt::Arena a;
    ice::Schema sch;
    const bolt::BoltBatch* b = wide_batch(&a, 150u, &sch);
    LimitGuard g(bolt::bolt_limits_id::iceberg_stat_columns, 100u);
    ice::FileStats st{};
    bolt::clear_resource_exhausted();
    EXPECT_FALSE(ice::file_metrics_from_batches(&b, 1u, &sch, ice::MetricsModeSpec{}, &a, &st));
    ASSERT_NE(bolt::last_resource_exhausted().knob, nullptr);
    EXPECT_STREQ(bolt::last_resource_exhausted().knob, "iceberg_stat_columns");
    EXPECT_EQ(bolt::last_resource_exhausted().requested, 101u);
}

TEST(IcebergMetricsBudget, FullBoundPastTheBudgetIsRefusedNeverCut) {
    bolt::Arena a;
    ice::FileStats st;
    LimitGuard g(bolt::bolt_limits_id::iceberg_bound_bytes, 64u);
    const uint8_t* buf = nullptr;
    uint64_t len = 0;
    ASSERT_TRUE(write_file(&a, &buf, &len));
    auto* meta = a.allocate_array<pq::PqMeta>(1);
    ASSERT_TRUE(pq::parquet_read_meta(buf, len, &a, meta));
    ice::MetricsLeaf leaves[6];
    for (uint32_t i = 0; i < 6; ++i) { leaves[i].field_id = static_cast<int32_t>(i + 1u); leaves[i].mode = {ice::MetricsMode::kFull, 0}; }
    std::memset(&st, 0, sizeof(st));
    bolt::clear_resource_exhausted();
    EXPECT_FALSE(ice::file_metrics_from_parquet(buf, len, meta, leaves, 6, &a, &st)) << "a 69-byte full bound over a 64-byte budget";
    ASSERT_NE(bolt::last_resource_exhausted().knob, nullptr);
    EXPECT_STREQ(bolt::last_resource_exhausted().knob, "iceberg_bound_bytes");
    for (uint32_t i = 0; i < 6; ++i) leaves[i].mode = ice::MetricsModeSpec{};   // truncate(16): 64 bytes at most, fits
    bolt::clear_resource_exhausted();
    EXPECT_TRUE(ice::file_metrics_from_parquet(buf, len, meta, leaves, 6, &a, &st));
}

TEST(IcebergMetricsBudget, LongBoundsSurviveManifestEncodeAndCacheCodec) {
    bolt::Arena a;
    ice::DataFileRef f{};
    f.status = ice::ManifestStatus::kAdded;
    f.content = ice::FileContent::kData;
    std::snprintf(f.file_path, sizeof(f.file_path), "s3://wh/t/data/a.parquet");
    f.stats.record_count = 1;
    f.stats.file_size_in_bytes = 10;
    const std::string lo_v(5000, 'a'), hi_v(7000, 'z');
    ice::ColumnStatEntry* e = ice::stats_add_col(&f.stats, &a, 3);
    ASSERT_NE(e, nullptr);
    e->null_count = 0;
    ASSERT_TRUE(ice::stats_set_bound(e, &a, true, lo_v.data(), static_cast<uint32_t>(lo_v.size())));
    ASSERT_TRUE(ice::stats_set_bound(e, &a, false, hi_v.data(), static_cast<uint32_t>(hi_v.size())));
    const int64_t splits[3] = {4, 1000, 9000};
    ASSERT_TRUE(ice::stats_set_splits(&f.stats, &a, splits, 3));
    static const char kSchema[] = "{\"type\":\"struct\",\"schema-id\":0,\"fields\":[]}";
    const uint8_t* mbuf = nullptr;
    uint64_t mlen = 0;
    ASSERT_TRUE(ice::manifest_write_avro(&f, 1, 9, 1, kSchema, sizeof(kSchema) - 1u, 0, &a, &mbuf, &mlen));
    ice::DataFileRef* out = nullptr;
    uint32_t n = 0;
    ASSERT_TRUE(ice::manifest_parse_avro_grow(mbuf, mlen, &a, 0, 1u, &out, &n));
    ASSERT_EQ(n, 1u);
    EXPECT_EQ(lo(find(out[0].stats, 3)), lo_v);
    EXPECT_EQ(hi(find(out[0].stats, 3)), hi_v);
    ASSERT_EQ(out[0].stats.n_split_offsets, 3u);
    EXPECT_EQ(out[0].stats.split_offsets[2], 9000);
    std::vector<uint8_t> enc(ice::stats_encoded_size(f.stats));
    uint64_t got = 0;
    ASSERT_TRUE(ice::stats_encode(f.stats, enc.data(), enc.size(), &got));
    EXPECT_FALSE(ice::stats_encode(f.stats, enc.data(), enc.size() - 1u, &got));
    bolt::Arena b;
    ice::FileStats back{};
    ASSERT_TRUE(ice::stats_decode(enc.data(), got, &b, &back));
    EXPECT_EQ(lo(find(back, 3)), lo_v);
    EXPECT_EQ(hi(find(back, 3)), hi_v);
    EXPECT_EQ(back.split_offsets[1], 1000);
    EXPECT_FALSE(ice::stats_decode(enc.data(), got - 1u, &b, &back)) << "a short encoding is refused";
}
