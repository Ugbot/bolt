// G2CHK-975: PqMeta::nanos_lane. TIMESTAMP(NANOS) keeps every nanosecond on
// an Int64 lane; INT96 floors to a microsecond Timestamp (DuckDB / Spark);
// without the opt-in a sub-microsecond digit is still refused.
#include <gtest/gtest.h>

#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_read.h"

namespace {

#include "test_bolt_parquet_nanos_fixtures.inc"

using bolt::Arena;
using bolt::BoltBatch;
using bolt::BoltType;
namespace pq = bolt::ingest::parquet;

bool valid(const bolt::BoltColumn& c, int r) {
    return c.validity == nullptr || ((c.validity[r >> 3] >> (r & 7)) & 1u) != 0;
}

TEST(BoltParquetNanos, NanosLaneKeepsEveryNanosecond) {
    Arena arena; BoltBatch b;
    ASSERT_TRUE(pq::parquet_read_file(k_ts_nanos, sizeof(k_ts_nanos), &arena, &b, nullptr, true));
    ASSERT_EQ(b.num_rows, 4);
    const auto& c = b.col(0);
    EXPECT_EQ(c.type, BoltType::Int64);
    const auto* v = static_cast<const int64_t*>(c.data);
    EXPECT_EQ(v[0], 1373892000123456789LL);
    EXPECT_EQ(v[1], -612947027621527012LL);
    EXPECT_FALSE(valid(c, 2));
    EXPECT_EQ(v[3], 5);
}

TEST(BoltParquetNanos, Int96FloorsToMicroseconds) {
    Arena arena; BoltBatch b;
    ASSERT_TRUE(pq::parquet_read_file(k_ts_int96, sizeof(k_ts_int96), &arena, &b, nullptr, true));
    const auto& c = b.col(0);
    EXPECT_EQ(c.type, BoltType::Timestamp);
    const auto* v = static_cast<const int64_t*>(c.data);
    EXPECT_EQ(v[0], 1373892000123456LL);
    EXPECT_EQ(v[1], -612947027621528LL);   // floor, as DuckDB reads INT96
    EXPECT_FALSE(valid(c, 2));
    EXPECT_EQ(v[3], 0);
}

TEST(BoltParquetNanos, WithoutTheLaneSubMicrosecondIsRefused) {
    Arena a1; BoltBatch b1;
    EXPECT_FALSE(pq::parquet_read_file(k_ts_nanos, sizeof(k_ts_nanos), &a1, &b1));
    Arena a2; BoltBatch b2;
    EXPECT_FALSE(pq::parquet_read_file(k_ts_int96, sizeof(k_ts_int96), &a2, &b2));
}

TEST(BoltParquetNanos, ColumnIsNanosNamesOnlyTimestampNanos) {
    Arena arena;
    pq::PqMeta m{};
    ASSERT_TRUE(pq::parquet_read_meta(k_ts_nanos, sizeof(k_ts_nanos), &arena, &m));
    EXPECT_TRUE(pq::parquet_column_is_nanos(&m, 0));
    pq::PqMeta m96{};
    ASSERT_TRUE(pq::parquet_read_meta(k_ts_int96, sizeof(k_ts_int96), &arena, &m96));
    EXPECT_FALSE(pq::parquet_column_is_nanos(&m96, 0));
}

}  // namespace
