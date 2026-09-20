// test_bolt_iceberg_manifest_write.cpp — the Avro manifest WRITE path.
//
// The oracle is bolt's own reader, and that is a genuinely strong one here:
// `manifest_parse_avro` / `manifest_list_parse_avro` were built and verified
// against manifests written by **pyiceberg 0.11.1** (see
// test_bolt_iceberg_real_avro.cpp, which asserts values from a real table).
// So "our bytes survive that reader" means our bytes are shaped the way a real
// Iceberg writer's are — not merely self-consistent.
//
// Every assertion below is a VALUE carried through the encode/decode round
// trip (paths, record counts, sizes, status), never a count of records, so a
// writer that mis-orders or drops a field cannot pass by accident. The
// negative cases exist because both failures they cover are SILENT: a
// partitioned file would encode into a schema that does not declare its
// partition tuple, and a non-null container would write no bytes at all,
// truncating the row and misaligning every field after it.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/ingest/bolt_avro.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/manifest_avro.h"

namespace ice = bolt::lakehouse::iceberg;
namespace ing = bolt::ingest;

namespace {

const char kSchemaJson[] =
    "{\"type\":\"struct\",\"schema-id\":0,\"fields\":["
    "{\"id\":1,\"name\":\"id\",\"required\":false,\"type\":\"long\"},"
    "{\"id\":2,\"name\":\"sym\",\"required\":false,\"type\":\"string\"}]}";

ice::DataFileRef make_file(const char* path, int64_t rows, int64_t bytes,
                           int64_t snap) {
    ice::DataFileRef d{};
    d.status  = ice::ManifestStatus::kAdded;
    d.content = ice::FileContent::kData;
    d.snapshot_id = snap;
    d.partition_spec_id = 0;
    d.n_partition = 0;
    std::snprintf(d.file_path, sizeof(d.file_path), "%s", path);
    d.stats.record_count       = rows;
    d.stats.file_size_in_bytes = bytes;
    return d;
}

}  // namespace

// A manifest we write parses back through the pyiceberg-verified reader with
// every value intact.
TEST(IcebergManifestWrite, ManifestRoundTrip) {
    bolt::Arena arena;
    ice::DataFileRef in[2] = {
        make_file("s3://wh/db/t/data/b-1.parquet", 1000, 165, 777),
        make_file("s3://wh/db/t/data/b-2.parquet", 2000, 170, 777),
    };

    const uint8_t* buf = nullptr;
    uint64_t len = 0;
    ASSERT_TRUE(ice::manifest_write_avro(in, 2, /*snapshot_id=*/777,
                                         /*sequence_number=*/3, kSchemaJson,
                                         sizeof(kSchemaJson) - 1u,
                                         /*partition_spec_id=*/0, &arena, &buf,
                                         &len));
    ASSERT_NE(buf, nullptr);
    ASSERT_GT(len, 0u);
    EXPECT_EQ(std::memcmp(buf, "Obj\x01", 4), 0);   // OCF magic

    ice::DataFileRef out[8]{};
    uint32_t n = 0;
    ASSERT_TRUE(ice::manifest_parse_avro(buf, len, &arena, /*default_spec_id=*/0,
                                         out, 8, &n));
    ASSERT_EQ(n, 2u);

    EXPECT_STREQ(out[0].file_path, "s3://wh/db/t/data/b-1.parquet");
    EXPECT_EQ(out[0].stats.record_count, 1000);
    EXPECT_EQ(out[0].stats.file_size_in_bytes, 165);
    EXPECT_EQ(out[0].status, ice::ManifestStatus::kAdded);
    EXPECT_EQ(out[0].content, ice::FileContent::kData);
    EXPECT_EQ(out[0].snapshot_id, 777);

    // The SECOND row is what catches a field-misalignment: a writer that drops
    // or reorders one field still tends to decode row 0 plausibly.
    EXPECT_STREQ(out[1].file_path, "s3://wh/db/t/data/b-2.parquet");
    EXPECT_EQ(out[1].stats.record_count, 2000);
    EXPECT_EQ(out[1].stats.file_size_in_bytes, 170);
}

// G2ICE-135 — per-column null_count + numeric bounds now round-trip through
// the SAME pyiceberg-verified reader the rest of this file trusts. Two
// columns, deliberately different: `id` has NO nulls (a real, certified
// ZERO — the case a "defaults to 0 on absence" bug could hide behind) and a
// real [10,50] bound; `score` has a NONZERO null_count and no bound (as a
// column with nulls-only, or a type this writer doesn't bound, would report)
// — so a writer that fabricates zeros, or that only ever emits one column's
// worth of stats, fails one half or the other.
TEST(IcebergManifestWrite, ColumnStatsRoundTrip) {
    bolt::Arena arena;
    ice::DataFileRef in = make_file("s3://wh/db/t/data/b-1.parquet", 5, 99, 1);
    in.stats.n_cols = 2;
    // id: field_id 1, zero nulls (real, certified), bound [10, 50].
    ice::ColumnStatEntry& c0 = in.stats.cols[0];
    c0.field_id   = 1;
    c0.null_count = 0;
    const int64_t id_lo = 10, id_hi = 50;
    std::memcpy(c0.lower, &id_lo, 8u); c0.lower_len = 8u; c0.has_lower = true;
    std::memcpy(c0.upper, &id_hi, 8u); c0.upper_len = 8u; c0.has_upper = true;
    // score: field_id 2, three real nulls, no bound recorded.
    ice::ColumnStatEntry& c1 = in.stats.cols[1];
    c1.field_id   = 2;
    c1.null_count = 3;
    c1.has_lower  = false;
    c1.has_upper  = false;

    const uint8_t* buf = nullptr;
    uint64_t len = 0;
    ASSERT_TRUE(ice::manifest_write_avro(&in, 1, 1, 1, kSchemaJson,
                                         sizeof(kSchemaJson) - 1u, 0, &arena,
                                         &buf, &len));

    ice::DataFileRef out[4]{};
    uint32_t n = 0;
    ASSERT_TRUE(ice::manifest_parse_avro(buf, len, &arena, /*default_spec_id=*/0,
                                         out, 4, &n));
    ASSERT_EQ(n, 1u);
    ASSERT_EQ(out[0].stats.n_cols, 2u);

    const ice::ColumnStatEntry* id = nullptr;
    const ice::ColumnStatEntry* score = nullptr;
    for (uint32_t i = 0; i < out[0].stats.n_cols; ++i) {
        if (out[0].stats.cols[i].field_id == 1) id = &out[0].stats.cols[i];
        if (out[0].stats.cols[i].field_id == 2) score = &out[0].stats.cols[i];
    }
    ASSERT_NE(id, nullptr);
    ASSERT_NE(score, nullptr);

    // The certified-zero case: must round-trip as an actual 0, not "absent".
    EXPECT_EQ(id->null_count, 0);
    ASSERT_TRUE(id->has_lower);
    ASSERT_TRUE(id->has_upper);
    ASSERT_EQ(id->lower_len, 8u);
    ASSERT_EQ(id->upper_len, 8u);
    int64_t rt_lo = 0, rt_hi = 0;
    std::memcpy(&rt_lo, id->lower, 8u);
    std::memcpy(&rt_hi, id->upper, 8u);
    EXPECT_EQ(rt_lo, id_lo);
    EXPECT_EQ(rt_hi, id_hi);

    // The nonzero-null, no-bound case.
    EXPECT_EQ(score->null_count, 3);
    EXPECT_FALSE(score->has_lower);
    EXPECT_FALSE(score->has_upper);
}

// n_cols == 0 (a file with no computed stats, e.g. an untouched DataFileRef
// from a caller this session's fix did not reach) must still encode the SIX
// stats maps as null, exactly as before — no empty-array vs. absent-map
// ambiguity introduced for the common "no stats" case.
TEST(IcebergManifestWrite, NoColumnStatsStaysNull) {
    bolt::Arena arena;
    ice::DataFileRef in = make_file("s3://wh/db/t/data/b-1.parquet", 5, 99, 1);
    ASSERT_EQ(in.stats.n_cols, 0u);

    const uint8_t* buf = nullptr;
    uint64_t len = 0;
    ASSERT_TRUE(ice::manifest_write_avro(&in, 1, 1, 1, kSchemaJson,
                                         sizeof(kSchemaJson) - 1u, 0, &arena,
                                         &buf, &len));
    ice::DataFileRef out[4]{};
    uint32_t n = 0;
    ASSERT_TRUE(ice::manifest_parse_avro(buf, len, &arena, 0, out, 4, &n));
    ASSERT_EQ(n, 1u);
    EXPECT_EQ(out[0].stats.n_cols, 0u);
}

TEST(IcebergManifestWrite, ManifestListRoundTrip) {
    bolt::Arena arena;
    ice::ManifestListEntry in[2]{};
    for (int i = 0; i < 2; ++i) {
        in[i].partition_spec_id = 0;
        in[i].content           = ice::ManifestContent::kDataManifest;
        in[i].manifest_length   = 4096 + i;
        in[i].added_snapshot_id = 777;
        in[i].added_files_count = 2 + i;
        std::snprintf(in[i].manifest_path, sizeof(in[i].manifest_path),
                      "s3://wh/db/t/metadata/m-%d.avro", i);
    }

    const uint8_t* buf = nullptr;
    uint64_t len = 0;
    ASSERT_TRUE(ice::manifest_list_write_avro(in, 2, 777, 3, &arena, &buf, &len));

    ice::ManifestListEntry out[8]{};
    uint32_t n = 0;
    ASSERT_TRUE(ice::manifest_list_parse_avro(buf, len, &arena, out, 8, &n));
    ASSERT_EQ(n, 2u);

    EXPECT_STREQ(out[0].manifest_path, "s3://wh/db/t/metadata/m-0.avro");
    EXPECT_EQ(out[0].manifest_length, 4096);
    EXPECT_EQ(out[0].added_files_count, 2);
    EXPECT_EQ(out[0].added_snapshot_id, 777);
    EXPECT_STREQ(out[1].manifest_path, "s3://wh/db/t/metadata/m-1.avro");
    EXPECT_EQ(out[1].manifest_length, 4097);
    EXPECT_EQ(out[1].added_files_count, 3);
}

// G2ICE-49 — added_rows_count (and its four siblings: existing_files_count,
// deleted_files_count, existing_rows_count, deleted_rows_count) used to be
// hardcoded 0 on write regardless of what the caller set, because
// `ManifestListEntry` had no fields to hold anything else. A reader that
// trusts the manifest-list alone (pyiceberg's `inspect.manifests()`, DuckDB,
// a snapshot-summary rollup) sees these WITHOUT opening the manifest, so a
// wrong value here silently corrupts a metadata-only COUNT(*) the same way a
// data file's own `record_count` would.
TEST(IcebergManifestWrite, ManifestListRowAndFileCountsRoundTrip) {
    bolt::Arena arena;
    ice::ManifestListEntry in[1]{};
    in[0].partition_spec_id    = 0;
    in[0].content              = ice::ManifestContent::kDataManifest;
    in[0].manifest_length      = 4096;
    in[0].added_snapshot_id    = 777;
    in[0].added_files_count    = 3;
    in[0].existing_files_count = 5;
    in[0].deleted_files_count  = 2;
    in[0].added_rows_count     = 30000;
    in[0].existing_rows_count  = 50000;
    in[0].deleted_rows_count   = 20000;
    std::snprintf(in[0].manifest_path, sizeof(in[0].manifest_path),
                  "s3://wh/db/t/metadata/m-0.avro");

    const uint8_t* buf = nullptr;
    uint64_t len = 0;
    ASSERT_TRUE(ice::manifest_list_write_avro(in, 1, 777, 3, &arena, &buf, &len));

    ice::ManifestListEntry out[4]{};
    uint32_t n = 0;
    ASSERT_TRUE(ice::manifest_list_parse_avro(buf, len, &arena, out, 4, &n));
    ASSERT_EQ(n, 1u);

    EXPECT_EQ(out[0].added_files_count, 3);
    EXPECT_EQ(out[0].existing_files_count, 5);
    EXPECT_EQ(out[0].deleted_files_count, 2);
    EXPECT_EQ(out[0].added_rows_count, 30000);
    EXPECT_EQ(out[0].existing_rows_count, 50000);
    EXPECT_EQ(out[0].deleted_rows_count, 20000);
}

// Synthesis is a pure function of its inputs: callers cache manifests and
// compare them by content, so a random sync marker or any other nondeterminism
// would break them.
TEST(IcebergManifestWrite, Deterministic) {
    bolt::Arena arena;
    ice::DataFileRef in = make_file("s3://wh/db/t/data/b-1.parquet", 5, 99, 1);

    const uint8_t* a = nullptr; uint64_t alen = 0;
    const uint8_t* b = nullptr; uint64_t blen = 0;
    ASSERT_TRUE(ice::manifest_write_avro(&in, 1, 1, 1, kSchemaJson,
                                         sizeof(kSchemaJson) - 1u, 0, &arena,
                                         &a, &alen));
    ASSERT_TRUE(ice::manifest_write_avro(&in, 1, 1, 1, kSchemaJson,
                                         sizeof(kSchemaJson) - 1u, 0, &arena,
                                         &b, &blen));
    ASSERT_EQ(alen, blen);
    EXPECT_EQ(std::memcmp(a, b, alen), 0);
}

// A partition tuple has no place in the flat unpartitioned schema this writer
// emits. Encoding one anyway would produce a manifest that parses and reports
// the WRONG partition, so it must fail closed.
TEST(IcebergManifestWrite, RejectsPartitionedFile) {
    bolt::Arena arena;
    ice::DataFileRef in = make_file("s3://wh/db/t/data/b-1.parquet", 5, 99, 1);
    in.n_partition = 1;
    in.partition[0].field_id = 0;
    in.partition[0].is_int   = true;
    in.partition[0].i64      = 42;

    const uint8_t* buf = nullptr;
    uint64_t len = 0;
    EXPECT_FALSE(ice::manifest_write_avro(&in, 1, 1, 1, kSchemaJson,
                                          sizeof(kSchemaJson) - 1u, 0, &arena,
                                          &buf, &len));
}

// A container field carries no per-field scalar in the flat model. Before
// G2ICE-135, ANY non-null container value was unconditionally rejected —
// which is why `manifest_write_avro` could never emit real
// null_value_counts/lower_bounds/upper_bounds, and every Iceberg manifest
// bolt wrote had those maps hard-null'd regardless of what `DataFileRef`
// carried. The updated contract (see bolt_avro.h): a non-null container's
// `bytes`/`bytes_len` must hold the CALLER's pre-encoded Avro array body,
// spliced in verbatim; a `bytes == nullptr` non-null value still fails
// (there is nothing to splice) rather than writing garbage or truncating.
TEST(IcebergManifestWrite, NonNullContainerNeedsPayloadBytes) {
    ing::AvroField f[2]{};
    std::snprintf(f[0].name, sizeof(f[0].name), "%s", "n");
    f[0].type = ing::AvroType::kLong;
    std::snprintf(f[1].name, sizeof(f[1].name), "%s", "arr");
    f[1].type     = ing::AvroType::kArray;
    f[1].nullable = true;

    ing::AvroValue row[2]{};
    row[0].num.i64 = 7;
    row[1].is_null = false;          // non-null, but no payload -> rejected

    const char kSchema[] = "{\"type\":\"record\",\"name\":\"r\",\"fields\":[]}";
    uint8_t sync[ing::kAvroSyncLen]{};
    uint8_t dst[4096];
    uint64_t len = sizeof(dst);
    EXPECT_FALSE(ing::avro_write_ex(f, 2, row, 1, kSchema,
                                    sizeof(kSchema) - 1u, nullptr, 0, sync,
                                    dst, &len));

    // The same row with the array NULL encodes fine — proving the rejection
    // above is about the missing payload, not the schema or the field.
    row[1].is_null = true;
    len = sizeof(dst);
    EXPECT_TRUE(ing::avro_write_ex(f, 2, row, 1, kSchema, sizeof(kSchema) - 1u,
                                   nullptr, 0, sync, dst, &len));

    // A non-null array WITH a real payload now succeeds and its bytes land
    // on the wire verbatim: one count-1 block holding the long `9`, then the
    // zero-count terminator — the exact shape `encode_null_count_map` /
    // `encode_bound_map` produce in iceberg_manifest_writer.cpp.
    uint8_t payload[3] = { 0x02 /*count=1*/, 0x12 /*zigzag(9)*/,
                           0x00 /*terminator*/ };
    row[1].is_null   = false;
    row[1].bytes     = payload;
    row[1].bytes_len = sizeof(payload);
    len = sizeof(dst);
    ASSERT_TRUE(ing::avro_write_ex(f, 2, row, 1, kSchema, sizeof(kSchema) - 1u,
                                   nullptr, 0, sync, dst, &len));
    // The payload bytes appear verbatim on the wire (splice, not re-encode).
    EXPECT_NE(std::search(dst, dst + len, payload, payload + sizeof(payload)),
             dst + len);
}
