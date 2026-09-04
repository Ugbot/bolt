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

// A container field carries no value in the flat model. Before this guard it
// fell through the value switch writing NOTHING — truncating the row and
// misaligning every field after it, with no error.
TEST(IcebergManifestWrite, NonNullContainerRejected) {
    ing::AvroField f[2]{};
    std::snprintf(f[0].name, sizeof(f[0].name), "%s", "n");
    f[0].type = ing::AvroType::kLong;
    std::snprintf(f[1].name, sizeof(f[1].name), "%s", "arr");
    f[1].type     = ing::AvroType::kArray;
    f[1].nullable = true;

    ing::AvroValue row[2]{};
    row[0].num.i64 = 7;
    row[1].is_null = false;          // a non-null array: unencodable

    const char kSchema[] = "{\"type\":\"record\",\"name\":\"r\",\"fields\":[]}";
    uint8_t sync[ing::kAvroSyncLen]{};
    uint8_t dst[4096];
    uint64_t len = sizeof(dst);
    EXPECT_FALSE(ing::avro_write_ex(f, 2, row, 1, kSchema,
                                    sizeof(kSchema) - 1u, nullptr, 0, sync,
                                    dst, &len));

    // The same row with the array NULL encodes fine — proving the rejection
    // above is about the unencodable value, not the schema or the field.
    row[1].is_null = true;
    len = sizeof(dst);
    EXPECT_TRUE(ing::avro_write_ex(f, 2, row, 1, kSchema, sizeof(kSchema) - 1u,
                                   nullptr, 0, sync, dst, &len));
}
