// Avro OCF writer + reader roundtrip (null codec): a record with int, string,
// and nullable-long fields.

#include "bolt/ingest/bolt_avro.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace bolt::ingest;

struct Collected {
    std::vector<int64_t>     ids;
    std::vector<std::string> names;
    std::vector<bool>        score_null;
    std::vector<int64_t>     scores;
};

bool collect(void* ctx, const AvroValue* v, uint32_t n,
             int64_t /*row*/) noexcept {
    Collected* c = static_cast<Collected*>(ctx);
    if (n != 3) return false;
    c->ids.push_back(v[0].num.i64);
    c->names.emplace_back(reinterpret_cast<const char*>(v[1].bytes),
                          v[1].bytes_len);
    c->score_null.push_back(v[2].is_null);
    c->scores.push_back(v[2].is_null ? 0 : v[2].num.i64);
    return true;
}

TEST(BoltAvro, WriteReadRoundtrip) {
    AvroField fields[3];
    std::memset(fields, 0, sizeof(fields));
    std::strcpy(fields[0].name, "id");    fields[0].type = AvroType::kLong;
    std::strcpy(fields[1].name, "name");  fields[1].type = AvroType::kString;
    std::strcpy(fields[2].name, "score"); fields[2].type = AvroType::kLong;
    fields[2].nullable = true;

    const char* n0 = "alice";
    const char* n1 = "bob";

    AvroValue rows[2 * 3];
    std::memset(rows, 0, sizeof(rows));
    // row 0: id=1, name="alice", score=99
    rows[0].type = AvroType::kLong;   rows[0].num.i64 = 1;
    rows[1].type = AvroType::kString; rows[1].bytes = reinterpret_cast<const uint8_t*>(n0);
    rows[1].bytes_len = 5;
    rows[2].type = AvroType::kLong;   rows[2].num.i64 = 99;
    // row 1: id=2, name="bob", score=null
    rows[3].type = AvroType::kLong;   rows[3].num.i64 = 2;
    rows[4].type = AvroType::kString; rows[4].bytes = reinterpret_cast<const uint8_t*>(n1);
    rows[4].bytes_len = 3;
    rows[5].type = AvroType::kLong;   rows[5].is_null = true;

    uint8_t sync[kAvroSyncLen];
    for (uint32_t i = 0; i < kAvroSyncLen; ++i) sync[i] = static_cast<uint8_t>(i + 1);

    const uint64_t cap = avro_write_max_len(fields, 3, 64, 2);
    std::vector<uint8_t> buf(static_cast<size_t>(cap));
    uint64_t out_len = cap;
    ASSERT_TRUE(avro_write(fields, 3, rows, 2, sync, buf.data(), &out_len));
    ASSERT_GT(out_len, 4u);
    EXPECT_EQ(0, std::memcmp(buf.data(), "Obj\x01", 4));

    bolt::Arena arena;
    Collected got;
    int64_t n_rows = 0;
    ASSERT_TRUE(avro_read(buf.data(), out_len, &arena, &got, collect, &n_rows));
    EXPECT_EQ(n_rows, 2);
    ASSERT_EQ(got.ids.size(), 2u);
    EXPECT_EQ(got.ids[0], 1);
    EXPECT_EQ(got.names[0], "alice");
    EXPECT_FALSE(got.score_null[0]);
    EXPECT_EQ(got.scores[0], 99);
    EXPECT_EQ(got.ids[1], 2);
    EXPECT_EQ(got.names[1], "bob");
    EXPECT_TRUE(got.score_null[1]);
}

TEST(BoltAvro, HeaderParse) {
    AvroField fields[2];
    std::memset(fields, 0, sizeof(fields));
    std::strcpy(fields[0].name, "x"); fields[0].type = AvroType::kInt;
    std::strcpy(fields[1].name, "y"); fields[1].type = AvroType::kString;

    AvroValue rows[1 * 2];
    std::memset(rows, 0, sizeof(rows));
    rows[0].type = AvroType::kInt; rows[0].num.i64 = 7;
    const char* s = "hi";
    rows[1].type = AvroType::kString;
    rows[1].bytes = reinterpret_cast<const uint8_t*>(s); rows[1].bytes_len = 2;

    uint8_t sync[kAvroSyncLen];
    std::memset(sync, 0xAB, sizeof(sync));
    const uint64_t cap = avro_write_max_len(fields, 2, 16, 1);
    std::vector<uint8_t> buf(static_cast<size_t>(cap));
    uint64_t out_len = cap;
    ASSERT_TRUE(avro_write(fields, 2, rows, 1, sync, buf.data(), &out_len));

    bolt::Arena arena;
    AvroHeader hdr;
    uint64_t body = 0;
    ASSERT_TRUE(avro_read_header(buf.data(), out_len, &arena, &hdr, &body));
    EXPECT_EQ(hdr.n_fields, 2u);
    EXPECT_STREQ(hdr.field[0].name, "x");
    EXPECT_STREQ(hdr.field[1].name, "y");
    EXPECT_EQ(hdr.codec, AvroCodec::kNull);
    EXPECT_EQ(0, std::memcmp(hdr.sync, sync, kAvroSyncLen));
}

TEST(BoltAvro, RejectsBadMagic) {
    const uint8_t bad[8] = {'X', 'X', 'X', 0x01, 0, 0, 0, 0};
    bolt::Arena arena;
    AvroHeader hdr;
    uint64_t body = 0;
    EXPECT_FALSE(avro_read_header(bad, sizeof(bad), &arena, &hdr, &body));
}

}  // namespace
