// Tests for the Hive Metastore Thrift Binary Protocol codec.

#include "bolt/lakehouse/catalog/hive_metastore.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

namespace hm = bolt::lakehouse::hive_metastore;

TEST(HiveThrift, StringFieldRoundTrip) {
    uint8_t buf[256];
    uint32_t c = 0;
    ASSERT_TRUE(hm::hms_encode_string_field(buf, sizeof(buf), &c, 1,
                                            "default", 7));
    ASSERT_TRUE(hm::hms_encode_struct_stop(buf, sizeof(buf), &c));
    const uint32_t written = c;

    // Decode.
    uint32_t r = 0;
    uint8_t type;
    int16_t fid;
    ASSERT_TRUE(hm::hms_decode_field_header(buf, written, &r, &type, &fid));
    EXPECT_EQ(type, 11);   // STRING
    EXPECT_EQ(fid, 1);
    char name[32];
    ASSERT_TRUE(hm::hms_decode_string(buf, written, &r, name, sizeof(name)));
    EXPECT_STREQ(name, "default");
    ASSERT_TRUE(hm::hms_decode_field_header(buf, written, &r, &type, &fid));
    EXPECT_EQ(type, 0);    // STOP
}

TEST(HiveThrift, TwoStringStructRoundTrip) {
    // Database struct: {1: name, 2: description}.
    uint8_t buf[256];
    uint32_t c = 0;
    ASSERT_TRUE(hm::hms_encode_string_field(buf, sizeof(buf), &c, 1, "sales", 5));
    ASSERT_TRUE(hm::hms_encode_string_field(buf, sizeof(buf), &c, 2, "Sales DB", 8));
    ASSERT_TRUE(hm::hms_encode_struct_stop(buf, sizeof(buf), &c));
    const uint32_t written = c;

    uint32_t r = 0;
    uint8_t type;
    int16_t fid;
    char s[64];
    ASSERT_TRUE(hm::hms_decode_field_header(buf, written, &r, &type, &fid));
    EXPECT_EQ(type, 11); EXPECT_EQ(fid, 1);
    ASSERT_TRUE(hm::hms_decode_string(buf, written, &r, s, sizeof(s)));
    EXPECT_STREQ(s, "sales");
    ASSERT_TRUE(hm::hms_decode_field_header(buf, written, &r, &type, &fid));
    EXPECT_EQ(type, 11); EXPECT_EQ(fid, 2);
    ASSERT_TRUE(hm::hms_decode_string(buf, written, &r, s, sizeof(s)));
    EXPECT_STREQ(s, "Sales DB");
    ASSERT_TRUE(hm::hms_decode_field_header(buf, written, &r, &type, &fid));
    EXPECT_EQ(type, 0);
}

TEST(HiveThrift, OverflowRejected) {
    uint8_t buf[4];
    uint32_t c = 0;
    // String field needs 1 (type) + 2 (fid) + 4 (len) + n bytes.
    EXPECT_FALSE(hm::hms_encode_string_field(buf, sizeof(buf), &c, 1,
                                             "hello", 5));
}

// ---- G2CLUS-179: high-level surface framing ----

TEST(HiveThrift, I32AndBoolFieldEncoding) {
    uint8_t buf[64];
    uint32_t c = 0;
    ASSERT_TRUE(hm::hms_encode_i32_field(buf, sizeof(buf), &c, 5, 0x01020304));
    ASSERT_TRUE(hm::hms_encode_bool_field(buf, sizeof(buf), &c, 6, true));
    // i32 field: type=8, fid=0,5, then 4 BE bytes.
    EXPECT_EQ(buf[0], 8u);
    EXPECT_EQ(buf[1], 0u); EXPECT_EQ(buf[2], 5u);
    EXPECT_EQ(buf[3], 0x01u); EXPECT_EQ(buf[4], 0x02u);
    EXPECT_EQ(buf[5], 0x03u); EXPECT_EQ(buf[6], 0x04u);
    // bool field: type=2, fid=0,6, then 1.
    EXPECT_EQ(buf[7], 2u);
    EXPECT_EQ(buf[8], 0u); EXPECT_EQ(buf[9], 6u);
    EXPECT_EQ(buf[10], 1u);
    EXPECT_EQ(c, 11u);
}

TEST(HiveThrift, StringListFieldEncoding) {
    uint8_t buf[128];
    uint32_t c = 0;
    const char* items[2] = { "a", "bc" };
    ASSERT_TRUE(hm::hms_encode_string_list_field(buf, sizeof(buf), &c, 3, items,
                                                 2u));
    // type=15(LIST), fid=0,3, element=11(STRING), size=0,0,0,2
    EXPECT_EQ(buf[0], 15u);
    EXPECT_EQ(buf[1], 0u); EXPECT_EQ(buf[2], 3u);
    EXPECT_EQ(buf[3], 11u);
    EXPECT_EQ(buf[4], 0u); EXPECT_EQ(buf[5], 0u);
    EXPECT_EQ(buf[6], 0u); EXPECT_EQ(buf[7], 2u);
    // first string: len=1, 'a'
    EXPECT_EQ(buf[11], 1u);
    EXPECT_EQ(buf[12], static_cast<uint8_t>('a'));
    // second string: len=2, 'b','c'
    EXPECT_EQ(buf[16], 2u);
    EXPECT_EQ(buf[17], static_cast<uint8_t>('b'));
    EXPECT_EQ(buf[18], static_cast<uint8_t>('c'));
}

TEST(HiveThrift, GetTableArgsFraming) {
    uint8_t buf[256];
    uint32_t n = hm::hms_build_get_table_args(buf, sizeof(buf), "sales",
                                              "orders");
    ASSERT_GT(n, 0u);
    // Decode: field1 string "sales", field2 string "orders", STOP.
    uint32_t r = 0;
    uint8_t type; int16_t fid; char s[32];
    ASSERT_TRUE(hm::hms_decode_field_header(buf, n, &r, &type, &fid));
    EXPECT_EQ(type, 11); EXPECT_EQ(fid, 1);
    ASSERT_TRUE(hm::hms_decode_string(buf, n, &r, s, sizeof(s)));
    EXPECT_STREQ(s, "sales");
    ASSERT_TRUE(hm::hms_decode_field_header(buf, n, &r, &type, &fid));
    EXPECT_EQ(type, 11); EXPECT_EQ(fid, 2);
    ASSERT_TRUE(hm::hms_decode_string(buf, n, &r, s, sizeof(s)));
    EXPECT_STREQ(s, "orders");
    ASSERT_TRUE(hm::hms_decode_field_header(buf, n, &r, &type, &fid));
    EXPECT_EQ(type, 0);   // STOP
}

TEST(HiveThrift, DropPartitionArgsFraming) {
    uint8_t buf[256];
    const char* vals[2] = { "2026", "06" };
    uint32_t n = hm::hms_build_drop_partition_args(buf, sizeof(buf), "db", "t",
                                                   vals, 2u, true);
    ASSERT_GT(n, 0u);
    uint32_t r = 0;
    uint8_t type; int16_t fid; char s[32];
    ASSERT_TRUE(hm::hms_decode_field_header(buf, n, &r, &type, &fid));
    EXPECT_EQ(type, 11); EXPECT_EQ(fid, 1);
    ASSERT_TRUE(hm::hms_decode_string(buf, n, &r, s, sizeof(s)));
    EXPECT_STREQ(s, "db");
    ASSERT_TRUE(hm::hms_decode_field_header(buf, n, &r, &type, &fid));
    EXPECT_EQ(type, 11); EXPECT_EQ(fid, 2);
    ASSERT_TRUE(hm::hms_decode_string(buf, n, &r, s, sizeof(s)));
    EXPECT_STREQ(s, "t");
    // field 3: LIST<STRING> partition values.
    ASSERT_TRUE(hm::hms_decode_field_header(buf, n, &r, &type, &fid));
    EXPECT_EQ(type, 15); EXPECT_EQ(fid, 3);
}

TEST(HiveThrift, GetTableArgsOverflowRejected) {
    uint8_t buf[6];   // too small for two string fields.
    EXPECT_EQ(hm::hms_build_get_table_args(buf, sizeof(buf), "sales", "orders"),
              0u);
}
