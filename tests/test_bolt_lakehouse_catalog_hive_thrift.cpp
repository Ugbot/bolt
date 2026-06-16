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
