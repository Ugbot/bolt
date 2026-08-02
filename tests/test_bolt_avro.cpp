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

// ---------------------------------------------------------------------------
// Bare-datum surface (avro_parse_schema / avro_decode_datum) — the streaming
// path: a schema as JSON text plus one record body, no OCF container.
// Datums below are hand-built so the expected bytes are independent of our
// own writer (a writer bug cannot mask a decoder bug).
// ---------------------------------------------------------------------------

// Append an Avro zig-zag varint long.
void put_long(std::vector<uint8_t>& v, int64_t x) {
    uint64_t u = (static_cast<uint64_t>(x) << 1) ^
                 static_cast<uint64_t>(x >> 63);
    do {
        uint8_t b = static_cast<uint8_t>(u & 0x7Fu);
        u >>= 7;
        if (u != 0u) b |= 0x80u;
        v.push_back(b);
    } while (u != 0u);
}

void put_str(std::vector<uint8_t>& v, const char* s) {
    const uint64_t n = std::strlen(s);
    put_long(v, static_cast<int64_t>(n));
    v.insert(v.end(), s, s + n);
}

TEST(BoltAvroDatum, ParseSchemaAndDecodePrimitives) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"i\",\"type\":\"int\"},"
        "{\"name\":\"l\",\"type\":\"long\"},"
        "{\"name\":\"f\",\"type\":\"float\"},"
        "{\"name\":\"d\",\"type\":\"double\"},"
        "{\"name\":\"b\",\"type\":\"boolean\"},"
        "{\"name\":\"s\",\"type\":\"string\"}]}";

    AvroField fields[8];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  fields, 8, &n));
    ASSERT_EQ(n, 6u);
    EXPECT_STREQ(fields[0].name, "i");
    EXPECT_EQ(fields[0].type, AvroType::kInt);
    EXPECT_EQ(fields[3].type, AvroType::kDouble);
    EXPECT_STREQ(fields[5].name, "s");
    EXPECT_EQ(fields[5].type, AvroType::kString);
    for (uint32_t k = 0; k < n; ++k) EXPECT_FALSE(fields[k].nullable);

    std::vector<uint8_t> d;
    put_long(d, 42);                                   // i
    put_long(d, -7);                                   // l
    const float fv = 1.5f;  d.insert(d.end(), reinterpret_cast<const uint8_t*>(&fv),
                                     reinterpret_cast<const uint8_t*>(&fv) + 4);
    const double dv = 2.25; d.insert(d.end(), reinterpret_cast<const uint8_t*>(&dv),
                                     reinterpret_cast<const uint8_t*>(&dv) + 8);
    d.push_back(1);                                    // b = true
    put_str(d, "hi");                                  // s

    AvroValue vals[8];
    uint64_t consumed = 0;
    ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), fields, n, vals, &consumed));
    EXPECT_EQ(consumed, d.size());                     // consumed exactly one datum
    EXPECT_EQ(vals[0].num.i64, 42);
    EXPECT_EQ(vals[1].num.i64, -7);
    EXPECT_DOUBLE_EQ(vals[2].num.f64, 1.5);
    EXPECT_DOUBLE_EQ(vals[3].num.f64, 2.25);
    EXPECT_EQ(vals[4].num.i64, 1);
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(vals[5].bytes),
                          vals[5].bytes_len), "hi");
}

// ["null","long"] — null is branch 0 (the common registry convention).
TEST(BoltAvroDatum, NullableNullFirst) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"id\",\"type\":\"long\"},"
        "{\"name\":\"opt\",\"type\":[\"null\",\"long\"]}]}";
    AvroField f[4];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 4, &n));
    ASSERT_EQ(n, 2u);
    EXPECT_FALSE(f[0].nullable);
    ASSERT_TRUE(f[1].nullable);
    EXPECT_EQ(f[1].type, AvroType::kLong);
    EXPECT_EQ(f[1].null_branch, 0u);

    AvroValue v[4];
    {   // present: branch 1 then the value
        std::vector<uint8_t> d;
        put_long(d, 5); put_long(d, 1); put_long(d, 900);
        ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, nullptr));
        EXPECT_EQ(v[0].num.i64, 5);
        EXPECT_FALSE(v[1].is_null);
        EXPECT_EQ(v[1].num.i64, 900);
    }
    {   // absent: branch 0, no value follows
        std::vector<uint8_t> d;
        put_long(d, 6); put_long(d, 0);
        ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, nullptr));
        EXPECT_EQ(v[0].num.i64, 6);
        EXPECT_TRUE(v[1].is_null);
    }
}

// ["long","null"] — null is branch 1. Before null_branch this schema parsed
// the field as NON-nullable, so the union index byte was read as the value and
// every subsequent field shifted: silent corruption, not a clean failure.
TEST(BoltAvroDatum, NullableValueFirstOrdering) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"opt\",\"type\":[\"long\",\"null\"]},"
        "{\"name\":\"tail\",\"type\":\"long\"}]}";
    AvroField f[4];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 4, &n));
    ASSERT_EQ(n, 2u);
    ASSERT_TRUE(f[0].nullable);
    EXPECT_EQ(f[0].type, AvroType::kLong);
    EXPECT_EQ(f[0].null_branch, 1u);        // null is the SECOND branch
    EXPECT_FALSE(f[1].nullable);

    AvroValue v[4];
    {   // present: branch 0 then value; `tail` must still line up
        std::vector<uint8_t> d;
        put_long(d, 0); put_long(d, 77); put_long(d, 123);
        ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, nullptr));
        EXPECT_FALSE(v[0].is_null);
        EXPECT_EQ(v[0].num.i64, 77);
        EXPECT_EQ(v[1].num.i64, 123);
    }
    {   // absent: branch 1, no value
        std::vector<uint8_t> d;
        put_long(d, 1); put_long(d, 456);
        ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, nullptr));
        EXPECT_TRUE(v[0].is_null);
        EXPECT_EQ(v[1].num.i64, 456);
    }
}

TEST(BoltAvroDatum, FailsClosedOnTruncationAndGarbage) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"a\",\"type\":\"long\"},"
        "{\"name\":\"s\",\"type\":\"string\"}]}";
    AvroField f[4];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 4, &n));
    ASSERT_EQ(n, 2u);

    std::vector<uint8_t> full;
    put_long(full, 9);
    put_str(full, "abcdef");

    AvroValue v[4];
    ASSERT_TRUE(avro_decode_datum(full.data(), full.size(), f, n, v, nullptr));

    // Every strict prefix must fail rather than read past the end.
    for (size_t cut = 0; cut < full.size(); ++cut) {
        EXPECT_FALSE(avro_decode_datum(full.data(), cut, f, n, v, nullptr))
            << "prefix of length " << cut << " should fail closed";
    }

    // A string length that overruns the buffer must be rejected, not trusted.
    std::vector<uint8_t> bad;
    put_long(bad, 1);
    put_long(bad, 9999);            // claims 9999 bytes; only a few follow
    bad.push_back('x');
    EXPECT_FALSE(avro_decode_datum(bad.data(), bad.size(), f, n, v, nullptr));

    EXPECT_FALSE(avro_decode_datum(full.data(), full.size(), f, 0, v, nullptr));
}

TEST(BoltAvroDatum, SchemaParseRespectsMaxFields) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"a\",\"type\":\"long\"},"
        "{\"name\":\"b\",\"type\":\"long\"},"
        "{\"name\":\"c\",\"type\":\"long\"}]}";
    AvroField f[2];
    uint32_t n = 0;
    // Cap of 2 must clamp, not overrun f[].
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 2, &n));
    EXPECT_EQ(n, 2u);
    EXPECT_STREQ(f[0].name, "a");
    EXPECT_STREQ(f[1].name, "b");

    uint32_t n2 = 0;
    EXPECT_FALSE(avro_parse_schema(reinterpret_cast<const uint8_t*>("not json"),
                                   8, f, 2, &n2));
}

// ---------------------------------------------------------------------------
// Nested records. Avro encodes a record's fields inline and in order with no
// delimiters, so a nested record's correct FLATTENING is exactly the parent's
// wire sequence — the leaves splice into the positional list, and the decoder
// needs no notion of nesting at all. The names are dotted so a nested leaf
// cannot collide with a top-level field of the same name.
// ---------------------------------------------------------------------------

TEST(BoltAvroDatum, NestedRecordFlattensPositionally) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"Event\",\"fields\":["
        "{\"name\":\"user_id\",\"type\":\"long\"},"
        "{\"name\":\"addr\",\"type\":{\"type\":\"record\",\"name\":\"Addr\","
        "\"fields\":[{\"name\":\"zip\",\"type\":\"long\"}]}},"
        "{\"name\":\"amount\",\"type\":\"double\"}]}";

    AvroField f[8];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 8, &n));
    // Three LEAVES on the wire. The old parser produced four (it committed
    // "addr" from the nested "record" type string, then took the nested
    // record's own "name":"Addr" as a further field), so every datum
    // under-ran and decoded nothing.
    ASSERT_EQ(n, 3u);
    EXPECT_STREQ(f[0].name, "user_id");
    EXPECT_EQ(f[0].type, AvroType::kLong);
    EXPECT_STREQ(f[1].name, "addr.zip");
    EXPECT_EQ(f[1].type, AvroType::kLong);
    EXPECT_STREQ(f[2].name, "amount");
    EXPECT_EQ(f[2].type, AvroType::kDouble);

    std::vector<uint8_t> d;
    put_long(d, 105);                                   // user_id
    put_long(d, 94107);                                 // addr.zip (inline)
    const double dv = 7.5;
    d.insert(d.end(), reinterpret_cast<const uint8_t*>(&dv),
             reinterpret_cast<const uint8_t*>(&dv) + 8);
    AvroValue v[8];
    uint64_t consumed = 0;
    ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, &consumed));
    EXPECT_EQ(consumed, d.size());                      // exact — no drift
    EXPECT_EQ(v[0].num.i64, 105);
    EXPECT_EQ(v[1].num.i64, 94107);
    EXPECT_DOUBLE_EQ(v[2].num.f64, 7.5);
}

// Depth 3, and a scalar AFTER the nested group: if the splice mis-counted by
// even one field, `tail` would decode from the wrong offset.
TEST(BoltAvroDatum, NestedRecordTwoLevelsThenScalar) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"A\",\"fields\":["
        "{\"name\":\"b\",\"type\":{\"type\":\"record\",\"name\":\"B\","
        "\"fields\":["
        "{\"name\":\"c\",\"type\":{\"type\":\"record\",\"name\":\"C\","
        "\"fields\":[{\"name\":\"d\",\"type\":\"long\"},"
        "{\"name\":\"e\",\"type\":\"string\"}]}},"
        "{\"name\":\"f\",\"type\":\"int\"}]}},"
        "{\"name\":\"tail\",\"type\":\"long\"}]}";

    AvroField f[8];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 8, &n));
    ASSERT_EQ(n, 4u);
    EXPECT_STREQ(f[0].name, "b.c.d");
    EXPECT_STREQ(f[1].name, "b.c.e");
    EXPECT_STREQ(f[2].name, "b.f");
    EXPECT_STREQ(f[3].name, "tail");

    std::vector<uint8_t> d;
    put_long(d, 11);            // b.c.d
    put_str(d, "xy");           // b.c.e
    put_long(d, 3);             // b.f  (int)
    put_long(d, 99);            // tail
    AvroValue v[8];
    uint64_t consumed = 0;
    ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, &consumed));
    EXPECT_EQ(consumed, d.size());
    EXPECT_EQ(v[0].num.i64, 11);
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(v[1].bytes),
                          v[1].bytes_len), "xy");
    EXPECT_EQ(v[2].num.i64, 3);
    EXPECT_EQ(v[3].num.i64, 99);     // the alignment proof
}

// A nested leaf and a top-level field may share a bare name; dotting keeps
// them distinct instead of silently producing two fields called "zip".
TEST(BoltAvroDatum, NestedLeafDoesNotCollideWithTopLevel) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"zip\",\"type\":\"long\"},"
        "{\"name\":\"addr\",\"type\":{\"type\":\"record\",\"name\":\"Addr\","
        "\"fields\":[{\"name\":\"zip\",\"type\":\"long\"}]}}]}";
    AvroField f[8];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 8, &n));
    ASSERT_EQ(n, 2u);
    EXPECT_STREQ(f[0].name, "zip");
    EXPECT_STREQ(f[1].name, "addr.zip");
}

// A nullable field INSIDE a nested record still carries its own union index.
TEST(BoltAvroDatum, NestedNullableLeaf) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"n\",\"type\":{\"type\":\"record\",\"name\":\"N\","
        "\"fields\":[{\"name\":\"opt\",\"type\":[\"null\",\"long\"]}]}},"
        "{\"name\":\"tail\",\"type\":\"long\"}]}";
    AvroField f[8];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 8, &n));
    ASSERT_EQ(n, 2u);
    EXPECT_STREQ(f[0].name, "n.opt");
    EXPECT_TRUE(f[0].nullable);
    EXPECT_EQ(f[0].null_branch, 0u);
    EXPECT_STREQ(f[1].name, "tail");

    std::vector<uint8_t> d;
    put_long(d, 0);             // n.opt -> null branch
    put_long(d, 42);            // tail
    AvroValue v[8];
    uint64_t consumed = 0;
    ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, &consumed));
    EXPECT_EQ(consumed, d.size());
    EXPECT_TRUE(v[0].is_null);
    EXPECT_EQ(v[1].num.i64, 42);
}

// Types the flat positional model cannot describe must FAIL the parse, not
// decode as something plausible. The old parser mapped every unrecognised
// type string to "string", so an enum (an int on the wire) was read as a
// length-prefixed byte string — silent corruption of that row and every
// row after it.
TEST(BoltAvroDatum, UnrepresentableTypesFailClosed) {
    const char* cases[] = {
        // enum: an int index on the wire, not a string.
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"s\",\"type\":{\"type\":\"enum\",\"name\":\"S\","
        "\"symbols\":[\"A\",\"B\"]}}]}",
        // fixed: N raw bytes, and the width has nowhere to live yet.
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"h\",\"type\":{\"type\":\"fixed\",\"name\":\"H\","
        "\"size\":16}}]}",
        // A named-type REFERENCE needs a definition table to resolve.
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"x\",\"type\":\"SomeOtherRecord\"}]}",
        // A nullable nested RECORD is not flattenable at all: the whole
        // group is present or absent per row, so no fixed positional list
        // can describe it.
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"a\",\"type\":[\"null\",{\"type\":\"record\","
        "\"name\":\"A\",\"fields\":[{\"name\":\"z\",\"type\":\"long\"}]}]}]}",
    };
    for (const char* json : cases) {
        AvroField f[8];
        uint32_t n = 0;
        EXPECT_FALSE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                       static_cast<uint32_t>(std::strlen(json)),
                                       f, 8, &n))
            << "should have failed closed: " << json;
    }
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
