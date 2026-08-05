// Avro OCF writer + reader roundtrip (null codec): a record with int, string,
// and nullable-long fields.

#include "bolt/ingest/bolt_avro.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
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

// ---------------------------------------------------------------------------
// Arrays and maps. Neither yields a value — both decode as NULL — but both are
// WALKED so the fields after them stay aligned. On the wire each is
//   block* 0,  block := count:long [size:long IF count<0] item*
// The negative-count form carries the block's byte size, so it can be skipped
// without knowing the element type; the positive-count form cannot.
// ---------------------------------------------------------------------------

// count > 0, elements walked one at a time.
void put_arr_counted(std::vector<uint8_t>& v,
                     const std::vector<int64_t>& items) {
    put_long(v, static_cast<int64_t>(items.size()));
    for (int64_t x : items) put_long(v, x);
    put_long(v, 0);                                     // terminator
}

// count < 0 followed by the block's BYTE SIZE.
void put_arr_sized(std::vector<uint8_t>& v, const std::vector<int64_t>& items) {
    std::vector<uint8_t> body;
    for (int64_t x : items) put_long(body, x);
    put_long(v, -static_cast<int64_t>(items.size()));
    put_long(v, static_cast<int64_t>(body.size()));
    v.insert(v.end(), body.begin(), body.end());
    put_long(v, 0);
}

TEST(BoltAvroDatum, ArrayOfPrimitiveSkippedAndScalarStaysAligned) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"head\",\"type\":\"long\"},"
        "{\"name\":\"tags\",\"type\":{\"type\":\"array\",\"items\":\"long\"}},"
        "{\"name\":\"tail\",\"type\":\"long\"}]}";
    AvroField f[8];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 8, &n));
    ASSERT_EQ(n, 3u);
    EXPECT_EQ(f[1].type, AvroType::kArray);
    EXPECT_EQ(f[1].item_type, static_cast<uint8_t>(AvroType::kLong));
    EXPECT_STREQ(f[2].name, "tail");

    // Both block forms, and an EMPTY array, must all land `tail` correctly —
    // this is the alignment proof that matters most.
    for (int form = 0; form < 3; ++form) {
        std::vector<uint8_t> d;
        put_long(d, 1);
        if (form == 0)      put_arr_counted(d, {7, 8, 9});
        else if (form == 1) put_arr_sized(d, {7, 8, 9});
        else                put_long(d, 0);             // empty: just the 0
        put_long(d, 4242);
        AvroValue v[8];
        uint64_t consumed = 0;
        ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, &consumed))
            << "form " << form;
        EXPECT_EQ(consumed, d.size()) << "form " << form;
        EXPECT_EQ(v[0].num.i64, 1)    << "form " << form;
        EXPECT_TRUE(v[1].is_null)     << "form " << form;   // no scalar value
        EXPECT_EQ(v[2].num.i64, 4242) << "form " << form;   // aligned
    }
}

// A multi-BLOCK array (writers chunk large arrays) must be walked to the
// terminating zero, not just through the first block.
TEST(BoltAvroDatum, ArrayMultiBlock) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"a\",\"type\":{\"type\":\"array\",\"items\":\"long\"}},"
        "{\"name\":\"tail\",\"type\":\"long\"}]}";
    AvroField f[4];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 4, &n));
    std::vector<uint8_t> d;
    put_long(d, 2); put_long(d, 1); put_long(d, 2);      // block 1: 2 items
    put_long(d, 1); put_long(d, 3);                      // block 2: 1 item
    put_long(d, 0);                                      // terminator
    put_long(d, 77);
    AvroValue v[4];
    uint64_t consumed = 0;
    ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, &consumed));
    EXPECT_EQ(consumed, d.size());
    EXPECT_EQ(v[1].num.i64, 77);
}

TEST(BoltAvroDatum, ArrayOfStringSkipped) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"a\",\"type\":{\"type\":\"array\",\"items\":\"string\"}},"
        "{\"name\":\"tail\",\"type\":\"long\"}]}";
    AvroField f[4];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 4, &n));
    EXPECT_EQ(f[0].item_type, static_cast<uint8_t>(AvroType::kString));
    std::vector<uint8_t> d;
    put_long(d, 2); put_str(d, "aa"); put_str(d, "bbbb");
    put_long(d, 0);
    put_long(d, 5);
    AvroValue v[4];
    uint64_t consumed = 0;
    ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, &consumed));
    EXPECT_EQ(consumed, d.size());
    EXPECT_EQ(v[1].num.i64, 5);
}

// A map element is a key STRING then the value — skipping only the value
// would desync immediately.
TEST(BoltAvroDatum, MapSkippedKeyAndValue) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"m\",\"type\":{\"type\":\"map\",\"values\":\"long\"}},"
        "{\"name\":\"tail\",\"type\":\"long\"}]}";
    AvroField f[4];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 4, &n));
    ASSERT_EQ(n, 2u);
    EXPECT_EQ(f[0].type, AvroType::kMap);
    EXPECT_EQ(f[0].item_type, static_cast<uint8_t>(AvroType::kLong));
    std::vector<uint8_t> d;
    put_long(d, 2);
    put_str(d, "k1"); put_long(d, 10);
    put_str(d, "k2"); put_long(d, 20);
    put_long(d, 0);
    put_long(d, 31337);
    AvroValue v[4];
    uint64_t consumed = 0;
    ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, &consumed));
    EXPECT_EQ(consumed, d.size());
    EXPECT_TRUE(v[0].is_null);
    EXPECT_EQ(v[1].num.i64, 31337);
}

// An array whose element is a RECORD is MODELLED: the element's own fields are
// spliced in after the container so the decoder can compute an element's width
// by walking them. That is the only way to advance past a POSITIVE-count block
// (no byte size to jump), which is the form Avro's Java writer emits -- and
// therefore the shape Iceberg's statistics maps actually take on disk.
TEST(BoltAvroDatum, ArrayOfRecordModelledAndSkips) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"parts\",\"type\":{\"type\":\"array\",\"items\":"
        "{\"type\":\"record\",\"name\":\"P\",\"fields\":["
        "{\"name\":\"k\",\"type\":\"int\"},{\"name\":\"v\",\"type\":\"long\"}]}}},"
        "{\"name\":\"tail\",\"type\":\"long\"}]}";
    AvroField f[8];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 8, &n));
    // The element's fields do NOT join the parent's positional sequence -- they
    // repeat per element -- so they follow the container as descriptors.
    ASSERT_EQ(n, 4u);
    EXPECT_STREQ(f[0].name, "parts");
    EXPECT_EQ(f[0].type, AvroType::kArray);
    EXPECT_EQ(f[0].item_type, kAvroItemRecord);
    EXPECT_EQ(f[0].elem_fields, 2u);
    EXPECT_STREQ(f[1].name, "parts[].k");
    EXPECT_EQ(f[1].type, AvroType::kInt);
    EXPECT_STREQ(f[2].name, "parts[].v");
    EXPECT_EQ(f[2].type, AvroType::kLong);
    EXPECT_STREQ(f[3].name, "tail");
    EXPECT_EQ(f[3].elem_fields, 0u);

    // POSITIVE count, no byte size: exactly the case that used to fail. The
    // element width comes from walking {k:int, v:long}.
    std::vector<uint8_t> d;
    put_long(d, 2);
    put_long(d, 1); put_long(d, 100);
    put_long(d, 2); put_long(d, 200);
    put_long(d, 0);
    put_long(d, 909);
    AvroValue v[8];
    uint64_t consumed = 0;
    ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, &consumed));
    EXPECT_EQ(consumed, d.size());
    // `tail` landing on 909 is the alignment proof: a wrong element width
    // would put it on any other byte.
    EXPECT_EQ(v[3].num.i64, 909);
    EXPECT_TRUE(v[0].is_null);          // the container itself carries no scalar
    EXPECT_TRUE(v[1].is_null);          // element descriptors publish as null
    EXPECT_TRUE(v[2].is_null);

    // The size-prefixed form still works (a different writer's encoding).
    std::vector<uint8_t> body;
    put_long(body, 1); put_long(body, 100);
    put_long(body, 2); put_long(body, 200);
    std::vector<uint8_t> sz;
    put_long(sz, -2);
    put_long(sz, static_cast<int64_t>(body.size()));
    sz.insert(sz.end(), body.begin(), body.end());
    put_long(sz, 0);
    put_long(sz, 909);
    ASSERT_TRUE(avro_decode_datum(sz.data(), sz.size(), f, n, v, nullptr));
    EXPECT_EQ(v[3].num.i64, 909);

    // An EMPTY array is just the terminator.
    std::vector<uint8_t> empty;
    put_long(empty, 0);
    put_long(empty, 77);
    ASSERT_TRUE(avro_decode_datum(empty.data(), empty.size(), f, n, v, nullptr));
    EXPECT_EQ(v[3].num.i64, 77);

    // Several blocks then the terminator.
    std::vector<uint8_t> multi;
    put_long(multi, 1); put_long(multi, 5); put_long(multi, 50);
    put_long(multi, 2); put_long(multi, 6); put_long(multi, 60);
                        put_long(multi, 7); put_long(multi, 70);
    put_long(multi, 0);
    put_long(multi, 4242);
    ASSERT_TRUE(avro_decode_datum(multi.data(), multi.size(), f, n, v, nullptr));
    EXPECT_EQ(v[3].num.i64, 4242);

    // Truncated mid-element must still fail closed, at every cut.
    for (size_t cut = 1; cut < d.size(); ++cut) {
        EXPECT_FALSE(avro_decode_datum(d.data(), cut, f, n, v, nullptr))
            << "cut=" << cut;
    }
}

// Two modelled containers with scalars on both sides and between them: the
// per-container element buffer must be reset and re-attached correctly, and
// every scalar must land on its own bytes.
TEST(BoltAvroDatum, TwoArrayOfRecordFieldsInterleaved) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"a\",\"type\":\"long\"},"
        "{\"name\":\"lo\",\"type\":{\"type\":\"array\",\"items\":"
        "{\"type\":\"record\",\"name\":\"kv1\",\"fields\":["
        "{\"name\":\"k\",\"type\":\"int\"},{\"name\":\"v\",\"type\":\"bytes\"}]}}},"
        "{\"name\":\"b\",\"type\":\"long\"},"
        "{\"name\":\"hi\",\"type\":{\"type\":\"array\",\"items\":"
        "{\"type\":\"record\",\"name\":\"kv2\",\"fields\":["
        "{\"name\":\"k\",\"type\":\"int\"},{\"name\":\"v\",\"type\":\"double\"},"
        "{\"name\":\"w\",\"type\":\"boolean\"}]}}},"
        "{\"name\":\"c\",\"type\":\"long\"}]}";
    AvroField f[16];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 16, &n));
    ASSERT_EQ(n, 10u);                 // a, lo, lo[].k, lo[].v, b, hi, +3, c
    EXPECT_STREQ(f[1].name, "lo");
    EXPECT_EQ(f[1].elem_fields, 2u);
    EXPECT_STREQ(f[2].name, "lo[].k");
    EXPECT_STREQ(f[3].name, "lo[].v");
    EXPECT_STREQ(f[4].name, "b");
    EXPECT_STREQ(f[5].name, "hi");
    EXPECT_EQ(f[5].elem_fields, 3u);
    EXPECT_STREQ(f[8].name, "hi[].w");
    EXPECT_STREQ(f[9].name, "c");

    std::vector<uint8_t> d;
    put_long(d, 11);                                    // a
    put_long(d, 1); put_long(d, 1);                     // lo: 1 elem, k=1
    put_long(d, 2); d.push_back('x'); d.push_back('y'); // v = 2 bytes
    put_long(d, 0);
    put_long(d, 22);                                    // b
    put_long(d, 1); put_long(d, 9);                     // hi: 1 elem, k=9
    for (int i = 0; i < 8; ++i) d.push_back(0);         // v = double
    d.push_back(1);                                     // w = true
    put_long(d, 0);
    put_long(d, 33);                                    // c
    AvroValue v[16];
    uint64_t consumed = 0;
    ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, &consumed));
    EXPECT_EQ(consumed, d.size());
    EXPECT_EQ(v[0].num.i64, 11);
    EXPECT_EQ(v[4].num.i64, 22);
    EXPECT_EQ(v[9].num.i64, 33);
}

// A nullable field INSIDE the element: the branch index is part of the
// element's width, so it must be consumed per element, per branch.
TEST(BoltAvroDatum, ArrayOfRecordNullableElementField) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"p\",\"type\":{\"type\":\"array\",\"items\":"
        "{\"type\":\"record\",\"name\":\"kv\",\"fields\":["
        "{\"name\":\"k\",\"type\":\"int\"},"
        "{\"name\":\"v\",\"type\":[\"null\",\"long\"]}]}}},"
        "{\"name\":\"tail\",\"type\":\"long\"}]}";
    AvroField f[8];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 8, &n));
    ASSERT_EQ(n, 4u);
    EXPECT_TRUE(f[2].nullable);
    EXPECT_EQ(f[2].null_branch, 0u);

    std::vector<uint8_t> d;
    put_long(d, 2);
    put_long(d, 1); put_long(d, 0);                 // k=1, v = null branch
    put_long(d, 2); put_long(d, 1); put_long(d, 5); // k=2, v = 5
    put_long(d, 0);
    put_long(d, 909);
    AvroValue v[8];
    uint64_t consumed = 0;
    ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, &consumed));
    EXPECT_EQ(consumed, d.size());
    EXPECT_EQ(v[3].num.i64, 909);
}

// map<string, record>: the key string precedes each element record.
TEST(BoltAvroDatum, MapOfRecordModelled) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"m\",\"type\":{\"type\":\"map\",\"values\":"
        "{\"type\":\"record\",\"name\":\"kv\",\"fields\":["
        "{\"name\":\"a\",\"type\":\"int\"},{\"name\":\"b\",\"type\":\"int\"}]}}},"
        "{\"name\":\"tail\",\"type\":\"long\"}]}";
    AvroField f[8];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 8, &n));
    ASSERT_EQ(n, 4u);
    EXPECT_EQ(f[0].type, AvroType::kMap);
    EXPECT_EQ(f[0].item_type, kAvroItemRecord);
    EXPECT_EQ(f[0].elem_fields, 2u);

    std::vector<uint8_t> d;
    put_long(d, 1);
    put_long(d, 2); d.push_back('h'); d.push_back('i');   // key "hi"
    put_long(d, 4); put_long(d, 5);                        // {a:4, b:5}
    put_long(d, 0);
    put_long(d, 909);
    AvroValue v[8];
    uint64_t consumed = 0;
    ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, &consumed));
    EXPECT_EQ(consumed, d.size());
    EXPECT_EQ(v[3].num.i64, 909);
}

// "items" BEFORE "type" is equally legal JSON. The element must survive it --
// resetting item_type when the "array" string arrived silently un-modelled
// every schema written in this order.
TEST(BoltAvroDatum, ArrayOfRecordItemsKeyBeforeTypeKey) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"p\",\"type\":{\"items\":"
        "{\"type\":\"record\",\"name\":\"kv\",\"fields\":["
        "{\"name\":\"k\",\"type\":\"int\"},{\"name\":\"v\",\"type\":\"long\"}]},"
        "\"type\":\"array\"}},"
        "{\"name\":\"tail\",\"type\":\"long\"}]}";
    AvroField f[8];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 8, &n));
    ASSERT_EQ(n, 4u);
    EXPECT_EQ(f[0].item_type, kAvroItemRecord);
    EXPECT_EQ(f[0].elem_fields, 2u);
    std::vector<uint8_t> d;
    put_long(d, 1); put_long(d, 1); put_long(d, 100);
    put_long(d, 0);
    put_long(d, 909);
    AvroValue v[8];
    ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, nullptr));
    EXPECT_EQ(v[3].num.i64, 909);
}

// An element whose OWN field is a container has no computable width from a
// flat descriptor list. Fail the PARSE -- never model a width we cannot walk.
TEST(BoltAvroDatum, ArrayOfRecordWithNestedContainerFailsParse) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"p\",\"type\":{\"type\":\"array\",\"items\":"
        "{\"type\":\"record\",\"name\":\"kv\",\"fields\":["
        "{\"name\":\"k\",\"type\":\"int\"},"
        "{\"name\":\"v\",\"type\":{\"type\":\"array\",\"items\":\"long\"}}]}}},"
        "{\"name\":\"tail\",\"type\":\"long\"}]}";
    AvroField f[8];
    uint32_t n = 0;
    EXPECT_FALSE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                   static_cast<uint32_t>(std::strlen(json)),
                                   f, 8, &n));
}

// An element wider than kAvroMaxElemFields fails the parse. Truncating it
// would under-state the element width and mis-align everything after.
TEST(BoltAvroDatum, ArrayOfRecordTooWideElementFailsParse) {
    std::string json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"p\",\"type\":{\"type\":\"array\",\"items\":"
        "{\"type\":\"record\",\"name\":\"kv\",\"fields\":[";
    for (uint32_t i = 0; i < kAvroMaxElemFields + 1u; ++i) {
        if (i) json += ",";
        json += "{\"name\":\"f" + std::to_string(i) + "\",\"type\":\"int\"}";
    }
    json += "]}}},{\"name\":\"tail\",\"type\":\"long\"}]}";
    AvroField f[kAvroMaxFields];
    uint32_t n = 0;
    EXPECT_FALSE(avro_parse_schema(
        reinterpret_cast<const uint8_t*>(json.data()),
        static_cast<uint32_t>(json.size()), f, kAvroMaxFields, &n));
}

// A truncated container must fail rather than run off the buffer.
TEST(BoltAvroDatum, ContainerTruncationFailsClosed) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"a\",\"type\":{\"type\":\"array\",\"items\":\"long\"}},"
        "{\"name\":\"tail\",\"type\":\"long\"}]}";
    AvroField f[4];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 4, &n));
    std::vector<uint8_t> full;
    put_long(full, 3); put_long(full, 1); put_long(full, 2); put_long(full, 3);
    put_long(full, 0);
    put_long(full, 6);
    AvroValue v[4];
    ASSERT_TRUE(avro_decode_datum(full.data(), full.size(), f, n, v, nullptr));
    for (size_t cut = 1; cut < full.size(); ++cut) {
        EXPECT_FALSE(avro_decode_datum(full.data(), cut, f, n, v, nullptr))
            << "cut " << cut;
    }
    // A size-prefixed block claiming more bytes than remain must fail too.
    std::vector<uint8_t> lie;
    put_long(lie, -1);
    put_long(lie, 9999);
    put_long(lie, 42);
    EXPECT_FALSE(avro_decode_datum(lie.data(), lie.size(), f, n, v, nullptr));
}

// `union{null, array<T>}` — how Iceberg declares EVERY optional repeated field
// (lower_bounds, split_offsets, partitions, ...). This used to fail closed,
// which made a real Iceberg manifest unparseable; it is now modelled as a
// nullable container. (G2FEAT-76)
TEST(BoltAvroDatum, NullableContainerIsModelled) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"a\",\"type\":[\"null\","
        "{\"type\":\"array\",\"items\":\"long\"}]},"
        "{\"name\":\"z\",\"type\":\"long\"}]}";
    AvroField f[4];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 4, &n));
    ASSERT_EQ(n, 2u);
    EXPECT_EQ(f[0].type, AvroType::kArray);
    EXPECT_TRUE(f[0].nullable);
    EXPECT_EQ(f[0].null_branch, 0u);
    EXPECT_EQ(f[0].item_type, static_cast<uint8_t>(AvroType::kLong));
    EXPECT_EQ(f[1].type, AvroType::kLong);

    // Branch 1 (present): one block of two longs, terminator, then `z`.
    std::vector<uint8_t> present;
    put_long(present, 1);            // union branch -> the array
    put_long(present, 2);            // block count
    put_long(present, 7);
    put_long(present, 8);
    put_long(present, 0);            // terminator
    put_long(present, 99);           // z
    AvroValue v[4];
    ASSERT_TRUE(avro_decode_datum(present.data(), present.size(), f, n, v,
                                  nullptr));
    EXPECT_TRUE(v[0].is_null);       // a container publishes no scalar
    EXPECT_EQ(v[1].num.i64, 99);     // the field AFTER it stays aligned

    // Branch 0 (absent): the array is not on the wire at all.
    std::vector<uint8_t> absent;
    put_long(absent, 0);             // union branch -> null
    put_long(absent, 123);           // z
    ASSERT_TRUE(avro_decode_datum(absent.data(), absent.size(), f, n, v,
                                  nullptr));
    EXPECT_TRUE(v[0].is_null);
    EXPECT_EQ(v[1].num.i64, 123);
}

// The [T,"null"] ordering must resolve the null branch to 1, not 0 — a complex
// branch occupies a branch index just like a named one.
TEST(BoltAvroDatum, NullableContainerValueFirstOrdering) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"a\",\"type\":[{\"type\":\"array\",\"items\":\"long\"},"
        "\"null\"]},"
        "{\"name\":\"z\",\"type\":\"long\"}]}";
    AvroField f[4];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 4, &n));
    ASSERT_EQ(n, 2u);
    EXPECT_TRUE(f[0].nullable);
    EXPECT_EQ(f[0].null_branch, 1u);          // NOT 0
    std::vector<uint8_t> absent;
    put_long(absent, 1);                      // branch 1 == null here
    put_long(absent, 55);
    AvroValue v[4];
    ASSERT_TRUE(avro_decode_datum(absent.data(), absent.size(), f, n, v,
                                  nullptr));
    EXPECT_EQ(v[1].num.i64, 55);
}

// An ABSENT nullable array<record> must still step over its element
// descriptors. Skipping them only on the present path mis-aligned every field
// after a null container — the exact shape of Iceberg's `lower_bounds`.
TEST(BoltAvroDatum, NullableArrayOfRecordAbsentStaysAligned) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"m\",\"type\":[\"null\",{\"type\":\"array\",\"items\":"
        "{\"type\":\"record\",\"name\":\"kv\",\"fields\":["
        "{\"name\":\"key\",\"type\":\"int\"},"
        "{\"name\":\"value\",\"type\":\"long\"}]}}]},"
        "{\"name\":\"tail\",\"type\":\"long\"}]}";
    AvroField f[8];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 8, &n));
    ASSERT_EQ(n, 4u);                         // m, m[].key, m[].value, tail
    EXPECT_EQ(f[0].elem_fields, 2u);
    EXPECT_TRUE(f[0].nullable);
    EXPECT_EQ(f[3].type, AvroType::kLong);

    AvroValue v[8];
    std::vector<uint8_t> absent;
    put_long(absent, 0);                      // null branch
    put_long(absent, 4242);                   // tail
    ASSERT_TRUE(avro_decode_datum(absent.data(), absent.size(), f, n, v,
                                  nullptr));
    EXPECT_EQ(v[3].num.i64, 4242);            // would be garbage if misaligned

    std::vector<uint8_t> present;
    put_long(present, 1);                     // the array branch
    put_long(present, 1);                     // one element
    put_long(present, 3);                     // key
    put_long(present, 77);                    // value
    put_long(present, 0);                     // terminator
    put_long(present, 4243);                  // tail
    ASSERT_TRUE(avro_decode_datum(present.data(), present.size(), f, n, v,
                                  nullptr));
    EXPECT_EQ(v[3].num.i64, 4243);
}

// avro_decode_datum takes a CALLER-supplied field table, so an out-of-range
// elem_fields must be rejected rather than overflowing the decoder's fixed
// element buffer. (Hand-built table — the schema parser cannot produce this.)
TEST(BoltAvroDatum, OversizedElemFieldsRejectedNotOverflowed) {
    AvroField f[4];
    std::memset(f, 0, sizeof(f));
    std::strcpy(f[0].name, "a");
    f[0].type        = AvroType::kArray;
    f[0].item_type   = kAvroItemRecord;
    f[0].elem_fields = static_cast<uint16_t>(kAvroMaxElemFields + 1u);
    std::strcpy(f[1].name, "k");
    f[1].type = AvroType::kLong;
    std::strcpy(f[2].name, "z");
    f[2].type = AvroType::kLong;

    std::vector<uint8_t> wire;
    put_long(wire, 1);
    put_long(wire, 1);
    put_long(wire, 0);
    AvroValue v[4];
    EXPECT_FALSE(avro_decode_datum(wire.data(), wire.size(), f, 3, v, nullptr));
}

// Still genuinely unsupported, and must stay fail-closed: a union carrying a
// container AND a named type is a real multi-type union, which one positional
// slot cannot describe.
TEST(BoltAvroDatum, UnionOfContainerAndScalarFailsClosed) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"a\",\"type\":[\"long\","
        "{\"type\":\"array\",\"items\":\"long\"}]}]}";
    AvroField f[4];
    uint32_t n = 0;
    EXPECT_FALSE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                   static_cast<uint32_t>(std::strlen(json)),
                                   f, 4, &n));
}

// A container INSIDE a nested record: the dotted leaf is still produced, and
// the group after it stays aligned.
TEST(BoltAvroDatum, ContainerInsideNestedRecord) {
    const char* json =
        "{\"type\":\"record\",\"name\":\"R\",\"fields\":["
        "{\"name\":\"g\",\"type\":{\"type\":\"record\",\"name\":\"G\","
        "\"fields\":[{\"name\":\"a\",\"type\":{\"type\":\"array\","
        "\"items\":\"long\"}},"
        "{\"name\":\"z\",\"type\":\"long\"}]}},"
        "{\"name\":\"tail\",\"type\":\"long\"}]}";
    AvroField f[8];
    uint32_t n = 0;
    ASSERT_TRUE(avro_parse_schema(reinterpret_cast<const uint8_t*>(json),
                                  static_cast<uint32_t>(std::strlen(json)),
                                  f, 8, &n));
    ASSERT_EQ(n, 3u);
    EXPECT_STREQ(f[0].name, "g.a");
    EXPECT_EQ(f[0].type, AvroType::kArray);
    EXPECT_STREQ(f[1].name, "g.z");
    EXPECT_STREQ(f[2].name, "tail");

    std::vector<uint8_t> d;
    put_arr_counted(d, {5, 6});
    put_long(d, 11);
    put_long(d, 22);
    AvroValue v[8];
    uint64_t consumed = 0;
    ASSERT_TRUE(avro_decode_datum(d.data(), d.size(), f, n, v, &consumed));
    EXPECT_EQ(consumed, d.size());
    EXPECT_TRUE(v[0].is_null);
    EXPECT_EQ(v[1].num.i64, 11);
    EXPECT_EQ(v[2].num.i64, 22);
}

// ---------------------------------------------------------------------------
// Real Iceberg-manifest-shaped files, written by the STOCK Apache Avro JAVA
// stack (GenericDatumWriter + DataFileWriter) — the same writer Iceberg's
// ManifestWriter goes through. Nothing in these bytes shares code with bolt's
// writer, so a bolt writer bug cannot mask a bolt reader bug.
//
// MEASURED (not assumed) from golden_avro_array_of_record.avro: that writer
// emits arrays as a POSITIVE count with NO byte size —
//     02          status = 1
//     04          array count = +2   <-- positive, no size follows
//     02 c8 01    {key:1, value:100}
//     04 90 03    {key:2, value:200}
//     00          terminator
//     9a 0e       tail = 909
// The negative-count-plus-size form comes from BlockingBinaryEncoder, which is
// not what Iceberg uses. So an array whose ELEMENT is a record cannot be
// skipped from a real manifest: there is no size to jump, and the element
// width is unknowable from one item_type byte. That case fails closed.
// ---------------------------------------------------------------------------

std::string data_path(const char* name) {
    return std::string(BOLT_TEST_DATA_DIR) + "/" + name;
}

bool read_file(const char* name, std::vector<uint8_t>* out) {
    std::FILE* f = std::fopen(data_path(name).c_str(), "rb");
    if (f == nullptr) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out->resize(static_cast<size_t>(n));
    const size_t got = std::fread(out->data(), 1, static_cast<size_t>(n), f);
    std::fclose(f);
    return got == static_cast<size_t>(n);
}

struct ManifestRows {
    std::vector<int64_t>     status;
    std::vector<bool>        snap_null;
    std::vector<int64_t>     snap;
    std::vector<std::string> path;
    std::vector<int64_t>     record_count;
    std::vector<int64_t>     sort_order_id;
    std::vector<int64_t>     seq;
    uint32_t                 width = 0;
};

bool collect_manifest(void* ctx, const AvroValue* v, uint32_t n,
                      int64_t /*row*/) noexcept {
    ManifestRows* m = static_cast<ManifestRows*>(ctx);
    m->width = n;
    if (n != 10) return false;
    m->status.push_back(v[0].num.i64);
    m->snap_null.push_back(v[1].is_null);
    m->snap.push_back(v[1].is_null ? 0 : v[1].num.i64);
    m->path.emplace_back(reinterpret_cast<const char*>(v[2].bytes),
                         v[2].bytes_len);
    m->record_count.push_back(v[4].num.i64);
    m->sort_order_id.push_back(v[8].num.i64);
    m->seq.push_back(v[9].num.i64);
    return true;
}

// The shape a manifest actually has: a nested data_file record carrying an
// array<long> (split_offsets) and a map<string,long> (value_counts), with
// scalars on BOTH sides of them, plus a nullable field. Every one of the three
// mechanisms — splice, array walk, map walk — has to be right simultaneously
// or the trailing columns land on the wrong bytes.
TEST(BoltAvroIceberg, RealJavaWrittenManifestShapeReads) {
    std::vector<uint8_t> buf;
    ASSERT_TRUE(read_file("golden_avro_manifest.avro", &buf))
        << "fixture missing";

    bolt::Arena arena;
    AvroHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    uint64_t body = 0;
    ASSERT_TRUE(avro_read_header(buf.data(), buf.size(), &arena, &hdr, &body));

    // 4 top-level fields flatten to 10 leaves: status, snapshot_id,
    // data_file.{file_path, file_format, record_count, file_size_in_bytes,
    // split_offsets, value_counts, sort_order_id}, sequence_number.
    ASSERT_EQ(hdr.n_fields, 10u);
    EXPECT_STREQ(hdr.field[0].name, "status");
    EXPECT_STREQ(hdr.field[1].name, "snapshot_id");
    EXPECT_TRUE(hdr.field[1].nullable);
    EXPECT_STREQ(hdr.field[2].name, "data_file.file_path");
    EXPECT_STREQ(hdr.field[6].name, "data_file.split_offsets");
    EXPECT_EQ(hdr.field[6].type, AvroType::kArray);
    EXPECT_EQ(hdr.field[6].item_type, static_cast<uint8_t>(AvroType::kLong));
    EXPECT_STREQ(hdr.field[7].name, "data_file.value_counts");
    EXPECT_EQ(hdr.field[7].type, AvroType::kMap);
    EXPECT_STREQ(hdr.field[8].name, "data_file.sort_order_id");
    EXPECT_STREQ(hdr.field[9].name, "sequence_number");

    ManifestRows m;
    int64_t rows = 0;
    ASSERT_TRUE(avro_read(buf.data(), buf.size(), &arena, &m,
                          collect_manifest, &rows));
    ASSERT_EQ(rows, 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(m.status[static_cast<size_t>(i)], 1 + i);
        // Alternating null exercises the union branch index inside the row.
        EXPECT_EQ(m.snap_null[static_cast<size_t>(i)], (i % 2) == 0);
        if ((i % 2) != 0) EXPECT_EQ(m.snap[static_cast<size_t>(i)], 900 + i);
        EXPECT_EQ(m.path[static_cast<size_t>(i)],
                  "s3://b/data/part-" + std::to_string(i) + ".parquet");
        EXPECT_EQ(m.record_count[static_cast<size_t>(i)], 1000 + i);
        // These two sit AFTER the array and the map. They are the proof the
        // container walks consumed exactly the right number of bytes.
        EXPECT_EQ(m.sort_order_id[static_cast<size_t>(i)], 7 + i);
        EXPECT_EQ(m.seq[static_cast<size_t>(i)], 42 + i);
    }
}

// The array<record> shape a REAL manifest uses, read from a file written by
// the stock Apache Avro JAVA stack -- the writer that emits a POSITIVE count
// with NO byte size. This test previously pinned the failure; it now pins the
// fix, against the exact same bytes.
struct AorRows {
    std::vector<int64_t> status;
    std::vector<int64_t> tail;
    std::vector<bool>    elem_null;
    uint32_t             width = 0;
};

bool collect_aor(void* ctx, const AvroValue* v, uint32_t n,
                 int64_t /*row*/) noexcept {
    AorRows* m = static_cast<AorRows*>(ctx);
    m->width = n;
    if (n != 5) return false;
    m->status.push_back(v[0].num.i64);
    // v[1] is the container, v[2]/v[3] its element descriptors: all null.
    m->elem_null.push_back(v[1].is_null && v[2].is_null && v[3].is_null);
    m->tail.push_back(v[4].num.i64);
    return true;
}

TEST(BoltAvroIceberg, RealJavaArrayOfRecordReads) {
    std::vector<uint8_t> buf;
    ASSERT_TRUE(read_file("golden_avro_array_of_record.avro", &buf))
        << "fixture missing";

    bolt::Arena arena;
    AvroHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    uint64_t body = 0;
    ASSERT_TRUE(avro_read_header(buf.data(), buf.size(), &arena, &hdr, &body));
    // status, parts, parts[].key, parts[].value, tail.
    ASSERT_EQ(hdr.n_fields, 5u);
    EXPECT_STREQ(hdr.field[1].name, "parts");
    EXPECT_EQ(hdr.field[1].type, AvroType::kArray);
    EXPECT_EQ(hdr.field[1].item_type, kAvroItemRecord);
    EXPECT_EQ(hdr.field[1].elem_fields, 2u);
    EXPECT_STREQ(hdr.field[2].name, "parts[].key");
    EXPECT_STREQ(hdr.field[3].name, "parts[].value");
    EXPECT_STREQ(hdr.field[4].name, "tail");

    AorRows m;
    int64_t rows = 0;
    ASSERT_TRUE(avro_read(buf.data(), buf.size(), &arena, &m,
                          collect_aor, &rows));
    ASSERT_EQ(rows, 1);
    EXPECT_EQ(m.status[0], 1);
    // 909 is the alignment proof: it sits after a 2-element array<record>
    // that carries no byte size, so only a correct element walk reaches it.
    EXPECT_EQ(m.tail[0], 909);
    EXPECT_TRUE(m.elem_null[0]);
}

namespace {
struct ElemCapture {
    uint32_t n;
    uint32_t field_index[16];
    int64_t  key[16];
    int64_t  value[16];
};
bool capture_elem(void* c, uint32_t fi, int64_t, int64_t ei,
                  const AvroValue* vals, uint32_t nv) noexcept {
    ElemCapture* e = static_cast<ElemCapture*>(c);
    if (e->n >= 16u || nv < 2u) return true;
    e->field_index[e->n] = fi;
    e->key[e->n]   = vals[0].num.i64;
    e->value[e->n] = vals[1].num.i64;
    ++e->n;
    (void)ei;
    return true;
}
}  // namespace

// Per-ELEMENT delivery over the same Java-written array<record> fixture. Until
// this landed the container walked correctly but published nothing, so a
// repeated field's values were unreachable to any caller. (G2FEAT-76)
TEST(BoltAvroIceberg, RealJavaArrayOfRecordDeliversElementValues) {
    std::vector<uint8_t> buf;
    ASSERT_TRUE(read_file("golden_avro_array_of_record.avro", &buf))
        << "fixture missing";
    bolt::Arena arena;
    AorRows m;
    ElemCapture e;
    std::memset(&e, 0, sizeof(e));
    int64_t rows = 0;
    ASSERT_TRUE(avro_read_ex(buf.data(), buf.size(), &arena, &m, collect_aor,
                             nullptr, &rows));
    ASSERT_EQ(rows, 1);
    // A null on_elem must behave exactly like avro_read (one walker, no drift).
    ASSERT_EQ(e.n, 0u);

    // The element sink is a separate ctx from the row sink in real callers, but
    // reusing one struct here keeps the fixture-shape assertions together.
    struct Both { AorRows* r; ElemCapture* e; } both{&m, &e};
    auto row_fn = [](void* c, const AvroValue* v, uint32_t n,
                     int64_t ri) noexcept -> bool {
        return collect_aor(static_cast<Both*>(c)->r, v, n, ri);
    };
    auto elem_fn = [](void* c, uint32_t fi, int64_t ri, int64_t ei,
                      const AvroValue* v, uint32_t nv) noexcept -> bool {
        return capture_elem(static_cast<Both*>(c)->e, fi, ri, ei, v, nv);
    };
    std::memset(&m, 0, sizeof(m));
    rows = 0;
    ASSERT_TRUE(avro_read_ex(buf.data(), buf.size(), &arena, &both, row_fn,
                             elem_fn, &rows));
    ASSERT_EQ(rows, 1);
    // The fixture's `parts` array holds 2 elements — previously invisible.
    ASSERT_EQ(e.n, 2u);
    EXPECT_EQ(e.field_index[0], 1u);          // the `parts` container field
    EXPECT_EQ(e.field_index[1], 1u);
    // Keys/values are distinct, so a decoder emitting the same element twice
    // (or re-reading one) cannot pass.
    EXPECT_NE(e.key[0], e.key[1]);
    // Alignment still holds with delivery switched on.
    EXPECT_EQ(m.tail[0], 909);
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

// Modelling an element record needed one more field on AvroField (a single
// item_type byte cannot say "record, N fields wide"). AvroField is multiplied
// by kAvroMaxFields inside AvroHeader, so the cost is pinned here rather than
// left to drift: 132 -> 134 bytes per field, 33,816 -> 34,328 per header.
TEST(BoltAvro, AbiSizesPinned) {
    static_assert(sizeof(AvroField) == 134, "AvroField ABI changed");
    static_assert(sizeof(AvroHeader) == 34328, "AvroHeader ABI changed");
    std::printf("sizeof(AvroField)=%zu sizeof(AvroHeader)=%zu\n",
                sizeof(AvroField), sizeof(AvroHeader));
    EXPECT_EQ(offsetof(AvroField, elem_fields), 132u);
}
