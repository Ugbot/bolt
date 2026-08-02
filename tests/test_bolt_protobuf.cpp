// test_bolt_protobuf.cpp — dynamic protobuf decoder (bolt_protobuf.h).
//
// Every message here is HAND-ENCODED: the test builds the exact wire bytes, so
// each assertion pins a specific byte sequence to a specific decoded value.
// That is deliberate — a round-trip through our own encoder would only prove
// self-consistency, not conformance to the protobuf wire format.

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/ingest/bolt_protobuf.h"

using namespace bolt::ingest;

namespace {

// ---- wire encoders (test-side; mirrors the spec, not our decoder) ----------

void put_varint(std::vector<uint8_t>& v, uint64_t u) {
    do {
        uint8_t b = static_cast<uint8_t>(u & 0x7Fu);
        u >>= 7;
        if (u != 0u) b |= 0x80u;
        v.push_back(b);
    } while (u != 0u);
}

void put_tag(std::vector<uint8_t>& v, int32_t field, PbWireType w) {
    put_varint(v, (static_cast<uint64_t>(field) << 3) |
                  static_cast<uint64_t>(w));
}

void put_varint_field(std::vector<uint8_t>& v, int32_t f, uint64_t u) {
    put_tag(v, f, PbWireType::kVarint);
    put_varint(v, u);
}

void put_zigzag_field(std::vector<uint8_t>& v, int32_t f, int64_t i) {
    put_tag(v, f, PbWireType::kVarint);
    put_varint(v, (static_cast<uint64_t>(i) << 1) ^
                  static_cast<uint64_t>(i >> 63));
}

void put_bytes_field(std::vector<uint8_t>& v, int32_t f,
                     const uint8_t* p, size_t n) {
    put_tag(v, f, PbWireType::kLenDelim);
    put_varint(v, n);
    v.insert(v.end(), p, p + n);
}

void put_string_field(std::vector<uint8_t>& v, int32_t f, const char* s) {
    put_bytes_field(v, f, reinterpret_cast<const uint8_t*>(s), std::strlen(s));
}

void put_fixed32_field(std::vector<uint8_t>& v, int32_t f, uint32_t raw) {
    put_tag(v, f, PbWireType::kFixed32);
    for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>(raw >> (8 * i)));
}

void put_fixed64_field(std::vector<uint8_t>& v, int32_t f, uint64_t raw) {
    put_tag(v, f, PbWireType::kFixed64);
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<uint8_t>(raw >> (8 * i)));
}

void put_float_field(std::vector<uint8_t>& v, int32_t f, float x) {
    uint32_t raw = 0; std::memcpy(&raw, &x, 4);
    put_fixed32_field(v, f, raw);
}

void put_double_field(std::vector<uint8_t>& v, int32_t f, double x) {
    uint64_t raw = 0; std::memcpy(&raw, &x, 8);
    put_fixed64_field(v, f, raw);
}

// ---- descriptor builders (descriptor.proto is itself protobuf) ------------

std::vector<uint8_t> make_field(const char* name, int32_t number,
                                PbType type, PbLabel label,
                                const char* type_name = nullptr) {
    std::vector<uint8_t> f;
    put_string_field(f, 1, name);                                  // name
    put_varint_field(f, 3, static_cast<uint64_t>(number));         // number
    put_varint_field(f, 4, static_cast<uint64_t>(label));          // label
    put_varint_field(f, 5, static_cast<uint64_t>(type));           // type
    if (type_name != nullptr) put_string_field(f, 6, type_name);   // type_name
    return f;
}

PbFieldDesc* find_field(PbMessageDesc* m, const char* name) {
    for (uint32_t i = 0; i < m->n_fields; ++i) {
        if (std::strcmp(m->field[i].name, name) == 0) return &m->field[i];
    }
    return nullptr;
}

}  // namespace

// ===========================================================================
// Layer 1 — wire reader
// ===========================================================================

TEST(BoltProtobufWire, VarintRoundTrip) {
    const uint64_t cases[] = {0u, 1u, 127u, 128u, 300u, 16383u, 16384u,
                              0x7FFFFFFFull, 0xFFFFFFFFull,
                              0xFFFFFFFFFFFFFFFFull};
    for (uint64_t want : cases) {
        std::vector<uint8_t> v;
        put_varint(v, want);
        uint64_t off = 0, got = 0;
        ASSERT_TRUE(pb_read_varint(v.data(), v.size(), &off, &got)) << want;
        EXPECT_EQ(got, want);
        EXPECT_EQ(off, v.size());
    }
}

TEST(BoltProtobufWire, VarintTruncatedFailsClosed) {
    // 0x80 sets the continuation bit but the buffer ends: must fail, not read on.
    const uint8_t buf[] = {0x80u};
    uint64_t off = 0, out = 0;
    EXPECT_FALSE(pb_read_varint(buf, sizeof(buf), &off, &out));
}

TEST(BoltProtobufWire, VarintOverlongFailsClosed) {
    // 11 continuation bytes — past the 10-byte 64-bit ceiling. Must terminate.
    const uint8_t buf[] = {0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u,
                           0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x01u};
    uint64_t off = 0, out = 0;
    EXPECT_FALSE(pb_read_varint(buf, sizeof(buf), &off, &out));
    EXPECT_LE(off, sizeof(buf));                 // never ran off the end
}

TEST(BoltProtobufWire, ZigZag) {
    EXPECT_EQ(pb_zigzag_decode(0u), 0);
    EXPECT_EQ(pb_zigzag_decode(1u), -1);
    EXPECT_EQ(pb_zigzag_decode(2u), 1);
    EXPECT_EQ(pb_zigzag_decode(3u), -2);
    EXPECT_EQ(pb_zigzag_decode(4294967294u), 2147483647);
}

TEST(BoltProtobufWire, TagAndSkipEveryWireType) {
    std::vector<uint8_t> v;
    put_varint_field(v, 1, 300u);
    put_fixed64_field(v, 2, 0x1122334455667788ull);
    put_string_field(v, 3, "hello");
    put_fixed32_field(v, 4, 0xDEADBEEFu);

    uint64_t off = 0;
    const int32_t    want_f[] = {1, 2, 3, 4};
    const PbWireType want_w[] = {PbWireType::kVarint, PbWireType::kFixed64,
                                 PbWireType::kLenDelim, PbWireType::kFixed32};
    for (int i = 0; i < 4; ++i) {
        int32_t f = 0; PbWireType w = PbWireType::kVarint;
        ASSERT_TRUE(pb_read_tag(v.data(), v.size(), &off, &f, &w)) << i;
        EXPECT_EQ(f, want_f[i]);
        EXPECT_EQ(static_cast<int>(w), static_cast<int>(want_w[i]));
        ASSERT_TRUE(pb_skip_field(v.data(), v.size(), &off, w)) << i;
    }
    EXPECT_EQ(off, v.size());                    // walked the whole message
}

TEST(BoltProtobufWire, GroupWireTypeRejected) {
    uint64_t off = 0;
    const uint8_t buf[] = {0x00u};
    EXPECT_FALSE(pb_skip_field(buf, sizeof(buf), &off, PbWireType::kStartGroup));
    off = 0;
    EXPECT_FALSE(pb_skip_field(buf, sizeof(buf), &off, PbWireType::kEndGroup));
}

TEST(BoltProtobufWire, FieldNumberZeroRejected) {
    const uint8_t buf[] = {0x00u};               // tag 0 ⇒ field 0, invalid
    uint64_t off = 0; int32_t f = 0; PbWireType w = PbWireType::kVarint;
    EXPECT_FALSE(pb_read_tag(buf, sizeof(buf), &off, &f, &w));
}

// ===========================================================================
// Layer 3 — decode against a descriptor
// ===========================================================================

namespace {

// A message with one field of every scalar type we support.
PbMessageDesc* make_scalar_desc(PbMessageDesc* m) {
    std::memset(m, 0, sizeof(*m));
    std::strcpy(m->full_name, "T");
    struct { const char* n; int32_t num; PbType t; } spec[] = {
        {"f_double",   1,  PbType::kDouble},
        {"f_float",    2,  PbType::kFloat},
        {"f_int64",    3,  PbType::kInt64},
        {"f_uint64",   4,  PbType::kUint64},
        {"f_int32",    5,  PbType::kInt32},
        {"f_fixed64",  6,  PbType::kFixed64},
        {"f_fixed32",  7,  PbType::kFixed32},
        {"f_bool",     8,  PbType::kBool},
        {"f_string",   9,  PbType::kString},
        {"f_bytes",    12, PbType::kBytes},
        {"f_uint32",   13, PbType::kUint32},
        {"f_enum",     14, PbType::kEnum},
        {"f_sfixed32", 15, PbType::kSfixed32},
        {"f_sfixed64", 16, PbType::kSfixed64},
        {"f_sint32",   17, PbType::kSint32},
        {"f_sint64",   18, PbType::kSint64},
    };
    for (const auto& s : spec) {
        PbFieldDesc* fd = &m->field[m->n_fields++];
        std::memset(fd, 0, sizeof(*fd));
        std::strcpy(fd->name, s.n);
        fd->number = s.num;
        fd->type = s.t;
        fd->label = PbLabel::kOptional;
        fd->nullable = true;
        fd->msg_index = -1;
    }
    return m;
}

}  // namespace

TEST(BoltProtobufDecode, EveryScalarType) {
    PbMessageDesc desc;
    make_scalar_desc(&desc);

    std::vector<uint8_t> v;
    put_double_field(v, 1, 3.5);
    put_float_field(v, 2, -1.25f);
    put_varint_field(v, 3, 1234567890123ull);
    put_varint_field(v, 4, 18446744073709551615ull);
    put_varint_field(v, 5, static_cast<uint64_t>(static_cast<uint32_t>(-7)));
    put_fixed64_field(v, 6, 0xAABBCCDDEEFF0011ull);
    put_fixed32_field(v, 7, 0x01020304u);
    put_varint_field(v, 8, 1u);
    put_string_field(v, 9, "hello");
    put_bytes_field(v, 12, reinterpret_cast<const uint8_t*>("\x01\x02\x03"), 3);
    put_varint_field(v, 13, 4000000000u);
    put_varint_field(v, 14, 2u);
    put_fixed32_field(v, 15, static_cast<uint32_t>(-5));
    put_fixed64_field(v, 16, static_cast<uint64_t>(-6));
    put_zigzag_field(v, 17, -12345);
    put_zigzag_field(v, 18, -9876543210ll);

    PbValue out[32];
    ASSERT_TRUE(pb_decode_message(v.data(), v.size(), &desc, nullptr, out, 32));

    EXPECT_DOUBLE_EQ(out[0].num.f64, 3.5);
    EXPECT_DOUBLE_EQ(out[1].num.f64, -1.25);
    EXPECT_EQ(out[2].num.i64, 1234567890123ll);
    EXPECT_EQ(out[3].num.u64, 18446744073709551615ull);
    EXPECT_EQ(out[4].num.i64, -7);
    EXPECT_EQ(out[5].num.u64, 0xAABBCCDDEEFF0011ull);
    EXPECT_EQ(out[6].num.u64, 0x01020304u);
    EXPECT_EQ(out[7].num.i64, 1);
    ASSERT_EQ(out[8].bytes_len, 5u);
    EXPECT_EQ(std::memcmp(out[8].bytes, "hello", 5), 0);
    ASSERT_EQ(out[9].bytes_len, 3u);
    EXPECT_EQ(out[9].bytes[2], 0x03u);
    EXPECT_EQ(out[10].num.u64, 4000000000u);
    EXPECT_EQ(out[11].num.i64, 2);
    EXPECT_EQ(out[12].num.i64, -5);
    EXPECT_EQ(out[13].num.i64, -6);
    EXPECT_EQ(out[14].num.i64, -12345);
    EXPECT_EQ(out[15].num.i64, -9876543210ll);
    for (int i = 0; i < 16; ++i) EXPECT_FALSE(out[i].is_null) << i;
}

TEST(BoltProtobufDecode, AbsentFieldIsNull) {
    PbMessageDesc desc;
    make_scalar_desc(&desc);
    std::vector<uint8_t> v;
    put_varint_field(v, 3, 42u);                 // only f_int64

    PbValue out[32];
    ASSERT_TRUE(pb_decode_message(v.data(), v.size(), &desc, nullptr, out, 32));
    EXPECT_FALSE(out[2].is_null);
    EXPECT_EQ(out[2].num.i64, 42);
    EXPECT_TRUE(out[0].is_null);                 // f_double absent
    EXPECT_EQ(out[0].rep_count, 0u);
    EXPECT_TRUE(out[8].is_null);                 // f_string absent
}

TEST(BoltProtobufDecode, FieldsOutOfOrder) {
    PbMessageDesc desc;
    make_scalar_desc(&desc);
    std::vector<uint8_t> v;
    put_string_field(v, 9, "last-declared-first-on-wire");
    put_varint_field(v, 5, 11u);
    put_double_field(v, 1, 2.5);
    put_varint_field(v, 3, 99u);

    PbValue out[32];
    ASSERT_TRUE(pb_decode_message(v.data(), v.size(), &desc, nullptr, out, 32));
    EXPECT_DOUBLE_EQ(out[0].num.f64, 2.5);       // matched by NUMBER, not order
    EXPECT_EQ(out[2].num.i64, 99);
    EXPECT_EQ(out[4].num.i64, 11);
    EXPECT_EQ(out[8].bytes_len, 27u);
}

TEST(BoltProtobufDecode, UnknownFieldSkippedWithoutCorruption) {
    PbMessageDesc desc;
    make_scalar_desc(&desc);
    std::vector<uint8_t> v;
    put_varint_field(v, 3, 7u);                       // known
    put_string_field(v, 900, "field from a newer schema");  // unknown, len-delim
    put_varint_field(v, 901, 123456u);                // unknown, varint
    put_fixed64_field(v, 902, 0xFFull);               // unknown, fixed64
    put_fixed32_field(v, 903, 0xAAu);                 // unknown, fixed32
    put_string_field(v, 9, "still-here");             // known, AFTER unknowns

    PbValue out[32];
    ASSERT_TRUE(pb_decode_message(v.data(), v.size(), &desc, nullptr, out, 32));
    EXPECT_EQ(out[2].num.i64, 7);
    ASSERT_EQ(out[8].bytes_len, 10u);
    EXPECT_EQ(std::memcmp(out[8].bytes, "still-here", 10), 0);
}

TEST(BoltProtobufDecode, RepeatedUnpacked) {
    PbMessageDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    std::strcpy(desc.full_name, "R");
    PbFieldDesc* fd = &desc.field[desc.n_fields++];
    std::strcpy(fd->name, "vals");
    fd->number = 1; fd->type = PbType::kInt64;
    fd->label = PbLabel::kRepeated; fd->repeated = true; fd->msg_index = -1;

    std::vector<uint8_t> v;
    for (uint64_t x : {10u, 20u, 30u}) put_varint_field(v, 1, x);

    PbValue out[4];
    ASSERT_TRUE(pb_decode_message(v.data(), v.size(), &desc, nullptr, out, 4));
    EXPECT_EQ(out[0].rep_count, 3u);
    EXPECT_EQ(out[0].num.i64, 10);               // FIRST occurrence, documented

    PbValue all[8];
    uint32_t n = 0;
    ASSERT_TRUE(pb_decode_repeated_field(v.data(), v.size(), 1, PbType::kInt64,
                                         all, 8, &n));
    ASSERT_EQ(n, 3u);
    EXPECT_EQ(all[0].num.i64, 10);
    EXPECT_EQ(all[1].num.i64, 20);
    EXPECT_EQ(all[2].num.i64, 30);
}

TEST(BoltProtobufDecode, RepeatedPacked) {
    PbMessageDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    std::strcpy(desc.full_name, "P");
    PbFieldDesc* fd = &desc.field[desc.n_fields++];
    std::strcpy(fd->name, "vals");
    fd->number = 1; fd->type = PbType::kInt64;
    fd->label = PbLabel::kRepeated; fd->repeated = true; fd->msg_index = -1;

    // packed = one length-delimited field holding a run of varints (proto3 default)
    std::vector<uint8_t> payload;
    for (uint64_t x : {1u, 300u, 70000u, 5u}) put_varint(payload, x);
    std::vector<uint8_t> v;
    put_bytes_field(v, 1, payload.data(), payload.size());

    PbValue out[4];
    ASSERT_TRUE(pb_decode_message(v.data(), v.size(), &desc, nullptr, out, 4));
    EXPECT_EQ(out[0].rep_count, 4u);
    EXPECT_EQ(out[0].num.i64, 1);
    EXPECT_FALSE(out[0].is_null);

    PbValue all[8];
    uint32_t n = 0;
    ASSERT_TRUE(pb_decode_repeated_field(v.data(), v.size(), 1, PbType::kInt64,
                                         all, 8, &n));
    ASSERT_EQ(n, 4u);
    EXPECT_EQ(all[0].num.i64, 1);
    EXPECT_EQ(all[1].num.i64, 300);
    EXPECT_EQ(all[2].num.i64, 70000);
    EXPECT_EQ(all[3].num.i64, 5);
}

TEST(BoltProtobufDecode, PackedFixed32Run) {
    PbMessageDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    PbFieldDesc* fd = &desc.field[desc.n_fields++];
    fd->number = 2; fd->type = PbType::kFloat;
    fd->label = PbLabel::kRepeated; fd->repeated = true; fd->msg_index = -1;

    std::vector<uint8_t> payload;
    for (float f : {1.5f, 2.5f, -3.5f}) {
        uint32_t raw = 0; std::memcpy(&raw, &f, 4);
        for (int i = 0; i < 4; ++i) payload.push_back(static_cast<uint8_t>(raw >> (8 * i)));
    }
    std::vector<uint8_t> v;
    put_bytes_field(v, 2, payload.data(), payload.size());

    PbValue all[8];
    uint32_t n = 0;
    ASSERT_TRUE(pb_decode_repeated_field(v.data(), v.size(), 2, PbType::kFloat,
                                         all, 8, &n));
    ASSERT_EQ(n, 3u);
    EXPECT_DOUBLE_EQ(all[0].num.f64, 1.5);
    EXPECT_DOUBLE_EQ(all[2].num.f64, -3.5);
}

TEST(BoltProtobufDecode, ScalarLastOneWins) {
    PbMessageDesc desc;
    make_scalar_desc(&desc);
    std::vector<uint8_t> v;
    put_varint_field(v, 3, 1u);
    put_varint_field(v, 3, 2u);
    put_varint_field(v, 3, 3u);                  // protobuf: last wins

    PbValue out[32];
    ASSERT_TRUE(pb_decode_message(v.data(), v.size(), &desc, nullptr, out, 32));
    EXPECT_EQ(out[2].num.i64, 3);
    EXPECT_EQ(out[2].rep_count, 3u);
}

TEST(BoltProtobufDecode, TruncatedAndGarbageFailClosed) {
    PbMessageDesc desc;
    make_scalar_desc(&desc);
    std::vector<uint8_t> good;
    put_string_field(good, 9, "abcdefghij");
    PbValue out[32];

    // Every proper prefix must fail or decode cleanly — never read OOB.
    for (size_t cut = 1; cut < good.size(); ++cut) {
        (void)pb_decode_message(good.data(), cut, &desc, nullptr, out, 32);
    }
    // A length prefix that overruns the buffer must be rejected.
    std::vector<uint8_t> bad;
    put_tag(bad, 9, PbWireType::kLenDelim);
    put_varint(bad, 9999u);                      // claims 9999 bytes, has 0
    EXPECT_FALSE(pb_decode_message(bad.data(), bad.size(), &desc, nullptr, out, 32));

    // Garbage bytes: must terminate (no infinite loop) and not crash.
    const uint8_t junk[] = {0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu,
                            0xFFu, 0xFFu, 0xFFu, 0xFFu, 0xFFu};
    EXPECT_FALSE(pb_decode_message(junk, sizeof(junk), &desc, nullptr, out, 32));
}

// ===========================================================================
// Layer 2 — descriptor parsing (a FileDescriptorSet is itself protobuf)
// ===========================================================================

TEST(BoltProtobufDescriptor, ParseNestedAndResolve) {
    // demo.User { id:1 int64, email:2 string, tags:3 repeated string,
    //             addr:4 .demo.User.Addr }  nested Addr { zip:1 int32 }
    std::vector<uint8_t> addr;
    put_string_field(addr, 1, "Addr");
    {
        auto f = make_field("zip", 1, PbType::kInt32, PbLabel::kOptional);
        put_bytes_field(addr, 2, f.data(), f.size());
    }

    std::vector<uint8_t> user;
    put_string_field(user, 1, "User");
    for (auto& f : {make_field("id", 1, PbType::kInt64, PbLabel::kOptional),
                    make_field("email", 2, PbType::kString, PbLabel::kOptional),
                    make_field("tags", 3, PbType::kString, PbLabel::kRepeated),
                    make_field("addr", 4, PbType::kMessage, PbLabel::kOptional,
                               ".demo.User.Addr")}) {
        put_bytes_field(user, 2, f.data(), f.size());
    }
    put_bytes_field(user, 3, addr.data(), addr.size());       // nested_type

    std::vector<uint8_t> file;
    put_string_field(file, 1, "demo.proto");
    put_string_field(file, 2, "demo");                        // package
    put_bytes_field(file, 4, user.data(), user.size());       // message_type

    std::vector<uint8_t> fds;
    put_bytes_field(fds, 1, file.data(), file.size());        // FileDescriptorSet.file

    // PbDescriptorSet is ~0.5 MB — heap, never a stack local (see header).
    PbDescriptorSet* set = new PbDescriptorSet();
    ASSERT_TRUE(pb_parse_descriptor_set(fds.data(), fds.size(), nullptr, set));

    ASSERT_EQ(set->n_msgs, 2u);                  // User + hoisted User.Addr
    const int32_t iu = pb_find_message(set, "demo.User");
    const int32_t ia = pb_find_message(set, ".demo.User.Addr");   // leading dot ok
    ASSERT_GE(iu, 0);
    ASSERT_GE(ia, 0);
    EXPECT_STREQ(set->msg[iu].full_name, "demo.User");
    EXPECT_STREQ(set->msg[ia].full_name, "demo.User.Addr");

    PbMessageDesc* u = &set->msg[iu];
    ASSERT_EQ(u->n_fields, 4u);
    PbFieldDesc* id = find_field(u, "id");
    ASSERT_NE(id, nullptr);
    EXPECT_EQ(id->number, 1);
    EXPECT_EQ(static_cast<int>(id->type), static_cast<int>(PbType::kInt64));
    EXPECT_FALSE(id->repeated);

    PbFieldDesc* tags = find_field(u, "tags");
    ASSERT_NE(tags, nullptr);
    EXPECT_TRUE(tags->repeated);
    EXPECT_EQ(static_cast<int>(tags->type), static_cast<int>(PbType::kString));

    PbFieldDesc* ad = find_field(u, "addr");
    ASSERT_NE(ad, nullptr);
    EXPECT_STREQ(ad->type_name, ".demo.User.Addr");
    EXPECT_EQ(ad->msg_index, ia);                // resolved to the nested type

    EXPECT_EQ(set->msg[ia].n_fields, 1u);
    EXPECT_STREQ(set->msg[ia].field[0].name, "zip");

    // --- and now decode a real message against the PARSED descriptor -------
    std::vector<uint8_t> addr_msg;
    put_varint_field(addr_msg, 1, 94107u);                    // zip

    std::vector<uint8_t> msg;
    put_varint_field(msg, 1, 4242u);                          // id
    put_string_field(msg, 2, "a@b.c");                        // email
    put_string_field(msg, 3, "x");                            // tags[0]
    put_string_field(msg, 3, "y");                            // tags[1]
    put_bytes_field(msg, 4, addr_msg.data(), addr_msg.size()); // addr

    PbValue out[8];
    ASSERT_TRUE(pb_decode_message(msg.data(), msg.size(), u, set, out, 8));
    EXPECT_EQ(out[0].num.i64, 4242);
    ASSERT_EQ(out[1].bytes_len, 5u);
    EXPECT_EQ(std::memcmp(out[1].bytes, "a@b.c", 5), 0);
    EXPECT_EQ(out[2].rep_count, 2u);                          // tags repeated
    EXPECT_EQ(out[3].msg_index, ia);                          // nested resolved

    // Nested message decodes recursively from its raw slice.
    PbValue nested[4];
    ASSERT_FALSE(out[3].is_null);
    ASSERT_TRUE(pb_decode_message(out[3].bytes, out[3].bytes_len,
                                  &set->msg[ia], set, nested, 4));
    EXPECT_EQ(nested[0].num.i64, 94107);

    delete set;
}

TEST(BoltProtobufDescriptor, MalformedSetFailsClosed) {
    const uint8_t junk[] = {0x0Au, 0xFFu, 0xFFu, 0xFFu};   // len prefix overruns
    PbDescriptorSet* set = new PbDescriptorSet();
    EXPECT_FALSE(pb_parse_descriptor_set(junk, sizeof(junk), nullptr, set));
    delete set;
}

// ===========================================================================
// Confluent Schema-Registry framing
// ===========================================================================

TEST(BoltProtobufConfluent, ShorthandZeroIndex) {
    // [0x00][id BE32][0x00 = shorthand for index array [0]][payload]
    std::vector<uint8_t> v = {0x00u, 0x00u, 0x00u, 0x01u, 0x2Cu, 0x00u};
    const uint8_t body[] = {0x08u, 0x2Au};                 // field1 varint 42
    v.insert(v.end(), body, body + sizeof(body));

    uint32_t id = 0, n_idx = 0;
    int32_t idx[kPbMaxMsgIndexes] = {0};
    const uint8_t* payload = nullptr;
    uint64_t payload_len = 0;
    ASSERT_TRUE(pb_confluent_parse(v.data(), v.size(), &id, idx,
                                   kPbMaxMsgIndexes, &n_idx,
                                   &payload, &payload_len));
    EXPECT_EQ(id, 300u);                                   // 0x0000012C
    EXPECT_EQ(n_idx, 1u);
    EXPECT_EQ(idx[0], 0);
    ASSERT_EQ(payload_len, 2u);
    EXPECT_EQ(payload[0], 0x08u);
}

TEST(BoltProtobufConfluent, ExplicitIndexPath) {
    std::vector<uint8_t> v = {0x00u, 0x00u, 0x00u, 0x00u, 0x07u};
    put_varint(v, 2u);                                     // 2 indexes
    put_varint(v, 4u);                                     // zigzag(2)
    put_varint(v, 2u);                                     // zigzag(1)
    v.push_back(0x08u); v.push_back(0x01u);

    uint32_t id = 0, n_idx = 0;
    int32_t idx[kPbMaxMsgIndexes] = {0};
    const uint8_t* payload = nullptr;
    uint64_t payload_len = 0;
    ASSERT_TRUE(pb_confluent_parse(v.data(), v.size(), &id, idx,
                                   kPbMaxMsgIndexes, &n_idx,
                                   &payload, &payload_len));
    EXPECT_EQ(id, 7u);
    ASSERT_EQ(n_idx, 2u);
    EXPECT_EQ(idx[0], 2);
    EXPECT_EQ(idx[1], 1);
    EXPECT_EQ(payload_len, 2u);
}

TEST(BoltProtobufConfluent, BadMagicAndShortBufferRejected) {
    uint32_t id = 0, n_idx = 0;
    int32_t idx[kPbMaxMsgIndexes] = {0};
    const uint8_t* p = nullptr;
    uint64_t pl = 0;

    const uint8_t bad_magic[] = {0x01u, 0, 0, 0, 1u, 0x00u};
    EXPECT_FALSE(pb_confluent_parse(bad_magic, sizeof(bad_magic), &id, idx,
                                    kPbMaxMsgIndexes, &n_idx, &p, &pl));
    const uint8_t too_short[] = {0x00u, 0x00u};
    EXPECT_FALSE(pb_confluent_parse(too_short, sizeof(too_short), &id, idx,
                                    kPbMaxMsgIndexes, &n_idx, &p, &pl));
}
