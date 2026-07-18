// ---------------------------------------------------------------------------
// test_bolt_json_writer — bolt::parse::json emitter (STATION-80 compact +
// STATION-87 nlohmann-faithful mode).
//
// The faithful-mode goldens below are the EXACT bytes produced by
// nlohmann::json v3.11.2 dump()/dump(2)/dump(-1,' ',true)/dump(2,' ',true) for
// the same value, captured from the real library. A live 4.2M-comparison fuzz
// harness (scratchpad/verify_emit.cpp, STATION-87) additionally proved
// byte-identity vs real nlohmann across random nested trees + bit-fuzzed
// doubles; these hardcoded cases lock the guarantee into CI without pulling
// nlohmann into bolt's dependency-free test build.
// ---------------------------------------------------------------------------
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#include "bolt/bolt_arena.h"
#include "bolt/parse/bolt_json_writer.h"

namespace bj = bolt::parse::json;

namespace {

// Serialize with the faithful (options) emitter and return the bytes.
std::string emit(bolt::Arena& a, const bj::JsonValue* v, int32_t indent,
                 bool ensure_ascii) {
    bj::EmitOptions opts;
    opts.indent = indent;
    opts.ensure_ascii = ensure_ascii;
    const uint8_t* out = nullptr;
    int32_t len = 0;
    EXPECT_TRUE(bj::serialize(v, &a, opts, &out, &len));
    return std::string(reinterpret_cast<const char*>(out), static_cast<size_t>(len));
}

// Compact byte-faithful emitter (the STATION-80 path — must stay unchanged).
std::string emit_compact_legacy(bolt::Arena& a, const bj::JsonValue* v) {
    const uint8_t* out = nullptr;
    int32_t len = 0;
    EXPECT_TRUE(bj::serialize(v, &a, &out, &len));
    return std::string(reinterpret_cast<const char*>(out), static_cast<size_t>(len));
}

bj::JsonValue* str(bolt::Arena& a, const char* s) {
    return bj::make_string(&a, s, static_cast<int32_t>(std::strlen(s)));
}

}  // namespace

// --------------------------- float dtoa (Grisu2) ---------------------------
TEST(BoltJsonWriter, FloatShortestRoundTrip) {
    bolt::Arena a;
    // Each golden is nlohmann::json(x).dump().
    EXPECT_EQ(emit(a, bj::make_double(&a, 0.1), -1, false), "0.1");
    EXPECT_EQ(emit(a, bj::make_double(&a, 3.14), -1, false), "3.14");
    EXPECT_EQ(emit(a, bj::make_double(&a, 1.0), -1, false), "1.0");
    EXPECT_EQ(emit(a, bj::make_double(&a, 100.0), -1, false), "100.0");
    EXPECT_EQ(emit(a, bj::make_double(&a, -2.5), -1, false), "-2.5");
    EXPECT_EQ(emit(a, bj::make_double(&a, 0.0), -1, false), "0.0");
    EXPECT_EQ(emit(a, bj::make_double(&a, -0.0), -1, false), "-0.0");
    EXPECT_EQ(emit(a, bj::make_double(&a, 1e100), -1, false), "1e+100");
    EXPECT_EQ(emit(a, bj::make_double(&a, 1e-7), -1, false), "1e-07");
    EXPECT_EQ(emit(a, bj::make_double(&a, 1e21), -1, false), "1e+21");
    EXPECT_EQ(emit(a, bj::make_double(&a, 1.7976931348623157e308), -1, false),
              "1.7976931348623157e+308");
    EXPECT_EQ(emit(a, bj::make_double(&a, 5e-324), -1, false), "5e-324");
    // NaN / Inf -> null (nlohmann default).
    EXPECT_EQ(emit(a, bj::make_double(&a, std::nan("")), -1, false), "null");
    EXPECT_EQ(emit(a, bj::make_double(&a, HUGE_VAL), -1, false), "null");
}

// --------------------------- integer edges ---------------------------------
TEST(BoltJsonWriter, IntegerEdges) {
    bolt::Arena a;
    EXPECT_EQ(emit(a, bj::make_int(&a, 0), -1, false), "0");
    EXPECT_EQ(emit(a, bj::make_int(&a, -1), -1, false), "-1");
    EXPECT_EQ(emit(a, bj::make_int(&a, 42), -1, false), "42");
    EXPECT_EQ(emit(a, bj::make_int(&a, INT64_MAX), -1, false),
              "9223372036854775807");
    EXPECT_EQ(emit(a, bj::make_int(&a, INT64_MIN), -1, false),
              "-9223372036854775808");
}

// --------------------------- escapes ---------------------------------------
TEST(BoltJsonWriter, Escapes) {
    bolt::Arena a;
    EXPECT_EQ(emit(a, str(a, "q\"b\\s\ttab\nnl\r\b\f"), -1, false),
              "\"q\\\"b\\\\s\\ttab\\nnl\\r\\b\\f\"");
    EXPECT_EQ(emit(a, str(a, "\x01\x1f"), -1, false), "\"\\u0001\\u001f\"");
    // 0x7F: verbatim without ensure_ascii,  with it.
    EXPECT_EQ(emit(a, str(a, "\x7f"), -1, false), "\"\x7f\"");
    EXPECT_EQ(emit(a, str(a, "\x7f"), -1, true), "\"\\u007f\"");
}

// --------------------------- ensure_ascii / unicode ------------------------
TEST(BoltJsonWriter, EnsureAsciiUnicode) {
    bolt::Arena a;
    // U+00E9 (café) — 2-byte UTF-8.
    EXPECT_EQ(emit(a, str(a, "caf\xC3\xA9"), -1, false), "\"caf\xC3\xA9\"");
    EXPECT_EQ(emit(a, str(a, "caf\xC3\xA9"), -1, true), "\"caf\\u00e9\"");
    // U+20AC (euro) — 3-byte.
    EXPECT_EQ(emit(a, str(a, "\xE2\x82\xAC"), -1, true), "\"\\u20ac\"");
    // U+1F600 (emoji) — 4-byte -> surrogate pair.
    EXPECT_EQ(emit(a, str(a, "\xF0\x9F\x98\x80"), -1, false),
              "\"\xF0\x9F\x98\x80\"");
    EXPECT_EQ(emit(a, str(a, "\xF0\x9F\x98\x80"), -1, true),
              "\"\\ud83d\\ude00\"");
    // Invalid UTF-8 under ensure_ascii -> serialize fails (returns false).
    bj::JsonValue* bad = str(a, "\xC3\x28");  // lead byte + non-continuation
    bj::EmitOptions opts;
    opts.ensure_ascii = true;
    const uint8_t* out = nullptr;
    int32_t len = 0;
    EXPECT_FALSE(bj::serialize(bad, &a, opts, &out, &len));
    // But the SAME bytes pass through fine without ensure_ascii.
    opts.ensure_ascii = false;
    EXPECT_TRUE(bj::serialize(bad, &a, opts, &out, &len));
}

// --------------------------- pretty-print whitespace -----------------------
TEST(BoltJsonWriter, PrettyPrintMatchesNlohmann) {
    bolt::Arena a;
    // {"a":{"b":[1,2,{"c":null}]},"d":[],"e":"x"}
    bj::JsonValue* inner = bj::make_object(&a);
    bj::object_add(&a, inner, "c", 1, bj::make_null(&a));
    bj::JsonValue* barr = bj::make_array(&a);
    bj::array_add(&a, barr, bj::make_int(&a, 1));
    bj::array_add(&a, barr, bj::make_int(&a, 2));
    bj::array_add(&a, barr, inner);
    bj::JsonValue* aobj = bj::make_object(&a);
    bj::object_add(&a, aobj, "b", 1, barr);
    bj::JsonValue* root = bj::make_object(&a);
    bj::object_add(&a, root, "a", 1, aobj);
    bj::object_add(&a, root, "d", 1, bj::make_array(&a));
    bj::object_add(&a, root, "e", 1, str(a, "x"));

    EXPECT_EQ(emit(a, root, -1, false),
              "{\"a\":{\"b\":[1,2,{\"c\":null}]},\"d\":[],\"e\":\"x\"}");
    EXPECT_EQ(
        emit(a, root, 2, false),
        "{\n  \"a\": {\n    \"b\": [\n      1,\n      2,\n      {\n        "
        "\"c\": null\n      }\n    ]\n  },\n  \"d\": [],\n  \"e\": \"x\"\n}");
}

TEST(BoltJsonWriter, EmptyContainers) {
    bolt::Arena a;
    EXPECT_EQ(emit(a, bj::make_object(&a), -1, false), "{}");
    EXPECT_EQ(emit(a, bj::make_object(&a), 2, false), "{}");
    EXPECT_EQ(emit(a, bj::make_array(&a), -1, false), "[]");
    EXPECT_EQ(emit(a, bj::make_array(&a), 2, false), "[]");
}

TEST(BoltJsonWriter, PrettyIndentWidthVariations) {
    bolt::Arena a;
    bj::JsonValue* arr = bj::make_array(&a);
    bj::array_add(&a, arr, bj::make_int(&a, 1));
    bj::array_add(&a, arr, bj::make_int(&a, 2));
    // indent=0 -> newlines, no leading spaces (nlohmann dump(0)).
    EXPECT_EQ(emit(a, arr, 0, false), "[\n1,\n2\n]");
    // indent=4.
    EXPECT_EQ(emit(a, arr, 4, false), "[\n    1,\n    2\n]");
}

// --------------------------- keys are escaped ------------------------------
TEST(BoltJsonWriter, ObjectKeysEscaped) {
    bolt::Arena a;
    bj::JsonValue* obj = bj::make_object(&a);
    bj::object_add(&a, obj, "a\"b", 3, bj::make_int(&a, 1));
    EXPECT_EQ(emit(a, obj, -1, false), "{\"a\\\"b\":1}");
    // ensure_ascii escapes non-ASCII keys too.
    bj::JsonValue* obj2 = bj::make_object(&a);
    bj::object_add(&a, obj2, "k\xC3\xA9", 3, bj::make_int(&a, 1));
    EXPECT_EQ(emit(a, obj2, -1, true), "{\"k\\u00e9\":1}");
}

// --------------------------- legacy compact mode unchanged -----------------
TEST(BoltJsonWriter, LegacyCompactPathUnchanged) {
    bolt::Arena a;
    // The no-options serialize() keeps object keys VERBATIM and no whitespace.
    bj::JsonValue* obj = bj::make_object(&a);
    bj::object_add(&a, obj, "n", 1, bj::make_int(&a, 5));
    bj::object_add(&a, obj, "s", 1, str(a, "hi"));
    EXPECT_EQ(emit_compact_legacy(a, obj), "{\"n\":5,\"s\":\"hi\"}");
}
