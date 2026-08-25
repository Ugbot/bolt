// test_bolt_parse_json.cpp — coverage for bolt::parse::json (Layer 1.3).
//
// Skip-aware JSON parser tests. STL is allowed in tests; production code
// stays Tiger-Style.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_types.h"
#include "bolt/parse/bolt_json.h"

using bolt::Arena;
using bolt::ArenaConfig;
using bolt::StringView;
using bolt::parse::json::Iterator;
using bolt::parse::json::PathFilter;
using bolt::parse::json::StructuralIndex;
using bolt::parse::json::Token;
using bolt::parse::json::TokenType;

namespace {

Arena make_arena() {
    ArenaConfig cfg{};
    cfg.initial_block_size = 64u * 1024u;
    cfg.max_block_size     = 4u * 1024u * 1024u;
    cfg.alignment          = 8u;
    return Arena(cfg);
}

// Copies the source into the ARENA before indexing.
//
// build_index borrows `src` and never copies it: every Token is an offset
// into those bytes, and iter_int64 / iter_float64 / iter_string read them on
// demand. Every call site below passes a string LITERAL, which materialises a
// temporary std::string that dies at the end of the full expression -- so
// without this copy the index points at freed stack memory the moment build()
// returns.
//
// It behaved for a long time because the short-string optimisation keeps such
// a literal inside the temporary object, and the dead stack slot usually
// still held the right bytes. At -O3 it does not: NumbersLazyMaterialise
// parsed 42 and -17 correctly and then failed on 3.14, because only part of
// the slot had been reused. Copying into the arena ties the bytes to the
// arena's lifetime, which outlives the index in every test here.
bool build(Arena& a, const std::string& s, StructuralIndex* out) {
    auto* buf = static_cast<uint8_t*>(
        a.allocate(s.size() + 1u, 1u));
    if (buf == nullptr) return false;
    if (!s.empty()) std::memcpy(buf, s.data(), s.size());
    buf[s.size()] = 0;
    return bolt::parse::json::build_index(
        buf, static_cast<int32_t>(s.size()), &a, out);
}

}  // namespace

TEST(BoltParseJson, EmptyDocument) {
    Arena a = make_arena();
    StructuralIndex idx;
    EXPECT_TRUE(build(a, "", &idx));
    ASSERT_EQ(idx.token_count, 1);
    EXPECT_EQ(idx.tokens[0].type, TokenType::End);

    EXPECT_TRUE(build(a, "   ", &idx));
    ASSERT_EQ(idx.token_count, 1);
    EXPECT_EQ(idx.tokens[0].type, TokenType::End);

    EXPECT_TRUE(build(a, "null", &idx));
    ASSERT_EQ(idx.token_count, 2);
    EXPECT_EQ(idx.tokens[0].type, TokenType::Null);
    EXPECT_EQ(idx.tokens[1].type, TokenType::End);
}

TEST(BoltParseJson, DeepNesting) {
    Arena a = make_arena();
    // 50 levels deep — within the cap of 64.
    std::string s(50, '[');
    s.append(50, ']');
    StructuralIndex idx;
    EXPECT_TRUE(build(a, s, &idx));
    int32_t begins = 0, ends = 0;
    for (int32_t i = 0; i < idx.token_count; ++i) {
        if (idx.tokens[i].type == TokenType::BeginArray) ++begins;
        if (idx.tokens[i].type == TokenType::EndArray)   ++ends;
    }
    EXPECT_EQ(begins, 50);
    EXPECT_EQ(ends, 50);

    // 80 levels — exceeds cap, must reject.
    std::string deep(80, '[');
    deep.append(80, ']');
    StructuralIndex idx2;
    EXPECT_FALSE(build(a, deep, &idx2));
}

TEST(BoltParseJson, EscapedStrings) {
    Arena a = make_arena();
    StructuralIndex idx;
    // {"k": "\"\\né"} — note the source bytes for é are the 6 ASCII
    // chars literally; UTF-8 decode happens lazily in the consumer.
    std::string s = "{\"k\": \"\\\"\\n\\u00e9\"}";
    ASSERT_TRUE(build(a, s, &idx));
    ASSERT_GE(idx.token_count, 5);
    EXPECT_EQ(idx.tokens[0].type, TokenType::BeginObject);
    EXPECT_EQ(idx.tokens[1].type, TokenType::Key);
    EXPECT_EQ(idx.tokens[2].type, TokenType::String);
    // Slice covers the source bytes between the quotes.
    const Token& str = idx.tokens[2];
    std::string slice(reinterpret_cast<const char*>(idx.src) + str.start, str.length);
    EXPECT_EQ(slice, "\\\"\\n\\u00e9");
}

TEST(BoltParseJson, UnicodeUtf8Validated) {
    Arena a = make_arena();
    StructuralIndex idx;
    // Valid UTF-8: U+00E9 LATIN SMALL LETTER E WITH ACUTE = 0xC3 0xA9.
    std::string ok;
    ok.push_back('"');
    ok.push_back(static_cast<char>(0xC3));
    ok.push_back(static_cast<char>(0xA9));
    ok.push_back('"');
    EXPECT_TRUE(build(a, ok, &idx));

    // Invalid UTF-8: overlong NUL via 0xC0 0x80.
    std::string bad;
    bad.push_back('"');
    bad.push_back(static_cast<char>(0xC0));
    bad.push_back(static_cast<char>(0x80));
    bad.push_back('"');
    StructuralIndex idx2;
    EXPECT_FALSE(build(a, bad, &idx2));
}

TEST(BoltParseJson, NumbersLazyMaterialise) {
    Arena a = make_arena();
    StructuralIndex idx;
    ASSERT_TRUE(build(a, "[42, -17, 3.14, 1.0e3]", &idx));
    Iterator it;
    ASSERT_TRUE(bolt::parse::json::iter_init(&idx, &it));
    EXPECT_EQ(bolt::parse::json::iter_peek(&it), TokenType::BeginArray);
    bolt::parse::json::iter_advance(&it);

    int64_t i64 = 0;
    ASSERT_TRUE(bolt::parse::json::iter_int64(&it, &i64));
    EXPECT_EQ(i64, 42);
    bolt::parse::json::iter_advance(&it);

    ASSERT_TRUE(bolt::parse::json::iter_int64(&it, &i64));
    EXPECT_EQ(i64, -17);
    bolt::parse::json::iter_advance(&it);

    double f = 0.0;
    ASSERT_TRUE(bolt::parse::json::iter_float64(&it, &f));
    EXPECT_DOUBLE_EQ(f, 3.14);
    bolt::parse::json::iter_advance(&it);

    ASSERT_TRUE(bolt::parse::json::iter_float64(&it, &f));
    EXPECT_DOUBLE_EQ(f, 1.0e3);
}

TEST(BoltParseJson, SkipToCloseEqualsSequentialAdvance) {
    Arena a = make_arena();
    StructuralIndex idx;
    const std::string src = "{\"a\":1,\"b\":[1,2,{\"x\":[true,false,null,\"s\"]}],\"c\":3}";
    ASSERT_TRUE(build(a, src, &idx));

    // Find the BeginArray for "b".
    int32_t b_begin = -1;
    for (int32_t i = 0; i < idx.token_count; ++i) {
        if (idx.tokens[i].type == TokenType::Key) {
            const Token& k = idx.tokens[i];
            if (k.length == 1 && idx.src[k.start] == 'b') {
                b_begin = i + 1;
                break;
            }
        }
    }
    ASSERT_GE(b_begin, 0);
    ASSERT_EQ(idx.tokens[b_begin].type, TokenType::BeginArray);

    // Skip-based.
    Iterator skip_it{ &idx, b_begin, 0 };
    ASSERT_TRUE(bolt::parse::json::iter_skip_to_close(&skip_it));

    // Advance-based: walk until depth returns to zero.
    Iterator adv_it{ &idx, b_begin, 0 };
    int32_t depth = 0;
    while (adv_it.cursor < idx.token_count) {
        TokenType t = idx.tokens[adv_it.cursor].type;
        if (t == TokenType::BeginObject || t == TokenType::BeginArray) ++depth;
        else if (t == TokenType::EndObject || t == TokenType::EndArray) {
            --depth;
            if (depth == 0) { ++adv_it.cursor; break; }
        }
        ++adv_it.cursor;
    }
    EXPECT_EQ(skip_it.cursor, adv_it.cursor);
}

TEST(BoltParseJson, PathFilterSubtreesSkipped) {
    Arena a = make_arena();
    const std::string doc =
        "{"
            "\"wanted\":{\"x\":1,\"y\":[10,20,30]},"
            "\"unwanted\":{\"a\":1,\"b\":[1,2,3,4,5,6,7,8,9,10]},"
            "\"also\":[{\"k\":\"v\"},{\"k\":\"v2\"}],"
            "\"trash\":[1,2,3,4,5,6,7,8,9,10]"
        "}";
    StructuralIndex full;
    ASSERT_TRUE(build(a, doc, &full));

    const char* paths[]    = { "/wanted", "/also/0" };
    const int32_t lens[]   = { 7, 7 };
    PathFilter pf{};
    ASSERT_TRUE(bolt::parse::json::compile_paths(paths, lens, 2, &a, &pf));

    StructuralIndex filt;
    ASSERT_TRUE(bolt::parse::json::build_index_filtered(
        reinterpret_cast<const uint8_t*>(doc.data()),
        static_cast<int32_t>(doc.size()), &pf, &a, &filt));

    EXPECT_LT(filt.token_count, full.token_count);

    // Verify "wanted" subtree is fully present (BeginObject/EndObject pair
    // for that subobject and all its leaves).
    int32_t wanted_keys = 0;
    int32_t unwanted_keys = 0;
    int32_t trash_keys = 0;
    for (int32_t i = 0; i < filt.token_count; ++i) {
        if (filt.tokens[i].type == TokenType::Key) {
            const Token& k = filt.tokens[i];
            std::string s(reinterpret_cast<const char*>(filt.src) + k.start, k.length);
            if (s == "wanted") ++wanted_keys;
            else if (s == "unwanted") ++unwanted_keys;
            else if (s == "trash") ++trash_keys;
        }
    }
    EXPECT_EQ(wanted_keys, 1);
    EXPECT_EQ(unwanted_keys, 0);
    EXPECT_EQ(trash_keys, 0);

    // Print delta for the report. (Test asserts the relationship; the bench
    // run captures the absolute numbers.)
    ::testing::Test::RecordProperty(
        "filtered_token_count", std::to_string(filt.token_count));
    ::testing::Test::RecordProperty(
        "full_token_count", std::to_string(full.token_count));
}

TEST(BoltParseJson, MalformedJsonRejected) {
    Arena a = make_arena();
    StructuralIndex idx;
    EXPECT_FALSE(build(a, "{\"a\":1", &idx));        // unterminated object
    EXPECT_FALSE(build(a, "[1,2,3", &idx));          // unterminated array
    EXPECT_FALSE(build(a, "{\"a\":}", &idx));        // missing value
    EXPECT_FALSE(build(a, "\"unterm", &idx));        // unterminated string
    EXPECT_FALSE(build(a, "[1,]", &idx));            // trailing comma
    EXPECT_FALSE(build(a, "{]}", &idx));             // bad bracket pair
    EXPECT_FALSE(build(a, "tru", &idx));             // truncated keyword
    EXPECT_FALSE(build(a, "1.", &idx));              // bad number
}

TEST(BoltParseJson, FilterTokenDelta) {
    // Standalone larger-document test that records the exact token-count
    // delta the report cares about.
    Arena a = make_arena();
    std::string doc = "{";
    doc += "\"wanted\":{\"x\":1,\"y\":2},";
    doc += "\"also\":[{\"keep\":true},{\"keep\":false},{\"keep\":null}],";
    doc += "\"big\":[";
    for (int i = 0; i < 64; ++i) {
        if (i) doc += ",";
        doc += "{\"id\":";
        doc += std::to_string(i);
        doc += ",\"name\":\"name_";
        doc += std::to_string(i);
        doc += "\",\"tags\":[\"a\",\"b\",\"c\"]}";
    }
    doc += "]}";

    StructuralIndex full;
    ASSERT_TRUE(build(a, doc, &full));

    const char* paths[] = { "/wanted", "/also/0" };
    const int32_t lens[] = { 7, 7 };
    PathFilter pf{};
    ASSERT_TRUE(bolt::parse::json::compile_paths(paths, lens, 2, &a, &pf));

    StructuralIndex filt;
    ASSERT_TRUE(bolt::parse::json::build_index_filtered(
        reinterpret_cast<const uint8_t*>(doc.data()),
        static_cast<int32_t>(doc.size()), &pf, &a, &filt));

    std::printf("[BoltParseJson] full_tokens=%d filtered_tokens=%d delta=%d\n",
                full.token_count, filt.token_count,
                full.token_count - filt.token_count);
    EXPECT_LT(filt.token_count, full.token_count);
}

// ===========================================================================
// fionn-parity wave: SAX streaming + NDJSON framing + SIMD plain-run.
// ===========================================================================

namespace {

// Event recorder — turns SAX callbacks into a comparable string trace.
struct SaxTrace {
    std::string events;
    int64_t     abort_after = -1;   // abort when event count reaches this
    int64_t     count = 0;

    bool tick() {
        ++count;
        return abort_after < 0 || count < abort_after;
    }
    static bool obj_b(void* c) noexcept { auto* t = static_cast<SaxTrace*>(c); t->events += "{"; return t->tick(); }
    static bool obj_e(void* c) noexcept { auto* t = static_cast<SaxTrace*>(c); t->events += "}"; return t->tick(); }
    static bool arr_b(void* c) noexcept { auto* t = static_cast<SaxTrace*>(c); t->events += "["; return t->tick(); }
    static bool arr_e(void* c) noexcept { auto* t = static_cast<SaxTrace*>(c); t->events += "]"; return t->tick(); }
    static bool key(void* c, const char* b, int32_t n) noexcept {
        auto* t = static_cast<SaxTrace*>(c);
        t->events += "K(" + std::string(b, static_cast<size_t>(n)) + ")";
        return t->tick();
    }
    static bool str(void* c, const char* b, int32_t n) noexcept {
        auto* t = static_cast<SaxTrace*>(c);
        t->events += "S(" + std::string(b, static_cast<size_t>(n)) + ")";
        return t->tick();
    }
    static bool i64(void* c, int64_t v, const char*, int32_t) noexcept {
        auto* t = static_cast<SaxTrace*>(c);
        t->events += "I(" + std::to_string(v) + ")";
        return t->tick();
    }
    static bool f64(void* c, double v, const char*, int32_t) noexcept {
        auto* t = static_cast<SaxTrace*>(c);
        t->events += "F(" + std::to_string(static_cast<int64_t>(v * 1000)) + ")";
        return t->tick();
    }
    static bool boolean(void* c, bool v) noexcept {
        auto* t = static_cast<SaxTrace*>(c);
        t->events += v ? "T" : "f";
        return t->tick();
    }
    static bool null_cb(void* c) noexcept {
        auto* t = static_cast<SaxTrace*>(c); t->events += "N"; return t->tick();
    }

    bolt::parse::json::SaxHandler handler() {
        bolt::parse::json::SaxHandler h{};
        h.ctx = this;
        h.on_begin_object = &obj_b; h.on_end_object = &obj_e;
        h.on_begin_array = &arr_b;  h.on_end_array = &arr_e;
        h.on_key = &key;            h.on_string = &str;
        h.on_int64 = &i64;          h.on_float64 = &f64;
        h.on_bool = &boolean;       h.on_null = &null_cb;
        return h;
    }
};

bool sax(const std::string& s, SaxTrace* t) {
    auto h = t->handler();
    return bolt::parse::json::sax_parse(
        reinterpret_cast<const uint8_t*>(s.data()),
        static_cast<int32_t>(s.size()), &h);
}

}  // namespace

TEST(BoltParseJsonSax, EventSequence) {
    SaxTrace t;
    ASSERT_TRUE(sax("{\"a\": 1, \"b\": [true, null, \"x\"], \"c\": 2.5}", &t));
    EXPECT_EQ(t.events, "{K(a)I(1)K(b)[TNS(x)]K(c)F(2500)}");
}

TEST(BoltParseJsonSax, EmptyAndScalars) {
    SaxTrace t1;
    EXPECT_TRUE(sax("", &t1));
    EXPECT_EQ(t1.events, "");
    SaxTrace t2;
    EXPECT_TRUE(sax("42", &t2));
    EXPECT_EQ(t2.events, "I(42)");
}

TEST(BoltParseJsonSax, MalformedRejects) {
    SaxTrace t;
    EXPECT_FALSE(sax("{\"a\": }", &t));
    SaxTrace t2;
    EXPECT_FALSE(sax("[1, 2", &t2));
}

TEST(BoltParseJsonSax, CallbackAbortStopsParse) {
    SaxTrace t;
    t.abort_after = 3;
    EXPECT_FALSE(sax("[1, 2, 3, 4, 5, 6, 7, 8]", &t));
    // Got the begin-array + first two ints, then aborted.
    EXPECT_EQ(t.events, "[I(1)I(2)");
}

TEST(BoltParseJsonSax, NullCallbacksSkipEvents) {
    // Only on_int64 wired; everything else nullptr — must not crash and
    // must still validate structure.
    struct Sum { int64_t total = 0; } s;
    bolt::parse::json::SaxHandler h{};
    h.ctx = &s;
    h.on_int64 = [](void* c, int64_t v, const char*, int32_t) noexcept {
        static_cast<Sum*>(c)->total += v;
        return true;
    };
    const std::string doc = "{\"a\": [1, 2, 3], \"b\": {\"c\": 4}}";
    EXPECT_TRUE(bolt::parse::json::sax_parse(
        reinterpret_cast<const uint8_t*>(doc.data()),
        static_cast<int32_t>(doc.size()), &h));
    EXPECT_EQ(s.total, 10);
}

TEST(BoltParseJsonSax, MatchesTapeOnLongStrings) {
    // SIMD plain-run stress: strings spanning the 8/32-byte boundaries
    // with escapes, quotes-at-every-offset, controls and UTF-8 mixed in.
    Arena a = make_arena();
    for (int pad = 0; pad < 40; ++pad) {
        std::string val(static_cast<size_t>(pad), 'x');
        val += "\\\"q\\\\";          // escaped quote + escaped backslash
        val += std::string(17, 'y');
        val += "\xC3\xA9";            // 2-byte UTF-8
        val += std::string(9, 'z');
        const std::string doc = "{\"k\": \"" + val + "\"}";

        StructuralIndex idx;
        ASSERT_TRUE(build(a, doc, &idx)) << "pad=" << pad;
        // tokens: BeginObject Key String EndObject End
        ASSERT_EQ(idx.token_count, 5) << "pad=" << pad;
        EXPECT_EQ(idx.tokens[2].type, TokenType::String);
        EXPECT_EQ(idx.tokens[2].length, static_cast<int32_t>(val.size()));

        SaxTrace t;
        ASSERT_TRUE(sax(doc, &t)) << "pad=" << pad;
        EXPECT_EQ(t.events, "{K(k)S(" + val + ")}") << "pad=" << pad;
    }
}

TEST(BoltParseJsonSax, RejectsBareControlAndBadUtf8AcrossRuns) {
    SaxTrace t1;
    std::string ctrl = "{\"k\": \"" + std::string(20, 'a');
    ctrl += '\x01';
    ctrl += "tail\"}";
    EXPECT_FALSE(sax(ctrl, &t1));

    SaxTrace t2;
    std::string bad = "{\"k\": \"" + std::string(20, 'a');
    bad += "\xC0\x80";   // overlong NUL
    bad += "\"}";
    EXPECT_FALSE(sax(bad, &t2));
}

TEST(BoltParseJsonNdjson, ForEachTape) {
    Arena scratch = make_arena();
    const std::string src =
        "{\"a\": 1}\n"
        "\n"
        "  {\"a\": 2}  \r\n"
        "{\"a\": 3}";   // no trailing newline
    struct Ctx { int64_t sum = 0; int64_t seen = 0; } c;
    bolt::parse::json::NdjsonStats st{};
    ASSERT_TRUE(bolt::parse::json::ndjson_for_each(
        reinterpret_cast<const uint8_t*>(src.data()),
        static_cast<int64_t>(src.size()),
        /*ignore_errors=*/false, &scratch, &c,
        [](void* ctx, const StructuralIndex* rec, int64_t) noexcept {
            auto* cc = static_cast<Ctx*>(ctx);
            // rec: BeginObject Key Int EndObject End
            Iterator it{rec, 2, 0};
            int64_t v = 0;
            if (!bolt::parse::json::iter_int64(&it, &v)) return false;
            cc->sum += v;
            cc->seen += 1;
            return true;
        },
        &st));
    EXPECT_EQ(st.records, 3);
    EXPECT_EQ(st.skipped, 0);
    EXPECT_EQ(c.sum, 6);
    EXPECT_EQ(c.seen, 3);
}

TEST(BoltParseJsonNdjson, IgnoreErrorsSkipsBadLines) {
    Arena scratch = make_arena();
    const std::string src =
        "{\"a\": 1}\n"
        "{BROKEN\n"
        "{\"a\": 3}\n";
    bolt::parse::json::NdjsonStats st{};
    int64_t sum = 0;
    ASSERT_TRUE(bolt::parse::json::ndjson_for_each(
        reinterpret_cast<const uint8_t*>(src.data()),
        static_cast<int64_t>(src.size()),
        /*ignore_errors=*/true, &scratch, &sum,
        [](void* ctx, const StructuralIndex* rec, int64_t) noexcept {
            Iterator it{rec, 2, 0};
            int64_t v = 0;
            if (!bolt::parse::json::iter_int64(&it, &v)) return false;
            *static_cast<int64_t*>(ctx) += v;
            return true;
        },
        &st));
    EXPECT_EQ(st.records, 2);
    EXPECT_EQ(st.skipped, 1);
    EXPECT_EQ(sum, 4);

    // Strict mode fails on the same input.
    EXPECT_FALSE(bolt::parse::json::ndjson_for_each(
        reinterpret_cast<const uint8_t*>(src.data()),
        static_cast<int64_t>(src.size()),
        /*ignore_errors=*/false, &scratch, &sum,
        [](void*, const StructuralIndex*, int64_t) noexcept { return true; },
        nullptr));
}

TEST(BoltParseJsonNdjson, ForEachSax) {
    const std::string src =
        "{\"v\": 10}\n"
        "{BROKEN}\n"
        "{\"v\": 32}\n";
    struct Sum { int64_t total = 0; } s;
    bolt::parse::json::SaxHandler h{};
    h.ctx = &s;
    h.on_int64 = [](void* c, int64_t v, const char*, int32_t) noexcept {
        static_cast<Sum*>(c)->total += v;
        return true;
    };
    bolt::parse::json::NdjsonStats st{};
    ASSERT_TRUE(bolt::parse::json::ndjson_for_each_sax(
        reinterpret_cast<const uint8_t*>(src.data()),
        static_cast<int64_t>(src.size()),
        /*ignore_errors=*/true, &h, &st));
    EXPECT_EQ(st.records, 2);
    EXPECT_EQ(st.skipped, 1);
    EXPECT_EQ(s.total, 42);
}

TEST(BoltParseJsonNdjson, SaxAbortIsNotSkipped) {
    // A consumer abort must fail the run even under ignore_errors —
    // it is not a malformed record.
    const std::string src = "{\"v\": 1}\n{\"v\": 2}\n";
    bolt::parse::json::SaxHandler h{};
    h.on_int64 = [](void*, int64_t, const char*, int32_t) noexcept {
        return false;   // abort immediately
    };
    bolt::parse::json::NdjsonStats st{};
    EXPECT_FALSE(bolt::parse::json::ndjson_for_each_sax(
        reinterpret_cast<const uint8_t*>(src.data()),
        static_cast<int64_t>(src.size()),
        /*ignore_errors=*/true, &h, &st));
}
