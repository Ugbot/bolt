// test_bolt_utf8.cpp — coverage for the UTF-8 native operator kernels.

#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_types.h"
#include "bolt/kernels/bolt_utf8.h"

using bolt::Arena;
using bolt::StringView;
namespace ku = bolt::kernels::utf8;

namespace {

// Build an inline StringView from a string literal (<= 12 bytes).
static StringView make_inline(const char* s) {
    return StringView::from_cstr(s);
}

// Build a spilled view backed by an external byte buffer.  Caller passes the
// buffer base and a byte offset; we set ref.offset accordingly.  Length must
// be > 12 to exercise the spilled path.
static StringView make_spilled(const char* bytes, uint32_t len, uint32_t offset) {
    StringView v;
    memset(&v, 0, sizeof(v));
    v.length = len;
    memcpy(v.prefix, bytes, 4);
    v.ref.buf_idx = 0;
    v.ref.offset = offset;
    return v;
}

class Utf8Test : public ::testing::Test {
protected:
    Arena arena_;
};

TEST_F(Utf8Test, CompareInline) {
    StringView a[3] = {
        make_inline("apple"),
        make_inline("banana"),
        make_inline("cherry"),
    };
    StringView b[3] = {
        make_inline("apple"),
        make_inline("banana_"),  // longer
        make_inline("cherryy"),  // longer
    };
    int32_t out[3];
    ku::utf8_compare(a, b, out, 3);
    EXPECT_EQ(out[0],  0);
    EXPECT_EQ(out[1], -1);
    EXPECT_EQ(out[2], -1);
}

TEST_F(Utf8Test, FilterEqInline) {
    StringView data[5] = {
        make_inline("foo"),
        make_inline("bar"),
        make_inline("foo"),
        make_inline("baz"),
        make_inline("foo"),
    };
    int32_t sel[5];
    auto needle = make_inline("foo");
    int64_t c = ku::utf8_filter_eq(data, 5, needle, sel);
    ASSERT_EQ(c, 3);
    EXPECT_EQ(sel[0], 0);
    EXPECT_EQ(sel[1], 2);
    EXPECT_EQ(sel[2], 4);
}

TEST_F(Utf8Test, FilterEqSpilled) {
    // 16-byte strings exercise spilled path.
    const char buf[] = "AAAAAAAAAAAAAAAA"  // 16
                       "BBBBBBBBBBBBBBBB"
                       "AAAAAAAAAAAAAAAA";
    StringView data[3] = {
        make_spilled(buf, 16, 0),
        make_spilled(buf + 16, 16, 16),
        make_spilled(buf + 32, 16, 32),
    };
    auto needle = make_spilled(buf, 16, 0);
    int32_t sel[3];
    int64_t c = ku::utf8_filter_eq(data, 3, needle, sel, buf, buf);
    ASSERT_EQ(c, 2);
    EXPECT_EQ(sel[0], 0);
    EXPECT_EQ(sel[1], 2);
}

TEST_F(Utf8Test, FilterNe) {
    StringView data[3] = {
        make_inline("foo"),
        make_inline("bar"),
        make_inline("foo"),
    };
    auto needle = make_inline("foo");
    int32_t sel[3];
    int64_t c = ku::utf8_filter_ne(data, 3, needle, sel);
    ASSERT_EQ(c, 1);
    EXPECT_EQ(sel[0], 1);
}

TEST_F(Utf8Test, HashStability) {
    auto a = make_inline("hello");
    auto b = make_inline("hello");
    uint64_t ha = ku::utf8_hash_one(a.prefix, a.length);
    uint64_t hb = ku::utf8_hash_one(b.prefix, b.length);
    EXPECT_EQ(ha, hb);
    auto c = make_inline("world");
    uint64_t hc = ku::utf8_hash_one(c.prefix, c.length);
    EXPECT_NE(ha, hc);
}

TEST_F(Utf8Test, HashArray) {
    StringView d[4] = {
        make_inline("foo"), make_inline("bar"),
        make_inline("foo"), make_inline("baz"),
    };
    uint64_t out[4];
    ku::utf8_hash(d, 4, out);
    EXPECT_EQ(out[0], out[2]);
    EXPECT_NE(out[0], out[1]);
    EXPECT_NE(out[0], out[3]);
}

TEST_F(Utf8Test, LikeCompileClassification) {
    ku::CompiledLike p;
    auto pat = make_inline("foo");
    ASSERT_TRUE(ku::utf8_like_compile(pat, &p));
    EXPECT_EQ(p.kind, ku::LikeKind::Exact);

    pat = make_inline("foo%");
    ASSERT_TRUE(ku::utf8_like_compile(pat, &p));
    EXPECT_EQ(p.kind, ku::LikeKind::Prefix);

    pat = make_inline("%BRASS");
    ASSERT_TRUE(ku::utf8_like_compile(pat, &p));
    EXPECT_EQ(p.kind, ku::LikeKind::Suffix);
    EXPECT_EQ(p.literal_len, 5u);

    pat = make_inline("%mid%");
    ASSERT_TRUE(ku::utf8_like_compile(pat, &p));
    EXPECT_EQ(p.kind, ku::LikeKind::Contains);

    pat = make_inline("a_b%c");
    ASSERT_TRUE(ku::utf8_like_compile(pat, &p));
    EXPECT_EQ(p.kind, ku::LikeKind::General);
}

TEST_F(Utf8Test, LikeFilterPrefixSuffixContainsExact) {
    StringView data[6] = {
        make_inline("brass"),
        make_inline("steel"),
        make_inline("copperBRASS"),
        make_inline("BRASS"),
        make_inline("aluminum"),
        make_inline("middle"),
    };
    ku::CompiledLike p;
    int32_t sel[6];

    auto suf = make_inline("%BRASS");
    ASSERT_TRUE(ku::utf8_like_compile(suf, &p));
    int64_t c = ku::utf8_filter_like(data, 6, &p, sel);
    ASSERT_EQ(c, 2);
    EXPECT_EQ(sel[0], 2);
    EXPECT_EQ(sel[1], 3);

    auto pre = make_inline("brass%");
    ASSERT_TRUE(ku::utf8_like_compile(pre, &p));
    c = ku::utf8_filter_like(data, 6, &p, sel);
    ASSERT_EQ(c, 1);
    EXPECT_EQ(sel[0], 0);

    auto contains = make_inline("%idd%");
    ASSERT_TRUE(ku::utf8_like_compile(contains, &p));
    c = ku::utf8_filter_like(data, 6, &p, sel);
    ASSERT_EQ(c, 1);
    EXPECT_EQ(sel[0], 5);

    auto exact = make_inline("steel");
    ASSERT_TRUE(ku::utf8_like_compile(exact, &p));
    c = ku::utf8_filter_like(data, 6, &p, sel);
    ASSERT_EQ(c, 1);
    EXPECT_EQ(sel[0], 1);
}

TEST_F(Utf8Test, LikeGeneral) {
    StringView data[5] = {
        make_inline("aXbYc"),
        make_inline("aXbZc"),
        make_inline("aXbc"),
        make_inline("ab"),
        make_inline("aXXbYYc"),
    };
    ku::CompiledLike p;
    auto pat = make_inline("a_b%c");
    ASSERT_TRUE(ku::utf8_like_compile(pat, &p));
    int32_t sel[5];
    int64_t c = ku::utf8_filter_like(data, 5, &p, sel);
    // Matches: "aXbYc" (X then "Yc"->%c absorbs "Y"+c?  needs Y then "c").
    // Let's enumerate: pattern = LIT(a), UND, LIT(b), PCT, LIT(c).
    //  "aXbYc" -> a,X,b,Yc?  after PCT we need final LIT 'c'.  "Y"+"c" => yes.
    //  "aXbZc" -> yes.
    //  "aXbc"  -> yes (% matches empty).
    //  "ab"    -> _ matches X but b doesn't follow.  no.
    //  "aXXbYYc"-> _ matches X, then need 'b' at offset 2 which is 'X'.  no.
    ASSERT_EQ(c, 3);
    EXPECT_EQ(sel[0], 0);
    EXPECT_EQ(sel[1], 1);
    EXPECT_EQ(sel[2], 2);
}

TEST_F(Utf8Test, LikeNotLike) {
    StringView data[3] = {
        make_inline("foo"), make_inline("bar"), make_inline("foo"),
    };
    ku::CompiledLike p;
    auto pat = make_inline("foo");
    ASSERT_TRUE(ku::utf8_like_compile(pat, &p));
    int32_t sel[3];
    int64_t c = ku::utf8_filter_not_like(data, 3, &p, sel);
    ASSERT_EQ(c, 1);
    EXPECT_EQ(sel[0], 1);
}

TEST_F(Utf8Test, Substring) {
    StringView in[3] = {
        make_inline("hello"),
        make_inline("worldwide!"),
        make_inline("abcdefghijkl"),  // 12 bytes (still inline)
    };
    StringView out[3];
    // SUBSTRING(s, 2, 3): 1-based start 2, length 3.
    ku::utf8_substring(in, 3, 2, 3, out, &arena_, reinterpret_cast<const char*>(&arena_));
    EXPECT_EQ(out[0].length, 3u);
    EXPECT_EQ(memcmp(out[0].prefix, "ell", 3), 0);
    EXPECT_EQ(out[1].length, 3u);
    EXPECT_EQ(memcmp(out[1].prefix, "orl", 3), 0);
    EXPECT_EQ(out[2].length, 3u);
    EXPECT_EQ(memcmp(out[2].prefix, "bcd", 3), 0);
}

TEST_F(Utf8Test, SubstringSaturatesAndClamps) {
    StringView in[2] = { make_inline("abcd"), make_inline("xy") };
    StringView out[2];
    // start past end → empty
    ku::utf8_substring(in, 2, 10, 5, out, &arena_, reinterpret_cast<const char*>(&arena_));
    EXPECT_EQ(out[0].length, 0u);
    EXPECT_EQ(out[1].length, 0u);
    // start < 1 → clamp to 1; length saturates at remaining.
    ku::utf8_substring(in, 2, 0, 100, out, &arena_, reinterpret_cast<const char*>(&arena_));
    EXPECT_EQ(out[0].length, 4u);
    EXPECT_EQ(out[1].length, 2u);
}

TEST_F(Utf8Test, Trim) {
    StringView in[3] = {
        make_inline("  hi  "),
        make_inline("nope"),
        make_inline("\tx\n"),
    };
    StringView lo[3], ro[3], to[3];
    auto anchor = reinterpret_cast<const char*>(&arena_);
    ku::utf8_ltrim(in, 3, lo, &arena_, anchor);
    ku::utf8_rtrim(in, 3, ro, &arena_, anchor);
    ku::utf8_trim (in, 3, to, &arena_, anchor);
    EXPECT_EQ(lo[0].length, 4u);
    EXPECT_EQ(memcmp(lo[0].prefix, "hi  ", 4), 0);
    EXPECT_EQ(ro[0].length, 4u);
    EXPECT_EQ(memcmp(ro[0].prefix, "  hi", 4), 0);
    EXPECT_EQ(to[0].length, 2u);
    EXPECT_EQ(memcmp(to[0].prefix, "hi", 2), 0);
    EXPECT_EQ(lo[1].length, 4u);  // unchanged
    EXPECT_EQ(to[2].length, 1u);
}

TEST_F(Utf8Test, UpperLower) {
    StringView in[3] = {
        make_inline("Hello"),
        make_inline("MIXED case"),
        make_inline("123"),
    };
    StringView up[3], lo[3];
    auto anchor = reinterpret_cast<const char*>(&arena_);
    ku::utf8_upper(in, 3, up, &arena_, anchor);
    ku::utf8_lower(in, 3, lo, &arena_, anchor);
    EXPECT_EQ(up[0].length, 5u);
    EXPECT_EQ(memcmp(up[0].prefix, "HELL", 4), 0);
    EXPECT_EQ(up[0].inline_data[0], 'O');
    EXPECT_EQ(lo[1].length, 10u);
    EXPECT_EQ(memcmp(lo[1].prefix, "mixe", 4), 0);
    EXPECT_EQ(up[2].length, 3u);
    EXPECT_EQ(memcmp(up[2].prefix, "123", 3), 0);
}

TEST_F(Utf8Test, ByteAndCharLength) {
    StringView in[3] = {
        make_inline("hi"),         // 2 ASCII
        make_inline("a\xc3\xa9"),  // 'a' + 'é' (2-byte UTF-8) = 3 bytes / 2 codepoints
        make_inline(""),
    };
    int32_t bl[3], cl[3];
    ku::utf8_byte_length(in, 3, bl);
    ku::utf8_char_length(in, 3, cl);
    EXPECT_EQ(bl[0], 2);  EXPECT_EQ(cl[0], 2);
    EXPECT_EQ(bl[1], 3);  EXPECT_EQ(cl[1], 2);
    EXPECT_EQ(bl[2], 0);  EXPECT_EQ(cl[2], 0);
}

TEST_F(Utf8Test, Position) {
    StringView hay[3] = {
        make_inline("hello world"),
        make_inline("nope"),
        make_inline("abcabc"),
    };
    int32_t pos[3];
    auto needle = make_inline("ab");
    ku::utf8_position(hay, 3, needle, pos);
    EXPECT_EQ(pos[0], 0);    // missing
    EXPECT_EQ(pos[1], 0);
    EXPECT_EQ(pos[2], 1);    // 1-based
    // Empty needle → position 1.
    auto empty = make_inline("");
    ku::utf8_position(hay, 1, empty, pos);
    EXPECT_EQ(pos[0], 1);
}

TEST_F(Utf8Test, Concat) {
    StringView a[2] = { make_inline("foo"), make_inline("hello_") };
    StringView b[2] = { make_inline("bar"), make_inline("world!!") };
    StringView out[2];
    auto anchor = reinterpret_cast<const char*>(&arena_);
    ku::utf8_concat(a, b, 2, out, &arena_, anchor);
    EXPECT_EQ(out[0].length, 6u);
    EXPECT_EQ(memcmp(out[0].prefix, "foob", 4), 0);
    EXPECT_EQ(out[0].inline_data[0], 'a');
    EXPECT_EQ(out[0].inline_data[1], 'r');
    EXPECT_EQ(out[1].length, 13u);  // spilled
    EXPECT_FALSE(out[1].is_inline());
}

TEST_F(Utf8Test, Replace) {
    StringView in[3] = {
        make_inline("ababab"),
        make_inline("nochange"),
        make_inline("xx"),
    };
    StringView out[3];
    auto anchor = reinterpret_cast<const char*>(&arena_);
    auto from = make_inline("a");
    auto to   = make_inline("Z");
    ku::utf8_replace(in, 3, from, to, out, &arena_, anchor);
    EXPECT_EQ(out[0].length, 6u);
    EXPECT_EQ(memcmp(out[0].prefix, "ZbZb", 4), 0);
    EXPECT_EQ(out[1].length, 8u);
    EXPECT_EQ(out[2].length, 2u);
    EXPECT_EQ(memcmp(out[2].prefix, "xx", 2), 0);
}

TEST_F(Utf8Test, ReplaceGrows) {
    StringView in[1] = { make_inline("aaa") };
    StringView out[1];
    auto anchor = reinterpret_cast<const char*>(&arena_);
    auto from = make_inline("a");
    auto to   = make_inline("ZZZZ");  // grows 3x
    ku::utf8_replace(in, 1, from, to, out, &arena_, anchor);
    EXPECT_EQ(out[0].length, 12u);  // 3 * 4 = 12 (still inline)
}

// ===========================================================================
// StringView::cmp_prefix + sv_bytelex_{min,max} — byte-lex semantics for
// typed GROUP BY MIN/MAX over Utf8 keys. Regression coverage for the
// "short string beats long string regardless of content" bug in the old
// length-first cmp_prefix shortcut.
// ===========================================================================
}  // namespace

#include "bolt/join/bolt_groupby.h"

namespace {

TEST_F(Utf8Test, CmpPrefix_ByteLexNotLengthFirst) {
    // "b" vs "aa" — byte-lex: "aa" < "b" because 'a' < 'b' on the first byte.
    // Old (broken) cmp_prefix returned 'b' < 'aa' (length-first: 1 < 2).
    auto sb  = make_inline("b");
    auto saa = make_inline("aa");
    EXPECT_LT(StringView::cmp_prefix(saa, sb), 0);
    EXPECT_GT(StringView::cmp_prefix(sb, saa), 0);

    // Equal-length prefixes that differ in content.
    auto a = make_inline("apple");
    auto b = make_inline("banana");
    EXPECT_LT(StringView::cmp_prefix(a, b), 0);

    // Identical 4-byte prefix, lengths differ → undecidable (0); caller
    // must fall back to a full byte compare on the spilled tail.
    auto p1 = make_inline("hello");
    auto p2 = make_inline("help");
    EXPECT_NE(StringView::cmp_prefix(p1, p2), 0);  // 5th byte not seen, but
                                                    // 4-byte prefix differs.

    auto same_prefix_short = make_inline("abcd");
    auto same_prefix_long  = make_inline("abcdef");
    EXPECT_EQ(StringView::cmp_prefix(same_prefix_short, same_prefix_long), 0);
}

// Helper: assert two StringViews are bit-equal (16-byte POD compare).
static void expect_sv_eq(const StringView& got, const StringView& want) {
    EXPECT_EQ(0, memcmp(&got, &want, sizeof(StringView)));
}

TEST_F(Utf8Test, SvBytelexMinMax_InlineByteLex) {
    auto apple  = make_inline("apple");
    auto banana = make_inline("banana");
    auto cherry = make_inline("cherry");

    StringView mn1 = bolt::sv_bytelex_min(banana, nullptr, apple,  nullptr);
    StringView mn2 = bolt::sv_bytelex_min(apple,  nullptr, banana, nullptr);
    StringView mx1 = bolt::sv_bytelex_max(apple,  nullptr, cherry, nullptr);
    expect_sv_eq(mn1, apple);
    expect_sv_eq(mn2, apple);
    expect_sv_eq(mx1, cherry);

    // Regression: short string vs long string with smaller first byte.
    // Byte-lex "aa" < "b"; the broken length-first shortcut said "b" < "aa".
    auto aa = make_inline("aa");
    auto b  = make_inline("b");
    StringView mn_ab = bolt::sv_bytelex_min(b,  nullptr, aa, nullptr);
    StringView mx_ab = bolt::sv_bytelex_max(aa, nullptr, b,  nullptr);
    expect_sv_eq(mn_ab, aa);
    expect_sv_eq(mx_ab, b);
}

TEST_F(Utf8Test, SvBytelexMinMax_TailDisambiguation) {
    // Identical 4-byte prefix; length differs → falls back to byte compare.
    // "abcd" vs "abcdef": "abcd" is a strict prefix → "abcd" < "abcdef".
    auto s4 = make_inline("abcd");
    auto s6 = make_inline("abcdef");
    StringView mn = bolt::sv_bytelex_min(s6, nullptr, s4, nullptr);
    StringView mx = bolt::sv_bytelex_max(s4, nullptr, s6, nullptr);
    expect_sv_eq(mn, s4);
    expect_sv_eq(mx, s6);
}

// Scalar reference: plain bytewise equality (no SIMD, no early width
// shortcut) — a kernel bug cannot paper over itself.
static bool ref_bytes_equal(const char* a, const char* b, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

// Bit-parity gate for bytes_equal_simd. Length 1021 is prime → exercises
// the SIMD body (16/32-lane) AND the scalar/memcmp tail. Random data with
// a single planted mismatch swept across both the body and the tail; the
// SIMD result must equal the all-scalar recompute at every position.
TEST_F(Utf8Test, BytesEqualSimdParityBodyAndTail) {
    constexpr uint32_t N = 1021;
    std::vector<char> a(N), b(N);
    std::mt19937 rng(0xB17E5);
    std::uniform_int_distribution<int> dist(0, 255);
    for (uint32_t i = 0; i < N; ++i) {
        a[i] = static_cast<char>(dist(rng));
        b[i] = a[i];                       // start fully equal
    }
    // Equal case.
    EXPECT_TRUE(ku::bytes_equal_simd(a.data(), b.data(), N));
    EXPECT_EQ(ref_bytes_equal(a.data(), b.data(), N), true);

    // Sweep a single-byte mismatch across body + tail boundaries.
    const uint32_t probes[] = {0u, 1u, 15u, 16u, 17u, 31u, 32u, 33u,
                               512u, 1008u, 1019u, 1020u};
    for (uint32_t p : probes) {
        const char saved = b[p];
        b[p] = static_cast<char>(saved ^ 0x01);     // flip one bit
        const bool got = ku::bytes_equal_simd(a.data(), b.data(), N);
        const bool ref = ref_bytes_equal(a.data(), b.data(), N);
        EXPECT_EQ(got, ref) << "mismatch at p=" << p;
        EXPECT_FALSE(got)   << "planted mismatch must be detected, p=" << p;
        b[p] = saved;                                // restore
    }
}

}  // namespace
