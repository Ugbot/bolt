// test_bolt_utf8.cpp — coverage for the UTF-8 native operator kernels.

#include <gtest/gtest.h>

#include <cmath>
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

// G2FEAT-341 — the inline masked-compare lane must be exact even when a
// producer leaves GARBAGE in an inline view's unused prefix/inline_data
// bytes (the mask, not zero-padding, carries correctness).
TEST_F(Utf8Test, FilterEqInlineGarbagePadding) {
    StringView data[4] = {
        make_inline("ZTEK"),
        make_inline("ZETA"),
        make_inline("ZTEK"),
        make_inline("ZT"),      // shorter — length leg must reject
    };
    // Poison every unused byte past each view's logical length.
    for (auto& v : data) {
        for (uint32_t b = v.length; b < 4u; ++b) v.prefix[b] = char(0xA5);
        const uint32_t tl = (v.length > 4u) ? v.length - 4u : 0u;
        for (uint32_t b = tl; b < 8u; ++b) v.inline_data[b] = char(0x5A);
    }
    auto needle = make_inline("ZTEK");
    for (uint32_t b = 0; b < 8u; ++b) needle.inline_data[b] = char(0x77);
    int32_t sel[4];
    int64_t c = ku::utf8_filter_eq(data, 4, needle, sel);
    ASSERT_EQ(c, 2);
    EXPECT_EQ(sel[0], 0);
    EXPECT_EQ(sel[1], 2);
    // Boundary lengths: 4 (prefix-only), 12 (full inline), 0 (empty).
    StringView d2[3] = { make_inline("ABCD"), make_inline("ABCDEFGHIJKL"),
                         make_inline("") };
    int64_t c4 = ku::utf8_filter_eq(d2, 3, make_inline("ABCD"), sel);
    ASSERT_EQ(c4, 1);
    EXPECT_EQ(sel[0], 0);
    int64_t c12 = ku::utf8_filter_eq(d2, 3, make_inline("ABCDEFGHIJKL"), sel);
    ASSERT_EQ(c12, 1);
    EXPECT_EQ(sel[0], 1);
    int64_t c0 = ku::utf8_filter_eq(d2, 3, make_inline(""), sel);
    ASSERT_EQ(c0, 1);
    EXPECT_EQ(sel[0], 2);
}

// G2FEAT-341 — sparse (_selected) variants: same results as the dense
// kernels restricted to the selection, with sel_out carrying ROW ids.
TEST_F(Utf8Test, FilterEqNeSelected) {
    StringView data[6] = {
        make_inline("L"),  make_inline("T"), make_inline("L"),
        make_inline("N"),  make_inline("L"), make_inline("LL"),
    };
    const int32_t sel_in[4] = {0, 2, 3, 5};   // window skips rows 1 and 4
    auto needle = make_inline("L");
    int32_t sel_out[6];
    int64_t c = ku::utf8_filter_eq_selected(data, sel_in, 4, needle, sel_out);
    ASSERT_EQ(c, 2);
    EXPECT_EQ(sel_out[0], 0);
    EXPECT_EQ(sel_out[1], 2);
    c = ku::utf8_filter_ne_selected(data, sel_in, 4, needle, sel_out);
    ASSERT_EQ(c, 2);
    EXPECT_EQ(sel_out[0], 3);
    EXPECT_EQ(sel_out[1], 5);
}

TEST_F(Utf8Test, FilterEqSelectedSpilledScalar) {
    const char buf[] = "AAAAAAAAAAAAAAAA"    // 16
                       "BBBBBBBBBBBBBBBB"
                       "AAAAAAAAAAAAAAAA";
    StringView data[3] = {
        make_spilled(buf, 16, 0),
        make_spilled(buf + 16, 16, 16),
        make_spilled(buf + 32, 16, 32),
    };
    const int32_t sel_in[3] = {0, 1, 2};
    auto needle = make_spilled(buf, 16, 0);
    int32_t sel_out[3];
    int64_t c = ku::utf8_filter_eq_selected(data, sel_in, 3, needle, sel_out,
                                            buf, buf);
    ASSERT_EQ(c, 2);
    EXPECT_EQ(sel_out[0], 0);
    EXPECT_EQ(sel_out[1], 2);
    c = ku::utf8_filter_ne_selected(data, sel_in, 3, needle, sel_out, buf, buf);
    ASSERT_EQ(c, 1);
    EXPECT_EQ(sel_out[0], 1);
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

// ===========================================================================
// utf8_substr_const (bolt_string.h) — vectorized fixed-window SUBSTRING.
// ===========================================================================
}  // namespace

#include "bolt/kernels/bolt_string.h"

namespace {

TEST_F(Utf8Test, SubstrConst_PrefixInline) {
    // The TPC-H Q22 shape: substring(x, 1, 2) — country code from a phone.
    StringView in[4] = {
        make_inline("13-456-789"),  // -> "13"
        make_inline("31-000"),      // -> "31"
        make_inline("7"),           // start past end of the 2-window -> "7"
        make_inline(""),            // empty -> empty
    };
    StringView out[4];
    bolt::kernels::utf8_substr_const(in, 4, nullptr, /*start1=*/1, /*take=*/2, out);
    EXPECT_EQ(out[0].length, 2u);
    EXPECT_EQ(memcmp(out[0].prefix, "13", 2), 0);
    EXPECT_EQ(out[1].length, 2u);
    EXPECT_EQ(memcmp(out[1].prefix, "31", 2), 0);
    EXPECT_EQ(out[2].length, 1u);             // saturated at remaining bytes
    EXPECT_EQ(out[2].prefix[0], '7');
    EXPECT_EQ(out[3].length, 0u);             // empty source -> empty
}

TEST_F(Utf8Test, SubstrConst_MidWindowAndClamp) {
    StringView in[3] = {
        make_inline("abcdefgh"),    // substring(.,3,4) -> "cdef"
        make_inline("ab"),          // start 3 past end -> empty
        make_inline("abcdefgh"),    // substring(.,7,9) -> "gh" (saturate)
    };
    StringView o1[3], o2[3];
    bolt::kernels::utf8_substr_const(in, 3, nullptr, 3, 4, o1);
    EXPECT_EQ(o1[0].length, 4u);
    EXPECT_EQ(memcmp(o1[0].prefix, "cdef", 4), 0);
    EXPECT_EQ(o1[1].length, 0u);
    bolt::kernels::utf8_substr_const(in, 3, nullptr, 7, 9, o2);
    EXPECT_EQ(o2[2].length, 2u);
    EXPECT_EQ(memcmp(o2[2].prefix, "gh", 2), 0);
}

TEST_F(Utf8Test, SubstrConst_SpilledSourcePrefixWindow) {
    // A spilled (>12B) source: a window inside the first 12 bytes must read
    // the inline prefix WITHOUT consulting the spill base (the inline-fast
    // path). prefix[] holds the first 4 bytes; the window [1,2] is in range.
    const char* big = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";   // 26 bytes
    StringView sp = make_spilled(big, 26, 0);
    StringView out[1];
    // start1=1 take=2 -> "AB", entirely in the 4-byte prefix, base unused.
    bolt::kernels::utf8_substr_const(&sp, 1, /*spilled_base=*/nullptr, 1, 2, out);
    EXPECT_EQ(out[0].length, 2u);
    EXPECT_EQ(memcmp(out[0].prefix, "AB", 2), 0);
}

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


// ===========================================================================
// G2FEAT-146 -- bytes_like must test '%' BEFORE the literal/`_` branch.
//
// `p[pi] == s[si]` is ALSO true when the haystack byte is itself a literal
// '%'. With the literal branch first, the pattern's WILDCARD gets consumed as
// a literal match, the star is never recorded, and the matcher can no longer
// backtrack -- it scans to the end and reports no match.
//
// Only haystacks containing a literal '%' are affected, which is why
// URL-encoded data exposed it and TPC-H/TAQ never did. A fuzzer whose HAYSTACK
// alphabet excludes '%' cannot find this: the original 400k-case sweep over
// {a,b} passed clean. The alphabet below deliberately includes both wildcards.
// ===========================================================================

// Reference matcher: exhaustive, obviously-correct, no backtracking cleverness.
bool ref_like(const char* s, uint32_t slen,
              const char* p, uint32_t plen) noexcept {
    if (plen == 0) return slen == 0;
    if (p[0] == '%') {
        for (uint32_t k = 0; k <= slen; ++k) {
            if (ref_like(s + k, slen - k, p + 1, plen - 1)) return true;
        }
        return false;
    }
    if (slen == 0) return false;
    if (p[0] != '_' && p[0] != s[0]) return false;
    return ref_like(s + 1, slen - 1, p + 1, plen - 1);
}

TEST_F(Utf8Test, BytesLike_WildcardBeatsLiteralPercentInHaystack) {
    // The exact ClickBench row: the byte after `.ru` is a literal '%'.
    const char* u = "https://produkty%2Fbonprix.ru%2Fkurtka-a-datch2867&pvno=2";
    const uint32_t ulen = static_cast<uint32_t>(std::strlen(u));

    EXPECT_TRUE(ku::bytes_like(u, ulen, "%.ru%", 5))
        << "pre-fix this returned false: the trailing '%' matched the "
           "haystack's literal '%' and destroyed the backtrack point";

    // Controls that passed even PRE-fix -- they pin that the fix did not
    // simply loosen matching. Both differ from the failing case only in
    // whether a '%' sits where the pattern's wildcard is tested.
    EXPECT_TRUE(ku::bytes_like(u, ulen, "%.r%", 4));        // next byte is 'u'
    EXPECT_TRUE(ku::bytes_like(u, ulen, "%.ru%2F%", 8));    // wildcard consumed, then matches
    EXPECT_TRUE(ku::bytes_like(u, ulen, "%bonprix%", 9));
    EXPECT_FALSE(ku::bytes_like(u, ulen, "%.rux%", 6));     // genuinely absent
    EXPECT_FALSE(ku::bytes_like(u, ulen, "%zzzz%", 6));

    // A literal '%' must still be matchable BY a literal '%' in the pattern
    // where that is the only reading -- `_` forces one byte, so the '%' that
    // follows has to match the haystack's '%' as a literal.
    EXPECT_TRUE(ku::bytes_like("a%b", 3, "a%b", 3));
    EXPECT_TRUE(ku::bytes_like("a%b", 3, "_%_", 3));
}

TEST_F(Utf8Test, BytesLike_ExhaustiveVsReferenceWithWildcardsInHaystack) {
    // Every string up to length 5 and every pattern up to length 4 over an
    // alphabet that INCLUDES both wildcard characters. This is the coverage
    // the original sweep lacked.
    const char alpha[] = {'a', 'b', '%', '_'};
    constexpr int kA = 4;
    char sbuf[6], pbuf[5];
    std::int64_t checked = 0, mismatches = 0;
    for (uint32_t sl = 0; sl <= 5u; ++sl) {
        const std::int64_t sn = static_cast<std::int64_t>(std::pow(kA, sl));
        for (std::int64_t sc = 0; sc < sn; ++sc) {
            std::int64_t t = sc;
            for (uint32_t i = 0; i < sl; ++i) { sbuf[i] = alpha[t % kA]; t /= kA; }
            for (uint32_t pl = 0; pl <= 4u; ++pl) {
                const std::int64_t pn = static_cast<std::int64_t>(std::pow(kA, pl));
                for (std::int64_t pc = 0; pc < pn; ++pc) {
                    std::int64_t v = pc;
                    for (uint32_t i = 0; i < pl; ++i) { pbuf[i] = alpha[v % kA]; v /= kA; }
                    const bool got = ku::bytes_like(sbuf, sl, pbuf, pl);
                    const bool want = ref_like(sbuf, sl, pbuf, pl);
                    ++checked;
                    if (got != want) {
                        ++mismatches;
                        if (mismatches <= 3) {
                            ADD_FAILURE() << "s='" << std::string(sbuf, sl)
                                          << "' p='" << std::string(pbuf, pl)
                                          << "' got=" << got << " want=" << want;
                        }
                    }
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0);
    EXPECT_GT(checked, 1000000) << "the sweep must actually be exhaustive";
}


// ===========================================================================
// The COMPILED LIKE path (utf8_like_compile + utf8_like_match_one) is what
// chukonu's Utf8 filter now dispatches through, because the general
// backtracking matcher was the leaf frame in a profile of ClickBench Q21
// (`count(*) WHERE url LIKE '%google%'` over 100M rows). Two implementations
// of one semantic is exactly how the two engines in bolt_regex.h drifted
// apart, so pin them together: the compiled path must agree with bytes_like
// on EVERY case, over an alphabet that includes the wildcards themselves.
// ===========================================================================

bool compiled_like(const char* p, uint32_t pl, const char* s, uint32_t sl) {
    ku::CompiledLike cl{};
    bolt::StringView pv{};
    const char* base = nullptr;
    if (pl <= 12u) {
        pv = ku::sv_make_inline(p, pl);
    } else {
        pv.length = pl; pv.ref.buf_idx = 0; pv.ref.offset = 0; base = p;
    }
    if (!ku::utf8_like_compile(pv, &cl, base)) return ku::bytes_like(s, sl, p, pl);
    return ku::utf8_like_match_one(&cl, s, sl);
}

TEST_F(Utf8Test, CompiledLikeAgreesWithGeneralMatcherExhaustively) {
    const char alpha[] = {'a', 'b', '%', '_'};
    constexpr int kA = 4;
    char sb[6], pb[5];
    std::int64_t checked = 0, mism = 0;
    for (uint32_t sl = 0; sl <= 5u; ++sl) {
        const std::int64_t sn = static_cast<std::int64_t>(std::pow(kA, sl));
        for (std::int64_t sc = 0; sc < sn; ++sc) {
            std::int64_t t = sc;
            for (uint32_t i = 0; i < sl; ++i) { sb[i] = alpha[t % kA]; t /= kA; }
            for (uint32_t pl = 0; pl <= 4u; ++pl) {
                const std::int64_t pn = static_cast<std::int64_t>(std::pow(kA, pl));
                for (std::int64_t pc = 0; pc < pn; ++pc) {
                    std::int64_t v = pc;
                    for (uint32_t i = 0; i < pl; ++i) { pb[i] = alpha[v % kA]; v /= kA; }
                    const bool a = ku::bytes_like(sb, sl, pb, pl);
                    const bool b = compiled_like(pb, pl, sb, sl);
                    ++checked;
                    if (a != b) {
                        ++mism;
                        if (mism <= 3) {
                            ADD_FAILURE() << "s='" << std::string(sb, sl)
                                          << "' p='" << std::string(pb, pl)
                                          << "' general=" << a << " compiled=" << b;
                        }
                    }
                }
            }
        }
    }
    EXPECT_EQ(mism, 0);
    EXPECT_GT(checked, 400000);
}

TEST_F(Utf8Test, BytesFindMemchrScanMatchesNaive) {
    // bytes_find's first-byte scan is memchr, not a byte loop. Same answers,
    // including the last legal start position and needles that appear only as
    // a prefix of a longer near-match.
    auto naive = [](const char* h, uint32_t hl, const char* n, uint32_t nl) {
        if (nl == 0) return 0;
        if (nl > hl) return -1;
        for (uint32_t k = 0; k + nl <= hl; ++k)
            if (std::memcmp(h + k, n, nl) == 0) return static_cast<int>(k);
        return -1;
    };
    const char* hays[] = {"", "a", "abcabcabd", "aaaaa", "http://a.ru/b",
                          "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxab"};
    const char* ndls[] = {"", "a", "ab", "abd", "aaa", ".ru", "b", "zz", "ab"};
    for (const char* h : hays) {
        for (const char* n : ndls) {
            const uint32_t hl = static_cast<uint32_t>(std::strlen(h));
            const uint32_t nl = static_cast<uint32_t>(std::strlen(n));
            EXPECT_EQ(ku::bytes_find(h, hl, n, nl), naive(h, hl, n, nl))
                << "h='" << h << "' n='" << n << "'";
        }
    }
}

}  // namespace
