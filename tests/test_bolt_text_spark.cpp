// test_bolt_text_spark.cpp — Spark string-core kernels (bolt_text_spark.h, bolt_text_numfmt.h,
// bolt_url.h). Expected values are Spark 4.0.1's answers (frozen through chukonu's
// tests/data/spark_string_core_expected.tsv, which runs these kernels through the engine).

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "bolt/kernels/bolt_text_numfmt.h"
#include "bolt/kernels/bolt_text_spark.h"
#include "bolt/kernels/bolt_url.h"

namespace tx = bolt::kernels::text;
namespace ur = bolt::kernels::url;

namespace {
uint32_t L(const std::string& s) { return static_cast<uint32_t>(s.size()); }

std::string range(const std::string& s, uint32_t b, uint32_t e) { return s.substr(b, e - b); }

std::string substring_index(const std::string& s, const std::string& d, int32_t n) {
    uint32_t b = 0, e = 0;
    tx::substring_index(s.data(), L(s), d.data(), L(d), n, &b, &e);
    return range(s, b, e);
}
std::string split_part(const std::string& s, const std::string& d, int64_t n) {
    uint32_t b = 0, e = 0;
    tx::split_part(s.data(), L(s), d.data(), L(d), n, &b, &e);
    return range(s, b, e);
}
std::string trim_set(const std::string& s, const std::string& set, int mode) {
    uint32_t b = 0, e = 0;
    tx::trim_set(s.data(), L(s), set.data(), L(set), mode, &b, &e);
    return range(s, b, e);
}
std::string overlay(const std::string& s, const std::string& r, int32_t pos, int32_t len) {
    char out[256];
    const int64_t w = tx::overlay(s.data(), L(s), r.data(), L(r), pos, len, out, sizeof(out));
    return w < 0 ? "<err>" : std::string(out, static_cast<size_t>(w));
}
std::string translate(const std::string& s, const std::string& f, const std::string& t) {
    char out[256];
    const int64_t w = tx::translate(s.data(), L(s), f.data(), L(f), t.data(), L(t), out, sizeof(out));
    return w < 0 ? "<err>" : std::string(out, static_cast<size_t>(w));
}
int64_t lev(const std::string& a, const std::string& b, int64_t k = 0, bool has = false) {
    static uint32_t work[tx::levenshtein_scratch(tx::kEditMaxChars)];
    return tx::levenshtein(a.data(), L(a), b.data(), L(b), k, has, work);
}
std::string mask(const std::string& s, uint32_t u, uint32_t l, uint32_t d, uint32_t o) {
    char out[512];
    const int64_t w = tx::mask(s.data(), L(s), u, l, d, o, out, sizeof(out));
    return w < 0 ? "<err>" : std::string(out, static_cast<size_t>(w));
}
std::string fmt_f64(double v, uint32_t d) {
    char out[512];
    const int64_t w = tx::format_number_f64(v, d, out, sizeof(out));
    return w < 0 ? "<err>" : std::string(out, static_cast<size_t>(w));
}
std::string fmt_i64(int64_t v, uint32_t d) {
    char out[512];
    const int64_t w = tx::format_number_i64(v, d, out, sizeof(out));
    return w < 0 ? "<err>" : std::string(out, static_cast<size_t>(w));
}
std::string to_char(const std::string& fmt, bool neg, const std::string& ip, const std::string& fp) {
    tx::NumFmt f;
    if (tx::numfmt_parse(fmt.data(), L(fmt), &f) != tx::NumFmtErr::Ok) return "<fmt>";
    tx::NumDigits v{};
    v.neg = neg;
    std::memcpy(v.ip, ip.data(), ip.size()); v.ni = L(ip);
    std::memcpy(v.fp, fp.data(), fp.size()); v.nf = L(fp);
    char out[256];
    const int64_t w = tx::to_char(f, v, out, sizeof(out));
    return w < 0 ? "<err>" : std::string(out, static_cast<size_t>(w));
}
std::string to_number(const std::string& fmt, const std::string& in) {
    tx::NumFmt f;
    if (tx::numfmt_parse(fmt.data(), L(fmt), &f) != tx::NumFmtErr::Ok) return "<fmt>";
    bool neg = false;
    char d[128];
    const int64_t n = tx::to_number(f, in.data(), L(in), &neg, d);
    return n < 0 ? "<mismatch>" : std::string(neg ? "-" : "") + std::string(d, static_cast<size_t>(n));
}
std::string url_decode(const std::string& s) {
    char out[256];
    const int64_t w = ur::url_decode(s.data(), L(s), out, L(s));
    return w < 0 ? "<err>" : std::string(out, static_cast<size_t>(w));
}
}  // namespace

TEST(BoltTextSpark, PositionsAndSlices) {
    EXPECT_EQ(tx::locate("a", 1, "banana", 6, 2), 2);
    EXPECT_EQ(tx::locate("a", 1, "banana", 6, 0), 0);
    EXPECT_EQ(tx::locate("", 0, "banana", 6, 10), 1);
    EXPECT_EQ(tx::locate("l", 1, "h\xC3\xA9llo", 6, 1), 3);
    EXPECT_EQ(tx::ascii_first("\xC3\xA9", 2), 233);
    EXPECT_EQ(tx::ascii_first("", 0), 0);
    EXPECT_EQ(tx::ascii_first("\xFF" "A", 2), 65533);
    char c[4];
    EXPECT_EQ(tx::chr_encode(200, c), 2u);
    EXPECT_EQ(tx::chr_encode(321, c), 1u);
    EXPECT_EQ(c[0], 'A');
    EXPECT_EQ(tx::chr_encode(-1, c), 0u);
    EXPECT_EQ(tx::left_bytes("h\xC3\xA9llo", 6, 2), 3u);
    EXPECT_EQ(tx::right_start("h\xC3\xA9llo", 6, 3), 3u);
    EXPECT_EQ(tx::find_in_set("b", 1, "a,b,c", 5), 2);
    EXPECT_EQ(tx::find_in_set("b,c", 3, "a,b,c", 5), 0);
    EXPECT_EQ(tx::find_in_set("", 0, "a,,c", 4), 2);
}

TEST(BoltTextSpark, SubstringIndexAndSplitPart) {
    EXPECT_EQ(substring_index("a.b.c.d", ".", 2), "a.b");
    EXPECT_EQ(substring_index("a.b.c.d", ".", -2), "c.d");
    EXPECT_EQ(substring_index("a.b.c.d", ".", 10), "a.b.c.d");
    EXPECT_EQ(substring_index("aaa", "aa", -1), "");
    EXPECT_EQ(substring_index("aaaa", "aa", -2), "a");
    EXPECT_EQ(substring_index("abcabc", "bc", 1), "a");
    EXPECT_EQ(split_part("a,b,c", ",", -1), "c");
    EXPECT_EQ(split_part("a,b,c", ",", 4), "");
    EXPECT_EQ(split_part("a,b,c", "", 1), "a,b,c");
    EXPECT_EQ(split_part("a,,c", ",", 2), "");
}

TEST(BoltTextSpark, TrimOverlayTranslate) {
    EXPECT_EQ(trim_set("xyabcyx", "xy", 0), "abc");
    EXPECT_EQ(trim_set("xxabcxx", "x", 1), "abcxx");
    EXPECT_EQ(trim_set("xxabcxx", "x", 2), "xxabc");
    EXPECT_EQ(trim_set("\xC3\xA9" "a" "\xC3\xA9", "\xC3\xA9", 0), "a");
    EXPECT_EQ(overlay("Spark SQL", "_", 6, -1), "Spark_SQL");
    EXPECT_EQ(overlay("abc", "XY", 2, -1), "aXY");
    EXPECT_EQ(overlay("abc", "X", 0, -1), "Xabc");
    EXPECT_EQ(overlay("abc", "X", 10, -1), "abcX");
    EXPECT_EQ(translate("AaBbCc", "aabc", "1234"), "A1B3C4");
    EXPECT_EQ(translate("abc", "abc", ""), "");
}

TEST(BoltTextSpark, LevenshteinSoundexMask) {
    EXPECT_EQ(lev("kitten", "sitting"), 3);
    EXPECT_EQ(lev("kitten", "sitting", 2, true), -1);
    EXPECT_EQ(lev("kitten", "sitting", 3, true), 3);
    EXPECT_EQ(lev("", "abc"), 3);
    char sx[4];
    EXPECT_EQ(tx::soundex("Robert", 6, sx), 4);
    EXPECT_EQ(std::string(sx, 4), "R163");
    EXPECT_EQ(tx::soundex("123", 3, sx), -1);
    EXPECT_EQ(std::string(sx, 4).size(), 4u);
    EXPECT_EQ(mask("AbCd123-@$#", 'X', 'x', 'n', tx::kMaskKeep), "XxXxnnn-@$#");
    EXPECT_EQ(mask("\xF0\x9D\x90\x80", 'X', 'x', 'n', tx::kMaskKeep), "\xF0\x9D\x90\x80");
    EXPECT_EQ(mask("\xF0\x9D\x90\x80", 'X', 'x', 'n', '*'), "**");
    // Spark classes by Character.getType (Lu / Ll / Nd), not isUpperCase / isLowerCase: U+00AA and
    // U+2167 carry Other_Lowercase / Other_Uppercase only, so they are "other".
    EXPECT_EQ(mask("\xC2\xAA\xE2\x85\xA7", 'X', 'x', 'n', 'o'), "oo");
    EXPECT_EQ(mask("\xD9\xA3", 'X', 'x', 'n', 'o'), "n");
}

TEST(BoltTextSpark, Utf8Validation) {
    EXPECT_TRUE(tx::utf8_is_valid("h\xC3\xA9llo", 6));
    EXPECT_FALSE(tx::utf8_is_valid("\xED\xA0\x80", 3));
    EXPECT_FALSE(tx::utf8_is_valid("\xC0\x80", 2));
    char out[64];
    int64_t w = tx::utf8_make_valid("\xE0\xA0" "A", 3, out, sizeof(out));
    EXPECT_EQ(std::string(out, static_cast<size_t>(w)), "\xEF\xBF\xBD" "A");
    w = tx::utf8_make_valid("\xED\xA0\x80", 3, out, sizeof(out));
    EXPECT_EQ(w, 9);
}

TEST(BoltTextNumfmt, FormatNumber) {
    EXPECT_EQ(fmt_f64(1234567.891, 2), "1,234,567.89");
    EXPECT_EQ(fmt_f64(0.15, 1), "0.1");                  // the exact binary value is below the tie
    EXPECT_EQ(fmt_f64(0.125, 2), "0.12");                // an exact tie goes to even
    EXPECT_EQ(fmt_f64(0.005, 2), "0.01");                // a lone 5 above the tie rounds up ...
    EXPECT_EQ(fmt_f64(0.0005, 3), "0.000");              // ... except at 5e-n, n >= 4 (the JVM's DecimalFormat)
    EXPECT_EQ(fmt_f64(-0.00005, 4), "-0.0000");
    EXPECT_EQ(fmt_f64(-0.4, 0), "-0");
    EXPECT_EQ(fmt_f64(0.0 / 0.0, 2), "NaN");
    EXPECT_EQ(fmt_f64(1.0 / 0.0, 2), "\xE2\x88\x9E");
    EXPECT_EQ(fmt_f64(999.995, 2), "1,000.00");
    EXPECT_EQ(fmt_i64(-12345, 0), "-12,345");
    EXPECT_EQ(fmt_i64(5, 2), "5.00");
    char out[64];
    const int64_t w = tx::format_number_dec(0, 26750, 4, 2, out, sizeof(out));   // 2.6750 -> 2.68 (HALF_EVEN, exact)
    EXPECT_EQ(std::string(out, static_cast<size_t>(w)), "2.68");
}

TEST(BoltTextNumfmt, ToCharAndToNumber) {
    EXPECT_EQ(to_char("9999", false, "454", ""), " 454");
    EXPECT_EQ(to_char("0000", false, "454", ""), "0454");
    EXPECT_EQ(to_char("99", false, "454", ""), "##");
    EXPECT_EQ(to_char("S999", true, "454", ""), "-454");
    EXPECT_EQ(to_char("999PR", true, "454", ""), "<454>");
    EXPECT_EQ(to_char("9,999.99", false, "1234", "5"), "1,234.50");
    EXPECT_EQ(to_char("S99.9", false, "", "5"), "  +.5");
    EXPECT_EQ(to_char("99.9", false, "", "5"), " 0.5");
    EXPECT_EQ(to_number("99,999", "12,454"), "12454");
    EXPECT_EQ(to_number("99,999", "12454"), "<mismatch>");
    EXPECT_EQ(to_number("S99.9", "-12.3"), "-123");
    EXPECT_EQ(to_number("99PR", "<12>"), "-12");
    EXPECT_EQ(to_number("$99.99", "$78.12"), "7812");
    EXPECT_EQ(to_number("00", "1"), "<mismatch>");
    tx::NumFmt f;
    EXPECT_EQ(tx::numfmt_parse("", 0, &f), tx::NumFmtErr::Empty);
    EXPECT_EQ(tx::numfmt_parse("9,", 2, &f), tx::NumFmtErr::ContThousandsSeps);
    EXPECT_EQ(tx::numfmt_parse("9.9.9", 5, &f), tx::NumFmtErr::WrongNumToken);
    EXPECT_EQ(tx::numfmt_parse("9$", 2, &f), tx::NumFmtErr::CurMustBeforeDigit);
    EXPECT_EQ(tx::numfmt_parse("99X", 3, &f), tx::NumFmtErr::UnexpectedToken);
    EXPECT_EQ(tx::numfmt_parse("S", 1, &f), tx::NumFmtErr::WrongNumDigit);
}

TEST(BoltUrl, EncodeDecodeParse) {
    char out[128];
    int64_t w = ur::url_encode("a b&c=d/\xC3\xA9~*-._", 15, out, sizeof(out));
    EXPECT_EQ(std::string(out, static_cast<size_t>(w)), "a+b%26c%3Dd%2F%C3%A9%7E*-._");
    EXPECT_EQ(url_decode("a+b%26c%3Dd%2F%C3%A9"), "a b&c=d/\xC3\xA9");
    EXPECT_EQ(url_decode("%E6%97" "a"), "\xEF\xBF\xBD" "a");
    EXPECT_EQ(url_decode("%ED%A0%80"), "\xEF\xBF\xBD");                     // an encoded surrogate: one U+FFFD (JDK)
    EXPECT_EQ(url_decode("%E2%82"), "\xEF\xBF\xBD");
    EXPECT_EQ(url_decode("%C0%AF"), "\xEF\xBF\xBD\xEF\xBF\xBD");
    EXPECT_EQ(url_decode("%4"), "<err>");
    EXPECT_EQ(url_decode("%-1"), "<err>");
    const std::string u = "http://user:pw@example.com:8080/a/b?x=1&y=2#frag";
    uint32_t b = 0, e = 0;
    EXPECT_EQ(ur::parse_url(u.data(), L(u), ur::UrlPart::Host, nullptr, 0, false, &b, &e), 1);
    EXPECT_EQ(range(u, b, e), "example.com");
    EXPECT_EQ(ur::parse_url(u.data(), L(u), ur::UrlPart::File, nullptr, 0, false, &b, &e), 1);
    EXPECT_EQ(range(u, b, e), "/a/b?x=1&y=2");
    EXPECT_EQ(ur::parse_url(u.data(), L(u), ur::UrlPart::Query, "y", 1, true, &b, &e), 1);
    EXPECT_EQ(range(u, b, e), "2");
    const std::string bad = "http://exa mple.com/";
    EXPECT_EQ(ur::parse_url(bad.data(), L(bad), ur::UrlPart::Host, nullptr, 0, false, &b, &e), static_cast<int>(tx::kTextInvalid));
    const std::string reg = "http://ex_ample.com/";
    EXPECT_EQ(ur::parse_url(reg.data(), L(reg), ur::UrlPart::Host, nullptr, 0, false, &b, &e), 0);
    EXPECT_EQ(ur::parse_url(reg.data(), L(reg), ur::UrlPart::Authority, nullptr, 0, false, &b, &e), 1);
    const std::string v6 = "http://[2001:db8::1]:8080/";
    EXPECT_EQ(ur::parse_url(v6.data(), L(v6), ur::UrlPart::Host, nullptr, 0, false, &b, &e), 1);
    EXPECT_EQ(range(v6, b, e), "[2001:db8::1]");
}
