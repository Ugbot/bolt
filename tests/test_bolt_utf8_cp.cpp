// test_bolt_utf8_cp.cpp — code-point primitives and the string kernels built
// on them (substring, LIKE '_', GLOB '?', regex '.', upper/lower). Expected
// case mappings are DuckDB's upper()/lower().

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_types.h"
#include "bolt/kernels/bolt_regex.h"
#include "bolt/kernels/bolt_string.h"
#include "bolt/kernels/bolt_utf8.h"

using bolt::Arena;
using bolt::StringView;
namespace ku = bolt::kernels::utf8;
namespace rx = bolt::kernels::regex;

namespace {

// Independent strict validator (not the kernel's decoder).
bool valid_utf8(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t n;
        uint32_t cp;
        if (c < 0x80) { ++i; continue; }
        if (c >= 0xC2 && c <= 0xDF) { n = 2; cp = c & 0x1F; }
        else if (c >= 0xE0 && c <= 0xEF) { n = 3; cp = c & 0x0F; }
        else if (c >= 0xF0 && c <= 0xF4) { n = 4; cp = c & 0x07; }
        else return false;
        if (i + n > s.size()) return false;
        for (size_t k = 1; k < n; ++k) {
            const unsigned char d = static_cast<unsigned char>(s[i + k]);
            if ((d & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (d & 0x3F);
        }
        if ((n == 3 && cp < 0x800) || (n == 4 && (cp < 0x10000 || cp > 0x10FFFF))) return false;
        if (cp >= 0xD800 && cp <= 0xDFFF) return false;
        i += n;
    }
    return true;
}

// Units as the reference sees them: a byte plus following continuations.
std::vector<std::string> units(const std::string& s) {
    std::vector<std::string> u;
    for (size_t i = 0; i < s.size();) {
        size_t k = i + 1;
        if (static_cast<unsigned char>(s[i]) >= 0x80) {
            while (k < s.size() && (static_cast<unsigned char>(s[k]) & 0xC0) == 0x80) ++k;
        }
        u.push_back(s.substr(i, k - i));
        i = k;
    }
    return u;
}

const std::vector<std::string>& valid_corpus() {
    static const std::vector<std::string> c = {
        "", "a", "é", "日本語", "Ω≈ç√", "İi", "ß", "Straße", "ÉCOLE", "héllo",
        "emoji \xF0\x9F\x98\x80", "ay-aé'é_-", "é日acz", "Kelvin \xE2\x84\xAA",
        "ǅǆǄ ςΣσ", "Привет МИР", "ÿŸ ıI ſS", "abcdefghijklmnopqrstuvwxyz0123456789"};
    return c;
}

const std::vector<std::string>& invalid_corpus() {
    static const std::vector<std::string> c = {
        "\xC3", "\x80" "abc", "a\xC3(b", "\xE6\x97", "\xF8\x88\x80\x80\x80",
        "\xED\xA0\x80", "\xC0\xAF", "x\xE6\x97\xA5\xA5y", "\xBF\xBF"};
    return c;
}

std::string case_map(const std::string& s, bool upper) {
    std::string out(ku::utf8_case_map_bound(static_cast<uint32_t>(s.size())) + 1, '\0');
    const uint32_t m = ku::utf8_case_map(s.data(), static_cast<uint32_t>(s.size()), upper, nullptr);
    const uint32_t w = ku::utf8_case_map(s.data(), static_cast<uint32_t>(s.size()), upper, &out[0]);
    EXPECT_EQ(m, w);
    out.resize(w);
    return out;
}

std::string substr(const std::string& s, uint32_t start1, uint32_t len) {
    const char* p = s.data();
    const uint32_t n = static_cast<uint32_t>(s.size());
    const uint32_t b = ku::utf8_cp_advance(p, n, 0, start1 < 1 ? 0 : start1 - 1);
    const uint32_t e = ku::utf8_cp_advance(p, n, b, len);
    return s.substr(b, e - b);
}

TEST(Utf8Cp, CaseMapMatchesDuckDB) {
    struct C { const char* in; bool up; const char* want; } cases[] = {
        {"İi", false, "ii"}, {"ß", true, "ẞ"}, {"ẞ", false, "ß"},
        {"Ω≈ç√", false, "ω≈ç√"}, {"ay-aé'é_-", true, "AY-AÉ'É_-"},
        {"é日acz", true, "É日ACZ"}, {"ÉCOLE", false, "école"},
        {"ı", true, "I"}, {"ſ", true, "S"}, {"\xE2\x84\xAA", false, "k"},
        {"ǅ", true, "Ǆ"}, {"ǅ", false, "ǆ"}, {"ς", true, "Σ"}, {"Σ", false, "σ"},
        {"Привет МИР", true, "ПРИВЕТ МИР"}, {"Привет МИР", false, "привет мир"},
        {"ÿ", true, "Ÿ"}, {"Ÿ", false, "ÿ"}, {"straße", true, "STRAẞE"},
        {"Hello", true, "HELLO"}, {"日本語", true, "日本語"}, {"ɐ", true, "Ɐ"},
    };
    for (const C& c : cases) {
        EXPECT_EQ(case_map(c.in, c.up), c.want) << c.in << (c.up ? " upper" : " lower");
    }
}

TEST(Utf8Cp, CaseMapBoundHoldsForEveryTableEntry) {
    for (uint32_t cp = 0; cp < 0x2200; ++cp) {
        if (cp >= 0xD800 && cp <= 0xDFFF) continue;
        char enc[4];
        const uint32_t n = ku::utf8_cp_encode(cp, enc);
        const std::string s(enc, n);
        for (bool up : {true, false}) {
            const std::string m = case_map(s, up);
            EXPECT_LE(m.size(), ku::utf8_case_map_bound(n)) << std::hex << cp;
            EXPECT_TRUE(valid_utf8(m)) << std::hex << cp;
            EXPECT_EQ(units(m).size(), 1u) << std::hex << cp;
        }
    }
}

TEST(Utf8Cp, OutputsOfValidInputAreValid) {
    for (const std::string& s : valid_corpus()) {
        ASSERT_TRUE(valid_utf8(s)) << s;
        EXPECT_TRUE(valid_utf8(case_map(s, true))) << s;
        EXPECT_TRUE(valid_utf8(case_map(s, false))) << s;
        const std::vector<std::string> u = units(s);
        EXPECT_EQ(ku::utf8_count_codepoints(s.data(), static_cast<uint32_t>(s.size())),
                  static_cast<int32_t>(u.size()));
        for (uint32_t st = 0; st <= u.size() + 1; ++st) {
            for (uint32_t ln = 0; ln <= u.size() + 1; ++ln) {
                const std::string got = substr(s, st, ln);
                std::string want;
                const uint32_t s0 = st < 1 ? 0 : st - 1;
                for (uint32_t k = s0; k < s0 + ln && k < u.size(); ++k) want += u[k];
                EXPECT_EQ(got, want) << s << " " << st << " " << ln;
                EXPECT_TRUE(valid_utf8(got));
            }
        }
    }
}

TEST(Utf8Cp, InvalidInputStaysInBoundsAndIsCopiedThrough) {
    for (const std::string& s : invalid_corpus()) {
        EXPECT_FALSE(valid_utf8(s));
        const std::vector<std::string> u = units(s);
        EXPECT_EQ(ku::utf8_count_codepoints(s.data(), static_cast<uint32_t>(s.size())),
                  static_cast<int32_t>(u.size()));
        std::string all;
        for (uint32_t k = 1; k <= u.size(); ++k) all += substr(s, k, 1);
        EXPECT_EQ(all, s);
        // No valid letter in the corpus's invalid units, so mapping is identity
        // apart from ASCII.
        std::string lower_ascii = s;
        for (char& c : lower_ascii) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
        EXPECT_EQ(case_map(s, false), lower_ascii);
    }
}

TEST(Utf8Cp, TicketCases) {
    EXPECT_EQ(substr("héllo", 2, 3), "éll");
    EXPECT_EQ(substr("日本語", 2, 3), "本語");
    EXPECT_TRUE(ku::bytes_like("héllo", 6, "h_llo%", 6));
    EXPECT_TRUE(ku::bytes_like("日本語", 9, "___", 3));
    EXPECT_FALSE(ku::bytes_like("日本語", 9, "__", 2));
    EXPECT_FALSE(ku::bytes_like("日本語", 9, "____", 4));
    EXPECT_TRUE(ku::bytes_like("日本語", 9, "%_語", 5));
    EXPECT_TRUE(ku::bytes_like("日本語", 9, "_\xE6\x9C\xAC%", 5));
}

// Reference LIKE over units, recursive (inputs are tiny).
bool ref_like(const std::vector<std::string>& s, size_t si,
              const std::vector<std::string>& p, size_t pi) {
    if (pi == p.size()) return si == s.size();
    if (p[pi] == "%") {
        for (size_t k = si; k <= s.size(); ++k) if (ref_like(s, k, p, pi + 1)) return true;
        return false;
    }
    if (si == s.size()) return false;
    if (p[pi] == "_" || p[pi] == s[si]) return ref_like(s, si + 1, p, pi + 1);
    return false;
}

TEST(Utf8Cp, LikeMatchersAgreeWithUnitReferenceExhaustively) {
    const std::vector<std::string> alpha = {"a", "é", "日", "%", "_"};
    std::vector<std::string> words;
    for (int len = 0; len <= 4; ++len) {
        int total = 1;
        for (int k = 0; k < len; ++k) total *= 5;
        for (int idx = 0; idx < total; ++idx) {
            std::string w;
            int t = idx;
            for (int k = 0; k < len; ++k) { w += alpha[t % 5]; t /= 5; }
            words.push_back(w);
        }
    }
    int64_t checked = 0;
    for (const std::string& s : words) {
        const std::vector<std::string> su = units(s);
        for (const std::string& p : words) {
            if (units(p).size() > 3) continue;
            const bool want = ref_like(su, 0, units(p), 0);
            EXPECT_EQ(ku::bytes_like(s.data(), static_cast<uint32_t>(s.size()),
                                     p.data(), static_cast<uint32_t>(p.size())), want)
                << s << " LIKE " << p;
            ku::CompiledLike cl{};
            ASSERT_LE(p.size(), 12u);
            const StringView pv = ku::sv_make_inline(p.data(), static_cast<uint32_t>(p.size()));
            ASSERT_TRUE(ku::utf8_like_compile(pv, &cl));
            EXPECT_EQ(ku::utf8_like_match_one(&cl, s.data(), static_cast<uint32_t>(s.size())),
                      want) << s << " LIKE(compiled) " << p;
            ++checked;
        }
    }
    EXPECT_GT(checked, 100000);
}

TEST(Utf8Cp, SubstringColumnKernel) {
    Arena arena{};
    const std::string big = "日本語日本語";   // 18 bytes, spilled
    StringView in[3];
    in[0] = ku::sv_make_inline("héllo", 6);
    in[1] = ku::sv_make_inline("日本語", 9);
    std::memset(&in[2], 0, sizeof(in[2]));
    in[2].length = 18;
    std::memcpy(in[2].prefix, big.data(), 4);
    in[2].ref.offset = 0;
    StringView out[3];
    char* anchor = static_cast<char*>(arena.allocate(1, 1));
    ku::utf8_substring(in, 3, 2, 5, out, &arena, anchor, big.data());
    EXPECT_EQ(std::string(out[0].prefix, out[0].length), "éllo");
    EXPECT_EQ(std::string(out[1].prefix, out[1].length), "本語");
    ASSERT_EQ(out[2].length, 15u);
    EXPECT_EQ(std::string(anchor + out[2].ref.offset, 15), "本語日本語");
}

TEST(Utf8Cp, SubstrConstCountsCharacters) {
    StringView in[2] = {ku::sv_make_inline("日本語", 9), ku::sv_make_inline("abc", 3)};
    StringView out[2];
    ASSERT_TRUE(bolt::kernels::utf8_substr_const(in, 2, nullptr, 2, 2, out));
    EXPECT_EQ(std::string(out[0].prefix, out[0].length), "本語");
    EXPECT_EQ(std::string(out[1].prefix, out[1].length), "bc");
    // Five 3-byte characters do not fit inline: the kernel declines.
    const std::string big = "日本語日本語";
    StringView sp;
    std::memset(&sp, 0, sizeof(sp));
    sp.length = 18;
    std::memcpy(sp.prefix, big.data(), 4);
    EXPECT_FALSE(bolt::kernels::utf8_substr_const(&sp, 1, big.data(), 1, 5, out));
    ASSERT_TRUE(bolt::kernels::utf8_substr_const(&sp, 1, big.data(), 3, 2, out));
    EXPECT_EQ(std::string(out[0].prefix, out[0].length), "語日");
}

TEST(Utf8Cp, UpperLowerColumnKernelsGrowAndShrink) {
    Arena arena{};
    char* anchor = static_cast<char*>(arena.allocate(1, 1));
    StringView in[3] = {ku::sv_make_inline("straße", 7), ku::sv_make_inline("İİİİİİ", 12),
                        ku::sv_make_inline("ÉCOLE", 6)};
    StringView up[3], lo[3];
    ku::utf8_upper(in, 3, up, &arena, anchor);
    ku::utf8_lower(in, 3, lo, &arena, anchor);
    EXPECT_EQ(std::string(up[0].prefix, up[0].length), "STRAẞE");
    EXPECT_EQ(std::string(lo[1].prefix, lo[1].length), "iiiiii");
    EXPECT_EQ(std::string(lo[2].prefix, lo[2].length), "école");
    // "ßßßßßß" (12 bytes inline) upper-cases to 18 bytes: spilled output.
    StringView g = ku::sv_make_inline("ßßßßßß", 12);
    ku::utf8_upper(&g, 1, up, &arena, anchor);
    ASSERT_EQ(up[0].length, 18u);
    EXPECT_EQ(std::string(anchor + up[0].ref.offset, 18), "ẞẞẞẞẞẞ");
}

TEST(Utf8Cp, RegexDotIsOneCharacter) {
    rx::CompiledPattern cp;
    rx::MatchResult mr;
    ASSERT_TRUE(rx::regex_compile("^.$", 3, &cp));
    EXPECT_TRUE(rx::regex_search(&cp, "日", 3, &mr));
    EXPECT_FALSE(rx::regex_search(&cp, "日本", 6, &mr));
    ASSERT_TRUE(rx::regex_compile("^(.)(.)$", 8, &cp));
    ASSERT_TRUE(rx::regex_search(&cp, "é日", 5, &mr));
    EXPECT_EQ(mr.g_end[1] - mr.g_start[1], 2);
    EXPECT_EQ(mr.g_end[2] - mr.g_start[2], 3);
    ASSERT_TRUE(rx::regex_compile("^[^a]+$", 7, &cp));
    EXPECT_TRUE(rx::regex_search(&cp, "éé", 4, &mr));
    ASSERT_TRUE(rx::regex_compile("b.d", 3, &cp));
    ASSERT_TRUE(rx::regex_search(&cp, "ab日d", 6, &mr));
    EXPECT_EQ(mr.g_start[0], 1);
    EXPECT_EQ(mr.g_end[0], 6);
    EXPECT_FALSE(rx::regex_compile("[é]", 4, &cp));
    EXPECT_FALSE(rx::regex_compile("é+", 3, &cp));
    EXPECT_TRUE(rx::regex_compile("é", 2, &cp));
}

TEST(Utf8Cp, Re2DotIsOneCharacter) {
    bool ok = false;
    EXPECT_TRUE(rx::re2_compile_and_full_match(".", 1, "日", 3, &ok));
    EXPECT_TRUE(ok);
    EXPECT_FALSE(rx::re2_compile_and_full_match("..", 2, "日", 3, &ok));
    EXPECT_FALSE(rx::re2_compile_and_full_match("...", 3, "日", 3, &ok));
    EXPECT_TRUE(rx::re2_compile_and_full_match(".{3}", 4, "日本語", 9, &ok));
    EXPECT_FALSE(rx::re2_compile_and_full_match(".{4}", 4, "日本語", 9, &ok));
    EXPECT_FALSE(rx::re2_compile_and_full_match("a.b.c", 5, "aébc", 5, &ok));
    EXPECT_TRUE(rx::re2_compile_and_full_match("a.b.c", 5, "aéb😀c", 9, &ok));
    EXPECT_TRUE(rx::re2_compile_and_full_match("[^x]\\W\\S", 8, "éΩ日", 7, &ok));
    EXPECT_FALSE(rx::re2_compile_and_full_match("\\w", 2, "é", 2, &ok));
    EXPECT_TRUE(rx::re2_compile_and_full_match("h.llo", 5, "héllo", 6, &ok));
    EXPECT_TRUE(rx::re2_compile_and_full_match("(?:.|x)*", 8, "日本x語", 10, &ok));
    EXPECT_FALSE(rx::re2_compile_and_full_match("é+", 3, "éé", 4, &ok));
    EXPECT_FALSE(ok);
    EXPECT_FALSE(rx::re2_compile_and_full_match("[é]", 4, "é", 2, &ok));
    EXPECT_FALSE(ok);
    EXPECT_TRUE(rx::re2_compile_and_full_match("é日", 5, "é日", 5, &ok));
    EXPECT_TRUE(ok);
}

TEST(Utf8Cp, GlobIsCharacterAware) {
    auto g = [](const char* s, const char* p, int32_t esc = -1) {
        return ku::utf8_glob_match(s, static_cast<uint32_t>(std::strlen(s)),
                                   p, static_cast<uint32_t>(std::strlen(p)), esc);
    };
    EXPECT_TRUE(g("é", "?"));
    EXPECT_FALSE(g("é", "??"));
    EXPECT_TRUE(g("日本語", "?本?"));
    EXPECT_TRUE(g("日本語", "*語"));
    EXPECT_FALSE(g("日本語", "*本"));
    EXPECT_TRUE(g("a*b", "a\\*b", '\\'));
    EXPECT_FALSE(g("axb", "a\\*b", '\\'));
    EXPECT_TRUE(g("axb", "a*b", '\\'));
    EXPECT_TRUE(g("", "*"));
    EXPECT_FALSE(g("a", "a\\", '\\'));
}

TEST(Utf8Cp, SpecialCasingFlagsFullMappingDivergence) {
    EXPECT_TRUE(ku::utf8_has_special_casing("Straße", 7));
    EXPECT_TRUE(ku::utf8_has_special_casing("İi", 3));
    EXPECT_TRUE(ku::utf8_has_special_casing("ΟΔΟΣ", 8));
    EXPECT_FALSE(ku::utf8_has_special_casing("ÉCOLE école Ω日", 17));
    EXPECT_FALSE(ku::utf8_has_special_casing("plain", 5));
}

TEST(Utf8Cp, AsciiCheckAndAdvance) {
    const std::string s = "abcdefghijklmnopqrstuvwxyz0123456789é";
    EXPECT_TRUE(ku::utf8_is_ascii(s.data(), 36));
    EXPECT_FALSE(ku::utf8_is_ascii(s.data(), 38));
    EXPECT_EQ(ku::utf8_cp_advance(s.data(), 38, 0, 36), 36u);
    EXPECT_EQ(ku::utf8_cp_advance(s.data(), 38, 0, 37), 38u);
    EXPECT_EQ(ku::utf8_cp_advance(s.data(), 38, 0, 99), 38u);
    EXPECT_EQ(ku::utf8_unit_back(s.data(), 0, 38), 36u);
}

}  // namespace
