// test_bolt_text_search.cpp — word / phrase / regex-search column kernels,
// differential against naive references over random strings.

#include "bolt/kernels/bolt_text_search.h"

#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <vector>

using namespace bolt;
using namespace bolt::kernels::text;
namespace rx = bolt::kernels::regex;

namespace {

bool wb(char ch) { const unsigned char c = static_cast<unsigned char>(ch); return (std::isalnum(static_cast<int>(c)) && c < 0x80) || c == '_'; }

// Reference: every occurrence of needle, boundary-checked, plain loops.
bool ref_phrase(const std::string& h, const std::string& n, bool ci) {
    if (n.empty() || h.size() < n.size()) return false;
    for (size_t i = 0; i + n.size() <= h.size(); ++i) {
        bool eq = true;
        for (size_t j = 0; j < n.size() && eq; ++j) {
            unsigned char a = static_cast<unsigned char>(h[i + j]), b = static_cast<unsigned char>(n[j]);
            if (ci) { a = (a >= 'A' && a <= 'Z') ? static_cast<unsigned char>(a + 32) : a; b = (b >= 'A' && b <= 'Z') ? static_cast<unsigned char>(b + 32) : b; }
            eq = a == b;
        }
        if (!eq) continue;
        if (wb(n.front()) && i > 0 && wb(h[i - 1])) continue;
        const size_t e = i + n.size();
        if (wb(n.back()) && e < h.size() && wb(h[e])) continue;
        return true;
    }
    return false;
}

// Reference regex search: some substring fully matches.
bool ref_regex(const rx::Re2Program& p, const std::string& h) {
    for (size_t i = 0; i <= h.size(); ++i)
        for (size_t j = i; j <= h.size(); ++j)
            if (rx::re2_full_match(&p, h.data() + i, static_cast<int32_t>(j - i))) return true;
    return false;
}

std::string rnd_string(std::mt19937& g, const char* alpha, size_t alen, size_t maxlen) {
    std::string s;
    const size_t n = g() % (maxlen + 1);
    for (size_t i = 0; i < n; ++i) s.push_back(alpha[g() % alen]);
    return s;
}

struct Col {
    std::vector<std::string> rows;
    std::vector<StringView> sv;
    std::vector<char> spill;
    // Build StringViews: inline when <= 12 bytes, else spilled.
    void build() {
        sv.assign(rows.size(), StringView{});
        size_t total = 0;
        for (auto& r : rows) if (r.size() > 12) total += r.size();
        spill.assign(total + 1, 0);
        size_t at = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            const std::string& r = rows[i];
            std::memset(&sv[i], 0, sizeof(StringView));
            sv[i].length = static_cast<uint32_t>(r.size());
            if (r.size() <= 12) {
                std::memcpy(sv[i].prefix, r.data(), r.size() > 4 ? 4 : r.size());
                if (r.size() > 4) std::memcpy(sv[i].inline_data, r.data() + 4, r.size() - 4);
            } else {
                std::memcpy(spill.data() + at, r.data(), r.size());
                std::memcpy(sv[i].prefix, r.data(), 4);
                sv[i].ref.buf_idx = 0;
                sv[i].ref.offset = static_cast<uint32_t>(at);
                at += r.size();
            }
        }
    }
};

}  // namespace

TEST(TextSearch, CompileRefusals) {
    TokenPattern p;
    EXPECT_FALSE(token_compile("", 0, false, &p));
    std::string big(kTokenMaxNeedle + 1, 'a');
    EXPECT_FALSE(token_compile(big.data(), static_cast<uint32_t>(big.size()), false, &p));
    EXPECT_TRUE(token_compile("gif", 3, false, &p));
    rx::Re2Program prog;
    RegexSearchPattern rp;
    EXPECT_FALSE(regex_search_compile("(a", 2, &rp));
    (void)prog;
}

TEST(TextSearch, KnownAnswers) {
    auto m = [](const char* n, const char* h, bool ci = false) {
        TokenPattern p;
        EXPECT_TRUE(token_compile(n, static_cast<uint32_t>(std::strlen(n)), ci, &p));
        return token_match_one(&p, h, static_cast<uint32_t>(std::strlen(h)));
    };
    EXPECT_TRUE(m("gif", "a.gif"));
    EXPECT_TRUE(m("gif", "gif"));
    EXPECT_FALSE(m("gif", "gifs"));
    EXPECT_FALSE(m("gif", "xgif"));
    EXPECT_FALSE(m("gif", "a_gif"));
    EXPECT_TRUE(m("gif", "\xC3\xA9gif"));          // non-ASCII bytes separate
    EXPECT_TRUE(m("GIF", "a.gif", true));
    EXPECT_FALSE(m("GIF", "a.gif", false));
    EXPECT_TRUE(m("images/hm", "GET images/hm HTTP"));
    EXPECT_FALSE(m("images/hm", "GET images/hm_bg.gif"));
    EXPECT_FALSE(m("images hm", "images/hm"));      // separators must match
    EXPECT_FALSE(m("/x", "a/xb"));
    EXPECT_TRUE(m("/x", "a/x b"));
    EXPECT_TRUE(m("/", "a/b"));                     // a non-word end needs no boundary
}

TEST(TextSearch, TokenDifferential) {
    std::mt19937 g(7);
    const char alpha[] = {'a', 'b', 'A', 'B', '_', '/', ' ', '.', '1', '\xC3', '\xA9'};
    for (int iter = 0; iter < 300; ++iter) {
        Col col;
        for (int i = 0; i < 64; ++i) col.rows.push_back(rnd_string(g, alpha, sizeof(alpha), iter % 3 == 0 ? 40 : 10));
        col.rows.push_back("");
        std::string nd = rnd_string(g, alpha, sizeof(alpha), 4);
        if (nd.empty()) nd = "a";
        // force needle at start and end in some rows
        col.rows.push_back(nd);
        col.rows.push_back(nd + " tail");
        col.rows.push_back("head " + nd);
        col.rows.push_back(nd.substr(0, nd.size() > 1 ? nd.size() - 1 : 0));   // value shorter than needle
        col.build();
        for (bool ci : {false, true}) {
            TokenPattern p;
            ASSERT_TRUE(token_compile(nd.data(), static_cast<uint32_t>(nd.size()), ci, &p));
            std::vector<int32_t> out(col.rows.size()), sel_out(col.rows.size());
            const int64_t cnt = utf8_filter_token(col.sv.data(), nullptr, static_cast<int64_t>(col.rows.size()), &p,
                                                  out.data(), col.spill.data());
            std::vector<int32_t> want;
            for (size_t i = 0; i < col.rows.size(); ++i) if (ref_phrase(col.rows[i], nd, ci)) want.push_back(static_cast<int32_t>(i));
            ASSERT_EQ(static_cast<size_t>(cnt), want.size()) << "needle=" << nd << " ci=" << ci;
            for (size_t i = 0; i < static_cast<size_t>(cnt); ++i) ASSERT_EQ(out[i], want[i]);
            // selection-vector input: every third row
            std::vector<int32_t> sin;
            for (size_t i = 0; i < col.rows.size(); i += 3) sin.push_back(static_cast<int32_t>(i));
            const int64_t c2 = utf8_filter_token(col.sv.data(), sin.data(), static_cast<int64_t>(sin.size()), &p,
                                                 sel_out.data(), col.spill.data());
            std::vector<int32_t> w2;
            for (int32_t r : sin) if (ref_phrase(col.rows[static_cast<size_t>(r)], nd, ci)) w2.push_back(r);
            ASSERT_EQ(static_cast<size_t>(c2), w2.size());
            for (size_t i = 0; i < static_cast<size_t>(c2); ++i) ASSERT_EQ(sel_out[i], w2[i]);
        }
    }
}

TEST(TextSearch, RegexDifferential) {
    const char* pats[] = {"hm_bg[.](jpg|gif)", "a+b", "(?i)ab", "x|yy", "a.c", "[a-c]{2}d",
                          "images/hm", "\\d+", "a*", "(ab)+c", "b\\.gif"};
    std::mt19937 g(11);
    const char alpha[] = {'a', 'b', 'c', 'd', 'x', 'y', '.', '/', '1', '_', 'h', 'm', 'g', 'i', 'f', 'A', 'B', '\n', '\xC3', '\xA9'};
    for (const char* pat : pats) {
        RegexSearchPattern rp;
        ASSERT_TRUE(regex_search_compile(pat, static_cast<uint32_t>(std::strlen(pat)), &rp)) << pat;
        Col col;
        for (int i = 0; i < 200; ++i) col.rows.push_back(rnd_string(g, alpha, sizeof(alpha), 24));
        col.rows.push_back("");
        col.rows.push_back("GET /images/hm_bg.gif HTTP/1.0");
        col.rows.push_back("hm_bg.jpg");
        col.rows.push_back("hm_bg.png");
        col.build();
        std::vector<int32_t> out(col.rows.size());
        const int64_t cnt = utf8_filter_regex_search(col.sv.data(), nullptr, static_cast<int64_t>(col.rows.size()), &rp,
                                                     out.data(), col.spill.data());
        std::vector<int32_t> want;
        for (size_t i = 0; i < col.rows.size(); ++i) if (ref_regex(rp.prog, col.rows[i])) want.push_back(static_cast<int32_t>(i));
        ASSERT_EQ(static_cast<size_t>(cnt), want.size()) << pat;
        for (size_t i = 0; i < static_cast<size_t>(cnt); ++i) ASSERT_EQ(out[i], want[i]) << pat;
    }
}

TEST(TextSearch, RegexAnchorsAndPrefilter) {
    auto m = [](const char* pat, const char* h) {
        RegexSearchPattern rp;
        EXPECT_TRUE(regex_search_compile(pat, static_cast<uint32_t>(std::strlen(pat)), &rp));
        return regex_search_one(&rp, h, static_cast<uint32_t>(std::strlen(h)));
    };
    EXPECT_TRUE(m("^ab", "abc"));
    EXPECT_FALSE(m("^ab", "cab"));
    EXPECT_TRUE(m("ab$", "cab"));
    EXPECT_FALSE(m("ab$", "abc"));
    EXPECT_TRUE(m("(?i)AB", "xab"));
    EXPECT_TRUE(m("hm_bg[.](jpg|gif)", "GET /hm_bg.gif"));
    EXPECT_FALSE(m("hm_bg[.](jpg|gif)", "GET /hm_bg.png"));
}

TEST(TextSearch, RequiredLiteral) {
    auto lit = [](const char* p) {
        char b[kRegexLiteralMax];
        const uint32_t n = regex_required_literal(p, static_cast<uint32_t>(std::strlen(p)), b, sizeof(b));
        return std::string(b, n);
    };
    EXPECT_EQ(lit("hm_bg[.](jpg|gif)"), "hm_bg");
    EXPECT_EQ(lit("abc"), "abc");
    EXPECT_EQ(lit("a|b"), "");
    EXPECT_EQ(lit("(?i)abc"), "");
    EXPECT_EQ(lit("ab*cdef"), "cdef");
    EXPECT_EQ(lit("ab+c"), "ab");
    EXPECT_EQ(lit("x\\.yz"), "x.yz");
    EXPECT_EQ(lit("[ab]cd"), "cd");
    EXPECT_EQ(lit("a{2,3}xy"), "xy");
}
