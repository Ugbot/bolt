// test_bolt_regex_backtrack.cpp — the capture-group backtracking engine
// (regex_compile / regex_search / regex_substitute) against DuckDB 1.5.5
// (RE2) answers for the same pattern and text (G2CHK-341).

#include "bolt/kernels/bolt_regex.h"

#include <cstring>
#include <string>
#include <gtest/gtest.h>

using namespace bolt::kernels::regex;

namespace {

struct Case { const char* pat; const char* text; bool want; };

// DuckDB: SELECT regexp_matches(text, pat).
const Case kMatch[] = {
    {"^a{3}$", "aaa", true},        {"^a{3}$", "a{3}", false},
    {"^a{3}$", "aa", false},        {"^a{3}$", "aaaa", false},
    {"^a{2,}$", "aaaa", true},      {"^a{2,}$", "a", false},
    {"^a{2,3}$", "aaaa", false},    {"^a{2,3}$", "aaa", true},
    {"^a{0}$", "", true},           {"^a{0}b$", "b", true},
    {"a+?b", "aab", true},          {"\\d+", "x12", true},
    {"^\\d{3}-\\d{4}$", "555-1234", true}, {"^\\d{3}-\\d{4}$", "55-1234", false},
    {"[\\d]", "5", true},           {"[\\d]", "x", false},
    {"^\\w+$", "ab_9", true},       {"^\\w+$", "a-b", false},
    {"^\\s$", " ", true},           {"^\\s$", "\v", false},
    {"\\D", "123", false},          {"\\S", "  x", true},
    {"^(ab){2}$", "abab", true},    {"^(ab){2}$", "ab", false},
    {"^(ab){2,}$", "ababab", true}, {"^(?:ab){1,2}c$", "ababc", true},
    {"^(?:ab){1,2}c$", "abababc", false},
    {"a{", "a{", true},             {"a{,2}", "a{,2}", true},
    {"a}", "a}", true},             {"\\{", "{", true},
    {"x{a}", "x{a}", true},
    {"^[a-c]{2}\\.[^.]{1,3}$", "ab.xyz", true},
    {"^[a-c]{2}\\.[^.]{1,3}$", "ab.xyzw", false},
    {"\\n", "a\nb", true},          {"\\t", "a\tb", true},
    {"^.{3}$", "日本語", true},      {"^(.){2}$", "日本", true},
    {"[\\]]", "]", true},           {"[a\\-z]", "-", true},
};

// DuckDB: SELECT regexp_replace(text, pat, repl) (first match only).
struct Repl { const char* text; const char* pat; const char* repl; const char* want; };
const Repl kReplace[] = {
    {"aaab", "a+?", "X", "Xaab"},
    {"<a><b>", "<.+?>", "X", "X<b>"},
    {"2024-01-05", "^(\\d{4})-(\\d{2})-(\\d{2})$", "\\3/\\2/\\1", "05/01/2024"},
    {"abbb", "ab{2,}?", "Z", "Zb"},
    {"xaaay", "a{1,2}", "-", "x-ay"},
};

// DuckDB refuses each ("invalid repetition size", "bad repetition
// operator", "missing argument", "invalid escape", "trailing \"); the
// last group is valid RE2 this engine does not implement.
const char* kRefused[] = {
    "{2}", "a{2,1}", "a{1001}", "a**", "a*+", "\\q", "a\\", "x{1}{2}", "*a",
    "(+a)", "\\1", "a*??", "a{2}+",
    "^*", "\\b", "[[:alpha:]]", "[\\D]", "(?i)a", "a|b", "(?=a)", "(ab){300}",
};

bool search(const char* pat, const char* text, MatchResult* mr) {
    CompiledPattern cp;
    const bool ok = regex_compile(pat, static_cast<uint32_t>(std::strlen(pat)), &cp);
    EXPECT_TRUE(ok) << "compile failed for /" << pat << "/";
    if (!ok) return false;
    const bool m = regex_search(&cp, text, static_cast<int32_t>(std::strlen(text)), mr);
    EXPECT_FALSE(mr->exhausted) << "/" << pat << "/ on '" << text << "'";
    return m;
}

TEST(RegexBacktrack, MatchesAgreeWithDuckDB) {
    for (const Case& c : kMatch) {
        MatchResult mr;
        EXPECT_EQ(search(c.pat, c.text, &mr), c.want)
            << "/" << c.pat << "/ on '" << c.text << "'";
    }
}

TEST(RegexBacktrack, ReplaceFirstAgreesWithDuckDB) {
    for (const Repl& r : kReplace) {
        MatchResult mr;
        ASSERT_TRUE(search(r.pat, r.text, &mr)) << r.pat;
        char buf[256];
        const int32_t n = regex_substitute(&mr, r.text, r.repl,
            static_cast<uint32_t>(std::strlen(r.repl)), buf, sizeof(buf));
        ASSERT_GE(n, 0);
        std::string out(r.text, static_cast<size_t>(mr.g_start[0]));
        out.append(buf, static_cast<size_t>(n));
        out.append(r.text + mr.g_end[0]);
        EXPECT_EQ(out, r.want) << "/" << r.pat << "/ on '" << r.text << "'";
    }
}

TEST(RegexBacktrack, UnsupportedSyntaxIsRefusedNotReadAsLiteral) {
    for (const char* p : kRefused) {
        CompiledPattern cp;
        EXPECT_FALSE(regex_compile(p, static_cast<uint32_t>(std::strlen(p)), &cp))
            << "/" << p << "/ compiled";
    }
}

TEST(RegexBacktrack, GroupRepetitionsPastTheCapAreExhaustionNotNoMatch) {
    std::string text;
    for (int i = 0; i < kRegexMaxGroupReps + 10; ++i) text += "ab";
    CompiledPattern cp;
    ASSERT_TRUE(regex_compile("^(ab)+$", 7, &cp));
    MatchResult mr;
    EXPECT_FALSE(regex_search(&cp, text.data(), static_cast<int32_t>(text.size()), &mr));
    EXPECT_TRUE(mr.exhausted);
    text.resize(static_cast<size_t>(kRegexMaxGroupReps) * 2u);
    EXPECT_TRUE(regex_search(&cp, text.data(), static_cast<int32_t>(text.size()), &mr));
    EXPECT_FALSE(mr.exhausted);
}

TEST(RegexBacktrack, StepBudgetExhaustionIsReported) {
    std::string text(4000, 'a');
    CompiledPattern cp;
    ASSERT_TRUE(regex_compile("^a*a*a*b", 8, &cp));
    MatchResult mr;
    EXPECT_FALSE(regex_search(&cp, text.data(), static_cast<int32_t>(text.size()), &mr));
    EXPECT_TRUE(mr.exhausted);
}

TEST(RegexBacktrack, StaleCaptureFromAFailedStartIsCleared) {
    MatchResult mr;
    ASSERT_TRUE(search("(x)?y", "xzy", &mr));
    EXPECT_EQ(mr.g_start[0], 2);
    EXPECT_EQ(mr.g_start[1], -1);
}

}  // namespace
