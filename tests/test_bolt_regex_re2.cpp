// test_bolt_regex_re2.cpp — RE2 common-subset anchored FULL-MATCH engine.
//
// Cross-checks bolt::kernels::regex::re2_{compile,full_match} against
// known-answer cases, including the exact PromQL matcher corpus patterns
// (RE2 semantics, both ends fully anchored). Every case is a whole-value
// membership decision: partial matches MUST reject.

#include "bolt/kernels/bolt_regex.h"

#include <cstring>
#include <gtest/gtest.h>

using namespace bolt::kernels::regex;

namespace {

// Compile `pat`; ASSERT it compiled; return match(text) as bool.
bool M(const char* pat, const char* text) {
    Re2Program prog;
    const bool ok = re2_compile(pat, static_cast<uint32_t>(std::strlen(pat)), &prog);
    EXPECT_TRUE(ok) << "compile failed for /" << pat << "/";
    if (!ok) return false;
    return re2_full_match(&prog, text, static_cast<int32_t>(std::strlen(text)));
}

// True iff `pat` fails to compile (unsupported / malformed / over cap).
bool CompileFails(const char* pat) {
    Re2Program prog;
    return !re2_compile(pat, static_cast<uint32_t>(std::strlen(pat)), &prog);
}

}  // namespace

// ---------------------------------------------------------------------
// Literals + anchoring (full match is the contract)
// ---------------------------------------------------------------------
TEST(Re2Literal, ExactWholeMatch) {
    EXPECT_TRUE(M("abc", "abc"));
    EXPECT_FALSE(M("abc", "abcd"));   // trailing char -> full match fails
    EXPECT_FALSE(M("abc", "xabc"));   // leading char
    EXPECT_FALSE(M("abc", "ab"));     // short
    EXPECT_FALSE(M("abc", "aXc"));
}

TEST(Re2Literal, EmptyPatternMatchesOnlyEmpty) {
    EXPECT_TRUE(M("", ""));
    EXPECT_FALSE(M("", "a"));
}

// ---------------------------------------------------------------------
// Alternation — the most important PromQL feature
// ---------------------------------------------------------------------
TEST(Re2Alt, PromqlNameAlternation) {
    // {__name__=~'testmetric1|testmetric2'}
    EXPECT_TRUE(M("testmetric1|testmetric2", "testmetric1"));
    EXPECT_TRUE(M("testmetric1|testmetric2", "testmetric2"));
    EXPECT_FALSE(M("testmetric1|testmetric2", "testmetric3"));
    EXPECT_FALSE(M("testmetric1|testmetric2", "testmetric1x"));  // full match
    EXPECT_FALSE(M("testmetric1|testmetric2", "xtestmetric1"));
    EXPECT_FALSE(M("testmetric1|testmetric2", ""));
}

TEST(Re2Alt, SingleDigitInstance) {
    // instance=~"0|1"
    EXPECT_TRUE(M("0|1", "0"));
    EXPECT_TRUE(M("0|1", "1"));
    EXPECT_FALSE(M("0|1", "2"));
    EXPECT_FALSE(M("0|1", "01"));   // full match: two chars
    EXPECT_FALSE(M("0|1", ""));
}

TEST(Re2Alt, ThreeArmsAndEmptyArm) {
    EXPECT_TRUE(M("a|b|c", "a"));
    EXPECT_TRUE(M("a|b|c", "b"));
    EXPECT_TRUE(M("a|b|c", "c"));
    EXPECT_FALSE(M("a|b|c", "d"));
    // empty arm: "a|" matches "a" or ""
    EXPECT_TRUE(M("a|", "a"));
    EXPECT_TRUE(M("a|", ""));
    EXPECT_FALSE(M("a|", "b"));
}

TEST(Re2Alt, GroupedAlternationWithSuffix) {
    EXPECT_TRUE(M("(cat|dog)s", "cats"));
    EXPECT_TRUE(M("(cat|dog)s", "dogs"));
    EXPECT_FALSE(M("(cat|dog)s", "cat"));    // missing 's'
    EXPECT_FALSE(M("(cat|dog)s", "birds"));
    EXPECT_TRUE(M("up|down|(left|right)", "left"));
    EXPECT_TRUE(M("up|down|(left|right)", "down"));
    EXPECT_FALSE(M("up|down|(left|right)", "middle"));
}

// ---------------------------------------------------------------------
// Quantifiers
// ---------------------------------------------------------------------
TEST(Re2Quant, StarPlusOptional) {
    EXPECT_TRUE(M("a*", ""));
    EXPECT_TRUE(M("a*", "aaaa"));
    EXPECT_FALSE(M("a*", "aaab"));
    EXPECT_FALSE(M("a+", ""));
    EXPECT_TRUE(M("a+", "a"));
    EXPECT_TRUE(M("a+", "aaa"));
    EXPECT_TRUE(M("ab?c", "ac"));
    EXPECT_TRUE(M("ab?c", "abc"));
    EXPECT_FALSE(M("ab?c", "abbc"));
}

TEST(Re2Quant, DotStar) {
    EXPECT_TRUE(M(".*", ""));
    EXPECT_TRUE(M(".*", "anything here"));
    EXPECT_TRUE(M("foo.*", "foobar"));
    EXPECT_TRUE(M("foo.*", "foo"));
    EXPECT_FALSE(M("foo.*", "barfoo"));
    // RE2 default: '.' excludes newline, so '.*' cannot span it.
    EXPECT_FALSE(M(".*", "a\nb"));
    EXPECT_TRUE(M(".*", "a b"));
}

TEST(Re2Quant, CountedBounds) {
    // \d{2,4}
    EXPECT_FALSE(M("\\d{2,4}", "1"));
    EXPECT_TRUE(M("\\d{2,4}", "12"));
    EXPECT_TRUE(M("\\d{2,4}", "123"));
    EXPECT_TRUE(M("\\d{2,4}", "1234"));
    EXPECT_FALSE(M("\\d{2,4}", "12345"));
    EXPECT_FALSE(M("\\d{2,4}", "12a"));
    // exact {m}
    EXPECT_TRUE(M("a{3}", "aaa"));
    EXPECT_FALSE(M("a{3}", "aa"));
    EXPECT_FALSE(M("a{3}", "aaaa"));
    // {m,}
    EXPECT_FALSE(M("a{2,}", "a"));
    EXPECT_TRUE(M("a{2,}", "aa"));
    EXPECT_TRUE(M("a{2,}", "aaaaaa"));
}

TEST(Re2Quant, QuantifiedGroup) {
    EXPECT_TRUE(M("(ab)+", "ababab"));
    EXPECT_FALSE(M("(ab)+", "aba"));
    EXPECT_FALSE(M("(ab)+", ""));
    EXPECT_TRUE(M("(ab){2}", "abab"));
    EXPECT_FALSE(M("(ab){2}", "ababab"));
}

// ---------------------------------------------------------------------
// Character classes + builtin escapes
// ---------------------------------------------------------------------
TEST(Re2Class, RangesAndSets) {
    EXPECT_TRUE(M("[a-z]+", "hello"));
    EXPECT_FALSE(M("[a-z]+", "hello1"));
    EXPECT_FALSE(M("[a-z]+", ""));
    EXPECT_TRUE(M("[abc]", "b"));
    EXPECT_FALSE(M("[abc]", "d"));
    EXPECT_TRUE(M("[A-Za-z0-9_]+", "Ident_9"));
    EXPECT_FALSE(M("[A-Za-z0-9_]+", "Ident-9"));
}

TEST(Re2Class, Negation) {
    EXPECT_TRUE(M("[^0-9]+", "abc"));
    EXPECT_FALSE(M("[^0-9]+", "ab1"));
    EXPECT_TRUE(M("[^/]+", "path_segment"));
    EXPECT_FALSE(M("[^/]+", "a/b"));
}

TEST(Re2Class, LeadingBracketAndDashLiteral) {
    EXPECT_TRUE(M("[-a]+", "a-a-"));      // '-' is literal at edges
    EXPECT_FALSE(M("[-a]+", "b"));
}

TEST(Re2Class, BuiltinEscapes) {
    EXPECT_TRUE(M("\\d+", "12345"));
    EXPECT_FALSE(M("\\d+", "12a45"));
    EXPECT_TRUE(M("\\w+", "abc_123"));
    EXPECT_FALSE(M("\\w+", "abc 123"));   // space not in \w
    EXPECT_TRUE(M("\\s+", "  \t "));
    EXPECT_FALSE(M("\\s+", " x "));
    // negated builtins
    EXPECT_TRUE(M("\\D+", "abc"));
    EXPECT_FALSE(M("\\D+", "ab3"));
    EXPECT_TRUE(M("\\S+", "nowhitespace"));
    EXPECT_FALSE(M("\\S+", "has space"));
}

TEST(Re2Class, BuiltinInsideClass) {
    EXPECT_TRUE(M("[\\w.]+", "a.b.c_1"));
    EXPECT_FALSE(M("[\\w.]+", "a b"));
    EXPECT_TRUE(M("[\\d-]+", "12-34"));
    EXPECT_FALSE(M("[\\d-]+", "12x34"));
}

// ---------------------------------------------------------------------
// Case-insensitive inline flags
// ---------------------------------------------------------------------
TEST(Re2ICase, ScopedGroupFlag) {
    // (?i:PRO).*
    EXPECT_TRUE(M("(?i:PRO).*", "protocol"));
    EXPECT_TRUE(M("(?i:PRO).*", "PROTOCOL"));
    EXPECT_TRUE(M("(?i:PRO).*", "PrOtocol"));
    EXPECT_FALSE(M("(?i:PRO).*", "xprotocol"));   // must start with pro
    // scope ends at group: the trailing literal stays case-sensitive
    EXPECT_TRUE(M("(?i:ab)CD", "AbCD"));
    EXPECT_FALSE(M("(?i:ab)CD", "ABcd"));         // 'cd' lower, but CD required
}

TEST(Re2ICase, PrefixFlag) {
    // (?i) applies to the rest of the pattern
    EXPECT_TRUE(M("(?i)hello", "HELLO"));
    EXPECT_TRUE(M("(?i)hello", "Hello"));
    EXPECT_TRUE(M("(?i)hello", "hello"));
    EXPECT_FALSE(M("(?i)hello", "hell"));
}

TEST(Re2ICase, ClassCaseInsensitive) {
    EXPECT_TRUE(M("(?i)[a-z]+", "ABC"));
    EXPECT_TRUE(M("(?i)[a-z]+", "aBcD"));
    EXPECT_FALSE(M("(?i)[a-z]+", "aB1"));
    // negated class + icase: (?i)[^a] must reject both 'a' and 'A'
    EXPECT_FALSE(M("(?i)[^a]", "A"));
    EXPECT_FALSE(M("(?i)[^a]", "a"));
    EXPECT_TRUE(M("(?i)[^a]", "b"));
}

// ---------------------------------------------------------------------
// Escaped metacharacters (literal)
// ---------------------------------------------------------------------
TEST(Re2Escape, LiteralMetachars) {
    EXPECT_TRUE(M("a\\.b", "a.b"));
    EXPECT_FALSE(M("a\\.b", "axb"));       // '.' is literal here
    EXPECT_TRUE(M("\\(x\\)", "(x)"));
    EXPECT_TRUE(M("a\\+", "a+"));
    EXPECT_TRUE(M("100\\%", "100%"));
    EXPECT_TRUE(M("a\\|b", "a|b"));        // escaped pipe is literal
    EXPECT_FALSE(M("a\\|b", "a"));
}

// ---------------------------------------------------------------------
// Realistic PromQL-shaped label matchers
// ---------------------------------------------------------------------
TEST(Re2Promql, LabelValueShapes) {
    // job=~"api|web|worker"
    EXPECT_TRUE(M("api|web|worker", "web"));
    EXPECT_FALSE(M("api|web|worker", "database"));
    // instance=~"10\\.0\\.0\\.[0-9]+:9090"
    EXPECT_TRUE(M("10\\.0\\.0\\.[0-9]+:9090", "10.0.0.5:9090"));
    EXPECT_FALSE(M("10\\.0\\.0\\.[0-9]+:9090", "10.0.0.5:9091"));
    EXPECT_FALSE(M("10\\.0\\.0\\.[0-9]+:9090", "10x0.0.5:9090"));
    // path=~"/api/v[0-9]+/.*"
    EXPECT_TRUE(M("/api/v[0-9]+/.*", "/api/v2/users"));
    EXPECT_FALSE(M("/api/v[0-9]+/.*", "/api/vX/users"));
    // status=~"[45][0-9]{2}"  (4xx/5xx codes)
    EXPECT_TRUE(M("[45][0-9]{2}", "404"));
    EXPECT_TRUE(M("[45][0-9]{2}", "503"));
    EXPECT_FALSE(M("[45][0-9]{2}", "200"));
    EXPECT_FALSE(M("[45][0-9]{2}", "4040"));
}

// ---------------------------------------------------------------------
// Bounded / reject-cleanly behavior (never a wrong match, never a hang)
// ---------------------------------------------------------------------
TEST(Re2Reject, UnsupportedConstructsFailCompile) {
    EXPECT_TRUE(CompileFails("(?P<name>x)"));   // named group
    EXPECT_TRUE(CompileFails("(?s)a.b"));        // unsupported 's' flag
    EXPECT_TRUE(CompileFails("(?m)^a"));         // unsupported 'm' flag
    EXPECT_TRUE(CompileFails("[a-z"));           // unterminated class
    EXPECT_TRUE(CompileFails("(abc"));           // unbalanced group
    EXPECT_TRUE(CompileFails("abc)"));           // stray ')'
    EXPECT_TRUE(CompileFails("a{5,2}"));         // inverted counted range
    EXPECT_TRUE(CompileFails("a{2000}"));        // over kRe2MaxRepeat
}

TEST(Re2Reject, EmptyLoopTerminates) {
    // Epsilon-loopable constructs must compile and decide membership without
    // hanging (visited-stamp in the NFA closure guarantees termination).
    EXPECT_TRUE(M("(a*)*", ""));
    EXPECT_TRUE(M("(a*)*", "aaa"));
    EXPECT_FALSE(M("(a*)*", "aab"));
    EXPECT_TRUE(M("()*", ""));
    EXPECT_TRUE(M("(|a)+", "aaa"));
}

TEST(Re2Reject, PathologicalNoCatastrophicBacktrack) {
    // The classic ReDoS shape (a+)+$ against a long non-matching string.
    // A backtracker would blow up; the Thompson NFA is linear -> just false.
    Re2Program prog;
    ASSERT_TRUE(re2_compile("(a+)+", 5, &prog));
    char buf[64];
    std::memset(buf, 'a', sizeof(buf));
    buf[40] = 'b';                       // force a full-match failure at the end
    EXPECT_FALSE(re2_full_match(&prog, buf, 60));
    // all-a's fully matches
    std::memset(buf, 'a', sizeof(buf));
    EXPECT_TRUE(re2_full_match(&prog, buf, 60));
}

TEST(Re2Anchor, ExplicitAnchorsHonored) {
    EXPECT_TRUE(M("^abc$", "abc"));
    EXPECT_FALSE(M("^abc$", "abcd"));
    EXPECT_TRUE(M("^a.*z$", "abcz"));
}
