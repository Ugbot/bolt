// test_bolt_setop.cpp — Coverage for the two-input sorted-merge set ops.
//
// Verifies SQL semantics for UNION / INTERSECT / EXCEPT (DISTINCT + ALL).
// Each kernel is checked against a reference std::multiset / std::set
// implementation over random sorted inputs of varied length and overlap.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "bolt/kernels/bolt_setop.h"

using namespace bolt::kernels::setop;

namespace {

// Reference implementations using std containers.
std::vector<int64_t> ref_union_distinct(const std::vector<int64_t>& a,
                                        const std::vector<int64_t>& b) {
    std::set<int64_t> s(a.begin(), a.end());
    s.insert(b.begin(), b.end());
    return {s.begin(), s.end()};
}

std::vector<int64_t> ref_union_all(const std::vector<int64_t>& a,
                                   const std::vector<int64_t>& b) {
    std::vector<int64_t> r;
    r.reserve(a.size() + b.size());
    r.insert(r.end(), a.begin(), a.end());
    r.insert(r.end(), b.begin(), b.end());
    std::sort(r.begin(), r.end());
    return r;
}

std::vector<int64_t> ref_intersect_distinct(const std::vector<int64_t>& a,
                                            const std::vector<int64_t>& b) {
    std::set<int64_t> sa(a.begin(), a.end());
    std::set<int64_t> sb(b.begin(), b.end());
    std::vector<int64_t> r;
    for (int64_t x : sa) if (sb.count(x)) r.push_back(x);
    return r;
}

std::vector<int64_t> ref_intersect_all(const std::vector<int64_t>& a,
                                       const std::vector<int64_t>& b) {
    std::vector<int64_t> r;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::back_inserter(r));
    return r;
}

std::vector<int64_t> ref_except_distinct(const std::vector<int64_t>& a,
                                         const std::vector<int64_t>& b) {
    std::set<int64_t> sa(a.begin(), a.end());
    std::set<int64_t> sb(b.begin(), b.end());
    std::vector<int64_t> r;
    for (int64_t x : sa) if (!sb.count(x)) r.push_back(x);
    return r;
}

std::vector<int64_t> ref_except_all(const std::vector<int64_t>& a,
                                    const std::vector<int64_t>& b) {
    std::vector<int64_t> r;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(r));
    return r;
}

// Driver: run a setop kernel and compare against ref.
template <typename Fn, typename Ref>
void check(Fn fn, Ref ref,
           const std::vector<int64_t>& a, const std::vector<int64_t>& b,
           const char* tag) {
    std::vector<int64_t> out(a.size() + b.size() + 1, 0);
    int64_t n = fn(a.data(), (int64_t)a.size(),
                   b.data(), (int64_t)b.size(),
                   out.data());
    std::vector<int64_t> got(out.begin(), out.begin() + n);
    std::vector<int64_t> exp = ref(a, b);
    ASSERT_EQ(got, exp) << tag << " na=" << a.size() << " nb=" << b.size();
}

std::vector<int64_t> mk_sorted(std::mt19937_64& rng, size_t n,
                               int64_t lo, int64_t hi) {
    std::uniform_int_distribution<int64_t> d(lo, hi);
    std::vector<int64_t> v(n);
    for (auto& x : v) x = d(rng);
    std::sort(v.begin(), v.end());
    return v;
}

}  // namespace

TEST(BoltSetOp, EmptyInputs) {
    std::vector<int64_t> empty;
    std::vector<int64_t> a = {1, 2, 3};
    check(set_union_distinct_sorted<int64_t>,    ref_union_distinct,
          empty, empty, "union_distinct empty/empty");
    check(set_union_distinct_sorted<int64_t>,    ref_union_distinct,
          a, empty, "union_distinct a/empty");
    check(set_union_all_sorted<int64_t>,         ref_union_all,
          empty, a, "union_all empty/a");
    check(set_intersect_sorted<int64_t>,         ref_intersect_distinct,
          empty, a, "intersect_distinct empty/a");
    check(set_intersect_all_sorted<int64_t>,     ref_intersect_all,
          a, empty, "intersect_all a/empty");
    check(set_except_sorted<int64_t>,            ref_except_distinct,
          a, empty, "except_distinct a/empty");
    check(set_except_all_sorted<int64_t>,        ref_except_all,
          empty, a, "except_all empty/a");
}

TEST(BoltSetOp, NoOverlap) {
    std::vector<int64_t> a = {1, 2, 3, 4};
    std::vector<int64_t> b = {10, 20, 30};
    check(set_union_distinct_sorted<int64_t>, ref_union_distinct, a, b, "ud");
    check(set_union_all_sorted<int64_t>,      ref_union_all,      a, b, "ua");
    check(set_intersect_sorted<int64_t>,      ref_intersect_distinct, a, b, "id");
    check(set_intersect_all_sorted<int64_t>,  ref_intersect_all,  a, b, "ia");
    check(set_except_sorted<int64_t>,         ref_except_distinct, a, b, "ed");
    check(set_except_all_sorted<int64_t>,     ref_except_all,     a, b, "ea");
}

TEST(BoltSetOp, FullOverlap) {
    std::vector<int64_t> a = {1, 2, 3, 4, 5};
    std::vector<int64_t> b = a;
    check(set_union_distinct_sorted<int64_t>, ref_union_distinct, a, b, "ud");
    check(set_intersect_sorted<int64_t>,      ref_intersect_distinct, a, b, "id");
    check(set_intersect_all_sorted<int64_t>,  ref_intersect_all,  a, b, "ia");
    check(set_except_sorted<int64_t>,         ref_except_distinct, a, b, "ed");
    check(set_except_all_sorted<int64_t>,     ref_except_all,     a, b, "ea");
}

TEST(BoltSetOp, DuplicatesWithin) {
    std::vector<int64_t> a = {1, 1, 1, 2, 3, 3};
    std::vector<int64_t> b = {1, 3, 3, 3, 4};
    check(set_union_distinct_sorted<int64_t>, ref_union_distinct, a, b, "ud");
    check(set_union_all_sorted<int64_t>,      ref_union_all,      a, b, "ua");
    check(set_intersect_sorted<int64_t>,      ref_intersect_distinct, a, b, "id");
    check(set_intersect_all_sorted<int64_t>,  ref_intersect_all,  a, b, "ia");
    check(set_except_sorted<int64_t>,         ref_except_distinct, a, b, "ed");
    check(set_except_all_sorted<int64_t>,     ref_except_all,     a, b, "ea");
}

TEST(BoltSetOp, RandomFuzz) {
    std::mt19937_64 rng(0xC0DEFEED);
    for (int trial = 0; trial < 200; ++trial) {
        const size_t na = rng() % 256;
        const size_t nb = rng() % 256;
        const int64_t lo = 0;
        const int64_t hi = 1 + (int64_t)(rng() % 50);  // small range → overlap
        auto a = mk_sorted(rng, na, lo, hi);
        auto b = mk_sorted(rng, nb, lo, hi);
        check(set_union_distinct_sorted<int64_t>, ref_union_distinct, a, b, "ud");
        check(set_union_all_sorted<int64_t>,      ref_union_all,      a, b, "ua");
        check(set_intersect_sorted<int64_t>,      ref_intersect_distinct, a, b, "id");
        check(set_intersect_all_sorted<int64_t>,  ref_intersect_all,  a, b, "ia");
        check(set_except_sorted<int64_t>,         ref_except_distinct, a, b, "ed");
        check(set_except_all_sorted<int64_t>,     ref_except_all,     a, b, "ea");
    }
}

TEST(BoltSetOp, Unbalanced) {
    std::mt19937_64 rng(42);
    auto a = mk_sorted(rng, 10000, 0, 1000000);
    auto b = mk_sorted(rng, 10,    0, 1000000);
    check(set_intersect_sorted<int64_t>,      ref_intersect_distinct, a, b, "id");
    check(set_except_sorted<int64_t>,         ref_except_distinct, a, b, "ed");
    check(set_union_distinct_sorted<int64_t>, ref_union_distinct, a, b, "ud");
}

// Adversarial perfectly-interleaved inputs — every step alternates which
// pointer advances, the worst case for the (now branchless) two-pointer loop.
// Exercises all four converted kernels against the std reference.
TEST(BoltSetOp, AlternatingInterleave) {
    // a = evens, b = odds → cmp alternates +/-, zero matches.
    std::vector<int64_t> a, b;
    for (int64_t x = 0; x < 256; x += 2) a.push_back(x);
    for (int64_t x = 1; x < 256; x += 2) b.push_back(x);
    check(set_intersect_sorted<int64_t>,     ref_intersect_distinct, a, b, "id");
    check(set_intersect_all_sorted<int64_t>, ref_intersect_all,      a, b, "ia");
    check(set_except_sorted<int64_t>,        ref_except_distinct,    a, b, "ed");
    check(set_except_all_sorted<int64_t>,    ref_except_all,         a, b, "ea");

    // Heavy interleave WITH matches + within-run duplicates on both sides.
    std::vector<int64_t> c = {0, 1, 1, 2, 4, 4, 4, 5, 7, 8, 8};
    std::vector<int64_t> d = {1, 2, 2, 3, 4, 6, 7, 7, 8};
    check(set_intersect_sorted<int64_t>,     ref_intersect_distinct, c, d, "id2");
    check(set_intersect_all_sorted<int64_t>, ref_intersect_all,      c, d, "ia2");
    check(set_except_sorted<int64_t>,        ref_except_distinct,    c, d, "ed2");
    check(set_except_all_sorted<int64_t>,    ref_except_all,         c, d, "ea2");
}

// Dedicated adversarial random fuzz with HIGH overlap probability — tight key
// range so cmp==0 lands ~50% of the time (max branch-mispredict in the old
// code) and the branchless rewrite must still match the std reference exactly.
TEST(BoltSetOp, RandomOverlapHeavyFuzz) {
    std::mt19937_64 rng(0xBADC0FFEE0DDF00DULL);
    for (int trial = 0; trial < 300; ++trial) {
        const size_t na = rng() % 300;
        const size_t nb = rng() % 300;
        const int64_t hi = 1 + (int64_t)(rng() % 12);  // very small → ~50% ties
        auto a = mk_sorted(rng, na, 0, hi);
        auto b = mk_sorted(rng, nb, 0, hi);
        check(set_intersect_sorted<int64_t>,     ref_intersect_distinct, a, b, "id");
        check(set_intersect_all_sorted<int64_t>, ref_intersect_all,      a, b, "ia");
        check(set_except_sorted<int64_t>,        ref_except_distinct,    a, b, "ed");
        check(set_except_all_sorted<int64_t>,    ref_except_all,         a, b, "ea");
    }
}

TEST(BoltSetOp, Int32AndDouble) {
    // Smoke-test alternative T instantiations: int32 + double.
    std::vector<int32_t> a32 = {1, 2, 2, 3};
    std::vector<int32_t> b32 = {2, 3, 4};
    std::vector<int32_t> out32(8, 0);
    int64_t n = set_union_distinct_sorted<int32_t>(
        a32.data(), 4, b32.data(), 3, out32.data());
    ASSERT_EQ(n, 4);
    EXPECT_EQ(out32[0], 1);
    EXPECT_EQ(out32[1], 2);
    EXPECT_EQ(out32[2], 3);
    EXPECT_EQ(out32[3], 4);

    std::vector<double> ad = {1.0, 2.5, 3.0};
    std::vector<double> bd = {2.5, 3.0, 4.0};
    std::vector<double> outd(8, 0.0);
    int64_t nd = set_intersect_sorted<double>(
        ad.data(), 3, bd.data(), 3, outd.data());
    ASSERT_EQ(nd, 2);
    EXPECT_DOUBLE_EQ(outd[0], 2.5);
    EXPECT_DOUBLE_EQ(outd[1], 3.0);
}
