#include "bolt/kernels/fintech/exact_depth.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #value); std::abort(); } } while (false)
namespace {
struct Level { std::int64_t price_ticks; std::uint32_t quantity; };
void test_depth() {
    using namespace bolt::fintech;
    Level ask[2]{{22, 15}, {23, 25}};
    ExactDepthResult out{};
    CHECK(exact_depth<2>(ask, 2, 40, out) == ExactDepthStatus::Ok);
    CHECK(out.weighted_ticks == 905 && out.capacity == 40 && out.consumed == 40);
    CHECK(out.marginal_ticks == 23);
    CHECK(exact_depth<2>(ask, 2, 41, out) == ExactDepthStatus::Insufficient);
    CHECK(out.capacity == 40 && out.weighted_ticks == 0 && out.consumed == 0);
    CHECK(exact_depth<2>(ask, 2, 0, out) == ExactDepthStatus::Invalid);
    CHECK(exact_depth<2>(ask, 3, 40, out) == ExactDepthStatus::Invalid);
    ask[0] = {-22, 15}; ask[1] = {-21, 25};
    CHECK(exact_depth<2>(ask, 2, 40, out) == ExactDepthStatus::Ok);
    CHECK(out.weighted_ticks == -855 && out.marginal_ticks == -21);
    ask[0] = {INT64_MAX, 2};
    CHECK(exact_depth<2>(ask, 1, 2, out) == ExactDepthStatus::Overflow);
    CHECK(out.weighted_ticks == 0 && out.consumed == 0);
}
void test_arithmetic() {
    using namespace bolt::fintech;
    std::int64_t out = 0;
    CHECK(!exact_add(INT64_MAX, 1, out));
    CHECK(!exact_add(INT64_MIN, -1, out));
    CHECK(exact_add(INT64_MIN, INT64_MAX, out) && out == -1);
    CHECK(!exact_sub(INT64_MIN, 1, out));
    CHECK(!exact_sub(INT64_MAX, -1, out));
    CHECK(exact_sub(INT64_MIN, INT64_MIN, out) && out == 0);
    CHECK(exact_sub(-1, INT64_MIN, out) && out == INT64_MAX);
    CHECK(exact_multiply(INT64_MIN, 0, out) && out == 0);
    CHECK(exact_multiply(INT64_MIN, 1, out) && out == INT64_MIN);
    CHECK(!exact_multiply(INT64_MIN, 2, out));
    CHECK(!exact_multiply(INT64_MAX, UINT32_MAX, out));
}
void test_text() {
    using namespace bolt::fintech;
    char text[64]{};
    bool exact = false;
    CHECK(exact_price_text(905, 40, text, sizeof(text), exact));
    CHECK(exact && std::strcmp(text, "0.22625") == 0);
    CHECK(exact_price_text(20000, 1, text, sizeof(text), exact));
    CHECK(exact && std::strcmp(text, "200.00") == 0);
    CHECK(exact_price_text(INT64_MIN, 1, text, sizeof(text), exact));
    CHECK(exact && std::strcmp(text, "-92233720368547758.08") == 0);
    CHECK(exact_price_text(INT64_MAX, 1, text, sizeof(text), exact));
    CHECK(exact && std::strcmp(text, "92233720368547758.07") == 0);
    CHECK(exact_price_text(1, 3, text, sizeof(text), exact));
    CHECK(!exact && std::strcmp(text, "0.00333333") == 0);
    CHECK(!exact_price_text(1, 0, text, sizeof(text), exact));
    CHECK(!exact_price_text(1, 1, text, 2, exact));
}
}
int main() {
    test_depth(); test_arithmetic(); test_text();
    std::puts("Bolt exact depth: depth/arithmetic/format tests passed");
    return 0;
}
