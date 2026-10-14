#include "bolt/kernels/fintech/exact_depth.h"
#include <algorithm>
#include <chrono>
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
void test_imbalance() {
    using namespace bolt::fintech;
    Level bids[2]{{INT64_MIN, 3}, {INT64_MAX, 0}};
    Level asks[2]{{INT64_MAX, 1}, {INT64_MIN, 0}};
    std::int8_t direction = 99;
    CHECK(exact_depth_imbalance<2>(bids, 2, asks, 2, 1, 2, direction) == ExactDepthStatus::Ok && direction == 0);
    CHECK(exact_depth_imbalance<2>(asks, 2, bids, 2, 1, 2, direction) == ExactDepthStatus::Ok && direction == 0);
    bids[0].quantity = 4;
    CHECK(exact_depth_imbalance<2>(bids, 2, asks, 2, 1, 2, direction) == ExactDepthStatus::Ok && direction == 1);
    CHECK(exact_depth_imbalance<2>(asks, 2, bids, 2, 1, 2, direction) == ExactDepthStatus::Ok && direction == -1);
    bids[0].quantity = asks[0].quantity = UINT32_MAX;
    bids[1].quantity = asks[1].quantity = UINT32_MAX;
    CHECK(exact_depth_imbalance<2>(bids, 2, asks, 2, UINT32_MAX, UINT32_MAX, direction) == ExactDepthStatus::Overflow && direction == 0);
    CHECK(exact_depth_imbalance<2>(bids, 2, asks, 2, 1, 1000000, direction) == ExactDepthStatus::Ok && direction == 0);
    asks[1].quantity = 0;
    CHECK(exact_depth_imbalance<2>(bids, 2, asks, 2, 0, UINT32_MAX, direction) == ExactDepthStatus::Overflow && direction == 0);
    CHECK(exact_depth_imbalance<2>(bids, 2, asks, 2, 0, 1, direction) == ExactDepthStatus::Ok && direction == 1);
    CHECK(exact_depth_imbalance<2>(bids, 2, asks, 2, 1, 0, direction) == ExactDepthStatus::Invalid && direction == 0);
    direction = 99;
    CHECK(exact_depth_imbalance<2>(bids, 2, asks, 2, 2, 1, direction) == ExactDepthStatus::Invalid && direction == 0);
    direction = 99;
    CHECK(exact_depth_imbalance<2>(bids, 3, asks, 2, 0, 1, direction) == ExactDepthStatus::Invalid && direction == 0);
    direction = 99;
    CHECK(exact_depth_imbalance<2>(bids, 0, asks, 2, 0, 1, direction) == ExactDepthStatus::Insufficient && direction == 0);
    asks[0].quantity = 0;
    direction = 99;
    CHECK(exact_depth_imbalance<2>(bids, 2, asks, 2, 0, 1, direction) == ExactDepthStatus::Insufficient && direction == 0);
}
std::int8_t scalar_direction(const Level* bids, const Level* asks, std::uint32_t numerator,
    std::uint32_t denominator) noexcept {
    std::int64_t b = 0, a = 0;
    for (std::uint32_t i = 0; i < 10; ++i) { b += bids[i].quantity; a += asks[i].quantity; }
    const auto lhs = (b - a) * denominator;
    const auto rhs = (b + a) * numerator;
    return static_cast<std::int8_t>((lhs > rhs) - (lhs < -rhs));
}
void bench_imbalance() {
    using Clock = std::chrono::steady_clock;
    constexpr std::uint32_t count = 10000;
    constexpr std::uint32_t batch = 64;
    Level bids[batch][10]{}, asks[batch][10]{};
    std::uint64_t samples[count]{};
    for (unsigned repeat = 0; repeat < 5; ++repeat) {
        for (unsigned order = 0; order < 2; ++order) {
            const bool kernel = ((order + repeat) % 2) != 0;
            for (std::uint32_t row = 0; row < batch; ++row)
                for (std::uint32_t i = 0; i < 10; ++i) {
                    bids[row][i] = {100 - i, 10 + (i + row) % 31};
                    asks[row][i] = {102 + i, 10 + i};
                }
            std::int64_t checksum = 0, expected = 0;
            for (std::uint32_t i = 0; i < count; ++i) {
                bids[i % batch][0].quantity = 10 + i % 31;
                for (std::uint32_t row = 0; row < batch; ++row) expected += scalar_direction(bids[row], asks[row], 1, 4);
                const auto begin = Clock::now();
                for (std::uint32_t row = 0; row < batch; ++row) {
                    std::int8_t direction = 0;
                    if (kernel) {
                        CHECK(bolt::fintech::exact_depth_imbalance<10>(bids[row], 10, asks[row], 10, 1, 4, direction) == bolt::fintech::ExactDepthStatus::Ok);
                    } else direction = scalar_direction(bids[row], asks[row], 1, 4);
                    checksum += direction;
                }
                const auto end = Clock::now();
                samples[i] = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
            }
            CHECK(checksum == expected);
            std::sort(samples, samples + count);
            std::printf("imbalance repeat=%u kernel=%u samples=%u batch=64 batch_p50_ns=%llu batch_p99_ns=%llu checksum=%lld timer_included=true heap_calls=0_by_inspection\n",
                repeat, unsigned(kernel), count, static_cast<unsigned long long>(samples[count / 2 - 1]),
                static_cast<unsigned long long>(samples[count * 99 / 100 - 1]), static_cast<long long>(checksum));
        }
    }
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
int main(int argc, char** argv) {
    test_depth(); test_imbalance(); test_arithmetic(); test_text();
    if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) bench_imbalance();
    std::puts("Bolt exact depth: depth/arithmetic/format tests passed");
    return 0;
}
