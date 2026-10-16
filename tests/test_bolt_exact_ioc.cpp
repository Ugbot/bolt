#include "bolt/kernels/fintech/exact_ioc.h"
#include "bolt/kernels/fintech/book_types.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#define CHECK(...) do { if (!(__VA_ARGS__)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #__VA_ARGS__); std::abort(); } } while (false)
namespace {
using namespace bolt::fintech;
using Side = ExactIocSide;
using Status = ExactIocStatus;
struct Level { std::int64_t price_ticks; std::uint32_t quantity; };
struct Wide { std::int64_t price_ticks; std::uint64_t quantity; };
struct Signed { std::int64_t price_ticks; std::int64_t quantity; };
ExactIocResult sentinel() noexcept {
    return {17, 11, 6, 9, -12345, -987, true, {1,2,3,4,5,6,7}};
}
void unchanged(const ExactIocResult& current, const ExactIocResult& previous) noexcept {
    CHECK(std::memcmp(&current, &previous, sizeof(current)) == 0);
}
void check_result(const ExactIocResult& result, std::uint32_t requested, std::uint32_t filled,
    std::int64_t weighted, std::int64_t marginal, std::uint32_t touched, bool limited) noexcept {
    CHECK(result.requested == requested && result.filled == filled && result.remaining == requested - filled);
    CHECK(result.weighted_ticks == weighted && result.marginal_ticks == marginal);
    CHECK(result.levels_touched == touched && result.price_limited == limited);
    for (std::uint32_t i = 0; i < 7; ++i) CHECK(result.reserved[i] == 0);
}
void buy_literals() noexcept {
    const Level levels[] = {{100,3},{102,4},{105,2}};
    std::uint64_t consumed[3]{}; std::uint32_t takes[3]{}; auto result = sentinel();
    CHECK(exact_ioc_plan<Side::Buy,3>(levels,3,consumed,3,8,105,2,takes,3,&result) == Status::Ok);
    check_result(result,8,7,722,104,2,true); // 3*102 + 4*104
    CHECK(takes[0] == 3 && takes[1] == 4 && takes[2] == 0);
    CHECK(consumed[0] == 0 && consumed[1] == 0 && consumed[2] == 0);
    CHECK(exact_ioc_apply<3>(levels,3,consumed,3,takes,3) == Status::Ok);
    CHECK(consumed[0] == 3 && consumed[1] == 4 && consumed[2] == 0);
    CHECK(exact_ioc_plan<Side::Buy,3>(levels,3,consumed,3,4,107,2,takes,3,&result) == Status::Ok);
    check_result(result,4,2,214,107,1,false); // only remaining displayed size, not synthetic replenishment
    CHECK(takes[0] == 0 && takes[1] == 0 && takes[2] == 2);
    CHECK(exact_ioc_apply<3>(levels,3,consumed,3,takes,3) == Status::Ok);
    CHECK(exact_ioc_plan<Side::Buy,3>(levels,3,consumed,3,1,INT64_MAX,2,takes,3,&result) == Status::Ok);
    check_result(result,1,0,0,0,0,false);
    CHECK(consumed[0] == 3 && consumed[1] == 4 && consumed[2] == 2);
}
void exact_and_stressed_limits() noexcept {
    const Level asks[] = {{100,3},{102,4},{105,2}};
    const std::uint64_t consumed[3]{}; std::uint32_t takes[3]{}; ExactIocResult result{};
    CHECK(exact_ioc_plan<Side::Buy,3>(asks,3,consumed,3,7,104,2,takes,3,&result) == Status::Ok);
    check_result(result,7,7,722,104,2,false);
    CHECK(exact_ioc_plan<Side::Buy,3>(asks,3,consumed,3,1,100,2,takes,3,&result) == Status::Ok);
    check_result(result,1,0,0,0,0,true); // displayed100 is allowed; stressed102 is not
    const Level bids[] = {{110,2},{108,3},{100,9}};
    CHECK(exact_ioc_plan<Side::Sell,3>(bids,3,consumed,3,6,106,1,takes,3,&result) == Status::Ok);
    check_result(result,6,5,539,107,2,true); // 2*109 + 3*107
    CHECK(takes[0] == 2 && takes[1] == 3 && takes[2] == 0);
    CHECK(exact_ioc_plan<Side::Sell,3>(bids,3,consumed,3,1,110,1,takes,3,&result) == Status::Ok);
    check_result(result,1,0,0,0,0,true);
}
void negatives_and_empty() noexcept {
    const Level asks[] = {{-10,2},{-5,3}}; const Level bids[] = {{-5,2},{-10,3}};
    std::uint64_t consumed[2]{}; std::uint32_t takes[2]{}; ExactIocResult result{};
    CHECK(exact_ioc_plan<Side::Buy,2>(asks,2,consumed,2,5,-3,2,takes,2,&result) == Status::Ok);
    check_result(result,5,5,-25,-3,2,false); // 2*(-8) + 3*(-3)
    CHECK(exact_ioc_plan<Side::Sell,2>(bids,2,consumed,2,5,-12,2,takes,2,&result) == Status::Ok);
    check_result(result,5,5,-50,-12,2,false); // 2*(-7) + 3*(-12)
    CHECK(exact_ioc_plan<Side::Buy,2,Level>(nullptr,0,nullptr,0,9,INT64_MIN,0,nullptr,0,&result) == Status::Ok);
    check_result(result,9,0,0,0,0,false);
    CHECK(exact_ioc_apply<2,Level>(nullptr,0,nullptr,0,nullptr,0) == Status::Ok);
}
void wide_quantities() noexcept {
    const Wide huge[] = {{1,std::uint64_t(UINT32_MAX)+9}};
    std::uint64_t consumed[1]{}; std::uint32_t takes[1]{}; ExactIocResult result{};
    CHECK(exact_ioc_plan<Side::Buy,1>(huge,1,consumed,1,UINT32_MAX,1,0,takes,1,&result) == Status::Ok);
    check_result(result,UINT32_MAX,UINT32_MAX,4294967295ll,1,1,false);
    CHECK(takes[0] == UINT32_MAX);
    CHECK(exact_ioc_apply<1>(huge,1,consumed,1,takes,1) == Status::Ok && consumed[0] == UINT32_MAX);
    CHECK(exact_ioc_plan<Side::Buy,1>(huge,1,consumed,1,20,1,0,takes,1,&result) == Status::Ok);
    check_result(result,20,9,9,1,1,false);
    const Wide maximum[] = {{-1,UINT64_MAX}}; consumed[0] = UINT64_MAX - 3;
    CHECK(exact_ioc_plan<Side::Sell,1>(maximum,1,consumed,1,5,-1,0,takes,1,&result) == Status::Ok);
    check_result(result,5,3,-3,-1,1,false);
    CHECK(exact_ioc_apply<1>(maximum,1,consumed,1,takes,1) == Status::Ok && consumed[0] == UINT64_MAX);
    const Signed signed_levels[] = {{0,INT64_MAX}}; consumed[0] = 0;
    CHECK(exact_ioc_plan<Side::Buy,1>(signed_levels,1,consumed,1,UINT32_MAX,0,0,takes,1,&result) == Status::Ok);
    check_result(result,UINT32_MAX,UINT32_MAX,0,0,1,false);
    const BookLevel native[] = {{.price_ticks=42,.quantity=2}}; consumed[0] = 0;
    CHECK(exact_ioc_plan<Side::Buy,1>(native,1,consumed,1,1,42,0,takes,1,&result) == Status::Ok);
    check_result(result,1,1,42,42,1,false);
}
void invalid_suffixes() noexcept {
    Level levels[] = {{10,5},{11,2},{12,1}};
    std::uint64_t consumed[3]{}; std::uint32_t takes[3]{9,9,9}; auto result = sentinel(); const auto saved = result;
    for (std::uint32_t mutation = 0; mutation < 5; ++mutation) {
        levels[2] = {12,1}; consumed[2] = 0;
        if (mutation == 0) levels[2].price_ticks = 11; // duplicate suffix
        if (mutation == 1) levels[2].price_ticks = 9; // reversed suffix
        if (mutation == 2) levels[2].quantity = 0;
        if (mutation == 3) consumed[2] = 2;
        if (mutation == 4) levels[1].price_ticks = 8;
        CHECK(exact_ioc_plan<Side::Buy,3>(levels,3,consumed,3,1,100,0,takes,3,&result) == Status::Invalid);
        unchanged(result,saved); CHECK(consumed[0] == 0 && consumed[1] == 0);
        levels[1] = {11,2};
    }
    const Signed negative[] = {{10,3},{11,-1}}; consumed[0] = consumed[1] = 0;
    CHECK(exact_ioc_plan<Side::Buy,3>(negative,2,consumed,3,1,100,0,takes,3,&result) == Status::Invalid);
    unchanged(result,saved);
    const Level malformed_bids[] = {{20,3},{21,1}};
    CHECK(exact_ioc_plan<Side::Sell,3>(malformed_bids,2,consumed,3,1,0,0,takes,3,&result) == Status::Invalid);
    unchanged(result,saved);
}
void arithmetic_refusal() noexcept {
    std::uint64_t consumed[2]{}; std::uint32_t takes[2]{}; auto result = sentinel(); const auto saved = result;
    const Level asks[] = {{1,1},{INT64_MAX,1}};
    CHECK(exact_ioc_plan<Side::Buy,2>(asks,2,consumed,2,1,INT64_MAX,1,takes,2,&result) == Status::Ok);
    check_result(result,1,1,2,2,1,false); // unused far adverse overflow is irrelevant
    result = saved;
    CHECK(exact_ioc_plan<Side::Buy,2>(asks,2,consumed,2,2,INT64_MAX,1,takes,2,&result) == Status::Overflow);
    unchanged(result,saved); CHECK(takes[0] == 1 && consumed[0] == 0 && consumed[1] == 0);
    const Level bids[] = {{0,1},{INT64_MIN,1}};
    CHECK(exact_ioc_plan<Side::Sell,2>(bids,2,consumed,2,2,INT64_MIN,1,takes,2,&result) == Status::Overflow);
    unchanged(result,saved); CHECK(takes[0] == 1);
    const Level sum[] = {{INT64_MAX/2+1,1},{INT64_MAX/2+2,1}};
    CHECK(exact_ioc_plan<Side::Buy,2>(sum,2,consumed,2,2,INT64_MAX,0,takes,2,&result) == Status::Overflow);
    unchanged(result,saved); CHECK(takes[0] == 1);
    const Level product[] = {{1,1},{INT64_MAX,2}};
    CHECK(exact_ioc_plan<Side::Buy,2>(product,2,consumed,2,3,INT64_MAX,0,takes,2,&result) == Status::Overflow);
    unchanged(result,saved); CHECK(takes[0] == 1);
    const Level negative_sum[] = {{INT64_MIN/2,1},{INT64_MIN/2-1,1}};
    CHECK(exact_ioc_plan<Side::Sell,2>(negative_sum,2,consumed,2,2,INT64_MIN,0,takes,2,&result) == Status::Overflow);
    unchanged(result,saved); CHECK(takes[0] == 1);
    const Level low[] = {{INT64_MIN,1}};
    CHECK(exact_ioc_plan<Side::Buy,2>(low,1,consumed,2,1,INT64_MIN,0,takes,2,&result) == Status::Ok);
    check_result(result,1,1,INT64_MIN,INT64_MIN,1,false);
    const Level high[] = {{INT64_MAX,1}};
    CHECK(exact_ioc_plan<Side::Sell,2>(high,1,consumed,2,1,INT64_MAX,0,takes,2,&result) == Status::Ok);
    check_result(result,1,1,INT64_MAX,INT64_MAX,1,false);
}
void unavailable_and_unused_prices() noexcept {
    const Level asks[] = {{1,1},{INT64_MAX,1}}, bids[] = {{0,1},{INT64_MIN,1}};
    std::uint64_t consumed[3]{0,1,0}; std::uint32_t takes[3]{9,9,88}; ExactIocResult result{};
    CHECK(exact_ioc_plan<Side::Buy,3>(asks,2,consumed,3,2,INT64_MAX,1,takes,3,&result) == Status::Ok);
    check_result(result,2,1,2,2,1,false); // unavailable far price is never stressed
    CHECK(takes[0] == 1 && takes[1] == 0 && takes[2] == 88);
    consumed[1] = 0;
    CHECK(exact_ioc_plan<Side::Sell,3>(bids,2,consumed,3,1,INT64_MIN,1,takes,3,&result) == Status::Ok);
    check_result(result,1,1,-1,-1,1,false); // filled prefix avoids unused far underflow
    consumed[1] = 1;
    CHECK(exact_ioc_plan<Side::Sell,3>(bids,2,consumed,3,2,INT64_MIN,1,takes,3,&result) == Status::Ok);
    check_result(result,2,1,-1,-1,1,false);
    CHECK(consumed[0] == 0 && consumed[1] == 1 && consumed[2] == 0 && takes[2] == 88);
}
void atomic_apply() noexcept {
    const Wide levels[] = {{10,3},{11,2}}; std::uint64_t consumed[2]{1,1};
    std::uint32_t takes[2]{1,2};
    CHECK(exact_ioc_apply<2>(levels,2,consumed,2,takes,2) == Status::Invalid);
    CHECK(consumed[0] == 1 && consumed[1] == 1); // valid first take must not leak through bad second
    takes[0] = 2; takes[1] = 1;
    CHECK(exact_ioc_apply<2>(levels,2,consumed,2,takes,2) == Status::Ok);
    CHECK(consumed[0] == 3 && consumed[1] == 2);
    CHECK(exact_ioc_apply<2>(levels,2,consumed,2,takes,2) == Status::Invalid);
    CHECK(consumed[0] == 3 && consumed[1] == 2);
    consumed[1] = 3;
    CHECK(exact_ioc_apply<2>(levels,2,consumed,2,takes,2) == Status::Invalid);
    CHECK(consumed[0] == 3 && consumed[1] == 3);
    const Signed negative[] = {{10,3},{11,-1}}; consumed[0] = consumed[1] = 0;
    CHECK(exact_ioc_apply<2>(negative,2,consumed,2,takes,2) == Status::Invalid);
    CHECK(consumed[0] == 0 && consumed[1] == 0);
}
void span_refusals() noexcept {
    Level levels[3]{{10,3},{11,2},{12,1}}, levels_before[3]{};
    std::memcpy(levels_before,levels,sizeof(levels));
    std::uint64_t consumed[8]{}; std::uint32_t takes[3]{8,8,8}; auto result = sentinel(); const auto saved = result;
    CHECK(exact_ioc_plan<Side::Buy,3>(levels,3,consumed,3,1,10,0,takes,2,&result) == Status::Invalid);
    CHECK(exact_ioc_plan<Side::Buy,3>(levels,3,consumed,2,1,10,0,takes,3,&result) == Status::Invalid);
    CHECK(exact_ioc_plan<Side::Buy,3>(levels,4,consumed,3,1,10,0,takes,3,&result) == Status::Invalid);
    CHECK(exact_ioc_plan<Side::Buy,3>(levels,3,consumed,3,0,10,0,takes,3,&result) == Status::Invalid);
    CHECK(exact_ioc_plan<Side::Buy,3>(levels,3,consumed,3,1,10,-1,takes,3,&result) == Status::Invalid);
    CHECK(exact_ioc_plan<Side::Buy,3>(levels,3,consumed,3,1,10,0,takes,3,nullptr) == Status::Invalid);
    CHECK(exact_ioc_plan<Side::Buy,3,Level>(nullptr,1,consumed,3,1,10,0,takes,3,&result) == Status::Invalid);
    CHECK(exact_ioc_plan<Side::Buy,3>(levels,3,consumed,3,1,10,0,takes,UINT32_MAX,&result) == Status::Invalid);
    CHECK(exact_ioc_plan<Side::Buy,3>(levels,3,consumed,3,1,10,0,reinterpret_cast<std::uint32_t*>(consumed),3,&result) == Status::Invalid);
    CHECK(exact_ioc_plan<Side::Buy,3>(levels,3,consumed,3,1,10,0,reinterpret_cast<std::uint32_t*>(levels),3,&result) == Status::Invalid);
    CHECK(exact_ioc_plan<Side::Buy,3>(levels,3,consumed,3,1,10,0,reinterpret_cast<std::uint32_t*>(&result),3,&result) == Status::Invalid);
    CHECK(exact_ioc_plan<Side::Buy,3>(levels,3,consumed,3,1,10,0,takes,3,reinterpret_cast<ExactIocResult*>(levels)) == Status::Invalid);
    CHECK(exact_ioc_plan<Side::Buy,3>(levels,3,consumed,3,1,10,0,takes,3,reinterpret_cast<ExactIocResult*>(consumed)) == Status::Invalid);
    auto* bad = reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::byte*>(takes)+1);
    CHECK(exact_ioc_plan<Side::Buy,3>(levels,3,consumed,3,1,10,0,bad,3,&result) == Status::Invalid);
    const auto* impossible = reinterpret_cast<const Level*>(std::numeric_limits<std::uintptr_t>::max()-7);
    CHECK(exact_ioc_plan<Side::Buy,3>(impossible,1,consumed,3,1,10,0,takes,3,&result) == Status::Invalid);
    unchanged(result,saved); CHECK(std::memcmp(levels,levels_before,sizeof(levels)) == 0);
    CHECK(takes[0] == 8 && takes[1] == 8 && takes[2] == 8);
    for (std::uint32_t i = 0; i < 8; ++i) CHECK(consumed[i] == 0);
    CHECK(exact_ioc_apply<3>(levels,3,consumed,3,reinterpret_cast<const std::uint32_t*>(consumed),3) == Status::Invalid);
    CHECK(exact_ioc_apply<3>(levels,3,reinterpret_cast<std::uint64_t*>(levels),3,takes,3) == Status::Invalid);
    for (std::uint32_t i = 0; i < 8; ++i) CHECK(consumed[i] == 0);
    CHECK(std::memcmp(levels,levels_before,sizeof(levels)) == 0);
}
void old_arithmetic_contract() noexcept {
    std::int64_t value = 77;
    CHECK(!exact_add(INT64_MAX,1,value) && value == 77);
    CHECK(!exact_sub(INT64_MIN,1,value) && value == 77);
    CHECK(!exact_multiply(INT64_MIN,2,value) && value == 77);
    CHECK(exact_multiply(INT64_MIN,1,value) && value == INT64_MIN);
}
} // namespace
int main() {
    buy_literals(); exact_and_stressed_limits(); negatives_and_empty(); wide_quantities();
    invalid_suffixes(); arithmetic_refusal(); unavailable_and_unused_prices();
    atomic_apply(); span_refusals(); old_arithmetic_contract();
    std::puts("Bolt exact IOC: literal/limits/overlay/wide/refusal/atomic-apply checks passed");
}
