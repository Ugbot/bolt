// test_bolt_query_knobs.cpp — Limits-registry knobs as per-query options
// (G2LAUNCH-156) and off/on/auto mode words (G2LAUNCH-143).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "bolt/bolt_budget.h"
#include "bolt/bolt_query_options.h"

namespace {

constexpr const char* kKnob = "query_claim_share_pct";

bolt::QueryOptionResult apply(const char* k, const char* v, bolt::QueryOptions* o) {
    return bolt::apply_query_option(k, std::strlen(k), v, std::strlen(v), o);
}

}  // namespace

TEST(QueryKnobs, RegistryKnobIsAQueryOption) {
    (void)bolt::bolt_limits_value(bolt::bolt_limits_id::query_claim_share_pct);
    bolt::QueryOptions o{};
    ASSERT_EQ(apply(kKnob, "25", &o), bolt::QueryOptionResult::Ok);
    EXPECT_NE(o.set & bolt::QueryOptions::kKnobs, 0u);
    uint64_t v = 0;
    bolt::QueryKnobSource src{};
    ASSERT_TRUE(bolt::query_knob_get(o.knobs, kKnob, &v, &src));
    EXPECT_EQ(v, 25u);
    EXPECT_EQ(src, bolt::QueryKnobSource::Query);
    EXPECT_EQ(apply("QUERY_CLAIM_SHARE_PCT", "30", &o), bolt::QueryOptionResult::Ok);
    ASSERT_TRUE(bolt::query_knob_get(o.knobs, kKnob, &v));
    EXPECT_EQ(v, 30u);
    EXPECT_EQ(o.knobs.count, 1u);
}

TEST(QueryKnobs, OutOfRangeUnknownAndInvariantRefuse) {
    bolt::QueryOptions o{};
    EXPECT_EQ(apply(kKnob, "0", &o), bolt::QueryOptionResult::BadValue);
    EXPECT_EQ(apply(kKnob, "101", &o), bolt::QueryOptionResult::BadValue);
    EXPECT_EQ(apply(kKnob, "lots", &o), bolt::QueryOptionResult::BadValue);
    EXPECT_EQ(apply("no_such_knob_anywhere", "1", &o), bolt::QueryOptionResult::UnknownKey);
    EXPECT_EQ(apply("max_workers", "2", &o), bolt::QueryOptionResult::UnknownKey);
    EXPECT_EQ(o.knobs.count, 0u);
    EXPECT_EQ(o.set & bolt::QueryOptions::kKnobs, 0u);
}

TEST(QueryKnobs, SetResetAndDefault) {
    bolt::QueryOptions o{};
    uint32_t reset = 0;
    char why[128] = {};
    const char* set = "SET query_claim_share_pct = 40";
    ASSERT_EQ(bolt::parse_query_set(set, std::strlen(set), &o, &reset, why, sizeof(why)),
              bolt::QuerySetResult::Applied);
    uint64_t v = 0;
    ASSERT_TRUE(bolt::query_knob_get(o.knobs, kKnob, &v));
    EXPECT_EQ(v, 40u);
    const char* rs = "RESET query_claim_share_pct";
    ASSERT_EQ(bolt::parse_query_set(rs, std::strlen(rs), &o, &reset, why, sizeof(why)),
              bolt::QuerySetResult::Applied);
    EXPECT_FALSE(bolt::query_knob_get(o.knobs, kKnob, &v));
    ASSERT_EQ(apply(kKnob, "60", &o), bolt::QueryOptionResult::Ok);
    ASSERT_EQ(apply(kKnob, "default", &o), bolt::QueryOptionResult::Ok);
    EXPECT_FALSE(bolt::query_knob_get(o.knobs, kKnob, &v));
    const char* bad = "SET query_claim_share_pct = 900";
    EXPECT_EQ(bolt::parse_query_set(bad, std::strlen(bad), &o, &reset, why, sizeof(why)),
              bolt::QuerySetResult::Invalid);
}

TEST(QueryKnobs, OptionsClauseCarriesKnobs) {
    const char* q = "SELECT 1 OPTIONS (query_claim_share_pct = 10, workers = 2)";
    bolt::QueryOptions o{};
    size_t body = 0;
    char why[128] = {};
    ASSERT_EQ(bolt::split_query_options(q, std::strlen(q), bolt::QueryDialect::Sql, &o, &body, why,
                                        sizeof(why)),
              bolt::QueryClauseResult::Found);
    EXPECT_EQ(body, 8u);
    uint64_t v = 0;
    ASSERT_TRUE(bolt::query_knob_get(o.knobs, kKnob, &v));
    EXPECT_EQ(v, 10u);
    EXPECT_EQ(o.workers, 2u);
}

TEST(QueryKnobs, OverlayQueryWinsOverSession) {
    bolt::QueryOptions session{}, query{};
    ASSERT_EQ(apply(kKnob, "20", &session), bolt::QueryOptionResult::Ok);
    bolt::query_knobs_mark(&session.knobs, bolt::QueryKnobSource::Session);
    ASSERT_EQ(apply(kKnob, "70", &query), bolt::QueryOptionResult::Ok);
    bolt::QueryOptions eff = session;
    bolt::query_options_overlay(&eff, query);
    uint64_t v = 0;
    bolt::QueryKnobSource src{};
    ASSERT_TRUE(bolt::query_knob_get(eff.knobs, kKnob, &v, &src));
    EXPECT_EQ(v, 70u);
    EXPECT_EQ(src, bolt::QueryKnobSource::Query);
}

TEST(QueryKnobs, QueryLimitValueReadsTheRunningQuery) {
    const bolt::LimitTable* t = nullptr;
    uint32_t row = 0;
    ASSERT_TRUE(bolt::limits_find(kKnob, &t, &row));
    const uint64_t process = bolt::limits_value(*t, row);
    EXPECT_EQ(bolt::query_limit_value(*t, row), process);
    bolt::QueryOptions o{};
    ASSERT_EQ(apply(kKnob, "5", &o), bolt::QueryOptionResult::Ok);
    bolt::QueryCtx* q = bolt::query_ctx_acquire(bolt::QueryLimits{}, nullptr);
    ASSERT_NE(q, nullptr);
    q->knobs = &o.knobs;
    {
        bolt::QueryCtxScope scope(q);
        EXPECT_EQ(bolt::query_limit_value(*t, row), 5u);
    }
    EXPECT_EQ(bolt::query_limit_value(*t, row), process);
    bolt::query_ctx_release(q);
}

TEST(LimitModes, WordsParseAnyCaseAndPrint) {
    uint64_t v = 9;
    EXPECT_TRUE(bolt::limits_parse_mode("off", 3, &v));
    EXPECT_EQ(v, bolt::kLimitModeOff);
    EXPECT_TRUE(bolt::limits_parse_mode("ON", 2, &v));
    EXPECT_EQ(v, bolt::kLimitModeOn);
    EXPECT_TRUE(bolt::limits_parse_mode("Auto", 4, &v));
    EXPECT_EQ(v, bolt::kLimitModeAuto);
    EXPECT_FALSE(bolt::limits_parse_mode("maybe", 5, &v));
    EXPECT_FALSE(bolt::limits_parse_mode("of", 2, &v));
    EXPECT_STREQ(bolt::limits_mode_name(bolt::kLimitModeAuto), "auto");
    EXPECT_STREQ(bolt::limits_mode_name(7), "?");
    bolt::LimitDesc d{};
    d.unit = bolt::kLimitUnitMode;
    EXPECT_TRUE(bolt::limits_is_mode(d));
    d.unit = "MiB";
    EXPECT_FALSE(bolt::limits_is_mode(d));
}
