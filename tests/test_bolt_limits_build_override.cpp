// test_bolt_limits_build_override.cpp — bolt's own Limits table compiled with
// BOLT_LIMITS_BUILD set (what -DBOLT_LIMIT_<NAME>=<value> generates).

#include <gtest/gtest.h>

#include <cstdlib>

#include "bolt/bolt_budget.h"
#include "bolt/bolt_limits.h"

TEST(LimitsBuildOverride, BoltTableTakesTheBuildDefault) {
#if defined(_WIN32)
    ::_putenv_s("BOLT_LAKE_METADATA_BUDGET_MB", "");
#else
    ::unsetenv("BOLT_LAKE_METADATA_BUDGET_MB");
#endif
    const bolt::LimitTable& t = bolt::bolt_limits();
    const uint32_t i = static_cast<uint32_t>(bolt::bolt_limits_id::lake_metadata_budget_mb);
    EXPECT_EQ(bolt::limits_value(t, i), 4096u);
    EXPECT_EQ(bolt::limits_source(t, i), bolt::LimitSource::kBuild);
    EXPECT_EQ(t.descs[i].def, 1024u);
    EXPECT_EQ(bolt::limits_first_error(), nullptr);
}
