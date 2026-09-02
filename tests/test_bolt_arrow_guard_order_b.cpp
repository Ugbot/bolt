// G2ARROW-17, order B: bolt FIRST, then the (spec-verbatim) Arrow C ABI
// header. This is the order that could not compile before the fix -- bolt's
// in-namespace definition claimed the global ARROW_C_DATA_INTERFACE guard,
// so the ABI header was skipped and no global ::ArrowSchema existed. See
// test_bolt_arrow_guard_order.cpp (order A) for the full story.

#include "bolt/bolt_arena.h"   // order B: bolt first
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"

#include "arrow_abi_mimic.h"   // then the ABI header

#include <gtest/gtest.h>

#include <cstring>

namespace {

TEST(BoltArrowGuardOrder, BoltFirstThenAbiFillsGlobalStructs) {
    bolt::Arena a;
    bolt::BoltColumn c = bolt::BoltColumn::make_flat_alloc(
        3, bolt::BoltType::Float64, &a);
    ASSERT_NE(c.data, nullptr);
    auto* p = static_cast<double*>(c.data);
    p[0] = 0.5; p[1] = 1.5; p[2] = 2.5;

    ::ArrowSchema s;
    ::ArrowArray arr;
    c.fill_arrow_schema(&s, "d");
    c.fill_arrow_array(&arr);
    EXPECT_STREQ(s.format, "g");
    EXPECT_EQ(arr.length, 3);
    ASSERT_GE(arr.n_buffers, 2);
    EXPECT_EQ(static_cast<const double*>(arr.buffers[1])[2], 2.5);
}

}  // namespace
