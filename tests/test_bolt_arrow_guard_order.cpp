// G2ARROW-17: bolt must share a translation unit with a real Arrow C ABI
// header IN EITHER INCLUDE ORDER. bolt_types.h used to define the standard
// ARROW_C_DATA_INTERFACE guard INSIDE namespace bolt, so including bolt
// first suppressed arrow/c/abi.h's global struct definitions while providing
// only namespaced ones -- "incomplete type ArrowSchema" in any mixed TU --
// which defeats the interface's stated purpose ("export to any Arrow
// consumer without linking libarrow"). The reverse order happened to work,
// which is exactly why it survived.
//
// This TU is order A: the (spec-verbatim) ABI header FIRST, then bolt. Its
// sibling test_bolt_arrow_guard_order_b.cpp is order B: bolt first, then the
// ABI header -- the order that failed to compile before the fix (proven by
// reverting bolt_types.h to the in-namespace block: order B stops building
// with "must use 'struct' tag / unknown type name" on ::ArrowSchema use
// while this TU still compiles).
//
// Both TUs link into ONE binary, so the types must also be identical enough
// not to trip the linker, and both exercise bolt's exporter through the
// GLOBAL ::ArrowSchema/::ArrowArray -- the type an external consumer hands us.

#include "arrow_abi_mimic.h"   // order A: ABI first

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"   // then bolt

#include <gtest/gtest.h>

#include <cstring>

// Compile-time proof the global-scope types are THE types bolt fills.
static_assert(sizeof(::ArrowSchema) == sizeof(void*) * 9,
              "spec layout: 9 pointer-sized fields");
static_assert(sizeof(::ArrowArray) == sizeof(void*) * 10,
              "spec layout: 10 pointer-sized fields");

namespace {

TEST(BoltArrowGuardOrder, AbiFirstThenBoltFillsGlobalStructs) {
    bolt::Arena a;
    bolt::BoltColumn c = bolt::BoltColumn::make_flat_alloc(
        4, bolt::BoltType::Int64, &a);
    ASSERT_NE(c.data, nullptr);
    auto* p = static_cast<std::int64_t*>(c.data);
    for (int i = 0; i < 4; ++i) p[i] = 10 + i;

    ::ArrowSchema s;   // the GLOBAL type, as an external consumer declares it
    ::ArrowArray arr;
    c.fill_arrow_schema(&s, "col");
    c.fill_arrow_array(&arr);
    EXPECT_STREQ(s.format, "l");
    EXPECT_STREQ(s.name, "col");
    EXPECT_EQ(arr.length, 4);
    ASSERT_GE(arr.n_buffers, 2);
    EXPECT_EQ(static_cast<const std::int64_t*>(arr.buffers[1])[3], 13);
}

}  // namespace
