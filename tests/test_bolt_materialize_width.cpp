// G2GRAPH-359 — materialize() of an encoded Int32 column must produce a Flat
// column as wide as its type. A Sequence's (and an int64 Constant's)
// type_size_bytes is the width of its stored seed, 8, so sizing the Flat
// buffer by it wrote 8-byte values under an Int32 type: every reader indexing
// by type then saw v0's low word, v0's high word, v1's low word, ...
// Found by chukonu's graph relation decoder fuzz (fuzz_graph_relation).

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

TEST(MaterializeWidth, Int32SequenceIsInt32Wide) {
    bolt::Arena arena;
    const bolt::BoltColumn seq = bolt::BoltColumn::make_sequence(41, 2, 5, bolt::BoltType::Int32);
    const bolt::BoltColumn flat = seq.materialize(&arena);
    ASSERT_EQ(flat.format, bolt::ColumnFormat::Flat);
    ASSERT_EQ(flat.type, bolt::BoltType::Int32);
    ASSERT_EQ(flat.type_size_bytes, 4u);
    const auto* v = static_cast<const int32_t*>(flat.data);
    for (int32_t i = 0; i < 5; ++i) EXPECT_EQ(v[i], 41 + 2 * i);
}

TEST(MaterializeWidth, Int64SequenceUnchanged) {
    bolt::Arena arena;
    const bolt::BoltColumn seq = bolt::BoltColumn::make_sequence(-7, 3, 4, bolt::BoltType::Int64);
    const bolt::BoltColumn flat = seq.materialize(&arena);
    ASSERT_EQ(flat.type_size_bytes, 8u);
    const auto* v = static_cast<const int64_t*>(flat.data);
    for (int64_t i = 0; i < 4; ++i) EXPECT_EQ(v[i], -7 + 3 * i);
}

TEST(MaterializeWidth, Int64ConstantTypedInt32IsInt32Wide) {
    bolt::Arena arena;
    const bolt::BoltColumn k = bolt::BoltColumn::make_constant<int64_t>(-123456, 6, bolt::BoltType::Int32);
    const bolt::BoltColumn flat = k.materialize(&arena);
    ASSERT_EQ(flat.type_size_bytes, 4u);
    const auto* v = static_cast<const int32_t*>(flat.data);
    for (int i = 0; i < 6; ++i) EXPECT_EQ(v[i], -123456);
}

}  // namespace
