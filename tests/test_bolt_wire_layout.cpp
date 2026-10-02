// test_bolt_wire_layout.cpp — B5: the persisted wire bytes sit where the
// layout constants say (bolt_wire_layout.h holds the compile-time half).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/wire/bolt_wire.h"
#include "bolt/wire/bolt_wire_layout.h"

using namespace bolt;
namespace L = bolt::wire::layout;

TEST(WireLayout, SchemaAndDescriptorBytesAtDeclaredOffsets) {
    Arena a;
    BoltBatch b;
    BoltBatch::init_empty(&b);
    ASSERT_TRUE(BoltBatch::alloc_columns(&b, &a, 1));
    b.num_rows = 3;
    b.schema.num_fields = 1;
    BoltField& f = b.schema.fields[0];
    std::memset(&f, 0, sizeof(f));
    f.set_name("price");
    f.type = BoltType::Decimal64;
    f.nullable = true;
    BoltColumn c = BoltColumn::make_flat_alloc(3, BoltType::Decimal64, &a);
    c.decimal_scale = 4;
    int64_t* v = static_cast<int64_t*>(c.data);
    v[0] = 1; v[1] = 2; v[2] = 3;
    b.columns[0][0] = c;
    b.columns[1][0] = c;

    const size_t n = wire::bolt_wire_size(&b);
    std::vector<uint8_t> buf(n);
    ASSERT_EQ(wire::bolt_wire_serialize(&b, buf.data(), n), n);
    const uint8_t* e = buf.data() + L::kHeaderBytes;
    EXPECT_STREQ(reinterpret_cast<const char*>(e), "price");
    EXPECT_EQ(e[L::kSchemaTypeOff], 44u);
    EXPECT_EQ(e[L::kSchemaFormatOff], 0u);
    EXPECT_EQ(e[L::kSchemaNullableOff], 1u);
    EXPECT_EQ(e[L::kSchemaScaleOff], 4u);
    const uint8_t* d = e + L::kSchemaEntryBytes;
    EXPECT_EQ(d[L::kDescFormatOff], 0u);
    uint64_t b1_len = 0;
    std::memcpy(&b1_len, d + 24, 8);
    EXPECT_EQ(b1_len, 24u);
}

TEST(WireLayout, BoltColumnIsFixedSize) {
    EXPECT_EQ(sizeof(BoltColumn), 256u);
    EXPECT_EQ(sizeof(StringView), 16u);
    EXPECT_EQ(sizeof(ZoneMap), 32u);
}
