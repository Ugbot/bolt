// bolt_nested_row.h: a nested row encodes to deterministic bytes and decodes back to the same value (re-encoding is byte-equal),
// for struct / list / map at depth, with NULL structs, NULL and empty lists, NULL elements and var leaves.
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include "bolt/bolt_nested_row.h"

namespace nr = bolt::nested_row;
using bolt::BoltColumn;
using bolt::BoltType;

namespace {

uint8_t* bits(bolt::Arena* a, const std::vector<int>& v) {
    auto* b = static_cast<uint8_t*>(a->allocate_zeroed(v.size() / 8 + 1));
    for (size_t i = 0; i < v.size(); ++i) if (v[i]) b[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
    return b;
}

BoltColumn i64(bolt::Arena* a, const std::vector<int64_t>& v, uint8_t* valid = nullptr) {
    auto* d = a->allocate_array<int64_t>(v.size() > 0 ? v.size() : 1);
    for (size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    return BoltColumn::make_flat(d, valid, static_cast<int64_t>(v.size()), BoltType::Int64);
}

BoltColumn str(bolt::Arena* a, const std::vector<const char*>& v, uint8_t* valid = nullptr) {
    auto* offs = a->allocate_array<int32_t>(v.size() + 1);
    size_t total = 0;
    for (const char* s : v) total += std::strlen(s);
    auto* bytes = static_cast<uint8_t*>(a->allocate(total + 1));
    offs[0] = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        const size_t l = std::strlen(v[i]);
        std::memcpy(bytes + offs[i], v[i], l);
        offs[i + 1] = offs[i] + static_cast<int32_t>(l);
    }
    return BoltColumn::make_var_binary(bytes, valid, offs, static_cast<int64_t>(v.size()), BoltType::Utf8, a);
}

std::vector<std::vector<uint8_t>> encode_all(const nr::TypeTree& t, const BoltColumn& c) {
    std::vector<std::vector<uint8_t>> rows;
    for (int64_t r = 0; r < c.length; ++r) {
        size_t need = nr::encode_row(t, c, r, nullptr, 0);
        EXPECT_GT(need, 0u) << "row " << r;
        std::vector<uint8_t> buf(need);
        EXPECT_EQ(nr::encode_row(t, c, r, buf.data(), buf.size()), need);
        rows.push_back(buf);
    }
    return rows;
}

void round_trip(const BoltColumn& c) {
    nr::TypeTree t;
    ASSERT_TRUE(nr::tree_from_column(c, &t));
    const auto rows = encode_all(t, c);
    bolt::Arena a;
    nr::Decoder d;
    ASSERT_TRUE(nr::decoder_init(&d, &t, &a));
    for (const auto& r : rows) ASSERT_TRUE(nr::decoder_push(&d, r.data(), r.size()));
    BoltColumn back{};
    ASSERT_TRUE(nr::decoder_finish(d, &back));
    ASSERT_EQ(back.length, c.length);
    const auto again = encode_all(t, back);
    ASSERT_EQ(again, rows);
}

}  // namespace

TEST(NestedRow, StructWithNullStructAndNullFields) {
    bolt::Arena a;
    BoltColumn f[2] = {i64(&a, {1, 2, 3, 4}, bits(&a, {1, 0, 1, 1})), str(&a, {"a", "", "ccc", "dd"}, bits(&a, {1, 1, 0, 1}))};
    BoltColumn s = BoltColumn::make_struct(f, 2, 4, bits(&a, {1, 1, 0, 1}), &a);
    round_trip(s);
}

TEST(NestedRow, ListOfStructWithNullAndEmptyLists) {
    bolt::Arena a;
    BoltColumn f[2] = {i64(&a, {10, 20, 30, 40, 50}), str(&a, {"x", "y", "z", "w", "v"}, bits(&a, {1, 0, 1, 1, 1}))};
    BoltColumn el = BoltColumn::make_struct(f, 2, 5, bits(&a, {1, 1, 0, 1, 1}), &a);
    static const int32_t offs[5] = {0, 2, 2, 2, 5};             // [2 items], null, [], [3 items]
    BoltColumn l = BoltColumn::make_list(&el, offs, 4, bits(&a, {1, 0, 1, 1}), &a);
    round_trip(l);
}

TEST(NestedRow, MapAndListOfList) {
    bolt::Arena a;
    BoltColumn kv[2] = {str(&a, {"k1", "k2", "k3"}), i64(&a, {7, 8, 9}, bits(&a, {1, 0, 1}))};
    BoltColumn ent = BoltColumn::make_struct(kv, 2, 3, nullptr, &a);
    static const int32_t moffs[4] = {0, 2, 2, 3};
    BoltColumn m = BoltColumn::make_list(&ent, moffs, 3, nullptr, &a);
    m.type = BoltType::Map;
    round_trip(m);
    BoltColumn leaf = i64(&a, {1, 2, 3, 4, 5, 6}, bits(&a, {1, 1, 0, 1, 1, 1}));
    static const int32_t ioffs[5] = {0, 3, 3, 4, 6};
    BoltColumn inner = BoltColumn::make_list(&leaf, ioffs, 4, bits(&a, {1, 0, 1, 1}), &a);
    static const int32_t ooffs[4] = {0, 2, 2, 4};
    BoltColumn outer = BoltColumn::make_list(&inner, ooffs, 3, bits(&a, {1, 1, 1}), &a);
    round_trip(outer);
}

TEST(NestedRow, MismatchAndCorruptInputAreRefused) {
    bolt::Arena a;
    BoltColumn f[1] = {i64(&a, {1, 2})};
    BoltColumn s = BoltColumn::make_struct(f, 1, 2, nullptr, &a);
    nr::TypeTree t;
    ASSERT_TRUE(nr::tree_from_column(s, &t));
    nr::Decoder d;
    ASSERT_TRUE(nr::decoder_init(&d, &t, &a));
    const uint8_t short_row[3] = {1, 2, 3};
    EXPECT_FALSE(nr::decoder_push(&d, short_row, 3));          // truncated
    EXPECT_FALSE(nr::decoder_push(&d, nullptr, 0));            // a NULL row of a required root
}
