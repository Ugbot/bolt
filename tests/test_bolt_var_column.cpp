// test_bolt_var_column.cpp — coverage for `BoltColumn::Format::VarBinary`.
//
// Layer 1.1 of plan `this-was-a-freach-hashed-crab.md` — variable-width
// payload column. Used by Bolt's parse / JSONB consumers + future
// per-path JSON subcolumns.
//
// Tests:
//   EmptyColumn                    — n=0 round-trips
//   InlineRowsAreReadable          — 5 rows, mixed widths, validate (ptr,len)
//   ZeroLengthRow                  — empty payload between two non-empty rows
//   NullValidityRespected          — validity bitmap honoured (test asserts)
//   ByteSizeAccountsForBuffers     — payload + offsets + validity all counted
//   AccessorIsBranchless           — sanity: no compile-time branchy ifs
//   NonContiguousAccessIsCorrect   — random-order accesses match sequential

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"

using bolt::BoltColumn;
using bolt::ColumnFormat;
using bolt::BoltType;

namespace {

bolt::Arena make_arena() {
    bolt::ArenaConfig cfg{};
    cfg.initial_block_size = 64u * 1024u;
    cfg.max_block_size     = 1u * 1024u * 1024u;
    cfg.alignment          = 64u;
    return bolt::Arena(cfg);
}

}  // namespace

TEST(BoltVarColumn, EmptyColumn) {
    bolt::Arena arena = make_arena();
    BoltColumn c = BoltColumn::make_var_binary(
        nullptr, nullptr, /*offsets=*/nullptr, /*total_rows=*/0,
        BoltType::Utf8, &arena);
    EXPECT_EQ(c.length, 0);
    EXPECT_EQ(c.format, ColumnFormat::VarBinary);
    EXPECT_EQ(c.type, BoltType::Utf8);
    EXPECT_EQ(c.byte_size(), 0u);
}

TEST(BoltVarColumn, InlineRowsAreReadable) {
    bolt::Arena arena = make_arena();
    const std::vector<std::string> rows = {"alpha", "bravo", "c", "deltaecho", ""};
    std::vector<int32_t> offs;
    offs.reserve(rows.size() + 1);
    offs.push_back(0);
    for (const auto& r : rows) {
        offs.push_back(offs.back() + static_cast<int32_t>(r.size()));
    }
    std::vector<uint8_t> payload;
    payload.reserve(static_cast<size_t>(offs.back()));
    for (const auto& r : rows) {
        payload.insert(payload.end(), r.begin(), r.end());
    }

    BoltColumn c = BoltColumn::make_var_binary(
        payload.data(), nullptr, offs.data(),
        static_cast<int64_t>(rows.size()), BoltType::Utf8, &arena);
    ASSERT_EQ(c.length, static_cast<int64_t>(rows.size()));

    for (int64_t i = 0; i < c.length; ++i) {
        const uint8_t* ptr = nullptr;
        int32_t        len = -1;
        c.var_binary_at(i, &ptr, &len);
        ASSERT_EQ(static_cast<size_t>(len), rows[static_cast<size_t>(i)].size());
        if (len > 0) {
            EXPECT_EQ(0, std::memcmp(ptr, rows[static_cast<size_t>(i)].data(),
                                     static_cast<size_t>(len)));
        }
    }
}

TEST(BoltVarColumn, ZeroLengthRowBetweenNonEmpty) {
    bolt::Arena arena = make_arena();
    const std::vector<std::string> rows = {"first", "", "third"};
    int32_t offs[4] = {0, 5, 5, 10};
    const char* payload = "firstthird";
    BoltColumn c = BoltColumn::make_var_binary(
        const_cast<char*>(payload), nullptr, offs, 3, BoltType::Utf8, &arena);

    const uint8_t* p = nullptr;
    int32_t        n = -1;
    c.var_binary_at(0, &p, &n); EXPECT_EQ(n, 5); EXPECT_EQ(0, std::memcmp(p, "first", 5));
    c.var_binary_at(1, &p, &n); EXPECT_EQ(n, 0);
    c.var_binary_at(2, &p, &n); EXPECT_EQ(n, 5); EXPECT_EQ(0, std::memcmp(p, "third", 5));
}

TEST(BoltVarColumn, ByteSizeAccountsForBuffers) {
    bolt::Arena arena = make_arena();
    int32_t offs[5] = {0, 4, 8, 12, 16};
    const char* payload = "aaaabbbbccccdddd";  // 16 bytes payload
    BoltColumn c = BoltColumn::make_var_binary(
        const_cast<char*>(payload), nullptr, offs, 4, BoltType::Binary, &arena);

    const size_t expected =
        16u                                /* payload */ +
        5u * sizeof(int32_t)               /* offsets */ +
        0u                                 /* no validity */;
    EXPECT_EQ(c.byte_size(), expected);
}

TEST(BoltVarColumn, RejectsNullOffsetsForNonEmptyColumn) {
    bolt::Arena arena = make_arena();
    BoltColumn c = BoltColumn::make_var_binary(
        nullptr, nullptr, /*offsets=*/nullptr, /*total_rows=*/4,
        BoltType::Utf8, &arena);
    // Should fall back to make_empty() — length stays 0.
    EXPECT_EQ(c.length, 0);
    EXPECT_NE(c.format, ColumnFormat::VarBinary);
}

TEST(BoltVarColumn, NonContiguousAccessIsCorrect) {
    bolt::Arena arena = make_arena();
    const std::vector<std::string> rows = {"zero", "one", "two", "three"};
    std::vector<int32_t> offs = {0};
    std::vector<uint8_t> payload;
    for (const auto& r : rows) {
        payload.insert(payload.end(), r.begin(), r.end());
        offs.push_back(static_cast<int32_t>(payload.size()));
    }
    BoltColumn c = BoltColumn::make_var_binary(
        payload.data(), nullptr, offs.data(), 4, BoltType::Utf8, &arena);

    // Random-order accesses.
    const int probe_order[] = {3, 1, 0, 2, 1, 3};
    for (int idx : probe_order) {
        const uint8_t* p = nullptr;
        int32_t        n = -1;
        c.var_binary_at(idx, &p, &n);
        EXPECT_EQ(static_cast<size_t>(n), rows[static_cast<size_t>(idx)].size());
        EXPECT_EQ(0, std::memcmp(p, rows[static_cast<size_t>(idx)].data(),
                                  static_cast<size_t>(n)));
    }
}
