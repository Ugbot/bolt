// test_bolt_partition_route.cpp — partition_route groups rows by key values
// (NULL its own value), first-seen partition order, stable rows per partition.

#include <gtest/gtest.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "bolt/kernels/bolt_partition_route.h"

namespace {

using bolt::BoltColumn;
using bolt::BoltType;
using bolt::PartitionRouteStatus;
using bolt::StringView;

TEST(PartitionRoute, CompositeKeysWithNullsGroupStably) {
    bolt::Arena ar;
    const size_t n = 1000;
    std::vector<int64_t> a(n);
    std::vector<uint8_t> a_valid((n + 7) / 8, 0xFF);
    std::vector<StringView> b(n);
    std::vector<std::string> strs(n);
    std::string over;
    for (size_t r = 0; r < n; ++r) {
        a[r] = static_cast<int64_t>(r % 7);
        if (r % 7 == 3) a_valid[r / 8] &= static_cast<uint8_t>(~(1u << (r % 8)));
        strs[r] = (r % 2 == 0) ? "even-and-longer-than-twelve" : "odd";
    }
    for (size_t r = 0; r < n; ++r) {
        std::memset(&b[r], 0, sizeof(StringView));
        b[r].length = static_cast<uint32_t>(strs[r].size());
        std::memcpy(b[r].prefix, strs[r].data(), std::min<size_t>(strs[r].size(), 12));
        if (strs[r].size() > 12) { b[r].ref.offset = static_cast<uint32_t>(over.size()); over += strs[r]; }
    }
    BoltColumn cols[3];
    cols[0] = BoltColumn::make_flat(a.data(), a_valid.data(), static_cast<int64_t>(n), BoltType::Int64);
    cols[1] = BoltColumn::make_flat(b.data(), nullptr, static_cast<int64_t>(n), BoltType::Utf8);
    cols[1].type_size_bytes = 16;
    cols[1].str_overflow_base = over.data();
    std::vector<double> payload(n, 1.0);
    cols[2] = BoltColumn::make_flat(payload.data(), nullptr, static_cast<int64_t>(n), BoltType::Float64);
    const uint32_t keys[2] = {0, 1};
    bolt::PartitionRoute pr;
    ASSERT_EQ(bolt::partition_route(cols, keys, 2, static_cast<int64_t>(n), 64, &ar, &pr), PartitionRouteStatus::kOk);
    // (r % 7, r % 2) has 14 combinations; r % 7 == 3 is NULL.
    EXPECT_EQ(pr.n_parts, 14u);
    std::map<std::pair<int64_t, int>, uint32_t> seen;
    for (uint32_t p = 0; p < pr.n_parts; ++p) {
        ASSERT_LT(pr.offsets[p], pr.offsets[p + 1]);
        const uint32_t f = pr.first_row[p];
        if (p > 0) EXPECT_GT(f, pr.first_row[p - 1]) << "first-seen order";
        for (uint32_t i = pr.offsets[p]; i < pr.offsets[p + 1]; ++i) {
            const uint32_t r = pr.rows[i];
            if (i > pr.offsets[p]) EXPECT_GT(r, pr.rows[i - 1]) << "stable";
            EXPECT_EQ(pr.part_of_row[r], p);
            EXPECT_EQ(r % 7, f % 7);
            EXPECT_EQ(r % 2, f % 2);
        }
        seen[{f % 7, static_cast<int>(f % 2)}] = p;
    }
    EXPECT_EQ(seen.size(), 14u);
    EXPECT_EQ(pr.offsets[pr.n_parts], n);
}

TEST(PartitionRoute, RefusesPastTheBudgetAndUnkeyableTypes) {
    bolt::Arena ar;
    std::vector<int64_t> a(100);
    for (size_t r = 0; r < a.size(); ++r) a[r] = static_cast<int64_t>(r);
    BoltColumn c = BoltColumn::make_flat(a.data(), nullptr, 100, BoltType::Int64);
    const uint32_t k = 0;
    bolt::PartitionRoute pr;
    EXPECT_EQ(bolt::partition_route(&c, &k, 1, 100, 99, &ar, &pr), PartitionRouteStatus::kTooManyParts);
    EXPECT_EQ(bolt::partition_route(&c, &k, 1, 100, 100, &ar, &pr), PartitionRouteStatus::kOk);
    EXPECT_EQ(pr.n_parts, 100u);
    EXPECT_EQ(bolt::partition_route(&c, &k, 1, 0, 1, &ar, &pr), PartitionRouteStatus::kOk);
    EXPECT_EQ(pr.n_parts, 0u);
}

}  // namespace
