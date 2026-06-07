// test_bolt_filter_null.cpp — null-bitmap-aware filter kernels.
//
// Golden coverage for SQL three-valued logic in
// bolt/kernels/bolt_filter_null.h. The kernel had ZERO tests; this file
// pins its observable behaviour so the (already-branchless) emit path can
// never silently drift. All data is deterministic — no RNG, no time.
//
// Contract under test (Arrow validity bitmap: LSB-first, bit i = row i,
// 1 = valid, 0 = NULL; nulls == nullptr ⇒ "no nulls" fast path):
//   * cmp<Op,T>      : a row passes iff (lhs OP rhs) AND both operands valid.
//                      A NULL on either side ⇒ never passes (3-valued logic).
//   * cmp_scalar<Op> : same against a broadcast scalar rhs.
//   * cmp_is_null    : passes iff the row is NULL (0 matches when no bitmap).
//   * cmp_is_not_null: passes iff the row is valid (all rows when no bitmap).
//   * Dense (in_sel == nullptr) and sparse (in_sel != nullptr) paths agree.
//   * All-valid bitmap path == no-validity (nullptr) path, bit-for-bit.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bolt/kernels/bolt_filter_null.h"

using namespace bolt::kernels::filter;
using bolt::branchless::CmpOp;

namespace {

// Build an Arrow-shape validity bitmap (1 = valid) from a per-row bool list.
std::vector<uint64_t> mk_bitmap(const std::vector<bool>& valid) {
    std::vector<uint64_t> bm((valid.size() + 63) / 64, 0ull);
    for (std::size_t i = 0; i < valid.size(); ++i) {
        if (valid[i]) bm[i >> 6] |= (1ull << (i & 63u));
    }
    return bm;
}

}  // namespace

// ---------------------------------------------------------------------------
// cmp<Op,T> col-vs-col, dense, with validity bitmap (3-valued logic).
// ---------------------------------------------------------------------------

TEST(BoltFilterNull, CmpColColDenseWithNulls) {
    //            row: 0   1   2   3   4   5
    const int64_t lhs[] = {5, 10, 10, 20, 30, 10};
    const int64_t rhs[] = {5,  9, 10, 25, 30, 10};
    // Mark rows 2 and 4 as NULL — even though lhs==rhs there, they must fail.
    const auto bm = mk_bitmap({true, true, false, true, false, true});

    uint32_t out[6] = {};
    // Eq: rows where lhs==rhs AND valid → row0(5==5,valid), row5(10==10,valid).
    // row2 (10==10) is NULL → excluded; row4 (30==30) is NULL → excluded.
    std::size_t k = cmp<CmpOp::Eq, int64_t>(
        out, /*in_sel=*/nullptr, /*in_sel_len=*/0, /*in_len=*/6,
        lhs, rhs, bm.data());
    ASSERT_EQ(k, 2u);
    EXPECT_EQ(out[0], 0u);
    EXPECT_EQ(out[1], 5u);

    // Gt: lhs>rhs AND valid → row1(10>9), row3(20>25 false). Only row1.
    uint32_t out2[6] = {};
    std::size_t k2 = cmp<CmpOp::Gt, int64_t>(
        out2, nullptr, 0, 6, lhs, rhs, bm.data());
    ASSERT_EQ(k2, 1u);
    EXPECT_EQ(out2[0], 1u);
}

// ---------------------------------------------------------------------------
// All-valid bitmap path must equal the nullptr (no-validity) fast path.
// ---------------------------------------------------------------------------

TEST(BoltFilterNull, AllValidEqualsNoValidity) {
    const int32_t lhs[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int32_t rhs[] = {1, 9, 3, 0, 5, 9, 7, 0};
    const auto bm = mk_bitmap(std::vector<bool>(8, true));  // all valid

    uint32_t a[8] = {}, b[8] = {};
    std::size_t ka = cmp<CmpOp::Eq, int32_t>(a, nullptr, 0, 8, lhs, rhs, bm.data());
    std::size_t kb = cmp<CmpOp::Eq, int32_t>(b, nullptr, 0, 8, lhs, rhs, nullptr);
    ASSERT_EQ(ka, kb);
    for (std::size_t i = 0; i < ka; ++i) EXPECT_EQ(a[i], b[i]) << "i=" << i;
    // Sanity: matches are rows 0,2,4,6.
    ASSERT_EQ(ka, 4u);
    EXPECT_EQ(a[0], 0u); EXPECT_EQ(a[1], 2u);
    EXPECT_EQ(a[2], 4u); EXPECT_EQ(a[3], 6u);
}

// ---------------------------------------------------------------------------
// Sparse (in_sel) path agrees with the dense path restricted to those rows.
// ---------------------------------------------------------------------------

TEST(BoltFilterNull, SparseSelPathWithNulls) {
    const int64_t lhs[] = {7, 7, 7, 7, 7, 7, 7, 7};
    const int64_t rhs[] = {7, 0, 7, 0, 7, 7, 7, 0};
    // rows 2 and 6 NULL.
    const auto bm = mk_bitmap({true, true, false, true, true, true, false, true});
    // Consider only odd rows via the input selection vector.
    const uint32_t in_sel[] = {1, 3, 5, 7};

    uint32_t out[4] = {};
    // Eq over {1,3,5,7}: lhs==rhs only at row5 (7==7); rows 1,3,7 have rhs=0;
    // none of these is NULL, so the pass set is exactly {5}.
    std::size_t k = cmp<CmpOp::Eq, int64_t>(
        out, in_sel, /*in_sel_len=*/4, /*in_len=*/0, lhs, rhs, bm.data());
    ASSERT_EQ(k, 1u);
    EXPECT_EQ(out[0], 5u);
}

// ---------------------------------------------------------------------------
// cmp_scalar — column vs broadcast literal, dense, with nulls.
// ---------------------------------------------------------------------------

TEST(BoltFilterNull, CmpScalarWithNulls) {
    const int32_t data[] = {10, 20, 30, 40, 50, 60};
    const auto bm = mk_bitmap({true, false, true, true, false, true});

    uint32_t out[6] = {};
    // Ge 30 AND valid → rows 2(30),3(40),5(60). row4(50) is NULL → excluded.
    std::size_t k = cmp_scalar<CmpOp::Ge, int32_t>(
        out, nullptr, 0, 6, data, /*rhs=*/30, bm.data());
    ASSERT_EQ(k, 3u);
    EXPECT_EQ(out[0], 2u);
    EXPECT_EQ(out[1], 3u);
    EXPECT_EQ(out[2], 5u);

    // No-bitmap fast path: same predicate, all rows valid.
    uint32_t out2[6] = {};
    std::size_t k2 = cmp_scalar<CmpOp::Ge, int32_t>(
        out2, nullptr, 0, 6, data, 30, nullptr);
    ASSERT_EQ(k2, 4u);  // rows 2,3,4,5
    EXPECT_EQ(out2[0], 2u); EXPECT_EQ(out2[1], 3u);
    EXPECT_EQ(out2[2], 4u); EXPECT_EQ(out2[3], 5u);
}

// ---------------------------------------------------------------------------
// cmp_is_null / cmp_is_not_null — pure bitmap kernels.
// ---------------------------------------------------------------------------

TEST(BoltFilterNull, IsNullAndIsNotNullDense) {
    const auto bm = mk_bitmap({true, false, false, true, true, false});

    uint32_t out[6] = {};
    std::size_t kn = cmp_is_null(out, nullptr, 0, 6, bm.data());
    ASSERT_EQ(kn, 3u);
    EXPECT_EQ(out[0], 1u);
    EXPECT_EQ(out[1], 2u);
    EXPECT_EQ(out[2], 5u);

    uint32_t out2[6] = {};
    std::size_t knn = cmp_is_not_null(out2, nullptr, 0, 6, bm.data());
    ASSERT_EQ(knn, 3u);
    EXPECT_EQ(out2[0], 0u);
    EXPECT_EQ(out2[1], 3u);
    EXPECT_EQ(out2[2], 4u);
}

TEST(BoltFilterNull, IsNullNoBitmap) {
    uint32_t out[4] = {};
    // No bitmap ⇒ IS NULL matches nothing.
    EXPECT_EQ(cmp_is_null(out, nullptr, 0, 4, nullptr), 0u);

    // No bitmap ⇒ IS NOT NULL matches all rows (identity selvec).
    uint32_t out2[4] = {};
    std::size_t k = cmp_is_not_null(out2, nullptr, 0, 4, nullptr);
    ASSERT_EQ(k, 4u);
    for (uint32_t i = 0; i < 4; ++i) EXPECT_EQ(out2[i], i);
}

TEST(BoltFilterNull, IsNullSparse) {
    const auto bm = mk_bitmap({true, false, true, false, true, false, true, false});
    const uint32_t in_sel[] = {1, 2, 5, 6};  // mix of null (1,5) and valid (2,6)

    uint32_t out[4] = {};
    std::size_t kn = cmp_is_null(out, in_sel, 4, 0, bm.data());
    ASSERT_EQ(kn, 2u);
    EXPECT_EQ(out[0], 1u);
    EXPECT_EQ(out[1], 5u);

    uint32_t out2[4] = {};
    std::size_t knn = cmp_is_not_null(out2, in_sel, 4, 0, bm.data());
    ASSERT_EQ(knn, 2u);
    EXPECT_EQ(out2[0], 2u);
    EXPECT_EQ(out2[1], 6u);
}

// ---------------------------------------------------------------------------
// i32-selvec convenience overloads route to the same kernel.
// ---------------------------------------------------------------------------

TEST(BoltFilterNull, I32OverloadsMatch) {
    const int32_t data[] = {1, 2, 3, 4, 5};
    const auto bm = mk_bitmap({true, true, false, true, true});

    int32_t out[5] = {};
    std::size_t k = cmp_scalar_i32<CmpOp::Gt, int32_t>(
        out, nullptr, 0, 5, data, /*rhs=*/2, bm.data());
    // >2 AND valid → rows 3(4),4(5). row2(3) is NULL → excluded.
    ASSERT_EQ(k, 2u);
    EXPECT_EQ(out[0], 3);
    EXPECT_EQ(out[1], 4);

    int32_t outn[5] = {};
    EXPECT_EQ(cmp_is_null_i32(outn, nullptr, 0, 5, bm.data()), 1u);
    EXPECT_EQ(outn[0], 2);

    int32_t outnn[5] = {};
    EXPECT_EQ(cmp_is_not_null_i32(outnn, nullptr, 0, 5, bm.data()), 4u);
}
