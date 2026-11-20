// test_bolt_vmem.cpp — reserved address space committed on demand (G2CHK-1140).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "bolt/io/bolt_vmem.h"

using namespace bolt::io;

TEST(Vmem, ReserveCommitWriteRelease) {
    const size_t page = vm_page_bytes();
    ASSERT_GE(page, 4096u);
    ASSERT_EQ(page & (page - 1u), 0u);
    const size_t big = size_t{1} << 30;                   // 1 GiB of address space
    auto* base = static_cast<uint8_t*>(vm_reserve(big));
    ASSERT_NE(base, nullptr);
    ASSERT_TRUE(vm_commit(base, 0, page));
    ASSERT_TRUE(vm_commit(base, big / 2, 4 * page));
    for (size_t i = 0; i < page; ++i) EXPECT_EQ(base[i], 0u) << "zero on first touch";
    std::memset(base, 0xAB, page);
    std::memset(base + big / 2, 0xCD, 4 * page);
    ASSERT_TRUE(vm_commit(base, 0, page));               // recommit keeps the bytes
    EXPECT_EQ(base[page - 1], 0xABu);
    EXPECT_EQ(base[big / 2 + 4 * page - 1], 0xCDu);
    vm_release(base, big);
}

TEST(Vmem, RefusesUnalignedCommitAndNull) {
    const size_t page = vm_page_bytes();
    void* base = vm_reserve(16 * page);
    ASSERT_NE(base, nullptr);
    EXPECT_FALSE(vm_commit(base, 1, page));
    EXPECT_FALSE(vm_commit(base, 0, page + 1));
    EXPECT_FALSE(vm_commit(nullptr, 0, page));
    EXPECT_TRUE(vm_commit(base, 0, 0));
    EXPECT_EQ(vm_reserve(0), nullptr);
    vm_release(base, 16 * page);
}
