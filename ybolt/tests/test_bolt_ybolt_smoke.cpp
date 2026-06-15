// bolt::ybolt smoke gate — confirm the BoltArenaAllocator binding plugs
// into ycpp's templated primitives end-to-end.
//
// The bigger end-to-end (Doc apply / encode-diff round trip) lands once
// ycpp W3 is in the static lib + interop-tested. For now we prove:
//
//   1. BoltArenaAllocator satisfies ycpp::Allocator (compile-time).
//   2. bolt::Arena allocations route through the adapter and tally on
//      arena->total_allocated() / adapter.bytes_in_use().
//   3. ycpp::Pool<Item, BoltArenaAllocator>     acquires + releases.
//   4. ycpp::DeleteSet<BoltArenaAllocator>      add/contains/encode/decode.
//   5. ycpp::StructStore<BoltArenaAllocator>    append/find_by_id.
//   6. ycpp::HashMap<K, V, BoltArenaAllocator>  insert/find.
//   7. ByteView adapters round-trip a C string into ycpp::ByteView.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/ybolt.h"

#include "ycpp/ycpp_byteview.h"
#include "ycpp/ycpp_delete_set.h"
#include "ycpp/ycpp_hashmap.h"
#include "ycpp/ycpp_id.h"
#include "ycpp/ycpp_item.h"
#include "ycpp/ycpp_pool.h"
#include "ycpp/ycpp_reader.h"
#include "ycpp/ycpp_status.h"
#include "ycpp/ycpp_struct_store.h"
#include "ycpp/ycpp_writer.h"

using bolt::ybolt::BoltArenaAllocator;
using bolt::ybolt::DeleteSet;
using bolt::ybolt::StructStore;

TEST(YboltSmoke, AllocatorConceptIsSatisfied) {
    static_assert(ycpp::Allocator<BoltArenaAllocator>,
                  "BoltArenaAllocator must satisfy ycpp::Allocator");
    SUCCEED();
}

TEST(YboltSmoke, ArenaAllocationsRouteThroughAdapter) {
    bolt::Arena               arena;
    BoltArenaAllocator        a{&arena};
    void* p1 = a.alloc(128, 16);
    void* p2 = a.alloc( 64,  8);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p1) % 16U, 0U);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p2) %  8U, 0U);
    EXPECT_GE(a.bytes_in_use(), 128U + 64U);
    EXPECT_EQ(a.bytes_in_use(), arena.total_allocated());
}

TEST(YboltSmoke, PoolUsingBoltArena) {
    bolt::Arena            arena;
    BoltArenaAllocator     a{&arena};
    ycpp::Pool<ycpp::Item, BoltArenaAllocator> pool{&a, 8};
    ycpp::Item* it = pool.acquire();
    ASSERT_NE(it, nullptr);
    EXPECT_EQ(pool.live(), 1U);
    pool.release(it);
    EXPECT_EQ(pool.live(), 0U);
}

TEST(YboltSmoke, DeleteSetUsingBoltArena) {
    bolt::Arena    arena;
    BoltArenaAllocator a{&arena};
    DeleteSet      ds{&a};
    ASSERT_EQ(ds.add(1, 0, 5),  ycpp::Status::kOk);
    ASSERT_EQ(ds.add(1, 5, 3),  ycpp::Status::kOk);   // touches → merge
    EXPECT_TRUE (ds.contains(ycpp::Id{1, 7}));
    EXPECT_FALSE(ds.contains(ycpp::Id{1, 8}));

    std::array<uint8_t, 64> buf{};
    ycpp::Writer w{buf.data(), buf.size()};
    ASSERT_EQ(ds.encode(w), ycpp::Status::kOk);

    bolt::Arena    arena2;
    BoltArenaAllocator a2{&arena2};
    DeleteSet      ds2{&a2};
    ycpp::Reader   r{ycpp::ByteView{buf.data(), w.pos()}};
    ASSERT_EQ(ds2.decode(r), ycpp::Status::kOk);
    EXPECT_TRUE (ds2.contains(ycpp::Id{1, 0}));
    EXPECT_TRUE (ds2.contains(ycpp::Id{1, 7}));
    EXPECT_FALSE(ds2.contains(ycpp::Id{1, 8}));
}

TEST(YboltSmoke, StructStoreUsingBoltArena) {
    bolt::Arena    arena;
    BoltArenaAllocator a{&arena};
    ycpp::Pool<ycpp::Item, BoltArenaAllocator> pool{&a, 8};
    StructStore    store{&a};

    auto fresh = [&](uint64_t client, uint64_t clock) {
        ycpp::Item* it = pool.acquire();
        it->id     = ycpp::Id{client, clock};
        it->length = 1;
        return it;
    };

    ASSERT_EQ(store.append(fresh(11, 0)), ycpp::Status::kOk);
    ASSERT_EQ(store.append(fresh(11, 1)), ycpp::Status::kOk);
    ASSERT_EQ(store.append(fresh(11, 2)), ycpp::Status::kOk);
    EXPECT_EQ(store.state(11), 3U);
    EXPECT_NE(store.find_by_id(ycpp::Id{11, 1}), nullptr);
    EXPECT_EQ(store.find_by_id(ycpp::Id{11, 9}), nullptr);
}

TEST(YboltSmoke, HashMapUsingBoltArena) {
    bolt::Arena    arena;
    BoltArenaAllocator a{&arena};
    ycpp::HashMap<uint64_t, uint64_t, BoltArenaAllocator> m{&a};
    for (uint64_t i = 0; i < 256; ++i) (void)m.insert(i, i * 13ULL);
    EXPECT_EQ(m.size(), 256U);
    auto* hit = m.find(42ULL);
    ASSERT_NE(hit, nullptr);
    EXPECT_EQ(hit->value, 42ULL * 13ULL);
}

TEST(YboltSmoke, FromCstrRoundTrip) {
    const ycpp::ByteView v = bolt::ybolt::from_cstr("ycpp");
    ASSERT_EQ(v.size, 4U);
    EXPECT_EQ(std::memcmp(v.data, "ycpp", 4), 0);

    const ycpp::ByteView empty = bolt::ybolt::from_cstr(nullptr);
    EXPECT_EQ(empty.size, 0U);
    EXPECT_EQ(empty.data, nullptr);
}
