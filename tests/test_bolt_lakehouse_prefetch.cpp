// Prefetcher — filesystem-backed ObjectStore, submit N requests, poll N
// results, verify slot_id round-trip and byte fidelity.

#include "bolt/lakehouse/object_store.h"
#include "bolt/lakehouse/object_store/prefetch.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

namespace {

using namespace bolt::lakehouse;

std::string temp_root() {
    auto p = std::filesystem::temp_directory_path() /
             ("bolt_lake_prefetch_" +
              std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
              "_" + std::to_string(static_cast<unsigned long long>(
                                    reinterpret_cast<uintptr_t>(&temp_root))));
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    std::filesystem::create_directories(p, ec);
    return p.generic_string();
}

void write_file(const std::string& root, const char* key,
                const uint8_t* data, size_t n) {
    auto path = std::filesystem::path(root) / key;
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::FILE* f = std::fopen(path.generic_string().c_str(), "wb");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(std::fwrite(data, 1, n, f), n);
    std::fclose(f);
}

TEST(BoltLakehousePrefetch, RoundTripFilesystem) {
    const std::string root = temp_root();
    FilesystemObjectStore fs{};
    ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.c_str(), &os));

    constexpr uint32_t kN = 8u;
    uint8_t bodies[kN][64];
    for (uint32_t i = 0; i < kN; ++i) {
        for (uint32_t j = 0; j < sizeof(bodies[i]); ++j) {
            bodies[i][j] = static_cast<uint8_t>((i * 31u + j) & 0xFFu);
        }
        char key[32];
        std::snprintf(key, sizeof(key), "blob_%u.bin", i);
        write_file(root, key, bodies[i], sizeof(bodies[i]));
    }

    bolt::Arena arena;
    prefetch::Prefetcher* p = nullptr;
    ASSERT_TRUE(prefetch::prefetcher_open(&p, &arena, &os,
                                          prefetch::kPrefetchDefaultDepth));
    ASSERT_NE(p, nullptr);

    // Submit. Some may bounce on a full ring — retry until accepted.
    for (uint32_t i = 0; i < kN; ++i) {
        prefetch::PrefetchRequest req{};
        std::snprintf(req.uri, sizeof(req.uri), "blob_%u.bin", i);
        req.range_lo = -1;
        req.range_hi = -1;
        req.slot_id  = 1000u + i;
        bool ok = false;
        for (int attempt = 0; attempt < 1000 && !ok; ++attempt) {
            ok = prefetch::prefetcher_submit(p, &req);
            if (!ok) {
                // Drain a result to free a slot.
                prefetch::PrefetchResult tmp{};
                if (prefetch::prefetcher_poll(p, &tmp, 5)) {
                    EXPECT_TRUE(tmp.ok);
                    // Verify before counting it consumed.
                    const uint32_t idx = tmp.slot_id - 1000u;
                    ASSERT_LT(idx, kN);
                    ASSERT_EQ(tmp.len, sizeof(bodies[idx]));
                    EXPECT_EQ(std::memcmp(tmp.buf, bodies[idx], tmp.len), 0);
                    bodies[idx][0] ^= 0x80u;   // sentinel: "this one's been seen"
                }
            }
        }
        ASSERT_TRUE(ok);
    }

    // Drain remaining results.
    uint32_t seen = 0;
    for (uint32_t i = 0; i < kN; ++i) {
        if (bodies[i][0] & 0x80u) {   // sentinel set above → already seen
            bodies[i][0] ^= 0x80u;
            ++seen;
        }
    }
    while (seen < kN) {
        prefetch::PrefetchResult res{};
        ASSERT_TRUE(prefetch::prefetcher_poll(p, &res, 5000));
        EXPECT_TRUE(res.ok);
        const uint32_t idx = res.slot_id - 1000u;
        ASSERT_LT(idx, kN);
        ASSERT_EQ(res.len, sizeof(bodies[idx]));
        EXPECT_EQ(std::memcmp(res.buf, bodies[idx], res.len), 0);
        ++seen;
    }
    EXPECT_EQ(seen, kN);

    prefetch::prefetcher_close(p);
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST(BoltLakehousePrefetch, OversizeDepthRejected) {
    bolt::Arena arena;
    FilesystemObjectStore fs{};
    ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, ".", &os));
    prefetch::Prefetcher* p = nullptr;
    EXPECT_FALSE(prefetch::prefetcher_open(
        &p, &arena, &os, prefetch::kPrefetchMaxDepth + 1u));
}

}  // namespace
