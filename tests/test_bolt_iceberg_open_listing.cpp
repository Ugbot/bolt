// G2ICE-245: finding the current metadata file lists metadata/ once in constant memory (list_visit), never the doubling
// relists of os_list_all; with a version hint only `metadata/v` keys are listed, and the hint stays advisory.
#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/iceberg/writer.h"

#include <gtest/gtest.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>

namespace {
using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

// Counts how the reader lists; with_visit = false hides list_visit so the os_list_all fallback is exercised.
struct CountingStore {
    FilesystemObjectStore fs{};
    ObjectStore under{};
    ObjectStore store{};
    ObjectStoreVT vt{};
    uint32_t lists = 0, visits = 0, keys = 0;
    std::string last_prefix;
    ObjectVisitFn inner_fn = nullptr;
    void* inner_ctx = nullptr;

    static CountingStore& self(void* p) { return *static_cast<CountingStore*>(p); }
    static int get(void* p, const char* k, bolt::Arena* a, const uint8_t** d, uint64_t* n) noexcept {
        return os_get(&self(p).under, k, a, d, n);
    }
    static int put(void* p, const char* k, const uint8_t* d, uint64_t n) noexcept { return os_put(&self(p).under, k, d, n); }
    static int list(void* p, const char* k, ObjectEntry* e, uint32_t c, uint32_t* n) noexcept {
        auto& s = self(p);
        ++s.lists;
        s.last_prefix = k;
        return os_list(&s.under, k, e, c, n);
    }
    static int del(void* p, const char* k) noexcept { return os_delete(&self(p).under, k); }
    static int head(void* p, const char* k, ObjectMeta* m) noexcept { return os_head(&self(p).under, k, m); }
    static int count_key(void* ctx, const char* k, uint64_t size) noexcept {
        auto& s = self(ctx);
        ++s.keys;
        return s.inner_fn(s.inner_ctx, k, size);
    }
    static int visit(void* p, const char* k, ObjectVisitFn fn, void* ctx) noexcept {
        auto& s = self(p);
        ++s.visits;
        s.last_prefix = k;
        s.inner_fn = fn;
        s.inner_ctx = ctx;
        return s.under.vt->list_visit(s.under.impl, k, &count_key, &s);
    }
    bool init(const char* root, bool with_visit) {
        if (!filesystem_object_store_init(&fs, root, &under)) return false;
        vt.get_object = get; vt.put_object = put; vt.list_prefix = list; vt.delete_object = del; vt.head_object = head;
        if (with_visit) vt.list_visit = visit;
        store = ObjectStore{&vt, this};
        return under.vt->list_visit != nullptr;
    }
};

class IcebergOpenListing : public ::testing::Test {
protected:
    std::string base;
    std::string meta_json;   // a valid metadata document; last-updated-ms is rewritten per file

    void SetUp() override {
        static std::atomic<uint32_t> next{0};
        base = (std::filesystem::temp_directory_path() /
                ("bolt_iceberg_open_listing_" + std::to_string(bolt_getpid()) + "_" + std::to_string(next.fetch_add(1))))
                   .generic_string();
        std::filesystem::create_directories(base + "/t/metadata");
        std::filesystem::create_directories(base + "/seed");
        bolt::Arena arena;
        FilesystemObjectStore fs{};
        ObjectStore os{};
        ASSERT_TRUE(filesystem_object_store_init(&fs, (base + "/seed").c_str(), &os));
        SchemaField f{};
        f.id = 1;
        f.required = true;
        std::strcpy(f.name, "id");
        std::strcpy(f.type, "long");
        Schema s{};
        s.fields = &f;
        s.cap_fields = 1;
        s.n_fields = 1;
        TableHandle* t = nullptr;
        ASSERT_TRUE(table_create(&t, &arena, &os, (base + "/seed").c_str(), &s, nullptr, nullptr, nullptr));
        const uint8_t* b = nullptr;
        uint64_t n = 0;
        ASSERT_EQ(os_get(&os, "metadata/v1.metadata.json", &arena, &b, &n), kOsOk);
        meta_json.assign(reinterpret_cast<const char*>(b), n);
        ASSERT_NE(meta_json.find("\"last-updated-ms\""), std::string::npos);
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(base, ec);
    }
    // Metadata file `name` whose last-updated-ms is `marker`, so the test sees which file the reader picked.
    void write_meta(const std::string& name, int64_t marker) {
        std::string m = meta_json;
        size_t at = m.find("\"last-updated-ms\"");
        at = m.find(':', at) + 1;
        while (m[at] == ' ') ++at;
        size_t end = at;
        while (end < m.size() && m[end] >= '0' && m[end] <= '9') ++end;
        m.replace(at, end - at, std::to_string(marker));
        write_file("t/metadata/" + name, m);
    }
    void write_file(const std::string& rel, const std::string& body) {
        std::FILE* f = std::fopen((base + "/" + rel).c_str(), "wb");
        ASSERT_NE(f, nullptr);
        ASSERT_EQ(std::fwrite(body.data(), 1, body.size(), f), body.size());
        std::fclose(f);
    }
    int64_t open_marker(CountingStore* cs, bool with_visit = true) {
        EXPECT_TRUE(cs->init(base.c_str(), with_visit));
        bolt::Arena arena;
        TableHandle* h = nullptr;
        if (!iceberg_table_open_store(&h, &arena, &cs->store, "t")) return -1;
        const int64_t v = iceberg_table_metadata(h)->last_updated_ms;
        iceberg_table_close(h);
        return v;
    }
};

TEST_F(IcebergOpenListing, CatalogManagedNamesManyObjectsListedOnceInConstantMemory) {
    write_meta("00001-aaaa.metadata.json", 1001);
    write_meta("00003-cccc.metadata.json", 1003);
    write_meta("00002-bbbb.metadata.json", 1002);
    for (int i = 0; i < 20000; ++i) write_file("t/metadata/" + std::to_string(i) + "-m0.avro", "x");
    CountingStore cs;
    EXPECT_EQ(open_marker(&cs), 1003);
    EXPECT_EQ(cs.visits, 1u);
    EXPECT_EQ(cs.lists, 0u) << "the reader must not materialize (and relist) the listing";
    EXPECT_EQ(cs.keys, 20003u);
    EXPECT_EQ(cs.last_prefix, "t/metadata/");
}

TEST_F(IcebergOpenListing, StaleHintIsAdvisoryAndOnlyHadoopNamesAreListed) {
    write_meta("v1.metadata.json", 1);
    write_meta("v2.metadata.json", 2);
    write_meta("v3.metadata.json", 3);
    for (int i = 0; i < 100; ++i) write_file("t/metadata/snap-" + std::to_string(i) + ".avro", "x");
    write_file("t/metadata/version-hint.text", "1");
    CountingStore cs;
    EXPECT_EQ(open_marker(&cs), 3);
    EXPECT_EQ(cs.last_prefix, "t/metadata/v");
    EXPECT_EQ(cs.keys, 4u);   // v1..v3 and version-hint.text, never the manifests
}

TEST_F(IcebergOpenListing, HintAtAMissingFileFindsTheRetainedLatest) {
    write_meta("v2.metadata.json", 2);
    write_meta("v5.metadata.json", 5);   // a retained version beyond a gap
    write_file("t/metadata/version-hint.text", "7");
    CountingStore cs;
    EXPECT_EQ(open_marker(&cs), 5);
}

TEST_F(IcebergOpenListing, CurrentHintStillPicksTheHintedVersion) {
    write_meta("v1.metadata.json", 1);
    write_meta("v2.metadata.json", 2);
    write_file("t/metadata/version-hint.text", "2\n");
    CountingStore cs;
    EXPECT_EQ(open_marker(&cs), 2);
}

TEST_F(IcebergOpenListing, GarbageHintFallsBackToTheWholeListing) {
    write_meta("00004-dddd.metadata.json", 4);
    write_meta("00002-bbbb.metadata.json", 2);
    write_file("t/metadata/version-hint.text", "not-a-version");
    CountingStore cs;
    EXPECT_EQ(open_marker(&cs), 4);
    EXPECT_EQ(cs.last_prefix, "t/metadata/");
}

TEST_F(IcebergOpenListing, GzipNamesRankWithPlainOnes) {
    write_meta("v1.metadata.json", 1);
    write_meta("v2.metadata.json", 2);
    write_meta("v3.gz.metadata.json", 3);   // plain bytes under a gzip name: the content decides how they are read
    write_file("t/metadata/version-hint.text", "2");
    CountingStore cs;
    EXPECT_EQ(open_marker(&cs), 3);
}

TEST_F(IcebergOpenListing, StoreWithoutListVisitAnswersTheSame) {
    write_meta("00001-aaaa.metadata.json", 1);
    write_meta("00002-bbbb.metadata.json", 2);
    for (int i = 0; i < 600; ++i) write_file("t/metadata/" + std::to_string(i) + "-m0.avro", "x");
    CountingStore cs;
    EXPECT_EQ(open_marker(&cs, false), 2);
    EXPECT_GE(cs.lists, 1u);
    EXPECT_EQ(cs.visits, 0u);
}

TEST_F(IcebergOpenListing, NoMetadataRefuses) {
    write_file("t/metadata/1-m0.avro", "x");
    CountingStore cs;
    EXPECT_EQ(open_marker(&cs), -1);
}

int stop_at_two(void* ctx, const char*, uint64_t) noexcept {
    uint32_t* n = static_cast<uint32_t*>(ctx);
    return ++*n == 2u ? kOsIoError : kOsOk;
}
int count_all(void* ctx, const char*, uint64_t) noexcept {
    ++*static_cast<uint32_t*>(ctx);
    return kOsOk;
}

TEST(IcebergOpenListingVisit, VisitorErrorStopsTheWalk) {
    const std::string dir =
        (std::filesystem::temp_directory_path() / ("bolt_visit_" + std::to_string(bolt_getpid()))).generic_string();
    std::filesystem::create_directories(dir + "/p");
    for (int i = 0; i < 5; ++i) {
        std::FILE* f = std::fopen((dir + "/p/" + std::to_string(i)).c_str(), "wb");
        ASSERT_NE(f, nullptr);
        std::fclose(f);
    }
    FilesystemObjectStore fs{};
    ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, dir.c_str(), &os));
    bolt::Arena arena;
    uint32_t seen = 0;
    EXPECT_EQ(os_list_visit(&os, "p/", &arena, &stop_at_two, &seen), kOsIoError);
    EXPECT_EQ(seen, 2u);
    seen = 0;
    EXPECT_EQ(os_list_visit(&os, "p/", &arena, &count_all, &seen), kOsOk);
    EXPECT_EQ(seen, 5u);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

}  // namespace
