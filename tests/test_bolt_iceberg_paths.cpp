// test_bolt_iceberg_paths.cpp — G2ICE-300: metadata naming, identifier-field-ids and exact path rules, on a copy of the
// pyiceberg-written golden table (9 rows, two snapshots, catalog-managed metadata names).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <map>
#include <fstream>
#include <memory>
#include <unistd.h>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_deflate.h"
#include "bolt/ingest/bolt_inflate.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/format.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"

namespace {

namespace fs = std::filesystem;
using namespace bolt;
using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

const char* kLatest = "00002-b00d0cb3-a57c-4208-9373-ab87144fd133";

std::string read_all(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void write_all(const fs::path& p, const std::string& s) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(s.data(), static_cast<std::streamsize>(s.size()));
}

// A fresh copy of the golden table; returns the warehouse root.
fs::path copy_table(const char* tag) {
    const fs::path dst = fs::temp_directory_path() / (std::string("bolt_g2ice300_") + tag + "_" + std::to_string(::getpid()));
    fs::remove_all(dst);
    fs::copy(fs::path(BOLT_TEST_DATA_DIR) / "golden_iceberg_table", dst, fs::copy_options::recursive);
    return dst;
}

fs::path meta_dir(const fs::path& root) { return root / "db" / "trades" / "metadata"; }

std::string gzip(const std::string& s) {
    auto st = std::make_unique<ingest::DeflateState>();
    std::vector<uint8_t> out(ingest::gzip_bound(s.size()));
    uint64_t n = 0;
    EXPECT_TRUE(ingest::gzip_compress(reinterpret_cast<const uint8_t*>(s.data()), s.size(), out.data(), out.size(), &n, st.get()));
    return std::string(reinterpret_cast<const char*>(out.data()), n);
}

// Rows of a full scan of db.trades under `root`, or -1 when the table does not open / scan.
int64_t scan_rows(const fs::path& root) {
    Arena arena;
    FilesystemCatalog fsc{};
    Catalog cat{};
    TableHandle* th = nullptr;
    ScanHandle* sh = nullptr;
    if (!filesystem_catalog_init(&fsc, root.c_str(), &cat) || !iceberg_table_open(&th, &arena, &cat, "db", "trades")) return -1;
    int64_t rows = -1;
    if (iceberg_scan_open(&sh, th, nullptr)) {
        rows = 0;
        for (uint32_t g = 0; g < 1024u; ++g) {
            BoltBatch b{};
            bool eof = false;
            if (!iceberg_scan_next_batch(sh, &b, &eof)) { rows = -1; break; }
            if (eof) break;
            rows += b.num_rows;
        }
        iceberg_scan_close(sh);
    }
    iceberg_table_close(th);
    return rows;
}

// Replace `from` with `to` in the latest metadata file (every occurrence).
void patch_latest(const fs::path& root, const std::string& from, const std::string& to) {
    const fs::path p = meta_dir(root) / (std::string(kLatest) + ".metadata.json");
    std::string s = read_all(p);
    for (size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size())) s.replace(at, from.size(), to);
    write_all(p, s);
}

}  // namespace

TEST(IcebergPaths, GzipMetadataUnderBothJavaNamings) {
    for (const char* suffix : {".gz.metadata.json", ".metadata.json.gz"}) {
        const fs::path root = copy_table("gz");
        const fs::path plain = meta_dir(root) / (std::string(kLatest) + ".metadata.json");
        write_all(meta_dir(root) / (std::string(kLatest) + suffix), gzip(read_all(plain)));
        fs::remove(plain);
        EXPECT_EQ(scan_rows(root), 9) << suffix;
        fs::remove_all(root);
    }
}

TEST(IcebergPaths, CorruptGzipMetadataIsRefusedNotMisread) {
    const fs::path root = copy_table("gzbad");
    const fs::path plain = meta_dir(root) / (std::string(kLatest) + ".metadata.json");
    std::string z = gzip(read_all(plain));
    z[z.size() - 6] = static_cast<char>(z[z.size() - 6] ^ 0x5A);     // the CRC32 trailer
    write_all(meta_dir(root) / (std::string(kLatest) + ".gz.metadata.json"), z);
    fs::remove(plain);
    EXPECT_EQ(scan_rows(root), -1) << "a gzip member whose CRC does not match is not metadata";
    fs::remove_all(root);
}

TEST(IcebergPaths, GunzipChecksTrailerAndBound) {
    const std::string raw(5000, 'q');
    const std::string z = gzip(raw);
    Arena a;
    auto alloc = [](void* c, uint64_t n) { return static_cast<uint8_t*>(static_cast<Arena*>(c)->allocate(n, 8)); };
    const uint8_t* out = nullptr;
    uint64_t n = 0;
    ASSERT_EQ(ingest::gunzip(reinterpret_cast<const uint8_t*>(z.data()), z.size(), 1u << 20, alloc, &a, &out, &n), ingest::kInflateOk);
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(out), n), raw);
    EXPECT_EQ(ingest::gunzip(reinterpret_cast<const uint8_t*>(z.data()), z.size(), 100u, alloc, &a, &out, &n), ingest::kInflateNeedMore);
    std::string bad = z;
    bad[bad.size() - 1] = static_cast<char>(bad[bad.size() - 1] ^ 1);  // ISIZE
    EXPECT_NE(ingest::gunzip(reinterpret_cast<const uint8_t*>(bad.data()), bad.size(), 1u << 20, alloc, &a, &out, &n), ingest::kInflateOk);
}

TEST(IcebergPaths, IdentifierFieldIdsArePreservedAndHonoured) {
    const fs::path root = copy_table("ident");
    patch_latest(root, "\"identifier-field-ids\":[]", "\"identifier-field-ids\":[1]");
    {
        Arena arena;
        FilesystemCatalog fsc{};
        Catalog cat{};
        TableHandle* th = nullptr;
        ASSERT_TRUE(filesystem_catalog_init(&fsc, root.c_str(), &cat));
        ASSERT_TRUE(iceberg_table_open(&th, &arena, &cat, "db", "trades"));
        const Schema* s = metadata_current_schema(iceberg_table_metadata(th));
        ASSERT_NE(s, nullptr);
        ASSERT_EQ(s->n_identifier, 1u);
        EXPECT_EQ(s->identifier_ids[0], 1);
        iceberg_table_close(th);
    }
    fs::remove_all(root);
}

TEST(IcebergPaths, WriterKeepsIdentifierFieldIdsAcrossEvolution) {
    const fs::path root = fs::temp_directory_path() / ("bolt_g2ice300_w_" + std::to_string(::getpid()));
    fs::remove_all(root);
    fs::create_directories(root / "db" / "t");
    Arena wa;
    FilesystemObjectStore fos{};
    ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fos, root.c_str(), &os));
    SchemaField fields[2]{};
    fields[0].id = 1; fields[0].required = true;  std::strcpy(fields[0].name, "id");  std::strcpy(fields[0].type, "long");
    fields[1].id = 2; fields[1].required = false; std::strcpy(fields[1].name, "v");   std::strcpy(fields[1].type, "double");
    int32_t ident[1] = {1};
    Schema sch{};
    sch.fields = fields; sch.cap_fields = 2; sch.n_fields = 2; sch.n_identifier = 1; sch.identifier_ids = ident;
    TableHandle* wh = nullptr;
    ASSERT_TRUE(table_create(&wh, &wa, &os, "db/t", &sch, nullptr, nullptr, nullptr));
    EXPECT_FALSE(table_drop_column(wh, "id")) << "Iceberg-Java refuses dropping an identifier field";
    ASSERT_TRUE(table_add_column(wh, "note", BoltType::Utf8, false));
    const Schema* cur = metadata_current_schema(table_metadata(wh));
    ASSERT_NE(cur, nullptr);
    ASSERT_EQ(cur->n_identifier, 1u);
    EXPECT_EQ(cur->identifier_ids[0], 1);
    table_close(wh);
    const fs::path mdir = root / "metadata";
    const fs::path latest = [&] {
        fs::path best;
        for (const auto& e : fs::directory_iterator(mdir))
            if (e.path().filename().string().find(".metadata.json") != std::string::npos &&
                (best.empty() || e.path().filename().string() > best.filename().string())) best = e.path();
        return best;
    }();
    const std::string j = read_all(latest);
    size_t hits = 0;
    for (size_t at = j.find("\"identifier-field-ids\":[1]"); at != std::string::npos; at = j.find("\"identifier-field-ids\":[1]", at + 1)) ++hits;
    EXPECT_EQ(hits, 2u) << "both schemas carry the identifier fields: " << latest;
    fs::remove_all(root);
}

TEST(IcebergPaths, LocationMatchIsSchemeInsensitive) {
    const fs::path root = copy_table("scheme");
    patch_latest(root, "\"location\":\"file://C:/", "\"location\":\"s3a://C:/");
    EXPECT_EQ(scan_rows(root), 9) << "s3a://<loc> covers file://<loc>/... paths";
    fs::remove_all(root);
}

TEST(IcebergPaths, UriOutsideTheLocationIsRefused) {
    const fs::path root = copy_table("outside");
    patch_latest(root, "\"location\":\"file://C:/Users/Ben/AppData/Local/Temp/bolt_iceberg_gen_0ynj69rb/db/trades\"",
                 "\"location\":\"s3://elsewhere/db/trades\"");
    EXPECT_EQ(scan_rows(root), -1) << "paths that name another location are never guessed at";
    fs::remove_all(root);
}

TEST(IcebergPaths, ReservedMetadataColumnsFollowTheTableColumns) {
    Arena arena;
    FilesystemCatalog fsc{};
    Catalog cat{};
    TableHandle* th = nullptr;
    ScanHandle* sh = nullptr;
    ASSERT_TRUE(filesystem_catalog_init(&fsc, (std::string(BOLT_TEST_DATA_DIR) + "/golden_iceberg_table").c_str(), &cat));
    ASSERT_TRUE(iceberg_table_open(&th, &arena, &cat, "db", "trades"));
    ReadOptions ro;
    read_options_init(&ro);
    ro.emit_metadata_columns = true;
    ASSERT_TRUE(iceberg_scan_open(&sh, th, &ro));
    std::map<std::string, std::vector<int64_t>> pos_by_file;
    int64_t rows = 0;
    for (uint32_t g = 0; g < 1024u; ++g) {
        BoltBatch b{};
        bool eof = false;
        ASSERT_TRUE(iceberg_scan_next_batch(sh, &b, &eof));
        if (eof) break;
        ASSERT_EQ(b.num_cols, 4u + 5u);
        const int fi = b.schema.find_field("_file"), pi = b.schema.find_field("_pos"), si = b.schema.find_field("_spec_id"),
                  di = b.schema.find_field("_deleted");
        ASSERT_EQ(fi, 4); ASSERT_EQ(pi, 5); ASSERT_EQ(si, 6); ASSERT_EQ(di, 7);
        const BoltColumn* c = b.columns[b.read_epoch];
        for (int64_t r = 0; r < b.num_rows; ++r) {
            const uint8_t* p = nullptr;
            int32_t n = 0;
            c[fi].utf8_at(r, &p, &n);
            const std::string file(reinterpret_cast<const char*>(p), static_cast<size_t>(n));
            EXPECT_EQ(file.rfind("file://", 0), 0u) << "the path as the manifest records it: " << file;
            pos_by_file[file].push_back(c[pi].typed_data<int64_t>()[r]);
            EXPECT_EQ(c[si].typed_data<int64_t>()[r], 0);
            EXPECT_EQ(c[di].typed_data<int64_t>()[r], 0);
            const int qi = b.schema.find_field("_partition");
            ASSERT_EQ(qi, 8);
            c[qi].utf8_at(r, &p, &n);
            const std::string part(reinterpret_cast<const char*>(p), static_cast<size_t>(n));
            EXPECT_EQ(part.rfind("{\"sym\":\"", 0), 0u) << part;
            EXPECT_EQ(part.back(), '}') << part;
        }
        rows += b.num_rows;
    }
    iceberg_scan_close(sh);
    iceberg_table_close(th);
    EXPECT_EQ(rows, 9);
    EXPECT_EQ(pos_by_file.size(), 5u) << "five data files";
    for (auto& [f, ps] : pos_by_file) {
        std::sort(ps.begin(), ps.end());
        for (size_t i = 0; i < ps.size(); ++i) EXPECT_EQ(ps[i], static_cast<int64_t>(i)) << f;
    }
}

TEST(IcebergPaths, RelativeOrOmittedLocationsAreNotAbsolute) {
    for (const char* loc : {"s3://b/t", "s3a://b/t", "file:/tmp/t", "file:///C:/t", "/tmp/t", "gs://b/t", "abfss://c@a/t"})
        EXPECT_TRUE(table_location_absolute(loc)) << loc;
    for (const char* loc : {"", "t", "warehouse/db/t", "./t", "1s3://b/t", ":x"})
        EXPECT_FALSE(table_location_absolute(loc)) << loc;
    EXPECT_FALSE(table_location_absolute(nullptr));
}
