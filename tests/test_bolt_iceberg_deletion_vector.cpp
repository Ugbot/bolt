// Iceberg v3 deletion-vector blobs: round trip, corruption refusals, and the multi-bucket (positions >= 2^32) layout.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_deflate.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/iceberg/deletion_vector.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

TEST(IcebergDeletionVector, RoundTripsPositionsAndRefusesCorruption) {
    bolt::Arena a;
    std::vector<uint32_t> pos = {0, 1, 7, 100, 65535, 65536, 70000, 1000000};
    const uint8_t* blob = nullptr; uint64_t len = 0;
    ASSERT_TRUE(iceberg_dv_encode(pos.data(), pos.size(), &a, &blob, &len));
    bolt::ingest::RoaringBitmap64 dv{};
    ASSERT_TRUE(iceberg_dv_decode(blob, len, &a, &dv));
    EXPECT_EQ(dv.cardinality, pos.size());
    for (uint32_t p : pos) EXPECT_TRUE(iceberg_dv_contains(&dv, p));
    for (uint32_t p : {2u, 6u, 99u, 65534u, 999999u}) EXPECT_FALSE(iceberg_dv_contains(&dv, p));

    std::vector<uint8_t> bad(blob, blob + len);
    bad[len / 2] ^= 0x01;  // payload bit flip: CRC must catch it
    bolt::ingest::RoaringBitmap64 x{};
    EXPECT_FALSE(iceberg_dv_decode(bad.data(), bad.size(), &a, &x));
    bad.assign(blob, blob + len); bad[4] = 0;  // wrong magic
    EXPECT_FALSE(iceberg_dv_decode(bad.data(), bad.size(), &a, &x));
    EXPECT_FALSE(iceberg_dv_decode(blob, len - 1, &a, &x));  // truncated
    bad.assign(blob, blob + len); bad[3] ^= 0x04;  // length field disagrees with the blob
    EXPECT_FALSE(iceberg_dv_decode(bad.data(), bad.size(), &a, &x));
}

TEST(IcebergDeletionVector, DecodesAMultiBucketBitmapFromAnotherWriter) {
    // Hand-built per the spec: two 32-bit buckets (high keys 0 and 1) so positions 5 and 2^32+9 are set.
    auto portable = [](uint16_t value) {  // one array container holding `value`
        std::vector<uint8_t> v = {0x3A, 0x30, 0, 0, 1, 0, 0, 0,  // cookie 12346, 1 container
                                  0, 0, 0, 0,                    // key 0, card-1 0
                                  0x10, 0, 0, 0,                 // offset header (16)
                                  static_cast<uint8_t>(value & 0xFF), static_cast<uint8_t>(value >> 8)};
        return v;
    };
    std::vector<uint8_t> bm;
    auto u64 = [&](uint64_t x) { for (int i = 0; i < 8; ++i) bm.push_back(static_cast<uint8_t>(x >> (8 * i))); };
    auto u32 = [&](uint32_t x) { for (int i = 0; i < 4; ++i) bm.push_back(static_cast<uint8_t>(x >> (8 * i))); };
    u64(2);
    u32(0); { auto p = portable(5); bm.insert(bm.end(), p.begin(), p.end()); }
    u32(1); { auto p = portable(9); bm.insert(bm.end(), p.begin(), p.end()); }
    std::vector<uint8_t> body = {0xD1, 0xD3, 0x39, 0x64};
    body.insert(body.end(), bm.begin(), bm.end());
    std::vector<uint8_t> blob(4);
    for (int i = 0; i < 4; ++i) blob[i] = static_cast<uint8_t>(body.size() >> (8 * (3 - i)));
    blob.insert(blob.end(), body.begin(), body.end());
    const uint32_t crc = bolt::ingest::crc32_ieee(body.data(), body.size(), 0u);
    for (int i = 0; i < 4; ++i) blob.push_back(static_cast<uint8_t>(crc >> (8 * (3 - i))));

    bolt::Arena a;
    bolt::ingest::RoaringBitmap64 dv{};
    ASSERT_TRUE(iceberg_dv_decode(blob.data(), blob.size(), &a, &dv));
    EXPECT_EQ(dv.cardinality, 2u);
    EXPECT_TRUE(iceberg_dv_contains(&dv, 5));
    EXPECT_TRUE(iceberg_dv_contains(&dv, (1ull << 32) + 9));
    EXPECT_FALSE(iceberg_dv_contains(&dv, 9));
    EXPECT_FALSE(iceberg_dv_contains(&dv, (1ull << 32) + 5));
}

namespace {

std::string slurp(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

void spit(const std::filesystem::path& p, const std::string& b) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(b.data(), static_cast<std::streamsize>(b.size()));
}

}  // namespace

// A table with a v3 deletion vector (Puffin blob located by offset/size in a delete manifest) reads without the DV's rows, and a
// blob that fails its CRC fails the scan rather than returning rows it should have deleted.
TEST(IcebergDeletionVector, ScanExcludesRowsNamedByADeletionVectorAndRefusesACorruptOne) {
    bolt::Arena arena;
    const std::filesystem::path base = std::filesystem::temp_directory_path() / "bolt_iceberg_dv_scan";
    std::error_code ec;
    std::filesystem::remove_all(base, ec);
    const std::filesystem::path root = base / "default" / "t";
    std::filesystem::create_directories(root, ec);

    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.generic_string().c_str(), &os));
    static SchemaField fields[1]{};
    Schema sch{}; sch.fields = fields; sch.cap_fields = 1; sch.n_fields = 1;
    fields[0].id = 1; fields[0].required = true;
    std::strncpy(fields[0].name, "id", sizeof(fields[0].name) - 1);
    std::strncpy(fields[0].type, "long", sizeof(fields[0].type) - 1);
    PartitionSpec spec{}; SortOrder sort{}; WriteOptions wo; write_options_init(&wo);
    TableHandle* wh = nullptr;
    ASSERT_TRUE(table_create(&wh, &arena, &os, root.generic_string().c_str(), &sch, &spec, &sort, &wo));
    AppendHandle* ah = nullptr;
    ASSERT_TRUE(append_open(&ah, wh));
    bolt::BoltBatch b{};
    bolt::BoltBatch::init_empty(&b);
    b.arena = &arena; b.num_rows = 10; b.num_cols = 1;
    bolt::BoltBatch::alloc_columns(&b, &arena, 1);
    b.schema.add_field("id", bolt::BoltType::Int64, false);
    auto* ids = arena.allocate_array<int64_t>(10);
    for (int i = 0; i < 10; ++i) ids[i] = 100 + i;
    b.columns[b.read_epoch][0].type = bolt::BoltType::Int64;
    b.columns[b.read_epoch][0].length = 10;
    b.columns[b.read_epoch][0].data = ids;
    ASSERT_TRUE(append_write(ah, &b));
    char data_file[kIcebergMaxPath]; int64_t rows = 0;
    ASSERT_TRUE(append_commit_ex(ah, data_file, kIcebergMaxPath, &rows));
    append_close(ah);
    table_close(wh);

    // Locate the current metadata and manifest list the writer produced.
    std::filesystem::path meta_path;
    for (auto& e : std::filesystem::recursive_directory_iterator(root))
        if (e.path().string().size() > 14 && e.path().string().rfind(".metadata.json") == e.path().string().size() - 14 &&
            (meta_path.empty() || e.path() > meta_path)) meta_path = e.path();
    ASSERT_FALSE(meta_path.empty());
    std::string meta = slurp(meta_path);
    const size_t ml_key = meta.find("\"manifest-list\"");
    ASSERT_NE(ml_key, std::string::npos);
    const size_t q1 = meta.find('"', meta.find(':', ml_key)), q2 = meta.find('"', q1 + 1);
    const std::string old_list = meta.substr(q1 + 1, q2 - q1 - 1);
    std::string old_list_path = old_list;
    if (old_list_path.rfind("file:", 0) == 0) old_list_path = old_list_path.substr(5);
    if (!std::filesystem::exists(old_list_path)) old_list_path = (root / old_list).generic_string();
    const std::string list_bytes = slurp(old_list_path);
    ASSERT_FALSE(list_bytes.empty());
    ManifestListEntry* mle = nullptr; uint32_t n_mle = 0;
    ASSERT_TRUE(manifest_list_parse_avro_grow(reinterpret_cast<const uint8_t*>(list_bytes.data()), list_bytes.size(), &arena, 8, &mle, &n_mle));
    ASSERT_GE(n_mle, 1u);

    // Puffin file: "PFA1" + blob + (unparsed) footer padding; the manifest locates the blob by offset/size.
    const uint32_t del_pos[] = {2, 5, 9};
    const uint8_t* blob = nullptr; uint64_t blob_len = 0;
    ASSERT_TRUE(iceberg_dv_encode(del_pos, 3, &arena, &blob, &blob_len));
    auto build = [&](bool corrupt) {
        std::string puffin = "PFA1";
        std::string bl(reinterpret_cast<const char*>(blob), blob_len);
        if (corrupt) bl[bl.size() / 2] ^= 0x01;
        puffin += bl; puffin += "PFA1{}";
        const std::string pf = (root / "data" / "dv-1.puffin").generic_string();
        spit(pf, puffin);
        const std::string dm = (root / "metadata" / "dv-manifest.json").generic_string();
        std::ostringstream m;
        m << "[{\"status\":1,\"data_file\":{\"content\":1,\"file_path\":\"" << pf << "\",\"file_format\":\"PUFFIN\","
          << "\"record_count\":3,\"file_size_in_bytes\":" << puffin.size() << ",\"content_offset\":4,\"content_size_in_bytes\":"
          << blob_len << ",\"referenced_data_file\":\"" << data_file << "\"}}]";
        spit(dm, m.str());
        const std::string ln = (root / "metadata" / "dv-list.json").generic_string();
        std::ostringstream l;
        l << "[";
        for (uint32_t i = 0; i < n_mle; ++i)
            l << (i ? "," : "") << "{\"manifest_path\":\"" << mle[i].manifest_path << "\",\"manifest_length\":" << mle[i].manifest_length
              << ",\"partition_spec_id\":" << mle[i].partition_spec_id << ",\"content\":0}";
        l << ",{\"manifest_path\":\"" << dm << "\",\"manifest_length\":" << m.str().size() << ",\"partition_spec_id\":0,\"content\":1,\"sequence_number\":100}]";
        spit(ln, l.str());
        std::string patched = meta;
        patched.replace(q1 + 1, q2 - q1 - 1, ln);
        const size_t fv = patched.find("\"format-version\"");
        if (fv != std::string::npos) { const size_t c = patched.find(':', fv); patched[patched.find_first_of("0123456789", c)] = '3'; }
        spit(meta_path, patched);
    };

    auto scan_ids = [&](std::vector<int64_t>* out) {
        FilesystemCatalog cat_fs{}; Catalog cat{};
        EXPECT_TRUE(filesystem_catalog_init(&cat_fs, base.generic_string().c_str(), &cat));
        bolt::Arena ra;
        TableHandle* rh = nullptr;
        if (!iceberg_table_open(&rh, &ra, &cat, "default", "t")) return false;
        ReadOptions ro{}; read_options_init(&ro);
        ScanHandle* sh = nullptr;
        if (!iceberg_scan_open(&sh, rh, &ro)) { iceberg_table_close(rh); return false; }
        bool ok = true;
        for (int g = 0; g < 64 && ok; ++g) {
            bolt::BoltBatch bb{}; bool eof = false;
            if (!iceberg_scan_next_batch(sh, &bb, &eof)) { ok = false; break; }
            if (eof) break;
            const auto* v = static_cast<const int64_t*>(bb.columns[bb.read_epoch][0].data);
            for (int64_t r = 0; r < bb.num_rows; ++r) out->push_back(v[r]);
        }
        iceberg_scan_close(sh); iceberg_table_close(rh);
        return ok;
    };

    build(false);
    std::vector<int64_t> got;
    ASSERT_TRUE(scan_ids(&got)) << "a table with a valid deletion vector must scan";
    std::vector<int64_t> want = {100, 101, 103, 104, 106, 107, 108};
    EXPECT_EQ(got, want) << "rows at positions 2, 5 and 9 must be absent, every other row present";

    build(true);
    std::vector<int64_t> bad;
    EXPECT_FALSE(scan_ids(&bad) && bad.size() == 10) << "a CRC-failing DV must not yield the undeleted rows";
}

// Negative control for column binding: a schema column that no data file carries is NULL when optional and a refusal (never invented
// values or silent nulls) when required.
TEST(IcebergSchemaBinding, RequiredColumnWithNoDataRefusesAndOptionalReadsNull) {
    bolt::Arena arena;
    const std::filesystem::path base = std::filesystem::temp_directory_path() / "bolt_iceberg_required_col";
    std::error_code ec;
    std::filesystem::remove_all(base, ec);
    const std::filesystem::path root = base / "default" / "t";
    std::filesystem::create_directories(root, ec);
    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.generic_string().c_str(), &os));
    static SchemaField fields[1]{};
    Schema sch{}; sch.fields = fields; sch.cap_fields = 1; sch.n_fields = 1;
    fields[0].id = 1; fields[0].required = true;
    std::strncpy(fields[0].name, "id", sizeof(fields[0].name) - 1);
    std::strncpy(fields[0].type, "long", sizeof(fields[0].type) - 1);
    PartitionSpec spec{}; SortOrder sort{}; WriteOptions wo; write_options_init(&wo);
    TableHandle* wh = nullptr;
    ASSERT_TRUE(table_create(&wh, &arena, &os, root.generic_string().c_str(), &sch, &spec, &sort, &wo));
    AppendHandle* ah = nullptr;
    ASSERT_TRUE(append_open(&ah, wh));
    bolt::BoltBatch b{};
    bolt::BoltBatch::init_empty(&b);
    b.arena = &arena; b.num_rows = 3; b.num_cols = 1;
    bolt::BoltBatch::alloc_columns(&b, &arena, 1);
    b.schema.add_field("id", bolt::BoltType::Int64, false);
    auto* ids = arena.allocate_array<int64_t>(3);
    for (int i = 0; i < 3; ++i) ids[i] = 7 + i;
    b.columns[b.read_epoch][0].type = bolt::BoltType::Int64;
    b.columns[b.read_epoch][0].length = 3;
    b.columns[b.read_epoch][0].data = ids;
    ASSERT_TRUE(append_write(ah, &b));
    ASSERT_TRUE(append_commit(ah));
    append_close(ah);
    table_close(wh);

    std::filesystem::path meta_path;
    for (auto& e : std::filesystem::recursive_directory_iterator(root)) {
        const std::string n = e.path().string();
        if (n.size() > 14 && n.rfind(".metadata.json") == n.size() - 14 && (meta_path.empty() || e.path() > meta_path)) meta_path = e.path();
    }
    ASSERT_FALSE(meta_path.empty());
    const std::string original = slurp(meta_path);

    auto scan = [&](std::vector<int64_t>* second_col_nulls) {
        FilesystemCatalog cat_fs{}; Catalog cat{};
        EXPECT_TRUE(filesystem_catalog_init(&cat_fs, base.generic_string().c_str(), &cat));
        bolt::Arena ra;
        TableHandle* rh = nullptr;
        if (!iceberg_table_open(&rh, &ra, &cat, "default", "t")) return 2;
        ReadOptions ro{}; read_options_init(&ro);
        ScanHandle* sh = nullptr;
        if (!iceberg_scan_open(&sh, rh, &ro)) { iceberg_table_close(rh); return 3; }
        int rc = 0;
        for (int g = 0; g < 16 && rc == 0; ++g) {
            bolt::BoltBatch bb{}; bool eof = false;
            if (!iceberg_scan_next_batch(sh, &bb, &eof)) { rc = 1; break; }
            if (eof) break;
            int extra = -1;
            for (int32_t c = 0; c < bb.num_cols; ++c) if (std::strcmp(bb.schema.fields[c].name, "extra") == 0) extra = c;
            for (int64_t r = 0; r < bb.num_rows; ++r)
                second_col_nulls->push_back(extra >= 0 && bb.columns[bb.read_epoch][extra].is_null(r) ? 1 : 0);
        }
        iceberg_scan_close(sh); iceberg_table_close(rh);
        return rc;
    };
    auto with_extra = [&](bool required) {
        std::string patched = original;
        const size_t at = patched.find("\"fields\":[");
        EXPECT_NE(at, std::string::npos);
        patched.insert(at + 10, std::string("{\"id\":3,\"name\":\"extra\",\"required\":") + (required ? "true" : "false") + ",\"type\":\"long\"},");
        spit(meta_path, patched);
    };

    with_extra(false);
    std::vector<int64_t> nulls;
    EXPECT_EQ(scan(&nulls), 0) << "an optional column added after the file was written reads as null";
    EXPECT_EQ(nulls.size(), 3u);
    for (int64_t n : nulls) EXPECT_EQ(n, 1);

    with_extra(true);
    std::vector<int64_t> none;
    EXPECT_NE(scan(&none), 0) << "a required column with no data must refuse, not read as null";

    // G2ICE-326: columns added with an initial-default of the remaining types read it (a file written before them lacks them).
    {
        std::string patched = original;
        const size_t at = patched.find("\"fields\":[");
        ASSERT_NE(at, std::string::npos);
        patched.insert(at + 10, std::string(
            "{\"id\":11,\"name\":\"tt\",\"required\":false,\"type\":\"time\",\"initial-default\":\"12:34:56.789012\"},"
            "{\"id\":12,\"name\":\"ts\",\"required\":false,\"type\":\"timestamp\",\"initial-default\":\"2024-01-02T03:04:05.000006\"},"
            "{\"id\":13,\"name\":\"d9\",\"required\":false,\"type\":\"decimal(9,2)\",\"initial-default\":\"-12.34\"},"
            "{\"id\":14,\"name\":\"d30\",\"required\":false,\"type\":\"decimal(30,2)\",\"initial-default\":\"-12345678901234567890.12\"},"
            "{\"id\":15,\"name\":\"uu\",\"required\":false,\"type\":\"uuid\",\"initial-default\":\"f79c3e09-677c-4bbd-a479-3f349cb785e7\"},"
            "{\"id\":16,\"name\":\"ff\",\"required\":false,\"type\":\"fixed[3]\",\"initial-default\":\"0a0b0c\"},"
            "{\"id\":17,\"name\":\"bb\",\"required\":false,\"type\":\"binary\",\"initial-default\":\"deadbeef\"},"
            "{\"id\":18,\"name\":\"ss\",\"required\":false,\"type\":\"string\",\"initial-default\":\"a\\\"b\\u00e9\\ud83d\\ude00\"},"));
        spit(meta_path, patched);
        FilesystemCatalog cat_fs{}; Catalog cat{};
        ASSERT_TRUE(filesystem_catalog_init(&cat_fs, base.generic_string().c_str(), &cat));
        bolt::Arena ra;
        TableHandle* rh = nullptr;
        ASSERT_TRUE(iceberg_table_open(&rh, &ra, &cat, "default", "t"));
        ReadOptions ro{}; read_options_init(&ro);
        ScanHandle* sh = nullptr;
        ASSERT_TRUE(iceberg_scan_open(&sh, rh, &ro));
        bolt::BoltBatch bb{}; bool eof = false;
        ASSERT_TRUE(iceberg_scan_next_batch(sh, &bb, &eof));
        ASSERT_FALSE(eof);
        ASSERT_EQ(bb.num_rows, 3);
        const auto col = [&](const char* n) -> const bolt::BoltColumn& {
            for (int32_t c = 0; c < bb.num_cols; ++c) if (std::strcmp(bb.schema.fields[c].name, n) == 0) return bb.columns[bb.read_epoch][c];
            ADD_FAILURE() << n;
            return bb.columns[bb.read_epoch][0];
        };
        for (int64_t r = 0; r < 3; ++r) {
            EXPECT_EQ(col("tt").type, bolt::BoltType::Duration);
            EXPECT_EQ(static_cast<const int64_t*>(col("tt").data)[r], int64_t{45296789012});
            EXPECT_EQ(static_cast<const int64_t*>(col("ts").data)[r], int64_t{1704164645000006});
            EXPECT_EQ(col("d9").type, bolt::BoltType::Decimal64);
            EXPECT_EQ(col("d9").decimal_scale, 2u);
            EXPECT_EQ(static_cast<const int64_t*>(col("d9").data)[r], int64_t{-1234});
            EXPECT_EQ(col("d30").type, bolt::BoltType::Decimal128);
            int64_t lo = 0, hi = 0;
            std::memcpy(&lo, static_cast<const uint8_t*>(col("d30").data) + r * 16, 8);
            std::memcpy(&hi, static_cast<const uint8_t*>(col("d30").data) + r * 16 + 8, 8);
            EXPECT_EQ(hi, -67);                                          // -1234567890123456789012 as a two's-complement int128
            EXPECT_EQ(static_cast<uint64_t>(lo), uint64_t{0x12edc4f427dfc5ecu});
            const uint8_t uuid[16] = {0xf7, 0x9c, 0x3e, 0x09, 0x67, 0x7c, 0x4b, 0xbd, 0xa4, 0x79, 0x3f, 0x34, 0x9c, 0xb7, 0x85, 0xe7};
            EXPECT_EQ(std::memcmp(static_cast<const uint8_t*>(col("uu").data) + r * 16, uuid, 16), 0);
            EXPECT_EQ(col("ff").fixed_width, 3u);
            EXPECT_EQ(std::memcmp(static_cast<const uint8_t*>(col("ff").data) + r * 16, "\x0a\x0b\x0c", 3), 0);
            const auto* bv = static_cast<const bolt::StringView*>(col("bb").data) + r;
            EXPECT_EQ(bv->length, 4u);
            EXPECT_EQ(std::memcmp(bv->prefix, "\xde\xad\xbe\xef", 4), 0);
            const auto* sv = static_cast<const bolt::StringView*>(col("ss").data) + r;
            EXPECT_EQ(sv->length, 9u);                                   // a " b, é (2 bytes), the emoji (4)
            EXPECT_EQ(std::memcmp(sv->prefix, "a\"b\xc3", 4), 0);
        }
        iceberg_scan_close(sh); iceberg_table_close(rh);
    }
    spit(meta_path, original);
}
