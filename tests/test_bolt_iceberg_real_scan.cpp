// test_bolt_iceberg_real_scan.cpp — SCAN a real Iceberg table, end to end.
//
// G2FEAT-76 proved the manifest DECODER against pyiceberg-written .avro files.
// This proves the SCAN: metadata.json -> snapshot -> manifest list -> manifests
// -> Parquet data files -> ROWS, through the public `iceberg_table_open` /
// `iceberg_scan_open` / `iceberg_scan_next_batch` surface, with no test-only
// entry point anywhere in the path.
//
// PROVENANCE (this is the whole point — a table bolt wrote itself would prove
// nothing about real-world compatibility): every byte under
// data/golden_iceberg_table/ was written by **pyiceberg 0.11.1** driving a real
// `SqlCatalog` warehouse — `create_table` + two `append(pyarrow.Table)` calls.
// Regenerate with data/make_iceberg_real_table.py.
//
// The warehouse is written at pyiceberg's UNTOUCHED DEFAULTS -- no table
// properties are set at all -- so the data files are **zstd**, pyiceberg's
// default Parquet codec. Until G2FEAT-134 gave bolt a self-contained zstd
// decoder this fixture had to pin `write.parquet.compression-codec=snappy`,
// which meant the happy path: the table the most common Python writer actually
// produces would NOT have read. It does now, and this test is what proves it.
// The one remaining disclosed choice is that absolute `file://` locations are
// left exactly as the writer recorded them (see the generator's docstring).
//
// THE SOURCE TABLE — db.trades (id:long, sym:string, price:double,
// active:boolean), format-version 2, identity-partitioned on `sym`:
//     append #1 -> 6 rows  AAPL(1,101.5) AAPL(2,102.25) MSFT(3,310.0)
//                          MSFT(4,311.75) GOOG(5,140.0) GOOG(6,NULL)  3 files
//     append #2 -> 3 rows  AAPL(7,103.0) TSLA(8,250.5) TSLA(9,251.0)   2 files
// so the current snapshot spans BOTH manifests: 5 data files, 9 rows.
//
// GROUND TRUTH, from `table.scan().to_arrow().to_pydict()` on that same table:
//     id     [7, 8, 9, 1, 2, 3, 4, 5, 6]
//     sym    [AAPL, TSLA, TSLA, AAPL, AAPL, MSFT, MSFT, GOOG, GOOG]
//     price  [103.0, 250.5, 251.0, 101.5, 102.25, 310.0, 311.75, 140.0, NULL]
//     active [T, F, T, T, F, T, T, NULL, F]
// Every assertion below is a VALUE from that, keyed by `id`, so a decoder that
// mis-aligns columns or files cannot pass by accident. File ORDER is not
// asserted — Iceberg does not promise one — but the row multiset is exact.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/format.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/iceberg/snapshot.h"
#include "bolt/lakehouse/object_store.h"

namespace {

using namespace bolt;
using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

const char* warehouse_root() {
    static const std::string root =
        std::string(BOLT_TEST_DATA_DIR) + "/golden_iceberg_table";
    return root.c_str();
}

// One scanned row, reduced to the four columns the table has.
struct Row {
    int64_t     id = 0;
    std::string sym;
    double      price = 0.0;
    bool        price_null = false;
    bool        active = false;
    bool        active_null = false;
};

// Drain a whole scan into rows keyed by `id`. Returns false on any scan error,
// so a silent short read cannot be mistaken for a small table.
bool drain(ScanHandle* s, std::map<int64_t, Row>* out, uint32_t* out_batches) {
    *out_batches = 0;
    for (uint32_t guard = 0; guard < 1024u; ++guard) {      // bounded
        BoltBatch b{};
        bool eof = false;
        if (!iceberg_scan_next_batch(s, &b, &eof)) return false;
        if (eof) return true;
        ++*out_batches;
        const int ci = b.schema.find_field("id");
        const int cs = b.schema.find_field("sym");
        const int cp = b.schema.find_field("price");
        const int ca = b.schema.find_field("active");
        if (ci < 0 || cs < 0 || cp < 0 || ca < 0) return false;
        const BoltColumn* cols = b.columns[b.read_epoch];
        for (int64_t r = 0; r < b.num_rows; ++r) {
            Row row;
            row.id = cols[ci].typed_data<int64_t>()[r];
            const uint8_t* sp = nullptr;
            int32_t sl = 0;
            cols[cs].utf8_at(r, &sp, &sl);
            row.sym.assign(reinterpret_cast<const char*>(sp),
                           static_cast<size_t>(sl));
            row.price_null = cols[cp].is_null(r);
            row.price = row.price_null ? 0.0 : cols[cp].typed_data<double>()[r];
            row.active_null = cols[ca].is_null(r);
            // bolt's Parquet reader materialises BOOLEAN as Int64 0/1.
            row.active = !row.active_null &&
                         cols[ca].typed_data<int64_t>()[r] != 0;
            (*out)[row.id] = row;
        }
    }
    return false;
}

// Open the committed warehouse and start a scan with `opts`.
bool open_scan(Arena* arena, FilesystemCatalog* fs, Catalog* cat,
               TableHandle** th, ScanHandle** sh, const ReadOptions* opts) {
    if (!filesystem_catalog_init(fs, warehouse_root(), cat)) return false;
    if (!iceberg_table_open(th, arena, cat, "db", "trades")) return false;
    return iceberg_scan_open(sh, *th, opts);
}

void set_str_pred(ReadOptions* o, const char* col, const char* val) {
    o->n_predicates = 1;
    o->predicates[0].op = PredicateOp::kEq;
    std::snprintf(o->predicates[0].column, kLakeMaxColName, "%s", col);
    o->predicates[0].value.type = BoltType::Utf8;
    std::snprintf(o->predicates[0].value.str, kLakeMaxValBytes, "%s", val);
    o->predicates[0].value.str_len =
        static_cast<uint32_t>(std::strlen(val));
}

}  // namespace

// ---------------------------------------------------------------------------
// The premise. Every W4 assumption the scan used to make is FALSE for this
// fixture; if these stop holding, the tests below stop proving what they claim.
// ---------------------------------------------------------------------------

TEST(IcebergRealScan, FixtureIsCatalogManagedNotHadoopLayout) {
    Arena arena;
    ObjectStore os{};
    FilesystemObjectStore fos{};
    ASSERT_TRUE(filesystem_object_store_init(&fos, warehouse_root(), &os));

    // No version-hint.text and no v<N>.metadata.json: a pyiceberg SqlCatalog
    // (like REST/Glue/Hive) writes neither, and the pre-G2FEAT-125 discovery
    // required both.
    const uint8_t* body = nullptr;
    uint64_t blen = 0;
    EXPECT_NE(os_get(&os, "db/trades/metadata/version-hint.text", &arena,
                     &body, &blen), kOsOk)
        << "fixture unexpectedly carries a Hadoop-catalog version hint";

    ObjectEntry* listing = arena.allocate_array<ObjectEntry>(64u);
    ASSERT_NE(listing, nullptr);
    uint32_t n = 0;
    ASSERT_EQ(os_list(&os, "db/trades/metadata/", listing, 64u, &n), kOsOk);
    uint32_t n_hadoop = 0, n_catalog_managed = 0, n_avro = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const std::string k(listing[i].key);
        const size_t slash = k.find_last_of('/');
        const std::string tail =
            slash == std::string::npos ? k : k.substr(slash + 1);
        if (tail.find(".metadata.json") != std::string::npos) {
            if (tail[0] == 'v') ++n_hadoop; else ++n_catalog_managed;
        } else if (tail.size() > 5 &&
                   tail.compare(tail.size() - 5, 5, ".avro") == 0) {
            const uint8_t* b = nullptr;
            uint64_t bl = 0;
            ASSERT_EQ(os_get(&os, listing[i].key, &arena, &b, &bl), kOsOk);
            ASSERT_GE(bl, 4u);
            // Avro object-container magic — these are NOT the JSON stand-ins
            // the W4 parser handled.
            EXPECT_EQ(std::memcmp(b, "Obj\x01", 4), 0) << k;
            ++n_avro;
        }
    }
    EXPECT_EQ(n_hadoop, 0u);
    EXPECT_EQ(n_catalog_managed, 3u);   // 00000-, 00001-, 00002-
    EXPECT_EQ(n_avro, 4u);              // 2 manifest lists + 2 manifests
}

TEST(IcebergRealScan, FixtureIsRelocatedFromWhereItWasWritten) {
    Arena arena;
    FilesystemCatalog fs{};
    Catalog cat{};
    ASSERT_TRUE(filesystem_catalog_init(&fs, warehouse_root(), &cat));
    TableHandle* th = nullptr;
    ASSERT_TRUE(iceberg_table_open(&th, &arena, &cat, "db", "trades"))
        << "catalog-managed <NNNNN>-<uuid>.metadata.json was not discovered";
    const Metadata* m = iceberg_table_metadata(th);
    ASSERT_NE(m, nullptr);

    // The writer's absolute location is baked into metadata.json and into
    // every manifest-list / manifest / data-file path. It is not where this
    // table now lives, so resolving ANY of them needs the relocation path.
    char here[kCatMaxPath];
    ASSERT_EQ(cat_table_path(&cat, "db", "trades", here, sizeof(here)),
              kCatOk);
    EXPECT_EQ(std::strstr(m->location, here), nullptr)
        << "fixture location " << m->location
        << " coincides with its checkout path " << here
        << " — relocation is not being exercised";
    EXPECT_EQ(std::strncmp(m->location, "file://", 7), 0) << m->location;
    ASSERT_EQ(m->n_snapshots, 2u);
    EXPECT_EQ(m->format_version, 2);
    iceberg_table_close(th);
}

// ---------------------------------------------------------------------------
// The acceptance test: rows out, values checked against pyiceberg's own scan.
// ---------------------------------------------------------------------------

TEST(IcebergRealScan, FullScanMatchesPyicebergValues) {
    Arena arena;
    FilesystemCatalog fs{};
    Catalog cat{};
    TableHandle* th = nullptr;
    ScanHandle* sh = nullptr;
    ASSERT_TRUE(open_scan(&arena, &fs, &cat, &th, &sh, nullptr));

    std::map<int64_t, Row> rows;
    uint32_t batches = 0;
    ASSERT_TRUE(drain(sh, &rows, &batches))
        << "scan reported an error — a live data file could not be produced";

    // 5 data files, one row group each, so 5 batches; 9 rows across BOTH
    // snapshots' manifests. A scan that read only the newest manifest would
    // report 2 files / 3 rows; one that dropped a file would report fewer.
    EXPECT_EQ(batches, 5u);
    ASSERT_EQ(rows.size(), 9u);

    struct Expect {
        int64_t id; const char* sym; double price; bool price_null;
        bool active; bool active_null;
    };
    const Expect want[] = {
        {1, "AAPL", 101.5,  false, true,  false},
        {2, "AAPL", 102.25, false, false, false},
        {3, "MSFT", 310.0,  false, true,  false},
        {4, "MSFT", 311.75, false, true,  false},
        {5, "GOOG", 140.0,  false, false, true },   // active NULL
        {6, "GOOG", 0.0,    true,  false, false},   // price NULL
        {7, "AAPL", 103.0,  false, true,  false},
        {8, "TSLA", 250.5,  false, false, false},
        {9, "TSLA", 251.0,  false, true,  false},
    };
    for (const Expect& e : want) {
        const auto it = rows.find(e.id);
        ASSERT_NE(it, rows.end()) << "missing id=" << e.id;
        const Row& r = it->second;
        EXPECT_EQ(r.sym, e.sym) << "id=" << e.id;
        EXPECT_EQ(r.price_null, e.price_null) << "id=" << e.id;
        if (!e.price_null) EXPECT_DOUBLE_EQ(r.price, e.price) << "id=" << e.id;
        EXPECT_EQ(r.active_null, e.active_null) << "id=" << e.id;
        if (!e.active_null) EXPECT_EQ(r.active, e.active) << "id=" << e.id;
    }
    iceberg_scan_close(sh);
    iceberg_table_close(th);
}

// Time travel: the FIRST snapshot saw only append #1 — 3 files, ids 1..6. This
// also proves the scan follows the requested snapshot's OWN manifest list
// rather than whatever the current metadata happens to point at.
TEST(IcebergRealScan, TimeTravelToFirstSnapshotSeesOnlySixRows) {
    Arena arena;
    FilesystemCatalog fs{};
    Catalog cat{};
    ASSERT_TRUE(filesystem_catalog_init(&fs, warehouse_root(), &cat));
    TableHandle* th = nullptr;
    ASSERT_TRUE(iceberg_table_open(&th, &arena, &cat, "db", "trades"));

    // Resolve the parent of the current snapshot from the table's own metadata
    // rather than hard-coding an id that regeneration would change.
    const Metadata* m = iceberg_table_metadata(th);
    ASSERT_NE(m, nullptr);
    const Snapshot* cur = snapshot_by_id(m, m->current_snapshot_id);
    ASSERT_NE(cur, nullptr);
    ASSERT_NE(cur->parent_snapshot_id, 0);

    ReadOptions opts{};
    read_options_init(&opts);
    opts.snapshot_id = cur->parent_snapshot_id;

    ScanHandle* sh = nullptr;
    ASSERT_TRUE(iceberg_scan_open(&sh, th, &opts));
    std::map<int64_t, Row> rows;
    uint32_t batches = 0;
    ASSERT_TRUE(drain(sh, &rows, &batches));
    EXPECT_EQ(batches, 3u);
    ASSERT_EQ(rows.size(), 6u);
    for (int64_t id = 1; id <= 6; ++id) {
        EXPECT_NE(rows.find(id), rows.end()) << "missing id=" << id;
    }
    EXPECT_EQ(rows.find(7), rows.end()) << "append #2 leaked into snapshot 1";
    EXPECT_EQ(rows[5].sym, "GOOG");
    EXPECT_TRUE(rows[6].price_null);
    iceberg_scan_close(sh);
    iceberg_table_close(th);
}

// ---------------------------------------------------------------------------
// Pruning. Both of these read EVERY file before G2FEAT-125 — partition pruning
// joined an ordinal against a v2 field-id (never matching) and stat pruning ran
// strtoll over little-endian bytes (never decoding). Asserting the row VALUES
// as well as the file count is what makes an over-eager prune — which would
// silently drop live rows — a failure rather than a speedup.
// ---------------------------------------------------------------------------

TEST(IcebergRealScan, PartitionPruneOnIdentityString) {
    Arena arena;
    FilesystemCatalog fs{};
    Catalog cat{};
    ReadOptions opts{};
    read_options_init(&opts);
    set_str_pred(&opts, "sym", "TSLA");

    TableHandle* th = nullptr;
    ScanHandle* sh = nullptr;
    ASSERT_TRUE(open_scan(&arena, &fs, &cat, &th, &sh, &opts));
    std::map<int64_t, Row> rows;
    uint32_t batches = 0;
    ASSERT_TRUE(drain(sh, &rows, &batches));

    // Exactly the one sym=TSLA data file: identity partitioning makes every
    // other file provably irrelevant. 5 batches would mean the ordinal
    // /field-id join silently failed open again.
    EXPECT_EQ(batches, 1u);
    ASSERT_EQ(rows.size(), 2u);
    ASSERT_NE(rows.find(8), rows.end());
    ASSERT_NE(rows.find(9), rows.end());
    EXPECT_EQ(rows[8].sym, "TSLA");
    EXPECT_DOUBLE_EQ(rows[8].price, 250.5);
    EXPECT_DOUBLE_EQ(rows[9].price, 251.0);
    iceberg_scan_close(sh);
    iceberg_table_close(th);
}

TEST(IcebergRealScan, StatPruneOnBinaryLongBounds) {
    Arena arena;
    FilesystemCatalog fs{};
    Catalog cat{};
    ReadOptions opts{};
    read_options_init(&opts);
    opts.n_predicates = 1;
    opts.predicates[0].op = PredicateOp::kGe;
    std::snprintf(opts.predicates[0].column, kLakeMaxColName, "%s", "id");
    opts.predicates[0].value.type = BoltType::Int64;
    opts.predicates[0].value.i64  = 7;

    TableHandle* th = nullptr;
    ScanHandle* sh = nullptr;
    ASSERT_TRUE(open_scan(&arena, &fs, &cat, &th, &sh, &opts));
    std::map<int64_t, Row> rows;
    uint32_t batches = 0;
    ASSERT_TRUE(drain(sh, &rows, &batches));

    // `id` upper bounds are 2, 4, 6 (append #1) and 7, 9 (append #2). Only the
    // last two files can hold id >= 7, so 3 of 5 are pruned on the manifest's
    // BINARY bounds alone. The scan does not filter rows, so both survivors
    // come back whole: ids {7} and {8,9}.
    EXPECT_EQ(batches, 2u);
    ASSERT_EQ(rows.size(), 3u);
    EXPECT_NE(rows.find(7), rows.end());
    EXPECT_NE(rows.find(8), rows.end());
    EXPECT_NE(rows.find(9), rows.end());
    EXPECT_EQ(rows.find(1), rows.end()) << "a pruned file was still scanned";
    iceberg_scan_close(sh);
    iceberg_table_close(th);
}

TEST(IcebergRealScan, StatPruneOnBinaryDoubleBounds) {
    Arena arena;
    FilesystemCatalog fs{};
    Catalog cat{};
    ReadOptions opts{};
    read_options_init(&opts);
    opts.n_predicates = 1;
    // price >= 300 keeps only the MSFT file (310.0 .. 311.75). This is exactly
    // the encoding the ticket calls out: 250.5 is 00 00 00 00 00 50 6F 40, so a
    // C-string read of that bound is empty and leaves every file unpruned.
    opts.predicates[0].op = PredicateOp::kGe;
    std::snprintf(opts.predicates[0].column, kLakeMaxColName, "%s", "price");
    opts.predicates[0].value.type = BoltType::Float64;
    opts.predicates[0].value.f64  = 300.0;

    TableHandle* th = nullptr;
    ScanHandle* sh = nullptr;
    ASSERT_TRUE(open_scan(&arena, &fs, &cat, &th, &sh, &opts));
    std::map<int64_t, Row> rows;
    uint32_t batches = 0;
    ASSERT_TRUE(drain(sh, &rows, &batches));
    EXPECT_EQ(batches, 1u);
    ASSERT_EQ(rows.size(), 2u);
    ASSERT_NE(rows.find(3), rows.end());
    ASSERT_NE(rows.find(4), rows.end());
    EXPECT_DOUBLE_EQ(rows[3].price, 310.0);
    EXPECT_DOUBLE_EQ(rows[4].price, 311.75);
    iceberg_scan_close(sh);
    iceberg_table_close(th);
}

// A predicate no file can satisfy must prune everything and end clean — not
// error, and not leak a stray batch.
TEST(IcebergRealScan, PruneEverythingIsCleanEof) {
    Arena arena;
    FilesystemCatalog fs{};
    Catalog cat{};
    ReadOptions opts{};
    read_options_init(&opts);
    set_str_pred(&opts, "sym", "NVDA");

    TableHandle* th = nullptr;
    ScanHandle* sh = nullptr;
    ASSERT_TRUE(open_scan(&arena, &fs, &cat, &th, &sh, &opts));
    std::map<int64_t, Row> rows;
    uint32_t batches = 0;
    ASSERT_TRUE(drain(sh, &rows, &batches));
    EXPECT_EQ(batches, 0u);
    EXPECT_TRUE(rows.empty());
    iceberg_scan_close(sh);
    iceberg_table_close(th);
}
