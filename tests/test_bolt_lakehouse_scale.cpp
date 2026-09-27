// LIMITS L5 (G2CHK-336) at scale, against an external oracle.
//
// scripts/l5_iceberg_scale.py writes a pyiceberg table (100,000 data files,
// 5,000 snapshots, 12 partition columns, schema evolved to 300 columns) and
// records pyiceberg's and DuckDB's row counts and sum(v) for the current and
// several time-travel snapshots. This reads the same table with bolt's own
// Iceberg reader and requires the same answers.
//
// Env-gated: BOLT_L5_SCALE_DIR=<the script's out_dir>. Skipped when unset.

#include "bolt/bolt_arena.h"
#include "bolt/bolt_budget.h"
#include "bolt/bolt_resource.h"
#include "bolt/bolt_column.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/scan.h"
#include "bolt/lakehouse/object_store.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

struct Check { int64_t snapshot_id; int64_t rows; int64_t sum_v; };

struct Oracle {
    std::string metadata;
    std::string location;
    uint32_t snapshots = 0, schema_fields = 0, data_files = 0;
    std::vector<Check> checks;
};

bool load_oracle(const std::string& dir, Oracle* o) {
    std::ifstream in(dir + "/oracle.txt");
    if (!in) return false;
    std::getline(in, o->metadata);
    std::getline(in, o->location);
    in >> o->snapshots >> o->schema_fields >> o->data_files;
    Check c{};
    while (in >> c.snapshot_id >> c.rows >> c.sum_v) o->checks.push_back(c);
    return !o->checks.empty();
}

std::string slurp(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

// Column index of `name` in the batch schema, or -1.
int32_t col_of(const bolt::BoltBatch& b, const char* name) {
    for (uint32_t i = 0; i < b.num_cols; ++i) {
        if (std::strcmp(b.schema.fields[i].name, name) == 0)
            return static_cast<int32_t>(i);
    }
    return -1;
}

}  // namespace

TEST(LakeScale, BoltMatchesPyicebergAndDuckDB) {
    const char* dir = std::getenv("BOLT_L5_SCALE_DIR");
    if (dir == nullptr) GTEST_SKIP() << "BOLT_L5_SCALE_DIR unset";
    Oracle o;
    ASSERT_TRUE(load_oracle(dir, &o)) << dir;
    const std::string mj = slurp(o.metadata);
    ASSERT_FALSE(mj.empty());

    FilesystemObjectStore fs{};
    ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, "/", &os));
    const std::string prefix = o.location.substr(1);   // store keys are root-relative

    for (const Check& c : o.checks) {
        bolt::Arena a;
        iceberg::TableHandle* h = nullptr;
        ASSERT_TRUE(iceberg_table_open_on_store(
            &h, &a, &os, prefix.c_str(),
            reinterpret_cast<const uint8_t*>(mj.data()),
            static_cast<uint32_t>(mj.size())));
        const Metadata* m = iceberg_table_metadata(h);
        ASSERT_EQ(m->n_snapshots, o.snapshots);
        ASSERT_EQ(metadata_current_schema(m)->n_fields, o.schema_fields);

        ReadOptions ro{};
        read_options_init(&ro);
        ro.snapshot_id = c.snapshot_id;
        ScanHandle* sh = nullptr;
        ASSERT_TRUE(iceberg_scan_open(&sh, h, &ro)) << c.snapshot_id;
        int64_t rows = 0, sum = 0;
        for (uint64_t guard = 0; guard < 100000000ull; ++guard) {
            bolt::BoltBatch b{};
            bool eof = false;
            ASSERT_TRUE(iceberg_scan_next_batch(sh, &b, &eof)) << c.snapshot_id;
            if (eof) break;
            const int32_t vi = col_of(b, "v");
            ASSERT_GE(vi, 0);
            const auto* v = static_cast<const int64_t*>(
                b.columns[b.read_epoch][vi].data);
            for (int64_t r = 0; r < b.num_rows; ++r) sum += v[r];
            rows += b.num_rows;
        }
        iceberg_scan_close(sh);
        EXPECT_EQ(rows, c.rows) << "snapshot " << c.snapshot_id;
        EXPECT_EQ(sum, c.sum_v) << "snapshot " << c.snapshot_id;
    }
}

// The same 100,000-file scan under a 1 MiB metadata budget is refused with
// ResourceExhausted naming the knob; it never returns a short or empty table.
TEST(LakeScale, TinyMetadataBudgetIsResourceExhausted) {
    const char* dir = std::getenv("BOLT_L5_SCALE_DIR");
    if (dir == nullptr) GTEST_SKIP() << "BOLT_L5_SCALE_DIR unset";
    Oracle o;
    ASSERT_TRUE(load_oracle(dir, &o)) << dir;
    const std::string mj = slurp(o.metadata);
    FilesystemObjectStore fs{};
    ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, "/", &os));
    const std::string prefix = o.location.substr(1);
    auto& lt = bolt::bolt_limits();
    const uint32_t id =
        static_cast<uint32_t>(bolt::bolt_limits_id::lake_metadata_budget_mb);
    const uint64_t saved = bolt::limits_value(lt, id);
    ASSERT_TRUE(bolt::limits_set(&lt, id, 1u, nullptr, 0));
    bolt::clear_resource_exhausted();

    bolt::Arena a;
    iceberg::TableHandle* h = nullptr;
    ASSERT_TRUE(iceberg_table_open_on_store(
        &h, &a, &os, prefix.c_str(),
        reinterpret_cast<const uint8_t*>(mj.data()),
        static_cast<uint32_t>(mj.size())));
    ReadOptions ro{};
    read_options_init(&ro);
    ScanHandle* sh = nullptr;
    int64_t rows = 0;
    bool failed = !iceberg_scan_open(&sh, h, &ro);
    for (uint64_t guard = 0; !failed && guard < 100000000ull; ++guard) {
        bolt::BoltBatch b{};
        bool eof = false;
        if (!iceberg_scan_next_batch(sh, &b, &eof)) { failed = true; break; }
        if (eof) break;
        rows += b.num_rows;
    }
    if (sh != nullptr) iceberg_scan_close(sh);
    ASSERT_TRUE(bolt::limits_set(&lt, id, saved, nullptr, 0));
    EXPECT_TRUE(failed) << "scan finished with " << rows << " rows";
    const bolt::ResourceExhausted re = bolt::last_resource_exhausted();
    ASSERT_NE(re.knob, nullptr);
    EXPECT_STREQ(re.knob, "lake_metadata_budget_mb");
}
