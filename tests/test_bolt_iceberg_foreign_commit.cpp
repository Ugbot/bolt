// G2ICE-279 — commit through bolt to a table another writer (pyiceberg) made, without losing its metadata.
//
// The C++ half only opens the table at BOLT_ICEBERG_FOREIGN_DIR and either appends one batch
// (BOLT_ICEBERG_FOREIGN_MODE=append) or checks the commit is refused (=refuse, a v1 table). The verdict that
// refs, sort orders, properties and statistics survived comes from scripts/iceberg_foreign_commit.py, which
// built the table with pyiceberg and reads the result back with it. Without the env the test skips.

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/writer.h"
#include "bolt/lakehouse/object_store.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

namespace {

using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;

void make_batch(bolt::Arena* a, bolt::BoltBatch* out) {
    constexpr int32_t n = 3;
    bolt::BoltBatch::init_empty(out);
    out->arena    = a;
    out->num_rows = n;
    out->num_cols = 2;
    bolt::BoltBatch::alloc_columns(out, a, 2);
    out->schema.add_field("id",    bolt::BoltType::Int64,   false);
    out->schema.add_field("score", bolt::BoltType::Float64, false);
    auto* cols = out->columns[out->read_epoch];
    auto* idata = a->allocate_array<int64_t>(n);
    auto* sdata = a->allocate_array<double>(n);
    for (int32_t k = 0; k < n; ++k) {
        idata[k] = 900 + k;
        sdata[k] = 0.25 * k;
    }
    cols[0].type = bolt::BoltType::Int64;   cols[0].length = n; cols[0].data = idata;
    cols[1].type = bolt::BoltType::Float64; cols[1].length = n; cols[1].data = sdata;
}

}  // namespace

TEST(IcebergForeignCommit, AppendKeepsForeignMetadataOrRefuses) {
    const char* dir = std::getenv("BOLT_ICEBERG_FOREIGN_DIR");
    const char* mode = std::getenv("BOLT_ICEBERG_FOREIGN_MODE");
    if (dir == nullptr || dir[0] == '\0' || mode == nullptr) GTEST_SKIP() << "driven by scripts/iceberg_foreign_commit.py";
    bolt::Arena arena;
    FilesystemObjectStore fs{}; ObjectStore os{};
    ASSERT_TRUE(filesystem_object_store_init(&fs, dir, &os));
    TableHandle* th = nullptr;
    ASSERT_TRUE(table_open(&th, &arena, &os, dir));
    bolt::BoltBatch b{};
    make_batch(&arena, &b);
    AppendHandle* ah = nullptr;
    if (std::strcmp(mode, "refuse") == 0) {
        EXPECT_FALSE(append_open(&ah, th));
        EXPECT_EQ(table_last_commit_error(th), CommitError::kUnimplemented);
        EXPECT_FALSE(std::filesystem::exists(std::filesystem::path(dir) / "metadata" / "v2.metadata.json"));
        return;
    }
    if (std::strcmp(mode, "delete") == 0) {   // G2ICE-280: drops pyiceberg's whole data file (ids 1..3)
        Predicate p{};
        std::strcpy(p.column, "id");
        p.op = PredicateOp::kLe;
        p.value.type = bolt::BoltType::Int64;
        p.value.i64 = 3;
        ASSERT_TRUE(table_delete(th, &p)) << "commit error " << static_cast<int>(table_last_commit_error(th));
        return;
    }
    ASSERT_TRUE(append_open(&ah, th));
    ASSERT_TRUE(append_write(ah, &b));
    ASSERT_TRUE(append_commit(ah)) << "commit error " << static_cast<int>(table_last_commit_error(th));
    append_close(ah);
}
