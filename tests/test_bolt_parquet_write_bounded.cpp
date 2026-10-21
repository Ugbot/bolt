#include "bolt/bolt_column.h"
#include "bolt/bolt_port.h"
#include "bolt/ingest/bolt_parquet_write.h"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <string>

namespace {
using namespace bolt::ingest::parquet;
class ParquetWriteBounded : public ::testing::Test {
protected:
    bolt::Arena arena;
    bolt::BoltBatch batch{};
    std::array<int64_t, 128> values{};
    ParquetWriteOpts options{};
    std::filesystem::path root;
    bool owns = false;
    void SetUp() override {
        static std::atomic<uint32_t> next{0};
        root = std::filesystem::temp_directory_path() / ("bolt_bounded_pq_" +
            std::to_string(bolt_getpid()) + "_" + std::to_string(next.fetch_add(1)));
        std::error_code ec;
        ASSERT_TRUE(std::filesystem::create_directory(root, ec)); owns = true;
        bolt::BoltBatch::init_empty(&batch);
        ASSERT_TRUE(bolt::BoltBatch::alloc_columns(&batch, &arena, 1));
        ASSERT_TRUE(batch.schema.add_field("id", bolt::BoltType::Int64, false));
        batch.num_cols = 1; batch.num_rows = static_cast<int64_t>(values.size());
        for (uint32_t i = 0; i < values.size(); ++i) values[i] = i * 131;
        auto& col = batch.columns[batch.read_epoch][0];
        col.type = bolt::BoltType::Int64; col.length = batch.num_rows; col.data = values.data();
        col.type_size_bytes = sizeof(int64_t); col.format = bolt::ColumnFormat::Flat; col.stats.all_valid = true;
        options.n_columns = 1; options.compression = 0;
        options.columns[0].type = bolt::BoltType::Int64;
        std::strcpy(options.columns[0].name, "id");
    }
    void TearDown() override {
        if (!owns) return;
        std::error_code ec; std::filesystem::remove_all(root, ec); EXPECT_FALSE(ec);
    }
    uint64_t baseline() {
        const auto file = (root / "baseline.parquet").generic_string();
        auto* writer = parquet_write_open(file.c_str(), &options);
        if (writer == nullptr) return 0;
        const bool wrote = parquet_write_row_group(writer, &batch);
        const bool closed = parquet_write_close(writer);
        if (!wrote || !closed) return 0;
        std::error_code ec;
        const uint64_t size = std::filesystem::file_size(file, ec);
        return ec ? 0 : size;
    }
};
TEST_F(ParquetWriteBounded, ExactSerializedBoundaryIncludesFooterAndOptionalIndexes) {
    for (const bool index : {false, true}) {
        options.emit_page_index = index;
        const uint64_t exact = baseline(); ASSERT_GT(exact, 12u);
        for (const uint64_t cap : {exact - 1u, exact}) {
            const auto file = (root / "bounded.parquet").generic_string();
            auto* writer = parquet_write_open_bounded(file.c_str(), &options, cap);
            ASSERT_NE(writer, nullptr);
            ASSERT_TRUE(parquet_write_row_group(writer, &batch));
            EXPECT_EQ(parquet_write_close(writer), cap == exact);
            if (cap == exact) {
                EXPECT_EQ(std::filesystem::file_size(file), exact);
                EXPECT_TRUE(std::filesystem::remove(file));
            } else EXPECT_FALSE(std::filesystem::exists(file));
        }
    }
}
TEST_F(ParquetWriteBounded, PageRefusalIsStickyAndPartialSourceIsRemoved) {
    const auto file = (root / "tiny.parquet").generic_string();
    auto* writer = parquet_write_open_bounded(file.c_str(), &options, 4);
    ASSERT_NE(writer, nullptr);
    EXPECT_FALSE(parquet_write_row_group(writer, &batch));
    EXPECT_FALSE(parquet_write_row_group(writer, &batch));
    EXPECT_FALSE(parquet_write_close(writer));
    EXPECT_FALSE(std::filesystem::exists(file));
}
TEST_F(ParquetWriteBounded, InvalidCapRefusesBeforeCreatingFile) {
    const auto file = (root / "invalid.parquet").generic_string();
    for (const uint64_t cap : {uint64_t{0}, uint64_t{3}, UINT64_MAX}) {
        EXPECT_EQ(parquet_write_open_bounded(file.c_str(), &options, cap), nullptr);
        EXPECT_FALSE(std::filesystem::exists(file));
    }
}
}  // namespace
