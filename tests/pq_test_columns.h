// pq_test_columns.h — column storage for tests that call pq_parse_file_meta
// directly (parquet_read_meta sizes it from an arena). Every call returns a
// fresh array, so two live PqMeta never alias.
#pragma once

#include <cstdint>
#include <vector>

#include "bolt/ingest/bolt_parquet_meta.h"

inline constexpr std::uint32_t kPqTestColumns = 256;

inline bolt::ingest::parquet::PqColumn* pq_test_columns() {
    static thread_local std::vector<std::vector<bolt::ingest::parquet::PqColumn>> pool;
    pool.emplace_back(kPqTestColumns);
    return pool.back().data();
}
