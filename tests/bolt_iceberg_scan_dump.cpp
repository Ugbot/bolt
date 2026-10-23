// Dev/oracle tool: scan an Iceberg table in a filesystem catalog with bolt's own reader and print every row as JSON
// ({"ids":[...first column...],"names":[column names],"rows":[[...]]}; null => null; unsupported column types print as "<type>").
//   bolt_iceberg_scan_dump <catalog_root> <namespace> <table>
// Used by scripts/oracles/verify_v3_external.py to compare bolt against Iceberg-Java on tables bolt did not write.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/scan.h"

int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: %s <catalog_root> <ns> <table>\n", argv[0]); return 2; }
    using namespace bolt::lakehouse;
    using namespace bolt::lakehouse::iceberg;
    bolt::Arena arena;
    FilesystemCatalog fc{}; Catalog cat{};
    if (!filesystem_catalog_init(&fc, argv[1], &cat)) { std::fprintf(stderr, "catalog init failed\n"); return 3; }
    TableHandle* th = nullptr;
    if (!iceberg_table_open(&th, &arena, &cat, argv[2], argv[3])) { std::fprintf(stderr, "table open failed\n"); return 4; }
    ReadOptions ro{}; read_options_init(&ro);
    ScanHandle* sh = nullptr;
    if (!iceberg_scan_open(&sh, th, &ro)) { std::fprintf(stderr, "scan open failed\n"); return 5; }
    std::string ids, rows, names;
    bool first_id = true, first_row = true, have_names = false;
    for (int g = 0; g < 100000; ++g) {
        bolt::BoltBatch b{}; bool eof = false;
        if (!iceberg_scan_next_batch(sh, &b, &eof)) { std::fprintf(stderr, "scan failed\n"); return 6; }
        if (eof) break;
        if (!have_names) {
            names = "[";
            for (int32_t c = 0; c < b.num_cols; ++c) names += (c ? ",\"" : "\"") + std::string(b.schema.fields[c].name) + "\"";
            names += "]";
            have_names = true;
        }
        for (int64_t r = 0; r < b.num_rows; ++r) {
            std::string row = "[";
            for (int32_t c = 0; c < b.num_cols; ++c) {
                const auto& col = b.columns[b.read_epoch][c];
                if (c) row += ",";
                if (col.is_null(r)) { row += "null"; continue; }
                char buf[64];
                switch (col.type) {
                    case bolt::BoltType::Int64: std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(static_cast<const int64_t*>(col.data)[r])); row += buf; break;
                    case bolt::BoltType::Int32: std::snprintf(buf, sizeof(buf), "%d", static_cast<const int32_t*>(col.data)[r]); row += buf; break;
                    case bolt::BoltType::Float64: std::snprintf(buf, sizeof(buf), "%.17g", static_cast<const double*>(col.data)[r]); row += buf; break;
                    case bolt::BoltType::Float32: std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(static_cast<const float*>(col.data)[r])); row += buf; break;
                    case bolt::BoltType::Bool: row += static_cast<const uint8_t*>(col.data)[r] ? "true" : "false"; break;
                    case bolt::BoltType::Utf8: {
                        const uint8_t* d = nullptr; int32_t n = 0;
                        col.utf8_at(r, &d, &n);
                        row += "\"";
                        for (int32_t i = 0; i < n; ++i) { const char ch = static_cast<char>(d[i]); if (ch == '"' || ch == '\\') row += '\\'; row += ch; }
                        row += "\"";
                        break;
                    }
                    default: row += "\"<unsupported>\""; break;
                }
            }
            row += "]";
            rows += (first_row ? "" : ",") + row;
            first_row = false;
            if (b.num_cols > 0 && b.columns[b.read_epoch][0].type == bolt::BoltType::Int64) {
                ids += (first_id ? "" : ",") + std::to_string(static_cast<const int64_t*>(b.columns[b.read_epoch][0].data)[r]);
                first_id = false;
            }
        }
    }
    std::printf("{\"ids\":[%s],\"names\":%s,\"rows\":[%s]}\n", ids.c_str(), have_names ? names.c_str() : "[]", rows.c_str());
    iceberg_scan_close(sh); iceberg_table_close(th);
    return 0;
}
